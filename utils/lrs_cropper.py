import cv2
import mediapipe as mp
from pathlib import Path
import subprocess
import json
import os
from datetime import datetime
import re
import numpy as np
import tempfile

TARGET_FPS = 25
FACE_CROP_SIZE = 224
FACE_MARGIN = 30
MIN_SEGMENT_WORDS = 2
MAX_SEGMENT_DURATION = 10.0
MIN_SEGMENT_DURATION = 0.4
ERROR_LOG_FILE = "crop_errors.log"

mp_face_mesh = mp.solutions.face_mesh


def log_error(message: str):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def get_face_bbox(landmarks, w, h, margin=FACE_MARGIN):
    xs = [int(p.x * w) for p in landmarks]
    ys = [int(p.y * h) for p in landmarks]
    x_min = max(0, min(xs) - margin)
    x_max = min(w, max(xs) + margin)
    y_min = max(0, min(ys) - margin)
    y_max = min(h, max(ys) + margin)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    side = max(x_max - x_min, y_max - y_min)
    half = side // 2
    x_min = max(0, cx - half)
    x_max = min(w, cx + half)
    y_min = max(0, cy - half)
    y_max = min(h, cy + half)
    return x_min, y_min, x_max, y_max


def crop_and_resize(frame, x_min, y_min, x_max, y_max, size):
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        crop = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    else:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop


def _save_raw_video(frames, out_path, fps):
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    for f in frames:
        writer.write(f)
    writer.release()
    return True


def _extract_audio(src_video, audio_out, start, duration):
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(src_video),
        "-t", str(duration),
        "-vn", "-c:a", "aac", "-ar", "16000", "-ac", "1",
        str(audio_out),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Audio extraction failed: {audio_out} – {e}")
        return False


def _mux(video_path, audio_path, output_path):
    tmp = str(Path(output_path).with_suffix(".muxed.mp4"))
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-r", str(TARGET_FPS),
        "-c:a", "aac", "-ar", "16000", "-ac", "1",
        "-shortest",
        tmp,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(tmp, output_path)
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Muxing failed for {output_path}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False


def _crop_frames(cap, face_mesh, start_frame, num_frames, bbox_fn, crop_size):
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks or len(results.multi_face_landmarks) != 1:
            continue
        lm = results.multi_face_landmarks[0].landmark
        x1, y1, x2, y2 = bbox_fn(lm, w, h)
        crop = crop_and_resize(frame, x1, y1, x2, y2, crop_size)
        frames.append(crop)
    return frames


def _write_lrs2_annotation(txt_path, sentence, words, clip_start):
    lines = [f"Text:  {sentence.upper()}"]
    for w in words:
        text = w["text"].strip()
        s = w["start"] - clip_start
        e = w["end"] - clip_start
        conf = w.get("confidence", 1.0)
        lines.append(f"{text.upper()} {s:.4f} {e:.4f} {conf:.4f}")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _cleanup(*paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            p.unlink()


def split_by_punctuation(words, max_duration=MAX_SEGMENT_DURATION):
    split_indices = [0]
    for i, w in enumerate(words):
        if re.match(r'[.,?!]', w['text']) and i + 1 < len(words):
            split_indices.append(i + 1)
    split_indices.append(len(words))
    chunks = []
    for start, end in zip(split_indices[:-1], split_indices[1:]):
        chunk = words[start:end]
        if not chunk:
            continue
        chunk_start = chunk[0]['start']
        chunk_end = chunk[-1]['end']
        chunk_dur = chunk_end - chunk_start
        if chunk_dur > max_duration and len(chunk) > 1:
            mid = len(chunk) // 2
            chunks.extend(split_by_punctuation(chunk[:mid], max_duration))
            chunks.extend(split_by_punctuation(chunk[mid:], max_duration))
        else:
            chunk_text = ' '.join(w['text'] for w in chunk)
            chunks.append((chunk, chunk_text))
    return chunks


def get_silence_timestamps(video_path, silence_thresh=-35, min_silence_len=0.3):
    import re
    import subprocess
    silence_cmd = [
        "ffmpeg", "-i", str(video_path),
        "-af", f"silencedetect=noise={silence_thresh}dB:d={min_silence_len}",
        "-f", "null", "-"
    ]
    result = subprocess.run(silence_cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    output = result.stderr
    silence_end_times = []
    for line in output.splitlines():
        m = re.search(r'silence_end: ([0-9.]+)', line)
        if m:
            silence_end_times.append(float(m.group(1)))
    return silence_end_times


def split_by_silence_no_word_cut(words, silence_times, max_duration=MAX_SEGMENT_DURATION, min_words=2):
    def safe_chunk_text(chunk):
        return ' '.join(x['text'] for x in chunk if x.get('text'))

    if not words:
        return []

    if not silence_times:
        chunks = []
        chunk = []
        chunk_start = words[0]['start']
        for w in words:
            if chunk and (w['end'] - chunk_start) > max_duration:
                chunk_text = safe_chunk_text(chunk)
                if chunk_text:
                    chunks.append((chunk.copy(), chunk_text))
                chunk = []
                chunk_start = w['start']
            if not chunk or (w['end'] - chunk_start) <= max_duration:
                chunk.append(w)
        if chunk:
            chunk_text = safe_chunk_text(chunk)
            if chunk_text:
                chunks.append((chunk.copy(), chunk_text))
        return chunks

    chunks = []
    chunk = []
    silence_idx = 0
    chunk_start = words[0]['start']
    for i, w in enumerate(words):
        if chunk and (w['end'] - chunk_start) > max_duration:
            if not chunk:
                chunk = []
                chunk_start = w['end']
                continue
            chunk_text = safe_chunk_text(chunk)
            if chunk_text:
                chunks.append((chunk.copy(), chunk_text))
            chunk = []
            chunk_start = w['start']

        if not chunk or (w['end'] - chunk_start) <= max_duration:
            chunk.append(w)

        while silence_idx < len(silence_times) and silence_times[silence_idx] < w['end']:
            if len(chunk) >= min_words:
                chunk_text = safe_chunk_text(chunk)
                if chunk_text:
                    chunks.append((chunk.copy(), chunk_text))
                chunk = []
                chunk_start = w['end']
            silence_idx += 1

    if chunk:
        chunk_text = safe_chunk_text(chunk)
        if chunk_text:
            chunks.append((chunk.copy(), chunk_text))
    return chunks


def _is_sentence_break_word(t: str) -> bool:
    if not t:
        return False
    t = t.strip()
    if t in {".", ",", "?", "!", "(", ")"}:
        return True
    return bool(re.search(r"[.,?!]$", t))


def _normalize_text_from_words(ws):
    txt = " ".join(w["text"].strip() for w in ws if w.get("text"))
    txt = re.sub(r"\s+([.,?!])", r"\1", txt).strip()
    return txt


def split_by_timestamp_rules(words, max_duration=MAX_SEGMENT_DURATION, min_words=MIN_SEGMENT_WORDS):
    ws = [
        w for w in words
        if w.get("start") is not None
        and w.get("end") is not None
        and w["end"] > w["start"]
        and w.get("text") is not None
        and str(w.get("text")).strip() != ""
    ]

    if len(ws) < min_words:
        return []

    chunks = []
    current = []
    cur_start = None

    for w in ws:
        t = str(w["text"]).strip()

        if not current:
            current = [w]
            cur_start = w["start"]
        else:
            prospective = w["end"] - cur_start
            if prospective > max_duration:
                if len(current) >= min_words:
                    chunks.append((current, _normalize_text_from_words(current)))
                return chunks
            current.append(w)

        if _is_sentence_break_word(t):
            if len(current) >= min_words:
                chunks.append((current, _normalize_text_from_words(current)))
            current = []
            cur_start = None

    if current and len(current) >= min_words:
        chunks.append((current, _normalize_text_from_words(current)))

    return chunks


def crop_sentences(
    video_path,
    json_path=None,
    output_dir=Path("videos/dataset/lrs2"),
    split="pretrain",
    crop_type="face"
):
    video_path = Path(video_path)
    if json_path is None:
        json_path = Path("timestamps") / f"{video_path.stem}_ts.json"
    json_path = Path(json_path)
    output_dir = Path(output_dir)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    if crop_type == "face":
        bbox_fn = lambda lm, w, h: get_face_bbox(lm, w, h)
        crop_size = FACE_CROP_SIZE
    else:
        raise NotImplementedError("Only face cropping is implemented.")

    if not json_path.exists():
        log_error(f"Missing JSON: {json_path}")
        raise FileNotFoundError(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps == 0:
        cap.release()
        raise RuntimeError(f"Cannot read FPS from {video_path}")

    video_id = video_path.stem
    video_out_dir = (output_dir / split) / video_id
    video_out_dir.mkdir(parents=True, exist_ok=True)

    file_list_entries = []
    utterance_idx = 0
    segments = data.get("segments", [])

    _ = get_silence_timestamps(video_path)

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        for seg in segments:
            words = seg.get("words", [])
            valid_words = [
                w for w in words
                if w.get("start") is not None and w.get("end") is not None and w["end"] > w["start"]
            ]
            if len(valid_words) < MIN_SEGMENT_WORDS:
                continue

            for chunk_words, chunk_text in split_by_timestamp_rules(
                valid_words,
                max_duration=MAX_SEGMENT_DURATION,
                min_words=MIN_SEGMENT_WORDS
            ):
                if not chunk_words:
                    continue

                chunk_start = chunk_words[0]["start"]
                chunk_end = chunk_words[-1]["end"]
                chunk_dur = chunk_end - chunk_start
                if chunk_dur < MIN_SEGMENT_DURATION:
                    continue

                clip_start = max(0.0, chunk_start - 0.04)
                clip_end = chunk_end + 0.04
                clip_duration = clip_end - clip_start

                if clip_duration > (MAX_SEGMENT_DURATION + 0.10):
                    chunk_words = [
                        w for w in chunk_words
                        if (w["end"] - chunk_words[0]["start"]) <= MAX_SEGMENT_DURATION
                    ]
                    if len(chunk_words) < MIN_SEGMENT_WORDS:
                        continue
                    chunk_start = chunk_words[0]["start"]
                    chunk_end = chunk_words[-1]["end"]
                    clip_start = max(0.0, chunk_start - 0.04)
                    clip_end = chunk_end + 0.04
                    clip_duration = clip_end - clip_start
                    if clip_duration > (MAX_SEGMENT_DURATION + 0.10):
                        continue

                start_frame = int(round(clip_start * src_fps))
                num_frames = int(round(clip_duration * src_fps))

                utterance_id = f"{utterance_idx:05d}"
                mp4_out = video_out_dir / f"{utterance_id}.mp4"
                txt_out = video_out_dir / f"{utterance_id}.txt"
                audio_tmp = video_out_dir / f"{utterance_id}_audio.m4a"
                video_tmp = video_out_dir / f"{utterance_id}_raw.mp4"

                frames = _crop_frames(cap, face_mesh, start_frame, num_frames, bbox_fn, crop_size)
                if len(frames) < 3:
                    log_error(f"Too few face frames for chunk '{chunk_text[:40]}…' in {video_path}")
                    continue

                if not _save_raw_video(frames, video_tmp, src_fps):
                    continue

                if not _extract_audio(video_path, audio_tmp, clip_start, clip_duration):
                    _cleanup(video_tmp, audio_tmp)
                    continue

                if not _mux(video_tmp, audio_tmp, mp4_out):
                    _cleanup(video_tmp, audio_tmp, mp4_out)
                    continue

                # Save standalone audio file for this chunk
                audio_out = video_out_dir / f"{utterance_id}.aac"
                if audio_tmp.exists():
                    audio_tmp.replace(audio_out)

                chunk_text = _normalize_text_from_words(chunk_words)
                _write_lrs2_annotation(txt_out, chunk_text, chunk_words, clip_start)

                _cleanup(video_tmp)  # Only remove temp video, keep audio_out

                file_list_entries.append(f"{video_id}/{utterance_id}")
                utterance_idx += 1
                print(f"  ✅ [{video_id}/{utterance_id}] ({clip_duration:.2f}s, {len(frames)} frames) '{chunk_text[:60]}'")

    cap.release()

    split_file = output_dir / f"{split}.txt"
    with open(split_file, "a", encoding="utf-8") as f:
        for entry in file_list_entries:
            f.write(entry + "\n")

    summary = f"✅ {video_path.name}: generated {utterance_idx} utterances → {video_out_dir}"
    print(summary)
    log_error(summary)
    return file_list_entries