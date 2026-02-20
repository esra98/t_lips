"""
Generate an LRS2-compatible sentence-level lip-reading dataset.

LRS2 dataset structure
======================
dataset/
  lrs2/
    pretrain/                     # (or train / val / test)
      <video_id>/
        <utterance_id>.mp4        # 25 fps, 224x224 face-cropped video (with audio)
        <utterance_id>.txt        # transcript + word-level timing
    pretrain.txt                  # list of  <video_id>/<utterance_id>  per line

Each .txt file
--------------
Text:  <FULL SENTENCE IN UPPERCASE>
WORD1 <start_offset> <end_offset> <confidence>
WORD2 <start_offset> <end_offset> <confidence>
...

Timing offsets are **relative to the clip start** (i.e. the clip begins at 0).

Each .mp4 file
--------------
Face-cropped & resized to 224x224, re-encoded at 25 fps, with AAC audio track.
"""

import cv2
import json
import os
import subprocess
import textwrap
from datetime import datetime
from pathlib import Path

import mediapipe as mp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_FPS = 25
FACE_CROP_SIZE = 224          # LRS2 standard face crop
LIP_CROP_SIZE = 96            # optional lip-only variant
FACE_MARGIN = 30              # extra pixels around detected face box
MIN_SEGMENT_WORDS = 2         # skip segments with fewer words
MAX_SEGMENT_DURATION = 12.0   # seconds – cap very long segments
MIN_SEGMENT_DURATION = 0.4    # seconds – skip very short segments
ERROR_LOG_FILE = "crop_errors.log"

mp_face_mesh = mp.solutions.face_mesh

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_error(message: str):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

# ---------------------------------------------------------------------------
# Face / lip bounding-box helpers (reuse from word-level pipeline)
# ---------------------------------------------------------------------------

def get_face_bbox(landmarks, w, h, margin=FACE_MARGIN):
    """Full-face bounding box with margin."""
    xs = [int(p.x * w) for p in landmarks]
    ys = [int(p.y * h) for p in landmarks]
    x_min = max(0, min(xs) - margin)
    x_max = min(w, max(xs) + margin)
    y_min = max(0, min(ys) - margin)
    y_max = min(h, max(ys) + margin)
    # make it square (LRS2 expects square crops)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    side = max(x_max - x_min, y_max - y_min)
    half = side // 2
    x_min = max(0, cx - half)
    x_max = min(w, cx + half)
    y_min = max(0, cy - half)
    y_max = min(h, cy + half)
    return x_min, y_min, x_max, y_max


def get_lip_bbox(landmarks, w, h):
    """Lower-face / lip bounding box (square, nose-to-chin)."""
    chin_bottom = landmarks[152]
    nose_tip = landmarks[1]
    nose_top = landmarks[168]
    left_face = landmarks[234]
    right_face = landmarks[454]

    chin_y = int(chin_bottom.y * h)
    nose_mid_y = (int(nose_tip.y * h) + int(nose_top.y * h)) // 2
    lip_mid_x = (int(left_face.x * w) + int(right_face.x * w)) // 2

    lip_height = chin_y - nose_mid_y
    half = lip_height // 2

    x_min = max(0, lip_mid_x - half)
    x_max = min(w, lip_mid_x + half)
    y_min = max(0, nose_mid_y)
    y_max = min(h, chin_y)
    return x_min, y_min, x_max, y_max


def crop_and_resize(frame, x_min, y_min, x_max, y_max, size):
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        crop = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    else:
        crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return crop

# ---------------------------------------------------------------------------
# Video I/O helpers
# ---------------------------------------------------------------------------

def _save_raw_video(frames, out_path, fps):
    """Write frames with OpenCV (no audio)."""
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
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Audio extraction failed: {audio_out} – {e}")
        return False


def _mux(video_path, audio_path, output_path):
    """Mux raw video + audio into final mp4 at TARGET_FPS."""
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
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        os.replace(tmp, output_path)
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"Muxing failed for {output_path}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False

# ---------------------------------------------------------------------------
# Frame-level face cropping for a time range
# ---------------------------------------------------------------------------

def _crop_frames(cap, face_mesh, start_frame, num_frames, bbox_fn, crop_size):
    """Read *num_frames* from *start_frame*, detect face, crop & resize."""
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
            # If face not detected or multiple faces, skip frame
            continue
        lm = results.multi_face_landmarks[0].landmark
        x1, y1, x2, y2 = bbox_fn(lm, w, h)
        crop = crop_and_resize(frame, x1, y1, x2, y2, crop_size)
        frames.append(crop)
    return frames

# ---------------------------------------------------------------------------
# LRS2 annotation writer
# ---------------------------------------------------------------------------

def _write_lrs2_annotation(txt_path, sentence, words, clip_start):
    """
    Write an LRS2-style annotation file.

    Format
    ------
    Text:  <SENTENCE>
    WORD start_offset end_offset confidence
    """
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

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_lrs2_dataset(
    video_path,
    json_path=None,
    output_dir="dataset/lrs2",
    split="pretrain",
    crop_type="face",        # "face" (224x224) or "lip" (96x96)
):
    """
    Process one source video into LRS2-format sentence clips.

    Parameters
    ----------
    video_path : str | Path
        Source video file.
    json_path : str | Path | None
        Whisper-timestamped JSON with word-level timestamps.
    output_dir : str | Path
        Root of the LRS2-style dataset tree.
    split : str
        One of pretrain / train / val / test.
    crop_type : str
        "face" → 224x224 full face crop (LRS2 default).
        "lip"  → 96x96 lip-only crop.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    split_dir = output_dir / split

    if json_path is None:
        json_path = Path("timestamps") / f"{video_path.stem}_ts.json"
    json_path = Path(json_path)

    if crop_type == "lip":
        bbox_fn = get_lip_bbox
        crop_size = LIP_CROP_SIZE
    else:
        bbox_fn = lambda lm, w, h: get_face_bbox(lm, w, h)
        crop_size = FACE_CROP_SIZE

    # ------------------------------------------------------------------
    # Load Whisper JSON
    # ------------------------------------------------------------------
    if not json_path.exists():
        log_error(f"Missing JSON: {json_path}")
        raise FileNotFoundError(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ------------------------------------------------------------------
    # Open source video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps == 0:
        cap.release()
        raise RuntimeError(f"Cannot read FPS from {video_path}")

    video_id = video_path.stem  # e.g. "video-1"
    video_out_dir = split_dir / video_id
    video_out_dir.mkdir(parents=True, exist_ok=True)

    file_list_entries = []  # for the split .txt file
    utterance_idx = 0

    # ------------------------------------------------------------------
    # Iterate over Whisper segments (≈ sentences)
    # ------------------------------------------------------------------
    segments = data.get("segments", [])

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        for seg_i, segment in enumerate(segments):
            words = segment.get("words", [])
            seg_text = segment.get("text", "").strip()

            # Filter words that have valid timing
            valid_words = [
                w for w in words
                if w.get("start") is not None
                and w.get("end") is not None
                and w["end"] > w["start"]
            ]

            if len(valid_words) < MIN_SEGMENT_WORDS:
                continue

            seg_start = valid_words[0]["start"]
            seg_end = valid_words[-1]["end"]
            seg_duration = seg_end - seg_start

            if seg_duration < MIN_SEGMENT_DURATION:
                continue

            # For very long segments, split them into sub-utterances
            sub_utterances = _split_if_long(valid_words, seg_text, MAX_SEGMENT_DURATION)

            for sub_words, sub_text in sub_utterances:
                clip_start = max(0.0, sub_words[0]["start"] - 0.04)
                clip_end = sub_words[-1]["end"] + 0.04
                clip_duration = clip_end - clip_start

                start_frame = int(clip_start * src_fps)
                num_frames = int(clip_duration * src_fps)

                utterance_id = f"{utterance_idx:05d}"
                mp4_out = video_out_dir / f"{utterance_id}.mp4"
                txt_out = video_out_dir / f"{utterance_id}.txt"
                audio_tmp = video_out_dir / f"{utterance_id}_audio.m4a"
                video_tmp = video_out_dir / f"{utterance_id}_raw.mp4"

                # Crop faces
                frames = _crop_frames(
                    cap, face_mesh, start_frame, num_frames,
                    bbox_fn, crop_size,
                )

                if len(frames) < 3:
                    log_error(
                        f"Too few face frames for segment {seg_i} "
                        f"('{sub_text[:40]}…') in {video_path}"
                    )
                    continue

                # Save raw cropped video
                if not _save_raw_video(frames, video_tmp, src_fps):
                    continue

                # Extract audio
                if not _extract_audio(video_path, audio_tmp, clip_start, clip_duration):
                    _cleanup(video_tmp, audio_tmp)
                    continue

                # Mux → final mp4 (re-encoded at 25 fps with audio)
                if not _mux(video_tmp, audio_tmp, mp4_out):
                    _cleanup(video_tmp, audio_tmp, mp4_out)
                    continue

                # Write LRS2 annotation
                _write_lrs2_annotation(txt_out, sub_text, sub_words, clip_start)

                # Cleanup temp files
                _cleanup(video_tmp, audio_tmp)

                file_list_entries.append(f"{video_id}/{utterance_id}")
                utterance_idx += 1

                print(
                    f"  ✅ [{video_id}/{utterance_id}] "
                    f"({clip_duration:.2f}s, {len(frames)} frames) "
                    f"'{sub_text[:60]}'"
                )

    cap.release()

    # ------------------------------------------------------------------
    # Append to split file  (e.g. dataset/lrs2/pretrain.txt)
    # ------------------------------------------------------------------
    split_file = output_dir / f"{split}.txt"
    with open(split_file, "a", encoding="utf-8") as f:
        for entry in file_list_entries:
            f.write(entry + "\n")

    summary = (
        f"✅ {video_path.name}: generated {utterance_idx} utterances "
        f"→ {split_dir / video_id}"
    )
    print(summary)
    log_error(summary)
    return file_list_entries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_if_long(words, text, max_dur):
    """
    If a segment is longer than *max_dur* seconds, split it into
    sub-utterances at natural word boundaries (roughly equal halves).
    Returns list of (sub_words, sub_text) tuples.
    """
    total_dur = words[-1]["end"] - words[0]["start"]
    if total_dur <= max_dur:
        return [(words, text)]

    # Split at the midpoint word
    mid = len(words) // 2
    left_words = words[:mid]
    right_words = words[mid:]
    left_text = " ".join(w["text"].strip() for w in left_words)
    right_text = " ".join(w["text"].strip() for w in right_words)

    # Recurse in case halves are still too long
    result = []
    result.extend(_split_if_long(left_words, left_text, max_dur))
    result.extend(_split_if_long(right_words, right_text, max_dur))
    return result


def _cleanup(*paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            p.unlink()
