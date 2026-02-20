# Word-level LRW cropper (refactored from generate_videos.py)
# Usage: crop_clips(video_path, json_path, output_roots, original_filename)

import cv2
import mediapipe as mp
from pathlib import Path
import subprocess
import textwrap
import json
import os
from datetime import datetime

word_sample_size = 500
mp_face_mesh = mp.solutions.face_mesh
ERROR_LOG_FILE = "crop_errors.log"

def log_error(message):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

def load_word_list(filepath="frequency_list_500.txt"):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return set(line.strip().lower() for line in f if line.strip())
    except Exception as e:
        log_error(f"Failed to load word list from {filepath}: {e}")
        return set()

ALLOWED_WORDS = load_word_list()

def get_next_index(word_dir: Path) -> int:
    if not word_dir.exists():
        return 1
    existing = [int(f.stem) for f in word_dir.glob("*.mp4") if f.stem.isdigit()]
    return max(existing, default=0) + 1

def valid_file(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def mux_video_audio(video_path, audio_path, output_path):
    temp_output = str(Path(output_path).with_suffix('.muxed.mp4'))
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        temp_output
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.replace(temp_output, output_path)
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg muxing failed for {output_path}: {e}"
        print(f"❌ {error_msg}")
        log_error(error_msg)
        if os.path.exists(temp_output):
            os.remove(temp_output)
        return False

def get_bbox(landmarks, w, h, indices=None, margin=0):
    if indices is None:
        points = [p for p in landmarks]
    else:
        points = [landmarks[i] for i in indices]
    xs = [int(p.x * w) for p in points]
    ys = [int(p.y * h) for p in points]
    x_min = max(0, min(xs) - margin)
    x_max = min(w, max(xs) + margin)
    y_min = max(0, min(ys) - margin)
    y_max = min(h, max(ys) + margin)
    return x_min, y_min, x_max, y_max

def get_face_lip_box(landmarks, w, h):
    chin_bottom_idx = 152
    nose_tip_idx = 1
    nose_top_idx = 168
    left_face_idx = 234
    right_face_idx = 454
    chin_bottom = landmarks[chin_bottom_idx]
    nose_tip = landmarks[nose_tip_idx]
    nose_top = landmarks[nose_top_idx]
    left_face = landmarks[left_face_idx]
    right_face = landmarks[right_face_idx]
    chin_bottom_px = int(chin_bottom.x * w), int(chin_bottom.y * h)
    nose_tip_px = int(nose_tip.x * w), int(nose_tip.y * h)
    nose_top_px = int(nose_top.x * w), int(nose_top.y * h)
    left_face_px = int(left_face.x * w), int(left_face.y * h)
    right_face_px = int(right_face.x * w), int(right_face.y * h)
    nose_vertical_mid = ((nose_tip_px[0] + nose_top_px[0]) // 2,
                         (nose_tip_px[1] + nose_top_px[1]) // 2)
    nose_horizontal_mid_x = (left_face_px[0] + right_face_px[0]) // 2
    lip_box_mid_x = nose_horizontal_mid_x
    lip_top_y = nose_vertical_mid[1]
    lip_bottom_y = chin_bottom_px[1]
    lip_height = lip_bottom_y - lip_top_y
    lip_left_x = int(lip_box_mid_x - lip_height // 2)
    lip_right_x = int(lip_box_mid_x + lip_height // 2)
    lip_left_x = max(0, lip_left_x)
    lip_right_x = min(w, lip_right_x)
    lip_top_y = max(0, lip_top_y)
    lip_bottom_y = min(h, lip_bottom_y)
    return lip_left_x, lip_top_y, lip_right_x, lip_bottom_y

def crop_and_resize(frame, x_min, y_min, x_max, y_max, output_size):
    crop = frame[y_min:y_max, x_min:x_max]
    if crop.size == 0 or (x_max-x_min) < 1 or (y_max-y_min) < 1:
        crop = cv2.resize(frame, (output_size, output_size), interpolation=cv2.INTER_AREA)
    else:
        crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return crop

def save_video(frames, out_path, fps):
    if len(frames) == 0:
        error_msg = f"No frames to save for {out_path}"
        log_error(error_msg)
        return False
    try:
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in frames:
            writer.write(frame)
        writer.release()
        return True
    except Exception as e:
        error_msg = f"Failed to save video {out_path}: {e}"
        log_error(error_msg)
        return False

def extract_audio(video_path, audio_out, clip_start, clip_duration):
    cmd_audio = [
        "ffmpeg", "-y",
        "-ss", str(clip_start),
        "-i", str(video_path),
        "-t", str(clip_duration),
        "-vn",
        "-c:a", "aac",
        str(audio_out)
    ]
    try:
        subprocess.run(cmd_audio, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError as e:
        error_msg = f"ffmpeg audio extraction failed for {audio_out}: {e}"
        print(f"❌ {error_msg}")
        log_error(error_msg)
        if os.path.exists(audio_out):
            os.remove(audio_out)
        return False

def save_meta(meta_out, meta_text):
    try:
        with open(meta_out, "w", encoding="utf-8") as f:
            f.write(meta_text.strip() + "\n")
    except Exception as e:
        error_msg = f"Failed to save metadata {meta_out}: {e}"
        log_error(error_msg)

def reached_sample_limit(word: str, output_roots) -> bool:
    target_filename = f"{word_sample_size}.mp4"
    for out_root in output_roots.values():
        if (out_root / word / target_filename).exists():
            return True
    return False

def process_clip(
    cap, face_mesh, start_frame, num_frames, bbox_fn, crop_size
):
    frames = []
    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks or len(results.multi_face_landmarks) != 1:
            continue
        landmarks = results.multi_face_landmarks[0].landmark
        x_min, y_min, x_max, y_max = bbox_fn(landmarks, w, h)
        crop = crop_and_resize(frame, x_min, y_min, x_max, y_max, crop_size)
        frames.append(crop)
    return frames

def crop_clips(
    video_path,
    json_path=None,
    output_roots=None,
    original_filename="unknown"
):
    video_path = Path(video_path)
    if json_path is None:
        json_path = video_path.with_name(video_path.stem + "_ts.json")
    if output_roots is None:
        output_roots = {
            "standart": Path("dataset/standart"),
            "standart_lip": Path("dataset/standart_lip"),
            "lip": Path("dataset/lip")
        }
    for out_root in output_roots.values():
        out_root.mkdir(parents=True, exist_ok=True)
    if not json_path.exists():
        error_msg = f"Missing JSON file: {json_path}"
        print(f"❌ {error_msg}")
        log_error(error_msg)
        return
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        error_msg = f"Failed to load JSON {json_path}: {e}"
        print(f"❌ {error_msg}")
        log_error(error_msg)
        return
    filtered_segments = []
    for segment in data.get("segments", []):
        filtered_words = [
            word_info for word_info in segment.get("words", [])
            if (
                word_info.get("text", "").strip().lower() in ALLOWED_WORDS
                and word_info.get("start") is not None
                and word_info.get("end") is not None
                and word_info.get("end") > word_info.get("start")
            )
        ]
        if filtered_words:
            filtered_segments.append({"words": filtered_words})
    if not filtered_segments:
        error_msg = f"No valid words to process in {video_path}"
        print(f"❌ {error_msg}")
        log_error(error_msg)
        return
    all_words = [
        word_info
        for segment in filtered_segments
        for word_info in segment.get("words", [])
    ]
    total_words = len(all_words)
    word_counter = 0
    successful_crops = 0
    datasets = [
        {
            "key": "standart",
            "bbox_fn": lambda lm, w, h: get_bbox(lm, w, h, indices=None, margin=5),
            "crop_size": 256,
            "clip_duration": 1.16,
            "meta_suffix": "standart_256x256_1.16s",
            "audio_length": lambda start, end: 1.16,
            "clip_start": lambda start, end: max(0, ((start + end) / 2) - 1.16 / 2),
            "start_frame": lambda start, end, fps: int(max(0, ((start + end) / 2) - 1.16 / 2) * fps),
            "num_frames": lambda start, end, fps: int(1.16 * fps)
        },
        {
            "key": "standart_lip",
            "bbox_fn": get_face_lip_box,
            "crop_size": 96,
            "clip_duration": 1.16,
            "meta_suffix": "standart_lip_96x96_1.16s",
            "audio_length": lambda start, end: 1.16,
            "clip_start": lambda start, end: max(0, ((start + end) / 2) - 1.16 / 2),
            "start_frame": lambda start, end, fps: int(max(0, ((start + end) / 2) - 1.16 / 2) * fps),
            "num_frames": lambda start, end, fps: int(1.16 * fps)
        },
        {
            "key": "lip",
            "bbox_fn": get_face_lip_box,
            "crop_size": 96,
            "clip_duration": None,
            "meta_suffix": "lip_96x96_wordlength",
            "audio_length": lambda start, end: (end - start) + 0.02,
            "clip_start": lambda start, end: max(0, ((start + end) / 2) - (end - start) / 2 - 0.01),
            "start_frame": lambda start, end, fps: int(max(0, ((start + end) / 2) - (end - start) / 2 - 0.01) * fps),
            "num_frames": lambda start, end, fps: int(((end - start) + 0.02) * fps)
        }
    ]
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            error_msg = f"Cannot read FPS from {video_path}"
            log_error(error_msg)
            raise RuntimeError(f"❌ {error_msg}")
    except Exception as e:
        error_msg = f"Failed to open video {video_path}: {e}"
        log_error(error_msg)
        raise
    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        for word_info in all_words:
            word_counter += 1
            word = word_info.get("text", "").strip().lower()
            start = word_info.get("start")
            end = word_info.get("end")
            conf = word_info.get("confidence", 1.0)
            print(f"Processing {word} ({word_counter}/{total_words})")
            idx = None
            all_success = True
            temp_outputs = []
            temp_files = []
            for ds in datasets:
                out_root = output_roots[ds["key"]] / word
                out_root.mkdir(parents=True, exist_ok=True)
                if idx is None:
                    idx = get_next_index(out_root)
                video_out = out_root / f"{idx}.mp4"
                audio_out = out_root / f"{idx}.m4a"
                meta_out = out_root / f"{idx}.txt"
                clip_start = ds["clip_start"](start, end)
                audio_length = ds["audio_length"](start, end)
                start_frame = ds["start_frame"](start, end, fps)
                num_frames = ds["num_frames"](start, end, fps)
                try:
                    frames = process_clip(
                        cap, face_mesh, start_frame, num_frames, ds["bbox_fn"], ds["crop_size"]
                    )
                    if len(frames) == 0:
                        log_error(f"No valid frames for word '{word}' in dataset '{ds['key']}'")
                        all_success = False
                        break
                    vid_success = save_video(frames, video_out, fps)
                    aud_success = extract_audio(video_path, audio_out, clip_start, audio_length)
                    mux_success = False
                    if vid_success and aud_success:
                        mux_success = mux_video_audio(video_out, audio_out, video_out)
                    if not (vid_success and aud_success and mux_success):
                        all_success = False
                        break
                    temp_files.extend([video_out, audio_out])
                    temp_outputs.append((meta_out, f"""
                        Spoken word: {word}
                        Starttime of utterance in seconds: {start:.3f}
                        Endtime of utterance in seconds: {end:.3f}
                        Duration of utterance in seconds: {audio_length:.3f}
                        Confidence: {conf:.3f}
                        Original filename: {original_filename}
                        Dataset: {ds["meta_suffix"]}
                    """))
                except Exception as e:
                    log_error(f"Unexpected error for word '{word}' in dataset '{ds['key']}']: {e}")
                    all_success = False
                    break
            if all_success and temp_outputs:
                for meta_out, meta_text in temp_outputs:
                    save_meta(meta_out, textwrap.dedent(meta_text))
                print(f"✅ Saved {word} in all datasets with index {idx}")
                successful_crops += 1
            else:
                for fpath in temp_files:
                    if valid_file(fpath):
                        os.remove(fpath)
                failed_datasets = []
                for ds in datasets:
                    video_out = output_roots[ds["key"]] / word / f"{idx}.mp4"
                    audio_out = output_roots[ds["key"]] / word / f"{idx}.m4a"
                    if not (valid_file(video_out) and valid_file(audio_out)):
                        failed_datasets.append(ds["key"])
                failed_ds_str = ", ".join(failed_datasets) if failed_datasets else "unknown"
                log_error(f"Skipped word '{word}' (start: {start:.3f}s, end: {end:.3f}s): datasets failed -> {failed_ds_str}")
                print(f"❌ Skipped {word}, datasets incompatible -> failed datasets: {failed_ds_str}")
    cap.release()
    summary_msg = f"Summary for {video_path}: Successfully saved {successful_crops} crops out of {total_words} attempted."
    print(f"\n{summary_msg}")
    log_error(summary_msg)
