import cv2
import mediapipe as mp
from pathlib import Path
import subprocess
import textwrap
import json

mp_face_mesh = mp.solutions.face_mesh

# FACEMESH_LIPS connections from MediaPipe documentation
FACEMESH_LIPS = frozenset([
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375),
    (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
    (37, 0), (0, 267), (267, 269), (269, 270), (270, 409), (409, 291),
    (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
    (14, 317), (317, 402), (402, 318), (318, 324),
    (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
    (82, 13), (13, 312), (312, 311), (311, 310),
    (310, 415), (415, 308)
])
# Extract all unique lip landmark indices from FACEMESH_LIPS
LIP_INDICES = sorted(set(idx for pair in FACEMESH_LIPS for idx in pair))

def load_word_list(filepath="frequency_list.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())

ALLOWED_WORDS = load_word_list()

def get_next_index(word_dir: Path) -> int:
    if not word_dir.exists():
        return 1
    existing = [int(f.stem) for f in word_dir.glob("*.mp4") if f.stem.isdigit()]
    return max(existing, default=0) + 1

def get_bbox(landmarks, w, h, indices=None, margin=0):
    """Get tight bounding box for given indices in landmarks (or all if indices=None)."""
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

def crop_and_resize(frame, x_min, y_min, x_max, y_max, output_size):
    crop = frame[y_min:y_max, x_min:x_max]
    # Fill with black if crop is invalid or zero-size
    if crop.size == 0 or (x_max-x_min) < 1 or (y_max-y_min) < 1:
        crop = cv2.resize(frame, (output_size, output_size), interpolation=cv2.INTER_AREA)
    else:
        crop = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return crop

def save_video(frames, out_path, fps):
    if len(frames) == 0:
        return
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()

def crop_clips(
    video_path,
    json_path=None,
    output_roots=None,  # {"standart": ..., "standart_lip": ..., "lip": ...}
    original_filename="unknown",
    titlelist_number="0000"
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
        print(f"❌ Missing JSON: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise RuntimeError("❌ Cannot read FPS")

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        for segment in data.get("segments", []):
            for word_info in segment.get("words", []):
                word = word_info.get("text", "").strip().lower()
                start = word_info.get("start")
                end = word_info.get("end")
                conf = word_info.get("confidence", 1.0)

                # Allow only words in frequency list
                if not word or word not in ALLOWED_WORDS:
                    continue
                if start is None or end is None or end <= start:
                    continue

                word_center = (start + end) / 2

                # --- 1. Dataset: Face tight crop and resize to 256x256, 1.16s ---
                clip_duration = 1.16
                clip_start = max(0, word_center - clip_duration / 2)
                start_frame = int(clip_start * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                ret, frame = cap.read()
                if not ret:
                    continue
                h, w, _ = frame.shape

                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if not results.multi_face_landmarks or len(results.multi_face_landmarks) != 1:
                    # Skip if not exactly one face
                    continue
                landmarks = results.multi_face_landmarks[0].landmark

                # --- Face crop ---
                x_min, y_min, x_max, y_max = get_bbox(landmarks, w, h, indices=None, margin=5)
                out_root = output_roots["standart"] / word
                out_root.mkdir(parents=True, exist_ok=True)
                idx = get_next_index(out_root)
                video_out = out_root / f"{idx}.mp4"
                audio_out = out_root / f"{idx}.m4a"
                meta_out = out_root / f"{idx}.txt"

                frames = []
                for i in range(int(clip_duration * fps)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
                    ret, frame_clip = cap.read()
                    if not ret:
                        break
                    h, w, _ = frame_clip.shape
                    results_clip = face_mesh.process(cv2.cvtColor(frame_clip, cv2.COLOR_BGR2RGB))
                    if not results_clip.multi_face_landmarks or len(results_clip.multi_face_landmarks) != 1:
                        continue
                    landmarks_clip = results_clip.multi_face_landmarks[0].landmark
                    x_min, y_min, x_max, y_max = get_bbox(landmarks_clip, w, h, indices=None, margin=5)
                    crop = crop_and_resize(frame_clip, x_min, y_min, x_max, y_max, 256)
                    frames.append(crop)
                save_video(frames, video_out, fps)
                # Extract audio
                cmd_audio = [
                    "ffmpeg", "-y",
                    "-ss", str(clip_start),
                    "-i", str(video_path),
                    "-t", str(clip_duration),
                    "-vn",
                    "-c:a", "aac",
                    str(audio_out)
                ]
                subprocess.run(cmd_audio, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                meta_text = textwrap.dedent(f"""
                    Spoken word: {word}
                    Starttime of utterance in seconds: {start:.3f}
                    Endtime of utterance in seconds: {end:.3f}
                    Duration of utterance in seconds: {end - start:.3f}
                    Confidence: {conf:.3f}
                    Original filename: {original_filename}
                    Corresponding number in titlelist.txt: {titlelist_number}
                    Dataset: standart_tight_256x256_1.16s
                """).strip()
                with open(meta_out, "w", encoding="utf-8") as f:
                    f.write(meta_text + "\n")

                # --- 2. Dataset: Lips tight crop and resize to 96x96, 1.16s (standart_lip) ---
                frames = []
                for i in range(int(clip_duration * fps)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + i)
                    ret, frame_clip = cap.read()
                    if not ret:
                        break
                    h, w, _ = frame_clip.shape
                    results_clip = face_mesh.process(cv2.cvtColor(frame_clip, cv2.COLOR_BGR2RGB))
                    if not results_clip.multi_face_landmarks or len(results_clip.multi_face_landmarks) != 1:
                        continue
                    landmarks_clip = results_clip.multi_face_landmarks[0].landmark
                    x_min, y_min, x_max, y_max = get_bbox(landmarks_clip, w, h, indices=LIP_INDICES, margin=2)
                    crop = crop_and_resize(frame_clip, x_min, y_min, x_max, y_max, 96)
                    frames.append(crop)
                out_root = output_roots["standart_lip"] / word
                out_root.mkdir(parents=True, exist_ok=True)
                idx = get_next_index(out_root)
                video_out = out_root / f"{idx}.mp4"
                audio_out = out_root / f"{idx}.m4a"
                meta_out = out_root / f"{idx}.txt"
                save_video(frames, video_out, fps)
                # Extract audio
                cmd_audio = [
                    "ffmpeg", "-y",
                    "-ss", str(clip_start),
                    "-i", str(video_path),
                    "-t", str(clip_duration),
                    "-vn",
                    "-c:a", "aac",
                    str(audio_out)
                ]
                subprocess.run(cmd_audio, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                meta_text = textwrap.dedent(f"""
                    Spoken word: {word}
                    Starttime of utterance in seconds: {start:.3f}
                    Endtime of utterance in seconds: {end:.3f}
                    Duration of utterance in seconds: {end - start:.3f}
                    Confidence: {conf:.3f}
                    Original filename: {original_filename}
                    Corresponding number in titlelist.txt: {titlelist_number}
                    Dataset: standart_lip_tight_96x96_1.16s
                """).strip()
                with open(meta_out, "w", encoding="utf-8") as f:
                    f.write(meta_text + "\n")

                # --- 3. Dataset: Lips tight crop, word duration (lip) ---
                margin_sec = 0.01  # Adds margin to start and end
                clip_duration_word = (end - start) + 2 * margin_sec
                clip_start_word = max(0, word_center - (end - start)/2 - margin_sec)
                start_frame_word = int(clip_start_word * fps)
                frames = []
                for i in range(int(clip_duration_word * fps)):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_word + i)
                    ret, frame_clip = cap.read()
                    if not ret:
                        break
                    h, w, _ = frame_clip.shape
                    results_clip = face_mesh.process(cv2.cvtColor(frame_clip, cv2.COLOR_BGR2RGB))
                    if not results_clip.multi_face_landmarks or len(results_clip.multi_face_landmarks) != 1:
                        continue
                    landmarks_clip = results_clip.multi_face_landmarks[0].landmark
                    x_min, y_min, x_max, y_max = get_bbox(landmarks_clip, w, h, indices=LIP_INDICES, margin=2)
                    crop = crop_and_resize(frame_clip, x_min, y_min, x_max, y_max, 96)
                    frames.append(crop)
                out_root = output_roots["lip"] / word
                out_root.mkdir(parents=True, exist_ok=True)
                idx = get_next_index(out_root)
                video_out = out_root / f"{idx}.mp4"
                audio_out = out_root / f"{idx}.m4a"
                meta_out = out_root / f"{idx}.txt"
                save_video(frames, video_out, fps)
                # Extract audio
                cmd_audio = [
                    "ffmpeg", "-y",
                    "-ss", str(clip_start_word),
                    "-i", str(video_path),
                    "-t", str(clip_duration_word),
                    "-vn",
                    "-c:a", "aac",
                    str(audio_out)
                ]
                subprocess.run(cmd_audio, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                meta_text = textwrap.dedent(f"""
                    Spoken word: {word}
                    Starttime of utterance in seconds: {start:.3f}
                    Endtime of utterance in seconds: {end:.3f}
                    Duration of utterance in seconds: {clip_duration_word:.3f}
                    Confidence: {conf:.3f}
                    Original filename: {original_filename}
                    Corresponding number in titlelist.txt: {titlelist_number}
                    Dataset: lip_tight_96x96_wordlength
                """).strip()
                with open(meta_out, "w", encoding="utf-8") as f:
                    f.write(meta_text + "\n")

                print(f"✅ Saved {word} in all datasets")

    cap.release()