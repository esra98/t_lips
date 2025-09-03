import cv2
import mediapipe as mp
from pathlib import Path
import subprocess
import textwrap
import json

mp_face_mesh = mp.solutions.face_mesh


def get_next_index(word_dir: Path) -> int:
    """Return the next available integer ID in this word folder."""
    if not word_dir.exists():
        return 1
    existing = [int(f.stem) for f in word_dir.glob("*.mp4") if f.stem.isdigit()]
    return max(existing, default=0) + 1


def crop_clips(
    video_path,
    json_path=None,
    output_root="videos/dataset",
    clip_duration=1.16,
    output_size=112,
    original_filename="unknown",
    titlelist_number="0000"
):
    """
    Process a video + its timestamp JSON and generate cropped LRW-style clips.
    """

    video_path = Path(video_path)
    if json_path is None:
        json_path = video_path.with_name(video_path.stem + "_ts.json")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

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
        max_num_faces=1,
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

                if not word or start is None or end is None or end <= start:
                    continue

                # --- Compute clip start ---
                word_center = (start + end) / 2
                clip_start = max(0, word_center - clip_duration / 2)

                # --- Jump to start frame ---
                start_frame = int(clip_start * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

                ret, frame = cap.read()
                if not ret:
                    continue
                h, w, _ = frame.shape

                # --- Mouth center detection ---
                results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    xs = [p.x * w for idx, p in enumerate(landmarks) if 61 <= idx <= 88]
                    ys = [p.y * h for idx, p in enumerate(landmarks) if 61 <= idx <= 88]
                    mouth_center_x = int(sum(xs) / len(xs))
                    mouth_center_y = int(sum(ys) / len(ys))
                else:
                    mouth_center_x, mouth_center_y = w // 2, h // 2

                # --- Fixed 112x112 crop ---
                half_size = output_size // 2
                x_min = max(0, mouth_center_x - half_size)
                y_min = max(0, mouth_center_y - half_size)
                x_max = min(w, mouth_center_x + half_size)
                y_max = min(h, mouth_center_y + half_size)

                crop_filter = f"crop={x_max-x_min}:{y_max-y_min}:{x_min}:{y_min}"

                # --- Output paths ---
                word_dir = output_root / word
                word_dir.mkdir(parents=True, exist_ok=True)

                idx = get_next_index(word_dir)

                video_out = word_dir / f"{idx}.mp4"
                audio_out = word_dir / f"{idx}.m4a"
                meta_out = word_dir / f"{idx}.txt"

                # --- Save cropped video ---
                cmd_video = [
                    "ffmpeg", "-y",
                    "-ss", str(clip_start),
                    "-i", str(video_path),
                    "-t", str(clip_duration),
                    "-vf", crop_filter,
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    str(video_out)
                ]
                subprocess.run(cmd_video, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                # --- Save audio ---
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

                # --- Save metadata ---
                meta_text = textwrap.dedent(f"""
                    Spoken word: {word}
                    Starttime of utterance in seconds: {start:.3f}
                    Endtime of utterance in seconds: {end:.3f}
                    Duration of utterance in seconds: {end - start:.3f}
                    Confidence: {conf:.3f}
                    Original filename: {original_filename}
                    Corresponding number in titlelist.txt: {titlelist_number}
                """).strip()

                with open(meta_out, "w", encoding="utf-8") as f:
                    f.write(meta_text + "\n")

                print(f"✅ Saved {word} -> {video_out.name}, {audio_out.name}, {meta_out.name}")

    cap.release()
