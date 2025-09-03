import cv2
import mediapipe as mp
from pathlib import Path
import subprocess
import textwrap

# ---- Config ----
input_video = "videos/original/test-1.mp4"
original_filename = "Mieterhöhungen u. unsoziale Modernisierungen bei GWH verhindern - 12.12.2019 - 28. Plenarsitzung"
titlelist_number = "0140"

word_info = {
    "text": "tam",
    "start": 0.66,
    "end": 1.46,
    "confidence": 0.76
}
clip_duration = 1.16  # seconds
output_size = 112  # LRW-style fixed crop
output_dir = Path(f"videos/dataset/{word_info['text']}")
output_dir.mkdir(parents=True, exist_ok=True)

video_out = output_dir / "1.mp4"
audio_out = output_dir / "1.m4a"
meta_out = output_dir / "1.txt"

# ---- Compute clip start ----
word_center = (word_info["start"] + word_info["end"]) / 2
clip_start = max(0, word_center - clip_duration / 2)

# ---- Open video ----
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    raise RuntimeError("Cannot read FPS from video")

start_frame = int(clip_start * fps)
end_frame = start_frame + int(clip_duration * fps)
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# ---- Mediapipe setup ----
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ---- Get mouth center from first frame ----
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame")
h, w, _ = frame.shape

results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
if results.multi_face_landmarks:
    landmarks = results.multi_face_landmarks[0].landmark
    xs = [p.x * w for idx, p in enumerate(landmarks) if 61 <= idx <= 88]
    ys = [p.y * h for idx, p in enumerate(landmarks) if 61 <= idx <= 88]
    mouth_center_x = int(sum(xs) / len(xs))
    mouth_center_y = int(sum(ys) / len(ys))
else:
    mouth_center_x, mouth_center_y = w // 2, h // 2

# ---- Compute crop box ----
half_size = output_size // 2
x_min = max(0, mouth_center_x - half_size)
y_min = max(0, mouth_center_y - half_size)
x_max = min(w, mouth_center_x + half_size)
y_max = min(h, mouth_center_y + half_size)

# ---- Save cropped video ----
crop_filter = f"crop={x_max-x_min}:{y_max-y_min}:{x_min}:{y_min}"
cmd_video = [
    "ffmpeg", "-y",
    "-ss", str(clip_start),
    "-i", input_video,
    "-t", str(clip_duration),
    "-vf", crop_filter,
    "-c:v", "libx264",
    "-c:a", "aac",
    "-movflags", "+faststart",
    str(video_out)
]
print("Running video command:", " ".join(cmd_video))
subprocess.run(cmd_video, check=True)

# ---- Save audio only (M4A) ----
cmd_audio = [
    "ffmpeg", "-y",
    "-ss", str(clip_start),
    "-i", input_video,
    "-t", str(clip_duration),
    "-vn",  # no video
    "-c:a", "aac",
    str(audio_out)
]
print("Running audio command:", " ".join(cmd_audio))
subprocess.run(cmd_audio, check=True)

# ---- Save metadata ----
meta_text = textwrap.dedent(f"""
    Spoken word: {word_info['text']}
    Starttime of utterance in seconds: {word_info['start']:.3f}
    Endtime of utterance in seconds: {word_info['end']:.3f}
    Duration of utterance in seconds: {word_info['end'] - word_info['start']:.3f}
    Original filename: {original_filename}
    Corresponding number in titlelist.txt: {titlelist_number}
""").strip()

with open(meta_out, "w", encoding="utf-8") as f:
    f.write(meta_text + "\n")

print(f"▶️ Saved video: {video_out}")
print(f"▶️ Saved audio: {audio_out}")
print(f"▶️ Saved metadata: {meta_out}")
