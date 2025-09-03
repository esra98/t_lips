import subprocess
from pathlib import Path
from collections import defaultdict
import json

def process_video(video_path, timestamps_dir="timestamps", output_dir="words"):
    """
    Crop word-level clips from a video based on its timestamp JSON.
    """
    video_path = Path(video_path)
    video_name = video_path.stem
    timestamps_dir = Path(timestamps_dir)
    output_dir = Path(output_dir)
    timestamps_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    json_path = timestamps_dir / f"{video_name}_timestamps.json"
    if not json_path.exists():
        print(f"❌ No timestamp file for {video_name}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    word_counts = defaultdict(int)

    for segment in data.get("segments", []):
        for word_info in segment.get("words", []):
            word = word_info.get("text", "").strip().lower()
            start = word_info.get("start")
            end = word_info.get("end")
            if not word or start is None or end is None or end <= start:
                continue
            duration = end - start
            word_base = "".join(c for c in word if c.isalnum() or c in "-_")
            word_counts[word_base] += 1
            filename = f"{video_name}_{word_base}_{word_counts[word_base]}.mp4"
            filepath = output_dir / filename
            crop_by_time(video_path, start, duration, filepath)

def crop_by_time(video_path, start, duration, output_path):
    """Crop a video segment between start and start+duration."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        "-c:v", "h264_nvenc",  # GPU encoder; change if needed
        "-c:a", "aac",
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"▶️ Cropped segment saved to {output_path}")

def crop_region(video_path, x, y, w, h, output_path):
    """Crop a specific rectangular area from the video."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-filter:v", f"crop={w}:{h}:{x}:{y}",
        "-c:v", "h264_nvenc",
        "-c:a", "aac",
        str(output_path)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"▶️ Cropped region saved to {output_path}")
