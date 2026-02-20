import csv
from pathlib import Path
from utils.generate_timestamp import generate_timestamp_json
from utils.lrw_cropper import crop_clips
from utils.lrs_cropper import crop_sentences


def parse_mp4_csv(csv_file):
    mp4_files = []
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            for item in row:
                item = item.strip()
                if item.endswith(".mp4"):
                    mp4_files.append(item)
    return mp4_files


def log_completion(log_file, video_name):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(video_name + "\n")


def process_videos(csv_file, log_file, mode="word"):
    videos = parse_mp4_csv(csv_file)

    for video in videos:
        video_path = Path(video)
        json_path = Path("timestamps") / f"{video_path.stem}_ts.json"

        print(f"Processing {video_path.name}...")

        try:
            if not json_path.exists():
                generate_timestamp_json(video_path)

            if mode == "word":
                crop_clips(
                    video_path=video_path,
                    json_path=json_path,
                    output_roots = {
                        "standart": Path("videos/dataset/standart"),
                        "standart_lip": Path("videos/dataset/standart_lip"),
                        "lip": Path("videos/dataset/lip")
                    },
                    original_filename=video_path.stem
                )
            elif mode == "sentence":
                crop_sentences(
                    video_path=video_path,
                    json_path=json_path,
                    output_dir=Path("videos/dataset/lrs2"),
                    split="pretrain",
                    crop_type="face"
                )

            log_completion(log_file, video)

        except Exception as e:
            print(f"❌ Error processing {video}: {e}")
            break


if __name__ == "__main__":
    # To run word-level: process_videos("videos/list.csv", "process_log.txt", mode="word")
    # To run sentence-level: process_videos("videos/list.csv", "process_log.txt", mode="sentence")
    process_videos("videos/list.csv", "process_log.txt", mode="word")
    process_videos("videos/list.csv", "process_log.txt", mode="sentence")
