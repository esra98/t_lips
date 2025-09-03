import json
from pathlib import Path
from whisper_timestamped import load_model, transcribe

MODEL = load_model("small")  

def generate_timestamp_json(video_path, output_dir="timestamps", language="turkish"):
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / f"{video_path.stem}_ts.json"

    print(f"🎙️ Transcribing {video_path.name} ...")
    result = transcribe(MODEL, str(video_path), language=language)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✅ Timestamp JSON saved to {json_path}")
    return json_path
