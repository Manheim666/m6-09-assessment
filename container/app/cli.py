"""Standardised CLI for the cat-detector container.
Two subcommands:
    info     -> prints /app/STUDENT.json to stdout
    predict  -> reads /data/input, writes /data/output/predictions.csv
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
STUDENT_JSON = Path("/app/STUDENT.json")
MODEL_PATH = Path("/app/models/best.onnx")
INPUT_DIR = Path("/data/input")
OUTPUT_DIR = Path("/data/output")
OUTPUT_CSV = OUTPUT_DIR / "predictions.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
CSV_HEADER = ["image_path", "xmin", "ymin", "xmax", "ymax", "confidence", "class"]
def cmd_info() -> int:
    with STUDENT_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    json.dump(data, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0
def iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path
def cmd_predict() -> int:
    from app.detector import CatDetector
    if not INPUT_DIR.exists():
        print(f"Input directory missing: {INPUT_DIR}", file=sys.stderr)
        return 2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    detector = CatDetector(MODEL_PATH, imgsz=1024, conf=0.25, class_names=("cat",))
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for img_path in iter_images(INPUT_DIR):
            rel = img_path.relative_to(INPUT_DIR).as_posix()
            try:
                boxes = detector.predict(img_path)
            except Exception as exc:
                print(f"WARN: failed on {rel}: {exc}", file=sys.stderr)
                boxes = []
            if not boxes:
                writer.writerow([rel, "", "", "", "", "", ""])
                continue
            for b in boxes:
                writer.writerow([
                    rel,
                    f"{b['xmin']:.2f}",
                    f"{b['ymin']:.2f}",
                    f"{b['xmax']:.2f}",
                    f"{b['ymax']:.2f}",
                    f"{b['confidence']:.4f}",
                    b["class"],
                ])
    return 0
def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: cli.py {info|predict}", file=sys.stderr)
        return 2
    cmd = argv[1]
    if cmd == "info":
        return cmd_info()
    if cmd == "predict":
        return cmd_predict()
    print(f"unknown subcommand: {cmd}", file=sys.stderr)
    return 2
if __name__ == "__main__":
    sys.exit(main(sys.argv))
