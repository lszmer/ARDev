#!/usr/bin/env python3
"""Brief YOLO .pt -> ONNX exporter using Ultralytics (ONNX backend).

Usage:
  python MLModels/export_yolo_to_onnx.py --pt MLModels/yolov12n.pt --subdir YOLO --imgsz 640
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pt", type=str, required=True, help="Path to YOLO .pt weights")
    p.add_argument("--subdir", type=str, default="YOLO", help="Assets/Models subfolder")
    p.add_argument("--imgsz", type=int, default=640, help="Input size (HxW), default 640")
    args = p.parse_args()

    weights = Path(args.pt)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    # Ultralytics handles ONNX export internally (uses ONNX toolchain). Fixed 1x3xHxW.
    try:
        onnx_out = YOLO(str(weights)).export(
            format="onnx",
            imgsz=args.imgsz,
            opset=12,
            dynamic=False,
            half=False,
            device="cpu",
            verbose=False,
            nms=False,
            optimize=False,
            batch=1,
        )
    except AttributeError as exc:
        raise RuntimeError(
            "Ultralytics export failed during forward (possible attention impl/version mismatch). "
            "Try: pip install -U ultralytics torch; or use a different ultralytics version that matches the weights.\n"
            f"Original error: {exc}"
        )

    # Move to Assets/Models/<subdir>/<name>.onnx
    repo_root = Path(__file__).resolve().parents[1]
    target_dir = repo_root / "Assets" / "Models" / args.subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / (weights.stem + ".onnx")
    shutil.move(str(onnx_out), str(target_path))
    print(f"Saved: {target_path}")


if __name__ == "__main__":
    main()


