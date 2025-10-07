#!/usr/bin/env python3
"""
Evaluate a YOLO-seg .pt model (e.g., yolo11n-seg.pt) on up to 100 sample images
and save visualizations with colored segmentation overlays and bounding boxes.

This mirrors eval_yolo_seg_onnx.py but uses Ultralytics directly to help
diagnose potential ONNX conversion issues.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. pip install opencv-python")
    sys.exit(1)

try:
    from ultralytics import YOLO  # type: ignore
    import torch  # type: ignore
except Exception:
    print("ERROR: ultralytics and torch are required. pip install ultralytics torch torchvision")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images" / "indoor objects" / "train" / "images"
RUNS_DIR = ROOT / "runs" / "eval"


def load_labels() -> List[str]:
    candidates = [
        ROOT / "yolo_classes.txt",
        ROOT / "yolo_classes 2.txt",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
    return [f"class_{i}" for i in range(80)]


def _select_torch_device() -> str:
    try:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        if torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


def color_for_class(c: int) -> Tuple[int, int, int]:
    rng = np.random.default_rng(seed=c * 9973)
    col = rng.integers(low=64, high=255, size=3, dtype=np.int32)
    return int(col[0]), int(col[1]), int(col[2])


def run_pt_model_on_images(model_path: Path, images: List[Path], output_dir: Path, labels: List[str], imgsz: int = 640) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = []

    device = _select_torch_device()
    print(f"\n=== Model: {model_path.name} (pt seg) ===")
    print(f"Using torch device: {device}")

    t0 = time.time()
    model = YOLO(str(model_path))
    load_ms = (time.time() - t0) * 1000
    print(f"Model loaded in {load_ms:.1f} ms")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARN: could not read {img_path}")
            continue

        t1 = time.time()
        results = model.predict(source=img, conf=0.25, iou=0.6, imgsz=imgsz, device=device, verbose=False)
        inf_ms = (time.time() - t1) * 1000
        timings.append(inf_ms)

        overlay = img.copy()
        if results and len(results) > 0:
            r = results[0]

            # Draw masks if available
            if hasattr(r, 'masks') and r.masks is not None and hasattr(r.masks, 'data') and r.masks.data is not None:
                # r.masks.data: (N, H, W) boolean/float mask in image size
                masks = r.masks.data
                if hasattr(masks, 'cpu'):
                    masks = masks.cpu().numpy()
                masks = masks.astype(bool)
            else:
                masks = None

            # Boxes
            if hasattr(r, 'boxes') and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.zeros((0, 4), dtype=np.float32)
                cls = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, 'cls') else np.zeros((0,), dtype=int)
                confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.zeros((0,), dtype=np.float32)
            else:
                xyxy = np.zeros((0, 4), dtype=np.float32)
                cls = np.zeros((0,), dtype=int)
                confs = np.zeros((0,), dtype=np.float32)

            # Blend masks first (per instance color)
            if masks is not None and len(masks) == len(xyxy):
                alpha = 0.5
                h0, w0 = overlay.shape[:2]
                for i, mask in enumerate(masks):
                    color = color_for_class(int(cls[i]) if i < len(cls) else 0)
                    m = mask
                    # Ensure 2D boolean mask at image size
                    if m.ndim == 3 and m.shape[0] == 1:
                        m = m[0]
                    if m.shape[0] != h0 or m.shape[1] != w0:
                        m = cv2.resize(m.astype(np.float32), (w0, h0), interpolation=cv2.INTER_NEAREST)
                    m = (m > 0.5)
                    if np.any(m):
                        overlay[m] = (overlay[m] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)

            # Draw boxes and labels
            for (x1, y1, x2, y2), c, sc in zip(xyxy.astype(int), cls, confs):
                color = color_for_class(int(c))
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label = labels[c] if 0 <= c < len(labels) else str(c)
                cv2.putText(overlay, f"{label}:{sc:.2f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out_file = output_dir / img_path.name
        cv2.imwrite(str(out_file), overlay)

    # Stats
    if timings:
        times = np.array(timings)
        summary = {
            "model": model_path.name,
            "images": len(timings),
            "load_ms": round(load_ms, 2),
            "avg_ms": round(float(times.mean()), 2),
            "p50_ms": round(float(np.percentile(times, 50)), 2),
            "p90_ms": round(float(np.percentile(times, 90)), 2),
            "min_ms": round(float(times.min()), 2),
            "max_ms": round(float(times.max()), 2),
            "fps": round(1000.0 / float(times.mean()), 2) if times.mean() > 0 else 0.0,
        }
    else:
        summary = {
            "model": model_path.name,
            "images": 0,
            "load_ms": round(load_ms, 2),
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "fps": 0.0,
        }
    print(f"Summary: {summary}")
    return summary


def discover_model() -> Path:
    for name in ["yolo11n-seg.pt", "yolov12n-seg.pt", "Yolo11n-seg.pt", "yolov12n-seg.pt"]:
        p = ROOT / name
        if p.exists():
            return p
    print("Seg .pt model not found in MLModels/. Place yolo11n-seg.pt there and re-run.")
    sys.exit(1)


def main():
    labels = load_labels()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in IMAGES_DIR.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        print(f"No images found in: {IMAGES_DIR}")
        sys.exit(1)
    images = images[:100]
    print(f"Found {len(images)} images. Running up to 100.")

    model_path = discover_model()
    out_dir = RUNS_DIR / (model_path.stem + "_pt")
    _ = run_pt_model_on_images(model_path, images, out_dir, labels, imgsz=640)
    print(f"\nAll done. Annotated outputs in: {out_dir}")


if __name__ == "__main__":
    main()


