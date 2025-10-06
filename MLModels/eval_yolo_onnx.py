#!/usr/bin/env python3
"""
Evaluate YOLO ONNX models on up to 100 images in images/train.

- Loads one or more ONNX models from MLModels (e.g., yolo12n.onnx, yolo12n-int8.onnx, yolov10_models/*.onnx)
- Runs inference with ONNX Runtime
- Performs simple postprocessing (argmax over classes, score filtering, NMS)
- Draws bounding boxes and labels, saves annotated images into MLModels/runs/eval/<model_name>/
- Records timing stats per model to MLModels/runs/eval/summary.csv

Assumptions:
- Model outputs a tensor shaped (1, 84, N) or (1, N, 84) where first 4 values are (cx, cy, w, h)
  in either normalized model-input space or pixels; remaining 80 are class logits/scores.
- If your model differs, adjust postprocess_yolo() accordingly.
"""

import os
import sys
import time
import csv
import glob
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is required. pip install onnxruntime")
    sys.exit(1)

# Validate onnxruntime API availability (some installs lack InferenceSession, e.g., wrong wheel)
if not hasattr(ort, "InferenceSession"):
    print(
        "ERROR: onnxruntime is installed but missing InferenceSession.\n"
        "On Apple Silicon/macOS, install the correct wheel:\n"
        "  pip install --upgrade onnxruntime-silicon\n"
        "Alternatively (Intel mac):\n"
        "  pip install --upgrade onnxruntime\n"
    )
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. pip install opencv-python")
    sys.exit(1)

# Optional: PyTorch/Ultralytics for evaluating .pt models locally
_ULTRALYTICS_AVAILABLE = True
try:
    from ultralytics import YOLO  # type: ignore
    import torch  # type: ignore
except Exception:
    _ULTRALYTICS_AVAILABLE = False


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images" / "indoor objects" / "train" / "images"
RUNS_DIR = ROOT / "runs" / "eval"
LABELS_CANDIDATES = [
    ROOT / "yolo_classes.txt",
    ROOT / "yolo_classes 2.txt",
]


def load_labels() -> List[str]:
    for p in LABELS_CANDIDATES:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                labels = [line.strip() for line in f if line.strip()]
            return labels
    # Fallback 80 COCO-like dummy labels if none found
    return [f"class_{i}" for i in range(80)]


def letterbox(image: np.ndarray, new_shape: Tuple[int, int], color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Resize image to a target shape with unchanged aspect ratio using padding.
    Returns padded image, scale, and padding (dw, dh)."""
    h, w = image.shape[:2]
    new_w, new_h = new_shape
    scale = min(new_w / w, new_h / h)
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
    pad_w = new_w - resized.shape[1]
    pad_h = new_h - resized.shape[0]
    dw = pad_w // 2
    dh = pad_h // 2
    out = cv2.copyMakeBorder(resized, dh, pad_h - dh, dw, pad_w - dw, cv2.BORDER_CONSTANT, value=color)
    return out, scale, (dw, dh)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    """Non-Maximum Suppression on corner boxes [x1,y1,x2,y2]. Returns indices of kept boxes."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1).clip(min=0) * (y2 - y1).clip(min=0)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = (xx2 - xx1).clip(min=0)
        h = (yy2 - yy1).clip(min=0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def postprocess_yolo(outputs: List[np.ndarray], img_shape_hw: Tuple[int, int], input_shape_wh: Tuple[int, int], scale: float, pad: Tuple[int, int], score_thr=0.25, iou_thr=0.6) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
    """
    Convert YOLO outputs to detections.
    Returns list of (class_id, score, (x1, y1, x2, y2)) in original image coordinates.
    """
    h0, w0 = img_shape_hw
    in_w, in_h = input_shape_wh

    # Pick the largest output tensor
    out = max(outputs, key=lambda x: x.size)
    out = np.squeeze(out)

    # Expect (N, 84). If (84, N), transpose.
    if out.ndim == 2:
        if out.shape[0] == 84:
            out = out.transpose(1, 0)
    elif out.ndim == 3:
        # (1,84,N) or (1,N,84)
        if out.shape[1] == 84:
            out = out[0].transpose(1, 0)
        else:
            out = out[0]
    else:
        raise RuntimeError(f"Unexpected output shape: {out.shape}")

    if out.shape[1] < 5:
        raise RuntimeError(f"Unexpected per-detection size: {out.shape}")

    boxes_cxcywh = out[:, :4].astype(np.float32)
    scores_all = out[:, 4:].astype(np.float32)
    # class score = max over classes, class id = argmax
    class_ids = np.argmax(scores_all, axis=1)
    class_scores = scores_all[np.arange(scores_all.shape[0]), class_ids]

    # Filter by score
    keep = class_scores >= score_thr
    if not np.any(keep):
        return []
    boxes_cxcywh = boxes_cxcywh[keep]
    class_ids = class_ids[keep]
    class_scores = class_scores[keep]

    # Determine if coords are normalized (0..1) or already in pixels
    # Heuristic: if most widths/heights <= 1.5, treat as normalized.
    normalized = np.mean(boxes_cxcywh[:, 2:4] <= 1.5) > 0.8

    if normalized:
        # Scale to model input pixels
        boxes_cxcywh[:, 0] *= in_w
        boxes_cxcywh[:, 1] *= in_h
        boxes_cxcywh[:, 2] *= in_w
        boxes_cxcywh[:, 3] *= in_h

    # Convert centers to corners in input space
    cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_in = np.stack([x1, y1, x2, y2], axis=1)

    # Map from input (after letterbox) back to original image coords
    dw, dh = pad
    # Remove padding, then divide by scale
    boxes_in[:, [0, 2]] -= dw
    boxes_in[:, [1, 3]] -= dh
    boxes_in[:, [0, 2]] /= (scale + 1e-9)
    boxes_in[:, [1, 3]] /= (scale + 1e-9)

    # Clip to image
    boxes_in[:, 0::2] = boxes_in[:, 0::2].clip(0, w0 - 1)
    boxes_in[:, 1::2] = boxes_in[:, 1::2].clip(0, h0 - 1)

    # NMS per class
    results: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
    for c in np.unique(class_ids):
        inds = np.where(class_ids == c)[0]
        keep_inds = nms(boxes_in[inds], class_scores[inds], iou_thr)
        for k in keep_inds:
            bi = inds[k]
            box = boxes_in[bi].astype(int)
            results.append((int(c), float(class_scores[bi]), (int(box[0]), int(box[1]), int(box[2]), int(box[3]))))
    return results


def run_model_on_images(model_path: Path, images: List[Path], output_dir: Path, labels: List[str]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = []

    print(f"\n=== Model: {model_path.name} ===")
    t0 = time.time()
    # Try CoreML first (if available), fallback to CPU. Be robust to older onnxruntime APIs.
    providers = ["CPUExecutionProvider"]
    try:
        avail = []
        # Try modern API
        if hasattr(ort, "get_available_providers"):
            avail = ort.get_available_providers() or []
        # Fallback to get_all_providers if present
        elif hasattr(ort, "get_all_providers"):
            avail = ort.get_all_providers() or []
        if "CoreMLExecutionProvider" in avail and "CoreMLExecutionProvider" not in providers:
            providers.insert(0, "CoreMLExecutionProvider")
    except Exception as _e:
        # Keep CPU only if detection failed
        pass

    # If providers argument isn't supported, fall back to default constructor
    try:
        sess = ort.InferenceSession(str(model_path), providers=providers)
        print("Using providers:", providers)
    except TypeError:
        sess = ort.InferenceSession(str(model_path))
        print("Using default provider configuration")
    load_ms = (time.time() - t0) * 1000
    print(f"Session loaded in {load_ms:.1f} ms")

    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    # Expected shape: [1,3,H,W] or [N,3,H,W]
    shape = input_meta.shape
    # Resolve dynamic dims (-1) with defaults
    if isinstance(shape[2], str) or shape[2] is None:
        in_h = 640
    else:
        in_h = int(shape[2])
    if isinstance(shape[3], str) or shape[3] is None:
        in_w = 640
    else:
        in_w = int(shape[3])
    print(f"Input shape resolved to [1,3,{in_h},{in_w}]")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARN: could not read {img_path}")
            continue
        h0, w0 = img.shape[:2]
        # Letterbox to preserve aspect
        resized, scale, (dw, dh) = letterbox(img, (in_w, in_h))
        # BGR->RGB, HWC->CHW, 0..255 -> 0..1
        inp = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        inp = np.expand_dims(inp, 0)

        t1 = time.time()
        outputs = sess.run(None, {input_name: inp})
        inf_ms = (time.time() - t1) * 1000
        timings.append(inf_ms)

        dets = postprocess_yolo(outputs, (h0, w0), (in_w, in_h), scale, (dw, dh), score_thr=0.25, iou_thr=0.6)

        # Draw
        for cls_id, score, (x1, y1, x2, y2) in dets:
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = labels[cls_id] if 0 <= cls_id < len(labels) else str(cls_id)
            txt = f"{label}:{score:.2f}"
            cv2.putText(img, txt, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out_file = output_dir / img_path.name
        cv2.imwrite(str(out_file), img)

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


def discover_models() -> List[Path]:
    # Only benchmark one local file on Mac: prefer 'yolov12n.onnx', fallback to 'yolo12n.onnx'
    for name in ["yolov12n.onnx", "yolo12n.onnx"]:
        p = ROOT / name
        if p.exists():
            return [p]
    return []


def _select_torch_device() -> str:
    try:
        if 'torch' in sys.modules and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        if 'torch' in sys.modules and torch.cuda.is_available():
            return 'cuda'
    except Exception:
        pass
    return 'cpu'


def run_pt_model_on_images(model_path: Path, images: List[Path], output_dir: Path, labels: List[str]) -> dict:
    """Evaluate a local YOLO .pt model using Ultralytics and save annotated outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = []

    if not _ULTRALYTICS_AVAILABLE:
        print("Ultralytics not available. Install with: pip install ultralytics torch torchvision")
        return {
            "model": model_path.name,
            "images": 0,
            "load_ms": 0.0,
            "avg_ms": 0.0,
            "p50_ms": 0.0,
            "p90_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "fps": 0.0,
        }

    print(f"\n=== Model: {model_path.name} (pt) ===")
    device = _select_torch_device()
    print(f"Using torch device: {device}")

    t0 = time.time()
    model = YOLO(str(model_path))
    load_ms = (time.time() - t0) * 1000
    print(f"Model loaded in {load_ms:.1f} ms")

    # Predict loop
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARN: could not read {img_path}")
            continue

        t1 = time.time()
        # conf and iou align roughly to the ONNX postproc
        results = model.predict(source=img, conf=0.25, iou=0.6, device=device, verbose=False)
        inf_ms = (time.time() - t1) * 1000
        timings.append(inf_ms)

        # Draw
        drawn = img.copy()
        if results and len(results) > 0:
            r = results[0]
            if hasattr(r, 'boxes') and r.boxes is not None:
                xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes, 'xyxy') else np.zeros((0, 4), dtype=np.float32)
                cls = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes, 'cls') else np.zeros((0,), dtype=int)
                confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, 'conf') else np.zeros((0,), dtype=np.float32)
                for (x1, y1, x2, y2), c, sc in zip(xyxy.astype(int), cls, confs):
                    color = (0, 128, 255)
                    cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
                    label = labels[c] if 0 <= c < len(labels) else str(c)
                    cv2.putText(drawn, f"{label}:{sc:.2f}", (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        out_file = output_dir / img_path.name
        cv2.imwrite(str(out_file), drawn)

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


def main():
    labels = load_labels()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in IMAGES_DIR.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        print(f"No images found in: {IMAGES_DIR}")
        sys.exit(1)
    images = images[:100]
    print(f"Found {len(images)} images. Running up to 100.")

    models = discover_models()
    if not models:
        print("Target ONNX 'yolov12n.onnx' (or 'yolo12n.onnx') not found in MLModels/. Place it there and re-run.")
        sys.exit(1)

    summaries = []

    # ONNX benchmark
    onnx_model = models[0]
    onnx_out_dir = RUNS_DIR / onnx_model.stem
    summaries.append(run_model_on_images(onnx_model, images, onnx_out_dir, labels))

    # .pt benchmark (yolo12n.pt) if available
    pt_model = ROOT / "yolo12n.pt"
    if pt_model.exists():
        pt_out_dir = RUNS_DIR / pt_model.stem
        summaries.append(run_pt_model_on_images(pt_model, images, pt_out_dir, labels))
    else:
        print("yolov12n.pt not found in MLModels/. Skipping .pt benchmark.")

    # Write CSV summary
    csv_path = RUNS_DIR / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "images", "load_ms", "avg_ms", "p50_ms", "p90_ms", "min_ms", "max_ms", "fps"
        ])
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    print(f"\nAll done. Annotated outputs in: {RUNS_DIR}\nSummary CSV: {csv_path}")


if __name__ == "__main__":
    main()


