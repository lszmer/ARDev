#!/usr/bin/env python3
"""
Evaluate a YOLO-seg ONNX model (e.g., yolo11n-seg.onnx) on up to 100 sample images
and save visualizations with colored segmentation overlays and bounding boxes.

Assumptions (heuristic, robust to slight layout differences):
- Model has two outputs: detections and mask protos, OR multiple outputs containing those.
- Detections contain [cx, cy, w, h] + NC class scores + NM mask coefficients.
- Protos have shape [1, NM, Mh, Mw].

If your model differs, adjust the detection/mask parsing code below.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is required. pip install onnxruntime")
    sys.exit(1)

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. pip install opencv-python")
    sys.exit(1)


ROOT = Path(__file__).resolve().parent
IMAGES_DIR = ROOT / "images" / "indoor objects" / "train" / "images"
RUNS_DIR = ROOT / "runs" / "eval"


def load_labels() -> List[str]:
    # Try common label files used in this repo
    candidates = [
        ROOT / "yolo_classes.txt",
        ROOT / "yolo_classes 2.txt",
    ]
    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
    # Fallback COCO-like dummy labels
    return [f"class_{i}" for i in range(80)]


def letterbox(image: np.ndarray, new_shape: Tuple[int, int], color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
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


def _identify_outputs(outputs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (dets, proto) tensors from ONNX outputs by heuristic rules."""
    dets = None
    proto = None
    # Prefer 4D proto (1, Cmask, Mh, Mw)
    for o in outputs:
        if o.ndim == 4:
            # pick the first 4D as proto
            proto = o
            break
    # For dets, prefer the 2D/3D tensor with per-detection width > 4
    cands = [o for o in outputs if o.ndim in (2, 3)]
    if cands:
        # Heuristic: prefer shape where trailing dim > leading (i.e., (N, L))
        def score(t):
            s = t.size
            shape = t.shape
            if t.ndim == 2:
                if shape[1] > shape[0]:
                    s *= 2
            elif t.ndim == 3:
                # (1, N, L) or (1, L, N) -> prefer L on last dim
                if shape[-1] > shape[-2]:
                    s *= 2
            return s
        dets = max(cands, key=score)
    if dets is None or proto is None:
        raise RuntimeError("Could not identify dets/proto tensors from model outputs")
    return dets, proto


def _resolve_input_shape(meta) -> Tuple[int, int]:
    shape = meta.shape
    in_h = 640 if (isinstance(shape[2], str) or shape[2] is None) else int(shape[2])
    in_w = 640 if (isinstance(shape[3], str) or shape[3] is None) else int(shape[3])
    return in_h, in_w


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid to avoid overflow warnings
    x_clipped = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def postprocess_yolo_seg(dets_raw: np.ndarray, proto_raw: np.ndarray, img_shape_hw: Tuple[int, int], input_shape_wh: Tuple[int, int], scale: float, pad: Tuple[int, int], score_thr=0.25, iou_thr=0.6, mask_thr=0.5):
    h0, w0 = img_shape_hw
    in_w, in_h = input_shape_wh

    # Squeeze batch dim if present
    dets = np.squeeze(dets_raw)
    proto = np.squeeze(proto_raw)  # [Cmask, Mh, Mw] expected

    # Normalize det tensor to shape (N, L)
    if dets.ndim == 3:
        # assume (1, N, L) or (1, L, N)
        if dets.shape[1] < dets.shape[2]:
            dets = dets[0]
        else:
            dets = dets[0].transpose(1, 0)
    elif dets.ndim != 2:
        raise RuntimeError(f"Unexpected dets shape: {dets.shape}")

    nm = int(proto.shape[0])  # mask coeff length equals proto channels
    L = int(dets.shape[1])
    if L <= 4 + nm:
        raise RuntimeError(f"Per-detection length too small: L={L}, nm={nm}")
    nc = L - 4 - nm

    raw_box = dets[:, :4].astype(np.float32)
    cls_scores_all = dets[:, 4:4+nc].astype(np.float32)
    mask_coeffs = dets[:, 4+nc:4+nc+nm].astype(np.float32)

    # Convert class logits to probabilities (robust to models emitting logits)
    cls_scores_all = 1.0 / (1.0 + np.exp(-np.clip(cls_scores_all, -80.0, 80.0)))

    class_ids = np.argmax(cls_scores_all, axis=1)
    class_scores = cls_scores_all[np.arange(cls_scores_all.shape[0]), class_ids]

    # Filter by score
    keep = class_scores >= score_thr
    if not np.any(keep):
        return []
    boxes_cxcywh = boxes_cxcywh[keep]
    class_ids = class_ids[keep]
    class_scores = class_scores[keep]
    mask_coeffs = mask_coeffs[keep]

    # Determine if raw_box is xyxy or cxcywh
    # Heuristic: if x2>x1 and y2>y1 for majority, treat as xyxy
    x1r, y1r, x2r, y2r = raw_box[:, 0], raw_box[:, 1], raw_box[:, 2], raw_box[:, 3]
    xyxy_like = np.mean((x2r > x1r) & (y2r > y1r)) > 0.6

    if xyxy_like:
        boxes_in = raw_box.copy()
        # Normalize check for xyxy: majority coords in [0,1]
        maxv = np.max(np.abs(boxes_in)) + 1e-6
        normalized = np.mean((boxes_in >= -0.01) & (boxes_in <= 1.01)) > 0.5
        if normalized:
            boxes_in[:, [0, 2]] *= in_w
            boxes_in[:, [1, 3]] *= in_h
    else:
        # Treat as cxcywh
        boxes_cxcywh = raw_box.copy()
        normalized = np.mean(boxes_cxcywh[:, 2:4] <= 1.5) > 0.8
        if normalized:
            boxes_cxcywh[:, 0] *= in_w
            boxes_cxcywh[:, 1] *= in_h
            boxes_cxcywh[:, 2] *= in_w
            boxes_cxcywh[:, 3] *= in_h
        cx, cy, w, h = boxes_cxcywh[:, 0], boxes_cxcywh[:, 1], boxes_cxcywh[:, 2], boxes_cxcywh[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_in = np.stack([x1, y1, x2, y2], axis=1)

    # Map to original image coords (invert letterbox)
    dw, dh = pad
    boxes_in[:, [0, 2]] -= dw
    boxes_in[:, [1, 3]] -= dh
    boxes_in[:, [0, 2]] /= (scale + 1e-9)
    boxes_in[:, [1, 3]] /= (scale + 1e-9)
    boxes_in[:, 0::2] = boxes_in[:, 0::2].clip(0, w0 - 1)
    boxes_in[:, 1::2] = boxes_in[:, 1::2].clip(0, h0 - 1)

    # NMS per class
    results = []
    for c in np.unique(class_ids):
        inds = np.where(class_ids == c)[0]
        keep_inds = nms(boxes_in[inds], class_scores[inds], iou_thr)
        for k in keep_inds:
            bi = inds[k]
            results.append((int(c), float(class_scores[bi]), boxes_in[bi].astype(int), mask_coeffs[bi]))

    # Build full-res masks per kept detection (mirror PT path)
    proto_ch, mh, mw = proto.shape
    proto_flat = proto.reshape(proto_ch, mh * mw)  # [Cmask, Mh*Mw]
    masks = []
    # Compute the valid resized region extents once (in input image space)
    in_w, in_h = input_shape_wh
    resized_w = int(round(w0 * scale))
    resized_h = int(round(h0 * scale))
    x0_in = int(dw)
    y0_in = int(dh)
    x1_in = x0_in + resized_w
    y1_in = y0_in + resized_h
    x0_in = max(0, min(x0_in, in_w))
    y0_in = max(0, min(y0_in, in_h))
    x1_in = max(0, min(x1_in, in_w))
    y1_in = max(0, min(y1_in, in_h))

    for _, _, box_xyxy, coeff in results:
        # Combine proto with coeffs -> [Mh*Mw]
        m = np.dot(coeff, proto_flat)  # [Mh*Mw]
        m = m.reshape(mh, mw)
        m = _sigmoid(m)
        # Upsample to network input size
        m_up = cv2.resize(m, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        # Crop to the resized image region (removes padding from letterbox)
        m_valid = m_up[y0_in:y1_in, x0_in:x1_in]
        if m_valid.size == 0:
            m_valid = m_up
        # Resize valid region back to original image size (nearest preserves hard edges)
        m_img = cv2.resize(m_valid, (w0, h0), interpolation=cv2.INTER_NEAREST)
        # Optionally crop by bbox
        x1i, y1i, x2i, y2i = [int(v) for v in box_xyxy]
        mask_full = np.zeros((h0, w0), dtype=np.float32)
        mask_roi = m_img[y1i:y2i, x1i:x2i]
        if mask_roi.size:
            mask_full[y1i:y2i, x1i:x2i] = mask_roi
        masks.append(mask_full > mask_thr)

    # Attach masks back to results in order
    results_with_masks = []
    for (c, sc, box, coeff), mask in zip(results, masks):
        results_with_masks.append((c, sc, box, mask))
    return results_with_masks


def color_for_class(c: int) -> Tuple[int, int, int]:
    # Deterministic color palette
    rng = np.random.default_rng(seed=c * 9973)
    col = rng.integers(low=64, high=255, size=3, dtype=np.int32)
    return int(col[0]), int(col[1]), int(col[2])


def run_model_on_images(model_path: Path, images: List[Path], output_dir: Path, labels: List[str]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    timings = []

    print(f"\n=== Model: {model_path.name} (seg) ===")
    t0 = time.time()
    providers = ["CPUExecutionProvider"]
    try:
        avail = ort.get_available_providers() if hasattr(ort, "get_available_providers") else []
        if "CoreMLExecutionProvider" in avail:
            providers.insert(0, "CoreMLExecutionProvider")
    except Exception:
        pass

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
    in_h, in_w = _resolve_input_shape(input_meta)
    print(f"Input shape resolved to [1,3,{in_h},{in_w}]")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"WARN: could not read {img_path}")
            continue
        h0, w0 = img.shape[:2]
        resized, scale, (dw, dh) = letterbox(img, (in_w, in_h))
        inp = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        inp = np.expand_dims(inp, 0)

        t1 = time.time()
        outputs = sess.run(None, {input_name: inp})
        dets_raw, proto_raw = _identify_outputs(outputs)
        results = postprocess_yolo_seg(dets_raw, proto_raw, (h0, w0), (in_w, in_h), scale, (dw, dh))
        inf_ms = (time.time() - t1) * 1000
        timings.append(inf_ms)

        # Draw overlays
        overlay = img.copy()
        alpha = 0.5
        for cls_id, score, (x1, y1, x2, y2), mask in results:
            color = color_for_class(cls_id)
            # ensure boolean mask
            m = mask
            if m.ndim == 3 and m.shape[0] == 1:
                m = m[0]
            if m.shape[:2] != overlay.shape[:2]:
                m = cv2.resize(m.astype(np.float32), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = (m > 0.5)
            if np.any(m):
                overlay[m] = (overlay[m] * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)
            # bbox and label for reference
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            label = labels[cls_id] if 0 <= cls_id < len(labels) else str(cls_id)
            txt = f"{label}:{score:.2f}"
            cv2.putText(overlay, txt, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

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


def discover_models() -> List[Path]:
    # Prefer yolo11n-seg.onnx or yolov12n-seg.onnx in MLModels
    for name in ["yolo11n-seg.onnx", "yolov12n-seg.onnx"]:
        p = ROOT / name
        if p.exists():
            return [p]
    return []


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
        print("Target ONNX 'yolo11n-seg.onnx' (or 'yolov12n-seg.onnx') not found in MLModels/. Place it there and re-run.")
        sys.exit(1)

    model = models[0]
    out_dir = RUNS_DIR / model.stem
    _ = run_model_on_images(model, images, out_dir, labels)
    print(f"\nAll done. Annotated outputs in: {out_dir}")


if __name__ == "__main__":
    main()


