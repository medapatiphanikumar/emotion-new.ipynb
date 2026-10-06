"""
STAGE 1 — RF-DETR Seg Medium: cassette_body + splice_connector.

Responsibilities
  1. run the RF-DETR instance-segmentation checkpoint on the full photo
  2. build the REGION OF INTEREST for the downstream fibre models:
        ROI = cassette_body mask  minus  (dilated) splice_connector masks
     cropped to the cassette bbox, with every pixel outside the ROI replaced by a neutral
     fill.  The fill is the mean colour of the ROI itself by default, so the shades-of-grey
     illuminant estimate and the retinex flattening in stage 3 are not biased by a large
     black rectangle.
  3. hand the splice_connector masks (in crop coordinates) to stage 2, where they become
     the port instances the fibres are bound to.
  4. draw the stage-1 overlay (contours, not boxes — same style as the original app).

Everything here works in RGB uint8 numpy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

CLASS_NAMES = ["cassette_body", "splice_connector"]
CASSETTE_ID, CONNECTOR_ID = 0, 1

# RGB colours (the overlay is drawn on an RGB array)
MASK_RGB    = {"cassette_body": (74, 181, 57),   "splice_connector": (220, 80, 80)}
CONTOUR_RGB = {"cassette_body": (30, 220, 30),   "splice_connector": (255, 60, 60)}
LABEL_BG_RGB = {"cassette_body": (30, 140, 30),  "splice_connector": (200, 40, 40)}


# ══════════════════════════════════════════════════════════════════════════════════════════
#  model wrapper
# ══════════════════════════════════════════════════════════════════════════════════════════
class CassetteSegmenter:
    """Thin wrapper around rfdetr.RFDETRSegMedium returning plain dict instances."""

    def __init__(self, ckpt_path: str, num_classes: int = 2, optimize: bool = True,
                 load_retries: int = 3, log=print, compile: bool = None):
        try:
            from retry_utils import load_retry
        except ImportError:                              # helper missing -> single attempt
            def load_retry(attempts=1, exc=(), log=print): return lambda f: f
        self.ckpt_path = ckpt_path
        self.num_classes = num_classes
        self.log = log

        @load_retry(load_retries, log=log)
        def _load():
            from rfdetr import RFDETRSegMedium      # imported lazily: heavy
            return RFDETRSegMedium(num_classes=num_classes, pretrain_weights=ckpt_path)
        self.model = _load()
        if optimize:
            # LATENCY: rfdetr can fuse / compile its graph for repeated same-size inference.
            # Signature differs between rfdetr releases, so try the richer call first.
            # RUNTIME: compile only when the profile asks for it (on CPU inductor compile of the
            # DINOv2 backbone is minutes of warm-up; fused eager is the better default there).
            want_compile = torch.cuda.is_available() if compile is None else bool(compile)
            for kw in ({"compile": want_compile, "batch_size": 1}, {"compile": False, "batch_size": 1}, {}):
                try:
                    self.model.optimize_for_inference(**kw); log(f"stage 1: optimize_for_inference({kw}) OK"); break
                except TypeError:
                    continue
                except Exception as e:
                    log(f"stage 1: optimize_for_inference skipped ({type(e).__name__}: {e})"); break

    def warmup(self, size=(640, 640)):
        """One dummy forward so CUDA init / compile / cudnn autotune are not billed to the first photo."""
        try:
            self.model.predict(Image.fromarray(np.zeros((size[0], size[1], 3), np.uint8)), threshold=0.9)
        except Exception as e:
            self.log(f"stage 1 warmup skipped: {e}")

    def predict(self, img_rgb: np.ndarray, threshold: float = 0.45) -> List[dict]:
        """
        Returns a list of {"cls", "name", "score", "bbox"(x0,y0,x1,y1), "mask"(HxW bool)}.
        Masks are full-frame booleans; a bbox rectangle is used if the model gives no mask.
        """
        H, W = img_rgb.shape[:2]
        det = self.model.predict(Image.fromarray(img_rgb), threshold=float(threshold))
        out = []
        if det is None or len(det) == 0:
            return out
        has_masks = getattr(det, "mask", None) is not None and len(det.mask) > 0
        for i in range(len(det)):
            cid = int(det.class_id[i]); conf = float(det.confidence[i])
            x0, y0, x1, y1 = [int(round(float(v))) for v in det.xyxy[i]]
            x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
            if has_masks and det.mask[i] is not None:
                m = np.asarray(det.mask[i]).astype(bool)
                if m.shape != (H, W):                       # be defensive about resolution
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                m = np.zeros((H, W), bool); m[y0:y1, x0:x1] = True
            if m.sum() == 0:
                continue
            out.append({"cls": cid,
                        "name": CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}",
                        "score": conf, "bbox": (x0, y0, x1, y1), "mask": m})
        # largest first so the primary cassette is instance 0
        out.sort(key=lambda k: (k["cls"], -int(k["mask"].sum())))
        return out


# ══════════════════════════════════════════════════════════════════════════════════════════
#  ROI extraction
# ══════════════════════════════════════════════════════════════════════════════════════════
@dataclass
class ROIResult:
    crop_raw:    np.ndarray                 # RGB crop of the cassette bbox, untouched
    crop_filled: np.ndarray                 # same crop, non-ROI pixels replaced by fill
    roi_local:   np.ndarray                 # bool mask (crop frame): what stages 2/3 may look at
    cassette_local: np.ndarray              # bool mask (crop frame): cassette only
    offset:      Tuple[int, int]            # (x0, y0) of the crop in the full image
    bbox:        Tuple[int, int, int, int]  # crop bbox in the full image
    fill_rgb:    Tuple[int, int, int]
    connector_local: Optional[np.ndarray] = None   # bool mask (crop frame): dilated splice connectors
    connectors_local: List[Tuple[Tuple[int, int, int, int], np.ndarray, float]] = field(default_factory=list)
    # [(bbox_local(x0,y0,x1,y1), local_bool_mask, score), ...]  -> port instances for stage 2
    n_cassette:  int = 0
    n_connector: int = 0
    fallback:    bool = False               # True when no cassette was found (whole image used)
    roi_px:      int = 0
    connector_px_removed: int = 0

    def to_full(self, x: float, y: float) -> Tuple[float, float]:
        return x + self.offset[0], y + self.offset[1]


def extract_roi(img_rgb: np.ndarray, insts: List[dict], cfg) -> ROIResult:
    H, W = img_rgb.shape[:2]
    cass = [k for k in insts if k["cls"] == CASSETTE_ID]
    conn = [k for k in insts if k["cls"] == CONNECTOR_ID]

    if cass:
        if cfg.cassette_select == "union":
            cmask = np.zeros((H, W), bool)
            for k in cass: cmask |= k["mask"]
        else:                                                    # "largest"
            cmask = max(cass, key=lambda k: int(k["mask"].sum()))["mask"].copy()
        fallback = False
    else:
        cmask = np.ones((H, W), bool); fallback = True

    conn_mask = np.zeros((H, W), bool)
    for k in conn: conn_mask |= k["mask"]
    if cfg.connector_dilate_px > 0 and conn_mask.any():
        r = int(cfg.connector_dilate_px)
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        conn_mask = cv2.dilate(conn_mask.astype(np.uint8), ker).astype(bool)

    mode = _removal_mode(cfg)
    # "before": the old behaviour - connectors are blanked from what stages 2/3 see.
    # "after" / "none": stages 2/3 see the whole cassette; removal (if any) is done on the
    # fibre masks and the output image once every model has run (pipeline.py).
    roi_mask = (cmask & ~conn_mask) if (mode == "before" and conn) else cmask

    ys, xs = np.nonzero(cmask)
    bx0, bx1 = int(xs.min()), int(xs.max()) + 1
    by0, by1 = int(ys.min()), int(ys.max()) + 1
    pad = int(round(cfg.roi_pad_frac * max(bx1 - bx0, by1 - by0)))
    x0, y0 = max(0, bx0 - pad), max(0, by0 - pad)
    x1, y1 = min(W, bx1 + pad), min(H, by1 + pad)

    crop = np.ascontiguousarray(img_rgb[y0:y1, x0:x1])
    roi_local = roi_mask[y0:y1, x0:x1]
    cass_local = cmask[y0:y1, x0:x1]

    if cfg.roi_fill == "keep":
        fill = None
    elif cfg.roi_fill == "black":
        fill = (0, 0, 0)
    else:                                                        # "mean"
        px = crop[roi_local]
        fill = tuple(int(v) for v in px.reshape(-1, 3).mean(0)) if len(px) else (128, 128, 128)
    filled = crop.copy()
    if fill is not None:
        filled[~roi_local] = np.array(fill, np.uint8)

    connectors_local = []
    for ci, k in enumerate(conn):
        kx0, ky0, kx1, ky1 = k["bbox"]
        lx0, ly0 = max(kx0, x0) - x0, max(ky0, y0) - y0
        lx1, ly1 = min(kx1, x1) - x0, min(ky1, y1) - y0
        if lx1 <= lx0 or ly1 <= ly0: continue
        lm = k["mask"][y0 + ly0:y0 + ly1, x0 + lx0:x0 + lx1]
        if lm.sum() == 0: continue
        connectors_local.append(((lx0, ly0, lx1, ly1), np.ascontiguousarray(lm), float(k["score"]), ci))

    return ROIResult(crop_raw=crop, crop_filled=filled, roi_local=roi_local,
                     cassette_local=cass_local, connector_local=conn_mask[y0:y1, x0:x1],
                     offset=(x0, y0), bbox=(x0, y0, x1, y1),
                     fill_rgb=fill or (0, 0, 0), connectors_local=connectors_local,
                     n_cassette=len(cass), n_connector=len(conn), fallback=fallback,
                     roi_px=int(roi_local.sum()),
                     connector_px_removed=int((cmask & conn_mask).sum()) if mode != "none" else 0)


def _removal_mode(cfg) -> str:
    mode = getattr(cfg, "connector_removal", "after")
    if not getattr(cfg, "remove_connectors", True): mode = "none"
    return mode if mode in ("after", "before", "none") else "after"


# ══════════════════════════════════════════════════════════════════════════════════════════
#  drawing
# ══════════════════════════════════════════════════════════════════════════════════════════
def _label_tag(img, text, cx, cy, bg, border, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX; th = max(1, int(scale * 2))
    (tw, tth), _ = cv2.getTextSize(text, font, scale, th)
    pad = max(3, int(scale * 6))
    cv2.rectangle(img, (cx - pad, cy - tth - pad * 2), (cx + tw + pad, cy + pad), bg, -1)
    cv2.rectangle(img, (cx - pad, cy - tth - pad * 2), (cx + tw + pad, cy + pad), border, 1)
    cv2.putText(img, text, (cx, cy), font, scale, (255, 255, 255), th, cv2.LINE_AA)


def annotate_stage1(img_rgb: np.ndarray, insts: List[dict], alpha: float = 0.35) -> np.ndarray:
    """Semi-transparent fill + thick contour + label tag at the centroid. No boxes."""
    overlay = img_rgb.copy()
    if not insts:
        return overlay
    H, W = img_rgb.shape[:2]
    scale = max(0.5, min(H, W) / 1600.0)
    thick = max(2, int(scale * 4))
    for k in insts:
        name = k["name"]
        mc = MASK_RGB.get(name, (150, 150, 150)); cc = CONTOUR_RGB.get(name, (200, 200, 200))
        bg = LABEL_BG_RGB.get(name, (100, 100, 100))
        m = k["mask"]
        sub = overlay[m]
        overlay[m] = (np.array(mc, np.float32) * alpha + sub.astype(np.float32) * (1 - alpha)).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, cc, thick)
        if cnts:
            big = max(cnts, key=cv2.contourArea); M = cv2.moments(big)
            if M["m00"] > 0: cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(big); cx, cy = x + w // 2, y + h // 2
            _label_tag(overlay, f"{name} {k['score']:.2f}", cx, cy, bg, cc, scale)
    return overlay


def annotate_roi(roi: ROIResult) -> np.ndarray:
    """The stage-2 input as the operator will see it: filled crop with the ROI outline."""
    out = roi.crop_filled.copy()
    cnts, _ = cv2.findContours(roi.roi_local.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = out.shape[:2]
    cv2.drawContours(out, cnts, -1, (30, 220, 30), max(1, int(min(h, w) / 800)))
    return out