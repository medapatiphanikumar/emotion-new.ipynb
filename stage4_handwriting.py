"""
STAGE 4 — handwritten label reading on the SPLICE-CONNECTOR regions only.

Wraps the uploaded `inference.py` (YOLOv8-OBB "Cassette-Handwritten" detector -> PARSeq-1
fine-tuned reader -> optional PARSeq-base -> confidence-gated selection). None of that logic
is duplicated here; `inference.process_image` is called as-is on a crop.

What this stage adds is the *restriction* and the *bookkeeping*:

  * one crop per RF-DETR `splice_connector` (bbox padded by `ocr_pad_frac`, optionally the
    pixels outside the dilated connector mask replaced by the ROI mean so the OBB detector
    can only fire on the connector itself);
  * every OBB polygon comes back in the crop's *rotated* frame (inference.py may re-orient
    the crop by 90/180/270 to read best) -> un-rotated and offset back to full-photo
    coordinates so the drawing and the JSON line up with the other stages;
  * each connector is linked to the port number stage 2 assigned to it, so a label reads as
    "Port 3: 'A12#B7'" next to that port's fibre colours;
  * one annotated image of the whole photo (connector boxes + OBB quads + text).

Requires beside the package:  inference.py (included), parseq_cassette_infer.py (yours),
YOLO best.pt and parseq_cassette_final.pt.  Extra deps: ultralytics, pytorch-lightning, timm.
"""
from __future__ import annotations

import time
from typing import List, Optional

import cv2
import numpy as np

from leftright import add_centroids, assign_sides, draw_split, split_sides


class HandwritingReader:
    def __init__(self, yolo_weights: str, parseq1_ckpt: str, parseq_repo: Optional[str] = None,
                 use_base: bool = False, max_lines: int = 3, log=print, prof=None):
        import importlib.util, os
        # ── preflight: say precisely what is missing instead of a bare ImportError ─────────
        missing = []
        for mod, pipname in (("ultralytics", "ultralytics"), ("torchvision", "torchvision"),
                             ("timm", "timm"), ("pytorch_lightning", "'pytorch-lightning>=2.2,<3'")):
            if importlib.util.find_spec(mod) is None: missing.append(pipname)
        if missing:
            raise RuntimeError("stage 4 needs: pip install " + " ".join(missing))
        # inference.py and parseq_cassette_infer.py are loaded from THIS file's folder by path,
        # not by module name: this cannot be shadowed by an installed package called
        # `inference` (Roboflow ships one) and the error names the exact path that was checked.
        import sys
        from pathlib import Path
        here = Path(__file__).resolve().parent
        inf_path = here / "inference.py"
        if not inf_path.exists():
            raise RuntimeError(f"inference.py not found at {inf_path} (copy it from the zip into that folder)")
        if not (here / "parseq_cassette_infer.py").exists():
            raise RuntimeError(f"parseq_cassette_infer.py (from your OCR project) not found at {here}")
        for p, what in ((yolo_weights, "YOLO-OBB best.pt"), (parseq1_ckpt, "parseq_cassette_final.pt")):
            if not p or not os.path.exists(p): raise FileNotFoundError(f"{what} not found at: {p!r}")
        if str(here) not in sys.path: sys.path.insert(0, str(here))     # so inference.py finds parseq_cassette_infer
        spec = importlib.util.spec_from_file_location("jacops_inference", inf_path)
        inf = importlib.util.module_from_spec(spec); sys.modules["jacops_inference"] = inf
        spec.loader.exec_module(inf)                 # the uploaded OCR module, unchanged
        self.inf = inf; self.log = log
        self.yolo = inf.load_yolo_model(yolo_weights)
        dev = None
        if prof is not None: dev = prof.device      # "cpu" / "cuda"
        self.reader = inf.load_parseq1_reader(parseq1_ckpt, device=dev, repo_dir=parseq_repo,
                                              max_lines=max_lines)
        if self.reader is None:
            raise RuntimeError(f"PARSeq-1 failed to load: {inf.get_parseq1_load_error()}")
        self.base_model = self.base_tf = None
        if use_base:
            self.base_model, self.base_tf = inf.load_parseq_model(device=dev, local_repo=parseq_repo)
            if self.base_model is None: log(f"stage 4: PARSeq-base unavailable ({inf.get_parseq_load_error()})")
        self.line_sep = getattr(self.reader, "line_sep", "#")
        try:                                          # warm the OBB detector (autotune / fuse)
            self.yolo.predict(source=np.full((256, 256, 3), 90, np.uint8), imgsz=256, verbose=False,
                              device=0 if dev == "cuda" else "cpu")
        except Exception:
            pass

    # ── one connector crop ────────────────────────────────────────────────────────────────
    def read_crop(self, crop_bgr: np.ndarray, cfg):
        ann, dets, meta = self.inf.process_image(
            crop_bgr, self.yolo, self.reader, self.base_model, self.base_tf,
            conf_threshold=cfg.ocr_conf, imgsz=cfg.ocr_imgsz, use_base=self.base_model is not None,
            annotate_text=cfg.ocr_annotate_text, orientation=cfg.ocr_orientation)
        return ann, dets, meta


# ══════════════════════════════════════════════════════════════════════════════════════════
#  geometry helpers
# ══════════════════════════════════════════════════════════════════════════════════════════
def unrotate_point(x: float, y: float, angle: int, w0: int, h0: int):
    """Map a point in the crop rotated by `angle` (cv2.rotate convention) back to the
    unrotated crop of size (w0, h0)."""
    a = int(angle) % 360
    if a == 0:   return x, y
    if a == 90:  return y, h0 - 1 - x                # ROTATE_90_CLOCKWISE:  x' = h0-1-y, y' = x
    if a == 180: return w0 - 1 - x, h0 - 1 - y
    if a == 270: return w0 - 1 - y, x                # ROTATE_90_COUNTERCLOCKWISE: x' = y, y' = w0-1-x
    return x, y


def connector_crops(img_rgb: np.ndarray, connectors: List[dict], cfg, fill_rgb=(128, 128, 128)):
    """[(conn_idx, bbox_full(x0,y0,x1,y1), crop_bgr)] one per splice_connector detection."""
    H, W = img_rgb.shape[:2]; out = []
    for ci, k in enumerate(connectors):
        x0, y0, x1, y1 = k["bbox"]
        pad = int(round(cfg.ocr_pad_frac * max(x1 - x0, y1 - y0)))
        cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
        if cx1 - cx0 < 8 or cy1 - cy0 < 8: continue
        crop = img_rgb[cy0:cy1, cx0:cx1].copy()
        if cfg.ocr_mask_outside:
            m = k["mask"][cy0:cy1, cx0:cx1]
            r = max(1, int(cfg.ocr_mask_dilate_px))
            m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
            crop[~m] = np.array(fill_rgb, np.uint8)
        out.append((ci, (cx0, cy0, cx1, cy1), np.ascontiguousarray(crop[:, :, ::-1])))   # BGR for inference.py
    return out


# ══════════════════════════════════════════════════════════════════════════════════════════
#  the stage
# ══════════════════════════════════════════════════════════════════════════════════════════
def run_stage4(reader: HandwritingReader, img_rgb: np.ndarray, connectors: List[dict], port_num_of: dict,
               cfg, fill_rgb=(128, 128, 128), log=print, cassette_bbox=None):
    """
    connectors   : RF-DETR splice_connector instances (full-frame masks/bboxes)
    port_num_of  : {conn_idx: port_num} from stage 2's injected ports
    cassette_bbox: (x0,y0,x1,y1) of the cassette from stage 1 -> reference frame for the
                   left/right split (its x-centre is the axis of symmetry)
    returns dict(labels=[...], per_connector=[...], split=..., n_regions, n_review, ...)
    """
    t0 = time.time(); labels, per_conn = [], []
    for ci, (cx0, cy0, cx1, cy1), crop_bgr in connector_crops(img_rgb, connectors, cfg, fill_rgb):
        w0, h0 = cx1 - cx0, cy1 - cy0
        try:
            ann_bgr, dets, meta = reader.read_crop(crop_bgr, cfg)
        except Exception as e:
            log(f"   stage 4: connector {ci} failed: {type(e).__name__}: {e}")
            per_conn.append({"conn_idx": ci, "port": port_num_of.get(ci), "bbox": [cx0, cy0, cx1, cy1],
                             "error": f"{type(e).__name__}: {e}", "n": 0}); continue
        rot = int(meta.get("rotation", 0) or 0)
        for d in dets:
            poly_full = [[float(v) for v in (lambda p: (p[0] + cx0, p[1] + cy0))(unrotate_point(px, py, rot, w0, h0))]
                         for px, py in d["poly"]]
            labels.append({
                "conn_idx": ci, "port": port_num_of.get(ci), "detection_id": d["detection_id"],
                "side_crop": d.get("side"), "side": None,       # crop-frame side (may be rotated); real side set below
                "det_conf": round(float(d["conf"]), 3),
                "parseq1_text": d.get("parseq1_text"), "parseq1_min_conf": round(float(d.get("parseq1_min_conf") or 0), 3),
                "parseq1_mean_conf": round(float(d.get("parseq1_mean_conf") or 0), 3),
                "parseq1_lines": d.get("parseq1_lines"), "parseq1_line_texts": d.get("parseq1_line_texts"),
                "parseq_base_text": d.get("parseq_base_text"), "parseq_base_conf": d.get("parseq_base_conf"),
                "final_text": d.get("final_text"), "final_engine": d.get("final_engine"),
                "review_status": d.get("review_status"), "rotation": rot, "poly": poly_full,
                "crop": d.get("crop"),                                       # BGR, kept for the Excel thumbnail
            })
        per_conn.append({"conn_idx": ci, "port": port_num_of.get(ci), "bbox": [cx0, cy0, cx1, cy1],
                         "rotation": rot, "split_method": meta.get("split_method"), "n": len(dets),
                         "texts": [d.get("final_text") for d in dets]})
    # ── LEFT / RIGHT on the whole cassette photo (after un-rotation) ─────────────────────
    H, W = img_rgb.shape[:2]
    if cassette_bbox is not None:
        ref_x0, ref_y0, ref_x1, ref_y1 = [int(v) for v in cassette_bbox]
    else:
        ref_x0, ref_y0, ref_x1, ref_y1 = 0, 0, W, H
    ref_w = max(1, ref_x1 - ref_x0); midline = ref_x0 + ref_w / 2.0
    override = None
    mode = getattr(cfg, "ocr_split_mode", "auto")
    if mode == "manual": override = ref_x0 + float(getattr(cfg, "ocr_split_frac", 0.5)) * ref_w
    elif mode == "cassette_mid": override = midline
    add_centroids(labels)
    margin = 0.03 * max(ref_w, ref_y1 - ref_y0)
    inside = [(ref_x0 - margin <= l["cx"] <= ref_x1 + margin) and (ref_y0 - margin <= l["cy"] <= ref_y1 + margin)
              for l in labels] if cassette_bbox is not None else None
    split_x, split_method, n_out = assign_sides(labels, ref_w, split_override=override, midline=midline, inside=inside)
    if mode == "cassette_mid": split_method = "cassette-midline (forced)"
    left, right, outl = split_sides(labels)
    split = {"x": round(float(split_x), 1), "method": split_method, "mode": mode,
             "ref_bbox": [ref_x0, ref_y0, ref_x1, ref_y1], "n_left": len(left), "n_right": len(right), "n_outlier": len(outl)}
    unread = sum(1 for l in labels if reader.inf.is_unreadable(l["final_text"]))
    return {"labels": labels, "per_connector": per_conn, "split": split, "n_regions": len(labels),
            "n_review": sum(1 for l in labels if l["review_status"] == "Needs Review"),
            "unreadable": unread, "n_connectors_read": len(per_conn), "seconds": round(time.time() - t0, 2)}


# ══════════════════════════════════════════════════════════════════════════════════════════
#  drawing + report
# ══════════════════════════════════════════════════════════════════════════════════════════
def render_stage4(img_rgb: np.ndarray, s4: dict, cfg) -> np.ndarray:
    ov = img_rgb.copy(); H, W = ov.shape[:2]
    fs = max(0.5, min(H, W) / 1600.0); th = max(1, int(fs * 2))
    for pc in s4["per_connector"]:
        x0, y0, x1, y1 = pc["bbox"]
        cv2.rectangle(ov, (x0, y0), (x1, y1), (255, 200, 40), max(1, int(fs * 2)))
        tag = f"Port {pc['port']}" if pc.get("port") is not None else f"holder {pc['conn_idx'] + 1}"
        if pc.get("rotation"): tag += f"  rot {pc['rotation']}"
        cv2.putText(ov, tag, (x0 + 4, max(14, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (255, 200, 40), th, cv2.LINE_AA)
    sp = s4.get("split")
    if sp and s4["labels"]:
        rx0, ry0, rx1, ry1 = sp["ref_bbox"]
        draw_split(ov, sp["x"], ry0, ry1, sp["method"], scale=max(1.0, fs * 1.2))
    for l in s4["labels"]:
        flagged = l["review_status"] == "Needs Review"
        col = (255, 165, 0) if flagged else (0, 255, 0)
        if l["side"] == "Outlier": col = (160, 160, 160)
        pts = np.array(l["poly"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(ov, [pts], True, col, max(2, int(fs * 3)))
        txt = f"ID{l['detection_id']} {l['det_conf']:.2f} {(l.get('side') or '?')[0]}"
        if cfg.ocr_annotate_text and l["final_text"]: txt += f"  {l['final_text']}"
        px, py = int(pts[0, 0, 0]), int(pts[0, 0, 1])
        for c_, t_ in [((0, 0, 0), th + 2), (col, th)]:
            cv2.putText(ov, txt, (px, max(14, py - 4)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, c_, t_, cv2.LINE_AA)
    return ov


def format_handwriting_report(s4: dict) -> str:
    if not s4["per_connector"]: return "(no splice connectors -> no handwriting read)"
    lines = []
    sp = s4.get("split")
    if sp and s4["labels"]:
        def _side(name):
            ls = [l for l in s4["labels"] if l.get("side") == name]
            return ", ".join(f"'{l['final_text']}'" + (" ?" if l["review_status"] == "Needs Review" else "") for l in ls) or "(none)"
        lines.append(f"LEFT : {_side('Left')}")
        lines.append(f"RIGHT: {_side('Right')}")
        if sp["n_outlier"]: lines.append(f"OFF-CASSETTE (review): {_side('Outlier')}")
        lines.append(f"split at x={sp['x']:.0f} ({sp['method']})")
    for pc in s4["per_connector"]:
        head = f"Port {pc['port']}" if pc.get("port") is not None else f"splice holder {pc['conn_idx'] + 1}"
        if pc.get("error"): lines.append(f"{head}: ERROR {pc['error']}"); continue
        ls = [l for l in s4["labels"] if l["conn_idx"] == pc["conn_idx"]]
        if not ls: lines.append(f"{head}: (no handwritten region)"); continue
        parts = []
        for l in ls:
            flag = " ?" if l["review_status"] == "Needs Review" else ""
            side = f"[{l['side'][0]}]" if l.get("side") else "[?]"
            parts.append(f"{side}'{l['final_text']}'{flag} ({l['final_engine']} {l['parseq1_min_conf']:.2f})")
        lines.append(f"{head}: " + ", ".join(parts))
    return "\n".join(lines)