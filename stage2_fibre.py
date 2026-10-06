"""
STAGE 2 — OneFormer fibre / port localisation  (port of FIBRE PIPELINE v6.4, stage 1).

Runs on the CASSETTE ROI produced by stage 1, not on the raw photo:
  tiled inference -> dedup -> endpoint chaining -> fibre instances -> port binding -> in/out.

Two integration points differ from the original single-cell script:
  * ribbon_override: the fibre ribbon width is normally derived from max(W, H) of the image.
    The ROI crop is smaller than the photo, so the pipeline computes the ribbon from the
    ORIGINAL photo size and passes it in (ribbon_from_dims + localise(..., ribbon_override)).
  * extra_ports: the splice_connector masks from RF-DETR are injected as port instances.
    They are blanked from the stage-2 input, so OneFormer cannot re-detect them; RF-DETR's
    masks are also better localised than OneFormer's port class, which fires on tray plastic.

Module-level parameters of the original are collected in DetParams so several detectors can
coexist and nothing is hidden in globals.
"""
from __future__ import annotations

import gc
import inspect
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from runtime import RuntimeProfile, optimize_module


# ══════════════════════════════════════════════════════════════════════════════════════════
#  parameters (defaults = the v6.4 cell; the checkpoint bundle overrides them)
# ══════════════════════════════════════════════════════════════════════════════════════════
@dataclass
class DetParams:
    class_names: List[str] = field(default_factory=lambda: ["fiber", "port"])
    io_names:    List[str] = field(default_factory=lambda: ["fiber_in", "fiber_out"])
    hf_checkpoint: str = "shi-labs/oneformer_ade20k_swin_tiny"
    num_queries: int = 50
    num_classes: int = 2
    no_object_weight: float = 0.1
    task_mode: str = "instance"
    train_num_points: int = 25088
    importance_sample_ratio: float = 0.9
    oversample_ratio: float = 3.0
    tile: int = 1024
    overlap_frac: float = 0.25
    batch_tiles: int = 8          # halved automatically on CUDA OOM
    min_tile_pix: int = 60
    skip_empty_tiles: bool = True
    tile_min_roi_px: int = 2000
    ribbon_scale: float = 0.0013
    ribbon_min: int = 3
    ribbon_max: int = 40
    score_thr: float = 0.30
    mask_thr: float = 0.50
    merge_gap_frac: float = 0.40
    merge_angle_deg: float = 25.0
    merge_iou: float = 0.50
    merge_iom: float = 0.15
    merge_seam_slack: float = 1.2
    tangent_px: int = 15
    max_len: float = 2300.0
    merge_anywhere: bool = True
    port_min_aspect: float = 3.0
    port_max_area_x: float = 4.0
    port_bind_slack: float = 0.6
    io_sign: int = +1
    ade_mean: np.ndarray = field(default_factory=lambda: np.array([123.675, 116.280, 103.530], np.float32) / 255.0)
    ade_std:  np.ndarray = field(default_factory=lambda: np.array([58.395, 57.120, 57.375], np.float32) / 255.0)
    amp_bf16: bool = True

    @property
    def fiber_id(self) -> int: return self.class_names.index("fiber")
    @property
    def port_id(self) -> int: return self.class_names.index("port")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  helpers
# ══════════════════════════════════════════════════════════════════════════════════════════
class UF:
    def __init__(s, n): s.p = list(range(n))
    def find(s, a):
        while s.p[a] != a: s.p[a] = s.p[s.p[a]]; a = s.p[a]
        return a
    def union(s, a, b):
        ra, rb = s.find(a), s.find(b)
        if ra != rb: s.p[rb] = ra


def dynamic_grid(W, H, tile, of):
    def ax(L):
        if L <= tile: return [0]
        s0 = max(1, int(round(tile * (1 - of)))); k = max(2, math.ceil((L - tile) / s0) + 1); st = (L - tile) / (k - 1)
        return [int(round(j * st)) for j in range(k)]
    return ax(W), ax(H)


def skel_path(mask, off=(0, 0)):
    sk = skeletonize(mask.astype(bool)); pts = np.argwhere(sk)
    if len(pts) < 2:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0: return None
        p = np.stack([xs + off[0], ys + off[1]], 1).astype(np.float32)
        return p[[0, -1]] if len(p) > 1 else None
    adj = defaultdict(list)
    for a, b in cKDTree(pts).query_pairs(1.5): adj[a].append(b); adj[b].append(a)
    def bfs(s):
        seen = {s: None}; q = [s]; last = s
        while q:
            nq = []
            for u in q:
                last = u
                for v in adj[u]:
                    if v not in seen: seen[v] = u; nq.append(v)
            q = nq
        return last, seen
    u, _ = bfs(0); v, par = bfs(u)
    chain = []; cur = v
    while cur is not None: chain.append(cur); cur = par[cur]
    return pts[chain][:, ::-1].astype(np.float32) + np.asarray(off, np.float32)


def endpoint_tangent(path, which, k):
    seg = path[:k] if which == 0 else path[-k:][::-1]
    if len(seg) < 2: return np.array([1., 0.], np.float32)
    d = seg[-1] - seg[0]; n = np.linalg.norm(d)
    return (d / n).astype(np.float32) if n > 1e-6 else np.array([1., 0.], np.float32)


def path_len(p): return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if p is not None and len(p) > 1 else 0.0


def ribbon_from_dims(W: int, H: int, P: DetParams) -> int:
    return int(np.clip(round(P.ribbon_scale * max(W, H)), P.ribbon_min, P.ribbon_max))


# ══════════════════════════════════════════════════════════════════════════════════════════
#  checkpoint loading (bundle -> DetParams, key alignment)
# ══════════════════════════════════════════════════════════════════════════════════════════
_SKIP = ("relative_position_index", "attn_mask", "position_ids", "num_batches_tracked")
_ATTN = [("attention.self.relative_position_bias_table", "attention.relative_position_bias_table"),
         ("attention.self.query", "attention.q_proj"), ("attention.self.key", "attention.k_proj"),
         ("attention.self.value", "attention.v_proj"), ("attention.output.dense", "attention.o_proj")]
_BLOCK = [("intermediate.dense", "mlp.fc1"), (".output.dense", ".mlp.fc2"),
          ("layernorm_before", "layer_norm1"), ("layernorm_after", "layer_norm2")]


def _strip(k):
    for p in ("module.", "_orig_mod."):
        while k.startswith(p): k = k[len(p):]
    return k


def _apply(k, rules, fwd):
    for x, y in rules:
        a, b = (x, y) if fwd else (y, x)
        if a in k: return k.replace(a, b, 1)
    return k


def _cands(mk):
    seen, out = set(), []
    for b in (mk, mk[6:] if mk.startswith("model.") else "model." + mk):
        for sw in (0, 1, 2):
            k1 = b
            if sw == 1: k1 = b.replace("pixel_level_module.encoder.", "pixel_level_module.encoder.swin.", 1)
            elif sw == 2: k1 = b.replace("pixel_level_module.encoder.swin.", "pixel_level_module.encoder.", 1)
            for fwd in (True, False):
                for at in (False, True):
                    k2 = _apply(k1, _ATTN, fwd) if at else k1
                    for bl in (False, True):
                        k3 = _apply(k2, _BLOCK, fwd) if bl and "attention" not in k2 else k2
                        if k3 not in seen: seen.add(k3); out.append(k3)
    return out


def load_oneformer(ckpt_path: str, device: torch.device, log=print, load_retries: int = 3):
    from transformers import OneFormerConfig, OneFormerForUniversalSegmentation
    try:
        from retry_utils import load_retry
    except ImportError:                              # helper missing -> single attempt
        def load_retry(attempts=1, exc=(), log=print): return lambda f: f
    _hub = load_retry(load_retries, log=log)          # transient hub / file errors only

    def _from_hub(fn, name):
        # Cache first: behind a corporate proxy every HEAD request to huggingface.co fails on
        # SSL and is retried 5x with backoff (~200 s) before transformers falls back to the
        # cache anyway. local_files_only skips that entirely once the files are cached.
        try:
            return fn(name, local_files_only=True)
        except Exception:
            log(f"   {name}: not in the local HF cache yet - fetching online (one-time)")
            return _hub(fn)(name)
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), f"detector checkpoint not found: {ckpt_path}"
    assert "seed" not in ckpt_path.name.lower() and "crossfibre" not in ckpt_path.name.lower(), (
        f"{ckpt_path.name} looks like the COLOUR model; the detector must be fiber_oneformer_best.pt")
    with open(ckpt_path, "rb") as fh: _h = fh.read(4)
    assert _h == b"PK\x03\x04" or _h[:2] == b"\x80\x02", f"{ckpt_path.name} is not a complete torch checkpoint"
    log(f"detector checkpoint: {ckpt_path.name} ({ckpt_path.stat().st_size / 1e6:.0f} MB)")
    raw_ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    P = DetParams()
    if isinstance(raw_ck, dict) and "model" in raw_ck and isinstance(raw_ck["model"], dict):
        sd = raw_ck["model"]
        P.hf_checkpoint = raw_ck.get("hf_checkpoint", P.hf_checkpoint)
        P.class_names   = list(raw_ck.get("class_names", P.class_names))
        P.io_names      = list(raw_ck.get("io_names", P.io_names))
        P.num_queries   = raw_ck.get("num_queries", P.num_queries)
        P.num_classes   = raw_ck.get("num_classes", P.num_classes)
        P.no_object_weight = raw_ck.get("no_object_weight", P.no_object_weight)
        P.tile          = raw_ck.get("tile", P.tile)
        P.overlap_frac  = raw_ck.get("overlap_frac", P.overlap_frac)
        P.ribbon_scale  = raw_ck.get("ribbon_scale", P.ribbon_scale)
        P.ribbon_min    = raw_ck.get("ribbon_min", P.ribbon_min)
        P.ribbon_max    = raw_ck.get("ribbon_max", P.ribbon_max)
        P.score_thr     = raw_ck.get("score_thr", P.score_thr)
        P.mask_thr      = raw_ck.get("mask_thr", P.mask_thr)
        P.task_mode     = raw_ck.get("task_mode", P.task_mode)
        if raw_ck.get("ade_mean"): P.ade_mean = np.array(raw_ck["ade_mean"], np.float32)
        if raw_ck.get("ade_std"):  P.ade_std  = np.array(raw_ck["ade_std"], np.float32)
        mg = raw_ck.get("merge", {}) or {}
        P.merge_gap_frac   = mg.get("gap_frac", P.merge_gap_frac)
        P.merge_angle_deg  = mg.get("angle_deg", P.merge_angle_deg)
        P.merge_iou        = mg.get("iou", P.merge_iou)
        P.merge_iom        = mg.get("iom", P.merge_iom)
        P.merge_seam_slack = mg.get("seam_slack", P.merge_seam_slack)
        P.tangent_px       = mg.get("tangent_px", P.tangent_px)
        P.max_len          = float(mg.get("max_len", P.max_len))
        log(f"bundle: epoch {raw_ck.get('epoch')} iter {raw_ck.get('iter')} best_f1 {raw_ck.get('best_f1')} | base {P.hf_checkpoint}")
    else:
        sd = raw_ck; log("plain state_dict - using the built-in defaults")

    sd = {_strip(k): v for k, v in sd.items()}
    assert not any(k.startswith(("backbone.stages", "head.", "fam_head.")) for k in sd), (
        "this checkpoint has ConvNeXt/CrossFibre keys - it's the colour model, not OneFormer")
    bad = [k for k, v in sd.items() if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all()]
    assert not bad, f"{len(bad)} NaN tensors, e.g. {bad[:3]}"
    sd = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in sd.items()}

    cfg = _from_hub(OneFormerConfig.from_pretrained, P.hf_checkpoint)
    cfg.num_labels = P.num_classes; cfg.num_queries = P.num_queries; cfg.no_object_weight = P.no_object_weight
    cfg.id2label = {i: n for i, n in enumerate(P.class_names)}; cfg.label2id = {n: i for i, n in enumerate(P.class_names)}
    for k, v in [("train_num_points", P.train_num_points), ("importance_sample_ratio", P.importance_sample_ratio),
                 ("oversample_ratio", P.oversample_ratio)]:
        if hasattr(cfg, k): setattr(cfg, k, v)
    if getattr(cfg, "contrastive_temperature", None) is None: cfg.contrastive_temperature = 0.07
    model = OneFormerForUniversalSegmentation(cfg)
    if hasattr(model, "model") and hasattr(model.model, "is_training"): model.model.is_training = False

    msd = model.state_dict()
    _mk = {k: v for k, v in msd.items() if not k.endswith(_SKIP)}
    _ck = {k: v for k, v in sd.items() if torch.is_tensor(v) and not k.endswith(_SKIP)}
    aligned, used, left = {}, set(), []
    for k, v in _mk.items():
        hit = next((c for c in _cands(k) if torch.is_tensor(_ck.get(c)) and tuple(_ck[c].shape) == tuple(v.shape)
                    and c not in used), None)
        if hit is None: left.append(k)
        else: aligned[k] = _ck[hit]; used.add(hit)
    log(f"detector align pass 1: {len(aligned)} matched | {len(left)} left")
    if left:
        def _sig(k, s): return (tuple(int(t) for t in k.split(".") if t.isdigit()), k.split(".")[-1], tuple(s))
        gm, gc_ = defaultdict(list), defaultdict(list)
        for k in left: gm[_sig(k, _mk[k].shape)].append(k)
        for k, v in _ck.items():
            if k not in used: gc_[_sig(k, v.shape)].append(k)
        n2 = 0
        for s, mks in gm.items():
            cks = gc_.get(s, [])
            if len(cks) == len(mks):
                for a, b in zip(mks, cks): aligned[a] = _ck[b]; used.add(b); n2 += 1
        left = [k for k in left if k not in aligned]
        log(f"detector align pass 2: {n2} matched | {len(left)} left")
    assert not left, f"{len(left)} detector tensors unaligned, e.g. {left[:3]}"
    miss, _unexp = model.load_state_dict(aligned, strict=False)
    miss = [k for k in miss if not k.endswith(_SKIP)]
    assert not miss, f"{len(miss)} still missing, e.g. {miss[:3]}"
    del sd, aligned, _mk, _ck, raw_ck; gc.collect()

    task = None
    if "task_inputs" in set(inspect.signature(type(model).forward).parameters):
        from transformers import AutoProcessor
        _p = _from_hub(AutoProcessor.from_pretrained, P.hf_checkpoint)
        task = _p.tokenizer([f"the task is {P.task_mode}"], padding="max_length", max_length=77,
                            truncation=True, return_tensors="pt").input_ids
    model.to(device).eval()
    log(f"detector ready | {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M params | "
        f"tile {P.tile} | score_thr {P.score_thr} | mask_thr {P.mask_thr}")
    return model, P, task


# ══════════════════════════════════════════════════════════════════════════════════════════
#  geometry: dedup / merge / spacing / ports / io
# ══════════════════════════════════════════════════════════════════════════════════════════
def _dedup(dets, P: DetParams):
    n = len(dets); uf = UF(n)
    for i in range(n):
        ax0, ay0, ax1, ay1 = dets[i]["bbox"]
        for j in range(i + 1, n):
            if dets[i]["cls"] != dets[j]["cls"]: continue
            bx0, by0, bx1, by1 = dets[j]["bbox"]
            ox0, oy0 = max(ax0, bx0), max(ay0, by0); ox1, oy1 = min(ax1, bx1), min(ay1, by1)
            if ox1 <= ox0 or oy1 <= oy0: continue
            ma = dets[i]["mask"][oy0 - ay0:oy1 - ay0, ox0 - ax0:ox1 - ax0]
            mb = dets[j]["mask"][oy0 - by0:oy1 - by0, ox0 - bx0:ox1 - bx0]
            inter = int(np.logical_and(ma, mb).sum())
            if inter == 0: continue
            sa, sb = int(dets[i]["mask"].sum()), int(dets[j]["mask"].sum())
            if inter / max(sa + sb - inter, 1) > P.merge_iou or inter / max(min(sa, sb), 1) > P.merge_iom: uf.union(i, j)
    comp = defaultdict(list)
    for i in range(n): comp[uf.find(i)].append(i)
    out = []
    for _, ix in comp.items():
        xs0 = min(dets[i]["bbox"][0] for i in ix); ys0 = min(dets[i]["bbox"][1] for i in ix)
        xs1 = max(dets[i]["bbox"][2] for i in ix); ys1 = max(dets[i]["bbox"][3] for i in ix)
        canvas = np.zeros((ys1 - ys0, xs1 - xs0), bool); sc = 0.0
        for i in ix:
            b0, c0, b1, c1 = dets[i]["bbox"]; canvas[c0 - ys0:c1 - ys0, b0 - xs0:b1 - xs0] |= dets[i]["mask"]; sc = max(sc, dets[i]["score"])
        out.append({"cls": dets[ix[0]]["cls"], "score": sc, "bbox": (xs0, ys0, xs1, ys1), "mask": canvas, "parts": len(ix)})
    return out


def merge_detections(dets, ribbon_px, spacing_p10, W, H, P: DetParams):
    if not dets: return []
    dets = _dedup(dets, P)
    for d in dets: d["path"] = skel_path(d["mask"], (d["bbox"][0], d["bbox"][1]))
    dets = [d for d in dets if d["path"] is not None and len(d["path"]) >= 2]
    if not dets: return []
    FIBER_ID = P.fiber_id
    spacing = spacing_p10 if spacing_p10 and spacing_p10 > 0 else 2.5 * ribbon_px
    gap = max(4.0, P.merge_gap_frac * spacing); tight = 0.5 * spacing
    xs, ys = dynamic_grid(W, H, P.tile, P.overlap_frac)
    seam_band = P.merge_seam_slack * P.overlap_frac * P.tile
    seams_x = np.array(sorted(set([x for x in xs] + [x + P.tile for x in xs])), float)
    seams_y = np.array(sorted(set([y for y in ys] + [y + P.tile for y in ys])), float)
    near_seam = lambda p: (np.abs(seams_x - p[0]).min() < seam_band) or (np.abs(seams_y - p[1]).min() < seam_band)
    eps = []
    for di, d in enumerate(dets):
        for w in (0, 1):
            pt = d["path"][0] if w == 0 else d["path"][-1]
            eps.append({"det": di, "w": w, "pt": pt, "t": endpoint_tangent(d["path"], w, P.tangent_px)})
    E = len(eps); BIG, UNMATCHED = 1e6, 1.0
    C = np.full((E, E), BIG); cosmax = math.cos(math.radians(P.merge_angle_deg))
    Pts = np.stack([e["pt"] for e in eps]) if E else np.zeros((0, 2))
    for i, j in (cKDTree(Pts).query_pairs(gap) if E else set()):
        a, b = eps[i], eps[j]
        if a["det"] == b["det"] or dets[a["det"]]["cls"] != dets[b["det"]]["cls"] or dets[a["det"]]["cls"] != FIBER_ID: continue
        dv = b["pt"] - a["pt"]; dist = float(np.linalg.norm(dv))
        if not (near_seam((a["pt"] + b["pt"]) / 2) or (P.merge_anywhere and dist < tight)): continue
        ang = float(-(a["t"] @ b["t"]))
        if ang < cosmax: continue
        if dist > 1e-3:
            u = dv / dist
            if float(a["t"] @ u) > -cosmax or float(b["t"] @ u) < cosmax: continue
        C[i, j] = C[j, i] = 0.6 * (dist / gap) + 0.4 * ((1 - ang) / (1 - cosmax + 1e-9))
    M = np.full((2 * E, 2 * E), BIG); M[:E, :E] = C
    M[:E, E:] = np.where(np.eye(E, dtype=bool), UNMATCHED, BIG); M[E:, :E] = np.where(np.eye(E, dtype=bool), UNMATCHED, BIG); M[E:, E:] = 0.0
    r, c = linear_sum_assignment(M); partner = {}
    for i, j in zip(r, c):
        if i < E and j < E and M[i, j] < UNMATCHED: partner[i] = j; partner[j] = i
    nbr = defaultdict(list)
    for i, j in partner.items():
        if i < j: a, b = eps[i]["det"], eps[j]["det"]; nbr[a].append(b); nbr[b].append(a)
    visited, chains = set(), []
    order = [d for d in range(len(dets)) if len(nbr[d]) <= 1] + list(range(len(dets)))
    for s in order:
        if s in visited: continue
        chain = []; prev = None; cur = s
        while cur is not None and cur not in visited:
            visited.add(cur); chain.append(cur)
            nxt = next((v for v in nbr[cur] if v != prev and v not in visited), None); prev, cur = cur, nxt
        acc, run = [], 0.0
        for k in chain:
            L = path_len(dets[k]["path"])
            if run + L > P.max_len and acc: chains.append(acc); acc, run = [], 0.0
            acc.append(k); run += L
        if acc: chains.append(acc)
    out = []
    for ix in chains:
        xs0 = min(dets[i]["bbox"][0] for i in ix); ys0 = min(dets[i]["bbox"][1] for i in ix)
        xs1 = max(dets[i]["bbox"][2] for i in ix); ys1 = max(dets[i]["bbox"][3] for i in ix)
        canvas = np.zeros((ys1 - ys0, xs1 - xs0), bool); votes = np.zeros(P.num_classes); sc = 0.0
        for i in ix:
            d = dets[i]; b0, c0, b1, c1 = d["bbox"]; canvas[c0 - ys0:c1 - ys0, b0 - xs0:b1 - xs0] |= d["mask"]
            votes[d["cls"]] += d["score"] * d["mask"].sum(); sc = max(sc, d["score"])
        path = skel_path(canvas, (xs0, ys0))
        out.append({"cls": int(votes.argmax()), "score": sc, "bbox": (xs0, ys0, xs1, ys1), "mask": canvas,
                    "path": path if path is not None else np.zeros((0, 2), np.float32),
                    "n_parts": sum(dets[i].get("parts", 1) for i in ix), "n_chained": len(ix),
                    "length": path_len(path), "source": "oneformer"})
    return out


def estimate_spacing(dets, fallback, P: DetParams):
    pts, own = [], []
    for gi, d in enumerate(dets):
        if d["cls"] != P.fiber_id: continue
        p = skel_path(d["mask"], (d["bbox"][0], d["bbox"][1]))
        if p is None or len(p) < 3: continue
        step = max(1, len(p) // 12)
        for q in p[::step]: pts.append(q); own.append(gi)
    if len(pts) < 40: return fallback
    pts = np.asarray(pts, float); own = np.asarray(own)
    dd, ii = cKDTree(pts).query(pts, k=min(10, len(pts)))
    best = [dd[r][own[ii[r]] != own[r]][0] for r in range(len(pts)) if (own[ii[r]] != own[r]).any()]
    return float(np.percentile(best, 10)) if best else fallback


def filter_ports(insts, P: DetParams, log=print):
    """Drop OneFormer port blobs that are not sleeve-shaped. Injected RF-DETR ports are kept."""
    ports = [k for k in insts if k["cls"] == P.port_id and k.get("source") != "rfdetr"]
    if len(ports) < 3: return insts
    dims = np.array([[k["bbox"][2] - k["bbox"][0], k["bbox"][3] - k["bbox"][1]] for k in ports], float)
    aspect = dims.max(1) / np.maximum(dims.min(1), 1); area = np.array([int(k["mask"].sum()) for k in ports], float)
    keep = (aspect >= P.port_min_aspect) & (area <= P.port_max_area_x * np.median(area))
    if (~keep).any(): log(f"   port filter: dropped {int((~keep).sum())}/{len(ports)} non-sleeve blobs")
    kept = {id(k) for k, ok in zip(ports, keep) if ok}
    return [k for k in insts if k["cls"] != P.port_id or k.get("source") == "rfdetr" or id(k) in kept]


def _dist_to_box(pt, bbox, slack):
    x0, y0, x1, y1 = bbox
    dx = max(x0 - slack - pt[0], 0.0, pt[0] - x1 - slack); dy = max(y0 - slack - pt[1], 0.0, pt[1] - y1 - slack)
    return math.hypot(dx, dy)


def assign_io(insts, W, H, P: DetParams):
    FIBER_ID, PORT_ID = P.fiber_id, P.port_id
    sign = P.io_sign
    ports = [k for k in insts if k["cls"] == PORT_ID]
    pc = np.array([[(k["bbox"][0] + k["bbox"][2]) / 2, (k["bbox"][1] + k["bbox"][3]) / 2] for k in ports], float) if ports else np.zeros((0, 2))
    if len(pc) >= 2: ctr = pc.mean(0); axis = np.linalg.svd(pc - ctr, full_matrices=False)[2][0]
    else: ctr, axis = np.array([W / 2., H / 2.]), np.array([0., 1.])
    for k in insts:
        k["io"] = None; k["port"] = None; k["port_dist"] = None
        if k["cls"] != FIBER_ID: continue
        p = np.asarray(k.get("path", []), float)
        if len(p) < 2: continue
        if ports:
            best = (None, np.inf, 0)
            for w_end, pt in ((0, p[0]), (1, p[-1])):
                for pi, pk in enumerate(ports):
                    short = max(1.0, min(pk["bbox"][2] - pk["bbox"][0], pk["bbox"][3] - pk["bbox"][1]))
                    d = _dist_to_box(pt, pk["bbox"], P.port_bind_slack * short)
                    if d < best[1]: best = (pi, d, w_end)
            pi, dmin, w_end = best; pk = ports[pi]
            short = max(1.0, min(pk["bbox"][2] - pk["bbox"][0], pk["bbox"][3] - pk["bbox"][1]))
            k["port_dist"] = round(float(dmin), 1)
            if dmin <= P.port_bind_slack * short: k["port"] = pi
            anchor = p[0] if w_end == 0 else p[-1]
            far = p[min(len(p) - 1, P.tangent_px)] if w_end == 0 else p[max(0, len(p) - 1 - P.tangent_px)]
            ref = pc[pi]
        else:
            anchor, far, ref = p[0], p[min(len(p) - 1, P.tangent_px)], ctr
        v = far - anchor; cross = float(axis[0] * v[1] - axis[1] * v[0])
        if abs(cross) < 1e-6: off = anchor - ref; cross = float(axis[0] * off[1] - axis[1] * off[0])
        k["io"] = 0 if sign * cross < 0 else 1
    if ports:
        order = np.argsort(pc @ axis) if len(pc) >= 2 else np.arange(len(ports))
        num = {int(i): n + 1 for n, i in enumerate(order)}
        for pi, k in enumerate(ports): k["port_num"] = num[pi]
        for k in insts:
            if k["cls"] == FIBER_ID: k["port_num"] = num[k["port"]] if k.get("port") is not None else None
    return insts


# ══════════════════════════════════════════════════════════════════════════════════════════
#  the detector
# ══════════════════════════════════════════════════════════════════════════════════════════
class FibreDetector:
    def __init__(self, ckpt_path: str, device: Optional[torch.device] = None, log=print,
                 load_retries: int = 3, oom_retries: int = 4, prof: Optional[RuntimeProfile] = None):
        self.prof = prof or RuntimeProfile(device="cuda" if torch.cuda.is_available() else "cpu",
                                           autocast_dtype="bf16" if torch.cuda.is_available() else None)
        self.device = device or self.prof.torch_device
        self.log = log; self.oom_retries = oom_retries
        self.model, self.P, task = load_oneformer(ckpt_path, self.device, log, load_retries)
        # RUNTIME: Swin/OneFormer is nn.Linear-heavy -> dynamic INT8 (opt-in) / ipex / compile.
        # Weights on disk are untouched; only the in-memory module is transformed.
        self.model = optimize_module(self.model, self.prof, "stage 2 OneFormer", conv_heavy=False,
                                     compile_dynamic=False, log=log)
        if not (self.P.amp_bf16): self.prof.autocast_dtype = None
        # LATENCY: task tokens live on the device once; repeated per batch size below
        self.task = task.to(self.device) if task is not None else None
        self._task_cache = {}
        self.last_tile_stats = (0, 0)      # (tiles run, tiles in grid) of the last call

    # ---- knobs from PipelineConfig ---------------------------------------------------------
    def apply_overrides(self, cfg):
        P = self.P
        if cfg.score_thr is not None: P.score_thr = float(cfg.score_thr)
        if cfg.mask_thr is not None:  P.mask_thr  = float(cfg.mask_thr)
        if cfg.max_len is not None:   P.max_len   = float(cfg.max_len)
        P.io_sign = int(cfg.io_sign); P.merge_anywhere = bool(cfg.merge_anywhere)
        P.port_min_aspect = float(cfg.port_min_aspect); P.port_max_area_x = float(cfg.port_max_area_x)
        P.port_bind_slack = float(cfg.port_bind_slack)
        P.skip_empty_tiles = bool(cfg.skip_empty_tiles); P.tile_min_roi_px = int(cfg.tile_min_roi_px)
        P.batch_tiles = max(1, int(cfg.batch_tiles) if cfg.batch_tiles else int(self.prof.batch_tiles))

    def warmup(self):
        """One forward of a single tile: cudnn autotune + CUDA context, off the first photo's clock."""
        try:
            self.infer_tiled(np.zeros((self.P.tile, self.P.tile, 3), np.uint8))
        except Exception as e:
            self.log(f"stage 2 warmup skipped: {e}")

    def _task_for(self, n):
        if self.task is None: return None
        if n not in self._task_cache: self._task_cache[n] = self.task.repeat(n, 1)
        return self._task_cache[n]

    # ---- tiled inference -----------------------------------------------------------------
    def _tile_origins(self, W, H, roi_mask):
        P = self.P; tile = P.tile
        xs, ys = dynamic_grid(W, H, tile, P.overlap_frac)
        origins = [(x0, y0) for y0 in ys for x0 in xs]
        n_grid = len(origins)
        if roi_mask is not None and P.skip_empty_tiles:
            # LATENCY: an integral image answers "how many ROI pixels in this tile" in O(1);
            # tiles that are pure fill (outside the cassette, or the corner padding) are skipped.
            integ = cv2.integral(roi_mask.astype(np.uint8))
            def cnt(x0, y0):
                x1, y1 = min(x0 + tile, W), min(y0 + tile, H)
                return int(integ[y1, x1] - integ[y0, x1] - integ[y1, x0] + integ[y0, x0])
            origins = [o for o in origins if cnt(*o) >= max(P.min_tile_pix, P.tile_min_roi_px)]
        self.last_tile_stats = (len(origins), n_grid)
        return origins

    def _forward_chunk(self, chunk, norm, H, W):
        """Run one batch of tiles. norm is the whole image already normalised (H, W, 3) float32."""
        P = self.P; tile = P.tile
        batch = np.zeros((len(chunk), 3, tile, tile), np.float32); shapes = []
        for bi, (x0, y0) in enumerate(chunk):
            c = norm[y0:min(y0 + tile, H), x0:min(x0 + tile, W)]
            ph, pw = c.shape[:2]; shapes.append((ph, pw))
            batch[bi, :, :ph, :pw] = c.transpose(2, 0, 1)
        px = torch.from_numpy(batch)
        if self.device.type == "cuda": px = px.pin_memory()
        px = px.to(self.device, non_blocking=True)
        kw = {"pixel_values": px}
        t = self._task_for(len(chunk))
        if t is not None: kw["task_inputs"] = t
        with self.prof.autocast():                       # cuda bf16 | cpu bf16 (AMX) | off
            o = self.model(**kw)
        cls_all = o.class_queries_logits.float().softmax(-1)                          # (B, Q, C+1)
        sc_all, lb_all = cls_all[:, :, :P.num_classes].max(-1)                        # (B, Q)
        keep_all = sc_all > P.score_thr
        dets = []
        if keep_all.any():
            mlg = o.masks_queries_logits.float()                                      # (B, Q, h, w)
            for bi, (x0, y0) in enumerate(chunk):
                keep = torch.nonzero(keep_all[bi]).flatten()
                if len(keep) == 0: continue
                ph, pw = shapes[bi]
                # LATENCY: upsample + threshold only the kept queries, on the device, then one transfer
                m_up = F.interpolate(mlg[bi, keep][None], (tile, tile), mode="bilinear", align_corners=False)[0]
                pk = (m_up.sigmoid() > P.mask_thr)
                pk[:, ph:, :] = False; pk[:, :, pw:] = False
                sums = pk.flatten(1).sum(1)
                ok = torch.nonzero(sums >= P.min_tile_pix).flatten()
                if len(ok) == 0: continue
                pk_np = pk[ok].cpu().numpy(); sc_np = sc_all[bi, keep[ok]].cpu().numpy(); lb_np = lb_all[bi, keep[ok]].cpu().numpy()
                for i in range(len(ok)):
                    m = pk_np[i]; yy, xx_ = np.nonzero(m)
                    b_0, b_1 = int(xx_.min()), int(xx_.max()) + 1; c_0, c_1 = int(yy.min()), int(yy.max()) + 1
                    dets.append({"cls": int(lb_np[i]), "score": float(sc_np[i]),
                                 "bbox": (x0 + b_0, y0 + c_0, x0 + b_1, y0 + c_1), "tile": (x0, y0),
                                 "mask": m[c_0:c_1, b_0:b_1].copy()})
            del mlg
        del o, cls_all, px
        return dets

    @torch.inference_mode()
    def infer_tiled(self, img_rgb: np.ndarray, roi_mask: Optional[np.ndarray] = None):
        try:
            from retry_utils import oom_retrying
        except ImportError:
            def oom_retrying(attempts, on_oom, log=print):
                class _Once:
                    def __enter__(self): return self
                    def __exit__(self, *a): return False
                return [_Once()]
        P = self.P
        img = np.ascontiguousarray(img_rgb); H, W = img.shape[:2]
        origins = self._tile_origins(W, H, roi_mask)
        # LATENCY: normalise the whole image once (vectorised) instead of per tile
        norm = ((img.astype(np.float32) / 255.0 - P.ade_mean) / P.ade_std)
        dets, b0 = [], 0
        def _shrink(): P.batch_tiles = max(1, P.batch_tiles // 2); self.log(f"   batch_tiles -> {P.batch_tiles}")
        while b0 < len(origins):
            out, n = [], 0
            for attempt in oom_retrying(self.oom_retries, _shrink, self.log):
                with attempt:
                    chunk = origins[b0:b0 + P.batch_tiles]        # re-sliced with the current batch size
                    out = self._forward_chunk(chunk, norm, H, W); n = len(chunk)
            dets += out; b0 += n
        return dets, (W, H)

    # ---- full localisation on one (ROI) image ----------------------------------------------
    def localise(self, img_rgb: np.ndarray, ribbon_override: Optional[int] = None,
                 extra_ports: Optional[List[Tuple[Tuple[int, int, int, int], np.ndarray, float]]] = None,
                 ports_from: str = "sleeves", roi_mask: Optional[np.ndarray] = None, cfg=None):
        """
        img_rgb        : the stage-1 ROI crop (filled)
        ribbon_override: ribbon width computed from the ORIGINAL photo dims
        extra_ports    : [(bbox, local_mask, score, conn_idx)] RF-DETR splice_connector regions
                         (crop coords) - the splice HOLDER, used only as a fallback port
        ports_from     : "sleeves"  OneFormer sleeve rows + lattice fit (fit_port_stack)   <- default
                         "auto"     sleeves; if no stack can be fitted fall back to the RF-DETR region
                         "rfdetr"   the RF-DETR splice_connector region is the (single) port
                         "legacy"   OneFormer ports with the old aspect/area filter + SVD numbering
        roi_mask       : bool mask (crop frame) used to skip tiles that contain no cassette pixels
        returns (insts, n_tile_dets, ribbon, spacing, stack)
        """
        insts, n_dets, ribbon, spacing, (W, H) = self.detect(img_rgb, ribbon_override, roi_mask)
        insts, stack = self.finish_ports(insts, W, H, extra_ports, ports_from, cfg)
        return insts, n_dets, ribbon, spacing, stack

    def detect(self, img_rgb: np.ndarray, ribbon_override: Optional[int] = None, roi_mask: Optional[np.ndarray] = None):
        """Model part only: tiled OneFormer -> dedup -> fibre chaining. Frame-agnostic."""
        P = self.P
        dets, (W, H) = self.infer_tiled(img_rgb, roi_mask)
        ribbon = int(ribbon_override) if ribbon_override else ribbon_from_dims(W, H, P)
        spacing = estimate_spacing(dets, 2.5 * ribbon, P)
        insts = merge_detections(dets, ribbon, spacing, W, H, P)
        return insts, len(dets), ribbon, spacing, (W, H)

    def finish_ports(self, insts, W, H, extra_ports=None, ports_from: str = "sleeves", cfg=None):
        """Geometry part: sleeve lattice / port numbering / IN-OUT. Run in the STANDARD frame."""
        from port_stack import assign_io_stack, fit_port_stack
        P = self.P; stack = None

        def _inject_rfdetr(insts):
            insts = [k for k in insts if k["cls"] != P.port_id]
            for ep in (extra_ports or []):
                bbox, m, sc = ep[0], ep[1], ep[2]; ci = ep[3] if len(ep) > 3 else None
                insts.append({"cls": P.port_id, "score": float(sc), "bbox": tuple(int(v) for v in bbox),
                              "mask": m.astype(bool), "path": np.zeros((0, 2), np.float32),
                              "n_parts": 1, "n_chained": 1, "length": 0.0, "source": "rfdetr", "conn_idx": ci})
            return insts

        if ports_from in ("sleeves", "auto"):
            # the horizontal sleeve rows ARE the ports: geometric gates + lattice numbering
            insts, stack = fit_port_stack(insts, P, cfg, self.log)
            if not stack.get("ok") and ports_from == "auto" and extra_ports:
                self.log("   port stack: no sleeve lattice -> falling back to the RF-DETR splice_connector region as one port")
                insts = _inject_rfdetr(insts); stack = None
            insts = assign_io_stack(insts, W, H, P, stack)
        elif ports_from == "rfdetr":
            insts = _inject_rfdetr(insts)
            insts = assign_io(insts, W, H, P)
        else:                                                    # "legacy"
            insts = filter_ports(insts, P, self.log)
            insts = assign_io(insts, W, H, P)
        return insts, stack