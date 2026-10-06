"""
STAGE 3 — CrossFibre ConvNeXt + CoordAtt colour model  (port of FIBRE PIPELINE v6.4, stage 2)
plus the splice consistency rule, port tally, text report and the final overlay renderer.

The colour model runs on the same filled ROI crop that stage 2 saw, so the fibre masks from
stage 2 line up pixel-for-pixel with the image the crops are cut from.
"""
from __future__ import annotations

import gc
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch

# ── v6.4 defaults, used only when inference_config.json is absent / pre-v6.3 ──────────────
V64_DEFAULT_CFG = {
    "classes": ["aqua", "black", "blue", "brown", "green", "orange", "pink", "red", "slate",
                "violet", "white", "yellow"],
    "families": ["achromatic", "cool", "green", "magenta", "warm", "yellow"],
    "positions": ["IN", "OUT", "NA"], "abstain_label": "uncertain",
    "img_size": 128, "feather_k": 5, "crop_pad_frac": 0.30, "canonical_orient": True,
    "min_crop_side": 1.0, "illum_mode": "shades_of_grey", "color_constancy": True,
    "flatten_shadow": True, "flatten_sigma_frac": 0.10, "flatten_work_px": 1024,
    "retinex_mode": "rgb", "cc_minkowski_p": 6,
    "imagenet_mean": [0.485, 0.456, 0.406], "imagenet_std": [0.229, 0.224, 0.225],
    "side_dim": 10, "use_chromaticity": True, "backbone": "convnext_tiny",
    "use_coordatt": True, "coordatt_stages": [1, 2], "coordatt_reduction": 32,
    "masked_pool": True, "masked_pool_mix": 0.75,
    "d_model": 256, "enc_layers": 3, "heads": 8, "ffn_dim": 512, "enc_dropout": 0.0,
    "chroma_abstain": 4.0, "low_conf_thr": 0.35, "use_tta": True, "version": "v6.4-defaults",
}
ACHROMATIC = {"black", "white", "slate", "grey"}

TIA_RGB = {"aqua": (140, 220, 226), "black": (40, 40, 44), "blue": (48, 70, 198), "brown": (120, 78, 50),
           "green": (38, 158, 60), "limegreen": (160, 220, 60), "orange": (232, 138, 40), "pink": (236, 150, 180),
           "rose": (236, 150, 180), "red": (206, 45, 40), "slate": (112, 120, 132), "violet": (150, 70, 190),
           "white": (240, 240, 240), "yellow": (232, 214, 58)}
ABSTAIN_RGB = (110, 110, 130); IO_COL = {0: (60, 220, 90), 1: (235, 70, 190)}; PORT_COL = (255, 200, 40)


# ══════════════════════════════════════════════════════════════════════════════════════════
#  classifier
# ══════════════════════════════════════════════════════════════════════════════════════════
class ColourClassifier:
    def __init__(self, ckpt_path: str, cfg_path: Optional[str] = None, bundle_dir: str = "_co_bundle",
                 device: Optional[torch.device] = None, log=print, load_retries: int = 3, oom_retries: int = 4,
                 prof=None):
        import crossfibre_infer as cfi
        try:
            from retry_utils import load_retry
        except ImportError:                              # helper missing -> single attempt
            def load_retry(attempts=1, exc=(), log=print): return lambda f: f
        from runtime import RuntimeProfile
        self.cfi = cfi; self.log = log; self.oom_retries = oom_retries
        self.prof = prof or RuntimeProfile(device="cuda" if torch.cuda.is_available() else "cpu",
                                           autocast_dtype="bf16" if torch.cuda.is_available() else None)
        self.device = device or self.prof.torch_device
        ckpt_path = Path(ckpt_path)
        assert ckpt_path.exists(), f"colour checkpoint not found: {ckpt_path}"
        assert "oneformer" not in ckpt_path.name.lower(), "colour_ckpt points at the detector"
        bdir = Path(bundle_dir); bdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(ckpt_path, bdir / "model.pth")

        # -- the state_dict is ground truth for the architecture: read it first --------------
        _sd = torch.load(bdir / "model.pth", map_location="cpu", weights_only=False)
        _sd = _sd.get("model", _sd) if isinstance(_sd, dict) and "model" in _sd else _sd
        _sd = {k[7:] if k.startswith("module.") else k: v for k, v in _sd.items()}
        _ca_stages = sorted({int(k.split(".")[2]) for k in _sd if k.startswith("backbone.stages.")
                             and k.split(".")[3] == "1" and k.split(".")[4] == "conv_h"})
        _from_w = {"use_coordatt": bool(_ca_stages), "coordatt_stages": _ca_stages or [1, 2],
                   "side_dim": int(_sd["side.0.weight"].shape[1]),
                   "use_chromaticity": int(_sd["side.0.weight"].shape[1]) > 6,
                   "d_model": int(_sd["proj.weight"].shape[0]),
                   "enc_layers": len({k.split(".")[2] for k in _sd if k.startswith("encoder.layers.")}),
                   "ffn_dim": int(_sd["encoder.layers.0.linear1.weight"].shape[0])}
        n_head = int(_sd["head.weight"].shape[0])
        del _sd; gc.collect()
        log(f"colour architecture read from the weights: {_from_w}")

        _cfg = None
        if cfg_path and Path(cfg_path).exists():
            _cfg = json.load(open(cfg_path))
            if "use_coordatt" not in _cfg:
                log(f"!! {Path(cfg_path).name} is a PRE-v6.3 export (no architecture keys). Ignoring it "
                    f"and using the embedded v6.4 defaults.")
                _cfg = None
        if _cfg is None:
            _cfg = dict(V64_DEFAULT_CFG); log("colour config: embedded v6.4 defaults")
        else:
            log("colour config: from inference_config.json")
        if n_head != len(_cfg["classes"]):
            log(f"!! head predicts {n_head} colours, config lists {len(_cfg['classes'])}")
        _diff = {k: (_cfg.get(k), v) for k, v in _from_w.items() if _cfg.get(k) != v}
        if _diff:
            log(f"!! config disagrees with the weights on {list(_diff)} -> weights win: {_diff}")
            _cfg.update(_from_w)
        json.dump(_cfg, open(bdir / "inference_config.json", "w"), indent=2)

        self.model, self.cfg = load_retry(load_retries, log=log)(cfi.load_bundle)(bdir, device=self.device)  # strict=True inside
        # RUNTIME: ConvNeXt is conv-heavy (channels_last) and its MLP / encoder are Linear
        # (int8-dynamic opt-in). Autocast policy for the harness follows the profile.
        from runtime import optimize_module
        self.model = optimize_module(self.model, self.prof, "stage 3 CrossFibre", conv_heavy=True,
                                     compile_dynamic=True, log=log)
        if hasattr(cfi, "set_autocast"):
            cfi.set_autocast(self.prof.amp_dtype is not None, self.prof.amp_dtype)
        else:   # older crossfibre_infer.py without the hook: keeps its built-in CUDA-only autocast
            log("stage 3: crossfibre_infer.py predates set_autocast() - CPU bf16 autocast not applied; "
                "replace it with the packaged copy to enable it")
        self.classes = list(self.cfg["classes"])
        self.abstain_label = self.cfg.get("abstain_label", "uncertain")
        self.low_conf_thr = float(self.cfg.get("low_conf_thr", 0.35))
        log(f"colour model: STRICT load OK | {len(self.classes)} classes | "
            f"CoordAtt={self.cfg.get('use_coordatt')} masked_pool={self.cfg.get('masked_pool')} "
            f"side_dim={self.cfg['side_dim']} retinex={self.cfg.get('retinex_mode')} | "
            f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f} M")

    def warmup(self):
        try:
            img = np.full((256, 256, 3), 128, np.uint8); m = np.zeros((60, 6), bool); m[:, 2:4] = True
            self.cfi.predict_from_masks(self.model, self.cfg, img, [((100, 100, 106, 160), m)], tta=False)
        except Exception as e:
            self.log(f"stage 3 warmup skipped: {e}")

    def colour_pass(self, raw_rgb: np.ndarray, fibres: List[dict], use_tta=True, max_fibres_fwd=48):
        for k in fibres:
            k.update({"colour": None, "predicted": None, "confidence": None, "abstained": True,
                      "reason": "no crop", "chroma": 0.0, "fibre_px": 0, "top3": {}, "low_conf": False})
        if not fibres: return
        try:
            from retry_utils import oom_retrying
        except ImportError:
            def oom_retrying(attempts, on_oom, log=print):
                class _Once:
                    def __enter__(self): return self
                    def __exit__(self, *a): return False
                return [_Once()]
        state = {"fwd": int(max_fibres_fwd)}
        def _shrink(): state["fwd"] = max(4, state["fwd"] // 2); self.log(f"   colour max_fibres_fwd -> {state['fwd']}")
        preds = []
        for attempt in oom_retrying(self.oom_retries, _shrink, self.log):
            with attempt:
                preds = self.cfi.predict_from_masks(self.model, self.cfg, raw_rgb,
                                                    [(k["bbox"], k["mask"]) for k in fibres],
                                                    tta=use_tta, max_fibres_fwd=state["fwd"])
        for o in preds:
            k = fibres[o["ann_index"]]; name = o["colour"]; abst = (name == self.abstain_label)
            k.update({"colour": None if abst else name.rstrip("?"), "predicted": o["predicted"],
                      "confidence": o["confidence"], "abstained": abst, "reason": o["reason"],
                      "chroma": o["chroma"], "fibre_px": o["fibre_px"], "top3": o["top3"],
                      "low_conf": name.endswith("?")})


# ══════════════════════════════════════════════════════════════════════════════════════════
#  splice rule, tally, report
# ══════════════════════════════════════════════════════════════════════════════════════════
def apply_colour_match(insts, fiber_id: int, enabled: bool = True):
    """Port k IN and OUT are the same spliced fibre -> colours must agree."""
    flags = []
    if not enabled: return flags
    by_port = defaultdict(lambda: {0: [], 1: []})
    for k in insts:
        if k["cls"] == fiber_id and k.get("port") is not None and k.get("io") is not None:
            by_port[k["port"]][k["io"]].append(k)
    for port, sides in by_port.items():
        if len(sides[0]) != 1 or len(sides[1]) != 1: continue
        a, b = sides[0][0], sides[1][0]; ca, cb = a.get("colour"), b.get("colour")
        if ca and cb and ca != cb:
            flags.append({"port": a.get("port_num"), "in": ca, "out": cb,
                          "in_conf": a.get("confidence"), "out_conf": b.get("confidence")})
            a["splice_flag"] = b["splice_flag"] = f"{ca} vs {cb}"
        elif ca and not cb: b.update({"colour": ca, "abstained": False, "reason": "resolved from IN partner"})
        elif cb and not ca: a.update({"colour": cb, "abstained": False, "reason": "resolved from OUT partner"})
    return flags


def tally_ports(insts, fiber_id: int, do_colour: bool = True):
    tally = defaultdict(lambda: {0: Counter(), 1: Counter()}); unassigned = {0: Counter(), 1: Counter()}
    for k in insts:
        if k["cls"] != fiber_id or k.get("io") is None: continue
        colour = (k.get("colour") or "unknown") if do_colour else "-"
        if k.get("port_num") is not None: tally[k["port_num"]][k["io"]][colour] += 1
        else: unassigned[k["io"]][colour] += 1
    return dict(sorted(tally.items())), unassigned


def format_port_report(tally, unassigned, flags):
    lines = []
    for num in sorted(tally):
        parts = [f"{n} {'in' if io == 0 else 'out'} {c}" for io in (0, 1)
                 for c, n in sorted(tally[num][io].items(), key=lambda t: (-t[1], t[0]))]
        fl = next((f for f in flags if f["port"] == num), None)
        lines.append(f"Port {num}: " + (", ".join(parts) if parts else "(no fibres)")
                     + (f"   !! MISMATCH {fl['in']} / {fl['out']}" if fl else ""))
    ua = [f"{n} {'in' if io == 0 else 'out'} {c}" for io in (0, 1)
          for c, n in sorted(unassigned[io].items(), key=lambda t: (-t[1], t[0]))]
    if ua: lines.append("Unassigned (no port within range): " + ", ".join(ua))
    return "\n".join(lines) if lines else "(no ports or fibres)"


# ══════════════════════════════════════════════════════════════════════════════════════════
#  post-stage connector removal
# ══════════════════════════════════════════════════════════════════════════════════════════
def remove_connector_region(insts, connector_local: np.ndarray, fiber_id: int, drop_frac: float = 0.9):
    """
    Called AFTER stage 2 (so port binding used the full fibre geometry) and BEFORE stage 3
    (so the colour crop is not contaminated by sleeve pixels). Every fibre mask loses its
    pixels inside the (dilated) connector region; a fibre that was almost entirely on the
    sleeve is a false detection and is dropped. Returns (kept_insts, dropped_insts).
    """
    if connector_local is None or not connector_local.any():
        for k in insts: k["px_in_connector"] = 0
        return insts, []
    H, W = connector_local.shape
    kept, dropped = [], []
    for k in insts:
        if k["cls"] != fiber_id:
            kept.append(k); continue
        x0, y0, x1, y1 = [int(v) for v in k["bbox"]]
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        m = k["mask"]; total = int(m.sum())
        if cx1 <= cx0 or cy1 <= cy0 or total == 0:
            k["px_in_connector"] = 0; kept.append(k); continue
        sub = connector_local[cy0:cy1, cx0:cx1]
        local = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
        inside = int((local & sub).sum()); k["px_in_connector"] = inside
        if inside >= drop_frac * total:
            k["dropped_reason"] = "inside splice connector"; dropped.append(k); continue
        if inside:
            m = m.copy(); m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] &= ~sub; k["mask"] = m
            k["length"] = float(k.get("length", 0.0)) * (1.0 - inside / total)
        kept.append(k)
    return kept, dropped


# ══════════════════════════════════════════════════════════════════════════════════════════
#  render
# ══════════════════════════════════════════════════════════════════════════════════════════
def offset_instances(insts, dx: int, dy: int):
    """Shift bboxes / paths from crop coordinates to full-image coordinates (masks stay local)."""
    out = []
    for k in insts:
        j = dict(k)
        x0, y0, x1, y1 = k["bbox"]; j["bbox"] = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
        p = k.get("path")
        j["path"] = (np.asarray(p, np.float32) + np.array([dx, dy], np.float32)) if p is not None and len(p) else p
        out.append(j)
    return out


def fibre_label_anchor(k):
    p = k.get("path")
    if p is not None and len(p) > 2: q = p[len(p) // 2]; return int(q[0]), int(q[1])
    x0, y0 = int(k["bbox"][0]), int(k["bbox"][1]); ys, xs = np.nonzero(k["mask"])
    return (x0 + int(xs.mean()), y0 + int(ys.mean())) if len(xs) else (x0, y0)


def render(img: np.ndarray, insts, tally, fiber_id: int, port_id: int, cfg, cassette_mask=None,
           connector_mask=None, fill_rgb=(128, 128, 128)):
    """
    connector_mask (full-image bool): when given, the splice-connector region is REMOVED from
    the output - its pixels are replaced (cfg.connector_out_fill) and no port fill is drawn
    over it; only a thin outline + the port label remain so the port is still identifiable.
    """
    ov = img.copy(); H, W = img.shape[:2]
    removed = connector_mask is not None and connector_mask.any()
    if removed and cfg.connector_out_fill != "keep":
        fill = (0, 0, 0) if cfg.connector_out_fill == "black" else tuple(int(v) for v in fill_rgb)
        ov[connector_mask] = np.array(fill, np.uint8)
    for k in insts:
        if k["cls"] == port_id and (not cfg.draw_ports or removed): continue   # no sleeve fill once removed
        x0, y0, x1, y1 = [int(v) for v in k["bbox"]]; x0, y0 = max(0, x0), max(0, y0); x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0: continue
        m = k["mask"][:y1 - y0, :x1 - x0]
        if k["cls"] == port_id: c = PORT_COL
        elif cfg.fill_by == "colour" and cfg.do_colour:
            c = ABSTAIN_RGB if k.get("abstained", True) else TIA_RGB.get(k.get("colour"), (120, 120, 255))
        else: c = IO_COL.get(k.get("io"), (120, 120, 255))
        sub = ov[y0:y1, x0:x1]
        sub[m] = (np.array(c) * cfg.alpha + sub[m] * (1 - cfg.alpha)).astype(np.uint8)
        if k["cls"] == port_id:                                  # sleeve bar (v6.4 overlay style)
            # the port mask can be thin at photo resolution; draw the full bar from the bbox so a
            # kept port is always a visible sleeve, then the mask contour on top
            cv2.rectangle(ov, (x0, y0), (x1 - 1, y1 - 1), tuple(int(v) for v in PORT_COL), -1)
            ov[y0:y1, x0:x1] = (np.array(PORT_COL) * cfg.alpha + img[y0:y1, x0:x1] * (1 - cfg.alpha)).astype(np.uint8)
            cv2.rectangle(ov, (x0, y0), (x1 - 1, y1 - 1), (255, 230, 120), max(1, int(min(H, W) / 1200)))
    if cassette_mask is not None and cfg.draw_cassette_outline:
        cnts, _ = cv2.findContours(cassette_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (30, 220, 30), max(2, int(min(H, W) / 700)))
    if removed and cfg.draw_ports:
        cnts, _ = cv2.findContours(connector_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, PORT_COL, max(1, int(min(H, W) / 1000)))
    fs = max(0.4, min(H, W) / 1800.0)
    if cfg.label_fibres:
        for k in insts:
            if k["cls"] != fiber_id: continue
            px, py = fibre_label_anchor(k)
            io = "" if k.get("io") is None else ("IN" if k["io"] == 0 else "OUT")
            if cfg.do_colour:
                txt = "?" if k.get("abstained", True) else (k.get("colour") or "?")
                if k.get("low_conf"): txt += "?"
                if k.get("splice_flag"): txt += " !"
            else: txt = io; io = ""
            if io: txt = f"{txt} {io}"
            for col, th in [((0, 0, 0), max(3, int(fs * 4))), ((255, 255, 255), max(1, int(fs * 2)))]:
                cv2.putText(ov, txt, (px + 5, py), cv2.FONT_HERSHEY_SIMPLEX, fs, col, th, cv2.LINE_AA)
    if cfg.label_ports and cfg.draw_ports:
        fp = max(0.5, min(H, W) / 1400.0)
        for k in insts:
            if k["cls"] != port_id or k.get("port_num") is None: continue
            x0, y0, x1, y1 = k["bbox"]; px, py = int((x0 + x1) / 2), int((y0 + y1) / 2)
            t = tally.get(k["port_num"]); nin = int(sum(t[0].values())) if t else 0; nout = int(sum(t[1].values())) if t else 0
            txt = (f"Port {k['port_num']}: {nin} in / {nout} out" if getattr(cfg, "port_label_counts", False)
                   else f"Port {k['port_num']}")                    # counts live in the report / Excel
            for col, th in [((0, 0, 0), max(4, int(fp * 5))), ((255, 255, 255), max(2, int(fp * 2)))]:
                cv2.putText(ov, txt, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, fp, col, th, cv2.LINE_AA)
    return ov