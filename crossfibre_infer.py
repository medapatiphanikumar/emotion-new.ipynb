# crossfibre_infer.py  -  self-contained inference for the cross-fibre colour model.
# GENERATED from the training notebook: every function below is the exact source that
# produced the training crops. Do not edit by hand; regenerate from the notebook.
#
# (This file is the module that FIBRE PIPELINE v6.4 embedded as CROSSFIBRE_INFER_SRC,
#  written out so the JACOPS package can import it directly.)
#
#     from crossfibre_infer import load_bundle, predict_photo, predict_from_masks
#     model, cfg = load_bundle("/path/to/export")           # strict=True load
#     out = predict_photo(model, cfg, image_rgb, coco_annotations)
#     out = predict_from_masks(model, cfg, image_rgb, [(bbox, local_bool_mask), ...])
import json, math
from pathlib import Path
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, timm

# ---- module globals; set by configure(cfg) so the copied functions see the same names --
IMG_SIZE = 128; FEATHER_K = 5; CROP_PAD_FRAC = 0.30; MIN_CROP_SIDE = 1.0
CANONICAL_ORIENT = True; ILLUM_MODE = "shades_of_grey"; COLOR_CONSTANCY = True
FLATTEN_SHADOW = True; FLATTEN_SIGMA_FRAC = 0.10; FLATTEN_WORK_PX = 1024
RETINEX_MODE = "rgb"
CC_MINKOWSKI_P = 6
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)
USE_COORDATT = True; COORDATT_STAGES = (1, 2); COORDATT_REDUCTION = 32
MASKED_POOL = True; MASKED_POOL_MIX = 0.75
USE_CHROMATICITY = True; SIDE_DIM = 10
D_MODEL = 256; N_ENC_LAYERS = 3; N_HEADS = 8; FFN_DIM = 512; ENC_DROPOUT = 0.0
EMB_NOISE_SD = 0.0; DROP_PATH = 0.0
NUM_CLASSES = 12; COLOUR_CLASSES = []; FAMILY_NAMES = []; POSITIONS = ["IN", "OUT", "NA"]
CHROMA_ABSTAIN = 4.0; LOW_CONF_THR = 0.35; ABSTAIN_LABEL = "uncertain"; USE_TTA = True
BACKBONE_CANDIDATES = ["convnext_tiny"]
ACHROMATIC = {"black", "white", "slate", "grey"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# JACOPS runtime: autocast policy for the forward pass. Defaults reproduce the original
# behaviour (autocast on CUDA only). set_autocast() lets the CPU profile enable bf16 on
# AMX / AVX512-BF16 machines without touching the model.
AUTOCAST_ENABLED = None      # None -> enabled iff device is cuda
AUTOCAST_DTYPE = None        # None -> torch default for the device


def set_autocast(enabled, dtype=None):
    g = globals(); g["AUTOCAST_ENABLED"] = enabled; g["AUTOCAST_DTYPE"] = dtype


def configure(cfg):
    g = globals()
    g["IMG_SIZE"] = int(cfg["img_size"]); g["FEATHER_K"] = int(cfg["feather_k"])
    g["CROP_PAD_FRAC"] = float(cfg["crop_pad_frac"])
    g["MIN_CROP_SIDE"] = float(cfg.get("min_crop_side", 1.0))
    g["CANONICAL_ORIENT"] = bool(cfg.get("canonical_orient", True))
    g["ILLUM_MODE"] = cfg.get("illum_mode", "shades_of_grey")
    g["COLOR_CONSTANCY"] = bool(cfg.get("color_constancy", True))
    g["FLATTEN_SHADOW"] = bool(cfg.get("flatten_shadow", True))
    g["FLATTEN_SIGMA_FRAC"] = float(cfg.get("flatten_sigma_frac", 0.10))
    g["FLATTEN_WORK_PX"] = int(cfg.get("flatten_work_px", 1024))
    g["RETINEX_MODE"] = cfg.get("retinex_mode", "L")   # "rgb" | "L" | "none"
    g["CC_MINKOWSKI_P"] = int(cfg.get("cc_minkowski_p", 6))
    g["IMAGENET_MEAN"] = np.array(cfg.get("imagenet_mean", [0.485, 0.456, 0.406]), np.float32)
    g["IMAGENET_STD"]  = np.array(cfg.get("imagenet_std",  [0.229, 0.224, 0.225]), np.float32)
    g["USE_COORDATT"] = bool(cfg.get("use_coordatt", False))
    g["COORDATT_STAGES"] = tuple(cfg.get("coordatt_stages", (1, 2)))
    g["COORDATT_REDUCTION"] = int(cfg.get("coordatt_reduction", 32))
    g["MASKED_POOL"] = bool(cfg.get("masked_pool", False))
    g["MASKED_POOL_MIX"] = float(cfg.get("masked_pool_mix", 0.75))
    g["USE_CHROMATICITY"] = bool(cfg.get("use_chromaticity", False))
    g["SIDE_DIM"] = int(cfg["side_dim"])
    g["D_MODEL"] = int(cfg["d_model"]); g["N_ENC_LAYERS"] = int(cfg["enc_layers"])
    g["N_HEADS"] = int(cfg["heads"]); g["FFN_DIM"] = int(cfg.get("ffn_dim", 512))
    g["ENC_DROPOUT"] = 0.0
    g["COLOUR_CLASSES"] = list(cfg["classes"]); g["NUM_CLASSES"] = len(cfg["classes"])
    g["FAMILY_NAMES"] = list(cfg.get("families", []))
    g["POSITIONS"] = list(cfg.get("positions", ["IN", "OUT", "NA"]))
    g["CHROMA_ABSTAIN"] = cfg.get("chroma_abstain", 4.0)
    g["LOW_CONF_THR"] = float(cfg.get("low_conf_thr", 0.35))
    g["ABSTAIN_LABEL"] = cfg.get("abstain_label", "uncertain")
    g["USE_TTA"] = bool(cfg.get("use_tta", True))
    g["BACKBONE_CANDIDATES"] = [cfg.get("backbone", "convnext_tiny")]


def create_backbone(drop_path=0.0):
    # inference: pretrained=False, weights come from the bundle
    m = timm.create_model(BACKBONE_CANDIDATES[0], pretrained=False, in_chans=4,
                          num_classes=0, drop_path_rate=0.0)
    if USE_COORDATT:
        m = attach_coordatt(m)
    return m, BACKBONE_CANDIDATES[0]


def ann_rings(ann):
    seg = ann.get("segmentation")
    rings = []
    if isinstance(seg, list):
        for r in seg:
            if isinstance(r, (list, tuple)) and len(r) >= 6:
                rings.append(np.asarray(r, dtype=np.float32).reshape(-1, 2))
    elif isinstance(seg, dict):
        try:
            from pycocotools import mask as maskutils
            m = maskutils.decode(seg).astype(np.uint8)
            if m.ndim == 3: m = m.max(2)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            rings = [c.reshape(-1, 2).astype(np.float32) for c in cnts if len(c) >= 3]
        except Exception:
            rings = []
    if not rings:
        b = ann.get("bbox")
        if b and b[2] >= MIN_CROP_SIDE and b[3] >= MIN_CROP_SIDE:
            x, y, w, h = b
            rings = [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)]
    return rings


def tight_bbox_from_rings(rings, img_w, img_h, pad_frac=CROP_PAD_FRAC):
    pts = np.concatenate(rings, axis=0)
    x0, y0 = pts.min(axis=0); x1, y1 = pts.max(axis=0)
    pad = max(x1 - x0, y1 - y0) * pad_frac
    return (max(0, x0 - pad), max(0, y0 - pad),
            min(img_w, x1 + pad), min(img_h, y1 + pad))


def letterbox_resize(img, size):
    h, w = img.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    if resized.ndim == 2: resized = resized[..., None]
    canvas = np.zeros((size, size, img.shape[2]), dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def flatten_illumination(img_rgb, sigma_frac=FLATTEN_SIGMA_FRAC, work=FLATTEN_WORK_PX):
    # The blur is a LOW-PASS filter, so it is computed on a downsampled copy (<= work px)
    # and resized back: same result to well under 1 L* unit, ~750x faster on large photos.
    # RETINEX_MODE "rgb" corrects coloured shadows; "L" only touches lightness; "none" is off.
    if RETINEX_MODE == "none":
        return img_rgb
    h, w = img_rgb.shape[:2]
    if RETINEX_MODE == "rgb":
        x = img_rgb.astype(np.float32) + 1e-6
        s = min(1.0, work / max(h, w))
        if s < 1.0:
            xs = cv2.resize(x, (max(1, int(w * s)), max(1, int(h * s))),
                            interpolation=cv2.INTER_AREA)
            ill = cv2.GaussianBlur(xs, (0, 0), max(2.0, sigma_frac * max(xs.shape[:2])))
            ill = cv2.resize(ill, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            ill = cv2.GaussianBlur(x, (0, 0), max(2.0, sigma_frac * max(h, w)))
        out = x / np.maximum(ill, 1e-3) * ill.mean(axis=(0, 1), keepdims=True)
        return np.clip(out, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    L = lab[..., 0].astype(np.float32)
    s = min(1.0, work / max(h, w))
    if s < 1.0:
        Ls = cv2.resize(L, (max(1, int(w * s)), max(1, int(h * s))),
                        interpolation=cv2.INTER_AREA)
        Lb = cv2.GaussianBlur(Ls, (0, 0), max(2.0, sigma_frac * max(Ls.shape)))
        Lb = cv2.resize(Lb, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        Lb = cv2.GaussianBlur(L, (0, 0), max(2.0, sigma_frac * max(h, w)))
    lab[..., 0] = np.clip(L - Lb + float(Lb.mean()), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def estimate_illuminant(img_rgb, p=CC_MINKOWSKI_P, max_side=1024):
    h, w = img_rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = (cv2.resize(img_rgb, (max(1, int(w * s)), max(1, int(h * s))),
                        interpolation=cv2.INTER_AREA) if s < 1.0 else img_rgb)
    x = small.astype(np.float32)
    illum = np.power(np.power(x, p).mean(axis=(0, 1)), 1.0 / p)
    return illum / (illum.mean() + 1e-6)


def fibre_mean_rgbs(img_rgb, anns, max_anns=48):
    out = []
    for ann in anns[:max_anns]:
        rings = ann_rings(ann)
        if not rings: continue
        pts = np.concatenate(rings, 0)
        x0, y0 = np.maximum(pts.min(0), 0).astype(int)
        x1, y1 = np.minimum(pts.max(0) + 1, [img_rgb.shape[1], img_rgb.shape[0]]).astype(int)
        if x1 <= x0 or y1 <= y0: continue
        sub = img_rgb[y0:y1, x0:x1]
        m = np.zeros(sub.shape[:2], np.uint8)
        for r in rings:
            cv2.fillPoly(m, [(r - [x0, y0]).astype(np.int32)], 255)
        if m.sum() == 0: continue
        out.append(sub[m > 0].reshape(-1, 3).mean(0))
    return np.array(out, dtype=np.float32) if out else np.zeros((0, 3), np.float32)


def background_illuminant(img_rgb, anns, p=CC_MINKOWSKI_P, max_side=1024):
    h, w = img_rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = (cv2.resize(img_rgb, (max(1, int(w * s)), max(1, int(h * s))),
                        interpolation=cv2.INTER_AREA) if s < 1.0 else img_rgb)
    m = np.ones(small.shape[:2], np.uint8)
    for a in anns:
        for r in ann_rings(a):
            cv2.fillPoly(m, [(r * s).astype(np.int32)], 0)
    px = small[m > 0].astype(np.float32)
    if len(px) < 500:
        return None
    illum = np.power(np.power(px, p).mean(0), 1.0 / p)
    if illum.mean() < 10:
        return None
    return illum / (illum.mean() + 1e-6)


def white_reference_illuminant(img_rgb, anns):
    means = fibre_mean_rgbs(img_rgb, anns)
    if len(means) >= 4:
        mx, mn = means.max(1), means.min(1)
        chroma = (mx - mn) / (mx + 1e-6)
        lum = means.mean(1)
        cand = means[(chroma < 0.14) & (lum > np.percentile(lum, 60))]
        if len(cand) >= 1:
            ref = np.median(cand, 0)
            if ref.mean() > 20:                      # ignore near-black false positives
                return ref / (ref.mean() + 1e-6)
    bg = background_illuminant(img_rgb, anns)
    if bg is not None:
        return bg
    return estimate_illuminant(img_rgb)


def photo_illuminant(img_rgb, anns):
    if ILLUM_MODE == "none":
        return np.ones(3, np.float32)
    if ILLUM_MODE == "shades_of_grey":
        return estimate_illuminant(img_rgb)
    return white_reference_illuminant(img_rgb, anns)


def apply_illuminant(rgb, illum):
    if illum is None:
        return rgb
    return np.clip(rgb.astype(np.float32) / illum[None, None, :], 0, 255).astype(np.uint8)


def extract_crop(img_rgb, ann, size=None, pad_frac=CROP_PAD_FRAC, illum=None):
    size = size or IMG_SIZE
    h, w = img_rgb.shape[:2]
    rings = ann_rings(ann)
    if not rings: return None
    x0, y0, x1, y1 = tight_bbox_from_rings(rings, w, h, pad_frac)
    if (x1 - x0) < MIN_CROP_SIDE or (y1 - y0) < MIN_CROP_SIDE: return None
    x0i, y0i, x1i, y1i = int(x0), int(y0), math.ceil(x1), math.ceil(y1)
    crop = img_rgb[y0i:y1i, x0i:x1i]
    if crop.size == 0: return None
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    for ring in rings:
        cv2.fillPoly(mask, [(ring - [x0i, y0i]).astype(np.int32)], 255)
    if mask.max() == 0: return None
    rgba = np.dstack([apply_illuminant(crop, illum), mask])
    # v6.0 CANONICAL ORIENTATION: rotate so the fibre's principal axis is vertical.
    if CANONICAL_ORIENT:
        _ys, _xs = np.nonzero(mask)
        if len(_xs) >= 8:
            _p = np.stack([_xs, _ys], 1).astype(np.float32); _p -= _p.mean(0)
            _vt = np.linalg.svd(_p, full_matrices=False)[2]
            _a = math.degrees(math.atan2(_vt[0, 1], _vt[0, 0])) % 180.0
            _h, _w = rgba.shape[:2]
            _M = cv2.getRotationMatrix2D((_w / 2.0, _h / 2.0), _a - 90.0, 1.0)
            _r = cv2.warpAffine(rgba, _M, (_w, _h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_REFLECT_101)
            if (_r[..., 3] > 0).sum() >= 16:
                rgba = _r
    out = letterbox_resize(rgba, size)
    # v5.5 FIX 2: feather kernel sized to the fibre's own width, capped by FEATHER_K.
    _a = out[..., 3]
    _rows = np.count_nonzero(_a.any(axis=1))
    _wpx = (_a > 0).sum() / max(1.0, _rows)
    _k = int(np.clip(round(_wpx / 3.0), 1, FEATHER_K)) | 1
    if _k > 1:
        out[..., 3] = cv2.GaussianBlur(_a, (_k, _k), 0)
    if out[..., 3].max() < 40:
        out[..., 3] = np.clip(out[..., 3].astype(np.float32) *
                              (255.0 / max(1, int(out[..., 3].max()))),
                              0, 255).astype(np.uint8)
    return out


def fibre_mask(alpha):
    # v5.5 FIX 1: relative threshold only, no absolute floor.
    mx = int(alpha.max())
    if mx == 0:
        return np.zeros(alpha.shape, bool)
    return alpha > max(1, int(mx * 0.5))


def chromaticity_from_crop(crop):
    # v6.2: r/(r+g+b), g/(r+g+b) over the fibre pixels - exposure invariant by construction.
    m = fibre_mask(crop[..., 3])
    if m.sum() < 8:
        return np.zeros(4, np.float32)
    px = crop[..., :3][m].astype(np.float64) + 1e-6
    ch = (px / px.sum(1, keepdims=True))[:, :2]
    return np.concatenate([np.median(ch, 0), ch.std(0)]).astype(np.float32)


def to_tensor(rgb, mask):
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    m = (mask.astype(np.float32) / 255.0)[..., None]
    return torch.from_numpy(np.ascontiguousarray(
        np.concatenate([x, m], -1).transpose(2, 0, 1)))


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class CoordAtt(nn.Module):
    # Coordinate Attention (Hou et al.): pool separately along H and W so the network can
    # encode WHERE ALONG THE STRAND'S AXIS the colour signal lives.
    def __init__(self, ch, reduction=COORDATT_REDUCTION):
        super().__init__()
        mip = max(8, ch // reduction)
        self.conv1 = nn.Conv2d(ch, mip, 1)
        self.norm  = nn.GroupNorm(1, mip)      # GroupNorm, not BatchNorm: ragged photo sets
        self.act   = nn.Hardswish()
        self.conv_h = nn.Conv2d(mip, ch, 1)
        self.conv_w = nn.Conv2d(mip, ch, 1)
        nn.init.zeros_(self.conv_h.weight); nn.init.zeros_(self.conv_h.bias)
        nn.init.zeros_(self.conv_w.weight); nn.init.zeros_(self.conv_w.bias)

    def forward(self, x):
        n, c, h, w = x.shape
        xh = x.mean(3, keepdim=True)                       # (n,c,h,1) along the fibre
        xw = x.mean(2, keepdim=True).permute(0, 1, 3, 2)   # (n,c,w,1) across it
        y = self.act(self.norm(self.conv1(torch.cat([xh, xw], dim=2))))
        yh, yw = torch.split(y, [h, w], dim=2)
        return x * torch.sigmoid(self.conv_h(yh)) * torch.sigmoid(
            self.conv_w(yw.permute(0, 1, 3, 2)))


def attach_coordatt(bb):
    # zero-init gates: starts as the identity, so pretrained weights are undisturbed
    try:
        dims = list(bb.feature_info.channels())
    except Exception:
        dims = [96, 192, 384, 768]
    n = 0
    for i in COORDATT_STAGES:
        if i < len(bb.stages):
            bb.stages[i] = nn.Sequential(bb.stages[i], CoordAtt(dims[i]))
            n += 1
    print(f"CoordAtt attached after {n} stage(s) {COORDATT_STAGES}, "
          f"channels {[dims[i] for i in COORDATT_STAGES if i < len(bb.stages)]}")
    return bb


class CrossFibreModel(nn.Module):
    def __init__(self, side_dim=SIDE_DIM, n_classes=NUM_CLASSES,
                 n_fam=len(FAMILY_NAMES), n_pos=len(POSITIONS), drop_path=DROP_PATH):
        super().__init__()
        self.backbone, self.backbone_name = create_backbone(drop_path)
        d = self.backbone.num_features
        self.side = nn.Sequential(nn.Linear(side_dim, 64), nn.GELU(),
                                  nn.Linear(64, 64), nn.GELU())
        self.norm_img  = nn.LayerNorm(d)
        self.norm_side = nn.LayerNorm(64)
        self.proj = nn.Linear(d + 64, D_MODEL)

        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=ENC_DROPOUT, activation="gelu",
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_ENC_LAYERS)
        self.post_norm = nn.LayerNorm(D_MODEL)

        self.head     = nn.Linear(D_MODEL, n_classes)
        self.fam_head = nn.Linear(D_MODEL, n_fam)
        self.pos_head = nn.Linear(D_MODEL, n_pos)
        self.aux_head = nn.Linear(D_MODEL, n_classes)   # pre-attention, deep supervision

    def head_parameters(self):
        for m in (self.side, self.norm_img, self.norm_side, self.proj,
                  self.encoder, self.post_norm,
                  self.head, self.fam_head, self.pos_head, self.aux_head):
            yield from m.parameters()

    def forward(self, imgs, side, mask):
        B, N = imgs.shape[:2]
        flat = imgs.flatten(0, 1)
        valid = mask.flatten(0, 1)
        if MASKED_POOL:
            # v6.2: pool over FIBRE pixels, blended with global pooling at MASKED_POOL_MIX
            fm = self.backbone.forward_features(flat[valid])       # (n, C, h, w)
            m = flat[valid][:, 3:4]
            m = F.interpolate(m, size=fm.shape[-2:], mode="area").clamp_min(0)
            w = m / m.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            out = (MASKED_POOL_MIX * (fm * w).sum(dim=(2, 3))
                   + (1.0 - MASKED_POOL_MIX) * fm.mean(dim=(2, 3)))
        else:
            out = self.backbone(flat[valid])
        feats = torch.zeros(B * N, out.shape[1], device=out.device, dtype=out.dtype)
        feats[valid] = out

        s = self.side(side.flatten(0, 1))
        tok = self.proj(torch.cat([self.norm_img(feats), self.norm_side(s)], 1)).view(B, N, -1)
        if self.training and EMB_NOISE_SD > 0:
            tok = tok + torch.randn_like(tok) * EMB_NOISE_SD

        aux = self.aux_head(tok)                                    # before attention
        z = self.encoder(tok, src_key_padding_mask=~mask)           # across fibres
        z = self.post_norm(tok + z)                                 # residual
        return self.head(z), self.fam_head(z), self.pos_head(z), aux


# =========================================================================================
#  bundle loading + prediction
# =========================================================================================
def load_bundle(export_dir, device=None):
    export_dir = Path(export_dir)
    cfg = json.load(open(export_dir / "inference_config.json"))
    configure(cfg)
    dev = device or DEVICE
    model = CrossFibreModel(side_dim=SIDE_DIM, n_classes=NUM_CLASSES,
                            n_fam=max(1, len(FAMILY_NAMES)), n_pos=len(POSITIONS),
                            drop_path=0.0)
    sd = torch.load(export_dir / "model.pth", map_location="cpu")
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    _extra = [k for k in sd if k.startswith("hist_head.")]     # v6.5 train-only auxiliary head
    if _extra and not any(n.startswith("hist_head") for n, _ in model.named_parameters()):
        sd = {k: v for k, v in sd.items() if not k.startswith("hist_head.")}
        print(f"  dropped {len(_extra)} hist_head tensors (training-only auxiliary, unused at inference)")
    model.load_state_dict(sd, strict=True)          # succeeds or fails loudly - no aligner
    model.to(dev).eval()
    return model, cfg


def decide_label(pred, conf, chroma):
    low = CHROMA_ABSTAIN is not None and chroma < CHROMA_ABSTAIN
    if low and pred not in ACHROMATIC:
        return ABSTAIN_LABEL, "low chroma contradicts a chromatic prediction"
    if conf < LOW_CONF_THR:
        return pred + "?", "low confidence"
    return pred, None


def masks_to_anns(instances):
    # [(bbox(x0,y0,x1,y1), local_bool_mask), ...] -> COCO-style polygon annotations,
    # so detector output goes through the IDENTICAL extract_crop the training set used.
    anns = []
    for bbox, m in instances:
        x0, y0 = int(bbox[0]), int(bbox[1])
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        rings = []
        for c in cnts:
            if len(c) < 3: continue
            r = c.reshape(-1, 2).astype(np.float32) + [x0, y0]
            rings.append(r.reshape(-1).tolist())
        anns.append({"segmentation": rings,
                     "bbox": [x0, y0, int(bbox[2]) - x0, int(bbox[3]) - y0]})
    return anns


@torch.no_grad()
def predict_photo(model, cfg, img_rgb, anns, tta=None, max_fibres_fwd=48):
    # classify every annotated fibre of ONE photo jointly (cross-fibre attention)
    tta = USE_TTA if tta is None else tta
    dev = next(model.parameters()).device
    img = flatten_illumination(img_rgb) if FLATTEN_SHADOW else img_rgb
    anns = [a for a in anns if ann_rings(a)]
    illum = photo_illuminant(img, anns)

    crops, keep = [], []
    for i, a in enumerate(anns):
        c = extract_crop(img, a)
        if c is not None: crops.append(c); keep.append(i)
    if not crops: return []

    raw_lab, chroma, npx = [], [], []
    for c in crops:
        m = fibre_mask(c[..., 3]); n = int(m.sum()); npx.append(n)
        px = c[..., :3][m] if n >= 8 else np.zeros((1, 3), np.uint8)
        lab = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        raw_lab.append(np.concatenate([np.median(lab, 0), lab.std(0)]))
        cor = apply_illuminant(px.reshape(-1, 1, 3), illum) if (COLOR_CONSTANCY and ILLUM_MODE != "none") else px.reshape(-1, 1, 3)
        cl = cv2.cvtColor(cor, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        med = np.median(cl, 0); chroma.append(float(np.linalg.norm(med[1:] - 128.0)))
    side = (np.stack(raw_lab) / 128.0 - 1.0).astype(np.float32)
    if USE_CHROMATICITY:
        chm = np.stack([chromaticity_from_crop(c) for c in crops]).astype(np.float32)
        side = np.concatenate([side, (chm - 0.333) * 6.0], axis=1)
    assert side.shape[1] == SIDE_DIM, f"side dim {side.shape[1]} != cfg {SIDE_DIM}"

    probs = np.zeros((len(crops), NUM_CLASSES), np.float32)
    variants = [1.0] + ([0.94, 1.06] if tta else [])
    for lo in range(0, len(crops), max_fibres_fwd):
        sl = slice(lo, min(lo + max_fibres_fwd, len(crops)))
        chunk = crops[sl]; logits = 0.0
        for v in variants:
            ims = []
            for c in chunk:
                rgb = apply_illuminant(c[..., :3], illum) if (COLOR_CONSTANCY and ILLUM_MODE != "none") else c[..., :3]
                rgb = np.clip(rgb.astype(np.float32) * v, 0, 255).astype(np.uint8)
                ims.append(to_tensor(rgb, c[..., 3]))
            x = torch.stack(ims).unsqueeze(0).to(dev)
            s = torch.from_numpy(side[sl]).unsqueeze(0).to(dev)
            mk = torch.ones(1, len(chunk), dtype=torch.bool, device=dev)
            _ac_on = (dev.type == "cuda") if AUTOCAST_ENABLED is None else bool(AUTOCAST_ENABLED)
            _ac_kw = {"dtype": AUTOCAST_DTYPE} if AUTOCAST_DTYPE is not None else {}
            with torch.autocast(device_type=dev.type, enabled=_ac_on, **_ac_kw):
                lc, _, _, _ = model(x, s, mk)
            logits = logits + lc[0].float().cpu().numpy()
        probs[sl] = softmax(logits / len(variants))

    out = []
    for j, i in enumerate(keep):
        b = int(probs[j].argmax()); conf = float(probs[j][b])
        label, reason = decide_label(COLOUR_CLASSES[b], conf, chroma[j])
        out.append({"ann_index": i, "colour": label, "predicted": COLOUR_CLASSES[b],
                    "confidence": round(conf, 3), "chroma": round(chroma[j], 2),
                    "fibre_px": int(npx[j]), "reason": reason,
                    "top3": {COLOUR_CLASSES[k]: round(float(probs[j][k]), 3)
                             for k in np.argsort(-probs[j])[:3]}})
    return out


def predict_from_masks(model, cfg, img_rgb, instances, **kw):
    return predict_photo(model, cfg, img_rgb, masks_to_anns(instances), **kw)