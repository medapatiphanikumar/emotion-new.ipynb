"""
JACOPS — Cassette → Fibres → Colours
Streamlit front-end for the three-stage pipeline.
Run: streamlit run app.py
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from config import PipelineConfig

st.set_page_config(page_title="JACOPS — Cassette Validator", page_icon="🔬", layout="wide",
                   initial_sidebar_state="expanded")

# ── CSS (dark industrial theme from the original app) ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.jacops-header { background: linear-gradient(135deg,#0d2137 0%,#1a3a5c 50%,#0d2137 100%); border:1px solid #1f6feb;
  border-radius:8px; padding:20px 28px 16px; margin-bottom:20px; position:relative; overflow:hidden; }
.jacops-header::before { content:''; position:absolute; top:0; left:0; right:0; height:3px;
  background: linear-gradient(90deg,#1f6feb,#58a6ff,#1f6feb); }
.jacops-title { font-family:'IBM Plex Mono',monospace; font-size:22px; font-weight:600; color:#58a6ff; letter-spacing:.05em; margin:0; }
.jacops-sub { font-size:13px; color:#8b949e; margin-top:4px; font-family:'IBM Plex Mono',monospace; }
.metric-row { display:grid; grid-template-columns:repeat(6,1fr); gap:10px; margin:14px 0; }
.metric-card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px 16px; text-align:center; }
.metric-card .metric-value { font-family:'IBM Plex Mono',monospace; font-size:24px; font-weight:600; color:#58a6ff; line-height:1; }
.metric-card .metric-label { font-size:11px; color:#8b949e; margin-top:4px; text-transform:uppercase; letter-spacing:.06em; }
.banner-success,.banner-warning,.banner-error { border-radius:6px; padding:10px 14px; font-family:'IBM Plex Mono',monospace; font-size:13px; margin:8px 0; }
.banner-success { background:#0e4429; border:1px solid #238636; color:#3fb950; }
.banner-warning { background:#2d1a00; border:1px solid #9e6a03; color:#e3b341; }
.banner-error   { background:#3d0e0e; border:1px solid #da3633; color:#ff7b72; }
.code-block { background:#161b22; border:1px solid #30363d; border-radius:6px; padding:12px 16px;
  font-family:'IBM Plex Mono',monospace; font-size:12px; color:#e6edf3; white-space:pre-wrap; overflow-x:auto; }
.stage-tag { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:11px; padding:2px 8px; border-radius:4px;
  background:#21262d; color:#58a6ff; border:1px solid #30363d; margin-bottom:6px; }
.stButton > button { background:#1f6feb; color:white; border:none; border-radius:6px; padding:8px 20px; font-weight:500; width:100%; }
.stButton > button:hover { background:#388bfd; }
.stSlider label,.stSelectbox label,.stFileUploader label,.stTextInput label,.stNumberInput label {
  color:#8b949e !important; font-size:12px !important; text-transform:uppercase !important; letter-spacing:.06em !important;
  font-family:'IBM Plex Mono',monospace !important; }
h1,h2,h3,h4 { font-family:'IBM Plex Mono',monospace !important; color:#e6edf3 !important; }
.stMarkdown p { color:#8b949e; font-size:13px; }
</style>
""", unsafe_allow_html=True)

CKPT_DIR = Path("_uploaded_ckpts"); CKPT_DIR.mkdir(exist_ok=True)


def banner(kind, text):
    st.markdown(f'<div class="banner-{kind}">{text}</div>', unsafe_allow_html=True)


def resolve_ckpt(label, key, default, types):
    """Sidebar widget: upload OR type a path. Returns a path string or None."""
    st.markdown(f"**{label}**")
    up = st.file_uploader(f"Upload {label}", type=types, key=f"up_{key}", label_visibility="collapsed")
    if up is not None:
        p = CKPT_DIR / up.name
        if not p.exists() or p.stat().st_size != up.size:
            p.write_bytes(up.getvalue())
        banner("success", f"✓ {up.name}")
        return str(p)
    path = st.text_input(f"{label} path", value=default, key=f"path_{key}", label_visibility="collapsed")
    if path and Path(path).exists():
        banner("success", f"✓ {Path(path).name}")
        return path
    banner("warning", f"⚠ not found: {path or '(empty)'}")
    return None


@st.cache_resource(show_spinner=False)
def load_pipeline(rfdetr, oneformer, colour, colour_cfg, do_colour, num_classes,
                  device="auto", threads=0, int8="auto", cpu_bf16="auto", compile_=False,
                  do_ocr=False, yolo_obb="", parseq1="", parseq_repo="", ocr_use_base=False):
    # runtime knobs are part of the cache key: changing them reloads (re-optimises) the models
    from pipeline import JacopsPipeline
    cfg = PipelineConfig(rfdetr_ckpt=rfdetr, oneformer_ckpt=oneformer, colour_ckpt=colour or "",
                         colour_cfg=colour_cfg or "", do_colour=do_colour, rfdetr_num_classes=num_classes,
                         device=device, cpu_threads=threads, int8_dynamic=int8, cpu_bf16=cpu_bf16,
                         torch_compile=compile_, do_ocr=do_ocr, yolo_obb_weights=yolo_obb or "",
                         parseq1_ckpt=parseq1 or "", parseq_repo=parseq_repo or "", ocr_use_base=ocr_use_base)
    logs = []
    pipe = JacopsPipeline(cfg, log=lambda s: logs.append(str(s)))
    return pipe, logs


# ══════════════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════════════════
defaults = PipelineConfig()
with st.sidebar:
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#58a6ff;letter-spacing:.08em;'
                'margin-bottom:16px;text-transform:uppercase">⬡ JACOPS CONFIG</div>', unsafe_allow_html=True)
    st.markdown("#### Checkpoints")
    rf_ckpt = resolve_ckpt("RF-DETR (stage 1)", "rf", defaults.rfdetr_ckpt, ["pth", "pt"])
    of_ckpt = resolve_ckpt("OneFormer (stage 2)", "of", defaults.oneformer_ckpt, ["pt", "pth"])
    do_colour = st.toggle("Run colour model (stage 3)", value=True)
    co_ckpt = co_cfg = None
    if do_colour:
        co_ckpt = resolve_ckpt("CrossFibre colour (stage 3)", "co", defaults.colour_ckpt, ["pth", "pt"])
        co_cfg = resolve_ckpt("inference_config.json (optional)", "cfg", defaults.colour_cfg, ["json"])
    num_classes = st.number_input("RF-DETR classes", 1, 10, 2)
    do_ocr = st.toggle("Read handwritten labels on connectors (stage 4)", value=True)
    yolo_ckpt = pq_ckpt = None; pq_repo = ""
    if do_ocr:
        yolo_ckpt = resolve_ckpt("YOLO-OBB best.pt (stage 4)", "yolo", defaults.yolo_obb_weights, ["pt"])
        pq_ckpt = resolve_ckpt("PARSeq-1 parseq_cassette_final.pt", "pq", defaults.parseq1_ckpt, ["pt", "pth", "ckpt"])
        pq_repo = st.text_input("Local baudm/parseq clone (offline, optional)", value=defaults.parseq_repo)
    st.markdown("---")

    st.markdown("#### Stage 1 · cassette")
    rf_thr = st.slider("RF-DETR confidence", 0.10, 0.90, 0.45, 0.05)
    orient_mode = st.selectbox("Standard orientation", ["auto", "none", "0", "90", "180", "270"], index=0,
                               help="auto: rotate the photo so the sleeve stack is horizontal on top and the splice "
                                    "holder below it (decided from the RF-DETR masks); a fixed angle forces it")
    conn_mode = st.selectbox("Splice connector removal", ["after (output only)", "before (stage-2/3 input)", "none"], index=0,
                             help="after: all models see the full cassette; the connector region is removed from the "
                                  "fibre masks and blanked in the final image once every stage has run")
    conn_out_fill = st.selectbox("Removed region shown as", ["fill", "black", "keep"], index=0)
    conn_dil = st.slider("Connector dilation px", 0, 15, 3)
    roi_fill = st.selectbox("Non-ROI fill", ["mean", "black", "keep"], index=0,
                            help="mean = ROI mean colour (neutral for the illuminant estimate)")
    cass_sel = st.selectbox("Cassette masks", ["largest", "union"], index=0)
    st.markdown("---")

    st.markdown("#### Stage 2 · fibres")
    use_bundle = st.toggle("Use detector bundle thresholds", value=True)
    score_thr = mask_thr = None
    if not use_bundle:
        score_thr = st.slider("Score threshold", 0.05, 0.90, 0.30, 0.05)
        mask_thr = st.slider("Mask threshold", 0.20, 0.80, 0.50, 0.05)
    ribbon = st.number_input("Ribbon px override (0 = auto)", 0, 60, 0)
    io_sign = st.selectbox("IN/OUT sign", [1, -1], index=0, help="flip if in/out looks mirrored")
    ports_from = st.selectbox("Ports = ", ["sleeves", "auto", "rfdetr", "legacy"], index=0,
                              help="sleeves: the horizontal splice-sleeve rows found by OneFormer, numbered on a fitted "
                                   "lattice (empty slots shown). auto: same, fallback to the RF-DETR holder region. "
                                   "rfdetr: the whole splice_connector region as one port. legacy: old v6.4 filter.")
    expected_slots = st.number_input("Holder capacity (sleeve slots, 0 = infer)", 0, 48, 0)
    st.markdown("---")

    st.markdown("#### Stage 3 · colour")
    tta_choice = st.selectbox("Brightness TTA (3 forwards)", ["auto (GPU on / CPU off)", "on", "off"], index=0)
    use_tta = {"on": True, "off": False}.get(tta_choice, "auto")
    match_rule = st.toggle("Splice IN/OUT must agree", value=True)
    fill_by = st.selectbox("Overlay fill", ["colour", "io"], index=0)
    port_counts = st.toggle("Show in/out counts on port labels", value=False)
    show_json = st.toggle("Show raw JSON", value=False)
    st.markdown("---")
    if do_ocr:
        st.markdown("#### Stage 4 · handwriting (connectors only)")
        ocr_orient = st.selectbox("Orientation search", ["auto", "none", "0", "90", "180", "270"], index=0,
                                  help="auto = detect+read at 4 rotations per connector, keep the best-reading one (4x cost)")
        ocr_conf = st.slider("YOLO-OBB confidence", 0.05, 0.90, 0.25, 0.05)
        ocr_pad = st.slider("Connector bbox padding (frac of long side)", 0.0, 0.5, 0.15, 0.05)
        ocr_mask = st.toggle("Blank pixels outside the connector mask", value=False,
                             help="strict ROI: the OBB detector can only fire on the connector itself")
        ocr_base = st.toggle("Also run stock PARSeq-base (secondary)", value=False)
        ocr_text = st.toggle("Draw recognised text on the image", value=True)
        st.markdown("**Left / right split** (on the full cassette photo)")
        split_mode = st.radio("Split position", ["auto", "cassette_mid", "manual"], index=0, horizontal=True,
                              format_func=lambda x: {"auto": "Auto (gap between lobes)", "cassette_mid": "Cassette midline",
                                                     "manual": "Manual"}[x], label_visibility="collapsed")
        split_frac = st.slider("Split at (fraction of cassette width)", 0.20, 0.80, 0.50, 0.01, disabled=split_mode != "manual")
        st.markdown("---")
    else:
        ocr_orient, ocr_conf, ocr_pad, ocr_mask, ocr_base, ocr_text = "auto", 0.25, 0.15, False, False, True
        split_mode, split_frac = "auto", 0.5
    st.markdown("#### Runtime")
    import torch as _torch
    _has_cuda = _torch.cuda.is_available()
    device_choice = st.selectbox("Device", ["auto", "cuda", "cpu"] if _has_cuda else ["auto", "cpu"], index=0)
    threads = st.number_input("CPU threads (0 = physical cores)", 0, 256, 0)
    int8_choice = st.selectbox("INT8 dynamic (CPU, Linear layers)", ["off", "on"], index=0,
                               help="~1.5-2.5x on the transformer blocks; numerics shift slightly - "
                                    "compare a few photos against fp32 before adopting")
    bf16_choice = st.selectbox("CPU bf16 autocast", ["auto (only with AMX / AVX512-BF16)", "on", "off"], index=0)
    compile_choice = st.toggle("torch.compile (slow first image, faster after)", value=False)
    skip_tiles = st.toggle("Skip tiles outside the cassette", value=True)
    batch_tiles = st.slider("Tiles per forward (0 = auto: 8 GPU / 2 CPU)", 0, 32, 0)
    st.caption("Model loading, optimisation and warm-up happen once per session (and again if a runtime knob changes).")
    st.markdown("---")
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#484f58;line-height:1.8">'
                'Stage 1 · RF-DETR Seg Medium<br>cassette_body / splice_connector<br><br>'
                'Stage 2 · OneFormer Swin-T<br>fiber / port on cassette ROI<br><br>'
                'Stage 3 · ConvNeXt-T + CoordAtt<br>12 TIA colours, cross-fibre attention</div>',
                unsafe_allow_html=True)

ready = bool(rf_ckpt and of_ckpt and (not do_colour or co_ckpt) and (not do_ocr or (yolo_ckpt and pq_ckpt)))


def get_pipe():
    pipe, logs = load_pipeline(rf_ckpt, of_ckpt, co_ckpt, co_cfg, do_colour, int(num_classes),
                               device=device_choice, threads=int(threads),
                               int8=(int8_choice == "on"),
                               cpu_bf16={"on": True, "off": False}.get(bf16_choice, "auto"),
                               compile_=bool(compile_choice),
                               do_ocr=do_ocr, yolo_obb=yolo_ckpt, parseq1=pq_ckpt, parseq_repo=pq_repo,
                               ocr_use_base=ocr_base)
    c = pipe.cfg
    if do_ocr and getattr(pipe, "ocr_error", None):
        banner("error", f"✗ Stage 4 (handwriting) failed to load: {pipe.ocr_error}")
    from pipeline import PIPELINE_VERSION as _PV
    _loaded = next((l for l in logs if l.startswith("pipeline version")), "pipeline version: (old build - restart Streamlit)")
    if _PV not in _loaded:
        banner("error", f"✗ Cached pipeline is stale ({_loaded}); file on disk is {_PV}. Click 'Reload models' below or restart Streamlit.")
    if st.button("⟳ Reload models (clear cache)", key="reload_models"):
        load_pipeline.clear(); st.rerun()
    with st.expander("Model load log", expanded=bool(do_ocr and getattr(pipe, "ocr_error", None))):
        st.code("\n".join(logs[-60:]) or "(empty)", language="text")
    c.ocr_orientation = ocr_orient; c.ocr_conf = float(ocr_conf); c.ocr_pad_frac = float(ocr_pad)
    c.ocr_mask_outside = ocr_mask; c.ocr_annotate_text = ocr_text
    c.ocr_split_mode = split_mode; c.ocr_split_frac = float(split_frac)
    c.rfdetr_threshold = float(rf_thr); c.connector_dilate_px = int(conn_dil); c.orientation_mode = orient_mode
    c.connector_removal = conn_mode.split(" ")[0]; c.remove_connectors = True; c.connector_out_fill = conn_out_fill
    c.roi_fill = roi_fill; c.cassette_select = cass_sel
    c.score_thr = score_thr; c.mask_thr = mask_thr; c.ribbon_override = int(ribbon) or None
    c.io_sign = int(io_sign); c.ports_from = ports_from; c.expected_slots = int(expected_slots)
    c.use_tta = use_tta; c.colour_match_rule = match_rule; c.fill_by = fill_by; c.port_label_counts = port_counts
    c.skip_empty_tiles = skip_tiles; c.batch_tiles = int(batch_tiles)
    # TTA is a per-run knob (no reload needed)
    if use_tta != "auto": pipe.prof.tta = bool(use_tta)
    return pipe, logs


def metric_cards(rec):
    cnt, s1 = rec["counts"], rec["stage1"]
    o = rec.get("orientation")
    if o is None:
        banner("error", "✗ This result came from an OLD pipeline build (no stage 0). Stop and restart Streamlit "
                        "(Ctrl+C in the terminal, then `streamlit run app.py`) so the cached models reload from the new files.")
    elif o.get("angle"):
        banner("success", f"↻ Stage 0: rotated {o['angle']}° to the standard position — {o['reason']}")
    else:
        banner("warning", f"Stage 0: no rotation — {o['reason']}")
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card"><div class="metric-value" style="color:#3fb950">{s1['n_cassette']}</div><div class="metric-label">Cassette</div></div>
      <div class="metric-card"><div class="metric-value" style="color:#ff7b72">{s1['n_connector']}</div><div class="metric-label">Connectors</div></div>
      <div class="metric-card"><div class="metric-value">{cnt['fiber']}</div><div class="metric-label">Fibres</div></div>
      <div class="metric-card"><div class="metric-value" style="color:#d2a8ff">{cnt['fiber_in']} / {cnt['fiber_out']}</div><div class="metric-label">In / Out</div></div>
      <div class="metric-card"><div class="metric-value" style="color:#e3b341">{cnt['answered']}/{cnt['fiber']}</div><div class="metric-label">Coloured</div></div>
      <div class="metric-card"><div class="metric-value" style="color:#8b949e">{rec['timing_s']['total']}s</div><div class="metric-label">Total</div></div>
    </div>""", unsafe_allow_html=True)
    t = rec["timing_s"]
    ps = rec.get("port_stack")
    if ps and ps.get("ok"):
        st.markdown(f'<div class="code-block">port stack: {ps["n_detected"]} sleeve candidates → {ps["n_kept"]} ports on a '
                    f'{ps["n_slots"]}-slot lattice (pitch {ps["pitch"]:.0f} px, angle {ps["theta_deg"]:.1f}°)'
                    + (f' · empty/missed slots {ps["missing"]}' if ps["missing"] else "")
                    + (f' · rejected {ps["rejected"]}' if ps.get("rejected") else "") + "</div>", unsafe_allow_html=True)
    elif ps is not None:
        st.markdown(f'<div class="code-block">port stack: no sleeve lattice fitted ({ps.get("n_detected", 0)} candidates, '
                    f'rejected {ps.get("rejected") or {}})</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="code-block">stage1 {t["stage1"]}s · stage2 {t["stage2"]}s '
                f'({rec.get("tiles_run", "?")}/{rec.get("tiles_grid", "?")} tiles run) · stage3 {t["stage3"]}s · '
                f'stage4 {t.get("stage4", 0)}s ({cnt.get("hw_regions", 0)} labels, {cnt.get("hw_review", 0)} review) · '
                f'render {t.get("render", 0)}s</div>', unsafe_allow_html=True)


def show_four(rec):
    keys = [("input", "INPUT", "Original photo")]
    if (rec.get("orientation") or {}).get("angle"):
        keys.append(("standard", "STAGE 0 · STANDARDISED", f"rotated {rec['orientation']['angle']}° — {rec['orientation']['reason']}"))
    keys += [
            ("stage1", "STAGE 1 · RF-DETR", "cassette_body / splice_connector"),
            ("roi", "STAGE 2 INPUT", "cassette ROI (models see the full cassette)"),
            ("overlay", "STAGE 2+3 · OneFormer + colour", "fibres, ports, colours; connector region removed")]
    if "handwriting" in rec["images"]:
        keys.append(("handwriting", "STAGE 4 · YOLO-OBB + PARSeq", "handwritten labels on the splice connectors"))
    cols = st.columns(len(keys))
    for col, (key, tag, cap) in zip(cols, keys):
        with col:
            st.markdown(f'<span class="stage-tag">{tag}</span>', unsafe_allow_html=True)
            st.image(rec["images"][key], caption=cap, use_container_width=True)


def show_report(rec):
    if rec["stage1"]["fallback_whole_image"]:
        banner("warning", "⚠ No cassette_body detected — stage 2 ran on the whole image")
    if rec["counts"]["splice_mismatch"]:
        banner("error", f"✗ {rec['counts']['splice_mismatch']} splice colour mismatch(es)")
    st.markdown(f'<div class="code-block">{rec["report"]}</div>', unsafe_allow_html=True)
    hw = rec.get("handwriting")
    if hw and hw.get("report"):
        st.markdown('<span class="stage-tag">HANDWRITTEN LABELS (per port)</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="code-block">{hw["report"]}</div>', unsafe_allow_html=True)
    if hw and hw.get("labels"):
        show_left_right(rec)


def show_left_right(rec):
    """Left / right differentiation of the handwritten labels on the cassette (leftright.py)."""
    from leftright import split_sides, side_table, highlight_status
    hw = rec["handwriting"]; sp = hw.get("split") or {}
    left, right, outl = split_sides(hw["labels"])
    how = {"lobes": "midpoint between the two label clusters",
           "midline": "image midline — no clear two-lobe separation found",
           "cassette-midline": "cassette midline — no clear two-lobe separation found",
           "cassette-midline (forced)": "cassette midline (forced)", "manual": "manual override"}.get(sp.get("method"), sp.get("method"))
    st.markdown('<span class="stage-tag">LEFT / RIGHT HANDWRITTEN TEXTS</span>', unsafe_allow_html=True)
    st.caption(f"Split at x={sp.get('x', 0):.0f} px on the full photo ({how}); sides are decided after un-rotating each connector crop.")
    m = st.columns(3)
    m[0].metric("Left", len(left)); m[1].metric("Right", len(right)); m[2].metric("Off-cassette", len(outl))
    t_left, t_right = st.columns(2)
    with t_left:
        st.markdown(f"**Left parameters** ({len(left)})")
        st.dataframe(side_table(left).style.apply(highlight_status, axis=1), hide_index=True, use_container_width=True)
    with t_right:
        st.markdown(f"**Right parameters** ({len(right)})")
        st.dataframe(side_table(right).style.apply(highlight_status, axis=1), hide_index=True, use_container_width=True)
    if outl:
        st.markdown(f"**Off-cassette / review** ({len(outl)})")
        st.dataframe(side_table(outl).style.apply(highlight_status, axis=1), hide_index=True, use_container_width=True)
    crops = {(ci, did): c for ci, did, c in (rec.get("_hw_crops") or [])}
    ordered = left + right + outl
    if crops and ordered:
        st.markdown("**Label crops** (left → right → off-cassette)")
        cols = st.columns(min(6, len(ordered)))
        for i, l in enumerate(ordered):
            c = crops.get((l["conn_idx"], l["detection_id"]))
            if c is None: continue
            with cols[i % len(cols)]:
                st.image(c[:, :, ::-1], caption=f"{l.get('side')} · ID{l['detection_id']} · {l.get('final_text') or '?'}", use_container_width=True)
    if rec["colours"]:
        st.markdown("Colours: " + ", ".join(f"**{c}** ×{n}" for c, n in sorted(rec["colours"].items(), key=lambda t: -t[1])))


def zip_results(recs, cfg, single_name=None):
    from jacops_excel import build_excel
    from pipeline import _js, write_rgb
    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="jacops_"))
    from pipeline import JacopsPipeline
    for r in recs:
        if "error" in r: continue
        stem = Path(r["file"]).stem
        for key, suf in [("input", "00_input"), ("standard", "00b_standard"), ("stage1", "01_cassette"), ("roi", "02_roi"), ("overlay", "03_fibres"),
                         ("handwriting", "04_handwriting")]:
            if key in r["images"]: write_rgb(tmp / f"{stem}_{suf}.jpg", r["images"][key])
        import cv2 as _cv2
        for ci, did, crop in (r.get("_hw_crops") or []):
            (tmp / "hw_crops").mkdir(exist_ok=True); _cv2.imwrite(str(tmp / "hw_crops" / f"{stem}_conn{ci}_id{did}.png"), crop)
        json.dump(_js({k: v for k, v in r.items() if k not in ("images", "_hw_crops")}), open(tmp / f"{stem}.json", "w"), indent=1)
        hw = (r.get("handwriting") or {}).get("report", "")
        open(tmp / f"{stem}_ports.txt", "w").write(r["report"] + "\n\n== handwritten labels ==\n" + hw + "\n")
    JacopsPipeline.write_summaries(recs, tmp)
    build_excel(recs, tmp / "jacops_report.xlsx", cfg=cfg)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(tmp.rglob("*")):
            if p.is_file(): z.write(p, p.relative_to(tmp))
    xls = (tmp / "jacops_report.xlsx").read_bytes()
    return buf.getvalue(), xls


# ══════════════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="jacops-header">
  <p class="jacops-title">JACOPS // CASSETTE VALIDATOR</p>
  <p class="jacops-sub">RF-DETR cassette + connector  →  OneFormer fibres on cassette ROI  →  CrossFibre colour  →  splice check</p>
</div>""", unsafe_allow_html=True)

tab_single, tab_batch, tab_val, tab_about = st.tabs(["  Single Image  ", "  Batch  ", "  Splicing Validation  ", "  About  "])

# ── single ────────────────────────────────────────────────────────────────────────────────
with tab_single:
    up = st.file_uploader("Choose a cassette photo", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"], key="single")
    if up is None:
        st.markdown('<div class="code-block">↑ Upload a cassette image, then click Run Pipeline</div>', unsafe_allow_html=True)
    elif not ready:
        banner("warning", "⚠ Set all checkpoint paths in the sidebar first")
    else:
        from PIL import ImageOps
        img = np.array(ImageOps.exif_transpose(Image.open(up)).convert("RGB"))
        st.markdown(f'<div class="code-block">file : {up.name}\nsize : {up.size // 1024} KB\nres  : {img.shape[1]} × {img.shape[0]} px</div>',
                    unsafe_allow_html=True)
        if st.button("▶  Run Pipeline", key="run_single"):
            try:
                with st.spinner("Loading models (first run only)..."):
                    pipe, logs = get_pipe()
                with st.spinner("Stage 1 → ROI → Stage 2 → Stage 3 ..."):
                    rec = pipe.process(img, up.name)
                st.session_state["single_rec"] = rec
                st.session_state["single_cfg"] = pipe.cfg
            except Exception as e:
                banner("error", f"✗ {type(e).__name__}: {e}")
                st.exception(e)
        rec = st.session_state.get("single_rec")
        if rec and rec["file"] == up.name:
            metric_cards(rec)
            show_four(rec)
            show_report(rec)
            zb, xb = zip_results([rec], st.session_state["single_cfg"])
            c1, c2 = st.columns(2)
            c1.download_button("⬇  Excel report (images embedded)", xb, f"jacops_{Path(up.name).stem}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            c2.download_button("⬇  All outputs (zip)", zb, f"jacops_{Path(up.name).stem}.zip", "application/zip")
            if show_json:
                st.code(json.dumps({k: v for k, v in rec.items() if k not in ("images", "_hw_crops")}, indent=2, default=str)[:20000], language="json")

# ── batch ─────────────────────────────────────────────────────────────────────────────────
with tab_batch:
    ups = st.file_uploader("Choose images", type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff"],
                           accept_multiple_files=True, key="batch")
    if ups:
        st.markdown(f"**{len(ups)} image(s) ready**")
        if not ready:
            banner("warning", "⚠ Set all checkpoint paths in the sidebar first")
        elif st.button("▶  Run Batch", key="run_batch"):
            try:
                with st.spinner("Loading models (first run only)..."):
                    pipe, logs = get_pipe()
                prog = st.progress(0); status = st.empty(); recs = []
                for i, f in enumerate(ups, 1):
                    status.markdown(f'<div class="banner-success">Processing {i}/{len(ups)}: {f.name}</div>', unsafe_allow_html=True)
                    try:
                        from PIL import ImageOps
                        recs.append(pipe.process(np.array(ImageOps.exif_transpose(Image.open(f)).convert("RGB")), f.name))
                    except Exception as e:
                        recs.append({"file": f.name, "error": f"{type(e).__name__}: {e}"})
                    prog.progress(i / len(ups))
                status.empty(); prog.empty()
                st.session_state["batch_recs"] = recs; st.session_state["batch_cfg"] = pipe.cfg
            except Exception as e:
                banner("error", f"✗ {type(e).__name__}: {e}"); st.exception(e)
        recs = st.session_state.get("batch_recs")
        if recs:
            ok = [r for r in recs if "error" not in r]
            tot = lambda k: sum(r["counts"][k] for r in ok)
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-card"><div class="metric-value">{len(ok)}/{len(recs)}</div><div class="metric-label">Images</div></div>
              <div class="metric-card"><div class="metric-value">{tot('fiber')}</div><div class="metric-label">Fibres</div></div>
              <div class="metric-card"><div class="metric-value">{tot('port')}</div><div class="metric-label">Ports</div></div>
              <div class="metric-card"><div class="metric-value" style="color:#e3b341">{tot('answered')}</div><div class="metric-label">Coloured</div></div>
              <div class="metric-card"><div class="metric-value" style="color:#ff7b72">{tot('splice_mismatch')}</div><div class="metric-label">Mismatches</div></div>
              <div class="metric-card"><div class="metric-value" style="color:#8b949e">{np.mean([r['timing_s']['total'] for r in ok]) if ok else 0:.1f}s</div><div class="metric-label">Avg time</div></div>
            </div>""", unsafe_allow_html=True)
            zb, xb = zip_results(recs, st.session_state["batch_cfg"])
            c1, c2 = st.columns(2)
            c1.download_button("⬇  Excel report (images embedded)", xb, "jacops_report.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_x")
            c2.download_button("⬇  All outputs (zip)", zb, "jacops_batch_results.zip", "application/zip", key="dl_z")
            for r in recs:
                with st.expander(f"{r['file']}" + (f"  —  FAILED" if "error" in r else
                                 f"  —  {r['counts']['fiber']} fibres, {r['counts']['port']} ports, {r['counts']['splice_mismatch']} mismatch")):
                    if "error" in r: banner("error", r["error"]); continue
                    show_four(r); show_report(r)

# ── splicing validation ───────────────────────────────────────────────────────────────────
with tab_val:
    from splice_validation import load_lasschema, validate_cassette, write_validation_workbook, TIA598
    st.markdown("#### Cassette photo vs lasschema (splice schema)")
    st.caption("T = tube, V = vezel. One cassette = one IN tube V1-12 (label on the IN side); the OUT side carries one "
               "T<tube>V<a>-<b> label per out tube. Vezel numbers map to fibre colours by the 12-colour code, so the schema "
               "also predicts the IN / OUT colour sets that stages 2+3 detected.")
    schema_up = st.file_uploader("Lasschema workbook (.xlsx)", type=["xlsx"], key="lasschema")
    single_rec = st.session_state.get("single_rec"); batch_recs = [r for r in (st.session_state.get("batch_recs") or []) if "error" not in r]
    if not schema_up:
        st.markdown('<div class="code-block">↑ Upload the lasschema Excel (e.g. 09_..._LASSCHEMA_DP020_....xlsx)</div>', unsafe_allow_html=True)
    elif not single_rec and not batch_recs:
        banner("warning", "⚠ Run a photo first (Single Image or Batch); the validation uses its fibre colours and handwritten labels")
    else:
        sp = CKPT_DIR / schema_up.name; sp.write_bytes(schema_up.getvalue())
        try:
            schema = load_lasschema(str(sp))
        except Exception as e:
            banner("error", f"✗ Could not parse the lasschema: {type(e).__name__}: {e}"); schema = None
        if schema:
            have = sorted(n for n, c in schema.cassettes.items() if c.rows)
            st.markdown(f'<div class="code-block">DP {schema.dp} · sheet {schema.sheet} · cassettes with splices: {have}</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            which = c1.selectbox("Photo(s) to validate", (["last single image"] if single_rec else []) + (["batch results"] if batch_recs else []))
            cas_choice = c2.selectbox("Cassette number", ["auto (handwritten tray number)"] + [str(n) for n in have])
            in_side = c3.selectbox("IN tube label side", ["Right", "Left"], index=0)
            recs_v = [single_rec] if which == "last single image" else batch_recs
            cas = None if cas_choice.startswith("auto") else int(cas_choice)
            if st.button("▶  Validate splicing", key="run_val"):
                results = [validate_cassette(r, schema, cas, in_side, TIA598) for r in recs_v]
                st.session_state["val_results"] = (results, recs_v, str(sp))
            got = st.session_state.get("val_results")
            if got and got[2] == str(sp):
                results, recs_v, _ = got
                col = {"OK": "success", "WARN": "warning", "MISMATCH": "error", "INFO": "success"}
                for res, r in zip(results, recs_v):
                    with st.expander(f"{r['file']}  —  cassette {res.get('cassette')} ({res.get('source')})  —  {res['overall']}", expanded=len(results) == 1):
                        banner(col[res["overall"]], f"{res['overall']}: {res['ai_comments']}")
                        import pandas as _pd
                        df = _pd.DataFrame(res["checks"])[["check", "status", "expected", "observed", "ai_comment"]]
                        df.columns = ["Check", "Status", "Expected (schema)", "Observed (photo)", "AI comment"]
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        if res.get("expected"):
                            e, o = res["expected"], res["observed"]
                            cc1, cc2 = st.columns(2)
                            cc1.markdown("**IN colours** — schema: " + ", ".join(f"{k}×{v}" for k, v in e["in_colours"].items()) +
                                         "  \nphoto: " + (", ".join(f"{k}×{v}" for k, v in o["in_colours"].items()) or "(none)") +
                                         (f" + {o['unknown']['in']} unknown" if o["unknown"]["in"] else ""))
                            cc2.markdown("**OUT colours** — schema: " + ", ".join(f"{k}×{v}" for k, v in e["out_colours"].items()) +
                                         "  \nphoto: " + (", ".join(f"{k}×{v}" for k, v in o["out_colours"].items()) or "(none)") +
                                         (f" + {o['unknown']['out']} unknown" if o["unknown"]["out"] else ""))
                        if res.get("labels"):
                            st.markdown("Handwriting parsed: " + ", ".join(
                                f"`{l['raw']}`→{l['kind']}" + (f" T{l['tube']} V{l['v0']}-{l['v1']} [{l['role'] or '?'}]" if l['kind'] == 'tube_range' else "")
                                for l in res["labels"]))
                        if "handwriting" in r.get("images", {}):
                            st.image(r["images"]["handwriting"], caption="stage 4 — labels on the splice holder", use_container_width=True)
                import tempfile
                outp = Path(tempfile.mkdtemp(prefix="jacops_val_")) / (Path(schema_up.name).stem + "_AI_validated.xlsx")
                write_validation_workbook(schema, results, recs_v, str(outp))
                st.download_button("⬇  Lasschema with AI status / AI comments columns + 'AI validation' sheet", outp.read_bytes(),
                                   outp.name, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="dl_val")

# ── about ─────────────────────────────────────────────────────────────────────────────────
with tab_about:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Pipeline")
        st.markdown("""<div class="code-block">1  RF-DETR Seg Medium
   cassette_body + splice_connector on the full photo
2  ROI = cassette_body − dilated splice_connector
   cropped to the cassette bbox, outside filled with
   the ROI mean colour (neutral for illuminant / retinex)
3  OneFormer (fiber / port) on the ROI only
   ribbon width derived from the ORIGINAL photo size
   splice_connector masks injected as the port instances
4  CrossFibre colour on the ROI: canonical orientation,
   shades-of-grey illuminant, RGB retinex, chromaticity
   side features, one joint forward pass per photo
5  Port k IN and OUT are one spliced fibre → colours must
   agree; a partner resolves an abstention, a disagreement
   is flagged
8  Splicing Validation tab: cassette number, T/V labels
   (IN = one tube V1-12, OUT = one label per out tube),
   IN/OUT fibre colours (vezel -> TIA-598 colour) and
   splice pairs compared with the lasschema Excel; the
   original workbook is returned with 'AI status' and
   'AI comments' columns and an 'AI validation' sheet
6  Overlay drawn back on the ORIGINAL photo; the splice
   connector region is removed from the output only after
   every stage has run
7  YOLO-OBB + PARSeq-1 (inference.py) read the handwritten
   labels on the SPLICE-CONNECTOR regions only, per port;
   Excel report embeds input / stage-1 / ROI / final /
   handwriting images and every label crop</div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("#### How to run")
        st.markdown("""<div class="code-block">pip install -r requirements.txt
streamlit run app.py

# headless
python run_batch.py --images ./photos --out ./results \\
   --rfdetr checkpoint_best_regular.pth \\
   --oneformer fiber_oneformer_best.pt \\
   --colour best_seed42.pth --colour-cfg inference_config.json

# env defaults (optional)
JACOPS_RFDETR_CKPT, JACOPS_ONEFORMER_CKPT,
JACOPS_COLOUR_CKPT, JACOPS_COLOUR_CFG</div>""", unsafe_allow_html=True)
        st.markdown("#### Outputs per image")
        st.markdown("""<div class="code-block">&lt;stem&gt;_00_input.jpg      original
&lt;stem&gt;_01_cassette.jpg   stage-1 contours
&lt;stem&gt;_02_roi.jpg        stage-2 input (ROI)
&lt;stem&gt;_03_fibres.jpg     fibres + colours + ports
&lt;stem&gt;_04_handwriting.jpg labels read on the connectors
hw_crops/                 every handwritten region crop
&lt;stem&gt;.json / _ports.txt
summary.csv · port_colour_table.csv · all_ports.txt
jacops_report.xlsx  (Summary · Stage1 · Fibres · Ports · Config)</div>""", unsafe_allow_html=True)