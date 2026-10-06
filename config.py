"""
JACOPS pipeline configuration.

One dataclass carries every knob from the three original scripts so the CLI, the Streamlit
app and the pipeline all read the same names.  Checkpoint paths default to environment
variables so nothing user-specific is hard-coded in the source.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class PipelineConfig:
    # ── checkpoints ──────────────────────────────────────────────────────────────────────
    rfdetr_ckpt:    str = os.environ.get("JACOPS_RFDETR_CKPT",    r"D:\splcing_complete\checkpoint_best_regular.pth")
    oneformer_ckpt: str = os.environ.get("JACOPS_ONEFORMER_CKPT", r"D:\splcing_complete\fiber_oneformer_best.pt")
    colour_ckpt:    str = os.environ.get("JACOPS_COLOUR_CKPT",    r"D:\splcing_complete\best_seed42 (2).pth")
    colour_cfg:     str = os.environ.get("JACOPS_COLOUR_CFG",     r"D:\splcing_complete\inference_config.json")
    work_dir:       str = os.environ.get("JACOPS_WORK_DIR",       "_jacops_work")
    yolo_obb_weights: str = os.environ.get("JACOPS_YOLO_OBB",     r"D:\splcing_complete\best.pt")                   # stage 4
    parseq1_ckpt:   str = os.environ.get("JACOPS_PARSEQ1_CKPT",   r"D:\splcing_complete\parseq_sp_vw.pt")  # stage 4
    parseq_repo:    str = os.environ.get("JACOPS_PARSEQ_REPO",    "")                          # local baudm/parseq clone (offline)

# local baudm/parseq clone (offline)
 
    # ── stage 0 : standard orientation (orientation.py) ─────────────────────────────────
    orientation_mode: str = "auto"          # "auto" (holder below the sleeve stack) | "none" | "0/90/180/270"
 
    # ── stage 1 : RF-DETR cassette / connector ───────────────────────────────────────────
    rfdetr_num_classes: int   = 2
    rfdetr_threshold:   float = 0.45
    rfdetr_optimize:    bool  = True        # rfdetr's optimize_for_inference() (fused/compiled graph)
    cassette_select:    str   = "largest"   # "largest" | "union"  (which cassette_body masks form the ROI)
    connector_removal:  str   = "after"     # "after"  : stages 2+3 see the whole cassette; the connector
                                            #            region is removed from fibre masks / the OUTPUT
                                            # "before" : blank connectors from the stage-2/3 input (old)
                                            # "none"   : never remove
    connector_dilate_px: int  = 3           # grow connector masks so the sleeve edge is removed too
    connector_out_fill: str   = "fill"      # output image: "fill" (ROI mean) | "black" | "keep" the pixels
    drop_fibre_in_connector_frac: float = 0.9   # a fibre with >= this fraction inside a connector is dropped
    remove_connectors:  bool  = True        # legacy alias: False -> connector_removal="none"
    roi_pad_frac:       float = 0.02        # padding around the cassette bbox when cropping
    roi_fill:           str   = "mean"      # "mean" | "black" | "keep"  fill for non-ROI pixels
 
    # ── stage 2 : OneFormer fibres / ports ───────────────────────────────────────────────
    score_thr:       Optional[float] = None   # None -> the detector bundle's own value
    mask_thr:        Optional[float] = None
    ribbon_override: Optional[int]   = None   # px; None -> derived from the ORIGINAL image size
    max_len:         Optional[float] = None
    io_sign:         int   = +1               # flip to -1 if in/out is mirrored on the overlays
    port_min_aspect: float = 3.0
    port_max_area_x: float = 4.0
    port_bind_slack: float = 0.6
    merge_anywhere:  bool  = True
    skip_empty_tiles: bool = True             # LATENCY: do not run tiles that contain no ROI pixels
    tile_min_roi_px: int   = 2000             # a tile needs at least this many ROI px to be run
    batch_tiles:     int   = 0                # tiles per forward; 0 = auto (8 GPU / 2 CPU); halved on CUDA OOM
    ports_from:      str   = "sleeves"        # "sleeves" (OneFormer sleeve rows + lattice, default) | "auto"
                                              # (sleeves, fallback RF-DETR region) | "rfdetr" | "legacy"
    # port stack (port_stack.py): the horizontal sleeve rows as ports
    port_angle_tol:  float = 10.0             # deg; a sleeve must be parallel to the dominant sleeve direction
    port_len_range:  tuple = (0.6, 1.5)       # x median sleeve length
    port_along_tol:  float = 0.5              # x median sleeve length; centre must sit in the sleeve column
    lattice_tol:     float = 0.30             # x pitch; max distance of a sleeve centre from its lattice slot
    max_empty_slots: int   = 2                # a run of more empty slots than this splits the stack
    expected_slots:  int   = 0                # holder capacity (e.g. 12 / 24) if known; 0 = infer
    draw_empty_slots: bool = True
    # "rfdetr": splice_connector masks from stage 1 become the port instances (they are also
    #           blanked from the stage-2 input, so OneFormer cannot see them anyway).
    # "oneformer": ignore stage-1 connectors for ports (only sensible with remove_connectors=False)
    # "both": union, then the usual sleeve-shape filter on the OneFormer ones.
 
    # ── stage 3 : CrossFibre colour ──────────────────────────────────────────────────────
    do_colour:         bool  = True
    use_tta:           object = "auto"        # brightness TTA = 3 forwards; auto: on GPU, off on CPU
    max_fibres_fwd:    int   = 48
    colour_match_rule: bool  = True           # port IN / OUT must agree
    min_fibre_px_warn: int   = 40
 
    # ── runtime / latency ────────────────────────────────────────────────────────────────
    cudnn_benchmark:  bool = True             # autotune convolutions (fixed tile size -> big win)
    allow_tf32:       bool = True             # TF32 matmuls on Ampere+
    prefetch_images:  bool = True             # decode the next photo on a thread while the GPU works
    warmup:           bool = True             # one dummy forward per model after loading
    oom_retries:      int  = 4                # tenacity: attempts per chunk, batch halved each time
    load_retries:     int  = 3                # tenacity: attempts for model / hub loading
 
    # ── runtime profile (runtime.py; "auto" = decided from the hardware) ─────────────────
    device:          str    = "auto"          # "auto" | "cuda" | "cpu"
    cpu_threads:     int    = 0               # 0 = physical cores
    int8_dynamic:    object = "auto"          # CPU: dynamic INT8 on nn.Linear (auto = off; opt-in, ~1.5-2.5x)
    cpu_bf16:        object = "auto"          # CPU: bf16 autocast only when AMX / AVX512-BF16 is present
    torch_compile:   bool   = False           # inductor; pays a compile cost on the first image
    use_ipex:        object = "auto"          # Intel Extension for PyTorch if installed
 
    # ── stage 4 : handwritten labels on the SPLICE-CONNECTOR regions only (inference.py) ─
    do_ocr:            bool  = True
    ocr_conf:          float = 0.25           # YOLO-OBB confidence
    ocr_imgsz:         int   = 640
    ocr_orientation:   str   = "auto"         # "auto" (4 rotations, best read wins) | "none" | "0/90/180/270"
    ocr_pad_frac:      float = 0.15           # pad the connector bbox by this fraction of its longer side
    ocr_mask_outside:  bool  = False          # fill pixels outside the connector mask (dilated) with the ROI mean
    ocr_mask_dilate_px: int  = 12
    ocr_use_base:      bool  = False          # stock PARSeq-base secondary (needs torch.hub / a local clone)
    ocr_max_lines:     int   = 3
    ocr_annotate_text: bool  = True
    ocr_split_mode:    str   = "auto"         # left/right split on the FULL photo: "auto" (gap between the two
                                              # label lobes, fallback = cassette midline) | "cassette_mid" | "manual"
    ocr_split_frac:    float = 0.5            # manual: fraction of the cassette bbox width
 
    # ── output ───────────────────────────────────────────────────────────────────────────
    draw_ports:   bool  = True
    fill_by:      str   = "colour"            # "colour" | "io"
    label_fibres: bool  = True
    label_ports:  bool  = True
    port_label_counts: bool = False           # False: draw "Port N" only; True: "Port N: a in / b out"
    alpha:        float = 0.55
    draw_cassette_outline: bool = True
    save_json:    bool  = True
    thumb_width:  int   = 320                 # Excel thumbnail width in px
 
    def to_dict(self) -> dict:
        return asdict(self)