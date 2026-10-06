"""
JACOPS batch runner (no UI).

    python run_batch.py --images ./photos --out ./results \
        --rfdetr checkpoint_best_regular.pth \
        --oneformer fiber_oneformer_best.pt \
        --colour best_seed42.pth --colour-cfg inference_config.json

Writes per image: <stem>_00_input.jpg, _01_cassette.jpg, _02_roi.jpg, _03_fibres.jpg, <stem>.json,
<stem>_ports.txt; and summary.csv, port_colour_table.csv, all_ports.txt, _summary.json,
jacops_report.xlsx (with embedded images).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import PipelineConfig
from pipeline import JacopsPipeline, list_images


def main(argv=None):
    ap = argparse.ArgumentParser(description="JACOPS: cassette -> fibres -> colours")
    ap.add_argument("--images", required=True, help="folder of images, or a single image")
    ap.add_argument("--out", default="results")
    ap.add_argument("--rfdetr"); ap.add_argument("--oneformer"); ap.add_argument("--colour"); ap.add_argument("--colour-cfg")
    ap.add_argument("--rfdetr-thr", type=float, default=0.45)
    ap.add_argument("--score-thr", type=float); ap.add_argument("--mask-thr", type=float)
    ap.add_argument("--ribbon", type=int, help="ribbon px override (magnification differs from training)")
    ap.add_argument("--io-sign", type=int, choices=[1, -1], default=1)
    ap.add_argument("--ports-from", choices=["rfdetr", "oneformer", "both"], default="rfdetr")
    ap.add_argument("--connector-removal", choices=["after", "before", "none"], default="after",
                    help="after = remove the connector region from masks + output once all stages ran (default)")
    ap.add_argument("--connector-out-fill", choices=["fill", "black", "keep"], default="fill")
    ap.add_argument("--roi-fill", choices=["mean", "black", "keep"], default="mean")
    ap.add_argument("--no-colour", action="store_true")
    # stage 4: handwriting on the splice-connector regions
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--yolo-obb", help="YOLOv8-OBB best.pt (Cassette-Handwritten)")
    ap.add_argument("--parseq1", help="parseq_cassette_final.pt")
    ap.add_argument("--parseq-repo", help="local baudm/parseq clone for offline use")
    ap.add_argument("--ocr-orientation", default="auto", help="auto | none | 0 | 90 | 180 | 270")
    ap.add_argument("--ocr-conf", type=float, default=0.25); ap.add_argument("--ocr-pad", type=float, default=0.15)
    ap.add_argument("--ocr-mask-outside", action="store_true", help="blank pixels outside the connector mask")
    ap.add_argument("--ocr-base", action="store_true", help="also run stock PARSeq-base as secondary")
    ap.add_argument("--tta", choices=["auto", "on", "off"], default="auto", help="auto: on GPU, off on CPU")
    # runtime
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (0 = physical cores)")
    ap.add_argument("--int8", action="store_true", help="CPU: dynamic INT8 on nn.Linear (opt-in)")
    ap.add_argument("--cpu-bf16", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--compile", action="store_true", help="torch.compile the models (slow first image)")
    ap.add_argument("--batch-tiles", type=int, default=0, help="0 = auto (8 GPU / 2 CPU)")
    ap.add_argument("--no-excel", action="store_true")
    # splicing validation against the lasschema
    ap.add_argument("--lasschema", help="lasschema .xlsx; adds AI status / AI comments columns -> <out>/<name>_AI_validated.xlsx")
    ap.add_argument("--cassette", type=int, help="cassette number (default: read from the handwritten tray number)")
    ap.add_argument("--in-side", choices=["Right", "Left"], default="Right")
    a = ap.parse_args(argv)

    cfg = PipelineConfig()
    if a.rfdetr: cfg.rfdetr_ckpt = a.rfdetr
    if a.oneformer: cfg.oneformer_ckpt = a.oneformer
    if a.colour: cfg.colour_ckpt = a.colour
    if a.colour_cfg: cfg.colour_cfg = a.colour_cfg
    cfg.rfdetr_threshold = a.rfdetr_thr; cfg.score_thr = a.score_thr; cfg.mask_thr = a.mask_thr
    cfg.ribbon_override = a.ribbon; cfg.io_sign = a.io_sign; cfg.ports_from = a.ports_from
    cfg.connector_removal = a.connector_removal; cfg.connector_out_fill = a.connector_out_fill; cfg.roi_fill = a.roi_fill
    cfg.do_colour = not a.no_colour
    cfg.do_ocr = not a.no_ocr
    if a.yolo_obb: cfg.yolo_obb_weights = a.yolo_obb
    if a.parseq1: cfg.parseq1_ckpt = a.parseq1
    if a.parseq_repo: cfg.parseq_repo = a.parseq_repo
    cfg.ocr_orientation = a.ocr_orientation; cfg.ocr_conf = a.ocr_conf; cfg.ocr_pad_frac = a.ocr_pad
    cfg.ocr_mask_outside = a.ocr_mask_outside; cfg.ocr_use_base = a.ocr_base
    cfg.use_tta = {"on": True, "off": False}.get(a.tta, "auto")
    cfg.device = a.device; cfg.cpu_threads = a.threads; cfg.int8_dynamic = bool(a.int8) or "auto"
    cfg.cpu_bf16 = {"on": True, "off": False}.get(a.cpu_bf16, "auto")
    cfg.torch_compile = a.compile; cfg.batch_tiles = a.batch_tiles

    src = Path(a.images)
    paths = [src] if src.is_file() else list_images(src)
    if not paths:
        print(f"no images found under {src}"); return 1
    pipe = JacopsPipeline(cfg)
    recs = pipe.process_batch(paths, a.out, progress=lambda i, n, f: print(f"[{i:3d}/{n}] ", end="", flush=True),
                              excel=not a.no_excel)
    if a.lasschema:
        from splice_validation import load_lasschema, validate_cassette, write_validation_workbook
        schema = load_lasschema(a.lasschema); ok = [r for r in recs if "error" not in r]
        results = [validate_cassette(r, schema, a.cassette, a.in_side) for r in ok]
        for r, res in zip(ok, results): print(f"{r['file']}: cassette {res.get('cassette')} -> {res['overall']} | {res['ai_comments']}")
        outp = Path(a.out) / (Path(a.lasschema).stem + "_AI_validated.xlsx")
        write_validation_workbook(schema, results, ok, str(outp)); print(f"validated workbook -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())