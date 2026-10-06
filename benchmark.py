"""
JACOPS latency benchmark.

Times each stage on real photos under one or more runtime profiles and, when several
profiles are compared, reports how far each one's *results* drift from the fp32 baseline
(fibre count, coloured count, colour agreement on matched fibres). Use it to decide whether
INT8 / bf16 / compile are worth switching on for your CPU.

    # single profile (whatever runtime.py auto-detects)
    python benchmark.py --images ./photos --n 3

    # compare on CPU
    python benchmark.py --images ./photos --compare fp32,int8,bf16 --device cpu
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from config import PipelineConfig
from pipeline import list_images, read_rgb

PROFILES = {
    "fp32":        dict(int8_dynamic=False, cpu_bf16=False, torch_compile=False),
    "bf16":        dict(int8_dynamic=False, cpu_bf16=True,  torch_compile=False),
    "int8":        dict(int8_dynamic=True,  cpu_bf16=False, torch_compile=False),
    "int8+bf16":   dict(int8_dynamic=True,  cpu_bf16=True,  torch_compile=False),
    "compile":     dict(int8_dynamic=False, cpu_bf16="auto", torch_compile=True),
    "int8+compile": dict(int8_dynamic=True, cpu_bf16="auto", torch_compile=True),
    "auto":        dict(),
}


def run_profile(name: str, base: PipelineConfig, images, n: int, log):
    from pipeline import JacopsPipeline
    import torch, gc
    cfg = PipelineConfig(**{**base.to_dict(), **PROFILES[name]})
    t = time.time(); pipe = JacopsPipeline(cfg, log=log); load_s = time.time() - t
    times = {"stage1": [], "stage2": [], "stage3": [], "total": []}
    results = {}
    for p in images:
        img = read_rgb(p)
        for _ in range(n):
            rec = pipe.process(img, Path(p).name)
            for k in times: times[k].append(rec["timing_s"][k])
        results[Path(p).name] = [(i["bbox"], i["colour"]) for i in rec["instances"] if i["class"] == "fiber"]
    del pipe; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"load_s": round(load_s, 1), "p50": {k: round(statistics.median(v), 2) for k, v in times.items()},
            "results": results}


def agreement(base, other):
    """Fraction of baseline fibres that have an IoU>0.5 bbox match with the same colour."""
    def iou(a, b):
        x0, y0 = max(a[0], b[0]), max(a[1], b[1]); x1, y1 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, x1 - x0) * max(0, y1 - y0)
        ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0
    n_b = n_match = n_col = 0
    for f, fibs in base.items():
        oth = other.get(f, [])
        for bb, col in fibs:
            n_b += 1
            best = max(oth, key=lambda o: iou(bb, o[0]), default=None)
            if best and iou(bb, best[0]) > 0.5:
                n_match += 1; n_col += (best[1] == col)
    return n_b, n_match, n_col


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True); ap.add_argument("--n", type=int, default=2, help="repeats per image")
    ap.add_argument("--max-images", type=int, default=3)
    ap.add_argument("--compare", default="", help="comma list of " + ",".join(PROFILES))
    ap.add_argument("--device", default="auto"); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--rfdetr"); ap.add_argument("--oneformer"); ap.add_argument("--colour"); ap.add_argument("--colour-cfg")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    base = PipelineConfig(device=a.device, cpu_threads=a.threads, save_json=False)
    if a.rfdetr: base.rfdetr_ckpt = a.rfdetr
    if a.oneformer: base.oneformer_ckpt = a.oneformer
    if a.colour: base.colour_ckpt = a.colour
    if a.colour_cfg: base.colour_cfg = a.colour_cfg
    src = Path(a.images); images = ([src] if src.is_file() else list_images(src))[:a.max_images]
    if not images: print("no images"); return 1
    log = (lambda s: None) if a.quiet else print
    names = [s.strip() for s in a.compare.split(",") if s.strip()] or ["auto"]

    out = {}
    for nm in names:
        print(f"\n=== profile {nm} ===")
        out[nm] = run_profile(nm, base, images, a.n, log)
        print(f"load {out[nm]['load_s']}s | p50 per image: {out[nm]['p50']}")

    print("\n" + "-" * 78)
    print(f"{'profile':14s} {'load s':>7s} {'s1':>7s} {'s2':>7s} {'s3':>7s} {'total':>7s}   fibres  matched  same-colour")
    ref = out[names[0]]["results"]
    for nm in names:
        r = out[nm]; p = r["p50"]
        n_b, n_m, n_c = agreement(ref, r["results"])
        print(f"{nm:14s} {r['load_s']:7.1f} {p['stage1']:7.2f} {p['stage2']:7.2f} {p['stage3']:7.2f} {p['total']:7.2f}   "
              f"{n_b:6d}  {n_m:7d}  {n_c:11d}")
    print(f"(agreement is measured against '{names[0]}'; a mismatch is an IoU<0.5 fibre or a different colour)")
    return 0


if __name__ == "__main__":
    sys.exit(main())