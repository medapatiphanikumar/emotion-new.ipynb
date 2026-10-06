"""
Runtime layer — make a GPU-designed pipeline run well on CPU (and better on GPU)
WITHOUT touching the checkpoints or the model architectures.

Every optimisation here is applied to the *loaded* module or to the *call path*:

  process level    torch/OMP/MKL thread counts sized to physical cores, no oversubscription
                   with OpenCV; denormal flushing (a real CPU stall on conv/transformer nets)
  precision        CUDA  -> bf16/fp16 autocast          (as before)
                   CPU   -> bf16 autocast ONLY when the CPU has AMX / AVX512-BF16
                            (on other CPUs bf16 emulation is slower than fp32, so it is off)
  weights          CPU   -> optional dynamic INT8 quantisation of nn.Linear
                            (transformer blocks in OneFormer/Swin, the CrossFibre encoder and
                            ConvNeXt MLPs are Linear-heavy; ~1.5–2.5x on those layers).
                            Weights on disk are untouched; quantised copies live in RAM only.
  memory layout    channels_last for conv-heavy nets (oneDNN fast path on CPU, TensorCores on GPU)
  graph            optional torch.compile (inductor) — fixed tile shape makes it worthwhile
                   for OneFormer; first call pays the compile cost
  vendor           optional Intel Extension for PyTorch (ipex.optimize) if installed
  warm-up          one dummy forward per model at load, so the first real image does not
                   pay oneDNN primitive creation / cudnn autotune / compile

Work reduction (the biggest CPU win) lives in the stages themselves and is driven by the
profile:  skip tiles that contain no ROI, small tile batches on CPU, TTA off on CPU.
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass, asdict
from typing import Callable, Optional

import cv2
import torch
import torch.nn as nn


# ══════════════════════════════════════════════════════════════════════════════════════════
#  detection
# ══════════════════════════════════════════════════════════════════════════════════════════
def physical_cores() -> int:
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n: return int(n)
    except Exception:
        pass
    n = os.cpu_count() or 4
    return max(1, n // 2) if n >= 4 else n          # assume SMT when psutil is absent


def cpu_flags() -> set:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if line.startswith("flags"):
                        return set(line.split(":")[1].split())
    except Exception:
        pass
    return set()


def cpu_supports_bf16() -> bool:
    f = cpu_flags()
    return bool({"amx_bf16", "avx512_bf16"} & f)


def cpu_supports_vnni() -> bool:
    f = cpu_flags()
    return bool({"avx512_vnni", "avx_vnni", "amx_int8"} & f)


def has_ipex() -> bool:
    try:
        import intel_extension_for_pytorch  # noqa: F401
        return True
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════════════════════════
#  profile
# ══════════════════════════════════════════════════════════════════════════════════════════
@dataclass
class RuntimeProfile:
    device: str = "cpu"                 # "cuda" | "cpu"
    threads: int = 4
    autocast_dtype: Optional[str] = None   # "bf16" | "fp16" | None
    int8_dynamic: bool = False          # CPU only
    channels_last: bool = True
    torch_compile: bool = False
    use_ipex: bool = False
    batch_tiles: int = 8                # OneFormer tiles per forward (8 GPU / 2 CPU)
    tta: bool = True                    # stage-3 brightness TTA (3 forwards) — off on CPU
    warmup: bool = True
    notes: str = ""

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def amp_dtype(self):
        return {"bf16": torch.bfloat16, "fp16": torch.float16}.get(self.autocast_dtype)

    def autocast(self):
        """Context manager for a forward pass on this profile's device."""
        dt = self.amp_dtype
        return torch.autocast(self.device, dtype=dt, enabled=dt is not None)

    def to_dict(self): return asdict(self)


def detect_runtime(cfg=None, log: Callable = print) -> RuntimeProfile:
    """Build a profile from the hardware, then let PipelineConfig.runtime_* override it."""
    want = (getattr(cfg, "device", "auto") or "auto").lower()
    cuda = torch.cuda.is_available() and want in ("auto", "cuda")
    notes = []
    if cuda:
        bf16 = torch.cuda.is_bf16_supported()
        prof = RuntimeProfile(device="cuda", threads=max(2, physical_cores() // 2),
                              autocast_dtype="bf16" if bf16 else "fp16",
                              int8_dynamic=False, channels_last=True, torch_compile=False,
                              batch_tiles=8, tta=True)
        notes.append(f"GPU {torch.cuda.get_device_name(0)}, autocast {prof.autocast_dtype}")
    else:
        cores = physical_cores()
        bf16 = cpu_supports_bf16()
        prof = RuntimeProfile(device="cpu", threads=cores,
                              autocast_dtype="bf16" if bf16 else None,
                              int8_dynamic=False,            # opt-in: numerics change slightly
                              channels_last=True, torch_compile=False,
                              use_ipex=has_ipex(),
                              batch_tiles=2, tta=False)
        notes.append(f"CPU {cores} physical cores, bf16 {'AMX/AVX512-BF16' if bf16 else 'not native -> fp32'}, "
                     f"VNNI int8 {'yes' if cpu_supports_vnni() else 'no'}, ipex {'yes' if prof.use_ipex else 'no'}")
    # ── user overrides from PipelineConfig ────────────────────────────────────────────────
    if cfg is not None:
        if getattr(cfg, "cpu_threads", 0): prof.threads = int(cfg.cpu_threads)
        v = getattr(cfg, "int8_dynamic", "auto")
        if v is not None and v != "auto": prof.int8_dynamic = bool(v) and prof.device == "cpu"
        v = getattr(cfg, "torch_compile", False)
        if v: prof.torch_compile = True
        v = getattr(cfg, "use_tta", "auto")
        if v is not None and v != "auto": prof.tta = bool(v)
        v = getattr(cfg, "batch_tiles", 0)
        if v: prof.batch_tiles = int(v)
        v = getattr(cfg, "cpu_bf16", "auto")
        if prof.device == "cpu" and v is not None and v != "auto":
            prof.autocast_dtype = "bf16" if v else None
        prof.warmup = bool(getattr(cfg, "warmup", True))
        v = getattr(cfg, "use_ipex", "auto")
        if v is not None and v != "auto": prof.use_ipex = bool(v) and has_ipex()
    prof.notes = "; ".join(notes)
    log(f"runtime: {prof.device} | threads {prof.threads} | autocast {prof.autocast_dtype} | int8 {prof.int8_dynamic} | "
        f"compile {prof.torch_compile} | batch_tiles {prof.batch_tiles} | "
        f"tta {prof.tta} | {prof.notes}")
    return prof


# ══════════════════════════════════════════════════════════════════════════════════════════
#  process-level settings
# ══════════════════════════════════════════════════════════════════════════════════════════
_APPLIED = False


def apply_process_settings(prof: RuntimeProfile, log: Callable = print):
    global _APPLIED
    if _APPLIED: return
    _APPLIED = True
    n = int(prof.threads)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, str(n))
    try:
        torch.set_num_threads(n)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)         # we do not run parallel graphs
    except RuntimeError:
        pass                                          # interop threads already initialised
    cv2.setNumThreads(max(1, n // 2))                 # OpenCV must not fight torch for cores
    torch.set_flush_denormal(True)                    # denormals stall CPU FMAs badly
    if prof.device == "cuda":
        torch.backends.cudnn.benchmark = True         # fixed tile shapes -> autotune once
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.backends.mkldnn.enabled = True
    log(f"process: torch threads {torch.get_num_threads()} | cv2 threads {cv2.getNumThreads()} | flush_denormal on")


# ══════════════════════════════════════════════════════════════════════════════════════════
#  module-level optimisation
# ══════════════════════════════════════════════════════════════════════════════════════════
def optimize_module(model: nn.Module, prof: RuntimeProfile, name: str = "model",
                    conv_heavy: bool = False, compile_dynamic: bool = False, log: Callable = print) -> nn.Module:
    """
    Apply the profile to an already-loaded module. Order matters:
      eval -> channels_last -> (ipex | int8) -> compile
    Nothing here writes to disk; the checkpoint is unchanged.
    """
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    applied = []

    if prof.channels_last and conv_heavy:
        model = model.to(memory_format=torch.channels_last); applied.append("channels_last")

    if prof.device == "cpu":
        if prof.use_ipex and has_ipex():
            try:
                import intel_extension_for_pytorch as ipex
                model = ipex.optimize(model, dtype=prof.amp_dtype or torch.float32, inplace=True)
                applied.append(f"ipex({prof.autocast_dtype or 'fp32'})")
            except Exception as e:
                log(f"   ipex.optimize failed on {name}: {e}")
        if prof.int8_dynamic:
            try:
                from torch.ao.quantization import quantize_dynamic
                model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=False)
                applied.append("int8-dynamic(Linear)")
            except Exception as e:
                log(f"   int8 dynamic quantisation failed on {name}: {e}")

    if prof.torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, dynamic=compile_dynamic,
                                  mode="max-autotune-no-cudagraphs" if prof.device == "cuda" else "default")
            applied.append("torch.compile")
        except Exception as e:
            log(f"   torch.compile failed on {name}: {e}")

    log(f"{name}: {', '.join(applied) if applied else 'no module-level changes'}")
    return model


def maybe_channels_last(x: torch.Tensor, prof: RuntimeProfile) -> torch.Tensor:
    return x.contiguous(memory_format=torch.channels_last) if (prof.channels_last and x.dim() == 4) else x


class StageTimer:
    """Tiny accumulator so the batch summary can report per-stage p50/mean."""
    def __init__(self): self.t = {}
    def add(self, k, s): self.t.setdefault(k, []).append(float(s))
    def summary(self):
        import numpy as np
        return {k: {"mean": round(float(np.mean(v)), 2), "p50": round(float(np.median(v)), 2), "n": len(v)}
                for k, v in self.t.items()}