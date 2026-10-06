"""
Retry policies (tenacity) shared by the three stages.

Two failure modes matter for wall-clock time:

  * loading      transient OSError / ConnectionError while transformers pulls the OneFormer base
                 config or a checkpoint is briefly locked by a sync client. Retried with
                 exponential back-off. Assertion errors (wrong checkpoint) are NOT retried.

  * CUDA OOM     one big tile batch or a photo with 200 fibres can exhaust memory half way
                 through a batch job. Instead of crashing (and re-running everything), the
                 chunk is retried with the batch size halved each attempt, cache emptied.
                 The caller re-slices its work inside the attempt so the new size applies.

tenacity is optional: if it is not installed the helpers degrade to a single attempt.
"""
from __future__ import annotations

import gc
from typing import Callable, Iterable, Type

try:
    from tenacity import (Retrying, retry, retry_if_exception_type, stop_after_attempt,
                          wait_exponential, wait_none)
    HAVE_TENACITY = True
except ImportError:                      # pragma: no cover
    HAVE_TENACITY = False


def _oom_types():
    try:
        import torch
        return (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, "OutOfMemoryError") else (RuntimeError,)
    except ImportError:
        return (RuntimeError,)


def is_oom(exc: BaseException) -> bool:
    try:
        import torch
        if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except ImportError:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass


def load_retry(attempts: int = 3, exc: Iterable[Type[BaseException]] = (OSError, ConnectionError, TimeoutError), log=print):
    """Decorator: retry a model/hub loading function on transient I/O errors."""
    if not HAVE_TENACITY or attempts <= 1:
        return lambda f: f
    exc = tuple(exc)

    def _before_sleep(rs):
        log(f"   retry {rs.attempt_number}/{attempts} after {type(rs.outcome.exception()).__name__}: "
            f"{rs.outcome.exception()}")

    return retry(stop=stop_after_attempt(attempts), wait=wait_exponential(multiplier=1, min=1, max=15),
                 retry=retry_if_exception_type(exc), before_sleep=_before_sleep, reraise=True)


def oom_retrying(attempts: int, on_oom: Callable[[], None], log=print):
    """
    Iterable of attempts for a CUDA-OOM-prone chunk:

        for attempt in oom_retrying(4, shrink_batch):
            with attempt:
                chunk = work[i:i + self.batch]      # re-sliced -> uses the shrunk batch
                run(chunk)

    on_oom() is called between attempts (halve the batch, free cache).
    """
    if not HAVE_TENACITY or attempts <= 1:
        class _Once:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return [_Once()]

    def _retry_pred(exc): return is_oom(exc)

    def _before_sleep(rs):
        log(f"   CUDA OOM on attempt {rs.attempt_number}/{attempts} - shrinking batch and retrying")
        free_cuda(); on_oom()

    from tenacity import retry_if_exception
    return Retrying(stop=stop_after_attempt(attempts), wait=wait_none(),
                    retry=retry_if_exception(_retry_pred), before_sleep=_before_sleep, reraise=True)