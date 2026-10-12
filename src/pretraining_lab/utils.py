"""Seeding, device/precision resolution, memory measurement and environment capture."""

from __future__ import annotations

import contextlib
import platform
import random
import sys
from typing import Any

import numpy as np
import torch

from .config import ConfigError

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA when available


def resolve_device(requested: str) -> torch.device:
    """``auto`` prefers CUDA when present. Explicitly asking for CUDA without it is an error."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigError("train.device is 'cuda' but CUDA is not available on this machine")
    return torch.device(requested)


def resolve_precision(requested: str, device: torch.device) -> tuple[str, str | None]:
    """Return ``(effective_precision, note)``; falls back to fp32 when unsupported."""
    if requested == "fp32":
        return "fp32", None
    if requested == "fp16" and device.type != "cuda":
        return "fp32", "fp16 autocast requires CUDA; fell back to fp32"
    if requested == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return "fp32", "bf16 is not supported on this GPU; fell back to fp32"
    return requested, None


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=_DTYPES[precision])


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> bool:
    """Reset the peak-memory counter. Returns False when it cannot be reset (CPU, non-Linux)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        return True
    try:  # Linux: writing "5" resets this process's peak RSS (VmHWM)
        with open("/proc/self/clear_refs", "w") as handle:
            handle.write("5")
        return True
    except OSError:
        return False


def peak_memory(device: torch.device) -> tuple[int | None, str | None]:
    """Return ``(bytes, kind)`` for the peak memory since the last reset, or ``(None, None)``."""
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device)), "cuda_max_memory_allocated"
    try:
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024, "cpu_peak_rss"
    except OSError:
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        scale = 1 if sys.platform == "darwin" else 1024
        return int(peak) * scale, "cpu_peak_rss_process_lifetime"
    except (ImportError, OSError):
        return None, None


def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def environment_info() -> dict[str, Any]:
    """Software and hardware details recorded alongside every benchmark result."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu_model": _cpu_model(),
        "cpu_logical_cores": __import__("os").cpu_count(),
        "torch_num_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "cuda_runtime": torch.version.cuda,
                "gpu_name": props.name,
                "gpu_total_memory_bytes": props.total_memory,
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return info
