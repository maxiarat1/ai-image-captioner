import os

def force_cpu_mode() -> bool:
    """Return True if CPU-only mode is requested via env var.

    Set TAGGER_FORCE_CPU=1 to force CPU and avoid CUDA initialization on startup.
    """
    return os.environ.get("TAGGER_FORCE_CPU", "0") == "1"


def pick_device(torch):
    """Choose a safe torch.device respecting forced CPU mode.

    - If TAGGER_FORCE_CPU=1, always return cpu.
    - Otherwise, prefer cuda if available; fall back to cpu.
    Using this helper avoids importing torch.cuda paths unnecessarily when CPU is forced.
    """
    if force_cpu_mode():
        return torch.device("cpu")
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        # If cuda probing triggers errors, safely fall back to CPU
        return torch.device("cpu")


def maybe_empty_cuda_cache(torch):
    """Empty CUDA cache if available and not in forced CPU mode."""
    if force_cpu_mode():
        return
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Ignore any CUDA cleanup errors on unsupported setups
        pass
