import os

def force_cpu_mode() -> bool:
    return os.environ.get("CAPTIONER_FORCE_CPU", "0") == "1"


def pick_device(torch):
    if force_cpu_mode():
        return torch.device("cpu")
    try:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return torch.device("cpu")


def maybe_empty_cuda_cache(torch):
    if force_cpu_mode():
        return
    try:
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
