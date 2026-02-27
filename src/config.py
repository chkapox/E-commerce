from __future__ import annotations
import torch


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


DEFAULT_MODEL_NAME = "Salesforce/blip-image-captioning-base"