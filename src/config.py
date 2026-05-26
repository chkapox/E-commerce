from __future__ import annotations
import os
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_HF_CACHE = PROJECT_ROOT / ".hf_cache"


def configure_project_hf_cache() -> None:
    if not PROJECT_HF_CACHE.exists():
        return
    os.environ.setdefault("HF_HOME", str(PROJECT_HF_CACHE))
    os.environ.setdefault("HF_HUB_CACHE", str(PROJECT_HF_CACHE / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_HF_CACHE / "transformers"))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_HF_CACHE / "xdg"))


def configure_hf_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


DEFAULT_MODEL_NAME = "Salesforce/blip-image-captioning-base"
DEFAULT_PRODUCT_PROMPT = None
