from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image, ImageFile, UnidentifiedImageError
from torch.utils.data import Dataset


ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


@dataclass
class Sample:
    image_path: str
    text: str
    prompt_text: str = ""


def image_can_load(path: str | Path) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except (OSError, UnidentifiedImageError):
        return False


class ProductCaptionDataset(Dataset):
    def __init__(self, jsonl_path: str | Path, validate_images: bool = False):
        self.rows = load_jsonl(jsonl_path)

        # keep only valid rows
        clean = []
        skipped_bad_images = 0
        for r in self.rows:
            ip = r.get("image_path")
            txt = r.get("text")
            if not ip or not txt:
                continue
            if validate_images and not image_can_load(ip):
                skipped_bad_images += 1
                continue
            clean.append(Sample(image_path=str(ip), text=str(txt), prompt_text=str(r.get("prompt_text", ""))))
        self.rows = clean
        if skipped_bad_images:
            print(f"Skipped bad images in {jsonl_path}: {skipped_bad_images}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Sample:
        return self.rows[idx]


def load_image_rgb(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")
