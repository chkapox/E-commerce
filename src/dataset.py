from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset


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


class ProductCaptionDataset(Dataset):
    def __init__(self, jsonl_path: str | Path):
        self.rows = load_jsonl(jsonl_path)

        # keep only valid rows
        clean = []
        for r in self.rows:
            ip = r.get("image_path")
            txt = r.get("text")
            if not ip or not txt:
                continue
            clean.append(Sample(image_path=str(ip), text=str(txt)))
        self.rows = clean

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Sample:
        return self.rows[idx]


def load_image_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")