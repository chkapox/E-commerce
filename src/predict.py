from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import DEFAULT_MODEL_NAME
from .model_wrapper import BlipCaptioner


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to an image file")
    group.add_argument("--jsonl", help="Path to jsonl with field 'image_path'")

    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HF model name")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--num_beams", type=int, default=3)

    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples for --jsonl (0 = all)")
    parser.add_argument(
        "--out",
        default="outputs/predictions/preds.jsonl",
        help="Where to save predictions (only for --jsonl mode)",
    )

    args = parser.parse_args()

    captioner = BlipCaptioner(args.model)

    # Single image mode
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        text = captioner.caption(
            image_path=str(img_path),
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        print(text)
        return

    # Batch jsonl mode
    in_path = Path(args.jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"jsonl not found: {in_path}")

    out_path = Path(args.out)
    preds = []

    n = 0
    for row in iter_jsonl(in_path):
        image_path = row.get("image_path")
        if not image_path:
            continue
        img_path = Path(image_path)
        if not img_path.exists():
            # сохраняем факт пропуска, чтобы отлаживать датасет
            preds.append({**row, "pred_text": None, "error": "image_not_found"})
            continue

        pred = captioner.caption(
            image_path=str(img_path),
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
        preds.append({**row, "pred_text": pred, "error": None})

        n += 1
        if args.limit and n >= args.limit:
            break

    write_jsonl(out_path, preds)
    print(f"Saved {len(preds)} predictions to {out_path}")


if __name__ == "__main__":
    main()