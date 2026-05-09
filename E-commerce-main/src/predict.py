from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, TextIO, Tuple

from .config import DEFAULT_MODEL_NAME, DEFAULT_PRODUCT_PROMPT

if TYPE_CHECKING:
    from .model_wrapper import BlipCaptioner


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_row(f: TextIO, row: Dict[str, Any]) -> None:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flush_pending(
    f: TextIO,
    captioner: BlipCaptioner,
    pending: List[Tuple[Dict[str, Any], str]],
    max_new_tokens: int,
    num_beams: int,
    prompt: str,
) -> int:
    if not pending:
        return 0

    rows, image_paths = zip(*pending)
    predictions = captioner.caption_batch(
        list(image_paths),
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        prompt=prompt,
    )
    for row, pred in zip(rows, predictions):
        write_row(f, {**row, "pred_text": pred, "error": None})
    pending.clear()
    return len(predictions)


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to an image file")
    group.add_argument("--jsonl", help="Path to jsonl with field 'image_path'")

    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HF model name")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--prompt", default=DEFAULT_PRODUCT_PROMPT, help="Optional text prompt for BLIP generation")

    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples for --jsonl (0 = all)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for --jsonl prediction")
    parser.add_argument(
        "--out",
        default="outputs/predictions/preds.jsonl",
        help="Where to save predictions (only for --jsonl mode)",
    )
    parser.add_argument("--adapter", default=None, help="Path to LoRA adapter dir (outputs/lora_adapter)")

    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    from .model_wrapper import BlipCaptioner

    captioner = BlipCaptioner(args.model, adapter_path=args.adapter)

    # Single image mode
    if args.image:
        img_path = Path(args.image)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        text = captioner.caption(
            image_path=str(img_path),
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            prompt=args.prompt,
        )
        print(text)
        return

    # Batch jsonl mode
    in_path = Path(args.jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"jsonl not found: {in_path}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_predicted = 0
    n_written = 0
    pending: List[Tuple[Dict[str, Any], str]] = []

    with out_path.open("w", encoding="utf-8") as f:
        for row in iter_jsonl(in_path):
            if args.limit and n_predicted >= args.limit:
                break

            image_path = row.get("image_path")
            if not image_path:
                continue
            img_path = Path(image_path)
            if not img_path.exists():
                write_row(f, {**row, "pred_text": None, "error": "image_not_found"})
                n_written += 1
                continue

            pending.append((row, str(img_path)))
            if len(pending) >= args.batch_size:
                n = flush_pending(
                    f,
                    captioner,
                    pending,
                    args.max_new_tokens,
                    args.num_beams,
                    args.prompt,
                )
                n_predicted += n
                n_written += n
                print(f"Predicted {n_predicted} samples")

        n = flush_pending(
            f,
            captioner,
            pending,
            args.max_new_tokens,
            args.num_beams,
            args.prompt,
        )
        n_predicted += n
        n_written += n

    print(f"Saved {n_written} rows to {out_path}")


if __name__ == "__main__":
    main()
