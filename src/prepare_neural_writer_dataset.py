from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable

from .config import DEFAULT_MODEL_NAME, DEFAULT_PRODUCT_PROMPT, configure_hf_offline_mode, configure_project_hf_cache
from .neural_writer_inputs import (
    build_neural_writer_input,
    clean_target_text,
    clean_value,
    facts_from_row,
    normalize_visible_caption,
)
from .ocr import extract_ocr_text_batch


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def get_target(row: dict[str, Any], max_chars: int) -> str:
    return clean_target_text(row.get("target_text") or row.get("text") or row.get("description"), max_chars=max_chars)


def row_id(row: dict[str, Any], fallback: int) -> str:
    for key in ("id", "asin", "image_path"):
        value = clean_value(row.get(key))
        if value:
            return value
    return str(fallback)


def dropout_facts(
    facts: dict[str, Any],
    *,
    rng: random.Random,
    keep_probs: dict[str, float],
) -> tuple[dict[str, Any], list[str]]:
    dropped: list[str] = []
    out = dict(facts)
    for field, keep_prob in keep_probs.items():
        if rng.random() < keep_prob:
            continue
        dropped.append(field)
        if field == "ocr":
            out["ocr_text"] = ""
            out["ocr_terms"] = []
        elif field == "visible_caption":
            out["visible_caption"] = ""
            out["raw_visible_caption"] = ""
        elif field == "brand":
            out["brand"] = ""
            out["brand_source"] = ""
            out["brand_matches"] = []
        else:
            out[field] = ""
    return out, dropped


def validate_probability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare neural writer rows from product metadata, BLIP-visible captions, OCR text, and real descriptions."
    )
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--image_adapter", default="outputs/lora_adapter_description_v2")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--caption_mode", choices=["blip", "field", "none"], default="blip")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--num_beams", type=int, default=3)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PRODUCT_PROMPT,
        help='Optional BLIP decoder prompt for visible captioning. Use "none" to disable.',
    )
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--ocr_backend", choices=["auto", "tesseract", "easyocr", "none"], default="auto")
    parser.add_argument("--ocr_min_confidence", type=float, default=50.0)
    parser.add_argument("--ocr_high_confidence", type=float, default=75.0)
    parser.add_argument("--ocr_batch_size", type=int, default=4)
    parser.add_argument("--easyocr_resize", type=int, default=1024)
    parser.add_argument("--ocr_languages", default="eng")
    parser.add_argument("--tesseract_cmd", default="")
    parser.add_argument("--easyocr_gpu", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--prompt_style",
        choices=["complete", "marketplace", "strict", "amazon_creative"],
        default="amazon_creative",
        help="Input prompt style for the neural writer.",
    )
    parser.add_argument("--max_ocr_terms", type=int, default=8)
    parser.add_argument("--max_target_chars", type=int, default=450)
    parser.add_argument("--min_target_words", type=int, default=8)
    parser.add_argument(
        "--metadata_dropout_variants",
        type=int,
        default=0,
        help="Extra random input variants per row with some facts removed. Targets stay as real descriptions.",
    )
    parser.add_argument("--include_full_variant", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dropout_seed", type=int, default=42)
    parser.add_argument("--keep_title_prob", type=float, default=0.9)
    parser.add_argument("--keep_visible_caption_prob", type=float, default=0.9)
    parser.add_argument("--keep_ocr_prob", type=float, default=0.75)
    parser.add_argument("--keep_brand_prob", type=float, default=0.65)
    parser.add_argument("--keep_category_prob", type=float, default=0.7)
    parser.add_argument("--keep_color_prob", type=float, default=0.6)
    parser.add_argument("--keep_material_prob", type=float, default=0.5)
    parser.add_argument("--keep_size_prob", type=float, default=0.5)
    parser.add_argument("--keep_style_prob", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    configure_project_hf_cache()
    args = parse_args()
    if args.local_files_only:
        configure_hf_offline_mode()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.ocr_batch_size <= 0:
        raise ValueError("ocr_batch_size must be positive")
    if args.metadata_dropout_variants < 0:
        raise ValueError("metadata_dropout_variants must be non-negative")
    keep_probs = {
        "title": args.keep_title_prob,
        "visible_caption": args.keep_visible_caption_prob,
        "ocr": args.keep_ocr_prob,
        "brand": args.keep_brand_prob,
        "category": args.keep_category_prob,
        "color": args.keep_color_prob,
        "material": args.keep_material_prob,
        "size": args.keep_size_prob,
        "style": args.keep_style_prob,
    }
    for name, value in keep_probs.items():
        validate_probability(f"keep_{name}_prob", value)
    rng = random.Random(args.dropout_seed)
    if isinstance(args.prompt, str) and args.prompt.strip().lower() in {"", "none", "null"}:
        args.prompt = None

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    captioner = None
    if args.caption_mode == "blip":
        from .model_wrapper import BlipCaptioner

        captioner = BlipCaptioner(args.model, adapter_path=args.image_adapter, local_files_only=args.local_files_only)

    pending: list[tuple[int, dict[str, Any], Path | None, str]] = []
    rows_out: list[dict[str, Any]] = []
    skipped = {
        "missing_image": 0,
        "missing_target": 0,
        "short_target": 0,
    }
    seen = 0

    def consume() -> None:
        nonlocal pending
        if not pending:
            return

        image_paths = [path for _, _, path, _ in pending if path is not None]
        raw_captions = [""] * len(pending)
        visible_captions = [""] * len(pending)

        if args.caption_mode == "blip" and image_paths:
            path_strings = [str(path) for path in image_paths]
            raw_by_image = captioner.caption_batch(
                path_strings,
                max_new_tokens=args.max_new_tokens,
                num_beams=args.num_beams,
                prompt=args.prompt,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                clean_output=False,
            )
            from .model_wrapper import clean_generation_artifacts, strip_prompt_echo

            clean_by_image = [
                normalize_visible_caption(clean_generation_artifacts(strip_prompt_echo(raw, args.prompt)))
                for raw in raw_by_image
            ]
            image_idx = 0
            for idx, (_, _, path, _) in enumerate(pending):
                if path is None:
                    continue
                raw_captions[idx] = raw_by_image[image_idx]
                visible_captions[idx] = clean_by_image[image_idx]
                image_idx += 1
        elif args.caption_mode == "field":
            for idx, (_, row, _, _) in enumerate(pending):
                visible_captions[idx] = normalize_visible_caption(row.get("visible_caption") or row.get("raw_visible_caption", ""))
                raw_captions[idx] = clean_value(row.get("raw_visible_caption") or visible_captions[idx])

        ocr_results = extract_ocr_text_batch(
            [path for _, _, path, _ in pending if path is not None],
            backend=args.ocr_backend if args.ocr else "none",
            min_confidence=args.ocr_min_confidence,
            high_confidence=args.ocr_high_confidence,
            languages=args.ocr_languages,
            tesseract_cmd=args.tesseract_cmd,
            easyocr_gpu=args.easyocr_gpu,
            easyocr_batch_size=args.ocr_batch_size,
            easyocr_resize=args.easyocr_resize,
        )
        ocr_by_pending: list[dict[str, Any]] = []
        image_idx = 0
        for _, _, path, _ in pending:
            if path is None:
                ocr_by_pending.append({"backend": "none", "text": "", "items": [], "warnings": ["image_missing"]})
                continue
            ocr_by_pending.append(ocr_results[image_idx])
            image_idx += 1

        for (source_idx, row, path, target), raw_caption, visible_caption, ocr_result in zip(
            pending,
            raw_captions,
            visible_captions,
            ocr_by_pending,
        ):
            facts = facts_from_row(
                row,
                visible_caption=visible_caption,
                raw_visible_caption=raw_caption,
                ocr_result=ocr_result,
            )
            base_id = row_id(row, source_idx)
            variants: list[tuple[str, dict[str, Any], list[str]]] = []
            if args.include_full_variant:
                variants.append(("full", facts, []))
            for variant_idx in range(args.metadata_dropout_variants):
                dropped_facts, dropped_fields = dropout_facts(facts, rng=rng, keep_probs=keep_probs)
                variants.append((f"dropout_{variant_idx + 1}", dropped_facts, dropped_fields))

            for variant_name, variant_facts, dropped_fields in variants:
                input_text = build_neural_writer_input(
                    variant_facts,
                    prompt_style=args.prompt_style,
                    max_ocr_terms=args.max_ocr_terms,
                )
                rows_out.append(
                    {
                        "id": base_id if variant_name == "full" else f"{base_id}::{variant_name}",
                        "base_id": base_id,
                        "image_path": str(path) if path is not None else "",
                        "input_text": input_text,
                        "target_text": target,
                        "text": target,
                        "source_target_text": target,
                        "facts": variant_facts,
                        "raw_facts": facts,
                        "raw_visible_caption": raw_caption,
                        "ocr": ocr_result,
                        "prompt_style": args.prompt_style,
                        "metadata_dropout_variant": variant_name,
                        "dropped_input_fields": dropped_fields,
                        "source": row.get("source", ""),
                        "split": row.get("split", ""),
                    }
                )

        pending = []
        print(f"Prepared {len(rows_out)} rows")

    for idx, row in enumerate(iter_jsonl(input_path), start=1):
        if args.limit and seen >= args.limit:
            break
        target = get_target(row, args.max_target_chars)
        if not target:
            skipped["missing_target"] += 1
            continue
        if len(target.split()) < args.min_target_words:
            skipped["short_target"] += 1
            continue

        image_path_value = clean_value(row.get("image_path"))
        image_path = Path(image_path_value) if image_path_value else None
        if args.caption_mode == "blip" or args.ocr:
            if image_path is None or not image_path.exists():
                skipped["missing_image"] += 1
                continue

        pending.append((idx, row, image_path, target))
        seen += 1
        if len(pending) >= args.batch_size:
            consume()

    consume()
    write_jsonl(Path(args.out), rows_out)

    summary = {
        "input_jsonl": args.input_jsonl,
        "out": args.out,
        "rows": len(rows_out),
        "seen": seen,
        "skipped": skipped,
        "caption_mode": args.caption_mode,
        "ocr": args.ocr,
        "prompt_style": args.prompt_style,
        "max_target_chars": args.max_target_chars,
        "min_target_words": args.min_target_words,
        "metadata_dropout_variants": args.metadata_dropout_variants,
        "include_full_variant": args.include_full_variant,
        "dropout_seed": args.dropout_seed,
        "keep_probs": keep_probs,
    }
    if args.summary:
        save_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
