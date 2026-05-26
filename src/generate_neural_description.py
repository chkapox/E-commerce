from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_MODEL_NAME,
    DEFAULT_PRODUCT_PROMPT,
    configure_hf_offline_mode,
    configure_project_hf_cache,
    pick_device,
)
from .neural_writer_inputs import (
    build_neural_writer_input,
    clean_value,
    facts_from_row,
    generated_description_warnings,
    normalize_visible_caption,
)
from .ocr import extract_ocr_text


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_generated_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def looks_like_field_fragment(text: str) -> bool:
    text = clean_generated_text(text)
    if not text:
        return True
    words = text.split()
    lowered = text.lower()
    field_prefixes = (
        "material:",
        "color:",
        "size:",
        "size or pack count:",
        "style:",
        "brand:",
        "category:",
    )
    if lowered.startswith(field_prefixes) and len(words) <= 12:
        return True
    return len(words) <= 4


def remove_sparse_metadata(facts: dict[str, Any]) -> dict[str, Any]:
    retry_facts = dict(facts)
    for key in ("color", "material", "size", "style"):
        retry_facts[key] = ""
    return retry_facts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a neural marketplace description from image caption, OCR, title, and metadata facts."
    )
    parser.add_argument("--image", required=True, help="Path to the product image")
    parser.add_argument("--title", default="", help="Trusted product title or product name")
    parser.add_argument("--brand", default="")
    parser.add_argument("--category", default="")
    parser.add_argument("--color", default="")
    parser.add_argument("--material", default="")
    parser.add_argument("--size", default="")
    parser.add_argument("--style", default="")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="HF BLIP model name")
    parser.add_argument("--image_adapter", default="outputs/lora_adapter_description_v2")
    parser.add_argument(
        "--text_writer",
        default="outputs/text_writer_flan_t5_base_lora_neural_marketplace_v3",
        help="Seq2seq writer model or LoRA adapter that generates the final description.",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=40, help="BLIP visible-caption token budget")
    parser.add_argument("--num_beams", type=int, default=3, help="BLIP beam count")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PRODUCT_PROMPT,
        help='Optional BLIP decoder prompt. Use "none" to disable.',
    )
    parser.add_argument("--ocr", action="store_true")
    parser.add_argument("--ocr_backend", choices=["auto", "tesseract", "easyocr", "none"], default="auto")
    parser.add_argument("--ocr_min_confidence", type=float, default=50.0)
    parser.add_argument("--ocr_high_confidence", type=float, default=75.0)
    parser.add_argument("--ocr_languages", default="eng")
    parser.add_argument("--tesseract_cmd", default="")
    parser.add_argument("--easyocr_gpu", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--prompt_style",
        choices=["complete", "marketplace", "strict", "amazon_creative"],
        default="complete",
        help="Prompt style used for the neural text writer.",
    )
    parser.add_argument("--max_ocr_terms", type=int, default=8)
    parser.add_argument("--writer_max_input_length", type=int, default=384)
    parser.add_argument("--writer_max_new_tokens", type=int, default=140)
    parser.add_argument("--writer_num_beams", type=int, default=4)
    parser.add_argument("--writer_no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--writer_repetition_penalty", type=float, default=1.15)
    parser.add_argument("--out", default="", help="Optional JSON output path")
    return parser.parse_args()


def main() -> None:
    configure_project_hf_cache()
    args = parse_args()
    if args.local_files_only:
        configure_hf_offline_mode()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if isinstance(args.prompt, str) and args.prompt.strip().lower() in {"", "none", "null"}:
        args.prompt = None

    from .model_wrapper import BlipCaptioner, clean_generation_artifacts, strip_prompt_echo
    from .predict_text_writer import generate_batch, load_writer_model

    captioner = BlipCaptioner(args.model, adapter_path=args.image_adapter, local_files_only=args.local_files_only)
    raw_visible_caption = captioner.caption(
        str(image_path),
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        prompt=args.prompt,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
        clean_output=False,
    )
    visible_caption = normalize_visible_caption(clean_generation_artifacts(strip_prompt_echo(raw_visible_caption, args.prompt)))

    ocr_result = (
        extract_ocr_text(
            image_path,
            backend=args.ocr_backend,
            min_confidence=args.ocr_min_confidence,
            high_confidence=args.ocr_high_confidence,
            languages=args.ocr_languages,
            tesseract_cmd=args.tesseract_cmd,
            easyocr_gpu=args.easyocr_gpu,
        )
        if args.ocr
        else {"backend": "none", "text": "", "items": [], "lines": [], "warnings": ["ocr_disabled"]}
    )

    row = {
        "image_path": str(image_path),
        "title": args.title,
        "brand": args.brand,
        "category": args.category,
        "color": args.color,
        "material": args.material,
        "size": args.size,
        "style": args.style,
    }
    facts = facts_from_row(
        row,
        visible_caption=visible_caption,
        raw_visible_caption=raw_visible_caption,
        ocr_result=ocr_result,
    )
    source_facts = dict(facts)
    input_text = build_neural_writer_input(
        facts,
        prompt_style=args.prompt_style,
        max_ocr_terms=args.max_ocr_terms,
    )

    device, _ = pick_device()
    tokenizer, writer_model = load_writer_model(args.text_writer, args.local_files_only, device)
    description = clean_generated_text(
        generate_batch(
            tokenizer=tokenizer,
            model=writer_model,
            device=device,
            input_texts=[input_text],
            max_input_length=args.writer_max_input_length,
            max_new_tokens=args.writer_max_new_tokens,
            num_beams=args.writer_num_beams,
            no_repeat_ngram_size=args.writer_no_repeat_ngram_size,
            repetition_penalty=args.writer_repetition_penalty,
        )[0]
    )
    retry_warnings: list[str] = []
    if looks_like_field_fragment(description) and any(facts.get(key) for key in ("color", "material", "size", "style")):
        retry_warnings.append("field_fragment_from_sparse_metadata")
        retry_facts = remove_sparse_metadata(facts)
        retry_input_text = build_neural_writer_input(
            retry_facts,
            prompt_style=args.prompt_style,
            max_ocr_terms=args.max_ocr_terms,
        )
        retry_description = clean_generated_text(
            generate_batch(
                tokenizer=tokenizer,
                model=writer_model,
                device=device,
                input_texts=[retry_input_text],
                max_input_length=args.writer_max_input_length,
                max_new_tokens=args.writer_max_new_tokens,
                num_beams=args.writer_num_beams,
                no_repeat_ngram_size=args.writer_no_repeat_ngram_size,
                repetition_penalty=args.writer_repetition_penalty,
            )[0]
        )
        retry_fact_warnings = generated_description_warnings(retry_description, source_facts)
        lost_trusted_optional_fact = any(
            f"{key}_not_preserved" in retry_fact_warnings
            for key in ("color", "material", "size", "style")
            if source_facts.get(key)
        )
        if retry_description and not looks_like_field_fragment(retry_description) and not lost_trusted_optional_fact:
            description = retry_description
            input_text = retry_input_text
            facts = retry_facts
        elif lost_trusted_optional_fact:
            retry_warnings.append("retry_rejected_missing_trusted_fact")

    result = {
        "description": description,
        "description_source": "neural_writer",
        "facts": facts,
        "source_facts": source_facts,
        "input_text": input_text,
        "raw_visible_caption": clean_value(raw_visible_caption),
        "visible_caption": visible_caption,
        "ocr": ocr_result,
        "warnings": list(
            dict.fromkeys(
                generated_description_warnings(description, facts)
                + [str(warning) for warning in ocr_result.get("warnings", [])]
                + retry_warnings
            )
        ),
        "models": {
            "image_model": args.model,
            "image_adapter": args.image_adapter,
            "text_writer": args.text_writer,
            "prompt_style": args.prompt_style,
        },
    }

    if args.out:
        save_json(Path(args.out), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
