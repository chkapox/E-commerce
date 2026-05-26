from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from .neural_writer_inputs import build_neural_writer_input, clean_target_text, clean_value, normalize_for_match


FACT_FIELDS = (
    "title",
    "brand",
    "brand_source",
    "brand_matches",
    "category",
    "color",
    "material",
    "size",
    "style",
    "visible_caption",
    "raw_visible_caption",
    "ocr_text",
    "raw_ocr_text",
    "ocr_terms",
)
OPTIONAL_DROPOUT_FIELDS = ("ocr", "brand", "category", "color", "material", "size", "style")
SPARSE_FIELDS = ("material", "color", "brand", "style", "size")
GENERIC_VISUAL_TOKENS = {
    "black",
    "blue",
    "brown",
    "gift",
    "green",
    "item",
    "orange",
    "pair",
    "pink",
    "product",
    "purple",
    "red",
    "set",
    "silver",
    "white",
    "yellow",
}


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


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def cached_facts(row: Mapping[str, Any]) -> dict[str, Any]:
    source = row.get("raw_facts") or row.get("facts")
    if not isinstance(source, Mapping):
        raise ValueError("Cached writer row does not contain raw_facts or facts.")
    facts = {field: source.get(field, [] if field in {"brand_matches", "ocr_terms"} else "") for field in FACT_FIELDS}
    if not isinstance(facts["brand_matches"], list):
        facts["brand_matches"] = []
    if not isinstance(facts["ocr_terms"], list):
        facts["ocr_terms"] = []
    return facts


def target_text(row: Mapping[str, Any], max_chars: int) -> str:
    return clean_target_text(
        row.get("source_target_text") or row.get("target_text") or row.get("text"),
        max_chars=max_chars,
    )


def meaningful_tokens(value: Any) -> list[str]:
    ignored = {
        "and",
        "with",
        "pack",
        "count",
        "inch",
        "inches",
        "ounce",
        "ounces",
        "size",
        "style",
        "the",
    }
    return [
        token
        for token in normalize_for_match(clean_value(value)).split()
        if len(token) >= 3 and token not in ignored and not token.isdigit()
    ]


def fact_supported_by_target(field: str, value: Any, target: str) -> bool:
    value_text = clean_value(value)
    if not value_text:
        return False
    normalized_target = normalize_for_match(target)
    normalized_value = normalize_for_match(value_text)
    if normalized_value and normalized_value in normalized_target:
        return True
    if field in {"brand", "size"}:
        return False
    tokens = meaningful_tokens(value_text)
    return bool(tokens) and all(token in normalized_target.split() for token in tokens)


def visible_caption_supported(facts: Mapping[str, Any], target: str) -> bool:
    caption_tokens = [
        token
        for token in meaningful_tokens(facts.get("visible_caption", ""))
        if token not in GENERIC_VISUAL_TOKENS
    ]
    if not caption_tokens:
        return False
    reference = normalize_for_match(f"{facts.get('title', '')} {target}").split()
    return any(token in reference for token in caption_tokens)


def looks_like_field_list_target(target: str) -> bool:
    field_lead = r"^(?:material|size(?:\s+and\s+packs?)?|dimensions|color|brand|style|specifications?)\s*[:\-]"
    if re.search(field_lead, target, flags=re.IGNORECASE):
        return True
    label_pattern = r"(?:^|[.!?]\s+)(?:material|size|dimensions|color|brand|style|features?|specifications?)\s*[:\-]"
    hits = re.findall(label_pattern, target, flags=re.IGNORECASE)
    return len(hits) >= 2 or (bool(hits) and len(target.split()) <= 12)


def clear_fact(facts: dict[str, Any], field: str) -> None:
    if field == "ocr":
        facts["ocr_text"] = ""
        facts["ocr_terms"] = []
    elif field == "brand":
        facts["brand"] = ""
        facts["brand_source"] = ""
        facts["brand_matches"] = []
    else:
        facts[field] = ""


def safe_dropout_facts(
    facts: Mapping[str, Any],
    *,
    target: str,
    rng: random.Random,
    keep_prob: float,
) -> tuple[dict[str, Any], list[str]]:
    variant = dict(facts)
    variant["brand_matches"] = list(facts.get("brand_matches", []))
    variant["ocr_terms"] = list(facts.get("ocr_terms", []))
    dropped: list[str] = []
    for field in OPTIONAL_DROPOUT_FIELDS:
        if field == "ocr":
            value = facts.get("ocr_text") or " ".join(str(term) for term in facts.get("ocr_terms", []))
        else:
            value = facts.get(field, "")
        if not clean_value(value):
            continue
        # A stated fact remains visible whenever the reference description relies on it.
        if fact_supported_by_target(field, value, target) or rng.random() < keep_prob:
            continue
        clear_fact(variant, field)
        dropped.append(field)
    return variant, dropped


def supported_sparse_variants(
    facts: Mapping[str, Any],
    target: str,
    limit: int,
) -> list[tuple[str, dict[str, Any], list[str]]]:
    if limit <= 0 or not clean_value(facts.get("visible_caption", "")) or not visible_caption_supported(facts, target):
        return []
    variants: list[tuple[str, dict[str, Any], list[str]]] = []
    for field in SPARSE_FIELDS:
        if not fact_supported_by_target(field, facts.get(field, ""), target):
            continue
        sparse = {key: [] if key in {"brand_matches", "ocr_terms"} else "" for key in FACT_FIELDS}
        sparse["visible_caption"] = facts.get("visible_caption", "")
        sparse["raw_visible_caption"] = facts.get("raw_visible_caption", "")
        sparse[field] = facts.get(field, "")
        if field == "brand":
            sparse["brand_source"] = facts.get("brand_source", "")
            sparse["brand_matches"] = list(facts.get("brand_matches", []))
        kept = {"visible_caption", "raw_visible_caption", field}
        dropped = [
            key for key in ("title", "brand", "category", "color", "material", "size", "style", "ocr") if key not in kept
        ]
        variants.append((f"supported_sparse_{field}", sparse, dropped))
        if len(variants) >= limit:
            break
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quickly rebuild neural-writer rows from cached facts; no image captioning or OCR is rerun."
    )
    parser.add_argument("--input_jsonl", required=True, help="Existing prepared writer JSONL containing raw_facts.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", default="")
    parser.add_argument("--prompt_style", choices=["complete", "marketplace", "strict", "amazon_creative"], default="complete")
    parser.add_argument("--max_target_chars", type=int, default=450)
    parser.add_argument("--min_target_words", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Maximum unique full source rows to rebuild; 0 means all.")
    parser.add_argument("--safe_dropout_variants", type=int, default=0)
    parser.add_argument("--dropout_keep_prob", type=float, default=0.45)
    parser.add_argument(
        "--supported_sparse_variants",
        type=int,
        default=0,
        help="Experimental: add caption-plus-one-fact rows when cached evidence appears to support both.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--filter_field_list_targets", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.safe_dropout_variants < 0 or args.supported_sparse_variants < 0:
        raise ValueError("Variant counts must be non-negative.")
    if not 0.0 <= args.dropout_keep_prob <= 1.0:
        raise ValueError("dropout_keep_prob must be between 0 and 1.")
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Cached writer JSONL not found: {input_path}")

    rng = random.Random(args.seed)
    rebuilt: list[dict[str, Any]] = []
    seen_base_ids: set[str] = set()
    skipped = {"duplicate_or_old_dropout": 0, "missing_target": 0, "short_target": 0, "field_list_target": 0}
    variant_counts: dict[str, int] = {}

    for row in iter_jsonl(input_path):
        variant_name = clean_value(row.get("metadata_dropout_variant", "full"))
        base_id = clean_value(row.get("base_id") or row.get("id"))
        if variant_name not in {"", "full"} or base_id in seen_base_ids:
            skipped["duplicate_or_old_dropout"] += 1
            continue
        if args.limit and len(seen_base_ids) >= args.limit:
            break
        seen_base_ids.add(base_id)
        target = target_text(row, args.max_target_chars)
        if not target:
            skipped["missing_target"] += 1
            continue
        if len(target.split()) < args.min_target_words:
            skipped["short_target"] += 1
            continue
        if args.filter_field_list_targets and looks_like_field_list_target(target):
            skipped["field_list_target"] += 1
            continue
        facts = cached_facts(row)
        variants: list[tuple[str, dict[str, Any], list[str]]] = [("full", facts, [])]
        for idx in range(args.safe_dropout_variants):
            dropped_facts, dropped = safe_dropout_facts(
                facts,
                target=target,
                rng=rng,
                keep_prob=args.dropout_keep_prob,
            )
            variants.append((f"safe_dropout_{idx + 1}", dropped_facts, dropped))
        variants.extend(supported_sparse_variants(facts, target, args.supported_sparse_variants))

        for name, variant_facts, dropped_fields in variants:
            variant_counts[name] = variant_counts.get(name, 0) + 1
            rebuilt.append(
                {
                    "id": base_id if name == "full" else f"{base_id}::{name}",
                    "base_id": base_id,
                    "image_path": row.get("image_path", ""),
                    "input_text": build_neural_writer_input(variant_facts, prompt_style=args.prompt_style),
                    "target_text": target,
                    "text": target,
                    "source_target_text": target,
                    "facts": variant_facts,
                    "raw_facts": facts,
                    "raw_visible_caption": row.get("raw_visible_caption", facts.get("raw_visible_caption", "")),
                    "ocr": row.get("ocr", {}),
                    "prompt_style": args.prompt_style,
                    "metadata_dropout_variant": name,
                    "dropped_input_fields": dropped_fields,
                    "source": row.get("source", ""),
                    "split": row.get("split", ""),
                    "rebuilt_from_cached_facts": True,
                }
            )

    write_jsonl(Path(args.out), rebuilt)
    summary = {
        "input_jsonl": str(input_path),
        "out": args.out,
        "prompt_style": args.prompt_style,
        "unique_source_rows": len(seen_base_ids),
        "rows": len(rebuilt),
        "variant_counts": variant_counts,
        "skipped": skipped,
        "safe_dropout_variants": args.safe_dropout_variants,
        "supported_sparse_variants": args.supported_sparse_variants,
        "filter_field_list_targets": args.filter_field_list_targets,
        "note": "Cached rebuild only; BLIP and OCR were not run.",
    }
    if args.summary:
        save_json(Path(args.summary), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
