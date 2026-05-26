from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .text_utils import COLOR_ALIASES, PRODUCT_ALIASES, extract_alias_matches, token_overlap, tokenize


GENERIC_FALLBACKS = {
    "product",
    "the product",
    "item",
    "the item",
    "this product",
    "a product",
    "an item",
}

PROMPT_ARTIFACTS = (
    "facts:",
    "description:",
    "product title:",
    "visible image description:",
    "ocr text:",
)

KNOWN_BAD_STRINGS = (
    "inflatori",
    "hydraulique",
    "hydrauilia",
    "cylindresses",
    "coloreded",
    "gogles",
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_for_contains(text: str) -> str:
    return " ".join(str(text).lower().split())


def normalize_value(value: str) -> str:
    return " ".join(tokenize(value))


def word_count(text: str) -> int:
    return len(tokenize(text))


def sentence_count(text: str) -> int:
    text = str(text).strip()
    if not text:
        return 0
    return max(1, text.count(".") + text.count("!") + text.count("?"))


def repeated_ngram_count(text: str, n: int = 3) -> int:
    tokens = tokenize(text)
    if len(tokens) < n:
        return 0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    return sum(count - 1 for count in counts.values() if count > 1)


def pick_reference(row: dict[str, Any]) -> str:
    for key in ("target_text", "text", "source_target_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def metadata_values(facts: dict[str, Any]) -> list[str]:
    metadata = facts.get("metadata")
    if not isinstance(metadata, dict):
        return []
    values: list[str] = []
    for value in metadata.values():
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return values


def ocr_terms(row: dict[str, Any], facts: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    facts_terms = facts.get("ocr_terms")
    if isinstance(facts_terms, list):
        terms.extend(str(term).strip() for term in facts_terms if str(term).strip())

    ocr = row.get("ocr")
    if isinstance(ocr, dict):
        high_confidence = ocr.get("high_confidence_terms")
        if isinstance(high_confidence, list):
            terms.extend(str(term).strip() for term in high_confidence if str(term).strip())

    return sorted(set(terms))


def value_is_preserved(prediction: str, value: str) -> bool:
    pred_norm = normalize_value(prediction)
    value_norm = normalize_value(value)
    return bool(value_norm and value_norm in pred_norm)


def summarize_run(label: str, path: Path, limit: int = 0) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for row in iter_jsonl(path):
        if limit and len(rows) >= limit:
            break
        if row.get("error") is not None:
            continue
        prediction = row.get("pred_text")
        if not isinstance(prediction, str) or not prediction.strip():
            continue
        rows.append(row)

    if not rows:
        raise ValueError(f"No valid predictions found in {path}")

    overlap_rows: list[dict[str, float]] = []
    word_counts: list[int] = []
    sentence_counts: list[int] = []
    repeated_rows = 0
    prompt_artifact_rows = 0
    known_bad_rows = 0
    generic_rows = 0
    exact_title_checks = 0
    exact_title_hits = 0
    metadata_checks = 0
    metadata_hits = 0
    ocr_checks = 0
    ocr_hits = 0
    color_scores: list[float] = []
    product_scores: list[float] = []
    warnings = Counter()
    samples: list[dict[str, Any]] = []

    for row in rows:
        prediction = str(row["pred_text"]).strip()
        reference = pick_reference(row)
        normalized_prediction = normalize_for_contains(prediction).strip(" .,:;!-")
        prediction_lower = prediction.lower()
        facts = row.get("facts") if isinstance(row.get("facts"), dict) else {}

        if reference:
            overlap_rows.append(token_overlap(prediction, reference))

            ref_colors = extract_alias_matches(reference, COLOR_ALIASES)
            pred_colors = extract_alias_matches(prediction, COLOR_ALIASES)
            if ref_colors:
                color_scores.append(len(ref_colors & pred_colors) / len(ref_colors))

            ref_products = extract_alias_matches(reference, PRODUCT_ALIASES)
            pred_products = extract_alias_matches(prediction, PRODUCT_ALIASES)
            if ref_products:
                product_scores.append(len(ref_products & pred_products) / len(ref_products))

        word_counts.append(word_count(prediction))
        sentence_counts.append(sentence_count(prediction))
        repeated = repeated_ngram_count(prediction)
        if repeated:
            repeated_rows += 1
        has_prompt_artifact = any(artifact in prediction_lower for artifact in PROMPT_ARTIFACTS)
        has_known_bad_string = any(bad in prediction_lower for bad in KNOWN_BAD_STRINGS)

        if has_prompt_artifact:
            prompt_artifact_rows += 1
        if has_known_bad_string:
            known_bad_rows += 1
        if normalized_prediction in GENERIC_FALLBACKS:
            generic_rows += 1

        title = facts.get("title") if isinstance(facts, dict) else None
        if isinstance(title, str) and title.strip():
            exact_title_checks += 1
            if value_is_preserved(prediction, title):
                exact_title_hits += 1

        for value in metadata_values(facts):
            metadata_checks += 1
            if value_is_preserved(prediction, value):
                metadata_hits += 1

        terms = ocr_terms(row, facts)
        if terms:
            ocr_checks += 1
            if any(value_is_preserved(prediction, term) for term in terms):
                ocr_hits += 1

        for warning_key in ("target_warnings", "template_warnings"):
            warning_values = row.get(warning_key)
            if isinstance(warning_values, list):
                warnings.update(str(value) for value in warning_values)

        repeated_mismatch = bool(repeated and reference and normalize_for_contains(prediction) != normalize_for_contains(reference))
        if len(samples) < 5 and (repeated_mismatch or has_known_bad_string or has_prompt_artifact):
            samples.append(
                {
                    "id": row.get("id"),
                    "reference": reference,
                    "prediction": prediction,
                }
            )

    n = len(rows)
    summary = {
        "label": label,
        "path": str(path),
        "n": n,
        "avg_pred_words": sum(word_counts) / n,
        "avg_pred_sentences": sum(sentence_counts) / n,
        "token_precision": avg(row["precision"] for row in overlap_rows),
        "token_recall": avg(row["recall"] for row in overlap_rows),
        "token_f1": avg(row["f1"] for row in overlap_rows),
        "color_recall": avg(color_scores),
        "color_eval_n": len(color_scores),
        "product_type_recall": avg(product_scores),
        "product_type_eval_n": len(product_scores),
        "generic_fallback_rate": generic_rows / n,
        "prompt_artifact_rate": prompt_artifact_rows / n,
        "repetition_rate": repeated_rows / n,
        "known_bad_string_rate": known_bad_rows / n,
        "exact_title_preservation_rate": exact_title_hits / exact_title_checks if exact_title_checks else None,
        "exact_title_eval_n": exact_title_checks,
        "metadata_value_preservation_rate": metadata_hits / metadata_checks if metadata_checks else None,
        "metadata_value_eval_n": metadata_checks,
        "ocr_term_use_rate": ocr_hits / ocr_checks if ocr_checks else None,
        "ocr_eval_n": ocr_checks,
        "top_warnings": warnings.most_common(10),
        "samples": samples,
    }
    return summary


def avg(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Run label is empty in {value!r}")
    return label, Path(raw_path.strip())


def format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        ("Run", "label"),
        ("n", "n"),
        ("Tok F1", "token_f1"),
        ("Words", "avg_pred_words"),
        ("Prompt Artifacts", "prompt_artifact_rate"),
        ("Repetition", "repetition_rate"),
        ("Bad Strings", "known_bad_string_rate"),
        ("Title Exact", "exact_title_preservation_rate"),
        ("Metadata Exact", "metadata_value_preservation_rate"),
        ("OCR Use", "ocr_term_use_rate"),
    ]

    lines = ["# Generation Run Comparison\n\n"]
    lines.append(
        "This report uses lightweight automatic checks as a triage aid. "
        "Final claims about hallucination, OCR usefulness, and product-page quality still need manual review.\n\n"
    )
    lines.append("| " + " | ".join(title for title, _ in columns) + " |\n")
    lines.append("| " + " | ".join("---" for _ in columns) + " |\n")
    for summary in summaries:
        lines.append("| " + " | ".join(format_metric(summary[key]) for _, key in columns) + " |\n")

    lines.append("\n## Notes By Run\n\n")
    for summary in summaries:
        lines.append(f"### {summary['label']}\n\n")
        lines.append(f"- path: `{summary['path']}`\n")
        lines.append(f"- color recall: {format_metric(summary['color_recall'])} ({summary['color_eval_n']} rows)\n")
        lines.append(
            f"- product type recall: {format_metric(summary['product_type_recall'])} "
            f"({summary['product_type_eval_n']} rows)\n"
        )
        lines.append(
            f"- exact title checks: {summary['exact_title_eval_n']}; "
            f"metadata value checks: {summary['metadata_value_eval_n']}; "
            f"OCR checks: {summary['ocr_eval_n']}\n"
        )
        if summary["top_warnings"]:
            warning_text = ", ".join(f"{name} ({count})" for name, count in summary["top_warnings"])
            lines.append(f"- common warnings: {warning_text}\n")
        if summary["samples"]:
            lines.append("\nPotential issue samples:\n\n")
            for sample in summary["samples"]:
                lines.append(f"- id `{sample.get('id')}`\n")
                lines.append(f"  - reference: {sample.get('reference', '')}\n")
                lines.append(f"  - prediction: {sample.get('prediction', '')}\n")
        lines.append("\n")

    path.write_text("".join(lines), encoding="utf-8")


def default_runs() -> list[str]:
    return [
        "v3_test=outputs/predictions/test_neural_marketplace_v3_full.jsonl",
        "v3_validation=outputs/predictions/val_neural_marketplace_v3_200.jsonl",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare product-description generation runs with lightweight groundedness checks."
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run spec in the form label=path. May be repeated. Defaults to current report comparison files.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows per run")
    parser.add_argument("--out_json", default="outputs/generation_run_comparison.json")
    parser.add_argument("--out_md", default="outputs/generation_run_comparison.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_specs = args.run or default_runs()
    summaries: list[dict[str, Any]] = []

    for spec in run_specs:
        label, path = parse_run(spec)
        if not path.exists():
            print(f"Skipping missing run {label}: {path}")
            continue
        summaries.append(summarize_run(label, path, limit=args.limit))

    if not summaries:
        raise ValueError("No runs were available to compare")

    save_json(Path(args.out_json), summaries)
    save_markdown(Path(args.out_md), summaries)
    print(f"Saved JSON comparison to {args.out_json}")
    print(f"Saved Markdown comparison to {args.out_md}")


if __name__ == "__main__":
    main()
