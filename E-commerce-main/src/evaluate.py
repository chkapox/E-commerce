from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .text_utils import COLOR_ALIASES, PRODUCT_ALIASES, extract_alias_matches, token_overlap


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, Any]:
    if not preds:
        raise ValueError("No valid predictions to evaluate")

    import evaluate as evaluate_lib

    # BLEU
    bleu = evaluate_lib.load("sacrebleu")
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])

    # ROUGE (rouge1/rouge2/rougeL/rougeLsum)
    rouge = evaluate_lib.load("rouge")
    rouge_res = rouge.compute(predictions=preds, references=refs)

    overlap_rows = [token_overlap(pred, ref) for pred, ref in zip(preds, refs)]
    color_scores = []
    product_scores = []

    for pred, ref in zip(preds, refs):
        ref_colors = extract_alias_matches(ref, COLOR_ALIASES)
        pred_colors = extract_alias_matches(pred, COLOR_ALIASES)
        if ref_colors:
            color_scores.append(len(ref_colors & pred_colors) / len(ref_colors))

        ref_products = extract_alias_matches(ref, PRODUCT_ALIASES)
        pred_products = extract_alias_matches(pred, PRODUCT_ALIASES)
        if ref_products:
            product_scores.append(len(ref_products & pred_products) / len(ref_products))

    return {
        "sacrebleu": float(bleu_res["score"]),
        "rouge1": float(rouge_res["rouge1"]),
        "rouge2": float(rouge_res["rouge2"]),
        "rougeL": float(rouge_res["rougeL"]),
        "rougeLsum": float(rouge_res["rougeLsum"]),
        "token_precision": float(sum(row["precision"] for row in overlap_rows) / len(overlap_rows)),
        "token_recall": float(sum(row["recall"] for row in overlap_rows) / len(overlap_rows)),
        "token_f1": float(sum(row["f1"] for row in overlap_rows) / len(overlap_rows)),
        "color_recall": float(sum(color_scores) / len(color_scores)) if color_scores else None,
        "color_eval_n": len(color_scores),
        "product_type_recall": float(sum(product_scores) / len(product_scores)) if product_scores else None,
        "product_type_eval_n": len(product_scores),
        "n": len(preds),
    }


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_samples_md(path: Path, rows: List[Tuple[str, str, str]], limit: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Samples (reference vs prediction)\n"]
    for i, (img, ref, pred) in enumerate(rows[:limit], start=1):
        lines.append(f"## {i}\n")
        lines.append(f"- image_path: `{img}`\n")
        lines.append(f"- reference: {ref}\n")
        lines.append(f"- prediction: {pred}\n")
        lines.append("\n")
    path.write_text("".join(lines), encoding="utf-8")


def save_diagnostics_md(path: Path, rows: List[Tuple[str, str, str]], limit: int = 80) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Prediction Diagnostics\n\n"]
    n_written = 0

    for img, ref, pred in rows:
        ref_colors = extract_alias_matches(ref, COLOR_ALIASES)
        pred_colors = extract_alias_matches(pred, COLOR_ALIASES)
        ref_products = extract_alias_matches(ref, PRODUCT_ALIASES)
        pred_products = extract_alias_matches(pred, PRODUCT_ALIASES)

        missing_colors = sorted(ref_colors - pred_colors)
        missing_products = sorted(ref_products - pred_products)
        extra_colors = sorted(pred_colors - ref_colors)
        extra_products = sorted(pred_products - ref_products)

        if not (missing_colors or missing_products or extra_colors or extra_products):
            continue

        n_written += 1
        lines.append(f"## {n_written}\n")
        lines.append(f"- image_path: `{img}`\n")
        lines.append(f"- reference: {ref}\n")
        lines.append(f"- prediction: {pred}\n")
        if missing_colors:
            lines.append(f"- missing reference colors: {', '.join(missing_colors)}\n")
        if missing_products:
            lines.append(f"- missing product types: {', '.join(missing_products)}\n")
        if extra_colors:
            lines.append(f"- extra predicted colors: {', '.join(extra_colors)}\n")
        if extra_products:
            lines.append(f"- extra predicted product types: {', '.join(extra_products)}\n")
        lines.append("\n")

        if n_written >= limit:
            break

    if n_written == 0:
        lines.append("No color or product-type mismatches found in the inspected rows.\n")

    path.write_text("".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="Path to preds jsonl (must have 'text' and 'pred_text')")
    parser.add_argument("--out_metrics", default="outputs/metrics_baseline.json")
    parser.add_argument("--out_samples", default="outputs/samples_baseline.md")
    parser.add_argument("--out_diagnostics", default=None)
    parser.add_argument("--max_samples", type=int, default=50)
    args = parser.parse_args()

    preds_path = Path(args.preds)
    if not preds_path.exists():
        raise FileNotFoundError(f"preds file not found: {preds_path}")

    preds: List[str] = []
    refs: List[str] = []
    samples: List[Tuple[str, str, str]] = []

    for row in iter_jsonl(preds_path):
        if row.get("error") is not None:
            continue
        ref = row.get("text")
        pred = row.get("pred_text")
        img = row.get("image_path", "")
        if not ref or not pred:
            continue
        refs.append(str(ref))
        preds.append(str(pred))
        samples.append((img, str(ref), str(pred)))

    metrics = compute_metrics(preds, refs)
    save_json(Path(args.out_metrics), metrics)
    save_samples_md(Path(args.out_samples), samples, limit=args.max_samples)
    if args.out_diagnostics:
        save_diagnostics_md(Path(args.out_diagnostics), samples, limit=args.max_samples)

    print("Saved metrics to:", args.out_metrics)
    print("Saved samples to:", args.out_samples)
    if args.out_diagnostics:
        print("Saved diagnostics to:", args.out_diagnostics)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
