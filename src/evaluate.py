from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import evaluate


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compute_metrics(preds: List[str], refs: List[str]) -> Dict[str, Any]:
    # BLEU
    bleu = evaluate.load("sacrebleu")
    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])

    # ROUGE (rouge1/rouge2/rougeL/rougeLsum)
    rouge = evaluate.load("rouge")
    rouge_res = rouge.compute(predictions=preds, references=refs)

    return {
        "sacrebleu": float(bleu_res["score"]),
        "rouge1": float(rouge_res["rouge1"]),
        "rouge2": float(rouge_res["rouge2"]),
        "rougeL": float(rouge_res["rougeL"]),
        "rougeLsum": float(rouge_res["rougeLsum"]),
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", required=True, help="Path to preds jsonl (must have 'text' and 'pred_text')")
    parser.add_argument("--out_metrics", default="outputs/metrics_baseline.json")
    parser.add_argument("--out_samples", default="outputs/samples_baseline.md")
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

    print("Saved metrics to:", args.out_metrics)
    print("Saved samples to:", args.out_samples)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()