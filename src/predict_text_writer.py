from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import configure_hf_offline_mode, configure_project_hf_cache, pick_device


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate product descriptions with a text-only writer model.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--jsonl", help="Prepared writer JSONL with input_text")
    group.add_argument("--input_text", help="Single prepared writer input")
    parser.add_argument("--model", default="outputs/text_writer_flan_t5_base_lora_neural_marketplace_v3")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--out", default="outputs/predictions/text_writer_preds.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_length", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    return parser.parse_args()


def load_writer_model(model_path: str, local_files_only: bool, device: str) -> tuple[Any, torch.nn.Module]:
    if local_files_only:
        configure_hf_offline_mode()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        from peft import PeftConfig, PeftModel

        peft_config = PeftConfig.from_pretrained(model_path, local_files_only=local_files_only)
        base_model_name = peft_config.base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, local_files_only=local_files_only)
        model = PeftModel.from_pretrained(model, model_path, local_files_only=local_files_only)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=local_files_only)

    model.to(device)
    model.eval()
    return tokenizer, model


def generate_batch(
    *,
    tokenizer: Any,
    model: torch.nn.Module,
    device: str,
    input_texts: list[str],
    max_input_length: int,
    max_new_tokens: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
) -> list[str]:
    inputs = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def main() -> None:
    configure_project_hf_cache()
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device, _ = pick_device()
    tokenizer, model = load_writer_model(args.model, args.local_files_only, device)

    if args.input_text:
        prediction = generate_batch(
            tokenizer=tokenizer,
            model=model,
            device=device,
            input_texts=[args.input_text],
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
        )[0]
        print(prediction)
        return

    in_path = Path(args.jsonl)
    if not in_path.exists():
        raise FileNotFoundError(f"JSONL not found: {in_path}")

    rows_out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    n_seen = 0

    def flush() -> None:
        nonlocal pending
        if not pending:
            return
        predictions = generate_batch(
            tokenizer=tokenizer,
            model=model,
            device=device,
            input_texts=[str(row["input_text"]) for row in pending],
            max_input_length=args.max_input_length,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
        )
        for row, prediction in zip(pending, predictions):
            rows_out.append({**row, "pred_text": prediction, "error": None})
        pending = []

    for row in iter_jsonl(in_path):
        if args.limit and n_seen >= args.limit:
            break
        input_text = str(row.get("input_text", "")).strip()
        if not input_text:
            rows_out.append({**row, "pred_text": None, "error": "missing_input_text"})
            continue
        pending.append(row)
        n_seen += 1
        if len(pending) >= args.batch_size:
            flush()
            print(f"Predicted {n_seen} rows")
    flush()

    write_jsonl(Path(args.out), rows_out)
    print(f"Saved {len(rows_out)} rows to {args.out}")


if __name__ == "__main__":
    main()
