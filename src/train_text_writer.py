from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from .config import configure_hf_offline_mode, configure_project_hf_cache, pick_device


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class TextWriterDataset(Dataset):
    def __init__(self, jsonl_path: str | Path):
        rows = load_jsonl(jsonl_path)
        self.rows = [
            {"input_text": str(row.get("input_text", "")).strip(), "target_text": str(row.get("target_text") or row.get("text") or "").strip()}
            for row in rows
        ]
        self.rows = [row for row in self.rows if row["input_text"] and row["target_text"]]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.rows[idx]


class TextWriterCollator:
    def __init__(self, tokenizer: Any, max_input_length: int, max_target_length: int):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __call__(self, batch: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            [row["input_text"] for row in batch],
            padding=True,
            truncation=True,
            max_length=self.max_input_length,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            text_target=[row["target_text"] for row in batch],
            padding=True,
            truncation=True,
            max_length=self.max_target_length,
            return_tensors="pt",
        )["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


def move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    amp_dtype: torch.dtype,
    precision: str,
) -> float:
    model.eval()
    losses: list[float] = []
    use_amp = precision in {"fp16", "bf16"} and device in {"cuda", "mps"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else amp_dtype
    for batch in loader:
        batch = move_batch(batch, device)
        if use_amp:
            with torch.autocast(device_type=device, dtype=autocast_dtype):
                out = model(**batch)
        else:
            out = model(**batch)
        losses.append(float(out.loss.detach().cpu()))
    model.train()
    return sum(losses) / max(1, len(losses))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a text-only product description writer.")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--val_jsonl", default="")
    parser.add_argument("--model", default="google/flan-t5-small")
    parser.add_argument("--out_dir", default="outputs/text_writer_flan_t5_small")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=1000, help="Optimizer steps; 0 means full epochs")
    parser.add_argument("--max_input_length", type=int, default=384)
    parser.add_argument("--max_target_length", type=int, default=160)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--precision",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
        help="Use fp32 by default because small seq2seq fine-tuning can become NaN in fp16.",
    )
    parser.add_argument("--use_lora", action="store_true", help="Train a PEFT LoRA adapter instead of full fine-tuning")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q,v",
        help="Comma-separated module names for LoRA. T5 attention projections usually use q,v.",
    )
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Trade speed for lower activation memory")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_project_hf_cache()
    if args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("batch_size and grad_accum must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    torch.manual_seed(args.seed)
    if args.local_files_only:
        configure_hf_offline_mode()
    device, amp_dtype = pick_device()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_cosine_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, local_files_only=args.local_files_only)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = [name.strip() for name in args.lora_target_modules.split(",") if name.strip()]
        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    model.to(device)

    train_ds = TextWriterDataset(args.train_jsonl)
    if len(train_ds) == 0:
        raise RuntimeError(f"No writer training rows found in {args.train_jsonl}")
    val_ds = TextWriterDataset(args.val_jsonl) if args.val_jsonl and Path(args.val_jsonl).exists() else None

    collator = TextWriterCollator(tokenizer, args.max_input_length, args.max_target_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=device == "cuda",
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=device == "cuda",
        )
        if val_ds is not None and len(val_ds) > 0
        else None
    )

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found")
    optim = torch.optim.AdamW(trainable_params, lr=args.lr)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    planned_steps = steps_per_epoch * args.epochs
    total_steps = planned_steps if args.max_train_steps == 0 else min(args.max_train_steps, planned_steps)
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=min(args.warmup_steps, total_steps // 5),
        num_training_steps=total_steps,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    model.train()
    use_amp = args.precision in {"fp16", "bf16"} and device in {"cuda", "mps"}
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else amp_dtype
    scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16" and device == "cuda")
    optim.zero_grad(set_to_none=True)

    optim_step = 0
    micro_step = 0
    pending_micro_steps = 0
    running_loss = 0.0

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)
            if use_amp:
                with torch.autocast(device_type=device, dtype=autocast_dtype):
                    out = model(**batch)
                    raw_loss = out.loss
                    loss = raw_loss / args.grad_accum
            else:
                out = model(**batch)
                raw_loss = out.loss
                loss = raw_loss / args.grad_accum

            if not torch.isfinite(raw_loss):
                raise RuntimeError(
                    f"Non-finite training loss at epoch={epoch + 1}, batch={batch_idx}. "
                    "Try --precision fp32, a lower learning rate, or inspect the training rows."
                )

            if use_amp and device == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(raw_loss.detach().cpu())
            micro_step += 1
            pending_micro_steps += 1

            is_accum_boundary = pending_micro_steps >= args.grad_accum
            is_last_batch = batch_idx == len(train_loader)
            if is_accum_boundary or is_last_batch:
                if args.max_grad_norm > 0:
                    if use_amp and device == "cuda":
                        scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if use_amp and device == "cuda":
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                sched.step()
                optim_step += 1

                if optim_step == 1 or (args.log_every and optim_step % args.log_every == 0):
                    log_row = {
                        "epoch": epoch + 1,
                        "optimizer_step": optim_step,
                        "total_steps": total_steps,
                        "micro_step": micro_step,
                        "train_loss": running_loss / max(1, pending_micro_steps),
                        "lr": sched.get_last_lr()[0],
                    }
                    print(
                        "step={optimizer_step}/{total_steps} "
                        "epoch={epoch} train_loss={train_loss:.4f} lr={lr:.2e}".format(**log_row)
                    )
                    append_jsonl(log_path, log_row)

                if val_loader is not None and args.eval_steps and optim_step % args.eval_steps == 0:
                    val_loss = evaluate_loss(model, val_loader, device, amp_dtype, args.precision)
                    log_row = {
                        "epoch": epoch + 1,
                        "optimizer_step": optim_step,
                        "total_steps": total_steps,
                        "val_loss": val_loss,
                    }
                    print(f"validation step={optim_step}/{total_steps} val_loss={val_loss:.4f}")
                    append_jsonl(log_path, log_row)

                running_loss = 0.0
                pending_micro_steps = 0

            if optim_step >= total_steps:
                break

        if val_loader is not None:
            val_loss = evaluate_loss(model, val_loader, device, amp_dtype, args.precision)
            log_row = {
                "epoch": epoch + 1,
                "optimizer_step": optim_step,
                "total_steps": total_steps,
                "val_loss": val_loss,
            }
            print(f"validation epoch={epoch + 1} val_loss={val_loss:.4f}")
            append_jsonl(log_path, log_row)

        if optim_step >= total_steps:
            break

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    save_json(
        out_dir / "training_args.json",
        {
            **vars(args),
            "device": device,
            "dtype": str(amp_dtype),
            "train_size": len(train_ds),
            "val_size": len(val_ds) if val_ds is not None else 0,
            "optimizer_steps_completed": optim_step,
        },
    )
    print("Saved text writer to:", out_dir)


if __name__ == "__main__":
    main()
