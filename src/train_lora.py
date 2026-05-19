from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from .config import DEFAULT_MODEL_NAME, pick_device
from .dataset import ProductCaptionDataset, load_image_rgb

class BlipCollator:
    def __init__(self, processor: BlipProcessor, max_length: int = 48):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch):
        images = [load_image_rgb(s.image_path) for s in batch]
        prompts = [s.prompt_text.strip() for s in batch]
        texts = [f"{prompt} {s.text}".strip() if prompt else s.text for prompt, s in zip(prompts, batch)]
        inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = inputs["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        for row_idx, prompt in enumerate(prompts):
            if not prompt:
                continue
            prompt_ids = self.processor.tokenizer(
                prompt,
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
            labels[row_idx, : min(len(prompt_ids), labels.shape[1])] = -100
        inputs["labels"] = labels
        return inputs


def move_batch(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    use_amp = device == "cuda" and amp_dtype == torch.float16

    for batch in loader:
        batch = move_batch(batch, device)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**batch)
        else:
            out = model(**batch)
        total_loss += float(out.loss.detach().cpu())
        total_batches += 1

    model.train()
    return total_loss / max(1, total_batches)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", default="data/train.jsonl")
    parser.add_argument("--val_jsonl", default="data/val.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--out_dir", default="outputs/lora_adapter_description")
    parser.add_argument("--base_adapter", default=None, help="Optional existing LoRA adapter to continue fine-tuning")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=2000, help="Optimizer steps; 0 means full epochs")
    parser.add_argument("--max_length", type=int, default=96)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--validate_images", action="store_true", help="Skip unreadable images before training")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--eval_steps", type=int, default=0, help="Validate every N optimizer steps; 0 = epoch/end only")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.batch_size <= 0 or args.grad_accum <= 0:
        raise ValueError("batch_size and grad_accum must be positive")
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    torch.manual_seed(args.seed)

    device, amp_dtype = pick_device()
    if device != "cuda":
        raise RuntimeError("Training should be run on CUDA (your RTX 3070 Ti).")

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import BlipForConditionalGeneration, BlipProcessor, get_cosine_schedule_with_warmup

    processor = BlipProcessor.from_pretrained(args.model)
    try:
        model = BlipForConditionalGeneration.from_pretrained(args.model, use_safetensors=True)
    except OSError:
        model = BlipForConditionalGeneration.from_pretrained(args.model, use_safetensors=False)
    model.to(device)

    if args.base_adapter:
        adapter_path = Path(args.base_adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(f"base_adapter not found: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    else:
        # LoRA: target text decoder attention projections
        lora_cfg = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            target_modules=["query", "value"],
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = ProductCaptionDataset(args.train_jsonl, validate_images=args.validate_images)
    if len(train_ds) == 0:
        raise RuntimeError(f"No training samples found in {args.train_jsonl}")

    val_ds = ProductCaptionDataset(args.val_jsonl, validate_images=args.validate_images) if Path(args.val_jsonl).exists() else None
    collator = BlipCollator(processor, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )
    else:
        print(f"Validation skipped: {args.val_jsonl} not found or empty")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum))
    planned_steps = steps_per_epoch * args.epochs
    total_steps = planned_steps if args.max_train_steps == 0 else min(args.max_train_steps, planned_steps)
    if total_steps <= 0:
        raise RuntimeError("No optimizer steps planned. Check dataset size and training arguments.")

    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=min(args.warmup_steps, total_steps // 5),
        num_training_steps=total_steps,
    )

    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    micro_step = 0
    optim_step = 0
    pending_micro_steps = 0
    running_loss = 0.0
    optim.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(train_loader, start=1):
            batch = move_batch(batch, device)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**batch)
                raw_loss = out.loss
                loss = raw_loss / args.grad_accum

            scaler.scale(loss).backward()
            running_loss += float(raw_loss.detach().cpu())
            micro_step += 1
            pending_micro_steps += 1

            is_accum_boundary = pending_micro_steps >= args.grad_accum
            is_last_batch = batch_idx == len(train_loader)
            if is_accum_boundary or is_last_batch:
                if args.max_grad_norm > 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()
                optim_step += 1

                should_log = optim_step == 1 or (args.log_every > 0 and optim_step % args.log_every == 0)
                if should_log:
                    avg_loss = running_loss / max(1, pending_micro_steps)
                    log_row = {
                        "epoch": epoch + 1,
                        "optimizer_step": optim_step,
                        "total_steps": total_steps,
                        "micro_step": micro_step,
                        "train_loss": avg_loss,
                        "lr": sched.get_last_lr()[0],
                    }
                    print(
                        "step={optimizer_step}/{total_steps} "
                        "epoch={epoch} train_loss={train_loss:.4f} lr={lr:.2e}".format(**log_row)
                    )
                    append_jsonl(log_path, log_row)

                if val_loader is not None and args.eval_steps and optim_step % args.eval_steps == 0:
                    val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
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
            val_loss = evaluate_loss(model, val_loader, device, amp_dtype)
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
    processor.save_pretrained(out_dir)
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
    print("Saved LoRA adapter to:", out_dir)


if __name__ == "__main__":
    main()
