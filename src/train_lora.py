from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

from .config import DEFAULT_MODEL_NAME, pick_device
from .dataset import ProductCaptionDataset, load_image_rgb

class BlipCollator:
    def __init__(self, processor: BlipProcessor):
        self.processor = processor

    def __call__(self, batch):
        images = [load_image_rgb(s.image_path) for s in batch]
        texts = [s.text for s in batch]
        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


def collate_fn(processor: BlipProcessor, batch):
    images = [load_image_rgb(s.image_path) for s in batch]
    texts = [s.text for s in batch]

    inputs = processor(images=images, text=texts, return_tensors="pt", padding=True, truncation=True)
    # For BLIP conditional generation, labels are input_ids
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", default="data/train.jsonl")
    parser.add_argument("--val_jsonl", default="data/val.jsonl")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--out_dir", default="outputs/lora_adapter")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--max_train_steps", type=int, default=2000)  # для первого прогона ограничим
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device, amp_dtype = pick_device()
    if device != "cuda":
        raise RuntimeError("Training should be run on CUDA (your RTX 3070 Ti).")

    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model, use_safetensors=True)
    model.to(device)

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

    train_ds = ProductCaptionDataset(args.train_jsonl)
    collator = BlipCollator(processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collator,
        pin_memory=True,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # total steps
    steps_per_epoch = max(1, len(train_loader) // args.grad_accum)
    total_steps = min(args.max_train_steps, steps_per_epoch * args.epochs)

    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=min(args.warmup_steps, total_steps // 5),
        num_training_steps=total_steps,
    )

    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    step = 0
    optim.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        for batch in train_loader:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(**batch)
                loss = out.loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                sched.step()

            step += 1
            if step % 50 == 0:
                print(f"step={step}/{total_steps} loss={loss.item():.4f}")

            if step >= total_steps:
                break
        if step >= total_steps:
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print("Saved LoRA adapter to:", out_dir)


if __name__ == "__main__":
    main()