# Automated Product Description Generation From Images

This project is a coursework prototype for generating short, grounded e-commerce product descriptions from product images. The current implementation uses BLIP as the captioning backbone and supports parameter-efficient LoRA fine-tuning for domain adaptation on fashion product images.

## What Is Included

- Dataset preparation from the Kaggle Fashion Product Images format (`styles.csv` + `images/`).
- Baseline and LoRA caption generation.
- Batched prediction for validation/test JSONL files.
- Automatic evaluation with BLEU, ROUGE, lexical overlap, color recall, and product-type recall.
- Training configured for a CUDA machine such as an RTX 3070 Ti.

## Recommended Workflow

Use the MacBook for code edits and quick syntax checks. Use the desktop PC for dataset preparation, training, full prediction, and evaluation.

## Setup

On macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt
python test_setup.py
```

On the Windows/CUDA PC:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-win-cuda.txt
python test_setup.py
```

If PyTorch CUDA wheels are not selected automatically on Windows, install PyTorch from the official CUDA command for your driver version, then install the project requirements again.

## Prepare Data

Expected raw Kaggle layout:

```text
data/raw/styles.csv
data/raw/images/1163.jpg
data/raw/images/1164.jpg
...
```

Create train/validation/test splits:

```bash
python -m src.prepare_dataset \
  --styles_csv data/raw/styles.csv \
  --image_dir data/raw/images \
  --out_dir data \
  --target_style visual
```

`target_style=visual` builds grounded references such as `blue jeans` from metadata fields that should be visually inferable. Use `--target_style title` only if you explicitly want product-title references, which may include brand or collection details that the image cannot prove.

## Run Baseline Predictions

```bash
python -m src.predict \
  --jsonl data/val.jsonl \
  --out outputs/predictions/val_baseline.jsonl \
  --batch_size 8 \
  --prompt "a product photo of"
```

Evaluate:

```bash
python -m src.evaluate \
  --preds outputs/predictions/val_baseline.jsonl \
  --out_metrics outputs/metrics_baseline.json \
  --out_samples outputs/samples_baseline.md \
  --out_diagnostics outputs/diagnostics_baseline.md
```

## Train LoRA On The RTX PC

Good first run for an RTX 3070 Ti:

```bash
python -m src.train_lora \
  --train_jsonl data/train.jsonl \
  --val_jsonl data/val.jsonl \
  --out_dir outputs/lora_adapter \
  --batch_size 8 \
  --grad_accum 4 \
  --epochs 3 \
  --lr 2e-4 \
  --max_train_steps 2000 \
  --num_workers 4
```

Then generate LoRA predictions:

```bash
python -m src.predict \
  --jsonl data/val.jsonl \
  --adapter outputs/lora_adapter \
  --out outputs/predictions/val_lora.jsonl \
  --batch_size 8 \
  --prompt "a product photo of"
```

Evaluate the LoRA run:

```bash
python -m src.evaluate \
  --preds outputs/predictions/val_lora.jsonl \
  --out_metrics outputs/metrics_lora.json \
  --out_samples outputs/samples_lora.md \
  --out_diagnostics outputs/diagnostics_lora.md
```

## Notes For The Report

The overlap metrics are useful for comparing runs, but they do not fully measure whether a description is truthful with respect to the image. The additional color and product-type recall metrics are a lightweight automatic proxy for grounding. The diagnostics Markdown file is meant for manual inspection of typical failure cases, which is important for the coursework discussion.
