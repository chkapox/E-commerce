# Automated Product Description Generation From Images

This project is a coursework prototype for generating short, grounded e-commerce product descriptions from product images. The main report outcome is an **image-only** pipeline: a product image goes in, and the model produces concise product-page copy that names the item and visible attributes without inventing unsupported details.

The implementation uses BLIP as the captioning backbone and supports parameter-efficient LoRA fine-tuning for domain adaptation on fashion product images. A separate metadata-assisted mode is included as a practical catalog extension, but it should be reported separately from the image-only experiment.

## Outcome

The desired output is not a plain object label such as:

```text
black shoes
```

The desired output is a short product description:

```text
A pair of black casual shoes.
```

When catalog metadata is available, the optional metadata-assisted mode can produce richer catalog copy:

```text
ADIDAS Originals Men Valley-Fdt Black Casual Shoes is a pair of black casual shoes for men. It is listed under the shoes category within footwear and suited to casual use.
```

## Current Best Run

The strongest image-only result in this workspace uses:

- adapter: `outputs/lora_adapter_description_v2`
- predictions: `outputs/predictions/val_lora_description_v2_noprompt.jsonl`
- metrics: `outputs/metrics_lora_description_v2_noprompt_300.json`
- generation prompt: none

Summary on the 304-row validation sample:

```text
sacrebleu: 61.3706
rouge1: 0.8245
rouge2: 0.6497
rougeL: 0.8245
token_f1: 0.8242
color_recall: 0.7842
product_type_recall: 0.8721
avg_pred_words: 4.2368
label_like_rate: 0.0
generic_fallback_rate: 0.0
```

The promptless run is important: adding prompts such as `a product photo of` or `a concise ecommerce product description of` made this fine-tuned BLIP adapter noticeably worse.

## Modes

- `model`: image-only generation. This is the main report baseline and LoRA result.
- `metadata`: deterministic catalog-assisted generation from JSONL fields. This is useful for a real store, but it is not image-only.
- `hybrid`: saves the model caption as `model_pred_text` and writes a catalog-assisted final description to `pred_text`.

For the report, compare `model` runs separately from `metadata` or `hybrid` runs.

## What Is Included

- Dataset preparation from the Kaggle Fashion Product Images format (`styles.csv` + `images/`).
- Description-style training targets for image-only LoRA fine-tuning.
- Baseline and LoRA generation for validation/test JSONL files.
- Optional metadata-assisted description generation.
- Automatic evaluation with BLEU, ROUGE, lexical overlap, color recall, product-type recall, output length, generic-output rate, and label-like-output rate.
- Training configured for a CUDA machine such as an RTX 3070 Ti.

## Setup

On the Windows/CUDA PC:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-win-cuda.txt
python test_setup.py
```

If PyTorch CUDA wheels are not selected automatically, install a CUDA-enabled PyTorch wheel from the official PyTorch command for your driver version, then run the project requirements again.

## Prepare Data

Expected raw Kaggle layout:

```text
data/raw/styles.csv
data/raw/images/1163.jpg
data/raw/images/1164.jpg
...
```

Create image-only description targets:

```bash
python -m src.prepare_dataset \
  --styles_csv data/raw/styles.csv \
  --image_dir data/raw/images \
  --out_dir data
```

The default `target_style=description` builds conservative one-sentence references from visually grounded fields, for example color and product type:

```text
A grey t-shirt.
```

Other target styles:

- `--target_style visual`: short labels such as `blue jeans`; useful only as a grounding check.
- `--target_style title`: product titles from `productDisplayName`.
- `--include_category_in_description`: adds a second category sentence. This is disabled by default because it can make the model repeat category templates.
- `--include_catalog_details`: adds metadata such as audience and usage. Use for metadata-assisted experiments, not the image-only baseline.
- `--include_title_in_description`: adds `productDisplayName`. Use only when product titles will also be available at inference time.

## Baseline Image-Only Predictions

```bash
python -m src.predict \
  --jsonl data/val.jsonl \
  --out outputs/predictions/val_baseline_description.jsonl \
  --batch_size 8 \
  --max_new_tokens 40
```

Evaluate:

```bash
python -m src.evaluate \
  --preds outputs/predictions/val_baseline_description.jsonl \
  --out_metrics outputs/metrics_baseline_description.json \
  --out_samples outputs/samples_baseline_description.md \
  --out_diagnostics outputs/diagnostics_baseline_description.md
```

## Train Description LoRA

Good first run for an RTX 3070 Ti:

```bash
python -m src.train_lora \
  --train_jsonl data/train.jsonl \
  --val_jsonl data/val.jsonl \
  --out_dir outputs/lora_adapter_description_v2 \
  --batch_size 8 \
  --grad_accum 4 \
  --epochs 3 \
  --lr 2e-4 \
  --max_train_steps 2000 \
  --max_length 64 \
  --num_workers 4
```

Generate LoRA predictions:

```bash
python -m src.predict \
  --jsonl data/val.jsonl \
  --adapter outputs/lora_adapter_description_v2 \
  --out outputs/predictions/val_lora_description_v2_noprompt.jsonl \
  --limit 300 \
  --batch_size 8 \
  --max_new_tokens 40
```

The fine-tuned adapter is trained without a decoder prompt, so prediction also runs without a prompt by default. If you experiment with prompts, compare them carefully; BLIP can become noticeably worse when the prompt style does not match fine-tuning.

Evaluate the LoRA run:

```bash
python -m src.evaluate \
  --preds outputs/predictions/val_lora_description_v2_noprompt.jsonl \
  --out_metrics outputs/metrics_lora_description_v2_noprompt_300.json \
  --out_samples outputs/samples_lora_description_v2_noprompt_300.md \
  --out_diagnostics outputs/diagnostics_lora_description_v2_noprompt_300.md \
  --max_samples 80
```

## Metadata-Assisted Catalog Mode

Use this when JSONL rows contain catalog fields and you want practical product-page descriptions immediately:

```bash
python -m src.predict \
  --jsonl data/val.jsonl \
  --out outputs/predictions/val_metadata_descriptions.jsonl \
  --description_mode metadata \
  --include_title_in_description
```

This mode is useful for a real e-commerce catalog, but it is not the same experiment as image-only generation. In the report, present it as a practical extension.

## Marketplace Dataset Extension

Richer marketplace-style descriptions need richer supervised targets than the current image-only Kaggle metadata can provide. A future dataset should include product images plus licensed or otherwise permitted title, brand, category, bullet, and description fields. Avoid scraping marketplaces unless the site's terms and robots policy clearly permit the intended use; prefer official APIs, licensed public datasets, or academic datasets.

Recommended row shape:

```json
{
  "id": "source-product-id",
  "source": "dataset-or-api-name",
  "image_path": "data/marketplace/images/example.jpg",
  "image_url": "https://example.com/image.jpg",
  "title": "Product title",
  "brand": "Brand if available",
  "category": "Category path",
  "bullets": ["bullet 1", "bullet 2"],
  "description": "Original marketplace description",
  "target_text": "Cleaned training target"
}
```

For this extension, keep image-only generation separate from metadata-assisted writing. A practical architecture is a two-stage pipeline: first predict visible attributes from the image, then generate the final catalog paragraph from those visible attributes plus trusted catalog fields.

### Amazon Metadata Preparation

The UCSD Amazon metadata format usually includes both `asin` and `imUrl`. Bestseller-style datasets may instead include a product URL, a product name, an extracted description, and one or more image links. The preparation script supports both styles.

If a row contains direct `m.media-amazon.com` image URLs, including URLs embedded inside an extracted image block such as `colorImages`, the script will use those URLs directly. It also tries higher-resolution variants by removing Amazon size tokens such as `_SX522_` or `_SL1024_`, then keeps the row only if the downloaded image passes the minimum-size check.

Do not use this script as an Amazon product-page crawler. Prefer direct image URLs already present in the dataset, extracted fields you are allowed to use, or an official API route.

Prepare a small fashion-focused experiment:

```bash
python -m src.prepare_amazon_dataset \
  --metadata_path data/raw/amazon/metadata.json.gz \
  --out_dir data/amazon \
  --image_dir data/amazon/images \
  --download_images \
  --category_filter Clothing \
  --category_filter Shoes \
  --limit 1000
```

This writes `data/amazon/train.jsonl`, `data/amazon/val.jsonl`, `data/amazon/test.jsonl`, and `data/amazon/amazon_dataset_summary.json`. Each row keeps the local `image_path`, original `image_url`, `asin`, title, brand, category, and a cleaned `text`/`target_text` field for training.

Prepare a bestseller CSV with product-page URLs and embedded/direct image links:

```bash
python -m src.prepare_amazon_dataset \
  --metadata_path data/raw/amazon_bestsellers.csv \
  --out_dir data/amazon_bestsellers \
  --image_dir data/amazon_bestsellers/images \
  --download_images \
  --min_image_side 224 \
  --limit 1000
```

Common column names such as `Product URL`, `Product Name`, `Image URL`, `Image Block`, `Description`, `Brand`, `Category`, and `Price` are handled automatically.

For the local bestseller dataset currently used in this workspace:

```bash
python -m src.prepare_amazon_dataset \
  --metadata_path "Q:\E-commerce\data\github_bestsellers\ecommerce-product-dataset-main\data\amazon_com\best_sellers\amazon_com_best_sellers_2025_01_27.csv" \
  --out_dir data/amazon_bestsellers \
  --image_dir data/amazon_bestsellers/images \
  --download_images \
  --min_image_side 224 \
  --max_target_words 80
```

The script understands the dataset columns `imageUrls`, `url`, `name`, and `description`. It can also use optional fields such as `brandName`, `breadcrumbs`, `salePrice`, and `listedPrice` when present.

## Notes For The Report

BLEU and ROUGE are useful for comparing runs, but they do not fully measure whether a description is truthful with respect to the image. Use the diagnostics Markdown file for manual inspection of failure cases:

- hallucinated attributes;
- incorrect product category;
- missing color or product type;
- overly generic text such as `the product`;
- label-like output that is too short for product-page copy.

The strongest final story is:

1. Baseline BLIP often produces generic captions.
2. Visual-label LoRA improves product recognition but still writes labels.
3. Description-target LoRA shifts the output toward short grounded product descriptions.
4. Metadata-assisted generation is more useful for a real catalog, but it relies on structured catalog data rather than image-only inference.
