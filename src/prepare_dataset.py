from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from .description_builder import build_product_description, build_visual_caption

MISSING_VALUES = {"", "nan", "none", "null", "na", "n/a"}
SPACE_RE = re.compile(r"\s+")


def clean_text(value: Any) -> str:
    text = SPACE_RE.sub(" ", str(value).replace("&", " and ")).strip()
    return "" if text.lower() in MISSING_VALUES else text


def normalize_caption(parts: Iterable[str]) -> str:
    seen = set()
    cleaned: List[str] = []
    for part in parts:
        value = clean_text(part).lower()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return SPACE_RE.sub(" ", " ".join(cleaned)).strip()


def build_caption(
    row: Dict[str, Any],
    target_style: str,
    include_title_in_description: bool,
    include_catalog_details: bool,
    include_category_in_description: bool,
) -> str:
    if target_style == "title":
        title = clean_text(row.get("productDisplayName", ""))
        if title:
            return title.lower()

    if target_style == "description":
        return build_product_description(
            row,
            include_title=include_title_in_description,
            include_catalog_details=include_catalog_details,
            include_category=include_category_in_description,
        )

    visual_caption = build_visual_caption(row)
    if visual_caption:
        return visual_caption

    fallback = clean_text(row.get("productDisplayName", ""))
    return fallback.lower()


def image_path_for(row: Dict[str, Any], image_dir: Path) -> Path:
    product_id = str(row["id"]).split(".")[0]
    return image_dir / f"{product_id}.jpg"


def portable_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def load_rows(
    styles_csv: Path,
    image_dir: Path,
    target_style: str,
    limit: int,
    include_title_in_description: bool,
    include_catalog_details: bool,
    include_category_in_description: bool,
) -> List[Dict[str, Any]]:
    frame = pd.read_csv(styles_csv, on_bad_lines="skip")
    frame = frame.fillna("")

    rows: List[Dict[str, Any]] = []
    skipped_missing_image = 0
    skipped_empty_text = 0

    for raw in frame.to_dict(orient="records"):
        image_path = image_path_for(raw, image_dir)
        if not image_path.exists():
            skipped_missing_image += 1
            continue

        caption = build_caption(
            raw,
            target_style,
            include_title_in_description,
            include_catalog_details,
            include_category_in_description,
        )
        if not caption:
            skipped_empty_text += 1
            continue

        rows.append(
            {
                "id": str(raw.get("id", "")).split(".")[0],
                "image_path": portable_path(image_path, Path.cwd()),
                "text": caption,
                "gender": clean_text(raw.get("gender", "")),
                "master_category": clean_text(raw.get("masterCategory", "")),
                "sub_category": clean_text(raw.get("subCategory", "")),
                "article_type": clean_text(raw.get("articleType", "")),
                "base_colour": clean_text(raw.get("baseColour", "")),
                "usage": clean_text(raw.get("usage", "")),
                "product_display_name": clean_text(raw.get("productDisplayName", "")),
            }
        )
        if limit and len(rows) >= limit:
            break

    print(f"Loaded {len(rows)} usable rows")
    print(f"Skipped missing images: {skipped_missing_image}")
    print(f"Skipped empty captions: {skipped_empty_text}")
    return rows


def split_rows(
    rows: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    stratify_key: str,
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row.get(stratify_key, "")) if stratify_key else "_all"
        groups[key or "_missing"].append(row)

    for group_rows in groups.values():
        rng.shuffle(group_rows)
        n = len(group_rows)
        if n == 1:
            splits["train"].extend(group_rows)
            continue

        n_train = max(1, int(round(n * train_ratio)))
        n_val = int(round(n * val_ratio))
        if n_train + n_val >= n:
            n_train = max(1, n - 1)
            n_val = 0 if n == 2 else 1

        splits["train"].extend(group_rows[:n_train])
        splits["val"].extend(group_rows[n_train : n_train + n_val])
        splits["test"].extend(group_rows[n_train + n_val :])

    for split_name, split_rows_ in splits.items():
        rng.shuffle(split_rows_)
        for row in split_rows_:
            row["split"] = split_name

    return splits


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_summary(out_dir: Path, splits: Dict[str, List[Dict[str, Any]]], target_style: str) -> None:
    summary = {
        "target_style": target_style,
        "splits": {name: len(rows) for name, rows in splits.items()},
        "sample_targets": [row["text"] for row in splits["train"][:5]],
        "top_article_types": Counter(
            row.get("article_type", "") for rows in splits.values() for row in rows
        ).most_common(25),
        "top_colours": Counter(
            row.get("base_colour", "") for rows in splits.values() for row in rows
        ).most_common(25),
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["splits"], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--styles_csv", default="data/raw/styles.csv")
    parser.add_argument("--image_dir", default="data/raw/images")
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--target_style", choices=["visual", "title", "description"], default="description")
    parser.add_argument(
        "--include_title_in_description",
        action="store_true",
        help=(
            "Include productDisplayName in description targets. Use only for metadata-assisted generation, "
            "not the image-only report baseline."
        ),
    )
    parser.add_argument(
        "--include_catalog_details",
        action="store_true",
        help=(
            "Include non-visual catalog details such as audience and usage in description targets. "
            "Leave off for image-only training."
        ),
    )
    parser.add_argument(
        "--include_category_in_description",
        action="store_true",
        help=(
            "Add a second category sentence to image-only description targets. "
            "Disabled by default because it can make BLIP repeat category templates."
        ),
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Limit usable rows for quick experiments")
    parser.add_argument("--stratify_key", default="article_type")
    args = parser.parse_args()

    styles_csv = Path(args.styles_csv)
    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_dir)

    if not styles_csv.exists():
        raise FileNotFoundError(f"styles_csv not found: {styles_csv}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Expected 0 < train_ratio and train_ratio + val_ratio < 1")

    rows = load_rows(
        styles_csv,
        image_dir,
        args.target_style,
        args.limit,
        args.include_title_in_description,
        args.include_catalog_details,
        args.include_category_in_description,
    )
    if not rows:
        raise RuntimeError("No usable rows found. Check paths and raw dataset layout.")

    splits = split_rows(rows, args.train_ratio, args.val_ratio, args.seed, args.stratify_key)
    for split_name, split_data in splits.items():
        write_jsonl(out_dir / f"{split_name}.jsonl", split_data)
    save_summary(out_dir, splits, args.target_style)


if __name__ == "__main__":
    main()
