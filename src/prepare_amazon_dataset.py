from __future__ import annotations

import argparse
import ast
import csv
import gzip
import html
import json
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, UnidentifiedImageError


MISSING_VALUES = {"", "nan", "none", "null", "na", "n/a", "unknown"}
SPACE_RE = re.compile(r"\s+")
HTML_TAG_RE = re.compile(r"<[^>]+>")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
MARKETPLACE_BOILERPLATE_RE = re.compile(
    r"\b(?:about this item|see more product details|show more|read more|from the manufacturer)\b",
    re.IGNORECASE,
)
AMAZON_IMAGE_URL_RE = re.compile(
    r"https?://(?:m\.media-amazon\.com|images-na\.ssl-images-amazon\.com)/images/I/[^\"'\\<>\s,}\]]+",
    re.IGNORECASE,
)
AMAZON_SIZE_TOKEN_RE = re.compile(r"\._[^/]*?(?=\.(?:jpg|jpeg|png|webp)(?:$|\?))", re.IGNORECASE)
ASIN_FROM_URL_RE = re.compile(r"/(?:dp|gp/product)/([A-Z0-9]{10})(?:[/?#]|$)", re.IGNORECASE)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = " ".join(clean_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        value = " ".join(clean_text(item) for item in value)
    if isinstance(value, str) and value.strip().startswith(("[", "{")):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = None
        if parsed is not None:
            return clean_text(parsed)
    text = html.unescape(str(value))
    text = HTML_TAG_RE.sub(" ", text)
    text = MARKETPLACE_BOILERPLATE_RE.sub(" ", text)
    text = text.replace("&", " and ")
    text = text.replace("\\n", " ").replace("\n", " ")
    text = SPACE_RE.sub(" ", text).strip(" \t\r\n-_:;,.")
    return "" if text.lower() in MISSING_VALUES else text


def parse_metadata_line(line: str) -> Dict[str, Any]:
    line = line.strip()
    if not line:
        return {}
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        row = ast.literal_eval(line)
    if not isinstance(row, dict):
        return {}
    return row


def normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def get_any(row: Dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and clean_text(row.get(name, "")):
            return row[name]
    normalized = {normalized_key(str(key)): value for key, value in row.items()}
    for name in names:
        value = normalized.get(normalized_key(name))
        if clean_text(value):
            return value
    return ""


def asin_from_url(value: Any) -> str:
    text = str(value or "")
    match = ASIN_FROM_URL_RE.search(text)
    return match.group(1).upper() if match else ""


def iter_metadata(path: Path) -> Iterable[Dict[str, Any]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    is_csv = path.name.lower().endswith(".csv") or path.name.lower().endswith(".csv.gz")
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        if is_csv:
            yield from csv.DictReader(f)
            return
        for line in f:
            try:
                row = parse_metadata_line(line)
            except (SyntaxError, ValueError):
                continue
            if row:
                yield row


def flatten_categories(value: Any) -> List[str]:
    categories: List[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            name = clean_text(item.get("name", ""))
            if name and name not in categories:
                categories.append(name)
            for key, child in item.items():
                if key != "name":
                    visit(child)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        if isinstance(item, str):
            raw = item.strip()
            if raw.startswith(("[", "{")):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    try:
                        parsed = ast.literal_eval(raw)
                    except (SyntaxError, ValueError):
                        parsed = None
                if parsed is not None:
                    visit(parsed)
                    return
        text = clean_text(item)
        if text and text not in categories:
            categories.append(text)

    visit(value)
    return categories


def first_sentence_block(text: str, max_sentences: int) -> str:
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(text) if part.strip()]
    if not sentences:
        return text
    return " ".join(sentences[:max_sentences])


def cap_words(text: str, max_words: int) -> str:
    if max_words <= 0:
        return text
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rstrip(" ,;:-")


def product_title(row: Dict[str, Any]) -> str:
    return clean_text(get_any(row, "title", "name", "product_name", "product title", "product name"))


def product_brand(row: Dict[str, Any]) -> str:
    return clean_text(get_any(row, "brand", "brandName", "brand_name", "brand name"))


def product_description_value(row: Dict[str, Any], prefer_features: bool = False) -> Any:
    feature_fields = (
        "feature",
        "features",
        "bullets",
        "bullet_points",
        "bullet points",
    )
    description_fields = (
        "description",
        "descriptions",
        "extracted_description",
        "extracted description",
        "product_description",
        "product description",
    )
    if prefer_features:
        value = get_any(row, *feature_fields)
        if clean_text(value):
            return value
    return get_any(
        row,
        *(description_fields + feature_fields),
    )


def compact_target(
    row: Dict[str, Any],
    max_sentences: int,
    max_words: int,
    target_mode: str,
    prefer_features: bool,
) -> str:
    title = product_title(row)
    brand = product_brand(row)
    descriptions = product_description_value(row, prefer_features=prefer_features)
    description = clean_text(descriptions)
    categories = flatten_categories(
        get_any(row, "categories", "category", "category_path", "category path", "breadcrumbs", "new_path", "nodeName")
    )

    parts: List[str] = []
    if title and target_mode in {"title_description", "title_only"}:
        parts.append(title)
    if description and target_mode in {"title_description", "description_only"} and description.lower() != title.lower():
        parts.append(description)

    if not parts:
        category_text = ", ".join(categories[-2:])
        if brand and category_text:
            parts.append(f"{brand} product in {category_text}")
        elif category_text:
            parts.append(f"Product in {category_text}")
        elif brand:
            parts.append(f"{brand} product")

    target = cap_words(first_sentence_block(" ".join(parts), max_sentences=max_sentences), max_words=max_words)
    if target and target[-1] not in ".!?":
        target = f"{target}."
    return target


def build_prompt_text(row: Dict[str, Any], categories: List[str]) -> str:
    title = product_title(row)
    brand = product_brand(row)
    category = " > ".join(categories[:3])
    color = clean_text(get_any(row, "color", "baseColour", "base_colour"))
    material = clean_text(get_any(row, "material", "fabric", "ingredients"))
    size = clean_text(get_any(row, "size", "dimensions", "product dimensions"))
    style = clean_text(get_any(row, "style"))
    parts: List[str] = []
    if title:
        parts.append(f"Product title: {title}.")
    if brand:
        parts.append(f"Brand: {brand}.")
    if category:
        parts.append(f"Category: {category}.")
    if color:
        parts.append(f"Color: {color}.")
    if material:
        parts.append(f"Material: {material}.")
    if size:
        parts.append(f"Size: {size}.")
    if style:
        parts.append(f"Style: {style}.")
    parts.append("Marketplace description:")
    return " ".join(parts)


def extract_amazon_image_urls(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        urls: List[str] = []
        for child in value.values():
            urls.extend(extract_amazon_image_urls(child))
        return urls
    if isinstance(value, (list, tuple)):
        urls = []
        for child in value:
            urls.extend(extract_amazon_image_urls(child))
        return urls

    text = html.unescape(str(value)).replace("\\/", "/")
    return [url.rstrip("\\") for url in AMAZON_IMAGE_URL_RE.findall(text)]


def high_res_image_candidates(urls: Sequence[str]) -> List[str]:
    candidates: List[str] = []

    def add(url: str) -> None:
        url = html.unescape(url).replace("\\/", "/").strip()
        if url and url not in candidates:
            candidates.append(url)

    for url in urls:
        unscaled = AMAZON_SIZE_TOKEN_RE.sub("", url)
        add(unscaled)
        if unscaled != url:
            for size in (1500, 1200, 1024, 800):
                add(AMAZON_SIZE_TOKEN_RE.sub(f"._SL{size}_", url))
        add(url)
    return candidates


def image_urls_from_row(row: Dict[str, Any]) -> List[str]:
    direct_fields = (
        "hiRes",
        "hires",
        "large",
        "image_url",
        "imageUrl",
        "image",
        "imgUrl",
        "img_url",
        "imUrl",
        "image link",
        "image links",
        "image url",
        "image urls",
        "picture",
        "picture url",
        "picture urls",
        "thumbnail",
        "thumb",
    )
    urls: List[str] = []
    for field in direct_fields:
        urls.extend(extract_amazon_image_urls(row.get(field)))
    for value in row.values():
        urls.extend(extract_amazon_image_urls(value))
    return high_res_image_candidates(urls)


def safe_filename(asin: str, image_url: str) -> str:
    suffix = Path(image_url.split("?", 1)[0]).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    return f"{asin}{suffix}"


def image_dimensions(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as img:
            return img.size
    except (OSError, UnidentifiedImageError):
        return (0, 0)


def image_is_valid(path: Path, min_image_side: int = 1) -> bool:
    width, height = image_dimensions(path)
    if width <= 0 or height <= 0:
        return False
    return min(width, height) >= min_image_side


def download_image(url: str, path: Path, timeout: int, sleep_seconds: float, min_image_side: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    request = Request(url, headers={"User-Agent": "coursework-research-dataset-prep/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response, tmp_path.open("wb") as f:
            shutil.copyfileobj(response, f)
        if not image_is_valid(tmp_path, min_image_side=min_image_side):
            tmp_path.unlink(missing_ok=True)
            return False
        tmp_path.replace(path)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        return True
    except (HTTPError, URLError, TimeoutError, OSError):
        tmp_path.unlink(missing_ok=True)
        return False


def download_first_valid_image(
    urls: Sequence[str],
    path: Path,
    timeout: int,
    sleep_seconds: float,
    min_image_side: int,
) -> str:
    for url in urls:
        if download_image(
            url,
            path,
            timeout=timeout,
            sleep_seconds=sleep_seconds,
            min_image_side=min_image_side,
        ):
            return url
    return ""


def download_row_image(
    image_urls: Sequence[str],
    image_path: Path,
    timeout: int,
    sleep_seconds: float,
    min_image_side: int,
) -> tuple[str, int, int]:
    downloaded_url = download_first_valid_image(
        image_urls,
        image_path,
        timeout=timeout,
        sleep_seconds=sleep_seconds,
        min_image_side=min_image_side,
    )
    if not downloaded_url:
        return ("", 0, 0)
    width, height = image_dimensions(image_path)
    return (downloaded_url, width, height)


def portable_path(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def category_matches(categories: List[str], filters: List[str]) -> bool:
    if not filters:
        return True
    haystack = " ".join(categories).lower()
    return any(term.lower() in haystack for term in filters)


def category_is_excluded(categories: List[str], excludes: List[str]) -> bool:
    if not excludes:
        return False
    haystack = " ".join(categories).lower()
    return any(term.lower() in haystack for term in excludes)


def build_rows(
    metadata_path: Path,
    image_dir: Path,
    root: Path,
    limit: int,
    download_images: bool,
    timeout: int,
    sleep_seconds: float,
    category_filters: List[str],
    category_excludes: List[str],
    max_target_sentences: int,
    max_target_words: int,
    target_mode: str,
    prefer_features: bool,
    min_image_side: int,
    progress_every: int,
    workers: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    counters: Counter[str] = Counter()
    started_at = time.time()

    def progress() -> None:
        elapsed = max(0.1, time.time() - started_at)
        rate = counters["seen"] / elapsed
        print(
            "progress "
            f"seen={counters['seen']} kept={len(rows)} "
            f"missing_image_url={counters['missing_image_url']} "
            f"download_failed={counters['download_failed']} "
            f"rate={rate:.1f} rows/s"
        )

    def row_payload(
        raw: Dict[str, Any],
        asin: str,
        image_path: Path,
        image_url: str,
        product_url: str,
        width: int,
        height: int,
        target: str,
        categories: List[str],
    ) -> Dict[str, Any]:
        return {
            "id": asin,
            "asin": asin,
            "image_path": portable_path(image_path, root),
            "image_url": image_url,
            "product_url": product_url,
            "image_width": width,
            "image_height": height,
            "text": target,
            "target_text": target,
            "prompt_text": build_prompt_text(raw, categories),
            "title": product_title(raw),
            "brand": product_brand(raw),
            "category": " > ".join(categories),
            "price": get_any(raw, "price", "salePrice", "listedPrice", "product_price", "product price"),
            "source": "amazon_bestsellers",
        }

    pending: dict[Future[tuple[str, int, int]], tuple[Dict[str, Any], str, Path, str, str, str, List[str]]] = {}

    def collect_done(done: Iterable[Future[tuple[str, int, int]]]) -> None:
        for future in done:
            raw, asin, image_path, fallback_url, product_url, target, categories = pending.pop(future)
            downloaded_url, width, height = future.result()
            if not downloaded_url:
                counters["download_failed"] += 1
                continue
            rows.append(row_payload(raw, asin, image_path, downloaded_url or fallback_url, product_url, width, height, target, categories))
            counters["kept"] += 1

    with ThreadPoolExecutor(max_workers=max(1, workers)) if download_images and workers > 1 else nullcontext() as executor:
        for raw in iter_metadata(metadata_path):
            counters["seen"] += 1
            product_url = clean_text(
                get_any(raw, "product_url", "product url", "url", "link", "product_link", "product link", "amazon_url", "amazon url")
            )
            asin = clean_text(get_any(raw, "asin", "product_id", "product id")) or asin_from_url(product_url)
            image_urls = image_urls_from_row(raw)
            image_url = image_urls[0] if image_urls else ""
            categories = flatten_categories(
                get_any(raw, "categories", "category", "category_path", "category path", "breadcrumbs", "new_path", "nodeName")
            )
            target = compact_target(
                raw,
                max_sentences=max_target_sentences,
                max_words=max_target_words,
                target_mode=target_mode,
                prefer_features=prefer_features,
            )

            if not asin:
                counters["missing_asin"] += 1
                continue
            if not image_url:
                counters["missing_image_url"] += 1
                continue
            if not target:
                counters["missing_target"] += 1
                continue
            if not category_matches(categories, category_filters):
                counters["filtered_category"] += 1
                continue
            if category_is_excluded(categories, category_excludes):
                counters["excluded_category"] += 1
                continue

            image_path = image_dir / safe_filename(asin, image_url)
            downloaded_url = image_url
            if image_path.exists() and not image_is_valid(image_path, min_image_side=min_image_side):
                counters["existing_image_too_small"] += 1
                image_path.unlink(missing_ok=True)

            if image_path.exists():
                width, height = image_dimensions(image_path)
                rows.append(row_payload(raw, asin, image_path, downloaded_url, product_url, width, height, target, categories))
                counters["kept"] += 1
            elif not download_images:
                counters["missing_local_image"] += 1
                continue
            elif executor is not None:
                future = executor.submit(
                    download_row_image,
                    image_urls,
                    image_path,
                    timeout,
                    sleep_seconds,
                    min_image_side,
                )
                pending[future] = (raw, asin, image_path, image_url, product_url, target, categories)
                if len(pending) >= workers * 4:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    collect_done(done)
            else:
                downloaded_url, width, height = download_row_image(
                    image_urls,
                    image_path,
                    timeout=timeout,
                    sleep_seconds=sleep_seconds,
                    min_image_side=min_image_side,
                )
                if not downloaded_url:
                    counters["download_failed"] += 1
                    continue
                rows.append(row_payload(raw, asin, image_path, downloaded_url, product_url, width, height, target, categories))
                counters["kept"] += 1

            if limit and len(rows) >= limit:
                break

            if progress_every and counters["seen"] % progress_every == 0:
                progress()

        while pending and not (limit and len(rows) >= limit):
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            collect_done(done)

    print("Dataset prep counters:", json.dumps(counters, ensure_ascii=False, indent=2))
    return rows


def split_rows(rows: List[Dict[str, Any]], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = row.get("category", "").split(" > ")[0] or "_missing"
        grouped[key].append(row)

    splits = {"train": [], "val": [], "test": []}
    for group_rows in grouped.values():
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


def save_summary(out_dir: Path, rows: List[Dict[str, Any]], splits: Dict[str, List[Dict[str, Any]]]) -> None:
    summary = {
        "source": "amazon_bestsellers",
        "rows": len(rows),
        "splits": {name: len(split_rows_) for name, split_rows_ in splits.items()},
        "top_categories": Counter(row.get("category", "").split(" > ")[0] for row in rows).most_common(25),
        "sample_targets": [row["text"] for row in rows[:5]],
    }
    (out_dir / "amazon_dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", required=True, help="Path to Amazon metadata .csv, .jsonl/.txt, or .gz file")
    parser.add_argument("--out_dir", default="data/amazon")
    parser.add_argument("--image_dir", default="data/amazon/images")
    parser.add_argument("--download_images", action="store_true", help="Download images from imUrl when missing locally")
    parser.add_argument("--limit", type=int, default=0, help="Limit kept rows for a quick experiment")
    parser.add_argument("--category_filter", action="append", default=[], help="Keep rows whose category contains this text")
    parser.add_argument("--category_exclude", action="append", default=[], help="Skip rows whose category contains this text")
    parser.add_argument("--max_target_sentences", type=int, default=3)
    parser.add_argument("--max_target_words", type=int, default=80)
    parser.add_argument(
        "--target_mode",
        choices=["title_description", "description_only", "title_only"],
        default="title_description",
        help="Which text fields become the training target",
    )
    parser.add_argument("--prefer_features", action="store_true", help="Prefer structured feature/bullet fields over long descriptions")
    parser.add_argument("--min_image_side", type=int, default=224, help="Skip downloaded images smaller than this side")
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--sleep_seconds", type=float, default=0.1, help="Delay between image downloads")
    parser.add_argument("--workers", type=int, default=1, help="Parallel image download workers")
    parser.add_argument("--progress_every", type=int, default=100, help="Print progress after this many scanned rows")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata_path = Path(args.metadata_path)
    out_dir = Path(args.out_dir)
    image_dir = Path(args.image_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata_path not found: {metadata_path}")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Expected 0 < train_ratio and train_ratio + val_ratio < 1")
    if args.max_target_sentences <= 0:
        raise ValueError("max_target_sentences must be positive")
    if args.max_target_words < 0:
        raise ValueError("max_target_words must be non-negative")
    if args.progress_every < 0:
        raise ValueError("progress_every must be non-negative")
    if args.workers <= 0:
        raise ValueError("workers must be positive")

    rows = build_rows(
        metadata_path=metadata_path,
        image_dir=image_dir,
        root=Path.cwd(),
        limit=args.limit,
        download_images=args.download_images,
        timeout=args.timeout,
        sleep_seconds=args.sleep_seconds,
        category_filters=args.category_filter,
        category_excludes=args.category_exclude,
        max_target_sentences=args.max_target_sentences,
        max_target_words=args.max_target_words,
        target_mode=args.target_mode,
        prefer_features=args.prefer_features,
        min_image_side=args.min_image_side,
        progress_every=args.progress_every,
        workers=args.workers,
    )
    if not rows:
        raise RuntimeError(
            "No usable rows found. Use --download_images to fetch imUrl files, "
            "or point --image_dir at an existing image cache."
        )

    splits = split_rows(rows, args.train_ratio, args.val_ratio, args.seed)
    for split_name, split_data in splits.items():
        write_jsonl(out_dir / f"{split_name}.jsonl", split_data)
    save_summary(out_dir, rows, splits)


if __name__ == "__main__":
    main()
