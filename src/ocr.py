from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any

from PIL import Image, ImageOps


LOW_VALUE_TERMS = {
    "a",
    "an",
    "and",
    "for",
    "last",
    "of",
    "the",
    "til",
    "to",
    "with",
}

OCR_TEXT_CORRECTIONS = {
    "nestla": "Nestle",
    "nestle": "Nestle",
    "nesquik": "Nesquik",
    "nol ingredient": "No.1 ingredient",
    "no1 ingredient": "No.1 ingredient",
    "no ingredient": "No.1 ingredient",
}

_EASYOCR_READERS: dict[tuple[tuple[str, ...], bool], Any] = {}


def normalize_ocr_text(text: str) -> str:
    text = " ".join(str(text).replace("\x0c", " ").split())
    return text.strip(" |,:;")


def _confidence(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _json_safe_box(box: Any) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point in box]


def should_use_easyocr_gpu(mode: str = "auto") -> bool:
    if mode == "cpu":
        return False
    if mode == "cuda":
        return True
    if mode != "auto":
        raise ValueError("easyocr_gpu must be one of: auto, cpu, cuda")
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _easyocr_language_key(languages: str) -> tuple[str, ...]:
    easyocr_languages = [lang.strip() for lang in languages.replace("+", ",").split(",") if lang.strip()]
    if easyocr_languages == ["eng"]:
        easyocr_languages = ["en"]
    return tuple(easyocr_languages or ["en"])


def _get_easyocr_reader(languages: str, easyocr_gpu: str) -> Any:
    import easyocr

    language_key = _easyocr_language_key(languages)
    use_gpu = should_use_easyocr_gpu(easyocr_gpu)
    reader_key = (language_key, use_gpu)
    if reader_key not in _EASYOCR_READERS:
        _EASYOCR_READERS[reader_key] = easyocr.Reader(list(language_key), gpu=use_gpu)
    return _EASYOCR_READERS[reader_key]


def clean_ocr_term(text: str) -> str:
    text = normalize_ocr_text(text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .,:;|/\\'\"")
    lowered = text.lower()
    if lowered in OCR_TEXT_CORRECTIONS:
        return OCR_TEXT_CORRECTIONS[lowered]
    return text


def is_low_value_ocr_term(text: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", text.lower())
    if not normalized:
        return True
    if normalized in LOW_VALUE_TERMS:
        return True
    return len(normalized) == 1 and not normalized.isdigit()


def dedupe_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for term in terms:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(term)
    return output


def enrich_ocr_result(result: dict[str, Any], *, high_confidence: float = 75.0) -> dict[str, Any]:
    items = result.get("items", [])
    high_confidence_terms: list[str] = []
    low_confidence_terms: list[str] = []
    discarded_terms: list[str] = []
    enriched_items: list[dict[str, Any]] = []

    for item in items:
        raw_text = normalize_ocr_text(item.get("text", ""))
        clean_text = clean_ocr_term(raw_text)
        confidence = _confidence(item.get("confidence"))
        use_for_description = bool(clean_text) and confidence >= high_confidence and not is_low_value_ocr_term(clean_text)
        reason = "used"
        if not clean_text:
            reason = "empty"
        elif confidence < high_confidence:
            reason = "low_confidence"
        elif is_low_value_ocr_term(clean_text):
            reason = "low_value_fragment"

        enriched_items.append(
            {
                **item,
                "text": raw_text,
                "clean_text": clean_text,
                "use_for_description": use_for_description,
                "filter_reason": reason,
            }
        )
        if use_for_description:
            high_confidence_terms.append(clean_text)
        elif reason == "low_confidence" and clean_text and not is_low_value_ocr_term(clean_text):
            low_confidence_terms.append(clean_text)
        elif clean_text:
            discarded_terms.append(clean_text)

    high_confidence_terms = dedupe_terms(high_confidence_terms)
    low_confidence_terms = dedupe_terms(low_confidence_terms)
    discarded_terms = dedupe_terms(discarded_terms)

    return {
        **result,
        "raw_text": normalize_ocr_text(result.get("text", "")),
        "clean_text": normalize_ocr_text(" ".join(high_confidence_terms)),
        "high_confidence_terms": high_confidence_terms,
        "low_confidence_terms": low_confidence_terms,
        "discarded_terms": discarded_terms,
        "items": enriched_items,
    }


def _easyocr_results_to_result(results: list[Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    lines: list[str] = []
    for box, raw_text, confidence in results:
        text = normalize_ocr_text(raw_text)
        if not text:
            continue
        lines.append(text)
        items.append({"text": text, "confidence": float(confidence) * 100.0, "box": _json_safe_box(box)})

    return {
        "backend": "easyocr",
        "text": normalize_ocr_text(" ".join(lines)),
        "lines": lines,
        "items": items,
        "warnings": [],
    }


def _prepare_image(image_path: str | Path) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    image = ImageOps.exif_transpose(image)
    grayscale = ImageOps.grayscale(image)
    width, height = grayscale.size
    shortest_side = min(width, height)
    if shortest_side and shortest_side < 900:
        scale = min(3.0, 900 / shortest_side)
        grayscale = grayscale.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)
    grayscale = ImageOps.autocontrast(grayscale)
    return grayscale


def _extract_tesseract(
    image_path: str | Path,
    *,
    min_confidence: float,
    languages: str,
    tesseract_cmd: str,
) -> dict[str, Any]:
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        return {
            "backend": "tesseract",
            "text": "",
            "lines": [],
            "items": [],
            "warnings": ["pytesseract_not_installed"],
        }

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        image = _prepare_image(image_path)
        data = pytesseract.image_to_data(image, lang=languages, output_type=Output.DICT)
    except Exception as exc:  # Tesseract can fail when the native binary or language data is missing.
        return {
            "backend": "tesseract",
            "text": "",
            "lines": [],
            "items": [],
            "warnings": [f"tesseract_failed:{type(exc).__name__}:{exc}"],
        }

    lines: dict[tuple[int, int, int, int], list[tuple[int, str]]] = defaultdict(list)
    items: list[dict[str, Any]] = []
    for idx, raw_text in enumerate(data.get("text", [])):
        text = normalize_ocr_text(raw_text)
        confidence = _confidence(data.get("conf", [])[idx])
        if not text or confidence < min_confidence:
            continue

        key = (
            int(data.get("page_num", [0])[idx]),
            int(data.get("block_num", [0])[idx]),
            int(data.get("par_num", [0])[idx]),
            int(data.get("line_num", [0])[idx]),
        )
        word_num = int(data.get("word_num", [idx])[idx])
        lines[key].append((word_num, text))
        items.append(
            {
                "text": text,
                "confidence": confidence,
                "box": [
                    int(data.get("left", [0])[idx]),
                    int(data.get("top", [0])[idx]),
                    int(data.get("width", [0])[idx]),
                    int(data.get("height", [0])[idx]),
                ],
            }
        )

    line_texts = []
    for key in sorted(lines):
        words = [word for _, word in sorted(lines[key])]
        line = normalize_ocr_text(" ".join(words))
        if line:
            line_texts.append(line)

    return {
        "backend": "tesseract",
        "text": normalize_ocr_text(" ".join(line_texts)),
        "lines": line_texts,
        "items": items,
        "warnings": [],
    }


def _extract_easyocr(
    image_path: str | Path,
    *,
    min_confidence: float,
    languages: str,
    easyocr_gpu: str,
) -> dict[str, Any]:
    try:
        import easyocr
    except ImportError:
        return {
            "backend": "easyocr",
            "text": "",
            "lines": [],
            "items": [],
            "warnings": ["easyocr_not_installed"],
        }

    try:
        reader = _get_easyocr_reader(languages, easyocr_gpu)
        results = reader.readtext(str(image_path), detail=1, paragraph=False)
    except Exception as exc:
        return {
            "backend": "easyocr",
            "text": "",
            "lines": [],
            "items": [],
            "warnings": [f"easyocr_failed:{type(exc).__name__}:{exc}"],
        }

    threshold = min_confidence / 100.0
    return _easyocr_results_to_result([result for result in results if float(result[2]) >= threshold])


def extract_ocr_text(
    image_path: str | Path,
    *,
    backend: str = "auto",
    min_confidence: float = 50.0,
    high_confidence: float = 75.0,
    languages: str = "eng",
    tesseract_cmd: str = "",
    easyocr_gpu: str = "auto",
) -> dict[str, Any]:
    """Extract visible text from an image with optional local OCR backends."""
    image_path = Path(image_path)
    if backend == "none":
        return enrich_ocr_result(
            {"backend": "none", "text": "", "lines": [], "items": [], "warnings": ["ocr_disabled"]},
            high_confidence=high_confidence,
        )

    if backend not in {"auto", "tesseract", "easyocr"}:
        raise ValueError("backend must be one of: auto, tesseract, easyocr, none")

    if backend in {"auto", "tesseract"}:
        result = _extract_tesseract(
            image_path,
            min_confidence=min_confidence,
            languages=languages,
            tesseract_cmd=tesseract_cmd,
        )
        if backend == "tesseract" or result.get("text"):
            return enrich_ocr_result(result, high_confidence=high_confidence)

    if backend in {"auto", "easyocr"}:
        easyocr_result = _extract_easyocr(
            image_path,
            min_confidence=min_confidence,
            languages=languages,
            easyocr_gpu=easyocr_gpu,
        )
        if backend == "easyocr" or easyocr_result.get("text"):
            return enrich_ocr_result(easyocr_result, high_confidence=high_confidence)
        if backend == "auto":
            warnings = list(result.get("warnings", [])) + list(easyocr_result.get("warnings", []))
            return enrich_ocr_result(
                {**easyocr_result, "backend": "auto", "warnings": warnings},
                high_confidence=high_confidence,
            )

    return enrich_ocr_result(result, high_confidence=high_confidence)


def extract_ocr_text_batch(
    image_paths: list[str | Path],
    *,
    backend: str = "auto",
    min_confidence: float = 50.0,
    high_confidence: float = 75.0,
    languages: str = "eng",
    tesseract_cmd: str = "",
    easyocr_gpu: str = "auto",
    easyocr_batch_size: int = 4,
    easyocr_resize: int = 1024,
) -> list[dict[str, Any]]:
    """Extract OCR for multiple images, using EasyOCR batched inference when possible."""
    if not image_paths:
        return []
    if backend == "none":
        return [
            extract_ocr_text(path, backend="none", high_confidence=high_confidence)
            for path in image_paths
        ]

    if backend == "easyocr":
        try:
            reader = _get_easyocr_reader(languages, easyocr_gpu)
            threshold = min_confidence / 100.0
            results: list[dict[str, Any]] = []
            for start in range(0, len(image_paths), max(1, easyocr_batch_size)):
                batch_paths = [str(path) for path in image_paths[start : start + max(1, easyocr_batch_size)]]
                resize_kwargs = {}
                if easyocr_resize > 0:
                    resize_kwargs = {"n_width": easyocr_resize, "n_height": easyocr_resize}
                batch_results = reader.readtext_batched(
                    batch_paths,
                    detail=1,
                    paragraph=False,
                    batch_size=max(1, easyocr_batch_size),
                    **resize_kwargs,
                )
                for image_results in batch_results:
                    filtered = [result for result in image_results if float(result[2]) >= threshold]
                    results.append(
                        enrich_ocr_result(
                            _easyocr_results_to_result(filtered),
                            high_confidence=high_confidence,
                        )
                    )
            return results
        except Exception as exc:
            warning = f"easyocr_batch_failed:{type(exc).__name__}:{exc}"
            results = [
                extract_ocr_text(
                    path,
                    backend="easyocr",
                    min_confidence=min_confidence,
                    high_confidence=high_confidence,
                    languages=languages,
                    tesseract_cmd=tesseract_cmd,
                    easyocr_gpu=easyocr_gpu,
                )
                for path in image_paths
            ]
            return [{**result, "warnings": [warning] + list(result.get("warnings", []))} for result in results]

    return [
        extract_ocr_text(
            path,
            backend=backend,
            min_confidence=min_confidence,
            high_confidence=high_confidence,
            languages=languages,
            tesseract_cmd=tesseract_cmd,
            easyocr_gpu=easyocr_gpu,
        )
        for path in image_paths
    ]
