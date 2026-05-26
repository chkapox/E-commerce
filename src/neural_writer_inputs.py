from __future__ import annotations

import re
import unicodedata
from typing import Any, Mapping

from .brand_lexicon import infer_brand_from_texts


MISSING_VALUES = {"", "nan", "none", "null", "na", "n/a"}


def clean_value(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return "" if text.lower() in MISSING_VALUES else text


def clean_brand(value: Any) -> str:
    text = clean_value(value)
    text = re.sub(r"^brand\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    return text


def clean_target_text(value: Any, max_chars: int = 0) -> str:
    text = clean_value(value)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.!?])(?=[A-Z0-9])", r"\1 ", text)
    if max_chars and len(text) > max_chars:
        cut = text[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
        text = cut if cut else text[:max_chars].rstrip(" ,;:")
        if text and text[-1] not in ".!?":
            text += "."
    return text


def parse_prompt_fields(prompt_text: Any) -> dict[str, str]:
    text = clean_value(prompt_text)
    labels = ["Product title", "Brand", "Category", "Color", "Material", "Size", "Style"]
    fields: dict[str, str] = {}
    for idx, label in enumerate(labels):
        marker = f"{label}:"
        start = text.find(marker)
        if start < 0:
            continue
        start += len(marker)
        following = [text.find(f"{next_label}:", start) for next_label in labels[idx + 1 :]]
        following.append(text.find("Marketplace description:", start))
        ends = [pos for pos in following if pos >= 0]
        end = min(ends) if ends else len(text)
        value = clean_value(text[start:end].strip(" ."))
        if value:
            fields[label.lower().replace(" ", "_")] = value
    return fields


def first_value(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = clean_value(row.get(name, ""))
        if value:
            return value
    return ""


def prompt_or_row_value(
    row: Mapping[str, Any],
    prompt_fields: Mapping[str, str],
    prompt_key: str,
    *row_names: str,
) -> str:
    return first_value(row, *row_names) or clean_value(prompt_fields.get(prompt_key, ""))


def leaf_category(category: str) -> str:
    parts = [part.strip() for part in clean_value(category).split(">") if part.strip()]
    return parts[-1] if parts else clean_value(category)


def normalize_visible_caption(value: Any) -> str:
    text = clean_value(value)
    text = re.sub(r"\b([a-zA-Z][\w'-]*)(?:\s+\1\b)+", r"\1", text, flags=re.IGNORECASE)
    return text.strip(" .,:;")


def facts_from_row(
    row: Mapping[str, Any],
    *,
    visible_caption: str = "",
    raw_visible_caption: str = "",
    ocr_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    prompt_fields = parse_prompt_fields(row.get("prompt_text"))
    ocr_result = ocr_result or {}
    ocr_terms = ocr_result.get("high_confidence_terms")
    if not isinstance(ocr_terms, list):
        row_terms = row.get("ocr_terms")
        ocr_terms = row_terms if isinstance(row_terms, list) else []
    ocr_terms = [clean_value(term) for term in ocr_terms if clean_value(term)]
    trusted_ocr_text = clean_value(ocr_result.get("clean_text") or row.get("ocr_text", ""))
    raw_ocr_text = clean_value(
        ocr_result.get("text")
        or ocr_result.get("raw_text")
        or row.get("raw_ocr_text", "")
    )

    brand = clean_brand(prompt_or_row_value(row, prompt_fields, "brand", "brand", "brandName", "brand_name"))
    brand_source = "metadata" if brand else ""
    brand_matches: list[str] = []
    if not brand:
        inferred_brand, brand_matches = infer_brand_from_texts(
            [
                row.get("title", ""),
                row.get("name", ""),
                visible_caption,
                raw_visible_caption,
                trusted_ocr_text,
                *ocr_terms,
            ]
        )
        if inferred_brand:
            brand = inferred_brand
            brand_source = "brand_lexicon"

    return {
        "title": prompt_or_row_value(
            row,
            prompt_fields,
            "product_title",
            "title",
            "name",
            "productDisplayName",
            "product_display_name",
        ),
        "brand": brand,
        "brand_source": brand_source,
        "brand_matches": brand_matches,
        "category": prompt_or_row_value(row, prompt_fields, "category", "category", "breadcrumbs", "category_path"),
        "color": prompt_or_row_value(row, prompt_fields, "color", "color", "colour", "baseColour", "base_colour"),
        "material": prompt_or_row_value(row, prompt_fields, "material", "material"),
        "size": prompt_or_row_value(row, prompt_fields, "size", "size"),
        "style": prompt_or_row_value(row, prompt_fields, "style", "style"),
        "visible_caption": normalize_visible_caption(visible_caption or row.get("visible_caption", "")),
        "raw_visible_caption": clean_value(raw_visible_caption or row.get("raw_visible_caption", "")),
        "ocr_text": trusted_ocr_text,
        "raw_ocr_text": raw_ocr_text,
        "ocr_terms": ocr_terms,
    }


def _append_fact(lines: list[str], label: str, value: str) -> None:
    value = clean_value(value)
    if value:
        lines.append(f"- {label}: {value}")


def build_neural_writer_input(
    facts: Mapping[str, Any],
    *,
    prompt_style: str = "marketplace",
    max_ocr_terms: int = 8,
) -> str:
    if prompt_style == "complete":
        lines = [
            "Write one complete product listing description using the evidence below.",
            "Write natural sentences, not field labels or bullet points.",
            "Naturally include supplied facts such as the product type, brand, color, or material.",
            "Do not state exact attributes unless they are supported by the evidence.",
            "",
            "Evidence:",
        ]
    elif prompt_style == "strict":
        lines = [
            "Generate a fluent e-commerce product description from the facts below.",
            "Use only supported facts. Do not invent exact materials, sizes, brands, compatibility, ingredients, claims, or pack counts.",
            "Write 1-3 natural sentences.",
            "",
            "Facts:",
        ]
    elif prompt_style == "amazon_creative":
        lines = [
            "Generate an Amazon-style product description from the available product facts.",
            "Use the title, image caption, OCR text, and metadata as evidence. Keep exact product names, brands, sizes, colors, and materials unchanged.",
            "Write concise, useful marketplace copy.",
            "",
            "Facts:",
        ]
    else:
        lines = [
            "Generate a valid and informative marketplace product description from the facts below.",
            "Use the visual caption, OCR text, title, and metadata as input evidence.",
            "Do not invent exact facts that are not supported by the input.",
            "Write 1-3 fluent sentences for a product listing.",
            "",
            "Facts:",
        ]

    _append_fact(lines, "Product title", clean_value(facts.get("title", "")))
    _append_fact(lines, "Visible image caption", clean_value(facts.get("visible_caption", "")))

    ocr_terms = facts.get("ocr_terms")
    if isinstance(ocr_terms, list) and ocr_terms:
        _append_fact(lines, "Visible OCR text", ", ".join(clean_value(term) for term in ocr_terms[:max_ocr_terms]))
    else:
        _append_fact(lines, "Visible OCR text", clean_value(facts.get("ocr_text", "")))

    _append_fact(lines, "Brand", clean_value(facts.get("brand", "")))
    category = clean_value(facts.get("category", ""))
    _append_fact(lines, "Category", leaf_category(category) if category else "")
    _append_fact(lines, "Color", clean_value(facts.get("color", "")))
    _append_fact(lines, "Material", clean_value(facts.get("material", "")))
    _append_fact(lines, "Size or pack count", clean_value(facts.get("size", "")))
    _append_fact(lines, "Style", clean_value(facts.get("style", "")))

    lines.append("")
    lines.append("Description:")
    return "\n".join(lines)


def normalize_for_match(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", clean_value(value).lower()))


def value_is_present(text: str, value: str) -> bool:
    value_norm = normalize_for_match(value)
    if not value_norm:
        return True
    return value_norm in normalize_for_match(text)


def generated_description_warnings(description: str, facts: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not clean_value(description):
        return ["empty_description"]

    for key in ("brand", "color", "material", "size"):
        value = clean_value(facts.get(key, ""))
        if value and not value_is_present(description, value):
            warnings.append(f"{key}_not_preserved")

    title = clean_value(facts.get("title", ""))
    if title:
        title_tokens = [token for token in normalize_for_match(title).split() if len(token) >= 4]
        if title_tokens:
            overlap = sum(1 for token in title_tokens if token in normalize_for_match(description).split())
            if overlap / len(title_tokens) < 0.35:
                warnings.append("title_weakly_preserved")

    return warnings
