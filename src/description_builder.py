from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


MISSING_VALUES = {"", "nan", "none", "null", "na", "n/a"}
SPACE_RE = re.compile(r"\s+")
TRAILING_NOISE_RE = re.compile(
    r"\b(?:person|people|amp|consp|con|contour|click|faux|big gift|gift|gifts|combo|combos|online shop|"
    r"my collection|my online shop|de|therm|interior|waistcoat|hi)\b.*$",
    re.IGNORECASE,
)
SYMBOL_NOISE_RE = re.compile(r"\s+(?:[_/]+|&)\s+.*$")
DANGLING_SUFFIXES = (" and", " the", " a", " an", " of", " with", " s")


def clean_text(value: Any) -> str:
    text = SPACE_RE.sub(" ", str(value).replace("&", " and ")).strip()
    return "" if text.lower() in MISSING_VALUES else text


def get_field(row: Mapping[str, Any], *names: str) -> str:
    for name in names:
        value = clean_text(row.get(name, ""))
        if value:
            return value
    return ""


def normalize_caption(parts: Iterable[str]) -> str:
    seen = set()
    cleaned: list[str] = []
    for part in parts:
        value = clean_text(part).lower()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return SPACE_RE.sub(" ", " ".join(cleaned)).strip()


def clean_model_phrase(text: str) -> str:
    text = clean_text(text).lower()
    text = text.replace(" - ", "-").replace("&amp;", " and ").replace("&", " and ")
    text = text.replace("'", " ")
    text = re.sub(r"[_/]+", " ", text)
    if text in {"me", "meena", "meena s", "me and me"}:
        return ""
    if text.startswith("me "):
        text = text[3:].strip()
    text = TRAILING_NOISE_RE.sub("", text)
    text = SYMBOL_NOISE_RE.sub("", text)
    text = SPACE_RE.sub(" ", text).strip(" .,:;!-")
    changed = True
    while changed:
        changed = False
        for suffix in DANGLING_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)].strip(" .,:;!-")
                changed = True
    for prefix in ("a pair of ", "an pair of ", "pair of ", "a ", "an ", "the "):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
    if text in {"me", "meena", "meena s", "me and me", "and me"}:
        return ""
    return text


def build_visual_caption(row: Mapping[str, Any]) -> str:
    return normalize_caption(
        [
            get_field(row, "baseColour", "base_colour"),
            get_field(row, "articleType", "article_type"),
        ]
    )


def indefinite_article(phrase: str) -> str:
    phrase = phrase.strip()
    if not phrase:
        return "a"
    return "an" if phrase[0].lower() in {"a", "e", "i", "o", "u"} else "a"


SINGULAR_TERMS = {
    "bags": "bag",
    "belts": "belt",
    "blazers": "blazer",
    "caps": "cap",
    "dresses": "dress",
    "handbags": "handbag",
    "jackets": "jacket",
    "kurtas": "kurta",
    "sarees": "saree",
    "shirts": "shirt",
    "skirts": "skirt",
    "sweaters": "sweater",
    "sweatshirts": "sweatshirt",
    "tshirts": "t-shirt",
    "tops": "top",
    "vests": "vest",
    "wallets": "wallet",
    "watches": "watch",
}

PAIR_TERMS = {
    "boots",
    "briefs",
    "casual shoes",
    "capris",
    "earrings",
    "flip flops",
    "formal shoes",
    "heels",
    "jeans",
    "leggings",
    "lounge pants",
    "pants",
    "sandals",
    "shoes",
    "shorts",
    "socks",
    "sports shoes",
    "sunglasses",
    "track pants",
    "trousers",
}


def replace_suffix(text: str, suffix: str, replacement: str) -> str:
    if text == suffix:
        return replacement
    marker = f" {suffix}"
    if text.endswith(marker):
        return text[: -len(marker)] + f" {replacement}"
    return text


def noun_phrase_with_determiner(product_phrase: str) -> str:
    product_phrase = clean_model_phrase(product_phrase)
    if not product_phrase:
        return "a product"
    for term in sorted(PAIR_TERMS, key=len, reverse=True):
        if product_phrase == term or product_phrase.endswith(f" {term}"):
            return f"a pair of {product_phrase}"

    singular = product_phrase
    for plural, replacement in sorted(SINGULAR_TERMS.items(), key=lambda item: len(item[0]), reverse=True):
        updated = replace_suffix(singular, plural, replacement)
        if updated != singular:
            singular = updated
            break

    return f"{indefinite_article(singular)} {singular}"


def join_phrases(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def gender_phrase(gender: str) -> str:
    normalized = gender.strip().lower()
    if normalized in {"men", "man", "male"}:
        return "for men"
    if normalized in {"women", "woman", "female"}:
        return "for women"
    if normalized in {"boys", "boy"}:
        return "for boys"
    if normalized in {"girls", "girl"}:
        return "for girls"
    if normalized == "unisex":
        return "for all genders"
    return ""


def category_phrase(master_category: str, sub_category: str) -> str:
    master = master_category.lower()
    sub = sub_category.lower()
    if master and sub and master != sub:
        return f"the {sub} category within {master}"
    if sub:
        return f"the {sub} category"
    if master:
        return f"the {master} category"
    return ""


def usage_phrase(usage: str) -> str:
    normalized = usage.strip().lower()
    if not normalized:
        return ""
    if normalized in {"casual", "formal", "ethnic", "sports", "smart casual", "travel"}:
        return f"suited to {normalized} use"
    return f"tagged for {normalized} use"


def product_phrase_from_row(row: Mapping[str, Any], fallback_visual: str = "") -> str:
    visual = build_visual_caption(row) or clean_text(fallback_visual).lower()
    article_type = get_field(row, "articleType", "article_type").lower()
    sub_category = get_field(row, "subCategory", "sub_category").lower()
    master_category = get_field(row, "masterCategory", "master_category").lower()
    return visual or article_type or sub_category or master_category or "product"


def build_visual_description(
    row: Mapping[str, Any],
    *,
    fallback_visual: str = "",
    include_category: bool = False,
) -> str:
    product_phrase = product_phrase_from_row(row, fallback_visual=fallback_visual)
    master_category = get_field(row, "masterCategory", "master_category")
    sub_category = get_field(row, "subCategory", "sub_category")

    sentences = [f"{noun_phrase_with_determiner(product_phrase).capitalize()}."]
    category = category_phrase(master_category, sub_category) if include_category else ""
    if include_category and category:
        sentences.append(f"It belongs to {category}.")
    return " ".join(sentences)


def build_description_from_model_text(text: str) -> str:
    phrase = clean_model_phrase(text)
    if not phrase:
        return "A product."
    return f"{noun_phrase_with_determiner(phrase).capitalize()}."


def build_catalog_description(
    row: Mapping[str, Any],
    *,
    include_title: bool = False,
    fallback_visual: str = "",
) -> str:
    title = get_field(row, "productDisplayName", "product_display_name") if include_title else ""
    product_phrase = product_phrase_from_row(row, fallback_visual=fallback_visual)
    master_category = get_field(row, "masterCategory", "master_category")
    sub_category = get_field(row, "subCategory", "sub_category")
    gender = gender_phrase(get_field(row, "gender"))
    usage = usage_phrase(get_field(row, "usage"))

    noun_phrase = noun_phrase_with_determiner(product_phrase)
    first_sentence = f"{title} is {noun_phrase}" if title else noun_phrase.capitalize()
    if gender:
        first_sentence = f"{first_sentence} {gender}"
    first_sentence = f"{first_sentence}."

    details: list[str] = []
    category = category_phrase(master_category, sub_category)
    if category:
        details.append(f"listed under {category}")
    if usage:
        details.append(usage)

    sentences = [first_sentence]
    if details:
        sentences.append(f"It is {join_phrases(details)}.")
    return " ".join(sentences)


def build_product_description(
    row: Mapping[str, Any],
    *,
    include_title: bool = False,
    include_catalog_details: bool = False,
    include_category: bool = False,
    fallback_visual: str = "",
) -> str:
    if include_catalog_details or include_title:
        return build_catalog_description(
            row,
            include_title=include_title,
            fallback_visual=fallback_visual,
        )
    return build_visual_description(
        row,
        fallback_visual=fallback_visual,
        include_category=include_category,
    )
