from __future__ import annotations

import re
from typing import Dict, Iterable, Set


TOKEN_RE = re.compile(r"[a-z0-9]+")

COLOR_ALIASES: Dict[str, tuple[str, ...]] = {
    "black": ("black",),
    "blue": ("blue", "navy", "teal", "denim"),
    "brown": ("brown", "tan", "beige", "camel", "khaki"),
    "green": ("green", "olive", "mint"),
    "grey": ("grey", "gray", "charcoal", "silver"),
    "orange": ("orange",),
    "pink": ("pink", "rose"),
    "purple": ("purple", "violet", "lavender"),
    "red": ("red", "maroon", "burgundy"),
    "white": ("white", "cream", "ivory"),
    "yellow": ("yellow", "gold", "mustard"),
}

PRODUCT_ALIASES: Dict[str, tuple[str, ...]] = {
    "backpack": ("backpack", "rucksack"),
    "bag": ("bag", "handbag", "purse", "tote"),
    "belt": ("belt",),
    "blouse": ("blouse",),
    "boots": ("boot", "boots"),
    "cap": ("cap", "hat"),
    "coat": ("coat", "overcoat"),
    "dress": ("dress", "gown"),
    "heels": ("heel", "heels", "pump", "pumps"),
    "hoodie": ("hoodie", "hooded sweatshirt"),
    "jacket": ("jacket", "blazer"),
    "jeans": ("jean", "jeans", "denim pants"),
    "kurta": ("kurta",),
    "pants": ("pant", "pants", "trouser", "trousers"),
    "sandals": ("sandal", "sandals"),
    "saree": ("saree", "sari"),
    "shirt": ("shirt",),
    "shoes": ("shoe", "shoes", "footwear"),
    "shorts": ("short", "shorts"),
    "skirt": ("skirt",),
    "sneakers": ("sneaker", "sneakers", "trainer", "trainers"),
    "sunglasses": ("sunglasses", "glasses"),
    "sweater": ("sweater", "jumper", "pullover"),
    "sweatshirt": ("sweatshirt",),
    "t-shirt": ("t shirt", "t-shirt", "tee", "tee shirt"),
    "top": ("top",),
    "wallet": ("wallet",),
    "watch": ("watch",),
}


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().replace("-", " ").split())


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def extract_alias_matches(text: str, aliases: Dict[str, Iterable[str]]) -> Set[str]:
    normalized = f" {normalize_text(text)} "
    found: Set[str] = set()
    for label, label_aliases in aliases.items():
        for alias in label_aliases:
            alias_norm = f" {normalize_text(alias)} "
            if alias_norm in normalized:
                found.add(label)
                break
    return found


def token_overlap(prediction: str, reference: str) -> dict[str, float]:
    pred_tokens = set(tokenize(prediction))
    ref_tokens = set(tokenize(reference))
    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = pred_tokens & ref_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}
