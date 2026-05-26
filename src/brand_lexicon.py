from __future__ import annotations

import re
from typing import Iterable


BRAND_ALIASES: dict[str, tuple[str, ...]] = {
    "Adidas": ("adidas", "adidas originals"),
    "Nike": ("nike",),
    "Puma": ("puma",),
    "Reebok": ("reebok",),
    "New Balance": ("new balance",),
    "Asics": ("asics",),
    "Under Armour": ("under armour", "under armor"),
    "Skechers": ("skechers",),
    "Converse": ("converse",),
    "Vans": ("vans",),
    "Fila": ("fila",),
    "Crocs": ("crocs",),
    "Timberland": ("timberland",),
    "The North Face": ("the north face", "north face"),
    "Columbia": ("columbia",),
    "Patagonia": ("patagonia",),
    "Levi's": ("levis", "levi's"),
    "Wrangler": ("wrangler",),
    "Calvin Klein": ("calvin klein",),
    "Tommy Hilfiger": ("tommy hilfiger",),
    "Hanes": ("hanes",),
    "Fruit of the Loom": ("fruit of the loom",),
    "Champion": ("champion",),
    "Carhartt": ("carhartt",),
    "Dickies": ("dickies",),
    "DeWalt": ("dewalt",),
    "Makita": ("makita",),
    "Bosch": ("bosch",),
    "Milwaukee": ("milwaukee",),
    "Ryobi": ("ryobi",),
    "Black+Decker": ("black decker", "black+decker", "black and decker"),
    "Craftsman": ("craftsman",),
    "Stanley": ("stanley",),
    "Klein Tools": ("klein tools",),
    "Permatex": ("permatex",),
    "3M": ("3m",),
    "Chemical Guys": ("chemical guys",),
    "Armor All": ("armor all",),
    "Meguiar's": ("meguiars", "meguiar's"),
    "Mobil 1": ("mobil 1",),
    "Castrol": ("castrol",),
    "Valvoline": ("valvoline",),
    "Apple": ("apple",),
    "Samsung": ("samsung",),
    "Sony": ("sony",),
    "LG": ("lg",),
    "HP": ("hp",),
    "Dell": ("dell",),
    "Lenovo": ("lenovo",),
    "Logitech": ("logitech",),
    "Anker": ("anker",),
    "Belkin": ("belkin",),
    "JBL": ("jbl",),
    "Bose": ("bose",),
    "Canon": ("canon",),
    "Nikon": ("nikon",),
    "KitchenAid": ("kitchenaid",),
    "Cuisinart": ("cuisinart",),
    "Ninja": ("ninja",),
    "Instant Pot": ("instant pot",),
    "Hamilton Beach": ("hamilton beach",),
    "OXO": ("oxo",),
    "Pyrex": ("pyrex",),
    "Rubbermaid": ("rubbermaid",),
    "Brita": ("brita",),
    "Purina": ("purina",),
    "Pedigree": ("pedigree",),
    "Iams": ("iams",),
    "Blue Buffalo": ("blue buffalo",),
    "Neutrogena": ("neutrogena",),
    "Dove": ("dove",),
    "Colgate": ("colgate",),
    "Crest": ("crest",),
    "Gillette": ("gillette",),
    "Maybelline": ("maybelline",),
    "L'Oreal": ("loreal", "l'oreal"),
    "Olay": ("olay",),
    "Nature Made": ("nature made",),
    "Nature's Bounty": ("natures bounty", "nature's bounty"),
    "Optimum Nutrition": ("optimum nutrition",),
    "Quest Nutrition": ("quest nutrition",),
    "Hammermill": ("hammermill",),
    "McCormick": ("mccormick",),
    "Fluidmaster": ("fluidmaster",),
    "Wilson": ("wilson",),
    "Gerber": ("gerber",),
    "Alesis": ("alesis",),
}


def text_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def alias_matches_text(alias: str, text: str) -> bool:
    alias_tokens = text_tokens(alias)
    if not alias_tokens:
        return False

    tokens = text_tokens(text)
    if len(alias_tokens) == 1:
        token = alias_tokens[0]
        if len(token) <= 3:
            return " ".join(tokens) == token
        return token in tokens

    haystack = f" {' '.join(tokens)} "
    needle = f" {' '.join(alias_tokens)} "
    return needle in haystack


def infer_brand_from_texts(texts: Iterable[str]) -> tuple[str, list[str]]:
    candidates = [str(text or "") for text in texts if str(text or "").strip()]
    matches: list[str] = []
    aliases = [
        (brand, alias)
        for brand, brand_aliases in BRAND_ALIASES.items()
        for alias in brand_aliases
    ]
    aliases.sort(key=lambda item: (len(text_tokens(item[1])), len(item[1])), reverse=True)

    for brand, alias in aliases:
        for text in candidates:
            if alias_matches_text(alias, text):
                if brand not in matches:
                    matches.append(brand)
                break

    return (matches[0] if matches else ""), matches
