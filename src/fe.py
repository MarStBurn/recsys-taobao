"""Feature specs: primitives (columns in the built parquet) and derived ratios over them.

A spec is either a primitive name, or "a / b" for a ratio of two primitives (NaN when b == 0).
"""
import numpy as np
from config import ENTITIES, EVENTS_MAP, WINDOWS

EVS = list(EVENTS_MAP)


def cnt(ent, ev, w):
    return f"{ent}__{ev}__cnt__{w}"


def needs(spec: str) -> list[str]:
    return [p.strip() for p in spec.split("/")]


def compute(spec: str, cols: dict) -> np.ndarray:
    """cols maps primitive name -> float32 array."""
    parts = needs(spec)
    if len(parts) == 1:
        return cols[parts[0]]
    a, b = cols[parts[0]], cols[parts[1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(b > 0, a / b, np.nan).astype(np.float32)


# Hand-crafted baseline: what a team would typically write by hand
# (1 / 3 / 6 day windows, standard conversions, recency, a few share features).
def baseline_specs() -> list[str]:
    s = []
    for ent in ("u", "i", "ui"):
        s += [cnt(ent, ev, w) for ev in ("pv", "fav", "cart", "buy") for w in (24, 72, 144)]
    s += [cnt("uc", ev, w) for ev in ("pv", "fav", "cart", "buy") for w in (24, 144)]
    s += [cnt("c", ev, w) for ev in ("pv", "buy") for w in (24, 144)]
    s += ["u__any__nuniq__144", "u__any__ndays__144", "i__any__nuniq__144", "i__buy__nuniq__144"]
    s += [f"ui__{ev}__rec" for ev in EVS] + ["uc__any__rec", "uc__buy__rec"]
    s += [f"{cnt(e, 'buy', 144)} / {cnt(e, 'pv', 144)}" for e in ("u", "i", "c")]
    s += [f"{cnt('i', 'cart', 144)} / {cnt('i', 'pv', 144)}"]
    s += [f"{cnt('ui', 'pv', 144)} / {cnt('u', 'pv', 144)}",
          f"{cnt('uc', 'pv', 144)} / {cnt('u', 'pv', 144)}",
          f"{cnt('ui', 'pv', 144)} / {cnt('uc', 'pv', 144)}",
          f"{cnt('i', 'buy', 144)} / {cnt('c', 'buy', 144)}"]
    return s


def search_space(primitives: list[str]) -> list[str]:
    """Every primitive, plus three generated ratio families:
    conversion (event over event, same entity & window), share (child entity over parent,
    same event & window) and trend (short window over long window, same entity & event)."""
    space = list(primitives)
    conv = [("buy", "pv"), ("cart", "pv"), ("fav", "pv"), ("buy", "cart"), ("buy", "any")]
    for ent in ENTITIES:
        for w in WINDOWS:
            space += [f"{cnt(ent, a, w)} / {cnt(ent, b, w)}" for a, b in conv]
    for child, parent in (("ui", "u"), ("uc", "u"), ("ui", "uc"), ("i", "c")):
        for ev in EVS:
            space += [f"{cnt(child, ev, w)} / {cnt(parent, ev, w)}" for w in WINDOWS]
    for ent in ENTITIES:
        for ev in EVS:
            for w2 in (24, 72, 144):
                space += [f"{cnt(ent, ev, w1)} / {cnt(ent, ev, w2)}" for w1 in WINDOWS if w1 < w2]
    return space
