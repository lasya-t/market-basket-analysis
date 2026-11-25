"""Synthetic mobile-store orders with a few planted buying patterns.

Each order is a set of items. Patterns are planted with known strength so mined rules can be checked:
  phone        -> case (0.65), screen_protector (0.55), protection_plan (0.40)
  case & screen_protector together are bought more when a phone is in the basket
  tablet       -> stylus (0.50), keyboard_cover (0.35)
  smartwatch   -> watch_band (0.45)
  earbuds and charger are popular on their own (high support, little lift)
"""
from __future__ import annotations

import numpy as np

ITEMS = ["phone", "case", "screen_protector", "protection_plan", "tablet", "stylus", "keyboard_cover",
         "smartwatch", "watch_band", "earbuds", "charger", "car_mount", "sim_kit", "gift_card"]


def make_orders(n: int = 20000, seed: int = 3) -> list[list[str]]:
    rng = np.random.default_rng(seed)
    orders = []
    for _ in range(n):
        b = set()
        if rng.random() < 0.30:
            b.add("phone")
            if rng.random() < 0.65:
                b.add("case")
            if rng.random() < 0.55:
                b.add("screen_protector")
            if rng.random() < 0.40:
                b.add("protection_plan")
            if rng.random() < 0.35:
                b.add("sim_kit")
        if rng.random() < 0.08:
            b.add("tablet")
            if rng.random() < 0.50:
                b.add("stylus")
            if rng.random() < 0.35:
                b.add("keyboard_cover")
        if rng.random() < 0.06:
            b.add("smartwatch")
            if rng.random() < 0.45:
                b.add("watch_band")
        # independent add-ons: popular, but unrelated to the rest
        for item, p in [("earbuds", 0.22), ("charger", 0.25), ("car_mount", 0.04), ("gift_card", 0.03),
                        ("case", 0.05), ("screen_protector", 0.04)]:
            if rng.random() < p:
                b.add(item)
        if b:
            orders.append(sorted(b))
    return orders
