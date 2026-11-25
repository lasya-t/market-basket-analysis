"""Check mined rules on held-out orders and use them to recommend add-ons."""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from basket.mine import one_hot, prune_redundant, rules


def split(orders: list, test_frac: float = 0.3, seed: int = 0) -> tuple[list, list]:
    idx = np.random.default_rng(seed).permutation(len(orders))
    cut = int(len(orders) * (1 - test_frac))
    return [orders[i] for i in idx[:cut]], [orders[i] for i in idx[cut:]]


def holdout_check(train_rules: pd.DataFrame, test: list) -> pd.DataFrame:
    """Recompute each rule's confidence and lift on the test orders."""
    oh = one_hot(test)
    out = train_rules[["rule", "antecedents", "consequents", "confidence", "lift"]].copy()

    def stats(a, c):
        has_a = oh[list(a)].all(axis=1)
        has_c = oh[list(c)].all(axis=1)
        conf = (has_a & has_c).sum() / max(has_a.sum(), 1)
        return conf, conf / has_c.mean()

    out[["test_confidence", "test_lift"]] = [stats(a, c) for a, c in zip(out.antecedents, out.consequents)]
    return out.drop(columns=["antecedents", "consequents"])


class RuleRecommender:
    """Recommend items whose single-item rules fire on the current basket, ranked by confidence."""

    def __init__(self, r: pd.DataFrame, fallback: list[str]):
        self.r = r[r.consequents.map(len) == 1]
        self.fallback = fallback

    def recommend(self, basket: set, k: int = 3) -> list[str]:
        fired = self.r[self.r.antecedents.map(lambda a: a <= basket)]
        scores: dict = {}
        for a, c, conf in zip(fired.antecedents, fired.consequents, fired.confidence):
            item = next(iter(c))
            if item not in basket:
                scores[item] = max(scores.get(item, 0), conf)
        ranked = sorted(scores, key=scores.get, reverse=True)
        ranked += [i for i in self.fallback if i not in basket and i not in ranked]
        return ranked[:k]


def hide_one_eval(test: list, recommend, k: int = 3, seed: int = 0) -> float:
    """Hide one random item from each multi-item test order; hit if it is in the top k."""
    rng = np.random.default_rng(seed)
    hits = n = 0
    for order in test:
        if len(order) < 2:
            continue
        hidden = order[rng.integers(len(order))]
        hits += hidden in recommend(set(order) - {hidden}, k)
        n += 1
    return hits / n


def compare_recommenders(orders: list, k: int = 3) -> dict:
    train, test = split(orders)
    popular = [i for i, _ in Counter(i for o in train for i in o).most_common()]
    r = prune_redundant(rules(one_hot(train), min_support=0.005, min_confidence=0.2, min_lift=1.2))
    rec = RuleRecommender(r, popular)
    return {"popularity": round(hide_one_eval(test, lambda b, k: [i for i in popular if i not in b][:k], k), 3),
            "rules": round(hide_one_eval(test, rec.recommend, k), 3), "k": k, "test_orders": len(test)}


def selection_bias_demo(orders: list) -> dict:
    """charger and phone are generated independently, yet lift < 1 because empty orders are never recorded."""
    oh = one_hot(orders)
    p_phone, p_charger = oh.phone.mean(), oh.charger.mean()
    both = (oh.phone & oh.charger).mean()
    return {"lift_charger_phone": round(float(both / (p_phone * p_charger)), 3),
            "p_phone": round(float(p_phone), 3), "p_phone_given_charger": round(float(both / p_charger), 3)}
