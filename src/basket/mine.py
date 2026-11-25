"""Frequent itemsets and association rules with mlxtend."""
from __future__ import annotations

import time

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder


def one_hot(orders: list[list[str]]) -> pd.DataFrame:
    te = TransactionEncoder()
    return pd.DataFrame(te.fit(orders).transform(orders), columns=te.columns_)


def itemsets(onehot: pd.DataFrame, min_support: float, algo: str = "fpgrowth") -> pd.DataFrame:
    fn = {"fpgrowth": fpgrowth, "apriori": apriori}[algo]
    return fn(onehot, min_support=min_support, use_colnames=True)


def timed_itemsets(onehot: pd.DataFrame, min_support: float) -> dict:
    """Run both algorithms, check they agree, and report wall time."""
    out = {}
    for algo in ("apriori", "fpgrowth"):
        t = time.perf_counter()
        res = itemsets(onehot, min_support, algo)
        out[algo] = (res, time.perf_counter() - t)
    a, f = (set(map(frozenset, out[k][0].itemsets)) for k in ("apriori", "fpgrowth"))
    return {"itemsets": len(a), "same_result": a == f,
            "apriori_s": round(out["apriori"][1], 3), "fpgrowth_s": round(out["fpgrowth"][1], 3)}


def rules(onehot: pd.DataFrame, min_support: float = 0.01, min_confidence: float = 0.3,
          min_lift: float = 1.2) -> pd.DataFrame:
    fi = itemsets(onehot, min_support)
    r = association_rules(fi, metric="confidence", min_threshold=min_confidence)
    r = r[r["lift"] >= min_lift].copy()
    r["rule"] = r.antecedents.map(lambda s: " + ".join(sorted(s))) + " -> " + r.consequents.map(
        lambda s: " + ".join(sorted(s)))
    cols = ["rule", "antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
    return r[cols].sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
