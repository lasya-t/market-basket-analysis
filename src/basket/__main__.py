"""python -m basket"""
import warnings

import pandas as pd

from basket.data import make_orders
from basket.evaluate import compare_recommenders, holdout_check, selection_bias_demo, split
from basket.mine import one_hot, prune_redundant, rules, timed_itemsets

warnings.filterwarnings("ignore", category=DeprecationWarning)
pd.set_option("display.width", 140)


def main() -> None:
    orders = make_orders()
    oh = one_hot(orders)
    print(f"orders={len(orders)}  items={oh.shape[1]}  avg basket={oh.sum(axis=1).mean():.2f}")

    print("\n== apriori vs fp-growth")
    for s in (0.01, 0.005, 0.002):
        print(f"min_support={s}", timed_itemsets(oh, s))

    r = rules(oh)
    p = prune_redundant(r)
    print(f"\n== rules (support>=1%, confidence>=30%, lift>=1.2): {len(r)} raw, {len(p)} after pruning")
    single = p[(p.antecedents.map(len) == 1) & (p.consequents.map(len) == 1)]
    print(single[["rule", "support", "confidence", "lift"]].round(3).to_string(index=False))

    train, test = split(orders)
    train_rules = prune_redundant(rules(one_hot(train)))
    h = holdout_check(train_rules, test)
    h["support_band"] = pd.cut(train_rules.support, [0, 0.02, 0.05, 1],
                               labels=["<2%", "2-5%", ">=5%"])
    h["abs_conf_change"] = (h.test_confidence - h.confidence).abs()
    print("\n== holdout: change in confidence on unseen orders, by rule support")
    print(h.groupby("support_band", observed=True).abs_conf_change.agg(["count", "mean", "max"]).round(3))
    print("rules whose lift fell below 1 on holdout:", int((h.test_lift < 1).sum()))

    print("\n== hide-one-item recommendation hit rate")
    for k in (1, 3):
        print(compare_recommenders(orders, k))

    print("\n== independent items, but only non-empty orders are recorded")
    print(selection_bias_demo(orders))


if __name__ == "__main__":
    main()
