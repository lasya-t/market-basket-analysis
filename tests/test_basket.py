import warnings

import pandas as pd

from basket.data import make_orders
from basket.evaluate import RuleRecommender, compare_recommenders, selection_bias_demo
from basket.mine import one_hot, prune_redundant, rules, timed_itemsets

warnings.filterwarnings("ignore", category=DeprecationWarning)
ORDERS = make_orders(6000)


def test_apriori_and_fpgrowth_agree():
    assert timed_itemsets(one_hot(ORDERS), 0.01)["same_result"]


def test_planted_rules_are_found():
    r = prune_redundant(rules(one_hot(ORDERS)))
    found = set(r.rule)
    for rule in ["watch_band -> smartwatch", "stylus -> tablet", "phone -> case"]:
        assert rule in found


def test_independent_popular_items_produce_no_rule():
    r = rules(one_hot(ORDERS))
    assert not r.rule.str.contains("earbuds -> charger|charger -> earbuds").any()


def test_pruning_drops_rule_with_unrelated_extra_item():
    r = pd.DataFrame({
        "rule": ["case -> phone", "case + charger -> phone"],
        "antecedents": [frozenset({"case"}), frozenset({"case", "charger"})],
        "consequents": [frozenset({"phone"})] * 2,
        "support": [0.3, 0.1], "confidence": [0.85, 0.86], "lift": [2.0, 2.0],
        "leverage": [0.1, 0.05], "conviction": [3.0, 3.0]})
    assert list(prune_redundant(r).rule) == ["case -> phone"]


def test_recommender_skips_items_already_in_basket():
    r = pd.DataFrame({"antecedents": [frozenset({"phone"})] * 2,
                      "consequents": [frozenset({"case"}), frozenset({"sim_kit"})],
                      "confidence": [0.7, 0.4]})
    rec = RuleRecommender(r, fallback=["charger", "earbuds"])
    assert rec.recommend({"phone", "case"}, k=2) == ["sim_kit", "charger"]


def test_rules_beat_popularity():
    res = compare_recommenders(ORDERS, k=3)
    assert res["rules"] > res["popularity"] + 0.05


def test_non_empty_orders_bias_lift_below_one():
    assert selection_bias_demo(ORDERS)["lift_charger_phone"] < 0.85
