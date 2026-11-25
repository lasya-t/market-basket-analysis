# market-basket-analysis

Association rules on mobile-store orders (phones, cases, protection plans, tablets, watches…) with mlxtend.
The project mines frequent itemsets, extracts and prunes rules, checks that they hold on unseen orders, and
uses them for a "frequently bought together" recommender that is compared against a popularity baseline.

## Data

Synthetic orders with planted patterns, so mined rules can be checked against the truth. That gives 13,954
non-empty orders over 14 items, with 2.47 items per order on average.

- phone → case 65%, screen protector 55%, protection plan 40%, SIM kit 35%
- tablet → stylus 50%, keyboard cover 35%
- smartwatch → watch band 45%
- earbuds (22%) and chargers (25%) are popular but bought independently of everything else

## Results (`PYTHONPATH=src python -m basket`)

**Apriori vs FP-Growth.** Both return identical itemsets (157 at 1% support, 545 at 0.2%). With only 14
items, apriori is actually faster here (0.015 s vs 0.029 s). FP-Growth's advantage shows up with large
catalogues, not small ones.

**Pruning.** At support ≥ 1%, confidence ≥ 30% and lift ≥ 1.2, mining produced **358 rules**. Most were a real
rule with an unrelated item bolted on (`case + charger → phone`). A rule is kept only if it beats every
simpler version by 5+ points of confidence, which leaves **59**. The single-item rules that remain match what
was planted:

| Rule | Support | Confidence | Lift |
|---|---|---|---|
| watch_band → smartwatch | 4.1% | 1.000 | 11.1 |
| stylus → tablet | 6.1% | 1.000 | 8.7 |
| tablet → stylus | 6.1% | 0.527 | 8.7 |
| phone → case | 28.7% | 0.670 | 2.0 |
| phone → screen_protector | 24.0% | 0.561 | 2.0 |
| phone → protection_plan | 16.8% | 0.392 | 2.3 |

No rule links earbuds and chargers. They're popular, but not related to each other.

**Holdout (70/30 split).** The rules were re-scored on orders they hadn't seen. Rules with support ≥ 5% moved
by 0.018 confidence on average. Rules under 2% support moved by 0.046 on average (up to 0.097). None of the
55 rules dropped below lift 1. Low-support rules are the noisy ones.

**Recommender (hide one item from each test order, can it be recovered?)**

| | Hit@1 | Hit@3 |
|---|---|---|
| Most popular items | 0.254 | 0.649 |
| Association rules (popularity as fallback) | **0.500** | **0.801** |

## Things that look like insights but aren't

- **Lift compounds on multi-item rules.** `case + stylus → phone + tablet` has lift 24, the highest of any rule.
  But phones and tablets are generated independently, so this lift is just the phone lift (~2.3) multiplied
  by the tablet lift (~8.7) at 1% support. Ranking by lift alone puts these at the top.
- **A shared cause looks like a link.** `sim_kit → case` (0.667 confidence) looks like a real rule, but it's
  phone → case (0.670) in disguise. Both items come with phones. Pruning only removes extra antecedent
  items, so it doesn't catch this.
- **Recording only non-empty orders creates negative lift.** Chargers and phones are generated independently,
  yet lift(charger, phone) = **0.681**. An order with only a charger is recorded, but a customer who bought
  nothing never appears. That raises P(phone) among recorded orders (0.429) above P(phone | charger) (0.292).
  Lift below 1 between unrelated popular items can come from this alone.

## Run

```bash
pip install -r requirements.txt
PYTHONPATH=src python -m basket
pytest -q
```
