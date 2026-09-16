# Machine learning

Five models, each with training, evaluation against a baseline, a saved artifact,
inference, API integration and UI integration.

All figures below come from the full dataset (1,200 users · 600 products ·
6,000 reviews · 136,000 events · 16,700 orders). Reproduce with
`make reset && make train`.

---

## Common contract

Every pipeline subclasses `ml/training/base.py::TrainingPipeline`:

```
load()      -> read from the operational database
validate()  -> row counts, value ranges, duplicates; raises DataValidationError
preprocess()-> feature engineering + temporal/stratified split
train()     -> fit
evaluate()  -> metrics AND a baseline comparison
save()      -> joblib artifact + metrics JSON + registry manifest
register()  -> mirror metadata into model_registry
```

Training reads from the same database the API serves from, so a model is never
trained on data the application cannot reproduce at inference time.

---

## 1. Recommendation

**Algorithm.** BM25-weighted implicit-feedback matrix → truncated SVD (32 factors)
for user/item embeddings; item-item cosine neighbourhood (top-50); TF-IDF+SVD
content embeddings; lift-weighted co-purchase graph; popularity and trending
aggregates.

**The blend.** Five signals, weights fitted on validation:

```
final = w_collab · residualise(collaborative, popularity)
      + w_content · residualise(content, popularity)
      + w_popularity · popularity
      + w_personal · affinity(category, brand)
      + w_business · (margin, conversion, availability)
final *= (0.35 + 0.65 · in_stock)      # damp out-of-stock items
```

**Evaluation.** Strict temporal split: the model sees only interactions before the
cut-off. The post-cutoff window is halved — the first half fits the weights, the
second half is the reported test set. Baselines use *identical* history-exclusion
rules (getting this wrong initially made the model look 58% worse than it was).

| Metric | Hybrid | Popularity | Random |
|---|---|---|---|
| NDCG@10 | **0.0638** | 0.0610 | 0.0065 |
| Precision@10 | 0.0219 | - | - |
| Recall@20 | 0.1216 | - | - |
| Catalogue coverage@10 | 0.21 | - | - |

+4.6% over popularity, 10x over random.

**Cold start.** `is_cold_start` (fewer than 3 interactions) or an unknown user →
popularity + trending + rating, scoped to the strongest category affinity when
one exists. New products are reachable immediately via content similarity.

**Explainability.** Every recommendation returns its per-signal contribution and a
sentence: *"Recommended because customers with similar taste bought this and it
matches your preferred categories."*

---

## 2. Review sentiment

**Algorithm.** Word (1-2 gram) + character (3-5 gram) TF-IDF union →
multinomial logistic regression with balanced class weights. Labels are derived
from star ratings (4-5 positive, 3 neutral, 1-2 negative) — the standard weak
supervision signal for review corpora — but the model reads **text only**, so it
can score reviews that have no rating attached.

| Metric | Value | Baseline (majority class) |
|---|---|---|
| Accuracy | **0.9147** | 0.7687 |
| F1 macro | 0.8384 | 0.2897 |
| Aspect detection F1 | 0.7730 | - |
| Aspect polarity accuracy | 0.9956 | - |

**A note on the 91%.** An earlier version scored a perfect 1.000. That was
leakage, not skill: review *titles* were generated deterministically from the
rating, handing the label to the classifier. The generator now draws titles from
a noisy tone variable with cross-bucket overlap, and 22% of reviews are written
in a tone one step away from their rating — as real reviewers do. There is a
regression test asserting accuracy stays **below** 1.0.

**Aspect polarity.** Polarity is decided from the signed score
`P(positive) - P(negative)` with a symmetric threshold, not the arg-max label:
training on whole reviews makes the label strongly positive-biased on short
clauses, which had collapsed polarity accuracy to 0.35. Explicit lexicon evidence
overrides the classifier when a clause contains a known polarity phrase.

> Caveat on the 0.996 figure: the synthetic reviews are composed from the same
> aspect lexicon the extractor consults, so lexicon matching is unusually
> effective here. On organic review text the classifier path would carry far more
> of the load and this number would be materially lower.

**Aspect extraction.** Reviews are split on contrastive conjunctions
(`but`, `however`, `although`), each clause is scored independently, and negation
terms invert the polarity. Ten aspects are tracked (battery life, sound quality,
comfort, build quality, value, shipping, durability, ease of use, design,
performance).

```
"Battery life is excellent but the ear cushions are uncomfortable."
  overall: positive
  battery_life: positive
  comfort: negative
```

---

## 3. Demand forecasting

**Grain.** Weekly. Daily per-SKU demand in a long-tail catalogue is dominated by
zeros, which makes daily point forecasts unstable and uninformative.

**Algorithm.** A seasonal-naive baseline (recent 4-week level × month-of-year
factor) plus a `HistGradientBoostingRegressor` trained on the **residual**, using
week lags 1-12, rolling mean/std/max, a 4-vs-12-week trend term, cyclical
calendar encodings, price ratio, promotion flag and product metadata.

| Metric | Hybrid | Seasonal naive |
|---|---|---|
| MAE | **1.0885** | 1.1902 |
| RMSE | 1.6800 | - |
| Improvement | **-8.5%** | - |

Plain regression on raw units *loses* to the naive baseline (MAE 1.089 raw-target vs 1.190 naive in earlier measurement) —
the residual formulation is what makes the learned model worth having. The
pipeline verifies this at training time and keeps the naive baseline if the
learned model fails to beat it.

**Serving.** Recursive weekly rollout, each prediction fed back as the next step's
lag, then expanded to daily points and 7/14/30-day cumulative totals. Prediction
intervals come from the empirical residual spread and widen with √horizon.
Stock-out risk compares projected demand against inventory on hand.

---

## 4. Price prediction

**Algorithm.** `HistGradientBoostingRegressor` on **log price** (errors are
multiplicative in retail), using brand reputation, rating, inventory, discount,
90-day units/revenue/velocity/views, conversion rate, leave-one-out category
price context, days since launch and catalogue metadata.

| Metric | Model | Baseline (category median) |
|---|---|---|
| MAE | **$32.36** | $66.13 |
| MAPE | 11.90% | - |
| R2 | 0.9680 | - |
| Within 20% | 82.5% | - |
| Improvement | **-51.1% MAE** | - |

**Two leakage fixes.** `base_cost` was excluded — the generator derives cost from
price (r=0.985), so including it pushed R² to 0.988 while teaching the model
nothing about the market. `category_price_rank` was also excluded: it is the
product's own price percentile within its category, i.e. the target in disguise.
Category median/quartiles are computed leave-one-out. There is a regression test
asserting neither feature reappears.

**Explainability.** Global: permutation importance on the hold-out set. Local:
per-product ablation against the category-median feature vector, reported as
"increases price" / "decreases price" contributions.

---

## 5. Customer segmentation

**Algorithm.** RFM plus behavioural features (recency, frequency, monetary, AOV,
product diversity, session frequency, discount affinity, tenure, review count),
log-compressed, `StandardScaler`, K-Means with k selected across 4-8.

**Selection rule.** Silhouette alone favours very coarse clusterings — it picked
k=4, collapsing seven business segments into four. The rule is now: take the
**largest** k whose silhouette is within 88% of the best, a one-standard-error
style trade-off that buys actionable granularity for a negligible cohesion cost.

| Metric | Value |
|---|---|
| Silhouette | 0.3371 |
| Davies-Bouldin | 1.0416 |
| Clusters | 6 |
| Largest segment share | 0.23 |

Centroids are mapped to business names greedily so no name is used twice:
`high_value`, `loyal_customer`, `frequent_buyer`, `discount_seeker`,
`occasional_buyer`, `at_risk`, `new_customer`.

---

## Feature store

Offline truth in `user_features` / `product_features` (versioned + timestamped),
online reads through the cache. Rebuilt by the background worker.

```
user_purchase_frequency   product_popularity
user_average_order_value  product_conversion_rate
user_total_spend          product_rating
user_recency_days         product_demand_30d
user_session_count        product_price_trend
user_distinct_products    product_view_count
user_discount_affinity    product_cart_rate
user_review_count         product_return_rate
```

---

## Monitoring

- **Registry** — every version with algorithm, dataset version, training rows,
  duration, metrics, params and feature names.
- **Serving stats** — call count, mean and p95 latency, error rate, captured on
  every inference call; 15% of predictions are sampled into `model_predictions`.
- **Drift** — live feature distributions compared against baselines captured at
  training time; PSI is available for numeric distributions. Relative change
  beyond `ML_DRIFT_THRESHOLD` (default 0.2) flags drift.
- **Hot reload** — `POST /api/v1/ml/reload` picks up newly trained versions
  without a restart; the worker checks periodically.

## Reproducibility

Every pipeline uses fixed seeds; the generator is fully deterministic given its
seed. `dataset_version` is a hash of the generator configuration and is recorded
on every model card, so a model can always be traced to the exact data that
produced it.
