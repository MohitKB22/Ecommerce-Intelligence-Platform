"""Evaluation metrics shared by every training pipeline.

Ranking metrics are implemented directly (rather than pulled from a library)
so their semantics are explicit and unit-testable.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def precision_at_k(recommended: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = set(relevant)
    if not rel:
        return 0.0
    top = list(recommended)[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in rel) / float(k)


def recall_at_k(recommended: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    top = list(recommended)[:k]
    return sum(1 for item in top if item in rel) / float(len(rel))


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(list(gains)[:k]))


def ndcg_at_k(recommended: Sequence[Any], relevant: Iterable[Any], k: int,
              gains: dict[Any, float] | None = None) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    gain_of = (lambda item: float(gains.get(item, 0.0))) if gains else (lambda item: 1.0 if item in rel else 0.0)
    actual = dcg_at_k([gain_of(i) for i in list(recommended)[:k]], k)
    ideal_gains = sorted((gain_of(i) for i in rel), reverse=True)
    ideal = dcg_at_k(ideal_gains, k)
    return actual / ideal if ideal > 0 else 0.0


def average_precision_at_k(recommended: Sequence[Any], relevant: Iterable[Any], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    hits, score = 0, 0.0
    for i, item in enumerate(list(recommended)[:k]):
        if item in rel:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(rel), k)


def map_at_k(all_recommended: dict[Any, Sequence[Any]], all_relevant: dict[Any, Iterable[Any]], k: int) -> float:
    keys = [key for key in all_recommended if all_relevant.get(key)]
    if not keys:
        return 0.0
    return float(np.mean([average_precision_at_k(all_recommended[key], all_relevant[key], k) for key in keys]))


def ranking_report(all_recommended: dict[Any, Sequence[Any]], all_relevant: dict[Any, Iterable[Any]],
                   ks: Sequence[int] = (5, 10, 20)) -> dict[str, float]:
    """Precision/Recall/NDCG/MAP at several cut-offs, averaged over users."""
    keys = [key for key in all_recommended if all_relevant.get(key)]
    report: dict[str, float] = {"evaluated_users": float(len(keys))}
    if not keys:
        for k in ks:
            report |= {f"precision@{k}": 0.0, f"recall@{k}": 0.0, f"ndcg@{k}": 0.0, f"map@{k}": 0.0}
        return report
    for k in ks:
        report[f"precision@{k}"] = round(
            float(np.mean([precision_at_k(all_recommended[u], all_relevant[u], k) for u in keys])), 5)
        report[f"recall@{k}"] = round(
            float(np.mean([recall_at_k(all_recommended[u], all_relevant[u], k) for u in keys])), 5)
        report[f"ndcg@{k}"] = round(
            float(np.mean([ndcg_at_k(all_recommended[u], all_relevant[u], k) for u in keys])), 5)
        report[f"map@{k}"] = round(map_at_k(all_recommended, all_relevant, k), 5)
    return report


def coverage(all_recommended: dict[Any, Sequence[Any]], catalogue_size: int, k: int = 10) -> float:
    """Fraction of the catalogue that appears in any top-k list."""
    if catalogue_size <= 0:
        return 0.0
    seen: set[Any] = set()
    for rec in all_recommended.values():
        seen.update(list(rec)[:k])
    return round(len(seen) / catalogue_size, 5)


def classification_report_dict(y_true: Sequence, y_pred: Sequence, labels: Sequence | None = None) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_true = list(y_true)
    y_pred = list(y_pred)
    if not y_true:
        return {"accuracy": 0.0, "support": 0}
    labels = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    per_class = {}
    p_each = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    r_each = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    f_each = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    for i, label in enumerate(labels):
        per_class[str(label)] = {
            "precision": round(float(p_each[i]), 5),
            "recall": round(float(r_each[i]), 5),
            "f1": round(float(f_each[i]), 5),
            "support": int(sum(1 for y in y_true if y == label)),
        }
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 5),
        "f1_weighted": round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 5),
        "labels": [str(label) for label in labels],
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
        "support": len(y_true),
    }


def forecast_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if yt.size == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0, "smape": 0.0, "bias": 0.0, "support": 0}
    err = yp - yt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nonzero = yt != 0
    mape = float(np.mean(np.abs(err[nonzero] / yt[nonzero])) * 100) if nonzero.any() else 0.0
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    smape = float(np.mean(np.where(denom == 0, 0.0, np.abs(err) / np.where(denom == 0, 1, denom))) * 100)
    return {
        "mae": round(mae, 5),
        "rmse": round(rmse, 5),
        "mape": round(mape, 4),
        "smape": round(smape, 4),
        "bias": round(float(np.mean(err)), 5),
        "support": int(yt.size),
    }


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    from sklearn.metrics import r2_score

    base = forecast_metrics(y_true, y_pred)
    if base["support"] > 1:
        base["r2"] = round(float(r2_score(list(y_true), list(y_pred))), 5)
    else:
        base["r2"] = 0.0
    return base


def population_stability_index(expected: Sequence[float], actual: Sequence[float], bins: int = 10) -> float:
    """PSI - the standard data-drift statistic. >0.2 conventionally means drift."""
    exp = np.asarray(expected, dtype=float)
    act = np.asarray(actual, dtype=float)
    if exp.size < 2 or act.size < 2:
        return 0.0
    quantiles = np.unique(np.quantile(exp, np.linspace(0, 1, bins + 1)))
    if quantiles.size < 3:
        return 0.0
    e_counts, _ = np.histogram(exp, bins=quantiles)
    a_counts, _ = np.histogram(act, bins=quantiles)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), 1e-6, None)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), 1e-6, None)
    return round(float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))), 5)
