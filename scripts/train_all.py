#!/usr/bin/env python3
"""Run every ML training pipeline in dependency order.

    python scripts/train_all.py [--continue-on-error]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.logging_config import configure_logging  # noqa: E402

from ml.training.train_forecasting import ForecastingPipeline  # noqa: E402
from ml.training.train_price_prediction import PricePredictionPipeline  # noqa: E402
from ml.training.train_recommendation import RecommendationPipeline  # noqa: E402
from ml.training.train_segmentation import SegmentationPipeline  # noqa: E402
from ml.training.train_sentiment import SentimentPipeline  # noqa: E402

PIPELINES = [
    ("sentiment", SentimentPipeline),
    ("segmentation", SegmentationPipeline),
    ("recommendation", RecommendationPipeline),
    ("forecasting", ForecastingPipeline),
    ("price_prediction", PricePredictionPipeline),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train every model.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="subset of pipeline names")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    configure_logging("WARNING" if args.quiet else "INFO", json_output=False)
    results: dict[str, object] = {}
    failures = 0
    for name, cls in PIPELINES:
        if args.only and name not in args.only:
            continue
        started = time.perf_counter()
        try:
            result = cls().run()
            results[name] = result.to_dict()
            print(f"[ok]   {name:18s} version={result.version} "
                  f"({time.perf_counter() - started:.1f}s)", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001 - orchestrator boundary
            failures += 1
            results[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"[FAIL] {name:18s} {type(exc).__name__}: {exc}", file=sys.stderr)
            if not args.continue_on_error:
                print(json.dumps(results, indent=2, default=str))
                return 1
    print(json.dumps(results, indent=2, default=str))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
