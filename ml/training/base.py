"""Shared scaffolding for every training pipeline.

Each pipeline follows the same contract: load -> validate -> preprocess ->
train -> evaluate -> save -> register. `TrainingResult` is what the CLI prints
and what the model registry stores.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.core.db import engine  # noqa: E402
from app.core.logging_config import configure_logging, get_logger  # noqa: E402

from ml.preprocessing.loaders import TrainingFrames, load_all  # noqa: E402
from ml.registry import ModelCard, get_registry  # noqa: E402


class DataValidationError(RuntimeError):
    """Raised when input data is not fit for training."""


@dataclass(slots=True)
class TrainingResult:
    model_name: str
    version: str
    algorithm: str
    metrics: dict[str, Any]
    params: dict[str, Any] = field(default_factory=dict)
    training_rows: int = 0
    duration_s: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "version": self.version,
            "algorithm": self.algorithm,
            "training_rows": self.training_rows,
            "duration_s": round(self.duration_s, 3),
            "metrics": self.metrics,
            "params": self.params,
            "notes": self.notes,
        }


class TrainingPipeline(ABC):
    """Template method: subclasses supply preprocess/train/evaluate/artifact."""

    name: str = "model"
    algorithm: str = ""
    requires: tuple[str, ...] = ()

    def __init__(self, registry_root: str | Path | None = None, dataset_version: str = ""):
        self.logger = get_logger(f"train.{self.name}")
        self.registry = get_registry(registry_root)
        self.dataset_version = dataset_version
        self.frames: TrainingFrames | None = None
        self._training_rows = 0

    # ---- steps -------------------------------------------------------
    def load(self) -> TrainingFrames:
        self.logger.info("loading_data")
        return load_all(engine)

    def validate(self, frames: TrainingFrames) -> None:
        problems = frames.validate(self.requires)
        if problems:
            raise DataValidationError(
                f"{self.name}: input data failed validation -> " + "; ".join(problems)
            )
        self.logger.info("data_validated", **{k: len(getattr(frames, k)) for k in self.requires})

    @abstractmethod
    def preprocess(self, frames: TrainingFrames) -> Any: ...

    @abstractmethod
    def train(self, prepared: Any) -> Any: ...

    @abstractmethod
    def evaluate(self, artifact: Any, prepared: Any) -> dict[str, Any]: ...

    def params(self, artifact: Any) -> dict[str, Any]:
        return {}

    def feature_names(self, artifact: Any) -> list[str]:
        return []

    def baseline_stats(self, artifact: Any, prepared: Any) -> dict[str, Any]:
        """Reference distributions captured at training time for drift detection."""
        return {}

    def notes(self, artifact: Any) -> str:
        return ""

    def persist_side_effects(self, artifact: Any, prepared: Any, version: str) -> None:  # noqa: B027
        """Optional hook to write derived rows (segments, forecasts...) to the DB.

        Intentionally a concrete no-op rather than abstract: most pipelines have
        no side effects and should not be forced to implement this.
        """

    # ---- orchestration -----------------------------------------------
    def run(self) -> TrainingResult:
        started = time.perf_counter()
        frames = self.load()
        self.frames = frames
        self.validate(frames)

        self.logger.info("preprocessing")
        prepared = self.preprocess(frames)

        self.logger.info("training")
        artifact = self.train(prepared)

        self.logger.info("evaluating")
        metrics = self.evaluate(artifact, prepared)

        duration = time.perf_counter() - started
        version = self.registry.new_version()
        dataset_version = self.dataset_version or self._infer_dataset_version()

        card: ModelCard = self.registry.save(
            self.name,
            artifact,
            version=version,
            algorithm=self.algorithm,
            dataset_version=dataset_version,
            training_rows=self._training_rows,
            training_duration_s=duration,
            metrics=metrics,
            params=self.params(artifact),
            feature_names=self.feature_names(artifact),
            baseline_stats=self.baseline_stats(artifact, prepared),
            notes=self.notes(artifact),
        )

        self.persist_side_effects(artifact, prepared, version)
        self.sync_registry_row(card)

        result = TrainingResult(
            model_name=self.name,
            version=card.version,
            algorithm=self.algorithm,
            metrics=metrics,
            params=card.params,
            training_rows=self._training_rows,
            duration_s=duration,
            notes=card.notes,
        )
        self.logger.info("training_complete", version=card.version, duration_s=round(duration, 2), **_flat(metrics))
        return result

    def _infer_dataset_version(self) -> str:
        manifest = REPO_ROOT / "data" / "seed" / "dataset_manifest.json"
        if manifest.exists():
            try:
                return json.loads(manifest.read_text(encoding="utf-8")).get("dataset_version", "unknown")
            except (json.JSONDecodeError, OSError):
                pass
        return "unknown"

    def sync_registry_row(self, card: ModelCard) -> None:
        """Mirror the registry entry into the database for the admin dashboard."""
        from app.models import ModelRegistryEntry
        from sqlalchemy.orm import Session

        try:
            with Session(engine) as session:
                session.query(ModelRegistryEntry).filter(
                    ModelRegistryEntry.name == card.name, ModelRegistryEntry.is_active.is_(True)
                ).update({"is_active": False, "stage": "archived"}, synchronize_session=False)
                session.add(
                    ModelRegistryEntry(
                        name=card.name,
                        version=card.version,
                        stage=card.stage,
                        algorithm=card.algorithm,
                        artifact_path=card.artifact_path,
                        dataset_version=card.dataset_version,
                        training_rows=card.training_rows,
                        trained_at=datetime.fromisoformat(card.trained_at),
                        training_duration_s=card.training_duration_s,
                        metrics=card.metrics,
                        params=card.params,
                        feature_names=card.feature_names,
                        baseline_stats=card.baseline_stats,
                        is_active=True,
                        notes=card.notes,
                    )
                )
                session.commit()
        except Exception as exc:  # noqa: BLE001 - registry mirror must never fail training
            self.logger.warning("registry_db_sync_failed", error=str(exc))


def _flat(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if isinstance(v, int | float | str)}


def run_cli(pipeline_cls: type[TrainingPipeline], description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--registry", default=None, help="override the model registry root")
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON only")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    configure_logging("WARNING" if (args.json or args.quiet) else "INFO", json_output=False)
    try:
        result = pipeline_cls(registry_root=args.registry).run()
    except DataValidationError as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(json.dumps({"status": "error", "reason": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 1

    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0
