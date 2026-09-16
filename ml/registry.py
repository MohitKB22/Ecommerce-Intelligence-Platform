"""Filesystem-backed model registry + artifact store.

Every training run writes an artifact (joblib), a metrics JSON and a manifest
entry. The backend reads the same manifest at inference time, so training and
serving never disagree about which version is live.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

MANIFEST_NAME = "registry.json"
_LOCK = threading.RLock()


def _default_model_root() -> Path:
    env = os.getenv("MODEL_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / "ml" / "models"


@dataclass(slots=True)
class ModelCard:
    """Immutable metadata describing one trained model version."""

    name: str
    version: str
    algorithm: str
    artifact_path: str
    dataset_version: str
    training_rows: int
    trained_at: str
    training_duration_s: float
    metrics: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    baseline_stats: dict[str, Any] = field(default_factory=dict)
    stage: str = "production"
    notes: str = ""
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


class ModelRegistry:
    def __init__(self, root: Path | str | None = None):
        self.root = Path(root) if root else _default_model_root()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / MANIFEST_NAME

    # ---- manifest io -------------------------------------------------
    def _read_manifest(self) -> dict[str, list[dict]]:
        if not self.manifest_path.exists():
            return {}
        try:
            with self.manifest_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_manifest(self, manifest: dict[str, list[dict]]) -> None:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.root), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, default=str)
            shutil.move(tmp_path, self.manifest_path)  # atomic replace
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ---- public api --------------------------------------------------
    @staticmethod
    def new_version(prefix: str = "v") -> str:
        return prefix + datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    def artifact_path(self, name: str, version: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{version}.joblib"

    def save(self, name: str, artifact: Any, *, version: str | None = None, algorithm: str = "",
             dataset_version: str = "", training_rows: int = 0, training_duration_s: float = 0.0,
             metrics: dict | None = None, params: dict | None = None, feature_names: list[str] | None = None,
             baseline_stats: dict | None = None, stage: str = "production", notes: str = "") -> ModelCard:
        version = version or self.new_version()
        path = self.artifact_path(name, version)
        joblib.dump(artifact, path, compress=3)

        card = ModelCard(
            name=name,
            version=version,
            algorithm=algorithm,
            artifact_path=str(path.relative_to(self.root)),
            dataset_version=dataset_version,
            training_rows=training_rows,
            trained_at=datetime.now(timezone.utc).isoformat(),
            training_duration_s=round(training_duration_s, 3),
            metrics=metrics or {},
            params=params or {},
            feature_names=feature_names or [],
            baseline_stats=baseline_stats or {},
            stage=stage,
            notes=notes,
        )

        with _LOCK:
            manifest = self._read_manifest()
            versions = manifest.setdefault(name, [])
            versions = [v for v in versions if v.get("version") != version]
            versions.append(card.to_dict())
            versions.sort(key=lambda v: v.get("trained_at", ""), reverse=True)
            manifest[name] = versions[:20]  # retain a bounded history
            self._write_manifest(manifest)

        metrics_path = self.root / name / f"{version}.metrics.json"
        metrics_path.write_text(json.dumps(card.to_dict(), indent=2, default=str), encoding="utf-8")
        return card

    def list_models(self) -> dict[str, list[ModelCard]]:
        manifest = self._read_manifest()
        return {name: [ModelCard.from_dict(v) for v in versions] for name, versions in manifest.items()}

    def history(self, name: str) -> list[ModelCard]:
        return [ModelCard.from_dict(v) for v in self._read_manifest().get(name, [])]

    def latest_card(self, name: str, stage: str = "production") -> ModelCard | None:
        for entry in self._read_manifest().get(name, []):
            if entry.get("stage") == stage:
                return ModelCard.from_dict(entry)
        return None

    def load(self, name: str, version: str | None = None) -> tuple[Any, ModelCard]:
        entries = self._read_manifest().get(name, [])
        if not entries:
            raise FileNotFoundError(f"No registered model named {name!r}")
        entry = next((e for e in entries if e.get("version") == version), None) if version else entries[0]
        if entry is None:
            raise FileNotFoundError(f"Model {name!r} version {version!r} is not registered")
        card = ModelCard.from_dict(entry)
        path = self.root / card.artifact_path
        if not path.exists():
            raise FileNotFoundError(f"Artifact missing for {name}:{card.version} at {path}")
        return joblib.load(path), card

    def exists(self, name: str) -> bool:
        entries = self._read_manifest().get(name, [])
        return bool(entries) and (self.root / entries[0]["artifact_path"]).exists()

    def promote(self, name: str, version: str, stage: str = "production") -> ModelCard:
        with _LOCK:
            manifest = self._read_manifest()
            entries = manifest.get(name, [])
            target = None
            for e in entries:
                if e.get("version") == version:
                    e["stage"] = stage
                    target = e
                elif e.get("stage") == stage:
                    e["stage"] = "archived"
            if target is None:
                raise FileNotFoundError(f"Model {name!r} version {version!r} is not registered")
            self._write_manifest(manifest)
            return ModelCard.from_dict(target)


_registry: ModelRegistry | None = None


def get_registry(root: Path | str | None = None) -> ModelRegistry:
    global _registry
    if root is not None:
        return ModelRegistry(root)
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
