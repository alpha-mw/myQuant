"""Calibration store — persists prediction-vs-outcome pairs for future tuning.

V1: preset calibration curves with no active learning loop.
Future: active learning from realized outcomes.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from quant_investor.logger import get_logger
from quant_investor.versioning import CALIBRATION_SCHEMA_VERSION

_logger = get_logger("CalibrationStore")

# Preset calibration buckets: branch_name -> {score_bucket: empirical_probability}
# These are initial estimates; the learning loop will refine them over time.
_PRESET_CALIBRATION: dict[str, dict[str, float]] = {
    "quant": {
        "strong_negative": 0.22,
        "negative": 0.35,
        "neutral": 0.50,
        "positive": 0.65,
        "strong_positive": 0.78,
    },
    "fundamental": {
        "strong_negative": 0.28,
        "negative": 0.40,
        "neutral": 0.50,
        "positive": 0.60,
        "strong_positive": 0.72,
    },
}


def _require_supported_branch(branch_name: str) -> str:
    resolved = str(branch_name).strip()
    if resolved not in _PRESET_CALIBRATION:
        raise ValueError(f"Unsupported calibration branch: {branch_name!r}.")
    return resolved


def _score_to_bucket(score: float) -> str:
    score = float(score)
    if not math.isfinite(score):
        raise ValueError(f"Calibration score must be finite: {score!r}.")
    if score <= -0.50:
        return "strong_negative"
    if score <= -0.15:
        return "negative"
    if score <= 0.15:
        return "neutral"
    if score <= 0.50:
        return "positive"
    return "strong_positive"


class CalibrationStore:
    """Manage calibration curves and evidence persistence.

    V1 is read-only from presets.  The ``record_outcome`` method persists
    evidence for future calibration but does not yet update the curves.
    """

    def __init__(self, store_path: str | None = None) -> None:
        self._store_path = Path(store_path or "data/bayesian_calibration/v14/calibration.json")
        self._curves = dict(_PRESET_CALIBRATION)
        self._outcome_stats: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
        self._load()
        self._load_outcomes()

    def _load(self) -> None:
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("Calibration store must be a JSON object.")
                schema_version = str(data.get("schema_version", ""))
                if schema_version != CALIBRATION_SCHEMA_VERSION:
                    raise ValueError(
                        "Calibration schema mismatch: "
                        f"expected {CALIBRATION_SCHEMA_VERSION!r}, got {schema_version!r}."
                    )
                curves = data.get("curves", {})
                if not isinstance(curves, dict):
                    raise ValueError("Calibration curves must be a JSON object.")
                unexpected = sorted(set(curves) - set(_PRESET_CALIBRATION))
                if unexpected:
                    raise ValueError(f"Unexpected calibration branches: {', '.join(unexpected)}")
                self._curves.update(curves)
            except Exception as exc:
                raise ValueError(f"Failed to load calibration store {self._store_path}: {exc}") from exc

    def get_calibration_curve(self, branch_name: str) -> dict[str, float]:
        branch_name = _require_supported_branch(branch_name)
        return dict(self._curves.get(branch_name, _PRESET_CALIBRATION[branch_name]))

    def _load_outcomes(self) -> None:
        outcomes_path = self._store_path.parent / "bayesian_outcomes.jsonl"
        if not outcomes_path.exists():
            return
        aggregates: dict[tuple[str, str], dict[str, float]] = defaultdict(
            lambda: {"sample_size": 0.0, "wins": 0.0, "recent_losses": 0.0, "recent_total": 0.0}
        )
        try:
            rows = outcomes_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            raise ValueError(f"Failed to read calibration outcomes {outcomes_path}: {exc}") from exc
        parsed_rows: list[tuple[int, dict[str, Any]]] = []
        for line_number, line in enumerate(rows, start=1):
            try:
                payload = json.loads(line)
            except Exception as exc:
                raise ValueError(
                    f"Malformed calibration outcome in {outcomes_path} at line {line_number}."
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Calibration outcome at line {line_number} must be a JSON object."
                )
            schema_version = str(payload.get("schema_version", ""))
            if schema_version != CALIBRATION_SCHEMA_VERSION:
                raise ValueError(
                    f"Calibration outcome schema mismatch at line {line_number}: "
                    f"expected {CALIBRATION_SCHEMA_VERSION!r}, got {schema_version!r}."
                )
            parsed_rows.append((line_number, payload))
        for line_number, payload in parsed_rows[-200:]:
            branch_name = str(payload.get("branch", "")).strip()
            bucket = str(payload.get("bucket", "")).strip() or _score_to_bucket(float(payload.get("score", 0.0)))
            if branch_name not in _PRESET_CALIBRATION:
                raise ValueError(f"Unexpected calibration branch at line {line_number}: {branch_name!r}.")
            if not bucket:
                raise ValueError(f"Missing calibration bucket at line {line_number}.")
            realized_return = float(payload.get("realized_return", 0.0) or 0.0)
            key = (branch_name, bucket)
            aggregates[key]["sample_size"] += 1.0
            if realized_return > 0.0:
                aggregates[key]["wins"] += 1.0
            aggregates[key]["recent_total"] += 1.0
            if realized_return <= 0.0:
                aggregates[key]["recent_losses"] += 1.0
        for (branch_name, bucket), stats in aggregates.items():
            sample_size = float(stats["sample_size"])
            if sample_size <= 0.0:
                continue
            self._outcome_stats.setdefault(branch_name, {})[bucket] = {
                "sample_size": sample_size,
                "empirical_probability": float(stats["wins"]) / sample_size,
                "recent_failure_rate": float(stats["recent_losses"]) / max(float(stats["recent_total"]), 1.0),
            }

    def calibration_stats(self, branch_name: str, score: float) -> dict[str, float | str]:
        branch_name = _require_supported_branch(branch_name)
        bucket = _score_to_bucket(score)
        curve = self._curves.get(branch_name, {})
        preset_probability = float(curve.get(bucket, 0.50))
        empirical = dict(self._outcome_stats.get(branch_name, {}).get(bucket, {}) or {})
        sample_size = float(empirical.get("sample_size", 0.0) or 0.0)
        empirical_probability = float(empirical.get("empirical_probability", preset_probability) or preset_probability)
        recent_failure_rate = float(empirical.get("recent_failure_rate", 0.0) or 0.0)
        if sample_size <= 0.0:
            probability = preset_probability
            source = "preset"
        else:
            empirical_weight = min(sample_size / 12.0, 1.0)
            probability = preset_probability * (1.0 - empirical_weight) + empirical_probability * empirical_weight
            source = "empirical_blend"
        return {
            "bucket": bucket,
            "probability": float(probability),
            "preset_probability": preset_probability,
            "empirical_probability": empirical_probability,
            "sample_size": sample_size,
            "recent_failure_rate": recent_failure_rate,
            "source": source,
        }

    def calibrated_probability(self, branch_name: str, score: float) -> float:
        """Map a branch score to a calibrated empirical probability."""
        return float(self.calibration_stats(branch_name, score)["probability"])

    def record_outcome(
        self,
        *,
        symbol: str,
        branch_name: str,
        predicted_score: float,
        realized_return: float,
        run_date: str = "",
    ) -> None:
        """Persist a prediction-vs-outcome pair for future calibration."""
        branch_name = _require_supported_branch(branch_name)
        record = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "symbol": symbol,
            "branch": branch_name,
            "score": predicted_score,
            "bucket": _score_to_bucket(predicted_score),
            "realized_return": realized_return,
            "run_date": run_date,
        }
        outcomes_path = self._store_path.parent / "bayesian_outcomes.jsonl"
        try:
            outcomes_path.parent.mkdir(parents=True, exist_ok=True)
            with outcomes_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            _logger.warning("Failed to record outcome: %s", exc)

    def save(self) -> None:
        """Persist current calibration curves."""
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "schema_version": CALIBRATION_SCHEMA_VERSION,
                "curves": self._curves,
            }
            self._store_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            _logger.warning("Failed to save calibration store: %s", exc)
