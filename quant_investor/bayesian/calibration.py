"""Calibration store — persists prediction-vs-outcome pairs for future tuning.

V1: preset calibration curves with no active learning loop.
Future: active learning from realized outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quant_investor.logger import get_logger

_logger = get_logger("CalibrationStore")

# Preset calibration buckets: branch_name -> {score_bucket: empirical_probability}
# These are initial estimates; the learning loop will refine them over time.
_PRESET_CALIBRATION: dict[str, dict[str, float]] = {
    "kline": {
        "strong_negative": 0.25,
        "negative": 0.38,
        "neutral": 0.50,
        "positive": 0.62,
        "strong_positive": 0.75,
    },
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
    "intelligence": {
        "strong_negative": 0.30,
        "negative": 0.42,
        "neutral": 0.50,
        "positive": 0.58,
        "strong_positive": 0.68,
    },
}


def _score_to_bucket(score: float) -> str:
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
        self._store_path = Path(store_path or "data/bayesian_calibration.json")
        self._curves = dict(_PRESET_CALIBRATION)
        self._load()

    def _load(self) -> None:
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and "curves" in data:
                    self._curves.update(data["curves"])
            except Exception as exc:
                _logger.warning("Failed to load calibration store: %s", exc)

    def get_calibration_curve(self, branch_name: str) -> dict[str, float]:
        return dict(self._curves.get(branch_name, _PRESET_CALIBRATION.get(branch_name, {})))

    def calibrated_probability(self, branch_name: str, score: float) -> float:
        """Map a branch score to a calibrated empirical probability."""
        bucket = _score_to_bucket(score)
        curve = self._curves.get(branch_name, {})
        return curve.get(bucket, 0.50)

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
        record = {
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
            data = {"curves": self._curves, "version": "v1"}
            self._store_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            _logger.warning("Failed to save calibration store: %s", exc)
