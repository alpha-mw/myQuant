"""Runtime scoring adapter for governed mined factors.

Only production factors that passed all eight governance gates are allowed to
feed the quant branch. If the registry is empty or no factor is selectable,
the Quant branch is governance-blocked; callers must not manufacture a legacy
proxy score.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from quant_investor.factors.governance import FactorLifecycleState, FactorRecord

DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[1] / "factor_registry" / "mined_factors.json"
)
PRODUCTION_RUNTIME_MODE = "production"
REPORT_ONLY_SHADOW_RUNTIME_MODE = "report_only_shadow"


def production_factor_set_sha256(names: Sequence[str]) -> str:
    """Hash the sorted selectable factor-name set for metadata/readback checks."""

    normalized = sorted({str(name).strip() for name in names if str(name).strip()})
    raw = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class MinedFactorRegistry:
    schema_version: str = "mined-factor-registry.v1"
    factors: list[FactorRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MinedFactorRegistry":
        return cls(
            schema_version=str(payload.get("schema_version", "mined-factor-registry.v1")),
            factors=[
                FactorRecord.from_dict(item)
                for item in payload.get("factors", [])
                if isinstance(item, Mapping) and str(item.get("name", "")).strip()
            ],
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    @classmethod
    def from_records(cls, records: Sequence[FactorRecord]) -> "MinedFactorRegistry":
        return cls(factors=list(records))

    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "MinedFactorRegistry":
        raw_path = path or os.getenv("MYQUANT_FACTOR_REGISTRY") or DEFAULT_REGISTRY_PATH
        registry_path = Path(raw_path).expanduser()
        if not registry_path.exists():
            return cls(metadata={"path": str(registry_path), "missing": True})
        try:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            registry = cls.from_dict(payload if isinstance(payload, Mapping) else {})
            registry.metadata.setdefault("path", str(registry_path))
            return registry
        except Exception as exc:
            return cls(metadata={"path": str(registry_path), "load_error": str(exc)})

    def selectable_factors(self) -> list[FactorRecord]:
        return [factor for factor in self.factors if factor.selectable_in_quant_branch()]

    def selectable_manifest(self) -> dict[str, Any]:
        names = sorted(factor.name for factor in self.selectable_factors())
        return {
            "production_factor_count": len(names),
            "production_factor_names": names,
            "production_factor_set_sha256": production_factor_set_sha256(names),
        }

    def non_selectable_reasons(self) -> dict[str, str]:
        reasons: dict[str, str] = {}
        for factor in self.factors:
            if factor.selectable_in_quant_branch():
                continue
            if factor.state != FactorLifecycleState.PRODUCTION_FACTOR:
                reasons[factor.name] = f"state={factor.state.value}"
            elif not factor.all_gates_passed():
                reasons[factor.name] = "not_all_gates_passed"
            elif not float(factor.weight):
                reasons[factor.name] = "zero_weight"
            elif factor.deprecated_reason:
                reasons[factor.name] = f"deprecated={factor.deprecated_reason}"
            else:
                reasons[factor.name] = "not_selectable"
        return reasons


@dataclass
class RuntimeFactorScore:
    symbol_scores: dict[str, float] = field(default_factory=dict)
    factor_count: int = 0
    factors_used: list[str] = field(default_factory=list)
    factor_weights: dict[str, float] = field(default_factory=dict)
    factor_coverages: dict[str, float] = field(default_factory=dict)
    skipped_factors: dict[str, str] = field(default_factory=dict)
    registry_metadata: dict[str, Any] = field(default_factory=dict)
    governance_status: str = "governance_blocked"
    factor_mode: str = "governance_blocked"
    confidence_multiplier: float = 0.0
    production_eligible: bool = False
    runtime_mode: str = PRODUCTION_RUNTIME_MODE
    runtime_blockers: list[str] = field(default_factory=list)

    @property
    def coverage_rate(self) -> float:
        if self.factor_coverages:
            values = [
                max(0.0, min(1.0, float(value)))
                for value in self.factor_coverages.values()
            ]
            return float(sum(values) / max(len(values), 1))
        if not self.symbol_scores:
            return 0.0
        non_zero = sum(
            1
            for value in self.symbol_scores.values()
            if abs(float(value)) > 1e-12
        )
        return non_zero / max(len(self.symbol_scores), 1)

    def to_metadata(self) -> dict[str, Any]:
        applied_to_score = bool(self.factor_count > 0 and self.factors_used)
        return {
            "factor_count": self.factor_count,
            "factors_used": list(self.factors_used),
            "factor_weights": dict(self.factor_weights),
            "factor_coverages": dict(self.factor_coverages),
            "skipped_factors": dict(self.skipped_factors),
            "coverage_rate": self.coverage_rate,
            "applied_to_score": applied_to_score,
            "score_weight": float(
                sum(abs(float(weight)) for weight in self.factor_weights.values())
            )
            if applied_to_score
            else 0.0,
            "registry": dict(self.registry_metadata),
            "governance_status": self.governance_status,
            "factor_mode": self.factor_mode,
            "confidence_multiplier": float(self.confidence_multiplier),
            "production_eligible": bool(self.production_eligible),
            "runtime_mode": self.runtime_mode,
            "runtime_blockers": list(self.runtime_blockers),
            "legacy_fallback_allowed": False,
        }


def _factor_window_from_name(name: str, default: int = 20) -> int:
    try:
        suffix = str(name).strip().rsplit("_", 1)[1]
        return max(int(suffix.removesuffix("d")), 1)
    except Exception:
        return int(default)


def _factor_window_pair_from_name(
    name: str,
    *,
    default: tuple[int, int] = (20, 5),
) -> tuple[int, int]:
    parts = str(name).strip().split("_")
    try:
        first = int(parts[-2].removesuffix("d"))
        second = int(parts[-1].removesuffix("d"))
        return max(first, 1), max(second, 1)
    except Exception:
        return default


def _price_volume_factor_lookback_rows(name: str) -> int:
    factor_name = str(name or "").strip()
    if not factor_name:
        return 0
    if factor_name.startswith("pv_blend_volstab19x2_mom90_amihud5_w"):
        return 91
    if factor_name.startswith("pv_volume_stability_smooth_"):
        base_window, smooth_window = _factor_window_pair_from_name(factor_name)
        return base_window + smooth_window
    if factor_name.startswith("pv_dollar_volume_growth_"):
        short_window, long_window = _factor_window_pair_from_name(
            factor_name,
            default=(20, 60),
        )
        return max(short_window, long_window)
    if factor_name.startswith(
        (
            "pv_momentum_",
            "pv_short_reversal_",
            "pv_volatility_penalty_",
            "pv_downside_volatility_",
            "pv_price_efficiency_",
            "pv_amihud_illiquidity_",
        )
    ):
        return _factor_window_from_name(factor_name) + 1
    if factor_name.startswith(
        (
            "pv_volume_stability_",
            "pv_low_dollar_volume_",
            "pv_high_dollar_volume_",
        )
    ):
        return _factor_window_from_name(factor_name)
    return 0


def _price_volume_required_lookback_rows(names: Sequence[str]) -> int:
    return max(
        (_price_volume_factor_lookback_rows(name) for name in names),
        default=0,
    )


class MinedFactorScorer:
    """Compute latest cross-sectional scores from governed production factors."""

    def __init__(
        self,
        registry: MinedFactorRegistry | None = None,
        *,
        runtime_mode: str = PRODUCTION_RUNTIME_MODE,
    ) -> None:
        self.registry = registry or MinedFactorRegistry.load()
        normalized_mode = str(runtime_mode or "").strip()
        if normalized_mode not in {
            PRODUCTION_RUNTIME_MODE,
            REPORT_ONLY_SHADOW_RUNTIME_MODE,
        }:
            raise ValueError(f"unsupported factor runtime mode: {runtime_mode!r}")
        self.runtime_mode = normalized_mode

    def _runtime_contract(self) -> tuple[list[FactorRecord], dict[str, Any]]:
        if self.runtime_mode == REPORT_ONLY_SHADOW_RUNTIME_MODE:
            candidates = (
                self.registry.factors
                if self.registry.metadata.get("historical_shadow_only") is True
                else self.registry.selectable_factors()
            )
            active = [
                record
                for record in candidates
                if str(record.name).strip()
                and str(record.implementation).strip()
                and abs(float(record.weight)) > 1e-12
            ]
            blockers = [] if active else ["report_only_shadow_factor_set_empty"]
            return active, {
                "status": "report_only" if active else "governance_blocked",
                "factor_mode": (
                    "historical_shadow_report_only"
                    if active
                    else "governance_blocked"
                ),
                "confidence_multiplier": 0.0,
                "production_eligible": False,
                "legacy_fallback_allowed": False,
                "blockers": blockers,
            }

        # Local import avoids the module-load cycle: protocol v2 owns the
        # complete production readiness contract and imports this registry type.
        from quant_investor.factors.governance_protocol_v2 import (
            governance_runtime_status,
        )

        status = governance_runtime_status(self.registry)
        active = self.registry.selectable_factors() if status["status"] == "ready" else []
        return active, {
            **status,
            "production_eligible": status["status"] == "ready",
        }

    def _empty_score(
        self,
        symbols: Sequence[str],
        *,
        skipped: Mapping[str, str],
        runtime_status: Mapping[str, Any],
    ) -> RuntimeFactorScore:
        return RuntimeFactorScore(
            symbol_scores={str(symbol): 0.0 for symbol in symbols},
            skipped_factors=dict(skipped),
            registry_metadata={
                **dict(self.registry.metadata),
                "governance_runtime": dict(runtime_status),
            },
            governance_status=str(
                runtime_status.get("status") or "governance_blocked"
            ),
            factor_mode=str(
                runtime_status.get("factor_mode") or "governance_blocked"
            ),
            confidence_multiplier=float(
                runtime_status.get("confidence_multiplier") or 0.0
            ),
            production_eligible=bool(
                runtime_status.get("production_eligible", False)
            ),
            runtime_mode=self.runtime_mode,
            runtime_blockers=[
                str(item)
                for item in runtime_status.get("blockers", []) or []
                if str(item)
            ],
        )

    def score(self, frames: Mapping[str, pd.DataFrame]) -> RuntimeFactorScore:
        symbols = [str(symbol) for symbol in frames if str(symbol).strip()]
        active, runtime_status = self._runtime_contract()
        non_selectable = self.registry.non_selectable_reasons()
        active_names = {record.name for record in active}
        skipped = {
            name: reason
            for name, reason in non_selectable.items()
            if name not in active_names
        }
        if not symbols:
            return self._empty_score(
                [],
                skipped=skipped,
                runtime_status=runtime_status,
            )

        if not active:
            return self._empty_score(
                symbols,
                skipped=skipped,
                runtime_status=runtime_status,
            )

        weighted_scores = pd.Series(0.0, index=symbols, dtype=float)
        total_weight = 0.0
        factors_used: list[str] = []
        factor_weights: dict[str, float] = {}
        factor_coverages: dict[str, float] = {}
        price_volume_prepared: Mapping[str, Any] | None = None
        price_volume_factor_cache: dict[str, Any] = {}
        price_volume_names = [
            str(factor.implementation or "").strip().split(":", 1)[1]
            for factor in active
            if str(factor.implementation or "").strip().startswith("price_volume:")
        ]
        price_volume_factor_cache["active_price_volume_names"] = tuple(price_volume_names)
        price_volume_lookback_rows = _price_volume_required_lookback_rows(
            price_volume_names
        )
        include_amihud_base = any(
            name.startswith("pv_amihud_illiquidity_")
            or name.startswith("pv_blend_volstab19x2_mom90_amihud5_w")
            for name in price_volume_names
        )

        for factor in active:
            try:
                impl = str(factor.implementation or "").strip()
                if impl.startswith("price_volume:"):
                    if price_volume_prepared is None:
                        from quant_investor.factors.price_volume import (
                            prepare_price_volume_frames,
                        )

                        price_volume_prepared = prepare_price_volume_frames(
                            frames,
                            include_amihud_base=include_amihud_base,
                            lookback_rows=price_volume_lookback_rows,
                        )
                    raw = self._price_volume_factor(
                        impl.split(":", 1)[1],
                        frames,
                        prepared_frames=price_volume_prepared,
                        factor_cache=price_volume_factor_cache,
                    )
                else:
                    raw = self._compute_factor(factor, frames)
            except Exception as exc:
                skipped[factor.name] = f"compute_error={exc}"
                continue
            valid = raw.replace([np.inf, -np.inf], np.nan).dropna()
            if valid.empty:
                skipped[factor.name] = "empty_factor_values"
                continue
            normalized = self._rank_normalize(raw.reindex(symbols))
            weight = float(factor.weight) * (1.0 if float(factor.direction) >= 0 else -1.0)
            if abs(weight) <= 1e-12:
                skipped[factor.name] = "zero_effective_weight"
                continue
            weighted_scores = weighted_scores.add(normalized.fillna(0.0) * weight, fill_value=0.0)
            total_weight += abs(weight)
            factors_used.append(factor.name)
            factor_weights[factor.name] = weight
            factor_coverages[factor.name] = float(
                valid.index.intersection(symbols).size / max(len(symbols), 1)
            )

        if total_weight <= 1e-12 or not factors_used:
            runtime_status = {
                **runtime_status,
                "status": "governance_blocked",
                "factor_mode": "governance_blocked",
                "confidence_multiplier": 0.0,
                "production_eligible": False,
                "blockers": [
                    *list(runtime_status.get("blockers", []) or []),
                    "no_runtime_factor_completed",
                ],
            }
            return self._empty_score(
                symbols,
                skipped=skipped,
                runtime_status=runtime_status,
            )

        symbol_scores = (weighted_scores / total_weight).clip(-1.0, 1.0).fillna(0.0)
        return RuntimeFactorScore(
            symbol_scores={symbol: float(symbol_scores.get(symbol, 0.0)) for symbol in symbols},
            factor_count=len(factors_used),
            factors_used=factors_used,
            factor_weights=factor_weights,
            factor_coverages=factor_coverages,
            skipped_factors=skipped,
            registry_metadata={
                **dict(self.registry.metadata),
                "governance_runtime": dict(runtime_status),
            },
            governance_status=str(runtime_status["status"]),
            factor_mode=str(runtime_status["factor_mode"]),
            confidence_multiplier=float(
                runtime_status.get("confidence_multiplier") or 0.0
            ),
            production_eligible=bool(
                runtime_status.get("production_eligible", False)
            ),
            runtime_mode=self.runtime_mode,
            runtime_blockers=[
                str(item)
                for item in runtime_status.get("blockers", []) or []
                if str(item)
            ],
        )

    def _compute_factor(
        self,
        factor: FactorRecord,
        frames: Mapping[str, pd.DataFrame],
    ) -> pd.Series:
        impl = str(factor.implementation or "").strip()
        if impl == "alpha158.FactorEngineer.cross_sectional_score":
            return self._alpha158_cross_sectional(frames)
        if impl.startswith("alpha_mining.FactorLibrary:"):
            return self._alpha_mining_factor(impl.split(":", 1)[1], frames)
        if impl.startswith("price_volume:"):
            return self._price_volume_factor(impl.split(":", 1)[1], frames)
        if impl.startswith("aquant_expression:"):
            return self._aquant_expression_factor(factor, impl.split(":", 1)[1], frames)
        if impl.startswith("builtin:"):
            return self._builtin_factor(impl.split(":", 1)[1], frames)
        # Backward-compatible convention: a registry factor named like a
        # FactorLibrary method can be used without a verbose implementation path.
        return self._alpha_mining_factor(factor.name, frames)

    @staticmethod
    def _alpha158_cross_sectional(frames: Mapping[str, pd.DataFrame]) -> pd.Series:
        from quant_investor.alpha158 import FactorEngineer

        engineer = FactorEngineer()
        scores = engineer.cross_sectional_score(dict(frames))
        return pd.Series(scores, dtype=float)

    def _alpha_mining_factor(self, name: str, frames: Mapping[str, pd.DataFrame]) -> pd.Series:
        from quant_investor.alpha_mining import FactorLibrary

        funcs = FactorLibrary.all_factor_funcs()
        func = funcs.get(str(name).strip())
        if func is None:
            raise ValueError(f"unknown FactorLibrary factor: {name}")
        combined = self._combined_frame(frames)
        if combined.empty:
            return pd.Series(dtype=float)
        values = func(combined)
        return self._latest_by_symbol(combined, values)

    @staticmethod
    def _price_volume_factor(
        name: str,
        frames: Mapping[str, pd.DataFrame],
        *,
        prepared_frames: Mapping[str, Any] | None = None,
        factor_cache: dict[str, Any] | None = None,
    ) -> pd.Series:
        from quant_investor.factors.price_volume import compute_price_volume_factor

        return compute_price_volume_factor(
            name,
            frames,
            prepared_frames=prepared_frames,
            factor_cache=factor_cache,
        )

    @staticmethod
    def _aquant_expression_factor(
        factor: FactorRecord,
        name: str,
        frames: Mapping[str, pd.DataFrame],
    ) -> pd.Series:
        from quant_investor.factors.aquant_expression import compute_aquant_expression_factor

        expression = str(factor.metadata.get("expression", "") or "").strip()
        metadata_dir = factor.metadata.get("metadata_dir")
        pit_series_path = factor.metadata.get("pit_series_path")
        fundamental_mart_root = factor.metadata.get("fundamental_mart_root")
        allow_legacy_fundamental_fallback = factor.metadata.get("allow_legacy_fundamental_fallback")
        return compute_aquant_expression_factor(
            str(name or factor.name),
            frames,
            expression=expression,
            metadata_dir=metadata_dir,
            pit_series_path=pit_series_path,
            fundamental_mart_root=fundamental_mart_root,
            allow_legacy_fundamental_fallback=allow_legacy_fundamental_fallback,
        )

    @staticmethod
    def _builtin_factor(name: str, frames: Mapping[str, pd.DataFrame]) -> pd.Series:
        values: dict[str, float] = {}
        for symbol, frame in frames.items():
            close = _close_series(frame)
            if close.empty:
                values[str(symbol)] = np.nan
                continue
            if name == "short_term_return":
                values[str(symbol)] = _window_return(close, 20)
            elif name == "volatility_penalty":
                returns = close.pct_change().dropna()
                values[str(symbol)] = (
                    -float(returns.tail(60).std()) if len(returns) >= 3 else np.nan
                )
            else:
                raise ValueError(f"unknown builtin factor: {name}")
        return pd.Series(values, dtype=float)

    @staticmethod
    def _combined_frame(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        for symbol, frame in frames.items():
            if frame is None or frame.empty:
                continue
            working = frame.copy()
            if "symbol" not in working.columns:
                working["symbol"] = str(symbol)
            if "date" not in working.columns:
                working["date"] = working.index
            chunks.append(working)
        if not chunks:
            return pd.DataFrame()
        combined = pd.concat(chunks, ignore_index=True)
        if "date" in combined.columns:
            combined = combined.sort_values(["date", "symbol"]).reset_index(drop=True)
        return combined

    @staticmethod
    def _latest_by_symbol(combined: pd.DataFrame, values: pd.Series) -> pd.Series:
        working = combined[["symbol"]].copy()
        working["__factor__"] = pd.to_numeric(values.reindex(combined.index), errors="coerce")
        latest: dict[str, float] = {}
        for symbol, group in working.groupby("symbol", sort=False):
            series = group["__factor__"].replace([np.inf, -np.inf], np.nan).dropna()
            latest[str(symbol)] = float(series.iloc[-1]) if not series.empty else np.nan
        return pd.Series(latest, dtype=float)

    @staticmethod
    def _rank_normalize(values: pd.Series) -> pd.Series:
        clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = clean.dropna()
        result = pd.Series(0.0, index=clean.index, dtype=float)
        if len(valid) <= 1:
            return result
        ranks = pd.Series(stats.rankdata(valid, method="average"), index=valid.index, dtype=float)
        normalized = ((ranks - ranks.mean()) / (ranks.std(ddof=0) + 1e-9)).clip(-3.0, 3.0) / 3.0
        result.loc[normalized.index] = normalized
        return result.clip(-1.0, 1.0)


def score_with_mined_factors(
    frames: Mapping[str, pd.DataFrame],
    registry: MinedFactorRegistry | None = None,
    *,
    runtime_mode: str = PRODUCTION_RUNTIME_MODE,
) -> RuntimeFactorScore:
    return MinedFactorScorer(
        registry=registry,
        runtime_mode=runtime_mode,
    ).score(frames)


def _close_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    close_col = "close" if "close" in frame.columns else "Close" if "Close" in frame.columns else ""
    if not close_col:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[close_col], errors="coerce").dropna()


def _window_return(close: pd.Series, window: int) -> float:
    if window <= 0 or len(close) <= window:
        return 0.0
    base = float(close.iloc[-window - 1])
    latest = float(close.iloc[-1])
    if abs(base) <= 1e-8:
        return 0.0
    return (latest / base) - 1.0


__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "MinedFactorRegistry",
    "MinedFactorScorer",
    "PRODUCTION_RUNTIME_MODE",
    "REPORT_ONLY_SHADOW_RUNTIME_MODE",
    "RuntimeFactorScore",
    "production_factor_set_sha256",
    "score_with_mined_factors",
]
