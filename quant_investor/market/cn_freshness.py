from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CNFreshnessConfig:
    mode: str = "stable"
    coverage_threshold: float = 0.95
    early_stop_sample_size: int = 10
    early_stop_stale_ratio: float = 0.80


@dataclass(frozen=True)
class CNFreshnessTargets:
    strict_trade_date: str
    stable_trade_date: str
    effective_target_trade_date: str
    mode: str = "stable"
    strict_coverage_ratio: float = 0.0
    switched_to_stable: bool = False


def compute_coverage_ratio(*, complete_count: int, expected_count: int) -> float:
    if expected_count <= 0:
        return 0.0
    return max(0.0, min(float(complete_count) / float(expected_count), 1.0))


def resolve_cn_freshness_targets(
    *,
    strict_trade_date: str,
    stable_trade_date: str,
    strict_complete_count: int,
    expected_count: int,
    config: CNFreshnessConfig,
) -> CNFreshnessTargets:
    strict_ratio = compute_coverage_ratio(
        complete_count=strict_complete_count,
        expected_count=expected_count,
    )
    if config.mode == "strict":
        return CNFreshnessTargets(
            strict_trade_date=strict_trade_date,
            stable_trade_date=stable_trade_date,
            effective_target_trade_date=strict_trade_date,
            mode="strict",
            strict_coverage_ratio=strict_ratio,
            switched_to_stable=False,
        )
    use_strict = strict_ratio >= config.coverage_threshold
    return CNFreshnessTargets(
        strict_trade_date=strict_trade_date,
        stable_trade_date=stable_trade_date,
        effective_target_trade_date=strict_trade_date if use_strict else stable_trade_date,
        mode="stable",
        strict_coverage_ratio=strict_ratio,
        switched_to_stable=not use_strict,
    )


def should_abort_strict_same_day(
    *,
    observed_statuses: Iterable[str],
    config: CNFreshnessConfig,
) -> bool:
    normalized = [str(item or "").strip().lower() for item in observed_statuses if str(item or "").strip()]
    if len(normalized) < max(1, config.early_stop_sample_size):
        return False
    stale_cached = sum(1 for item in normalized if item == "stale_cached")
    stale_ratio = stale_cached / max(len(normalized), 1)
    return stale_ratio >= config.early_stop_stale_ratio
