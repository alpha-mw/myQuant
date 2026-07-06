from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_investor.themes.storage import ThemeSnapshotStore
from quant_investor.themes.types import ThemePhase


_TIGHTEN_FLAGS = {"theme_distribution_risk"}
_WATCH_FLAGS = {
    "theme_overextended",
    "theme_overextended_no_chase",
    "theme_fake_breakout_risk",
}


@dataclass(frozen=True)
class ThemeHoldingSignal:
    symbol: str
    primary_theme_id: str = ""
    primary_theme_name: str = ""
    phase: str = ""
    risk_flags: list[str] = field(default_factory=list)
    guard_level: str = "none"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "primary_theme_id": self.primary_theme_id,
            "primary_theme_name": self.primary_theme_name,
            "phase": self.phase,
            "risk_flags": list(self.risk_flags),
            "guard_level": self.guard_level,
            "reasons": list(self.reasons),
        }


def load_latest_holding_theme_payload(
    *,
    market: str = "CN",
    universe_key: str = "full_a",
    as_of: str | None = None,
    root_dir: str | Path = "results/theme_snapshots",
) -> dict[str, Any]:
    latest = ThemeSnapshotStore(root_dir).load_latest_with_path(
        market=market,
        universe_key=universe_key,
        as_of=as_of,
    )
    if latest is None:
        return {}
    return dict(latest[1])


def evaluate_holding_theme_guard(
    holding_symbols: Sequence[str],
    theme_payload: Mapping[str, Any] | None,
) -> dict[str, ThemeHoldingSignal]:
    symbols = [_normalize_symbol(symbol) for symbol in holding_symbols if _normalize_symbol(symbol)]
    rotation = _theme_rotation_payload(theme_payload)
    if not rotation:
        return {
            symbol: ThemeHoldingSignal(
                symbol=symbol,
                guard_level="none",
                reasons=["theme_snapshot_unavailable"],
            )
            for symbol in symbols
        }

    primary_by_symbol = _mapping(rotation.get("symbol_primary_theme"))
    phase_by_symbol = _mapping(rotation.get("symbol_phase"))
    flags_by_symbol = _mapping(rotation.get("symbol_risk_flags"))
    theme_scores = _mapping(rotation.get("theme_scores"))

    signals: dict[str, ThemeHoldingSignal] = {}
    for symbol in symbols:
        theme_id = str(primary_by_symbol.get(symbol) or "").strip()
        flags = _risk_flags(flags_by_symbol.get(symbol))
        if not theme_id:
            signals[symbol] = ThemeHoldingSignal(
                symbol=symbol,
                risk_flags=flags,
                guard_level="none",
                reasons=["no_theme_mapping"],
            )
            continue

        theme_score = _mapping(theme_scores.get(theme_id))
        phase = _normalize_phase(phase_by_symbol.get(symbol) or theme_score.get("phase"))
        theme_name = str(
            theme_score.get("theme_name")
            or theme_score.get("name")
            or theme_score.get("theme_id")
            or theme_id
        ).strip()
        guard_level, reasons = _classify_guard(phase=phase, risk_flags=flags)
        signals[symbol] = ThemeHoldingSignal(
            symbol=symbol,
            primary_theme_id=theme_id,
            primary_theme_name=theme_name,
            phase=phase,
            risk_flags=flags,
            guard_level=guard_level,
            reasons=reasons,
        )
    return signals


def _theme_rotation_payload(theme_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _mapping(theme_payload)
    rotation = _mapping(payload.get("theme_rotation"))
    if rotation:
        return rotation
    if any(key in payload for key in ("symbol_primary_theme", "symbol_phase", "symbol_risk_flags")):
        return payload
    return {}


def _classify_guard(*, phase: str, risk_flags: list[str]) -> tuple[str, list[str]]:
    lowered_flags = {str(flag).strip().lower() for flag in risk_flags if str(flag).strip()}
    if phase == ThemePhase.DISTRIBUTION.value:
        return "tighten", ["phase_distribution"]
    distribution_flags = sorted(lowered_flags & _TIGHTEN_FLAGS)
    if distribution_flags:
        return "tighten", [f"flag_{flag}" for flag in distribution_flags]

    if phase == ThemePhase.OVEREXTENDED.value:
        return "watch", ["phase_overextended"]
    watch_flags = sorted(lowered_flags & _WATCH_FLAGS)
    if watch_flags:
        return "watch", [f"flag_{flag}" for flag in watch_flags]

    if not phase or phase == ThemePhase.UNCLASSIFIED.value:
        return "none", ["phase_unclassified"]
    return "none", [f"phase_{phase}"]


def _risk_flags(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or value == "":
        return []
    return [str(value).strip()]


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _normalize_phase(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    valid = {phase.value for phase in ThemePhase}
    return text if text in valid else text


def _normalize_symbol(value: Any) -> str:
    return str(value or "").strip().upper()
