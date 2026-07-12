from __future__ import annotations

import math
import re
from collections import defaultdict
from statistics import mean, median
from typing import Any, Mapping, Sequence

import pandas as pd

from quant_investor.themes.membership import active_memberships_by_symbol
from quant_investor.themes.policy import PolicyCatalystScanner
from quant_investor.themes.smoothing import (
    ThemeSmoothingConfig,
    smooth_numeric_series,
    smooth_theme_series,
)
from quant_investor.themes.types import ThemePhase, ThemeScanResult, ThemeScore, clamp


_INVALID_THEME_NAMES = {"", "unknown", "none", "nan", "null"}
_DATE_COLUMNS = ("trade_date", "date", "Date", "datetime", "time")
_CLOSE_COLUMNS = ("close", "Close")
_HIGH_COLUMNS = ("high", "High")
_VOLUME_COLUMNS = ("volume", "vol")
_AMOUNT_COLUMNS = ("amount", "Amount", "turnover")
_PCT_CHG_COLUMNS = ("pct_chg", "pct_change", "pctChange", "change_pct")
_NAME_COLUMNS = ("name", "stock_name", "sec_name", "证券简称")
_ST_COLUMNS = ("is_st", "st", "is_ST", "risk_warning")
_ST_LIMIT_CHANGE_DATE = "20260706"
_THEME_CROWDING_WEIGHTS: dict[str, float] = {
    "turnover_share_stretch": 0.45,
    "limitup_norm": 0.35,
    "member_turnover_concentration": 0.20,
}


class ThemeScanner:
    def scan(
        self,
        *,
        frames: Mapping[str, pd.DataFrame],
        industry_map: Mapping[str, str],
        symbol_market_state: Mapping[str, Mapping[str, Any]] | None = None,
        market: str = "CN",
        universe_key: str = "",
        as_of: str = "",
        min_member_count: int = 5,
        top_n: int = 20,
        smoothing_window: int = 10,
        smoothing_min_observations: int = 5,
        policy_catalyst_enabled: bool | None = None,
        policy_catalyst_weight: float | None = None,
        policy_lookback_days: int | None = None,
        policy_event_path: str | None = None,
        crowding_enabled: bool | None = None,
        crowding_min_universe: int | None = None,
        snapshot_history: Sequence[Mapping[str, Any]] | None = None,
        concept_membership_enabled: bool | None = None,
        concept_primary_margin: float | None = None,
        theme_memberships: Sequence[Mapping[str, Any]] | None = None,
        membership_v2_enabled: bool = False,
        theme_memberships_v2: Sequence[Mapping[str, Any]] | None = None,
    ) -> ThemeScanResult:
        state_map = symbol_market_state or {}
        min_count = max(0, int(min_member_count))
        limit = max(0, int(top_n))
        policy_config = _resolve_policy_config(
            policy_catalyst_enabled=policy_catalyst_enabled,
            policy_catalyst_weight=policy_catalyst_weight,
            policy_lookback_days=policy_lookback_days,
            policy_event_path=policy_event_path,
        )
        crowding_config = _resolve_crowding_config(
            crowding_enabled=crowding_enabled,
            crowding_min_universe=crowding_min_universe,
        )
        concept_config = _resolve_concept_config(
            concept_membership_enabled=concept_membership_enabled,
            concept_primary_margin=concept_primary_margin,
        )
        smoothing_config = ThemeSmoothingConfig(
            window=max(int(smoothing_window or 10), 1),
            min_observations=max(int(smoothing_min_observations or 5), 1),
        )
        themes: dict[str, dict[str, Any]] = {}
        members_by_theme: dict[str, list[dict[str, Any]]] = defaultdict(list)
        history_members_by_theme: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        theme_ids_by_symbol: dict[str, list[str]] = {}
        membership_details_by_symbol: dict[str, list[dict[str, Any]]] = {}
        scanned_symbol_count = 0
        universe_members: list[dict[str, Any]] = []
        concept_diagnostic_notes: list[str] = []
        active_concept_memberships = (
            active_memberships_by_symbol(theme_memberships or (), as_of=as_of)
            if concept_config["enabled"]
            else {}
        )
        active_membership_v2 = (
            active_memberships_by_symbol(theme_memberships_v2 or (), as_of=as_of)
            if membership_v2_enabled
            else {}
        )
        concept_membership_count = sum(
            len(memberships) for memberships in active_concept_memberships.values()
        )
        concept_status = "disabled"
        if concept_config["enabled"]:
            concept_status = "success" if concept_membership_count else "empty"
            if not concept_membership_count:
                concept_diagnostic_notes.append("theme_membership_no_active_members")
        membership_v2_count = sum(
            len(memberships) for memberships in active_membership_v2.values()
        )
        membership_v2_status = "disabled"
        if membership_v2_enabled:
            membership_v2_status = "success" if membership_v2_count else "empty"
        all_symbols = set(str(symbol) for symbol in dict(industry_map or {}))
        all_symbols.update(active_concept_memberships)
        all_symbols.update(active_membership_v2)

        for symbol in sorted(all_symbols):
            state = state_map.get(symbol, {})
            frame = frames.get(symbol)
            try:
                metrics = _symbol_metrics(frame, state, symbol=symbol)
            except Exception:
                metrics = _neutral_symbol_metrics()
            metrics["symbol"] = symbol
            assigned_theme_ids: list[str] = []
            assigned_membership_details: list[dict[str, Any]] = []

            theme_name = _valid_theme_name(industry_map.get(symbol))
            if theme_name is not None:
                theme_id = f"industry::{_normalize_theme_id(theme_name)}"
                themes.setdefault(
                    theme_id,
                    {
                        "theme_id": theme_id,
                        "theme_name": theme_name,
                        "theme_type": "industry",
                        "membership_source": "industry_map",
                        "pit_membership": False,
                    },
                )
                industry_metrics = {
                    **dict(metrics),
                    "theme_type": "industry",
                    "membership_source": "industry_map",
                    "pit_membership": False,
                }
                members_by_theme[theme_id].append(industry_metrics)
                assigned_theme_ids.append(theme_id)
                assigned_membership_details.append(
                    {
                        "schema_version": "industry_map.v1",
                        "theme_id": theme_id,
                        "theme_name": theme_name,
                        "theme_type": "industry",
                        "symbol": symbol,
                        "supply_chain_role": "unknown",
                        "revenue_exposure": None,
                        "pit_membership": False,
                    }
                )

            canonical_theme_ids: set[str] = set()
            for membership in active_membership_v2.get(symbol, []):
                canonical_theme_id = str(membership.theme_id or "").strip()
                if not canonical_theme_id:
                    continue
                canonical_theme_ids.add(canonical_theme_id)
                canonical_theme_name = str(
                    membership.theme_name or canonical_theme_id
                )
                themes.setdefault(
                    canonical_theme_id,
                    {
                        "theme_id": canonical_theme_id,
                        "theme_name": canonical_theme_name,
                        "theme_type": str(
                            membership.theme_type or "technology"
                        ),
                        "membership_source": "canonical_theme_membership.v2",
                        "pit_membership": True,
                    },
                )
                canonical_metrics = {
                    **dict(metrics),
                    "theme_type": str(membership.theme_type or "technology"),
                    "membership_source": "canonical_theme_membership.v2",
                    "pit_membership": True,
                    "membership_id": str(membership.membership_id or ""),
                    "membership_confidence": clamp(membership.confidence),
                }
                members_by_theme[canonical_theme_id].append(canonical_metrics)
                assigned_theme_ids.append(canonical_theme_id)
                assigned_membership_details.append(
                    {
                        **membership.to_dict(),
                        "pit_membership": True,
                        "canonical_membership_v2": True,
                    }
                )

            for membership in active_concept_memberships.get(symbol, []):
                concept_theme_id = str(membership.theme_id or "").strip()
                if not concept_theme_id or concept_theme_id in canonical_theme_ids:
                    continue
                concept_theme_name = str(membership.theme_name or concept_theme_id)
                themes.setdefault(
                    concept_theme_id,
                    {
                        "theme_id": concept_theme_id,
                        "theme_name": concept_theme_name,
                        "theme_type": str(membership.theme_type or "concept"),
                        "membership_source": "theme_membership.v2",
                        "pit_membership": True,
                    },
                )
                concept_metrics = {
                    **dict(metrics),
                    "theme_type": str(membership.theme_type or "concept"),
                    "membership_source": "theme_membership.v2",
                    "pit_membership": True,
                    "membership_id": str(membership.membership_id or ""),
                    "membership_confidence": clamp(membership.confidence),
                }
                members_by_theme[concept_theme_id].append(concept_metrics)
                assigned_theme_ids.append(concept_theme_id)
                assigned_membership_details.append(
                    {
                        **membership.to_dict(),
                        "pit_membership": True,
                        "canonical_membership_v2": False,
                    }
                )

            if not assigned_theme_ids:
                continue
            assigned_theme_ids = list(dict.fromkeys(assigned_theme_ids))
            theme_ids_by_symbol[symbol] = list(assigned_theme_ids)
            membership_details_by_symbol[symbol] = assigned_membership_details
            universe_members.append(dict(metrics))
            for offset in range(smoothing_config.window):
                if offset == 0:
                    historical_metrics = dict(metrics)
                else:
                    historical_metrics = _symbol_metrics_at_offset(
                        frame,
                        state,
                        offset,
                        symbol=symbol,
                    )
                    historical_metrics["symbol"] = symbol
                for assigned_theme_id in assigned_theme_ids:
                    history_members_by_theme[assigned_theme_id][offset].append(
                        dict(historical_metrics)
                    )
            scanned_symbol_count += 1

        universe_amount = sum(
            max(0.0, _safe_float(member.get("latest_amount"), 0.0))
            for member in universe_members
        )
        scored: list[ThemeScore] = []
        for theme_id in sorted(members_by_theme):
            members = members_by_theme[theme_id]
            if len(members) < min_count:
                continue
            scored.append(
                _score_theme(
                    theme_id=theme_id,
                    theme_name=themes.get(theme_id, {}).get("theme_name", theme_id),
                    theme_type=themes.get(theme_id, {}).get("theme_type", "industry"),
                    membership_source=themes.get(theme_id, {}).get(
                        "membership_source",
                        "industry_map",
                    ),
                    pit_membership=bool(themes.get(theme_id, {}).get("pit_membership", False)),
                    members=members,
                    min_member_count=min_count,
                    crowding_enabled=crowding_config["enabled"],
                    crowding_min_universe=crowding_config["min_universe"],
                    scanned_symbol_count=scanned_symbol_count,
                    universe_amount=universe_amount,
                    snapshot_history=snapshot_history or (),
                    smoothing_config=smoothing_config,
                )
            )

        policy_metadata = _apply_policy_catalysts(
            theme_scores=scored,
            members_by_theme=members_by_theme,
            as_of=as_of,
            enabled=policy_config["enabled"],
            weight=policy_config["weight"],
            lookback_days=policy_config["lookback_days"],
            event_path=policy_config["event_path"],
        )
        for theme_score in scored:
            _apply_smoothing_to_theme_score(
                theme_score=theme_score,
                history_members_by_offset=history_members_by_theme.get(theme_score.theme_id, {}),
                min_member_count=min_count,
                config=smoothing_config,
            )
        selected = sorted(
            scored,
            key=lambda item: (
                -float(item.effective_score if item.effective_score is not None else item.score),
                item.theme_id,
            ),
        )[:limit]
        theme_scores = {score.theme_id: score for score in selected}

        symbol_scores: dict[str, float] = {}
        symbol_smoothed_scores: dict[str, float] = {}
        symbol_primary_theme: dict[str, str] = {}
        symbol_theme_memberships: dict[str, list[str]] = {}
        symbol_theme_membership_details: dict[str, list[dict[str, Any]]] = {}
        symbol_phase: dict[str, str] = {}
        symbol_risk_flags: dict[str, list[str]] = {}
        for symbol in sorted(theme_ids_by_symbol):
            memberships = [
                theme_id
                for theme_id in theme_ids_by_symbol.get(symbol, [])
                if theme_id in theme_scores
            ]
            if not memberships:
                continue
            primary = _choose_primary_theme(
                memberships=memberships,
                theme_scores=theme_scores,
                concept_primary_margin=concept_config["primary_margin"],
            )
            if primary is None:
                continue
            symbol_theme_memberships[symbol] = list(memberships)
            symbol_theme_membership_details[symbol] = [
                dict(detail)
                for detail in membership_details_by_symbol.get(symbol, [])
                if str(detail.get("theme_id") or "") in memberships
            ]
            # Preserve raw and smoothed symbol maps as separate contracts.
            # Formal theme gating consumes ThemeScore.effective_score instead.
            symbol_scores[symbol] = clamp(primary.score / 100.0)
            if primary.smoothed_score is not None:
                symbol_smoothed_scores[symbol] = clamp(primary.smoothed_score / 100.0)
            symbol_primary_theme[symbol] = primary.theme_id
            symbol_phase[symbol] = primary.phase.value
            symbol_risk_flags[symbol] = sorted(
                {
                    flag
                    for theme_id in memberships
                    for flag in theme_scores[theme_id].risk_flags
                }
            )

        return ThemeScanResult(
            market=market,
            universe_key=universe_key,
            as_of=as_of,
            theme_scores=theme_scores,
            symbol_scores=symbol_scores,
            symbol_smoothed_scores=symbol_smoothed_scores,
            symbol_primary_theme=symbol_primary_theme,
            symbol_theme_memberships=symbol_theme_memberships,
            symbol_theme_membership_details=symbol_theme_membership_details,
            symbol_phase=symbol_phase,
            symbol_risk_flags=symbol_risk_flags,
            metadata={
                "theme_count": len(theme_scores),
                "scanned_symbol_count": scanned_symbol_count,
                "member_count_min": min_count,
                "top_n": limit,
                "smoothing_method": "sma",
                "smoothing_window": smoothing_config.window,
                "smoothing_min_observations": smoothing_config.min_observations,
                "policy_catalyst_status": policy_metadata["status"],
                "policy_catalyst_enabled": policy_metadata["enabled"],
                "policy_catalyst_weight": policy_metadata["weight"],
                "policy_catalyst_event_path": policy_metadata["event_path"],
                "policy_catalyst_matched_theme_count": policy_metadata["matched_theme_count"],
                "policy_catalyst_diagnostic_notes": policy_metadata["diagnostic_notes"],
                "crowding_enabled": crowding_config["enabled"],
                "crowding_min_universe": crowding_config["min_universe"],
                "crowding_universe_amount": universe_amount,
                "crowding_weight_model": dict(_THEME_CROWDING_WEIGHTS),
                "crowding_diagnostic_notes": _crowding_result_notes(scored),
                "concept_membership_enabled": concept_config["enabled"],
                "concept_membership_status": concept_status,
                "concept_membership_count": concept_membership_count,
                "concept_primary_margin": concept_config["primary_margin"],
                "concept_membership_diagnostic_notes": concept_diagnostic_notes,
                "membership_v2_enabled": bool(membership_v2_enabled),
                "membership_v2_status": membership_v2_status,
                "membership_v2_count": membership_v2_count,
                "deterministic": True,
                "no_llm": True,
                "no_network": True,
            },
        )


def _valid_theme_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _INVALID_THEME_NAMES:
        return None
    return text


def _normalize_theme_id(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^0-9a-z_\-\u4e00-\u9fff]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "unclassified"


def _symbol_metrics(
    frame: pd.DataFrame | None,
    state: Mapping[str, Any] | None,
    *,
    symbol: str = "",
) -> dict[str, Any]:
    metrics = _neutral_symbol_metrics()
    state = state if isinstance(state, Mapping) else {}
    market_data = _ordered_market_data(frame)
    close = market_data["close"]
    high = market_data["high"] or close
    volume = market_data["volume"]

    if close:
        metrics["return_3d"] = _window_return(close, 3)
        metrics["return_5d"] = _window_return(close, 5)
        metrics["return_20d"] = _window_return(close, 20)
        metrics["return_60d"] = _window_return(close, 60)
        metrics["return_120d"] = _window_return(close, 120)
        metrics["attention_return_5d"] = _window_return_optional(close, 5)
        metrics["attention_return_20d"] = _window_return_optional(close, 20)
        metrics["attention_return_60d"] = _window_return_optional(close, 60)
        metrics["attention_return_120d"] = _window_return_optional(close, 120)
        metrics["has_ma20"] = len(close) >= 20
        metrics["above_ma20"] = bool(len(close) >= 20 and close[-1] > _mean(close[-20:]))
        metrics["data_coverage"] = clamp(len(close) / 21.0)
        metrics["attention_history_coverage"] = clamp(len(close) / 121.0)
        metrics["fake_breakout_proxy"] = _fake_breakout_proxy(close)
    else:
        metrics["return_3d"] = _state_float(state, "return_3d", 0.0)
        metrics["return_5d"] = _state_float(state, "return_5d", 0.0)
        metrics["return_20d"] = _state_float(state, "return_20d", 0.0)
        metrics["return_60d"] = _state_float(state, "return_60d", 0.0)
        metrics["return_120d"] = _state_float(state, "return_120d", 0.0)
        metrics["attention_return_5d"] = _state_optional_float(state, "return_5d")
        metrics["attention_return_20d"] = _state_optional_float(state, "return_20d")
        metrics["attention_return_60d"] = _state_optional_float(state, "return_60d")
        metrics["attention_return_120d"] = _state_optional_float(state, "return_120d")

    if volume:
        tail = volume[-20:]
        average_volume = _mean(tail)
        if average_volume > 0:
            ratio = volume[-1] / average_volume
            metrics["volume_ratio"] = ratio
            metrics["symbol_volume_confirmation"] = clamp((ratio - 1.0) / 1.5)

    if close:
        metrics["latest_close"] = close[-1]
    if high:
        metrics["latest_high"] = high[-1]
    if close and len(high) >= 120:
        metrics["new_high_120d"] = bool(
            close[-1] >= max(high[-120:]) * (1.0 - 0.001)
        )
    if volume:
        metrics["latest_volume"] = volume[-1]
    amount, approximated = _latest_amount(close=close, volume=volume, amount=market_data["amount"])
    metrics["latest_amount"] = amount
    metrics["amount_approximated"] = approximated
    metrics["pct_chg_derived"] = bool(market_data["pct_chg_derived"])
    pct_chg = market_data["pct_chg"]
    normalized_pct_chg = _pct_chg_values_for_unit(
        pct_chg,
        unit=str(market_data.get("pct_chg_unit") or "auto"),
    )
    if normalized_pct_chg:
        metrics["latest_pct_chg"] = normalized_pct_chg[-1]
    metrics["limitup_hit"] = _is_limitup_latest(
        symbol,
        close=close,
        high=high,
        pct_chg=pct_chg,
        pct_chg_unit=str(market_data.get("pct_chg_unit") or "auto"),
        trade_date=market_data.get("latest_trade_date"),
        name=market_data.get("latest_name"),
        is_st=market_data.get("latest_is_st"),
    )

    state_fake_risk = _state_float(state, "fake_breakout_risk", math.nan)
    metrics["fake_breakout_risk"] = (
        clamp(state_fake_risk)
        if math.isfinite(state_fake_risk)
        else metrics["fake_breakout_proxy"]
    )
    return metrics


def _symbol_metrics_at_offset(
    frame: pd.DataFrame | None,
    state: Mapping[str, Any] | None,
    trailing_offset: int,
    *,
    symbol: str = "",
) -> dict[str, Any]:
    if trailing_offset <= 0:
        return _symbol_metrics(frame, state, symbol=symbol)
    prefix = _frame_prefix(frame, trailing_offset)
    if prefix is None or prefix.empty:
        return _neutral_symbol_metrics()
    return _symbol_metrics(prefix, {}, symbol=symbol)


def _frame_prefix(frame: pd.DataFrame | None, trailing_offset: int) -> pd.DataFrame | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    ordered = frame
    for date_col in _DATE_COLUMNS:
        if date_col in frame.columns:
            try:
                ordered = frame.sort_values(date_col, kind="mergesort")
            except Exception:
                ordered = frame
            break
    keep_count = len(ordered) - max(int(trailing_offset), 0)
    if keep_count <= 0:
        return None
    return ordered.iloc[:keep_count]


def _neutral_symbol_metrics() -> dict[str, Any]:
    return {
        "return_3d": 0.0,
        "return_5d": 0.0,
        "return_20d": 0.0,
        "return_60d": 0.0,
        "return_120d": 0.0,
        "attention_return_5d": None,
        "attention_return_20d": None,
        "attention_return_60d": None,
        "attention_return_120d": None,
        "above_ma20": False,
        "has_ma20": False,
        "volume_ratio": None,
        "symbol_volume_confirmation": 0.0,
        "fake_breakout_proxy": 0.0,
        "fake_breakout_risk": 0.0,
        "data_coverage": 0.0,
        "attention_history_coverage": 0.0,
        "new_high_120d": None,
        "latest_close": 0.0,
        "latest_high": 0.0,
        "latest_volume": 0.0,
        "latest_amount": 0.0,
        "latest_pct_chg": 0.0,
        "amount_approximated": False,
        "pct_chg_derived": False,
        "limitup_hit": False,
    }


def _ordered_series(frame: pd.DataFrame | None) -> tuple[list[float], list[float]]:
    market_data = _ordered_market_data(frame)
    return market_data["close"], market_data["volume"]


def _ordered_market_data(frame: pd.DataFrame | None) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {
            "close": [],
            "high": [],
            "volume": [],
            "amount": [],
            "pct_chg": [],
            "pct_chg_derived": False,
            "pct_chg_unit": "auto",
            "latest_trade_date": "",
            "latest_name": "",
            "latest_is_st": None,
        }

    close_col = _first_column(frame, _CLOSE_COLUMNS)
    if close_col is None:
        return {
            "close": [],
            "high": [],
            "volume": [],
            "amount": [],
            "pct_chg": [],
            "pct_chg_derived": False,
            "pct_chg_unit": "auto",
            "latest_trade_date": "",
            "latest_name": "",
            "latest_is_st": None,
        }

    ordered = frame
    latest_trade_date: Any = ""
    for date_col in _DATE_COLUMNS:
        if date_col in frame.columns:
            try:
                ordered = frame.sort_values(date_col, kind="mergesort")
            except Exception:
                ordered = frame
            non_null_dates = ordered[date_col].dropna()
            if not non_null_dates.empty:
                latest_trade_date = non_null_dates.iloc[-1]
            break

    close = _finite_values(ordered[close_col])
    high_col = _first_column(ordered, _HIGH_COLUMNS)
    high = _finite_values(ordered[high_col]) if high_col is not None else list(close)
    volume_col = _first_column(ordered, _VOLUME_COLUMNS)
    volume = _finite_values(ordered[volume_col]) if volume_col is not None else []
    amount_col = _first_column(ordered, _AMOUNT_COLUMNS)
    amount = _finite_values(ordered[amount_col]) if amount_col is not None else []
    pct_col = _first_column(ordered, _PCT_CHG_COLUMNS)
    pct_chg_derived = pct_col is None
    pct_chg = _finite_values(ordered[pct_col]) if pct_col is not None else _pct_chg_from_close(close)
    name_col = _first_column(ordered, _NAME_COLUMNS)
    latest_name = ""
    if name_col is not None:
        names = ordered[name_col].dropna()
        if not names.empty:
            latest_name = str(names.iloc[-1])
    st_col = _first_column(ordered, _ST_COLUMNS)
    latest_is_st = None
    if st_col is not None:
        st_values = ordered[st_col].dropna()
        if not st_values.empty:
            latest_is_st = _coerce_st_flag(st_values.iloc[-1])
    return {
        "close": close,
        "high": high,
        "volume": [value for value in volume if value >= 0.0],
        "amount": [value for value in amount if value >= 0.0],
        "pct_chg": pct_chg,
        "pct_chg_derived": pct_chg_derived,
        "pct_chg_unit": "percent" if pct_chg_derived else "auto",
        "latest_trade_date": latest_trade_date,
        "latest_name": latest_name,
        "latest_is_st": latest_is_st,
    }


def _first_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def _finite_values(series: pd.Series) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").dropna().tolist()
    return [float(value) for value in values if math.isfinite(float(value))]


def _latest_amount(
    *,
    close: Sequence[float],
    volume: Sequence[float],
    amount: Sequence[float],
) -> tuple[float, bool]:
    amount_values = [float(value) for value in amount if math.isfinite(float(value))]
    if amount_values:
        return max(0.0, amount_values[-1]), False
    if close and volume:
        latest_close = _safe_float(close[-1], 0.0)
        latest_volume = _safe_float(volume[-1], 0.0)
        if latest_close > 0.0 and latest_volume >= 0.0:
            return latest_close * latest_volume, True
    return 0.0, False


def _normalized_pct_chg_values(values: Sequence[float]) -> list[float]:
    pct_values = [
        float(value)
        for value in values
        if math.isfinite(_safe_float(value, math.nan))
    ]
    if not pct_values:
        return []
    max_abs = max(abs(value) for value in pct_values)
    if max_abs <= 1.0:
        return [value * 100.0 for value in pct_values]
    return pct_values


def _pct_chg_values_for_unit(values: Sequence[float], *, unit: str = "auto") -> list[float]:
    pct_values = [
        float(value)
        for value in values
        if math.isfinite(_safe_float(value, math.nan))
    ]
    if str(unit or "auto").strip().lower() == "percent":
        return pct_values
    return _normalized_pct_chg_values(pct_values)


def _pct_chg_from_close(close: Sequence[float]) -> list[float]:
    values = [float(value) for value in close if math.isfinite(float(value))]
    if len(values) < 2:
        return []
    changes: list[float] = []
    for previous, current in zip(values[:-1], values[1:]):
        if previous <= 0.0:
            changes.append(0.0)
            continue
        changes.append((current / previous - 1.0) * 100.0)
    return changes


def _trade_date_key(trade_date: Any) -> str:
    if trade_date is None:
        return ""
    if hasattr(trade_date, "strftime"):
        try:
            return str(trade_date.strftime("%Y%m%d"))
        except Exception:
            pass
    digits = re.sub(r"\D", "", str(trade_date or ""))
    return digits[:8] if len(digits) >= 8 else digits


def st_limit_ratio(trade_date: Any) -> float:
    trade_date_key = _trade_date_key(trade_date)
    if trade_date_key and trade_date_key >= _ST_LIMIT_CHANGE_DATE:
        return 0.10
    return 0.05


def _coerce_st_flag(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "st", "*st", "风险警示"}:
        return True
    if text in {"0", "false", "no", "n", "", "nan", "none"}:
        return False
    try:
        return bool(float(text))
    except (TypeError, ValueError):
        return None


def _is_st_designated(*, name: Any = None, is_st: Any = None) -> bool:
    coerced = _coerce_st_flag(is_st)
    if coerced is not None:
        return coerced
    # ``ts_code`` alone does not encode historical ST status; require name/flag evidence.
    name_text = str(name or "").strip().upper()
    return bool(name_text and ("ST" in name_text or "风险警示" in name_text))


def _limitup_threshold_pct(
    symbol: str,
    *,
    trade_date: Any = None,
    name: Any = None,
    is_st: Any = None,
) -> float:
    if _is_st_designated(name=name, is_st=is_st):
        ratio = st_limit_ratio(trade_date)
        return 4.9 if ratio <= 0.05 else 9.5
    code = str(symbol or "").strip().split(".", 1)[0]
    if code.startswith(("688", "689", "300", "301")):
        return 19.5
    if code.startswith(("8", "4")):
        return 29.5
    return 9.5


def _is_limitup_latest(
    symbol: str,
    *,
    close: Sequence[float],
    high: Sequence[float],
    pct_chg: Sequence[float],
    pct_chg_unit: str = "auto",
    trade_date: Any = None,
    name: Any = None,
    is_st: Any = None,
) -> bool:
    pct_values = _pct_chg_values_for_unit(pct_chg, unit=pct_chg_unit)
    if not close or not pct_values:
        return False
    latest_close = _safe_float(close[-1], 0.0)
    latest_high = _safe_float(high[-1] if high else close[-1], 0.0)
    if latest_close <= 0.0 or latest_high <= 0.0:
        return False
    return (
        pct_values[-1] >= _limitup_threshold_pct(
            symbol,
            trade_date=trade_date,
            name=name,
            is_st=is_st,
        )
        and latest_close >= latest_high * (1.0 - 0.002)
    )


def _window_return(values: list[float], lookback: int) -> float:
    if len(values) <= lookback:
        return 0.0
    start = values[-1 - lookback]
    end = values[-1]
    if not math.isfinite(start) or not math.isfinite(end) or start <= 0:
        return 0.0
    return end / start - 1.0


def _window_return_optional(values: list[float], lookback: int) -> float | None:
    if len(values) <= lookback:
        return None
    start = values[-1 - lookback]
    end = values[-1]
    if not math.isfinite(start) or not math.isfinite(end) or start <= 0:
        return None
    return end / start - 1.0


def _score_theme(
    *,
    theme_id: str,
    theme_name: str,
    theme_type: str = "industry",
    membership_source: str = "industry_map",
    pit_membership: bool = False,
    members: list[dict[str, Any]],
    min_member_count: int,
    crowding_enabled: bool = False,
    crowding_min_universe: int = 30,
    scanned_symbol_count: int = 0,
    universe_amount: float = 0.0,
    snapshot_history: Sequence[Mapping[str, Any]] = (),
    smoothing_config: ThemeSmoothingConfig | None = None,
) -> ThemeScore:
    member_count = len(members)
    return_20d = _median_metric(members, "return_20d")
    return_5d = _median_metric(members, "return_5d")
    return_60d = _median_metric(members, "return_60d")
    return_120d = _median_metric(members, "return_120d")
    momentum = clamp((return_20d + 0.10) / 0.30)
    volume_confirmation = _theme_volume_confirmation(members)
    acceleration_base = clamp(((return_5d - return_20d / 4.0) + 0.03) / 0.12)
    acceleration = clamp(0.75 * acceleration_base + 0.25 * volume_confirmation)
    breadth = _theme_breadth(members)
    fake_breakout_risk = _median_metric(members, "fake_breakout_risk")
    crowding = _theme_crowding_metrics(
        theme_id=theme_id,
        members=members,
        enabled=crowding_enabled,
        min_universe=crowding_min_universe,
        scanned_symbol_count=scanned_symbol_count,
        universe_amount=universe_amount,
        snapshot_history=snapshot_history,
        smoothing_config=smoothing_config or ThemeSmoothingConfig(),
    )
    overextension_risk = _theme_overextension_risk(
        return_5d,
        breadth,
        volume_confirmation,
        crowding_risk=crowding["crowding_risk"],
        crowding_enabled=crowding_enabled,
    )
    raw = (
        0.30 * momentum
        + 0.24 * acceleration
        + 0.22 * breadth
        + 0.16 * volume_confirmation
        - 0.08 * fake_breakout_risk
        - 0.10 * overextension_risk
    )
    score = clamp(raw, 0.0, 1.0) * 100.0
    data_coverage = _median_metric(members, "data_coverage")
    attention_history_coverage = _median_metric(
        members,
        "attention_history_coverage",
    )
    confidence = clamp(
        0.20
        + min(member_count, 20) / 20.0 * 0.25
        + breadth * 0.15
        + min(data_coverage, 1.0) * 0.15
        + min(attention_history_coverage, 1.0) * 0.15
        - fake_breakout_risk * 0.10,
        0.0,
        0.90,
    )
    phase = _infer_phase(
        score=score,
        breadth=breadth,
        acceleration=acceleration,
        volume_confirmation=volume_confirmation,
        overextension_risk=overextension_risk,
        fake_breakout_risk=fake_breakout_risk,
        momentum=momentum,
    )
    risk_flags = _risk_flags(
        score=score,
        breadth=breadth,
        overextension_risk=overextension_risk,
        fake_breakout_risk=fake_breakout_risk,
        member_count=member_count,
        min_member_count=min_member_count,
        crowding_risk=crowding["crowding_risk"],
        limitup_ratio=crowding["theme_limitup_ratio"],
    )
    top_symbols = _top_symbols(members)
    attention_5d = _normalized_attention_return(
        _median_optional_metric(members, "attention_return_5d"),
        floor=-0.03,
        span=0.12,
    )
    attention_20d = _normalized_attention_return(
        _median_optional_metric(members, "attention_return_20d"),
        floor=-0.05,
        span=0.25,
    )
    attention_60d = _normalized_attention_return(
        _median_optional_metric(members, "attention_return_60d"),
        floor=-0.08,
        span=0.40,
    )
    attention_120d = _normalized_attention_return(
        _median_optional_metric(members, "attention_return_120d"),
        floor=-0.10,
        span=0.60,
    )
    attention_turnover_share = _theme_turnover_share(
        members,
        universe_amount=universe_amount,
    )
    new_high_rate = _optional_boolean_rate(members, "new_high_120d")
    leader_persistence = _leader_persistence(members)
    attention = _weighted_optional_score(
        (
            (attention_5d, 0.12),
            (attention_20d, 0.18),
            (attention_60d, 0.18),
            (attention_120d, 0.17),
            (
                clamp(attention_turnover_share / 0.10)
                if attention_turnover_share is not None
                else None,
                0.10,
            ),
            (breadth, 0.10),
            (new_high_rate, 0.075),
            (leader_persistence, 0.075),
        )
    )
    market_confirmation = _weighted_optional_score(
        (
            (breadth, 0.30),
            (momentum, 0.20),
            (volume_confirmation, 0.15),
            (acceleration, 0.15),
            (new_high_rate, 0.10),
            (leader_persistence, 0.10),
        )
    )

    evidence = [
        f"member_count={member_count}",
        f"momentum={momentum:.3f}",
        f"breadth={breadth:.3f}",
        f"volume_confirmation={volume_confirmation:.3f}",
        f"attention_history_coverage={attention_history_coverage:.3f}",
        f"crowding_risk={crowding['crowding_risk']:.3f}",
    ]
    return ThemeScore(
        theme_id=theme_id,
        theme_name=theme_name,
        theme_type=str(theme_type or "industry"),
        membership_source=str(membership_source or "industry_map"),
        pit_membership=bool(pit_membership),
        phase=phase,
        score=score,
        confidence=confidence,
        member_count=member_count,
        breadth=breadth,
        momentum=momentum,
        acceleration=acceleration,
        volume_confirmation=volume_confirmation,
        overextension_risk=overextension_risk,
        fake_breakout_risk=fake_breakout_risk,
        top_symbols=top_symbols,
        risk_flags=risk_flags,
        evidence=evidence,
        attention=attention,
        attention_5d=attention_5d,
        attention_20d=attention_20d,
        attention_60d=attention_60d,
        attention_120d=attention_120d,
        attention_turnover_share=attention_turnover_share,
        new_high_rate=new_high_rate,
        leader_persistence=leader_persistence,
        attention_history_coverage=attention_history_coverage,
        market_confirmation=market_confirmation,
        evidence_confidence=confidence,
        theme_turnover_share=crowding["theme_turnover_share"],
        turnover_share_sma10=crowding["turnover_share_sma10"],
        turnover_share_stretch=crowding["turnover_share_stretch"],
        turnover_share_delta_5d=crowding["turnover_share_delta_5d"],
        turnover_share_trend=crowding["turnover_share_trend"],
        theme_limitup_ratio=crowding["theme_limitup_ratio"],
        limitup_norm=crowding["limitup_norm"],
        member_turnover_concentration=crowding["member_turnover_concentration"],
        crowding_risk=crowding["crowding_risk"],
        crowding_status=crowding["crowding_status"],
        crowding_diagnostic_notes=list(crowding["crowding_diagnostic_notes"]),
        metadata={
            "theme_return_5d": return_5d,
            "theme_return_20d": return_20d,
            "theme_return_60d": return_60d,
            "theme_return_120d": return_120d,
            "attention_horizons": [5, 20, 60, 120],
            "crowding_weight_model": dict(_THEME_CROWDING_WEIGHTS),
        },
    )


def _apply_smoothing_to_theme_score(
    *,
    theme_score: ThemeScore,
    history_members_by_offset: Mapping[int, list[dict[str, Any]]],
    min_member_count: int,
    config: ThemeSmoothingConfig,
) -> None:
    score_history: list[float] = []
    for offset in sorted(history_members_by_offset, reverse=True):
        members = list(history_members_by_offset.get(offset, []) or [])
        if len(members) < min_member_count:
            continue
        if _median_metric(members, "data_coverage") < 0.50:
            continue
        try:
            historical_score = _score_theme(
                theme_id=theme_score.theme_id,
                theme_name=theme_score.theme_name,
                members=members,
                min_member_count=min_member_count,
            )
        except Exception:
            continue
        score_history.append(float(historical_score.score))

    smoothing = smooth_theme_series(score_history, config)
    theme_score.raw_score = float(theme_score.score)
    theme_score.smoothed_score = smoothing.smoothed_score
    theme_score.effective_score = (
        float(smoothing.smoothed_score)
        if smoothing.smoothed_score is not None
        else float(theme_score.score)
    )
    theme_score.heat_10d = smoothing.heat_10d
    theme_score.heat_delta_5d = smoothing.heat_delta_5d
    theme_score.persistence_count = int(smoothing.persistence_count)
    theme_score.trend_state = smoothing.trend_state
    theme_score.smoothing_observation_count = int(smoothing.observation_count)
    theme_score.smoothing_status = smoothing.status
    theme_score.metadata = {
        **dict(theme_score.metadata or {}),
        "smoothing_method": "sma",
        "smoothing_window": int(config.window),
        "smoothing_min_observations": int(config.min_observations),
        "smoothing_diagnostic_notes": list(smoothing.diagnostic_notes),
    }


def _choose_primary_theme(
    *,
    memberships: Sequence[str],
    theme_scores: Mapping[str, ThemeScore],
    concept_primary_margin: float,
) -> ThemeScore | None:
    candidates = [
        theme_scores[theme_id]
        for theme_id in list(memberships or [])
        if theme_id in theme_scores
    ]
    if not candidates:
        return None

    industry = next(
        (
            score
            for score in candidates
            if str(score.theme_type or "").strip().lower() == "industry"
        ),
        None,
    )
    if industry is None:
        return sorted(candidates, key=lambda score: (-score.score, score.theme_id))[0]

    margin_points = clamp(_safe_float(concept_primary_margin, 0.05)) * 100.0
    concept_candidates = [
        score
        for score in candidates
        if str(score.theme_type or "").strip().lower() != "industry"
        and score.phase != ThemePhase.DISTRIBUTION
    ]
    if not concept_candidates:
        return industry

    best_concept = sorted(
        concept_candidates,
        key=lambda score: (-score.score, score.theme_id),
    )[0]
    if best_concept.score >= industry.score + margin_points:
        return best_concept
    return industry


def _resolve_concept_config(
    *,
    concept_membership_enabled: bool | None,
    concept_primary_margin: float | None,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": False,
        "primary_margin": 0.05,
    }
    try:
        from quant_investor.config import Config

        defaults.update(
            {
                "enabled": bool(getattr(Config, "THEME_CONCEPT_MEMBERSHIP_ENABLED", False)),
                "primary_margin": _safe_float(
                    getattr(Config, "THEME_CONCEPT_PRIMARY_MARGIN", 0.05),
                    0.05,
                ),
            }
        )
    except Exception:
        pass

    if concept_membership_enabled is not None:
        defaults["enabled"] = bool(concept_membership_enabled)
    if concept_primary_margin is not None:
        defaults["primary_margin"] = _safe_float(
            concept_primary_margin,
            defaults["primary_margin"],
        )
    defaults["primary_margin"] = clamp(defaults["primary_margin"])
    return defaults


def _resolve_crowding_config(
    *,
    crowding_enabled: bool | None,
    crowding_min_universe: int | None,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": False,
        "min_universe": 30,
    }
    try:
        from quant_investor.config import Config

        defaults.update(
            {
                "enabled": bool(getattr(Config, "THEME_CROWDING_ENABLED", False)),
                "min_universe": max(
                    int(getattr(Config, "THEME_CROWDING_MIN_UNIVERSE", 30) or 30),
                    1,
                ),
            }
        )
    except Exception:
        pass

    if crowding_enabled is not None:
        defaults["enabled"] = bool(crowding_enabled)
    if crowding_min_universe is not None:
        defaults["min_universe"] = max(int(crowding_min_universe or 30), 1)
    return defaults


def _theme_crowding_metrics(
    *,
    theme_id: str,
    members: list[dict[str, Any]],
    enabled: bool,
    min_universe: int,
    scanned_symbol_count: int,
    universe_amount: float,
    snapshot_history: Sequence[Mapping[str, Any]],
    smoothing_config: ThemeSmoothingConfig,
) -> dict[str, Any]:
    neutral = _neutral_crowding_metrics(status="disabled")
    if not enabled:
        return neutral

    notes: list[str] = []
    min_universe = max(int(min_universe or 30), 1)
    if int(scanned_symbol_count or 0) < min_universe:
        return {
            **_neutral_crowding_metrics(status="insufficient_universe"),
            "crowding_diagnostic_notes": [
                f"insufficient_universe:{int(scanned_symbol_count or 0)}/{min_universe}"
            ],
        }

    total_amount = max(0.0, _safe_float(universe_amount, 0.0))
    if total_amount <= 0.0:
        return {
            **_neutral_crowding_metrics(status="unavailable"),
            "crowding_diagnostic_notes": ["universe_amount_unavailable"],
        }

    member_amounts = [
        max(0.0, _safe_float(member.get("latest_amount"), 0.0))
        for member in list(members or [])
    ]
    theme_amount = sum(member_amounts)
    turnover_share = clamp(theme_amount / total_amount)
    if any(bool(member.get("amount_approximated")) for member in list(members or [])):
        notes.append("amount_approximated")
    if any(bool(member.get("pct_chg_derived")) for member in list(members or [])):
        notes.append("pct_chg_derived_from_close")

    share_history = _turnover_share_history(theme_id, snapshot_history)
    share_sma = smooth_numeric_series(
        share_history,
        lower=0.0,
        upper=1.0,
        config=smoothing_config,
    )
    if share_sma is None or share_sma <= 0.0:
        stretch = 0.0
        notes.append("turnover_share_smoothing_status=insufficient_history")
    else:
        stretch = clamp((turnover_share / share_sma - 1.0) / 1.0)
    delta_5d = _turnover_share_delta_5d(share_history)
    trend = _turnover_share_trend(delta_5d)

    member_count = len(members)
    limitup_count = sum(1 for member in list(members or []) if bool(member.get("limitup_hit")))
    limitup_ratio = (limitup_count / member_count) if member_count else 0.0
    limitup_norm = clamp(limitup_ratio / 0.30)
    if member_count < 4:
        concentration = 1.0
        notes.append("member_turnover_concentration_small_theme")
    elif theme_amount <= 0.0:
        concentration = 0.0
        notes.append("theme_amount_unavailable")
    else:
        concentration = clamp(sum(sorted(member_amounts, reverse=True)[:3]) / theme_amount)

    crowding_risk = clamp(
        _THEME_CROWDING_WEIGHTS["turnover_share_stretch"] * stretch
        + _THEME_CROWDING_WEIGHTS["limitup_norm"] * limitup_norm
        + _THEME_CROWDING_WEIGHTS["member_turnover_concentration"] * concentration
    )
    return {
        "theme_turnover_share": round(turnover_share, 6),
        "turnover_share_sma10": share_sma,
        "turnover_share_stretch": round(stretch, 6),
        "turnover_share_delta_5d": delta_5d,
        "turnover_share_trend": trend,
        "theme_limitup_ratio": round(limitup_ratio, 6),
        "limitup_norm": round(limitup_norm, 6),
        "member_turnover_concentration": round(concentration, 6),
        "crowding_risk": round(crowding_risk, 6),
        "crowding_status": "success",
        "crowding_diagnostic_notes": _dedupe_texts(notes),
    }


def _neutral_crowding_metrics(*, status: str) -> dict[str, Any]:
    return {
        "theme_turnover_share": 0.0,
        "turnover_share_sma10": None,
        "turnover_share_stretch": 0.0,
        "turnover_share_delta_5d": None,
        "turnover_share_trend": status,
        "theme_limitup_ratio": 0.0,
        "limitup_norm": 0.0,
        "member_turnover_concentration": 0.0,
        "crowding_risk": 0.0,
        "crowding_status": status,
        "crowding_diagnostic_notes": [],
    }


def _turnover_share_history(
    theme_id: str,
    snapshot_history: Sequence[Mapping[str, Any]],
) -> list[float]:
    values: list[float] = []
    for snapshot in list(snapshot_history or []):
        if not isinstance(snapshot, Mapping):
            continue
        rotation = snapshot.get("theme_rotation")
        payload = rotation if isinstance(rotation, Mapping) else snapshot
        scores = payload.get("theme_scores") if isinstance(payload, Mapping) else None
        if not isinstance(scores, Mapping):
            continue
        theme_payload = scores.get(theme_id)
        if not isinstance(theme_payload, Mapping):
            continue
        numeric = _safe_float(theme_payload.get("theme_turnover_share"), math.nan)
        if math.isfinite(numeric):
            values.append(clamp(numeric))
    return values


def _turnover_share_delta_5d(values: Sequence[float]) -> float | None:
    cleaned = [
        clamp(_safe_float(value, math.nan))
        for value in list(values or [])
        if math.isfinite(_safe_float(value, math.nan))
    ]
    if len(cleaned) <= 5:
        return None
    previous = cleaned[-6:-1]
    if not previous:
        return None
    return round(cleaned[-1] - mean(previous), 6)


def _turnover_share_trend(delta_5d: float | None) -> str:
    if delta_5d is None:
        return "insufficient_history"
    if delta_5d >= 0.02:
        return "warming"
    if delta_5d <= -0.02:
        return "cooling"
    return "stable"


def _crowding_result_notes(theme_scores: Sequence[ThemeScore]) -> list[str]:
    notes: list[str] = []
    for score in list(theme_scores or []):
        notes.extend(str(note) for note in list(score.crowding_diagnostic_notes or []))
    return _dedupe_texts(notes)


def _resolve_policy_config(
    *,
    policy_catalyst_enabled: bool | None,
    policy_catalyst_weight: float | None,
    policy_lookback_days: int | None,
    policy_event_path: str | None,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "enabled": False,
        "weight": 0.16,
        "lookback_days": 30,
        "event_path": "data/theme_policy_events.jsonl",
    }
    try:
        from quant_investor.config import Config

        defaults.update(
            {
                "enabled": bool(getattr(Config, "THEME_POLICY_CATALYST_ENABLED", False)),
                "weight": _safe_float(
                    getattr(Config, "THEME_POLICY_CATALYST_WEIGHT", 0.16),
                    0.16,
                ),
                "lookback_days": max(
                    int(getattr(Config, "THEME_POLICY_LOOKBACK_DAYS", 30) or 30),
                    1,
                ),
                "event_path": str(
                    getattr(
                        Config,
                        "THEME_POLICY_EVENT_PATH",
                        "data/theme_policy_events.jsonl",
                    )
                    or "data/theme_policy_events.jsonl"
                ),
            }
        )
    except Exception:
        pass

    if policy_catalyst_enabled is not None:
        defaults["enabled"] = bool(policy_catalyst_enabled)
    if policy_catalyst_weight is not None:
        defaults["weight"] = _safe_float(policy_catalyst_weight, defaults["weight"])
    if policy_lookback_days is not None:
        defaults["lookback_days"] = max(int(policy_lookback_days or 30), 1)
    if policy_event_path is not None:
        defaults["event_path"] = str(policy_event_path or defaults["event_path"])
    defaults["weight"] = clamp(defaults["weight"])
    return defaults


def _apply_policy_catalysts(
    *,
    theme_scores: list[ThemeScore],
    members_by_theme: Mapping[str, list[dict[str, Any]]],
    as_of: str,
    enabled: bool,
    weight: float,
    lookback_days: int,
    event_path: str,
) -> dict[str, Any]:
    metadata = {
        "enabled": bool(enabled),
        "status": "disabled",
        "weight": clamp(weight),
        "event_path": str(event_path or ""),
        "matched_theme_count": 0,
        "diagnostic_notes": [],
    }
    if not enabled:
        return metadata

    scanner = PolicyCatalystScanner(
        event_path=event_path or "data/theme_policy_events.jsonl",
        lookback_days=lookback_days,
    )
    events = scanner.load_events()
    metadata["diagnostic_notes"] = list(scanner.diagnostic_notes)
    if scanner.status != "success":
        metadata["status"] = "unavailable"
        for theme_score in theme_scores:
            theme_score.policy_stage = "unavailable"
        return metadata

    metadata["status"] = "success"
    matched = 0
    for theme_score in theme_scores:
        member_symbols = [
            str(member.get("symbol", ""))
            for member in list(members_by_theme.get(theme_score.theme_id, []) or [])
            if str(member.get("symbol", "")).strip()
        ]
        catalyst = scanner.score_theme(
            theme_id=theme_score.theme_id,
            theme_name=theme_score.theme_name,
            member_symbols=member_symbols,
            as_of=as_of,
            events=events,
        )
        _apply_policy_score_to_theme(theme_score, catalyst.to_dict(), weight=weight)
        if catalyst.policy_stage not in {"no_match", "unavailable"}:
            matched += 1

    metadata["matched_theme_count"] = matched
    return metadata


def _apply_policy_score_to_theme(
    theme_score: ThemeScore,
    catalyst: Mapping[str, Any],
    *,
    weight: float,
) -> None:
    policy_score = clamp(_safe_float(catalyst.get("policy_score"), 0.0))
    policy_boost = policy_score * clamp(weight) * 100.0
    theme_score.score = clamp(_safe_float(theme_score.score, 0.0) + policy_boost, 0.0, 100.0)
    theme_score.policy_catalyst_score = policy_score
    theme_score.policy_confidence = clamp(_safe_float(catalyst.get("confidence"), 0.0))
    theme_score.policy_stage = str(catalyst.get("policy_stage") or "no_match")
    theme_score.policy_evidence = [
        str(item)
        for item in list(catalyst.get("evidence", []) or [])
        if str(item).strip()
    ]
    theme_score.policy_risk_flags = [
        str(flag)
        for flag in list(catalyst.get("risk_flags", []) or [])
        if str(flag).startswith("policy_")
    ]
    theme_score.risk_flags = _dedupe_texts(
        [*list(theme_score.risk_flags or []), *theme_score.policy_risk_flags]
    )
    theme_score.metadata = {
        **dict(theme_score.metadata or {}),
        "policy_catalyst": {
            "policy_score": policy_score,
            "confidence": theme_score.policy_confidence,
            "stage": theme_score.policy_stage,
            "score_component": policy_boost,
            "weight_cap": clamp(weight),
        },
    }


def _dedupe_texts(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _median_metric(members: list[dict[str, Any]], key: str) -> float:
    values = [_safe_float(member.get(key), 0.0) for member in members]
    return float(median(values)) if values else 0.0


def _median_optional_metric(
    members: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [
        _safe_float(member.get(key), math.nan)
        for member in members
        if member.get(key) is not None
    ]
    finite = [value for value in values if math.isfinite(value)]
    return float(median(finite)) if finite else None


def _normalized_attention_return(
    value: float | None,
    *,
    floor: float,
    span: float,
) -> float | None:
    if value is None or span <= 0:
        return None
    return clamp((value - floor) / span)


def _theme_turnover_share(
    members: list[dict[str, Any]],
    *,
    universe_amount: float,
) -> float | None:
    denominator = _safe_float(universe_amount, 0.0)
    if denominator <= 0.0:
        return None
    member_amount = sum(
        max(0.0, _safe_float(member.get("latest_amount"), 0.0))
        for member in members
    )
    return clamp(member_amount / denominator)


def _optional_boolean_rate(
    members: list[dict[str, Any]],
    key: str,
) -> float | None:
    values = [member.get(key) for member in members if member.get(key) is not None]
    if not values:
        return None
    return clamp(sum(1 for value in values if bool(value)) / len(values))


def _leader_persistence(members: list[dict[str, Any]]) -> float | None:
    eligible = [
        member
        for member in members
        if member.get("attention_return_5d") is not None
        and member.get("attention_return_20d") is not None
    ]
    if len(eligible) < 3:
        return None
    leader_count = max(1, int(math.ceil(len(eligible) * 0.20)))
    leaders_5d = {
        str(member.get("symbol") or "")
        for member in sorted(
            eligible,
            key=lambda item: (
                -_safe_float(item.get("attention_return_5d"), 0.0),
                str(item.get("symbol") or ""),
            ),
        )[:leader_count]
    }
    leaders_20d = {
        str(member.get("symbol") or "")
        for member in sorted(
            eligible,
            key=lambda item: (
                -_safe_float(item.get("attention_return_20d"), 0.0),
                str(item.get("symbol") or ""),
            ),
        )[:leader_count]
    }
    return clamp(len(leaders_5d.intersection(leaders_20d)) / leader_count)


def _weighted_optional_score(
    components: Sequence[tuple[float | None, float]],
) -> float:
    available = [
        (clamp(value), max(0.0, float(weight)))
        for value, weight in components
        if value is not None and math.isfinite(_safe_float(value, math.nan))
    ]
    denominator = sum(weight for _, weight in available)
    if denominator <= 0.0:
        return 0.0
    return clamp(sum(value * weight for value, weight in available) / denominator)


def _theme_volume_confirmation(members: list[dict[str, Any]]) -> float:
    ratios = [
        _safe_float(member.get("volume_ratio"), math.nan)
        for member in members
        if member.get("volume_ratio") is not None
    ]
    ratios = [ratio for ratio in ratios if math.isfinite(ratio)]
    if not ratios:
        return 0.0
    return clamp((float(median(ratios)) - 1.0) / 1.5)


def _theme_breadth(members: list[dict[str, Any]]) -> float:
    if not members:
        return 0.0
    member_count = len(members)
    positive_20d = sum(1 for member in members if _safe_float(member.get("return_20d"), 0.0) > 0)
    positive_5d = sum(1 for member in members if _safe_float(member.get("return_5d"), 0.0) > 0)
    ma_members = [member for member in members if bool(member.get("has_ma20"))]
    ma_ratio = (
        sum(1 for member in ma_members if bool(member.get("above_ma20"))) / len(ma_members)
        if ma_members
        else 0.0
    )
    return clamp(((positive_20d / member_count) + (positive_5d / member_count) + ma_ratio) / 3.0)


def _theme_overextension_risk(
    theme_return_5d: float,
    breadth: float,
    volume_confirmation: float,
    *,
    crowding_risk: float = 0.0,
    crowding_enabled: bool = False,
) -> float:
    risk = clamp((theme_return_5d - 0.08) / 0.12)
    if volume_confirmation > 0.85 and breadth < 0.50:
        risk = clamp(risk + 0.15)
    if crowding_enabled:
        risk = clamp(risk + 0.30 * clamp(crowding_risk))
    return risk


def _infer_phase(
    *,
    score: float,
    breadth: float,
    acceleration: float,
    volume_confirmation: float,
    overextension_risk: float,
    fake_breakout_risk: float,
    momentum: float,
) -> ThemePhase:
    if overextension_risk >= 0.70:
        return ThemePhase.OVEREXTENDED
    if fake_breakout_risk >= 0.70 and momentum < 0.45:
        return ThemePhase.DISTRIBUTION
    if score < 35:
        return ThemePhase.UNCLASSIFIED
    if score >= 70 and breadth >= 0.55 and overextension_risk < 0.60:
        return ThemePhase.CONFIRMED_ROTATION
    if score >= 55 and acceleration >= 0.55 and volume_confirmation >= 0.35:
        return ThemePhase.EARLY_ACCELERATION
    if score >= 40 and breadth >= 0.40:
        return ThemePhase.ACCUMULATION
    return ThemePhase.UNCLASSIFIED


def _risk_flags(
    *,
    score: float,
    breadth: float,
    overextension_risk: float,
    fake_breakout_risk: float,
    member_count: int,
    min_member_count: int,
    crowding_risk: float = 0.0,
    limitup_ratio: float = 0.0,
) -> list[str]:
    flags: list[str] = []
    if overextension_risk >= 0.70:
        flags.append("theme_overextended")
    if fake_breakout_risk >= 0.65:
        flags.append("theme_fake_breakout_risk")
    if score >= 55 and breadth < 0.35:
        flags.append("theme_low_breadth")
    if member_count < min_member_count:
        flags.append("theme_low_member_count")
    if crowding_risk >= 0.70:
        flags.append("theme_crowded")
    if limitup_ratio >= 0.20 and breadth < 0.40:
        flags.append("theme_narrow_leadership")
    return flags


def _top_symbols(members: list[dict[str, Any]]) -> list[str]:
    ranked: list[tuple[float, str]] = []
    for member in members:
        return_20d = clamp((_safe_float(member.get("return_20d"), 0.0) + 0.10) / 0.30)
        return_5d = clamp((_safe_float(member.get("return_5d"), 0.0) + 0.03) / 0.12)
        volume_confirmation = clamp(_safe_float(member.get("symbol_volume_confirmation"), 0.0))
        fake_breakout_risk = clamp(_safe_float(member.get("fake_breakout_risk"), 0.0))
        symbol_score = (
            0.45 * return_20d
            + 0.25 * return_5d
            + 0.20 * volume_confirmation
            - 0.10 * fake_breakout_risk
        )
        ranked.append((clamp(symbol_score), str(member.get("symbol", ""))))
    return [symbol for _, symbol in sorted(ranked, key=lambda item: (-item[0], item[1]))[:5]]


def _fake_breakout_proxy(close: list[float]) -> float:
    if not close:
        return 0.0
    latest = close[-1]
    high_60 = max(close[-60:])
    recent_peak = max(close[-10:])
    if high_60 <= 0 or recent_peak <= 0:
        return 0.0
    distance_from_high = max(0.0, high_60 - latest) / high_60
    recent_pullback = max(0.0, recent_peak - latest) / recent_peak
    distance_risk = clamp((distance_from_high - 0.03) / 0.12)
    pullback_risk = clamp((recent_pullback - 0.02) / 0.10)
    return clamp(0.60 * distance_risk + 0.40 * pullback_risk)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _state_float(state: Mapping[str, Any], key: str, default: float) -> float:
    if not isinstance(state, Mapping):
        return default
    return _safe_float(state.get(key), default)


def _state_optional_float(state: Mapping[str, Any], key: str) -> float | None:
    if not isinstance(state, Mapping) or state.get(key) is None:
        return None
    numeric = _safe_float(state.get(key), math.nan)
    return numeric if math.isfinite(numeric) else None
