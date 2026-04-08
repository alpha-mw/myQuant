"""GlobalContext builder — assembles the once-per-run shared market state.

Responsibilities:
- Full-A logical universe resolution
- Stable CN freshness target selection
- Latest stable trade date
- Data completeness / quarantine
- Macro regime detection
- Model capability map
- Symbol -> company_name map
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from quant_investor.agent_protocol import DataQualityIssue, GlobalContext
from quant_investor.config import config
from quant_investor.logger import get_logger
from quant_investor.market.data_quality import build_data_quality_diagnostics
from quant_investor.market.provider_health import detect_provider_health
from quant_investor.model_roles import resolve_model_role

_logger = get_logger("GlobalContextBuilder")

# Default regime prior base-rates (used when regime detector is unavailable).
_REGIME_DEFAULTS: dict[str, dict[str, Any]] = {
    "趋势上涨": {"max_position": 0.90, "stop_loss_pct": -0.10, "rebalance_freq": "W"},
    "趋势下跌": {"max_position": 0.40, "stop_loss_pct": -0.05, "rebalance_freq": "W"},
    "震荡低波": {"max_position": 0.80, "stop_loss_pct": -0.08, "rebalance_freq": "M"},
    "震荡高波": {"max_position": 0.60, "stop_loss_pct": -0.06, "rebalance_freq": "W"},
    "未知": {"max_position": 0.60, "stop_loss_pct": -0.07, "rebalance_freq": "W"},
}


def _universe_hash(symbols: list[str]) -> str:
    payload = ",".join(sorted(symbols))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _load_stock_name_map(market: str) -> dict[str, str]:
    """Load symbol -> company_name from the local cache."""
    if market == "CN":
        candidates = [
            Path(config.DATA_DIR) / "cn_universe" / "stock_names.json",
            Path("data") / "cn_universe" / "stock_names.json",
        ]
    else:
        candidates = [
            Path(config.DATA_DIR) / "us_universe" / "stock_names.json",
            Path("data") / "us_universe" / "stock_names.json",
        ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


def _detect_macro_regime(symbol_data: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
    """Run the regime detector and return (regime_label, regime_params)."""
    try:
        from quant_investor.regime_detector import RegimeDetector
        detector = RegimeDetector()
        if symbol_data:
            result = detector.detect(symbol_data)
        else:
            result = detector.detect_from_index()
        regime_label = result.regime.value if hasattr(result, "regime") else str(result.regime)
        params = asdict(result) if hasattr(result, "__dataclass_fields__") else {}
        return regime_label, params
    except Exception as exc:
        _logger.warning("Regime detection failed, using default: %s", exc)
        return "未知", _REGIME_DEFAULTS["未知"]


def _build_model_capability_map(
    *,
    agent_model: str,
    master_model: str,
    agent_fallback_model: str,
    master_fallback_model: str,
) -> dict[str, dict[str, Any]]:
    """Build a model capability map by probing provider availability."""
    roles: dict[str, tuple[str, str]] = {
        "branch": (agent_model, agent_fallback_model),
        "master": (master_model, master_fallback_model),
    }
    cap_map: dict[str, dict[str, Any]] = {}
    for role, (primary, fallback) in roles.items():
        resolution = resolve_model_role(role=role, primary_model=primary, fallback_model=fallback)
        cap_map[role] = {
            "primary_model": primary,
            "fallback_model": fallback,
            "resolved_model": resolution.resolved_model,
            "provider_available": not resolution.fallback_used,
            "fallback_used": resolution.fallback_used,
            "fallback_reason": resolution.fallback_reason,
        }
    return cap_map


class GlobalContextBuilder:
    """Assembles a GlobalContext object once per run."""

    def build(
        self,
        *,
        stock_pool: list[str],
        market: str = "CN",
        universe_key: str = "full_a",
        total_capital: float = 1_000_000.0,
        risk_level: str = "中等",
        agent_model: str = "",
        master_model: str = "",
        agent_fallback_model: str = "",
        master_fallback_model: str = "",
        freshness_mode: str = "stable",
        latest_trade_date: str = "",
        effective_target_trade_date: str = "",
        data_quality_issues: list[DataQualityIssue] | None = None,
        symbol_data: dict[str, Any] | None = None,
        config: Any = None,
        data_bundle: Any = None,
    ) -> GlobalContext:
        _logger.info(
            "Building GlobalContext: market=%s universe=%s symbols=%d",
            market, universe_key, len(stock_pool),
        )

        # Macro regime
        regime_label, regime_params = _detect_macro_regime(symbol_data)

        # Model capability
        cap_map = _build_model_capability_map(
            agent_model=agent_model,
            master_model=master_model,
            agent_fallback_model=agent_fallback_model,
            master_fallback_model=master_fallback_model,
        )
        cap_map.update(
            detect_provider_health(
                agent_model=agent_model or cap_map.get("branch", {}).get("resolved_model", ""),
                master_model=master_model or cap_map.get("master", {}).get("resolved_model", ""),
            )
        )

        # Symbol names
        name_map = _load_stock_name_map(market)

        # Data quality quarantine — symbols with known issues
        quarantine: list[str] = []
        quality_issues = list(data_quality_issues or [])
        for issue in quality_issues:
            if issue.severity in ("error", "critical") and issue.symbol:
                if issue.symbol not in quarantine:
                    quarantine.append(issue.symbol)

        # Universe tiers
        researchable = [s for s in stock_pool if s not in quarantine]
        universe_tiers = {
            "total": list(stock_pool),
            "researchable": researchable,
            "shortlistable": [],  # populated after funnel
            "final_selected": [],  # populated after portfolio decision
        }
        diagnostics = build_data_quality_diagnostics(
            total_symbols=stock_pool,
            researchable_symbols=researchable,
            shortlistable_symbols=[],
            final_selected_symbols=[],
            quarantined_symbols=quarantine,
            issues=quality_issues,
        )

        return GlobalContext(
            market=market,
            universe_key=universe_key,
            latest_trade_date=latest_trade_date,
            universe_symbols=list(stock_pool),
            universe_hash=_universe_hash(stock_pool),
            macro_regime=regime_label,
            regime_params=regime_params,
            risk_budget={
                "total_capital": total_capital,
                "risk_level": risk_level,
            },
            data_quality_issues=quality_issues,
            data_quality_quarantine=quarantine,
            model_capability_map=cap_map,
            symbol_name_map=name_map,
            freshness_mode=freshness_mode,
            effective_target_trade_date=effective_target_trade_date or latest_trade_date,
            universe_tiers=universe_tiers,
            data_quality_diagnostics=diagnostics,
            metadata={
                "provider_health": cap_map,
            },
        )
