"""Run a V8 analysis job and write a normalized JSON result."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _project_paths() -> tuple[Path, Path]:
    project_root = Path(__file__).resolve().parents[2]
    package_root = project_root / "quant_investor"
    return project_root, package_root


def _prepare_imports() -> None:
    return None


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types to plain Python."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    type_name = type(obj).__name__
    if "int" in type_name.lower() and hasattr(obj, "item"):
        return int(obj)
    if "float" in type_name.lower() and hasattr(obj, "item"):
        return float(obj)
    if "bool" in type_name.lower() and hasattr(obj, "item"):
        return bool(obj)
    if hasattr(obj, "to_dict"):
        return None
    try:
        return str(obj)
    except Exception:
        return None


def _branch_summary(branch: Any, settings: dict[str, Any], assignments: list[dict[str, Any]]) -> dict[str, Any]:
    top_symbols = [
        symbol
        for symbol, _ in sorted(
            branch.symbol_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
    ]
    return {
        "branch_name": branch.branch_name,
        "enabled": True,
        "score": float(branch.score),
        "confidence": float(branch.confidence),
        "explanation": branch.explanation,
        "risks": [str(item) for item in branch.risks[:5]],
        "top_symbols": top_symbols,
        "branch_mode": str(branch.signals.get("branch_mode", "")) or None,
        "settings": settings,
        "model_assignment": assignments if branch.branch_name == "llm_debate" else [],
        "signals": _sanitize_for_json(dict(branch.signals)) if branch.signals else {},
        "metadata": _sanitize_for_json(dict(branch.metadata)) if hasattr(branch, "metadata") and branch.metadata else {},
    }


def _risk_summary(result: Any, request: dict[str, Any]) -> dict[str, Any]:
    risk_result = result.risk_results
    if risk_result is None:
        return {
            "risk_level": "unknown",
            "volatility": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "warnings": [],
            "max_single_position": float(request["risk"].get("max_single_position", 0.2)),
            "max_drawdown_limit": float(request["risk"].get("max_drawdown_limit", 0.15)),
            "default_stop_loss": float(request["risk"].get("default_stop_loss", 0.08)),
            "keep_cash_buffer": bool(request["risk"].get("keep_cash_buffer", True)),
            "stress_test": "",
        }
    warnings = [str(item) for item in risk_result.risk_warnings]
    return {
        "risk_level": risk_result.risk_level,
        "volatility": float(risk_result.risk_metrics.volatility),
        "max_drawdown": float(risk_result.risk_metrics.max_drawdown),
        "sharpe_ratio": float(risk_result.risk_metrics.sharpe_ratio),
        "warnings": warnings,
        "max_single_position": float(request["risk"].get("max_single_position", 0.2)),
        "max_drawdown_limit": float(request["risk"].get("max_drawdown_limit", 0.15)),
        "default_stop_loss": float(request["risk"].get("default_stop_loss", 0.08)),
        "keep_cash_buffer": bool(request["risk"].get("keep_cash_buffer", True)),
        "stress_test": warnings[-1] if warnings else "",
    }


def _trade_summary(trade: Any) -> dict[str, Any]:
    if is_dataclass(trade):
        payload = asdict(trade)
    elif isinstance(trade, dict):
        payload = dict(trade)
    else:
        payload = {
            key: getattr(trade, key)
            for key in dir(trade)
            if not key.startswith("_") and not callable(getattr(trade, key))
        }
    risk_flags = [str(item) for item in payload["risk_flags"]]
    rationale = [
        f"共识得分 {float(payload['consensus_score']):+.2f}",
        f"支持分支 {int(payload['branch_positive_count'])}/5",
    ]
    if payload["trend_regime"]:
        rationale.append(f"趋势状态 {payload['trend_regime']}")
    if risk_flags:
        rationale.append(f"风险提示：{'；'.join(risk_flags[:2])}")
    return {
        "symbol": payload["symbol"],
        "action": payload["action"],
        "current_price": float(payload["current_price"]),
        "recommended_entry_price": float(payload["recommended_entry_price"]),
        "target_price": float(payload["target_price"]),
        "stop_loss_price": float(payload["stop_loss_price"]),
        "suggested_weight": float(payload["suggested_weight"]),
        "suggested_amount": float(payload["suggested_amount"]),
        "suggested_shares": int(payload["suggested_shares"]),
        "confidence": float(payload["confidence"]),
        "consensus_score": float(payload["consensus_score"]),
        "branch_positive_count": int(payload["branch_positive_count"]),
        "trend_regime": payload["trend_regime"],
        "risk_flags": risk_flags,
        "rationale": "；".join(rationale),
    }


def _default_preset() -> dict[str, Any]:
    return {
        "mode": "single",
        "targets": [],
        "preset": "quick_scan",
        "market": "CN",
        "branches": {
            "kline": {"enabled": True, "settings": {"prediction_horizon": "20d", "trend_window": "60d", "backend": "heuristic"}},
            "quant": {"enabled": True, "settings": {"factor_pack": "core", "rebalance": "monthly"}},
            "llm_debate": {"enabled": True, "settings": {"rounds": 2}},
            "intelligence": {"enabled": True, "settings": {"event_risk": True, "capital_flow": True}},
            "macro": {"enabled": True, "settings": {"overlay_strength": "medium"}},
        },
        "risk": {
            "capital": 1_000_000.0,
            "risk_level": "中等",
            "max_single_position": 0.2,
            "max_drawdown_limit": 0.15,
            "default_stop_loss": 0.08,
            "keep_cash_buffer": True,
        },
        "portfolio": {"candidate_limit": 10, "allocation_mode": "target_weight", "allow_cash_buffer": True},
        "llm_debate": {
            "enabled": True,
            "models": [],
            "rounds": 2,
            "assignment_mode": "random_balanced",
            "judge_mode": "auto",
            "judge_model": None,
            "assignments": [],
        },
    }


def _normalize_targets(payload: dict[str, Any]) -> list[str]:
    targets = payload.get("targets") or payload.get("stocks") or []
    return list(dict.fromkeys(str(item).strip().upper() for item in targets if str(item).strip()))


def _build_llm_assignments(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = [str(item).strip() for item in config.get("models", []) if str(item).strip()]
    if not models:
        return []
    shuffled = models[:]
    random.shuffle(shuffled)
    if len(shuffled) == 1:
        return [{"model": shuffled[0], "role": "solo"}]
    if len(shuffled) == 2:
        return [
            {"model": shuffled[0], "role": "bull"},
            {"model": shuffled[1], "role": "bear"},
            {"model": "ensemble", "role": "judge"},
        ]
    judge = shuffled[-1]
    remaining = shuffled[:-1]
    assignments = []
    for model in remaining[::2]:
        assignments.append({"model": model, "role": "bull"})
    for model in remaining[1::2]:
        assignments.append({"model": model, "role": "bear"})
    assignments.append({"model": judge, "role": "judge"})
    return assignments


def _normalize_request(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = _default_preset()
    normalized["targets"] = _normalize_targets(payload)
    normalized["stocks"] = normalized["targets"]
    normalized["mode"] = str(payload.get("mode") or ("portfolio" if len(normalized["targets"]) > 1 else "single"))
    normalized["preset"] = str(payload.get("preset", normalized["preset"]))
    normalized["market"] = str(payload.get("market", normalized["market"])).upper()

    for branch_name, config in (payload.get("branches") or {}).items():
        existing = normalized["branches"].setdefault(branch_name, {"enabled": True, "settings": {}})
        if "enabled" in config:
            existing["enabled"] = bool(config["enabled"])
        existing["settings"].update(config.get("settings", {}))

    legacy_keys = {
        "enable_macro": "macro",
        "enable_kronos": "kline",
        "enable_kline": "kline",
        "enable_intelligence": "intelligence",
        "enable_llm_debate": "llm_debate",
    }
    for legacy_key, branch_name in legacy_keys.items():
        if payload.get(legacy_key) is not None:
            normalized["branches"][branch_name]["enabled"] = bool(payload[legacy_key])

    risk = dict(normalized["risk"])
    risk.update(payload.get("risk", {}))
    if payload.get("capital") is not None:
        risk["capital"] = float(payload["capital"])
    if payload.get("risk_level") is not None:
        risk["risk_level"] = str(payload["risk_level"])
    normalized["risk"] = risk
    normalized["capital"] = float(risk["capital"])
    normalized["risk_level"] = str(risk["risk_level"])

    portfolio = dict(normalized["portfolio"])
    portfolio.update(payload.get("portfolio", {}))
    normalized["portfolio"] = portfolio

    llm_debate = dict(normalized["llm_debate"])
    llm_debate.update(payload.get("llm_debate", {}))
    llm_debate["enabled"] = bool(normalized["branches"]["llm_debate"]["enabled"])
    llm_debate["assignments"] = _build_llm_assignments(llm_debate)
    normalized["llm_debate"] = llm_debate

    normalized["enable_macro"] = bool(normalized["branches"]["macro"]["enabled"])
    normalized["enable_kline"] = bool(normalized["branches"]["kline"]["enabled"])
    normalized["enable_kronos"] = normalized["enable_kline"]  # 向后兼容
    normalized["enable_intelligence"] = bool(normalized["branches"]["intelligence"]["enabled"])
    normalized["enable_llm_debate"] = bool(normalized["branches"]["llm_debate"]["enabled"])
    return normalized


def _final_decision(strategy: Any, risk_summary: dict[str, Any]) -> str:
    exposure = float(strategy.target_exposure)
    risk_level = str(risk_summary.get("risk_level", "unknown"))
    if not strategy.candidate_symbols:
        return "继续观察，当前没有形成明确候选池。"
    if exposure >= 0.6:
        return f"建议积极配置候选标的，当前风险状态为 {risk_level}。"
    if exposure >= 0.3:
        return f"建议分批建仓，当前风险状态为 {risk_level}。"
    return f"建议轻仓试错或继续观察，当前风险状态为 {risk_level}。"


def run_job(payload: dict[str, Any]) -> dict[str, Any]:
    _prepare_imports()

    from quant_investor.pipeline import QuantInvestor

    normalized = _normalize_request(payload)
    analysis_id = payload.get("analysis_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
    created_at = datetime.now().isoformat(timespec="seconds")

    investor = QuantInvestor(
        stock_pool=normalized["stocks"],
        market=normalized["market"],
        lookback_years=1.0,
        total_capital=normalized["capital"],
        risk_level=normalized["risk_level"],
        enable_macro=normalized["enable_macro"],
        enable_quant=bool(normalized["branches"].get("quant", {}).get("enabled", True)),
        enable_kline=normalized["enable_kline"],
        kline_backend=normalized["branches"].get("kline", {}).get("settings", {}).get("backend", "heuristic"),
        enable_fundamental=bool(normalized["branches"].get("fundamental", {}).get("enabled", True)),
        enable_intelligence=normalized["enable_intelligence"],
        enable_agent_layer=bool(normalized["llm_debate"].get("enabled", False)),
        verbose=False,
    )
    result = investor.run()
    strategy = result.final_strategy
    risk_summary = _risk_summary(result, normalized)
    decisions = [_trade_summary(item) for item in strategy.trade_recommendations]

    branches = []
    from web.services.analysis_service import BRANCH_ORDER

    for branch_name in BRANCH_ORDER:
        branch_result = result.branch_results.get(branch_name)
        if branch_result is None:
            branches.append(
                {
                    "branch_name": branch_name,
                    "enabled": bool(normalized["branches"].get(branch_name, {}).get("enabled", False)),
                    "score": 0.0,
                    "confidence": 0.0,
                    "explanation": "本次任务未启用该分支。",
                    "risks": [],
                    "top_symbols": [],
                    "branch_mode": None,
                    "settings": normalized["branches"].get(branch_name, {}).get("settings", {}),
                    "model_assignment": normalized["llm_debate"]["assignments"] if branch_name == "llm_debate" else [],
                }
            )
            continue
        branches.append(
            _branch_summary(
                branch_result,
                normalized["branches"].get(branch_name, {}).get("settings", {}),
                normalized["llm_debate"]["assignments"],
            )
        )

    return {
        "analysis_id": analysis_id,
        "created_at": created_at,
        "source": "web",
        "request": normalized,
        "total_time": float(result.total_time),
        "research_mode": strategy.research_mode,
        "final_decision": _final_decision(strategy, risk_summary),
        "target_exposure": float(strategy.target_exposure),
        "style_bias": strategy.style_bias,
        "sector_preferences": [str(item) for item in strategy.sector_preferences],
        "candidate_symbols": [str(item) for item in strategy.candidate_symbols],
        "execution_notes": [str(item) for item in strategy.execution_notes],
        "branches": branches,
        "risk": risk_summary,
        "execution_plan": {
            "capital": float(normalized["capital"]),
            "target_exposure": float(strategy.target_exposure),
            "investable_capital": float(normalized["capital"] * strategy.target_exposure),
            "reserved_cash": float(normalized["capital"] - normalized["capital"] * strategy.target_exposure),
            "symbol_decisions": decisions,
        },
        "trade_recommendations": decisions,
        "report_markdown": result.final_report,
        "execution_log": [str(item) for item in result.execution_log[-80:]],
        "llm_assignments": normalized["llm_debate"]["assignments"],
        "config_applied": {
            "preset": normalized["preset"],
            "mode": normalized["mode"],
            "enabled_branches": [
                name
                for name, config in normalized["branches"].items()
                if config.get("enabled")
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run web analysis job")
    parser.add_argument("--payload-file", required=True)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.payload_file).read_text(encoding="utf-8"))
    result = run_job(payload)
    Path(args.output_file).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
