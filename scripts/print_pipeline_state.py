#!/usr/bin/env python3
"""Print a read-only local CN aggressive pipeline state snapshot.

This is the required first step before any AI thread answers buy/sell
questions. It uses only local artifacts and does not call providers, LLMs,
brokers, or execution APIs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORD_ROOT = (
    PROJECT_ROOT / "results" / "strategy_records" / "CN" / "aggressive_tech_manufacturing"
)
DEFAULT_REGIME_HISTORY = PROJECT_ROOT / "results" / "regime" / "markov_regime_history.jsonl"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {str(key or "").strip(): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value if value is not None else "").replace(",", "").strip()
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value if value is not None else "").replace(",", "").strip()
        return int(float(text)) if text else default
    except (TypeError, ValueError):
        return default


def _latest_record(root: Path) -> Path:
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and (path / "pnl_summary.csv").exists()
    ] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"no strategy records found under {root}")
    return sorted(candidates, key=lambda path: path.name)[-1]


def _latest_regime(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    if not rows:
        return {}
    return sorted(rows, key=lambda row: str(row.get("as_of") or ""))[-1]


def _effective_holdings(record_dir: Path) -> tuple[list[dict[str, str]], str]:
    for filename in ("ledger_after_manual_switch.csv", "holdings_review.csv"):
        rows = [
            row for row in _read_csv(record_dir / filename)
            if _safe_int(row.get("shares") or row.get("shares_before")) > 0
        ]
        if rows:
            return rows, filename
    return [], ""


def _exposure_components(
    record_dir: Path,
    pnl: dict[str, str],
    effective_rows: list[dict[str, str]],
    effective_source: str,
) -> dict[str, Any]:
    total_value = _safe_float(pnl.get("total_value_after"))
    review_rows = [
        row for row in _read_csv(record_dir / "holdings_review.csv")
        if _safe_int(row.get("shares") or row.get("shares_before")) > 0
    ]
    review_value = sum(_safe_float(row.get("current_value")) for row in review_rows)
    effective_value = sum(_safe_float(row.get("current_value")) for row in effective_rows)
    return {
        "denominator_total_value_after": total_value,
        "pnl_market_value_after": _safe_float(pnl.get("market_value_after")),
        "pnl_cash_after": _safe_float(pnl.get("cash_after")),
        "phase13_effective_source": effective_source,
        "phase13_effective_symbols": [row.get("symbol") for row in effective_rows if row.get("symbol")],
        "phase13_market_value_numerator": effective_value,
        "phase13_exposure": effective_value / total_value if total_value > 0 else None,
        "holdings_review_symbols": [row.get("symbol") for row in review_rows if row.get("symbol")],
        "holdings_review_market_value_numerator": review_value,
        "holdings_review_exposure": review_value / total_value if total_value > 0 else None,
        "difference_symbols_review_not_effective": sorted(
            set(row.get("symbol") for row in review_rows if row.get("symbol"))
            - set(row.get("symbol") for row in effective_rows if row.get("symbol"))
        ),
        "conclusion": (
            "effective ledger exposure is authoritative; holdings_review is review-state metadata and can include pre-manual-switch rows"
            if effective_source == "ledger_after_manual_switch.csv"
            else "fallback exposure source used because effective ledger was unavailable"
        ),
    }


def _shortlist_diagnosis(
    candidate_rows: list[dict[str, str]],
    theme_audit: dict[str, Any],
    market_snapshot: dict[str, Any],
) -> dict[str, Any]:
    if candidate_rows:
        return {"status": "available", "conclusion": "shortlist present"}
    summary = theme_audit.get("summary") if isinstance(theme_audit.get("summary"), dict) else {}
    blocker = market_snapshot.get("blocker") or market_snapshot.get("candidate_generation_status")
    if blocker == "theme_pool_hard_filter_regression" or _safe_int(summary.get("residual_symbol_count")) > 0:
        conclusion = "shortlist=N/A is a guardrail/blocker outcome, not a normal regime-gate empty pool."
        status = "guardrail_blocked"
    elif market_snapshot.get("candidate_generation_status") == "empty":
        conclusion = "shortlist=N/A is a normal empty candidate set from the production pipeline."
        status = "empty_normal"
    else:
        conclusion = "shortlist=N/A cause is undetermined from local artifacts; inspect candidate generation diagnostics."
        status = "unknown"
    return {
        "status": status,
        "candidate_generation_status": market_snapshot.get("candidate_generation_status"),
        "blocker": market_snapshot.get("blocker"),
        "theme_pool_status": summary.get("status") or summary.get("theme_pool_status"),
        "core_symbol_count": summary.get("core_symbol_count"),
        "residual_symbol_count": summary.get("residual_symbol_count"),
        "policy_regime": summary.get("policy_regime"),
        "conclusion": conclusion,
    }


def build_state(
    record_root: Path = DEFAULT_RECORD_ROOT,
    regime_history: Path = DEFAULT_REGIME_HISTORY,
    record_id: str | None = None,
) -> dict[str, Any]:
    record_dir = record_root / record_id if record_id else _latest_record(record_root)
    if not record_dir.exists():
        raise FileNotFoundError(f"strategy record not found: {record_dir}")
    pnl_rows = _read_csv(record_dir / "pnl_summary.csv")
    pnl = pnl_rows[-1] if pnl_rows else {}
    effective_holdings, effective_source = _effective_holdings(record_dir)
    review_by_symbol = {row.get("symbol"): row for row in _read_csv(record_dir / "holdings_review.csv") if row.get("symbol")}
    candidates = _read_csv(record_dir / "candidate_pool.csv")
    market_snapshot = _read_json(record_dir / "market_snapshot.json")
    theme_audit = _read_json(record_dir / "theme_pool_audit.json")
    theme_snapshot = _read_json(record_dir / "theme_snapshot.json")
    regime = _latest_regime(regime_history)
    total_value = _safe_float(pnl.get("total_value_after"))
    market_value = sum(_safe_float(row.get("current_value")) for row in effective_holdings)
    return {
        "record": record_dir.name,
        "record_path": str(record_dir),
        "regime": {
            "as_of": regime.get("as_of"),
            "dominant_regime": regime.get("dominant_regime"),
            "suggested_gross_exposure_cap": regime.get("suggested_gross_exposure_cap"),
            "transition_risk": regime.get("transition_risk"),
            "scope": regime.get("regime_scope"),
        },
        "actual_total_exposure": market_value / total_value if total_value > 0 else None,
        "exposure_components": _exposure_components(record_dir, pnl, effective_holdings, effective_source),
        "holdings": [
            {
                "symbol": row.get("symbol"),
                "name": row.get("name"),
                "weight": _safe_float(row.get("market_weight")),
                "stage_stop_price": _safe_float(row.get("stage_stop_price") or review_by_symbol.get(row.get("symbol"), {}).get("stage_stop_price")),
                "recommended_action": review_by_symbol.get(row.get("symbol"), {}).get("recommended_action") or row.get("recommended_action"),
                "theme_phase": (
                    review_by_symbol.get(row.get("symbol"), {}).get("theme_phase")
                    or review_by_symbol.get(row.get("symbol"), {}).get("primary_theme_phase")
                    or row.get("theme_phase")
                    or row.get("primary_theme_phase")
                    or ""
                ),
                "crowding_flags": (
                    review_by_symbol.get(row.get("symbol"), {}).get("crowding_flags")
                    or review_by_symbol.get(row.get("symbol"), {}).get("risk_flags")
                    or row.get("crowding_flags")
                    or row.get("risk_flags")
                    or ""
                ),
            }
            for row in effective_holdings
            if row.get("symbol")
        ],
        "candidate_head": [
            {
                "symbol": row.get("symbol"),
                "name": row.get("name"),
                "rank": row.get("candidate_rank"),
                "target_weight": _safe_float(row.get("portfolio_target_weight")),
                "score": _safe_float(row.get("codex_recommendation_score")),
            }
            for row in candidates[:5]
        ],
        "theme_snapshot_status": theme_snapshot.get("status") or theme_snapshot.get("metadata", {}).get("status"),
        "shortlist_diagnosis": _shortlist_diagnosis(candidates, theme_audit, market_snapshot),
    }


def render_state(state: dict[str, Any]) -> str:
    lines = [
        f"record: {state['record']}",
        (
            "regime: "
            f"{state['regime'].get('dominant_regime') or 'N/A'} "
            f"as_of={state['regime'].get('as_of') or 'N/A'} "
            f"gross_cap={state['regime'].get('suggested_gross_exposure_cap')}"
        ),
        (
            "actual_total_exposure: "
            f"{state.get('actual_total_exposure')} "
            f"source={state.get('exposure_components', {}).get('phase13_effective_source')} "
            f"numerator={state.get('exposure_components', {}).get('phase13_market_value_numerator')} "
            f"denominator={state.get('exposure_components', {}).get('denominator_total_value_after')}"
        ),
        "holdings:",
    ]
    for row in state.get("holdings", []):
        lines.append(
            "  - "
            f"{row.get('symbol')} {row.get('name')} "
            f"weight={row.get('weight')} stop={row.get('stage_stop_price')} "
            f"action={row.get('recommended_action')} theme_phase={row.get('theme_phase') or 'N/A'} "
            f"crowding={row.get('crowding_flags') or 'N/A'}"
        )
    lines.append("shortlist_head:")
    for row in state.get("candidate_head", []):
        lines.append(
            "  - "
            f"{row.get('rank')} {row.get('symbol')} {row.get('name')} "
            f"target_weight={row.get('target_weight')} score={row.get('score')}"
        )
    if not state.get("candidate_head"):
        lines.append("  - N/A")
    diagnosis = state.get("shortlist_diagnosis", {})
    lines.append(f"shortlist_diagnosis: {diagnosis.get('conclusion')}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--regime-history", type=Path, default=DEFAULT_REGIME_HISTORY)
    parser.add_argument("--record-id")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    state = build_state(args.record_root, args.regime_history, args.record_id)
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_state(state))


if __name__ == "__main__":
    main()
