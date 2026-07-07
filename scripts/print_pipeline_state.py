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


def build_state(record_root: Path = DEFAULT_RECORD_ROOT, regime_history: Path = DEFAULT_REGIME_HISTORY) -> dict[str, Any]:
    record_dir = _latest_record(record_root)
    pnl_rows = _read_csv(record_dir / "pnl_summary.csv")
    pnl = pnl_rows[-1] if pnl_rows else {}
    holdings = _read_csv(record_dir / "holdings_review.csv") or _read_csv(record_dir / "ledger_after_manual_switch.csv")
    candidates = _read_csv(record_dir / "candidate_pool.csv")
    theme_snapshot = _read_json(record_dir / "theme_snapshot.json")
    regime = _latest_regime(regime_history)
    total_value = _safe_float(pnl.get("total_value_after"))
    market_value = sum(_safe_float(row.get("current_value")) for row in holdings)
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
        "holdings": [
            {
                "symbol": row.get("symbol"),
                "name": row.get("name"),
                "weight": _safe_float(row.get("market_weight")),
                "stage_stop_price": _safe_float(row.get("stage_stop_price")),
                "recommended_action": row.get("recommended_action"),
                "theme_phase": row.get("theme_phase") or row.get("primary_theme_phase") or "",
                "crowding_flags": row.get("crowding_flags") or row.get("risk_flags") or "",
            }
            for row in holdings
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
        f"actual_total_exposure: {state.get('actual_total_exposure')}",
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
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--regime-history", type=Path, default=DEFAULT_REGIME_HISTORY)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    state = build_state(args.record_root, args.regime_history)
    if args.json:
        print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(render_state(state))


if __name__ == "__main__":
    main()
