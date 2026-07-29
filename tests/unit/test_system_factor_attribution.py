from __future__ import annotations

import json
import os
from pathlib import Path

from quant_investor.monitoring.system_factor_attribution import (
    ATTRIBUTION_BUCKETS,
    build_system_factor_attribution,
    render_system_factor_attribution_markdown,
    system_factor_attribution_bytes,
    system_factor_attribution_reference,
)


def _write_json(path: Path, payload: dict[str, object], *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.chmod(path, mode)


def _readiness_payload() -> dict[str, object]:
    return {
        "schema_version": "factor-governance-readiness.v4",
        "protocol_version": "v4",
        "protocol_hash": "a" * 64,
        "as_of": "2026-07-28",
        "status": "no_new_risk",
        "factor_governance_ready": False,
        "new_risk_eligible": False,
        "new_risk_authorized": False,
        "production_apply_enabled": False,
        "production_apply_blocker": "factor_v4_production_apply_not_authorized",
        "production_factor_target": 10,
        "production_factor_count": 2,
        "healthy_factor_count": 2,
        "production_family_count": 2,
        "factors": [
            {
                "name": "quality_alpha",
                "family": "quality",
                "health_status": "healthy",
            },
            {
                "name": "momentum_alpha",
                "family": "momentum",
                "health_status": "healthy",
            },
        ],
        "normalized_abs_weights": {
            "quality_alpha": 0.6,
            "momentum_alpha": 0.4,
        },
        "family_normalized_abs_weights": {
            "quality": 0.6,
            "momentum": 0.4,
        },
        "blockers": ["production_factor_count_below_5"],
    }


def _write_manual_manifest(
    base_dir: Path,
    *,
    run_id: str,
    symbol: str,
    name: str,
    action: str,
    pnl: float,
    reason: str,
) -> None:
    _write_json(
        base_dir / run_id / "manual_execution_manifest.json",
        {
            "schema_version": "cn_aggressive_manual_execution.v2",
            "recorded_at": f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]} 15:00:00 CST",
            "applied_local_trades": [
                {
                    "symbol": symbol,
                    "name": name,
                    "action": action,
                    "realized_pnl": pnl,
                    "reason": reason,
                }
            ],
            "no_broker_api_called": True,
        },
    )


def test_builds_report_only_history_and_factor_v4_analysis(tmp_path: Path) -> None:
    base_dir = tmp_path / "strategy_records"
    _write_manual_manifest(
        base_dir,
        run_id="20260701_1500",
        symbol="600001.SH",
        name="测试一",
        action="sell",
        pnl=100.0,
        reason="portfolio rebalance target weight",
    )
    _write_manual_manifest(
        base_dir,
        run_id="20260702_1500",
        symbol="600002.SH",
        name="测试二",
        action="clear_risk_sell",
        pnl=-50.0,
        reason="risk_control stop loss",
    )
    _write_manual_manifest(
        base_dir,
        run_id="20260703_1500",
        symbol="600003.SH",
        name="测试三",
        action="sell_for_switch_now",
        pnl=25.0,
        reason="factor_signal documented exit",
    )
    _write_manual_manifest(
        base_dir,
        run_id="20260704_1500",
        symbol="600004.SH",
        name="测试四",
        action="sell",
        pnl=0.0,
        reason="manual correction",
    )
    _write_json(
        base_dir / "20260705_1500" / "manual_execution_manifest.json",
        {
            "schema_version": "cn_aggressive_manual_execution.v3",
            "financial_state_sha256": "b" * 64,
            "ledger_after_manual_switch_csv_sha256": "c" * 64,
            "applied_local_trades": [
                {
                    "symbol": "600005.SH",
                    "name": "测试五",
                    "action": "sell",
                    "realized_pnl": 999.0,
                    "reason": "portfolio rebalance",
                }
            ],
        },
    )
    readiness_path = tmp_path / "results" / "factor_governance" / "readiness.json"
    _write_json(readiness_path, _readiness_payload(), mode=0o600)

    payload = build_system_factor_attribution(
        project_root=tmp_path,
        base_dir=base_dir,
        run_id="20260728_0900",
        generated_at="2026-07-28T09:00:00+08:00",
        trade_date="2026-07-28",
        analysis_trade_date="2026-07-27",
        completeness_passed=False,
        decision_data_sufficient=False,
        action_taken_today=False,
        execution_rejections=[],
        factor_readiness_path=readiness_path,
    )

    history = payload["historical_trade_attribution"]
    assert history["realized_sell_count"] == 4
    assert history["evidence_quality"] == "legacy_manifest_limited"
    assert history["manifest_integrity_levels"] == {"legacy_unsealed_manifest": 4}
    assert history["skipped_reasons"]["v3_manifest_financial_seal_invalid"] == 1
    assert set(history["buckets"]) == set(ATTRIBUTION_BUCKETS)
    assert history["buckets"]["portfolio_decision"]["realized_pnl"] == 100.0
    assert history["buckets"]["risk_control"]["realized_pnl"] == -50.0
    assert history["buckets"]["factor_signal"]["realized_pnl"] == 25.0
    assert history["buckets"]["insufficient_evidence"]["flat_count"] == 1
    factor = payload["factor_v4_analysis"]
    assert factor["status"] == "available"
    assert factor["production_factor_count"] == 2
    assert [row["name"] for row in factor["factors"]] == [
        "quality_alpha",
        "momentum_alpha",
    ]
    assert factor["contribution_evidence"]["status"] == "unavailable"
    assert factor["legacy_or_shadow_fallback_used"] is False
    assert payload["today_diagnosis"]["primary_driver"] == "market_data_staleness"
    assert payload["source_policy"]["ledger_csv_fallback_used"] is False
    assert payload["execution_boundary"] == {
        "broker_connected_by_analysis": False,
        "order_created_by_analysis": False,
        "trade_executed_by_analysis": False,
    }

    reference = system_factor_attribution_reference(payload)
    assert len(reference["sha256"]) == 64
    assert system_factor_attribution_bytes(payload).endswith(b"\n")
    factor_lines, history_lines = render_system_factor_attribution_markdown(payload)
    factor_text = "\n".join(factor_lines)
    history_text = "\n".join(history_lines)
    assert "quality_alpha" in factor_text
    assert "authoritative_factor_v4_contribution_evidence_missing" in factor_text
    assert "600002.SH 测试二" in history_text
    assert "单一笔或少量交易" not in history_text
    assert "同一笔或少量交易不能证明" in history_text


def test_missing_factor_v4_readiness_fails_closed_without_fallback(
    tmp_path: Path,
) -> None:
    payload = build_system_factor_attribution(
        project_root=tmp_path,
        base_dir=tmp_path / "missing_records",
        run_id="20260728_0900",
        generated_at="2026-07-28T09:00:00+08:00",
        trade_date="2026-07-28",
        analysis_trade_date="2026-07-27",
        completeness_passed=True,
        decision_data_sufficient=True,
        action_taken_today=False,
        execution_rejections=[],
    )

    factor = payload["factor_v4_analysis"]
    assert factor["status"] == "unavailable_fail_closed"
    assert factor["factor_verdict"] == "insufficient_evidence"
    assert factor["factors"] == []
    assert factor["legacy_or_shadow_fallback_used"] is False
    assert payload["today_diagnosis"]["primary_driver"] == "insufficient_evidence"


def test_execution_rejection_is_today_driver_when_data_is_ready(
    tmp_path: Path,
) -> None:
    readiness_path = tmp_path / "results" / "factor_governance" / "readiness.json"
    _write_json(readiness_path, _readiness_payload(), mode=0o600)

    payload = build_system_factor_attribution(
        project_root=tmp_path,
        base_dir=tmp_path / "strategy_records",
        run_id="20260728_0900",
        generated_at="2026-07-28T09:00:00+08:00",
        trade_date="2026-07-28",
        analysis_trade_date="2026-07-27",
        completeness_passed=True,
        decision_data_sufficient=True,
        action_taken_today=False,
        execution_rejections=[{"reason": "fresh_quote_missing"}],
        factor_readiness_path=readiness_path,
    )

    assert payload["today_diagnosis"]["primary_driver"] == "execution"
    assert payload["today_diagnosis"]["execution_rejection_count"] == 1
