from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_investor.factors.governance_protocol_v4 import (
    assess_factor_governance_readiness_v4,
    protocol_hash,
    semantic_sha256,
)
from quant_investor.factors.governance_transaction_v4 import (
    activation_receipt_sha256,
)
from quant_investor.factors.runtime import production_factor_set_sha256
from quant_investor.v16.runtime import (
    V16Stage1RuntimeError,
    build_stage1_package_from_market_context,
    prepare_v16_stage1_pending,
)

NOW = datetime(2026, 7, 17, 8, 0, tzinfo=timezone.utc)


def _factor_readiness() -> dict[str, object]:
    records = []
    for index in range(5):
        name = f"factor_{index}"
        runtime_contract = {"name": name, "formula": f"primitive_{index}"}
        records.append(
            {
                "name": name,
                "family": f"family_{index}",
                "slot": f"family_{index}::slot",
                "state": "production_factor",
                "weight": 1.0,
                "gate_results": {str(gate): True for gate in range(1, 9)},
                "maturity": {
                    "month_end_rankic_dates": [f"2025-{month:02d}-28" for month in range(1, 13)],
                    "forward_cohorts": [],
                },
                "bh_q_value": 0.05,
                "fdr_method": "benjamini_hochberg_by_family",
                "runtime_contract": runtime_contract,
                "runtime_contract_sha256": semantic_sha256(runtime_contract),
                "runtime_contract_status": "verified",
                "evidence": {
                    "schema_version": "factor-governance-replay-evidence.v4",
                    "status": "verified",
                    "replay_semantic_sha256": sha256(name.encode()).hexdigest(),
                },
                "health": {
                    "status": "healthy",
                    "fresh": True,
                    "data_blocked": False,
                },
            }
        )
    registry_sha = "b" * 64
    factor_set_sha = production_factor_set_sha256(sorted(record["name"] for record in records))
    runtime_sha = semantic_sha256(sorted(record["runtime_contract_sha256"] for record in records))
    context = {
        "protocol_hash": protocol_hash(),
        "transaction_plan_sha256": "c" * 64,
        "registry_file_sha256": registry_sha,
        "production_factor_set_sha256": factor_set_sha,
        "runtime_contracts_sha256": runtime_sha,
        "as_of": "2026-07-17",
    }
    receipt = {
        "schema_version": "factor-governance-activation-receipt.v4",
        "protocol_version": "v4",
        "protocol_hash": protocol_hash(),
        "receipt_id": "receipt-runtime-test",
        "status": "activated",
        "authorization_scope": "factor_v4_production_activation",
        "authorized_by": "Maxwell",
        "activated_at": "2026-07-17T09:00:00+08:00",
        "as_of": "2026-07-17",
        "transaction_plan_sha256": "c" * 64,
        "registry_file_sha256": registry_sha,
        "production_factor_set_sha256": factor_set_sha,
        "runtime_contracts_sha256": runtime_sha,
        "activation_context_sha256": semantic_sha256(context),
        "activation_performed": True,
    }
    receipt["receipt_sha256"] = activation_receipt_sha256(receipt)
    return assess_factor_governance_readiness_v4(
        records,
        as_of="2026-07-17",
        registry_file_sha256=registry_sha,
        production_factor_set_sha256=factor_set_sha,
        activation_receipt=receipt,
    )


def _context(pointer: Path, *, count: int = 2, funnel_count: int = 1):
    symbols = [f"{index:06d}.SZ" for index in range(count)]
    scores = {
        symbol: -1.0 + (2.0 * index / max(count - 1, 1)) for index, symbol in enumerate(symbols)
    }
    return SimpleNamespace(
        researchable_symbols=symbols,
        candidate_symbols=symbols[-funnel_count:],
        quant_result=SimpleNamespace(
            success=True,
            symbol_scores=scores,
            signals={"factor_contract": "v4-receipt-bound"},
        ),
        branch_data_readiness={
            "readiness": {
                branch: {"status": "pass"} for branch in ("quant", "fundamental", "macro")
            }
        },
        branch_data_payload={
            "fundamentals": {symbol: {"as_of": "2026-07-17", "quality": 0.5} for symbol in symbols},
            "macro_data": {"as_of": "2026-07-17", "regime": "neutral"},
        },
        global_context=SimpleNamespace(
            metadata={
                "symbol_market_state": {
                    symbol: {"latest_close": 10.0 + index} for index, symbol in enumerate(symbols)
                }
            }
        ),
        resolver_snapshot={"latest_pointer_path": str(pointer)},
    )


def test_full_market_fact_package_is_after_quant_and_funnel(tmp_path: Path) -> None:
    pointer = tmp_path / "_latest.json"
    pointer.write_text('{"generation":"g1"}\n', encoding="utf-8")
    context = _context(pointer, count=601, funnel_count=500)
    package = build_stage1_package_from_market_context(
        context_state=context,
        market="CN",
        mode="batch",
        pit_pointer_sha256=sha256(pointer.read_bytes()).hexdigest(),
        factor_readiness=_factor_readiness(),
        cutoff_at=NOW,
        expires_at=NOW + timedelta(hours=24),
    )

    assert len(package.rows) == 601
    assert len(package.funnel_symbols) == 500
    assert package.rows[0].formal_quant_score == -1.0
    assert set(package.stratum_counts) == {
        "quant_quintile_1",
        "quant_quintile_2",
        "quant_quintile_3",
        "quant_quintile_4",
        "quant_quintile_5",
    }


def test_formal_stage1_fails_closed_without_v4_factor_receipt(tmp_path: Path) -> None:
    pointer = tmp_path / "_latest.json"
    pointer.write_text("{}\n", encoding="utf-8")
    factor = _factor_readiness()
    factor["activation_receipt"] = {"valid": False}
    with pytest.raises(V16Stage1RuntimeError, match="activation_receipt"):
        build_stage1_package_from_market_context(
            context_state=_context(pointer),
            market="CN",
            mode="batch",
            pit_pointer_sha256=sha256(pointer.read_bytes()).hexdigest(),
            factor_readiness=factor,
            cutoff_at=NOW,
            expires_at=NOW + timedelta(hours=24),
        )


def test_no_agent_layer_is_diagnostic_and_does_not_read_factor_artifact(
    tmp_path: Path,
) -> None:
    pointer = tmp_path / "_latest.json"
    pointer.write_text("{}\n", encoding="utf-8")
    result = prepare_v16_stage1_pending(
        context_state=_context(pointer),
        market="CN",
        mode="batch",
        enable_agent_layer=False,
        factor_readiness_path=tmp_path / "missing.json",
        model_id="codex",
        now=NOW,
    )
    assert result["status"] == "diagnostic_only"
    assert result["formal_shortlist_generated"] is False
    assert result["new_risk_authorized"] is False


def test_agent_layer_creates_only_s1_prepared_pending_run(tmp_path: Path) -> None:
    pointer = tmp_path / "_latest.json"
    pointer.write_text('{"generation":"g1"}\n', encoding="utf-8")
    factor_path = tmp_path / "factor-readiness.json"
    factor_path.write_text(
        json.dumps(_factor_readiness(), ensure_ascii=False),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.json"
    config_path.write_text('{"architecture":"16.0.0"}\n', encoding="utf-8")
    prompt_path = tmp_path / "prompt.md"
    prompt_path.write_text("sealed local PIT facts only\n", encoding="utf-8")

    result = prepare_v16_stage1_pending(
        context_state=_context(pointer),
        market="CN",
        mode="batch",
        enable_agent_layer=True,
        factor_readiness_path=factor_path,
        review_root=tmp_path / "review",
        config_path=config_path,
        prompt_path=prompt_path,
        model_id="codex-test",
        repo_path=Path(__file__).resolve().parents[2],
        run_id="run-stage1",
        now=NOW,
    )

    assert result["status"] == "pending_codex_stage1"
    assert result["review"]["state"] == "S1_PREPARED"
    assert result["formal_shortlist_generated"] is False
    assert result["new_risk_authorized"] is False
    request_path = tmp_path / "review" / "run-stage1" / result["review"]["stage1_request_path"]
    assert request_path.is_file()
    assert request_path.stat().st_mode & 0o777 == 0o600
