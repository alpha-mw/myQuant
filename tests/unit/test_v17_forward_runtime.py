from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from quant_investor.factors.forward_evaluator import (
    annualized_rank_ic_ir,
    pearson_ic,
    spearman_rank_ic,
)
from quant_investor.v17_v4_runtime.themes import (
    ThemeExposure,
    ThemeExposureType,
    score_theme_exposure,
)
from quant_investor.v17_v4_contract import (
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v4_runtime.orchestrator import (
    ForwardEvidenceError,
    StageResult,
    publish_forward_request,
    run_forward,
)
from quant_investor.v17_v4_runtime.forward_scoring_v3 import (
    score_quant_forward_v3,
)
from quant_investor.v17_v4_runtime.source_storage import GovernedStore

CUTOFF = "2026-07-29T07:00:00Z"
SESSION = "2026-07-29"
STRATEGY_ID = "runtime-contract"
SYMBOLS = ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ")
NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}


def _ref(
    artifact_id: str,
    artifact_version: str,
    relative_path: str,
    *,
    byte_sha256: str = "a" * 64,
    semantic_sha256: str = "b" * 64,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": artifact_version,
        "byte_sha256": byte_sha256,
        "cutoff": CUTOFF,
        "relative_path": relative_path,
        "semantic_sha256": semantic_sha256,
        "strategy_id": STRATEGY_ID,
    }


def _request(
    *,
    factor_ref: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "authority": dict(NO_AUTHORITY),
        "created_at": CUTOFF,
        "cutoff": CUTOFF,
        "decision_session": SESSION,
        "factor_refs": [
            factor_ref
            or _ref(
                "factor-set",
                "myquant.v17.v4.synthetic-factor-set.v1",
                "data/private/v17_v4_sources/factor-set.json",
            )
        ],
        "protocol_version": "myquant.v17.v4",
        "request_profile": "FORWARD_EVIDENCE",
        "source_refs": [
            _ref(
                "source-snapshot",
                "myquant.v17.v4.source-snapshot.v1",
                "data/private/v17_v4_sources/source-snapshot.json",
            )
        ],
        "strategy_id": STRATEGY_ID,
    }


def _callbacks(
    *,
    source_result: Any | None = None,
) -> dict[str, Any]:
    payloads = {
        "source": (
            {"pit_valid": True, "source_snapshot": "frozen"}
            if source_result is None
            else source_result
        ),
        "allocation": {"allocation": "Core"},
        "quant": {"rows": len(SYMBOLS)},
        "factor_universe_observation": {"observed": True},
        "fusion": {"rows": len(SYMBOLS)},
        "strategy_pool_observation": {"observed": True},
        "final": {"forward_label": "PENDING"},
    }
    return {stage: (lambda _context, value=value: value) for stage, value in payloads.items()}


def _materialize_request_refs(
    workspace: Path,
    request: dict[str, Any],
) -> dict[str, Any]:
    for field in ("source_refs", "factor_refs"):
        for reference in request[field]:
            target = workspace / reference["relative_path"]
            if target.exists():
                raw = target.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == reference["byte_sha256"]
                continue
            document = seal_semantic(
                {
                    "bound_artifact_id": reference["artifact_id"],
                    "cutoff": reference["cutoff"],
                    "strategy_id": reference["strategy_id"],
                    "version": reference["artifact_version"],
                }
            )
            raw = canonical_resource_bytes(document)
            GovernedStore(workspace).write_exact_once(
                reference["relative_path"],
                raw,
            )
            reference["byte_sha256"] = hashlib.sha256(raw).hexdigest()
            reference["semantic_sha256"] = document["semantic_sha256"]
    return request


def _run(
    workspace: Path,
    *,
    callbacks: dict[str, Any] | None = None,
    request: dict[str, Any] | None = None,
    factor_pointer_reread: Any = lambda: True,
    event_hook: Any | None = None,
) -> dict[str, Any]:
    bound_request = _materialize_request_refs(
        workspace,
        request or _request(),
    )
    published = publish_forward_request(workspace, bound_request)
    return run_forward(
        workspace,
        request_path=published["request_path"],
        request_sha256=published["request_sha256"],
        stage_callbacks=callbacks or _callbacks(),
        factor_pointer_reread=factor_pointer_reread,
        event_hook=event_hook,
    )


def _neutralizers() -> dict[str, dict[str, dict[str, Any]]]:
    return {
        symbol: {
            "industry": {
                "available_at": CUTOFF,
                "value": f"industry-{index % 2}",
            },
            "log_market_cap": {
                "available_at": CUTOFF,
                "value": 10 + index,
            },
            "beta_252d": {
                "available_at": CUTOFF,
                "value": 0.8 + index * 0.1,
            },
            "amihud_20d": {
                "available_at": CUTOFF,
                "value": 0.01 + index * 0.01,
            },
        }
        for index, symbol in enumerate(SYMBOLS)
    }


def _stage_receipts(
    workspace: Path,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    run = json.loads((workspace / result["run_ref"]["relative_path"]).read_text())
    return [
        json.loads((workspace / reference["relative_path"]).read_text())
        for reference in run["stage_receipt_refs"]
    ]


def test_quant_can_run_independently() -> None:
    result = score_quant_forward_v3(
        symbols=SYMBOLS,
        selected_factors=(
            {"family": "quality", "name": "quality-a"},
            {"family": "value", "name": "value-a"},
        ),
        factor_values={
            "quality-a": {symbol: index for index, symbol in enumerate(SYMBOLS)},
            "value-a": {symbol: 10 - index for index, symbol in enumerate(SYMBOLS)},
        },
        neutralizer_inputs=_neutralizers(),
        cutoff=CUTOFF,
    )

    assert len(result["records"]) == len(SYMBOLS)
    assert all(row["status"] == "AVAILABLE" for row in result["records"])


def test_fundamental_missing_does_not_block(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result["global_activation_state"] == "INACTIVE"
    assert result["run_state"] == "FORWARD_EVIDENCE_ACTIVE"
    assert result["research_runtime_default"] is False
    assert result["formal_activation_eligible"] is False
    fundamental = next(
        row for row in _stage_receipts(tmp_path, result) if row["stage_id"] == "fundamental"
    )
    assert fundamental["execution_outcome"] == "SKIPPED"
    assert fundamental["completeness"] == "UNAVAILABLE"


def test_deep_missing_does_not_block_observation(tmp_path: Path) -> None:
    result = _run(tmp_path)

    assert result["global_activation_state"] == "INACTIVE"
    assert result["run_state"] == "FORWARD_EVIDENCE_ACTIVE"
    stages = {row["stage_id"]: row for row in _stage_receipts(tmp_path, result)}
    assert stages["deep"]["completeness"] == "UNAVAILABLE"
    assert stages["factor_universe_observation"]["completeness"] == "COMPLETE"
    assert stages["strategy_pool_observation"]["completeness"] == "COMPLETE"


def test_pit_violation_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ForwardEvidenceError, match="stage_contract"):
        _run(
            tmp_path,
            callbacks=_callbacks(
                source_result=StageResult(
                    payload={"pit_valid": False},
                    pit_valid=False,
                )
            ),
        )


def test_future_data_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ForwardEvidenceError, match="stage_contract"):
        _run(
            tmp_path,
            callbacks=_callbacks(
                source_result=StageResult(
                    payload={"future_data_present": True},
                    future_data_present=True,
                )
            ),
        )


def test_factor_pointer_is_frozen_and_reread(tmp_path: Path) -> None:
    document = seal_semantic(
        {
            "cutoff": CUTOFF,
            "factor_set_id": "factor-set",
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.synthetic-factor-set.v1",
        }
    )
    raw = canonical_resource_bytes(document)
    relative_path = "data/private/v17_v4_sources/factor-set.json"
    target = tmp_path / relative_path
    GovernedStore(tmp_path).write_exact_once(relative_path, raw)
    factor_ref = _ref(
        "factor-set",
        str(document["version"]),
        relative_path,
        byte_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=str(document["semantic_sha256"]),
    )

    def tamper(event: str, _context: Any) -> None:
        if event == "after_run":
            target.write_bytes(raw.replace(b"factor-set", b"factor-drift", 1))

    with pytest.raises(ForwardEvidenceError, match="factor_pointer_reread"):
        _run(
            tmp_path,
            request=_request(factor_ref=factor_ref),
            factor_pointer_reread=None,
            event_hook=tamper,
        )


def test_factor_evaluation_is_reproducible() -> None:
    exposures = [0.1, 0.3, 0.2, 0.9]
    labels = [0.0, 0.4, 0.1, 0.8]

    first = (
        pearson_ic(exposures, labels),
        spearman_rank_ic(exposures, labels),
        annualized_rank_ic_ir([0.01, 0.02, 0.03]),
    )
    second = (
        pearson_ic(exposures, labels),
        spearman_rank_ic(exposures, labels),
        annualized_rank_ic_ir([0.01, 0.02, 0.03]),
    )

    assert first == second


def test_theme_evidence_binding_is_required() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ThemeExposure(
            symbol="000001.SZ",
            theme_id="ai-infrastructure",
            exposure_type=ThemeExposureType.DIRECT_BENEFICIARY,
            revenue_exposure=0.8,
            product_exposure=None,
            customer_exposure=None,
            supply_chain_position="upstream",
            confidence=0.9,
            evidence_refs=(),
        )

    exposure = ThemeExposure(
        symbol="000001.SZ",
        theme_id="ai-infrastructure",
        exposure_type=ThemeExposureType.DIRECT_BENEFICIARY,
        revenue_exposure=0.8,
        product_exposure=None,
        customer_exposure=None,
        supply_chain_position="upstream",
        confidence=0.9,
        evidence_refs=("evidence/theme-a.json",),
    )
    assert score_theme_exposure(exposure).score > 0
