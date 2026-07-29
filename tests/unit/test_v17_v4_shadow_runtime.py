from __future__ import annotations

from datetime import date
from decimal import Decimal, localcontext
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import quant_investor.factors.production_control_v1 as production_control
from quant_investor.factors.production_control_v1 import (
    ProductionControlStore,
)
from quant_investor.v17_v4_contract import (
    canonical_resource_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v4_contract.schema_validation import (
    artifact_identity_field,
)
from quant_investor.v17_v4_runtime.calibration import (
    build_calibration_receipt,
    build_fusion_promotion_receipt,
)
from quant_investor.v17_v4_runtime.deep_control import (
    FusionTop24Input,
    build_fusion_top24,
)
from quant_investor.v17_v4_runtime.deep_v2 import (
    ASSESSMENT_VERSION,
    compile_deep_v2,
)
from quant_investor.v17_v4_runtime.formal_activation import (
    factor_artifact_ref,
)
from quant_investor.v17_v4_runtime.portfolio_control import (
    build_holdings_snapshot,
)
from quant_investor.v17_v4_runtime.research_quant import (
    RESEARCH_FACTOR_NAMES,
    build_research_quant_branch,
)
from quant_investor.v17_v4_runtime.shadow_runtime import (
    ShadowRuntimeError,
    publish_shadow_run,
    read_shadow_session,
)
from quant_investor.v17_v4_runtime.source_storage import (
    ExactReferenceReader,
    GovernedStore,
)
from tests.unit.test_factor_production_control_v1 import _artifacts
from tests.unit.test_v17_v4_calibration import (
    _fixture,
    _run,
    _store_v4_artifact,
)
from tests.unit.test_v17_v4_deep_control import (
    _calendar_rows,
    _pit_catalog,
    _sessions_ending,
)

NO_AUTHORITY = {
    "broker": False,
    "execution": False,
    "formal_research_publication": False,
    "order": False,
    "research_runtime_default": False,
    "trade": False,
}
STRATEGY_ID = "quant-first"
SHADOW_RUN_ID = "calibration-run-1"
CUTOFF = "2026-07-28T07:00:00Z"
SESSION = "2026-07-28"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(
    store: GovernedStore,
    *,
    path: str,
    raw: bytes,
) -> None:
    store.write_exact_once(path, raw)


def _store_artifact(
    store: GovernedStore,
    artifact: dict[str, Any],
    *,
    path: str,
) -> dict[str, str]:
    raw = canonical_resource_bytes(artifact)
    _write(store, path=path, raw=raw)
    identity_field = artifact_identity_field(artifact["version"])
    return {
        "artifact_id": artifact[identity_field],
        "artifact_version": artifact["version"],
        "byte_sha256": _sha(raw),
        "cutoff": artifact["cutoff"],
        "relative_path": path,
        "semantic_sha256": artifact["semantic_sha256"],
        "strategy_id": artifact["strategy_id"],
    }


def _factor_closure(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, str], dict[str, str]]:
    def readback(evidence: dict[str, Any]) -> dict[str, Any]:
        return {
            "complete_chain_hash_binding_verified": True,
            "context_bindings_readback_verified": True,
            "evidence": dict(evidence),
            "local_bytes_readback_verified": True,
            "quantitative_evidence_hash_binding_verified": True,
            "replay": {
                "replay_semantic_sha256": evidence["replay_semantic_sha256"],
            },
            "replay_file_sha256": evidence["replay_file_sha256"],
        }

    monkeypatch.setattr(
        production_control,
        "readback_v4_evidence",
        readback,
    )
    (
        registry,
        eligibility,
        authorization,
        transaction,
        source_artifacts,
    ) = _artifacts()
    control_root = root / "data/private/factor_governance_production_control_v1"
    store = ProductionControlStore(control_root.resolve())
    receipt = store.apply(
        transaction,
        registry=registry,
        pre_activation_eligibility=eligibility,
        authorization_receipt=authorization,
        source_artifacts=source_artifacts,
    )
    active = json.loads(store.active_set_path.read_bytes())
    active_path = store.active_set_path.relative_to(root).as_posix()
    receipt_path = (
        (control_root / "receipts/control_activations" / f"{receipt['receipt_id']}.json")
        .relative_to(root)
        .as_posix()
    )
    return (
        factor_artifact_ref(
            active,
            relative_path=active_path,
            strategy_id=STRATEGY_ID,
            cutoff=CUTOFF,
        ),
        factor_artifact_ref(
            receipt,
            relative_path=receipt_path,
            strategy_id=STRATEGY_ID,
            cutoff=CUTOFF,
        ),
    )


def _calibration_promotion(
    store: GovernedStore,
) -> tuple[dict[str, Any], dict[str, str]]:
    rows, sessions, active_cutoff, cutoff, artifacts = _fixture()
    closure = _run(
        rows,
        sessions,
        active_cutoff,
        cutoff,
        artifacts,
    )
    loader = lambda reference: artifacts[reference["byte_sha256"]]
    inventory_ref = _store_v4_artifact(
        dict(closure.origin_inventory),
        path=("data/private/v17_v4_runs/calibration-run-1/" "origin_inventory.json"),
        artifacts=artifacts,
    )
    quant = build_calibration_receipt(
        closure,
        calibration_kind="QUANT_TIMING",
        receipt_id="quant-calibration-shadow",
        created_at=cutoff,
        origin_inventory_ref=inventory_ref,
        artifact_loader=loader,
    )
    fundamental = build_calibration_receipt(
        closure,
        calibration_kind="FUNDAMENTAL_FORWARD",
        receipt_id="fundamental-calibration-shadow",
        created_at=cutoff,
        origin_inventory_ref=inventory_ref,
        artifact_loader=loader,
    )
    quant_ref = _store_v4_artifact(
        quant,
        path=("data/private/v17_v4_runs/calibration-run-1/" "quant_calibration.json"),
        artifacts=artifacts,
    )
    fundamental_ref = _store_v4_artifact(
        fundamental,
        path=("data/private/v17_v4_runs/calibration-run-1/" "fundamental_calibration.json"),
        artifacts=artifacts,
    )
    promotion = build_fusion_promotion_receipt(
        closure,
        receipt_id="fusion-promotion-shadow",
        created_at=cutoff,
        origin_inventory_ref=inventory_ref,
        quant_calibration_receipt_ref=quant_ref,
        fundamental_calibration_receipt_ref=fundamental_ref,
        artifact_loader=loader,
    )
    promotion_ref = _store_v4_artifact(
        promotion,
        path=("data/private/v17_v4_runs/calibration-run-1/" "fusion_promotion.json"),
        artifacts=artifacts,
    )
    for reference in (
        inventory_ref,
        quant_ref,
        fundamental_ref,
        promotion_ref,
    ):
        _write(
            store,
            path=reference["relative_path"],
            raw=artifacts[reference["byte_sha256"]],
        )
    return promotion, promotion_ref


def _current_model_inputs(
    root: Path,
    store: GovernedStore,
    promotion: dict[str, Any],
    promotion_ref: dict[str, str],
) -> dict[str, dict[str, str]]:
    sessions = _sessions_ending(SESSION, 30)
    calendar_rows = _calendar_rows(
        sessions,
        available_at=CUTOFF,
    )
    calendar_raw = b"canonical calendar bytes"
    calendar_path = "data/private/v17_v4_sources/cn_open_day_calendar/" "current.bin"
    _write(store, path=calendar_path, raw=calendar_raw)
    calendar_ref = {
        "artifact_id": "calendar-current",
        "artifact_version": ("myquant.v17.v4.dataset.cn_open_day_calendar.v1"),
        "byte_sha256": _sha(calendar_raw),
        "cutoff": CUTOFF,
        "relative_path": calendar_path,
        "semantic_sha256": "4" * 64,
        "strategy_id": STRATEGY_ID,
    }
    catalog = _pit_catalog(
        strategy_id=STRATEGY_ID,
        cutoff=CUTOFF,
        sessions=sessions,
        calendar_rows=calendar_rows,
        calendar_ref=calendar_ref,
    )
    market_raw = b"dataset-bytes:market_bars"
    market_ref = catalog["dataset_refs"]["market_bars"]
    assert _sha(market_raw) == market_ref["byte_sha256"]
    _write(
        store,
        path=market_ref["relative_path"],
        raw=market_raw,
    )
    catalog_ref = _store_artifact(
        store,
        catalog,
        path=("data/private/v17_v4_runs/calibration-run-1/" "pit_catalog.json"),
    )
    locator = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "cutoff": CUTOFF,
            "locator_id": "preselect-shadow-current",
            "origin": SESSION,
            "pit_catalog_ref": catalog_ref,
            "protocol_version": "myquant.v17.v4",
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.preselect-locator.v1",
        }
    )
    locator_ref = _store_artifact(
        store,
        locator,
        path=("data/private/v17_v4_runs/calibration-run-1/" "preselect_locator.json"),
    )
    pool = [f"{index:06d}.SZ" for index in range(1, 25)]
    initial = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "cutoff": CUTOFF,
            "ordered_pool": pool,
            "origin": SESSION,
            "output_id": "initial-pool-shadow-current",
            "preselect_locator_ref": locator_ref,
            "protocol_version": "myquant.v17.v4",
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.initial-pool-output.v1",
        }
    )
    initial_ref = _store_artifact(
        store,
        initial,
        path=("data/private/v17_v4_runs/calibration-run-1/" "initial_pool.json"),
    )
    dates = pd.bdate_range(end=SESSION, periods=270)
    market_rows: list[dict[str, Any]] = []
    for symbol_index, symbol in enumerate(
        [f"{index:06d}.SZ" for index in range(1, 31)],
        start=1,
    ):
        steps = np.arange(len(dates), dtype=float)
        returns = (
            0.0004
            + 0.004 * np.sin(steps / (5.0 + symbol_index / 10.0))
            + 0.002 * np.cos(steps / (11.0 + symbol_index / 7.0))
            + symbol_index * 0.000002
        )
        prices = 10.0 * np.cumprod(1.0 + returns)
        market_rows.extend(
            {
                "adj_close": float(price),
                "available_at": CUTOFF,
                "symbol": symbol,
                "trade_date": session.date().isoformat(),
            }
            for session, price in zip(dates, prices, strict=True)
        )
    stream = BytesIO()
    pd.DataFrame(market_rows).to_parquet(stream, index=False)
    research_market_raw = stream.getvalue()
    research_market_path = (
        "data/private/v17_v4_sources/research-quant/market.parquet"
    )
    _write(
        store,
        path=research_market_path,
        raw=research_market_raw,
    )
    research_market_sha = _sha(research_market_raw)
    quant_branch = build_research_quant_branch(
        initial_pool=initial,
        initial_pool_ref=initial_ref,
        market_slice_ref={
            "artifact_id": "shadow-current-quant-market-slice",
            "artifact_version": (
                "myquant.v17.v4.dataset.quant-factor-input.v1"
            ),
            "byte_sha256": research_market_sha,
            "cutoff": CUTOFF,
            "relative_path": research_market_path,
            "semantic_sha256": research_market_sha,
            "strategy_id": STRATEGY_ID,
        },
        market_slice_raw=research_market_raw,
        output_id="quant-branch-shadow-current",
    )
    branch_refs: dict[str, dict[str, str]] = {
        "quant_branch": _store_artifact(
            store,
            quant_branch,
            path=(
                "data/private/v17_v4_runs/calibration-run-1/"
                "quant_branch.json"
            ),
        )
    }
    fundamental_branch = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "branch_kind": "FUNDAMENTAL",
            "cutoff": CUTOFF,
            "initial_pool_ref": initial_ref,
            "origin": SESSION,
            "output_id": "fundamental-branch-shadow-current",
            "protocol_version": "myquant.v17.v4",
            "score_rows": [
                {
                    "score": str(index),
                    "symbol": symbol,
                }
                for index, symbol in enumerate(pool, start=1)
            ],
            "strategy_id": STRATEGY_ID,
            "version": "myquant.v17.v4.branch-output.v1",
        }
    )
    branch_refs["fundamental_branch"] = _store_artifact(
        store,
        fundamental_branch,
        path=(
            "data/private/v17_v4_runs/calibration-run-1/"
            "fundamental_branch.json"
        ),
    )
    count = Decimal(len(pool))
    quant_scores = {
        row["symbol"]: Decimal(row["score"])
        for row in quant_branch["score_rows"]
    }
    ordered_quant = sorted(quant_scores.values())
    quant_percentiles = {
        symbol: Decimal(
            sum(candidate <= score for candidate in ordered_quant)
        )
        / count
        for symbol, score in quant_scores.items()
    }
    fundamental_percentiles = {
        symbol: Decimal(index) / count
        for index, symbol in enumerate(pool, start=1)
    }
    weight = Decimal(promotion["active_quant_weight"])
    with localcontext() as context:
        context.prec = 40
        fused = {
            symbol: (
                weight * quant_percentiles[symbol]
                + (Decimal("1") - weight)
                * fundamental_percentiles[symbol]
            )
            for symbol in pool
        }
    ordered = sorted(pool, key=lambda symbol: (-fused[symbol], symbol))
    reader = ExactReferenceReader(root)
    top24 = build_fusion_top24(
        [
            FusionTop24Input(
                symbol=symbol,
                fused_score=str(fused[symbol]),
                base_target="0.03",
            )
            for symbol in ordered
        ],
        output_id="fusion-top24-shadow-current",
        run_id=SHADOW_RUN_ID,
        strategy_id=STRATEGY_ID,
        cutoff=CUTOFF,
        created_at=CUTOFF,
        promotion_receipt_ref=promotion_ref,
        artifact_loader=lambda reference: reader.read(
            reference["relative_path"],
            reference["byte_sha256"],
        ),
    )
    fusion_ref = _store_artifact(
        store,
        top24,
        path=("data/private/v17_v4_runs/calibration-run-1/" "fusion_top24.json"),
    )
    assessment = seal_semantic(
        {
            "authority": dict(NO_AUTHORITY),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "fusion_top24_ref": fusion_ref,
            "protocol_version": "myquant.v17.v4",
            "request_id": "deep-assessment-shadow-current",
            "rows": [
                {
                    "blocker_codes": ["official_evidence_unavailable"],
                    "event_flags": [],
                    "modules": [],
                    "raw_documents": [],
                    "status": "UNAVAILABLE",
                    "symbol": row["symbol"],
                }
                for row in top24["rows"]
            ],
            "run_id": SHADOW_RUN_ID,
            "strategy_id": STRATEGY_ID,
            "version": ASSESSMENT_VERSION,
        }
    )
    validate_artifact(assessment)
    assessment_ref = _store_artifact(
        store,
        assessment,
        path=("data/private/v17_v4_runs/calibration-run-1/" "deep_assessment_manifest.json"),
    )
    deep_result = compile_deep_v2(
        str(root),
        assessment_manifest_path=assessment_ref["relative_path"],
        expected_assessment_manifest_sha256=assessment_ref["byte_sha256"],
        created_at=CUTOFF,
    )
    holdings = build_holdings_snapshot(
        run_id=SHADOW_RUN_ID,
        strategy_id=STRATEGY_ID,
        cutoff=CUTOFF,
        as_of_session=SESSION,
        available_at=CUTOFF,
        nav="1000000",
        cash="1000000",
        positions=[],
    )
    holdings_ref = _store_artifact(
        store,
        holdings,
        path=("data/private/v17_v4_runs/calibration-run-1/" "holdings_snapshot.json"),
    )
    return {
        "deep_bundle": deep_result["bundle_ref"],
        "fundamental_branch": branch_refs["fundamental_branch"],
        "fusion_top24": fusion_ref,
        "holdings_snapshot": holdings_ref,
        "initial_pool": initial_ref,
        "quant_branch": branch_refs["quant_branch"],
        "source_locator": locator_ref,
    }


@pytest.fixture
def shadow_inputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("v17-v4-shadow").resolve()
    store = GovernedStore(root)
    store.initialize()
    promotion, promotion_ref = _calibration_promotion(store)
    refs = _current_model_inputs(
        root,
        store,
        promotion,
        promotion_ref,
    )
    return {"refs": refs, "root": root}


def test_missing_factor_v4_writes_only_blocker_readiness(
    tmp_path: Path,
) -> None:
    result = publish_shadow_run(
        str(tmp_path.resolve()),
        readiness_id="readiness-missing-factor",
        shadow_run_id=SHADOW_RUN_ID,
        strategy_id=STRATEGY_ID,
        cutoff=CUTOFF,
        decision_session=SESSION,
        created_at=CUTOFF,
        factor_active_set_path=None,
        factor_active_set_sha256=None,
        factor_control_receipt_path=None,
        factor_control_receipt_sha256=None,
    )
    assert result["state"] == "FACTOR_V4_BLOCKED"
    assert result["model_output_present"] is False
    assert not (tmp_path / "results/v17_v4_shadow/strategies/quant-first/runs").exists()
    assert not (tmp_path / "results/v17_v4_shadow/strategies/quant-first/sessions").exists()


def test_factor_gated_shadow_run_replays_same_pool_fusion_and_deep(
    shadow_inputs: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_ref, receipt_ref = _factor_closure(
        shadow_inputs["root"],
        monkeypatch,
    )
    refs = shadow_inputs["refs"]
    result = publish_shadow_run(
        str(shadow_inputs["root"]),
        readiness_id="readiness-shadow-complete",
        shadow_run_id=SHADOW_RUN_ID,
        strategy_id=STRATEGY_ID,
        cutoff=CUTOFF,
        decision_session=SESSION,
        created_at=CUTOFF,
        factor_active_set_path=active_ref["relative_path"],
        factor_active_set_sha256=active_ref["byte_sha256"],
        factor_control_receipt_path=receipt_ref["relative_path"],
        factor_control_receipt_sha256=receipt_ref["byte_sha256"],
        source_locator_path=refs["source_locator"]["relative_path"],
        source_locator_sha256=refs["source_locator"]["byte_sha256"],
        initial_pool_path=refs["initial_pool"]["relative_path"],
        initial_pool_sha256=refs["initial_pool"]["byte_sha256"],
        quant_branch_path=refs["quant_branch"]["relative_path"],
        quant_branch_sha256=refs["quant_branch"]["byte_sha256"],
        fundamental_branch_path=refs["fundamental_branch"]["relative_path"],
        fundamental_branch_sha256=refs["fundamental_branch"]["byte_sha256"],
        fusion_top24_path=refs["fusion_top24"]["relative_path"],
        fusion_top24_sha256=refs["fusion_top24"]["byte_sha256"],
        deep_bundle_path=refs["deep_bundle"]["relative_path"],
        deep_bundle_sha256=refs["deep_bundle"]["byte_sha256"],
        holdings_snapshot_path=refs["holdings_snapshot"]["relative_path"],
        holdings_snapshot_sha256=refs["holdings_snapshot"]["byte_sha256"],
    )
    assert result["state"] == "SHADOW_COMPLETE"
    assert result["formal_activation_eligible"] is False
    assert result["canary_evidence_eligible"] is False
    status = read_shadow_session(
        str(shadow_inputs["root"]),
        strategy_id=STRATEGY_ID,
        decision_session=SESSION,
        expected_sha256=result["session_ref"]["byte_sha256"],
    )
    assert status["shadow_run"]["model_output_present"] is True
    assert len(status["shadow_run"]["production_factor_names"]) == 5
    assert status["shadow_run"]["research_quant_factor_names"] == list(
        RESEARCH_FACTOR_NAMES
    )
    assert (
        status["shadow_run"]["deep_bundle_ref"]["artifact_version"]
        == "myquant.v17.v4.deep-evidence-bundle.v2"
    )


def test_research_factor_shadow_run_replays_without_formal_factor_refs(
    shadow_inputs: dict[str, Any],
) -> None:
    refs = shadow_inputs["refs"]
    result = publish_shadow_run(
        str(shadow_inputs["root"]),
        readiness_id="readiness-shadow-research",
        shadow_run_id=SHADOW_RUN_ID,
        strategy_id=STRATEGY_ID,
        cutoff=CUTOFF,
        decision_session=SESSION,
        created_at=CUTOFF,
        factor_active_set_path=None,
        factor_active_set_sha256=None,
        factor_control_receipt_path=None,
        factor_control_receipt_sha256=None,
        research_factor_shadow_only_override_id=(
            "operator-shadow-trio-20260728"
        ),
        source_locator_path=refs["source_locator"]["relative_path"],
        source_locator_sha256=refs["source_locator"]["byte_sha256"],
        initial_pool_path=refs["initial_pool"]["relative_path"],
        initial_pool_sha256=refs["initial_pool"]["byte_sha256"],
        quant_branch_path=refs["quant_branch"]["relative_path"],
        quant_branch_sha256=refs["quant_branch"]["byte_sha256"],
        fundamental_branch_path=refs["fundamental_branch"]["relative_path"],
        fundamental_branch_sha256=refs["fundamental_branch"]["byte_sha256"],
        fusion_top24_path=refs["fusion_top24"]["relative_path"],
        fusion_top24_sha256=refs["fusion_top24"]["byte_sha256"],
        deep_bundle_path=refs["deep_bundle"]["relative_path"],
        deep_bundle_sha256=refs["deep_bundle"]["byte_sha256"],
        holdings_snapshot_path=refs["holdings_snapshot"]["relative_path"],
        holdings_snapshot_sha256=refs["holdings_snapshot"]["byte_sha256"],
    )
    assert result["state"] == "SHADOW_COMPLETE"
    status = read_shadow_session(
        str(shadow_inputs["root"]),
        strategy_id=STRATEGY_ID,
        decision_session=SESSION,
        expected_sha256=result["session_ref"]["byte_sha256"],
    )
    run = status["shadow_run"]
    assert run["version"] == "myquant.v17.v4.shadow-run.v2"
    assert status["session"]["version"] == (
        "myquant.v17.v4.shadow-session-ref.v2"
    )
    assert run["factor_evidence_mode"] == "RESEARCH_TRIO_SHADOW_ONLY"
    assertion_ref = run["research_factor_shadow_assertion_ref"]
    assert assertion_ref == result["research_factor_shadow_assertion_ref"]
    assertion = ExactReferenceReader(shadow_inputs["root"]).read(
        assertion_ref["relative_path"],
        assertion_ref["byte_sha256"],
    )
    assertion_payload = json.loads(assertion)
    assert assertion_payload["override_id"] == "operator-shadow-trio-20260728"
    assert assertion_payload["shadow_run_id"] == SHADOW_RUN_ID
    assert assertion_payload["factor_names"] == list(RESEARCH_FACTOR_NAMES)
    assert "factor_control_active_set_ref" not in run
    assert "factor_control_receipt_ref" not in run
    assert "factor_set_sha256" not in run
    assert "production_factor_names" not in run


def test_research_factor_shadow_rejects_mixed_formal_refs(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ShadowRuntimeError,
        match="research_factor_shadow_mixed_formal_refs",
    ):
        publish_shadow_run(
            str(tmp_path.resolve()),
            readiness_id="readiness-shadow-mixed",
            shadow_run_id=SHADOW_RUN_ID,
            strategy_id=STRATEGY_ID,
            cutoff=CUTOFF,
            decision_session=SESSION,
            created_at=CUTOFF,
            factor_active_set_path="data/private/factor.json",
            factor_active_set_sha256="0" * 64,
            factor_control_receipt_path=None,
            factor_control_receipt_sha256=None,
            research_factor_shadow_only_override_id=(
                "operator-shadow-trio-20260728"
            ),
        )


@pytest.mark.parametrize(
    "strategy_id",
    (
        "Quant-First",
        "quant_first",
        "quant-first/alias",
    ),
)
def test_shadow_strategy_path_identity_rejects_aliases(
    tmp_path: Path,
    strategy_id: str,
) -> None:
    with pytest.raises((ShadowRuntimeError, ValueError)):
        publish_shadow_run(
            str(tmp_path.resolve()),
            readiness_id="readiness-bad-strategy",
            shadow_run_id=SHADOW_RUN_ID,
            strategy_id=strategy_id,
            cutoff=CUTOFF,
            decision_session=SESSION,
            created_at=CUTOFF,
            factor_active_set_path=None,
            factor_active_set_sha256=None,
            factor_control_receipt_path=None,
            factor_control_receipt_sha256=None,
        )
