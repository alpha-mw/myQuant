from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
from io import BytesIO
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

from quant_investor.v17_v3_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
)
from quant_investor.v17_v3_contract.resources import load_packaged_json
from quant_investor.v17_v3_runtime.artifacts import (
    RuntimeArtifact,
    load_typed_artifact,
    runtime_artifact,
    seal_typed_artifact,
    write_typed_exact_once,
)
from quant_investor.v17_v3_runtime.authority import (
    PROTOCOL_VERSION,
    authority_envelope,
)
from quant_investor.v17_v3_runtime.pipeline import (
    FORMAL_RESEARCH_MODE,
    PipelineRequest,
    SHADOW_MODE,
    run_pipeline,
)
from quant_investor.v17_v3_runtime.redaction import (
    PublicEnvelopeError,
    assert_public_envelope_safe,
    redact_public,
)
from quant_investor.v17_v3_runtime.service import analyze, build_initial_pool
from quant_investor.v17_v3_runtime.sources import (
    SourceAdmissionError,
    admit_source_locator,
)
from quant_investor.v17_v3_runtime.storage import (
    GOVERNED_ROOTS,
    PRIVATE_RUNS_ROOT,
    PRIVATE_SOURCES_ROOT,
    SecureStore,
    StorageSecurityError,
)

CUTOFF = "2026-07-25T07:00:00Z"
STRATEGY = "strategy-1"
RUN_ID = "run-1"


@dataclass(frozen=True)
class StagedClosure:
    root: Path
    store: SecureStore
    preselect_locator: RuntimeArtifact
    analyze_locator: RuntimeArtifact
    initial_pool: RuntimeArtifact
    quant_branch: RuntimeArtifact
    fundamental_branch: RuntimeArtifact
    readiness: RuntimeArtifact
    factor_baseline: RuntimeArtifact


def _workspace(tmp_path: Path) -> tuple[Path, SecureStore]:
    root = tmp_path / "private-workspace"
    root.mkdir(mode=0o700, parents=True)
    root.chmod(0o700)
    store = SecureStore(root)
    store.initialize()
    return root, store


def _typed(
    store: SecureStore,
    path: str,
    payload: Mapping[str, Any],
) -> RuntimeArtifact:
    artifact = runtime_artifact(
        relative_path=path,
        document=seal_typed_artifact(payload),
    )
    write_typed_exact_once(store, artifact)
    return artifact


def _raw_ref(store: SecureStore, role: str) -> dict[str, str]:
    if role == "holdings_snapshot":
        path = f"{PRIVATE_SOURCES_ROOT}/raw/{role}.json"
        raw = canonical_resource_bytes(
            {
                "role": role,
                "strategy_id": STRATEGY,
                "as_of_session": "2026-07-25",
                "available_at": CUTOFF,
            }
        )
    elif role == "deep_evidence":
        path = f"{PRIVATE_SOURCES_ROOT}/raw/{role}.json"
        raw = canonical_resource_bytes({"role": role, "available_at": CUTOFF})
    elif role == "cn_open_day_calendar":
        path = f"{PRIVATE_SOURCES_ROOT}/raw/{role}.json"
        raw = canonical_resource_bytes(
            {
                "role": role,
                "available_at": CUTOFF,
                "sessions": ["2026-07-25"],
            }
        )
    elif role == "official_delisting_cash":
        import pyarrow as arrow
        import pyarrow.parquet as parquet

        path = f"{PRIVATE_SOURCES_ROOT}/raw/{role}.parquet"
        buffer = BytesIO()
        parquet.write_table(
            arrow.table(
                {
                    "symbol": arrow.array(["000001.SZ"], type=arrow.string()),
                    "event_date": arrow.array(
                        [date(2026, 7, 25)],
                        type=arrow.date32(),
                    ),
                    "cash_per_share": arrow.array(
                        [0.25],
                        type=arrow.float64(),
                    ),
                    "announced_at": arrow.array(
                        [datetime(2026, 7, 25, 6, tzinfo=timezone.utc)],
                        type=arrow.timestamp("us", tz="UTC"),
                    ),
                    "available_at": arrow.array(
                        [datetime(2026, 7, 25, 7, tzinfo=timezone.utc)],
                        type=arrow.timestamp("us", tz="UTC"),
                    ),
                }
            ),
            buffer,
        )
        raw = buffer.getvalue()
    else:
        path = f"{PRIVATE_SOURCES_ROOT}/raw/{role}.parquet"
        raw = b"PAR1PAR1"
    store.write_exact_once(path, raw)
    return {
        "artifact_id": f"raw-{role.replace('_', '-')}",
        "artifact_version": (
            "myquant.v17.v3.dataset.official-delisting-cash.v1"
            if role == "official_delisting_cash"
            else "myquant.v17.v3.raw-source.v1"
        ),
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": CUTOFF,
        "relative_path": path,
        "semantic_sha256": hashlib.sha256(role.encode("ascii")).hexdigest(),
        "strategy_id": STRATEGY,
    }


def _source_manifest(
    store: SecureStore,
    *,
    manifest_id: str,
    phase: str,
    sources: list[dict[str, Any]],
    parent: RuntimeArtifact | None = None,
    raw_profile: str | None = None,
) -> RuntimeArtifact:
    payload: dict[str, Any] = {
        "version": "myquant.v17.v3.source-manifest.v1",
        "protocol_version": PROTOCOL_VERSION,
        "manifest_id": manifest_id,
        "strategy_id": STRATEGY,
        "cutoff": CUTOFF,
        "created_at": CUTOFF,
        "phase": phase,
        "closure_kind": "RAW" if parent is None else "DERIVED_CLOSURE",
        "sources": sorted(sources, key=lambda row: row["role"]),
        "authority": authority_envelope(),
    }
    if parent is not None:
        payload["parent_raw_manifest_ref"] = parent.reference
    elif raw_profile is not None:
        payload["raw_profile"] = raw_profile
    return _typed(
        store,
        f"{PRIVATE_SOURCES_ROOT}/manifests/{manifest_id}.json",
        payload,
    )


def _source_locator(
    store: SecureStore,
    *,
    locator_id: str,
    manifest: RuntimeArtifact,
    preselection: RuntimeArtifact | None,
) -> RuntimeArtifact:
    return _typed(
        store,
        f"{PRIVATE_SOURCES_ROOT}/locators/{locator_id}.json",
        {
            "version": "myquant.v17.v3.source-locator.v1",
            "protocol_version": PROTOCOL_VERSION,
            "locator_id": locator_id,
            "strategy_id": STRATEGY,
            "cutoff": CUTOFF,
            "created_at": CUTOFF,
            "source_manifest_ref": manifest.reference,
            "preselection_locator_ref": (None if preselection is None else preselection.reference),
            "authority": authority_envelope(),
        },
    )


def build_staged_closure(
    tmp_path: Path,
    *,
    shadow_current: bool = False,
) -> StagedClosure:
    root, store = _workspace(tmp_path)
    baseline_roles = (
        "cn_open_day_calendar",
        "corporate_actions",
        "market_bars",
        "pit_fundamentals",
        "universe_membership",
    )
    raw_roles = (
        *baseline_roles,
        "benchmark_total_return",
        "deep_evidence",
        *(("holdings_snapshot",) if not shadow_current else ()),
        "official_delisting_cash",
    )
    raw_refs = {role: _raw_ref(store, role) for role in raw_roles}
    readiness = _typed(
        store,
        f"{PRIVATE_SOURCES_ROOT}/raw/factor-governance-readiness.json",
        {
            "version": "myquant.v17.v3.factor-governance-readiness.v1",
            "protocol_version": PROTOCOL_VERSION,
            "readiness_id": "factor-readiness-1",
            "role": "factor_governance_readiness",
            "strategy_id": STRATEGY,
            "cutoff": CUTOFF,
            "created_at": CUTOFF,
            "available_at": CUTOFF,
            "source_as_of": "2026-07-25",
            "source_schema_version": "factor-governance-readiness.v4",
            "source_byte_sha256": "9" * 64,
            "readiness_status": ("FACTOR_V4_NOT_READY" if shadow_current else "FACTOR_V4_READY"),
            "factor_governance_ready": not shadow_current,
            "healthy_factor_count": 0 if shadow_current else 5,
            "production_factor_count": 0 if shadow_current else 5,
            "production_family_count": 0 if shadow_current else 3,
            "activation_receipt_valid": not shadow_current,
            "blockers": (["factor_v4_not_ready"] if shadow_current else []),
            "authority": authority_envelope(),
        },
    )
    raw_refs["factor_governance_readiness"] = readiness.reference
    provisional_policy = load_packaged_json("resources/provisional_factor_baseline_policy.v1.json")
    factor_baseline = (
        _typed(
            store,
            f"{PRIVATE_SOURCES_ROOT}/derived/provisional-factor-baseline.json",
            {
                "version": ("myquant.v17.v3.provisional-factor-baseline.v1"),
                "protocol_version": PROTOCOL_VERSION,
                "baseline_id": "provisional-baseline-1",
                "role": "provisional_factor_baseline",
                "strategy_id": STRATEGY,
                "cutoff": CUTOFF,
                "created_at": CUTOFF,
                "factor_baseline_mode": "PROVISIONAL_RESEARCH",
                "factor_governance_readiness_ref": readiness.reference,
                "policy_sha256": provisional_policy["semantic_sha256"],
                "preselector_factors": provisional_policy["preselector_factors"],
                "quant_factors": provisional_policy["quant_factors"],
                "authority": authority_envelope(),
            },
        )
        if shadow_current
        else readiness
    )
    preselect_raw_roles = [
        *baseline_roles,
        "benchmark_total_return",
        "factor_governance_readiness",
        "official_delisting_cash",
    ]
    raw_manifest = _source_manifest(
        store,
        manifest_id="raw-manifest-1",
        phase="RAW",
        sources=[{"role": role, "artifact_ref": raw_refs[role]} for role in preselect_raw_roles],
        raw_profile=("SHADOW_CURRENT" if shadow_current else "HISTORICAL_FORMAL"),
    )
    preselection_inputs = _typed(
        store,
        f"{PRIVATE_SOURCES_ROOT}/derived/quant-preselection-inputs.json",
        {
            "version": "myquant.v17.v3.quant-preselection-inputs.v1",
            "protocol_version": PROTOCOL_VERSION,
            "input_id": "preselection-input-1",
            "run_id": RUN_ID,
            "role": "quant_preselection_inputs",
            "strategy_id": STRATEGY,
            "cutoff": CUTOFF,
            "created_at": CUTOFF,
            "factor_baseline_mode": (
                "PROVISIONAL_RESEARCH" if shadow_current else "FACTOR_V4_PRODUCTION"
            ),
            "factor_baseline_ref": factor_baseline.reference,
            "payload": {
                "factor_contract": [
                    {
                        "definition_hash": row["definition_sha256"],
                        "family": row["family_id"],
                        "lineage": row["lineage_id"],
                        "lookback": row["lookback_open_days"],
                        "minimum_coverage": "0.90",
                        "name": row["factor_id"],
                        "warmup": row["lookback_open_days"],
                        "weight": row["weight"],
                    }
                    for row in provisional_policy["preselector_factors"]
                ],
                "observations": [
                    {
                        "data_ready": True,
                        "factor_values": [
                            {
                                "factor_id": row["factor_id"],
                                "value": value,
                            }
                            for row in provisional_policy["preselector_factors"]
                        ],
                        "history_count": 120,
                        "liquid": True,
                        "research_eligible": True,
                        "symbol": symbol,
                        "tradable": True,
                    }
                    for symbol, value in ((f"{index:06d}.SZ", str(index)) for index in range(1, 25))
                ],
                "policy_sha256": load_packaged_json("resources/preselector_policy.v1.json")[
                    "semantic_sha256"
                ],
                "quant_branch_inventory": [
                    {
                        "definition_hash": row["definition_sha256"],
                        "family": row["family_id"],
                        "lineage": row["lineage_id"],
                        "name": row["factor_id"],
                    }
                    for row in provisional_policy["quant_factors"]
                ],
            },
            "authority": authority_envelope(),
        },
    )
    preselect_manifest = _source_manifest(
        store,
        manifest_id="preselect-manifest-1",
        phase=("SHADOW_CURRENT_PRESELECT" if shadow_current else "PRESELECT"),
        sources=[
            *(
                [
                    {
                        "role": "provisional_factor_baseline",
                        "artifact_ref": factor_baseline.reference,
                    }
                ]
                if shadow_current
                else []
            ),
            {
                "role": "quant_preselection_inputs",
                "artifact_ref": preselection_inputs.reference,
            },
        ],
        parent=raw_manifest,
    )
    preselect_locator = _source_locator(
        store,
        locator_id="preselect-locator-1",
        manifest=preselect_manifest,
        preselection=None,
    )
    initial_outcome = build_initial_pool(
        workspace_root=root,
        locator_path=str(preselect_locator.relative_path),
        expected_locator_sha256=preselect_locator.byte_sha256,
    )
    initial_raw = store.read(
        initial_outcome.relative_path,
        initial_outcome.byte_sha256,
    )
    initial_document = load_typed_artifact(
        initial_raw,
        label="initial pool",
        expected_version="myquant.v17.v3.initial-pool-output.v1",
    )
    initial_pool = runtime_artifact(
        relative_path=initial_outcome.relative_path,
        document=initial_document,
    )
    pool = list(initial_document["selected_symbols"])
    pool_order_sha = hashlib.sha256(canonical_bytes(pool)).hexdigest()

    def branch(name: str) -> RuntimeArtifact:
        return _typed(
            store,
            f"{PRIVATE_SOURCES_ROOT}/derived/{name}-branch.json",
            {
                "version": "myquant.v17.v3.branch-output.v1",
                "protocol_version": PROTOCOL_VERSION,
                "output_id": f"{name}-branch-1",
                "run_id": RUN_ID,
                "branch": name,
                "strategy_id": STRATEGY,
                "cutoff": CUTOFF,
                "created_at": CUTOFF,
                "state": "BRANCHES_COMPLETE",
                "source_locator_ref": preselect_locator.reference,
                "initial_pool_ref": initial_pool.reference,
                "initial_pool_count": len(pool),
                "initial_pool_symbol_order_sha256": pool_order_sha,
                "policy_sha256": load_packaged_json(f"resources/{name}_branch_policy.v1.json")[
                    "semantic_sha256"
                ],
                "ordered_domain": pool,
                "records": [
                    {
                        "symbol": symbol,
                        "status": "READY",
                        "score": str(len(pool) - index),
                        "reason": None,
                    }
                    for index, symbol in enumerate(pool)
                ],
                "authority": authority_envelope(),
            },
        )

    quant_branch = branch("quant")
    fundamental_branch = branch("fundamental")
    deep_inputs = _typed(
        store,
        f"{PRIVATE_SOURCES_ROOT}/derived/deep-inputs.json",
        {
            "version": "myquant.v17.v3.deep-research-inputs.v1",
            "protocol_version": PROTOCOL_VERSION,
            "input_id": "deep-input-1",
            "run_id": RUN_ID,
            "role": "deep_research_inputs",
            "strategy_id": STRATEGY,
            "cutoff": CUTOFF,
            "created_at": CUTOFF,
            "payload": [
                *[
                    {
                        "symbol": symbol,
                        "lane": "SELECTION_POOL",
                        "held": False,
                        "available": True,
                        "signal": "0",
                        "veto_buy": False,
                        "base_target": "0.03",
                        "current_target": "0",
                        "evidence_refs": [raw_refs["deep_evidence"]],
                    }
                    for symbol in pool
                ],
                *(
                    [
                        {
                            "symbol": "000025.SZ",
                            "lane": "REVIEW_ONLY_HOLDING",
                            "held": True,
                            "available": False,
                            "signal": None,
                            "veto_buy": True,
                            "base_target": "0.1",
                            "current_target": "0.1",
                            "evidence_refs": [],
                        }
                    ]
                    if not shadow_current
                    else []
                ),
            ],
            "authority": authority_envelope(),
        },
    )
    permissions = _typed(
        store,
        f"{PRIVATE_SOURCES_ROOT}/derived/permissions.json",
        {
            "version": "myquant.v17.v3.pretrade-permissions.v1",
            "protocol_version": PROTOCOL_VERSION,
            "permissions_id": "permissions-1",
            "run_id": RUN_ID,
            "role": "permissions",
            "strategy_id": STRATEGY,
            "cutoff": CUTOFF,
            "created_at": CUTOFF,
            "canonical_calendar_ref": raw_refs["cn_open_day_calendar"],
            "decision_session": "2026-07-25",
            "portfolio_basis": (
                "MODEL_ONLY_NO_PRIVATE_HOLDINGS" if shadow_current else "HOLDINGS_AWARE"
            ),
            "holdings_snapshot_as_of_session": (None if shadow_current else "2026-07-25"),
            "holdings_snapshot_age_sessions": (None if shadow_current else 0),
            "holdings_snapshot_ref": (None if shadow_current else raw_refs["holdings_snapshot"]),
            "payload": sorted(
                [
                    *[
                        {
                            "symbol": symbol,
                            "lane": "SELECTION_POOL",
                            "held": False,
                            "can_buy": not shadow_current,
                            "current_target": "0",
                        }
                        for symbol in pool
                    ],
                    *(
                        [
                            {
                                "symbol": "000025.SZ",
                                "lane": "REVIEW_ONLY_HOLDING",
                                "held": True,
                                "can_buy": False,
                                "current_target": "0.1",
                            }
                        ]
                        if not shadow_current
                        else []
                    ),
                ],
                key=lambda row: row["symbol"],
            ),
            "authority": authority_envelope(),
        },
    )
    portfolio_raw_manifest = _source_manifest(
        store,
        manifest_id="portfolio-raw-manifest-1",
        phase="RAW",
        sources=[
            {"role": role, "artifact_ref": raw_refs[role]}
            for role in (
                *baseline_roles,
                "benchmark_total_return",
                "deep_evidence",
                "factor_governance_readiness",
                *(("holdings_snapshot",) if not shadow_current else ()),
                "official_delisting_cash",
            )
        ],
        raw_profile=("SHADOW_CURRENT" if shadow_current else "HISTORICAL_FORMAL"),
    )
    analyze_manifest = _source_manifest(
        store,
        manifest_id="analyze-manifest-1",
        phase=("SHADOW_CURRENT_MODEL_PORTFOLIO" if shadow_current else "PORTFOLIO"),
        sources=[
            {"role": "deep_research_inputs", "artifact_ref": deep_inputs.reference},
            {
                "role": "fundamental_branch_output",
                "artifact_ref": fundamental_branch.reference,
            },
            {"role": "initial_pool_output", "artifact_ref": initial_pool.reference},
            {"role": "permissions", "artifact_ref": permissions.reference},
            *(
                [
                    {
                        "role": "provisional_factor_baseline",
                        "artifact_ref": factor_baseline.reference,
                    }
                ]
                if shadow_current
                else []
            ),
            {"role": "quant_branch_output", "artifact_ref": quant_branch.reference},
            {
                "role": "quant_preselection_inputs",
                "artifact_ref": preselection_inputs.reference,
            },
        ],
        parent=portfolio_raw_manifest,
    )
    analyze_locator = _source_locator(
        store,
        locator_id="analyze-locator-1",
        manifest=analyze_manifest,
        preselection=preselect_locator,
    )
    return StagedClosure(
        root,
        store,
        preselect_locator,
        analyze_locator,
        initial_pool,
        quant_branch,
        fundamental_branch,
        readiness,
        factor_baseline,
    )


def test_secure_store_enforces_owner_private_regular_single_link_and_roots(
    tmp_path: Path,
) -> None:
    root, store = _workspace(tmp_path)
    for relative_root in GOVERNED_ROOTS:
        assert (root / relative_root).stat().st_mode & 0o777 == 0o700
    path = f"{PRIVATE_RUNS_ROOT}/run-1/value.json"
    raw = canonical_resource_bytes({"value": 1})
    store.write_exact_once(path, raw)
    target = root / path
    assert target.stat().st_mode & 0o777 == 0o600
    assert target.stat().st_nlink == 1
    hardlink = target.with_name("hardlink.json")
    os.link(target, hardlink)
    with pytest.raises(StorageSecurityError, match="hard link"):
        store.read(path)
    hardlink.unlink()


def test_public_envelope_drops_holdings_symbols_and_market_values() -> None:
    private = {
        "holdings": {"600000.SH": 100},
        "nav": 1000,
        "rows": [{"symbol": "600000.SH", "price": 10.0}],
    }
    redacted = redact_public(private)
    assert_public_envelope_safe(redacted)
    with pytest.raises(PublicEnvelopeError):
        assert_public_envelope_safe(private)


def test_staged_preselect_analyze_roundtrip_and_review_only_boundary(
    tmp_path: Path,
) -> None:
    closure = build_staged_closure(tmp_path)
    admitted = admit_source_locator(
        closure.store,
        locator_path=str(closure.analyze_locator.relative_path),
        expected_locator_sha256=closure.analyze_locator.byte_sha256,
    )
    result = run_pipeline(PipelineRequest(mode=SHADOW_MODE, admitted_sources=admitted))
    assert result.terminal.state == "SHADOW_COMPLETE"
    assert result.calibration_label == "UNCALIBRATED_50_50"
    assert result.terminal_artifact is not None
    assert result.terminal_artifact.document["analyze_locator_ref"] == (
        closure.analyze_locator.reference
    )
    portfolio = next(
        artifact
        for artifact in result.artifacts
        if artifact.document["version"] == "myquant.v17.v3.portfolio-output.v1"
    )
    assert "000025.SZ" not in result.fusion.selected_symbols
    review = next(row for row in portfolio.document["targets"] if row["symbol"] == "000025.SZ")
    assert review["lane"] == "REVIEW_ONLY_HOLDING"
    assert Decimal(review["final_target"]) <= Decimal(review["current_target"])
    outcome = analyze(
        workspace_root=closure.root,
        mode=SHADOW_MODE,
        locator_path=str(closure.analyze_locator.relative_path),
        expected_locator_sha256=closure.analyze_locator.byte_sha256,
    )
    assert outcome.result.terminal.state == "SHADOW_COMPLETE"


def test_staged_lineage_replay_drift_fails_closed(tmp_path: Path) -> None:
    closure = build_staged_closure(tmp_path)
    stored = closure.store.read_optional(closure.initial_pool.relative_path)
    assert stored is not None
    tampered = dict(closure.initial_pool.document)
    tampered["output_id"] = "initial-pool-tampered"
    tampered.pop("semantic_sha256")
    replacement = canonical_resource_bytes(seal_typed_artifact(tampered))
    closure.store.replace_cas(
        closure.initial_pool.relative_path,
        stored.byte_sha256,
        replacement,
    )
    with pytest.raises(SourceAdmissionError, match="exact-byte"):
        admit_source_locator(
            closure.store,
            locator_path=str(closure.analyze_locator.relative_path),
            expected_locator_sha256=closure.analyze_locator.byte_sha256,
        )


def test_formal_pipeline_has_no_caller_array_bypass(tmp_path: Path) -> None:
    closure = build_staged_closure(tmp_path)
    admitted = admit_source_locator(
        closure.store,
        locator_path=str(closure.analyze_locator.relative_path),
        expected_locator_sha256=closure.analyze_locator.byte_sha256,
    )
    result = run_pipeline(
        PipelineRequest(
            mode=FORMAL_RESEARCH_MODE,
            admitted_sources=admitted,
        )
    )
    assert result.terminal.state == "HARD_STOP_INVALID_EVIDENCE"
    with pytest.raises(TypeError):
        run_pipeline({"mode": FORMAL_RESEARCH_MODE})  # type: ignore[arg-type]


def test_shadow_current_model_only_is_all_cash_and_propagates_profile(
    tmp_path: Path,
) -> None:
    closure = build_staged_closure(tmp_path, shadow_current=True)
    admitted = admit_source_locator(
        closure.store,
        locator_path=str(closure.analyze_locator.relative_path),
        expected_locator_sha256=closure.analyze_locator.byte_sha256,
    )
    result = run_pipeline(PipelineRequest(mode=SHADOW_MODE, admitted_sources=admitted))
    assert result.terminal.state == "SHADOW_COMPLETE"
    assert result.factor_baseline_mode == "PROVISIONAL_RESEARCH"
    assert result.portfolio_basis == "MODEL_ONLY_NO_PRIVATE_HOLDINGS"
    assert [stage["status"] for stage in result.overlay_stages] == [
        "UNAVAILABLE_NO_OP",
        "UNAVAILABLE_NO_OP",
    ]
    portfolio = next(
        artifact.document
        for artifact in result.artifacts
        if artifact.document["version"] == "myquant.v17.v3.portfolio-output.v1"
    )
    assert portfolio["cash_weight"] == "1"
    assert portfolio["gross_weight"] == "0"
    assert len(portfolio["targets"]) == 24
    assert all(row["final_target"] == "0" for row in portfolio["targets"])
    assert portfolio["holdings_snapshot_ref"] is None
    public = result.to_public_wire()
    assert public["factor_baseline_mode"] == "PROVISIONAL_RESEARCH"
    assert public["portfolio_basis"] == "MODEL_ONLY_NO_PRIVATE_HOLDINGS"
    assert public["formal_research_publication_authority"] is False
    assert public["execution_authority"] is False


def test_shadow_current_and_deep_fail_closed_with_exact_blockers(
    tmp_path: Path,
) -> None:
    closure = build_staged_closure(tmp_path, shadow_current=True)
    admitted = admit_source_locator(
        closure.store,
        locator_path=str(closure.analyze_locator.relative_path),
        expected_locator_sha256=closure.analyze_locator.byte_sha256,
    )
    formal = run_pipeline(
        PipelineRequest(
            mode=FORMAL_RESEARCH_MODE,
            admitted_sources=admitted,
        )
    )
    assert formal.terminal.blockers == ("shadow_current_phase_requires_shadow_mode",)
    deep_inputs = admitted.documents["deep_research_inputs"]
    assert isinstance(deep_inputs, dict)
    deep_inputs["payload"] = deep_inputs["payload"][:-1]
    missing = run_pipeline(PipelineRequest(mode=SHADOW_MODE, admitted_sources=admitted))
    assert missing.terminal.blockers == ("deep_top24_row_missing",)
