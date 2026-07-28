from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal
import hashlib
from pathlib import Path
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from quant_investor.v17_v3_contract.resources import (
    PACKAGE_MANIFEST_SHA256,
    PackageResourceError,
    load_packaged_json,
)
from quant_investor.v17_v3_contract.canonical import (
    canonical_bytes,
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v3_runtime.activation import (
    ACTIVE,
    ACTIVATION_REJECTED,
    NO_CURRENT_ACTIVE_FORMAL_RESULT,
    REVOKED,
    ActivationError,
    ActivationPublisher,
)
from quant_investor.v17_v3_runtime.artifacts import (
    RuntimeArtifact,
    load_typed_artifact,
    runtime_artifact,
    write_typed_exact_once,
)
from quant_investor.v17_v3_runtime.authority import (
    PROTOCOL_VERSION,
    authority_envelope,
)
from quant_investor.v17_v3_runtime.pipeline import (
    FORMAL_RESEARCH_MODE,
    PipelineRequest,
    run_pipeline,
)
from quant_investor.v17_v3_runtime.sources import (
    AdmittedSources,
    admit_source_locator,
)
from quant_investor.v17_v3_runtime.storage import StorageError
from quant_investor.v17_v3_runtime import activation as runtime_activation
from quant_investor.v17_v3_runtime import algorithms
from quant_investor.v17_v3_runtime import service as runtime_service

from test_v17_v3_runtime import (
    CUTOFF,
    RUN_ID,
    STRATEGY,
    StagedClosure,
    _source_locator,
    _source_manifest,
    build_staged_closure,
)


def _write(
    closure: StagedClosure,
    path: str,
    payload: dict,
) -> RuntimeArtifact:
    document = seal_semantic(payload)
    raw = canonical_resource_bytes(document)
    artifact = RuntimeArtifact(
        relative_path=PurePosixPath(path),
        document=document,
        raw=raw,
        byte_sha256=hashlib.sha256(raw).hexdigest(),
    )
    write_typed_exact_once(closure.store, artifact)
    return artifact


def _write_fast(
    closure: StagedClosure,
    path: str,
    payload: dict,
) -> RuntimeArtifact:
    """Write immutable fixture bytes without an fsync per historical artifact."""

    document = seal_semantic(payload)
    raw = canonical_resource_bytes(document)
    artifact = RuntimeArtifact(
        relative_path=PurePosixPath(path),
        document=document,
        raw=raw,
        byte_sha256=hashlib.sha256(raw).hexdigest(),
    )
    target = closure.root / artifact.relative_path
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent = target.parent
    while parent != closure.root:
        parent.chmod(0o700)
        parent = parent.parent
    if target.exists():
        assert target.read_bytes() == artifact.raw
    else:
        target.write_bytes(artifact.raw)
        target.chmod(0o600)
    return artifact


def _fixture_reference(artifact: RuntimeArtifact) -> dict[str, str]:
    """Build a known-valid test reference without revalidating fixture bytes."""

    document = artifact.document
    identity_field = next(
        field
        for field in (
            "output_id",
            "input_id",
            "manifest_id",
            "locator_id",
        )
        if field in document
    )
    return {
        "artifact_id": str(document[identity_field]),
        "artifact_version": str(document["version"]),
        "byte_sha256": artifact.byte_sha256,
        "cutoff": str(document["cutoff"]),
        "relative_path": str(artifact.relative_path),
        "semantic_sha256": str(document["semantic_sha256"]),
        "strategy_id": str(document["strategy_id"]),
    }


def _calibration_input(
    closure: StagedClosure,
    *,
    persist_history: bool = True,
) -> RuntimeArtifact:
    origins = [(date(2026, 3, 27) + timedelta(days=index)).isoformat() for index in range(120)]
    history: dict[str, dict[str, RuntimeArtifact]] = {}
    baseline_roles = (
        "cn_open_day_calendar",
        "corporate_actions",
        "market_bars",
        "pit_fundamentals",
        "universe_membership",
    )
    raw_paths = {
        role: ("data/private/v17_v3_sources/raw/calibration/" f"{role}.parquet")
        for role in baseline_roles
    }
    if persist_history:
        for path in raw_paths.values():
            closure.store.write_exact_once(path, b"PAR1PAR1")

    def ensure_history(origin: str) -> dict[str, RuntimeArtifact]:
        existing = history.get(origin)
        if existing is not None:
            return existing
        cutoff = f"{origin}T07:00:00Z"
        origin_id = origin.replace("-", "")
        run_id = f"calibration-{origin_id}"
        pool = ["000001.SZ"]
        factor_baseline_ref = {
            "artifact_id": f"calibration-{origin_id}-factor-readiness",
            "artifact_version": ("myquant.v17.v3.factor-governance-readiness.v1"),
            "byte_sha256": "a" * 64,
            "cutoff": cutoff,
            "relative_path": (
                "data/private/v17_v3_sources/raw/calibration/" f"{origin_id}-factor-readiness.json"
            ),
            "semantic_sha256": "b" * 64,
            "strategy_id": STRATEGY,
        }
        raw_refs = {
            role: {
                "artifact_id": f"raw-{role.replace('_', '-')}",
                "artifact_version": "myquant.v17.v3.raw-source.v1",
                "byte_sha256": hashlib.sha256(b"PAR1PAR1").hexdigest(),
                "cutoff": cutoff,
                "relative_path": raw_paths[role],
                "semantic_sha256": hashlib.sha256(role.encode("ascii")).hexdigest(),
                "strategy_id": STRATEGY,
            }
            for role in baseline_roles
        }
        raw_manifest = _write_fast(
            closure,
            ("data/private/v17_v3_sources/manifests/calibration/" f"{origin_id}-raw.json"),
            {
                "version": "myquant.v17.v3.source-manifest.v1",
                "protocol_version": PROTOCOL_VERSION,
                "manifest_id": f"calibration-{origin_id}-raw",
                "strategy_id": STRATEGY,
                "cutoff": cutoff,
                "created_at": cutoff,
                "phase": "RAW",
                "closure_kind": "RAW",
                "sources": [
                    {"role": role, "artifact_ref": raw_refs[role]} for role in baseline_roles
                ],
                "authority": authority_envelope(),
            },
        )
        preselection_inputs = _write_fast(
            closure,
            (
                "data/private/v17_v3_sources/derived/calibration/"
                f"{origin_id}-preselection-inputs.json"
            ),
            {
                "version": "myquant.v17.v3.quant-preselection-inputs.v1",
                "protocol_version": PROTOCOL_VERSION,
                "input_id": f"calibration-{origin_id}-preselection-inputs",
                "run_id": run_id,
                "role": "quant_preselection_inputs",
                "strategy_id": STRATEGY,
                "cutoff": cutoff,
                "created_at": cutoff,
                "factor_baseline_mode": "FACTOR_V4_PRODUCTION",
                "factor_baseline_ref": factor_baseline_ref,
                "payload": {
                    "factor_contract": [
                        {
                            "definition_hash": "1" * 64,
                            "family": "pre-family",
                            "lineage": "pre-lineage",
                            "lookback": 120,
                            "minimum_coverage": "0.90",
                            "name": "pre-factor",
                            "warmup": 120,
                            "weight": "1",
                        }
                    ],
                    "observations": [
                        {
                            "data_ready": True,
                            "factor_values": [
                                {
                                    "factor_id": "pre-factor",
                                    "value": "1",
                                }
                            ],
                            "history_count": 120,
                            "liquid": True,
                            "research_eligible": True,
                            "symbol": pool[0],
                            "tradable": True,
                        }
                    ],
                    "policy_sha256": load_packaged_json("resources/preselector_policy.v1.json")[
                        "semantic_sha256"
                    ],
                    "quant_branch_inventory": [
                        {
                            "definition_hash": "3" * 64,
                            "family": "quant-family",
                            "lineage": "quant-lineage",
                            "name": "quant-factor",
                        }
                    ],
                },
                "authority": authority_envelope(),
            },
        )
        preselect_manifest = _write_fast(
            closure,
            ("data/private/v17_v3_sources/manifests/calibration/" f"{origin_id}-preselect.json"),
            {
                "version": "myquant.v17.v3.source-manifest.v1",
                "protocol_version": PROTOCOL_VERSION,
                "manifest_id": f"calibration-{origin_id}-preselect",
                "strategy_id": STRATEGY,
                "cutoff": cutoff,
                "created_at": cutoff,
                "phase": "PRESELECT",
                "closure_kind": "DERIVED_CLOSURE",
                "sources": [
                    {
                        "role": "quant_preselection_inputs",
                        "artifact_ref": _fixture_reference(preselection_inputs),
                    }
                ],
                "parent_raw_manifest_ref": _fixture_reference(raw_manifest),
                "authority": authority_envelope(),
            },
        )
        locator = _write_fast(
            closure,
            ("data/private/v17_v3_sources/locators/calibration/" f"{origin_id}-preselect.json"),
            {
                "version": "myquant.v17.v3.source-locator.v1",
                "protocol_version": PROTOCOL_VERSION,
                "locator_id": f"calibration-{origin_id}-preselect",
                "strategy_id": STRATEGY,
                "cutoff": cutoff,
                "created_at": cutoff,
                "source_manifest_ref": _fixture_reference(preselect_manifest),
                "preselection_locator_ref": None,
                "authority": authority_envelope(),
            },
        )
        pool_order_sha = hashlib.sha256(canonical_bytes(pool)).hexdigest()
        initial_pool = _write_fast(
            closure,
            ("data/private/v17_v3_runs/calibration/" f"{origin}/initial-pool.json"),
            {
                "version": "myquant.v17.v3.initial-pool-output.v1",
                "protocol_version": PROTOCOL_VERSION,
                "output_id": f"calibration-{origin_id}-pool",
                "run_id": run_id,
                "strategy_id": STRATEGY,
                "cutoff": cutoff,
                "created_at": cutoff,
                "state": "PRESELECT_COMPLETE",
                "status": "READY",
                "source_locator_ref": _fixture_reference(locator),
                "raw_source_manifest_ref": _fixture_reference(raw_manifest),
                "policy_sha256": load_packaged_json("resources/preselector_policy.v1.json")[
                    "semantic_sha256"
                ],
                "factor_baseline_mode": "FACTOR_V4_PRODUCTION",
                "factor_baseline_ref": factor_baseline_ref,
                "history_required": 120,
                "ordered_domain": pool,
                "ready_domain": pool,
                "selected_symbols": pool,
                "pool_count": 1,
                "pool_symbol_order_sha256": pool_order_sha,
                "dispositions": [
                    {
                        "symbol": pool[0],
                        "status": "READY",
                        "score": "1",
                        "selected": True,
                        "reasons": [],
                    }
                ],
                "factor_coverage": [
                    {
                        "factor_id": "pre-factor",
                        "coverage": "1",
                    }
                ],
                "blockers": [],
                "authority": authority_envelope(),
            },
        )

        def branch(name: str) -> RuntimeArtifact:
            return _write_fast(
                closure,
                ("data/private/v17_v3_runs/calibration/" f"{origin}/{name}.json"),
                {
                    "version": "myquant.v17.v3.branch-output.v1",
                    "protocol_version": PROTOCOL_VERSION,
                    "output_id": f"{name}-{origin_id}",
                    "run_id": run_id,
                    "branch": name,
                    "strategy_id": STRATEGY,
                    "cutoff": cutoff,
                    "created_at": cutoff,
                    "state": "BRANCHES_COMPLETE",
                    "source_locator_ref": _fixture_reference(locator),
                    "initial_pool_ref": _fixture_reference(initial_pool),
                    "initial_pool_count": 1,
                    "initial_pool_symbol_order_sha256": pool_order_sha,
                    "policy_sha256": load_packaged_json(f"resources/{name}_branch_policy.v1.json")[
                        "semantic_sha256"
                    ],
                    "ordered_domain": pool,
                    "records": [
                        {
                            "symbol": pool[0],
                            "status": "READY",
                            "score": "1",
                            "reason": None,
                        }
                    ],
                    "authority": authority_envelope(),
                },
            )

        completed = {
            "raw_manifest": raw_manifest,
            "locator": locator,
            "initial_pool": initial_pool,
            "quant": branch("quant"),
            "fundamental": branch("fundamental"),
        }
        history[origin] = completed
        return completed

    def historical_ref(origin: str, branch: str) -> dict[str, str]:
        relative_path = f"data/private/v17_v3_runs/calibration/{origin}/{branch}.json"
        if persist_history:
            return _fixture_reference(ensure_history(origin)[branch])
        cutoff = f"{origin}T07:00:00Z"
        return {
            "artifact_id": f"{branch}-{origin}",
            "artifact_version": "myquant.v17.v3.branch-output.v1",
            "byte_sha256": "4" * 64 if branch == "quant" else "5" * 64,
            "cutoff": cutoff,
            "relative_path": relative_path,
            "semantic_sha256": "6" * 64 if branch == "quant" else "7" * 64,
            "strategy_id": STRATEGY,
        }

    return _write(
        closure,
        f"data/private/v17_v3_runs/{RUN_ID}/fusion_calibration_inputs.json",
        {
            "version": "myquant.v17.v3.fusion-calibration-inputs.v1",
            "protocol_version": PROTOCOL_VERSION,
            "input_id": "fusion-calibration-input-1",
            "run_id": RUN_ID,
            "role": "fusion_calibration",
            "strategy_id": STRATEGY,
            "cutoff": CUTOFF,
            "created_at": CUTOFF,
            "payload": {
                "active_cutoff": origins[-1],
                "canonical_sessions": origins,
                "scheduled_origins": origins,
                "months": [
                    {
                        "origin": origin,
                        "label_252_end_session": origin,
                        "label_252_mature": True,
                        "ordered_pool": ["000001.SZ"],
                        "quant_branch_ref": historical_ref(origin, "quant"),
                        "fundamental_branch_ref": historical_ref(origin, "fundamental"),
                        "forward_return_60": [{"symbol": "000001.SZ", "value": "0.1"}],
                        "forward_return_252": [{"symbol": "000001.SZ", "value": "0.2"}],
                    }
                    for origin in origins
                ],
            },
            "authority": authority_envelope(),
        },
    )


def _preselect_raw_manifest(
    closure: StagedClosure,
) -> tuple[RuntimeArtifact, AdmittedSources]:
    admission = admit_source_locator(
        closure.store,
        locator_path=str(closure.preselect_locator.relative_path),
        expected_locator_sha256=closure.preselect_locator.byte_sha256,
    )
    raw_manifest = admission.documents["raw_source_manifest"]
    calibration_raw = _source_manifest(
        closure.store,
        manifest_id="calibration-raw-manifest-1",
        phase="RAW",
        sources=[
            dict(row)
            for row in raw_manifest["sources"]
            if row["role"] != "factor_governance_readiness"
        ],
    )
    return (
        calibration_raw,
        admission,
    )


def _fusion_raw_manifest(
    closure: StagedClosure,
    *,
    calibration_raw: RuntimeArtifact,
    preselect_admission: AdmittedSources,
    manifest_id: str,
) -> RuntimeArtifact:
    return _source_manifest(
        closure.store,
        manifest_id=manifest_id,
        phase="RAW",
        sources=[
            *calibration_raw.document["sources"],
            {
                "role": "factor_governance_readiness",
                "artifact_ref": preselect_admission.reference_for_role(
                    "factor_governance_readiness"
                ),
            },
        ],
        raw_profile="HISTORICAL_FORMAL",
    )


def _calibration_gate_locator(
    closure: StagedClosure,
    *,
    kind: str,
    raw_manifest: RuntimeArtifact,
) -> RuntimeArtifact:
    phase, role = {
        "QUANT_TIMING": (
            "QUANT_TIMING_CALIBRATION",
            "quant_timing_calibration_inputs",
        ),
        "FUNDAMENTAL_FORWARD": (
            "FUNDAMENTAL_FORWARD_CALIBRATION",
            "fundamental_forward_calibration_inputs",
        ),
    }[kind]
    slug = kind.casefold().replace("_", "-")
    gate_inputs = _write(
        closure,
        f"data/private/v17_v3_sources/derived/{slug}-inputs.json",
        {
            "version": "myquant.v17.v3.calibration-gate-inputs.v1",
            "protocol_version": PROTOCOL_VERSION,
            "input_id": f"{slug}-inputs-1",
            "role": role,
            "calibration_kind": kind,
            "strategy_id": STRATEGY,
            "cutoff": CUTOFF,
            "created_at": CUTOFF,
            "observation_start_at": "2021-01-01T07:00:00Z",
            "observation_end_at": CUTOFF,
            "authority": authority_envelope(),
        },
    )
    manifest = _source_manifest(
        closure.store,
        manifest_id=f"{slug}-manifest-1",
        phase=phase,
        sources=[
            {
                "role": role,
                "artifact_ref": gate_inputs.reference,
            }
        ],
        parent=raw_manifest,
    )
    return _source_locator(
        closure.store,
        locator_id=f"{slug}-locator-1",
        manifest=manifest,
        preselection=None,
    )


def _fusion_gate_locator(
    closure: StagedClosure,
    *,
    decision_tag: str,
    calibration_input: RuntimeArtifact,
    quant_receipt: RuntimeArtifact,
    fundamental_receipt: RuntimeArtifact,
    raw_manifest: RuntimeArtifact,
    preselect_admission: AdmittedSources,
) -> RuntimeArtifact:
    manifest = _source_manifest(
        closure.store,
        manifest_id=f"fusion-promotion-{decision_tag}-manifest-1",
        phase="FUSION_PROMOTION",
        sources=[
            {
                "role": "fundamental_branch_output",
                "artifact_ref": closure.fundamental_branch.reference,
            },
            {
                "role": "fundamental_forward_calibration",
                "artifact_ref": fundamental_receipt.reference,
            },
            {
                "role": "fusion_calibration",
                "artifact_ref": calibration_input.reference,
            },
            {
                "role": "initial_pool_output",
                "artifact_ref": closure.initial_pool.reference,
            },
            {
                "role": "quant_branch_output",
                "artifact_ref": closure.quant_branch.reference,
            },
            {
                "role": "quant_preselection_inputs",
                "artifact_ref": preselect_admission.reference_for_role("quant_preselection_inputs"),
            },
            {
                "role": "quant_timing_calibration",
                "artifact_ref": quant_receipt.reference,
            },
        ],
        parent=raw_manifest,
    )
    return _source_locator(
        closure.store,
        locator_id=f"fusion-promotion-{decision_tag}-locator-1",
        manifest=manifest,
        preselection=closure.preselect_locator,
    )


def _promotion(
    closure: StagedClosure,
    *,
    accepted: bool,
    persist_history: bool = True,
) -> RuntimeArtifact:
    calibration_input = _calibration_input(
        closure,
        persist_history=persist_history,
    )
    decision_tag = "accepted" if accepted else "rejected"
    raw_manifest, preselect_admission = _preselect_raw_manifest(closure)
    phase_locators = {
        kind: _calibration_gate_locator(
            closure,
            kind=kind,
            raw_manifest=raw_manifest,
        )
        for kind in ("QUANT_TIMING", "FUNDAMENTAL_FORWARD")
    }
    fusion_raw_manifest = _fusion_raw_manifest(
        closure,
        calibration_raw=raw_manifest,
        preselect_admission=preselect_admission,
        manifest_id=f"fusion-{decision_tag}-raw-manifest-1",
    )
    receipt_artifacts: list[RuntimeArtifact] = []
    for index, kind in enumerate(("QUANT_TIMING", "FUNDAMENTAL_FORWARD")):
        receipt_artifacts.append(
            _write(
                closure,
                (
                    f"data/private/v17_v3_runs/{RUN_ID}/calibration/"
                    f"{decision_tag}-{kind.casefold()}.json"
                ),
                {
                    "version": ("myquant.v17.v3.fusion-calibration-receipt.v1"),
                    "protocol_version": PROTOCOL_VERSION,
                    "calibration_id": f"calibration-{decision_tag}-{index}",
                    "calibration_kind": kind,
                    "strategy_id": STRATEGY,
                    "cutoff": CUTOFF,
                    "created_at": CUTOFF,
                    "observation_end_at": CUTOFF,
                    "accepted": True,
                    "evidence_refs": [phase_locators[kind].reference],
                    "authority": authority_envelope(),
                },
            )
        )
    fusion_locator = _fusion_gate_locator(
        closure,
        decision_tag=decision_tag,
        calibration_input=calibration_input,
        quant_receipt=receipt_artifacts[0],
        fundamental_receipt=receipt_artifacts[1],
        raw_manifest=fusion_raw_manifest,
        preselect_admission=preselect_admission,
    )
    receipt_artifacts.append(
        _write(
            closure,
            (
                f"data/private/v17_v3_runs/{RUN_ID}/calibration/"
                f"{decision_tag}-fusion_promotion.json"
            ),
            {
                "version": "myquant.v17.v3.fusion-calibration-receipt.v1",
                "protocol_version": PROTOCOL_VERSION,
                "calibration_id": f"calibration-{decision_tag}-2",
                "calibration_kind": "FUSION_PROMOTION",
                "strategy_id": STRATEGY,
                "cutoff": CUTOFF,
                "created_at": CUTOFF,
                "observation_end_at": CUTOFF,
                "accepted": accepted,
                "evidence_refs": sorted(
                    [
                        calibration_input.reference,
                        fusion_locator.reference,
                    ],
                    key=lambda ref: (
                        ref["relative_path"],
                        ref["byte_sha256"],
                    ),
                ),
                "authority": authority_envelope(),
            },
        )
    )
    origins = [(date(2026, 3, 27) + timedelta(days=index)).isoformat() for index in range(120)]
    training = origins[:60]
    outer = origins[60:]
    folds = [
        {
            "fold_index": index + 1,
            "training_origins": training,
            "oos_origins": outer[index * 12 : (index + 1) * 12],
            "selected_quant_weight": "0.50",
        }
        for index in range(5)
    ]
    evidence = sorted(
        [
            closure.initial_pool.reference,
            closure.quant_branch.reference,
            closure.fundamental_branch.reference,
            calibration_input.reference,
        ],
        key=lambda ref: (ref["relative_path"], ref["byte_sha256"]),
    )
    common = {
        "version": "myquant.v17.v3.fusion-promotion-receipt.v1",
        "protocol_version": PROTOCOL_VERSION,
        "promotion_id": ("promotion-accepted" if accepted else "promotion-rejected"),
        "strategy_id": STRATEGY,
        "cutoff": CUTOFF,
        "created_at": CUTOFF,
        "observation_end_at": CUTOFF,
        "bootstrap_matrix_sha256": "8" * 64,
        "calibration_receipt_refs": [artifact.reference for artifact in receipt_artifacts],
        "evidence_refs": evidence,
        "contract_package_manifest_sha256": PACKAGE_MANIFEST_SHA256,
        "preselector_policy_sha256": load_packaged_json("resources/preselector_policy.v1.json")[
            "semantic_sha256"
        ],
        "quant_branch_policy_sha256": load_packaged_json("resources/quant_branch_policy.v1.json")[
            "semantic_sha256"
        ],
        "fundamental_branch_policy_sha256": load_packaged_json(
            "resources/fundamental_branch_policy.v1.json"
        )["semantic_sha256"],
        "fusion_policy_sha256": load_packaged_json("resources/fusion_policy.v1.json")[
            "semantic_sha256"
        ],
        "evidence_bound": "research_screening_bound",
        "effective_outer_blocks": 5,
        "active_refit_origins": origins[-60:],
        "outer_oos_origins": outer,
        "fold_inventory": folds,
        "oos_mean_hit60": "0.60",
        "oos_mean_q25_252": "0.10",
        "oos_p5_hit60": "0.51",
        "oos_p5_q25_252": "0.01",
        "authority": authority_envelope(),
    }
    if accepted:
        common.update(
            {
                "status": "PROMOTED",
                "accepted": True,
                "active_formal_research_weight": "0.50",
            }
        )
    else:
        common.update(
            {
                "status": "PROMOTION_REJECTED",
                "accepted": False,
                "evaluated_quant_weight": "0.50",
                "rejection_reasons": ["fusion_threshold_not_met"],
            }
        )
    return _write(
        closure,
        f"data/private/v17_v3_runs/{RUN_ID}/{common['promotion_id']}.json",
        common,
    )


def _formal_output(
    closure: StagedClosure,
    promotion: RuntimeArtifact,
) -> RuntimeArtifact:
    admission = admit_source_locator(
        closure.store,
        locator_path=str(closure.analyze_locator.relative_path),
        expected_locator_sha256=closure.analyze_locator.byte_sha256,
    )
    current_manifest = admission.documents["source_manifest"]
    raw_manifest = runtime_artifact(
        relative_path=current_manifest["parent_raw_manifest_ref"]["relative_path"],
        document=admission.documents["raw_source_manifest"],
    )
    formal_manifest = _source_manifest(
        closure.store,
        manifest_id="formal-analyze-manifest-1",
        phase="PORTFOLIO",
        sources=[
            *current_manifest["sources"],
            {
                "role": "fusion_calibration",
                "artifact_ref": next(
                    reference
                    for reference in promotion.document["evidence_refs"]
                    if reference["artifact_version"]
                    == "myquant.v17.v3.fusion-calibration-inputs.v1"
                ),
            },
            {
                "role": "fusion_promotion_receipt",
                "artifact_ref": promotion.reference,
            },
        ],
        parent=raw_manifest,
    )
    formal_locator = _source_locator(
        closure.store,
        locator_id="formal-analyze-locator-1",
        manifest=formal_manifest,
        preselection=closure.preselect_locator,
    )
    formal_admission = admit_source_locator(
        closure.store,
        locator_path=str(formal_locator.relative_path),
        expected_locator_sha256=formal_locator.byte_sha256,
    )
    formal_result = run_pipeline(
        PipelineRequest(
            mode=FORMAL_RESEARCH_MODE,
            admitted_sources=formal_admission,
        )
    )
    for artifact in formal_result.artifacts:
        write_typed_exact_once(closure.store, artifact)
    assert formal_result.terminal_artifact is not None
    return formal_result.terminal_artifact


def _skip_history_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep lifecycle-only tests focused after one full-history acceptance test."""

    monkeypatch.setattr(
        ActivationPublisher,
        "_validate_calibration_history",
        lambda self, calibration_input, *, resolved: None,
    )


def test_typed_active_idempotent_revoke_and_current_revalidation(
    tmp_path: Path,
) -> None:
    closure = build_staged_closure(tmp_path)
    promotion = _promotion(closure, accepted=True)
    formal = _formal_output(closure, promotion)
    publisher = ActivationPublisher(closure.store)
    active = publisher.activate(
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        promotion_receipt_bytes=promotion.raw,
        promotion_receipt_path=promotion.relative_path,
        expected_promotion_receipt_sha256=promotion.byte_sha256,
        formal_output_bytes=formal.raw,
        formal_output_path=formal.relative_path,
        expected_formal_output_sha256=formal.byte_sha256,
    )
    assert active.status == ACTIVE
    assert publisher.current_active(STRATEGY).status == ACTIVE
    retry = publisher.activate(
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        promotion_receipt_bytes=promotion.raw,
        promotion_receipt_path=promotion.relative_path,
        expected_promotion_receipt_sha256=promotion.byte_sha256,
        formal_output_bytes=formal.raw,
        formal_output_path=formal.relative_path,
        expected_formal_output_sha256=formal.byte_sha256,
    )
    assert retry.status == ACTIVE
    assert retry.idempotent is True
    revoked = publisher.revoke(
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        expected_active_receipt_sha256=str(active.receipt_sha256),
        reason="risk-change",
    )
    assert revoked.status == REVOKED
    assert publisher.current_active(STRATEGY).status == (NO_CURRENT_ACTIVE_FORMAL_RESULT)


def test_rejected_promotion_writes_only_activation_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_history_replay(monkeypatch)
    closure = build_staged_closure(tmp_path)
    promotion = _promotion(
        closure,
        accepted=False,
        persist_history=False,
    )
    publisher = ActivationPublisher(closure.store)
    outcome = publisher.activate(
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        promotion_receipt_bytes=promotion.raw,
        promotion_receipt_path=promotion.relative_path,
        expected_promotion_receipt_sha256=promotion.byte_sha256,
    )
    assert outcome.status == ACTIVATION_REJECTED
    assert closure.store.read_optional(publisher._activation_pointer(STRATEGY, CUTOFF)) is None
    assert closure.store.read_optional(publisher._latest_path(STRATEGY)) is None


def test_active_pointer_before_latest_is_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_history_replay(monkeypatch)
    closure = build_staged_closure(tmp_path)
    promotion = _promotion(
        closure,
        accepted=True,
        persist_history=False,
    )
    formal = _formal_output(closure, promotion)
    publisher = ActivationPublisher(closure.store)
    original = closure.store.replace_cas
    failed = False

    def fail_latest(path, expected_sha256, raw):
        nonlocal failed
        if path == publisher._latest_path(STRATEGY) and not failed:
            failed = True
            raise StorageError("injected latest failure")
        return original(path, expected_sha256, raw)

    monkeypatch.setattr(closure.store, "replace_cas", fail_latest)
    with pytest.raises(StorageError, match="injected"):
        publisher.activate(
            strategy_id=STRATEGY,
            cutoff=CUTOFF,
            promotion_receipt_bytes=promotion.raw,
            promotion_receipt_path=promotion.relative_path,
            expected_promotion_receipt_sha256=promotion.byte_sha256,
            formal_output_bytes=formal.raw,
            formal_output_path=formal.relative_path,
            expected_formal_output_sha256=formal.byte_sha256,
        )
    assert closure.store.read_optional(publisher._activation_pointer(STRATEGY, CUTOFF)) is not None
    assert publisher.current_active(STRATEGY).status == (NO_CURRENT_ACTIVE_FORMAL_RESULT)
    monkeypatch.setattr(closure.store, "replace_cas", original)
    repaired = publisher.activate(
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        promotion_receipt_bytes=promotion.raw,
        promotion_receipt_path=promotion.relative_path,
        expected_promotion_receipt_sha256=promotion.byte_sha256,
        formal_output_bytes=formal.raw,
        formal_output_path=formal.relative_path,
        expected_formal_output_sha256=formal.byte_sha256,
    )
    assert repaired.status == ACTIVE
    assert repaired.idempotent is True


def test_revoked_pointer_before_tombstone_fails_closed_and_repairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_history_replay(monkeypatch)
    closure = build_staged_closure(tmp_path)
    promotion = _promotion(
        closure,
        accepted=True,
        persist_history=False,
    )
    formal = _formal_output(closure, promotion)
    publisher = ActivationPublisher(closure.store)
    active = publisher.activate(
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        promotion_receipt_bytes=promotion.raw,
        promotion_receipt_path=promotion.relative_path,
        expected_promotion_receipt_sha256=promotion.byte_sha256,
        formal_output_bytes=formal.raw,
        formal_output_path=formal.relative_path,
        expected_formal_output_sha256=formal.byte_sha256,
    )
    latest_path = closure.root / publisher._latest_path(STRATEGY)
    latest_path.unlink()
    assert publisher.current_active(STRATEGY).status == (NO_CURRENT_ACTIVE_FORMAL_RESULT)
    original = closure.store.replace_cas
    failed = False

    def fail_tombstone(path, expected_sha256, raw):
        nonlocal failed
        if path == publisher._latest_path(STRATEGY) and not failed:
            failed = True
            raise StorageError("injected tombstone failure")
        return original(path, expected_sha256, raw)

    monkeypatch.setattr(closure.store, "replace_cas", fail_tombstone)
    with pytest.raises(StorageError, match="injected"):
        publisher.revoke(
            strategy_id=STRATEGY,
            cutoff=CUTOFF,
            expected_active_receipt_sha256=str(active.receipt_sha256),
            reason="risk-change",
        )
    assert publisher.current_active(STRATEGY).status == (NO_CURRENT_ACTIVE_FORMAL_RESULT)
    monkeypatch.setattr(closure.store, "replace_cas", original)
    repaired = publisher.revoke(
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        expected_active_receipt_sha256=str(active.receipt_sha256),
        reason="risk-change",
    )
    assert repaired.status == REVOKED
    assert repaired.idempotent is True
    latest = load_typed_artifact(
        latest_path.read_bytes(),
        label="repaired REVOKED latest",
        expected_version="myquant.v17.v3.formal-latest.v1",
    )
    assert latest["status"] == REVOKED


def test_revoked_publish_is_zero_write_and_all_lifecycle_json_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_history_replay(monkeypatch)
    closure = build_staged_closure(tmp_path)
    promotion = _promotion(
        closure,
        accepted=True,
        persist_history=False,
    )
    formal = _formal_output(closure, promotion)
    publisher = ActivationPublisher(closure.store)
    active = publisher.activate(
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        promotion_receipt_bytes=promotion.raw,
        promotion_receipt_path=promotion.relative_path,
        expected_promotion_receipt_sha256=promotion.byte_sha256,
        formal_output_bytes=formal.raw,
        formal_output_path=formal.relative_path,
        expected_formal_output_sha256=formal.byte_sha256,
    )
    publisher.revoke(
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        expected_active_receipt_sha256=str(active.receipt_sha256),
        reason="risk-change",
    )
    replay = publisher.publish_formal_result(
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        expected_active_receipt_sha256=str(active.receipt_sha256),
        deterministic_core_bytes=formal.raw,
    )
    assert replay.status == NO_CURRENT_ACTIVE_FORMAL_RESULT
    assert replay.write_count == 0
    for path in (closure.root / publisher._strategy_root(STRATEGY)).rglob("*.json"):
        load_typed_artifact(path.read_bytes(), label=path.name)


def test_activation_drift_has_zero_authority_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closure = build_staged_closure(tmp_path)
    promotion = _promotion(closure, accepted=True)
    formal = _formal_output(closure, promotion)
    publisher = ActivationPublisher(closure.store)

    calibration_ref = next(
        reference
        for reference in promotion.document["evidence_refs"]
        if reference["artifact_version"] == "myquant.v17.v3.fusion-calibration-inputs.v1"
    )
    calibration_raw = closure.store.read(
        calibration_ref["relative_path"],
        calibration_ref["byte_sha256"],
    )
    calibration = load_typed_artifact(
        calibration_raw,
        label="calibration input",
        expected_version="myquant.v17.v3.fusion-calibration-inputs.v1",
    )
    dangling_ref = calibration["payload"]["months"][0]["quant_branch_ref"]
    dangling_raw = closure.store.read(
        dangling_ref["relative_path"],
        dangling_ref["byte_sha256"],
    )
    dangling_path = closure.root / dangling_ref["relative_path"]
    dangling_path.unlink()
    with pytest.raises((ActivationError, StorageError)):
        publisher.activate(
            strategy_id=STRATEGY,
            cutoff=CUTOFF,
            promotion_receipt_bytes=promotion.raw,
            promotion_receipt_path=promotion.relative_path,
            expected_promotion_receipt_sha256=promotion.byte_sha256,
            formal_output_bytes=formal.raw,
            formal_output_path=formal.relative_path,
            expected_formal_output_sha256=formal.byte_sha256,
        )
    closure.store.write_exact_once(
        dangling_ref["relative_path"],
        dangling_raw,
    )

    first_month = calibration["payload"]["months"][0]
    historical_quant_raw = closure.store.read(
        first_month["quant_branch_ref"]["relative_path"],
        first_month["quant_branch_ref"]["byte_sha256"],
    )
    historical_quant = load_typed_artifact(
        historical_quant_raw,
        label="historical Quant branch",
        expected_version="myquant.v17.v3.branch-output.v1",
    )
    pool_ref = historical_quant["initial_pool_ref"]
    pool_raw = closure.store.read(
        pool_ref["relative_path"],
        pool_ref["byte_sha256"],
    )
    pool_path = closure.root / pool_ref["relative_path"]
    pool_path.unlink()
    with pytest.raises((ActivationError, StorageError)):
        publisher.activate(
            strategy_id=STRATEGY,
            cutoff=CUTOFF,
            promotion_receipt_bytes=promotion.raw,
            promotion_receipt_path=promotion.relative_path,
            expected_promotion_receipt_sha256=promotion.byte_sha256,
            formal_output_bytes=formal.raw,
            formal_output_path=formal.relative_path,
            expected_formal_output_sha256=formal.byte_sha256,
        )
    closure.store.write_exact_once(pool_ref["relative_path"], pool_raw)

    locator_ref = historical_quant["source_locator_ref"]
    locator_raw = closure.store.read(
        locator_ref["relative_path"],
        locator_ref["byte_sha256"],
    )
    locator_path = closure.root / locator_ref["relative_path"]
    locator_path.unlink()
    with pytest.raises((ActivationError, StorageError)):
        publisher.activate(
            strategy_id=STRATEGY,
            cutoff=CUTOFF,
            promotion_receipt_bytes=promotion.raw,
            promotion_receipt_path=promotion.relative_path,
            expected_promotion_receipt_sha256=promotion.byte_sha256,
            formal_output_bytes=formal.raw,
            formal_output_path=formal.relative_path,
            expected_formal_output_sha256=formal.byte_sha256,
        )
    closure.store.write_exact_once(locator_ref["relative_path"], locator_raw)

    pool_document = load_typed_artifact(
        pool_raw,
        label="historical initial pool",
        expected_version="myquant.v17.v3.initial-pool-output.v1",
    )
    distinct_pool_payload = dict(pool_document)
    distinct_pool_payload["output_id"] = "distinct-same-symbol-pool"
    distinct_pool_payload.pop("semantic_sha256")
    distinct_pool = _write(
        closure,
        str(Path(pool_ref["relative_path"]).with_name("distinct-same-symbol-pool.json")),
        distinct_pool_payload,
    )
    fundamental_ref = first_month["fundamental_branch_ref"]
    fundamental_raw = closure.store.read(
        fundamental_ref["relative_path"],
        fundamental_ref["byte_sha256"],
    )
    fundamental_document = load_typed_artifact(
        fundamental_raw,
        label="historical Fundamental branch",
        expected_version="myquant.v17.v3.branch-output.v1",
    )
    distinct_fundamental_payload = dict(fundamental_document)
    distinct_fundamental_payload["output_id"] = "distinct-pool-fundamental"
    distinct_fundamental_payload["initial_pool_ref"] = distinct_pool.reference
    distinct_fundamental_payload.pop("semantic_sha256")
    distinct_fundamental = _write(
        closure,
        str(Path(fundamental_ref["relative_path"]).with_name("distinct-pool-fundamental.json")),
        distinct_fundamental_payload,
    )
    drifted_calibration_payload = dict(calibration)
    drifted_calibration_payload["input_id"] = "distinct-pool-calibration-input"
    drifted_calibration_payload["payload"] = dict(calibration["payload"])
    drifted_calibration_payload["payload"]["months"] = [
        dict(month) for month in calibration["payload"]["months"]
    ]
    drifted_calibration_payload["payload"]["months"][0][
        "fundamental_branch_ref"
    ] = distinct_fundamental.reference
    drifted_calibration_payload.pop("semantic_sha256")
    distinct_calibration = _write(
        closure,
        str(
            Path(calibration_ref["relative_path"]).with_name("distinct-pool-calibration-input.json")
        ),
        drifted_calibration_payload,
    )
    with pytest.raises(ActivationError, match="same exact pool/locator"):
        publisher._validate_calibration_history(
            distinct_calibration,
            resolved={},
        )

    quant_receipt_ref = promotion.document["calibration_receipt_refs"][0]
    quant_receipt_raw = closure.store.read(
        quant_receipt_ref["relative_path"],
        quant_receipt_ref["byte_sha256"],
    )
    quant_receipt = load_typed_artifact(
        quant_receipt_raw,
        label="Quant calibration receipt",
        expected_version="myquant.v17.v3.fusion-calibration-receipt.v1",
    )
    substituted_receipt_payload = dict(quant_receipt)
    substituted_receipt_payload["calibration_id"] = "preselect-substituted-quant"
    substituted_receipt_payload["evidence_refs"] = [closure.preselect_locator.reference]
    substituted_receipt_payload.pop("semantic_sha256")
    substituted_receipt = _write(
        closure,
        str(Path(quant_receipt_ref["relative_path"]).with_name("preselect-substituted-quant.json")),
        substituted_receipt_payload,
    )
    substituted_promotion_payload = dict(promotion.document)
    substituted_promotion_payload["promotion_id"] = "preselect-substituted-promotion"
    substituted_promotion_payload["calibration_receipt_refs"] = [
        substituted_receipt.reference,
        *promotion.document["calibration_receipt_refs"][1:],
    ]
    substituted_promotion_payload.pop("semantic_sha256")
    substituted_promotion = _write(
        closure,
        str(promotion.relative_path.with_name("preselect-substituted-promotion.json")),
        substituted_promotion_payload,
    )
    with pytest.raises(ActivationError, match="exact phase locator"):
        publisher._validate_promotion_closure(substituted_promotion)

    def fail_runtime_build() -> dict[str, str]:
        raise PackageResourceError("injected runtime source drift")

    with monkeypatch.context() as context:
        context.setattr(
            runtime_activation,
            "verify_runtime_build",
            fail_runtime_build,
        )
        with pytest.raises(ActivationError, match="runtime build identity"):
            publisher.activate(
                strategy_id=STRATEGY,
                cutoff=CUTOFF,
                promotion_receipt_bytes=promotion.raw,
                promotion_receipt_path=promotion.relative_path,
                expected_promotion_receipt_sha256=promotion.byte_sha256,
                formal_output_bytes=formal.raw,
                formal_output_path=formal.relative_path,
                expected_formal_output_sha256=formal.byte_sha256,
            )

    drifted = dict(formal.document)
    refs = [dict(reference) for reference in drifted["artifact_refs"]]
    refs[0]["byte_sha256"] = "0" * 64
    drifted["artifact_refs"] = refs
    drifted["output_id"] = "formal-run-drifted"
    drifted.pop("semantic_sha256")
    drifted_artifact = _write(
        closure,
        str(formal.relative_path.with_name("formal_output_drifted.json")),
        drifted,
    )
    with monkeypatch.context() as context:
        _skip_history_replay(context)
        with pytest.raises((ActivationError, StorageError)):
            publisher.activate(
                strategy_id=STRATEGY,
                cutoff=CUTOFF,
                promotion_receipt_bytes=promotion.raw,
                promotion_receipt_path=promotion.relative_path,
                expected_promotion_receipt_sha256=promotion.byte_sha256,
                formal_output_bytes=drifted_artifact.raw,
                formal_output_path=drifted_artifact.relative_path,
                expected_formal_output_sha256=drifted_artifact.byte_sha256,
            )
    assert closure.store.read_optional(publisher._activation_pointer(STRATEGY, CUTOFF)) is None
    assert closure.store.read_optional(publisher._latest_path(STRATEGY)) is None
    assert closure.store.read_optional(publisher._receipt_path(STRATEGY, CUTOFF, ACTIVE)) is None


def test_rejected_then_promoted_same_cutoff_is_terminal_zero_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _skip_history_replay(monkeypatch)
    closure = build_staged_closure(tmp_path)
    rejected = _promotion(
        closure,
        accepted=False,
        persist_history=False,
    )
    promoted = _promotion(
        closure,
        accepted=True,
        persist_history=False,
    )
    formal = _formal_output(closure, promoted)
    publisher = ActivationPublisher(closure.store)
    assert (
        publisher.activate(
            strategy_id=STRATEGY,
            cutoff=CUTOFF,
            promotion_receipt_bytes=rejected.raw,
            promotion_receipt_path=rejected.relative_path,
            expected_promotion_receipt_sha256=rejected.byte_sha256,
        ).status
        == ACTIVATION_REJECTED
    )
    blocked = publisher.activate(
        strategy_id=STRATEGY,
        cutoff=CUTOFF,
        promotion_receipt_bytes=promoted.raw,
        promotion_receipt_path=promoted.relative_path,
        expected_promotion_receipt_sha256=promoted.byte_sha256,
        formal_output_bytes=formal.raw,
        formal_output_path=formal.relative_path,
        expected_formal_output_sha256=formal.byte_sha256,
    )
    assert blocked.status == "SAME_CUTOFF_TERMINAL_CONFLICT"
    assert blocked.write_count == 0
    assert closure.store.read_optional(publisher._activation_pointer(STRATEGY, CUTOFF)) is None


@pytest.mark.parametrize(
    ("fundamental_accepted", "expected_status"),
    ((True, "FUSION_CALIBRATED"), (False, "PROMOTION_REJECTED")),
)
def test_calibrate_consumes_upstream_receipts_and_mints_only_fusion_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fundamental_accepted: bool,
    expected_status: str,
) -> None:
    closure = build_staged_closure(tmp_path)
    calibration_input = _calibration_input(
        closure,
        persist_history=False,
    )
    raw_manifest, preselect_admission = _preselect_raw_manifest(closure)
    phase_locators = {
        kind: _calibration_gate_locator(
            closure,
            kind=kind,
            raw_manifest=raw_manifest,
        )
        for kind in ("QUANT_TIMING", "FUNDAMENTAL_FORWARD")
    }

    def upstream(kind: str, accepted: bool) -> RuntimeArtifact:
        return _write(
            closure,
            ("data/private/v17_v3_sources/derived/" f"{kind.casefold()}-receipt.json"),
            {
                "version": "myquant.v17.v3.fusion-calibration-receipt.v1",
                "protocol_version": PROTOCOL_VERSION,
                "calibration_id": f"upstream-{kind.casefold()}",
                "calibration_kind": kind,
                "strategy_id": STRATEGY,
                "cutoff": CUTOFF,
                "created_at": CUTOFF,
                "observation_end_at": CUTOFF,
                "accepted": accepted,
                "evidence_refs": [phase_locators[kind].reference],
                "authority": authority_envelope(),
            },
        )

    quant_receipt = upstream("QUANT_TIMING", True)
    fundamental_receipt = upstream(
        "FUNDAMENTAL_FORWARD",
        fundamental_accepted,
    )
    fusion_raw_manifest = _fusion_raw_manifest(
        closure,
        calibration_raw=raw_manifest,
        preselect_admission=preselect_admission,
        manifest_id=(
            "fusion-calibration-raw-accepted"
            if fundamental_accepted
            else "fusion-calibration-raw-rejected"
        ),
    )
    fusion_manifest = _source_manifest(
        closure.store,
        manifest_id=(
            "fusion-calibration-manifest-accepted"
            if fundamental_accepted
            else "fusion-calibration-manifest-rejected"
        ),
        phase="FUSION_PROMOTION",
        sources=[
            {
                "role": "fundamental_branch_output",
                "artifact_ref": closure.fundamental_branch.reference,
            },
            {
                "role": "fundamental_forward_calibration",
                "artifact_ref": fundamental_receipt.reference,
            },
            {
                "role": "fusion_calibration",
                "artifact_ref": calibration_input.reference,
            },
            {
                "role": "initial_pool_output",
                "artifact_ref": closure.initial_pool.reference,
            },
            {
                "role": "quant_branch_output",
                "artifact_ref": closure.quant_branch.reference,
            },
            {
                "role": "quant_preselection_inputs",
                "artifact_ref": preselect_admission.reference_for_role("quant_preselection_inputs"),
            },
            {
                "role": "quant_timing_calibration",
                "artifact_ref": quant_receipt.reference,
            },
        ],
        parent=fusion_raw_manifest,
    )
    fusion_locator = _source_locator(
        closure.store,
        locator_id=(
            "fusion-calibration-locator-accepted"
            if fundamental_accepted
            else "fusion-calibration-locator-rejected"
        ),
        manifest=fusion_manifest,
        preselection=closure.preselect_locator,
    )

    original_read_exact = runtime_service._read_exact_artifact_ref

    def fake_month_ref(
        store,
        reference,
        *,
        expected_version,
        expected_strategy_id,
        expected_cutoff,
        label,
    ):
        if expected_version != "myquant.v17.v3.branch-output.v1":
            return original_read_exact(
                store,
                reference,
                expected_version=expected_version,
                expected_strategy_id=expected_strategy_id,
                expected_cutoff=expected_cutoff,
                label=label,
            )
        return {
            "branch": ("quant" if "Quant" in label else "fundamental"),
            "cutoff": expected_cutoff,
            "ordered_domain": ["000001.SZ"],
        }

    dates = [date(2026, 3, 27) + timedelta(days=index) for index in range(120)]
    folds = tuple(
        SimpleNamespace(
            index=index + 1,
            training_origins=tuple(dates[:60]),
            oos_origins=tuple(dates[60 + index * 12 : 72 + index * 12]),
            selected_weight=Decimal("0.50"),
        )
        for index in range(5)
    )
    result = SimpleNamespace(
        promoted=True,
        active_weight=Decimal("0.50"),
        bootstrap_matrix_sha256="8" * 64,
        blockers=(),
        folds=folds,
        evidence_bound="research_screening_bound",
        effective_outer_blocks=5,
        oos_mean_hit60=Decimal("0.60"),
        oos_mean_q25_252=Decimal("0.10"),
        oos_p5_hit60=Decimal("0.51"),
        oos_p5_q25_252=Decimal("0.01"),
    )
    monkeypatch.setattr(
        runtime_service,
        "_read_exact_artifact_ref",
        fake_month_ref,
    )
    monkeypatch.setattr(algorithms, "calibrate_fusion", lambda *args, **kwargs: result)
    outcome = runtime_service.calibrate(
        workspace_root=closure.root,
        locator_path=str(fusion_locator.relative_path),
        expected_locator_sha256=fusion_locator.byte_sha256,
    )
    assert outcome.status == expected_status
    promotion_raw = closure.store.read(
        str(outcome.promotion_path),
        str(outcome.promotion_sha256),
    )
    promotion = load_typed_artifact(
        promotion_raw,
        label="persisted promotion",
        expected_version="myquant.v17.v3.fusion-promotion-receipt.v1",
    )
    assert promotion["status"] == ("PROMOTED" if fundamental_accepted else "PROMOTION_REJECTED")
    calibration_dir = closure.root / "data/private/v17_v3_runs" / RUN_ID / "calibration"
    assert [path.name for path in calibration_dir.glob("*.json")] == ["fusion_promotion.json"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "factor_baseline_mode",
            "PROVISIONAL_RESEARCH",
            "activation_rejects_provisional_factor_baseline",
        ),
        (
            "portfolio_basis",
            "MODEL_ONLY_NO_PRIVATE_HOLDINGS",
            "activation_rejects_model_only_portfolio",
        ),
    ],
)
def test_activation_rejects_nonformal_profiles_before_any_write(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    closure = build_staged_closure(tmp_path)
    promotion = _promotion(closure, accepted=True)
    formal = _formal_output(closure, promotion)
    document = dict(formal.document)
    document[field] = value
    document.pop("semantic_sha256")
    raw = canonical_resource_bytes(seal_semantic(document))
    path = f"data/private/v17_v3_runs/{RUN_ID}/{field}-rejected.json"
    closure.store.write_exact_once(path, raw)
    before = sorted(
        item.relative_to(closure.root) for item in closure.root.rglob("*") if item.is_file()
    )
    with pytest.raises(ActivationError, match=message):
        ActivationPublisher(closure.store).activate(
            strategy_id=STRATEGY,
            cutoff=CUTOFF,
            promotion_receipt_bytes=promotion.raw,
            promotion_receipt_path=promotion.relative_path,
            expected_promotion_receipt_sha256=promotion.byte_sha256,
            formal_output_bytes=raw,
            formal_output_path=path,
            expected_formal_output_sha256=hashlib.sha256(raw).hexdigest(),
        )
    after = sorted(
        item.relative_to(closure.root) for item in closure.root.rglob("*") if item.is_file()
    )
    assert after == before
