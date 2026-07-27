from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import shutil

import pytest

from quant_investor.v17_v3_contract import (
    FORMAL_RESEARCH_RESULTS_ROOT,
    PROTOCOL_VERSION,
    RUNS_ROOT,
    SHADOW_RESULTS_ROOT,
    SOURCES_ROOT,
    artifact_identity_field,
    build_artifact_ref,
    canonical_bytes,
    verify_package,
)
from quant_investor.v17_v3_contract.action_matrix import (
    ACTIVATION_STATUSES,
    STATES,
    TERMINAL_STATES,
    decide_action,
)
from quant_investor.v17_v3_contract.canonical import (
    canonical_resource_bytes,
    seal_semantic,
)
from quant_investor.v17_v3_contract.policy import (
    SOURCE_PHASES,
    source_role_matrix,
    source_role_registries,
)
from quant_investor.v17_v3_contract.namespace import (
    derive_namespace_path,
    formal_run_path,
    shadow_run_path,
)
from quant_investor.v17_v3_contract.resources import (
    PackageResourceError,
    load_packaged_json,
    package_resource_session,
    verify_runtime_build,
)
from quant_investor.v17_v3_contract.validators import (
    ArtifactContractError,
    validate_activation_transition,
    validate_branch_same_pool_binding,
    validate_portfolio_output,
    validate_staged_analysis_lineage,
)

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "quant_investor" / "v17_v3_contract"
CUTOFF = "2026-07-25T07:00:00Z"
STRATEGY = "quant-first"


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _authority(*, formal: bool = False) -> dict[str, bool]:
    return {
        "broker_authority": False,
        "execution_authority": False,
        "formal_research_publication_authority": formal,
        "order_authority": False,
        "production_default": False,
        "trade_authority": False,
    }


def _policy_sha(name: str) -> str:
    return str(load_packaged_json(f"resources/{name}.v1.json")["semantic_sha256"])


def _factor_v4_readiness_ref() -> dict[str, str]:
    return _ref(
        "factor-readiness-1",
        "myquant.v17.v3.factor-governance-readiness.v1",
        "data/private/v17_v3_sources/objects/factor-readiness.json",
    )


def _ref(
    artifact_id: str,
    version: str,
    path: str,
    *,
    semantic: str | None = None,
) -> dict[str, str]:
    return {
        "artifact_id": artifact_id,
        "artifact_version": version,
        "byte_sha256": _sha(f"bytes:{artifact_id}"),
        "cutoff": CUTOFF,
        "relative_path": path,
        "semantic_sha256": semantic or _sha(f"semantic:{artifact_id}"),
        "strategy_id": STRATEGY,
    }


def _active_receipt() -> dict[str, object]:
    return seal_semantic(
        {
            "activated_at": "2026-07-25T07:10:00Z",
            "authority": _authority(formal=True),
            "cutoff": CUTOFF,
            "formal_output_ref": _ref(
                "formal-1",
                "myquant.v17.v3.formal-research-output.v1",
                "results/v17_v3_formal_research/strategies/quant-first/runs/run-1/formal.json",
            ),
            "promotion_receipt_ref": _ref(
                "promotion-1",
                "myquant.v17.v3.fusion-promotion-receipt.v1",
                "data/private/v17_v3_runs/run-1/promotion.json",
            ),
            "protocol_version": PROTOCOL_VERSION,
            "receipt_id": "active-1",
            "status": "ACTIVE",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.activation-receipt.v1",
        }
    )


def _active_pointer(active: dict[str, object]) -> dict[str, object]:
    return seal_semantic(
        {
            "active_receipt_ref": _ref(
                "active-1",
                "myquant.v17.v3.activation-receipt.v1",
                "results/v17_v3_formal_research/strategies/quant-first/activations/cutoff/receipts/active.json",
                semantic=str(active["semantic_sha256"]),
            ),
            "authority": _authority(formal=True),
            "cutoff": CUTOFF,
            "formal_output_ref": active["formal_output_ref"],
            "pointer_id": "pointer-1",
            "protocol_version": PROTOCOL_VERSION,
            "status": "ACTIVE",
            "strategy_id": STRATEGY,
            "updated_at": "2026-07-25T07:10:01Z",
            "version": "myquant.v17.v3.activation-pointer.v1",
        }
    )


def test_stable_surface_fixed_roots_and_package_seal() -> None:
    assert PROTOCOL_VERSION == "myquant.v17.v3"
    assert SOURCES_ROOT == "data/private/v17_v3_sources"
    assert RUNS_ROOT == "data/private/v17_v3_runs"
    assert SHADOW_RESULTS_ROOT == "results/v17_v3_shadow"
    assert FORMAL_RESEARCH_RESULTS_ROOT == "results/v17_v3_formal_research"
    assert (
        load_packaged_json("resources/preselector_policy.v1.json")["max_initial_pool_size"] == 500
    )
    assert canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert verify_package()


def test_runtime_build_manifest_detects_source_drift(tmp_path: Path) -> None:
    copied_root = tmp_path / "quant_investor"
    copied_contract = copied_root / "v17_v3_contract"
    copied_runtime = copied_root / "v17_v3_runtime"
    shutil.copytree(PACKAGE_ROOT, copied_contract)
    shutil.copytree(ROOT / "quant_investor" / "v17_v3_runtime", copied_runtime)
    assert verify_runtime_build(package_root=copied_contract)
    target = copied_runtime / "algorithms" / "branch_fusion.py"
    target.write_bytes(target.read_bytes() + b"\n# injected drift\n")
    with pytest.raises(PackageResourceError, match="byte SHA-256 mismatch"):
        verify_runtime_build(package_root=copied_contract)


def test_package_resource_session_does_not_cache_across_transactions(
    tmp_path: Path,
) -> None:
    copied_contract = tmp_path / "v17_v3_contract"
    shutil.copytree(PACKAGE_ROOT, copied_contract)
    with package_resource_session():
        assert (
            load_packaged_json(
                "schemas/common.v1.schema.json",
                package_root=copied_contract,
            )["$id"]
            == "myquant.v17.v3.common.schema.v1"
        )
    schema = copied_contract / "schemas/common.v1.schema.json"
    schema.write_bytes(schema.read_bytes() + b" ")
    with pytest.raises(PackageResourceError, match="byte SHA-256 mismatch"):
        load_packaged_json(
            "schemas/common.v1.schema.json",
            package_root=copied_contract,
        )


def test_contract_package_has_no_v2_or_runtime_imports() -> None:
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
        assert all("v17_v2" not in name for name in imported)
        assert all("v17_v3_runtime" not in name for name in imported)


def test_state_and_role_registries_are_closed() -> None:
    assert {"ACTIVE", "REVOKED"}.isdisjoint(STATES)
    assert ACTIVATION_STATUSES == ("ACTIVATION_REJECTED", "ACTIVE", "REVOKED")
    assert "ACTIVATION_REJECTED" in TERMINAL_STATES
    assert {"ACTIVE", "REVOKED"}.isdisjoint(TERMINAL_STATES)
    assert decide_action(
        action="REVOKE_FORMAL_RESEARCH",
        state="ACTIVE",
    ).targets == ("REVOKED",)
    raw, derived = source_role_registries()
    assert not set(raw) & set(derived)
    assert {
        "initial_pool_output",
        "quant_preselection_inputs",
        "quant_branch_output",
        "fundamental_branch_output",
        "fusion_calibration",
        "quant_timing_calibration_inputs",
        "fundamental_forward_calibration_inputs",
        "permissions",
    }.issubset(derived)
    matrix = source_role_matrix()
    assert tuple(matrix) == SOURCE_PHASES
    registry = set(raw) | set(derived)
    for requirement in matrix.values():
        partition = (
            *requirement.required_roles,
            *requirement.optional_roles,
            *requirement.forbidden_roles,
        )
        assert len(partition) == len(set(partition))
        assert set(partition) == registry


def test_namespace_templates_derive_exact_strategy_scoped_topology() -> None:
    digest = _sha("receipt")
    assert formal_run_path(
        strategy_id=STRATEGY,
        run_id="run-1",
    ).as_posix() == ("results/v17_v3_formal_research/strategies/quant-first/runs/run-1")
    assert (
        shadow_run_path(
            strategy_id=STRATEGY,
            run_id="run-1",
        ).as_posix()
        == "results/v17_v3_shadow/strategies/quant-first/runs/run-1"
    )
    assert (
        derive_namespace_path(
            "SHADOW_LATEST",
            strategy_id=STRATEGY,
        ).as_posix()
        == "results/v17_v3_shadow/strategies/quant-first/_latest.json"
    )
    assert derive_namespace_path(
        "FORMAL_ACTIVATION_RECEIPT",
        strategy_id=STRATEGY,
        cutoff_id="20260725t070000z",
        status="active",
    ).as_posix() == (
        "results/v17_v3_formal_research/strategies/quant-first/activations/"
        "20260725t070000z/receipts/active.json"
    )
    assert derive_namespace_path(
        "FORMAL_UNPUBLISHED_EVIDENCE",
        strategy_id=STRATEGY,
        byte_sha256=digest,
    ).as_posix() == (
        "results/v17_v3_formal_research/strategies/quant-first/unpublished/" f"{digest}.json"
    )


def test_single_identity_registry_builds_exact_seven_field_reference() -> None:
    manifest_ref = _ref(
        "raw-1",
        "myquant.v17.v3.source-manifest.v1",
        "data/private/v17_v3_sources/manifests/raw-1.json",
    )
    locator = seal_semantic(
        {
            "authority": _authority(),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "locator_id": "locator-1",
            "preselection_locator_ref": None,
            "protocol_version": PROTOCOL_VERSION,
            "source_manifest_ref": manifest_ref,
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.source-locator.v1",
        }
    )
    raw = canonical_resource_bytes(locator)
    ref = build_artifact_ref(
        locator,
        raw,
        "data/private/v17_v3_sources/locators/quant-first/preselect.json",
    )
    assert artifact_identity_field(locator["version"]) == "locator_id"
    assert set(ref) == {
        "artifact_id",
        "artifact_version",
        "byte_sha256",
        "cutoff",
        "relative_path",
        "semantic_sha256",
        "strategy_id",
    }
    assert ref["artifact_id"] == "locator-1"


def test_branch_complete_domain_and_strong_same_pool_binding() -> None:
    locator_ref = _ref(
        "preselect-locator",
        "myquant.v17.v3.source-locator.v1",
        "data/private/v17_v3_sources/locators/quant-first/preselect.json",
    )
    pool_ref = _ref(
        "pool-1",
        "myquant.v17.v3.initial-pool-output.v1",
        "data/private/v17_v3_runs/run-1/initial_pool.json",
    )
    symbols = ["000001.SZ", "600000.SH"]
    branch = seal_semantic(
        {
            "authority": _authority(),
            "branch": "quant",
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "initial_pool_count": 2,
            "initial_pool_ref": pool_ref,
            "initial_pool_symbol_order_sha256": hashlib.sha256(
                canonical_bytes(symbols)
            ).hexdigest(),
            "ordered_domain": symbols,
            "output_id": "quant-branch-1",
            "policy_sha256": _policy_sha("quant_branch_policy"),
            "protocol_version": PROTOCOL_VERSION,
            "records": [
                {"reason": None, "score": "0.9", "status": "READY", "symbol": "000001.SZ"},
                {
                    "reason": "missing",
                    "score": None,
                    "status": "UNAVAILABLE",
                    "symbol": "600000.SH",
                },
            ],
            "run_id": "run-1",
            "source_locator_ref": locator_ref,
            "state": "BRANCHES_COMPLETE",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.branch-output.v1",
        }
    )
    expected = {
        "initial_pool_count": 2,
        "initial_pool_ref": pool_ref,
        "initial_pool_symbol_order_sha256": branch["initial_pool_symbol_order_sha256"],
        "policy_sha256": _policy_sha("quant_branch_policy"),
        "source_locator_ref": locator_ref,
    }
    assert (
        validate_branch_same_pool_binding(
            branch,
            expected_bindings=expected,
        ).payload["branch"]
        == "quant"
    )
    policy_drift = dict(branch)
    policy_drift["policy_sha256"] = _sha("unpackaged-policy")
    policy_drift.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="packaged policy"):
        validate_branch_same_pool_binding(
            seal_semantic(policy_drift),
            expected_bindings=expected,
        )
    branch["records"] = list(reversed(branch["records"]))
    branch.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="exactly one"):
        validate_branch_same_pool_binding(seal_semantic(branch), expected_bindings=expected)


def test_activation_absent_active_rejected_and_active_revoked_pointer_cas() -> None:
    active = _active_receipt()
    active_pointer = _active_pointer(active)
    assert (
        validate_activation_transition(
            active,
            proposed_pointer=active_pointer,
        ).next_pointer
        is not None
    )
    predecessor_ref = active_pointer["active_receipt_ref"]
    revoked = seal_semantic(
        {
            "authority": _authority(),
            "cutoff": CUTOFF,
            "predecessor_active_receipt_ref": predecessor_ref,
            "protocol_version": PROTOCOL_VERSION,
            "reason": "evidence revoked",
            "receipt_id": "revoked-1",
            "revoked_at": "2026-07-25T07:20:00Z",
            "status": "REVOKED",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.activation-receipt.v1",
        }
    )
    revoked_pointer = seal_semantic(
        {
            "authority": _authority(),
            "cutoff": CUTOFF,
            "pointer_id": "pointer-1",
            "predecessor_active_receipt_ref": predecessor_ref,
            "protocol_version": PROTOCOL_VERSION,
            "revocation_receipt_ref": _ref(
                "revoked-1",
                "myquant.v17.v3.activation-receipt.v1",
                "results/v17_v3_formal_research/strategies/quant-first/activations/cutoff/receipts/revoked.json",
                semantic=str(revoked["semantic_sha256"]),
            ),
            "status": "REVOKED",
            "strategy_id": STRATEGY,
            "updated_at": "2026-07-25T07:20:01Z",
            "version": "myquant.v17.v3.activation-pointer.v1",
        }
    )
    transition = validate_activation_transition(
        revoked,
        predecessor_active=active,
        current_pointer=active_pointer,
        proposed_pointer=revoked_pointer,
    )
    assert transition.next_pointer is not None
    assert transition.next_pointer.payload["status"] == "REVOKED"
    rejected = seal_semantic(
        {
            "authority": _authority(),
            "cutoff": CUTOFF,
            "promotion_receipt_ref": active["promotion_receipt_ref"],
            "protocol_version": PROTOCOL_VERSION,
            "receipt_id": "rejected-1",
            "rejected_at": "2026-07-25T07:09:00Z",
            "rejection_reasons": ["lower_bound_failed"],
            "status": "ACTIVATION_REJECTED",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.activation-receipt.v1",
        }
    )
    assert validate_activation_transition(rejected).next_pointer is None
    with pytest.raises(ArtifactContractError, match="cannot be reactivated"):
        validate_activation_transition(
            active,
            proposed_pointer=active_pointer,
            history=[revoked],
        )


def test_review_only_holding_cannot_enter_pool_or_receive_positive_delta() -> None:
    portfolio = seal_semantic(
        {
            "allocation_policy_sha256": _policy_sha("portfolio_allocation_policy"),
            "authority": _authority(),
            "blockers": [],
            "cash_weight": "0.70",
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "deep_output_ref": _ref(
                "deep-1",
                "myquant.v17.v3.deep-output.v1",
                "data/private/v17_v3_runs/run-1/deep.json",
            ),
            "fusion_output_ref": _ref(
                "fusion-1",
                "myquant.v17.v3.fusion-output.v1",
                "data/private/v17_v3_runs/run-1/fusion.json",
            ),
            "factor_baseline_mode": "FACTOR_V4_PRODUCTION",
            "factor_baseline_ref": _factor_v4_readiness_ref(),
            "gross_weight": "0.30",
            "holdings_snapshot_ref": _ref(
                "holdings-1",
                "myquant.v17.v3.holdings-snapshot.v1",
                "data/private/v17_v3_sources/objects/holdings.json",
            ),
            "output_id": "portfolio-1",
            "overlay_stages": [
                {
                    "overlay_ref": None,
                    "stage": "MACRO",
                    "status": "UNAVAILABLE_NO_OP",
                },
                {
                    "overlay_ref": None,
                    "stage": "MARKOV",
                    "status": "UNAVAILABLE_NO_OP",
                },
            ],
            "permissions_ref": _ref(
                "permissions-1",
                "myquant.v17.v3.pretrade-permissions.v1",
                "data/private/v17_v3_runs/run-1/permissions.json",
            ),
            "portfolio_basis": "HOLDINGS_AWARE",
            "protocol_version": PROTOCOL_VERSION,
            "review_only_holdings": ["600000.SH"],
            "run_id": "run-1",
            "selection_pool_symbols": ["000001.SZ"],
            "status": "COMPLETE",
            "strategy_id": STRATEGY,
            "targets": [
                {
                    "current_target": "0",
                    "final_target": "0.20",
                    "lane": "SELECTION_POOL",
                    "symbol": "000001.SZ",
                },
                {
                    "current_target": "0.15",
                    "final_target": "0.10",
                    "lane": "REVIEW_ONLY_HOLDING",
                    "symbol": "600000.SH",
                },
            ],
            "version": "myquant.v17.v3.portfolio-output.v1",
        }
    )
    assert validate_portfolio_output(portfolio).payload["status"] == "COMPLETE"
    portfolio["targets"][1]["final_target"] = "0.16"
    portfolio["gross_weight"] = "0.36"
    portfolio["cash_weight"] = "0.64"
    portfolio.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="positive target delta"):
        validate_portfolio_output(seal_semantic(portfolio))


def test_cycle_free_staged_locator_pool_branch_lineage() -> None:
    raw_ref = _ref(
        "raw-1",
        "myquant.v17.v3.source-manifest.v1",
        "data/private/v17_v3_sources/manifests/raw-1.json",
    )
    preselect_ref = _ref(
        "preselect-locator",
        "myquant.v17.v3.source-locator.v1",
        "data/private/v17_v3_sources/locators/quant-first/preselect.json",
    )
    symbols = ["000001.SZ"]
    preselector_inventory = load_packaged_json("resources/preselector_policy.v1.json")[
        "factor_inventory"
    ]
    pool = seal_semantic(
        {
            "authority": _authority(),
            "blockers": [],
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "dispositions": [
                {
                    "reasons": [],
                    "score": "1",
                    "selected": True,
                    "status": "READY",
                    "symbol": "000001.SZ",
                }
            ],
            "factor_baseline_mode": "FACTOR_V4_PRODUCTION",
            "factor_baseline_ref": _factor_v4_readiness_ref(),
            "factor_coverage": [
                {"coverage": "1", "factor_id": row["factor_id"]} for row in preselector_inventory
            ],
            "history_required": 120,
            "ordered_domain": symbols,
            "output_id": "pool-1",
            "policy_sha256": _policy_sha("preselector_policy"),
            "pool_count": 1,
            "pool_symbol_order_sha256": hashlib.sha256(canonical_bytes(symbols)).hexdigest(),
            "protocol_version": PROTOCOL_VERSION,
            "raw_source_manifest_ref": raw_ref,
            "ready_domain": symbols,
            "run_id": "run-1",
            "selected_symbols": symbols,
            "source_locator_ref": preselect_ref,
            "state": "PRESELECT_COMPLETE",
            "status": "READY",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.initial-pool-output.v1",
        }
    )
    pool_ref = _ref(
        "pool-1",
        "myquant.v17.v3.initial-pool-output.v1",
        "data/private/v17_v3_runs/run-1/pool.json",
        semantic=str(pool["semantic_sha256"]),
    )

    def branch(kind: str) -> dict[str, object]:
        return seal_semantic(
            {
                "authority": _authority(),
                "branch": kind,
                "created_at": CUTOFF,
                "cutoff": CUTOFF,
                "initial_pool_count": 1,
                "initial_pool_ref": pool_ref,
                "initial_pool_symbol_order_sha256": pool["pool_symbol_order_sha256"],
                "ordered_domain": symbols,
                "output_id": f"{kind}-1",
                "policy_sha256": _policy_sha(f"{kind}_branch_policy"),
                "protocol_version": PROTOCOL_VERSION,
                "records": [
                    {"reason": None, "score": "1", "status": "READY", "symbol": "000001.SZ"}
                ],
                "run_id": "run-1",
                "source_locator_ref": preselect_ref,
                "state": "BRANCHES_COMPLETE",
                "strategy_id": STRATEGY,
                "version": "myquant.v17.v3.branch-output.v1",
            }
        )

    quant = branch("quant")
    fundamental = branch("fundamental")
    role_refs = {
        "deep_research_inputs": _ref(
            "deep-inputs-1",
            "myquant.v17.v3.deep-research-inputs.v1",
            "data/private/v17_v3_runs/run-1/deep-inputs.json",
        ),
        "fundamental_branch_output": _ref(
            "fundamental-1",
            "myquant.v17.v3.branch-output.v1",
            "data/private/v17_v3_runs/run-1/fundamental.json",
            semantic=str(fundamental["semantic_sha256"]),
        ),
        "initial_pool_output": pool_ref,
        "permissions": _ref(
            "permissions-1",
            "myquant.v17.v3.pretrade-permissions.v1",
            "data/private/v17_v3_runs/run-1/permissions.json",
        ),
        "quant_branch_output": _ref(
            "quant-1",
            "myquant.v17.v3.branch-output.v1",
            "data/private/v17_v3_runs/run-1/quant.json",
            semantic=str(quant["semantic_sha256"]),
        ),
        "quant_preselection_inputs": _ref(
            "pre-inputs-1",
            "myquant.v17.v3.quant-preselection-inputs.v1",
            "data/private/v17_v3_runs/run-1/pre-inputs.json",
        ),
    }
    manifest = seal_semantic(
        {
            "authority": _authority(),
            "closure_kind": "DERIVED_CLOSURE",
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "manifest_id": "analyze-manifest-1",
            "parent_raw_manifest_ref": raw_ref,
            "phase": "PORTFOLIO",
            "protocol_version": PROTOCOL_VERSION,
            "sources": [
                {"artifact_ref": role_refs[role], "role": role} for role in sorted(role_refs)
            ],
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.source-manifest.v1",
        }
    )
    locator = seal_semantic(
        {
            "authority": _authority(),
            "created_at": CUTOFF,
            "cutoff": CUTOFF,
            "locator_id": "analyze-locator-1",
            "preselection_locator_ref": preselect_ref,
            "protocol_version": PROTOCOL_VERSION,
            "source_manifest_ref": _ref(
                "analyze-manifest-1",
                "myquant.v17.v3.source-manifest.v1",
                "data/private/v17_v3_sources/manifests/analyze-1.json",
                semantic=str(manifest["semantic_sha256"]),
            ),
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v3.source-locator.v1",
        }
    )
    assert (
        validate_staged_analysis_lineage(
            analyze_locator=locator,
            derived_manifest=manifest,
            initial_pool=pool,
            quant_branch=quant,
            fundamental_branch=fundamental,
        )[0].payload["locator_id"]
        == "analyze-locator-1"
    )
    pool_policy_drift = dict(pool)
    pool_policy_drift["policy_sha256"] = _sha("unpackaged-policy")
    pool_policy_drift.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="packaged preselector policy"):
        validate_staged_analysis_lineage(
            analyze_locator=locator,
            derived_manifest=manifest,
            initial_pool=seal_semantic(pool_policy_drift),
            quant_branch=quant,
            fundamental_branch=fundamental,
        )
    quant["source_locator_ref"] = _ref(
        "wrong",
        "myquant.v17.v3.source-locator.v1",
        "data/private/v17_v3_sources/locators/quant-first/wrong.json",
    )
    quant.pop("semantic_sha256")
    with pytest.raises(ArtifactContractError, match="PRESELECT locator drift"):
        validate_staged_analysis_lineage(
            analyze_locator=locator,
            derived_manifest=manifest,
            initial_pool=pool,
            quant_branch=seal_semantic(quant),
            fundamental_branch=fundamental,
        )
