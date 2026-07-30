from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from quant_investor.v17_v4_contract import canonical_resource_bytes, seal_semantic
from quant_investor.v17_v5_contract import (
    artifact_identity_field,
    load_factor_regime_diagnostic_policy,
    validate_artifact,
)
from quant_investor.v17_v5_contract.resources import (
    FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH,
    read_packaged_asset,
)
from quant_investor.v17_v5_contract.validators import (
    FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH as POLICY_REFERENCE_PATH,
    NO_AUTHORITY,
)
from quant_investor.v17_v5_runtime import cli as cli_module
from quant_investor.v17_v5_runtime.cli import main
from quant_investor.v17_v5_runtime.factor_regime_diagnostics import (
    build_unavailable_regime_conditioned_factor_diagnostic,
)

CUTOFF = "2026-07-29T08:00:00Z"
CREATED_AT = "2026-07-29T08:00:01Z"
STRATEGY = "quant-first"
REGIME_PATH = "data/private/v17_v4_runs/run-1/regime.json"

_V3_TEST_PATH = Path(__file__).with_name("test_v17_v4_regime_evidence_v3.py")
_V3_SPEC = importlib.util.spec_from_file_location(
    "_v4_v3_regime_cli_test_support",
    _V3_TEST_PATH,
)
assert _V3_SPEC is not None and _V3_SPEC.loader is not None
_V3_SUPPORT = importlib.util.module_from_spec(_V3_SPEC)
sys.modules[_V3_SPEC.name] = _V3_SUPPORT
_V3_SPEC.loader.exec_module(_V3_SUPPORT)


def _policy_ref() -> dict[str, str]:
    policy = load_factor_regime_diagnostic_policy()
    raw = read_packaged_asset(FACTOR_REGIME_DIAGNOSTIC_POLICY_PATH)
    return {
        "artifact_id": policy["artifact_id"],
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "relative_path": POLICY_REFERENCE_PATH,
        "semantic_sha256": policy["semantic_sha256"],
        "version": policy["version"],
    }


def _regime_artifact() -> dict[str, Any]:
    return seal_semantic(
        {
            "authority": {
                "broker": False,
                "execution": False,
                "formal_research_publication": False,
                "order": False,
                "research_runtime_default": False,
                "trade": False,
            },
            "available_at": "2026-07-29T07:59:00Z",
            "created_at": CREATED_AT,
            "cutoff": CUTOFF,
            "evidence_id": "regime-evidence-1",
            "gross_multiplier": "0.8",
            "protocol_version": "myquant.v17.v4",
            "role": "markov_evidence",
            "run_id": "run-1",
            "status": "AVAILABLE",
            "strategy_id": STRATEGY,
            "version": "myquant.v17.v4.regime-evidence.v1",
        }
    )


def _tree(root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def _schema_property_names(value: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            names.update(properties)
        for child in value.values():
            names.update(_schema_property_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(_schema_property_names(child))
    return names


def test_policy_is_content_addressed_descriptive_only_and_source_ineligible() -> None:
    policy = load_factor_regime_diagnostic_policy()

    assert policy["conditioning_dimension"] == "ORIGIN_REGIME"
    assert policy["horizon_sessions"] == 20
    assert policy["minimum_descriptive_origins"] == 20
    assert policy["minimum_stability_origins"] == 60
    assert policy["newey_west_lag"] == 19
    assert policy["accepted_regime_source_versions"] == ["myquant.v17.v4.regime-evidence.v3"]
    assert policy["required_inference_kind"] == "FILTERED_CAUSAL"
    assert policy["required_smoothing_used"] is False
    assert policy["conditioning_ineligible_states"] == ["未知"]
    assert policy["conditioning_eligible_continuity"] == ["CONTIGUOUS", "ROLLOVER"]
    assert policy["conditioning_ineligible_continuity"] == ["GENESIS", "RECOVERY"]
    assert policy["authority"] == NO_AUTHORITY


def test_new_schemas_register_identity_and_exclude_governance_fields() -> None:
    root = Path(__file__).resolve().parents[2] / "quant_investor/v17_v5_contract/schemas"
    versions = {
        "myquant.v17.v5.factor-regime-origin-inventory.v1": (
            "factor_regime_origin_inventory.v1.schema.json",
            "inventory_id",
        ),
        "myquant.v17.v5.factor-regime-origin-inventory.v2": (
            "factor_regime_origin_inventory.v2.schema.json",
            "inventory_id",
        ),
        "myquant.v17.v5.factor-regime-origin-inventory.v3": (
            "factor_regime_origin_inventory.v3.schema.json",
            "inventory_id",
        ),
        "myquant.v17.v5.regime-conditioned-factor-diagnostic.v1": (
            "regime_conditioned_factor_diagnostic.v1.schema.json",
            "diagnostic_id",
        ),
        "myquant.v17.v5.regime-conditioned-factor-diagnostic.v2": (
            "regime_conditioned_factor_diagnostic.v2.schema.json",
            "diagnostic_id",
        ),
        "myquant.v17.v5.regime-conditioned-factor-diagnostic.v3": (
            "regime_conditioned_factor_diagnostic.v3.schema.json",
            "diagnostic_id",
        ),
    }
    forbidden = {
        "factor_weight",
        "recommended_weight",
        "target_weight",
        "portfolio_weight",
        "production_weight",
        "tier",
        "lifecycle_action",
        "promotion_eligible",
        "validity_verdict",
    }
    for version, (filename, identity) in versions.items():
        schema = json.loads((root / filename).read_bytes())
        assert artifact_identity_field(version) == identity
        assert _schema_property_names(schema).isdisjoint(forbidden)


def test_unavailable_diagnostic_validates_without_fake_factor_sha() -> None:
    diagnostic = build_unavailable_regime_conditioned_factor_diagnostic(
        strategy_id=STRATEGY,
        factor_name="cn-factor",
        factor_implementation_sha256=None,
        policy_ref=_policy_ref(),
        cutoff=CUTOFF,
        created_at=CREATED_AT,
        unavailable_prerequisites=(
            "V4_FACTOR_EVIDENCE_UNAVAILABLE",
            "V4_REGIME_EVIDENCE_V3_UNAVAILABLE",
        ),
    )

    assert validate_artifact(diagnostic) == diagnostic
    assert diagnostic["status"] == "UNAVAILABLE"
    assert diagnostic["factor_implementation_sha256"] is None
    assert diagnostic["authority"] == NO_AUTHORITY


def test_cli_unavailable_is_stdout_only_and_attests_no_authority(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = _tree(tmp_path)
    assert (
        main(
            [
                "factor-regime-diagnostics",
                "--workspace-root",
                str(tmp_path),
                "--strategy-id",
                STRATEGY,
                "--factor-name",
                "cn-factor",
                "--evaluation-cutoff",
                CUTOFF,
                "--created-at",
                CREATED_AT,
                "--output-id",
                "diagnostic-request-1",
                "--factor-evidence-unavailable",
                "--regime-evidence-unavailable",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "UNAVAILABLE"
    assert payload["global_activation_state"] == "INACTIVE"
    assert payload["run_state"] == "INACTIVE"
    assert payload["default_protocol_state"] == "V15_DEFAULT"
    assert payload["provider_calls"] is False
    assert all(payload[field] is False for field in NO_AUTHORITY)
    assert _tree(tmp_path) == before


def test_cli_exact_v4_regime_v1_stays_unavailable_without_inference(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / REGIME_PATH
    target.parent.mkdir(parents=True)
    raw = canonical_resource_bytes(_regime_artifact())
    target.write_bytes(raw)
    before = _tree(tmp_path)

    assert (
        main(
            [
                "factor-regime-diagnostics",
                "--workspace-root",
                str(tmp_path),
                "--strategy-id",
                STRATEGY,
                "--factor-name",
                "cn-factor",
                "--evaluation-cutoff",
                CUTOFF,
                "--created-at",
                CREATED_AT,
                "--output-id",
                "diagnostic-request-2",
                "--factor-evidence-unavailable",
                "--regime-evidence-path",
                REGIME_PATH,
                "--regime-evidence-sha256",
                hashlib.sha256(raw).hexdigest(),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "UNAVAILABLE"
    assert (
        "REGIME_EVIDENCE_V1_NOT_CONDITIONING_ELIGIBLE" in payload["diagnostic"]["limitation_codes"]
    )
    assert _tree(tmp_path) == before


def test_cli_exact_finalized_v3_reports_metadata_without_persisting(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sessions = _V3_SUPPORT._business_sessions("2026-07-29", 8)
    factory = _V3_SUPPORT.V3Factory(tmp_path, sessions)
    genesis, _ = factory.build(
        observed=sessions[0],
        effective=sessions[1],
        created_at="2026-07-30T00:01:00Z",
    )
    contiguous, _ = factory.build(
        observed=sessions[1],
        effective=sessions[2],
        created_at="2026-07-30T15:21:00Z",
        prior=genesis,
    )
    before = _tree(tmp_path)

    assert (
        main(
            [
                "factor-regime-diagnostics",
                "--workspace-root",
                str(tmp_path),
                "--strategy-id",
                str(contiguous.document["strategy_id"]),
                "--factor-name",
                "cn-factor",
                "--evaluation-cutoff",
                str(contiguous.document["cutoff"]),
                "--created-at",
                "2026-07-31T08:00:00Z",
                "--output-id",
                "diagnostic-request-v3",
                "--factor-evidence-unavailable",
                "--regime-evidence-path",
                contiguous.evidence_path,
                "--regime-evidence-sha256",
                contiguous.evidence_sha256,
                "--regime-checkpoint-path",
                contiguous.chain_checkpoint_path,
                "--regime-checkpoint-sha256",
                contiguous.chain_checkpoint_sha256,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "UNAVAILABLE"
    assert payload["regime_source_version"] == "myquant.v17.v4.regime-evidence.v3"
    assert payload["regime_finalized"] is True
    assert payload["continuity_kind"] == "CONTIGUOUS"
    assert payload["regime_conditioning_eligibility"] is True
    assert payload["predecessor_source_commit"] == ("73c5b6eea6c60d9a31865e176646687ffeee9d6a")
    assert payload["diagnostic"]["limitation_codes"] == ["V4_FACTOR_EVIDENCE_UNAVAILABLE"]
    assert _tree(tmp_path) == before


def test_cli_explicit_sha_mismatch_and_mode_conflict_exit_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = tmp_path / REGIME_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes(canonical_resource_bytes(_regime_artifact()))

    base = [
        "factor-regime-diagnostics",
        "--workspace-root",
        str(tmp_path),
        "--strategy-id",
        STRATEGY,
        "--factor-name",
        "cn-factor",
        "--evaluation-cutoff",
        CUTOFF,
        "--created-at",
        CREATED_AT,
        "--output-id",
        "diagnostic-request-3",
        "--factor-evidence-unavailable",
    ]
    assert (
        main(
            [
                *base,
                "--regime-evidence-path",
                REGIME_PATH,
                "--regime-evidence-sha256",
                "0" * 64,
            ]
        )
        == 2
    )
    assert '"verified":false' in capsys.readouterr().out

    assert (
        main(
            [
                *base,
                "--factor-evidence-sha256",
                "0" * 64,
                "--regime-evidence-unavailable",
            ]
        )
        == 2
    )
    assert "cannot accompany an unavailable declaration" in capsys.readouterr().out


def test_cli_rejects_exact_factor_receipt_for_another_factor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli_module,
        "read_v4_artifact",
        lambda *args, **kwargs: SimpleNamespace(
            document={
                "lineage_key": {"factor_name": "other-factor"},
                "subject_id": "other-factor",
                "version": "myquant.v17.v4.forward-evaluation-receipt.v1",
            }
        ),
    )

    assert (
        main(
            [
                "factor-regime-diagnostics",
                "--workspace-root",
                str(tmp_path),
                "--strategy-id",
                STRATEGY,
                "--factor-name",
                "cn-factor",
                "--evaluation-cutoff",
                CUTOFF,
                "--created-at",
                CREATED_AT,
                "--output-id",
                "diagnostic-request-factor-mismatch",
                "--factor-evidence-path",
                "data/private/v17_v4_runs/factor-receipt.json",
                "--factor-evidence-sha256",
                "1" * 64,
                "--regime-evidence-unavailable",
            ]
        )
        == 2
    )
    assert "does not match --factor-name" in capsys.readouterr().out


def test_cli_exact_eligible_inputs_fail_closed_until_origin_assembly_is_enabled(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reads = iter(
        (
            SimpleNamespace(
                document={
                    "lineage_key": {"factor_name": "cn-factor"},
                    "subject_id": "cn-factor",
                    "version": "myquant.v17.v4.forward-evaluation-receipt.v1",
                }
            ),
            SimpleNamespace(document={"version": "myquant.v17.v4.regime-evidence.v3"}),
        )
    )
    monkeypatch.setattr(cli_module, "read_v4_artifact", lambda *args, **kwargs: next(reads))
    monkeypatch.setattr(
        cli_module,
        "adapt_v4_regime_evidence",
        lambda read, **kwargs: SimpleNamespace(
            conditioning_eligible=True,
            conditioning_ineligibility_reason=None,
            continuity_kind="CONTIGUOUS",
            finalized=True,
            hard_state="趋势上涨",
            inference_kind="FILTERED_CAUSAL",
            publication_phase="PRIOR_SESSION_EFFECTIVE_NEXT_SESSION",
            scope_kind="FULL_MARKET",
            smoothing_used=False,
            source_version="myquant.v17.v4.regime-evidence.v3",
        ),
    )

    assert (
        main(
            [
                "factor-regime-diagnostics",
                "--workspace-root",
                str(tmp_path),
                "--strategy-id",
                STRATEGY,
                "--factor-name",
                "cn-factor",
                "--evaluation-cutoff",
                CUTOFF,
                "--created-at",
                CREATED_AT,
                "--output-id",
                "diagnostic-request-observed-not-enabled",
                "--factor-evidence-path",
                "data/private/v17_v4_runs/factor-receipt.json",
                "--factor-evidence-sha256",
                "1" * 64,
                "--regime-evidence-path",
                "data/private/v17_v4_runs/regime-evidence-v3.json",
                "--regime-evidence-sha256",
                "2" * 64,
                "--regime-checkpoint-path",
                "data/private/v17_v4_runs/regime-state-checkpoint.json",
                "--regime-checkpoint-sha256",
                "3" * 64,
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "UNAVAILABLE"
    assert payload["origin_binding_result"] == "NOT_ENABLED"
    assert (
        "OBSERVED_FACTOR_REGIME_CLI_PATH_NOT_ENABLED" in payload["diagnostic"]["limitation_codes"]
    )
