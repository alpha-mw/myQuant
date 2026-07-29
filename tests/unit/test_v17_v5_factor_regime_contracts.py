from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
    assert policy["regime_source_versions"]["registered"] == ["myquant.v17.v4.regime-evidence.v1"]
    assert policy["regime_source_versions"]["conditioning_eligible"] == []
    assert policy["authority"] == NO_AUTHORITY


def test_new_schemas_register_identity_and_exclude_governance_fields() -> None:
    root = Path(__file__).resolve().parents[2] / "quant_investor/v17_v5_contract/schemas"
    versions = {
        "myquant.v17.v5.factor-regime-origin-inventory.v1": (
            "factor_regime_origin_inventory.v1.schema.json",
            "inventory_id",
        ),
        "myquant.v17.v5.regime-conditioned-factor-diagnostic.v1": (
            "regime_conditioned_factor_diagnostic.v1.schema.json",
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
            "V4_REGIME_EVIDENCE_UNAVAILABLE",
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
    assert "REGIME_HARD_STATE_UNAVAILABLE" in payload["diagnostic"]["limitation_codes"]
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
