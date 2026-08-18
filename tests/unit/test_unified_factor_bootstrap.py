from __future__ import annotations

import copy
import hashlib

import numpy as np
import pandas as pd
import pytest

from quant_investor.contracts import (
    ContractError,
    artifact_byte_sha256,
    canonical_json_bytes,
    seal_artifact,
)
from quant_investor.factors.governance import (
    BLEND_W75_CONTROL,
    BLEND_W80,
    CANONICAL_PARQUET,
    LOW_DOLLAR_VOLUME,
    FactorGovernanceError,
    bootstrap_factor_definitions,
    build_bootstrap_exception_evidence,
    build_bootstrap_factor_set,
    compute_bootstrap_signals,
    validate_bootstrap_exception_evidence,
    validate_bootstrap_factor_set,
)
from quant_investor.factors.governance.bootstrap import (
    BOOTSTRAP_REQUIRED_SOURCE_ROLES,
    FactorSourceRole,
    derive_active_required_source_roles,
    required_source_roles_for_factor,
)
from quant_investor.factors.governance.implementations import installed_semantic_row

STAMP = "2026-01-02T00:00:00Z"


def _ref(artifact: dict) -> dict[str, str]:
    return {
        "kind": artifact["kind"],
        "contract_sha256": artifact["contract_sha256"],
        "artifact_id": artifact["artifact_id"],
        "semantic_sha256": artifact["semantic_sha256"],
        "byte_sha256": artifact_byte_sha256(artifact),
    }


def _source_object(source_id: str, raw_sha256: str) -> dict:
    return seal_artifact(
        "system.source_object",
        {
            "source_object_id": source_id,
            "source_root_id": "test-root",
            "relative_path": f"tests/fixtures/{source_id}.json",
            "media_type": "application/json",
            "source_format": "JSON",
            "byte_sha256": raw_sha256,
        },
        created_at=STAMP,
    )


def _bundle(bundle_id: str, sources: list[dict] | None = None) -> dict:
    return seal_artifact(
        "system.source_bundle",
        {
            "source_bundle_id": bundle_id,
            "state": "SYNTHETIC_TEST_ONLY",
            "sources": list(sources or []),
        },
        created_at=STAMP,
    )


def _decision_bytes() -> bytes:
    return canonical_json_bytes(
        {
            "kind": "factor.bootstrap_decision",
            "decision_source_id": "user-approved-unified-runtime-cutover",
            "admission_route": "BOOTSTRAP_EXCEPTION",
            "producer_identity": "NOT_CLAIMED",
            "factor_weights": [
                {
                    "factor_id": BLEND_W80,
                    "weight": "0.500000000000",
                },
                {
                    "factor_id": LOW_DOLLAR_VOLUME,
                    "weight": "0.500000000000",
                },
            ],
            "control_factor_ids": [BLEND_W75_CONTROL],
            "prospective_evidence_claimed": False,
            "activation_authorized": False,
        }
    )


def _evidence_inputs() -> tuple[bytes, dict[str, dict], str, str]:
    decision_bytes = _decision_bytes()
    decision_sha = hashlib.sha256(decision_bytes).hexdigest()
    implementation_sha = "d" * 64
    decision_object = _source_object("bootstrap-decision", decision_sha)
    implementation_object = _source_object("implementation-tree-manifest", implementation_sha)
    release = seal_artifact(
        "system.release",
        {
            "release_id": "release-test",
            "state": "READY",
            "code_sha256": "c" * 64,
            "wheel_sha256": "e" * 64,
            "code_manifest_sha256": "f" * 64,
        },
        created_at=STAMP,
    )
    artifacts = {
        "code": release,
        "decision_source": _bundle(
            "decision-bundle",
            [{"role": "bootstrap_decision", "source_ref": _ref(decision_object)}],
        ),
        "exchange_calendar": _bundle("calendar-bundle"),
        "implementation": _bundle(
            "implementation-bundle",
            [
                {
                    "role": "implementation_tree_manifest",
                    "source_ref": _ref(implementation_object),
                }
            ],
        ),
        "market": _bundle("market-bundle"),
        "pit_universe": _bundle("universe-bundle"),
        "recomputation": _bundle("recomputation-bundle"),
        "source_generation": _bundle("generation-bundle"),
    }
    return decision_bytes, artifacts, implementation_sha, decision_sha


def test_bootstrap_definitions_are_exact_two_plus_w75_control() -> None:
    rows = bootstrap_factor_definitions()
    assert [row["factor_id"] for row in rows] == sorted(
        [BLEND_W75_CONTROL, BLEND_W80, LOW_DOLLAR_VOLUME]
    )
    by_id = {row["factor_id"]: row for row in rows}
    assert by_id[LOW_DOLLAR_VOLUME]["formula"] == "-log(mean(amount[t-4:t]))"
    assert by_id[LOW_DOLLAR_VOLUME]["input_fields"] == ["amount"]
    assert by_id[LOW_DOLLAR_VOLUME]["bootstrap_weight"] == "0.500000000000"
    assert by_id[BLEND_W80]["parameters"]["outer_volume_stability_weight"] == ("0.800000000000")
    assert by_id[BLEND_W80]["bootstrap_weight"] == "0.500000000000"
    assert by_id[BLEND_W75_CONTROL]["role"] == "CONTROL_ONLY"
    assert by_id[BLEND_W75_CONTROL]["selectable"] is False
    assert by_id[BLEND_W75_CONTROL]["bootstrap_weight"] == "0.000000000000"
    assert all(row["direction"] == "HIGHER_IS_BETTER" for row in rows)
    assert all(row["producer_identity"] == "NOT_CLAIMED" for row in rows)
    assert [role.value for role in FactorSourceRole] == [
        "EXCHANGE_CALENDAR",
        "MARKET",
        "PIT_MEMBERSHIP",
        "FUNDAMENTAL",
    ]
    assert list(BOOTSTRAP_REQUIRED_SOURCE_ROLES) == [
        "EXCHANGE_CALENDAR",
        "MARKET",
        "PIT_MEMBERSHIP",
    ]
    assert all(
        row["required_source_roles"] == list(BOOTSTRAP_REQUIRED_SOURCE_ROLES) for row in rows
    )
    assert all(
        forbidden not in row
        for row in rows
        for forbidden in ("version", "schema_version", "protocol_version")
    )


def test_bootstrap_exception_evidence_binds_raw_shas_but_authorizes_nothing() -> None:
    decision, sources, implementation_sha, decision_sha = _evidence_inputs()
    assert decision_sha == "f4add792c25eafa61730dfc839e1e5d6cd9c81de25b3c47b455359d26fb2ce95"
    evidence = build_bootstrap_exception_evidence(
        decision_source_bytes=decision,
        source_artifacts=sources,
        implementation_source_sha256=implementation_sha,
        created_at=STAMP,
    )
    assert validate_bootstrap_exception_evidence(evidence) == evidence
    payload = evidence["payload"]
    assert payload["decision_source_sha256"] == decision_sha
    assert payload["authorizes_readiness"] is False
    assert payload["authorizes_selectability"] is False
    assert payload["reader_contract"] == {
        "reader": "MarketDataReader",
        "market": "CN",
        "mode_policy": "strict",
        "source_format": "PARQUET",
        "fallback_allowed": False,
    }
    assert [row["role"] for row in payload["source_refs"]] == [
        "code",
        "decision_source",
        "exchange_calendar",
        "implementation",
        "market",
        "pit_universe",
        "recomputation",
        "source_generation",
    ]
    for row in payload["factor_rows"]:
        assert row["code_sha256"] == "c" * 64
        assert row["code_sha256"] != artifact_byte_sha256(sources["code"])
        assert row["implementation_sha256"] == implementation_sha
        assert row["implementation_sha256"] != artifact_byte_sha256(sources["implementation"])
        assert row["required_source_roles"] == list(BOOTSTRAP_REQUIRED_SOURCE_ROLES)


def test_bootstrap_evidence_requires_locked_inner_source_roles() -> None:
    decision, sources, implementation_sha, _ = _evidence_inputs()
    wrong = dict(sources)
    wrong["implementation"] = _bundle("wrong-implementation-bundle")
    with pytest.raises(FactorGovernanceError, match="implementation_tree_manifest"):
        build_bootstrap_exception_evidence(
            decision_source_bytes=decision,
            source_artifacts=wrong,
            implementation_source_sha256=implementation_sha,
            created_at=STAMP,
        )


def test_bootstrap_set_binds_evidence_but_never_activates() -> None:
    decision, sources, implementation_sha, _ = _evidence_inputs()
    evidence = build_bootstrap_exception_evidence(
        decision_source_bytes=decision,
        source_artifacts=sources,
        implementation_source_sha256=implementation_sha,
        created_at=STAMP,
    )
    factor_set = build_bootstrap_factor_set(
        bootstrap_exception_evidence=evidence,
        created_at="2026-01-03T00:00:00Z",
    )
    assert validate_bootstrap_factor_set(factor_set) == factor_set
    payload = factor_set["payload"]
    assert payload["admission_route"] == "BOOTSTRAP_EXCEPTION"
    assert payload["producer_identity"] == "NOT_CLAIMED"
    assert payload["weighting_method"] == "EQUAL_WEIGHT"
    assert payload["weight_total"] == "1.000000000000"
    assert payload["activation_authorized"] is False
    assert [row["factor_id"] for row in payload["factor_rows"]] == sorted(
        [BLEND_W80, LOW_DOLLAR_VOLUME]
    )
    assert (
        derive_active_required_source_roles(
            factor_set,
            implementation_rows=[
                installed_semantic_row(BLEND_W80),
                installed_semantic_row(LOW_DOLLAR_VOLUME),
            ],
        )
        == BOOTSTRAP_REQUIRED_SOURCE_ROLES
    )


@pytest.mark.parametrize(
    ("factor_id", "mutated_roles"),
    [
        (BLEND_W75_CONTROL, ["EXCHANGE_CALENDAR", "MARKET"]),
        (
            BLEND_W80,
            ["EXCHANGE_CALENDAR", "MARKET", "PIT_MEMBERSHIP", "FUNDAMENTAL"],
        ),
        (LOW_DOLLAR_VOLUME, ["MARKET", "EXCHANGE_CALENDAR", "PIT_MEMBERSHIP"]),
        (LOW_DOLLAR_VOLUME, ["EXCHANGE_CALENDAR", "MARKET", "UNKNOWN"]),
        (
            BLEND_W75_CONTROL,
            ["EXCHANGE_CALENDAR", "FUNDAMENTAL", "MARKET", "PIT_MEMBERSHIP"],
        ),
    ],
)
def test_bootstrap_rejects_source_role_mutation_with_reused_identities(
    factor_id: str,
    mutated_roles: list[str],
) -> None:
    decision, sources, implementation_sha, _ = _evidence_inputs()
    evidence = build_bootstrap_exception_evidence(
        decision_source_bytes=decision,
        source_artifacts=sources,
        implementation_source_sha256=implementation_sha,
        created_at=STAMP,
    )
    factor_set = build_bootstrap_factor_set(
        bootstrap_exception_evidence=evidence,
        created_at="2026-01-03T00:00:00Z",
    )
    payload = copy.deepcopy(factor_set["payload"])
    target = next(row for row in payload["factor_definitions"] if row["factor_id"] == factor_id)
    original_spec_id = target["spec_id"]
    original_factor_set_sha = payload["factor_set_sha256"]
    target["required_source_roles"] = mutated_roles
    matching = next(
        row
        for row in [*payload["factor_rows"], *payload["control_rows"]]
        if row["factor_id"] == target["factor_id"]
    )
    matching["required_source_roles"] = mutated_roles
    assert target["spec_id"] == original_spec_id
    assert payload["factor_set_sha256"] == original_factor_set_sha
    with pytest.raises((ContractError, FactorGovernanceError)):
        validate_bootstrap_factor_set(
            seal_artifact(
                "factor.bootstrap_set",
                payload,
                created_at=factor_set["created_at"],
            )
        )


def test_source_dependencies_are_code_owned_and_w75_cannot_mutate() -> None:
    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80, BLEND_W75_CONTROL):
        assert required_source_roles_for_factor(factor_id) == (
            "EXCHANGE_CALENDAR",
            "MARKET",
            "PIT_MEMBERSHIP",
        )
    with pytest.raises(FactorGovernanceError, match="not installed"):
        required_source_roles_for_factor("caller-supplied-factor")

    definitions = bootstrap_factor_definitions()
    w75 = next(row for row in definitions if row["factor_id"] == BLEND_W75_CONTROL)
    old_spec_id = w75["spec_id"]
    old_body = {
        key: value for key, value in w75.items() if key not in {"spec_id", "required_source_roles"}
    }
    assert hashlib.sha256(canonical_json_bytes(old_body)).hexdigest() != old_spec_id


@pytest.mark.parametrize(
    "mutated_roles",
    [
        ["EXCHANGE_CALENDAR", "MARKET"],
        ["EXCHANGE_CALENDAR", "FUNDAMENTAL", "MARKET", "PIT_MEMBERSHIP"],
        ["MARKET", "EXCHANGE_CALENDAR", "PIT_MEMBERSHIP"],
        ["EXCHANGE_CALENDAR", "MARKET", "PIT_MEMBERSHIP", "UNKNOWN"],
    ],
)
def test_active_dependency_union_rejects_installed_role_mutation(
    mutated_roles: list[str],
) -> None:
    decision, sources, implementation_sha, _ = _evidence_inputs()
    evidence = build_bootstrap_exception_evidence(
        decision_source_bytes=decision,
        source_artifacts=sources,
        implementation_source_sha256=implementation_sha,
        created_at=STAMP,
    )
    factor_set = build_bootstrap_factor_set(
        bootstrap_exception_evidence=evidence,
        created_at="2026-01-03T00:00:00Z",
    )
    installed_rows = [
        installed_semantic_row(BLEND_W80),
        installed_semantic_row(LOW_DOLLAR_VOLUME),
    ]
    installed_rows[0]["required_source_roles"] = mutated_roles
    with pytest.raises(FactorGovernanceError, match="semantic identity differs"):
        derive_active_required_source_roles(
            factor_set,
            implementation_rows=installed_rows,
        )


def test_bootstrap_signal_helper_rejects_csv_and_amount_fallbacks() -> None:
    dates = pd.bdate_range("2025-01-01", periods=100)
    frame = pd.DataFrame(
        {
            "trade_date": dates,
            "amount": np.linspace(100.0, 200.0, len(dates)),
            "adj_close": np.linspace(10.0, 15.0, len(dates)),
            "vol": np.linspace(1000.0, 1200.0, len(dates)),
        }
    )
    signals = compute_bootstrap_signals(
        {"000001.SZ": frame, "600000.SH": frame.assign(amount=lambda x: x.amount * 2)},
        source_format=CANONICAL_PARQUET,
    )
    assert set(signals) == {LOW_DOLLAR_VOLUME, BLEND_W80, BLEND_W75_CONTROL}
    with pytest.raises(FactorGovernanceError, match="canonical Parquet"):
        compute_bootstrap_signals({"000001.SZ": frame}, source_format="CSV")
    no_amount = frame.drop(columns="amount").assign(close_times_volume=1.0)
    with pytest.raises(FactorGovernanceError, match="missing amount"):
        compute_bootstrap_signals({"000001.SZ": no_amount}, source_format=CANONICAL_PARQUET)
