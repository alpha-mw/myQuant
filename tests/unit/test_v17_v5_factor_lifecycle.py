from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.v17_v4_contract.schema_validation import validate_instance_against_schema
from quant_investor.v17_v5_contract import canonical_bytes, seal_semantic
from quant_investor.v17_v5_runtime.factor_diagnostics import (
    FactorOriginSample,
    FactorSampleStratum,
    build_factor_diagnostic,
    build_unavailable_factor_diagnostic,
)
from quant_investor.v17_v5_runtime.factor_lifecycle import (
    FactorLifecycleDiagnosticError,
    build_factor_lifecycle_diagnostic,
    build_unavailable_factor_lifecycle_diagnostic,
    validate_factor_lifecycle_diagnostic_replay,
)
from quant_investor.v17_v5_contract.validators import (
    V4_FACTOR_EVIDENCE_ADAPTER_POLICY_BYTE_SHA256,
)


def _sessions(count: int = 100) -> tuple[str, ...]:
    start = date(2026, 1, 1)
    return tuple((start + timedelta(days=index)).isoformat() for index in range(count))


def _calendar_sha(sessions: tuple[str, ...]) -> str:
    return hashlib.sha256(canonical_bytes(list(sessions))).hexdigest()


def _stratum(
    sessions: tuple[str, ...],
    *,
    factor_name: str = "cn_factor_test",
    factor_set_sha256: str = "3" * 64,
) -> FactorSampleStratum:
    return FactorSampleStratum(
        strategy_id="cn-factor-lifecycle",
        factor_name=factor_name,
        factor_definition_sha256="1" * 64,
        factor_implementation_sha256="2" * 64,
        factor_set_sha256=factor_set_sha256,
        quant_policy_sha256="4" * 64,
        adapter_policy_byte_sha256=V4_FACTOR_EVIDENCE_ADAPTER_POLICY_BYTE_SHA256,
        source_lineage_series_sha256="6" * 64,
        market_calendar_sha256=_calendar_sha(sessions),
    )


def _origin(sessions: tuple[str, ...], index: int) -> FactorOriginSample:
    symbols = tuple(f"{symbol_index:06d}.SZ" for symbol_index in range(1, 101))
    end = sessions[index + 20]
    return FactorOriginSample(
        origin_id=f"origin.{index:03d}",
        decision_session=sessions[index],
        horizon_end_session=end,
        label_available_at=f"{end}T08:00:00Z",
        evidence_lineage_sha256=hashlib.sha256(f"lineage-{index}".encode()).hexdigest(),
        factor_values={symbol: str(rank) for rank, symbol in enumerate(symbols, start=1)},
        forward_returns={symbol: str(rank) for rank, symbol in enumerate(symbols, start=1)},
    )


def _diagnostic(
    sessions: tuple[str, ...],
    origin_indexes: list[int],
    *,
    factor_name: str = "cn_factor_test",
    cutoff: str = "2026-12-31T23:59:59Z",
) -> dict[str, object]:
    return build_factor_diagnostic(
        stratum=_stratum(sessions, factor_name=factor_name),
        evaluation_cutoff=cutoff,
        open_sessions=sessions,
        origins=[_origin(sessions, index) for index in origin_indexes],
    )


def _schema() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    return json.loads(
        (
            root / "quant_investor/v17_v5_contract/schemas/"
            "factor_lifecycle_diagnostic.v1.schema.json"
        ).read_text()
    )


def test_lifecycle_is_deterministic_dedups_inputs_and_origin_rows() -> None:
    sessions = _sessions()
    first = _diagnostic(sessions, [0, 1])
    duplicate_origins = _diagnostic(sessions, [1, 2])
    expected = build_factor_lifecycle_diagnostic(
        factor_diagnostics=[first, duplicate_origins],
        evaluation_cutoff="2027-01-01T00:00:00Z",
        expected_factor_name="cn_factor_test",
    )
    observed = build_factor_lifecycle_diagnostic(
        factor_diagnostics=[duplicate_origins, first, first],
        evaluation_cutoff="2027-01-01T00:00:00Z",
        expected_factor_name="cn_factor_test",
    )

    assert observed == expected
    assert observed["status"] == "ACCUMULATING"
    assert observed["factor_name"] == "cn_factor_test"
    assert observed["unique_origin_count"] == 3
    assert observed["first_decision_session"] == sessions[0]
    assert observed["last_decision_session"] == sessions[2]
    assert observed["input_factor_diagnostic_semantic_sha256s"] == sorted(
        {
            str(first["semantic_sha256"]),
            str(duplicate_origins["semantic_sha256"]),
        }
    )
    validate_instance_against_schema(observed, _schema())
    assert (
        validate_factor_lifecycle_diagnostic_replay(
            observed,
            factor_diagnostics=[first, duplicate_origins],
            evaluation_cutoff="2027-01-01T00:00:00Z",
            expected_factor_name="cn_factor_test",
        )
        == observed
    )


def test_all_unobserved_remains_unobserved() -> None:
    sessions = _sessions()
    artifact = build_factor_lifecycle_diagnostic(
        factor_diagnostics=[_diagnostic(sessions, [])],
        evaluation_cutoff="2026-12-31T23:59:59Z",
    )

    assert artifact["status"] == "UNOBSERVED"
    assert artifact["unique_origin_count"] == 0
    assert artifact["first_decision_session"] is None
    assert artifact["last_decision_session"] is None
    assert artifact["stratum"] is not None
    assert "lifecycle_no_observed_origins" in artifact["blockers"]


def test_all_unavailable_factor_diagnostics_remain_unavailable() -> None:
    first = build_unavailable_factor_diagnostic(
        factor_name="cn_factor_test",
        evaluation_cutoff="2026-12-31T23:59:59Z",
        unavailable_prerequisites=["calendar_missing"],
    )
    second = build_unavailable_factor_diagnostic(
        factor_name="cn_factor_test",
        evaluation_cutoff="2026-12-30T23:59:59Z",
        unavailable_prerequisites=["label_contract_missing"],
    )

    artifact = build_factor_lifecycle_diagnostic(
        factor_diagnostics=[first, second],
        evaluation_cutoff="2027-01-01T00:00:00Z",
    )

    assert artifact["status"] == "UNAVAILABLE"
    assert artifact["stratum"] is None
    assert artifact["stratum_sha256"] is None
    assert artifact["input_factor_diagnostic_semantic_sha256s"] == sorted(
        {first["semantic_sha256"], second["semantic_sha256"]}
    )
    assert "lifecycle_inputs_unavailable" in artifact["blockers"]


def test_mixed_unavailable_and_observed_fails_closed() -> None:
    sessions = _sessions()
    observed = _diagnostic(sessions, [0])
    unavailable = build_unavailable_factor_diagnostic(
        factor_name="cn_factor_test",
        evaluation_cutoff="2026-12-31T23:59:59Z",
        unavailable_prerequisites=["calendar_missing"],
    )

    with pytest.raises(FactorLifecycleDiagnosticError, match="mix unavailable"):
        build_factor_lifecycle_diagnostic(
            factor_diagnostics=[observed, unavailable],
            evaluation_cutoff="2027-01-01T00:00:00Z",
        )


def test_mixed_exact_stratum_fails_closed() -> None:
    sessions = _sessions()
    other_stratum = build_factor_diagnostic(
        stratum=_stratum(sessions, factor_set_sha256="9" * 64),
        evaluation_cutoff="2026-12-31T23:59:59Z",
        open_sessions=sessions,
        origins=[_origin(sessions, 1)],
    )

    with pytest.raises(FactorLifecycleDiagnosticError, match="exact stratum"):
        build_factor_lifecycle_diagnostic(
            factor_diagnostics=[
                _diagnostic(sessions, [0]),
                other_stratum,
            ],
            evaluation_cutoff="2027-01-01T00:00:00Z",
        )


def test_conflicting_origin_or_same_session_alias_fails_closed() -> None:
    sessions = _sessions()
    first = _diagnostic(sessions, [0])
    origin = _origin(sessions, 0)
    conflicting = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff="2026-12-31T23:59:59Z",
        open_sessions=sessions,
        origins=[replace(origin, evidence_lineage_sha256="9" * 64)],
    )
    alias = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff="2026-12-31T23:59:59Z",
        open_sessions=sessions,
        origins=[replace(origin, origin_id="origin.alias")],
    )

    with pytest.raises(FactorLifecycleDiagnosticError, match="conflicting duplicate"):
        build_factor_lifecycle_diagnostic(
            factor_diagnostics=[first, conflicting],
            evaluation_cutoff="2027-01-01T00:00:00Z",
        )
    with pytest.raises(FactorLifecycleDiagnosticError, match="decision session"):
        build_factor_lifecycle_diagnostic(
            factor_diagnostics=[first, alias],
            evaluation_cutoff="2027-01-01T00:00:00Z",
        )


def test_empty_factor_diagnostics_expected_name_and_cutoff_fail_closed() -> None:
    sessions = _sessions()
    with pytest.raises(FactorLifecycleDiagnosticError, match="nonempty"):
        build_factor_lifecycle_diagnostic(
            factor_diagnostics=[],
            evaluation_cutoff="2027-01-01T00:00:00Z",
        )
    with pytest.raises(FactorLifecycleDiagnosticError, match="expected_factor_name"):
        build_factor_lifecycle_diagnostic(
            factor_diagnostics=[_diagnostic(sessions, [0])],
            evaluation_cutoff="2027-01-01T00:00:00Z",
            expected_factor_name="cn_factor_other",
        )
    with pytest.raises(FactorLifecycleDiagnosticError, match="precedes"):
        build_factor_lifecycle_diagnostic(
            factor_diagnostics=[_diagnostic(sessions, [0], cutoff="2027-01-01T00:00:00Z")],
            evaluation_cutoff="2026-12-31T23:59:59Z",
        )


def test_explicit_unavailable_lifecycle_has_no_authority_and_replays() -> None:
    artifact = build_unavailable_factor_lifecycle_diagnostic(
        factor_name="cn_factor_test",
        evaluation_cutoff="2026-12-31T23:59:59Z",
        prerequisites=["calendar_missing", "label_contract_missing"],
    )

    assert artifact["status"] == "UNAVAILABLE"
    assert all(value is False for value in artifact["authority"].values())
    assert artifact["input_factor_diagnostic_semantic_sha256s"] == []
    assert artifact["unique_origin_count"] == 0
    assert artifact["lifecycle_conclusion"] is None
    assert artifact["lifecycle_action"] is None
    assert artifact["effectiveness_claimed"] is False
    assert artifact["factor_tier_change_eligible"] is False
    assert artifact["factor_weight_change_eligible"] is False
    assert artifact["promotion_eligible"] is False
    validate_instance_against_schema(artifact, _schema())
    assert (
        validate_factor_lifecycle_diagnostic_replay(
            artifact,
            factor_name="cn_factor_test",
            evaluation_cutoff="2026-12-31T23:59:59Z",
            prerequisites=["label_contract_missing", "calendar_missing"],
        )
        == artifact
    )


def test_lifecycle_validator_rejects_authority_or_semantic_drift() -> None:
    artifact = build_unavailable_factor_lifecycle_diagnostic(
        factor_name="cn_factor_test",
        evaluation_cutoff="2026-12-31T23:59:59Z",
        prerequisites=["calendar_missing"],
    )
    authority_mutation = dict(artifact)
    authority_mutation.pop("semantic_sha256")
    authority_mutation["authority"] = {**artifact["authority"], "promotion": True}
    with pytest.raises(FactorLifecycleDiagnosticError, match="grants authority"):
        validate_factor_lifecycle_diagnostic_replay(
            seal_semantic(authority_mutation),
            factor_name="cn_factor_test",
            evaluation_cutoff="2026-12-31T23:59:59Z",
            prerequisites=["calendar_missing"],
        )
    semantic_mutation = dict(artifact)
    semantic_mutation["semantic_sha256"] = "0" * 64
    with pytest.raises(FactorLifecycleDiagnosticError):
        validate_factor_lifecycle_diagnostic_replay(
            semantic_mutation,
            factor_name="cn_factor_test",
            evaluation_cutoff="2026-12-31T23:59:59Z",
            prerequisites=["calendar_missing"],
        )
