from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import hashlib
from pathlib import Path

import pytest

from quant_investor.v17_v5_contract import (
    canonical_bytes,
    seal_semantic,
    validate_artifact,
)
from quant_investor.v17_v5_contract.validators import ArtifactContractError
from quant_investor.v17_v5_runtime.factor_diagnostics import (
    FactorDiagnosticError,
    FactorOriginSample,
    FactorSampleStratum,
    build_factor_diagnostic,
    build_unavailable_factor_diagnostic,
    validate_factor_diagnostic_replay,
)


def _sessions(count: int = 100) -> tuple[str, ...]:
    start = date(2026, 1, 1)
    return tuple((start + timedelta(days=index)).isoformat() for index in range(count))


def _calendar_sha(sessions: tuple[str, ...]) -> str:
    return hashlib.sha256(canonical_bytes(list(sessions))).hexdigest()


def _stratum(sessions: tuple[str, ...]) -> FactorSampleStratum:
    return FactorSampleStratum(
        strategy_id="cn-factor-diagnostic",
        factor_name="cn_factor_test",
        factor_definition_sha256="1" * 64,
        factor_implementation_sha256="2" * 64,
        factor_set_sha256="3" * 64,
        quant_policy_sha256="4" * 64,
        adapter_policy_byte_sha256="5" * 64,
        source_lineage_series_sha256="6" * 64,
        market_calendar_sha256=_calendar_sha(sessions),
    )


def _symbols(count: int) -> tuple[str, ...]:
    return tuple(f"{index:06d}.SZ" for index in range(1, count + 1))


def _origin(
    sessions: tuple[str, ...],
    index: int,
    *,
    symbol_count: int = 100,
    constant_factor: bool = False,
    constant_return: bool = False,
) -> FactorOriginSample:
    symbols = _symbols(symbol_count)
    factor_values = {
        symbol: ("1" if constant_factor else str(rank))
        for rank, symbol in enumerate(symbols, start=1)
    }
    returns = {
        symbol: ("1" if constant_return else str(rank))
        for rank, symbol in enumerate(symbols, start=1)
    }
    end = sessions[index + 20]
    return FactorOriginSample(
        origin_id=f"origin.{index:03d}",
        decision_session=sessions[index],
        horizon_end_session=end,
        label_available_at=f"{end}T08:00:00Z",
        evidence_lineage_sha256=hashlib.sha256(f"origin-lineage-{index}".encode()).hexdigest(),
        factor_values=factor_values,
        forward_returns=returns,
    )


def _cutoff() -> str:
    return "2026-12-31T23:59:59Z"


def _reseal(document: dict[str, object], **changes: object) -> dict[str, object]:
    value = dict(document)
    value.pop("semantic_sha256")
    value.update(changes)
    return seal_semantic(value)


def _tree(root: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (path.relative_to(root).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def test_zero_origins_is_unobserved_and_has_no_statistics() -> None:
    sessions = _sessions()
    artifact = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[],
    )

    assert artifact["status"] == "UNOBSERVED"
    assert artifact["matured_origin_count"] == 0
    assert artifact["statistics"] is None
    assert artifact["descriptive_coverage_minimum_met"] is False
    assert artifact["inference_gate_passed"] is False
    assert artifact["inference_eligible"] is False
    assert artifact["effectiveness_claimed"] is False


@pytest.mark.parametrize("origin_count", [1, 59])
def test_subminimum_origins_are_accumulating(origin_count: int) -> None:
    sessions = _sessions()
    artifact = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[_origin(sessions, index) for index in range(origin_count)],
    )

    assert artifact["status"] == "ACCUMULATING"
    assert artifact["matured_origin_count"] == origin_count
    assert artifact["rank_ic_available_origin_count"] == origin_count
    assert artifact["descriptive_coverage_minimum_met"] is False
    assert artifact["statistics"]["rank_ic_mean"] == "1.000000000000"


def test_exact_descriptive_coverage_minimum_is_not_an_inference_gate() -> None:
    sessions = _sessions()
    artifact = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[_origin(sessions, index) for index in range(60)],
    )

    assert artifact["status"] == "ACCUMULATING"
    assert artifact["descriptive_coverage_minimum_met"] is True
    assert artifact["gate_scope"] == "DESCRIPTIVE_ONLY"
    assert artifact["inference_gate_passed"] is False
    assert artifact["effectiveness_conclusion"] is None
    assert artifact["factor_tier_change_eligible"] is False
    assert artifact["factor_weight_change_eligible"] is False
    assert artifact["promotion_eligible"] is False
    assert artifact["blockers"] == ["inference_not_implemented"]


def test_one_undersized_origin_prevents_descriptive_coverage_minimum() -> None:
    sessions = _sessions()
    origins = [_origin(sessions, index) for index in range(59)]
    origins.append(_origin(sessions, 59, symbol_count=99))

    artifact = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=origins,
    )

    assert artifact["rank_ic_available_origin_count"] == 60
    assert artifact["minimum_comparable_symbol_count"] == 99
    assert artifact["descriptive_coverage_minimum_met"] is False


def test_constant_vectors_are_descriptively_unavailable_not_malformed() -> None:
    sessions = _sessions()
    factor_constant = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[_origin(sessions, 0, constant_factor=True)],
    )
    return_constant = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[_origin(sessions, 0, constant_return=True)],
    )

    assert factor_constant["origin_diagnostics"][0]["blockers"] == ["constant_factor"]
    assert return_constant["origin_diagnostics"][0]["blockers"] == ["constant_return"]
    assert factor_constant["rank_ic_available_origin_count"] == 0
    assert factor_constant["statistics"] is None
    assert factor_constant["status"] == "ACCUMULATING"


def test_ties_use_deterministic_average_rank_and_normalize_output() -> None:
    sessions = _sessions()
    symbols = _symbols(3)
    origin = _origin(sessions, 0, symbol_count=3)
    tied = replace(
        origin,
        factor_values={
            symbols[0]: "1",
            symbols[1]: "2",
            symbols[2]: "2",
        },
        forward_returns={
            symbols[0]: "3",
            symbols[1]: "2",
            symbols[2]: "1",
        },
    )

    artifact = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[tied],
    )

    assert artifact["origin_diagnostics"][0]["rank_ic"] == "-0.866025403784"
    assert artifact["statistics"]["rank_ic_population_stddev"] == "0.000000000000"
    assert "-0.000000000000" not in str(artifact)


def test_permutation_and_identical_duplicate_are_exact_replay() -> None:
    sessions = _sessions()
    first = _origin(sessions, 0)
    second = _origin(sessions, 1)
    reversed_first = replace(
        first,
        factor_values=dict(reversed(list(first.factor_values.items()))),
        forward_returns=dict(reversed(list(first.forward_returns.items()))),
    )
    expected = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[first, second],
    )
    observed = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[second, reversed_first, reversed_first],
    )

    assert observed == expected
    assert (
        validate_factor_diagnostic_replay(
            observed,
            stratum=_stratum(sessions),
            evaluation_cutoff=_cutoff(),
            open_sessions=sessions,
            origins=[first, second],
        )
        == observed
    )


def test_conflicting_duplicate_and_same_session_alias_fail_closed() -> None:
    sessions = _sessions()
    origin = _origin(sessions, 0)
    conflicting = replace(origin, forward_returns={**origin.forward_returns, "000001.SZ": "2"})
    alias = replace(origin, origin_id="origin.alias")

    with pytest.raises(FactorDiagnosticError, match="conflicting duplicate"):
        build_factor_diagnostic(
            stratum=_stratum(sessions),
            evaluation_cutoff=_cutoff(),
            open_sessions=sessions,
            origins=[origin, conflicting],
        )
    with pytest.raises(FactorDiagnosticError, match="decision session"):
        build_factor_diagnostic(
            stratum=_stratum(sessions),
            evaluation_cutoff=_cutoff(),
            open_sessions=sessions,
            origins=[origin, alias],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"label_available_at": "2027-01-01T00:00:00Z"}, "not naturally matured"),
        ({"label_available_at": "2026-01-20T08:00:00Z"}, "before its horizon end"),
        ({"horizon_end_session": "2026-01-20"}, "exact 20-session"),
        ({"factor_values": {"000001.SZ": "NaN"}}, "canonical finite decimal"),
        ({"factor_values": {"000001.SZ": "1.0"}}, "canonical finite decimal"),
    ],
)
def test_future_maturity_wrong_horizon_and_noncanonical_numbers_fail_closed(
    mutation: dict[str, object],
    message: str,
) -> None:
    sessions = _sessions()
    origin = replace(_origin(sessions, 0), **mutation)

    with pytest.raises(FactorDiagnosticError, match=message):
        build_factor_diagnostic(
            stratum=_stratum(sessions),
            evaluation_cutoff=_cutoff(),
            open_sessions=sessions,
            origins=[origin],
        )


def test_calendar_and_exact_stratum_are_hash_bound() -> None:
    sessions = _sessions()
    stratum = _stratum(sessions)
    other_stratum = replace(stratum, factor_set_sha256="9" * 64)
    first = build_factor_diagnostic(
        stratum=stratum,
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[],
    )
    second = build_factor_diagnostic(
        stratum=other_stratum,
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[],
    )

    assert first["stratum_sha256"] != second["stratum_sha256"]
    assert first["diagnostic_id"] != second["diagnostic_id"]
    with pytest.raises(FactorDiagnosticError, match="calendar SHA"):
        build_factor_diagnostic(
            stratum=stratum,
            evaluation_cutoff=_cutoff(),
            open_sessions=(*sessions[:-1], "2027-01-01"),
            origins=[],
        )


def test_unavailable_is_non_receipt_and_contains_no_evidence_metrics() -> None:
    artifact = build_unavailable_factor_diagnostic(
        factor_name="cn_factor_test",
        evaluation_cutoff=_cutoff(),
        unavailable_prerequisites=["label_contract_missing", "calendar_missing"],
    )

    assert artifact["status"] == "UNAVAILABLE"
    assert artifact["stratum"] is None
    assert artifact["stratum_sha256"] is None
    assert artifact["statistics"] is None
    assert artifact["origin_diagnostics"] == []
    assert artifact["descriptive_coverage_minimum_met"] is False
    assert "receipt" not in artifact["version"]
    assert (
        validate_factor_diagnostic_replay(
            artifact,
            factor_name="cn_factor_test",
            evaluation_cutoff=_cutoff(),
            unavailable_prerequisites=["calendar_missing", "label_contract_missing"],
        )
        == artifact
    )


def test_unavailable_does_not_hide_malformed_subject_or_cutoff() -> None:
    with pytest.raises(FactorDiagnosticError):
        build_unavailable_factor_diagnostic(
            factor_name="CN BAD",
            evaluation_cutoff="bad",
            unavailable_prerequisites=["lineage_missing"],
        )
    with pytest.raises(FactorDiagnosticError, match="reserved diagnostic blocker"):
        build_unavailable_factor_diagnostic(
            factor_name="cn_factor_test",
            evaluation_cutoff=_cutoff(),
            unavailable_prerequisites=["inference_not_implemented"],
        )


def test_schema_and_semantic_validator_reject_authority_policy_or_state_drift() -> None:
    sessions = _sessions()
    artifact = build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[],
    )
    with pytest.raises(Exception):
        validate_artifact({**artifact, "status": "ACCUMULATING"})
    with pytest.raises(Exception):
        validate_artifact(
            _reseal(
                artifact,
                authority={**artifact["authority"], "factor_governance_write": True},
            )
        )
    policy_ref = {**artifact["policy_ref"], "byte_sha256": "0" * 64}
    with pytest.raises(ArtifactContractError, match="policy identity"):
        validate_artifact(_reseal(artifact, policy_ref=policy_ref))
    with pytest.raises(Exception):
        validate_artifact(_reseal(artifact, inference_gate_passed=True))
    with pytest.raises(Exception):
        validate_artifact(_reseal(artifact, effectiveness_claimed=True))


def test_runtime_resource_limit_and_no_write_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions = _sessions()
    origin = _origin(sessions, 0)
    monkeypatch.chdir(tmp_path)
    before = _tree(tmp_path)

    with pytest.raises(FactorDiagnosticError, match="resource limit"):
        build_factor_diagnostic(
            stratum=_stratum(sessions),
            evaluation_cutoff=_cutoff(),
            open_sessions=sessions,
            origins=[origin] * 4_097,
        )
    build_factor_diagnostic(
        stratum=_stratum(sessions),
        evaluation_cutoff=_cutoff(),
        open_sessions=sessions,
        origins=[origin],
    )
    assert _tree(tmp_path) == before
