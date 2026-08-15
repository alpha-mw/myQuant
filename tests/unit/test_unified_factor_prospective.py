from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quant_investor.contracts import get_contract, seal_artifact
import quant_investor.factors.governance as governance
from quant_investor.factors.governance import FactorGovernanceError
from quant_investor.factors.governance.bootstrap import (
    BLEND_W75_CONTROL,
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
)
from quant_investor.factors.governance.common import (
    business_identity,
    canonical_a_share_symbol,
    decimal_text,
)
from quant_investor.factors.governance.custody import build_composite_state
from quant_investor.factors.governance.implementations import installed_semantic_row
from quant_investor.factors.governance.prospective import (
    _build_configuration_selection,
    _build_observation,
    _build_preregistration,
    _build_signal_capture,
    validate_configuration_selection,
    validate_observation,
    validate_preregistration,
    validate_signal_capture,
)

STAMP = "2026-08-16T00:00:00Z"


def _ref(kind: str, artifact_id: str) -> dict[str, str]:
    return {
        "kind": kind,
        "contract_sha256": get_contract(kind).contract_sha256,
        "artifact_id": artifact_id,
        "semantic_sha256": "1" * 64,
        "byte_sha256": "2" * 64,
    }


def _candidate(factor_id: str, configuration_id: str) -> dict:
    installed = installed_semantic_row(factor_id)
    body = {
        "configuration_id": configuration_id,
        "factor_id": factor_id,
        "implementation_id": installed["implementation_id"],
        "implementation_component_ref": _ref(
            "system.installed_component_manifest", f"component-{configuration_id}"
        ),
        "implementation_sha256": installed["code_sha256"],
        "family": installed["family"],
        "primitive": installed["primitive"],
        "direction": installed["direction"],
        "formula": installed["formula"],
        "normalized_expression": installed["normalized_expression"],
        "parameters_json": installed["parameters_json"],
        "input_fields": installed["input_fields"],
        "role": "PRIMARY",
    }
    return {
        "candidate_spec_id": business_identity("factor-candidate-spec", body),
        **body,
    }


def _calendar() -> tuple[list[str], list[dict]]:
    first = date(2026, 8, 17)
    sessions = [(first + timedelta(days=index)).isoformat() for index in range(390)]
    rows = []
    for ordinal, session in enumerate(sessions):
        current = date.fromisoformat(session)
        opens = datetime.combine(current, datetime.min.time(), tzinfo=timezone.utc) + timedelta(
            hours=1
        )
        closes = opens + timedelta(hours=6)
        next_opens = opens + timedelta(days=1)
        rows.append(
            {
                "ordinal": ordinal,
                "open_session": session,
                "opens_at_utc": opens.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "closes_at_utc": closes.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "next_opens_at_utc": next_opens.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
        )
    return sessions, rows


def _preregistration() -> dict:
    sessions, windows = _calendar()
    return _build_preregistration(
        open_sessions=sessions,
        session_windows=windows,
        candidates=[
            _candidate(LOW_DOLLAR_VOLUME, "config-low"),
            _candidate(BLEND_W80, "config-w80"),
        ],
        exchange_calendar_ref=_ref("system.source_object", "calendar"),
        implementation_manifest_ref=_ref("system.source_object", "implementations"),
        source_decode_attestation_ref=_ref(
            "factor.source_decode_attestation", "prereg-attestation"
        ),
        factor_validator_manifest_ref=_ref("factor.validator_manifest", "factor-validator"),
        trusted_at=STAMP,
    )


def _selection_summary(candidate: dict, *, selection_complete: int = 80) -> dict:
    return {
        "configuration_id": candidate["configuration_id"],
        "factor_id": candidate["factor_id"],
        "normalized_input_sha256": "3" * 64,
        "signal_sha256": "4" * 64,
        "finite_signal_count": 90,
        "required_input_complete_count": 85,
        "selection_complete_count": selection_complete,
        "denominator_count": 100,
        "signal_coverage": decimal_text(Decimal("0.90")),
        "selection_coverage": decimal_text(Decimal(selection_complete) / Decimal(100)),
        "coverage_gate": "PASSED" if selection_complete >= 80 else "FAILED",
    }


def _selection(preregistration: dict) -> dict:
    candidates = preregistration["payload"]["candidates"]
    return _build_configuration_selection(
        preregistration=preregistration,
        source_decode_attestation_ref=_ref(
            "factor.source_decode_attestation", "signal-zero-attestation"
        ),
        configuration_summary_rows=[_selection_summary(candidate) for candidate in candidates],
        selected_configurations=[
            {
                "primary_configuration_id": candidate["configuration_id"],
                "selected_configuration_id": candidate["configuration_id"],
                "selected_factor_id": candidate["factor_id"],
                "used_alternate": False,
            }
            for candidate in candidates
        ],
        trusted_at="2026-08-17T07:00:00Z",
    )


def _capture_rows(selection: dict, *, numerator: int = 80) -> list[dict]:
    rows = []
    for selected in selection["payload"]["selected_configurations"]:
        rows.append(
            {
                "configuration_id": selected["selected_configuration_id"],
                "factor_id": selected["selected_factor_id"],
                "signal_values_sha256": "5" * 64,
                "finite_signal_count": numerator,
                "coverage_numerator_count": numerator,
                "coverage_denominator_count": 100,
                "coverage": decimal_text(Decimal(numerator) / Decimal(100)),
                "coverage_gate": "PASSED" if numerator >= 80 else "FAILED",
                "portfolio_weights_sha256": "6" * 64,
                "nonzero_weight_count": 2,
                "long_weight": decimal_text(Decimal("0.5")),
                "short_weight": decimal_text(Decimal("0")),
                "gross_weight": decimal_text(Decimal("0.5")),
                "net_weight": decimal_text(Decimal("0.5")),
            }
        )
    return rows


def _capture(preregistration: dict, selection: dict) -> dict:
    return _build_signal_capture(
        preregistration=preregistration,
        selection=selection,
        previous_signal_capture=None,
        source_decode_attestation_ref=selection["payload"]["source_decode_attestation_ref"],
        ordinal=0,
        pit_universe_count=100,
        pit_universe_sha256="7" * 64,
        configuration_rows=_capture_rows(selection),
        trusted_at="2026-08-17T07:00:00Z",
    )


def _observation_rows(selection: dict, *, complete_cases: int = 20) -> list[dict]:
    rows = []
    for selected in selection["payload"]["selected_configurations"]:
        valid = complete_cases >= 20
        rows.append(
            {
                "configuration_id": selected["selected_configuration_id"],
                "factor_id": selected["selected_factor_id"],
                "signal_values_sha256": "5" * 64,
                "coverage_numerator_count": 80,
                "coverage_denominator_count": 100,
                "coverage": decimal_text(Decimal("0.8")),
                "coverage_gate": "PASSED",
                "complete_case_count": complete_cases,
                "held_nonzero_symbol_count": 2,
                "held_missing_label_count": 0,
                "gross_labeled_return_symbol_count": 2,
                "gross_labeled_return": decimal_text(Decimal("0.01")),
                "rank_ic": decimal_text(Decimal("0.1")) if valid else None,
                "rank_ic_p_value": decimal_text(Decimal("0.05")) if valid else None,
                "valid_daily_rankic": valid,
            }
        )
    return rows


def test_prospective_artifacts_replay_exact_compact_contracts() -> None:
    preregistration = _preregistration()
    selection = _selection(preregistration)
    capture = _capture(preregistration, selection)
    observation = _build_observation(
        preregistration=preregistration,
        selection=selection,
        signal_capture=capture,
        previous_observation=None,
        source_decode_attestation_ref=_ref(
            "factor.source_decode_attestation", "label-zero-attestation"
        ),
        pit_universe_sha256="7" * 64,
        label_values_sha256="8" * 64,
        label_finite_pair_count=100,
        configuration_rows=_observation_rows(selection),
        trusted_at="2026-09-16T07:00:00Z",
    )

    assert validate_preregistration(preregistration) == preregistration
    assert validate_configuration_selection(selection, preregistration=preregistration) == selection
    assert (
        validate_signal_capture(
            capture,
            preregistration=preregistration,
            selection=selection,
        )
        == capture
    )
    assert (
        validate_observation(
            observation,
            preregistration=preregistration,
            selection=selection,
            signal_capture=capture,
        )
        == observation
    )
    assert "pit_universe_rows" not in selection["payload"]
    assert "source_refs" not in observation["payload"]


def test_initial_selection_uses_full_denominator_and_exact_point_eight_gate() -> None:
    preregistration = _preregistration()
    candidates = preregistration["payload"]["candidates"]
    selection = _selection(preregistration)
    assert all(
        row["selection_coverage"] == "0.800000000000" and row["coverage_gate"] == "PASSED"
        for row in selection["payload"]["configuration_summary_rows"]
    )

    with pytest.raises(FactorGovernanceError, match="fail initial coverage"):
        _build_configuration_selection(
            preregistration=preregistration,
            source_decode_attestation_ref=_ref(
                "factor.source_decode_attestation", "signal-zero-attestation"
            ),
            configuration_summary_rows=[
                _selection_summary(candidate, selection_complete=79) for candidate in candidates
            ],
            selected_configurations=[
                {
                    "primary_configuration_id": candidate["configuration_id"],
                    "selected_configuration_id": candidate["configuration_id"],
                    "selected_factor_id": candidate["factor_id"],
                    "used_alternate": False,
                }
                for candidate in candidates
            ],
            trusted_at="2026-08-17T07:00:00Z",
        )


def test_signal_coverage_is_signal_only_and_full_pit_denominator() -> None:
    preregistration = _preregistration()
    selection = _selection(preregistration)
    capture = _capture(preregistration, selection)
    assert all(
        row["coverage_numerator_count"] == 80
        and row["coverage_denominator_count"] == 100
        and row["coverage_gate"] == "PASSED"
        for row in capture["payload"]["configuration_rows"]
    )

    forged = deepcopy(capture)
    forged["payload"]["configuration_rows"][0]["coverage_denominator_count"] = 80
    forged = seal_artifact(
        "factor.signal_capture",
        forged["payload"],
        created_at=forged["created_at"],
    )
    with pytest.raises(FactorGovernanceError, match="coverage count"):
        validate_signal_capture(
            forged,
            preregistration=preregistration,
            selection=selection,
        )


@pytest.mark.parametrize("complete_cases, expected_valid", [(19, False), (20, True)])
def test_observation_rankic_needs_twenty_complete_pairs(
    complete_cases: int, expected_valid: bool
) -> None:
    preregistration = _preregistration()
    selection = _selection(preregistration)
    capture = _capture(preregistration, selection)
    observation = _build_observation(
        preregistration=preregistration,
        selection=selection,
        signal_capture=capture,
        previous_observation=None,
        source_decode_attestation_ref=_ref(
            "factor.source_decode_attestation", "label-zero-attestation"
        ),
        pit_universe_sha256="7" * 64,
        label_values_sha256="8" * 64,
        label_finite_pair_count=complete_cases,
        configuration_rows=_observation_rows(selection, complete_cases=complete_cases),
        trusted_at="2026-09-16T07:00:00Z",
    )
    assert all(
        row["valid_daily_rankic"] is expected_valid
        for row in observation["payload"]["configuration_rows"]
    )


def test_candidate_identity_and_installed_control_boundary_fail_closed() -> None:
    sessions, windows = _calendar()
    candidate = _candidate(LOW_DOLLAR_VOLUME, "config-low")
    candidate["candidate_spec_id"] = "factor-candidate-spec-" + "0" * 64
    with pytest.raises(FactorGovernanceError, match="candidate business identity"):
        _build_preregistration(
            open_sessions=sessions,
            session_windows=windows,
            candidates=[candidate, _candidate(BLEND_W80, "config-w80")],
            exchange_calendar_ref=_ref("system.source_object", "calendar"),
            implementation_manifest_ref=_ref("system.source_object", "implementations"),
            source_decode_attestation_ref=_ref(
                "factor.source_decode_attestation", "prereg-attestation"
            ),
            factor_validator_manifest_ref=_ref("factor.validator_manifest", "factor-validator"),
            trusted_at=STAMP,
        )


def test_old_timestamp_and_pandas_builders_are_absent_from_stable_api() -> None:
    for name in (
        "build_preregistration",
        "select_initial_configuration",
        "record_observation",
    ):
        assert not hasattr(governance, name)


def test_coverage_below_080_is_terminal_without_switch_or_stitch() -> None:
    preregistration = _preregistration()
    selection = _selection(preregistration)
    failed = _build_signal_capture(
        preregistration=preregistration,
        selection=selection,
        previous_signal_capture=None,
        source_decode_attestation_ref=selection["payload"]["source_decode_attestation_ref"],
        ordinal=0,
        pit_universe_count=100,
        pit_universe_sha256="7" * 64,
        configuration_rows=_capture_rows(selection, numerator=79),
        trusted_at="2026-08-17T07:00:00Z",
    )
    assert all(
        row["coverage"] == "0.790000000000" and row["coverage_gate"] == "FAILED"
        for row in failed["payload"]["configuration_rows"]
    )
    terminal = build_composite_state(
        custody_namespace_id="factor-terminal-coverage-test",
        preregistration_ref=_ref("factor.preregistration", "preregistration"),
        cycle_state="TERMINAL_INCOMPLETE",
        transaction_sequence=1,
        previous_composite_state_ref=None,
        transaction_id="factor-custody-transaction-" + "a" * 64,
        custody_record_count=2,
        custody_head_ref=_ref("factor.custody_record", "coverage-failure"),
        selection_ref=_ref("factor.configuration_selection", "selection"),
        signal_capture_count=1,
        signal_capture_head_ref=_ref("factor.signal_capture", "failed-capture"),
        observation_count=0,
        observation_head_ref=None,
        execution_evidence_ref=None,
        evaluation_ref=None,
        admitted_set_ref=None,
        intrinsic_receipt_ref=None,
        resolved_signal_slot_count=1,
        resolved_label_slot_count=0,
        slot_tree_sha256="b" * 64,
        terminal=True,
        blockers=["SIGNAL_COVERAGE_BELOW_MINIMUM"],
        last_stored_at="2026-08-17T07:00:00Z",
    )
    assert terminal["payload"]["terminal"] is True
    assert terminal["payload"]["blockers"] == ["SIGNAL_COVERAGE_BELOW_MINIMUM"]
    assert selection["payload"]["substitution_allowed"] is False


def test_low_dollar_and_w80_may_reenter_but_w75_is_permanent_control() -> None:
    preregistration = _preregistration()
    assert {row["factor_id"] for row in preregistration["payload"]["candidates"]} == {
        LOW_DOLLAR_VOLUME,
        BLEND_W80,
    }
    with pytest.raises(FactorGovernanceError, match="implementation is not installed"):
        _candidate(BLEND_W75_CONTROL, "config-w75")


@pytest.mark.parametrize("symbol", ["000001.sz", "000001", "SH.600000", "ABCDEF.SH"])
def test_lowercase_or_noncanonical_symbols_fail_closed(symbol: str) -> None:
    with pytest.raises(FactorGovernanceError, match="canonical A-share symbol"):
        canonical_a_share_symbol(symbol, label="prospective.symbol")
    assert canonical_a_share_symbol("000001.SZ", label="prospective.symbol") == "000001.SZ"


def test_observation_uses_only_selected_ids_real_symbols_and_exact_coverage() -> None:
    preregistration = _preregistration()
    selection = _selection(preregistration)
    capture = _capture(preregistration, selection)
    observation = _build_observation(
        preregistration=preregistration,
        selection=selection,
        signal_capture=capture,
        previous_observation=None,
        source_decode_attestation_ref=_ref(
            "factor.source_decode_attestation", "label-zero-attestation"
        ),
        pit_universe_sha256="7" * 64,
        label_values_sha256="8" * 64,
        label_finite_pair_count=100,
        configuration_rows=_observation_rows(selection),
        trusted_at="2026-09-16T07:00:00Z",
    )
    assert canonical_a_share_symbol("600000.SH", label="observation.symbol") == "600000.SH"
    forged = deepcopy(observation)
    forged["payload"]["configuration_rows"][0]["configuration_id"] = "not-selected"
    forged = seal_artifact(
        "factor.prospective_observation", forged["payload"], created_at=forged["created_at"]
    )
    with pytest.raises(FactorGovernanceError, match="configuration rows are not canonical"):
        validate_observation(
            forged,
            preregistration=preregistration,
            selection=selection,
            signal_capture=capture,
        )
    forged = deepcopy(observation)
    forged["payload"]["configuration_rows"][0]["coverage"] = "0.810000000000"
    forged = seal_artifact(
        "factor.prospective_observation", forged["payload"], created_at=forged["created_at"]
    )
    with pytest.raises(FactorGovernanceError, match="coverage projection"):
        validate_observation(
            forged,
            preregistration=preregistration,
            selection=selection,
            signal_capture=capture,
        )


def test_preregistration_has_exact_390_360_30_and_explicit_300_12_8_units() -> None:
    payload = _preregistration()["payload"]
    assert len(payload["open_sessions"]) == 390
    assert len(payload["signal_sessions"]) == 360
    assert len(payload["maturity_sessions"]) == 30
    assert payload["label_contract"]["maturity_offset_open_sessions"] == 30
    assert payload["maturity_contract"] == {
        "conjunctive": True,
        "minimum_valid_daily_rankic_sessions": 300,
        "minimum_closed_calendar_month_end_observations": 12,
        "minimum_disjoint_30_open_session_cohort_means": 8,
        "cohort_open_sessions": 30,
    }


def test_selection_seals_only_first_signal_and_locks_single_alternate() -> None:
    sessions, windows = _calendar()
    low = _candidate(LOW_DOLLAR_VOLUME, "config-low")
    alternate = _candidate(LOW_DOLLAR_VOLUME, "config-low-alt")
    alternate_body = dict(alternate)
    alternate_body["factor_id"] = "pv_low_dollar_volume_alternate"
    alternate_body["role"] = "ALTERNATE_FOR:config-low"
    identity_body = {
        key: value for key, value in alternate_body.items() if key != "candidate_spec_id"
    }
    alternate_body["candidate_spec_id"] = business_identity(
        "factor-candidate-spec", identity_body
    )
    w80 = _candidate(BLEND_W80, "config-w80")
    preregistration = _build_preregistration(
        open_sessions=sessions,
        session_windows=windows,
        candidates=[low, alternate_body, w80],
        exchange_calendar_ref=_ref("system.source_object", "calendar"),
        implementation_manifest_ref=_ref("system.source_object", "implementations"),
        source_decode_attestation_ref=_ref(
            "factor.source_decode_attestation", "prereg-attestation"
        ),
        factor_validator_manifest_ref=_ref("factor.validator_manifest", "factor-validator"),
        trusted_at=STAMP,
    )
    candidates = preregistration["payload"]["candidates"]
    summaries = [
        _selection_summary(row, selection_complete=(79 if row["configuration_id"] == "config-low" else 80))
        for row in candidates
    ]
    selection = _build_configuration_selection(
        preregistration=preregistration,
        source_decode_attestation_ref=_ref(
            "factor.source_decode_attestation", "signal-zero-attestation"
        ),
        configuration_summary_rows=summaries,
        selected_configurations=[
            {
                "primary_configuration_id": "config-low",
                "selected_configuration_id": "config-low-alt",
                "selected_factor_id": "pv_low_dollar_volume_alternate",
                "used_alternate": True,
            },
            {
                "primary_configuration_id": "config-w80",
                "selected_configuration_id": "config-w80",
                "selected_factor_id": BLEND_W80,
                "used_alternate": False,
            },
        ],
        trusted_at="2026-08-17T07:00:00Z",
    )
    assert selection["payload"]["first_signal_session"] == sessions[0]
    assert selection["payload"]["substitution_allowed"] is False
    assert preregistration["payload"]["alternate_policy"] == {
        "maximum_per_primary": 1,
        "selection_deadline": "FIRST_SIGNAL_CAPTURE",
        "midstream_substitution_allowed": False,
    }
