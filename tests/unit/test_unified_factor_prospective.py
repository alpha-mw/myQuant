from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from quant_investor.contracts import get_contract, seal_artifact
import quant_investor.factors.governance as governance
from quant_investor.factors.governance import FactorGovernanceError
from quant_investor.factors.governance.bootstrap import BLEND_W80, LOW_DOLLAR_VOLUME
from quant_investor.factors.governance.common import business_identity, decimal_text
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
