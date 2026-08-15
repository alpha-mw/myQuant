from __future__ import annotations

from copy import deepcopy
from decimal import Decimal

import pytest

from quant_investor.contracts import seal_artifact
import quant_investor.factors.governance as governance
from quant_investor.factors.governance import FactorGovernanceError
from quant_investor.factors.governance.common import decimal_text
from quant_investor.factors.governance.execution import (
    _build_execution_turnover_evidence,
    validate_execution_turnover_evidence,
)
from test_unified_factor_prospective import _preregistration, _ref, _selection


def _capture_artifact(preregistration: dict, selection: dict, ordinal: int) -> dict:
    payload = {
        "signal_capture_id": f"capture-{ordinal:03d}",
        "observation_lineage_id": "observation-lineage-test",
        "previous_signal_capture_ref": None,
        "preregistration_id": preregistration["payload"]["preregistration_id"],
        "selection_id": selection["payload"]["selection_id"],
        "ordinal": ordinal,
        "signal_session": preregistration["payload"]["signal_sessions"][ordinal],
        "source_decode_attestation_ref": _ref(
            "factor.source_decode_attestation", f"signal-attestation-{ordinal:03d}"
        ),
        "pit_universe_count": 100,
        "pit_universe_sha256": "7" * 64,
        "configuration_rows": [],
        "coverage_minimum": "0.800000000000",
        "label_inputs_used": False,
        "unlisted_universe_weight": "EXACT_ZERO",
        "backfill": False,
        "authority": "NON_AUTHORIZING",
    }
    return seal_artifact(
        "factor.signal_capture",
        payload,
        created_at="2026-08-17T07:00:00Z",
    )


def _observation_artifact(preregistration: dict, selection: dict, ordinal: int) -> dict:
    payload = {
        "observation_id": f"observation-{ordinal:03d}",
        "observation_lineage_id": "observation-lineage-test",
        "previous_observation_ref": None,
        "preregistration_id": preregistration["payload"]["preregistration_id"],
        "selection_id": selection["payload"]["selection_id"],
        "signal_capture_ref": _ref("factor.signal_capture", f"capture-{ordinal:03d}"),
        "source_decode_attestation_ref": _ref(
            "factor.source_decode_attestation", f"label-attestation-{ordinal:03d}"
        ),
        "ordinal": ordinal,
        "signal_session": preregistration["payload"]["signal_sessions"][ordinal],
        "label_start_session": preregistration["payload"]["open_sessions"][ordinal + 1],
        "label_end_session": preregistration["payload"]["open_sessions"][ordinal + 30],
        "label_formula": "adj_close[t+30]/adj_close[t+1]-1",
        "neutralization_method": "PIT_INDUSTRY_PLUS_LOG_TOTAL_MV_OLS",
        "coverage_minimum": "0.800000000000",
        "pit_universe_count": 100,
        "pit_universe_sha256": "7" * 64,
        "label_values_sha256": "8" * 64,
        "label_finite_pair_count": 100,
        "configuration_rows": [],
        "backfill": False,
        "substitution": False,
    }
    return seal_artifact(
        "factor.prospective_observation",
        payload,
        created_at="2026-09-16T07:00:00Z",
    )


def _closure() -> tuple[dict, dict, list[dict], list[dict]]:
    preregistration = _preregistration()
    selection = _selection(preregistration)
    captures = [_capture_artifact(preregistration, selection, ordinal) for ordinal in range(360)]
    observations = [
        _observation_artifact(preregistration, selection, ordinal) for ordinal in range(360)
    ]
    return preregistration, selection, captures, observations


def _configuration_rows(selection: dict, *, gross_count: int = 360) -> list[dict]:
    rows = []
    for selected in selection["payload"]["selected_configurations"]:
        rows.append(
            {
                "configuration_id": selected["selected_configuration_id"],
                "factor_id": selected["selected_factor_id"],
                "session_summary_sha256": "9" * 64,
                "session_summary_count": 360,
                "initial_entry_turnover": decimal_text(Decimal("1")),
                "rebalance_turnover": decimal_text(Decimal("0.5")),
                "terminal_exit_turnover": decimal_text(Decimal("0.5")),
                "total_turnover": decimal_text(Decimal("2")),
                "annualized_turnover": decimal_text(Decimal("1.4")),
                "total_estimated_cost": decimal_text(Decimal("0.0001")),
                "gross_labeled_return_count": gross_count,
                "gross_labeled_return_sum": (
                    decimal_text(Decimal("0.36")) if gross_count else None
                ),
                "net_labeled_return_sum": (
                    decimal_text(Decimal("0.3599")) if gross_count else None
                ),
            }
        )
    return rows


def test_execution_evidence_replays_daily_turnover_cost_and_annualization() -> None:
    preregistration, selection, captures, observations = _closure()
    evidence = _build_execution_turnover_evidence(
        preregistration=preregistration,
        selection=selection,
        signal_captures=captures,
        observations=observations,
        configuration_rows=_configuration_rows(selection),
        execution_state="COMPLETE",
        blockers=[],
        trusted_at="2027-09-10T07:00:00Z",
    )

    assert (
        validate_execution_turnover_evidence(
            evidence,
            preregistration=preregistration,
            selection=selection,
            signal_captures=captures,
            observations=observations,
        )
        == evidence
    )
    assert evidence["payload"]["signal_session_count"] == 360
    assert len(evidence["payload"]["signal_capture_refs"]) == 360
    assert len(evidence["payload"]["observation_refs"]) == 360
    for row in evidence["payload"]["configuration_rows"]:
        assert row["total_turnover"] == "2.000000000000"
        assert row["annualized_turnover"] == "1.400000000000"
        assert row["total_estimated_cost"] == "0.000100000000"
        assert row["net_labeled_return_sum"] == "0.359900000000"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("total_turnover", "1.000000000000", "components do not sum"),
        ("annualized_turnover", "1.399999999999", "annualized turnover differs"),
        ("total_estimated_cost", "0.000050000000", "cost differs"),
        ("net_labeled_return_sum", "0.360000000000", "net return"),
    ],
)
def test_execution_evidence_rejects_forged_aggregates(field: str, value: str, match: str) -> None:
    preregistration, selection, captures, observations = _closure()
    evidence = _build_execution_turnover_evidence(
        preregistration=preregistration,
        selection=selection,
        signal_captures=captures,
        observations=observations,
        configuration_rows=_configuration_rows(selection),
        execution_state="COMPLETE",
        blockers=[],
        trusted_at="2027-09-10T07:00:00Z",
    )
    forged = deepcopy(evidence)
    forged["payload"]["configuration_rows"][0][field] = value
    forged = seal_artifact(
        "factor.execution_turnover_evidence",
        forged["payload"],
        created_at=forged["created_at"],
    )
    with pytest.raises(FactorGovernanceError, match=match):
        validate_execution_turnover_evidence(
            forged,
            preregistration=preregistration,
            selection=selection,
        )


def test_execution_incomplete_is_explicit_and_cannot_claim_complete() -> None:
    preregistration, selection, captures, observations = _closure()
    incomplete = _build_execution_turnover_evidence(
        preregistration=preregistration,
        selection=selection,
        signal_captures=captures,
        observations=observations,
        configuration_rows=_configuration_rows(selection, gross_count=0),
        execution_state="INCOMPLETE",
        blockers=["EXECUTION_HELD_LABEL_MISSING"],
        trusted_at="2027-09-10T07:00:00Z",
    )
    assert incomplete["payload"]["execution_state"] == "INCOMPLETE"

    forged = deepcopy(incomplete)
    forged["payload"]["execution_state"] = "COMPLETE"
    forged["payload"]["blockers"] = []
    forged = seal_artifact(
        "factor.execution_turnover_evidence",
        forged["payload"],
        created_at=forged["created_at"],
    )
    with pytest.raises(FactorGovernanceError, match="lacks all matured returns"):
        validate_execution_turnover_evidence(
            forged,
            preregistration=preregistration,
            selection=selection,
        )


def test_execution_requires_exact_360_capture_and_observation_refs() -> None:
    preregistration, selection, captures, observations = _closure()
    with pytest.raises(FactorGovernanceError, match="exactly 360 refs"):
        _build_execution_turnover_evidence(
            preregistration=preregistration,
            selection=selection,
            signal_captures=captures[:-1],
            observations=observations,
            configuration_rows=_configuration_rows(selection),
            execution_state="COMPLETE",
            blockers=[],
            trusted_at="2027-09-10T07:00:00Z",
        )


def test_old_caller_weight_execution_builder_is_absent() -> None:
    assert not hasattr(governance, "build_execution_turnover_evidence")
