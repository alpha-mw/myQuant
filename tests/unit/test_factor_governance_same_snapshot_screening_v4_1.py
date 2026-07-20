from __future__ import annotations

import copy
import hashlib
import stat
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest

from quant_investor.factors import governance_private_bundle_io as private_io
from quant_investor.factors import governance_same_snapshot_screening_v4_1 as subject
from quant_investor.factors import governance_screening_v4 as screening_v4


DIAGNOSTIC_NEW_NAMES = tuple(f"new_diag_{index:03d}" for index in range(27))
TURNOVER_NEW_NAMES = tuple(
    sorted(subject.PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES)
)
FUNDAMENTAL_NEW_NAMES = tuple(
    sorted(subject.PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES)
)
ALL_NEW_NAMES = (
    *DIAGNOSTIC_NEW_NAMES,
    *TURNOVER_NEW_NAMES,
    *FUNDAMENTAL_NEW_NAMES,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _candidate(name: str, primitive_ids: list[str]) -> dict[str, Any]:
    return {
        "name": name,
        "implementation": "synthetic.strict.v1",
        "expression": f"rank({name})",
        "direction": 1.0,
        "params": {"window": 20},
        "lookback": 20,
        "slot": f"slot:{name}",
        "input_fields": ["close"],
        "primitive_ids": primitive_ids,
    }


def _catalog_input(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: copy.deepcopy(row[key])
        for key in (
            "name",
            "implementation",
            "expression",
            "direction",
            "params",
            "lookback",
            "slot",
            "input_fields",
            "primitive_ids",
        )
    }


def _reseal(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop(field, None)
    result[field] = subject.semantic_sha256_v4_1(result)
    return result


def _screening_reseal(value: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(value)
    result.pop("semantic_sha256", None)
    result["semantic_sha256"] = screening_v4.canonical_semantic_sha256(result)
    return result


@pytest.fixture(scope="module")
def evidence() -> dict[str, Any]:
    base_primitives = [
        {"primitive_id": f"base_p_{index:02d}", "family": f"base_f_{index:02d}"}
        for index in range(subject.BASE_PRIMITIVE_COUNT)
    ]
    new_primitives = [
        {"primitive_id": f"new_p_{index:02d}", "family": f"new_f_{index:02d}"}
        for index in range(5)
    ]
    base_ontology = screening_v4.build_primitive_ontology_v4(base_primitives)
    formal_ontology = screening_v4.build_primitive_ontology_v4(
        [*base_primitives, *new_primitives]
    )
    base_inputs = [
        _candidate(f"base_{index:03d}", [f"base_p_{index % 13:02d}"])
        for index in range(subject.BASE_CANDIDATE_COUNT)
    ]
    new_inputs = []
    new_primitive_names = [row["primitive_id"] for row in new_primitives]
    for index, name in enumerate(ALL_NEW_NAMES):
        # The 27 evaluated rows have unique primitive sets, so correlation—not
        # primitive duplication—drives the deterministic shortlist in tests.
        mask = (index % 31) + 1
        primitive_ids = [
            name
            for bit, name in enumerate(new_primitive_names)
            if mask & (1 << bit)
        ]
        new_inputs.append(_candidate(name, primitive_ids))
    base_catalog = screening_v4.build_candidate_catalog_v4(
        ontology=base_ontology, candidates=base_inputs
    )
    formal_catalog = screening_v4.build_candidate_catalog_v4(
        ontology=formal_ontology,
        candidates=[*[_catalog_input(row) for row in base_catalog["candidates"]], *new_inputs],
    )

    evaluations: list[dict[str, Any]] = []
    diagnostic_signals: dict[str, str] = {}
    for row in formal_catalog["candidates"]:
        name = row["name"]
        if name.startswith("base_"):
            evaluations.append(
                {
                    "name": name,
                    "status": subject.STATUS_COMPUTE_FAILED,
                    "signal_sha256": None,
                    "finite_ratio": None,
                    "raw_p_value": None,
                    "failure_reason": "synthetic compute failure",
                }
            )
            continue
        index = ALL_NEW_NAMES.index(name)
        if name in DIAGNOSTIC_NEW_NAMES:
            signal_sha = _sha(f"signal:{name}")
            diagnostic_signals[name] = signal_sha
            evaluations.append(
                {
                    "name": name,
                    "status": subject.STATUS_EVALUATED,
                    "signal_sha256": signal_sha,
                    "finite_ratio": 0.8,
                    "raw_p_value": float(0.0001 + index * 0.000001),
                    "failure_reason": None,
                }
            )
        elif name in subject.PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES:
            evaluations.append(
                {
                    "name": name,
                    "status": subject.STATUS_TURNOVER_BLOCKED,
                    "signal_sha256": None,
                    "finite_ratio": None,
                    "raw_p_value": None,
                    "failure_reason": "turnover unavailable",
                }
            )
        else:
            evaluations.append(
                {
                    "name": name,
                    "status": subject.STATUS_FUNDAMENTAL_BLOCKED,
                    "signal_sha256": None,
                    "finite_ratio": None,
                    "raw_p_value": None,
                    "failure_reason": "fundamental semantics unproven",
                }
            )

    source_bindings = {
        field: _sha(f"source:{field}")
        for field in screening_v4.SOURCE_BINDING_FIELDS
    }
    closed_dates = ["2026-01-30", "2026-02-27", "2026-03-31"]
    matrix_context = {
        "date_axis_sha256": _sha("date-axis"),
        "symbol_axis_sha256": _sha("symbol-axis"),
        "eligibility_matrix_sha256": _sha("eligibility-mask"),
        "session_scope_sha256": _sha("session-scope"),
        "calendar_sha256": source_bindings["calendar_sha256"],
        "closed_month_end_dates": closed_dates,
        "closed_month_end_axis_sha256": subject.semantic_sha256_v4_1(closed_dates),
        "session_count": 1000,
        "symbol_count": 5000,
    }
    screening = subject.build_same_snapshot_screening_v4_1(
        cycle_id="cycle-synthetic-001",
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
        evaluations=evaluations,
        diagnostic_signal_sha256_by_name=diagnostic_signals,
        matrix_context=matrix_context,
        source_bindings=source_bindings,
    )
    evaluated = [
        row["name"] for row in screening["rows"] if row["status"] == subject.STATUS_EVALUATED
    ]
    monthly_inputs = [
        {
            "left_name": left,
            "right_name": right,
            "month_end": month_end,
            "abs_spearman": 0.2,
            "valid_common_symbol_count": 100,
        }
        for left, right in combinations(evaluated, 2)
        for month_end in closed_dates
    ]
    correlation = subject.build_correlation_diagnostic_v4_1(
        cycle_id="cycle-synthetic-001",
        base_ontology=base_ontology,
        formal_ontology=formal_ontology,
        base_catalog=base_catalog,
        formal_catalog=formal_catalog,
        screening=screening,
        monthly_rows=monthly_inputs,
    )
    return {
        "base_ontology": base_ontology,
        "formal_ontology": formal_ontology,
        "base_catalog": base_catalog,
        "formal_catalog": formal_catalog,
        "evaluations": evaluations,
        "diagnostic_signals": diagnostic_signals,
        "matrix_context": matrix_context,
        "source_bindings": source_bindings,
        "screening": screening,
        "monthly_inputs": monthly_inputs,
        "correlation": correlation,
    }


def _screening_kwargs(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_ontology": evidence["base_ontology"],
        "formal_ontology": evidence["formal_ontology"],
        "base_catalog": evidence["base_catalog"],
        "formal_catalog": evidence["formal_catalog"],
    }


def _profile_evaluations(
    evidence: dict[str, Any], profile: str
) -> list[dict[str, Any]]:
    result = copy.deepcopy(evidence["evaluations"])
    resolve_names = set(subject.PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES)
    if profile == subject.ACCOUNTING_PROFILE_FULLY_RESOLVED:
        resolve_names |= set(subject.PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES)
    for index, row in enumerate(result):
        if row["name"] not in resolve_names:
            continue
        result[index] = {
            "name": row["name"],
            "status": subject.STATUS_EVALUATED,
            "signal_sha256": _sha(f"resolved-signal:{row['name']}"),
            "finite_ratio": 0.75,
            "raw_p_value": 0.25,
            "failure_reason": None,
        }
    return result


def _build_profile_screening(
    evidence: dict[str, Any], profile: str
) -> dict[str, Any]:
    return subject.build_same_snapshot_screening_v4_1(
        cycle_id="cycle-synthetic-001",
        evaluations=_profile_evaluations(evidence, profile),
        diagnostic_signal_sha256_by_name=evidence["diagnostic_signals"],
        matrix_context=evidence["matrix_context"],
        source_bindings=evidence["source_bindings"],
        input_resolution_semantic_sha256=_sha("input-resolution"),
        **_screening_kwargs(evidence),
    )


def test_exact_267_ledger_full_family_bh_and_no_authority(evidence: dict[str, Any]) -> None:
    screening = subject.validate_same_snapshot_screening_v4_1(
        evidence["screening"], **_screening_kwargs(evidence)
    )
    assert screening["catalog_accounting"] == {
        "base_candidate_count": 230,
        "new_candidate_count": 37,
        "candidate_count": 267,
    }
    assert len(screening["rows"]) == 267
    assert [row["name"] for row in screening["rows"]] == [
        row["name"] for row in evidence["formal_catalog"]["candidates"]
    ]
    assert screening["status_accounting"] == {
        "evaluated_count": 27,
        "compute_failed_count": 230,
        "blocked_count": 10,
        "turnover_data_blocked_count": 2,
        "fundamental_semantic_blocked_count": 8,
        "bh_denominator_count": 267,
    }
    assert all(value is False for value in screening["authority"].values())
    assert all(value is False for value in screening["side_effects"].values())
    assert screening["rebalance_policy"] == subject.REBALANCE_POLICY
    failed = [
        row
        for row in screening["screening_evidence"]["rows"]
        if row["evaluation_status"] == screening_v4.COMPUTE_FAILED_STATUS
    ]
    assert len(failed) == 240
    assert all(row["bh_input_p_value"] == 1.0 for row in failed)
    new_rows = [row for row in screening["rows"] if row["catalog_role"] == "new"]
    assert sum(row["diagnostic_signal_reproduced"] is True for row in new_rows) == 27
    assert screening["authority"]["runtime_equivalence_verified"] is False


@pytest.mark.parametrize(
    ("profile", "schema_version", "evaluated", "turnover", "fundamental"),
    [
        (
            subject.ACCOUNTING_PROFILE_FUNDAMENTAL_RESOLVED,
            subject.FUNDAMENTAL_RESOLVED_SCREENING_SCHEMA_VERSION,
            35,
            2,
            0,
        ),
        (
            subject.ACCOUNTING_PROFILE_FULLY_RESOLVED,
            subject.FULLY_RESOLVED_SCREENING_SCHEMA_VERSION,
            37,
            0,
            0,
        ),
    ],
)
def test_resolved_profiles_are_exact_proof_bound_and_keep_original_diagnostics(
    evidence: dict[str, Any],
    profile: str,
    schema_version: str,
    evaluated: int,
    turnover: int,
    fundamental: int,
) -> None:
    screening = _build_profile_screening(evidence, profile)
    validated = subject.validate_same_snapshot_screening_v4_1(
        screening, **_screening_kwargs(evidence)
    )

    assert validated["schema_version"] == schema_version
    assert validated["accounting_profile"] == profile
    assert validated["input_resolution_semantic_sha256"] == _sha(
        "input-resolution"
    )
    assert validated["status_accounting"] == {
        "evaluated_count": evaluated,
        "compute_failed_count": 230,
        "blocked_count": turnover + fundamental,
        "turnover_data_blocked_count": turnover,
        "fundamental_semantic_blocked_count": fundamental,
        "bh_denominator_count": 267,
    }
    new_rows = [row for row in validated["rows"] if row["catalog_role"] == "new"]
    assert {
        row["name"]
        for row in new_rows
        if row["diagnostic_signal_reproduced"] is True
    } == set(DIAGNOSTIC_NEW_NAMES)
    assert all(
        row["diagnostic_signal_reproduced"] is False
        for row in new_rows
        if row["name"] in subject.PREDECLARED_BLOCKED_CANDIDATE_NAMES
    )
    assert all(value is False for value in validated["authority"].values())
    assert all(value is False for value in validated["side_effects"].values())


def test_resolved_profiles_reject_missing_proof_partial_statuses_and_diagnostic_drift(
    evidence: dict[str, Any],
) -> None:
    with pytest.raises(
        subject.FactorGovernanceSameSnapshotScreeningV4_1Error,
        match="legacy accounting profile",
    ):
        subject.build_same_snapshot_screening_v4_1(
            cycle_id="cycle-synthetic-001",
            evaluations=evidence["evaluations"],
            diagnostic_signal_sha256_by_name=evidence["diagnostic_signals"],
            matrix_context=evidence["matrix_context"],
            source_bindings=evidence["source_bindings"],
            input_resolution_semantic_sha256=_sha("input-resolution"),
            **_screening_kwargs(evidence),
        )

    fully_resolved = _profile_evaluations(
        evidence, subject.ACCOUNTING_PROFILE_FULLY_RESOLVED
    )
    with pytest.raises(
        subject.FactorGovernanceSameSnapshotScreeningV4_1Error,
        match="input-resolution semantic SHA",
    ):
        subject.build_same_snapshot_screening_v4_1(
            cycle_id="cycle-synthetic-001",
            evaluations=fully_resolved,
            diagnostic_signal_sha256_by_name=evidence["diagnostic_signals"],
            matrix_context=evidence["matrix_context"],
            source_bindings=evidence["source_bindings"],
            **_screening_kwargs(evidence),
        )

    partial = copy.deepcopy(evidence["evaluations"])
    target = next(
        row
        for row in partial
        if row["name"] in subject.PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES
    )
    target.update(
        {
            "status": subject.STATUS_EVALUATED,
            "signal_sha256": _sha("partial"),
            "finite_ratio": 0.5,
            "raw_p_value": 0.5,
            "failure_reason": None,
        }
    )
    with pytest.raises(
        subject.FactorGovernanceSameSnapshotScreeningV4_1Error,
        match="predefined accounting profile",
    ):
        subject.build_same_snapshot_screening_v4_1(
            cycle_id="cycle-synthetic-001",
            evaluations=partial,
            diagnostic_signal_sha256_by_name=evidence["diagnostic_signals"],
            matrix_context=evidence["matrix_context"],
            source_bindings=evidence["source_bindings"],
            input_resolution_semantic_sha256=_sha("input-resolution"),
            **_screening_kwargs(evidence),
        )

    swapped = copy.deepcopy(evidence["evaluations"])
    turnover_row = next(
        row
        for row in swapped
        if row["name"] in subject.PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES
    )
    fundamental_row = next(
        row
        for row in swapped
        if row["name"] in subject.PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES
    )
    turnover_row["status"] = subject.STATUS_FUNDAMENTAL_BLOCKED
    fundamental_row["status"] = subject.STATUS_TURNOVER_BLOCKED
    with pytest.raises(
        subject.FactorGovernanceSameSnapshotScreeningV4_1Error,
        match="predefined accounting profile",
    ):
        subject.build_same_snapshot_screening_v4_1(
            cycle_id="cycle-synthetic-001",
            evaluations=swapped,
            diagnostic_signal_sha256_by_name=evidence["diagnostic_signals"],
            matrix_context=evidence["matrix_context"],
            source_bindings=evidence["source_bindings"],
            **_screening_kwargs(evidence),
        )

    diagnostic_drift = copy.deepcopy(evidence["diagnostic_signals"])
    diagnostic_drift.pop(DIAGNOSTIC_NEW_NAMES[0])
    diagnostic_drift[next(iter(subject.PREDECLARED_BLOCKED_CANDIDATE_NAMES))] = _sha(
        "forged-resolved-diagnostic"
    )
    with pytest.raises(
        subject.FactorGovernanceSameSnapshotScreeningV4_1Error,
        match="exact 27",
    ):
        subject.build_same_snapshot_screening_v4_1(
            cycle_id="cycle-synthetic-001",
            evaluations=fully_resolved,
            diagnostic_signal_sha256_by_name=diagnostic_drift,
            matrix_context=evidence["matrix_context"],
            source_bindings=evidence["source_bindings"],
            input_resolution_semantic_sha256=_sha("input-resolution"),
            **_screening_kwargs(evidence),
        )


def test_rejects_duplicate_membership_wrong_family_and_subset_bh_after_reseal(
    evidence: dict[str, Any],
) -> None:
    duplicate = copy.deepcopy(evidence["evaluations"])
    duplicate[-1] = copy.deepcopy(duplicate[0])
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.build_same_snapshot_screening_v4_1(
            cycle_id="cycle-synthetic-001",
            evaluations=duplicate,
            diagnostic_signal_sha256_by_name=evidence["diagnostic_signals"],
            matrix_context=evidence["matrix_context"],
            source_bindings=evidence["source_bindings"],
            **_screening_kwargs(evidence),
        )

    wrong_family = copy.deepcopy(evidence["screening"])
    wrong_family["rows"][0]["family"] = "forged-family"
    wrong_family["rows"][0] = _reseal(wrong_family["rows"][0], "row_semantic_sha256")
    wrong_family = _reseal(wrong_family, "screening_semantic_sha256")
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.validate_same_snapshot_screening_v4_1(
            wrong_family, **_screening_kwargs(evidence)
        )

    subset_bh = copy.deepcopy(evidence["screening"])
    subset_bh["screening_evidence"]["rows"].pop()
    subset_bh["screening_evidence"] = _screening_reseal(
        subset_bh["screening_evidence"]
    )
    subset_bh = _reseal(subset_bh, "screening_semantic_sha256")
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.validate_same_snapshot_screening_v4_1(
            subset_bh, **_screening_kwargs(evidence)
        )


def test_rejects_nonfinite_mixed_snapshot_and_authority_flip(evidence: dict[str, Any]) -> None:
    nonfinite = copy.deepcopy(evidence["evaluations"])
    evaluated = next(row for row in nonfinite if row["status"] == subject.STATUS_EVALUATED)
    evaluated["raw_p_value"] = float("nan")
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.build_same_snapshot_screening_v4_1(
            cycle_id="cycle-synthetic-001",
            evaluations=nonfinite,
            diagnostic_signal_sha256_by_name=evidence["diagnostic_signals"],
            matrix_context=evidence["matrix_context"],
            source_bindings=evidence["source_bindings"],
            **_screening_kwargs(evidence),
        )

    mixed_snapshot = copy.deepcopy(evidence["matrix_context"])
    mixed_snapshot["calendar_sha256"] = _sha("different-calendar")
    with pytest.raises(
        subject.FactorGovernanceSameSnapshotScreeningV4_1Error,
        match="calendar SHA differs",
    ):
        subject.build_same_snapshot_screening_v4_1(
            cycle_id="cycle-synthetic-001",
            evaluations=evidence["evaluations"],
            diagnostic_signal_sha256_by_name=evidence["diagnostic_signals"],
            matrix_context=mixed_snapshot,
            source_bindings=evidence["source_bindings"],
            **_screening_kwargs(evidence),
        )

    authority = copy.deepcopy(evidence["screening"])
    authority["authority"]["qualification_authority"] = True
    authority = _reseal(authority, "screening_semantic_sha256")
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.validate_same_snapshot_screening_v4_1(
            authority, **_screening_kwargs(evidence)
        )


def test_correlation_complete_pairs_policy_axes_and_shortlist(evidence: dict[str, Any]) -> None:
    correlation = subject.validate_correlation_diagnostic_v4_1(
        evidence["correlation"], screening=evidence["screening"], **_screening_kwargs(evidence)
    )
    assert correlation["expected_pair_count"] == 351
    assert correlation["observed_pair_count"] == 351
    assert len(correlation["monthly_pair_evidence"]) == 351
    assert all(
        len(row["abs_spearman_by_month"]) == 3
        and len(row["valid_common_symbol_count_by_month"]) == 3
        for row in correlation["monthly_pair_evidence"]
    )
    assert all(row["valid_month_count"] == 3 for row in correlation["pair_summaries"])
    assert correlation["correlation_contract"] == {
        "metric": subject.CORRELATION_METRIC,
        "threshold": 0.70,
        "minimum_valid_symbol_count_per_month": 20,
        "minimum_valid_month_count": 3,
        "rebalance_policy": subject.REBALANCE_POLICY,
        "axes_bound": True,
        "eligibility_mask_bound": True,
        "formal_dedup_authority": False,
    }
    assert correlation["matrix_context"] == evidence["screening"]["matrix_context"]
    assert len(correlation["exploratory_new_candidate_shortlist"]) == 10
    assert [row["rank"] for row in correlation["exploratory_new_candidate_shortlist"]] == list(
        range(1, 11)
    )
    assert all(value is False for value in correlation["authority"].values())
    assert all(value is False for value in correlation["side_effects"].values())


def test_resolved_provenance_threads_through_correlation_and_readback(
    tmp_path: Path,
    evidence: dict[str, Any],
) -> None:
    screening = _build_profile_screening(
        evidence, subject.ACCOUNTING_PROFILE_FULLY_RESOLVED
    )
    evaluated = [
        row["name"]
        for row in screening["rows"]
        if row["status"] == subject.STATUS_EVALUATED
    ]
    monthly_rows = [
        {
            "left_name": left,
            "right_name": right,
            "month_end": month_end,
            "abs_spearman": 0.2,
            "valid_common_symbol_count": 100,
        }
        for left, right in combinations(evaluated, 2)
        for month_end in evidence["matrix_context"]["closed_month_end_dates"]
    ]
    correlation = subject.build_correlation_diagnostic_v4_1(
        cycle_id="cycle-synthetic-001",
        screening=screening,
        monthly_rows=monthly_rows,
        **_screening_kwargs(evidence),
    )
    artifacts = {
        subject.SCREENING_FILENAME: screening,
        subject.CORRELATION_FILENAME: correlation,
    }
    bindings = []
    for filename in subject.BUNDLE_INPUT_FILENAMES:
        raw = subject.canonical_file_bytes_v4_1(artifacts[filename])
        bindings.append(
            {
                "filename": filename,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "mode": 0o600,
                "uid": 501,
                "nlink": 1,
            }
        )
    readback = subject.build_readback_report_v4_1(
        run_id="resolved-test",
        artifacts=artifacts,
        artifact_bindings=bindings,
        **_screening_kwargs(evidence),
    )

    assert correlation["schema_version"] == (
        subject.FULLY_RESOLVED_CORRELATION_SCHEMA_VERSION
    )
    assert readback["schema_version"] == subject.FULLY_RESOLVED_READBACK_SCHEMA_VERSION
    for payload in (screening, correlation, readback):
        assert payload["accounting_profile"] == (
            subject.ACCOUNTING_PROFILE_FULLY_RESOLVED
        )
        assert payload["input_resolution_semantic_sha256"] == _sha(
            "input-resolution"
        )
        assert all(value is False for value in payload["authority"].values())
        assert all(value is False for value in payload["side_effects"].values())
    assert correlation["evaluated_new_candidate_count"] == 37
    assert correlation["expected_pair_count"] == 666

    bundle_contract = subject.build_private_bundle_contract_v4_1(
        expected_artifacts=artifacts,
        **_screening_kwargs(evidence),
    )
    root = tmp_path.joinpath(*subject.PRIVATE_ROOT_SUFFIX)
    root.mkdir(parents=True)
    root.chmod(0o700)
    published = private_io.publish_private_bundle(
        private_root=root,
        run_id="resolved-proof-threading",
        artifacts=artifacts,
        contract=bundle_contract,
        revalidate_inputs=lambda: None,
    )
    independent = private_io.readback_private_bundle(
        published["bundle_path"], contract=bundle_contract
    )
    assert independent["accepted"] is True
    assert independent["artifacts"][subject.READBACK_FILENAME][
        "input_resolution_semantic_sha256"
    ] == _sha("input-resolution")

    drift = copy.deepcopy(correlation)
    drift["input_resolution_semantic_sha256"] = _sha("other-resolution")
    drift = _reseal(drift, "correlation_semantic_sha256")
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.validate_correlation_diagnostic_v4_1(
            drift,
            screening=screening,
            **_screening_kwargs(evidence),
        )


def test_compact_correlation_payload_scales_under_64_mib() -> None:
    axis = [f"2024-{month:02d}-28" for month in range(1, 13)] + [
        f"2025-{month:02d}-28" for month in range(1, 13)
    ] + [f"2026-{month:02d}-28" for month in range(1, 12 + 1)] + [
        f"2027-{month:02d}-28" for month in range(1, 11 + 1)
    ]
    assert len(axis) == 47
    pair_rows = [
        subject._seal(
            {
                "left_name": f"factor_{pair_index:04d}",
                "right_name": f"new_{pair_index % 27:03d}",
                "left_signal_sha256": _sha(f"left:{pair_index}"),
                "right_signal_sha256": _sha(f"right:{pair_index}"),
                "abs_spearman_by_month": [0.2] * len(axis),
                "valid_common_symbol_count_by_month": [100] * len(axis),
            },
            "row_semantic_sha256",
        )
        for pair_index in range(6561)
    ]
    payload = {
        "monthly_axis_source": "matrix_context.closed_month_end_dates",
        "closed_month_end_dates": axis,
        "monthly_pair_evidence": pair_rows,
    }
    assert len(subject.canonical_file_bytes_v4_1(payload)) < 64 * 1024 * 1024


def test_correlation_rejects_missing_duplicate_invalid_month_and_cycle(
    evidence: dict[str, Any],
) -> None:
    first_pair = (
        evidence["monthly_inputs"][0]["left_name"],
        evidence["monthly_inputs"][0]["right_name"],
    )
    missing_pair = [
        row
        for row in evidence["monthly_inputs"]
        if (row["left_name"], row["right_name"]) != first_pair
    ]
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.build_correlation_diagnostic_v4_1(
            cycle_id="cycle-synthetic-001",
            screening=evidence["screening"],
            monthly_rows=missing_pair,
            **_screening_kwargs(evidence),
        )
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.build_correlation_diagnostic_v4_1(
            cycle_id="cycle-synthetic-001",
            screening=evidence["screening"],
            monthly_rows=[*evidence["monthly_inputs"], evidence["monthly_inputs"][0]],
            **_screening_kwargs(evidence),
        )
    invalid_month = copy.deepcopy(evidence["monthly_inputs"])
    invalid_month[0]["month_end"] = "2026-03-30"
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.build_correlation_diagnostic_v4_1(
            cycle_id="cycle-synthetic-001",
            screening=evidence["screening"],
            monthly_rows=invalid_month,
            **_screening_kwargs(evidence),
        )
    with pytest.raises(
        subject.FactorGovernanceSameSnapshotScreeningV4_1Error,
        match="cycle_id mismatch",
    ):
        subject.build_correlation_diagnostic_v4_1(
            cycle_id="cycle-other",
            screening=evidence["screening"],
            monthly_rows=evidence["monthly_inputs"],
            **_screening_kwargs(evidence),
        )


def test_correlation_rejects_threshold_axes_and_shortlist_drift_after_reseal(
    evidence: dict[str, Any],
) -> None:
    threshold = copy.deepcopy(evidence["correlation"])
    threshold["correlation_contract"]["threshold"] = 0.9
    threshold = _reseal(threshold, "correlation_semantic_sha256")
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.validate_correlation_diagnostic_v4_1(
            threshold, screening=evidence["screening"], **_screening_kwargs(evidence)
        )

    axes = copy.deepcopy(evidence["correlation"])
    axes["matrix_context"]["eligibility_matrix_sha256"] = _sha("forged-mask")
    axes = _reseal(axes, "correlation_semantic_sha256")
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.validate_correlation_diagnostic_v4_1(
            axes, screening=evidence["screening"], **_screening_kwargs(evidence)
        )

    shortlist = copy.deepcopy(evidence["correlation"])
    shortlist["exploratory_new_candidate_shortlist"][0]["rank"] = 2
    shortlist["exploratory_new_candidate_shortlist"][0] = _reseal(
        shortlist["exploratory_new_candidate_shortlist"][0], "row_semantic_sha256"
    )
    shortlist = _reseal(shortlist, "correlation_semantic_sha256")
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.validate_correlation_diagnostic_v4_1(
            shortlist, screening=evidence["screening"], **_screening_kwargs(evidence)
        )


def test_owner_private_exact_once_bundle_readback(tmp_path: Path, evidence: dict[str, Any]) -> None:
    artifacts = {
        subject.SCREENING_FILENAME: evidence["screening"],
        subject.CORRELATION_FILENAME: evidence["correlation"],
    }
    contract = subject.build_private_bundle_contract_v4_1(
        expected_artifacts=artifacts, **_screening_kwargs(evidence)
    )
    root = tmp_path.joinpath(*subject.PRIVATE_ROOT_SUFFIX)
    root.mkdir(parents=True)
    root.chmod(0o700)
    published = private_io.publish_private_bundle(
        private_root=root,
        run_id="same-snapshot-test",
        artifacts=artifacts,
        contract=contract,
        revalidate_inputs=lambda: None,
    )
    bundle = Path(published["bundle_path"])
    assert stat.S_IMODE(bundle.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in bundle.iterdir())
    readback = private_io.readback_private_bundle(bundle, contract=contract)
    assert readback["accepted"] is True
    assert set(readback["artifacts"]) == set(subject.BUNDLE_FILENAMES)
    with pytest.raises(private_io.FactorGovernancePrivateBundleIOError):
        private_io.publish_private_bundle(
            private_root=root,
            run_id="same-snapshot-test",
            artifacts=artifacts,
            contract=contract,
            revalidate_inputs=lambda: None,
        )


def test_readback_report_rejects_resealed_authority_and_artifact_substitution(
    evidence: dict[str, Any],
) -> None:
    artifacts = {
        subject.SCREENING_FILENAME: evidence["screening"],
        subject.CORRELATION_FILENAME: evidence["correlation"],
    }
    bindings = []
    for filename in subject.BUNDLE_INPUT_FILENAMES:
        raw = subject.canonical_file_bytes_v4_1(artifacts[filename])
        bindings.append(
            {
                "filename": filename,
                "byte_sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "mode": 0o600,
                "uid": 501,
                "nlink": 1,
            }
        )
    report = subject.build_readback_report_v4_1(
        run_id="same-snapshot-test",
        artifacts=artifacts,
        artifact_bindings=bindings,
        **_screening_kwargs(evidence),
    )
    drift = copy.deepcopy(report)
    drift["authority"]["formal_dedup_authority"] = True
    drift = _reseal(drift, "report_semantic_sha256")
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.validate_readback_report_v4_1(
            drift, artifacts=artifacts, **_screening_kwargs(evidence)
        )
    substituted = copy.deepcopy(artifacts)
    substituted[subject.SCREENING_FILENAME]["cycle_id"] = "cycle-substituted"
    substituted[subject.SCREENING_FILENAME] = _reseal(
        substituted[subject.SCREENING_FILENAME], "screening_semantic_sha256"
    )
    with pytest.raises(subject.FactorGovernanceSameSnapshotScreeningV4_1Error):
        subject.validate_readback_report_v4_1(
            report, artifacts=substituted, **_screening_kwargs(evidence)
        )
