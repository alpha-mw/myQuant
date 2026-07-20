from __future__ import annotations

import copy
import hashlib
import math
import statistics
from datetime import date, timedelta
from pathlib import Path

import pytest
from scipy.stats import t as student_t

from quant_investor.factors.governance_canonical_replay_v4 import (
    ARM_NAMES,
    CONTROL_CHAIN_STAGES,
    CanonicalReplayV4Error,
    canonical_file_bytes,
    readback_v4_evidence,
    semantic_sha256,
    stage_byte_sha256,
    validate_canonical_replay_v4,
    validate_v4_evidence,
)
from quant_investor.factors.runtime import production_factor_set_sha256


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _factor_record(
    name: str,
    *,
    challenger: str,
    incumbent: str | None,
    comparison_family: str,
    comparison_slot: str,
    occupy_add_slot: bool,
) -> dict:
    index = int(name.split("_")[-1]) if name.startswith("factor_") else 0
    if name == challenger or name == incumbent or (occupy_add_slot and name == "factor_0"):
        family = comparison_family
        slot = comparison_slot
    else:
        family = f"family_{index}"
        slot = f"family_{index}::slot"
    return {
        "name": name,
        "family": family,
        "slot": slot,
        "state": "production_candidate" if name == challenger else "production_factor",
        "registry_record_sha256": _digest(f"record:{name}"),
    }


def _arm(gross_return: float, *, turnover: float, cost_rate: float = 0.001) -> dict:
    cost_return = turnover * cost_rate
    return {
        "gross_return": gross_return,
        "turnover": turnover,
        "cost_rate": cost_rate,
        "cost_return": cost_return,
        "net_return": gross_return - cost_return,
    }


def _business_dates(count: int) -> list[str]:
    result: list[str] = []
    current = date(2024, 1, 1)
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def _cohorts(
    *,
    count: int,
    calendar_sha256: str,
    calendar_open_session_dates: list[str],
    pit_sha256: str,
    delta_shift: float,
) -> list[dict]:
    universe_sha256 = _digest("full-a-universe")
    result: list[dict] = []
    for index in range(count):
        first_index = index * 35
        open_session_dates = calendar_open_session_dates[
            first_index : first_index + 30
        ]
        incumbent_gross = 0.01 + index * 0.0003
        challenger_gross = incumbent_gross + delta_shift + (index - 3.5) * 0.0002
        result.append(
            {
                "cohort_id": f"cohort-{index:02d}",
                "open_session_dates": open_session_dates,
                "universe_sha256": universe_sha256,
                "calendar_sha256": calendar_sha256,
                "pit_sha256": pit_sha256,
                "arms": {
                    "A": _arm(incumbent_gross, turnover=0.10),
                    "B": _arm(incumbent_gross - 0.001, turnover=0.08),
                    "C": _arm(challenger_gross - 0.0005, turnover=0.11),
                    "D": _arm(challenger_gross, turnover=0.10),
                },
            }
        )
    return result


def _comparison_statistics(cohorts: list[dict]) -> dict:
    deltas = [
        cohort["arms"]["D"]["net_return"]
        - cohort["arms"]["A"]["net_return"]
        for cohort in cohorts
    ]
    n = len(deltas)
    mean = statistics.fmean(deltas)
    stddev = statistics.stdev(deltas)
    standard_error = stddev / math.sqrt(n)
    critical_value = float(student_t.ppf(0.975, n - 1))
    return {
        "n": n,
        "mean": mean,
        "stddev": stddev,
        "standard_error": standard_error,
        "critical_value": critical_value,
        "method": "student_t",
        "sidedness": "two_sided",
        "incremental_edge_ci95_lower": mean - critical_value * standard_error,
    }


def _replay(
    *,
    transition_mode: str = "replace",
    factor_count: int | None = None,
    delta_shift: float = 0.01,
    cohort_count: int = 8,
    occupy_add_slot: bool = False,
) -> dict:
    if factor_count is None:
        factor_count = 10 if transition_mode == "replace" else 9
    factor_set = [f"factor_{index}" for index in range(factor_count)]
    incumbent = "factor_0" if transition_mode == "replace" else None
    challenger = "challenger"
    comparison_family = "family_0" if transition_mode == "replace" else "candidate_family"
    comparison_slot = (
        "family_0::slot_0" if transition_mode == "replace" else "candidate_family::slot"
    )
    registry_sha = _digest("registry")
    factor_set_sha = production_factor_set_sha256(factor_set)
    latest_pointer_sha256 = _digest("latest-pointer")
    manifest_sha256 = _digest("manifest")
    calendar_open_session_dates = _business_dates(320)
    calendar = {
        "schema_version": "factor-governance-open-session-calendar.v4",
        "market": "CN",
        "source": "strict_parquet_observed_trade_dates",
        "latest_pointer_sha256": latest_pointer_sha256,
        "manifest_sha256": manifest_sha256,
        "open_session_dates": calendar_open_session_dates,
    }
    calendar_sha256 = semantic_sha256(calendar)
    pit_sha256 = _digest("pit")
    cohorts = _cohorts(
        count=cohort_count,
        calendar_sha256=calendar_sha256,
        calendar_open_session_dates=calendar_open_session_dates,
        pit_sha256=pit_sha256,
        delta_shift=delta_shift,
    )
    comparison = {
        "transition_mode": transition_mode,
        "incumbent": incumbent,
        "challenger": challenger,
        "slot": comparison_slot,
        "cohorts": cohorts,
        **_comparison_statistics(cohorts),
    }
    quantitative_evidence_sha256 = semantic_sha256(
        {
            key: comparison[key]
            for key in ("transition_mode", "incumbent", "challenger", "slot", "cohorts")
        }
    )
    context = {
        "eligibility_contract_sha256": _digest("eligibility-contract"),
        "calendar_sha256": calendar_sha256,
        "pit_sha256": pit_sha256,
        "runtime_contract_sha256": _digest("runtime-contract"),
        "latest_pointer_sha256": latest_pointer_sha256,
        "manifest_sha256": manifest_sha256,
        "market_data_input_sha256": _digest("market-data-input"),
        "candidate_catalog_sha256": _digest("candidate-catalog"),
        "screening_evidence_sha256": _digest("screening-evidence"),
        "dedup_evidence_sha256": _digest("dedup-evidence"),
        "quantitative_evidence_sha256": quantitative_evidence_sha256,
    }
    context_sha = semantic_sha256(
        {
            "registry_file_sha256": registry_sha,
            "production_factor_set_sha256": factor_set_sha,
            **context,
        }
    )
    if transition_mode == "replace":
        without_incumbent = [name for name in factor_set if name != incumbent]
        factor_sets = {
            "A": factor_set,
            "B": without_incumbent,
            "C": sorted([*without_incumbent, challenger]),
            "D": sorted([*without_incumbent, challenger]),
        }
    else:
        with_challenger = sorted([*factor_set, challenger])
        factor_sets = {
            "A": factor_set,
            "B": factor_set,
            "C": with_challenger,
            "D": with_challenger,
        }
    stages: list[dict] = []
    for arm in ARM_NAMES:
        predecessor = {
            "kind": "genesis",
            "byte_sha256": "0" * 64,
            "semantic_sha256": "0" * 64,
        }
        semantic_by_stage: dict[str, str] = {}
        ic_output_hashes: dict[str, str] = {}
        for stage in CONTROL_CHAIN_STAGES:
            if stage == "eligibility":
                output = {
                    "schema_version": "factor-governance-eligibility-output.v4",
                    "eligible_symbols": ["AAA"],
                    "eligibility_contract_sha256": context["eligibility_contract_sha256"],
                }
            elif stage == "quant":
                output = {
                    "schema_version": "factor-governance-quant-output.v4",
                    "scored_symbols": ["AAA"],
                    "selected_factors": factor_sets[arm],
                    "factor_records": [
                        _factor_record(
                            name,
                            challenger=challenger,
                            incumbent=incumbent,
                            comparison_family=comparison_family,
                            comparison_slot=comparison_slot,
                            occupy_add_slot=occupy_add_slot,
                        )
                        for name in factor_sets[arm]
                    ],
                }
            elif stage == "funnel":
                output = {
                    "schema_version": "factor-governance-funnel-output.v4",
                    "eligible_symbols": ["AAA"],
                }
            elif stage == "codex_s1":
                output = {
                    "schema_version": "factor-governance-codex-s1-output.v4",
                    "advisory_scores": {"AAA": 0.6},
                }
            elif stage == "bayesian":
                output = {
                    "schema_version": "factor-governance-bayesian-output.v4",
                    "posterior_scores": {"AAA": 0.7},
                    "codex_s1_semantic_sha256": semantic_by_stage["codex_s1"],
                }
            elif stage == "risk_advisor":
                output = {
                    "schema_version": "factor-governance-risk-advisor-output.v4",
                    "advisory_only": True,
                    "decisions": {"AAA": "reject"},
                    "bayesian_semantic_sha256": semantic_by_stage["bayesian"],
                }
            elif stage == "codex_ic":
                ic_input = {
                    "symbol": "AAA",
                    "upstream_stage_sha256s": {
                        name: semantic_by_stage[name]
                        for name in CONTROL_CHAIN_STAGES[: CONTROL_CHAIN_STAGES.index("codex_ic")]
                    },
                    "ic_hints": {},
                }
                decision = {
                    "schema_version": "codex-ic-decision.v4",
                    "symbol": "AAA",
                    "action": "buy",
                }
                ic_output_hashes = {"AAA": semantic_sha256(decision)}
                output = {
                    "schema_version": "factor-governance-codex-ic-output.v4",
                    "inputs": {"AAA": ic_input},
                    "input_sha256s": {"AAA": semantic_sha256(ic_input)},
                    "decisions": {"AAA": decision},
                    "output_sha256s": ic_output_hashes,
                }
            else:
                output = {
                    "schema_version": "factor-governance-portfolio-output.v4",
                    "target_weights": {"AAA": 0.5},
                    "codex_ic_decision_sha256s": ic_output_hashes,
                }
            semantic_sha = semantic_sha256(output)
            byte_sha = stage_byte_sha256(
                arm=arm,
                stage=stage,
                context_sha256=context_sha,
                predecessor=predecessor,
                output=output,
            )
            stages.append(
                {
                    "schema_version": "factor-governance-canonical-stage.v4",
                    "arm": arm,
                    "stage": stage,
                    "context_sha256": context_sha,
                    "byte_sha256": byte_sha,
                    "semantic_sha256": semantic_sha,
                    "predecessor": predecessor,
                    "output": output,
                }
            )
            semantic_by_stage[stage] = semantic_sha
            predecessor = {
                "kind": "stage",
                "byte_sha256": byte_sha,
                "semantic_sha256": semantic_sha,
            }
    return {
        "schema_version": "factor-governance-canonical-replay.v4",
        "protocol_version": "v4",
        "run_id": "v4-replay-test",
        "as_of": "2026-07-17",
        "registry_file_sha256": registry_sha,
        "production_factor_set_sha256": factor_set_sha,
        "context": context,
        "context_sha256": context_sha,
        "calendar": calendar,
        "factor_set": factor_set,
        "comparison": comparison,
        "stages": stages,
    }


def _evidence(replay: dict, path: str) -> dict:
    raw = canonical_file_bytes(replay)
    return {
        "schema_version": "factor-governance-replay-evidence.v4",
        "status": "verified",
        "factor_name": "challenger",
        "registry_file_sha256": replay["registry_file_sha256"],
        "replay_path": path,
        "replay_file_sha256": hashlib.sha256(raw).hexdigest(),
        "replay_semantic_sha256": semantic_sha256(replay),
        **replay["context"],
        "replay": replay,
    }


def test_v4_replay_binds_exact_chain_and_risk_advisor_is_not_positive_weight_gate() -> None:
    replay = _replay()
    normalized = validate_canonical_replay_v4(replay)
    assert normalized["comparison"]["transition_mode"] == "replace"
    assert normalized["comparison"]["n"] == 8
    assert normalized["comparison"]["method"] == "student_t"
    assert normalized["comparison"]["sidedness"] == "two_sided"
    assert normalized["calendar"]["source"] == "strict_parquet_observed_trade_dates"
    assert all(
        date.fromisoformat(session).weekday() < 5
        for session in normalized["calendar"]["open_session_dates"]
    )
    assert normalized["positive_weight_depends_on_risk_advisor_approval"] is False
    assert all(
        normalized["arms"][arm]["risk_advisor"]["decisions"]["AAA"] == "reject" for arm in ARM_NAMES
    )
    assert all(
        normalized["arms"][arm]["portfolio_constructor"]["target_weights"]["AAA"] == 0.5
        for arm in ARM_NAMES
    )


def test_v4_target_10_replacement_requires_positive_incremental_edge_lower_bound() -> None:
    with pytest.raises(CanonicalReplayV4Error, match="incremental edge"):
        validate_canonical_replay_v4(_replay(delta_shift=-0.01))


def test_v4_underfilled_add_records_nonpositive_edge_without_using_it_as_a_gate() -> None:
    normalized = validate_canonical_replay_v4(
        _replay(transition_mode="add", delta_shift=-0.01)
    )
    comparison = normalized["comparison"]
    assert comparison["transition_mode"] == "add"
    assert comparison["incumbent"] is None
    assert comparison["incremental_edge_ci95_lower"] < 0.0
    assert normalized["arms"]["A"]["quant"]["selected_factors"] == normalized["factor_set"]
    assert normalized["arms"]["B"]["quant"]["selected_factors"] == normalized["factor_set"]
    expected = sorted([*normalized["factor_set"], comparison["challenger"]])
    assert normalized["arms"]["C"]["quant"]["selected_factors"] == expected
    assert normalized["arms"]["D"]["quant"]["selected_factors"] == expected


def test_v4_transition_modes_enforce_size_incumbent_and_slot_rules() -> None:
    with pytest.raises(CanonicalReplayV4Error, match="exactly 10"):
        validate_canonical_replay_v4(_replay(transition_mode="replace", factor_count=9))
    with pytest.raises(CanonicalReplayV4Error, match="underfilled"):
        validate_canonical_replay_v4(_replay(transition_mode="add", factor_count=10))

    non_null_incumbent = _replay(transition_mode="add")
    non_null_incumbent["comparison"]["incumbent"] = "factor_0"
    with pytest.raises(CanonicalReplayV4Error, match="incumbent must be null"):
        validate_canonical_replay_v4(non_null_incumbent)

    with pytest.raises(CanonicalReplayV4Error, match="empty slot"):
        validate_canonical_replay_v4(
            _replay(transition_mode="add", occupy_add_slot=True)
        )


def test_v4_paired_cohorts_recompute_cost_net_delta_and_student_t_statistics() -> None:
    replay = _replay()
    expected = _comparison_statistics(replay["comparison"]["cohorts"])
    normalized = validate_canonical_replay_v4(replay)
    for field in (
        "n",
        "mean",
        "stddev",
        "standard_error",
        "critical_value",
        "incremental_edge_ci95_lower",
    ):
        assert normalized["comparison"][field] == pytest.approx(expected[field], abs=1e-15)


def test_v4_paired_cohorts_require_eight_nonoverlapping_sorted_30_session_windows() -> None:
    with pytest.raises(CanonicalReplayV4Error, match="at least 8"):
        validate_canonical_replay_v4(_replay(cohort_count=7))

    overlapping = _replay()
    overlapping["comparison"]["cohorts"][1]["open_session_dates"] = copy.deepcopy(
        overlapping["comparison"]["cohorts"][0]["open_session_dates"]
    )
    with pytest.raises(CanonicalReplayV4Error, match="must not overlap"):
        validate_canonical_replay_v4(overlapping)

    short = _replay()
    short["comparison"]["cohorts"][0]["open_session_dates"].pop()
    with pytest.raises(CanonicalReplayV4Error, match="exactly 30"):
        validate_canonical_replay_v4(short)

    unsorted = _replay()
    dates = unsorted["comparison"]["cohorts"][0]["open_session_dates"]
    dates[0], dates[1] = dates[1], dates[0]
    with pytest.raises(CanonicalReplayV4Error, match="sorted and distinct"):
        validate_canonical_replay_v4(unsorted)


@pytest.mark.parametrize("field", ["cost_return", "net_return"])
def test_v4_paired_cohorts_reject_cost_or_net_recomputation_drift(field: str) -> None:
    replay = _replay()
    replay["comparison"]["cohorts"][0]["arms"]["D"][field] += 0.001
    with pytest.raises(CanonicalReplayV4Error, match=rf"{field} recomputation mismatch"):
        validate_canonical_replay_v4(replay)


@pytest.mark.parametrize(
    "field",
    [
        "n",
        "mean",
        "stddev",
        "standard_error",
        "critical_value",
        "incremental_edge_ci95_lower",
    ],
)
def test_v4_comparison_declared_statistics_must_match_recomputation(field: str) -> None:
    replay = _replay()
    replay["comparison"][field] += 1 if field == "n" else 0.001
    with pytest.raises(CanonicalReplayV4Error, match=rf"{field} recomputation mismatch"):
        validate_canonical_replay_v4(replay)


def test_v4_quantitative_hash_and_cohort_source_bindings_are_fail_closed() -> None:
    replay = _replay()
    replay["context"]["quantitative_evidence_sha256"] = _digest("forged-quantitative")
    replay["context_sha256"] = semantic_sha256(
        {
            "registry_file_sha256": replay["registry_file_sha256"],
            "production_factor_set_sha256": replay["production_factor_set_sha256"],
            **replay["context"],
        }
    )
    with pytest.raises(CanonicalReplayV4Error, match="quantitative evidence semantic SHA"):
        validate_canonical_replay_v4(replay)

    calendar_drift = _replay()
    calendar_drift["comparison"]["cohorts"][0]["calendar_sha256"] = _digest(
        "other-calendar"
    )
    with pytest.raises(CanonicalReplayV4Error, match="calendar SHA binding"):
        validate_canonical_replay_v4(calendar_drift)

    universe_drift = _replay()
    universe_drift["comparison"]["cohorts"][0]["universe_sha256"] = _digest(
        "other-universe"
    )
    with pytest.raises(CanonicalReplayV4Error, match="universe SHA binding"):
        validate_canonical_replay_v4(universe_drift)


def test_v4_calendar_rejects_weekends_noncalendar_dates_and_nonconsecutive_sessions() -> None:
    weekend = _replay()
    weekend["calendar"]["open_session_dates"][0] = "2024-01-06"
    with pytest.raises(CanonicalReplayV4Error, match="weekend"):
        validate_canonical_replay_v4(weekend)

    outside_calendar = _replay()
    outside_calendar["comparison"]["cohorts"][0]["open_session_dates"][0] = (
        "2023-12-29"
    )
    with pytest.raises(CanonicalReplayV4Error, match="exist in the bound calendar"):
        validate_canonical_replay_v4(outside_calendar)

    nonconsecutive = _replay()
    calendar_dates = nonconsecutive["calendar"]["open_session_dates"]
    nonconsecutive["comparison"]["cohorts"][0]["open_session_dates"] = [
        *calendar_dates[:15],
        *calendar_dates[16:31],
    ]
    with pytest.raises(CanonicalReplayV4Error, match="30 consecutive calendar sessions"):
        validate_canonical_replay_v4(nonconsecutive)


def test_v4_calendar_hash_source_and_context_bindings_are_fail_closed() -> None:
    hash_drift = _replay()
    hash_drift["context"]["calendar_sha256"] = _digest("forged-calendar")
    hash_drift["context_sha256"] = semantic_sha256(
        {
            "registry_file_sha256": hash_drift["registry_file_sha256"],
            "production_factor_set_sha256": hash_drift[
                "production_factor_set_sha256"
            ],
            **hash_drift["context"],
        }
    )
    with pytest.raises(CanonicalReplayV4Error, match="calendar SHA"):
        validate_canonical_replay_v4(hash_drift)

    source_drift = _replay()
    source_drift["calendar"]["source"] = "inferred_weekdays"
    with pytest.raises(CanonicalReplayV4Error, match="calendar source"):
        validate_canonical_replay_v4(source_drift)

    for field, expected in (
        ("latest_pointer_sha256", "pointer SHA"),
        ("manifest_sha256", "manifest SHA"),
    ):
        binding_drift = _replay()
        binding_drift["calendar"][field] = _digest(f"forged:{field}")
        binding_drift["context"]["calendar_sha256"] = semantic_sha256(
            binding_drift["calendar"]
        )
        binding_drift["context_sha256"] = semantic_sha256(
            {
                "registry_file_sha256": binding_drift["registry_file_sha256"],
                "production_factor_set_sha256": binding_drift[
                    "production_factor_set_sha256"
                ],
                **binding_drift["context"],
            }
        )
        with pytest.raises(CanonicalReplayV4Error, match=expected):
            validate_canonical_replay_v4(binding_drift)


def test_v4_portfolio_positive_weight_still_requires_hash_bound_codex_ic_buy() -> None:
    replay = _replay()
    ic_index = next(
        index
        for index, row in enumerate(replay["stages"])
        if row["arm"] == "A" and row["stage"] == "codex_ic"
    )
    ic_stage = replay["stages"][ic_index]
    portfolio_stage = replay["stages"][ic_index + 1]
    decision = ic_stage["output"]["decisions"]["AAA"]
    decision["action"] = "hold"
    decision_sha = semantic_sha256(decision)
    ic_stage["output"]["output_sha256s"]["AAA"] = decision_sha
    portfolio_stage["output"]["codex_ic_decision_sha256s"]["AAA"] = decision_sha
    for index in (ic_index, ic_index + 1):
        row = replay["stages"][index]
        if index > ic_index:
            previous = replay["stages"][index - 1]
            row["predecessor"] = {
                "kind": "stage",
                "byte_sha256": previous["byte_sha256"],
                "semantic_sha256": previous["semantic_sha256"],
            }
        row["semantic_sha256"] = semantic_sha256(row["output"])
        row["byte_sha256"] = stage_byte_sha256(
            arm=row["arm"],
            stage=row["stage"],
            context_sha256=row["context_sha256"],
            predecessor=row["predecessor"],
            output=row["output"],
        )
    with pytest.raises(CanonicalReplayV4Error, match="CodexIC BUY"):
        validate_canonical_replay_v4(replay)


def test_v4_evidence_rejects_v2_v3_and_supports_exact_0600_readback(
    tmp_path: Path,
) -> None:
    for version in ("v2", "v3"):
        with pytest.raises(CanonicalReplayV4Error, match="unsupported"):
            validate_v4_evidence(
                {"schema_version": f"factor-governance-replay-evidence.{version}"}
            )
        with pytest.raises(CanonicalReplayV4Error, match="unsupported"):
            validate_canonical_replay_v4(
                {"schema_version": f"factor-governance-canonical-replay.{version}"}
            )
    replay = _replay()
    path = (tmp_path / "canonical-replay.v4.json").resolve()
    path.write_bytes(canonical_file_bytes(replay))
    path.chmod(0o600)
    evidence = _evidence(replay, str(path))
    assert validate_v4_evidence(evidence)["factor_name"] == "challenger"
    readback = readback_v4_evidence(evidence)
    assert readback["complete_chain_hash_binding_verified"] is True
    assert readback["context_bindings_readback_verified"] is True
    assert readback["quantitative_evidence_hash_binding_verified"] is True
    assert readback["positive_weight_depends_on_risk_advisor_approval"] is False

    forged = copy.deepcopy(evidence)
    forged["replay_semantic_sha256"] = _digest("forged")
    with pytest.raises(CanonicalReplayV4Error, match="semantic SHA"):
        validate_v4_evidence(forged)


@pytest.mark.parametrize(
    "field",
    [
        "eligibility_contract_sha256",
        "calendar_sha256",
        "pit_sha256",
        "runtime_contract_sha256",
        "latest_pointer_sha256",
        "manifest_sha256",
        "market_data_input_sha256",
        "candidate_catalog_sha256",
        "screening_evidence_sha256",
        "dedup_evidence_sha256",
        "quantitative_evidence_sha256",
    ],
)
def test_v4_evidence_envelope_binds_all_replay_context_hashes(field: str) -> None:
    replay = _replay()
    evidence = _evidence(replay, "/tmp/not-read-for-envelope-validation.json")
    evidence[field] = _digest(f"forged:{field}")
    with pytest.raises(CanonicalReplayV4Error, match=rf"{field} mismatch"):
        validate_v4_evidence(evidence)
