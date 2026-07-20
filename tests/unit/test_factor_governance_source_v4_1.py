from __future__ import annotations

import copy
import hashlib
from datetime import date, timedelta

import pytest

from quant_investor.factors.governance_source_v4_1 import (
    CUTOFF_SCOPE,
    DESIGN_CUTOFF_DATE,
    DESIGN_HISTORY_SCOPE,
    GENESIS_SHA256,
    HOLDOUT_SCOPE,
    FactorGovernanceSourceV41Error,
    append_holdout_source_node_v4_1,
    assess_holdout_calendar_readiness_v4_1,
    build_design_source_node_v4_1,
    build_session_scope_descriptor_v4_1,
    byte_sha256,
    validate_calendar_prefix_v4_1,
    validate_design_source_node_v4_1,
    validate_holdout_source_node_v4_1,
    validate_pit_records_v4_1,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _set_digest(symbols: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(symbols)).encode("ascii")).hexdigest()


def _row(
    symbol: str,
    status: str,
    effective_from: str,
    effective_to: str | None = None,
    *,
    quality: str = "ok",
) -> dict:
    return {
        "schema_version": "cn_pit_universe.v1",
        "symbol": symbol,
        "source_list_status": status,
        "effective_from": effective_from,
        "effective_to": effective_to or "",
        "list_date": effective_from,
        "delist_date": effective_to or "",
        "membership_quality": quality,
    }


def _records() -> list[dict]:
    return [
        _row("000001.SZ", "L", "2026-07-01"),
        _row("000002.SZ", "D", "2026-07-01", "2026-07-20"),
        _row("000003.SZ", "P", DESIGN_CUTOFF_DATE),
        _row("000004.BJ", "L", "2026-07-01"),
        _row("T600018.SH", "D", "2000-07-19", "2006-10-20"),
        _row("TS0018.SH", "D", "2000-07-19", "2006-10-20"),
    ]


def _components() -> list[str]:
    return ["000001.SZ", "000002.SZ", "000003.SZ"]


def _design_calendar() -> list[str]:
    return ["2026-07-15", "2026-07-16", DESIGN_CUTOFF_DATE]


def _business_after(start: str, count: int) -> list[str]:
    current = date.fromisoformat(start) + timedelta(days=1)
    result: list[str] = []
    while len(result) < count:
        if current.weekday() < 5:
            result.append(current.isoformat())
        current += timedelta(days=1)
    return result


def _actual_month_ends(calendar: list[str]) -> list[str]:
    by_month: dict[str, list[str]] = {}
    for session in calendar:
        by_month.setdefault(session[:7], []).append(session)
    latest_month = calendar[-1][:7]
    return [
        by_month[month][-1]
        for month in sorted(by_month)
        if month < latest_month and by_month[month][-1] > DESIGN_CUTOFF_DATE
    ]


def _design() -> dict:
    return build_design_source_node_v4_1(
        cycle_id="cycle-source-v41",
        pit_records=_records(),
        component_symbols=_components(),
        calendar_sessions=_design_calendar(),
        market_binding_sha256=_digest("design-table"),
        source_binding_sha256=_digest("design-source"),
        expected_component_count=3,
    )


def _cas(previous: dict | None) -> dict[str, str]:
    if previous is None:
        return {
            "predecessor_byte_sha256": GENESIS_SHA256,
            "expected_predecessor_byte_sha256": GENESIS_SHA256,
            "expected_predecessor_semantic_sha256": GENESIS_SHA256,
        }
    observed = byte_sha256(previous)
    return {
        "predecessor_byte_sha256": observed,
        "expected_predecessor_byte_sha256": observed,
        "expected_predecessor_semantic_sha256": previous["semantic_sha256"],
    }


def _append(
    appended: list[str],
    *,
    previous: dict | None = None,
    components: list[str] | None = None,
    inventory: int = 5728,
    node_id: str = "holdout-001",
) -> dict:
    current = previous
    for index, session in enumerate(appended):
        step_id = node_id if len(appended) == 1 else f"{node_id}-{index:03d}"
        prior_calendar = (
            current["cumulative_calendar_sessions"]
            if current is not None
            else _design_calendar()
        )
        current = append_holdout_source_node_v4_1(
            design_node=_design(),
            previous_node=current,
            design_pit_records=_records(),
            node_pit_records=_records(),
            component_symbols=components or _components(),
            appended_sessions=[session],
            node_id=step_id,
            observed_at="2026-07-18T01:02:03Z",
            serving_inventory_count=inventory,
            market_binding_sha256=_digest(f"table:{step_id}"),
            source_binding_sha256=_digest(f"source:{step_id}"),
            actual_month_end_sessions=_actual_month_ends([*prior_calendar, session]),
            expected_design_component_count=3,
            **_cas(current),
        )
    assert current is not None
    return current


def _ready_chain() -> tuple[dict, dict]:
    previous: dict | None = None
    for index, session in enumerate(_business_after(DESIGN_CUTOFF_DATE, 300)):
        current = _append(
            [session], previous=previous, node_id=f"ready-{index:03d}"
        )
        if current["ready"]:
            assert previous is not None
            return previous, current
        previous = current
    raise AssertionError("synthetic calendar never reached readiness")


def test_pit_half_open_boundaries_and_l_d_p_semantics() -> None:
    records = [
        _row("000001.SZ", "L", "2026-01-06"),
        _row("000002.SZ", "D", "2026-01-05", "2026-01-07"),
        _row("000003.SZ", "P", "2026-01-06"),
    ]
    components = ["000001.SZ", "000002.SZ", "000003.SZ"]

    before = build_session_scope_descriptor_v4_1(
        records, "2026-01-05", CUTOFF_SCOPE, components
    )
    listed = build_session_scope_descriptor_v4_1(
        records, "2026-01-06", CUTOFF_SCOPE, components
    )
    delisted = build_session_scope_descriptor_v4_1(
        records, "2026-01-07", CUTOFF_SCOPE, components
    )

    assert before["research_eligible_count"] == 1
    assert before["tradable_count"] == 1
    assert listed["research_eligible_count"] == 3
    assert listed["tradable_count"] == 2
    assert listed["research_eligible_symbols_newline_sha256"] == _set_digest(components)
    assert listed["tradable_symbols_newline_sha256"] == _set_digest(
        ["000001.SZ", "000002.SZ"]
    )
    assert delisted["research_eligible_count"] == 2
    assert delisted["tradable_count"] == 1


def test_historical_aliases_are_retained_hashed_and_reported() -> None:
    normalized = validate_pit_records_v4_1(_records())
    assert [row["symbol"] for row in normalized][-2:] == ["T600018.SH", "TS0018.SH"]

    design = _design()
    report = design["out_of_bound_calendar_nonparticipating"]
    assert report["count"] == 2
    assert [row["symbol"] for row in report["records"]] == [
        "T600018.SH",
        "TS0018.SH",
    ]
    assert all(row["active_bound_session_count"] == 0 for row in report["records"])
    assert design["pit_record_count"] == 6


def test_historical_alias_overlapping_bound_session_blocks() -> None:
    records = _records()
    records[-1] = _row("TS0018.SH", "L", "2026-01-01")
    with pytest.raises(FactorGovernanceSourceV41Error, match="alias overlaps"):
        build_session_scope_descriptor_v4_1(
            records, DESIGN_CUTOFF_DATE, DESIGN_HISTORY_SCOPE
        )


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda rows: rows[0].pop("effective_from"), "effective_from"),
        (lambda rows: rows[0].update(effective_from="bad", list_date="bad"), "invalid"),
        (
            lambda rows: rows[0].update(
                effective_to="2026-01-01", delist_date="2026-01-01"
            ),
            "interval order",
        ),
        (lambda rows: rows[0].update(source_list_status="X"), "unsupported"),
        (lambda rows: rows[0].update(membership_quality="conflict"), "not ok"),
        (lambda rows: rows[0].update(symbol="bad"), "symbol"),
        (lambda rows: rows[0].update(schema_version="wrong"), "source schema"),
        (
            lambda rows: rows[0].update(list_date="2026-01-02"),
            "list/effective_from conflict",
        ),
    ],
)
def test_bad_pit_row_blocks_whole_source(mutator, match: str) -> None:
    rows = [_row("000001.SZ", "L", "2026-01-01")]
    mutator(rows)
    with pytest.raises(FactorGovernanceSourceV41Error, match=match):
        validate_pit_records_v4_1(rows)


def test_duplicate_and_conflicting_status_rows_block() -> None:
    duplicate = [
        _row("000001.SZ", "L", "2026-01-01"),
        _row("000001.SZ", "P", "2026-01-02"),
    ]
    with pytest.raises(FactorGovernanceSourceV41Error, match="duplicate"):
        validate_pit_records_v4_1(duplicate)

    conflict = _row("000001.SZ", "L", "2026-01-01")
    conflict["status"] = "P"
    with pytest.raises(FactorGovernanceSourceV41Error, match="conflicting"):
        validate_pit_records_v4_1([conflict])


def test_delisted_row_without_end_blocks() -> None:
    with pytest.raises(FactorGovernanceSourceV41Error, match="no effective_to"):
        validate_pit_records_v4_1([_row("000001.SZ", "D", "2026-01-01")])


def test_component_scope_requires_exact_sorted_contained_six_digit_symbols() -> None:
    with pytest.raises(FactorGovernanceSourceV41Error, match="missing from PIT"):
        build_session_scope_descriptor_v4_1(
            _records(), DESIGN_CUTOFF_DATE, CUTOFF_SCOPE, ["999999.SH"]
        )
    with pytest.raises(FactorGovernanceSourceV41Error, match="sorted and distinct"):
        build_session_scope_descriptor_v4_1(
            _records(), DESIGN_CUTOFF_DATE, CUTOFF_SCOPE, list(reversed(_components()))
        )
    with pytest.raises(FactorGovernanceSourceV41Error, match="component symbol"):
        build_session_scope_descriptor_v4_1(
            _records(), DESIGN_CUTOFF_DATE, CUTOFF_SCOPE, ["T600018.SH"]
        )


def test_design_history_is_full_pit_not_retrofiltered_by_cutoff_components() -> None:
    design = _design()
    history, _, cutoff = design["session_scope_descriptors"]
    assert history["scope_kind"] == DESIGN_HISTORY_SCOPE
    assert history["component_symbols_semantic_sha256"] is None
    assert history["tradable_count"] == 3  # includes non-component 000004.BJ
    assert cutoff["scope_kind"] == CUTOFF_SCOPE
    assert cutoff["component_symbols_semantic_sha256"] == _set_digest(_components())
    assert cutoff["tradable_count"] == 2
    assert cutoff["research_eligible_count"] == 3


def test_descriptor_is_canonical_under_pit_row_order() -> None:
    forward = build_session_scope_descriptor_v4_1(
        _records(), DESIGN_CUTOFF_DATE, CUTOFF_SCOPE, _components()
    )
    reverse = build_session_scope_descriptor_v4_1(
        list(reversed(_records())), DESIGN_CUTOFF_DATE, CUTOFF_SCOPE, _components()
    )
    assert forward == reverse


def test_calendar_prefix_rejects_insert_delete_reclassification_and_empty_append() -> None:
    previous = _design_calendar()
    appended = ["2026-07-20", "2026-07-21"]
    assert validate_calendar_prefix_v4_1(
        previous, previous + appended, appended
    ) == previous + appended
    with pytest.raises(FactorGovernanceSourceV41Error, match="prefix"):
        validate_calendar_prefix_v4_1(
            previous, [previous[0], previous[2], *appended], appended
        )
    with pytest.raises(FactorGovernanceSourceV41Error):
        validate_calendar_prefix_v4_1(previous, previous, [])
    with pytest.raises(FactorGovernanceSourceV41Error):
        validate_calendar_prefix_v4_1(previous, previous + ["2026-07-16"], ["2026-07-16"])


def test_holdout_node_rejects_multi_session_component_retrofilter() -> None:
    sessions = _business_after(DESIGN_CUTOFF_DATE, 2)
    with pytest.raises(FactorGovernanceSourceV41Error, match="exactly one session"):
        append_holdout_source_node_v4_1(
            design_node=_design(),
            previous_node=None,
            design_pit_records=_records(),
            node_pit_records=_records(),
            component_symbols=_components(),
            appended_sessions=sessions,
            node_id="bad-batch",
            observed_at="2026-07-18T01:02:03Z",
            serving_inventory_count=5728,
            market_binding_sha256=_digest("bad-batch-table"),
            source_binding_sha256=_digest("bad-batch-source"),
            actual_month_end_sessions=[],
            expected_design_component_count=3,
            **_cas(None),
        )


def test_embargo_off_by_one_and_actual_month_close() -> None:
    design = _design_calendar()
    thirty = _business_after(DESIGN_CUTOFF_DATE, 30)
    at_30_calendar = design + thirty
    at_30 = assess_holdout_calendar_readiness_v4_1(
        design, at_30_calendar, _actual_month_ends(at_30_calendar)
    )
    at_31_calendar = design + _business_after(DESIGN_CUTOFF_DATE, 31)
    at_31 = assess_holdout_calendar_readiness_v4_1(
        design, at_31_calendar, _actual_month_ends(at_31_calendar)
    )
    assert at_30["embargo_session_count"] == 30
    assert at_30["post_embargo_session_count"] == 0
    assert at_31["post_embargo_session_count"] == 1

    sparse = design + thirty + ["2026-09-30", "2026-10-01"]
    closed = assess_holdout_calendar_readiness_v4_1(
        design, sparse, ["2026-09-30"]
    )
    assert closed["closed_post_embargo_month_end_dates"] == ["2026-09-30"]


def test_240_post_embargo_sessions_without_12_closed_months_is_not_ready() -> None:
    sessions = _business_after(DESIGN_CUTOFF_DATE, 30 + 240)
    cumulative = _design_calendar() + sessions
    result = assess_holdout_calendar_readiness_v4_1(
        _design_calendar(), cumulative, _actual_month_ends(cumulative)
    )
    assert result["post_embargo_session_count"] == 240
    assert result["closed_post_embargo_month_end_count"] < 12
    assert result["ready"] is False


def test_ready_requires_both_thresholds() -> None:
    sessions = _business_after(DESIGN_CUTOFF_DATE, 300)
    cumulative = _design_calendar() + sessions
    result = assess_holdout_calendar_readiness_v4_1(
        _design_calendar(), cumulative, _actual_month_ends(cumulative)
    )
    assert result["post_embargo_session_count"] >= 240
    assert result["closed_post_embargo_month_end_count"] >= 12
    assert result["ready"] is True


def test_sparse_calendar_cannot_invent_actual_month_ends() -> None:
    sessions = _business_after(DESIGN_CUTOFF_DATE, 330)
    cumulative = _design_calendar() + sessions
    result = assess_holdout_calendar_readiness_v4_1(
        _design_calendar(), cumulative, []
    )
    assert result["post_embargo_session_count"] >= 240
    assert result["closed_post_embargo_month_end_count"] == 0
    assert result["ready"] is False

    october = [session for session in cumulative if session.startswith("2026-10")]
    with pytest.raises(FactorGovernanceSourceV41Error, match="last supplied"):
        assess_holdout_calendar_readiness_v4_1(
            _design_calendar(), cumulative, [october[-2]]
        )


def test_design_node_validates_and_default_production_count_is_5502() -> None:
    design = _design()
    assert validate_design_source_node_v4_1(
        design, pit_records=_records(), expected_component_count=3
    ) == design
    with pytest.raises(FactorGovernanceSourceV41Error, match="component count"):
        validate_design_source_node_v4_1(design, pit_records=_records())


def test_safe_append_allows_future_component_count_change_and_5728_inventory() -> None:
    first_sessions = _business_after(DESIGN_CUTOFF_DATE, 5)
    first = _append(first_sessions)
    future_components = ["000001.SZ", "000002.SZ", "000003.SZ", "000004.BJ"]
    next_sessions = _business_after(first_sessions[-1], 1)
    second = _append(
        next_sessions,
        previous=first,
        components=future_components,
        node_id="holdout-002",
    )
    assert first["serving_inventory_count"] == 5728
    assert first["serving_inventory_eligibility_prohibited"] is True
    assert second["component_count"] == 4
    assert second["session_scope_descriptors"][: len(first["session_scope_descriptors"])] == first[
        "session_scope_descriptors"
    ]
    assert validate_holdout_source_node_v4_1(
        second,
        design_node=_design(),
        previous_node=first,
        design_pit_records=_records(),
        node_pit_records=_records(),
        actual_month_end_sessions=second["actual_month_end_sessions"],
        expected_design_component_count=3,
        **_cas(first),
    ) == second


def test_future_pit_inventory_can_extend_without_recomputing_frozen_design() -> None:
    node_records = _records() + [_row("000005.SH", "L", "2026-07-20")]
    components = [*_components(), "000005.SH"]
    node = append_holdout_source_node_v4_1(
        design_node=_design(),
        previous_node=None,
        design_pit_records=_records(),
        node_pit_records=node_records,
        component_symbols=components,
        appended_sessions=["2026-07-20"],
        node_id="holdout-new-listing",
        observed_at="2026-07-20T09:00:00Z",
        serving_inventory_count=5728,
        market_binding_sha256=_digest("new-listing-table"),
        source_binding_sha256=_digest("new-listing-source"),
        actual_month_end_sessions=[],
        expected_design_component_count=3,
        **_cas(None),
    )
    assert node["pit_record_count"] == len(node_records)
    assert node["component_count"] == 4
    assert node["session_scope_descriptors"][-1]["research_eligible_count"] == 3


def test_non_genesis_bindings_reject_zero_sha() -> None:
    with pytest.raises(FactorGovernanceSourceV41Error, match="genesis SHA"):
        build_design_source_node_v4_1(
            cycle_id="cycle-zero-sha",
            pit_records=_records(),
            component_symbols=_components(),
            calendar_sessions=_design_calendar(),
            market_binding_sha256=GENESIS_SHA256,
            source_binding_sha256=_digest("source"),
            expected_component_count=3,
        )


def test_serving_inventory_cannot_become_eligibility_source() -> None:
    node = _append(_business_after(DESIGN_CUTOFF_DATE, 2))
    node["eligibility_source"] = "serving_inventory"
    with pytest.raises(FactorGovernanceSourceV41Error, match="eligibility source"):
        validate_holdout_source_node_v4_1(
            node,
            design_node=_design(),
            previous_node=None,
            design_pit_records=_records(),
            node_pit_records=_records(),
            actual_month_end_sessions=node["actual_month_end_sessions"],
            expected_design_component_count=3,
            **_cas(None),
        )


def test_prior_calendar_or_mapping_mutation_blocks() -> None:
    first = _append(_business_after(DESIGN_CUTOFF_DATE, 3))
    next_sessions = _business_after(first["cumulative_calendar_sessions"][-1], 2)
    second = _append(next_sessions, previous=first, node_id="holdout-002")

    calendar_drift = copy.deepcopy(second)
    calendar_drift["cumulative_calendar_sessions"][0] = "2026-07-14"
    with pytest.raises(FactorGovernanceSourceV41Error):
        validate_holdout_source_node_v4_1(
            calendar_drift,
            design_node=_design(),
            previous_node=first,
            design_pit_records=_records(),
            node_pit_records=_records(),
            actual_month_end_sessions=calendar_drift[
                "actual_month_end_sessions"
            ],
            expected_design_component_count=3,
            **_cas(first),
        )

    mapping_drift = copy.deepcopy(second)
    mapping_drift["session_scope_descriptors"][0]["tradable_count"] += 1
    with pytest.raises(FactorGovernanceSourceV41Error):
        validate_holdout_source_node_v4_1(
            mapping_drift,
            design_node=_design(),
            previous_node=first,
            design_pit_records=_records(),
            node_pit_records=_records(),
            actual_month_end_sessions=mapping_drift[
                "actual_month_end_sessions"
            ],
            expected_design_component_count=3,
            **_cas(first),
        )


def test_stale_predecessor_dual_cas_blocks() -> None:
    first = _append(_business_after(DESIGN_CUTOFF_DATE, 2))
    with pytest.raises(FactorGovernanceSourceV41Error, match="dual CAS"):
        append_holdout_source_node_v4_1(
            design_node=_design(),
            previous_node=first,
            design_pit_records=_records(),
            node_pit_records=_records(),
            component_symbols=_components(),
            appended_sessions=_business_after(first["cumulative_calendar_sessions"][-1], 1),
            node_id="holdout-002",
            observed_at="2026-07-18T01:02:04Z",
            serving_inventory_count=5728,
            market_binding_sha256=_digest("next-table"),
            source_binding_sha256=_digest("next-source"),
            actual_month_end_sessions=first["actual_month_end_sessions"],
            predecessor_byte_sha256=byte_sha256(first),
            expected_predecessor_byte_sha256=_digest("stale"),
            expected_predecessor_semantic_sha256=first["semantic_sha256"],
            expected_design_component_count=3,
        )


def test_cross_cycle_design_root_and_historical_source_drift_block() -> None:
    first = _append(_business_after(DESIGN_CUTOFF_DATE, 2))
    for key, value in (
        ("cycle_id", "other-cycle"),
        ("design_source_root_sha256", _digest("other-design")),
        ("historical_source_binding_sha256", _digest("other-source")),
    ):
        drift = copy.deepcopy(first)
        drift[key] = value
        with pytest.raises(FactorGovernanceSourceV41Error):
            validate_holdout_source_node_v4_1(
                drift,
                design_node=_design(),
                previous_node=None,
                design_pit_records=_records(),
                node_pit_records=_records(),
                actual_month_end_sessions=drift[
                    "actual_month_end_sessions"
                ],
                expected_design_component_count=3,
                **_cas(None),
            )


def test_ready_node_is_terminal_and_cannot_append() -> None:
    previous, ready = _ready_chain()
    assert ready["ready"] is True
    assert ready["terminal_holdout_source_root_sha256"] == ready["semantic_sha256"]
    unsealed = copy.deepcopy(ready)
    unsealed["terminal_holdout_source_root_sha256"] = None
    with pytest.raises(FactorGovernanceSourceV41Error, match="terminal holdout root"):
        validate_holdout_source_node_v4_1(
            unsealed,
            design_node=_design(),
            previous_node=previous,
            design_pit_records=_records(),
            node_pit_records=_records(),
            actual_month_end_sessions=unsealed["actual_month_end_sessions"],
            expected_design_component_count=3,
            **_cas(previous),
        )
    with pytest.raises(FactorGovernanceSourceV41Error, match="terminal"):
        _append(
            _business_after(ready["cumulative_calendar_sessions"][-1], 1),
            previous=ready,
            node_id="holdout-002",
        )
