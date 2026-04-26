from __future__ import annotations

import json

import pytest

from quant_investor.data_quality_contract import (
    ISSUE_LOOKAHEAD_EFFECTIVE_DATE,
    ISSUE_LOOKAHEAD_OBSERVED_AT,
    ISSUE_MISSING_PROVENANCE,
    ISSUE_MISSING_REQUIRED_FIELD,
    ISSUE_OUTLIER_FIELD,
    ISSUE_SEVERITY_BLOCKER,
    ISSUE_SEVERITY_INFO,
    ISSUE_SEVERITY_WARNING,
    ISSUE_STALE_FIELD,
    ISSUE_UNTRADABLE,
    TRADABILITY_DELISTED,
    TRADABILITY_LOW_LIQUIDITY,
    TRADABILITY_SUSPENDED,
    DataQualityAssessment,
    DataQualityContractStore,
    DataQualityIssue,
    FieldProvenance,
    PointInTimeSnapshot,
    TradabilityStatus,
    assess_point_in_time_snapshot,
    build_global_context_quality_patch,
    build_tradability_status,
    days_between,
    generate_data_quality_issues,
    iso_value_after,
    make_assessment_id,
    make_issue_id,
    make_snapshot_id,
    parse_iso_date_or_datetime,
)


def _provenance(
    field_name: str,
    *,
    effective_date: str | None = "2026-04-25",
    observed_at: str | None = "2026-04-25T15:00:00",
) -> FieldProvenance:
    return FieldProvenance(
        field_name=field_name,
        source="unit-fixture",
        as_of="2026-04-26",
        effective_date=effective_date,
        observed_at=observed_at,
        is_point_in_time=True,
    )


def _snapshot(
    *,
    symbol: str = "000001.SZ",
    fields: dict[str, object] | None = None,
    provenance: dict[str, FieldProvenance] | None = None,
    quality_issues: list[DataQualityIssue] | None = None,
    tradability_status: TradabilityStatus | None = None,
) -> PointInTimeSnapshot:
    resolved_fields = fields if fields is not None else {"close": 10.0, "volume": 1000}
    resolved_provenance = provenance if provenance is not None else {
        "close": _provenance("close"),
        "volume": _provenance("volume"),
    }
    return PointInTimeSnapshot(
        snapshot_id=make_snapshot_id(
            symbol=symbol,
            market="CN",
            as_of="2026-04-26",
            latest_trade_date="2026-04-25",
        ),
        symbol=symbol,
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        fields=resolved_fields,
        provenance=resolved_provenance,
        quality_issues=quality_issues or [],
        tradability_status=tradability_status,
        metadata={"fixture": True},
    )


def test_dataclass_round_trips() -> None:
    provenance = _provenance("close")
    issue = DataQualityIssue(
        issue_id=make_issue_id(
            symbol="000001.SZ",
            market="CN",
            as_of="2026-04-26",
            issue_type=ISSUE_MISSING_PROVENANCE,
            field_name="close",
        ),
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        field_name="close",
        issue_type=ISSUE_MISSING_PROVENANCE,
        severity=ISSUE_SEVERITY_WARNING,
        message="missing provenance",
    )
    tradability = build_tradability_status(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
    )
    snapshot = _snapshot(quality_issues=[issue], tradability_status=tradability)
    assessment = assess_point_in_time_snapshot(snapshot)

    assert FieldProvenance.from_dict(provenance.to_dict()).to_dict() == provenance.to_dict()
    assert DataQualityIssue.from_dict(issue.to_dict()).to_dict() == issue.to_dict()
    assert TradabilityStatus.from_dict(tradability.to_dict()).to_dict() == tradability.to_dict()
    assert PointInTimeSnapshot.from_dict(snapshot.to_dict()).to_dict() == snapshot.to_dict()
    assert DataQualityAssessment.from_dict(assessment.to_dict()).to_dict() == assessment.to_dict()

    with pytest.raises(ValueError, match="severity"):
        DataQualityIssue(severity="urgent")


def test_deterministic_ids() -> None:
    snapshot_args = {
        "symbol": "000001.SZ",
        "market": "CN",
        "as_of": "2026-04-26",
        "latest_trade_date": "2026-04-25",
    }
    issue_args = {
        "symbol": "000001.SZ",
        "market": "CN",
        "as_of": "2026-04-26",
        "issue_type": ISSUE_MISSING_REQUIRED_FIELD,
        "field_name": "close",
    }
    snapshot_id = make_snapshot_id(**snapshot_args)

    assert make_snapshot_id(**snapshot_args) == snapshot_id
    assert make_issue_id(**issue_args) == make_issue_id(**issue_args)
    assert make_assessment_id(snapshot_id=snapshot_id) == make_assessment_id(snapshot_id=snapshot_id)


def test_date_helpers() -> None:
    assert parse_iso_date_or_datetime("2026-04-26").date().isoformat() == "2026-04-26"
    assert parse_iso_date_or_datetime("2026-04-26T15:30:00").hour == 15
    assert iso_value_after("2026-04-27", "2026-04-26") is True
    assert iso_value_after(None, "2026-04-26") is False
    assert days_between("2026-04-20", "2026-04-26") == 6
    assert days_between("2026-04-26", "2026-04-20") == -6

    with pytest.raises(ValueError, match="Invalid ISO"):
        parse_iso_date_or_datetime("not-a-date")


def test_tradability_helper_validation_and_reasons() -> None:
    clean = build_tradability_status(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
    )
    suspended = build_tradability_status(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        is_suspended=True,
    )
    delisted = build_tradability_status(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        is_delisted=True,
    )
    low_liquidity = build_tradability_status(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        liquidity_score=0.10,
        min_liquidity_score=0.20,
    )

    assert clean.is_tradable is True
    assert suspended.is_tradable is False
    assert TRADABILITY_SUSPENDED in suspended.reasons
    assert delisted.is_tradable is False
    assert TRADABILITY_DELISTED in delisted.reasons
    assert low_liquidity.is_tradable is False
    assert TRADABILITY_LOW_LIQUIDITY in low_liquidity.reasons

    with pytest.raises(ValueError, match="adv"):
        build_tradability_status(
            symbol="000001.SZ",
            market="CN",
            as_of="2026-04-26",
            latest_trade_date="2026-04-25",
            adv=-1.0,
        )
    with pytest.raises(ValueError, match="liquidity_score"):
        build_tradability_status(
            symbol="000001.SZ",
            market="CN",
            as_of="2026-04-26",
            latest_trade_date="2026-04-25",
            liquidity_score=1.1,
        )


def test_issue_generation_covers_required_provenance_lookahead_stale_outlier_and_tradability() -> None:
    existing_duplicate = DataQualityIssue(
        issue_id=make_issue_id(
            symbol="000001.SZ",
            market="CN",
            as_of="2026-04-26",
            issue_type=ISSUE_MISSING_REQUIRED_FIELD,
            field_name="required_missing",
        ),
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        field_name="required_missing",
        issue_type=ISSUE_MISSING_REQUIRED_FIELD,
        severity=ISSUE_SEVERITY_BLOCKER,
        message="preexisting duplicate",
    )
    tradability = build_tradability_status(
        symbol="000001.SZ",
        market="CN",
        as_of="2026-04-26",
        latest_trade_date="2026-04-25",
        is_suspended=True,
    )
    snapshot = _snapshot(
        fields={
            "close": 10.0,
            "volume": 1000,
            "future_eps": 1.2,
            "late_price": 11.0,
            "stale_field": 5.0,
            "outlier_field": 999.0,
        },
        provenance={
            "close": _provenance("close"),
            "future_eps": _provenance("future_eps", effective_date="2026-04-27"),
            "late_price": _provenance("late_price", observed_at="2026-04-26T16:01:00"),
            "stale_field": _provenance("stale_field", observed_at="2026-04-01"),
            "outlier_field": _provenance("outlier_field"),
        },
        quality_issues=[existing_duplicate],
        tradability_status=tradability,
    )

    issues = generate_data_quality_issues(
        snapshot,
        required_fields=["close", "required_missing"],
        freshness_rules_days={"stale_field": 5},
        outlier_flags={"outlier_field": True},
    )
    issue_types = [issue.issue_type for issue in issues]

    assert ISSUE_MISSING_REQUIRED_FIELD in issue_types
    assert ISSUE_MISSING_PROVENANCE in issue_types
    assert ISSUE_LOOKAHEAD_EFFECTIVE_DATE in issue_types
    assert ISSUE_LOOKAHEAD_OBSERVED_AT in issue_types
    assert ISSUE_STALE_FIELD in issue_types
    assert ISSUE_OUTLIER_FIELD in issue_types
    assert ISSUE_UNTRADABLE in issue_types
    assert issue_types.count(ISSUE_MISSING_REQUIRED_FIELD) == 1
    assert any(issue.metadata.get("reasons") == [TRADABILITY_SUSPENDED] for issue in issues)


def test_assessment_quarantine_scoring_and_clean_snapshot() -> None:
    blocked_snapshot = _snapshot(
        fields={},
        provenance={},
        tradability_status=build_tradability_status(
            symbol="000001.SZ",
            market="CN",
            as_of="2026-04-26",
            latest_trade_date="2026-04-25",
            is_suspended=True,
        ),
    )
    blocked = assess_point_in_time_snapshot(
        blocked_snapshot,
        required_fields=["close", "volume", "amount"],
    )
    clean = assess_point_in_time_snapshot(_snapshot(), required_fields=["close", "volume"])

    assert blocked.quarantine is True
    assert blocked.is_researchable is False
    assert blocked.data_quality_score == 0.0
    assert TRADABILITY_SUSPENDED in blocked.tradability_reasons
    assert clean.quarantine is False
    assert clean.is_researchable is True
    assert clean.data_quality_score == pytest.approx(1.0)


def test_global_context_quality_patch_is_deterministic_and_serializable() -> None:
    clean = assess_point_in_time_snapshot(_snapshot(symbol="000002.SZ"))
    blocked = assess_point_in_time_snapshot(
        _snapshot(
            symbol="000001.SZ",
            fields={},
            provenance={},
            tradability_status=build_tradability_status(
                symbol="000001.SZ",
                market="CN",
                as_of="2026-04-26",
                latest_trade_date="2026-04-25",
                is_suspended=True,
            ),
        ),
        required_fields=["close"],
    )

    patch = build_global_context_quality_patch([clean, blocked])

    assert patch["data_quality_quarantine"] == ["000001.SZ"]
    assert patch["tradability_blocked_symbols"] == ["000001.SZ"]
    assert list(patch["symbol_quality_scores"]) == ["000001.SZ", "000002.SZ"]
    assert patch["symbol_issue_counts"]["000001.SZ"] == blocked.issue_count
    json.dumps(patch, ensure_ascii=False)


def test_data_quality_contract_store_round_trip_and_duplicate_protection(tmp_path) -> None:
    store = DataQualityContractStore(tmp_path)
    snapshot = _snapshot()
    assessment = assess_point_in_time_snapshot(snapshot)

    store.append_snapshot(snapshot)
    assert store.read_snapshots()[0].snapshot_id == snapshot.snapshot_id
    with pytest.raises(ValueError, match="Duplicate snapshot_id"):
        store.append_snapshot(snapshot)

    store.append_assessment(assessment)
    assert store.read_assessments()[0].assessment_id == assessment.assessment_id
    with pytest.raises(ValueError, match="Duplicate assessment_id"):
        store.append_assessment(assessment)
    assert store.append_assessments([]) == 0

    bad_store = DataQualityContractStore(tmp_path / "bad")
    bad_store.snapshots_path.parent.mkdir(parents=True, exist_ok=True)
    bad_store.snapshots_path.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON"):
        bad_store.read_snapshots()
