from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from types import SimpleNamespace

import pandas as pd

import quant_investor.market.cn_history_audit as history_audit_module
from quant_investor.market.cn_history_audit import (
    _missing_pit_components,
    _query_open_trade_dates,
    _read_suspend_evidence_cache,
    _select_suspension_continuity_symbols,
    _suspension_continuity_cache_path,
    build_cn_history_audit,
    run_cn_history_audit,
)
from quant_investor.market.cn_nontrading_evidence import (
    canonical_json_sha256,
    symbol_set_sha256,
)
from quant_investor.market.pit_universe import PITUniverseRecord


def _dates(count: int) -> list[str]:
    start = datetime(2026, 1, 1)
    return [
        (start + timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(count)
    ]


def _record(symbol: str, *, delist_date: str = "") -> PITUniverseRecord:
    return PITUniverseRecord(
        symbol=symbol,
        list_date="20200101",
        delist_date=delist_date,
        source_list_status="D" if delist_date else "L",
    )


def test_full_100_date_audit_excludes_target_day_delist_and_accepts_typed_nontrading():
    dates = _dates(100)
    delist_date = dates[60]
    nontrading_date = dates[30]
    rows = []
    for trade_date in dates:
        rows.append(
            {"ts_code": "000001.SZ", "trade_date": trade_date, "adj_factor": 1.0}
        )
        if trade_date != nontrading_date:
            rows.append(
                {"ts_code": "000002.SZ", "trade_date": trade_date, "adj_factor": 1.0}
            )
        if trade_date < delist_date:
            rows.append(
                {"ts_code": "000003.SZ", "trade_date": trade_date, "adj_factor": 1.0}
            )

    audit = build_cn_history_audit(
        bars=pd.DataFrame(rows),
        trade_dates=dates,
        component_symbols=["000001.SZ", "000002.SZ", "000003.SZ"],
        pit_records_by_symbol={
            "000001.SZ": _record("000001.SZ"),
            "000002.SZ": _record("000002.SZ"),
            "000003.SZ": _record("000003.SZ", delist_date=delist_date),
        },
        suspended_evidence_by_date={},
        nontrading_evidence_by_date={
            nontrading_date: {"verified_symbols": ["000002.SZ"]}
        },
    )

    assert audit["history_audit_status"] == "passed"
    assert audit["audited_trade_dates_count"] == 100
    assert audit["per_date_count"] == 100
    assert audit["prior_trade_dates_reused"] == 0
    assert audit["full_window_recomputed"] is True
    assert audit["history_unresolved_gap_dates"] == []
    assert audit["history_primary_absence_dates"] == [nontrading_date]
    delist_row = next(
        row for row in audit["per_date"] if row["trade_date"] == delist_date
    )
    assert delist_row["excluded_delisted_on_target_symbols"] == ["000003.SZ"]
    assert delist_row["verified_inactive_or_prelisting_absent"] == [
        "000003.SZ"
    ]
    assert delist_row["true_missing_symbols"] == []
    nontrading_row = next(
        row for row in audit["per_date"] if row["trade_date"] == nontrading_date
    )
    assert nontrading_row["verified_nontrading_bak_daily_zero"] == [
        "000002.SZ"
    ]


def test_terminal_sidecar_boundary_keeps_last_session_and_excludes_delist_day():
    dates = ["20260713", "20260714"]
    audit = build_cn_history_audit(
        bars=pd.DataFrame(
            [
                {
                    "ts_code": "000004.SZ",
                    "trade_date": "20260713",
                    "adj_factor": 1.0,
                }
            ]
        ),
        trade_dates=dates,
        component_symbols=["000004.SZ"],
        pit_records_by_symbol={"000004.SZ": _record("000004.SZ")},
        suspended_evidence_by_date={},
        nontrading_evidence_by_date={},
        terminal_delist_dates_by_symbol={"000004.SZ": "20260714"},
    )

    assert audit["history_audit_status"] == "passed"
    last_session, delist_day = audit["per_date"]
    assert last_session["observed_active_count"] == 1
    assert last_session["excluded_delisted_symbols"] == []
    assert delist_day["excluded_delisted_on_target_symbols"] == [
        "000004.SZ"
    ]
    assert delist_day["verified_terminal_delisting_absent"] == [
        "000004.SZ"
    ]
    assert delist_day["verified_inactive_or_prelisting_absent"] == [
        "000004.SZ"
    ]
    assert delist_day["true_missing_symbols"] == []


def test_unexplained_active_primary_absence_remains_fail_closed():
    trade_date = "20260707"
    audit = build_cn_history_audit(
        bars=pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "adj_factor": 1.0,
                }
            ]
        ),
        trade_dates=[trade_date],
        component_symbols=["000001.SZ", "000002.SZ"],
        pit_records_by_symbol={
            "000001.SZ": _record("000001.SZ"),
            "000002.SZ": _record("000002.SZ"),
        },
        suspended_evidence_by_date={},
        nontrading_evidence_by_date={},
    )

    assert audit["history_audit_status"] == "blocked"
    assert audit["history_unresolved_gap_dates"] == [trade_date]
    assert audit["history_true_missing_symbols_by_date"] == {
        trade_date: ["000002.SZ"]
    }
    assert audit["per_date"][0]["blockers"] == ["true_missing:1"]


def test_missing_pit_record_is_a_blocker_not_an_active_gap():
    trade_date = "20260707"
    audit = build_cn_history_audit(
        bars=pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "adj_factor": 1.0,
                }
            ]
        ),
        trade_dates=[trade_date],
        component_symbols=["000001.SZ"],
        pit_records_by_symbol={},
        suspended_evidence_by_date={},
        nontrading_evidence_by_date={},
    )

    assert audit["history_audit_status"] == "blocked"
    assert audit["per_date"][0]["unknown_membership_symbols"] == ["000001.SZ"]
    assert audit["per_date"][0]["blockers"] == ["pit_membership_unknown:1"]


def test_pending_pit_record_is_not_cleared_as_active_nontrading():
    trade_date = "20260707"
    audit = build_cn_history_audit(
        bars=pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"]),
        trade_dates=[trade_date],
        component_symbols=["000001.SZ"],
        pit_records_by_symbol={
            "000001.SZ": PITUniverseRecord(
                symbol="000001.SZ",
                list_date="20260701",
                source_list_status="P",
            )
        },
        suspended_evidence_by_date={},
        nontrading_evidence_by_date={
            trade_date: {"verified_symbols": ["000001.SZ"]}
        },
    )

    assert audit["history_audit_status"] == "blocked"
    assert audit["per_date"][0]["unknown_membership_reasons"] == {
        "000001.SZ": "pending"
    }
    assert audit["per_date"][0]["verified_nontrading_bak_daily_zero"] == []


def test_trade_calendar_requires_exact_open_nonempty_dates():
    dates = pd.bdate_range("2026-01-01", periods=100).strftime("%Y%m%d").tolist()

    class _Provider:
        def trade_cal(self, **kwargs):
            assert kwargs["is_open"] == "1"
            return pd.DataFrame(
                [
                    *({"cal_date": value, "is_open": 1} for value in dates),
                    {"cal_date": None, "is_open": 1},
                    {"cal_date": "20260701", "is_open": 0},
                ]
            )

    selected, evidence = _query_open_trade_dates(
        _Provider(),
        end_date=dates[-1],
        days=100,
    )

    assert selected == dates
    assert evidence["raw_row_count"] == 102


def test_trade_calendar_does_not_count_empty_date_toward_100_days():
    dates = pd.bdate_range("2026-01-01", periods=99).strftime("%Y%m%d").tolist()

    class _Provider:
        def trade_cal(self, **_kwargs):
            return pd.DataFrame(
                [
                    *({"cal_date": value, "is_open": 1} for value in dates),
                    {"cal_date": None, "is_open": 1},
                ]
            )

    import pytest

    with pytest.raises(RuntimeError, match="fewer than 100"):
        _query_open_trade_dates(
            _Provider(),
            end_date=dates[-1],
            days=100,
        )


def test_trade_calendar_requires_is_open_column():
    dates = pd.bdate_range("2026-01-01", periods=100).strftime("%Y%m%d")

    class _Provider:
        def trade_cal(self, **_kwargs):
            return pd.DataFrame({"cal_date": dates})

    import pytest

    with pytest.raises(RuntimeError, match="lacks is_open"):
        _query_open_trade_dates(
            _Provider(),
            end_date=str(dates[-1]),
            days=100,
        )


def test_current_full_a_component_missing_from_pit_is_detected():
    assert _missing_pit_components(
        ["000001.SZ", "000002.SZ"],
        ["000001.SZ", "600000.SH"],
    ) == ["000002.SZ"]


def test_suspension_continuity_is_independent_and_disjoint():
    trade_date = "20260707"
    audit = build_cn_history_audit(
        bars=pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "trade_date": trade_date,
                    "adj_factor": 1.0,
                }
            ]
        ),
        trade_dates=[trade_date],
        component_symbols=["000001.SZ", "000002.SZ", "000003.SZ"],
        pit_records_by_symbol={
            "000001.SZ": _record("000001.SZ"),
            "000002.SZ": _record("000002.SZ"),
            "000003.SZ": _record("000003.SZ"),
        },
        suspended_evidence_by_date={trade_date: ["000002.SZ"]},
        suspension_continuity_by_date={trade_date: ["000003.SZ"]},
        nontrading_evidence_by_date={},
    )

    assert audit["history_audit_status"] == "passed"
    row = audit["per_date"][0]
    assert row["verified_exact_suspended_absent"] == ["000002.SZ"]
    assert row["verified_suspension_continuity_absent"] == ["000003.SZ"]
    assert row["verified_inactive_or_prelisting_absent"] == []
    assert row["verified_suspended_absent"] == ["000002.SZ", "000003.SZ"]
    assert row["classification_sets_disjoint"] is True
    assert row["classification_union_complete"] is True


def test_continuity_selector_rejects_current_resume_or_nonzero_bak():
    base = {
        "unresolved_symbols": ["000001.SZ", "000002.SZ"],
        "bak_daily_payload": {
            "rejected_symbols": {
                "000001.SZ": ["exact_row_missing"],
                "000002.SZ": ["vol_nonzero_or_invalid"],
            }
        },
        "previous_suspended_symbols": ["000001.SZ", "000002.SZ"],
        "next_suspended_symbols": ["000001.SZ", "000002.SZ"],
    }

    assert _select_suspension_continuity_symbols(
        **base,
        current_event_symbols=[],
    ) == ["000001.SZ"]
    assert _select_suspension_continuity_symbols(
        **base,
        current_event_symbols=["000001.SZ"],
    ) == []


def test_suspension_continuity_cache_isolated_by_full_pit_sha(tmp_path):
    first = _suspension_continuity_cache_path(
        tmp_path,
        trade_date="20260707",
        symbols=["000001.SZ"],
        pit_membership_sha256="a" * 64,
    )
    second = _suspension_continuity_cache_path(
        tmp_path,
        trade_date="20260707",
        symbols=["000001.SZ"],
        pit_membership_sha256="b" * 64,
    )

    assert first != second
    assert f"pit_{'a' * 64}" in first.parts
    assert f"pit_{'b' * 64}" in second.parts

    import pytest

    with pytest.raises(ValueError, match="complete 64-character"):
        _suspension_continuity_cache_path(
            tmp_path,
            trade_date="20260707",
            symbols=["000001.SZ"],
            pit_membership_sha256="short",
        )


def test_suspend_v5_readback_replays_all_exact_event_types(tmp_path):
    trade_date = "20260707"
    records = [
        {
            "ts_code": "000001.SZ",
            "trade_date": trade_date,
            "suspend_type": "R",
        }
    ]
    payload = {
        "version": 5,
        "trade_date": trade_date,
        "query_run_id": "unit-test-run",
        "query_succeeded": True,
        "query_variant": "trade_date",
        "query_params": {"trade_date": trade_date},
        "source": "tushare.suspend_d",
        "semantic_scope": "exact_date_suspend_events_only",
        "continuation_state_complete": False,
        "exact_date_rows_validated": True,
        "raw_row_count": 1,
        "raw_rows_sha256": "a" * 64,
        "matched_row_count": 0,
        "symbols": [],
        "matched_symbols_sha256": symbol_set_sha256([]),
        "resume_symbols": ["000001.SZ"],
        "resume_symbols_sha256": symbol_set_sha256(["000001.SZ"]),
        "other_event_symbols": [],
        "other_event_symbols_sha256": symbol_set_sha256([]),
        "exact_event_row_count": 1,
        "exact_event_records": records,
        "exact_event_records_sha256": canonical_json_sha256(records),
        "updated_at": "2026-07-13T00:00:00Z",
    }
    payload["payload_sha256"] = canonical_json_sha256(payload)
    path = tmp_path / ".suspend_20260707.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    symbols, readback, blockers = _read_suspend_evidence_cache(
        path,
        trade_date=trade_date,
    )

    assert blockers == []
    assert symbols == set()
    assert readback["resume_symbols"] == ["000001.SZ"]

    readback["resume_symbols"] = []
    forged = dict(readback)
    forged.pop("payload_sha256")
    readback["payload_sha256"] = canonical_json_sha256(forged)
    path.write_text(json.dumps(readback), encoding="utf-8")
    _symbols, _payload, blockers = _read_suspend_evidence_cache(
        path,
        trade_date=trade_date,
    )
    assert "suspend_evidence_resume_records_mismatch" in blockers


def test_candidate_root_audit_is_explicit_and_preserves_active_pointer(
    tmp_path,
    monkeypatch,
):
    active_root = tmp_path / "active-data"
    active_pointer = active_root / "parquet" / "cn" / "_latest.json"
    active_pointer.parent.mkdir(parents=True)
    active_pointer_bytes = b'{"snapshot_id":"protected-active"}\n'
    active_pointer.write_bytes(active_pointer_bytes)

    candidate_root = tmp_path / "private-candidate-data"
    candidate_pointer = candidate_root / "prepared" / "candidate-pointer.json"
    candidate_pointer.parent.mkdir(parents=True)
    components_path = (
        candidate_root / "cn_universe" / "cn_index_components.json"
    )
    components_path.parent.mkdir(parents=True)
    components_path.write_text(
        json.dumps({"full_a": ["000001.SZ"]}),
        encoding="utf-8",
    )
    pit_path = candidate_root / "pit.parquet"
    pit_path.write_bytes(b"sealed-pit")
    pit_sha256 = hashlib.sha256(pit_path.read_bytes()).hexdigest()
    manifest_path = candidate_root / "candidate-manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    candidate_payload = {
        "snapshot_id": "candidate-20260818",
        "latest_complete_trade_date": "20260818",
        "manifest_path": str(manifest_path),
        "coverage": {
            "coverage_schema_version": "cn-full-a-coverage.v4",
            "pit_membership_path": str(pit_path),
            "pit_membership_sha256": pit_sha256,
        },
    }
    candidate_pointer.write_text(
        json.dumps(candidate_payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pointer_reads = []

    def _validate_latest(store):
        assert store.reader.latest_pointer_path == candidate_pointer
        return {"status": "passed", "blockers": []}

    def _load_latest_payload(reader, *, refresh=False):
        pointer_reads.append(reader.latest_pointer_path)
        return json.loads(
            reader.latest_pointer_path.read_text(encoding="utf-8")
        )

    monkeypatch.setattr(
        history_audit_module.MarketDataStore,
        "validate_latest",
        _validate_latest,
    )
    monkeypatch.setattr(
        history_audit_module.MarketDataReader,
        "_load_latest_payload",
        _load_latest_payload,
    )
    monkeypatch.setattr(
        history_audit_module.MarketDataReader,
        "coverage_bound_pit",
        lambda _reader: {
            "status": "passed",
            "blockers": [],
            "canonical_sha256": pit_sha256,
            "records": {"000001.SZ": _record("000001.SZ")},
        },
    )
    monkeypatch.setattr(
        history_audit_module.MarketDataReader,
        "_require_snapshot",
        lambda _reader: SimpleNamespace(
            table_root=candidate_root / "table",
            serving_root=candidate_root / "serving",
        ),
    )
    monkeypatch.setattr(
        history_audit_module,
        "_read_canonical_window",
        lambda *_args, **_kwargs: (
            pd.DataFrame(
                [
                    {
                        "ts_code": "000001.SZ",
                        "trade_date": "20260818",
                        "open": 1.0,
                        "high": 1.0,
                        "low": 1.0,
                        "close": 1.0,
                        "vol": 1.0,
                        "amount": 1.0,
                        "adj_factor": 1.0,
                    }
                ]
            ),
            {
                "table_sha256": "a" * 64,
                "serving_sha256": "a" * 64,
                "table_serving_match": True,
            },
        ),
    )

    report, output_path = run_cn_history_audit(
        data_root=active_root,
        candidate_data_root=candidate_root,
        candidate_pointer_path=candidate_pointer,
        output_root=candidate_root / "audit-output",
        days=1,
        end_date="20260818",
        trade_dates=["20260818"],
    )

    assert output_path.is_relative_to(candidate_root)
    assert pointer_reads and set(pointer_reads) == {candidate_pointer}
    assert active_pointer.read_bytes() == active_pointer_bytes
    assert report["audit_input_kind"] == "candidate"
    assert report["audit_method"] == "full_recompute_from_candidate"
    assert report["candidate"]["latest_path"] == str(candidate_pointer)
    assert report["pit_membership_evidence"]["binding_source"] == (
        "candidate_market_coverage"
    )
    assert report["candidate_data_ready"] is True
    assert report["latest_canonical_ready"] is False
    assert report["portfolio_data_ready"] is False
    assert report["protected_active_pointer"]["unchanged"] is True
    assert report["protected_active_pointer"]["sha256_before"] == (
        hashlib.sha256(active_pointer_bytes).hexdigest()
    )


def test_history_audit_script_accepts_explicit_candidate_input_flags():
    from scripts.run_cn_history_audit import _parse_args

    args = _parse_args(
        [
            "--candidate-data-root",
            "/private/tmp/candidate-data",
            "--candidate-pointer-path",
            "/private/tmp/candidate-data/candidate-pointer.json",
            "--output-root",
            "/private/tmp/candidate-audit",
        ]
    )

    assert args.candidate_data_root == "/private/tmp/candidate-data"
    assert args.candidate_pointer_path.endswith("candidate-pointer.json")
