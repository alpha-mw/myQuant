from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_investor.market.cn_nontrading_evidence import (
    canonical_json_sha256,
    dataframe_sha256,
)
from quant_investor.market.cn_terminal_delisting_evidence import (
    build_terminal_delisting_evidence,
    read_terminal_delisting_evidence,
    resolve_terminal_delisting_evidence,
    select_terminal_delisting_candidates,
    terminal_delisting_cache_path,
    validate_terminal_delisting_evidence,
)
from quant_investor.market.pit_universe import PITUniverseRecord

OPEN_DATES = [
    "20260623",
    "20260624",
    "20260625",
    "20260626",
    "20260629",
    "20260630",
    "20260701",
    "20260702",
    "20260703",
    "20260706",
    "20260707",
    "20260708",
    "20260709",
    "20260710",
    "20260713",
    "20260714",
]


def _pit(symbol: str, name: str, *, status: str = "L", delist_date: str = "") -> PITUniverseRecord:
    return PITUniverseRecord(
        symbol=symbol,
        name=name,
        source_list_status=status,
        list_date="20200101",
        delist_date=delist_date,
        observed_at="2026-07-15T00:00:00Z",
        source_run_id="unit-test",
    )


class _Provider:
    def __init__(self, *, daily_count: int = 15, reason: str = "退市整理期") -> None:
        self.daily_count = daily_count
        self.reason = reason
        self.calls: list[tuple[str, dict[str, object]]] = []

    def stock_basic(self, **kwargs):
        self.calls.append(("stock_basic", kwargs))
        symbol = str(kwargs["ts_code"])
        name = "国华退" if symbol == "000004.SZ" else "恒久退"
        return pd.DataFrame(
            [
                {
                    "ts_code": symbol,
                    "name": name,
                    "list_status": "L",
                    "list_date": "20200101",
                    "delist_date": None,
                }
            ]
        )

    def namechange(self, **kwargs):
        self.calls.append(("namechange", kwargs))
        symbol = str(kwargs["ts_code"])
        name = "国华退" if symbol == "000004.SZ" else "恒久退"
        return pd.DataFrame(
            [
                {
                    "ts_code": symbol,
                    "name": name,
                    "start_date": "20260623",
                    "end_date": None,
                    "ann_date": "20260613",
                    "change_reason": self.reason,
                }
            ]
        )

    def daily(self, **kwargs):
        self.calls.append(("daily", kwargs))
        return pd.DataFrame(
            [
                {
                    "ts_code": kwargs["ts_code"],
                    "trade_date": trade_date,
                    "close": 1.0,
                }
                for trade_date in OPEN_DATES[: self.daily_count]
            ]
        )

    def trade_cal(self, **kwargs):
        self.calls.append(("trade_cal", kwargs))
        return pd.DataFrame([{"cal_date": trade_date, "is_open": 1} for trade_date in OPEN_DATES])

    def suspend_d(self, **kwargs):
        self.calls.append(("suspend_d", kwargs))
        return pd.DataFrame(columns=["ts_code", "trade_date", "suspend_type"])

    def bak_daily(self, **kwargs):
        self.calls.append(("bak_daily", kwargs))
        return pd.DataFrame(columns=["ts_code", "trade_date"])


def _records() -> dict[str, PITUniverseRecord]:
    return {
        "000004.SZ": _pit("000004.SZ", "国华退"),
        "002808.SZ": _pit("002808.SZ", "恒久退"),
    }


def test_terminal_delisting_evidence_verifies_both_symbols() -> None:
    payload = build_terminal_delisting_evidence(
        _Provider(),
        target_trade_date="20260714",
        candidate_symbols=["000004.SZ", "002808.SZ"],
        pit_records_by_symbol=_records(),
        pit_membership_path="data/parquet/cn/reference/stock_basic_membership.parquet",
        pit_membership_sha256="a" * 64,
    )

    assert payload["all_candidates_verified"] is True
    assert payload["verified_symbols"] == ["000004.SZ", "002808.SZ"]
    assert payload["inferred_delist_dates"] == {
        "000004.SZ": "20260714",
        "002808.SZ": "20260714",
    }
    assert (
        validate_terminal_delisting_evidence(
            payload,
            target_trade_date="20260714",
            candidate_symbols=["000004.SZ", "002808.SZ"],
            pit_membership_path="data/parquet/cn/reference/stock_basic_membership.parquet",
            pit_membership_sha256="a" * 64,
        )
        == []
    )


@pytest.mark.parametrize(
    ("provider", "reason_fragment"),
    [
        (_Provider(daily_count=14), "terminal_daily_session_count_mismatch"),
        (_Provider(daily_count=16), "terminal_daily_session_count_mismatch"),
        (_Provider(reason="改名"), "terminal_namechange_active_row_count_mismatch"),
    ],
)
def test_terminal_delisting_evidence_fails_closed_on_ambiguous_window(
    provider: _Provider,
    reason_fragment: str,
) -> None:
    payload = build_terminal_delisting_evidence(
        provider,
        target_trade_date="20260714",
        candidate_symbols=["000004.SZ"],
        pit_records_by_symbol=_records(),
        pit_membership_path="pit.parquet",
        pit_membership_sha256="a" * 64,
    )

    assert payload["all_candidates_verified"] is False
    assert any(reason_fragment in reason for reason in payload["rejected_symbols"]["000004.SZ"])


def test_terminal_candidate_selection_enforces_outer_pit_gates() -> None:
    records = {
        **_records(),
        "600001.SH": _pit("600001.SH", "沪市退"),
        "000005.SZ": _pit("000005.SZ", "普通股份"),
        "000006.SZ": _pit("000006.SZ", "已退", status="D"),
        "000007.SZ": _pit("000007.SZ", "日期退", delist_date="20260714"),
    }

    assert select_terminal_delisting_candidates(
        records,
        target_trade_date="20260714",
        pit_records_by_symbol=records,
    ) == ["000004.SZ", "002808.SZ"]


def test_resolver_reuses_only_valid_positive_cache_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    provider = _Provider()
    kwargs = {
        "cache_root": tmp_path,
        "target_trade_date": "20260714",
        "candidate_symbols": ["000004.SZ", "002808.SZ"],
        "pit_records_by_symbol": _records(),
        "pit_membership_path": "pit.parquet",
        "pit_membership_sha256": "a" * 64,
    }
    first = resolve_terminal_delisting_evidence(provider, **kwargs)
    assert first["status"] == "passed"
    assert first["cache_reused"] is False
    call_count = len(provider.calls)

    second = resolve_terminal_delisting_evidence(provider, **kwargs)
    assert second["cache_reused"] is True
    assert len(provider.calls) == call_count

    evidence_path = Path(second["evidence_path"])
    tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
    tampered["inferred_delist_dates"]["000004.SZ"] = "20260715"
    evidence_path.write_text(json.dumps(tampered), encoding="utf-8")
    _payload, blockers = read_terminal_delisting_evidence(
        evidence_path,
        target_trade_date="20260714",
        candidate_symbols=["000004.SZ", "002808.SZ"],
        pit_membership_path="pit.parquet",
        pit_membership_sha256="a" * 64,
    )
    assert "payload_sha256_mismatch" in blockers

    refreshed = resolve_terminal_delisting_evidence(provider, **kwargs)
    assert refreshed["status"] == "passed"
    assert refreshed["cache_reused"] is False
    assert len(provider.calls) > call_count


def test_terminal_cache_isolated_by_full_pit_sha_without_overwriting_old_bytes(
    tmp_path: Path,
) -> None:
    candidates = ["000004.SZ", "002808.SZ"]
    legacy_path = terminal_delisting_cache_path(
        tmp_path,
        target_trade_date="20260714",
        candidate_symbols=candidates,
    )
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_bytes = b"legacy-unversioned-cache\n"
    legacy_path.write_bytes(legacy_bytes)

    common = {
        "cache_root": tmp_path,
        "target_trade_date": "20260714",
        "candidate_symbols": candidates,
        "pit_records_by_symbol": _records(),
        "pit_membership_path": "pit.parquet",
    }
    first = resolve_terminal_delisting_evidence(
        _Provider(),
        **common,
        pit_membership_sha256="a" * 64,
    )
    first_path = Path(first["evidence_path"])
    first_bytes = first_path.read_bytes()

    second = resolve_terminal_delisting_evidence(
        _Provider(),
        **common,
        pit_membership_sha256="b" * 64,
    )
    second_path = Path(second["evidence_path"])

    assert first["status"] == second["status"] == "passed"
    assert first_path != second_path
    assert f"pit_{'a' * 64}" in first_path.parts
    assert f"pit_{'b' * 64}" in second_path.parts
    assert legacy_path.read_bytes() == legacy_bytes
    assert first_path.read_bytes() == first_bytes


def test_terminal_cache_path_rejects_incomplete_explicit_pit_sha(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="complete 64-character SHA-256"):
        terminal_delisting_cache_path(
            tmp_path,
            target_trade_date="20260714",
            candidate_symbols=["000004.SZ"],
            pit_membership_sha256="a" * 63,
        )


def test_validator_rejects_resigned_raw_semantic_tamper() -> None:
    payload = build_terminal_delisting_evidence(
        _Provider(),
        target_trade_date="20260714",
        candidate_symbols=["000004.SZ"],
        pit_records_by_symbol=_records(),
        pit_membership_path="pit.parquet",
        pit_membership_sha256="a" * 64,
    )
    stock_query = payload["symbol_proofs"]["000004.SZ"]["queries"]["stock_basic"]
    stock_query["raw_records"][0]["name"] = "普通股份"
    stock_query["raw_rows_sha256"] = dataframe_sha256(
        pd.DataFrame(
            stock_query["raw_records"],
            columns=stock_query["columns"],
        )
    )
    unsigned = dict(payload)
    unsigned.pop("payload_sha256")
    payload["payload_sha256"] = canonical_json_sha256(unsigned)

    blockers = validate_terminal_delisting_evidence(
        payload,
        target_trade_date="20260714",
        candidate_symbols=["000004.SZ"],
        pit_membership_path="pit.parquet",
        pit_membership_sha256="a" * 64,
    )

    assert "symbol_stock_basic_semantic_mismatch:000004.SZ" in blockers
