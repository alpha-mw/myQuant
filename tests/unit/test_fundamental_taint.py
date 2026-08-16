from __future__ import annotations

from decimal import Decimal
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_investor.market.fundamental_incremental import (
    FINANCIAL_TABLES,
    replay_successor_event_trace,
)
from quant_investor.market.fundamental_successor_source import (
    acquire_successor_support,
    build_successor_support_plan,
    load_capture_support_tables,
)
from quant_investor.market.fundamental_taint import (
    FundamentalTaintError,
    analyze_deferred_fundamental_taints,
    validate_taint_analysis_result,
)
from quant_investor.market.tushare_transport import (
    TushareResponse,
    replay_tushare_response_bytes,
)


SYMBOL = "001236.SZ"
PARENT = "20220103"
TARGET = "20220104"
TAINT_END = "20210630"
TAINT_AVAILABILITY = "20220102"
SUPPORTED_END = "20211231"


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _table(api_name: str) -> str:
    return {
        "balancesheet_vip": "balancesheet",
        "cashflow_vip": "cashflow",
        "daily_basic": "daily_basic",
        "fina_indicator_vip": "fina_indicator",
        "forecast_vip": "forecast",
        "income_vip": "income",
    }[api_name]


def _base_row(
    api_name: str,
    params: Mapping[str, Any],
    fields: Sequence[str],
    *,
    end_date: str,
    comp_type: str = "1",
) -> dict[str, Any]:
    table = _table(api_name)
    availability = (
        params["trade_date"]
        if table == "daily_basic"
        else params.get("ann_date") or params.get("start_date")
    )
    values: dict[str, Any] = {
        "ts_code": SYMBOL,
        "ann_date": availability,
        "f_ann_date": availability,
        "end_date": end_date,
        "trade_date": availability,
        "report_type": "1",
        "comp_type": comp_type,
        "update_flag": "1",
        "type": "预增",
        "summary": "support",
        "change_reason": "operations",
    }
    for field in fields:
        values.setdefault(field, Decimal("1"))
    return values


class _Client:
    def __init__(self, *, reachable: bool = False) -> None:
        self.calls = 0
        self.reachable = reachable

    def request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ) -> TushareResponse:
        self.calls += 1
        table = _table(api_name)
        availability = (
            params.get("trade_date")
            or params.get("ann_date")
            or params.get("start_date")
        )
        rows: list[Mapping[str, Any]] = []
        if table == "daily_basic":
            rows = [
                _base_row(
                    api_name,
                    params,
                    expected_fields,
                    end_date=SUPPORTED_END,
                )
            ]
        elif table == "balancesheet" and availability == TAINT_AVAILABILITY:
            rows = [
                _base_row(
                    api_name,
                    params,
                    expected_fields,
                    end_date=TAINT_END,
                    comp_type="7",
                )
            ]
        elif table in FINANCIAL_TABLES and availability == PARENT:
            rows = [
                _base_row(
                    api_name,
                    params,
                    expected_fields,
                    end_date=SUPPORTED_END,
                )
            ]
        elif (
            self.reachable
            and table in FINANCIAL_TABLES
            and availability == TARGET
        ):
            rows = [
                _base_row(
                    api_name,
                    params,
                    expected_fields,
                    end_date=TAINT_END,
                )
            ]
        physical = [
            [row[field] for field in expected_fields] for row in rows
        ]

        def scalar(value: Any) -> Any:
            if isinstance(value, Decimal):
                return int(value)
            return value

        raw = _canonical_bytes(
            {
                "code": 0,
                "data": {
                    "count": len(physical),
                    "fields": list(expected_fields),
                    "has_more": False,
                    "items": [
                        [scalar(value) for value in row] for row in physical
                    ],
                },
                "detail": "",
                "msg": "",
                "request_id": f"taint-{self.calls}",
            }
        )
        return replay_tushare_response_bytes(
            raw,
            api_name=api_name,
            expected_fields=expected_fields,
            strict_decimal_decode=True,
        )


def _capture(root: Path, *, reachable: bool = False) -> None:
    plan = build_successor_support_plan(
        support_start=TAINT_AVAILABILITY,
        target_date=TARGET,
        open_sessions=(TARGET,),
        symbols=(SYMBOL,),
        canonical_subject_scope_authority_sha256="d" * 64,
    )
    acquire_successor_support(
        plan=plan,
        client=_Client(reachable=reachable),
        fileset_root=root,
        captured_pointer_bytes={
            "predecessor": b'{"generation":"parent"}\n',
            "market": b'{"snapshot":"target"}\n',
            "pit": b'{"generation":"pit"}\n',
        },
        immutable_refs={"fixture": {"sha256": "c" * 64}},
        implementation_sha256="a" * 64,
        captured_at="2026-08-15T00:00:00Z",
        max_attempts=1,
        retry_backoff_seconds=(),
        requests_per_second=8.0,
        sleeper=lambda _seconds: None,
    )


def _authorities(tmp_path: Path, *, reachable: bool = False) -> tuple[Path, ...]:
    root = tmp_path / "capture"
    _capture(root, reachable=reachable)
    tables = load_capture_support_tables(root)
    trace = replay_successor_event_trace(
        financial_rows={
            table: tables[table].loc[
                tables[table]["ts_code"].eq(SYMBOL)
            ].to_dict("records")
            for table in FINANCIAL_TABLES
        },
        symbol=SYMBOL,
        parent_cutoff=PARENT,
        target_cutoff=TARGET,
        support_start=TAINT_AVAILABILITY,
    )
    boundary = dict(trace["boundary_winner"])
    boundary["trade_date"] = PARENT
    parent_daily = tmp_path / "fundamental_daily.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame([boundary])), parent_daily)
    parent_period = tmp_path / "fundamental_period.parquet"
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                [
                    {
                        "ts_code": SYMBOL,
                        "end_date": TAINT_END,
                        "availability_date": pd.Timestamp("2022-01-02"),
                        "fin_debt_to_assets": 0.5,
                    }
                ]
            )
        ),
        parent_period,
    )
    membership = tmp_path / "membership.parquet"
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(
                [
                    {
                        "symbol": SYMBOL,
                        "name": "弘业期货",
                        "list_date": "20220805",
                        "effective_from": "20220805",
                        "effective_to": "",
                        "source_list_status": "L",
                    }
                ]
            )
        ),
        membership,
    )
    return root, parent_period, parent_daily, membership


def _analyze(tmp_path: Path, *, reachable: bool = False) -> dict[str, Any]:
    root, period, daily, membership = _authorities(
        tmp_path,
        reachable=reachable,
    )
    return analyze_deferred_fundamental_taints(
        fileset_root=root,
        parent_period_path=period,
        parent_daily_path=daily,
        membership_path=membership,
        parent_cutoff=PARENT,
        target_cutoff=TARGET,
        support_start=TAINT_AVAILABILITY,
        authority_bindings={
            "predecessor": {"sha256": "1" * 64},
            "market": {"sha256": "2" * 64},
            "pit": {"sha256": "3" * 64},
        },
    )


def test_prelisting_unpaired_observation_is_target_non_reachable(
    tmp_path: Path,
) -> None:
    result = _analyze(tmp_path)
    report = result["report"]
    proof = report["proofs"][0]
    assert report["taint_analysis_status"] == "PASS", proof
    assert report["deferred_observation_count"] == 1
    assert proof["symbol"] == SYMBOL
    assert proof["observation_state"] == "TAINTED_NON_REACHABLE"
    assert proof["historical_derived_row_still_present"] is True
    assert proof["current_provider_comp_type7_authority_accepted"] is False
    assert proof["tainted_source_row_entered_suffix"] is False
    assert proof["tainted_state_reachable_through_target"] is False
    assert report["staging_eligible"] is False
    assert report["promotion_eligible"] is False
    assert report["usable_for_investment_research"] is False


def test_post_seam_same_period_event_blocks_taint_proof(tmp_path: Path) -> None:
    result = _analyze(tmp_path, reachable=True)
    report = result["report"]
    proof = report["proofs"][0]
    assert report["taint_analysis_status"] == "BLOCKED"
    assert proof["observation_state"] == "BLOCKING_UNKNOWN"
    assert "TAINT_STATE_REACHABLE_POST_SEAM" in proof["blocking_reasons"]


def test_production_modules_contain_no_symbol_or_date_allowlist() -> None:
    for name in (
        "fundamental_taint.py",
        "fundamental_successor_source.py",
        "fundamental_successor.py",
    ):
        text = (
            Path(__file__).parents[2]
            / "quant_investor"
            / "market"
            / name
        ).read_text(encoding="utf-8")
        assert SYMBOL not in text
        assert TAINT_AVAILABILITY not in text


def test_taint_proof_tamper_is_rejected(tmp_path: Path) -> None:
    result = _analyze(tmp_path)
    result["report"]["proofs"][0]["blocking_reasons"] = ["RESEALED_FAKE"]
    with pytest.raises(FundamentalTaintError, match="TAINT_RECEIPT_SEAL_INVALID"):
        validate_taint_analysis_result(result)
