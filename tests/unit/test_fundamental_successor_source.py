from __future__ import annotations

from collections.abc import Mapping, Sequence
import base64
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import pyarrow.parquet as pq
import quant_investor.market.fundamental_successor_source as source_module

from quant_investor.market.fundamental_successor_source import (
    FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA,
    FundamentalSuccessorSourceError,
    SUCCESSOR_ENDPOINT_CAPABILITIES,
    acquire_successor_support,
    build_successor_support_plan,
    capture_successor_support_evidence,
    load_capture_support_tables,
    load_unsupported_inventory,
    open_capture_support_tables,
    replay_successor_support_requests,
    successor_support_evidence_paths,
    validate_successor_failure_evidence,
    validate_successor_capture_fileset,
    validate_successor_support_fileset,
)
from quant_investor.market.fundamental_provider_contract import frame_fingerprint
from quant_investor.v17_v4_runtime.tushare_https import (
    TushareResponse,
    replay_tushare_response_bytes,
)

TARGET = "20260807"
CAPTURED_AT = "2026-08-07T17:00:00Z"
IMPLEMENTATION_SHA256 = "a" * 64
OPAQUE_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "fundamental_successor_balancesheet_20190819_comp_type7.json"
)
OPAQUE_471_FIXTURE_PATH = (
    Path(__file__).parents[1]
    / "fixtures"
    / "fundamental_successor_balancesheet_20190828_comp_type7_471.json"
)
SYMBOLS = ["600000.SH", "000001.SZ"]
ENDPOINT_TABLE = {
    "balancesheet_vip": "balancesheet",
    "cashflow_vip": "cashflow",
    "daily_basic": "daily_basic",
    "fina_indicator_vip": "fina_indicator",
    "forecast_vip": "forecast",
    "income_vip": "income",
}


def _plan(
    *,
    support_start: str = TARGET,
    target: str = TARGET,
    open_sessions: Sequence[str] = (TARGET,),
    income_support_dependencies: Sequence[Mapping[str, Any]] = (),
    financial_support_dependencies: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return build_successor_support_plan(
        support_start=support_start,
        target_date=target,
        open_sessions=open_sessions,
        symbols=SYMBOLS,
        canonical_subject_scope_authority_sha256="d" * 64,
        income_support_dependencies=income_support_dependencies,
        financial_support_dependencies=financial_support_dependencies,
    )


def _row(
    *,
    api_name: str,
    params: Mapping[str, Any],
    fields: Sequence[str],
) -> dict[str, Any]:
    table = ENDPOINT_TABLE[api_name]
    if table == "daily_basic":
        availability = params["trade_date"]
    elif table in {"fina_indicator", "forecast"}:
        availability = params["ann_date"]
    else:
        availability = params["start_date"]
    values: dict[str, Any] = {
        "ts_code": "000001.SZ",
        "ann_date": availability,
        "f_ann_date": availability,
        "end_date": "20260630",
        "trade_date": availability,
        "report_type": "1",
        "comp_type": "1",
        "update_flag": "1",
        "type": "预增",
        "summary": "expected improvement",
        "change_reason": "operations",
    }
    for field in fields:
        values.setdefault(field, Decimal("1.25"))
    return values


RowFactory = Callable[[str, Mapping[str, Any], Sequence[str]], Sequence[Mapping[str, Any]]]


class FakeClient:
    def __init__(
        self,
        *,
        row_factory: RowFactory | None = None,
        has_more: bool = False,
        request_id_prefix: str = "request",
        reverse_rows: bool = False,
    ) -> None:
        self.calls = 0
        self.row_factory = row_factory
        self.has_more = has_more
        self.request_id_prefix = request_id_prefix
        self.reverse_rows = reverse_rows

    def request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ) -> TushareResponse:
        self.calls += 1
        factory = self.row_factory
        rows = (
            list(factory(api_name, params, expected_fields))
            if factory is not None
            else [_row(api_name=api_name, params=params, fields=expected_fields)]
        )
        if self.reverse_rows:
            rows.reverse()
        physical = tuple(tuple(row[field] for field in expected_fields) for row in rows)
        request_id = f"{self.request_id_prefix}-{self.calls}"

        def json_value(value: Any) -> Any:
            if isinstance(value, Decimal):
                return int(value) if value == value.to_integral_value() else float(value)
            return value

        raw_body = _canonical_bytes(
            {
                "code": 0,
                "data": {
                    "count": len(physical),
                    "fields": list(expected_fields),
                    "has_more": self.has_more,
                    "items": [
                        [json_value(value) for value in row] for row in physical
                    ],
                },
                "detail": "",
                "msg": "",
                "request_id": request_id,
            }
        )
        return replay_tushare_response_bytes(
            raw_body,
            api_name=api_name,
            expected_fields=expected_fields,
        )


class ExactOpaqueBalancesheetClient:
    def __init__(self, *, reverse_fixture_rows: bool = False) -> None:
        fixture_bytes = OPAQUE_FIXTURE_PATH.read_bytes()
        fixture = json.loads(fixture_bytes)
        if reverse_fixture_rows:
            fixture["data"]["items"].reverse()
            fixture_bytes = _canonical_bytes(fixture)
        self.fixture_bytes = fixture_bytes
        self.delegate = FakeClient(row_factory=self._support_row)

    @staticmethod
    def _support_row(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        if ENDPOINT_TABLE[api_name] != "daily_basic":
            row["end_date"] = "20190630"
        return [row]

    def request(
        self,
        *,
        api_name: str,
        params: Mapping[str, Any],
        expected_fields: Sequence[str],
    ) -> TushareResponse:
        if api_name == "balancesheet_vip":
            assert params == {"end_date": "20190819", "start_date": "20190819"}
            return replay_tushare_response_bytes(
                self.fixture_bytes,
                api_name=api_name,
                expected_fields=expected_fields,
                strict_decimal_decode=True,
            )
        return self.delegate.request(
            api_name=api_name,
            params=params,
            expected_fields=expected_fields,
        )


def _acquire(
    root: Path,
    *,
    client: Any | None = None,
    plan: Mapping[str, Any] | None = None,
    immutable_refs: Mapping[str, Any] | None = None,
    pointers: Mapping[str, bytes] | None = None,
    max_attempts: int = 1,
) -> dict[str, Any]:
    return acquire_successor_support(
        plan=plan or _plan(),
        client=client or FakeClient(),
        fileset_root=root,
        captured_pointer_bytes=pointers
        or {
            "predecessor": b'{"schema_version":"v3","generation":"old"}\n',
            "market": b'{"snapshot":"20260807T170000Z"}\n',
            "pit": b'{"generation":"pit-20260807"}\n',
        },
        immutable_refs=immutable_refs
        or {
            "parent_metadata": {
                "schema_version": "myquant-fundamental-provider-manifest.v3",
                "manifest_sha256": "b" * 64,
            },
            "scope_ref": {"sha256": "c" * 64},
        },
        implementation_sha256=IMPLEMENTATION_SHA256,
        captured_at=CAPTURED_AT,
        max_attempts=max_attempts,
        retry_backoff_seconds=tuple(0.0 for _ in range(max_attempts - 1)),
        requests_per_second=8.0,
        sleeper=lambda _seconds: None,
    )


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


def _opaque_fixture_plan() -> dict[str, Any]:
    fixture = json.loads(OPAQUE_FIXTURE_PATH.read_text(encoding="utf-8"))
    symbols = sorted(
        {str(row[0]) for row in fixture["data"]["items"]}
        | {"000001.SZ"}
    )
    return build_successor_support_plan(
        support_start="20190819",
        target_date="20190819",
        open_sessions=("20190819",),
        symbols=symbols,
        canonical_subject_scope_authority_sha256="d" * 64,
    )


def test_plan_is_permutation_invariant_and_forecast_is_exact_announcement() -> None:
    plan = _plan(
        support_start="20260807",
        target="20260808",
        open_sessions=("20260808", "20260807"),
    )
    permuted = build_successor_support_plan(
        support_start="20260807",
        target_date="20260808",
        open_sessions=("20260807", "20260808"),
        symbols=tuple(reversed(SYMBOLS)),
        canonical_subject_scope_authority_sha256="d" * 64,
    )

    assert plan == permuted
    assert plan["planned_request_count"] == 12
    requests = replay_successor_support_requests(plan)
    assert [row["ordinal"] for row in requests] == list(range(12))
    forecast = [row for row in requests if row["table"] == "forecast"]
    assert [row["params"] for row in forecast] == [
        {"ann_date": "20260807"},
        {"ann_date": "20260808"},
    ]
    statements = [row for row in requests if row["table"] in {"income", "cashflow", "balancesheet"}]
    assert all(row["params"]["start_date"] == row["params"]["end_date"] for row in statements)


def test_capability_matrix_and_f_ann_date_selection_are_production_strict(
    tmp_path: Path,
) -> None:
    statement_fields = SUCCESSOR_ENDPOINT_CAPABILITIES["income"]["expected_fields"]
    assert {"report_type", "comp_type", "update_flag"}.issubset(statement_fields)
    assert "update_flag" in SUCCESSOR_ENDPOINT_CAPABILITIES["fina_indicator"]["expected_fields"]
    assert not {
        "update_flag",
        "report_type",
        "comp_type",
    }.intersection(SUCCESSOR_ENDPOINT_CAPABILITIES["forecast"]["expected_fields"])

    def final_announcement(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        if ENDPOINT_TABLE[api_name] in {"balancesheet", "cashflow", "income"}:
            row["ann_date"] = params["start_date"]
            row["f_ann_date"] = "20260808"
            if params["start_date"] == "20260807":
                row["end_date"] = "20260331"
        return [row]

    manifest = _acquire(
        tmp_path / "fileset",
        client=FakeClient(row_factory=final_announcement),
        plan=_plan(
            support_start="20260807",
            target="20260808",
            open_sessions=("20260807", "20260808"),
        ),
    )
    tables = load_capture_support_tables(tmp_path / "fileset")

    assert manifest["schema_version"] == FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA
    delayed = tables["income"].loc[tables["income"]["ann_date"] == "20260807"]
    assert delayed.iloc[0]["availability_date"] == "20260808"
    assert manifest["provider_accounting"]["requests_failed"] == 0
    assert manifest["provider_accounting"]["malformed_requests"] == 0


def test_statement_partition_uses_ann_date_not_selected_availability(
    tmp_path: Path,
) -> None:
    def wrong_partition(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        if ENDPOINT_TABLE[api_name] in {"balancesheet", "cashflow", "income"}:
            row["ann_date"] = "20260806"
            row["f_ann_date"] = params["start_date"]
        return [row]

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_RESPONSE_PARTITION_SCOPE_MISMATCH",
    ):
        _acquire(
            tmp_path / "wrong-partition",
            client=FakeClient(row_factory=wrong_partition),
        )


def test_exact_income_support_dependency_is_bounded_and_replayable(
    tmp_path: Path,
) -> None:
    dependency = {"ts_code": "000001.SZ", "end_date": "20250630"}
    plan = _plan(income_support_dependencies=(dependency,))
    requests = replay_successor_support_requests(plan)
    assert requests[-1]["partition_type"] == "EXACT_SYMBOL_REPORT_PERIOD_SUPPORT"
    assert requests[-1]["params"] == {
        "period": "20250630",
        "ts_code": "000001.SZ",
    }

    def exact_period(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        if params.get("period"):
            row = _row(
                api_name=api_name,
                params={"end_date": "20250830", "start_date": "20250830"},
                fields=fields,
            )
            row.update(
                {
                    "ts_code": params["ts_code"],
                    "end_date": params["period"],
                }
            )
            return [row]
        return [_row(api_name=api_name, params=params, fields=fields)]

    root = tmp_path / "bounded-income-support"
    manifest = _acquire(root, client=FakeClient(row_factory=exact_period), plan=plan)
    income = load_capture_support_tables(root)["income"]

    assert manifest["provider_accounting"]["requests_terminal"] == len(requests)
    assert (
        (income["ts_code"] == "000001.SZ")
        & (income["end_date"] == "20250630")
    ).any()


def test_financial_support_plan_allows_only_statement_fallback_lanes() -> None:
    plan = _plan(
        financial_support_dependencies=(
            {
                "table": "cashflow",
                "ts_code": "000001.SZ",
                "end_date": "20260630",
            },
        )
    )
    request = replay_successor_support_requests(plan)[-1]
    assert request["table"] == "cashflow"
    assert request["params"] == {
        "period": "20260630",
        "ts_code": "000001.SZ",
    }

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_FINANCIAL_SUPPORT_TABLE_INVALID",
    ):
        _plan(
            financial_support_dependencies=(
                {
                    "table": "fina_indicator",
                    "ts_code": "000001.SZ",
                    "end_date": "20260630",
                },
            )
        )


def test_exact_income_support_dependency_rejects_cross_symbol_response(
    tmp_path: Path,
) -> None:
    plan = _plan(
        income_support_dependencies=(
            {"ts_code": "000001.SZ", "end_date": "20250630"},
        )
    )

    def wrong_symbol(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        if params.get("period"):
            row = _row(
                api_name=api_name,
                params={"end_date": "20250830", "start_date": "20250830"},
                fields=fields,
            )
            row.update({"ts_code": "600000.SH", "end_date": params["period"]})
            return [row]
        return [_row(api_name=api_name, params=params, fields=fields)]

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_RESPONSE_PARTITION_SCOPE_MISMATCH",
    ):
        _acquire(tmp_path / "cross-symbol-support", client=FakeClient(row_factory=wrong_symbol), plan=plan)


def test_fina_indicator_exact_announcement_captures_pre_anchor_report_period(
    tmp_path: Path,
) -> None:
    def prior_period(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        if ENDPOINT_TABLE[api_name] == "fina_indicator":
            row["end_date"] = "20181231"
        return [row]

    root = tmp_path / "prior-period"
    manifest = _acquire(root, client=FakeClient(row_factory=prior_period))
    fina = load_capture_support_tables(root)["fina_indicator"]

    assert fina.iloc[0]["end_date"] == "20181231"
    assert manifest["binding"]["plan"]["report_period_envelope"] == {
        "capture_policy": "ALL_REPORT_PERIODS_FOR_EXACT_ANNOUNCEMENT_DATE",
        "lower_bound": None,
        "upper_bound_policy": "END_DATE_NOT_AFTER_AVAILABILITY",
    }


def test_observation_payload_and_logical_identities_are_separate(tmp_path: Path) -> None:
    def duplicated(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        return [row, dict(row)]

    first = _acquire(
        tmp_path / "first",
        client=FakeClient(
            row_factory=duplicated,
            request_id_prefix="observation-a",
        ),
    )
    second = _acquire(
        tmp_path / "second",
        client=FakeClient(
            row_factory=duplicated,
            request_id_prefix="observation-b",
        ),
    )

    first_receipt = first["request_receipts"][0]
    second_receipt = second["request_receipts"][0]
    assert first_receipt["observation_sha256"] != second_receipt["observation_sha256"]
    assert first_receipt["payload_sha256"] == second_receipt["payload_sha256"]
    assert first_receipt["logical_sha256"] == second_receipt["logical_sha256"]
    assert (
        len(
            {
                first_receipt["observation_sha256"],
                first_receipt["payload_sha256"],
                first_receipt["logical_sha256"],
            }
        )
        == 3
    )


def test_global_response_scope_is_preserved_before_frozen_subject_projection(
    tmp_path: Path,
) -> None:
    def global_partition(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        in_scope = _row(api_name=api_name, params=params, fields=fields)
        out_of_scope = dict(in_scope)
        out_of_scope["ts_code"] = "999999.SH"
        return [in_scope, out_of_scope]

    root = tmp_path / "fileset"
    manifest = _acquire(root, client=FakeClient(row_factory=global_partition))
    tables = load_capture_support_tables(root)

    assert manifest["provider_accounting"]["scope_excluded_rows"] == 6
    assert manifest["provider_accounting"]["scope_exclusion_requests"] == 6
    assert manifest["scope_projection"]["out_of_scope_observation_count"] == 6
    assert all(set(frame["ts_code"]) == {"000001.SZ"} for frame in tables.values())
    assert b"999999.SH" in (root / "requests" / "000000.json").read_bytes()

    record_path = root / "requests" / "000000.json"
    record = json.loads(record_path.read_text())
    receipt = dict(record["receipt"])
    receipt.pop("receipt_sha256")
    receipt["out_of_scope_observation_count"] += 1
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    record["receipt"] = receipt
    record.pop("record_sha256")
    record["record_sha256"] = hashlib.sha256(_canonical_bytes(record)).hexdigest()
    record_payload = _canonical_bytes(record)
    record_path.write_bytes(record_payload)

    manifest_path = root / "provider_manifest.json"
    forged_manifest = json.loads(manifest_path.read_text())
    forged_manifest["request_receipts"][0] = receipt
    forged_manifest["record_files"][0]["byte_length"] = len(record_payload)
    forged_manifest["record_files"][0]["sha256"] = hashlib.sha256(
        record_payload
    ).hexdigest()
    forged_manifest.pop("manifest_sha256")
    forged_manifest["manifest_sha256"] = hashlib.sha256(
        _canonical_bytes(forged_manifest)
    ).hexdigest()
    manifest_path.write_bytes(_canonical_bytes(forged_manifest))

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_RECEIPT_SCOPE_IDENTITY_MISMATCH",
    ):
        validate_successor_capture_fileset(root)


@pytest.mark.parametrize(
    "external_symbol",
    ["A21651.SH", "X25192.BJ", "833243!1.BJ"],
)
def test_provider_external_symbol_is_exact_out_of_scope_evidence_only(
    tmp_path: Path,
    external_symbol: str,
) -> None:
    def provisional_partition(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        in_scope = _row(api_name=api_name, params=params, fields=fields)
        provisional = dict(in_scope)
        provisional["ts_code"] = external_symbol
        return [in_scope, provisional]

    root = tmp_path / hashlib.sha256(external_symbol.encode()).hexdigest()
    manifest = _acquire(
        root,
        client=FakeClient(row_factory=provisional_partition),
    )
    tables = load_capture_support_tables(root)

    assert manifest["provider_accounting"]["scope_excluded_rows"] == 6
    assert all(set(frame["ts_code"]) == {"000001.SZ"} for frame in tables.values())
    record = json.loads((root / "requests" / "000000.json").read_text())
    assert any(
        encoded[0] == {"kind": "text", "value": external_symbol}
        for encoded in record["observed_rows"]
    )
    assert all(
        encoded[0] != {"kind": "text", "value": external_symbol}
        for encoded in record["rows"]
    )


@pytest.mark.parametrize(
    "provider_symbol",
    [
        " A21651.SH",
        "a21651.sh",
        "A2165.SH",
        "AA2165.SH",
        "833243!0.BJ",
        "833243!.BJ",
        "833243!1000.BJ",
    ],
)
def test_provider_external_symbol_alias_or_unknown_form_blocks(
    tmp_path: Path,
    provider_symbol: str,
) -> None:
    def invalid_provider_symbol(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        row["ts_code"] = provider_symbol
        return [row]

    with pytest.raises(FundamentalSuccessorSourceError):
        _acquire(
            tmp_path / hashlib.sha256(provider_symbol.encode()).hexdigest(),
            client=FakeClient(row_factory=invalid_provider_symbol),
        )


def test_out_of_scope_material_conflict_is_evidence_not_a_canonical_winner(
    tmp_path: Path,
) -> None:
    def global_conflict(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        in_scope = _row(api_name=api_name, params=params, fields=fields)
        out_one = dict(in_scope)
        out_one["ts_code"] = "999999.SH"
        if ENDPOINT_TABLE[api_name] != "income":
            return [in_scope, out_one]
        out_two = dict(out_one)
        out_two["n_income"] = Decimal("999")
        return [in_scope, out_one, out_two]

    root = tmp_path / "fileset"
    manifest = _acquire(root, client=FakeClient(row_factory=global_conflict))
    income = load_capture_support_tables(root)["income"]

    assert len(income) == 1
    assert income.iloc[0]["ts_code"] == "000001.SZ"
    assert manifest["provider_accounting"]["scope_excluded_rows"] == 7
    assert manifest["provider_accounting"]["requests_failed"] == 0


def _dominance_rows(
    api_name: str,
    params: Mapping[str, Any],
    fields: Sequence[str],
) -> Sequence[Mapping[str, Any]]:
    base = _row(api_name=api_name, params=params, fields=fields)
    if ENDPOINT_TABLE[api_name] != "income":
        return [base]
    old = dict(base)
    old["update_flag"] = "0"
    old["n_income"] = Decimal("1")
    latest_one = dict(base)
    latest_one["update_flag"] = "1"
    latest_one["n_income"] = Decimal("2")
    latest_two = dict(latest_one)
    latest_two["comp_type"] = "2"
    return [old, latest_one, latest_two]


def test_update_dominance_projection_collapse_and_response_permutation(
    tmp_path: Path,
) -> None:
    first = _acquire(
        tmp_path / "first",
        client=FakeClient(row_factory=_dominance_rows),
    )
    second = _acquire(
        tmp_path / "second",
        client=FakeClient(row_factory=_dominance_rows, reverse_rows=True),
    )
    first_table = load_capture_support_tables(tmp_path / "first")["income"]
    second_table = load_capture_support_tables(tmp_path / "second")["income"]

    assert first["manifest_sha256"] != second["manifest_sha256"]
    assert first["table_fingerprints"] == second["table_fingerprints"]
    assert first_table.equals(second_table)
    assert len(first_table) == 1
    assert first_table.iloc[0]["update_flag"] == "1"
    assert first_table.iloc[0]["comp_type"] == "1"
    assert first_table.iloc[0]["n_income"] == Decimal("2")
    income_receipt = next(row for row in first["request_receipts"] if row["table"] == "income")
    assert income_receipt["canonicalization_counters"] == {
        "exact_duplicates_collapsed": 0,
        "projection_equivalent_duplicates_collapsed": 1,
        "superseded_updates_discarded": 1,
        "deferred_opaque_observations": 0,
    }


def test_exact_comp_type_seven_fixture_is_opaque_equivalence_only(
    tmp_path: Path,
) -> None:
    fixture_bytes = OPAQUE_FIXTURE_PATH.read_bytes()
    fixture = json.loads(fixture_bytes)
    assert len(fixture_bytes) == 2_419
    assert hashlib.sha256(fixture_bytes).hexdigest() == (
        "a7c87103025b434432fab4a02e1ac48e008c9a20fbd65d7c2bcb86c11da730cb"
    )
    assert fixture["data"]["fields"] == [
        "ts_code",
        "ann_date",
        "f_ann_date",
        "end_date",
        "total_liab",
        "total_assets",
        "update_flag",
        "report_type",
        "comp_type",
    ]
    assert len(fixture["data"]["items"]) == 25
    assert fixture["data"]["count"] == 0
    assert fixture["data"]["has_more"] is False
    plan = _opaque_fixture_plan()
    first = _acquire(
        tmp_path / "fixture-order",
        plan=plan,
        client=ExactOpaqueBalancesheetClient(),
    )
    second = _acquire(
        tmp_path / "reverse-order",
        plan=plan,
        client=ExactOpaqueBalancesheetClient(reverse_fixture_rows=True),
    )
    first_table = load_capture_support_tables(tmp_path / "fixture-order")[
        "balancesheet"
    ]
    second_table = load_capture_support_tables(tmp_path / "reverse-order")[
        "balancesheet"
    ]
    first_receipt = next(
        row for row in first["request_receipts"]
        if row["table"] == "balancesheet"
    )
    second_receipt = next(
        row for row in second["request_receipts"]
        if row["table"] == "balancesheet"
    )

    assert first["table_fingerprints"] == second["table_fingerprints"]
    assert first_table.equals(second_table)
    assert first_receipt["canonicalization_counters"] == {
        "exact_duplicates_collapsed": 0,
        "projection_equivalent_duplicates_collapsed": 4,
        "superseded_updates_discarded": 3,
        "deferred_opaque_observations": 0,
    }
    assert first_receipt["raw_response_byte_length"] == len(fixture_bytes)
    assert first_receipt["raw_response_sha256"] == hashlib.sha256(
        fixture_bytes
    ).hexdigest()
    opaque = first_receipt["opaque_comp_type_evidence"]
    assert opaque["opaque_comp_type_observation_count"] == 4
    assert opaque["opaque_comp_type_business_key_count"] == 4
    assert opaque["opaque_unpaired_count"] == 0
    assert opaque["opaque_material_conflict_count"] == 0
    accounting = first["provider_accounting"]["opaque_comp_type"]
    assert accounting["opaque_comp_type_observation_count"] == 4
    assert accounting["opaque_comp_type_business_key_count"] == 4
    assert accounting["opaque_unpaired_count"] == 0
    assert accounting["opaque_material_conflict_count"] == 0

    canonical = first_table[first_table["ts_code"] == "002961.SZ"]
    assert set(canonical["end_date"]) == {
        "20161231",
        "20171231",
        "20181231",
        "20190630",
    }
    assert set(canonical["comp_type"]) == {"2"}
    assert dict(zip(canonical["end_date"], canonical["update_flag"], strict=True)) == {
        "20161231": "1",
        "20171231": "1",
        "20181231": "1",
        "20190630": "0",
    }
    expected_ratios = {
        "20161231": 0.7873876250652059,
        "20171231": 0.7497458402964471,
        "20181231": 0.6841780690952511,
        "20190630": 0.726604351077235,
    }
    observed_ratios = {
        str(row.end_date): float(row.total_liab / row.total_assets)
        for row in canonical.itertuples(index=False)
    }
    assert observed_ratios == pytest.approx(expected_ratios)

    invariant_fields = (
        "full_response_observation_multiset_sha256",
        "in_scope_canonical_payload_multiset_sha256",
        "in_scope_observation_multiset_sha256",
        "logical_sha256",
        "observation_sha256",
        "payload_sha256",
    )
    assert all(
        first_receipt[field] == second_receipt[field]
        for field in invariant_fields
    )
    assert (
        first_receipt["raw_item_order_sha256"]
        != second_receipt["raw_item_order_sha256"]
    )


def test_exact_471_row_blocker_is_captured_as_one_deferred_observation() -> None:
    fixture_bytes = OPAQUE_471_FIXTURE_PATH.read_bytes()
    fixture = json.loads(fixture_bytes)
    fields = fixture["data"]["fields"]
    symbols = sorted({str(row[0]) for row in fixture["data"]["items"]})
    assert len(fixture_bytes) == 41_026
    assert hashlib.sha256(fixture_bytes).hexdigest() == (
        "c3bf8cc8e3f3e65eb309a7256d5ae7b73ebbeb5674412587f4ae448cb42c1f8a"
    )
    assert len(fixture["data"]["items"]) == 471
    assert fixture["data"]["count"] == 0
    assert fixture["data"]["has_more"] is False

    class Exact471Client:
        def __init__(self) -> None:
            self.calls = 0

        def request(
            self,
            *,
            api_name: str,
            params: Mapping[str, Any],
            expected_fields: Sequence[str],
        ) -> TushareResponse:
            self.calls += 1
            if api_name == "balancesheet_vip":
                assert params == {
                    "end_date": "20190828",
                    "start_date": "20190828",
                }
                assert list(expected_fields) == fields
                return replay_tushare_response_bytes(
                    fixture_bytes,
                    api_name=api_name,
                    expected_fields=expected_fields,
                    strict_decimal_decode=True,
                )
            empty = _canonical_bytes(
                {
                    "code": 0,
                    "data": {
                        "count": 0,
                        "fields": list(expected_fields),
                        "has_more": False,
                        "items": [],
                    },
                    "detail": "",
                    "msg": "",
                    "request_id": f"empty-{self.calls}",
                }
            )
            return replay_tushare_response_bytes(
                empty,
                api_name=api_name,
                expected_fields=expected_fields,
                strict_decimal_decode=True,
            )

    plan = build_successor_support_plan(
        support_start="20190828",
        target_date="20190828",
        open_sessions=("20190828",),
        symbols=symbols,
        canonical_subject_scope_authority_sha256="e" * 64,
    )
    request = next(
        row
        for row in replay_successor_support_requests(plan)
        if row["table"] == "balancesheet"
    )
    plan_for_replay = dict(plan)
    plan_for_replay["target_date"] = "20200821"
    receipt, _observed, logical, raw = source_module._fetch_request(
        plan=plan_for_replay,
        request=request,
        client=Exact471Client(),
        symbols=frozenset(symbols),
        max_attempts=1,
        retry_backoff_seconds=(),
        pacer=source_module._Pacer(
            8.0,
            sleeper=lambda _seconds: None,
            monotonic=lambda: 0.0,
        ),
        sleeper=lambda _seconds: None,
    )
    inventory = source_module._unsupported_inventory([receipt])
    assert raw == fixture_bytes
    assert len(logical) == 468
    assert receipt["raw_response_sha256"] == hashlib.sha256(
        fixture_bytes
    ).hexdigest()
    assert receipt["raw_response_byte_length"] == 41_026
    assert receipt["item_count"] == 471
    assert receipt["provider_reported_count"] == 0
    assert receipt["has_more"] is False
    assert receipt["classification_partition"]["classification_counts"] == {
        "authoritative_supported": 470,
        "opaque_equivalent": 0,
        "out_of_scope_excluded": 0,
        "source_blocking": 0,
        "tainted_deferred": 1,
    }
    assert inventory["deferred_observation_count"] == 1
    assert inventory["entries"][0]["business_key"] == [
        "001236.SZ",
        "20190630",
        "20190828",
    ]


@pytest.mark.parametrize("mode", ["value", "update_winner"])
def test_opaque_comp_type_requires_supported_projection_equivalent_peer(
    tmp_path: Path,
    mode: str,
) -> None:
    def rows(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        base = _row(api_name=api_name, params=params, fields=fields)
        if ENDPOINT_TABLE[api_name] != "balancesheet":
            return [base]
        supported = dict(base)
        supported["comp_type"] = "2"
        opaque = dict(base)
        opaque["comp_type"] = "7"
        if mode == "value":
            opaque["total_assets"] = Decimal("999")
            return [supported, opaque]
        opaque["update_flag"] = "0"
        opaque_latest = dict(opaque)
        opaque_latest["update_flag"] = "1"
        opaque_latest["total_assets"] = Decimal("999")
        return [supported, opaque, opaque_latest]

    root = tmp_path / mode
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_OPAQUE_COMP_TYPE_EQUIVALENCE_UNCLOSED",
    ):
        _acquire(root, client=FakeClient(row_factory=rows))
    assert not (root / "provider_manifest.json").exists()


def test_unpaired_opaque_balancesheet_is_deferred_not_authoritative(
    tmp_path: Path,
) -> None:
    def rows(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        base = _row(api_name=api_name, params=params, fields=fields)
        if ENDPOINT_TABLE[api_name] == "balancesheet":
            base["comp_type"] = "7"
        return [base]

    root = tmp_path / "unpaired"
    manifest = _acquire(root, client=FakeClient(row_factory=rows))
    inventory = load_unsupported_inventory(root)
    assert manifest["authority_state"] == "DEFERRED_UNSUPPORTED_OBSERVATIONS"
    assert manifest["authoritative_source_ready"] is False
    assert inventory["deferred_observation_count"] == 1
    assert inventory["entries"][0]["classification"] == (
        "TAINTED_PENDING_ANALYSIS"
    )
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_DEFERRED_CAPTURE_NOT_AUTHORITATIVE",
    ):
        validate_successor_support_fileset(root)


def test_opaque_comp_type_availability_mismatch_is_deferred(
    tmp_path: Path,
) -> None:
    plan = build_successor_support_plan(
        support_start="20260807",
        target_date="20260808",
        open_sessions=("20260807", "20260808"),
        symbols=SYMBOLS,
        canonical_subject_scope_authority_sha256="d" * 64,
    )

    def rows(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        base = _row(api_name=api_name, params=params, fields=fields)
        if (
            ENDPOINT_TABLE[api_name] != "balancesheet"
            or params["start_date"] != "20260807"
        ):
            return [base]
        supported = dict(base)
        supported["comp_type"] = "2"
        opaque = dict(base)
        opaque["comp_type"] = "7"
        opaque["f_ann_date"] = "20260808"
        return [supported, opaque]

    root = tmp_path / "availability"
    _acquire(root, plan=plan, client=FakeClient(row_factory=rows))
    inventory = load_unsupported_inventory(root)
    assert inventory["deferred_observation_count"] == 1
    assert inventory["entries"][0]["business_key"][-1] == "20260808"


@pytest.mark.parametrize("table", ["income", "cashflow"])
def test_opaque_comp_type_is_not_authorized_for_other_statements(
    tmp_path: Path,
    table: str,
) -> None:
    def rows(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        if ENDPOINT_TABLE[api_name] == table:
            row["comp_type"] = "7"
        return [row]

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_STATEMENT_PHYSICAL_CLASS_INVALID",
    ):
        _acquire(tmp_path / table, client=FakeClient(row_factory=rows))


def test_material_duplicate_conflict_blocks_without_manifest(tmp_path: Path) -> None:
    def conflict(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        first = _row(api_name=api_name, params=params, fields=fields)
        if ENDPOINT_TABLE[api_name] != "income":
            return [first]
        second = dict(first)
        second["n_income"] = Decimal("999")
        return [first, second]

    root = tmp_path / "fileset"
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_MATERIAL_DUPLICATE_CONFLICT",
    ):
        _acquire(root, client=FakeClient(row_factory=conflict))
    assert not (root / "provider_manifest.json").exists()


@pytest.mark.parametrize("reverse_rows", [False, True])
def test_lower_rank_cross_comp_type_material_conflict_blocks(
    tmp_path: Path,
    reverse_rows: bool,
) -> None:
    def cross_comp(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        first = _row(api_name=api_name, params=params, fields=fields)
        if ENDPOINT_TABLE[api_name] != "income":
            return [first]
        first["comp_type"] = "1"
        first["update_flag"] = "1"
        first["n_income"] = Decimal("2")
        second = dict(first)
        second["comp_type"] = "2"
        second["update_flag"] = "0"
        second["n_income"] = Decimal("1")
        return [first, second]

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_MATERIAL_DUPLICATE_CONFLICT",
    ):
        _acquire(
            tmp_path / f"cross-comp-{reverse_rows}",
            client=FakeClient(row_factory=cross_comp, reverse_rows=reverse_rows),
        )


@pytest.mark.parametrize("versioned_flag", ["0", "1"])
@pytest.mark.parametrize("reverse_rows", [False, True])
def test_fina_unversioned_material_conflict_with_versioned_row_blocks(
    tmp_path: Path,
    versioned_flag: str,
    reverse_rows: bool,
) -> None:
    def blank_vs_versioned(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        first = _row(api_name=api_name, params=params, fields=fields)
        if ENDPOINT_TABLE[api_name] != "fina_indicator":
            return [first]
        first["update_flag"] = None
        first["roe"] = Decimal("1")
        second = dict(first)
        second["update_flag"] = versioned_flag
        second["roe"] = Decimal("2")
        return [first, second]

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_MATERIAL_DUPLICATE_CONFLICT",
    ):
        _acquire(
            tmp_path / f"fina-{versioned_flag}-{reverse_rows}",
            client=FakeClient(
                row_factory=blank_vs_versioned,
                reverse_rows=reverse_rows,
            ),
        )


def test_has_more_and_terminal_provider_failure_are_hard_blockers(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_PROVIDER_HAS_MORE",
    ):
        _acquire(tmp_path / "has-more", client=FakeClient(has_more=True))

    secret = "TOKEN_MUST_NOT_APPEAR_1234567890"

    class FailingClient:
        calls = 0

        def request(self, **_kwargs: Any) -> TushareResponse:
            self.calls += 1
            raise RuntimeError(f"transport failed with {secret}")

    client = FailingClient()
    root = tmp_path / "failure"
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_PROVIDER_REQUEST_FAILED",
    ) as error:
        _acquire(root, client=client, max_attempts=2)
    assert client.calls == 2
    assert secret not in str(error.value)
    assert all(
        secret.encode() not in path.read_bytes() for path in root.rglob("*") if path.is_file()
    )
    assert not (root / "provider_manifest.json").exists()

    def malformed_numeric(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        numeric_field = next(
            field
            for field in fields
            if field
            not in {
                "ann_date",
                "change_reason",
                "comp_type",
                "end_date",
                "f_ann_date",
                "report_type",
                "summary",
                "trade_date",
                "ts_code",
                "type",
                "update_flag",
            }
        )
        row[numeric_field] = "not-a-number"
        return [row]

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_RESPONSE_FIELD_TYPE_INVALID",
    ):
        _acquire(
            tmp_path / "malformed",
            client=FakeClient(row_factory=malformed_numeric),
        )


def test_invalid_provider_classification_seals_exact_failure_response(
    tmp_path: Path,
) -> None:
    def invalid_classification(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        row["comp_type"] = "5"
        return [row]

    root = tmp_path / "invalid-classification"
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_CLASSIFICATION_VALUE_INVALID",
    ):
        _acquire(
            root,
            client=FakeClient(row_factory=invalid_classification),
        )

    failure_root = tmp_path / "invalid-classification-failures"
    raw_path = failure_root / "000000.raw.json"
    failure_path = failure_root / "000000.failure.json"
    assert raw_path.exists()
    assert failure_path.exists()
    raw_bytes = raw_path.read_bytes()
    failure_bytes = failure_path.read_bytes()
    failure = json.loads(failure_bytes)
    assert failure_bytes == _canonical_bytes(failure)
    assert failure["schema_version"] == (
        "myquant-fundamental-successor-failure-evidence.v1"
    )
    assert failure["status"] == "BLOCKED"
    assert failure["error_code"] == (
        "SUCCESSOR_CLASSIFICATION_VALUE_INVALID"
    )
    assert failure["request"]["ordinal"] == 0
    assert failure["raw_response_ref"] == {
        "path": "000000.raw.json",
        "byte_length": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    replayed = replay_tushare_response_bytes(
        raw_bytes,
        api_name="balancesheet_vip",
        expected_fields=failure["request"]["expected_fields"],
    )
    comp_type_index = replayed.fields.index("comp_type")
    assert replayed.rows[0][comp_type_index] == "5"
    assert validate_successor_failure_evidence(
        failure_root,
        ordinal=0,
    )["failure_sha256"] == failure["failure_sha256"]
    assert not (root / "provider_manifest.json").exists()


@pytest.mark.parametrize("comp_type", ["6", "8", "9", None])
def test_unknown_or_null_statement_comp_type_blocks(
    tmp_path: Path,
    comp_type: str | None,
) -> None:
    def invalid_classification(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        row["comp_type"] = comp_type
        return [row]

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_CLASSIFICATION_VALUE_INVALID",
    ):
        _acquire(
            tmp_path / f"invalid-{comp_type}",
            client=FakeClient(row_factory=invalid_classification),
        )


def test_failure_raw_tamper_is_detected_by_independent_validator(
    tmp_path: Path,
) -> None:
    def invalid_classification(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        row["comp_type"] = "5"
        return [row]

    root = tmp_path / "tampered-failure"
    with pytest.raises(FundamentalSuccessorSourceError):
        _acquire(root, client=FakeClient(row_factory=invalid_classification))
    failure_root = tmp_path / "tampered-failure-failures"
    raw_path = failure_root / "000000.raw.json"
    raw_path.write_bytes(raw_path.read_bytes() + b" ")
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_FAILURE_EVIDENCE_INVALID",
    ):
        validate_successor_failure_evidence(failure_root, ordinal=0)


@pytest.mark.parametrize(
    "mutation",
    [
        "error_code",
        "request_params",
        "request_fields",
        "request_endpoint",
        "request_ordinal",
        "binding",
    ],
)
def test_resealed_failure_claim_must_match_binding_plan_and_replay(
    tmp_path: Path,
    mutation: str,
) -> None:
    def invalid_classification(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        row["comp_type"] = "5"
        return [row]

    root = tmp_path / f"resealed-{mutation}"
    with pytest.raises(FundamentalSuccessorSourceError):
        _acquire(root, client=FakeClient(row_factory=invalid_classification))
    failure_root = tmp_path / f"resealed-{mutation}-failures"
    failure_path = failure_root / "000000.failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    if mutation == "error_code":
        failure["error_code"] = (
            "SUCCESSOR_OPAQUE_COMP_TYPE_EQUIVALENCE_UNCLOSED"
        )
    elif mutation == "request_params":
        failure["request"]["params"]["end_date"] = "20260808"
    elif mutation == "request_fields":
        failure["request"]["expected_fields"].reverse()
    elif mutation == "request_endpoint":
        failure["request"]["endpoint"] = "cashflow_vip"
    elif mutation == "request_ordinal":
        failure["request"]["ordinal"] = 1
    else:
        failure["binding_sha256"] = "b" * 64
    failure.pop("failure_sha256")
    failure["failure_sha256"] = hashlib.sha256(
        _canonical_bytes(failure)
    ).hexdigest()
    failure_path.write_bytes(_canonical_bytes(failure))

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_FAILURE_EVIDENCE_INVALID",
    ):
        validate_successor_failure_evidence(failure_root, ordinal=0)


def test_resealed_failure_with_now_valid_raw_response_is_rejected(
    tmp_path: Path,
) -> None:
    def invalid_classification(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        row["comp_type"] = "5"
        return [row]

    root = tmp_path / "now-valid"
    with pytest.raises(FundamentalSuccessorSourceError):
        _acquire(root, client=FakeClient(row_factory=invalid_classification))
    failure_root = tmp_path / "now-valid-failures"
    failure_path = failure_root / "000000.failure.json"
    raw_path = failure_root / "000000.raw.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_bytes())
    comp_type_index = raw["data"]["fields"].index("comp_type")
    raw["data"]["items"][0][comp_type_index] = "1"
    raw_bytes = _canonical_bytes(raw)
    raw_path.write_bytes(raw_bytes)
    failure["raw_response_ref"] = {
        "path": "000000.raw.json",
        "byte_length": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    failure["response_evidence"]["raw_response_byte_length"] = len(raw_bytes)
    failure["response_evidence"]["raw_response_sha256"] = hashlib.sha256(
        raw_bytes
    ).hexdigest()
    failure.pop("failure_sha256")
    failure["failure_sha256"] = hashlib.sha256(
        _canonical_bytes(failure)
    ).hexdigest()
    failure_path.write_bytes(_canonical_bytes(failure))

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_FAILURE_EVIDENCE_INVALID",
    ):
        validate_successor_failure_evidence(failure_root, ordinal=0)


@pytest.mark.parametrize("mutation", ["has_more", "count", "row_ceiling"])
def test_failure_validator_replays_precanonical_response_gates(
    tmp_path: Path,
    mutation: str,
) -> None:
    def invalid_classification(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        row["comp_type"] = "5"
        return [row]

    root = tmp_path / f"envelope-{mutation}"
    with pytest.raises(FundamentalSuccessorSourceError):
        _acquire(root, client=FakeClient(row_factory=invalid_classification))
    failure_root = tmp_path / f"envelope-{mutation}-failures"
    failure_path = failure_root / "000000.failure.json"
    raw_path = failure_root / "000000.raw.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    raw = json.loads(raw_path.read_bytes())
    if mutation == "has_more":
        raw["data"]["has_more"] = True
    elif mutation == "count":
        raw["data"]["count"] = 2
    else:
        raw["data"]["items"] = raw["data"]["items"] * failure["request"][
            "row_ceiling"
        ]
        raw["data"]["count"] = 0
    raw_bytes = _canonical_bytes(raw)
    raw_path.write_bytes(raw_bytes)
    failure["raw_response_ref"] = {
        "path": "000000.raw.json",
        "byte_length": len(raw_bytes),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    failure["response_evidence"].update(
        {
            "has_more": raw["data"]["has_more"],
            "item_count": len(raw["data"]["items"]),
            "provider_reported_count": raw["data"]["count"],
            "raw_response_byte_length": len(raw_bytes),
            "raw_response_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }
    )
    failure.pop("failure_sha256")
    failure["failure_sha256"] = hashlib.sha256(
        _canonical_bytes(failure)
    ).hexdigest()
    failure_path.write_bytes(_canonical_bytes(failure))

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_FAILURE_EVIDENCE_INVALID",
    ):
        validate_successor_failure_evidence(failure_root, ordinal=0)


def test_failure_root_cannot_rebind_to_a_different_sibling_plan(
    tmp_path: Path,
) -> None:
    def invalid_classification(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        row = _row(api_name=api_name, params=params, fields=fields)
        row["comp_type"] = "5"
        return [row]

    first = tmp_path / "first"
    second = tmp_path / "second"
    with pytest.raises(FundamentalSuccessorSourceError):
        _acquire(first, client=FakeClient(row_factory=invalid_classification))
    with pytest.raises(FundamentalSuccessorSourceError):
        _acquire(
            second,
            client=FakeClient(row_factory=invalid_classification),
            immutable_refs={"different": {"sha256": "e" * 64}},
        )
    first_failure = tmp_path / "first-failures"
    second_failure = tmp_path / "second-failures"
    for name in ("000000.failure.json", "000000.raw.json"):
        (second_failure / name).write_bytes((first_failure / name).read_bytes())

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_FAILURE_EVIDENCE_INVALID",
    ):
        validate_successor_failure_evidence(second_failure, ordinal=0)


def test_v2_canonicalization_plan_cannot_replay_under_v3_contract() -> None:
    plan = _plan()
    plan.pop("plan_sha256")
    plan["canonicalization_policy"] = (
        "myquant-fundamental-successor-canonicalization.v2"
    )
    plan["plan_sha256"] = hashlib.sha256(_canonical_bytes(plan)).hexdigest()

    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_PLAN_CONTRACT_MISMATCH",
    ):
        replay_successor_support_requests(plan)


def test_resume_binding_and_record_tamper_fail_before_network(tmp_path: Path) -> None:
    root = tmp_path / "fileset"
    _acquire(root)

    class NetworkForbidden:
        def request(self, **_kwargs: Any) -> TushareResponse:
            raise AssertionError("resume attempted network")

    resumed = _acquire(root, client=NetworkForbidden())
    assert resumed["status"] == "COMPLETE"
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_RESUME_BINDING_MISMATCH",
    ):
        _acquire(
            root,
            client=NetworkForbidden(),
            immutable_refs={"parent_metadata": {"schema_version": "v2"}},
        )
    expanded_plan = build_successor_support_plan(
        support_start=TARGET,
        target_date=TARGET,
        open_sessions=(TARGET,),
        symbols=(*SYMBOLS, "999999.SH"),
        canonical_subject_scope_authority_sha256="e" * 64,
    )
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_RESUME_BINDING_MISMATCH",
    ):
        _acquire(root, client=NetworkForbidden(), plan=expanded_plan)

    record_path = root / "requests" / "000000.json"
    record_path.write_bytes(record_path.read_bytes() + b" ")
    with pytest.raises(FundamentalSuccessorSourceError):
        validate_successor_capture_fileset(root)


def _forge_record_raw_response(root: Path, mutation: str) -> None:
    record_path = root / "requests" / "000000.json"
    record = json.loads(record_path.read_text())
    raw = json.loads(
        base64.b64decode(record["raw_response_bytes_base64"], validate=True)
    )
    if mutation == "rows":
        raw["data"]["items"][0][4] = 9.75
    elif mutation == "request_id":
        raw["request_id"] += "-tampered"
    elif mutation == "count":
        raw["data"]["count"] += 1
    elif mutation == "fields":
        raw["data"]["fields"] = list(reversed(raw["data"]["fields"]))
    else:  # pragma: no cover - closed by the parameter set
        raise AssertionError(mutation)
    raw_bytes = _canonical_bytes(raw)
    record["raw_response_bytes_base64"] = base64.b64encode(raw_bytes).decode("ascii")
    record["raw_response_byte_length"] = len(raw_bytes)
    record["raw_response_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    record.pop("record_sha256")
    record["record_sha256"] = hashlib.sha256(_canonical_bytes(record)).hexdigest()
    record_payload = _canonical_bytes(record)
    record_path.write_bytes(record_payload)

    manifest_path = root / "provider_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["record_files"][0]["byte_length"] = len(record_payload)
    manifest["record_files"][0]["sha256"] = hashlib.sha256(record_payload).hexdigest()
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
    manifest_path.write_bytes(_canonical_bytes(manifest))


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("rows", "SUCCESSOR_RECORD_RAW_ROWS_MISMATCH"),
        ("request_id", "SUCCESSOR_RECEIPT_INVALID"),
        ("count", "SUCCESSOR_RECEIPT_INVALID"),
        ("fields", "SUCCESSOR_RECORD_RAW_RESPONSE_REPLAY_FAILED"),
    ],
)
def test_exact_raw_response_is_independently_replayed(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    root = tmp_path / mutation
    _acquire(root)
    _forge_record_raw_response(root, mutation)

    with pytest.raises(FundamentalSuccessorSourceError, match=error):
        validate_successor_capture_fileset(root)


def test_exact_pointer_bytes_are_durable_and_pointer_tamper_is_detected(
    tmp_path: Path,
) -> None:
    predecessor = b'{"schema_version":"v2","opaque":"\\u4e2d\\u6587"}\n'
    root = tmp_path / "fileset"
    _acquire(
        root,
        pointers={
            "predecessor": predecessor,
            "market": b'{"market":"exact"}\n',
            "pit": b'{"pit":"exact"}\n',
        },
    )
    manifest = validate_successor_capture_fileset(root)
    pointer = manifest["captured_pointers"]["predecessor"]
    assert base64.b64decode(pointer["bytes_base64"]) == predecessor
    assert pointer["sha256"] == hashlib.sha256(predecessor).hexdigest()

    binding_path = root / "binding.json"
    binding = json.loads(binding_path.read_text())
    binding["captured_pointers"]["market"]["bytes_base64"] = base64.b64encode(b"forged").decode(
        "ascii"
    )
    binding.pop("binding_sha256")
    binding["binding_sha256"] = hashlib.sha256(_canonical_bytes(binding)).hexdigest()
    binding_path.write_bytes(_canonical_bytes(binding))
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_CAPTURED_POINTER_EVIDENCE_INVALID",
    ):
        validate_successor_capture_fileset(root)


def test_staging_capture_helper_accepts_zero_deferred_fileset(tmp_path: Path) -> None:
    root = tmp_path / "fileset"
    manifest = _acquire(root)
    evidence = capture_successor_support_evidence(root)

    assert manifest["authority_state"] == "AUTHORITATIVE_DELTA_COMPLETE"
    assert manifest["authoritative_source_ready"] is True
    assert manifest["staging_eligible"] is True
    assert evidence["provider_manifest.json"] == (root / "provider_manifest.json").read_bytes()


def test_path_backed_evidence_and_lazy_tables_preserve_resource_accounting(
    tmp_path: Path,
) -> None:
    root = tmp_path / "fileset"
    manifest = _acquire(root)
    paths = successor_support_evidence_paths(root)
    tables = open_capture_support_tables(root)

    assert all(path.is_absolute() for path in paths.values())
    assert paths["provider_manifest.json"] == root / "provider_manifest.json"
    assert set(tables) == set(manifest["table_files"])
    assert sum(len(batch) for batch in tables.iter_batches("daily_basic")) == 1
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_FULL_TABLE_ACCESS_FORBIDDEN",
    ):
        tables["daily_basic"]
    resource = manifest["resource_accounting"]
    assert resource["status"] == "PASS"
    assert resource["maximum_estimated_table_memory_bytes"] <= resource["policy"][
        "table_memory_limit_bytes"
    ]
    materialized = load_capture_support_tables(root)
    assert manifest["table_fingerprints"] == {
        table: frame_fingerprint(frame) for table, frame in materialized.items()
    }


def test_low_available_memory_blocks_before_provider(tmp_path: Path) -> None:
    client = FakeClient()
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_RESOURCE_POLICY_INVALID",
    ):
        acquire_successor_support(
            plan=_plan(),
            client=client,
            fileset_root=tmp_path / "fileset",
            captured_pointer_bytes={
                "predecessor": b"predecessor",
                "market": b"market",
                "pit": b"pit",
            },
            immutable_refs={"authority": {"sha256": "a" * 64}},
            implementation_sha256=IMPLEMENTATION_SHA256,
            captured_at=CAPTURED_AT,
            max_attempts=1,
            retry_backoff_seconds=(),
            requests_per_second=8.0,
            physical_memory_bytes=16 * 1024 * 1024 * 1024,
            available_memory_bytes=16 * 1024 * 1024,
            rlimit_headroom_bytes=16 * 1024 * 1024,
            sleeper=lambda _seconds: None,
        )
    assert client.calls == 0


def test_high_cardinality_capture_uses_bounded_parquet_row_groups(
    tmp_path: Path,
) -> None:
    symbols = [f"{value:06d}.SZ" for value in range(1, 4_102)]
    plan = build_successor_support_plan(
        support_start=TARGET,
        target_date=TARGET,
        open_sessions=(TARGET,),
        symbols=symbols,
        canonical_subject_scope_authority_sha256="d" * 64,
    )

    def rows(
        api_name: str,
        params: Mapping[str, Any],
        fields: Sequence[str],
    ) -> Sequence[Mapping[str, Any]]:
        template = _row(api_name=api_name, params=params, fields=fields)
        return [{**template, "ts_code": symbol} for symbol in symbols]

    root = tmp_path / "fileset"
    manifest = _acquire(root, client=FakeClient(row_factory=rows), plan=plan)
    store = open_capture_support_tables(root)
    for table, ref in manifest["table_files"].items():
        metadata = ref["metadata"]
        assert metadata["row_count"] == len(symbols)
        assert metadata["observed_maximum_batch_rows"] <= 2_048
        assert metadata["observed_maximum_batch_bytes"] <= 16 * 1024 * 1024
        parquet = pq.ParquetFile(root / ref["path"])
        assert all(
            parquet.metadata.row_group(group).num_rows <= 2_048
            for group in range(parquet.num_row_groups)
        )
        assert sum(len(batch) for batch in store.iter_batches(table)) == len(symbols)


def test_external_sort_disk_reserve_blocks_after_resumable_request_capture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeClient()

    def blocked(*_args: Any, **_kwargs: Any) -> int:
        raise FundamentalSuccessorSourceError(
            "SUCCESSOR_EXTERNAL_SORT_DISK_RESERVE_EXHAUSTED"
        )

    monkeypatch.setattr(source_module, "_require_external_sort_reserve", blocked)
    root = tmp_path / "fileset"
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_EXTERNAL_SORT_DISK_RESERVE_EXHAUSTED",
    ):
        _acquire(root, client=client)
    assert client.calls == len(replay_successor_support_requests(_plan()))
    assert list((root / "requests").glob("*.json"))
    assert not (root / "provider_manifest.json").exists()
    assert not list((root / "tables").glob("*.parquet"))


def test_validator_reconstruction_disk_reserve_is_same_device_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "fileset"
    _acquire(root)
    manifest_bytes = (root / "provider_manifest.json").read_bytes()

    def blocked(*_args: Any, **_kwargs: Any) -> int:
        raise FundamentalSuccessorSourceError(
            "SUCCESSOR_EXTERNAL_SORT_DISK_RESERVE_EXHAUSTED"
        )

    monkeypatch.setattr(source_module, "_require_external_sort_reserve", blocked)
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_EXTERNAL_SORT_DISK_RESERVE_EXHAUSTED",
    ):
        validate_successor_capture_fileset(root)
    assert (root / "provider_manifest.json").read_bytes() == manifest_bytes
    assert not list(tmp_path.glob(f".{root.name}.validation-*"))


def test_disk_resource_preflight_blocks_before_provider(tmp_path: Path) -> None:
    client = FakeClient()
    with pytest.raises(
        FundamentalSuccessorSourceError,
        match="SUCCESSOR_DISK_RESERVE_EXHAUSTED",
    ):
        acquire_successor_support(
            plan=_plan(),
            client=client,
            fileset_root=tmp_path / "fileset",
            captured_pointer_bytes={
                "predecessor": b"predecessor",
                "market": b"market",
                "pit": b"pit",
            },
            immutable_refs={"authority": {"sha256": "a" * 64}},
            implementation_sha256=IMPLEMENTATION_SHA256,
            captured_at=CAPTURED_AT,
            max_attempts=1,
            retry_backoff_seconds=(),
            requests_per_second=8.0,
            minimum_free_disk_bytes=10**18,
            sleeper=lambda _seconds: None,
        )
    assert client.calls == 0
