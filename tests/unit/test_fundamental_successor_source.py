from __future__ import annotations

from collections.abc import Mapping, Sequence
import base64
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pytest

from quant_investor.market.fundamental_successor_source import (
    FUNDAMENTAL_SUCCESSOR_PROVIDER_MANIFEST_SCHEMA,
    FundamentalSuccessorSourceError,
    SUCCESSOR_ENDPOINT_CAPABILITIES,
    acquire_successor_support,
    build_successor_support_plan,
    capture_successor_support_evidence,
    load_support_tables,
    replay_successor_support_requests,
    validate_successor_support_fileset,
)
from quant_investor.v17_v4_runtime.tushare_https import (
    TushareResponse,
    replay_tushare_response_bytes,
)

TARGET = "20260807"
CAPTURED_AT = "2026-08-07T17:00:00Z"
IMPLEMENTATION_SHA256 = "a" * 64
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
) -> dict[str, Any]:
    return build_successor_support_plan(
        support_start=support_start,
        target_date=target,
        open_sessions=open_sessions,
        symbols=SYMBOLS,
        canonical_subject_scope_authority_sha256="d" * 64,
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
    tables = load_support_tables(tmp_path / "fileset")

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
    fina = load_support_tables(root)["fina_indicator"]

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
    tables = load_support_tables(root)

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
        validate_successor_support_fileset(root)


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
    income = load_support_tables(root)["income"]

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
    first_table = load_support_tables(tmp_path / "first")["income"]
    second_table = load_support_tables(tmp_path / "second")["income"]

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
    }


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
        validate_successor_support_fileset(root)


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
    manifest = validate_successor_support_fileset(root)
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
        validate_successor_support_fileset(root)


def test_capture_helper_returns_only_validated_exact_files(tmp_path: Path) -> None:
    root = tmp_path / "fileset"
    manifest = _acquire(root)
    captured = capture_successor_support_evidence(root)

    assert captured["provider_manifest.json"] == (root / "provider_manifest.json").read_bytes()
    assert captured["binding.json"] == (root / "binding.json").read_bytes()
    assert len(captured) == 2 + len(manifest["record_files"]) + len(manifest["table_files"])
    assert all(Path(relative).is_absolute() is False for relative in captured)
