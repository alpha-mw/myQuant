from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from typing import Any

import pytest

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.market.exchange_calendar_closure import (
    runtime_json_bytes,
    runtime_parquet_bytes,
)
from quant_investor.market.tushare_calendar_authority import (
    EXPECTED_FIELDS,
    SOURCE_LIMITATIONS,
    build_calendar_authority_policy,
    build_trusted_provider_calendar_capability,
    build_trusted_provider_calendar_capture,
    build_trusted_provider_calendar_capture_transaction,
    build_trusted_provider_calendar_compilation,
    capture_trusted_provider_calendar_evidence,
    decode_trade_cal_documentation,
    validate_trusted_provider_calendar_compilation,
    validate_trusted_provider_calendar_capture_transaction,
)
from quant_investor.market.tushare_transport import replay_tushare_response_bytes
from quant_investor.system import SystemContractError, SystemPreconditionError
from quant_investor.system.store import object_ref_for_artifact
from tests.unit.test_unified_production_bootstrap_operator import (
    _byte_ref,
    _inputs,
    _request,
    _seed_workspace,
    _write,
)
from quant_investor.factors.governance.production import assemble_production_bootstrap

CREATED_AT = "2026-08-17T00:00:00Z"
CAPTURE_START = date(2023, 12, 1)
CUTOFF = date(2025, 7, 1)


def _docs(*, output_exchange_text: str = "SSE上交所 SZSE深交所") -> bytes:
    return f"""<!doctype html><html><body>
<h2>交易日历</h2><p>接口：trade_cal</p>
<h3>输入参数</h3><table>
<tr><th>名称</th><th>描述</th></tr>
<tr><td>exchange</td><td>SSE上交所,SZSE深交所,CFFEX 中金所,SHFE 上期所,CZCE 郑商所,DCE 大商所,INE 上能源</td></tr>
</table>
<h3>输出参数</h3><table>
<tr><th>名称</th><th>描述</th></tr>
<tr><td>exchange</td><td>{output_exchange_text}</td></tr>
<tr><td>cal_date</td><td>日历日期</td></tr>
<tr><td>is_open</td><td>是否交易</td></tr>
<tr><td>pretrade_date</td><td>上一个交易日</td></tr>
</table></body></html>""".encode()


def _ref(relative_path: str, raw: bytes) -> dict[str, str]:
    return {
        "relative_path": relative_path,
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _provider_raw(
    exchange: str,
    *,
    force_nonempty: bool = False,
    cutoff: date = CUTOFF,
) -> bytes:
    items: list[list[Any]] = []
    if exchange != "BSE" or force_nonempty:
        prior_open = ""
        cursor = CAPTURE_START
        while cursor <= cutoff:
            opened = cursor.weekday() < 5
            items.append(
                [
                    exchange,
                    cursor.strftime("%Y%m%d"),
                    1 if opened else 0,
                    prior_open,
                ]
            )
            if opened:
                prior_open = cursor.strftime("%Y%m%d")
            cursor += timedelta(days=1)
    return json.dumps(
        {
            "code": 0,
            "data": {
                "count": 0,
                "fields": list(EXPECTED_FIELDS),
                "has_more": False,
                "items": items,
            },
            "detail": "",
            "msg": "",
            "request_id": f"request-{exchange.lower()}",
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _runtime(*, cutoff: date = CUTOFF) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = date(2024, 1, 1)
    while cursor <= cutoff:
        opened = cursor.weekday() < 5
        rows.append(
            {
                "date": cursor.isoformat(),
                "status": "OPEN" if opened else "CLOSED",
                "opens_at_utc": (
                    datetime.combine(cursor, time(1, 30), tzinfo=timezone.utc).isoformat()
                    if opened
                    else None
                ),
                "closes_at_utc": (
                    datetime.combine(cursor, time(7), tzinfo=timezone.utc).isoformat()
                    if opened
                    else None
                ),
            }
        )
        cursor += timedelta(days=1)
    return rows


def _case() -> dict[str, Any]:
    docs = _docs()
    docs_ref = _ref("raw/tushare-trade-cal-doc.html", docs)
    capability = build_trusted_provider_calendar_capability(
        docs_raw=docs,
        docs_raw_file_ref=docs_ref,
        docs_captured_at=CREATED_AT,
        docs_http_status=200,
        docs_tls_verified=True,
        docs_redirect_chain=[],
        docs_response_headers={"content-type": "text/html; charset=utf-8"},
        created_at=CREATED_AT,
    )
    raw_by_ref: dict[bytes, bytes] = {canonical_json_bytes(docs_ref): docs}
    captures = []
    for exchange in ("SSE", "SZSE", "BSE"):
        raw = _provider_raw(exchange)
        raw_ref = _ref(f"raw/trade-cal-{exchange.lower()}.json", raw)
        raw_by_ref[canonical_json_bytes(raw_ref)] = raw
        captures.append(
            build_trusted_provider_calendar_capture(
                exchange_id=exchange,
                raw=raw,
                raw_file_ref=raw_ref,
                capability=capability,
                docs_raw=docs,
                captured_at=CREATED_AT,
                capture_start_date=CAPTURE_START.isoformat(),
                cutoff_date=CUTOFF.isoformat(),
                request_parameters_sanitized={
                    "end_date": CUTOFF.strftime("%Y%m%d"),
                    "exchange": exchange,
                    "start_date": CAPTURE_START.strftime("%Y%m%d"),
                },
                response_headers={"content-type": "application/json"},
                created_at=CREATED_AT,
            )
        )
    runtime = _runtime()
    runtime_json = runtime_json_bytes(runtime)
    runtime_parquet = runtime_parquet_bytes(runtime)
    json_ref = _ref("strict/exchange-calendar.json", runtime_json)
    parquet_ref = _ref("strict/exchange-calendar.parquet", runtime_parquet)
    raw_by_ref[canonical_json_bytes(json_ref)] = runtime_json
    raw_by_ref[canonical_json_bytes(parquet_ref)] = runtime_parquet
    release = seal_artifact(
        "system.release",
        {
            "release_id": "test-tushare-calendar-release",
            "state": "OPERATIONAL",
            "code_sha256": "1" * 64,
            "wheel_sha256": "2" * 64,
            "code_manifest_sha256": "3" * 64,
        },
        created_at=CREATED_AT,
    )
    return {
        "capability": capability,
        "captures": captures,
        "docs": docs,
        "json_ref": json_ref,
        "market_sessions": [row["date"] for row in runtime if row["status"] == "OPEN"],
        "parquet_ref": parquet_ref,
        "policy": build_calendar_authority_policy(created_at=CREATED_AT),
        "raw_by_ref": raw_by_ref,
        "release_ref": object_ref_for_artifact(release),
    }


def _raw_resolver(case: dict[str, Any]):
    def resolve(reference: dict[str, Any]) -> bytes:
        return case["raw_by_ref"][canonical_json_bytes(reference)]

    return resolve


def test_trusted_provider_calendar_compiles_with_truthful_bse_projection() -> None:
    case = _case()
    compilation = build_trusted_provider_calendar_compilation(
        compilation_id="trusted-provider-test",
        policy=case["policy"],
        capability=case["capability"],
        capture_documents=case["captures"],
        docs_raw=case["docs"],
        raw_resolver=_raw_resolver(case),
        release_ref=case["release_ref"],
        pit_exchange_ids=["BSE", "SSE", "SZSE"],
        market_session_dates=case["market_sessions"],
        cutoff_date=CUTOFF.isoformat(),
        calendar_json_file_ref=case["json_ref"],
        calendar_parquet_file_ref=case["parquet_ref"],
        created_at=CREATED_AT,
    )
    payload = compilation["payload"]
    assert payload["source_limitations"] == list(SOURCE_LIMITATIONS)
    bse = next(row for row in payload["exchange_rows"] if row["exchange_id"] == "BSE")
    assert bse["calendar_row_origin"] == "POLICY_PROJECTED"
    assert bse["provider_direct"] is False
    assert bse["exchange_official"] is False
    assert (
        validate_trusted_provider_calendar_compilation(
            compilation,
            policy=case["policy"],
            capability=case["capability"],
            capture_documents=case["captures"],
            docs_raw=case["docs"],
            raw_resolver=_raw_resolver(case),
            expected_release_ref=case["release_ref"],
            pit_exchange_ids=["BSE", "SSE", "SZSE"],
            market_session_dates=case["market_sessions"],
        )
        == compilation
    )


def test_capture_transaction_rejects_leaf_binding_drift() -> None:
    case = _case()
    docs_ref = case["capability"]["payload"]["docs_raw_file_ref"]
    capability_ref = _ref(
        "artifacts/provider-capability.json",
        canonical_json_bytes(case["capability"]),
    )
    policy_ref = _ref(
        "artifacts/calendar-authority-policy.json",
        canonical_json_bytes(case["policy"]),
    )
    capture_refs = [
        _ref(
            f"artifacts/{capture['payload']['exchange_id'].lower()}-capture.json",
            canonical_json_bytes(capture),
        )
        for capture in case["captures"]
    ]
    raw_refs = [capture["payload"]["raw_file_ref"] for capture in case["captures"]]
    transaction = build_trusted_provider_calendar_capture_transaction(
        capture_root_name="capture-transaction-test",
        capture_start_date=CAPTURE_START.isoformat(),
        cutoff_date=CUTOFF.isoformat(),
        captured_at=CREATED_AT,
        documentation_raw_file_ref=docs_ref,
        capability_file_ref=capability_ref,
        policy_file_ref=policy_ref,
        provider_raw_file_refs=raw_refs,
        provider_capture_file_refs=capture_refs,
    )
    drifted_raw_refs = [*raw_refs]
    drifted_raw_refs[0] = {**drifted_raw_refs[0], "byte_sha256": "f" * 64}
    with pytest.raises(SystemContractError, match="transaction binding differs"):
        validate_trusted_provider_calendar_capture_transaction(
            transaction,
            documentation_raw_file_ref=docs_ref,
            capability_file_ref=capability_ref,
            policy_file_ref=policy_ref,
            provider_raw_file_refs=drifted_raw_refs,
            provider_capture_file_refs=capture_refs,
        )


def test_documentation_decoder_blocks_bse_direct_support_or_output_drift() -> None:
    with pytest.raises(SystemContractError, match="output exchange"):
        decode_trade_cal_documentation(_docs(output_exchange_text="SSE SZSE BSE"))


def test_bse_nonempty_probe_cannot_confer_direct_calendar_authority() -> None:
    case = _case()
    raw = _provider_raw("BSE", force_nonempty=True)
    with pytest.raises(SystemPreconditionError, match="exact-empty"):
        build_trusted_provider_calendar_capture(
            exchange_id="BSE",
            raw=raw,
            raw_file_ref=_ref("raw/bse-nonempty.json", raw),
            capability=case["capability"],
            docs_raw=case["docs"],
            captured_at=CREATED_AT,
            capture_start_date=CAPTURE_START.isoformat(),
            cutoff_date=CUTOFF.isoformat(),
            request_parameters_sanitized={
                "end_date": CUTOFF.strftime("%Y%m%d"),
                "exchange": "BSE",
                "start_date": CAPTURE_START.strftime("%Y%m%d"),
            },
            response_headers={"content-type": "application/json"},
            created_at=CREATED_AT,
        )


def test_market_bar_on_provider_closed_date_blocks_without_calendar_mutation() -> None:
    case = _case()
    closed = next(row["date"] for row in _runtime() if row["status"] == "CLOSED")
    sessions = sorted([*case["market_sessions"], closed])
    with pytest.raises(SystemPreconditionError, match="contradict"):
        build_trusted_provider_calendar_compilation(
            compilation_id="trusted-provider-contradiction",
            policy=case["policy"],
            capability=case["capability"],
            capture_documents=case["captures"],
            docs_raw=case["docs"],
            raw_resolver=_raw_resolver(case),
            release_ref=case["release_ref"],
            pit_exchange_ids=["BSE", "SSE", "SZSE"],
            market_session_dates=sessions,
            cutoff_date=CUTOFF.isoformat(),
            calendar_json_file_ref=case["json_ref"],
            calendar_parquet_file_ref=case["parquet_ref"],
            created_at=CREATED_AT,
        )


def test_capture_transaction_is_four_call_atomic_and_secret_free(tmp_path) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)

    class FakeClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def request(self, *, api_name: str, params: dict[str, str], expected_fields):
            assert api_name == "trade_cal"
            assert tuple(expected_fields) == EXPECTED_FIELDS
            exchange = params["exchange"]
            self.calls.append(exchange)
            return replay_tushare_response_bytes(
                _provider_raw(exchange),
                api_name="trade_cal",
                expected_fields=EXPECTED_FIELDS,
                strict_decimal_decode=True,
            )

    client = FakeClient()
    result = capture_trusted_provider_calendar_evidence(
        capture_parent=parent,
        capture_root_name="capture-20250801",
        cutoff_date=CUTOFF.isoformat(),
        captured_at=CREATED_AT,
        client=client,
        documentation_fetcher=lambda: (
            _docs(),
            200,
            {"content-type": "text/html; charset=utf-8"},
            True,
            [],
        ),
    )
    assert result["network_call_count"] == 4
    assert client.calls == ["SSE", "SZSE", "BSE"]
    capture_root = parent / "capture-20250801"
    assert capture_root.is_dir()
    assert not list(parent.glob(".capture-20250801.staging-*"))
    combined = b"".join(path.read_bytes() for path in capture_root.rglob("*") if path.is_file())
    assert b"TUSHARE_TOKEN" not in combined
    assert b"token" not in combined.lower()
    assert all(
        (path.stat().st_mode & 0o777) == 0o600 for path in capture_root.rglob("*") if path.is_file()
    )


def test_capture_failure_never_publishes_success_root(tmp_path) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)

    class FailingClient:
        def request(self, **kwargs):
            del kwargs
            raise RuntimeError("provider cells must never escape")

    with pytest.raises(SystemPreconditionError, match="CAPTURE_FAILED"):
        capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name="capture-fails",
            cutoff_date=CUTOFF.isoformat(),
            captured_at=CREATED_AT,
            client=FailingClient(),
            documentation_fetcher=lambda: (
                _docs(),
                200,
                {"content-type": "text/html; charset=utf-8"},
                True,
                [],
            ),
        )
    assert not (parent / "capture-fails").exists()
    failures = list(parent.glob(".capture-fails.failed-*"))
    assert len(failures) == 1
    failure_raw = (failures[0] / "capture-failure.json").read_bytes()
    assert b"provider cells" not in failure_raw


def test_production_bootstrap_accepts_only_sealed_provider_route(tmp_path) -> None:
    production_cutoff = date(2026, 8, 7)
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-provider-inputs"
    files = _inputs(input_root)
    docs = _docs()
    docs_ref = _ref("provider/raw/trade-cal-documentation.html", docs)
    _write(input_root, docs_ref["relative_path"], docs)
    capability = build_trusted_provider_calendar_capability(
        docs_raw=docs,
        docs_raw_file_ref=docs_ref,
        docs_captured_at="2026-08-14T00:00:00Z",
        docs_http_status=200,
        docs_tls_verified=True,
        docs_redirect_chain=[],
        docs_response_headers={"content-type": "text/html; charset=utf-8"},
        created_at="2026-08-14T00:00:00Z",
    )
    policy = build_calendar_authority_policy(created_at="2026-08-14T00:00:00Z")
    raw_by_ref: dict[bytes, bytes] = {canonical_json_bytes(docs_ref): docs}
    raw_refs = [docs_ref]
    capture_paths: list[str] = []
    captures = []
    for exchange in ("SSE", "SZSE", "BSE"):
        raw = _provider_raw(exchange, cutoff=production_cutoff)
        raw_ref = _ref(f"provider/raw/trade-cal-{exchange.lower()}.json", raw)
        _write(input_root, raw_ref["relative_path"], raw)
        raw_by_ref[canonical_json_bytes(raw_ref)] = raw
        raw_refs.append(raw_ref)
        capture = build_trusted_provider_calendar_capture(
            exchange_id=exchange,
            raw=raw,
            raw_file_ref=raw_ref,
            capability=capability,
            docs_raw=docs,
            captured_at="2026-08-14T00:00:00Z",
            capture_start_date=CAPTURE_START.isoformat(),
            cutoff_date=production_cutoff.isoformat(),
            request_parameters_sanitized={
                "end_date": production_cutoff.strftime("%Y%m%d"),
                "exchange": exchange,
                "start_date": CAPTURE_START.strftime("%Y%m%d"),
            },
            response_headers={"content-type": "application/json"},
            created_at="2026-08-14T00:00:00Z",
        )
        path = f"provider/artifacts/trade-cal-{exchange.lower()}-capture.json"
        _write(input_root, path, canonical_json_bytes(capture))
        capture_paths.append(path)
        captures.append(capture)
    capability_path = "provider/artifacts/trade-cal-capability.json"
    policy_path = "provider/artifacts/calendar-authority-policy.json"
    _write(input_root, capability_path, canonical_json_bytes(capability))
    _write(input_root, policy_path, canonical_json_bytes(policy))
    runtime = _runtime(cutoff=production_cutoff)
    calendar_json = runtime_json_bytes(runtime)
    calendar_parquet = runtime_parquet_bytes(runtime)
    _write(input_root, "strict/exchange_calendar.json", calendar_json)
    _write(input_root, "strict/exchange_calendar.parquet", calendar_parquet)
    json_ref = _byte_ref(input_root, "strict/exchange_calendar.json")
    parquet_ref = _byte_ref(input_root, "strict/exchange_calendar.parquet")
    raw_by_ref[canonical_json_bytes(json_ref)] = calendar_json
    raw_by_ref[canonical_json_bytes(parquet_ref)] = calendar_parquet

    def raw_resolver(reference: dict[str, Any]) -> bytes:
        return raw_by_ref[canonical_json_bytes(reference)]

    market_sessions = [
        row["date"] for row in runtime if row["status"] == "OPEN" and row["date"] >= "2024-01-02"
    ][-100:]
    compilation = build_trusted_provider_calendar_compilation(
        compilation_id="provider-production-bootstrap",
        policy=policy,
        capability=capability,
        capture_documents=captures,
        docs_raw=docs,
        raw_resolver=raw_resolver,
        release_ref=release_ref,
        pit_exchange_ids=["SSE", "SZSE"],
        market_session_dates=market_sessions,
        cutoff_date=production_cutoff.isoformat(),
        calendar_json_file_ref=json_ref,
        calendar_parquet_file_ref=parquet_ref,
        created_at="2026-08-14T00:00:00Z",
    )
    compilation_path = "provider/artifacts/calendar-compilation.json"
    _write(input_root, compilation_path, canonical_json_bytes(compilation))
    files["exchange_calendar_file_ref"] = parquet_ref
    files["calendar_runtime_json_file_ref"] = json_ref
    files["calendar_compilation_file_ref"] = _byte_ref(input_root, compilation_path)
    files["calendar_authority_policy_file_ref"] = _byte_ref(input_root, policy_path)
    files["official_calendar_raw_file_refs"] = []
    files["official_calendar_capture_file_refs"] = []
    files["official_calendar_decoder_admission_file_refs"] = []
    files["official_calendar_index_closure_file_refs"] = []
    files["trusted_provider_calendar_raw_file_refs"] = sorted(
        raw_refs, key=lambda row: (row["relative_path"], row["byte_sha256"])
    )
    files["trusted_provider_calendar_capture_file_refs"] = [
        _byte_ref(input_root, path) for path in sorted(capture_paths)
    ]
    files["trusted_provider_calendar_capability_file_ref"] = _byte_ref(input_root, capability_path)
    transaction = build_trusted_provider_calendar_capture_transaction(
        capture_root_name="provider-production-capture",
        capture_start_date=CAPTURE_START.isoformat(),
        cutoff_date=production_cutoff.isoformat(),
        captured_at="2026-08-14T00:00:00Z",
        documentation_raw_file_ref=docs_ref,
        capability_file_ref=files["trusted_provider_calendar_capability_file_ref"],
        policy_file_ref=files["calendar_authority_policy_file_ref"],
        provider_raw_file_refs=[capture["payload"]["raw_file_ref"] for capture in captures],
        provider_capture_file_refs=files["trusted_provider_calendar_capture_file_refs"],
    )
    transaction_path = "provider/capture-transaction.json"
    _write(input_root, transaction_path, canonical_json_bytes(transaction))
    files["trusted_provider_calendar_capture_transaction_file_ref"] = _byte_ref(
        input_root,
        transaction_path,
    )
    result = assemble_production_bootstrap(
        workspace_root=workspace,
        input_root=input_root,
        request_raw=_request(
            workspace_root=workspace,
            release_ref=release_ref,
            files=files,
            operation_id="provider-production-bootstrap",
        ),
    )
    assert result["generation"]["generation_state"] == "OPERATIONAL"
    assert result["generation"]["calendar_authority_confidence"] == "DEGRADED"
    assert result["generation"]["calendar_authority_route"] == ("TRUSTED_PROVIDER_DEGRADED")
    receipt = result["generation"]["research"][0]["payload"]
    assert receipt["calendar_source_limitations"] == list(SOURCE_LIMITATIONS)
    assert receipt["calendar_capability_ref"] is not None
