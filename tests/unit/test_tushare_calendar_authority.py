from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
from typing import Any

import pytest

import quant_investor.market.tushare_calendar_authority as calendar_authority
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
    validate_published_trusted_provider_calendar_capture_failure_root,
    validate_trusted_provider_calendar_capture_failure,
    validate_trusted_provider_calendar_compilation,
    validate_trusted_provider_calendar_capture_transaction,
)
from quant_investor.market.tushare_transport import (
    TushareHttpsError,
    replay_tushare_response_bytes,
)
from quant_investor.system import (
    SystemContractError,
    SystemPreconditionError,
    SystemSecurityError,
    SystemStorageError,
    SystemStore,
)
from quant_investor.system.release_install import build_release_install_evidence
from quant_investor.system.store import object_ref_for_artifact
from tests.unit.test_unified_production_bootstrap_operator import (
    _byte_ref,
    _inputs,
    _request,
    _seed_workspace,
    _write,
)
import tests.unit.test_unified_production_bootstrap_operator as production_test_module
from quant_investor.factors.governance.production import assemble_production_bootstrap

CREATED_AT = "2026-08-17T00:00:00Z"
CAPTURE_START = date(2023, 12, 1)
CUTOFF = date(2025, 7, 1)


def test_execution_and_success_sealers_have_one_production_writer() -> None:
    assert "build_trusted_provider_calendar_capture_execution" not in calendar_authority.__all__
    assert "build_trusted_provider_calendar_capture_success" not in calendar_authority.__all__
    assert "build_trusted_provider_calendar_capture_failure" not in calendar_authority.__all__
    assert "publish_trusted_provider_calendar_capture_failure" not in calendar_authority.__all__
    source = inspect.getsource(calendar_authority)
    tree = ast.parse(source)
    targets = {
        "_build_trusted_provider_calendar_capture_execution",
        "_build_trusted_provider_calendar_capture_failure",
        "_build_trusted_provider_calendar_capture_success",
        "_publish_trusted_provider_calendar_capture_failure",
    }
    callers: dict[str, set[str]] = {target: set() for target in targets}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id in targets
            ):
                callers[child.func.id].add(node.name)
    assert callers == {
        "_build_trusted_provider_calendar_capture_execution": {
            "capture_trusted_provider_calendar_evidence"
        },
        "_build_trusted_provider_calendar_capture_success": {
            "capture_trusted_provider_calendar_evidence"
        },
        "_build_trusted_provider_calendar_capture_failure": {"_publish_capture_failure_root"},
        "_publish_trusted_provider_calendar_capture_failure": {"_raise_recorded_capture_failure"},
    }
    assert {
        "CAPTURE_FAILURE_KIND",
        "validate_published_trusted_provider_calendar_capture_failure_root",
        "validate_trusted_provider_calendar_capture_failure",
    } <= set(calendar_authority.__all__)


@pytest.mark.parametrize("root_name", ["run.failure", "run.FAILURE", "run.FaIlUrE"])
def test_capture_root_namespace_reserves_failure_suffix(
    tmp_path: Path,
    root_name: str,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    with pytest.raises(SystemSecurityError, match="root name is invalid"):
        calendar_authority._capture_root_name(root_name)
    with pytest.raises(SystemSecurityError, match="root name is invalid"):
        calendar_authority._publish_capture_tree(
            parent=parent,
            root_name=root_name,
            files={"payload.raw": b"payload"},
            success_builder=lambda _completed, _root_stat: b"success",
        )
    with pytest.raises(SystemSecurityError, match="root name is invalid"):
        calendar_authority._build_trusted_provider_calendar_capture_failure(
            capture_root_name=root_name,
            failed_at=CREATED_AT,
            error_code="TRUSTED_PROVIDER_CALENDAR_DOCUMENTATION_FETCH_FAILED",
            success_root_published=False,
            published_root_device=1,
            published_root_inode=1,
        )
    assert list(parent.iterdir()) == []


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
        "policy": build_calendar_authority_policy(
            created_at=CREATED_AT,
            pit_exchange_ids=["BSE", "SSE", "SZSE"],
            provider_capability=capability,
        ),
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


def test_direct_provider_literal_count_must_remain_zero() -> None:
    case = _case()
    value = json.loads(_provider_raw("SSE"))
    value["data"]["count"] = len(value["data"]["items"])
    raw = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    with pytest.raises(SystemContractError, match="direct response exchange differs"):
        build_trusted_provider_calendar_capture(
            exchange_id="SSE",
            raw=raw,
            raw_file_ref=_ref("raw/sse-nonzero-count.json", raw),
            capability=case["capability"],
            docs_raw=case["docs"],
            captured_at=CREATED_AT,
            capture_start_date=CAPTURE_START.isoformat(),
            cutoff_date=CUTOFF.isoformat(),
            request_parameters_sanitized={
                "end_date": CUTOFF.strftime("%Y%m%d"),
                "exchange": "SSE",
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


def test_public_capture_api_has_no_caller_time_or_transport_injection() -> None:
    import inspect

    parameters = inspect.signature(capture_trusted_provider_calendar_evidence).parameters
    assert "captured_at" not in parameters
    assert "client" not in parameters
    assert "documentation_fetcher" not in parameters
    assert {
        "capture_parent",
        "capture_root_name",
        "cutoff_date",
        "release_install_input_raw",
        "expected_release_install_input_sha256",
        "release_repository_root",
    } == set(parameters)


def test_capture_failure_never_calls_network_before_release_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    network_called = False

    def forbidden_fetch():
        nonlocal network_called
        network_called = True
        raise AssertionError("network must not run")

    monkeypatch.setattr(
        "quant_investor.market.tushare_calendar_authority._official_documentation_fetch",
        forbidden_fetch,
    )
    invalid = canonical_json_bytes({"not": "release-install-input"})
    with pytest.raises(SystemContractError, match="fields are not exact"):
        capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name="capture-fails",
            cutoff_date=CUTOFF.isoformat(),
            release_install_input_raw=invalid,
            expected_release_install_input_sha256=hashlib.sha256(invalid).hexdigest(),
            release_repository_root=tmp_path,
        )
    assert not (parent / "capture-fails").exists()
    assert network_called is False


def test_calendar_capture_cli_reports_immutable_failure_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from quant_investor.cli.main import main

    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    request_raw = canonical_json_bytes({"release": "input"})
    request_path = workspace / "release-install-input.json"
    request_path.write_bytes(request_raw)
    request_path.chmod(0o600)
    failure_ref = {
        "relative_path": "cli-capture.failure/capture-failure.json",
        "byte_sha256": "1" * 64,
    }

    def fail_capture(**_: Any) -> dict[str, Any]:
        error = SystemPreconditionError(
            "recorded capture failure",
            code="TRUSTED_PROVIDER_CALENDAR_DOCUMENTATION_FETCH_FAILED",
        )
        error.public_fields = {
            "capture_failure_id": "tushare-calendar-failure-cli",
            "capture_failure_root": str(tmp_path / "cli-capture.failure"),
            "capture_failure_file_ref": failure_ref,
            "success_root_published": False,
        }
        raise error

    monkeypatch.setattr(
        calendar_authority,
        "capture_trusted_provider_calendar_evidence",
        fail_capture,
    )
    with pytest.raises(SystemExit) as caught:
        main(
            [
                "system",
                "calendar-capture",
                "--workspace-root",
                str(workspace),
                "--capture-parent",
                str(tmp_path),
                "--capture-root-name",
                "cli-capture",
                "--cutoff-date",
                CUTOFF.isoformat(),
                "--release-repository-root",
                str(tmp_path),
                "--release-install-input",
                request_path.name,
                "--expected-release-install-input-sha256",
                hashlib.sha256(request_raw).hexdigest(),
            ]
        )
    assert caught.value.code == 2
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "status": "BLOCKED",
        "blocker_code": "TRUSTED_PROVIDER_CALENDAR_DOCUMENTATION_FETCH_FAILED",
        "capture_failure_id": "tushare-calendar-failure-cli",
        "capture_failure_root": str(tmp_path / "cli-capture.failure"),
        "capture_failure_file_ref": failure_ref,
        "success_root_published": False,
    }


def _fake_release_install_closure(
    tmp_path: Path,
    *,
    release: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    release = release or seal_artifact(
        "system.release",
        {
            "release_id": "installed-calendar-capture-test",
            "state": "OPERATIONAL",
            "code_sha256": "1" * 64,
            "wheel_sha256": "2" * 64,
            "code_manifest_sha256": "3" * 64,
        },
        created_at=CREATED_AT,
    )
    archive = tmp_path / "release.tar.gz"
    wheel = tmp_path / "release.whl"
    archive.write_bytes(b"archive")
    wheel.write_bytes(b"wheel")
    evidence = build_release_install_evidence(
        final_commit="4" * 40,
        final_tree="5" * 40,
        code_tree_sha256_value="6" * 64,
        git_code_manifest_sha256_value="7" * 64,
        release_ref=object_ref_for_artifact(release),
        source_archive={
            "path": str(archive),
            "byte_sha256": hashlib.sha256(b"archive").hexdigest(),
            "size": len(b"archive"),
        },
        wheel={
            "path": str(wheel),
            "byte_sha256": hashlib.sha256(b"wheel").hexdigest(),
            "size": len(b"wheel"),
        },
        install_root=str(tmp_path / "installed"),
        python_executable=str(tmp_path / "installed/bin/python"),
        python_executable_sha256="8" * 64,
        import_origin=str(tmp_path / "installed/quant_investor/__init__.py"),
        installed_code_manifest_sha256="9" * 64,
        contract_catalog_sha256_value="a" * 64,
        lockfile_sha256="b" * 64,
        created_at=CREATED_AT,
    )
    exact_input = canonical_json_bytes(
        {"release_install_evidence": evidence, "deployed_release": release}
    )
    verification = {
        "wheel_sha256": evidence["payload"]["wheel"]["byte_sha256"],
        "installed_code_manifest_sha256": evidence["payload"]["installed_code_manifest_sha256"],
        "contract_catalog_sha256": evidence["payload"]["contract_catalog_sha256"],
        "import_origin": evidence["payload"]["import_origin"],
    }
    return exact_input, release, {"evidence": evidence, "verification": verification}


def _patch_capture_release_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> bytes:
    exact_input, release, closure = _fake_release_install_closure(tmp_path)

    def fake_release_components(raw: bytes, **_: Any):
        assert raw == exact_input
        return closure["evidence"], release, closure["verification"]

    monkeypatch.setattr(
        calendar_authority,
        "_release_install_components",
        fake_release_components,
    )
    return exact_input


def _assert_recorded_capture_failure(
    *,
    parent: Path,
    caught: pytest.ExceptionInfo[SystemPreconditionError],
    capture_root_name: str,
    expected_code: str,
    success_root_published: bool,
) -> dict[str, Any]:
    assert caught.value.code == expected_code
    public = caught.value.public_fields
    assert public["success_root_published"] is success_root_published
    assert public["capture_failure_root"] == str(parent / f"{capture_root_name}.failure")
    reference = public["capture_failure_file_ref"]
    failure_raw = (parent / reference["relative_path"]).read_bytes()
    failure = validate_trusted_provider_calendar_capture_failure(failure_raw)
    assert failure["payload"] == {
        "capture_failure_id": public["capture_failure_id"],
        "state": "FAILED",
        "capture_root_name": capture_root_name,
        "failed_at": failure["created_at"],
        "error_code": expected_code,
        "success_root_published": success_root_published,
        "published_root_device": failure["payload"]["published_root_device"],
        "published_root_inode": failure["payload"]["published_root_inode"],
    }
    assert (
        validate_published_trusted_provider_calendar_capture_failure_root(
            capture_parent=parent,
            capture_failure=failure,
            capture_failure_file_ref=reference,
        )
        == failure_raw
    )
    assert sorted(path.name for path in (parent / f"{capture_root_name}.failure").iterdir()) == [
        "capture-failure.json"
    ]
    assert b"secret" not in failure_raw.lower()
    assert b"authorization" not in failure_raw.lower()
    return failure


def test_invalid_cutoff_is_rejected_before_release_or_network_and_consumes_no_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    exact_input = b"release-input"
    release_calls = 0
    documentation_calls = 0

    def forbidden_release(*_: Any, **__: Any):
        nonlocal release_calls
        release_calls += 1
        raise AssertionError("release closure must not run")

    def forbidden_documentation():
        nonlocal documentation_calls
        documentation_calls += 1
        raise AssertionError("network must not run")

    monkeypatch.setattr(calendar_authority, "_release_install_components", forbidden_release)
    monkeypatch.setattr(
        calendar_authority, "_official_documentation_fetch", forbidden_documentation
    )
    with pytest.raises(SystemPreconditionError, match="cutoff is invalid"):
        capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name="invalid-cutoff",
            cutoff_date="2023-12-31",
            release_install_input_raw=exact_input,
            expected_release_install_input_sha256=hashlib.sha256(exact_input).hexdigest(),
            release_repository_root=tmp_path,
        )
    assert release_calls == 0
    assert documentation_calls == 0
    assert list(parent.iterdir()) == []


@pytest.mark.parametrize(
    ("failure_stage", "expected_code", "expected_calls"),
    [
        (
            "DOCS",
            "TRUSTED_PROVIDER_CALENDAR_DOCUMENTATION_FETCH_FAILED",
            [],
        ),
        (
            "SSE",
            "TRUSTED_PROVIDER_CALENDAR_SSE_TUSHARE_TRANSPORT_ERROR",
            ["SSE"],
        ),
        (
            "SZSE",
            "TRUSTED_PROVIDER_CALENDAR_SZSE_TUSHARE_TRANSPORT_ERROR",
            ["SSE", "SZSE"],
        ),
        (
            "BSE",
            "TRUSTED_PROVIDER_CALENDAR_BSE_TUSHARE_TRANSPORT_ERROR",
            ["SSE", "SZSE", "BSE"],
        ),
    ],
)
def test_post_release_capture_failures_publish_sanitized_sibling_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    expected_code: str,
    expected_calls: list[str],
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    exact_input = _patch_capture_release_closure(tmp_path, monkeypatch)
    calls: list[str] = []
    secret = "do-not-retain-this-secret"

    if failure_stage == "DOCS":

        def docs_failure():
            raise SystemPreconditionError(f"documentation failed {secret}")

        monkeypatch.setattr(
            calendar_authority,
            "_official_documentation_fetch",
            docs_failure,
        )
    else:
        monkeypatch.setattr(
            calendar_authority,
            "_official_documentation_fetch",
            lambda: (
                _docs(),
                200,
                {"content-type": "text/html; charset=utf-8"},
                True,
                [],
            ),
        )

    class FakeClient:
        def __init__(self, **_: Any) -> None:
            pass

        def request(self, *, api_name: str, params: dict[str, str], **_: Any):
            exchange = params["exchange"]
            calls.append(exchange)
            if exchange == failure_stage:
                raise TushareHttpsError("TUSHARE_TRANSPORT_ERROR")
            return replay_tushare_response_bytes(
                _provider_raw(exchange),
                api_name=api_name,
                expected_fields=EXPECTED_FIELDS,
                strict_decimal_decode=True,
            )

    monkeypatch.setattr(calendar_authority, "OfficialTushareHttpsClient", FakeClient)
    with pytest.raises(SystemPreconditionError) as caught:
        capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name=f"capture-{failure_stage.lower()}",
            cutoff_date=CUTOFF.isoformat(),
            release_install_input_raw=exact_input,
            expected_release_install_input_sha256=hashlib.sha256(exact_input).hexdigest(),
            release_repository_root=tmp_path,
        )
    _assert_recorded_capture_failure(
        parent=parent,
        caught=caught,
        capture_root_name=f"capture-{failure_stage.lower()}",
        expected_code=expected_code,
        success_root_published=False,
    )
    assert calls == expected_calls
    assert not (parent / f"capture-{failure_stage.lower()}").exists()


@pytest.mark.parametrize(
    "documentation_error",
    [
        SystemPreconditionError("TRUSTED_PROVIDER_DOCUMENTATION_HTTP_FAILED"),
        SystemSecurityError("TRUSTED_PROVIDER_DOCUMENTATION_TLS_FAILED"),
        SystemSecurityError("TRUSTED_PROVIDER_DOCUMENTATION_REDIRECT_BLOCKED"),
    ],
)
def test_documentation_http_tls_and_redirect_failures_are_terminally_recorded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    documentation_error: Exception,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    exact_input = _patch_capture_release_closure(tmp_path, monkeypatch)

    def fail_documentation():
        raise documentation_error

    monkeypatch.setattr(
        calendar_authority,
        "_official_documentation_fetch",
        fail_documentation,
    )
    root_name = "capture-docs-" + documentation_error.args[0].lower().replace("_", "-")
    with pytest.raises(SystemPreconditionError) as caught:
        capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name=root_name,
            cutoff_date=CUTOFF.isoformat(),
            release_install_input_raw=exact_input,
            expected_release_install_input_sha256=hashlib.sha256(exact_input).hexdigest(),
            release_repository_root=tmp_path,
        )
    _assert_recorded_capture_failure(
        parent=parent,
        caught=caught,
        capture_root_name=root_name,
        expected_code="TRUSTED_PROVIDER_CALENDAR_DOCUMENTATION_FETCH_FAILED",
        success_root_published=False,
    )


def test_arbitrary_exception_cannot_forge_success_publication_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    exact_input = _patch_capture_release_closure(tmp_path, monkeypatch)
    forged = SystemPreconditionError("sensitive forged publication state")
    setattr(forged, "_trusted_provider_capture_root_published", True)
    monkeypatch.setattr(
        calendar_authority,
        "_official_documentation_fetch",
        lambda: (_ for _ in ()).throw(forged),
    )
    with pytest.raises(SystemPreconditionError) as caught:
        capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name="forged-publication-state",
            cutoff_date=CUTOFF.isoformat(),
            release_install_input_raw=exact_input,
            expected_release_install_input_sha256=hashlib.sha256(exact_input).hexdigest(),
            release_repository_root=tmp_path,
        )
    failure = _assert_recorded_capture_failure(
        parent=parent,
        caught=caught,
        capture_root_name="forged-publication-state",
        expected_code="TRUSTED_PROVIDER_CALENDAR_DOCUMENTATION_FETCH_FAILED",
        success_root_published=False,
    )
    assert b"sensitive" not in canonical_json_bytes(failure).lower()


@pytest.mark.parametrize(
    ("fault", "success_root_published", "success_marker_exists"),
    [
        ("PRE_RENAME", False, False),
        ("RENAME_COMPLETED_THEN_RAISED", True, False),
        ("POST_RENAME", True, False),
        ("POST_SUCCESS_MARKER", True, True),
    ],
)
def test_publication_failure_records_exact_publication_state_without_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    success_root_published: bool,
    success_marker_exists: bool,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    exact_input = _patch_capture_release_closure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        calendar_authority,
        "_official_documentation_fetch",
        lambda: (
            _docs(),
            200,
            {"content-type": "text/html; charset=utf-8"},
            True,
            [],
        ),
    )

    class FakeClient:
        def __init__(self, **_: Any) -> None:
            pass

        def request(self, *, api_name: str, params: dict[str, str], **_: Any):
            return replay_tushare_response_bytes(
                _provider_raw(params["exchange"]),
                api_name=api_name,
                expected_fields=EXPECTED_FIELDS,
                strict_decimal_decode=True,
            )

    monkeypatch.setattr(calendar_authority, "OfficialTushareHttpsClient", FakeClient)
    original_rename = calendar_authority._rename_no_replace
    original_read = calendar_authority._read_directory_files
    rename_calls = 0
    read_calls = 0

    def fail_first_rename(parent_fd: int, source: str, target: str) -> None:
        nonlocal rename_calls
        rename_calls += 1
        if fault == "PRE_RENAME" and rename_calls == 1:
            raise SystemSecurityError("sensitive pre-rename failure")
        original_rename(parent_fd, source, target)
        if fault == "RENAME_COMPLETED_THEN_RAISED" and rename_calls == 1:
            raise KeyboardInterrupt("sensitive rename-boundary interrupt")

    def fail_selected_read(*args: Any, **kwargs: Any) -> None:
        nonlocal read_calls
        read_calls += 1
        fail_at = 2 if fault == "POST_RENAME" else 3
        if fault in {"POST_RENAME", "POST_SUCCESS_MARKER"} and read_calls == fail_at:
            raise SystemSecurityError("sensitive post-publication failure")
        original_read(*args, **kwargs)

    monkeypatch.setattr(calendar_authority, "_rename_no_replace", fail_first_rename)
    monkeypatch.setattr(calendar_authority, "_read_directory_files", fail_selected_read)
    root_name = f"capture-{fault.lower().replace('_', '-')}"
    with pytest.raises(SystemPreconditionError) as caught:
        capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name=root_name,
            cutoff_date=CUTOFF.isoformat(),
            release_install_input_raw=exact_input,
            expected_release_install_input_sha256=hashlib.sha256(exact_input).hexdigest(),
            release_repository_root=tmp_path,
        )
    _assert_recorded_capture_failure(
        parent=parent,
        caught=caught,
        capture_root_name=root_name,
        expected_code="TRUSTED_PROVIDER_CALENDAR_PUBLICATION_FAILED",
        success_root_published=success_root_published,
    )
    success_root = parent / root_name
    assert success_root.exists() is success_root_published
    assert (success_root / "capture-success.json").exists() is success_marker_exists
    before = (
        {path.name: path.read_bytes() for path in success_root.iterdir()}
        if success_root.exists()
        else {}
    )
    validate_published_trusted_provider_calendar_capture_failure_root(
        capture_parent=parent,
        capture_failure=json.loads(
            (parent / f"{root_name}.failure/{calendar_authority._FAILURE_LEAF}").read_bytes()
        ),
        capture_failure_file_ref=caught.value.public_fields["capture_failure_file_ref"],
    )
    after = (
        {path.name: path.read_bytes() for path in success_root.iterdir()}
        if success_root.exists()
        else {}
    )
    assert after == before


@pytest.mark.parametrize(
    ("fault", "expected_code"),
    [
        ("DOCUMENTATION_DECODER", "TRUSTED_PROVIDER_CALENDAR_EVIDENCE_INVALID"),
        (
            "PROVIDER_REPLAY",
            "TRUSTED_PROVIDER_CALENDAR_SSE_TUSHARE_RESPONSE_INVALID",
        ),
    ],
)
def test_decode_failures_publish_only_sanitized_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    expected_code: str,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    exact_input = _patch_capture_release_closure(tmp_path, monkeypatch)
    docs = (
        b"not a valid trade-cal documentation contract"
        if fault == "DOCUMENTATION_DECODER"
        else _docs()
    )
    monkeypatch.setattr(
        calendar_authority,
        "_official_documentation_fetch",
        lambda: (
            docs,
            200,
            {"content-type": "text/html; charset=utf-8"},
            True,
            [],
        ),
    )

    class FakeClient:
        def __init__(self, **_: Any) -> None:
            pass

        def request(self, *, api_name: str, params: dict[str, str], **_: Any):
            raw = (
                b'{"authorization":"sensitive","broken":true}'
                if fault == "PROVIDER_REPLAY" and params["exchange"] == "SSE"
                else _provider_raw(params["exchange"])
            )
            return replay_tushare_response_bytes(
                raw,
                api_name=api_name,
                expected_fields=EXPECTED_FIELDS,
                strict_decimal_decode=True,
            )

    monkeypatch.setattr(calendar_authority, "OfficialTushareHttpsClient", FakeClient)
    root_name = f"capture-{fault.lower().replace('_', '-')}"
    with pytest.raises(SystemPreconditionError) as caught:
        capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name=root_name,
            cutoff_date=CUTOFF.isoformat(),
            release_install_input_raw=exact_input,
            expected_release_install_input_sha256=hashlib.sha256(exact_input).hexdigest(),
            release_repository_root=tmp_path,
        )
    failure = _assert_recorded_capture_failure(
        parent=parent,
        caught=caught,
        capture_root_name=root_name,
        expected_code=expected_code,
        success_root_published=False,
    )
    encoded = canonical_json_bytes(failure).lower()
    assert b"sensitive" not in encoded
    assert b"broken" not in encoded


def test_public_capture_publishes_success_last_with_exact_owner_only_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    exact_input, release, closure = _fake_release_install_closure(tmp_path)
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_release_components(raw: bytes, **_: Any):
        assert raw == exact_input
        return closure["evidence"], release, closure["verification"]

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["max_response_bytes"] == 4 * 1024 * 1024
            assert kwargs["max_response_items"] == 2_000

        def request(self, *, api_name: str, params: dict[str, str], **_: Any):
            calls.append((api_name, dict(params)))
            return replay_tushare_response_bytes(
                _provider_raw(params["exchange"]),
                api_name=api_name,
                expected_fields=EXPECTED_FIELDS,
                strict_decimal_decode=True,
            )

    monkeypatch.setattr(calendar_authority, "_release_install_components", fake_release_components)
    monkeypatch.setattr(
        calendar_authority,
        "_official_documentation_fetch",
        lambda: (
            _docs(),
            200,
            {"content-type": "text/html; charset=utf-8"},
            True,
            [],
        ),
    )
    monkeypatch.setattr(calendar_authority, "OfficialTushareHttpsClient", FakeClient)

    result = capture_trusted_provider_calendar_evidence(
        capture_parent=parent,
        capture_root_name="capture-success",
        cutoff_date=CUTOFF.isoformat(),
        release_install_input_raw=exact_input,
        expected_release_install_input_sha256=hashlib.sha256(exact_input).hexdigest(),
        release_repository_root=tmp_path,
    )

    root = parent / "capture-success"
    assert result["status"] == "CAPTURED"
    assert [exchange for _, params in calls for exchange in [params["exchange"]]] == [
        "SSE",
        "SZSE",
        "BSE",
    ]
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    leaves = sorted(path.name for path in root.iterdir())
    assert leaves == sorted(
        [
            "capability.json",
            "capture-bse.json",
            "capture-execution.json",
            "capture-sse.json",
            "capture-success.json",
            "capture-szse.json",
            "capture-transaction.json",
            "documentation.raw",
            "policy.json",
            "release-install-input.json",
            "response-bse.raw",
            "response-sse.raw",
            "response-szse.raw",
        ]
    )
    for leaf in root.iterdir():
        observed = leaf.stat()
        assert stat.S_IMODE(observed.st_mode) == 0o600
        assert observed.st_nlink == 1
        assert b"token" not in leaf.read_bytes().lower()
    assert (
        result["capture_success"]["payload"]["observed_completed_at"]
        >= result["capture_execution"]["payload"]["observed_completed_at"]
    )


def _captured_provider_production_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, dict[str, str], dict[str, Any], dict[str, Any]]:
    production_cutoff = date(2026, 8, 7)
    monkeypatch.setattr(
        production_test_module,
        "SYMBOLS",
        ["000001.SZ", "000002.SZ", "430001.BJ", "600000.SH", "600001.SH"],
    )
    workspace, release_ref = _seed_workspace(tmp_path)
    release = SystemStore(workspace).get_object(release_ref)
    input_root = tmp_path / "provider-production-inputs"
    files = _inputs(input_root)
    exact_input, _release, closure = _fake_release_install_closure(
        tmp_path,
        release=release,
    )
    release_root = tmp_path / "detached-release-root"
    release_root.mkdir(mode=0o700)

    def fake_release_components(raw: bytes, **_: Any):
        assert raw == exact_input
        return closure["evidence"], release, closure["verification"]

    class FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def request(self, *, api_name: str, params: dict[str, str], **_: Any):
            return replay_tushare_response_bytes(
                _provider_raw(params["exchange"], cutoff=production_cutoff),
                api_name=api_name,
                expected_fields=EXPECTED_FIELDS,
                strict_decimal_decode=True,
            )

    monkeypatch.setattr(calendar_authority, "_release_install_components", fake_release_components)
    monkeypatch.setattr(calendar_authority, "_utc_now", lambda: "2026-08-14T00:00:00Z")
    monkeypatch.setattr(
        calendar_authority,
        "_official_documentation_fetch",
        lambda: (
            _docs(),
            200,
            {"content-type": "text/html; charset=utf-8"},
            True,
            [],
        ),
    )
    monkeypatch.setattr(calendar_authority, "OfficialTushareHttpsClient", FakeClient)
    capture = capture_trusted_provider_calendar_evidence(
        capture_parent=input_root,
        capture_root_name="provider-production-capture",
        cutoff_date=production_cutoff.isoformat(),
        release_install_input_raw=exact_input,
        expected_release_install_input_sha256=hashlib.sha256(exact_input).hexdigest(),
        release_repository_root=release_root,
    )
    runtime = _runtime(cutoff=production_cutoff)
    calendar_json = runtime_json_bytes(runtime)
    calendar_parquet = runtime_parquet_bytes(runtime)
    _write(input_root, "strict/exchange_calendar.json", calendar_json)
    _write(input_root, "strict/exchange_calendar.parquet", calendar_parquet)
    json_ref = _byte_ref(input_root, "strict/exchange_calendar.json")
    parquet_ref = _byte_ref(input_root, "strict/exchange_calendar.parquet")
    raw_by_ref = {
        canonical_json_bytes(reference): (input_root / reference["relative_path"]).read_bytes()
        for reference in capture["trusted_provider_calendar_raw_file_refs"]
    }
    raw_by_ref[canonical_json_bytes(json_ref)] = calendar_json
    raw_by_ref[canonical_json_bytes(parquet_ref)] = calendar_parquet

    def raw_resolver(reference: dict[str, Any]) -> bytes:
        return raw_by_ref[canonical_json_bytes(reference)]

    market_sessions = [
        row["date"] for row in runtime if row["status"] == "OPEN" and row["date"] >= "2024-01-02"
    ][-100:]
    policy_document = json.loads(
        (input_root / capture["calendar_authority_policy_file_ref"]["relative_path"])
        .read_bytes()
        .decode()
    )
    capability_document = json.loads(
        (input_root / capture["trusted_provider_calendar_capability_file_ref"]["relative_path"])
        .read_bytes()
        .decode()
    )
    capture_documents = [
        json.loads((input_root / reference["relative_path"]).read_bytes().decode())
        for reference in capture["trusted_provider_calendar_capture_file_refs"]
    ]
    docs_raw = (
        input_root / capture["trusted_provider_calendar_raw_file_refs"][0]["relative_path"]
    ).read_bytes()
    compilation = build_trusted_provider_calendar_compilation(
        compilation_id="provider-production-compilation",
        policy=policy_document,
        capability=capability_document,
        capture_documents=capture_documents,
        docs_raw=docs_raw,
        raw_resolver=raw_resolver,
        release_ref=release_ref,
        pit_exchange_ids=["BSE", "SSE", "SZSE"],
        market_session_dates=market_sessions,
        cutoff_date=production_cutoff.isoformat(),
        calendar_json_file_ref=json_ref,
        calendar_parquet_file_ref=parquet_ref,
        created_at="2026-08-14T00:00:00Z",
    )
    compilation_path = "closure/provider-calendar-compilation.json"
    _write(input_root, compilation_path, canonical_json_bytes(compilation))
    files.update(
        {
            "exchange_calendar_file_ref": parquet_ref,
            "calendar_runtime_json_file_ref": json_ref,
            "calendar_compilation_file_ref": _byte_ref(input_root, compilation_path),
            "calendar_authority_policy_file_ref": capture["calendar_authority_policy_file_ref"],
            "official_calendar_raw_file_refs": [],
            "official_calendar_capture_file_refs": [],
            "official_calendar_decoder_admission_file_refs": [],
            "official_calendar_index_closure_file_refs": [],
            "trusted_provider_calendar_raw_file_refs": sorted(
                capture["trusted_provider_calendar_raw_file_refs"],
                key=lambda row: row["relative_path"],
            ),
            "trusted_provider_calendar_capture_file_refs": sorted(
                capture["trusted_provider_calendar_capture_file_refs"],
                key=lambda row: row["relative_path"],
            ),
            "trusted_provider_calendar_capability_file_ref": capture[
                "trusted_provider_calendar_capability_file_ref"
            ],
            "trusted_provider_calendar_capture_transaction_file_ref": capture[
                "capture_transaction_file_ref"
            ],
            "trusted_provider_calendar_capture_execution_file_ref": capture[
                "capture_execution_file_ref"
            ],
            "trusted_provider_calendar_capture_success_file_ref": capture[
                "capture_success_file_ref"
            ],
            "trusted_provider_release_install_input_file_ref": capture[
                "release_install_input_file_ref"
            ],
        }
    )
    return workspace, input_root, release_ref, files, capture


def test_fixed_capture_root_enters_provider_production_assembly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, input_root, release_ref, files, _capture = _captured_provider_production_case(
        tmp_path,
        monkeypatch,
    )
    result = assemble_production_bootstrap(
        workspace_root=workspace,
        input_root=input_root,
        request_raw=_request(
            workspace_root=workspace,
            release_ref=release_ref,
            files=files,
            operation_id="provider-production-positive",
        ),
    )
    assert result["status"] == "OFFLINE_VERIFIED"
    assert result["generation"]["verified"] is True
    assert result["generation"]["calendar_authority_route"] == "TRUSTED_PROVIDER_DEGRADED"
    assert result["generation"]["calendar_source_limitations"] == list(SOURCE_LIMITATIONS)
    assert not (workspace / "results/system/_active.json").exists()


def test_rehomed_execution_and_success_artifacts_cannot_enter_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace, input_root, _release_ref, files, _capture = _captured_provider_production_case(
        tmp_path,
        monkeypatch,
    )
    rehomed = tmp_path / "rehomed-provider-inputs"
    shutil.copytree(input_root, rehomed)
    rehomed.chmod(0o700)
    rehomed_case = tmp_path / "rehomed-workspace-case"
    rehomed_case.mkdir(mode=0o700)
    workspace, release_ref = _seed_workspace(rehomed_case)
    with pytest.raises(SystemSecurityError, match="published root identity differs"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=rehomed,
            request_raw=_request(
                workspace_root=workspace,
                release_ref=release_ref,
                files=files,
                operation_id="provider-production-rehomed",
            ),
        )
    assert not (workspace / "results/system/_active.json").exists()


def test_failure_artifact_or_mixed_success_root_cannot_enter_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, input_root, release_ref, files, capture = _captured_provider_production_case(
        tmp_path,
        monkeypatch,
    )
    failure = calendar_authority._publish_trusted_provider_calendar_capture_failure(
        capture_parent=input_root,
        capture_root_name=capture["capture_transaction"]["payload"]["capture_root_name"],
        failed_at=CREATED_AT,
        error_code="TRUSTED_PROVIDER_CALENDAR_SUCCESS_VALIDATION_FAILED",
        success_root_published=True,
    )
    original_success_ref = files["trusted_provider_calendar_capture_success_file_ref"]
    files["trusted_provider_calendar_capture_success_file_ref"] = failure[
        "capture_failure_file_ref"
    ]
    with pytest.raises((SystemContractError, SystemPreconditionError, SystemSecurityError)):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(
                workspace_root=workspace,
                release_ref=release_ref,
                files=files,
                operation_id="provider-production-failure-substitution",
            ),
        )
    files["trusted_provider_calendar_capture_success_file_ref"] = original_success_ref
    shutil.rmtree(Path(failure["capture_failure_root"]))
    success_root = input_root / capture["capture_transaction"]["payload"]["capture_root_name"]
    mixed_leaf = success_root / "capture-failure.json"
    mixed_leaf.write_bytes(canonical_json_bytes(failure["capture_failure"]))
    mixed_leaf.chmod(0o600)
    with pytest.raises(SystemSecurityError, match="topology differs"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(
                workspace_root=workspace,
                release_ref=release_ref,
                files=files,
                operation_id="provider-production-mixed-root",
            ),
        )
    assert not (workspace / "results/system/_active.json").exists()
    assert not (workspace / "results/system/_migration_complete.json").exists()


def test_capture_publication_is_no_replace_with_exactly_one_winner(tmp_path: Path) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    files = {"payload.raw": b"payload"}

    def publish() -> str:
        try:
            calendar_authority._publish_capture_tree(
                parent=parent,
                root_name="one-winner",
                files=files,
                success_builder=lambda _completed, _root_stat: b"success",
            )
        except SystemPreconditionError:
            return "BLOCKED"
        return "PUBLISHED"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(lambda _: publish(), range(2)))
    assert outcomes == ["BLOCKED", "PUBLISHED"]
    assert (parent / "one-winner/payload.raw").read_bytes() == b"payload"
    assert (parent / "one-winner/capture-success.json").read_bytes() == b"success"


def test_post_rename_readback_failure_leaves_no_success_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    original = calendar_authority._read_directory_files
    readbacks = 0

    def fail_after_rename(*args: Any, **kwargs: Any) -> None:
        nonlocal readbacks
        readbacks += 1
        if readbacks == 2:
            raise SystemSecurityError("forced post-rename readback failure")
        original(*args, **kwargs)

    monkeypatch.setattr(calendar_authority, "_read_directory_files", fail_after_rename)
    with pytest.raises(calendar_authority._CapturePublicationFailure) as caught:
        calendar_authority._publish_capture_tree(
            parent=parent,
            root_name="incomplete",
            files={"payload.raw": b"payload"},
            success_builder=lambda _completed, _root_stat: b"success",
        )
    assert caught.value.success_root_published is True
    assert isinstance(caught.value.cause, SystemSecurityError)
    assert "forced post-rename" in str(caught.value.cause)
    assert "forced post-rename" not in str(caught.value)
    assert (parent / "incomplete/payload.raw").read_bytes() == b"payload"
    assert not (parent / "incomplete/capture-success.json").exists()


def test_rename_boundary_inode_mismatch_is_ambiguous_and_not_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    original_rename = calendar_authority._rename_no_replace

    def replace_target_after_rename(parent_fd: int, source: str, target: str) -> None:
        original_rename(parent_fd, source, target)
        os.rename(
            target,
            f"{target}.displaced",
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir(target, mode=0o700, dir_fd=parent_fd)
        raise KeyboardInterrupt("sensitive ambiguous rename boundary")

    monkeypatch.setattr(
        calendar_authority,
        "_rename_no_replace",
        replace_target_after_rename,
    )
    with pytest.raises(calendar_authority._CapturePublicationFailure) as caught:
        calendar_authority._publish_capture_tree(
            parent=parent,
            root_name="ambiguous-boundary",
            files={"payload.raw": b"payload"},
            success_builder=lambda _completed, _root_stat: b"success",
        )
    assert caught.value.success_root_published is None
    assert isinstance(caught.value.cause, SystemStorageError)
    assert (parent / "ambiguous-boundary").is_dir()
    assert (parent / "ambiguous-boundary.displaced/payload.raw").read_bytes() == b"payload"
    assert not (parent / "ambiguous-boundary.failure").exists()


def test_public_capture_does_not_seal_false_state_for_ambiguous_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    exact_input = _patch_capture_release_closure(tmp_path, monkeypatch)
    monkeypatch.setattr(
        calendar_authority,
        "_official_documentation_fetch",
        lambda: (
            _docs(),
            200,
            {"content-type": "text/html; charset=utf-8"},
            True,
            [],
        ),
    )

    class FakeClient:
        def __init__(self, **_: Any) -> None:
            pass

        def request(self, *, api_name: str, params: dict[str, str], **_: Any):
            return replay_tushare_response_bytes(
                _provider_raw(params["exchange"]),
                api_name=api_name,
                expected_fields=EXPECTED_FIELDS,
                strict_decimal_decode=True,
            )

    monkeypatch.setattr(calendar_authority, "OfficialTushareHttpsClient", FakeClient)
    original_rename = calendar_authority._rename_no_replace

    def replace_target_after_rename(parent_fd: int, source: str, target: str) -> None:
        original_rename(parent_fd, source, target)
        os.rename(
            target,
            f"{target}.displaced",
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir(target, mode=0o700, dir_fd=parent_fd)
        raise KeyboardInterrupt("sensitive public ambiguity")

    monkeypatch.setattr(
        calendar_authority,
        "_rename_no_replace",
        replace_target_after_rename,
    )
    with pytest.raises(SystemStorageError, match="outcome is ambiguous"):
        capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name="public-ambiguous",
            cutoff_date=CUTOFF.isoformat(),
            release_install_input_raw=exact_input,
            expected_release_install_input_sha256=hashlib.sha256(exact_input).hexdigest(),
            release_repository_root=tmp_path,
        )
    assert not (parent / "public-ambiguous.failure").exists()


def test_failure_publication_rename_boundary_preserves_published_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    original_rename = calendar_authority._rename_no_replace

    def rename_then_interrupt(parent_fd: int, source: str, target: str) -> None:
        original_rename(parent_fd, source, target)
        raise KeyboardInterrupt("sensitive failure publication boundary")

    monkeypatch.setattr(
        calendar_authority,
        "_rename_no_replace",
        rename_then_interrupt,
    )
    with pytest.raises(KeyboardInterrupt, match="failure publication boundary"):
        _publish_failure_case(parent, "failure-boundary")
    failure_root = parent / "failure-boundary.failure"
    failure_raw = (failure_root / calendar_authority._FAILURE_LEAF).read_bytes()
    failure = validate_trusted_provider_calendar_capture_failure(failure_raw)
    failure_ref = {
        "relative_path": "failure-boundary.failure/capture-failure.json",
        "byte_sha256": hashlib.sha256(failure_raw).hexdigest(),
    }
    assert (
        validate_published_trusted_provider_calendar_capture_failure_root(
            capture_parent=parent,
            capture_failure=failure,
            capture_failure_file_ref=failure_ref,
        )
        == failure_raw
    )
    assert not any(
        path.name.startswith(".failure-boundary.failure.staging-") for path in parent.iterdir()
    )


def test_failure_publication_inode_mismatch_is_ambiguous_and_not_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    original_rename = calendar_authority._rename_no_replace

    def replace_target_after_rename(parent_fd: int, source: str, target: str) -> None:
        original_rename(parent_fd, source, target)
        os.rename(
            target,
            f"{target}.displaced",
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        os.mkdir(target, mode=0o700, dir_fd=parent_fd)
        raise KeyboardInterrupt("sensitive failure ambiguity")

    monkeypatch.setattr(
        calendar_authority,
        "_rename_no_replace",
        replace_target_after_rename,
    )
    with pytest.raises(SystemStorageError, match="outcome is ambiguous"):
        _publish_failure_case(parent, "failure-ambiguous")
    assert (parent / "failure-ambiguous.failure").is_dir()
    assert (
        parent / "failure-ambiguous.failure.displaced" / calendar_authority._FAILURE_LEAF
    ).is_file()


def test_capture_publication_rejects_unsafe_parent_mode(tmp_path: Path) -> None:
    parent = tmp_path / "unsafe"
    parent.mkdir(mode=0o755)
    with pytest.raises(SystemSecurityError, match="directory is unsafe"):
        calendar_authority._publish_capture_tree(
            parent=parent,
            root_name="blocked",
            files={"payload.raw": b"payload"},
            success_builder=lambda _completed, _root_stat: b"success",
        )
    assert not (parent / "blocked").exists()


def _publish_failure_case(parent: Path, root_name: str = "failed-capture") -> dict[str, Any]:
    return calendar_authority._publish_trusted_provider_calendar_capture_failure(
        capture_parent=parent,
        capture_root_name=root_name,
        failed_at=CREATED_AT,
        error_code="TRUSTED_PROVIDER_CALENDAR_DOCUMENTATION_FETCH_FAILED",
        success_root_published=False,
    )


def test_failure_builder_rejects_tamper_sensitive_code_and_rehoming(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    result = _publish_failure_case(parent)
    failure = result["capture_failure"]
    validate_trusted_provider_calendar_capture_failure(failure)
    tampered = json.loads(canonical_json_bytes(failure))
    tampered["payload"]["success_root_published"] = True
    with pytest.raises(SystemContractError):
        validate_trusted_provider_calendar_capture_failure(tampered)
    with pytest.raises(SystemSecurityError, match="sensitive"):
        calendar_authority._build_trusted_provider_calendar_capture_failure(
            capture_root_name="sensitive-code",
            failed_at=CREATED_AT,
            error_code="AUTHORIZATION_SECRET",
            success_root_published=False,
            published_root_device=1,
            published_root_inode=1,
        )
    rehomed = tmp_path / "rehomed"
    rehomed.mkdir(mode=0o700)
    shutil.copytree(
        parent / "failed-capture.failure",
        rehomed / "failed-capture.failure",
    )
    with pytest.raises(SystemSecurityError, match="root identity differs"):
        validate_published_trusted_provider_calendar_capture_failure_root(
            capture_parent=rehomed,
            capture_failure=failure,
            capture_failure_file_ref=result["capture_failure_file_ref"],
        )


def test_failure_publication_rejects_symlink_targets_and_ancestors(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(SystemSecurityError, match="traversal is unsafe"):
        _publish_failure_case(linked_parent, "symlink-ancestor")
    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    (real_parent / "symlink-target.failure").symlink_to(
        target,
        target_is_directory=True,
    )
    with pytest.raises(SystemPreconditionError, match="destination exists"):
        _publish_failure_case(real_parent, "symlink-target")
    assert not any(
        path.name.startswith(".symlink-target.failure.staging") for path in real_parent.iterdir()
    )


@pytest.mark.parametrize("mutation", ["LEAF_MODE", "ROOT_MODE", "HARDLINK", "OWNER"])
def test_failure_readback_rejects_mode_link_and_owner_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    result = _publish_failure_case(parent, f"failure-{mutation.lower()}")
    root = Path(result["capture_failure_root"])
    leaf = root / "capture-failure.json"
    if mutation == "LEAF_MODE":
        leaf.chmod(0o644)
    elif mutation == "ROOT_MODE":
        root.chmod(0o755)
    elif mutation == "HARDLINK":
        os.link(leaf, tmp_path / "failure-hardlink-copy")
    else:
        monkeypatch.setattr(calendar_authority.os, "geteuid", lambda: os.getuid() + 1)
    with pytest.raises(SystemSecurityError, match="unsafe"):
        validate_published_trusted_provider_calendar_capture_failure_root(
            capture_parent=parent,
            capture_failure=result["capture_failure"],
            capture_failure_file_ref=result["capture_failure_file_ref"],
        )


def test_failure_readback_rejects_path_inode_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    result = _publish_failure_case(parent, "failure-inode-race")
    original_stat = calendar_authority.os.stat

    def drifted_stat(path: Any, *args: Any, **kwargs: Any):
        observed = original_stat(path, *args, **kwargs)
        if path == "capture-failure.json" and kwargs.get("follow_symlinks") is False:
            values = list(observed)
            values[1] += 1
            return os.stat_result(values)
        return observed

    monkeypatch.setattr(calendar_authority.os, "stat", drifted_stat)
    with pytest.raises(SystemSecurityError, match="readback differs"):
        validate_published_trusted_provider_calendar_capture_failure_root(
            capture_parent=parent,
            capture_failure=result["capture_failure"],
            capture_failure_file_ref=result["capture_failure_file_ref"],
        )


def test_failure_root_rejects_success_marker_or_extra_topology(tmp_path: Path) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    result = _publish_failure_case(parent, "failure-mixed-topology")
    root = Path(result["capture_failure_root"])
    extra = root / "capture-success.json"
    extra.write_bytes(b"not-success")
    extra.chmod(0o600)
    with pytest.raises(SystemSecurityError, match="topology differs"):
        validate_published_trusted_provider_calendar_capture_failure_root(
            capture_parent=parent,
            capture_failure=result["capture_failure"],
            capture_failure_file_ref=result["capture_failure_file_ref"],
        )


def test_failure_publication_has_exactly_one_concurrent_winner(tmp_path: Path) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)

    def publish() -> str:
        try:
            _publish_failure_case(parent, "failure-one-winner")
        except SystemPreconditionError:
            return "BLOCKED"
        return "PUBLISHED"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(lambda _: publish(), range(2)))
    assert outcomes == ["BLOCKED", "PUBLISHED"]


def test_failed_capture_retry_requires_fresh_success_root_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path / "captures"
    parent.mkdir(mode=0o700)
    exact_input = _patch_capture_release_closure(tmp_path, monkeypatch)
    patched_release = calendar_authority._release_install_components
    release_calls = 0
    documentation_calls = 0

    def counted_release(*args: Any, **kwargs: Any):
        nonlocal release_calls
        release_calls += 1
        return patched_release(*args, **kwargs)

    def unavailable_documentation():
        nonlocal documentation_calls
        documentation_calls += 1
        raise SystemPreconditionError("docs unavailable")

    monkeypatch.setattr(calendar_authority, "_release_install_components", counted_release)
    monkeypatch.setattr(
        calendar_authority,
        "_official_documentation_fetch",
        unavailable_documentation,
    )

    def capture(root_name: str) -> dict[str, Any]:
        return capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name=root_name,
            cutoff_date=CUTOFF.isoformat(),
            release_install_input_raw=exact_input,
            expected_release_install_input_sha256=hashlib.sha256(exact_input).hexdigest(),
            release_repository_root=tmp_path,
        )

    with pytest.raises(SystemPreconditionError):
        capture("retry-name")
    assert (release_calls, documentation_calls) == (1, 1)
    with pytest.raises(SystemPreconditionError, match="failure destination exists"):
        capture("retry-name")
    assert (release_calls, documentation_calls) == (1, 1)

    def available_documentation():
        nonlocal documentation_calls
        documentation_calls += 1
        return (
            _docs(),
            200,
            {"content-type": "text/html; charset=utf-8"},
            True,
            [],
        )

    monkeypatch.setattr(
        calendar_authority,
        "_official_documentation_fetch",
        available_documentation,
    )

    class FakeClient:
        def __init__(self, **_: Any) -> None:
            pass

        def request(self, *, api_name: str, params: dict[str, str], **_: Any):
            return replay_tushare_response_bytes(
                _provider_raw(params["exchange"]),
                api_name=api_name,
                expected_fields=EXPECTED_FIELDS,
                strict_decimal_decode=True,
            )

    monkeypatch.setattr(calendar_authority, "OfficialTushareHttpsClient", FakeClient)
    assert capture("retry-name-fresh")["status"] == "CAPTURED"
    assert (release_calls, documentation_calls) == (4, 2)
    assert (parent / "retry-name.failure").is_dir()
    assert (parent / "retry-name-fresh/capture-success.json").is_file()


def test_builder_only_provider_artifacts_cannot_enter_production(tmp_path) -> None:
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
    policy = build_calendar_authority_policy(
        created_at="2026-08-14T00:00:00Z",
        pit_exchange_ids=["BSE", "SSE", "SZSE"],
        provider_capability=capability,
    )
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
        pit_exchange_ids=["BSE", "SSE", "SZSE"],
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
    with pytest.raises(SystemContractError, match="route tombstones are not exact"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(
                workspace_root=workspace,
                release_ref=release_ref,
                files=files,
                operation_id="provider-production-bootstrap",
            ),
        )
    assert not (workspace / "results/system/_active.json").exists()
