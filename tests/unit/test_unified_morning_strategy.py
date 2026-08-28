from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pytest
import pandas as pd

import quant_investor
from quant_investor.cli.main import _build_parser
from quant_investor.contracts import canonical_json_bytes
from quant_investor.intelligence import morning

SHANGHAI = ZoneInfo("Asia/Shanghai")
ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts/capture_sina_cn_quotes.py"
SPEC = importlib.util.spec_from_file_location("capture_sina_cn_quotes_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
SINA = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SINA
SPEC.loader.exec_module(SINA)
LAUNCHER_PATH = ROOT / "scripts/operations/run_cn_daily_slot.sh"


def _write(path: Path, value: dict | bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else canonical_json_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def _quote_capture(tmp_path: Path, *, run_date: str = "20260827") -> tuple[str, str]:
    raw_path = tmp_path / f"data/private/cn_public_quotes/{run_date}/sina-0945/raw.gb18030.txt"
    raw = b"provider raw bytes"
    raw_sha = _write(raw_path, raw)
    capture = {
        "schema_version": morning.SINA_CAPTURE_SCHEMA,
        "provider": "SINA",
        "request_time": "2026-08-27T01:45:00Z",
        "response_time": "2026-08-27T01:45:01Z",
        "encoding": "GB18030",
        "raw_ref": {
            "path": raw_path.relative_to(tmp_path).as_posix(),
            "sha256": raw_sha,
            "size": len(raw),
        },
        "field_definitions": {
            "amount": "provider cumulative turnover CNY",
            "price": "provider current price CNY",
            "volume": "provider cumulative shares",
        },
        "symbol_mapping": [{"provider_symbol": "sz002463", "symbol": "002463.SZ"}],
        "quote_rows": [
            {
                "amount": "1000",
                "high": "50.00",
                "low": "48.00",
                "name": "Test",
                "open": "49.00",
                "previous_close": "48.50",
                "price": "49.50",
                "provider_date": "2026-08-27",
                "provider_time": "09:45:00",
                "symbol": "002463.SZ",
                "volume": "100",
            }
        ],
        "reasonable": True,
        "broker": False,
        "order": False,
        "execution": False,
    }
    capture_path = raw_path.parent / "capture.json"
    capture_sha = _write(capture_path, capture)
    return capture_path.relative_to(tmp_path).as_posix(), capture_sha


def _store(tmp_path: Path) -> tuple[dict, str]:
    pointer = {
        "active_closure": {"ledger_path": "record/ledger_after_manual_switch.parquet"},
        "active_record_id": "record",
    }
    path = tmp_path / morning.STORE_POINTER_RELATIVE
    sha = _write(path, pointer)
    return pointer, sha


def _morning_request(tmp_path: Path, *, action: str) -> dict:
    quote_path, quote_sha = _quote_capture(tmp_path)
    _pointer, store_sha = _store(tmp_path)
    return {
        "action": action,
        "automation_id": "automation",
        "run_date": "20260827",
        "previous_trade_date": "20260826",
        "expected_factor_pointer_sha256": "a" * 64,
        "low_observation_path": "results/factors/low.json",
        "low_observation_sha256": "b" * 64,
        "w80_observation_path": "results/factors/w80.json",
        "w80_observation_sha256": "c" * 64,
        "expected_store_pointer_sha256": store_sha,
        "quote_capture_path": quote_path,
        "quote_capture_sha256": quote_sha,
        "pool_manifest_path": None,
        "pool_manifest_sha256": None,
        "output_path": None,
        "output_sha256": None,
    }


def _patch_inputs(monkeypatch, pointer):
    monkeypatch.setattr(
        morning,
        "verify_factor_production",
        lambda _root: {
            "verified": True,
            "factor_authority": "ACTIVE",
            "factor_readiness": "READY",
            "as_of": "20260826",
            "factor_pointer_byte_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(
        morning,
        "_observation",
        lambda *_args, **_kwargs: {"payload": {"state": "OPEN"}},
    )
    monkeypatch.setattr(morning, "load_registered_catalog", lambda _root: (pointer, {}))
    monkeypatch.setattr(
        morning,
        "validate_stable_artifact",
        lambda value, expected_kind=None: value,
    )


def test_morning_preflight_allows_auxiliary_top100_gap(tmp_path: Path, monkeypatch) -> None:
    request = _morning_request(tmp_path, action="PREFLIGHT")
    pointer = json.loads((tmp_path / morning.STORE_POINTER_RELATIVE).read_bytes())
    _patch_inputs(monkeypatch, pointer)

    result = morning.run_morning_strategy(
        workspace_root=tmp_path,
        request=request,
        now=datetime(2026, 8, 27, 9, 45, tzinfo=SHANGHAI),
    )

    assert result["command_status"] == "PREFLIGHT_COMPLETE"
    assert result["status"] == "PARTIAL"
    assert result["core_blockers"] == []
    assert result["auxiliary_blockers"] == ["TOP100_UNAVAILABLE"]
    assert result["factor_status"] == "READY"
    assert result["holdings_status"] == "AVAILABLE"


def test_morning_seal_publishes_exact_no_authority_receipt(tmp_path: Path, monkeypatch) -> None:
    request = _morning_request(tmp_path, action="SEAL")
    pointer = json.loads((tmp_path / morning.STORE_POINTER_RELATIVE).read_bytes())
    _patch_inputs(monkeypatch, pointer)
    output_path = tmp_path / "results/operations/morning_strategy/CN/20260827/0945-strategy.md"
    output_path.parent.mkdir(parents=True, mode=0o700)
    output_path.write_text(
        "research_only=true\nbroker=false\nlive_order=false\n" "actual_holdings_mutation=false\n",
        encoding="utf-8",
    )
    output_path.chmod(0o600)
    request["output_path"] = output_path.relative_to(tmp_path).as_posix()
    request["output_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()

    first = morning.run_morning_strategy(
        workspace_root=tmp_path,
        request=request,
        now=datetime(2026, 8, 27, 9, 45, tzinfo=SHANGHAI),
    )
    second = morning.run_morning_strategy(
        workspace_root=tmp_path,
        request=request,
        now=datetime(2026, 8, 27, 9, 45, tzinfo=SHANGHAI),
    )

    assert first["command_status"] == "PUBLISHED"
    assert second["command_status"] == "NO_ACTION"
    receipt = json.loads((tmp_path / first["receipt_path"]).read_bytes())
    assert receipt["schema_version"] == morning.MORNING_RECEIPT_SCHEMA
    assert receipt["broker"] is False
    assert receipt["live_order"] is False
    assert receipt["live_execution"] is False
    assert receipt["actual_holdings_mutation"] is False


def test_quote_window_rejects_non_0945_capture(tmp_path: Path) -> None:
    quote_path, _sha = _quote_capture(tmp_path)
    path = tmp_path / quote_path
    capture = json.loads(path.read_bytes())
    capture["request_time"] = "2026-08-27T02:00:00Z"
    capture["response_time"] = "2026-08-27T02:00:01Z"
    raw = (path.parent / "raw.gb18030.txt").read_bytes()
    with pytest.raises(morning.IntelligenceError, match="outside"):
        morning.validate_sina_quote_capture(capture, raw=raw, run_date="20260827")


def _sina_raw() -> bytes:
    fields = ["Test", "49", "48.5", "49.5", "50", "48", "49", "49.5", "100", "1000"]
    fields.extend(["0"] * 20)
    fields.extend(["2026-08-27", "09:45:00", "00"])
    return f'var hq_str_sz002463="{",".join(fields)}";\n'.encode("gb18030")


def test_sina_capture_dry_run_and_live_fake(tmp_path: Path) -> None:
    request_path = tmp_path / "request.json"
    request_sha = _write(
        request_path,
        {"run_date": "20260827", "symbols": ["002463.SZ"]},
    )
    output = tmp_path / "data/private/cn_public_quotes/20260827/sina-0945"
    args = argparse.Namespace(
        allow_live=False,
        workspace_root=str(tmp_path),
        request_path=str(request_path),
        request_sha256=request_sha,
        output_root=str(output),
    )
    dry = SINA.run(args)
    assert dry["status"] == "DRY_RUN_VALIDATED"
    assert not output.exists()
    args.allow_live = True
    moments = iter(
        [
            datetime(2026, 8, 27, 1, 45, 0, tzinfo=timezone.utc),
            datetime(2026, 8, 27, 1, 45, 1, tzinfo=timezone.utc),
        ]
    )
    live = SINA.run(args, fetcher=lambda _url: _sina_raw(), now=lambda: next(moments))
    assert live["status"] == "CAPTURED"
    assert live["broker"] is False
    assert (output / "capture.json").is_file()


def test_launcher_reads_only_project_env_without_secret_logging() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert "/usr/bin/security" not in source
    assert "find-generic-password" not in source
    assert "Keychain" not in source
    assert 'env_file="$workspace_root/.env"' in source
    assert "read_project_env_token" in source
    assert "CN_SLOT_LAUNCHER_ENV_UNAVAILABLE" in source
    assert "set +x" in source
    assert 'TUSHARE_URL="https://api.tushare.pro/dataapi"' in source
    assert "hash token" not in source.lower()
    assert "echo $slot_token" not in source
    assert "print -- $slot_token" not in source


def test_cli_registers_morning_and_veto_recovery_commands() -> None:
    parser = _build_parser()
    morning_args = parser.parse_args(
        [
            "research",
            "morning-strategy",
            "--request",
            "request.json",
            "--expected-request-sha256",
            "a" * 64,
        ]
    )
    cutover_args = parser.parse_args(
        [
            "research",
            "morning-cutover",
            "--request",
            "request.json",
            "--expected-request-sha256",
            "a" * 64,
        ]
    )
    evaluate_args = parser.parse_args(
        [
            "research",
            "morning-evaluate",
            "--request",
            "request.json",
            "--expected-request-sha256",
            "a" * 64,
        ]
    )
    recover_args = parser.parse_args(
        [
            "market",
            "recover-transient-write-veto",
            "--run-root",
            "/private/tmp/cn-daily",
            "--expected-veto-sha256",
            "a" * 64,
            "--credential-preflight-receipt",
            "/private/tmp/preflight.json",
            "--expected-credential-preflight-sha256",
            "b" * 64,
        ]
    )
    macro_ready_args = parser.parse_args(
        [
            "market",
            "macro-readiness-seal",
            "--request",
            "request.json",
            "--expected-request-sha256",
            "f" * 64,
        ]
    )
    assert morning_args.research_command == "morning-strategy"
    assert cutover_args.research_command == "morning-cutover"
    assert evaluate_args.research_command == "morning-evaluate"
    assert recover_args.market_command == "recover-transient-write-veto"
    assert macro_ready_args.market_command == "macro-readiness-seal"


def _cutover_request(tmp_path: Path, store_sha: str, calendar_path: Path) -> dict:
    maintenance = {
        "mode": "execute",
        "attempt_slot": "2020",
        "target_date": "20260826",
        "factor_input_readiness": "READY",
        "core_blockers": [],
        "macro_blockers": ["MACRO_WRITE_VETO_ACTIVE"],
        "fundamental_integrity_status": "READY",
    }
    maintenance_path = tmp_path / "data/private/attempt.json"
    maintenance_sha = _write(maintenance_path, maintenance)
    calendar_sha = _write(
        calendar_path,
        {
            "kind": "system.trusted_provider_calendar_capture_success",
            "payload": {"state": "COMPLETE"},
        },
    )
    return {
        "target_date": "20260826",
        "maintenance_receipt_path": maintenance_path.relative_to(tmp_path).as_posix(),
        "maintenance_receipt_sha256": maintenance_sha,
        "calendar_success_path": str(calendar_path),
        "calendar_success_sha256": calendar_sha,
        "factor_rollover_status": "ACTIVATED",
        "expected_factor_pointer_sha256": "a" * 64,
        "low_observation_path": "results/factors/low.json",
        "low_observation_sha256": "b" * 64,
        "w80_observation_path": "results/factors/w80.json",
        "w80_observation_sha256": "c" * 64,
        "expected_store_pointer_sha256": store_sha,
        "expected_import_root": str(Path(quant_investor.__file__).resolve().parent.parent),
        "scheduler_origin_verified": True,
        "current_schedule_state": "EVENING_PRIMARY",
        "morning_receipts": [],
        "auxiliary_blockers": [],
    }


def test_macro_partial_does_not_block_dual_run_cutover(tmp_path: Path, monkeypatch) -> None:
    pointer, store_sha = _store(tmp_path)
    _patch_inputs(monkeypatch, pointer)
    request = _cutover_request(tmp_path, store_sha, tmp_path / "calendar.json")

    result = morning.evaluate_morning_cutover(workspace_root=tmp_path, request=request)

    assert result["morning_strategy_cutover_eligible"] is True
    assert result["core_production_status"] == "COMPLETE"
    assert result["holdings_status"] == "COMPLETE"
    assert result["auxiliary_status"] == "PARTIAL"
    assert result["next_schedule_state"] == "DUAL_RUN"
    assert result["schedule_action"] == "ENABLE_0945_CREATE_2100_FALLBACK_KEEP_2130"


def test_unverified_scheduler_keeps_evening_fallback(tmp_path: Path, monkeypatch) -> None:
    pointer, store_sha = _store(tmp_path)
    _patch_inputs(monkeypatch, pointer)
    request = _cutover_request(tmp_path, store_sha, tmp_path / "calendar.json")
    request["scheduler_origin_verified"] = False

    result = morning.evaluate_morning_cutover(workspace_root=tmp_path, request=request)

    assert result["morning_strategy_cutover_eligible"] is False
    assert result["next_schedule_state"] == "EVENING_PRIMARY"
    assert result["schedule_action"] == "KEEP_FALLBACK"
    assert result["core_blockers"] == ["SCHEDULER_ORIGIN_UNVERIFIED"]


def _successful_morning_receipt(run_date: str, previous_trade_date: str) -> dict:
    return {
        "schema_version": morning.MORNING_RECEIPT_SCHEMA,
        "run_date": run_date,
        "previous_trade_date": previous_trade_date,
        "status": "PARTIAL",
        "core_blockers": [],
        "quote_provider": "SINA",
        "quote_capture_ref": {"path": "quote.json", "sha256": "e" * 64},
        "quote_raw_sha256": "d" * 64,
        "broker": False,
        "live_order": False,
        "live_execution": False,
        "actual_holdings_mutation": False,
    }


def test_morning_eod_separates_operational_success_from_auxiliary_quality(
    tmp_path: Path, monkeypatch
) -> None:
    quote_path, quote_sha = _quote_capture(tmp_path)
    quote = json.loads((tmp_path / quote_path).read_bytes())
    morning_receipt = _successful_morning_receipt("20260827", "20260826")
    morning_receipt["quote_raw_sha256"] = quote["raw_ref"]["sha256"]
    morning_receipt["quote_capture_ref"] = {"path": quote_path, "sha256": quote_sha}
    receipt_path = tmp_path / "results/operations/morning_strategy/CN/20260827/0945-run.v1.json"
    receipt_sha = _write(receipt_path, morning_receipt)
    market_pointer_path = tmp_path / morning.MARKET_POINTER_RELATIVE
    market_pointer_sha = _write(
        market_pointer_path,
        {"snapshot_id": "20260827T080000Z", "status": "OK"},
    )

    class FakeReader:
        def __init__(self, **_kwargs):
            pass

        def clean_snapshot_gate(self, *, refresh=False):
            assert refresh is True
            return {"healthy": True, "latest_complete_trade_date": "20260827"}

        def read_cross_section(self, trade_date, *, columns):
            assert trade_date == "20260827"
            assert "close" in columns
            return pd.DataFrame([{"symbol": "002463.SZ", "trade_date": "20260827", "close": 50}])

    import quant_investor.market.market_data_reader as reader_module

    monkeypatch.setattr(reader_module, "MarketDataReader", FakeReader)
    request = {
        "action": "PREFLIGHT",
        "run_date": "20260827",
        "morning_receipt_path": receipt_path.relative_to(tmp_path).as_posix(),
        "morning_receipt_sha256": receipt_sha,
        "quote_capture_path": quote_path,
        "quote_capture_sha256": quote_sha,
        "expected_market_pointer_sha256": market_pointer_sha,
        "benchmark_symbol": None,
        "output_path": None,
        "output_sha256": None,
    }
    preflight = morning.evaluate_morning_strategy_eod(
        workspace_root=tmp_path,
        request=request,
    )
    assert preflight["command_status"] == "PREFLIGHT_COMPLETE"
    assert preflight["operational_success"] is True
    assert preflight["decision_quality"] == "PARTIAL_AUXILIARY"
    assert preflight["auxiliary_blockers"] == ["BENCHMARK_UNAVAILABLE"]
    assert preflight["instrument_outcomes"][0]["return_0945_to_close"] == ("0.010101010101")

    output_path = tmp_path / "results/operations/morning_strategy/CN/20260827/eod-evaluation.md"
    output_path.write_text(
        "research_only=true\nbroker=false\nlive_order=false\n" "actual_holdings_mutation=false\n",
        encoding="utf-8",
    )
    output_path.chmod(0o600)
    request["action"] = "SEAL"
    request["output_path"] = output_path.relative_to(tmp_path).as_posix()
    request["output_sha256"] = hashlib.sha256(output_path.read_bytes()).hexdigest()
    sealed = morning.evaluate_morning_strategy_eod(
        workspace_root=tmp_path,
        request=request,
    )
    repeated = morning.evaluate_morning_strategy_eod(
        workspace_root=tmp_path,
        request=request,
    )
    assert sealed["command_status"] == "PUBLISHED"
    assert repeated["command_status"] == "NO_ACTION"
    assert sealed["paper_fill"] is False
    assert sealed["actual_holdings_mutation"] is False


def test_morning_eod_rejects_quote_binding_drift(tmp_path: Path) -> None:
    quote_path, quote_sha = _quote_capture(tmp_path)
    receipt = _successful_morning_receipt("20260827", "20260826")
    receipt_path = tmp_path / "results/operations/morning_strategy/CN/20260827/0945-run.v1.json"
    receipt_sha = _write(receipt_path, receipt)
    market_sha = _write(tmp_path / morning.MARKET_POINTER_RELATIVE, {"status": "OK"})
    request = {
        "action": "PREFLIGHT",
        "run_date": "20260827",
        "morning_receipt_path": receipt_path.relative_to(tmp_path).as_posix(),
        "morning_receipt_sha256": receipt_sha,
        "quote_capture_path": quote_path,
        "quote_capture_sha256": quote_sha,
        "expected_market_pointer_sha256": market_sha,
        "benchmark_symbol": None,
        "output_path": None,
        "output_sha256": None,
    }
    with pytest.raises(morning.IntelligenceError, match="binding differs"):
        morning.evaluate_morning_strategy_eod(workspace_root=tmp_path, request=request)


def test_two_successful_mornings_promote_and_pause_dashboard(tmp_path: Path, monkeypatch) -> None:
    pointer, store_sha = _store(tmp_path)
    _patch_inputs(monkeypatch, pointer)
    request = _cutover_request(tmp_path, store_sha, tmp_path / "calendar.json")
    refs = []
    for run_date, previous in (("20260825", "20260824"), ("20260826", "20260825")):
        path = tmp_path / f"results/operations/morning_strategy/CN/{run_date}/0945-run.v1.json"
        digest = _write(path, _successful_morning_receipt(run_date, previous))
        refs.append({"path": path.relative_to(tmp_path).as_posix(), "sha256": digest})
    request["current_schedule_state"] = "DUAL_RUN"
    request["morning_receipts"] = refs

    result = morning.evaluate_morning_cutover(workspace_root=tmp_path, request=request)

    assert result["next_schedule_state"] == "MORNING_PRIMARY"
    assert result["schedule_action"] == "PAUSE_2100_FALLBACK_PAUSE_2130"
    assert result["consecutive_morning_success_count"] == 2


def test_missing_current_morning_restores_evening_fallback(tmp_path: Path, monkeypatch) -> None:
    pointer, store_sha = _store(tmp_path)
    _patch_inputs(monkeypatch, pointer)
    request = _cutover_request(tmp_path, store_sha, tmp_path / "calendar.json")
    request["current_schedule_state"] = "MORNING_PRIMARY"

    result = morning.evaluate_morning_cutover(workspace_root=tmp_path, request=request)

    assert result["next_schedule_state"] == "DUAL_RUN"
    assert result["schedule_action"] == "RESUME_2100_FALLBACK_KEEP_0945_RESUME_2130"
