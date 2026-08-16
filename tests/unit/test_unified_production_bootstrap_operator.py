from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pyarrow as pa
import pytest

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.factors.governance import (
    BLEND_W75_CONTROL,
    BLEND_W80,
    FactorGovernanceError,
    LOW_DOLLAR_VOLUME,
)
from quant_investor.factors.governance.production import (
    assemble_production_bootstrap,
)
import quant_investor.factors.governance.production as production_module
from quant_investor.market.exchange_calendar_official import (
    decode_capture_projection as production_decode_capture_projection,
    decoder_code_sha256,
    decoder_id as production_decoder_id,
)
from quant_investor.system import (
    SystemActivationAuthorizationError,
    SystemContractError,
    SystemPreconditionError,
    SystemStore,
    decode_assembly_request,
    installed_code_manifest_sha256,
)
from unified_activation_helpers import prepare_initial_activation

from tests.unit.test_unified_system_bootstrap import (
    BASE,
    _decision_bytes,
    _write,
    _write_parquet,
)
from quant_investor.factors.governance.source import role_schema
from quant_investor.market.fundamental_incremental import stage_successor_generation
from tests.unit.test_fundamental_incremental_successor import _path_backed_case

SYMBOLS = ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"]


def _test_only_synthetic_daily_bytes(
    exchange: str, rows: list[dict[str, str]]
) -> tuple[bytes, str]:
    if exchange == "SSE":
        body = {
            "result": [
                {"tradeDate": row["date"], "isOpen": row["status"] == "OPEN"} for row in rows
            ]
        }
        return b"sseCalendarCallback(" + canonical_json_bytes(body) + b");", (
            "application/javascript"
        )
    if exchange == "SZSE":
        body = {
            "code": "0",
            "data": [
                {
                    "tradeDate": row["date"].replace("-", ""),
                    "tradeFlag": "1" if row["status"] == "OPEN" else "0",
                }
                for row in rows
            ],
        }
        return canonical_json_bytes(body), "application/json"
    if exchange == "BSE":
        body = {
            "result": [
                {
                    "market_status": "TRADING" if row["status"] == "OPEN" else "CLOSED",
                    "trade_date": row["date"],
                }
                for row in rows
            ],
            "success": True,
        }
        return canonical_json_bytes(body), "application/json"
    raise AssertionError(exchange)


def _test_only_synthetic_session_bytes(exchange: str) -> bytes:
    if exchange == "SSE":
        body = {
            "continuousAuction": [
                {"beginTime": "09:30:00", "endTime": "11:30:00"},
                {"beginTime": "13:00:00", "endTime": "15:00:00"},
            ],
            "market": "SSE",
        }
    elif exchange == "SZSE":
        body = {
            "exchange": "SZSE",
            "sessions": [
                {"end": "11:30:00", "start": "09:30:00", "type": "CONTINUOUS"},
                {"end": "15:00:00", "start": "13:00:00", "type": "CONTINUOUS"},
            ],
        }
    elif exchange == "BSE":
        body = {
            "code": 0,
            "data": {
                "continuousTradingPeriods": ["09:30:00/11:30:00", "13:00:00/15:00:00"],
                "marketCode": "BSE",
            },
        }
    else:
        raise AssertionError(exchange)
    return canonical_json_bytes(body)


def _test_only_decoder_id(exchange: str, role: str) -> str:
    return f"test-only.synthetic-calendar.{exchange.lower()}.{role.lower()}"


def _test_only_decode_projection(
    exchange: str, role: str, raw: bytes, *, media_type: str
) -> dict[str, object]:
    del media_type
    if role == "DAILY_STATUS":
        if exchange == "SSE":
            prefix = b"sseCalendarCallback("
            payload_start = len(prefix)
            value = json.loads(raw[payload_start:-2])
            rows = [
                {
                    "date": row["tradeDate"],
                    "status": "OPEN" if row["isOpen"] else "CLOSED",
                }
                for row in value["result"]
            ]
        elif exchange == "SZSE":
            value = json.loads(raw)
            rows = [
                {
                    "date": (
                        f"{row['tradeDate'][:4]}-{row['tradeDate'][4:6]}-" f"{row['tradeDate'][6:]}"
                    ),
                    "status": "OPEN" if row["tradeFlag"] == "1" else "CLOSED",
                }
                for row in value["data"]
            ]
        else:
            value = json.loads(raw)
            rows = [
                {
                    "date": row["trade_date"],
                    "status": "OPEN" if row["market_status"] == "TRADING" else "CLOSED",
                }
                for row in value["result"]
            ]
        return {"daily_status_rows": rows}
    value = json.loads(raw)
    if exchange == "SSE":
        intervals = [
            {"opens_local": row["beginTime"], "closes_local": row["endTime"]}
            for row in value["continuousAuction"]
        ]
    elif exchange == "SZSE":
        intervals = [
            {"opens_local": row["start"], "closes_local": row["end"]} for row in value["sessions"]
        ]
    else:
        intervals = [
            {"opens_local": row.split("/")[0], "closes_local": row.split("/")[1]}
            for row in value["data"]["continuousTradingPeriods"]
        ]
    return {"session_intervals": intervals}


@pytest.fixture(autouse=True)
def _synthetic_calendar_decoder_is_test_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(production_module, "decoder_id", _test_only_decoder_id)
    monkeypatch.setattr(
        production_module,
        "decode_capture_projection",
        _test_only_decode_projection,
    )


def _raw_market_rows(sessions: list[date]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ordinal, trade_date in enumerate(sessions[-100:]):
        for symbol_index, symbol in enumerate(SYMBOLS):
            rows.append(
                {
                    "ts_code": symbol,
                    "trade_date": trade_date.strftime("%Y%m%d"),
                    "adj_close": 10.0 + symbol_index + ordinal * (0.01 + symbol_index * 0.001),
                    "amount": 1000.0 + symbol_index * 100.0 + ordinal,
                    "vol": 100.0 + symbol_index * 10.0 + ordinal / 7.0,
                    "total_mv": 1_000_000.0 + symbol_index * 100_000.0,
                }
            )
    return rows


RAW_MARKET_SCHEMA = pa.schema(
    [
        pa.field("ts_code", pa.string(), nullable=False),
        pa.field("trade_date", pa.string(), nullable=False),
        pa.field("adj_close", pa.float64(), nullable=True),
        pa.field("amount", pa.float64(), nullable=True),
        pa.field("vol", pa.float64(), nullable=True),
        pa.field("total_mv", pa.float64(), nullable=True),
    ]
)


RAW_PIT_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("industry", pa.string(), nullable=True),
        pa.field("source_list_status", pa.string(), nullable=False),
        pa.field("list_date", pa.string(), nullable=False),
        pa.field("delist_date", pa.string(), nullable=False),
        pa.field("membership_quality", pa.string(), nullable=False),
    ]
)


def _byte_ref(root: Path, relative: str) -> dict[str, str]:
    raw = (root / relative).read_bytes()
    return {
        "relative_path": relative,
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _seed_workspace(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    workspace = tmp_path / "workspace"
    seed_source = tmp_path / "seed-source"
    workspace.mkdir(mode=0o700)
    seed_source.mkdir(mode=0o700)
    store = SystemStore(
        workspace,
        source_root=seed_source,
        source_root_id="production-bootstrap-seed-source",
    )
    release = seal_artifact(
        "system.release",
        {
            "release_id": "production-bootstrap-release",
            "state": "OPERATIONAL",
            "code_sha256": "3" * 64,
            "wheel_sha256": "4" * 64,
            "code_manifest_sha256": installed_code_manifest_sha256(),
        },
        created_at=BASE,
    )
    return workspace, store.put_object(release)


def _inputs(root: Path) -> dict[str, dict[str, str] | list[dict[str, str]]]:
    root.mkdir(mode=0o700)
    first = date(2024, 1, 2)
    cutoff = date(2026, 8, 7)
    calendar_rows: list[dict[str, Any]] = []
    cursor = first
    while cursor <= cutoff:
        if cursor.weekday() < 5:
            opens = datetime.combine(cursor, datetime.min.time(), tzinfo=timezone.utc)
            calendar_rows.append(
                {
                    "ordinal": len(calendar_rows),
                    "open_session": cursor,
                    "opens_at_utc": opens + timedelta(hours=1, minutes=30),
                    "closes_at_utc": opens + timedelta(hours=7),
                }
            )
        cursor += timedelta(days=1)
    _write_parquet(
        root,
        "strict/exchange_calendar.parquet",
        calendar_rows,
        role_schema("exchange_calendar"),
    )
    (root / "closure").mkdir(mode=0o700, exist_ok=True)
    _write(
        root,
        "closure/market_scope.json",
        canonical_json_bytes(
            {
                "full_a": SYMBOLS,
                "full_market": SYMBOLS,
                "all_a": SYMBOLS,
                "all": SYMBOLS,
                "stats": {"full_a": len(SYMBOLS)},
            }
        ),
    )
    market_rows = _raw_market_rows([row["open_session"] for row in calendar_rows])
    _write_parquet(
        root,
        "closure/market-2026.parquet",
        market_rows,
        RAW_MARKET_SCHEMA,
    )
    _write_parquet(
        root,
        "closure/pit_membership.parquet",
        [
            {
                "symbol": symbol,
                "industry": "industry-a" if index < 2 else "industry-b",
                "source_list_status": "L",
                "list_date": "19910101",
                "delist_date": "",
                "membership_quality": "ok",
            }
            for index, symbol in enumerate(SYMBOLS)
        ],
        RAW_PIT_SCHEMA,
    )
    scope_sha = hashlib.sha256("\n".join(SYMBOLS).encode("utf-8")).hexdigest()
    pit_sha = _byte_ref(root, "closure/pit_membership.parquet")["byte_sha256"]
    pit_manifest_body = {
        "generation_id": "pit-test",
        "canonical_sha256": pit_sha,
        "row_count": len(SYMBOLS),
        "status_counts": {"L": len(SYMBOLS)},
        "membership_quality_counts": {"ok": len(SYMBOLS)},
    }
    _write(
        root,
        "closure/pit_generation_manifest.json",
        canonical_json_bytes(pit_manifest_body),
    )
    _write(
        root,
        "closure/pit_pointer.json",
        canonical_json_bytes(
            {
                "generation_id": "pit-test",
                "membership_path": "closure/pit_membership.parquet",
            }
        ),
    )
    pit_manifest_sha = _byte_ref(root, "closure/pit_generation_manifest.json")["byte_sha256"]
    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "complete": True,
        "coverage_ratio": 1.0,
        "coverage_complete_count": len(SYMBOLS),
        "expected_scope_count": len(SYMBOLS),
        "observed_bar_count": len(SYMBOLS),
        "blocking_incomplete_count": 0,
        "categories_checked": ["full_a"],
        "latest_complete_trade_date": cutoff.strftime("%Y%m%d"),
        "coverage_trade_date": cutoff.strftime("%Y%m%d"),
        "upsert_target_trade_date": cutoff.strftime("%Y%m%d"),
        "expected_scope_sha256": scope_sha,
        "suspended_symbols": [],
        "inactive_symbols": [],
        "non_blocking_absent_symbols": [],
        "true_missing_symbols": [],
        "classification_sets_disjoint": True,
        "pit_membership_sha256": pit_sha,
        "pit_generation_id": "pit-test",
        "pit_generation_manifest_sha256": pit_manifest_sha,
    }
    market_pointer = {
        "snapshot_id": "snapshot-test",
        "status": "OK",
        "manifest_path": "data/parquet/cn/_snapshots/snapshot-test.json",
        "latest_complete_trade_date": cutoff.strftime("%Y%m%d"),
        "latest_trade_date": cutoff.strftime("%Y%m%d"),
        "coverage": coverage,
        "blockers": [],
    }
    _write(
        root,
        "closure/market_pointer.json",
        canonical_json_bytes(market_pointer),
    )
    _write(
        root,
        "closure/market_snapshot_manifest.json",
        canonical_json_bytes(
            {
                "snapshot_id": "snapshot-test",
                "status": "OK",
                "latest_complete_trade_date": cutoff.strftime("%Y%m%d"),
                "latest_trade_date": cutoff.strftime("%Y%m%d"),
                "coverage": coverage,
                "blockers": [],
            }
        ),
    )
    fundamental_fixture = root.parent / f"{root.name}-fundamental-fixture"
    fundamental_fixture.mkdir(mode=0o700)
    bundle, _, support_files, provider = _path_backed_case(fundamental_fixture)
    target_bindings = {
        "market_pointer": {
            "path": str((root / "closure/market_pointer.json").resolve()),
            "sha256": _byte_ref(root, "closure/market_pointer.json")["byte_sha256"],
            "as_of": cutoff.strftime("%Y%m%d"),
            "immutable_refs": [
                {
                    "path": str((root / "closure/market_snapshot_manifest.json").resolve()),
                    "sha256": _byte_ref(root, "closure/market_snapshot_manifest.json")[
                        "byte_sha256"
                    ],
                }
            ],
        },
        "pit_pointer": {
            "path": str((root / "closure/pit_pointer.json").resolve()),
            "sha256": _byte_ref(root, "closure/pit_pointer.json")["byte_sha256"],
            "as_of": cutoff.strftime("%Y%m%d"),
            "immutable_refs": [
                {
                    "path": str((root / "closure/pit_generation_manifest.json").resolve()),
                    "sha256": _byte_ref(root, "closure/pit_generation_manifest.json")[
                        "byte_sha256"
                    ],
                }
            ],
        },
        "pit_membership": {
            "path": str((root / "closure/pit_membership.parquet").resolve()),
            "sha256": _byte_ref(root, "closure/pit_membership.parquet")["byte_sha256"],
            "as_of": cutoff.strftime("%Y%m%d"),
        },
        "expected_scope": {
            "path": str((root / "closure/market_scope.json").resolve()),
            "sha256": _byte_ref(root, "closure/market_scope.json")["byte_sha256"],
            "as_of": cutoff.strftime("%Y%m%d"),
        },
    }
    capture = stage_successor_generation(
        bundle,
        staging_root=root.parent / f"{root.name}-fundamental-capture",
        generation_id="staged_successor",
        provider_manifest=provider,
        target_bindings=target_bindings,
        provider_evidence_files=support_files,
    )
    for source in sorted(capture.staging_root.rglob("*")):
        if source.is_file():
            _write(root, source.relative_to(capture.staging_root).as_posix(), source.read_bytes())
    _write(root, "decision/bootstrap-decision.json", _decision_bytes())
    calendar_file_ref = _byte_ref(root, "strict/exchange_calendar.parquet")
    session_set = {row["open_session"].isoformat() for row in calendar_rows}
    daily_status_rows: list[dict[str, str]] = []
    status_cursor = date(2024, 1, 1)
    while status_cursor <= cutoff:
        status_date = status_cursor.isoformat()
        daily_status_rows.append(
            {
                "date": status_date,
                "status": "OPEN" if status_date in session_set else "CLOSED",
            }
        )
        status_cursor += timedelta(days=1)
    session_sha = hashlib.sha256(
        canonical_json_bytes([row["open_session"].isoformat() for row in calendar_rows])
    ).hexdigest()
    intervals = [
        {"opens_local": "09:30:00", "closes_local": "11:30:00"},
        {"opens_local": "13:00:00", "closes_local": "15:00:00"},
    ]
    code_sha = decoder_code_sha256()
    raw_by_exchange_role: dict[tuple[str, str], dict[str, str]] = {}
    capture_by_exchange_role: dict[tuple[str, str], dict[str, str]] = {}
    source_by_exchange_role: dict[tuple[str, str], str] = {}
    for exchange, issuer, host in (
        ("SSE", "SSE_OFFICIAL", "www.sse.com.cn"),
        ("SZSE", "SZSE_OFFICIAL", "www.szse.cn"),
    ):
        for role in ("DAILY_STATUS", "SESSION_RULE"):
            role_name = "daily" if role == "DAILY_STATUS" else "rules"
            if role == "DAILY_STATUS":
                raw_body, media_type = _test_only_synthetic_daily_bytes(exchange, daily_status_rows)
            else:
                raw_body, media_type = (
                    _test_only_synthetic_session_bytes(exchange),
                    "application/json",
                )
            raw_relative = f"closure/calendar-{role_name}-raw-{exchange.lower()}.bin"
            _write(root, raw_relative, raw_body)
            raw_ref = _byte_ref(root, raw_relative)
            source_url = f"https://{host}/official/calendar/{role_name}"
            projection = dict(
                _test_only_decode_projection(exchange, role, raw_body, media_type=media_type)
            )
            capture = seal_artifact(
                "system.exchange_calendar_capture",
                {
                    "calendar_capture_id": (
                        f"official-calendar-{role_name}-capture-{exchange.lower()}"
                    ),
                    "state": "IMMUTABLE",
                    "evidence_role": role,
                    "exchange_id": exchange,
                    "issuer": issuer,
                    "source_url": source_url,
                    "request_url": source_url,
                    "effective_url": source_url,
                    "redirect_chain": [],
                    "http_status": 200,
                    "issuer_host": host,
                    "tls_verified": True,
                    "captured_at": BASE,
                    "raw_file_ref": raw_ref,
                    "raw_sha256": raw_ref["byte_sha256"],
                    "raw_byte_length": len(raw_body),
                    "raw_media_type": media_type,
                    "decoder_id": _test_only_decoder_id(exchange, role),
                    "decoder_sha256": code_sha,
                    "timezone": "Asia/Shanghai",
                    "session_intervals": projection.get("session_intervals", []),
                    "coverage_start_date": "2024-01-01",
                    "cutoff_date": cutoff.isoformat(),
                    "daily_status_rows": projection.get("daily_status_rows", []),
                    "projection_sha256": hashlib.sha256(
                        canonical_json_bytes(projection)
                    ).hexdigest(),
                    "transform_code_sha256": code_sha,
                },
                created_at=BASE,
            )
            capture_relative = f"closure/calendar-{role_name}-capture-{exchange.lower()}.json"
            _write(root, capture_relative, canonical_json_bytes(capture))
            key = (exchange, role)
            raw_by_exchange_role[key] = raw_ref
            capture_by_exchange_role[key] = _byte_ref(root, capture_relative)
            source_by_exchange_role[key] = source_url
    raw_calendar_refs = sorted(
        raw_by_exchange_role.values(), key=lambda value: value["relative_path"]
    )
    capture_refs = sorted(
        capture_by_exchange_role.values(), key=lambda value: value["relative_path"]
    )
    calendar_manifest = seal_artifact(
        "system.exchange_calendar_manifest",
        {
            "calendar_manifest_id": "official-calendar-test",
            "state": "IMMUTABLE",
            "coverage_start_date": "2024-01-01",
            "cutoff_date": cutoff.isoformat(),
            "timezone": "Asia/Shanghai",
            "calendar_file_ref": calendar_file_ref,
            "transform_code_sha256": code_sha,
            "exchange_rows": [
                {
                    "exchange_id": exchange,
                    "issuer": issuer,
                    "daily_status_source_url": source_by_exchange_role[(exchange, "DAILY_STATUS")],
                    "daily_status_captured_at": BASE,
                    "daily_status_raw_file_ref": raw_by_exchange_role[(exchange, "DAILY_STATUS")],
                    "daily_status_capture_file_ref": capture_by_exchange_role[
                        (exchange, "DAILY_STATUS")
                    ],
                    "session_rule_source_url": source_by_exchange_role[(exchange, "SESSION_RULE")],
                    "session_rule_captured_at": BASE,
                    "session_rule_raw_file_ref": raw_by_exchange_role[(exchange, "SESSION_RULE")],
                    "session_rule_capture_file_ref": capture_by_exchange_role[
                        (exchange, "SESSION_RULE")
                    ],
                    "open_session_count": len(calendar_rows),
                    "open_session_sha256": session_sha,
                    "session_intervals": intervals,
                }
                for exchange, issuer in (
                    ("SSE", "SSE_OFFICIAL"),
                    ("SZSE", "SZSE_OFFICIAL"),
                )
            ],
        },
        created_at=BASE,
    )
    _write(
        root,
        "closure/calendar_manifest.json",
        canonical_json_bytes(calendar_manifest),
    )
    scalars = {
        "exchange_calendar_file_ref": "strict/exchange_calendar.parquet",
        "market_scope_file_ref": "closure/market_scope.json",
        "market_pointer_file_ref": "closure/market_pointer.json",
        "market_snapshot_manifest_file_ref": "closure/market_snapshot_manifest.json",
        "pit_pointer_file_ref": "closure/pit_pointer.json",
        "pit_generation_manifest_file_ref": "closure/pit_generation_manifest.json",
        "pit_membership_file_ref": "closure/pit_membership.parquet",
        "calendar_manifest_file_ref": "closure/calendar_manifest.json",
        "fundamental_pointer_file_ref": "_fundamental_latest.json",
        "fundamental_generation_manifest_file_ref": (
            "_fundamental_generations/staged_successor/manifest.json"
        ),
        "bootstrap_decision_file_ref": "decision/bootstrap-decision.json",
    }
    result: dict[str, dict[str, str] | list[dict[str, str]]] = {
        field: _byte_ref(root, relative) for field, relative in scalars.items()
    }
    result["calendar_raw_file_refs"] = raw_calendar_refs
    result["calendar_capture_file_refs"] = capture_refs
    result["market_table_file_refs"] = [_byte_ref(root, "closure/market-2026.parquet")]
    result["fundamental_table_file_refs"] = [
        _byte_ref(root, f"_fundamental_generations/staged_successor/{name}.parquet")
        for name in (
            "fundamental_daily",
            "fundamental_period",
            "fundamental_quarantine",
        )
    ]
    evidence_root = root / "_fundamental_generations/staged_successor/provider_evidence"
    result["fundamental_evidence_file_refs"] = [
        _byte_ref(root, path.relative_to(root).as_posix())
        for path in sorted(evidence_root.rglob("*"))
        if path.is_file()
    ]
    return result


def _replace_synthetic_calendar_capture(
    root: Path,
    files: dict[str, dict[str, str] | list[dict[str, str]]],
    *,
    exchange: str,
    role: str,
    raw_body: bytes,
) -> None:
    role_name = "daily" if role == "DAILY_STATUS" else "rules"
    raw_relative = f"closure/calendar-{role_name}-raw-{exchange.lower()}.bin"
    capture_relative = f"closure/calendar-{role_name}-capture-{exchange.lower()}.json"
    _write(root, raw_relative, raw_body)
    raw_ref = _byte_ref(root, raw_relative)
    capture_path = root / capture_relative
    capture = json.loads(capture_path.read_bytes())
    payload = capture["payload"]
    projection = dict(
        _test_only_decode_projection(
            exchange,
            role,
            raw_body,
            media_type=payload["raw_media_type"],
        )
    )
    payload["raw_file_ref"] = raw_ref
    payload["raw_sha256"] = raw_ref["byte_sha256"]
    payload["raw_byte_length"] = len(raw_body)
    payload["daily_status_rows"] = projection.get("daily_status_rows", [])
    payload["session_intervals"] = projection.get("session_intervals", [])
    payload["projection_sha256"] = hashlib.sha256(canonical_json_bytes(projection)).hexdigest()
    _write(
        root,
        capture_relative,
        canonical_json_bytes(
            seal_artifact("system.exchange_calendar_capture", payload, created_at=BASE)
        ),
    )
    capture_ref = _byte_ref(root, capture_relative)
    raw_refs = [
        raw_ref if row["relative_path"] == raw_relative else row
        for row in files["calendar_raw_file_refs"]
    ]
    capture_refs = [
        capture_ref if row["relative_path"] == capture_relative else row
        for row in files["calendar_capture_file_refs"]
    ]
    files["calendar_raw_file_refs"] = sorted(raw_refs, key=lambda value: value["relative_path"])
    files["calendar_capture_file_refs"] = sorted(
        capture_refs, key=lambda value: value["relative_path"]
    )
    manifest_path = root / "closure/calendar_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    row = next(
        item for item in manifest["payload"]["exchange_rows"] if item["exchange_id"] == exchange
    )
    prefix = "daily_status" if role == "DAILY_STATUS" else "session_rule"
    row[f"{prefix}_raw_file_ref"] = raw_ref
    row[f"{prefix}_capture_file_ref"] = capture_ref
    _write(
        root,
        "closure/calendar_manifest.json",
        canonical_json_bytes(
            seal_artifact(
                "system.exchange_calendar_manifest",
                manifest["payload"],
                created_at=BASE,
            )
        ),
    )
    files["calendar_manifest_file_ref"] = _byte_ref(root, "closure/calendar_manifest.json")


def _request(
    *,
    release_ref: dict[str, str],
    files: dict[str, dict[str, str] | list[dict[str, str]]],
    operation_id: str = "production-bootstrap-test",
) -> bytes:
    payload: dict[str, Any] = {
        "bootstrap_operation_id": operation_id,
        "state": "SEALED",
        "source_root_id": "production-bootstrap-source-test",
        "release_manifest_ref": release_ref,
        "skill_tree_sha256": "1" * 64,
        "automation_semantic_sha256": "2" * 64,
        "source_blockers": ["FUNDAMENTAL_SOURCE_STALE"],
        "trusted_at": BASE,
        **files,
    }
    return canonical_json_bytes(
        seal_artifact("system.bootstrap_operator_request", payload, created_at=BASE)
    )


def test_production_bootstrap_materializes_and_offline_verifies(tmp_path: Path) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    raw = _request(release_ref=release_ref, files=_inputs(input_root))

    result = assemble_production_bootstrap(
        workspace_root=workspace,
        input_root=input_root,
        request_raw=raw,
    )

    assert result["status"] == "OFFLINE_VERIFIED"
    assert result["generation"]["generation_state"] == "OPERATIONAL"
    readiness = result["generation"]["readiness"]["payload"]
    assert readiness["factor_state"] == "READY"
    assert readiness["mainline_state"] == "UNINITIALIZED"
    assert readiness["investment_state"] == "BLOCKED"
    active = result["generation"]["factor_active_set"]["payload"]
    assert active["producer_identity"] == "NOT_CLAIMED"
    assert active["admission_route"] == "BOOTSTRAP_EXCEPTION"
    assert {row["factor_id"]: row["weight"] for row in active["factor_rows"]} == {
        LOW_DOLLAR_VOLUME: "0.500000000000",
        BLEND_W80: "0.500000000000",
    }
    assert len(active["control_rows"]) == 1
    control = active["control_rows"][0]
    assert control["factor_id"] == BLEND_W75_CONTROL
    assert control["role"] == "CONTROL_ONLY"
    assert control["selectable"] is False
    assert control["weight"] == "0.000000000000"
    assert all(row["finite_count"] > 0 for row in result["signal_statistics"])
    assert all(row["distinct_finite_count"] > 1 for row in result["signal_statistics"])
    assert result["active_pointer_write_count"] == 0
    assert result["marker_write_count"] == 0
    assert result["generation"]["manifest"]["payload"]["research_refs"] == [
        result["production_bootstrap_receipt_ref"]
    ]
    assert not (workspace / "results/system/_active.json").exists()
    assert not (workspace / "results/system/_migration_complete.json").exists()
    for path in Path(result["source_root"]).rglob("*"):
        if path.is_file():
            assert path.stat().st_mode & 0o777 == 0o600
            assert path.stat().st_nlink == 1


def test_production_bootstrap_receipt_allows_exact_isolated_initial_activation(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    result = assemble_production_bootstrap(
        workspace_root=workspace,
        input_root=input_root,
        request_raw=_request(release_ref=release_ref, files=_inputs(input_root)),
    )
    store = SystemStore(
        workspace,
        source_root=result["source_root"],
        source_root_id="production-bootstrap-source-test",
    )
    prepared = prepare_initial_activation(
        store,
        result["generation"],
        release_ref,
    )

    activated = store.activate_initial_generation(**prepared)

    assert activated["activation"]["cas_performed"] is True
    assert activated["generation_state"] == "OPERATIONAL"
    assert activated["factor_status"]["payload"]["readiness"] == "READY"
    assert activated["migration_completion"]["marker"]["payload"]["status"] == "COMPLETE"


def test_production_bootstrap_receipt_binding_mutations_block_initial_activation(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    result = assemble_production_bootstrap(
        workspace_root=workspace,
        input_root=input_root,
        request_raw=_request(release_ref=release_ref, files=_inputs(input_root)),
    )
    store = SystemStore(
        workspace,
        source_root=result["source_root"],
        source_root_id="production-bootstrap-source-test",
    )
    receipt = store.get_object(result["production_bootstrap_receipt_ref"])
    assembly_request = store.get_object(result["assembly_request_ref"])
    base_kwargs = decode_assembly_request(assembly_request)

    def reseal(mutator: Any) -> dict[str, str]:
        payload = dict(receipt["payload"])
        mutator(payload)
        body = {
            key: payload[key] for key in sorted(payload) if key != "production_bootstrap_receipt_id"
        }
        payload["production_bootstrap_receipt_id"] = (
            "production-bootstrap-" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()
        )
        return store.put_object(
            seal_artifact(
                "system.production_bootstrap_receipt",
                payload,
                created_at=receipt["created_at"],
            )
        )

    def mutate_statistics(payload: dict[str, Any]) -> None:
        rows = [dict(row) for row in payload["signal_statistics"]]
        rows[0]["finite_count"] += 1
        payload["signal_statistics"] = rows
        payload["signal_statistics_sha256"] = hashlib.sha256(canonical_json_bytes(rows)).hexdigest()

    mutations = [
        lambda payload: payload.__setitem__(
            "deployed_release_ref", result["production_bootstrap_receipt_ref"]
        ),
        lambda payload: payload.__setitem__(
            "source_refs", [result["production_bootstrap_receipt_ref"]]
        ),
        lambda payload: payload.__setitem__(
            "factor_policy_ref", result["production_bootstrap_receipt_ref"]
        ),
        lambda payload: payload.__setitem__(
            "readiness_matrix_ref", result["production_bootstrap_receipt_ref"]
        ),
        lambda payload: payload.__setitem__("emergency_controller_sha256", "f" * 64),
        lambda payload: payload.__setitem__("assembler_code_sha256", "e" * 64),
        mutate_statistics,
    ]
    for mutate in mutations:
        mutated_ref = reseal(mutate)
        generation = store.assemble_generation(**{**base_kwargs, "research_refs": [mutated_ref]})
        prepared = prepare_initial_activation(store, generation, release_ref)
        with pytest.raises(
            SystemActivationAuthorizationError,
            match="lacks valid production bootstrap closure",
        ):
            store.activate_initial_generation(**prepared)
        assert not (workspace / "results/system/_active.json").exists()
        assert not (workspace / "results/system/_migration_complete.json").exists()

    wrong_kind_generation = store.assemble_generation(
        **{**base_kwargs, "research_refs": [release_ref]}
    )
    wrong_kind_prepared = prepare_initial_activation(store, wrong_kind_generation, release_ref)
    with pytest.raises(
        SystemActivationAuthorizationError,
        match="lacks valid production bootstrap closure",
    ):
        store.activate_initial_generation(**wrong_kind_prepared)
    with pytest.raises(SystemContractError, match="duplicate|sorted and unique"):
        store.assemble_generation(
            **{
                **base_kwargs,
                "research_refs": [
                    result["production_bootstrap_receipt_ref"],
                    result["production_bootstrap_receipt_ref"],
                ],
            }
        )


def test_production_source_drift_after_receipt_blocks_before_initial_cas(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    result = assemble_production_bootstrap(
        workspace_root=workspace,
        input_root=input_root,
        request_raw=_request(release_ref=release_ref, files=_inputs(input_root)),
    )
    store = SystemStore(
        workspace,
        source_root=result["source_root"],
        source_root_id="production-bootstrap-source-test",
    )
    receipt = store.get_object(result["production_bootstrap_receipt_ref"])
    raw_market_row = next(
        row
        for row in receipt["payload"]["input_source_rows"]
        if row["field"] == "market_table_file_refs"
    )
    source = store.get_object(raw_market_row["source_object_ref"])
    source_path = Path(result["source_root"]) / source["payload"]["relative_path"]
    source_path.write_bytes(source_path.read_bytes() + b"tamper")
    source_path.chmod(0o600)
    prepared = prepare_initial_activation(
        store,
        result["generation"],
        release_ref,
    )

    with pytest.raises(SystemContractError, match="byte hash|byte binding|source"):
        store.activate_initial_generation(**prepared)

    assert not (workspace / "results/system/_active.json").exists()
    assert not (workspace / "results/system/_migration_complete.json").exists()


def test_production_bootstrap_rejects_input_drift(tmp_path: Path) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    raw = _request(release_ref=release_ref, files=files)
    _write(root=input_root, relative="closure/market_pointer.json", raw=b'{"drift":true}')

    with pytest.raises(Exception, match="exact hash"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=raw,
        )


def test_production_bootstrap_rejects_non_strict_market(tmp_path: Path) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    bad_path = input_root / "closure/market-2026.parquet"
    table = pq.read_table(bad_path).drop(["total_mv"])
    pq.write_table(table, bad_path)
    bad_path.chmod(0o600)
    files["market_table_file_refs"] = [_byte_ref(input_root, "closure/market-2026.parquet")]
    raw = _request(release_ref=release_ref, files=files)

    with pytest.raises((SystemContractError, ValueError), match="schema|column"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=raw,
        )


def test_production_bootstrap_rejects_constant_active_signals(tmp_path: Path) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    raw = pq.read_table(input_root / "closure/market-2026.parquet")
    constant_rows = [
        {**row, "adj_close": 10.0, "amount": 1_000.0, "vol": 100.0} for row in raw.to_pylist()
    ]
    _write_parquet(
        input_root,
        "closure/market-2026.parquet",
        constant_rows,
        RAW_MARKET_SCHEMA,
    )
    files["market_table_file_refs"] = [_byte_ref(input_root, "closure/market-2026.parquet")]
    raw = _request(release_ref=release_ref, files=files)

    with pytest.raises(FactorGovernanceError, match="constant"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=raw,
        )


def test_production_bootstrap_rejects_unofficial_calendar_authority(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    manifest_path = input_root / "closure/calendar_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    payload = manifest["payload"]
    payload["exchange_rows"][1]["daily_status_source_url"] = "https://example.com/calendar"
    tampered = seal_artifact(
        "system.exchange_calendar_manifest",
        payload,
        created_at=BASE,
    )
    _write(
        input_root,
        "closure/calendar_manifest.json",
        canonical_json_bytes(tampered),
    )
    files["calendar_manifest_file_ref"] = _byte_ref(input_root, "closure/calendar_manifest.json")
    raw = _request(release_ref=release_ref, files=files)

    with pytest.raises(SystemContractError, match="source authority"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=raw,
        )


def test_production_bootstrap_requires_explicit_daily_open_closed_evidence(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    capture_path = input_root / "closure/calendar-daily-capture-sse.json"
    capture = json.loads(capture_path.read_bytes())
    capture_payload = capture["payload"]
    capture_payload["daily_status_rows"].pop(10)
    _write(
        input_root,
        "closure/calendar-daily-capture-sse.json",
        canonical_json_bytes(
            seal_artifact(
                "system.exchange_calendar_capture",
                capture_payload,
                created_at=BASE,
            )
        ),
    )
    new_capture_ref = _byte_ref(input_root, "closure/calendar-daily-capture-sse.json")
    capture_refs = list(files["calendar_capture_file_refs"])
    capture_refs = [
        (
            new_capture_ref
            if row["relative_path"] == "closure/calendar-daily-capture-sse.json"
            else row
        )
        for row in capture_refs
    ]
    files["calendar_capture_file_refs"] = sorted(
        capture_refs, key=lambda value: value["relative_path"]
    )
    manifest_path = input_root / "closure/calendar_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest_payload = manifest["payload"]
    manifest_payload["exchange_rows"][0]["daily_status_capture_file_ref"] = new_capture_ref
    _write(
        input_root,
        "closure/calendar_manifest.json",
        canonical_json_bytes(
            seal_artifact(
                "system.exchange_calendar_manifest",
                manifest_payload,
                created_at=BASE,
            )
        ),
    )
    files["calendar_manifest_file_ref"] = _byte_ref(input_root, "closure/calendar_manifest.json")

    with pytest.raises(SystemContractError, match="projection binding"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(release_ref=release_ref, files=files),
        )
    assert not (workspace / "results/system/objects/system.production_bootstrap_receipt").exists()
    assert not (workspace / "results/system/generations").exists()
    assert not (workspace / "results/system/_active.json").exists()


def test_production_bootstrap_rejects_unadmitted_synthetic_calendar_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    monkeypatch.setattr(production_module, "decoder_id", production_decoder_id)
    monkeypatch.setattr(
        production_module,
        "decode_capture_projection",
        production_decode_capture_projection,
    )
    with pytest.raises(
        SystemContractError,
        match="OFFICIAL_CALENDAR_WIRE_CONTRACT_UNVERIFIED",
    ):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(release_ref=release_ref, files=files),
        )
    assert not (workspace / "results/system/objects/system.production_bootstrap_receipt").exists()
    assert not (workspace / "results/system/generations").exists()
    assert not (workspace / "results/system/_active.json").exists()


def test_production_bootstrap_rejects_synthetic_daily_date_gap(tmp_path: Path) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    raw_path = input_root / "closure/calendar-daily-raw-szse.bin"
    native = json.loads(raw_path.read_bytes())
    native["data"].pop(10)
    _replace_synthetic_calendar_capture(
        input_root,
        files,
        exchange="SZSE",
        role="DAILY_STATUS",
        raw_body=canonical_json_bytes(native),
    )

    with pytest.raises(SystemContractError, match="coverage is not consecutive"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(release_ref=release_ref, files=files),
        )


def test_production_bootstrap_rejects_cross_exchange_calendar_disagreement(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    raw_path = input_root / "closure/calendar-daily-raw-szse.bin"
    native = json.loads(raw_path.read_bytes())
    open_row = next(row for row in native["data"] if row["tradeFlag"] == "1")
    open_row["tradeFlag"] = "0"
    _replace_synthetic_calendar_capture(
        input_root,
        files,
        exchange="SZSE",
        role="DAILY_STATUS",
        raw_body=canonical_json_bytes(native),
    )

    with pytest.raises(SystemContractError, match="OPEN/CLOSED calendar differs"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(release_ref=release_ref, files=files),
        )


def test_production_bootstrap_rejects_unsupported_official_session_hours(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    raw_path = input_root / "closure/calendar-rules-raw-sse.bin"
    native = json.loads(raw_path.read_bytes())
    native["continuousAuction"][0]["beginTime"] = "09:31:00"
    _replace_synthetic_calendar_capture(
        input_root,
        files,
        exchange="SSE",
        role="SESSION_RULE",
        raw_body=canonical_json_bytes(native),
    )

    with pytest.raises(SystemContractError, match="session-rule projection binding"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(release_ref=release_ref, files=files),
        )


def test_production_bootstrap_rejects_decoder_code_identity_drift(tmp_path: Path) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    capture_relative = "closure/calendar-daily-capture-sse.json"
    capture_path = input_root / capture_relative
    capture = json.loads(capture_path.read_bytes())
    capture["payload"]["decoder_sha256"] = "0" * 64
    _write(
        input_root,
        capture_relative,
        canonical_json_bytes(
            seal_artifact(
                "system.exchange_calendar_capture",
                capture["payload"],
                created_at=BASE,
            )
        ),
    )
    capture_ref = _byte_ref(input_root, capture_relative)
    files["calendar_capture_file_refs"] = sorted(
        [
            capture_ref if row["relative_path"] == capture_relative else row
            for row in files["calendar_capture_file_refs"]
        ],
        key=lambda value: value["relative_path"],
    )
    manifest_path = input_root / "closure/calendar_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["payload"]["exchange_rows"][0]["daily_status_capture_file_ref"] = capture_ref
    _write(
        input_root,
        "closure/calendar_manifest.json",
        canonical_json_bytes(
            seal_artifact(
                "system.exchange_calendar_manifest",
                manifest["payload"],
                created_at=BASE,
            )
        ),
    )
    files["calendar_manifest_file_ref"] = _byte_ref(input_root, "closure/calendar_manifest.json")

    with pytest.raises(SystemContractError, match="decoder identity differs"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(release_ref=release_ref, files=files),
        )


def test_production_bootstrap_rejects_fundamental_table_manifest_drift(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    relative = "_fundamental_generations/staged_successor/fundamental_daily.parquet"
    table_path = input_root / relative
    _write(input_root, relative, table_path.read_bytes() + b"x")
    table_refs = list(files["fundamental_table_file_refs"])
    table_refs[0] = _byte_ref(input_root, relative)
    files["fundamental_table_file_refs"] = table_refs

    with pytest.raises(SystemContractError, match="Fundamental table byte binding"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(release_ref=release_ref, files=files),
        )


def test_production_bootstrap_rejects_fundamental_wrong_parquet_schema(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    relative = "_fundamental_generations/staged_successor/fundamental_daily.parquet"
    _write_parquet(
        input_root,
        relative,
        [{"wrong_column": "000001.SZ"}],
        pa.schema([pa.field("wrong_column", pa.string(), nullable=False)]),
    )
    daily_ref = _byte_ref(input_root, relative)
    table_refs = list(files["fundamental_table_file_refs"])
    table_refs[0] = daily_ref
    files["fundamental_table_file_refs"] = table_refs
    with pytest.raises(SystemContractError, match="Fundamental table byte binding"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(release_ref=release_ref, files=files),
        )


def test_production_bootstrap_rejects_homogeneous_v1_fundamental_claim(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    pointer_relative = "_fundamental_latest.json"
    manifest_relative = "_fundamental_generations/staged_successor/manifest.json"
    pointer = json.loads((input_root / pointer_relative).read_bytes())
    manifest = json.loads((input_root / manifest_relative).read_bytes())
    homogeneous = {
        "schema_version": "cn-fundamental-primary-provenance.v1",
        "status": "verified_live_tushare",
        "source_provenance": "live_tushare_explicit",
        "output_parquet_sha256": {name: row["sha256"] for name, row in manifest["tables"].items()},
    }
    pointer["primary_provenance"] = homogeneous
    manifest["primary_provenance"] = homogeneous
    _write(input_root, pointer_relative, canonical_json_bytes(pointer))
    _write(input_root, manifest_relative, canonical_json_bytes(manifest))
    files["fundamental_pointer_file_ref"] = _byte_ref(input_root, pointer_relative)
    files["fundamental_generation_manifest_file_ref"] = _byte_ref(input_root, manifest_relative)

    with pytest.raises(
        SystemContractError,
        match="Fundamental safe-successor provenance validation failed",
    ):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(release_ref=release_ref, files=files),
        )


def test_production_bootstrap_requires_complete_fundamental_evidence_fileset(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    files["fundamental_evidence_file_refs"] = list(files["fundamental_evidence_file_refs"])[1:]

    with pytest.raises(
        SystemContractError,
        match="Fundamental safe-successor provenance validation failed",
    ):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(release_ref=release_ref, files=files),
        )


def test_production_bootstrap_rejects_fundamental_target_binding_drift(
    tmp_path: Path,
) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    files = _inputs(input_root)
    relative = "closure/market_pointer.json"
    pointer = json.loads((input_root / relative).read_bytes())
    _write(input_root, relative, canonical_json_bytes(pointer) + b"\n")
    files["market_pointer_file_ref"] = _byte_ref(input_root, relative)

    with pytest.raises(SystemContractError, match="target source binding differs"):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=_request(release_ref=release_ref, files=files),
        )


def test_production_bootstrap_requires_empty_pointer(tmp_path: Path) -> None:
    workspace, release_ref = _seed_workspace(tmp_path)
    input_root = tmp_path / "sealed-inputs"
    raw = _request(release_ref=release_ref, files=_inputs(input_root))
    active = workspace / "results/system/_active.json"
    active.write_bytes(b"{}")
    active.chmod(0o600)

    with pytest.raises((SystemPreconditionError, SystemContractError)):
        assemble_production_bootstrap(
            workspace_root=workspace,
            input_root=input_root,
            request_raw=raw,
        )
