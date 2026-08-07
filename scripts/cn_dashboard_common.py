"""Shared fail-closed helpers for the CN aggressive holdings Dashboard.

This module is intentionally offline and read-only.  It consumes archived
strategy records and a local benchmark file; it never calls a provider, builds
portfolio candidates, mutates a strategy pointer, or creates execution state.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import stat
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "cn_aggressive_dashboard.v1"
MARKET = "CN"
STRATEGY = "aggressive_tech_manufacturing"
RECORD_NAME_RE = re.compile(r"^[0-9]{8}_[0-9]{4}$")
SYMBOL_RE = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_BENCHMARK_SOURCES = {
    "eastmoney.push2his.kline",
    "tushare.index_daily",
}
ALLOWED_BENCHMARK_COVERAGE = {"exact_close", "previous_trading_day_ffill"}
REQUIRED_LEDGER_COLUMNS = {
    "symbol",
    "name",
    "shares",
    "avg_cost",
    "cost_basis",
    "current_price",
    "current_value",
    "unrealized_pnl",
    "equity_sleeve_weight",
    "nav_weight",
}
ACCOUNTING_FIELDS = (
    "cash_after",
    "market_value_after",
    "total_value_after",
    "portfolio_pnl_after",
    "realized_pnl_from_rebalance",
)
HISTORICAL_ACCOUNTING_FIELDS = (
    "cash_after",
    "market_value_after",
    "total_value_after",
    "portfolio_pnl_after",
    "realized_pnl_from_rebalance",
)
AUTHORITY_FLAGS = {
    "benchmark_provider_calls": False,
    "broker_calls": False,
    "candidate_generation": False,
    "holdings_writes": False,
    "order_calls": False,
    "portfolio_recomputation": False,
    "provider_calls": False,
    "strategy_record_writes": False,
    "trade_calls": False,
    "v17_pointer_mutation": False,
}


class DashboardInputError(RuntimeError):
    """Raised when a required Dashboard input cannot be closed exactly."""


@dataclass(frozen=True)
class StableArtifact:
    path: Path
    relative_path: str
    data: bytes
    sha256: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _relative_to_project(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise DashboardInputError(f"artifact_outside_project:{path}") from exc


def stable_read(path: Path, project_root: Path) -> StableArtifact:
    """Read a regular non-symlink file twice and bind its exact bytes."""

    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise DashboardInputError(f"artifact_missing:{path}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise DashboardInputError(f"artifact_not_regular_non_symlink:{path}")
    first = path.read_bytes()
    second = path.read_bytes()
    if first != second:
        raise DashboardInputError(f"artifact_unstable_double_read:{path}")
    return StableArtifact(
        path=path,
        relative_path=_relative_to_project(path, project_root),
        data=first,
        sha256=sha256_bytes(first),
    )


def load_json(artifact: StableArtifact) -> Any:
    try:
        return json.loads(artifact.data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DashboardInputError(
            f"json_unreadable:{artifact.relative_path}"
        ) from exc


def _safe_same_record_path(
    record_dir: Path, declared: Any, label: str
) -> Path:
    if not isinstance(declared, str) or not declared.strip():
        raise DashboardInputError(f"{label}_missing")
    candidate = Path(declared)
    if candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        candidate = (record_dir / candidate).resolve()
    record_resolved = record_dir.resolve()
    if candidate.parent != record_resolved:
        raise DashboardInputError(f"{label}_outside_record:{declared}")
    return candidate


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DashboardInputError(f"{label}_not_numeric") from exc
    if not math.isfinite(result):
        raise DashboardInputError(f"{label}_not_finite")
    return result


def _almost_equal(left: Any, right: Any, *, tolerance: float = 0.01) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _csv_rows(artifact: StableArtifact) -> list[dict[str, str]]:
    try:
        text = artifact.data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DashboardInputError(
            f"csv_unreadable:{artifact.relative_path}"
        ) from exc
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        raise DashboardInputError(f"csv_empty:{artifact.relative_path}")
    return rows


def _ledger_rows(artifact: StableArtifact) -> list[dict[str, Any]]:
    if artifact.path.suffix == ".csv":
        return _csv_rows(artifact)
    if artifact.path.suffix != ".parquet":
        raise DashboardInputError(
            f"effective_ledger_format_invalid:{artifact.relative_path}"
        )
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pq.read_table(pa.BufferReader(artifact.data))
        rows = table.to_pylist()
    except Exception as exc:
        raise DashboardInputError(
            f"parquet_unreadable:{artifact.relative_path}"
        ) from exc
    if not rows:
        raise DashboardInputError(f"parquet_empty:{artifact.relative_path}")
    return rows


def _execution_kind(status_value: Any) -> str:
    status_text = str(status_value or "").strip().lower()
    if not status_text:
        raise DashboardInputError("manual_manifest_status_missing")
    if "no_action" in status_text or "carry_forward" in status_text:
        return "carry_forward"
    if any(
        token in status_text
        for token in ("filled", "success", "applied", "executed")
    ):
        if any(token in status_text for token in ("pending", "rejected")):
            raise DashboardInputError("manual_manifest_status_ambiguous")
        return "applied_effective_ledger"
    raise DashboardInputError(
        f"manual_manifest_status_not_effective:{status_text}"
    )


def _record_date(value: Any, label: str) -> str:
    text = str(value or "")
    try:
        return datetime.strptime(text, "%Y%m%d").date().isoformat()
    except ValueError as exc:
        raise DashboardInputError(f"{label}_invalid:{text}") from exc


def _validate_funding(
    manual: dict[str, Any], record_dir: Path, project_root: Path
) -> tuple[dict[str, Any] | None, list[StableArtifact]]:
    embedded = manual.get("manual_funding_supplement")
    if embedded in (None, {}):
        return None, []
    if not isinstance(embedded, dict):
        raise DashboardInputError("manual_funding_supplement_invalid")
    required = {
        "amount",
        "cash_before",
        "cash_after",
        "total_value_before",
        "total_value_after",
        "record_id",
        "schema_version",
        "status",
    }
    if not required.issubset(embedded):
        raise DashboardInputError("manual_funding_supplement_incomplete")
    if embedded["record_id"] != record_dir.name:
        raise DashboardInputError("manual_funding_record_id_mismatch")
    if (
        embedded["schema_version"]
        != "cn_aggressive_manual_funding_supplement.v1"
    ):
        raise DashboardInputError("manual_funding_schema_invalid")
    if embedded["status"] != "local_manual_funding_recorded_no_broker_api":
        raise DashboardInputError("manual_funding_status_invalid")
    amount = _number(embedded["amount"], "manual_funding_amount")
    if amount == 0:
        raise DashboardInputError("manual_funding_amount_zero")
    if not _almost_equal(
        _number(embedded["total_value_before"], "manual_funding_total_before")
        + amount,
        embedded["total_value_after"],
    ):
        raise DashboardInputError("manual_funding_total_reconciliation_failed")
    path_value = manual.get("manual_funding_supplement_path")
    if not path_value:
        raise DashboardInputError("manual_funding_supplement_path_missing")
    supplement_path = _safe_same_record_path(
        record_dir, path_value, "manual_funding_supplement"
    )
    supplement_artifact = stable_read(supplement_path, project_root)
    if load_json(supplement_artifact) != embedded:
        raise DashboardInputError(
            "manual_funding_supplement_readback_mismatch"
        )
    result = {
        "amount": amount,
        "total_value_before": _number(
            embedded["total_value_before"], "funding_total_before"
        ),
        "total_value_after": _number(
            embedded["total_value_after"], "funding_total_after"
        ),
        "evidence_path": supplement_artifact.relative_path,
        "evidence_sha256": supplement_artifact.sha256,
    }
    return result, [supplement_artifact]


def validate_record(
    record_dir: Path, record_root: Path, project_root: Path
) -> dict[str, Any]:
    """Validate one timestamped record and return its closed snapshot."""

    if not RECORD_NAME_RE.fullmatch(record_dir.name):
        raise DashboardInputError("record_name_invalid")
    manifest_artifact = stable_read(record_dir / "manifest.json", project_root)
    manifest = load_json(manifest_artifact)
    if not isinstance(manifest, dict):
        raise DashboardInputError("manifest_not_object")
    if manifest.get("market") != MARKET:
        raise DashboardInputError("manifest_market_mismatch")
    if manifest.get("strategy") != STRATEGY:
        raise DashboardInputError("manifest_strategy_mismatch")
    if manifest.get("timestamp") != record_dir.name:
        raise DashboardInputError("manifest_timestamp_mismatch")

    source_record = manifest.get("source_record")
    if source_record:
        if not isinstance(source_record, str) or not RECORD_NAME_RE.fullmatch(
            source_record
        ):
            raise DashboardInputError("manifest_source_record_invalid")
        source_dir = record_root / source_record
        if not source_dir.is_dir() or source_dir.is_symlink():
            raise DashboardInputError("manifest_source_record_missing")

    files = manifest.get("files")
    if not isinstance(files, dict):
        raise DashboardInputError("manifest_file_refs_missing")
    manual_path = _safe_same_record_path(
        record_dir,
        files.get("manual_execution_manifest"),
        "manifest_manual_ref",
    )
    manual_artifact = stable_read(manual_path, project_root)
    manual = load_json(manual_artifact)
    if not isinstance(manual, dict):
        raise DashboardInputError("manual_manifest_not_object")
    if manual.get("record_timestamp") != record_dir.name:
        raise DashboardInputError("manual_manifest_timestamp_mismatch")
    embedded_manual = manifest.get("manual_execution")
    if embedded_manual != manual:
        raise DashboardInputError(
            "manifest_manual_execution_readback_mismatch"
        )
    execution_kind = _execution_kind(
        manual.get("status") or manual.get("execution_status")
    )
    if manual.get("no_broker_api_called") is not True:
        raise DashboardInputError("manual_manifest_no_broker_proof_missing")

    ledger_declared = manual.get("effective_manual_ledger_path") or manual.get(
        "next_ledger_path"
    )
    ledger_path = _safe_same_record_path(
        record_dir, ledger_declared, "effective_ledger"
    )
    if ledger_path.name not in {
        "ledger_after_manual_switch.csv",
        "ledger_after_manual_switch.parquet",
    }:
        raise DashboardInputError("effective_ledger_name_invalid")
    ledger_artifact = stable_read(ledger_path, project_root)
    declared_ledger_sha = (
        manual.get("next_ledger_sha256")
        or manual.get(
            "ledger_after_manual_switch_parquet_sha256"
            if ledger_path.suffix == ".parquet"
            else "ledger_after_manual_switch_csv_sha256"
        )
        or (manual.get("ledger_provenance") or {}).get("declared_sha256")
    )
    if not isinstance(declared_ledger_sha, str) or not SHA256_RE.fullmatch(
        declared_ledger_sha
    ):
        raise DashboardInputError("effective_ledger_sha_not_declared")
    if declared_ledger_sha != ledger_artifact.sha256:
        raise DashboardInputError("effective_ledger_sha_mismatch")
    provenance = manual.get("ledger_provenance")
    if not isinstance(provenance, dict):
        raise DashboardInputError("ledger_provenance_missing")
    if not all(
        provenance.get(key) is True
        for key in (
            "contained_in_run_directory",
            "regular_non_symlink_file",
            "stable_double_read",
        )
    ):
        raise DashboardInputError("ledger_provenance_flags_invalid")
    if provenance.get("declared_sha256") != ledger_artifact.sha256:
        raise DashboardInputError("ledger_provenance_sha_mismatch")

    parquet_name = manual.get("ledger_after_manual_switch_parquet")
    parquet_artifact: StableArtifact | None = None
    if parquet_name:
        parquet_path = _safe_same_record_path(
            record_dir, parquet_name, "ledger_parquet"
        )
        parquet_artifact = stable_read(parquet_path, project_root)
        declared_parquet_sha = manual.get(
            "ledger_after_manual_switch_parquet_sha256"
        )
        if declared_parquet_sha != parquet_artifact.sha256:
            raise DashboardInputError("ledger_parquet_sha_mismatch")

    rows = _ledger_rows(ledger_artifact)
    if not REQUIRED_LEDGER_COLUMNS.issubset(rows[0]):
        raise DashboardInputError("effective_ledger_columns_missing")
    positions: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        symbol = str(row.get("symbol") or "").strip()
        name = str(row.get("name") or "").strip()
        if (
            not SYMBOL_RE.fullmatch(symbol)
            or not name
            or symbol in seen_symbols
        ):
            raise DashboardInputError(
                f"effective_ledger_identity_invalid:{row_number}"
            )
        seen_symbols.add(symbol)
        shares = _number(row.get("shares"), f"shares:{symbol}")
        if shares <= 0:
            raise DashboardInputError(f"shares_not_positive:{symbol}")
        position = {
            "symbol": symbol,
            "name": name,
            "shares": shares,
            "avg_cost": _number(row.get("avg_cost"), f"avg_cost:{symbol}"),
            "cost_basis": _number(
                row.get("cost_basis"), f"cost_basis:{symbol}"
            ),
            "recorded_price": _number(
                row.get("current_price"), f"recorded_price:{symbol}"
            ),
            "market_value": _number(
                row.get("current_value"), f"market_value:{symbol}"
            ),
            "unrealized_pnl": _number(
                row.get("unrealized_pnl"), f"unrealized_pnl:{symbol}"
            ),
            "realized_pnl": None,
            "nav_weight": _number(
                row.get("nav_weight"), f"nav_weight:{symbol}"
            ),
            "equity_weight": _number(
                row.get("equity_sleeve_weight"), f"equity_weight:{symbol}"
            ),
            "thesis_status": str(row.get("thesis_status") or "UNKNOWN").strip()
            or "UNKNOWN",
        }
        positions.append(position)
    positions.sort(key=lambda item: item["symbol"])
    expected_count = manual.get("effective_manual_holding_count")
    if expected_count is None or int(expected_count) != len(positions):
        raise DashboardInputError("effective_holding_count_mismatch")

    pnl_path = _safe_same_record_path(
        record_dir, files.get("pnl_summary"), "manifest_pnl_ref"
    )
    pnl_artifact = stable_read(pnl_path, project_root)
    pnl_rows = _csv_rows(pnl_artifact)
    pnl = pnl_rows[-1]
    accounting: dict[str, float] = {}
    for field in ACCOUNTING_FIELDS:
        if field not in manual or field not in pnl:
            raise DashboardInputError(f"accounting_field_missing:{field}")
        manual_value = _number(manual[field], f"manual_{field}")
        if not _almost_equal(manual_value, pnl[field]):
            raise DashboardInputError(
                f"pnl_manual_reconciliation_failed:{field}"
            )
        accounting[field] = manual_value
    if not _almost_equal(
        accounting["cash_after"] + accounting["market_value_after"],
        accounting["total_value_after"],
    ):
        raise DashboardInputError("accounting_cash_market_total_mismatch")
    ledger_market_value = sum(item["market_value"] for item in positions)
    if not _almost_equal(
        ledger_market_value, accounting["market_value_after"]
    ):
        raise DashboardInputError("ledger_market_value_accounting_mismatch")
    if not isinstance(
        manual.get("financial_state_sha256"), str
    ) or not SHA256_RE.fullmatch(manual["financial_state_sha256"]):
        raise DashboardInputError("financial_state_sha_missing_or_invalid")

    data_snapshot = manifest.get("data_snapshot")
    if not isinstance(data_snapshot, dict):
        raise DashboardInputError("manifest_data_snapshot_missing")
    data_date = _record_date(
        data_snapshot.get("analysis_trade_date"), "analysis_trade_date"
    )
    funding, funding_artifacts = _validate_funding(
        manual, record_dir, project_root
    )

    artifacts = [
        manifest_artifact,
        manual_artifact,
        ledger_artifact,
        pnl_artifact,
    ]
    if parquet_artifact is not None:
        artifacts.append(parquet_artifact)
    artifacts.extend(funding_artifacts)
    source_refs = [
        {"path": artifact.relative_path, "sha256": artifact.sha256}
        for artifact in artifacts
    ]
    return {
        "record": record_dir.name,
        "recorded_at": manifest.get("recorded_at")
        or manual.get("recorded_at"),
        "source_record": source_record,
        "data_date": data_date,
        "execution_status": str(
            manual.get("status") or manual.get("execution_status")
        ),
        "execution_kind": execution_kind,
        "manifest_path": manifest_artifact.relative_path,
        "manifest_sha256": manifest_artifact.sha256,
        "manual_manifest_path": manual_artifact.relative_path,
        "manual_manifest_sha256": manual_artifact.sha256,
        "ledger_path": ledger_artifact.relative_path,
        "ledger_sha256": ledger_artifact.sha256,
        "pnl_path": pnl_artifact.relative_path,
        "pnl_sha256": pnl_artifact.sha256,
        "financial_state_sha256": manual["financial_state_sha256"],
        "positions": positions,
        "accounting": accounting,
        "funding": funding,
        "source_refs": source_refs,
    }


def scan_valid_records(
    record_root: Path, project_root: Path
) -> tuple[list[dict[str, Any]], list[str], str | None]:
    if not record_root.is_dir() or record_root.is_symlink():
        raise DashboardInputError(
            f"record_root_missing_or_invalid:{record_root}"
        )
    valid: list[dict[str, Any]] = []
    warnings: list[str] = []
    candidates = sorted(
        path
        for path in record_root.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and RECORD_NAME_RE.fullmatch(path.name)
    )
    latest_seen = candidates[-1].name if candidates else None
    for record_dir in candidates:
        try:
            valid.append(
                validate_record(record_dir, record_root, project_root)
            )
        except DashboardInputError as exc:
            warnings.append(f"{record_dir.name}:{exc}")
    return valid, warnings, latest_seen


def _validate_historical_manifest(
    record_dir: Path, record_root: Path, project_root: Path
) -> tuple[dict[str, Any], StableArtifact, str | None]:
    manifest_artifact = stable_read(record_dir / "manifest.json", project_root)
    manifest = load_json(manifest_artifact)
    if not isinstance(manifest, dict):
        raise DashboardInputError("historical_manifest_not_object")
    if manifest.get("market") != MARKET:
        raise DashboardInputError("historical_manifest_market_mismatch")
    if manifest.get("strategy") != STRATEGY:
        raise DashboardInputError("historical_manifest_strategy_mismatch")
    if manifest.get("timestamp") != record_dir.name:
        raise DashboardInputError("historical_manifest_timestamp_mismatch")
    source_record = manifest.get("source_record")
    if source_record:
        if not isinstance(source_record, str) or not RECORD_NAME_RE.fullmatch(
            source_record
        ):
            raise DashboardInputError("historical_source_record_invalid")
        source_dir = record_root / source_record
        if not source_dir.is_dir() or source_dir.is_symlink():
            raise DashboardInputError("historical_source_record_missing")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise DashboardInputError("historical_manifest_file_refs_missing")
    return manifest, manifest_artifact, source_record


def _historical_valuation_date(
    *,
    record_dir: Path,
    manifest: dict[str, Any],
    pnl: dict[str, Any] | None,
    strict_record: dict[str, Any] | None,
) -> str:
    quote_snapshot = str((pnl or {}).get("quote_snapshot") or "")
    quote_match = re.match(r"^(20[0-9]{6})", quote_snapshot)
    if quote_match:
        return _record_date(quote_match.group(1), "historical_quote_date")
    intraday_snapshot = str(
        (manifest.get("data_snapshot") or {}).get(
            "intraday_quote_snapshot"
        )
        or ""
    )
    intraday_match = re.match(
        r"^(20[0-9]{2})-([0-9]{2})-([0-9]{2})", intraday_snapshot
    )
    if intraday_match:
        return date(
            int(intraday_match.group(1)),
            int(intraday_match.group(2)),
            int(intraday_match.group(3)),
        ).isoformat()
    if strict_record is not None:
        return strict_record["data_date"]
    if pnl is None and manifest.get("source_record") in (None, ""):
        return _record_date(record_dir.name[:8], "historical_baseline_date")
    raise DashboardInputError("historical_valuation_date_unverified")


def _historical_funding(
    *,
    manifest: dict[str, Any],
    record_dir: Path,
    project_root: Path,
) -> tuple[dict[str, Any] | None, list[StableArtifact]]:
    embedded_manual = manifest.get("manual_execution")
    files = manifest.get("files") or {}
    manual_ref = files.get("manual_execution_manifest")
    embedded_has_funding = isinstance(
        embedded_manual, dict
    ) and bool(embedded_manual.get("manual_funding_supplement"))
    if not manual_ref:
        if embedded_has_funding:
            raise DashboardInputError("historical_manual_ref_missing")
        return None, []
    manual_path = _safe_same_record_path(
        record_dir,
        manual_ref,
        "historical_manual_ref",
    )
    manual_artifact = stable_read(manual_path, project_root)
    manual = load_json(manual_artifact)
    if not isinstance(manual, dict):
        raise DashboardInputError("historical_manual_not_object")
    manual_has_funding = bool(manual.get("manual_funding_supplement"))
    if not manual_has_funding:
        if embedded_has_funding:
            raise DashboardInputError(
                "historical_embedded_funding_missing_from_manual"
            )
        return None, []
    if embedded_has_funding and manual != embedded_manual:
        raise DashboardInputError("historical_manual_readback_mismatch")
    if manual.get("record_timestamp") != record_dir.name:
        raise DashboardInputError("historical_manual_timestamp_mismatch")
    funding, artifacts = _validate_funding(manual, record_dir, project_root)
    assert funding is not None
    funding["binding_status"] = (
        "MANIFEST_EMBED_AND_INDEPENDENT_MANUAL_MATCH"
        if embedded_has_funding
        else "INDEPENDENT_MANUAL_AND_EXACT_SUPPLEMENT_ONLY"
    )
    return funding, [manual_artifact, *artifacts]


def _validate_historical_record(
    *,
    record_dir: Path,
    record_root: Path,
    project_root: Path,
    strict_record: dict[str, Any] | None,
) -> dict[str, Any]:
    manifest, manifest_artifact, source_record = _validate_historical_manifest(
        record_dir, record_root, project_root
    )
    files = manifest["files"]
    source_artifacts = [manifest_artifact]
    ledger_artifact: StableArtifact | None = None
    ledger_ref = files.get("ledger_after_manual_switch") or files.get(
        "ledger"
    )
    if ledger_ref:
        ledger_path = _safe_same_record_path(
            record_dir, ledger_ref, "historical_ledger_ref"
        )
        ledger_artifact = stable_read(ledger_path, project_root)
        source_artifacts.append(ledger_artifact)

    pnl_ref = files.get("pnl_summary")
    if not pnl_ref:
        if source_record not in (None, ""):
            raise DashboardInputError("historical_pnl_ref_missing")
        if ledger_artifact is None:
            raise DashboardInputError("historical_baseline_ledger_missing")
        capital = _number(
            manifest.get("capital_cny"), "historical_baseline_capital"
        )
        if capital <= 0:
            raise DashboardInputError(
                "historical_baseline_capital_not_positive"
            )
        ledger_rows = _ledger_rows(ledger_artifact)
        if "current_value" not in ledger_rows[0]:
            raise DashboardInputError(
                "historical_baseline_ledger_value_missing"
            )
        market_value = sum(
            _number(row.get("current_value"), "historical_baseline_value")
            for row in ledger_rows
        )
        cash = capital - market_value
        if cash < -0.01:
            raise DashboardInputError(
                "historical_baseline_negative_implied_cash"
            )
        valuation_date = _historical_valuation_date(
            record_dir=record_dir,
            manifest=manifest,
            pnl=None,
            strict_record=strict_record,
        )
        return {
            "record": record_dir.name,
            "source_record": None,
            "valuation_date": valuation_date,
            "accounting": {
                "cash_after": max(cash, 0.0),
                "market_value_after": market_value,
                "total_value_after": capital,
                "portfolio_pnl_after": 0.0,
                "realized_pnl_from_rebalance": 0.0,
            },
            "capital_base": capital,
            "funding": None,
            "evidence_status": "ARCHIVE_INCEPTION_EXACT_BYTES_NO_DECLARED_SHA",
            "manifest_path": manifest_artifact.relative_path,
            "manifest_sha256": manifest_artifact.sha256,
            "ledger_path": ledger_artifact.relative_path,
            "ledger_sha256": ledger_artifact.sha256,
            "pnl_path": None,
            "pnl_sha256": None,
            "formal_record": manifest.get("formal_record") is True,
            "completeness_passed": manifest.get("completeness_passed")
            is True,
            "source_refs": [
                {"path": artifact.relative_path, "sha256": artifact.sha256}
                for artifact in source_artifacts
            ],
        }

    pnl_path = _safe_same_record_path(
        record_dir, pnl_ref, "historical_pnl_ref"
    )
    pnl_artifact = stable_read(pnl_path, project_root)
    pnl_rows = _csv_rows(pnl_artifact)
    pnl = pnl_rows[-1]
    source_artifacts.append(pnl_artifact)
    accounting: dict[str, float] = {}
    for field in HISTORICAL_ACCOUNTING_FIELDS:
        if field not in pnl:
            raise DashboardInputError(
                f"historical_accounting_field_missing:{field}"
            )
        accounting[field] = _number(pnl[field], f"historical_{field}")
    if not _almost_equal(
        accounting["cash_after"] + accounting["market_value_after"],
        accounting["total_value_after"],
    ):
        raise DashboardInputError("historical_cash_market_total_mismatch")
    capital_value = pnl.get("initial_capital")
    if capital_value in (None, ""):
        capital_value = manifest.get("capital_cny")
    capital_base = (
        _number(capital_value, "historical_capital_base")
        if capital_value not in (None, "")
        else None
    )
    if capital_base is not None and not _almost_equal(
        accounting["total_value_after"] - capital_base,
        accounting["portfolio_pnl_after"],
    ):
        raise DashboardInputError("historical_portfolio_pnl_mismatch")
    funding, funding_artifacts = _historical_funding(
        manifest=manifest,
        record_dir=record_dir,
        project_root=project_root,
    )
    source_artifacts.extend(funding_artifacts)
    if funding and not _almost_equal(
        funding["total_value_after"], accounting["total_value_after"]
    ):
        raise DashboardInputError("historical_funding_total_after_mismatch")
    valuation_date = _historical_valuation_date(
        record_dir=record_dir,
        manifest=manifest,
        pnl=pnl,
        strict_record=strict_record,
    )
    return {
        "record": record_dir.name,
        "source_record": source_record,
        "valuation_date": valuation_date,
        "accounting": accounting,
        "capital_base": capital_base,
        "funding": funding,
        "evidence_status": (
            "HASH_BOUND_CURRENT_CLOSURE"
            if strict_record is not None
            else "LEGACY_EXACT_BYTES_NO_DECLARED_SHA"
        ),
        "manifest_path": manifest_artifact.relative_path,
        "manifest_sha256": manifest_artifact.sha256,
        "ledger_path": (
            ledger_artifact.relative_path if ledger_artifact else None
        ),
        "ledger_sha256": ledger_artifact.sha256 if ledger_artifact else None,
        "pnl_path": pnl_artifact.relative_path,
        "pnl_sha256": pnl_artifact.sha256,
        "formal_record": manifest.get("formal_record") is True,
        "completeness_passed": manifest.get("completeness_passed") is True,
        "source_refs": [
            {"path": artifact.relative_path, "sha256": artifact.sha256}
            for artifact in source_artifacts
        ],
    }


def scan_historical_performance_records(
    *,
    record_root: Path,
    project_root: Path,
    strict_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    strict_by_record = {record["record"]: record for record in strict_records}
    records: list[dict[str, Any]] = []
    rejected: list[str] = []
    candidates = sorted(
        path
        for path in record_root.iterdir()
        if path.is_dir()
        and not path.is_symlink()
        and RECORD_NAME_RE.fullmatch(path.name)
    )
    for record_dir in candidates:
        try:
            records.append(
                _validate_historical_record(
                    record_dir=record_dir,
                    record_root=record_root,
                    project_root=project_root,
                    strict_record=strict_by_record.get(record_dir.name),
                )
            )
        except DashboardInputError as exc:
            rejected.append(f"{record_dir.name}:{exc}")
    if len(records) < 2:
        raise DashboardInputError(
            "fewer_than_two_historical_performance_records"
        )
    previous_capital: float | None = None
    for record in records:
        capital = record["capital_base"]
        if (
            previous_capital is not None
            and capital is not None
            and not _almost_equal(previous_capital, capital)
            and record.get("funding") is None
        ):
            raise DashboardInputError(
                "historical_capital_change_without_funding:"
                + record["record"]
            )
        if capital is not None:
            previous_capital = capital
    return records, rejected


def _read_benchmark_rows(
    artifact: StableArtifact, ts_code: str
) -> dict[str, dict[str, Any]]:
    rows = _csv_rows(artifact)
    required = {
        "date",
        "ts_code",
        "close",
        "source_system",
        "value_date",
        "coverage",
    }
    if not required.issubset(rows[0]):
        raise DashboardInputError("benchmark_columns_missing")
    selected: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=2):
        if row.get("ts_code") != ts_code:
            continue
        date_value = str(row.get("date") or "")
        value_date = str(row.get("value_date") or "")
        try:
            current_date = date.fromisoformat(date_value)
            source_date = date.fromisoformat(value_date)
        except ValueError as exc:
            raise DashboardInputError(
                f"benchmark_date_invalid:{row_number}"
            ) from exc
        source_system = str(row.get("source_system") or "")
        coverage = str(row.get("coverage") or "")
        if source_system not in ALLOWED_BENCHMARK_SOURCES:
            raise DashboardInputError(
                f"benchmark_source_forbidden:{source_system}"
            )
        if coverage not in ALLOWED_BENCHMARK_COVERAGE:
            raise DashboardInputError(f"benchmark_coverage_invalid:{coverage}")
        if coverage == "exact_close" and source_date != current_date:
            raise DashboardInputError(
                f"benchmark_exact_value_date_mismatch:{date_value}"
            )
        if (
            coverage == "previous_trading_day_ffill"
            and source_date >= current_date
        ):
            raise DashboardInputError(
                f"benchmark_ffill_value_date_invalid:{date_value}"
            )
        if date_value in selected:
            raise DashboardInputError(f"benchmark_duplicate_date:{date_value}")
        selected[date_value] = {
            "date": date_value,
            "close": _number(
                row.get("close"), f"benchmark_close:{date_value}"
            ),
            "source_system": source_system,
            "value_date": value_date,
            "coverage": coverage,
        }
    return selected


def _max_drawdown(values: Iterable[float]) -> float:
    peak: float | None = None
    result = 0.0
    for value in values:
        peak = value if peak is None else max(peak, value)
        if peak > 0:
            result = min(result, value / peak - 1.0)
    return result


def _changes(
    current: dict[str, Any], previous: dict[str, Any]
) -> list[dict[str, Any]]:
    current_by_symbol = {row["symbol"]: row for row in current["positions"]}
    previous_by_symbol = {row["symbol"]: row for row in previous["positions"]}
    result: list[dict[str, Any]] = []
    for symbol in sorted(set(current_by_symbol) | set(previous_by_symbol)):
        now = current_by_symbol.get(symbol)
        before = previous_by_symbol.get(symbol)
        before_shares = before["shares"] if before else 0.0
        now_shares = now["shares"] if now else 0.0
        delta = now_shares - before_shares
        if before is None:
            change_type = "NEW"
        elif now is None:
            change_type = "CLOSED"
        elif delta > 0:
            change_type = "INCREASED"
        elif delta < 0:
            change_type = "REDUCED"
        else:
            change_type = "UNCHANGED"
        row = now or before
        result.append(
            {
                "symbol": symbol,
                "name": row["name"],
                "change_type": change_type,
                "previous_shares": before_shares,
                "current_shares": now_shares,
                "share_delta": delta,
                "nav_weight_delta": (now["nav_weight"] if now else 0.0)
                - (before["nav_weight"] if before else 0.0),
                "equity_weight_delta": (now["equity_weight"] if now else 0.0)
                - (before["equity_weight"] if before else 0.0),
            }
        )
    return result


def _unitize(records_by_date: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first = records_by_date[0]
    first_total = first["accounting"]["total_value_after"]
    if first_total <= 0:
        raise DashboardInputError("unitization_initial_total_not_positive")
    units = first_total
    timeline = [
        {
            "date": first.get("valuation_date") or first["data_date"],
            "record": first["record"],
            "unit_nav": 1.0,
            "total_value": first_total,
            "evidence_status": first.get(
                "evidence_status", "HASH_BOUND_CURRENT_CLOSURE"
            ),
        }
    ]
    for record in records_by_date[1:]:
        total_after = record["accounting"]["total_value_after"]
        funding = record.get("funding")
        if funding:
            flow_nav = funding["total_value_before"] / units
            if flow_nav <= 0:
                raise DashboardInputError("unitization_flow_nav_not_positive")
            units += funding["amount"] / flow_nav
        unit_nav = total_after / units
        if unit_nav <= 0 or not math.isfinite(unit_nav):
            raise DashboardInputError("unitization_nav_invalid")
        timeline.append(
            {
                "date": record.get("valuation_date")
                or record["data_date"],
                "record": record["record"],
                "unit_nav": unit_nav,
                "total_value": total_after,
                "evidence_status": record.get(
                    "evidence_status", "HASH_BOUND_CURRENT_CLOSURE"
                ),
            }
        )
    return timeline


def build_bundle(
    *,
    project_root: Path,
    record_root: Path,
    benchmark_path: Path,
    generated_at: str,
    today: date,
) -> dict[str, Any]:
    valid, rejected, latest_seen = scan_valid_records(
        record_root, project_root
    )
    if len(valid) < 2:
        raise DashboardInputError("fewer_than_two_hash_bound_valid_records")
    latest = valid[-1]
    previous = valid[-2]
    historical_records, historical_rejected = (
        scan_historical_performance_records(
            record_root=record_root,
            project_root=project_root,
            strict_records=valid,
        )
    )
    unitized_raw = _unitize(historical_records)
    collapsed_unitized: dict[str, dict[str, Any]] = {}
    for point in unitized_raw:
        collapsed_unitized[point["date"]] = point
    unitized = [
        collapsed_unitized[key] for key in sorted(collapsed_unitized)
    ]
    if len(unitized) < 2:
        raise DashboardInputError(
            "portfolio_performance_has_no_comparable_interval"
        )

    benchmark_artifact = stable_read(benchmark_path, project_root)
    benchmark_rows = _read_benchmark_rows(benchmark_artifact, "000300.SH")
    required_dates = [row["date"] for row in unitized]
    missing_dates = [
        value for value in required_dates if value not in benchmark_rows
    ]
    if missing_dates:
        raise DashboardInputError(
            "csi300_benchmark_missing_dates:" + ",".join(missing_dates)
        )
    selected_benchmark = [benchmark_rows[value] for value in required_dates]
    first_close = selected_benchmark[0]["close"]
    if first_close <= 0:
        raise DashboardInputError("csi300_initial_close_not_positive")
    benchmark_nav = [row["close"] / first_close for row in selected_benchmark]

    portfolio_nav = [row["unit_nav"] for row in unitized]
    cumulative_twr = portfolio_nav[-1] / portfolio_nav[0] - 1.0
    benchmark_return = benchmark_nav[-1] / benchmark_nav[0] - 1.0
    changes = _changes(latest, previous)
    gross = (
        latest["accounting"]["market_value_after"]
        / latest["accounting"]["total_value_after"]
    )
    cash_weight = (
        latest["accounting"]["cash_after"]
        / latest["accounting"]["total_value_after"]
    )
    equity_weights = sorted(
        (position["equity_weight"] for position in latest["positions"]),
        reverse=True,
    )
    top1 = sum(equity_weights[:1])
    top3 = sum(equity_weights[:3])
    hhi = sum(weight * weight for weight in equity_weights)
    current_unrealized = sum(
        position["unrealized_pnl"] for position in latest["positions"]
    )
    data_age_days = (today - date.fromisoformat(latest["data_date"])).days
    rejection_counts = Counter(
        (item.split(":", 1)[1] if ":" in item else item).split(":", 1)[0]
        for item in rejected
    )
    historical_rejection_counts = Counter(
        (item.split(":", 1)[1] if ":" in item else item).split(":", 1)[0]
        for item in historical_rejected
    )
    legacy_history_count = sum(
        record["evidence_status"]
        in {
            "ARCHIVE_INCEPTION_EXACT_BYTES_NO_DECLARED_SHA",
            "LEGACY_EXACT_BYTES_NO_DECLARED_SHA",
        }
        for record in historical_records
    )
    funding_events = [
        {
            "record": record["record"],
            "date": record["valuation_date"],
            "amount": record["funding"]["amount"],
            "total_value_before": record["funding"]["total_value_before"],
            "total_value_after": record["funding"]["total_value_after"],
            "evidence_path": record["funding"]["evidence_path"],
            "evidence_sha256": record["funding"]["evidence_sha256"],
            "binding_status": record["funding"]["binding_status"],
        }
        for record in historical_records
        if record.get("funding") is not None
    ]
    warnings = [
        "trade_fee_and_net_of_fee_basis_unknown",
        "per_position_realized_pnl_unavailable",
        "current_quote_unavailable_recorded_prices_only",
        "industry_and_theme_exposure_not_hash_bound_in_effective_ledger",
    ]
    if data_age_days > 3:
        warnings.append(f"latest_data_stale_calendar_days:{data_age_days}")
    if latest_seen != latest["record"]:
        warnings.append(f"newer_unusable_record_exists:{latest_seen}")
    if rejected:
        warnings.append(f"current_holdings_records_rejected:{len(rejected)}")
    if legacy_history_count:
        warnings.append(
            "historical_performance_legacy_exact_bytes_without_declared_sha:"
            f"{legacy_history_count}"
        )
    if historical_rejected:
        warnings.append(
            "historical_performance_records_rejected:"
            f"{len(historical_rejected)}"
        )
    status = "PARTIAL" if warnings else "FRESH"

    source_refs: list[dict[str, str]] = []
    seen_refs: set[tuple[str, str]] = set()
    for record in valid:
        for ref in record["source_refs"]:
            identity = (ref["path"], ref["sha256"])
            if identity not in seen_refs:
                source_refs.append(ref)
                seen_refs.add(identity)
    for record in historical_records:
        for ref in record["source_refs"]:
            identity = (ref["path"], ref["sha256"])
            if identity not in seen_refs:
                source_refs.append(ref)
                seen_refs.add(identity)
    source_refs.append(
        {
            "path": benchmark_artifact.relative_path,
            "sha256": benchmark_artifact.sha256,
        }
    )

    for position in latest["positions"]:
        position["price_date"] = latest["data_date"]
        position["evidence_status"] = "HASH_BOUND_EFFECTIVE_LEDGER"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": status,
        "market": MARKET,
        "strategy_label": STRATEGY,
        "strategy_id_kind": "HISTORICAL_DISPLAY_LABEL_NOT_V17_CANONICAL_ID",
        "read_only": True,
        "authority_flags": dict(AUTHORITY_FLAGS),
        "latest_record_seen": latest_seen,
        "latest_valid_record": latest["record"],
        "previous_valid_record": previous["record"],
        "latest_data_date": latest["data_date"],
        "data_age_calendar_days": data_age_days,
        "current_evidence": {
            key: latest[key]
            for key in (
                "manifest_path",
                "manifest_sha256",
                "manual_manifest_path",
                "manual_manifest_sha256",
                "ledger_path",
                "ledger_sha256",
                "pnl_path",
                "pnl_sha256",
                "financial_state_sha256",
                "execution_status",
                "execution_kind",
            )
        },
        "previous_evidence": {
            key: previous[key]
            for key in (
                "manifest_path",
                "manifest_sha256",
                "manual_manifest_path",
                "manual_manifest_sha256",
                "ledger_path",
                "ledger_sha256",
                "pnl_path",
                "pnl_sha256",
                "financial_state_sha256",
                "execution_status",
                "execution_kind",
            )
        },
        "history": {
            "archive_start_record": historical_records[0]["record"],
            "archive_start_date": unitized[0]["date"],
            "first_pnl_record": next(
                record["record"]
                for record in historical_records
                if record["pnl_path"] is not None
            ),
            "first_pnl_date": next(
                record["valuation_date"]
                for record in historical_records
                if record["pnl_path"] is not None
            ),
            "latest_performance_record": unitized[-1]["record"],
            "latest_performance_date": unitized[-1]["date"],
            "included_record_count": len(historical_records),
            "performance_point_count": len(unitized),
            "legacy_exact_byte_record_count": legacy_history_count,
            "hash_bound_current_record_count": sum(
                record["evidence_status"]
                == "HASH_BOUND_CURRENT_CLOSURE"
                for record in historical_records
            ),
            "funding_events": funding_events,
            "net_external_flow": sum(
                event["amount"] for event in funding_events
            ),
            "rejected_record_count": len(historical_rejected),
            "rejected_record_reason_counts": dict(
                sorted(historical_rejection_counts.items())
            ),
            "rejected_record_samples": historical_rejected[-12:],
            "evidence_status": (
                "PARTIAL_LEGACY_EXACT_BYTES_NO_DECLARED_SHA"
                if legacy_history_count
                else "HASH_BOUND_CURRENT_CLOSURE_ONLY"
            ),
            "baseline_manifest_path": historical_records[0][
                "manifest_path"
            ],
            "baseline_manifest_sha256": historical_records[0][
                "manifest_sha256"
            ],
            "baseline_ledger_path": historical_records[0]["ledger_path"],
            "baseline_ledger_sha256": historical_records[0][
                "ledger_sha256"
            ],
        },
        "positions": latest["positions"],
        "changes": changes,
        "portfolio": {
            "cash": latest["accounting"]["cash_after"],
            "market_value": latest["accounting"]["market_value_after"],
            "total_value": latest["accounting"]["total_value_after"],
            "cash_weight": cash_weight,
            "gross_exposure": gross,
            "portfolio_pnl": latest["accounting"]["portfolio_pnl_after"],
            "current_unrealized_pnl": current_unrealized,
            "latest_record_realized_pnl_from_rebalance": latest["accounting"][
                "realized_pnl_from_rebalance"
            ],
            "cumulative_realized_pnl": None,
            "realized_pnl_evidence_status": "UNKNOWN",
            "cumulative_twr": cumulative_twr,
            "latest_record_interval_return": latest["accounting"][
                "total_value_after"
            ]
            / previous["accounting"]["total_value_after"]
            - 1.0,
            "max_drawdown": _max_drawdown(portfolio_nav),
            "latest_interval_turnover": 0.5
            * sum(abs(change["nav_weight_delta"]) for change in changes),
            "return_method": "funding_aware_time_weighted_unitization",
            "gross_or_net": "UNKNOWN",
            "fee_basis": "UNKNOWN",
            "performance_start_date": unitized[0]["date"],
            "performance_end_date": unitized[-1]["date"],
            "performance_points": [
                {
                    "date": row["date"],
                    "record": row["record"],
                    "total_value": row["total_value"],
                    "portfolio_unit_nav": row["unit_nav"],
                    "portfolio_cumulative_return": row["unit_nav"]
                    / portfolio_nav[0]
                    - 1.0,
                    "csi300_nav": benchmark_nav[index],
                    "csi300_cumulative_return": benchmark_nav[index]
                    / benchmark_nav[0]
                    - 1.0,
                    "cumulative_excess_return": row["unit_nav"]
                    / portfolio_nav[0]
                    - benchmark_nav[index] / benchmark_nav[0],
                    "benchmark_coverage": selected_benchmark[index][
                        "coverage"
                    ],
                    "benchmark_value_date": selected_benchmark[index][
                        "value_date"
                    ],
                    "evidence_status": row["evidence_status"],
                }
                for index, row in enumerate(unitized)
            ],
        },
        "benchmarks": [
            {
                "id": "CSI300",
                "name": "沪深300",
                "ts_code": "000300.SH",
                "source_path": benchmark_artifact.relative_path,
                "source_sha256": benchmark_artifact.sha256,
                "start_date": required_dates[0],
                "end_date": required_dates[-1],
                "return": benchmark_return,
                "excess_return": cumulative_twr - benchmark_return,
                "max_drawdown": _max_drawdown(benchmark_nav),
                "missing_dates": [],
                "coverage": [row["coverage"] for row in selected_benchmark],
            }
        ],
        "concentration": {
            "top1_equity_weight": top1,
            "top3_equity_weight": top3,
            "equity_hhi": hhi,
            "holding_count": len(latest["positions"]),
            "thesis_status_counts": {
                value: sum(
                    1
                    for row in latest["positions"]
                    if row["thesis_status"] == value
                )
                for value in sorted(
                    {row["thesis_status"] for row in latest["positions"]}
                )
            },
        },
        "risks": [
            {
                "code": "RECORDED_PRICE_STALE",
                "severity": "HIGH" if data_age_days > 5 else "MEDIUM",
                "detail": f"持仓价格与估值数据日期为 {latest['data_date']}，不是当前行情。",
            },
            {
                "code": "EQUITY_CONCENTRATION",
                "severity": "HIGH" if top3 >= 0.8 else "MEDIUM",
                "detail": f"权益仓前三大权重 {top3:.2%}，权益 HHI {hhi:.4f}。",
            },
            {
                "code": "THESIS_STATUS_PRESSURE",
                "severity": "HIGH",
                "detail": "有效 ledger 中多数持仓的 thesis_status 为降级观察。",
            },
            {
                "code": "FEE_AND_REALIZED_PNL_UNKNOWN",
                "severity": "MEDIUM",
                "detail": "费用毛净口径与累计已实现盈亏缺少 hash-bound 分项证据。",
            },
            {
                "code": "LEGACY_HISTORY_EVIDENCE_PARTIAL",
                "severity": "MEDIUM",
                "detail": (
                    f"自归档起点纳入 {legacy_history_count} 条旧记录；"
                    "文件已 exact-byte 绑定，但旧 manifest 未声明 SHA，"
                    "不得作为当前持仓权威。"
                ),
            },
        ],
        "i1_research": None,
        "i1_display_status": "NOT_DISPLAYED_NO_EXACT_HASH_BOUND_I1_ARTIFACT",
        "blockers": [],
        "warnings": sorted(set(warnings)),
        "valid_record_count": len(valid),
        "rejected_record_count": len(rejected),
        "rejected_record_reason_counts": dict(
            sorted(rejection_counts.items())
        ),
        "rejected_record_samples": rejected[-12:],
        "source_refs": sorted(
            source_refs, key=lambda item: (item["path"], item["sha256"])
        ),
    }
    payload["content_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def validate_bundle_shape(bundle: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(bundle, dict):
        return ["bundle_not_object"]
    if bundle.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version_invalid")
    if bundle.get("status") not in {"FRESH", "PARTIAL", "BLOCKED"}:
        errors.append("status_invalid")
    if (
        bundle.get("market") != MARKET
        or bundle.get("strategy_label") != STRATEGY
    ):
        errors.append("identity_invalid")
    if bundle.get("read_only") is not True:
        errors.append("read_only_invalid")
    flags = bundle.get("authority_flags")
    if flags != AUTHORITY_FLAGS:
        errors.append("authority_flags_invalid")
    if bundle.get("status") in {"FRESH", "PARTIAL"}:
        if bundle.get("blockers") != []:
            errors.append("usable_bundle_has_blockers")
        if not bundle.get("positions"):
            errors.append("usable_bundle_has_no_positions")
        if len(bundle.get("benchmarks") or []) < 1:
            errors.append("usable_bundle_has_no_benchmark")
        history = bundle.get("history")
        portfolio = bundle.get("portfolio")
        if not isinstance(history, dict):
            errors.append("usable_bundle_has_no_history")
        elif not isinstance(portfolio, dict):
            errors.append("usable_bundle_has_no_portfolio")
        else:
            points = portfolio.get("performance_points")
            if not isinstance(points, list) or len(points) < 2:
                errors.append("usable_bundle_has_no_performance_history")
            elif (
                portfolio.get("performance_start_date")
                != history.get("archive_start_date")
                or points[0].get("record")
                != history.get("archive_start_record")
            ):
                errors.append("performance_history_start_mismatch")
    if bundle.get("i1_research") is None and bundle.get(
        "i1_display_status"
    ) != ("NOT_DISPLAYED_NO_EXACT_HASH_BOUND_I1_ARTIFACT"):
        errors.append("i1_absence_status_invalid")
    content_sha = bundle.get("content_sha256")
    if not isinstance(content_sha, str) or not SHA256_RE.fullmatch(
        content_sha
    ):
        errors.append("content_sha256_invalid")
    else:
        without_hash = dict(bundle)
        without_hash.pop("content_sha256", None)
        if sha256_bytes(canonical_json_bytes(without_hash)) != content_sha:
            errors.append("content_sha256_mismatch")
    return errors


def verify_source_refs(
    bundle: dict[str, Any], project_root: Path
) -> list[str]:
    errors: list[str] = []
    refs = bundle.get("source_refs")
    if not isinstance(refs, list) or not refs:
        return ["source_refs_missing"]
    seen: set[str] = set()
    for index, ref in enumerate(refs):
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            errors.append(f"source_ref_shape_invalid:{index}")
            continue
        relative_path = ref.get("path")
        declared_sha = ref.get("sha256")
        if (
            not isinstance(relative_path, str)
            or relative_path.startswith("/")
            or ".." in Path(relative_path).parts
        ):
            errors.append(f"source_ref_path_invalid:{index}")
            continue
        if relative_path in seen:
            errors.append(f"source_ref_duplicate:{relative_path}")
            continue
        seen.add(relative_path)
        try:
            artifact = stable_read(project_root / relative_path, project_root)
        except DashboardInputError as exc:
            errors.append(str(exc))
            continue
        if declared_sha != artifact.sha256:
            errors.append(f"source_ref_sha_mismatch:{relative_path}")
    return errors
