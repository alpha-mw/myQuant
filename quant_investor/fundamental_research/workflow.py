"""Offline prepare/import/status workflow for external Codex research."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Annotated, Literal, Mapping
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import Field, StringConstraints

from .governance import (
    ActivationGateEvidenceV2,
    ELIGIBLE_HOLDINGS_STATUSES,
    HoldingsScopeSnapshotV1,
    activation_readiness,
    build_activation_gate_evidence,
    verify_recomputed_evidence,
)
from . import governance as _governance
from .ledger import HashChainLedger, validate_job_transition
from .models import (
    Dimension,
    FundamentalOverlayV1,
    LocalFundamentalContextV1,
    FundamentalResearchRequestV1,
    FundamentalResearchResponseV1,
    JobEventV1,
    JobState,
    SourceEligibilityPolicyV1,
    StrictModel,
    compute_base_score_sha256,
    compute_source_policy_sha256,
)
from .scoring import DIMENSION_WEIGHTS
from .service import validate_response
from .storage import (
    MAX_JSON_BYTES,
    atomic_write_json_model,
    canonical_json_bytes,
    load_json_model,
    model_sha256,
    sha256_bytes,
)

DEFAULT_ROOT = Path("results/fundamental_research")
ANALYSIS_MANIFEST_NAME = "analysis_run_manifest.v1.json"
PREPARE_MANIFEST_NAME = "manifest.v1.json"
PROMPT_VERSION = "fundamental-dossier-v1"
POLICY_VERSION = "v13.2-fundamental-research"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,64}$")
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
_UNKNOWN_NAMES = {
    "",
    "unknown",
    "unconfirmed",
    "n/a",
    "na",
    "none",
    "未知",
    "未知公司",
}


class WorkflowInputError(ValueError):
    """An input lineage or workflow invariant failed closed."""


class PrepareBlockerV1(StrictModel):
    symbol: Annotated[str, StringConstraints(strip_whitespace=True, max_length=32)] = ""
    code: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)]
    detail: Annotated[str, StringConstraints(max_length=500)] = ""


class PreparedRequestRecordV1(StrictModel):
    request_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    symbol: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    company_name: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    selection_reasons: list[str]
    request_path: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    request_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    task_path: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    task_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    response_path: Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class ResearchDimensionSpecV1(StrictModel):
    dimension: Dimension
    weight: float = Field(gt=0.0, le=1.0)


class FundamentalResearchTaskPacketV1(StrictModel):
    schema_version: Literal["fundamental-research-task.v1"] = "fundamental-research-task.v1"
    request: FundamentalResearchRequestV1
    request_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    response_path: Annotated[str, StringConstraints(min_length=1, max_length=1024)]
    dimensions: list[ResearchDimensionSpecV1] = Field(min_length=6, max_length=6)
    source_taxonomy: dict[str, list[str]]
    research_constraints: list[str]
    context_blockers: list[str] = Field(default_factory=list)
    allowed_signals: list[str]
    response_schema: Literal["FundamentalResearchResponseV1"] = "FundamentalResearchResponseV1"
    response_json_schema: dict[str, Any]


class FundamentalResearchPrepareManifestV1(StrictModel):
    schema_version: Literal["fundamental-research-prepare.v1"] = "fundamental-research-prepare.v1"
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    market: Literal["CN"] = "CN"
    created_at: datetime
    analysis_manifest_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    manual_manifest_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    manual_ledger_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    requested: list[PreparedRequestRecordV1] = Field(default_factory=list)
    blockers: list[PrepareBlockerV1] = Field(default_factory=list)


class ImportReportV1(StrictModel):
    schema_version: Literal["fundamental-research-import.v1"] = "fundamental-research-import.v1"
    request_id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    symbol: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    imported_at: datetime
    status: Literal["VALIDATED", "REJECTED"]
    error: Annotated[str, StringConstraints(max_length=2000)] = ""
    request_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    response_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")] | None = None
    dossier_path: Annotated[str, StringConstraints(max_length=1024)] = ""
    overlay_path: Annotated[str, StringConstraints(max_length=1024)] = ""


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(value)))


def _read_stable_bytes(path: Path, *, max_bytes: int | None = None) -> bytes:
    if path.is_symlink():
        raise WorkflowInputError(f"symlink input is forbidden: {path}")
    before = path.stat()
    if max_bytes is not None and before.st_size > max_bytes:
        raise WorkflowInputError(f"input exceeds {max_bytes} bytes: {path}")
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise WorkflowInputError(f"input changed during read: {path}")
    return payload


def _read_json_mapping(
    path: Path, *, max_bytes: int = 100 * 1024 * 1024
) -> tuple[dict[str, Any], bytes]:
    payload = _read_stable_bytes(path, max_bytes=max_bytes)
    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise WorkflowInputError(f"invalid JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkflowInputError(f"JSON input must be an object: {path}")
    return value, payload


def _resolve_analysis_manifest(path: str | Path) -> Path:
    target = Path(path)
    if target.is_dir():
        target = target / ANALYSIS_MANIFEST_NAME
    if not target.is_file():
        raise WorkflowInputError(f"analysis manifest not found: {target}")
    return target


def _git_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise WorkflowInputError("unable to verify current Git SHA") from exc
    return completed.stdout.strip().lower()


def _validate_analysis_manifest(path: Path) -> tuple[dict[str, Any], str]:
    manifest, _ = _read_json_mapping(path)
    if manifest.get("schema_version") != "analysis-run-manifest.v1":
        raise WorkflowInputError("unsupported analysis manifest schema")
    if str(manifest.get("market", "")).upper() != "CN":
        raise WorkflowInputError("fundamental research v1 only supports CN")
    meta = manifest.get("analysis_meta")
    if not isinstance(meta, dict):
        raise WorkflowInputError("analysis_meta must be an object")
    expected_meta = str(manifest.get("analysis_meta_sha256", "")).lower()
    if not _SHA256_RE.fullmatch(expected_meta) or _canonical_mapping_sha256(meta) != expected_meta:
        raise WorkflowInputError("analysis_meta_sha256 mismatch")
    sealed = dict(manifest)
    supplied_hash = str(sealed.pop("manifest_sha256", "")).lower()
    if (
        not _SHA256_RE.fullmatch(supplied_hash)
        or _canonical_mapping_sha256(sealed) != supplied_hash
    ):
        raise WorkflowInputError("analysis manifest_sha256 mismatch")
    declared_git = str(manifest.get("git_sha", "")).strip().lower()
    if not _GIT_SHA_RE.fullmatch(declared_git):
        raise WorkflowInputError("analysis git_sha is missing or invalid")
    current_git = _git_sha()
    if not current_git.startswith(declared_git):
        raise WorkflowInputError(
            f"analysis git_sha {declared_git} does not match current checkout {current_git}"
        )
    run_id = str(manifest.get("run_id", "")).strip()
    if not run_id or len(run_id) > 128 or not _IDENTIFIER_RE.fullmatch(run_id):
        raise WorkflowInputError("analysis run_id is invalid")
    try:
        generated_at = datetime.fromisoformat(str(manifest.get("generated_at", "")))
    except ValueError as exc:
        raise WorkflowInputError("analysis generated_at is invalid") from exc
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise WorkflowInputError("analysis generated_at must be timezone-aware")
    if str(meta.get("market", "")).upper() != "CN":
        raise WorkflowInputError("analysis_meta market mismatch")
    return manifest, supplied_hash


def _resolve_contained_input(manifest_path: Path, declared: str) -> Path:
    base = manifest_path.parent.resolve(strict=True)
    candidate = Path(declared)
    candidate = candidate if candidate.is_absolute() else manifest_path.parent / candidate
    if candidate.is_symlink():
        raise WorkflowInputError("manual next_ledger_path cannot be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise WorkflowInputError("manual next_ledger_path escapes manifest directory") from exc
    if not resolved.is_file():
        raise WorkflowInputError("manual next_ledger_path is not a file")
    if resolved.name not in {
        "ledger_after_manual_switch.csv",
        "ledger_after_manual_switch.parquet",
    }:
        raise WorkflowInputError(
            "manual next_ledger_path must select ledger_after_manual_switch.csv or .parquet"
        )
    return resolved


def _parse_manifest_time(manifest: Mapping[str, Any]) -> datetime:
    raw = next(
        (
            str(manifest.get(key, "")).strip()
            for key in ("recorded_at", "record_timestamp", "timestamp")
            if str(manifest.get(key, "")).strip()
        ),
        "",
    )
    if not raw:
        raise WorkflowInputError("manual manifest timestamp is missing")
    try:
        value = datetime.fromisoformat(raw)
    except ValueError as exc:
        try:
            value = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S CST").replace(
                tzinfo=ZoneInfo("Asia/Shanghai")
            )
        except ValueError:
            raise WorkflowInputError(
                "manual manifest timestamp must be ISO-8601 or legacy CN CST"
            ) from exc
    if value.tzinfo is None or value.utcoffset() is None:
        raise WorkflowInputError("manual manifest timestamp must be timezone-aware")
    return value


def _load_manual_holdings(
    path: Path,
) -> tuple[list[dict[str, str]], str, str, Path]:
    manifest, manifest_bytes = _read_json_mapping(path, max_bytes=MAX_JSON_BYTES)
    status = str(manifest.get("status", "")).strip().casefold()
    if not status:
        raise WorkflowInputError("manual manifest status is missing")
    if status not in ELIGIBLE_HOLDINGS_STATUSES:
        raise WorkflowInputError("manual manifest status is not eligible")
    _parse_manifest_time(manifest)
    next_ledger = str(
        manifest.get("ledger_after_manual_switch_parquet")
        or manifest.get("next_ledger_path")
        or manifest.get("effective_manual_ledger_path")
        or ""
    ).strip()
    if not next_ledger:
        raise WorkflowInputError("manual manifest next_ledger_path is missing")
    ledger_path = _resolve_contained_input(path, next_ledger)
    before_bytes = _read_stable_bytes(ledger_path)
    ledger_sha = hashlib.sha256(before_bytes).hexdigest()
    extension_key = "ledger_after_manual_switch_parquet_sha256"
    declared_hashes = [
        str(manifest.get(key, "")).strip().lower()
        for key in ("next_ledger_sha256", "ledger_sha256", extension_key)
        if str(manifest.get(key, "")).strip()
    ]
    if any(not _SHA256_RE.fullmatch(item) or item != ledger_sha for item in declared_hashes):
        raise WorkflowInputError("manual ledger declared sha256 mismatch")
    if ledger_path.suffix.lower() != ".parquet":
        raise WorkflowInputError("manual ledger must use the canonical Parquet sidecar")
    frame = pd.read_parquet(ledger_path)
    if hashlib.sha256(_read_stable_bytes(ledger_path)).hexdigest() != ledger_sha:
        raise WorkflowInputError("manual ledger changed during parse")
    symbol_column = next(
        (item for item in ("symbol", "ts_code", "code") if item in frame.columns), ""
    )
    name_column = next(
        (item for item in ("name", "company_name", "stock_name") if item in frame.columns), ""
    )
    shares_column = next(
        (item for item in ("shares", "quantity", "position") if item in frame.columns), ""
    )
    if not symbol_column:
        raise WorkflowInputError("manual ledger symbol column is missing")
    holdings: list[dict[str, str]] = []
    for row in frame.to_dict(orient="records"):
        if shares_column:
            try:
                if float(row.get(shares_column, 0.0) or 0.0) <= 0.0:
                    continue
            except (TypeError, ValueError):
                continue
        symbol = str(row.get(symbol_column, "") or "").strip().upper()
        if not symbol or symbol in {item["symbol"] for item in holdings}:
            continue
        holdings.append(
            {
                "symbol": symbol,
                "company_name": str(row.get(name_column, "") or "").strip() if name_column else "",
            }
        )
    return holdings, hashlib.sha256(manifest_bytes).hexdigest(), ledger_sha, ledger_path


def _data_snapshot(meta: Mapping[str, Any]) -> Mapping[str, Any]:
    snapshot = meta.get("data_snapshot", {})
    if not isinstance(snapshot, Mapping):
        snapshot = {}
    if snapshot:
        return snapshot
    global_context = meta.get("global_context", {})
    if not isinstance(global_context, Mapping):
        return {}
    nested = global_context.get("metadata", {})
    nested = nested if isinstance(nested, Mapping) else {}
    snapshot = nested.get("data_snapshot", {})
    return snapshot if isinstance(snapshot, Mapping) else {}


def _identifier_value(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if len(text) <= 128 and _IDENTIFIER_RE.fullmatch(text):
        return text
    return f"generation-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]}"


def _source_policy() -> tuple[SourceEligibilityPolicyV1, str]:
    policy = SourceEligibilityPolicyV1()
    extra: set[str] = set()
    for value in os.environ.get("FUNDAMENTAL_RESEARCH_EXTRA_PRIMARY_HOSTNAMES", "").split(","):
        hostname = value.strip().lower().rstrip(".")
        if not hostname:
            continue
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?", hostname):
            raise WorkflowInputError(
                "FUNDAMENTAL_RESEARCH_EXTRA_PRIMARY_HOSTNAMES contains an invalid hostname"
            )
        extra.add(hostname)
    if extra:
        policy = policy.model_copy(
            update={"primary_hostnames": set(policy.primary_hostnames) | extra}
        )
    return policy, compute_source_policy_sha256(policy)


def _analysis_data_cutoff(meta: Mapping[str, Any]) -> datetime:
    snapshot = _data_snapshot(meta)
    global_context = meta.get("global_context", {})
    global_context = global_context if isinstance(global_context, Mapping) else {}
    raw = next(
        (
            str(value).strip()
            for value in (
                snapshot.get("stable_trade_date"),
                snapshot.get("local_latest_trade_date"),
                snapshot.get("latest_trade_date"),
                global_context.get("latest_trade_date"),
            )
            if str(value or "").strip()
        ),
        "",
    )
    if not raw:
        raise WorkflowInputError("analysis data cutoff is missing")
    normalized = raw.replace("-", "")
    try:
        date_value = datetime.strptime(normalized[:8], "%Y%m%d")
    except ValueError as exc:
        raise WorkflowInputError("analysis data cutoff is invalid") from exc
    return date_value.replace(hour=15, tzinfo=timezone(timedelta(hours=8)))


def _parse_decision_cutoff(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        cutoff = value
    else:
        try:
            cutoff = datetime.fromisoformat(str(value).strip())
        except ValueError as exc:
            raise WorkflowInputError("--as-of must be an ISO-8601 timestamp") from exc
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise WorkflowInputError("--as-of must be timezone-aware")
    return cutoff


def _base_score_name_and_context(
    meta: Mapping[str, Any], symbol: str
) -> tuple[float | None, str, LocalFundamentalContextV1]:
    bases = meta.get("fundamental_deterministic_bases", {})
    item = bases.get(symbol, {}) if isinstance(bases, Mapping) else {}
    if not isinstance(item, Mapping):
        return None, "", LocalFundamentalContextV1()
    company_name = str(item.get("company_name", "") or "").strip()
    industry_raw = str(item.get("industry", "") or "").strip()
    industry_confirmed = bool(industry_raw) and industry_raw.casefold() not in _UNKNOWN_NAMES
    peer_values = item.get("peer_symbols", [])
    peers = []
    if isinstance(peer_values, list):
        for value in peer_values:
            peer = str(value or "").strip().upper()
            if peer and peer != symbol and peer not in peers:
                peers.append(peer)
    audit = item.get("runtime_audit", {})
    audit = audit if isinstance(audit, Mapping) else {}
    confidence_raw = item.get("base_confidence", audit.get("reliability", 0.0))
    try:
        confidence = float(confidence_raw)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(confidence, 1.0)) if math.isfinite(confidence) else 0.0
    available = [str(value) for value in item.get("available_modules", []) if str(value)]
    missing = [str(value) for value in item.get("missing_modules", []) if str(value)]
    try:
        valuation_price = float(item.get("valuation_price"))
        if not math.isfinite(valuation_price) or valuation_price <= 0.0:
            valuation_price = None
    except (TypeError, ValueError):
        valuation_price = None
    try:
        valuation_price_as_of = date.fromisoformat(str(item.get("valuation_price_as_of", "")))
    except ValueError:
        valuation_price_as_of = None
    if valuation_price is None or valuation_price_as_of is None:
        valuation_price = None
        valuation_price_as_of = None
    declared_peer_status = str(item.get("peer_set_status", "")).strip().lower()
    peer_confirmed = declared_peer_status == "confirmed" and bool(peers)
    context = LocalFundamentalContextV1(
        industry=industry_raw if industry_confirmed else "UNCONFIRMED",
        industry_status="confirmed" if industry_confirmed else "unconfirmed",
        peer_set_status="confirmed" if peer_confirmed else "unconfirmed",
        peer_symbols=peers if peer_confirmed else [],
        base_confidence=confidence,
        available_modules=available,
        missing_modules=missing,
        valuation_price=valuation_price,
        valuation_price_as_of=valuation_price_as_of,
    )
    if str(item.get("status", "")).strip().upper() not in {"SUCCESS", "DEGRADED"}:
        return None, company_name, context
    raw_score = item.get("base_score")
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None, company_name, context
    if not math.isfinite(score) or not -1.0 <= score <= 1.0:
        return None, company_name, context
    expected_hash = compute_base_score_sha256(score)
    if str(item.get("base_score_sha256", "")).strip().lower() != expected_hash:
        return None, company_name, context
    return score, company_name, context


def _relative_artifact(root: Path, path: Path) -> str:
    return path.resolve(strict=False).relative_to(root.resolve(strict=True)).as_posix()


def _artifact_input(root: Path, path: str | Path) -> Path:
    target = Path(path)
    if target.is_absolute():
        return target
    absolute = target.absolute()
    try:
        absolute.relative_to(root.absolute())
        return absolute
    except ValueError:
        return root / target


def _append_job_event(root: Path, event: JobEventV1) -> str:
    ledger = HashChainLedger(root, root / "state" / "jobs.v1.jsonl")
    previous: JobState | None = None
    for record in ledger.read_records():
        payload = dict(record.get("event", {}))
        if str(payload.get("request_id", "")) == event.request_id:
            previous = JobState(str(payload.get("state")))
    validate_job_transition(previous, event.state)
    return ledger.append(event, expected_head=ledger.head())


def prepare_research_requests(
    *,
    market: str,
    as_of: str | datetime,
    analysis_run: str | Path,
    holdings_manifest: str | Path,
    root: str | Path = DEFAULT_ROOT,
    now: datetime | None = None,
    prompt_version: str = PROMPT_VERSION,
    policy_version: str = POLICY_VERSION,
) -> FundamentalResearchPrepareManifestV1:
    """Build private, hash-bound request packets from explicit local manifests."""
    if str(market).upper() != "CN":
        raise WorkflowInputError("fundamental research v1 only supports CN")
    created_at = now or datetime.now(timezone.utc)
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise WorkflowInputError("now must be timezone-aware")
    analysis_path = _resolve_analysis_manifest(analysis_run)
    analysis, analysis_hash = _validate_analysis_manifest(analysis_path)
    analysis_generated_at = datetime.fromisoformat(str(analysis["generated_at"]))
    if created_at.astimezone(timezone.utc) < analysis_generated_at.astimezone(timezone.utc):
        raise WorkflowInputError("request creation cannot precede analysis generation")
    holdings_path = Path(holdings_manifest)
    if not holdings_path.is_file():
        raise WorkflowInputError(f"manual execution manifest not found: {holdings_path}")
    holdings, manual_hash, ledger_hash, _ = _load_manual_holdings(holdings_path)
    meta = dict(analysis["analysis_meta"])
    run_id = str(analysis["run_id"])
    decision_cutoff = _parse_decision_cutoff(as_of)
    data_cutoff = _analysis_data_cutoff(meta)
    if decision_cutoff.astimezone(timezone.utc) != data_cutoff.astimezone(timezone.utc):
        raise WorkflowInputError(
            "--as-of must equal the analysis decision cutoff " f"{data_cutoff.isoformat()}"
        )
    source_policy, source_policy_sha = _source_policy()
    shortlist = meta.get("shortlist", [])
    shortlist = shortlist if isinstance(shortlist, list) else []
    shortlist_by_symbol: dict[str, dict[str, Any]] = {}
    for item in shortlist:
        if not isinstance(item, Mapping):
            continue
        symbol = str(item.get("symbol", "") or "").strip().upper()
        if symbol and symbol not in shortlist_by_symbol:
            shortlist_by_symbol[symbol] = dict(item)
    candidates: list[tuple[str, list[str], str]] = []
    for holding in holdings:
        symbol = holding["symbol"]
        reasons = ["current_holding"]
        if symbol in shortlist_by_symbol:
            reasons.append("analysis_shortlist")
        candidates.append((symbol, reasons, holding["company_name"]))
    seen = {item[0] for item in candidates}
    for symbol, item in shortlist_by_symbol.items():
        if symbol not in seen:
            candidates.append(
                (symbol, ["analysis_shortlist"], str(item.get("company_name", "") or "").strip())
            )
            seen.add(symbol)

    root_path = Path(root)
    run_dir = root_path / "CN" / decision_cutoff.date().isoformat() / run_id
    requests_dir = run_dir / "requests"
    requested: list[PreparedRequestRecordV1] = []
    blockers: list[PrepareBlockerV1] = []
    for symbol, reasons, holding_name in candidates:
        base_score, packet_name, local_context = _base_score_name_and_context(meta, symbol)
        shortlist_name = str(
            shortlist_by_symbol.get(symbol, {}).get("company_name", "") or ""
        ).strip()
        company_name = packet_name or shortlist_name or holding_name
        if company_name.strip().casefold() in _UNKNOWN_NAMES:
            blockers.append(PrepareBlockerV1(symbol=symbol, code="company_name_missing"))
            continue
        if base_score is None:
            blockers.append(
                PrepareBlockerV1(symbol=symbol, code="deterministic_fundamental_base_missing")
            )
            continue
        if local_context.peer_set_status == "unconfirmed":
            blockers.append(
                PrepareBlockerV1(
                    symbol=symbol,
                    code="peer_set_unconfirmed",
                    detail="peer_symbols empty; external worker must not select substitutes",
                )
            )
        if local_context.industry_status == "unconfirmed":
            blockers.append(PrepareBlockerV1(symbol=symbol, code="industry_unconfirmed"))
        if local_context.valuation_price is None:
            blockers.append(PrepareBlockerV1(symbol=symbol, code="valuation_price_unconfirmed"))
        base_records = meta.get("fundamental_deterministic_bases", {})
        base_record = base_records.get(symbol, {}) if isinstance(base_records, Mapping) else {}
        data_generation = _identifier_value(
            base_record.get("data_generation") if isinstance(base_record, Mapping) else ""
        )
        if not data_generation:
            blockers.append(
                PrepareBlockerV1(symbol=symbol, code="fundamental_data_generation_missing")
            )
            continue
        identity = hashlib.sha256(
            canonical_json_bytes(
                {
                    "analysis_manifest_sha256": analysis_hash,
                    "run_id": run_id,
                    "symbol": symbol,
                    "base_score": base_score,
                }
            )
        ).hexdigest()
        request = FundamentalResearchRequestV1(
            request_id=f"fr-{identity[:24]}",
            run_id=run_id,
            symbol=symbol,
            company_name=company_name,
            decision_cutoff=decision_cutoff,
            created_at=created_at,
            expires_at=created_at + timedelta(days=30),
            base_score=base_score,
            base_score_sha256=compute_base_score_sha256(base_score),
            git_sha=str(analysis["git_sha"]),
            data_generation=data_generation,
            selection_reasons=reasons,
            prompt_version=prompt_version,
            policy_version=policy_version,
            source_policy_sha256=source_policy_sha,
            local_context=local_context,
        )
        request_path = requests_dir / f"{symbol}.request.v1.json"
        existing_request: FundamentalResearchRequestV1 | None = None
        if request_path.exists():
            existing_request = load_json_model(
                root_path, request_path, FundamentalResearchRequestV1
            )
            comparable_fields = (
                "request_id",
                "run_id",
                "symbol",
                "company_name",
                "market",
                "decision_cutoff",
                "base_score",
                "base_score_sha256",
                "git_sha",
                "data_generation",
                "selection_reasons",
                "prompt_version",
                "policy_version",
                "source_policy_sha256",
                "budget",
                "local_context",
            )
            if any(
                getattr(existing_request, field) != getattr(request, field)
                for field in comparable_fields
            ):
                raise WorkflowInputError(f"existing request lineage mismatch for {symbol}")
            request = existing_request
            request_sha = model_sha256(request)
        else:
            request_sha = atomic_write_json_model(root_path, request_path, request)
        response_path = run_dir / "responses" / f"{symbol}.response.v1.json"
        task_path = requests_dir / f"{symbol}.task.v1.json"
        task = FundamentalResearchTaskPacketV1(
            request=request,
            request_sha256=request_sha,
            response_path=_relative_artifact(root_path, response_path),
            dimensions=[
                ResearchDimensionSpecV1(
                    dimension=dimension,
                    weight=DIMENSION_WEIGHTS[dimension],
                )
                for dimension in Dimension
            ],
            source_taxonomy={
                "primary": [
                    "exchange_or_regulator",
                    "statutory_filing",
                    "company_ir",
                    "government_or_industry_association",
                    "competitor_statutory_filing",
                ],
                "secondary": [
                    "reliable_broker_research",
                    "industry_database",
                    "major_financial_media",
                ],
                "ineligible": ["social_media", "forum", "self_media", "search_snippet"],
                "primary_hostname_allowlist": sorted(source_policy.primary_hostnames),
            },
            research_constraints=[
                "decision_cutoff is the PIT ceiling; do not use later evidence",
                "budget is 60 minutes, 20 searches, and 25 deduplicated documents",
                "local industry, peer_symbols, and valuation price are authoritative",
                "if peer_set_status is unconfirmed, keep it unknown; never select web substitutes",
                "separate fact, judgment, and unknown and cite every material claim",
                "do not propose score_delta or any portfolio or execution action",
                "return strict FundamentalResearchResponseV1 JSON with no extra fields",
                "company IR is primary only when its hostname is in primary_hostname_allowlist",
            ],
            context_blockers=[
                blocker.code
                for blocker in blockers
                if blocker.symbol == symbol
                and blocker.code
                in {
                    "industry_unconfirmed",
                    "peer_set_unconfirmed",
                    "valuation_price_unconfirmed",
                }
            ],
            allowed_signals=[
                "strong_negative",
                "negative",
                "neutral",
                "positive",
                "strong_positive",
                "unknown",
            ],
            response_json_schema=FundamentalResearchResponseV1.model_json_schema(),
        )
        if task_path.exists():
            existing_task = load_json_model(root_path, task_path, FundamentalResearchTaskPacketV1)
            if existing_task != task:
                raise WorkflowInputError(f"existing task packet lineage mismatch for {symbol}")
            task_sha = model_sha256(existing_task)
        else:
            task_sha = atomic_write_json_model(root_path, task_path, task)
        record = PreparedRequestRecordV1(
            request_id=request.request_id,
            symbol=symbol,
            company_name=company_name,
            selection_reasons=reasons,
            request_path=_relative_artifact(root_path, request_path),
            request_sha256=request_sha,
            task_path=_relative_artifact(root_path, task_path),
            task_sha256=task_sha,
            response_path=_relative_artifact(root_path, response_path),
        )
        requested.append(record)
        if existing_request is None:
            for state in (JobState.PREPARED, JobState.EXPORTED):
                _append_job_event(
                    root_path,
                    JobEventV1(
                        event_id=f"{request.request_id}:{state.value.lower()}",
                        request_id=request.request_id,
                        state=state,
                        occurred_at=created_at,
                        reason=(
                            "request_written"
                            if state == JobState.PREPARED
                            else "external_handoff_ready"
                        ),
                    ),
                )
    prepare_manifest = FundamentalResearchPrepareManifestV1(
        run_id=run_id,
        created_at=created_at,
        analysis_manifest_sha256=analysis_hash,
        manual_manifest_sha256=manual_hash,
        manual_ledger_sha256=ledger_hash,
        requested=requested,
        blockers=blockers,
    )
    manifest_path = run_dir / PREPARE_MANIFEST_NAME
    if manifest_path.exists():
        existing_manifest = load_json_model(
            root_path, manifest_path, FundamentalResearchPrepareManifestV1
        )
        current_payload = prepare_manifest.model_dump(mode="json")
        existing_payload = existing_manifest.model_dump(mode="json")
        current_payload.pop("created_at", None)
        existing_payload.pop("created_at", None)
        if current_payload != existing_payload:
            raise WorkflowInputError("existing prepare manifest lineage mismatch")
        return existing_manifest
    atomic_write_json_model(root_path, manifest_path, prepare_manifest)
    return prepare_manifest


def _run_dir_from_request(root: Path, request_path: Path) -> Path:
    resolved_root = root.resolve(strict=True)
    resolved_request = request_path.resolve(strict=True)
    try:
        relative = resolved_request.relative_to(resolved_root)
    except ValueError as exc:
        raise WorkflowInputError("request path escapes configured root") from exc
    if len(relative.parts) < 5 or relative.parts[-2] != "requests":
        raise WorkflowInputError("request path is not in a canonical run requests directory")
    return resolved_request.parent.parent


def _validate_prepared_request_binding(
    *,
    root: Path,
    run_dir: Path,
    request_path: Path,
    request: FundamentalResearchRequestV1,
) -> PreparedRequestRecordV1:
    manifest_path = run_dir / PREPARE_MANIFEST_NAME
    try:
        manifest = load_json_model(root, manifest_path, FundamentalResearchPrepareManifestV1)
    except Exception as exc:
        raise WorkflowInputError("prepare manifest is missing or invalid") from exc
    if manifest.market != "CN" or manifest.run_id != request.run_id:
        raise WorkflowInputError("prepare manifest run lineage mismatch")
    matching = [
        item
        for item in manifest.requested
        if item.request_id == request.request_id and item.symbol == request.symbol
    ]
    if len(matching) != 1:
        raise WorkflowInputError("request is not registered in prepare manifest")
    record = matching[0]
    if record.request_path != _relative_artifact(
        root, request_path
    ) or record.request_sha256 != model_sha256(request):
        raise WorkflowInputError("request does not match prepare manifest binding")
    task_path = root / record.task_path
    try:
        task = load_json_model(root, task_path, FundamentalResearchTaskPacketV1)
    except Exception as exc:
        raise WorkflowInputError("prepare manifest task binding is invalid") from exc
    if model_sha256(task) != record.task_sha256 or task.request_sha256 != record.request_sha256:
        raise WorkflowInputError("prepare manifest task hash mismatch")
    return record


def import_research_response(
    *,
    request_path: str | Path,
    response_path: str | Path,
    root: str | Path = DEFAULT_ROOT,
    validate_only: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate an untrusted response and optionally persist dossier/overlay/report."""
    imported_at = now or datetime.now(timezone.utc)
    if imported_at.tzinfo is None or imported_at.utcoffset() is None:
        raise WorkflowInputError("now must be timezone-aware")
    root_path = Path(root)
    request_target = _artifact_input(root_path, request_path)
    response_target = _artifact_input(root_path, response_path)
    request = load_json_model(root_path, request_target, FundamentalResearchRequestV1)
    source_policy, _ = _source_policy()
    run_dir = _run_dir_from_request(root_path, request_target)
    _validate_prepared_request_binding(
        root=root_path,
        run_dir=run_dir,
        request_path=request_target,
        request=request,
    )
    request_sha = model_sha256(request)
    response_sha: str | None = None
    if validate_only:
        response = load_json_model(root_path, response_target, FundamentalResearchResponseV1)
        response_sha = model_sha256(response)
        overlay = validate_response(
            request,
            response,
            imported_at=imported_at,
            source_policy=source_policy,
        )
        return {
            "status": "VALIDATED",
            "validate_only": True,
            "request_id": request.request_id,
            "symbol": request.symbol,
            "response_sha256": response_sha,
            "overlay": overlay.model_dump(mode="json"),
        }

    _append_job_event(
        root_path,
        JobEventV1(
            event_id=f"{request.request_id}:received:{int(imported_at.timestamp() * 1_000_000)}",
            request_id=request.request_id,
            state=JobState.RECEIVED,
            occurred_at=imported_at,
            reason="response_import_started",
        ),
    )
    report_path = run_dir / "import_reports" / f"{request.request_id}.import-report.v1.json"
    try:
        response = load_json_model(root_path, response_target, FundamentalResearchResponseV1)
        response_sha = model_sha256(response)
        overlay = validate_response(
            request,
            response,
            imported_at=imported_at,
            source_policy=source_policy,
        )
        canonical_response_path = run_dir / "responses" / f"{request.symbol}.response.v1.json"
        dossier_path = run_dir / "dossiers" / f"{request.symbol}.dossier.v1.json"
        overlay_path = run_dir / "overlays" / f"{request.symbol}.overlay.v1.json"
        response_sha = atomic_write_json_model(root_path, canonical_response_path, response)
        dossier_sha = atomic_write_json_model(root_path, dossier_path, response.dossier)
        overlay_sha = atomic_write_json_model(root_path, overlay_path, overlay)
        report = ImportReportV1(
            request_id=request.request_id,
            symbol=request.symbol,
            imported_at=imported_at,
            status="VALIDATED",
            request_sha256=request_sha,
            response_sha256=response_sha,
            dossier_path=_relative_artifact(root_path, dossier_path),
            overlay_path=_relative_artifact(root_path, overlay_path),
        )
        atomic_write_json_model(root_path, report_path, report)
        _append_job_event(
            root_path,
            JobEventV1(
                event_id=f"{request.request_id}:validated:{int(imported_at.timestamp() * 1_000_000)}",
                request_id=request.request_id,
                state=JobState.VALIDATED,
                occurred_at=imported_at,
                reason="response_schema_binding_and_evidence_validated",
                request_sha256=request_sha,
                response_sha256=response_sha,
                dossier_sha256=dossier_sha,
                overlay_sha256=overlay_sha,
            ),
        )
        return {
            "status": "VALIDATED",
            "validate_only": False,
            "request_id": request.request_id,
            "symbol": request.symbol,
            "response_sha256": response_sha,
            "dossier_path": report.dossier_path,
            "overlay_path": report.overlay_path,
            "import_report_path": _relative_artifact(root_path, report_path),
            "overlay": overlay.model_dump(mode="json"),
        }
    except Exception as exc:
        report = ImportReportV1(
            request_id=request.request_id,
            symbol=request.symbol,
            imported_at=imported_at,
            status="REJECTED",
            error=str(exc)[:2000],
            request_sha256=request_sha,
            response_sha256=response_sha,
        )
        atomic_write_json_model(root_path, report_path, report)
        _append_job_event(
            root_path,
            JobEventV1(
                event_id=f"{request.request_id}:rejected:{int(imported_at.timestamp() * 1_000_000)}",
                request_id=request.request_id,
                state=JobState.REJECTED,
                occurred_at=imported_at,
                reason="response_validation_failed",
            ),
        )
        raise WorkflowInputError(f"response rejected: {exc}") from exc


def research_status(
    *,
    root: str | Path = DEFAULT_ROOT,
    market: str = "CN",
    run_id: str = "",
    symbol: str = "",
    state: str = "",
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return filtered, read-only latest job status without mutating artifacts."""
    if str(market).upper() != "CN":
        raise WorkflowInputError("fundamental research v1 only supports CN")
    root_path = Path(root)
    if not root_path.exists():
        return {"market": "CN", "count": 0, "jobs": []}
    ledger = HashChainLedger(root_path, root_path / "state" / "jobs.v1.jsonl")
    records = ledger.read_records()
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        event = dict(record.get("event", {}))
        request_id = str(event.get("request_id", ""))
        if request_id:
            latest[request_id] = event
    observed_at = now or datetime.now(timezone.utc)
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise WorkflowInputError("now must be timezone-aware")
    request_index: dict[str, dict[str, Any]] = {}
    market_root = root_path / "CN"
    if market_root.exists():
        for request_file in market_root.glob("*/*/requests/*.request.v1.json"):
            try:
                request = load_json_model(root_path, request_file, FundamentalResearchRequestV1)
            except Exception:
                continue
            request_index[request.request_id] = {
                "run_id": request.run_id,
                "symbol": request.symbol,
                "company_name": request.company_name,
                "expires_at": request.expires_at.isoformat(),
                "request_file": request_file,
            }
    wanted_symbol = symbol.strip().upper()
    wanted_state = state.strip().upper()
    jobs: list[dict[str, Any]] = []
    for request_id, event in latest.items():
        identity = request_index.get(request_id, {})
        request_file = identity.get("request_file")
        public_identity = {key: value for key, value in identity.items() if key != "request_file"}
        ledger_state = str(event.get("state", ""))
        derived_state = ledger_state
        expires_text = str(identity.get("expires_at", ""))
        if expires_text and ledger_state in {
            JobState.PREPARED.value,
            JobState.EXPORTED.value,
            JobState.RECEIVED.value,
        }:
            expires_at = datetime.fromisoformat(expires_text)
            if observed_at.astimezone(timezone.utc) > expires_at.astimezone(timezone.utc):
                derived_state = JobState.EXPIRED.value
        row = {
            **public_identity,
            **event,
            "ledger_state": ledger_state,
            "derived_state": derived_state,
            "import_status": "",
            "import_blocker": "",
            "overlay_eligible": None,
            "overlay_delta": None,
            "overlay_blockers": [],
        }
        if isinstance(request_file, Path):
            run_dir = request_file.parent.parent
            report_path = run_dir / "import_reports" / f"{request_id}.import-report.v1.json"
            overlay_path = run_dir / "overlays" / f"{identity.get('symbol', '')}.overlay.v1.json"
            if report_path.exists():
                try:
                    report = load_json_model(root_path, report_path, ImportReportV1)
                    row["import_status"] = report.status
                    if report.status == "REJECTED":
                        row["import_blocker"] = "response_rejected"
                except Exception:
                    row["import_status"] = "INVALID_REPORT"
                    row["import_blocker"] = "import_report_invalid"
            if overlay_path.exists():
                try:
                    overlay = load_json_model(root_path, overlay_path, FundamentalOverlayV1)
                    row["overlay_eligible"] = overlay.eligible
                    row["overlay_delta"] = overlay.computed_delta
                    row["overlay_blockers"] = list(overlay.blockers)
                except Exception:
                    row["import_blocker"] = "overlay_invalid"
        if run_id and identity.get("run_id") != run_id:
            continue
        if wanted_symbol and identity.get("symbol", "").upper() != wanted_symbol:
            continue
        if wanted_state and derived_state.upper() != wanted_state:
            continue
        jobs.append(row)
    jobs.sort(
        key=lambda item: (str(item.get("occurred_at", "")), str(item.get("request_id", ""))),
        reverse=True,
    )
    gate_status: dict[str, Any] = {
        "available": False,
        "recomputed": False,
        "eligible_modes": ["shadow"],
    }
    evidence_files = sorted((root_path / "state").glob("activation-evidence-*.v2.json"))
    if evidence_files:
        try:
            evidence = load_json_model(root_path, evidence_files[-1], ActivationGateEvidenceV2)
            recompute_blockers = verify_recomputed_evidence(root=root_path, evidence=evidence)
            current_evidence = build_activation_gate_evidence(
                root=root_path,
                holdings_snapshot_path=root_path / evidence.holdings_snapshot_path,
                generated_at=observed_at,
            )
            gate_status = {
                "available": True,
                "generated_at": evidence.generated_at.isoformat(),
                "evaluated_at": observed_at.isoformat(),
                "recomputed": not recompute_blockers,
                "recompute_blockers": recompute_blockers,
                **activation_readiness(current_evidence),
            }
            if recompute_blockers:
                gate_status["eligible_modes"] = ["shadow"]
        except Exception as exc:
            gate_status = {
                "available": True,
                "recomputed": False,
                "eligible_modes": ["shadow"],
                "recompute_blockers": [f"gate_evidence_invalid:{type(exc).__name__}"],
            }
    return {
        "market": "CN",
        "count": len(jobs),
        "jobs": jobs,
        "activation_gate": gate_status,
    }


def generate_activation_gate_evidence(
    *,
    holdings_manifest: str | Path,
    root: str | Path = DEFAULT_ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write a readback-verified gate snapshot recomputed from private ledgers."""

    generated_at = now or datetime.now(timezone.utc)
    if generated_at.tzinfo is None or generated_at.utcoffset() is None:
        raise WorkflowInputError("now must be timezone-aware")
    root_path = Path(root)
    holdings_path = Path(holdings_manifest)
    holdings, manual_manifest_sha, manual_ledger_sha, ledger_path = _load_manual_holdings(
        holdings_path
    )
    try:
        manifest_repo_path = holdings_path.resolve(strict=True).relative_to(
            _governance.REPO_ROOT.resolve(strict=True)
        )
        ledger_repo_path = ledger_path.resolve(strict=True).relative_to(
            _governance.REPO_ROOT.resolve(strict=True)
        )
    except ValueError as exc:
        raise WorkflowInputError(
            "holdings manifest and ledger must be inside the repository"
        ) from exc
    stamp = generated_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    holdings_snapshot_path = root_path / "state" / f"holdings-scope-{stamp}.v1.json"
    holdings_snapshot = HoldingsScopeSnapshotV1(
        generated_at=generated_at,
        symbols=sorted({item["symbol"] for item in holdings}),
        manual_manifest_repo_path=manifest_repo_path.as_posix(),
        manual_ledger_repo_path=ledger_repo_path.as_posix(),
        manual_manifest_sha256=manual_manifest_sha,
        manual_ledger_sha256=manual_ledger_sha,
    )
    holdings_snapshot_sha = atomic_write_json_model(
        root_path, holdings_snapshot_path, holdings_snapshot
    )
    evidence = build_activation_gate_evidence(
        root=root_path,
        holdings_snapshot_path=holdings_snapshot_path,
        generated_at=generated_at,
    )
    target = root_path / "state" / f"activation-evidence-{stamp}.v2.json"
    if target.exists() or target.is_symlink():
        raise WorkflowInputError("activation evidence target already exists")
    digest = atomic_write_json_model(root_path, target, evidence)
    readiness = activation_readiness(evidence)
    return {
        "schema_version": evidence.schema_version,
        "generated_at": evidence.generated_at.isoformat(),
        "evidence_path": _relative_artifact(root_path, target),
        "evidence_sha256": digest,
        "manual_manifest_sha256": manual_manifest_sha,
        "manual_ledger_sha256": manual_ledger_sha,
        "holdings_snapshot_path": _relative_artifact(root_path, holdings_snapshot_path),
        "holdings_snapshot_sha256": holdings_snapshot_sha,
        "validated_dossiers": len(evidence.validated_request_ids),
        "distinct_companies": len(evidence.validated_company_names),
        "distinct_industries": len(evidence.validated_industries),
        "shadow_trading_days": len(evidence.shadow_trading_dates),
        "limited_trading_days": len(evidence.limited_trading_dates),
        "target_weight_counterfactual_days": len(evidence.target_weight_counterfactual_dates),
        "nav_attribution_days": len(evidence.nav_attribution_dates),
        "holdings_coverage_passed": evidence.holdings_coverage_passed,
        "recent_validation_success_rate": evidence.recent_validation_success_rate,
        "critical_error_codes": list(evidence.critical_error_codes),
        **readiness,
    }


__all__ = [
    "DEFAULT_ROOT",
    "FundamentalResearchPrepareManifestV1",
    "ImportReportV1",
    "WorkflowInputError",
    "import_research_response",
    "generate_activation_gate_evidence",
    "prepare_research_requests",
    "research_status",
]
