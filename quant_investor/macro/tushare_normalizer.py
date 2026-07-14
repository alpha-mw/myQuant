"""Compile offline Tushare macro rows against independently captured evidence.

Date-only ``cn_schedule`` rows are diagnostic evidence only.  They never imply
an availability timestamp and therefore can never make an observation
promotable.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlparse
from zoneinfo import ZoneInfo

from quant_investor.macro.contracts import (
    MacroObservation,
    canonical_hash,
    parse_timestamp,
)
from quant_investor.macro.registry import NATIONAL_INDICATORS
from quant_investor.macro.store import publish_observations

RAW_BUNDLE_SCHEMA = "macro-tushare-raw-bundle.v1"
BACKFILL_PLAN_SCHEMA = "macro-backfill-plan.v1"
EVIDENCE_CAPTURE_SCHEMA = "macro-release-evidence-capture.v1"
NORMALIZATION_SCHEMA = "macro-tushare-normalization.v1"
AVAILABILITY_POLICY = "verified_capture_timestamp_evidence_required"
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_KEY_FRAGMENTS = frozenset(
    {
        "apikey",
        "accesskey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "secret",
        "token",
    }
)
_SHANGHAI = ZoneInfo("Asia/Shanghai")


class MacroNormalizationError(RuntimeError):
    """Raised when an input or normalization artifact is unsafe."""


@dataclass(frozen=True)
class TushareIndicatorSpec:
    endpoint: str
    indicator_id: str
    period_field: str
    value_field: str
    frequency: str
    unit: str
    max_release_lag_days: int
    documentation_url: str
    evidence_source_system: str
    evidence_domains: tuple[str, ...]
    value_aliases: tuple[str, ...] = ()

    @property
    def accepted_value_fields(self) -> tuple[str, ...]:
        return (self.value_field, *self.value_aliases)


_NBS = ("stats.gov.cn", "www.stats.gov.cn")
_PBOC = ("pbc.gov.cn", "www.pbc.gov.cn")
SPECS: tuple[TushareIndicatorSpec, ...] = (
    TushareIndicatorSpec(
        "cn_gdp",
        "cn.gdp_yoy",
        "quarter",
        "gdp_yoy",
        "quarterly",
        "%",
        60,
        "https://tushare.pro/document/2?doc_id=227",
        "nbs_official",
        _NBS,
    ),
    TushareIndicatorSpec(
        "cn_cpi",
        "cn.cpi_yoy",
        "month",
        "nt_yoy",
        "monthly",
        "%",
        45,
        "https://tushare.pro/document/2?doc_id=228",
        "nbs_official",
        _NBS,
    ),
    TushareIndicatorSpec(
        "cn_ppi",
        "cn.ppi_yoy",
        "month",
        "ppi_yoy",
        "monthly",
        "%",
        45,
        "https://tushare.pro/document/2?doc_id=245",
        "nbs_official",
        _NBS,
    ),
    TushareIndicatorSpec(
        "cn_m",
        "cn.m1_yoy",
        "month",
        "m1_yoy",
        "monthly",
        "%",
        45,
        "https://tushare.pro/document/2?doc_id=242",
        "pboc_official",
        _PBOC,
    ),
    TushareIndicatorSpec(
        "cn_m",
        "cn.m2_yoy",
        "month",
        "m2_yoy",
        "monthly",
        "%",
        45,
        "https://tushare.pro/document/2?doc_id=242",
        "pboc_official",
        _PBOC,
    ),
    TushareIndicatorSpec(
        "sf_month",
        "cn.social_financing_flow",
        "month",
        "inc_month",
        "monthly",
        "CNY_100M",
        45,
        "https://tushare.pro/document/2?doc_id=310",
        "pboc_official",
        _PBOC,
    ),
    TushareIndicatorSpec(
        "cn_pmi",
        "cn.pmi_manufacturing",
        "month",
        "pmi010000",
        "monthly",
        "index",
        15,
        "https://tushare.pro/document/2?doc_id=325",
        "nbs_official",
        _NBS,
        ("PMI010000",),
    ),
)
_SPEC_BY_ID = {spec.indicator_id: spec for spec in SPECS}


@dataclass(frozen=True)
class MacroNormalizationResult:
    observations: tuple[MacroObservation, ...]
    quarantine: tuple[Mapping[str, Any], ...]
    receipts: tuple[Mapping[str, Any], ...]
    manifest: Mapping[str, Any]


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _assert_hash(value: str, error: str) -> str:
    normalized = str(value or "").lower()
    if not _SHA256.fullmatch(normalized):
        raise MacroNormalizationError(error)
    return normalized


def _sensitive_key(value: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", str(value).strip().lower())
    return any(fragment in normalized for fragment in _SECRET_KEY_FRAGMENTS)


def _assert_no_secret_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if _sensitive_key(key):
                raise MacroNormalizationError(
                    "macro_input_secret_key_rejected"
                )
            _assert_no_secret_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_secret_keys(child)
    elif isinstance(value, str) and value.strip().lower().startswith(
        ("http://", "https://")
    ):
        parsed = urlparse(value.strip())
        if parsed.username is not None or parsed.password is not None:
            raise MacroNormalizationError("macro_input_url_userinfo_rejected")
        if any(_sensitive_key(key) for key, _ in parse_qsl(parsed.query)):
            raise MacroNormalizationError(
                "macro_input_url_secret_query_rejected"
            )


def _month_end(value: str) -> date:
    text = str(value or "").strip()
    if not re.fullmatch(r"\d{6}", text):
        raise ValueError("period_month_invalid")
    year, month = int(text[:4]), int(text[4:])
    return date(year, month, calendar.monthrange(year, month)[1])


def _quarter_end(value: str) -> date:
    match = re.fullmatch(r"(\d{4})Q([1-4])", str(value or "").strip().upper())
    if match is None:
        raise ValueError("period_quarter_invalid")
    year, month = int(match.group(1)), int(match.group(2)) * 3
    return date(year, month, calendar.monthrange(year, month)[1])


def _period_end(spec: TushareIndicatorSpec, row: Mapping[str, Any]) -> date:
    raw = str(row.get(spec.period_field) or "")
    return (
        _quarter_end(raw) if spec.frequency == "quarterly" else _month_end(raw)
    )


def _value(spec: TushareIndicatorSpec, row: Mapping[str, Any]) -> float:
    values: list[float] = []
    for field in spec.accepted_value_fields:
        if field in row and row.get(field) is not None:
            value = float(row[field])
            if not math.isfinite(value):
                raise ValueError("value_non_finite")
            values.append(value)
    if not values:
        raise ValueError("value_field_missing")
    if len(set(values)) != 1:
        raise ValueError("value_field_alias_conflict")
    return values[0]


def _schedule_has_date_only_match(
    rows: Sequence[Mapping[str, Any]],
    spec: TushareIndicatorSpec,
    period_end: date,
) -> bool:
    limit = period_end + timedelta(days=spec.max_release_lag_days)
    for row in rows:
        if str(row.get("data_api") or "").strip() != spec.endpoint:
            continue
        try:
            published = datetime.strptime(
                str(row.get("publish_date") or ""), "%Y%m%d"
            ).date()
        except ValueError:
            continue
        if period_end <= published <= limit:
            return True
    return False


def _scope_from_plan(plan: Mapping[str, Any]) -> set[tuple[str, str]]:
    if (
        plan.get("schema_version") != BACKFILL_PLAN_SCHEMA
        or plan.get("market") != "CN"
    ):
        raise MacroNormalizationError("macro_backfill_plan_schema_invalid")
    payload = plan.get("requested_scope")
    if (
        not isinstance(payload, list)
        or not payload
        or not all(isinstance(x, Mapping) for x in payload)
    ):
        raise MacroNormalizationError("macro_backfill_plan_scope_invalid")
    scope: set[tuple[str, str]] = set()
    for item in payload:
        indicator_id = str(item.get("indicator_id") or "").strip()
        if indicator_id not in _SPEC_BY_ID:
            raise MacroNormalizationError(
                "macro_backfill_plan_scope_unsupported"
            )
        try:
            period_end = date.fromisoformat(
                str(item.get("period_end") or "")
            ).isoformat()
        except ValueError as exc:
            raise MacroNormalizationError(
                "macro_backfill_plan_period_invalid"
            ) from exc
        scope.add((indicator_id, period_end))
    if len(scope) != len(payload):
        raise MacroNormalizationError("macro_backfill_plan_scope_duplicate")
    return scope


def _evidence_records(capture: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if capture.get("schema_version") != EVIDENCE_CAPTURE_SCHEMA:
        raise MacroNormalizationError("macro_evidence_capture_schema_invalid")
    try:
        parse_timestamp(capture.get("captured_at"), field_name="captured_at")
    except ValueError as exc:
        raise MacroNormalizationError(str(exc)) from exc
    records = capture.get("records")
    if not isinstance(records, list) or not all(
        isinstance(row, Mapping) for row in records
    ):
        raise MacroNormalizationError("macro_evidence_capture_records_invalid")
    return [dict(row) for row in records]


def _availability_evidence(
    records: Sequence[Mapping[str, Any]],
    spec: TushareIndicatorSpec,
    period_end: date,
) -> Mapping[str, Any]:
    matches = [
        dict(row)
        for row in records
        if str(row.get("indicator_id") or "").strip() == spec.indicator_id
        and str(row.get("period_end") or "").strip() == period_end.isoformat()
    ]
    by_hash = {canonical_hash(row): row for row in matches}
    if not by_hash:
        raise ValueError("availability_evidence_missing")
    if len(by_hash) != 1:
        raise ValueError("availability_evidence_conflict")
    evidence = next(iter(by_hash.values()))
    if (
        str(evidence.get("time_precision") or "").strip().lower()
        != "timestamp"
    ):
        raise ValueError("availability_evidence_timestamp_required")
    if (
        str(evidence.get("evidence_source_system") or "").strip()
        != spec.evidence_source_system
    ):
        raise ValueError("availability_evidence_issuer_mismatch")
    record_id = str(evidence.get("evidence_record_id") or "").strip()
    if not record_id:
        raise ValueError("availability_evidence_record_id_missing")
    url = str(evidence.get("evidence_url") or "").strip()
    parsed = urlparse(url)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("availability_evidence_url_userinfo_rejected")
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() not in spec.evidence_domains
    ):
        raise ValueError("availability_evidence_domain_mismatch")
    return evidence


def _quarantine(
    spec: TushareIndicatorSpec,
    row_index: int,
    row: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        "endpoint": spec.endpoint,
        "indicator_id": spec.indicator_id,
        "row_index": row_index,
        "raw_row_hash": canonical_hash(dict(row)),
        "reason": reason,
        "promotable": False,
    }


def normalize_tushare_bundle(
    payload: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    evidence_capture: Mapping[str, Any],
    raw_bundle_file_sha256: str,
    plan_file_sha256: str,
    evidence_capture_sha256: str,
) -> MacroNormalizationResult:
    """Normalize inputs; publishing always re-verifies their bytes."""

    raw_hash = _assert_hash(
        raw_bundle_file_sha256, "macro_raw_bundle_hash_invalid"
    )
    plan_hash = _assert_hash(
        plan_file_sha256, "macro_backfill_plan_hash_invalid"
    )
    capture_hash = _assert_hash(
        evidence_capture_sha256, "macro_evidence_capture_hash_invalid"
    )
    for item in (payload, plan, evidence_capture):
        _assert_no_secret_keys(item)
    if (
        payload.get("schema_version") != RAW_BUNDLE_SCHEMA
        or str(payload.get("provider") or "").lower() != "tushare"
    ):
        raise MacroNormalizationError("macro_raw_bundle_schema_invalid")
    try:
        fetched_at = parse_timestamp(
            payload.get("fetched_at"), field_name="fetched_at"
        )
        captured_at = parse_timestamp(
            evidence_capture.get("captured_at"), field_name="captured_at"
        )
    except ValueError as exc:
        raise MacroNormalizationError(str(exc)) from exc
    tables = payload.get("tables")
    if not isinstance(tables, Mapping):
        raise MacroNormalizationError("macro_raw_bundle_tables_invalid")
    for table_name, rows in tables.items():
        if (
            not isinstance(table_name, str)
            or not isinstance(rows, list)
            or not all(isinstance(row, Mapping) for row in rows)
        ):
            raise MacroNormalizationError("macro_raw_bundle_table_invalid")
    expected_scope = _scope_from_plan(plan)
    evidence_rows = _evidence_records(evidence_capture)
    schedules = [dict(row) for row in tables.get("cn_schedule", [])]
    accepted: list[
        tuple[
            MacroObservation,
            Mapping[str, Any],
            TushareIndicatorSpec,
            int,
            Mapping[str, Any],
        ]
    ] = []
    quarantine: list[Mapping[str, Any]] = []
    receipts: list[Mapping[str, Any]] = []
    for spec in SPECS:
        for row_index, raw_row in enumerate(tables.get(spec.endpoint, [])):
            row = dict(raw_row)
            try:
                period_end = _period_end(spec, row)
                value = _value(spec, row)
                try:
                    evidence = _availability_evidence(
                        evidence_rows, spec, period_end
                    )
                except ValueError as exc:
                    missing = str(exc) == "availability_evidence_missing"
                    has_date = _schedule_has_date_only_match(
                        schedules, spec, period_end
                    )
                    if missing and has_date:
                        raise ValueError(
                            "cn_schedule_date_only_not_promotable"
                        ) from exc
                    raise
                release_at = parse_timestamp(
                    evidence.get("release_at"), field_name="release_at"
                )
                available_at = parse_timestamp(
                    evidence.get("available_at"), field_name="available_at"
                )
                if release_at.astimezone(_SHANGHAI).date() < period_end:
                    raise ValueError("release_before_period_end")
                if release_at > available_at:
                    raise ValueError("release_at_after_available_at")
                if available_at > fetched_at:
                    raise ValueError("availability_after_raw_fetch")
                if available_at > captured_at:
                    raise ValueError("availability_after_evidence_capture")
                row_hash = canonical_hash(row)
                evidence_record_hash = canonical_hash(dict(evidence))
                observation = MacroObservation.from_mapping(
                    {
                        "indicator_id": spec.indicator_id,
                        "dimension_type": "national",
                        "industry_chain": "",
                        "period_end": period_end.isoformat(),
                        "release_at": release_at.isoformat(),
                        "available_at": available_at.isoformat(),
                        "vintage_id": f"observed:{row_hash[:16]}",
                        "value": value,
                        "unit": spec.unit,
                        "frequency": spec.frequency,
                        "source_system": "tushare_fallback",
                        "source_record_id": (
                            f"{spec.endpoint}:{row.get(spec.period_field)}:"
                            f"{row_hash[:16]}"
                        ),
                        "source_url": spec.documentation_url,
                        "fetched_at": fetched_at.isoformat(),
                        "quality_status": "pass",
                    }
                )
                receipt = {
                    "status": "accepted",
                    "endpoint": spec.endpoint,
                    "indicator_id": spec.indicator_id,
                    "period_end": period_end.isoformat(),
                    "raw_record_hash": row_hash,
                    "evidence_capture_sha256": capture_hash,
                    "evidence_record_hash": evidence_record_hash,
                    "evidence_source_system": spec.evidence_source_system,
                    "evidence_record_id": str(evidence["evidence_record_id"]),
                    "observation_content_hash": observation.content_hash,
                }
                accepted.append((observation, receipt, spec, row_index, row))
            except (TypeError, ValueError) as exc:
                rejected = _quarantine(spec, row_index, row, str(exc))
                quarantine.append(rejected)
                receipts.append({**rejected, "status": "quarantine"})

    by_scope: dict[
        tuple[str, str],
        list[
            tuple[
                MacroObservation,
                Mapping[str, Any],
                TushareIndicatorSpec,
                int,
                Mapping[str, Any],
            ]
        ],
    ] = {}
    for entry in accepted:
        by_scope.setdefault(
            (entry[0].indicator_id, entry[0].period_end), []
        ).append(entry)
    conflicting_scope: list[tuple[str, str]] = []
    observations: list[MacroObservation] = []
    for scope, entries in sorted(by_scope.items()):
        distinct = {entry[0].content_hash for entry in entries}
        if len(distinct) > 1:
            conflicting_scope.append(scope)
            for conflict in entries:
                _, _, conflict_spec, row_index, conflict_row = conflict
                rejected = _quarantine(
                    conflict_spec,
                    row_index,
                    conflict_row,
                    "conflicting_values_without_revision_evidence",
                )
                quarantine.append(rejected)
                receipts.append({**rejected, "status": "quarantine"})
        else:
            observation, accepted_receipt, _, _, _ = sorted(
                entries, key=lambda x: x[0].content_hash
            )[0]
            observations.append(observation)
            receipts.append(accepted_receipt)

    normalized = tuple(sorted(observations, key=lambda x: x.content_hash))
    quarantine_sorted = tuple(
        sorted(
            quarantine,
            key=lambda x: (
                str(x["indicator_id"]),
                int(x["row_index"]),
                str(x["raw_row_hash"]),
            ),
        )
    )
    receipts_sorted = tuple(
        sorted(
            receipts,
            key=lambda x: (
                str(x.get("status")),
                str(x.get("indicator_id")),
                str(x.get("period_end", "")),
                str(x.get("raw_record_hash", x.get("raw_row_hash", ""))),
            ),
        )
    )
    accepted_scope = {
        (item.indicator_id, item.period_end) for item in normalized
    }
    missing_scope = sorted(expected_scope - accepted_scope)
    unexpected_scope = sorted(accepted_scope - expected_scope)
    status = (
        "OK"
        if normalized
        and not quarantine_sorted
        and not missing_scope
        and not unexpected_scope
        and not conflicting_scope
        else ("degraded" if normalized else "blocked")
    )
    supported_ids = sorted(_SPEC_BY_ID)
    registry_ids = sorted(item.indicator_id for item in NATIONAL_INDICATORS)
    scope_rows = [
        {"indicator_id": x, "period_end": y} for x, y in sorted(expected_scope)
    ]
    manifest = {
        "schema_version": NORMALIZATION_SCHEMA,
        "status": status,
        "raw_bundle_file_sha256": raw_hash,
        "backfill_plan_sha256": plan_hash,
        "evidence_capture_sha256": capture_hash,
        "fetched_at": fetched_at.isoformat(),
        "availability_policy": AVAILABILITY_POLICY,
        "supported_indicator_ids": supported_ids,
        "unsupported_indicator_ids": sorted(
            set(registry_ids) - set(supported_ids)
        ),
        "national_registry_coverage": len(supported_ids) / len(registry_ids),
        "observation_count": len(normalized),
        "quarantine_count": len(quarantine_sorted),
        "receipt_count": len(receipts_sorted),
        "expected_scope": scope_rows,
        "expected_scope_hash": canonical_hash({"scope": scope_rows}),
        "missing_scope": [
            {"indicator_id": x, "period_end": y} for x, y in missing_scope
        ],
        "unexpected_scope": [
            {"indicator_id": x, "period_end": y} for x, y in unexpected_scope
        ],
        "conflicting_scope": [
            {"indicator_id": x, "period_end": y} for x, y in conflicting_scope
        ],
        "receipt_set_hash": canonical_hash(
            {"receipts": [dict(x) for x in receipts_sorted]}
        ),
        "observer_only": True,
        "production_eligible": False,
        "applied": False,
    }
    return MacroNormalizationResult(
        normalized, quarantine_sorted, receipts_sorted, manifest
    )


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unsafe_path(path: Path) -> bool:
    current = path.absolute()
    while True:
        if current.exists() and current.is_symlink():
            return True
        if current.parent == current:
            return False
        current = current.parent


def persist_tushare_normalization(
    result: MacroNormalizationResult,
    *,
    input_artifacts: Mapping[str, bytes],
    output_root: str | Path = "results/macro_normalization",
    run_id: str,
) -> dict[str, str]:
    if not _SAFE_RUN_ID.fullmatch(run_id) or run_id in {".", ".."}:
        raise MacroNormalizationError("macro_normalization_run_id_unsafe")
    required_inputs = {
        "raw_bundle.json",
        "backfill_plan.json",
        "evidence_capture.json",
    }
    if set(input_artifacts) != required_inputs:
        raise MacroNormalizationError(
            "macro_normalization_input_artifacts_invalid"
        )
    root = Path(output_root).expanduser()
    if _unsafe_path(root):
        raise MacroNormalizationError(
            "macro_normalization_root_symlink_rejected"
        )
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    market_root = root / "CN"
    if _unsafe_path(market_root):
        raise MacroNormalizationError(
            "macro_normalization_market_root_symlink_rejected"
        )
    market_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    final = market_root / run_id
    if final.exists():
        raise MacroNormalizationError("macro_normalization_output_exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=market_root))
    os.chmod(staging, 0o700)
    try:
        serialized = {
            "observations.jsonl": "".join(
                json.dumps(x.to_dict(), ensure_ascii=False, sort_keys=True)
                + "\n"
                for x in result.observations
            ).encode(),
            "quarantine.jsonl": "".join(
                json.dumps(dict(x), ensure_ascii=False, sort_keys=True) + "\n"
                for x in result.quarantine
            ).encode(),
            "normalization_receipts.jsonl": "".join(
                json.dumps(dict(x), ensure_ascii=False, sort_keys=True) + "\n"
                for x in result.receipts
            ).encode(),
            **dict(input_artifacts),
        }
        for name, payload in serialized.items():
            _atomic_bytes(staging / name, payload)
        artifact_hashes = {
            name: _sha256((staging / name).read_bytes())
            for name in sorted(serialized)
        }
        persisted_manifest = {
            **dict(result.manifest),
            "artifact_sha256": artifact_hashes,
        }
        _atomic_bytes(
            staging / "normalization_manifest.json",
            json.dumps(
                persisted_manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode(),
        )
        report = "\n".join(
            (
                "# CN Macro Tushare Normalization",
                "",
                f"- Status: `{result.manifest['status']}`",
                f"- Observations: `{len(result.observations)}`",
                f"- Quarantine: `{len(result.quarantine)}`",
                "- Observer only: `true`",
                "- Production eligible: `false`",
                "- Applied: `false`",
                "",
            )
        )
        _atomic_bytes(staging / "normalization_report.md", report.encode())
        _fsync_dir(staging)
        os.replace(staging, final)
        _fsync_dir(market_root)
        return {
            "observations": str(final / "observations.jsonl"),
            "quarantine": str(final / "quarantine.jsonl"),
            "receipts": str(final / "normalization_receipts.jsonl"),
            "raw_bundle": str(final / "raw_bundle.json"),
            "backfill_plan": str(final / "backfill_plan.json"),
            "evidence_capture": str(final / "evidence_capture.json"),
            "manifest": str(final / "normalization_manifest.json"),
            "report": str(final / "normalization_report.md"),
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _read_json_object(
    path: str | Path, error: str
) -> tuple[Mapping[str, Any], bytes]:
    source = Path(path).expanduser()
    if not source.exists() or _unsafe_path(source) or not source.is_file():
        raise MacroNormalizationError(f"{error}_missing_or_unsafe")
    try:
        payload_bytes = source.read_bytes()
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise MacroNormalizationError(f"{error}_json_invalid") from exc
    if not isinstance(payload, Mapping):
        raise MacroNormalizationError(f"{error}_not_object")
    return payload, payload_bytes


def normalize_tushare_bundle_file(
    path: str | Path,
    *,
    plan_path: str | Path,
    evidence_path: str | Path,
    output_root: str | Path = "results/macro_normalization",
    run_id: str,
) -> dict[str, Any]:
    raw, raw_bytes = _read_json_object(path, "macro_raw_bundle")
    plan, plan_bytes = _read_json_object(plan_path, "macro_backfill_plan")
    evidence, evidence_bytes = _read_json_object(
        evidence_path, "macro_evidence_capture"
    )
    result = normalize_tushare_bundle(
        raw,
        plan=plan,
        evidence_capture=evidence,
        raw_bundle_file_sha256=_sha256(raw_bytes),
        plan_file_sha256=_sha256(plan_bytes),
        evidence_capture_sha256=_sha256(evidence_bytes),
    )
    artifacts = persist_tushare_normalization(
        result,
        input_artifacts={
            "raw_bundle.json": raw_bytes,
            "backfill_plan.json": plan_bytes,
            "evidence_capture.json": evidence_bytes,
        },
        output_root=output_root,
        run_id=run_id,
    )
    manifest_sha = _sha256(Path(artifacts["manifest"]).read_bytes())
    return {
        **dict(result.manifest),
        "normalization_manifest_sha256": manifest_sha,
        "artifacts": artifacts,
        "promoted": False,
    }


def _load_jsonl_bytes(payload: bytes) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise MacroNormalizationError(
                "macro_normalization_jsonl_row_invalid"
            )
        rows.append(dict(row))
    return rows


def publish_tushare_normalization(
    manifest_path: str | Path,
    *,
    observations_root: str | Path,
    run_id: str,
    expected_pointer_sha256: str,
    expected_manifest_sha256: str,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Recompile a hash-bound evidence bundle, then publish through CAS."""

    if expected_pointer_sha256 is None:
        raise MacroNormalizationError(
            "macro_normalization_expected_pointer_required"
        )
    if expected_pointer_sha256 and not _SHA256.fullmatch(
        str(expected_pointer_sha256).lower()
    ):
        raise MacroNormalizationError(
            "macro_normalization_expected_pointer_hash_invalid"
        )
    expected_manifest = _assert_hash(
        expected_manifest_sha256,
        "macro_normalization_expected_manifest_hash_invalid",
    )
    expected_plan = _assert_hash(
        expected_plan_sha256, "macro_normalization_expected_plan_hash_invalid"
    )
    path = Path(manifest_path).expanduser()
    if not path.exists() or _unsafe_path(path) or not path.is_file():
        raise MacroNormalizationError(
            "macro_normalization_manifest_missing_or_unsafe"
        )
    manifest_bytes = path.read_bytes()
    if _sha256(manifest_bytes) != expected_manifest:
        raise MacroNormalizationError(
            "macro_normalization_manifest_hash_mismatch"
        )
    try:
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        raise MacroNormalizationError(
            "macro_normalization_manifest_invalid"
        ) from exc
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != NORMALIZATION_SCHEMA
    ):
        raise MacroNormalizationError(
            "macro_normalization_manifest_shape_invalid"
        )
    parent = path.parent
    names = (
        "observations.jsonl",
        "quarantine.jsonl",
        "normalization_receipts.jsonl",
        "raw_bundle.json",
        "backfill_plan.json",
        "evidence_capture.json",
    )
    declared_hashes = manifest.get("artifact_sha256")
    if not isinstance(declared_hashes, Mapping):
        raise MacroNormalizationError(
            "macro_normalization_artifact_hashes_missing"
        )
    payloads: dict[str, bytes] = {}
    for name in names:
        artifact = parent / name
        if (
            not artifact.exists()
            or _unsafe_path(artifact)
            or not artifact.is_file()
            or artifact.stat().st_mode & 0o077
        ):
            raise MacroNormalizationError(
                "macro_normalization_artifact_missing_or_unsafe"
            )
        payloads[name] = artifact.read_bytes()
        if _sha256(payloads[name]) != str(declared_hashes.get(name) or ""):
            raise MacroNormalizationError(
                "macro_normalization_artifact_hash_mismatch"
            )
    if _sha256(payloads["backfill_plan.json"]) != expected_plan:
        raise MacroNormalizationError("macro_normalization_plan_hash_mismatch")
    try:
        raw = json.loads(payloads["raw_bundle.json"])
        plan = json.loads(payloads["backfill_plan.json"])
        evidence = json.loads(payloads["evidence_capture.json"])
    except Exception as exc:
        raise MacroNormalizationError(
            "macro_normalization_input_artifact_invalid"
        ) from exc
    recomputed = normalize_tushare_bundle(
        raw,
        plan=plan,
        evidence_capture=evidence,
        raw_bundle_file_sha256=_sha256(payloads["raw_bundle.json"]),
        plan_file_sha256=_sha256(payloads["backfill_plan.json"]),
        evidence_capture_sha256=_sha256(payloads["evidence_capture.json"]),
    )
    observation_rows = _load_jsonl_bytes(payloads["observations.jsonl"])
    quarantine_rows = _load_jsonl_bytes(payloads["quarantine.jsonl"])
    receipt_rows = _load_jsonl_bytes(payloads["normalization_receipts.jsonl"])
    if (
        manifest.get("status") != "OK"
        or recomputed.manifest.get("status") != "OK"
    ):
        raise MacroNormalizationError(
            "macro_normalization_bundle_not_publishable"
        )
    if quarantine_rows or recomputed.quarantine:
        raise MacroNormalizationError(
            "macro_normalization_quarantine_not_empty"
        )
    if observation_rows != [
        item.to_dict() for item in recomputed.observations
    ] or receipt_rows != [dict(x) for x in recomputed.receipts]:
        raise MacroNormalizationError("macro_normalization_recompile_mismatch")
    core_manifest = {
        key: value
        for key, value in manifest.items()
        if key != "artifact_sha256"
    }
    if core_manifest != dict(recomputed.manifest):
        raise MacroNormalizationError(
            "macro_normalization_manifest_recompile_mismatch"
        )
    return publish_observations(
        recomputed.observations,
        root=observations_root,
        run_id=run_id,
        expected_pointer_sha256=expected_pointer_sha256,
        metadata={
            "normalization_manifest_sha256": expected_manifest,
            "backfill_plan_sha256": expected_plan,
            "raw_bundle_file_sha256": recomputed.manifest[
                "raw_bundle_file_sha256"
            ],
            "evidence_capture_sha256": recomputed.manifest[
                "evidence_capture_sha256"
            ],
            "receipt_set_hash": recomputed.manifest["receipt_set_hash"],
            "observer_only": True,
            "production_eligible": False,
            "applied": False,
        },
    )


__all__ = [
    "AVAILABILITY_POLICY",
    "BACKFILL_PLAN_SCHEMA",
    "EVIDENCE_CAPTURE_SCHEMA",
    "MacroNormalizationError",
    "MacroNormalizationResult",
    "RAW_BUNDLE_SCHEMA",
    "SPECS",
    "normalize_tushare_bundle",
    "normalize_tushare_bundle_file",
    "persist_tushare_normalization",
    "publish_tushare_normalization",
]
