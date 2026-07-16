"""Offline, observer-only acquisition tasks derived from a coverage audit."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import pandas as pd

from quant_investor.macro.contracts import canonical_hash
from quant_investor.macro.registry import (
    INDUSTRY_CHAINS,
    INDUSTRY_COMPONENT_WEIGHTS,
    NATIONAL_INDICATORS,
)

ACQUISITION_PLAN_SCHEMA = "macro-acquisition-plan.v1"
ACQUISITION_MANIFEST_SCHEMA = "macro-acquisition-plan-manifest.v1"
COVERAGE_AUDIT_SCHEMA = "macro-coverage-audit.v1.1"
DEFAULT_OUTPUT_ROOT = Path("results/macro_acquisition_plan")
_UTC = ZoneInfo("UTC")
_SAFE_SLUG = re.compile(r"^[A-Za-z0-9_.-]+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_SOURCE_ROUTES: dict[str, tuple[str, list[str]]] = {
    "nbs_official": ("confirmed", ["stats.gov.cn"]),
    "pbc_official": ("confirmed", ["pbc.gov.cn"]),
    "pboc_official": ("confirmed", ["pbc.gov.cn"]),
    "customs_official": ("confirmed", ["customs.gov.cn"]),
    "mof_official": ("confirmed", ["mof.gov.cn"]),
    "ndrc_official": ("confirmed", ["ndrc.gov.cn"]),
    "local_strict_parquet": ("confirmed_local", []),
}

_EXPECTED_AUTHORITY_BY_NATIONAL = {
    "cn.gdp_yoy": "nbs_official",
    "cn.industrial_value_added_yoy": "nbs_official",
    "cn.pmi_manufacturing": "nbs_official",
    "cn.retail_sales_yoy": "nbs_official",
    "cn.fixed_asset_investment_yoy": "nbs_official",
    "cn.m1_yoy": "pboc_official",
    "cn.m2_yoy": "pboc_official",
    "cn.social_financing_flow": "pboc_official",
    "cn.cpi_yoy": "nbs_official",
    "cn.ppi_yoy": "nbs_official",
    "cn.fiscal_expenditure_yoy": "mof_official",
    "cn.property_investment_yoy": "nbs_official",
    # The reviewed NBS national-economy release republishes these exact CNY
    # monthly rates and identifies Customs as the upstream data authority.
    "cn.exports_yoy": "nbs_official",
    "cn.imports_yoy": "nbs_official",
    "market.breadth": "local_strict_parquet",
    "market.volatility_percentile": "local_strict_parquet",
}

_OFFICIAL_REQUIREMENTS = [
    "immutable_raw_capture_sha256",
    "issuer_bound_https_url",
    "source_record_id",
    "timezone_release_at",
    "timezone_available_at",
    "captured_at_not_before_available_at",
    "period_value_unit_frequency_exact_match",
    "zero_quarantine_recompile",
]

_LOCAL_REQUIREMENTS = [
    "strict_parquet_snapshot_sha256",
    "trade_date_and_cutoff_bound",
    "source_table_lineage",
    "deterministic_recompute",
    "zero_quarantine_recompile",
]


class MacroAcquisitionError(RuntimeError):
    """Raised when acquisition planning inputs or outputs are unsafe."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unsafe_path(path: Path) -> bool:
    current = path.absolute()
    while True:
        if current.exists() and current.is_symlink():
            return True
        if current.parent == current:
            return False
        current = current.parent


def _load_coverage(
    value: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], str | None]:
    file_sha256: str | None = None
    if isinstance(value, Mapping):
        payload = dict(value)
    else:
        path = Path(value).expanduser()
        if _unsafe_path(path) or not path.is_file():
            raise MacroAcquisitionError("coverage_input_unsafe")
        source = path.read_bytes()
        file_sha256 = _sha256(source)
        try:
            parsed = json.loads(source)
        except Exception as exc:
            raise MacroAcquisitionError("coverage_input_invalid") from exc
        if not isinstance(parsed, Mapping):
            raise MacroAcquisitionError("coverage_input_invalid")
        payload = dict(parsed)
    if payload.get("schema_version") != COVERAGE_AUDIT_SCHEMA:
        raise MacroAcquisitionError("coverage_schema_unsupported")
    if str(payload.get("market") or "").upper() != "CN":
        raise MacroAcquisitionError("coverage_market_unsupported")
    audit_hash = str(payload.get("audit_hash") or "")
    semantic = dict(payload)
    semantic.pop("audit_hash", None)
    if (
        not _SHA256.fullmatch(audit_hash)
        or canonical_hash(semantic) != audit_hash
    ):
        raise MacroAcquisitionError("coverage_hash_mismatch")
    if not isinstance(payload.get("national"), list) or not isinstance(
        payload.get("industry"), list
    ):
        raise MacroAcquisitionError("coverage_tasks_missing")
    return payload, file_sha256


def _action_for(row: Mapping[str, Any], *, dimension: str) -> str:
    status = str(row.get("status") or "")
    authority = str(row.get("expected_authority") or "UNCONFIRMED")
    if status == "pit_signal_ready":
        return "none"
    if authority == "UNCONFIRMED":
        return "confirm_authority_and_mapping"
    if status == "observation_present_not_signal_ready":
        return "repair_observation_quality"
    if status == "raw_present_pit_evidence_missing":
        return "bind_timestamp_release_evidence"
    if status == "mapped_raw_not_usable_as_of":
        return "repair_raw_capture_as_of"
    if status == "mapped_raw_missing":
        return "acquire_raw_and_release_evidence"
    if status == "mapping_not_implemented":
        if authority == "local_strict_parquet":
            return "build_local_strict_parquet_observation"
        return "implement_official_mapping"
    if dimension == "industry":
        return "confirm_authority_and_mapping"
    return "review_unknown_status"


def _task(row: Mapping[str, Any], *, dimension: str) -> dict[str, Any]:
    indicator_id = str(row.get("indicator_id") or "").strip()
    if not indicator_id:
        raise MacroAcquisitionError("coverage_indicator_id_missing")
    authority = str(row.get("expected_authority") or "UNCONFIRMED")
    if dimension == "national":
        expected_authority = _EXPECTED_AUTHORITY_BY_NATIONAL.get(indicator_id)
        if expected_authority != authority:
            raise MacroAcquisitionError("coverage_authority_mismatch")
    else:
        parts = indicator_id.split(".")
        if (
            len(parts) != 3
            or str(row.get("industry_chain") or "") != parts[1]
            or str(row.get("component") or "") != parts[2]
        ):
            raise MacroAcquisitionError(
                "coverage_industry_metadata_mismatch"
            )
        if authority != "UNCONFIRMED":
            raise MacroAcquisitionError("coverage_authority_mismatch")
    authority_status, domains = _SOURCE_ROUTES.get(
        authority, ("UNCONFIRMED", [])
    )
    action = _action_for(row, dimension=dimension)
    task_status = "satisfied" if action == "none" else "open"
    requirements = [] if task_status == "satisfied" else list(
        _OFFICIAL_REQUIREMENTS
    )
    if authority == "local_strict_parquet" and task_status == "open":
        requirements = list(_LOCAL_REQUIREMENTS)
    if authority == "UNCONFIRMED":
        requirements = ["authority_owner_and_domain_confirmed", *requirements]
    return {
        "indicator_id": indicator_id,
        "dimension": dimension,
        "industry_chain": str(row.get("industry_chain") or ""),
        "component": str(row.get("component") or ""),
        "frequency": str(row.get("frequency") or "monthly"),
        "unit": str(row.get("unit") or ""),
        "current_status": str(row.get("status") or "unknown"),
        "current_blockers": sorted(
            str(item) for item in (row.get("blockers") or [])
        ),
        "expected_authority": authority,
        "authority_status": authority_status,
        "allowed_domains": domains,
        "mapping_endpoint": row.get("mapping_endpoint"),
        "task_status": task_status,
        "action": action,
        "priority": 0 if task_status == "satisfied" else (
            1 if action in {
                "bind_timestamp_release_evidence",
                "repair_observation_quality",
                "repair_raw_capture_as_of",
            } else 2
        ),
        "acceptance_requirements": requirements,
        "observation_emitted": False,
        "production_eligible": False,
    }


def build_macro_acquisition_plan(
    coverage_audit: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Build tasks only; this function never fetches or emits observations."""

    coverage, coverage_file_sha256 = _load_coverage(coverage_audit)
    national = sorted(
        (_task(row, dimension="national") for row in coverage["national"]),
        key=lambda row: row["indicator_id"],
    )
    industry = sorted(
        (_task(row, dimension="industry") for row in coverage["industry"]),
        key=lambda row: row["indicator_id"],
    )
    ids = [row["indicator_id"] for row in [*national, *industry]]
    if len(ids) != len(set(ids)):
        raise MacroAcquisitionError("coverage_indicator_duplicate")
    expected_national = {item.indicator_id for item in NATIONAL_INDICATORS}
    expected_industry = {
        f"industry.{chain}.{component}"
        for chain in INDUSTRY_CHAINS
        for component in INDUSTRY_COMPONENT_WEIGHTS
    }
    if (
        {row["indicator_id"] for row in national} != expected_national
        or {row["indicator_id"] for row in industry} != expected_industry
    ):
        raise MacroAcquisitionError("coverage_scope_mismatch")
    open_count = sum(
        row["task_status"] == "open" for row in [*national, *industry]
    )
    semantic = {
        "schema_version": ACQUISITION_PLAN_SCHEMA,
        "market": "CN",
        "as_of": str(coverage.get("as_of") or ""),
        "coverage_audit_hash": coverage["audit_hash"],
        "coverage_audit_file_sha256": coverage_file_sha256,
        "status": "ready" if open_count == 0 else "blocked",
        "task_count": len(national) + len(industry),
        "open_task_count": open_count,
        "satisfied_task_count": len(national) + len(industry) - open_count,
        "national_tasks": national,
        "industry_tasks": industry,
        "observation_count": 0,
        "observer_only": True,
        "production_eligible": False,
        "activation_authorized": False,
        "applied": False,
    }
    return {**semantic, "plan_hash": canonical_hash(semantic)}


def _report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# CN Macro Acquisition Plan",
        "",
        f"- Status: `{payload['status']}`",
        f"- As of: `{payload['as_of']}`",
        f"- Coverage audit: `{payload['coverage_audit_hash']}`",
        f"- Plan hash: `{payload['plan_hash']}`",
        (
            f"- Open tasks: `{payload['open_task_count']}/"
            f"{payload['task_count']}`"
        ),
        "- Observations emitted: `0`",
        "- Production eligible: `false`",
        "",
        "## Task summary",
        "",
        "| Dimension | Indicator | Status | Action | Authority |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in [*payload["national_tasks"], *payload["industry_tasks"]]:
        lines.append(
            f"| {row['dimension']} | {row['indicator_id']} | "
            f"{row['task_status']} | {row['action']} | "
            f"{row['expected_authority']} |"
        )
    return "\n".join(lines) + "\n"


def _artifacts(payload: Mapping[str, Any]) -> dict[str, bytes]:
    tasks = pd.DataFrame(
        [*payload["national_tasks"], *payload["industry_tasks"]]
    )
    tasks = tasks.reindex(sorted(tasks.columns), axis=1)
    return {
        "acquisition_plan.json": json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2
        ).encode("utf-8"),
        "acquisition_tasks.csv": tasks.to_csv(index=False).encode("utf-8"),
        "acquisition_report.md": _report(payload).encode("utf-8"),
    }


def _write(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
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


def persist_macro_acquisition_plan(
    payload: Mapping[str, Any],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    root = Path(output_root).expanduser()
    if _unsafe_path(root):
        raise MacroAcquisitionError("output_root_symlink_rejected")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    as_of = str(payload.get("as_of") or "").split("T", 1)[0]
    plan_hash = str(payload.get("plan_hash") or "")
    if not _SAFE_SLUG.fullmatch(as_of) or not _SHA256.fullmatch(plan_hash):
        raise MacroAcquisitionError("output_slug_invalid")
    date_root = root / "CN" / as_of
    if _unsafe_path(date_root):
        raise MacroAcquisitionError("output_path_symlink_rejected")
    date_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(date_root.parent, 0o700)
    os.chmod(date_root, 0o700)
    final = date_root / plan_hash
    if final.exists():
        if _unsafe_path(final) or final.stat().st_mode & 0o077:
            raise MacroAcquisitionError("existing_output_unsafe")
        plan_path = final / "acquisition_plan.json"
        manifest_path = final / "manifest.json"
        if any(
            _unsafe_path(path)
            or not path.is_file()
            or path.stat().st_mode & 0o077
            for path in (plan_path, manifest_path)
        ):
            raise MacroAcquisitionError("existing_output_unsafe")
        try:
            persisted = json.loads(plan_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise MacroAcquisitionError("existing_output_invalid") from exc
        semantic = dict(persisted)
        persisted_hash = str(semantic.pop("plan_hash", ""))
        if (
            persisted_hash != plan_hash
            or canonical_hash(semantic) != plan_hash
        ):
            raise MacroAcquisitionError("existing_plan_hash_mismatch")
        expected = _artifacts(persisted)
        hashes = manifest.get("artifact_sha256", {})
        if (
            manifest.get("schema_version") != ACQUISITION_MANIFEST_SCHEMA
            or manifest.get("plan_hash") != plan_hash
            or set(hashes) != set(expected)
        ):
            raise MacroAcquisitionError("existing_output_mismatch")
        for name, content in expected.items():
            path = final / name
            if (
                _unsafe_path(path)
                or not path.is_file()
                or path.stat().st_mode & 0o077
                or path.read_bytes() != content
                or hashes[name] != _sha256(content)
            ):
                raise MacroAcquisitionError("artifact_mismatch")
        return {
            **dict(payload),
            "output_dir": str(final),
            "idempotent": True,
        }
    staging = Path(
        tempfile.mkdtemp(prefix=f".{plan_hash[:12]}.", dir=date_root)
    )
    os.chmod(staging, 0o700)
    try:
        artifacts = _artifacts(payload)
        for name, content in artifacts.items():
            _write(staging / name, content)
        manifest = {
            "schema_version": ACQUISITION_MANIFEST_SCHEMA,
            "plan_hash": plan_hash,
            "generated_at": datetime.now(_UTC).isoformat(),
            "artifact_sha256": {
                name: _sha256(content) for name, content in artifacts.items()
            },
            "observer_only": True,
            "production_eligible": False,
            "applied": False,
        }
        _write(
            staging / "manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8"),
        )
        _fsync_dir(staging)
        os.replace(staging, final)
        _fsync_dir(date_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {**dict(payload), "output_dir": str(final), "idempotent": False}


def run_macro_acquisition_plan(
    *,
    market: str = "CN",
    coverage_audit: Mapping[str, Any] | str | Path,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    if str(market).upper() != "CN":
        raise MacroAcquisitionError("market_unsupported")
    return persist_macro_acquisition_plan(
        build_macro_acquisition_plan(coverage_audit), output_root=output_root
    )


__all__ = [
    "ACQUISITION_PLAN_SCHEMA",
    "MacroAcquisitionError",
    "build_macro_acquisition_plan",
    "persist_macro_acquisition_plan",
    "run_macro_acquisition_plan",
]
