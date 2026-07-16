"""Offline coverage audit for national and industry Macro v2 observations."""

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

from quant_investor.macro.contracts import (
    canonical_hash,
    parse_timestamp,
    published_cutoff,
)
from quant_investor.macro.observer import load_macro_observation_generation
from quant_investor.macro.registry import (
    INDUSTRY_CHAINS,
    INDUSTRY_COMPONENT_WEIGHTS,
    NATIONAL_INDICATORS,
)
from quant_investor.macro.snapshot import build_macro_snapshot
from quant_investor.macro.tushare_normalizer import SPECS

COVERAGE_AUDIT_SCHEMA = "macro-coverage-audit.v1.1"
COVERAGE_MANIFEST_SCHEMA = "macro-coverage-manifest.v1.1"
DEFAULT_RAW_ROOT = Path("data/parquet/cn/dag_core_raw")
DEFAULT_OBSERVATIONS_ROOT = Path("data/parquet/cn/macro_observations")
DEFAULT_OUTPUT_ROOT = Path("results/macro_coverage_audit")
_UTC = ZoneInfo("UTC")
_SAFE_SLUG = re.compile(r"^[A-Za-z0-9_.-]+$")

_AUTHORITY_BY_INDICATOR = {
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
    "cn.exports_yoy": "nbs_official",
    "cn.imports_yoy": "nbs_official",
    "market.breadth": "local_strict_parquet",
    "market.volatility_percentile": "local_strict_parquet",
}


class MacroCoverageError(RuntimeError):
    """Raised when coverage inputs or outputs are structurally unsafe."""


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


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _raw_inventory(
    root_value: str | Path, *, cutoff: datetime
) -> tuple[dict[str, Any], list[str]]:
    root = Path(root_value).expanduser()
    if _unsafe_path(root):
        raise MacroCoverageError("macro_coverage_raw_root_symlink_rejected")
    blockers: list[str] = []
    inventory: dict[str, Any] = {}
    specs_by_endpoint: dict[str, list[Any]] = {}
    for spec in SPECS:
        specs_by_endpoint.setdefault(spec.endpoint, []).append(spec)
    for endpoint, endpoint_specs in sorted(specs_by_endpoint.items()):
        path = root / f"table={endpoint}" / "part.parquet"
        indicator_ids = sorted(spec.indicator_id for spec in endpoint_specs)
        base: dict[str, Any] = {
            "endpoint": endpoint,
            "indicator_ids": indicator_ids,
            "path_present": False,
            "row_count": 0,
            "row_count_as_of": 0,
            "columns": [],
            "period_min": None,
            "period_max": None,
            "fetched_at_min": None,
            "fetched_at_max": None,
            "source_systems": [],
            "source_snapshot_ids": [],
            "file_sha256": None,
            "pit_promotable": False,
            "raw_available_by_audit_cutoff": False,
            "pit_blocker": "raw_table_missing",
        }
        if not path.exists():
            blockers.append(f"raw_table_missing:{endpoint}")
            inventory[endpoint] = base
            continue
        if _unsafe_path(path) or not path.is_file():
            raise MacroCoverageError("macro_coverage_raw_table_unsafe")
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            base["pit_blocker"] = "raw_table_unreadable"
            blockers.append(f"raw_table_unreadable:{endpoint}:{exc}")
            inventory[endpoint] = base
            continue
        period_field = endpoint_specs[0].period_field
        required = {
            period_field,
            "fetched_at",
            "source",
            "source_snapshot_id",
        }
        missing = sorted(required - set(frame.columns))
        missing_value_groups = [
            "/".join(spec.accepted_value_fields)
            for spec in endpoint_specs
            if not set(spec.accepted_value_fields).intersection(frame.columns)
        ]
        missing.extend(f"value:{group}" for group in missing_value_groups)
        base.update(
            {
                "path_present": True,
                "row_count": len(frame),
                "columns": sorted(str(column) for column in frame.columns),
                "file_sha256": _sha256(path.read_bytes()),
            }
        )
        if frame.empty:
            base["pit_blocker"] = "raw_table_empty"
            blockers.append(f"raw_table_empty:{endpoint}")
            inventory[endpoint] = base
            continue
        if missing:
            base["pit_blocker"] = f"raw_schema_missing:{','.join(missing)}"
            blockers.append(
                f"raw_schema_missing:{endpoint}:{','.join(missing)}"
            )
            inventory[endpoint] = base
            continue
        if frame["fetched_at"].isna().any():
            base["pit_blocker"] = "fetched_at_missing"
            blockers.append(f"raw_fetched_at_invalid:{endpoint}")
            inventory[endpoint] = base
            continue
        fetched = frame["fetched_at"].astype(str)
        try:
            parsed_fetched = [
                parse_timestamp(value, field_name="fetched_at")
                for value in fetched
            ]
        except ValueError as exc:
            base["pit_blocker"] = str(exc)
            blockers.append(f"raw_fetched_at_invalid:{endpoint}")
            inventory[endpoint] = base
            continue
        eligible = frame.loc[
            [value <= cutoff for value in parsed_fetched]
        ]
        if eligible.empty:
            base.update(
                {
                    "fetched_at_min": min(fetched),
                    "fetched_at_max": max(fetched),
                    "pit_blocker": "raw_capture_after_audit_cutoff",
                }
            )
            blockers.append(f"raw_capture_after_audit_cutoff:{endpoint}")
            inventory[endpoint] = base
            continue
        source_values = eligible["source"].dropna().astype(str).str.strip()
        snapshot_values = (
            eligible["source_snapshot_id"].dropna().astype(str).str.strip()
        )
        if (
            len(source_values) != len(eligible)
            or not source_values.astype(bool).all()
            or len(snapshot_values) != len(eligible)
            or not snapshot_values.astype(bool).all()
        ):
            base["pit_blocker"] = "raw_provenance_missing"
            blockers.append(f"raw_provenance_missing:{endpoint}")
            inventory[endpoint] = base
            continue
        periods = eligible[period_field].dropna().astype(str)
        if periods.empty:
            base["pit_blocker"] = "raw_period_empty"
            blockers.append(f"raw_period_empty:{endpoint}")
            inventory[endpoint] = base
            continue
        period_pattern = (
            r"^\d{4}Q[1-4]$"
            if any(spec.frequency == "quarterly" for spec in endpoint_specs)
            else r"^\d{6}$"
        )
        if not periods.str.fullmatch(period_pattern).all():
            base["pit_blocker"] = "raw_period_format_invalid"
            blockers.append(f"raw_period_format_invalid:{endpoint}")
            inventory[endpoint] = base
            continue
        alias_conflicts: list[str] = []
        for spec in endpoint_specs:
            fields = [
                field
                for field in spec.accepted_value_fields
                if field in eligible.columns
            ]
            if len(fields) < 2:
                continue
            for _, row in eligible[fields].iterrows():
                values = {
                    float(row[field])
                    for field in fields
                    if pd.notna(row[field])
                }
                if len(values) > 1:
                    alias_conflicts.append(spec.indicator_id)
                    break
        if alias_conflicts:
            joined = ",".join(sorted(set(alias_conflicts)))
            base["pit_blocker"] = f"raw_value_alias_conflict:{joined}"
            blockers.append(f"raw_value_alias_conflict:{endpoint}:{joined}")
            inventory[endpoint] = base
            continue
        value_counts = {
            spec.indicator_id: max(
                int(eligible[field].notna().sum())
                for field in spec.accepted_value_fields
                if field in eligible.columns
            )
            for spec in endpoint_specs
        }
        empty_values = sorted(
            indicator_id
            for indicator_id, count in value_counts.items()
            if count == 0
        )
        if empty_values:
            base["pit_blocker"] = (
                f"raw_value_column_empty:{','.join(empty_values)}"
            )
            blockers.append(
                f"raw_value_column_empty:{endpoint}:{','.join(empty_values)}"
            )
            inventory[endpoint] = base
            continue
        base.update(
            {
                "row_count_as_of": len(eligible),
                "period_min": min(periods) if not periods.empty else None,
                "period_max": max(periods) if not periods.empty else None,
                "fetched_at_min": min(fetched) if not fetched.empty else None,
                "fetched_at_max": max(fetched) if not fetched.empty else None,
                "source_systems": sorted(set(source_values)),
                "source_snapshot_ids": sorted(set(snapshot_values)),
                "raw_available_by_audit_cutoff": True,
                "pit_blocker": (
                    "timestamp_level_availability_evidence_not_bound"
                ),
            }
        )
        blockers.append(f"raw_pit_evidence_missing:{endpoint}")
        inventory[endpoint] = base
    return inventory, sorted(blockers)


def _load_observations(
    path_value: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any], str, list[str]]:
    path = Path(path_value).expanduser()
    if _unsafe_path(path):
        raise MacroCoverageError(
            "macro_coverage_observations_symlink_rejected"
        )
    if not path.exists():
        return [], {}, "missing", ["observation_input_missing"]
    try:
        rows, generation = load_macro_observation_generation(
            path,
            allow_standalone_offline=True,
        )
    except Exception as exc:
        return [], {}, "blocked", [f"observation_input_blocked:{exc}"]
    if path.is_file():
        generation = {
            **dict(generation),
            "standalone_file_sha256": _sha256(path.read_bytes()),
        }
    status = "OK" if rows else "empty"
    blockers = [] if rows else ["observation_input_empty"]
    return rows, generation, status, blockers


def _national_rows(
    snapshot: Any, raw_inventory: Mapping[str, Any]
) -> list[dict[str, Any]]:
    spec_by_indicator = {spec.indicator_id: spec for spec in SPECS}
    stale = set(snapshot.freshness.get("stale_availability", []))
    stale_periods = set(snapshot.freshness.get("stale_periods", []))
    insufficient = set(snapshot.freshness.get("insufficient_history", []))
    rows: list[dict[str, Any]] = []
    for definition in NATIONAL_INDICATORS:
        indicator_id = definition.indicator_id
        lineage = dict(snapshot.source_lineage.get(indicator_id, {}) or {})
        spec = spec_by_indicator.get(indicator_id)
        raw = dict(raw_inventory.get(spec.endpoint, {}) or {}) if spec else {}
        reasons: list[str] = []
        if indicator_id in stale:
            reasons.append("stale_availability")
        if indicator_id in stale_periods:
            reasons.append("stale_period")
        if indicator_id in insufficient:
            reasons.append("insufficient_history")
        if lineage and not reasons:
            status = "pit_signal_ready"
        elif lineage:
            status = "observation_present_not_signal_ready"
        elif raw.get("raw_available_by_audit_cutoff"):
            status = "raw_present_pit_evidence_missing"
            reasons.append(
                str(raw.get("pit_blocker") or "pit_evidence_missing")
            )
        elif spec and raw.get("path_present"):
            status = "mapped_raw_not_usable_as_of"
            reasons.append(str(raw.get("pit_blocker") or "raw_not_usable"))
        elif spec:
            status = "mapped_raw_missing"
            reasons.append("raw_table_missing")
        else:
            status = "mapping_not_implemented"
            reasons.append("reviewed_raw_mapping_missing")
        rows.append(
            {
                "indicator_id": indicator_id,
                "domain": definition.domain,
                "frequency": definition.frequency,
                "unit": definition.unit,
                "expected_authority": _AUTHORITY_BY_INDICATOR[indicator_id],
                "mapping_endpoint": spec.endpoint if spec else None,
                "status": status,
                "observation_present": bool(lineage),
                "latest_period_end": lineage.get("period_end"),
                "latest_available_at": lineage.get("available_at"),
                "source_system": lineage.get("source_system"),
                "raw_table_present": bool(raw.get("path_present")),
                "raw_available_by_audit_cutoff": bool(
                    raw.get("raw_available_by_audit_cutoff")
                ),
                "raw_period_min": raw.get("period_min"),
                "raw_period_max": raw.get("period_max"),
                "blockers": sorted(set(reasons)),
                "production_eligible": False,
            }
        )
    return rows


def _industry_rows(snapshot: Any) -> list[dict[str, Any]]:
    stale = set(snapshot.freshness.get("stale_availability", []))
    stale_periods = set(snapshot.freshness.get("stale_periods", []))
    insufficient = set(snapshot.freshness.get("insufficient_history", []))
    rows: list[dict[str, Any]] = []
    for chain in INDUSTRY_CHAINS:
        for component in INDUSTRY_COMPONENT_WEIGHTS:
            indicator_id = f"industry.{chain}.{component}"
            lineage = dict(snapshot.source_lineage.get(indicator_id, {}) or {})
            reasons: list[str] = []
            if indicator_id in stale:
                reasons.append("stale_availability")
            if indicator_id in stale_periods:
                reasons.append("stale_period")
            if indicator_id in insufficient:
                reasons.append("insufficient_history")
            if lineage and not reasons:
                status = "pit_signal_ready"
            elif lineage:
                status = "observation_present_not_signal_ready"
            else:
                status = "mapping_not_implemented"
                reasons.append("official_industry_source_not_mapped")
            rows.append(
                {
                    "industry_chain": chain,
                    "component": component,
                    "indicator_id": indicator_id,
                    "component_weight": INDUSTRY_COMPONENT_WEIGHTS[component],
                    "status": status,
                    "observation_present": bool(lineage),
                    "latest_period_end": lineage.get("period_end"),
                    "latest_available_at": lineage.get("available_at"),
                    "source_system": lineage.get("source_system"),
                    "expected_authority": "UNCONFIRMED",
                    "blockers": sorted(set(reasons)),
                    "production_eligible": False,
                }
            )
    return rows


def build_macro_coverage_audit(
    *,
    market: str = "CN",
    as_of: str,
    observations_path: str | Path = DEFAULT_OBSERVATIONS_ROOT,
    raw_root: str | Path = DEFAULT_RAW_ROOT,
) -> dict[str, Any]:
    """Build a truthful coverage matrix without fetching or inferring data."""

    if str(market).upper() != "CN":
        raise MacroCoverageError("macro_coverage_market_unsupported")
    cutoff = published_cutoff(as_of)
    observations, generation, input_status, input_blockers = (
        _load_observations(observations_path)
    )
    try:
        snapshot = build_macro_snapshot(observations, market="CN", as_of=as_of)
    except Exception as exc:
        input_status = "blocked"
        input_blockers.append(f"snapshot_build_blocked:{exc}")
        snapshot = build_macro_snapshot([], market="CN", as_of=as_of)
    raw_inventory, raw_blockers = _raw_inventory(raw_root, cutoff=cutoff)
    national = _national_rows(snapshot, raw_inventory)
    industry = _industry_rows(snapshot)
    national_ready = sum(
        row["status"] == "pit_signal_ready" for row in national
    )
    industry_ready = sum(
        row["status"] == "pit_signal_ready" for row in industry
    )
    ready_chains = sum(
        float(snapshot.coverage.get("industry_chains", {}).get(chain, 0.0))
        >= 0.70
        for chain in INDUSTRY_CHAINS
    )
    blockers = sorted(
        set(input_blockers)
        | set(raw_blockers)
        | set(snapshot.blockers)
        | {
            (
                "national_signal_coverage_below_80pct"
                if national_ready / len(national) < 0.80
                else ""
            )
        }
        | ({"no_industry_chain_at_70pct"} if ready_chains == 0 else set())
    )
    blockers = [item for item in blockers if item]
    semantic = {
        "schema_version": COVERAGE_AUDIT_SCHEMA,
        "market": "CN",
        "as_of": str(as_of),
        "published_cutoff": cutoff.isoformat(),
        "status": "ready" if not blockers else "blocked",
        "observation_input_status": input_status,
        "observation_generation": dict(generation),
        "snapshot_hash": snapshot.snapshot_hash,
        "snapshot_readiness_status": snapshot.readiness_status,
        "national_indicator_count": len(national),
        "national_pit_signal_ready_count": national_ready,
        "national_pit_signal_coverage": national_ready / len(national),
        "industry_chain_count": len(INDUSTRY_CHAINS),
        "industry_component_count": len(industry),
        "industry_pit_signal_ready_count": industry_ready,
        "industry_pit_signal_coverage": industry_ready / len(industry),
        "industry_chain_70pct_ready_count": ready_chains,
        "national": national,
        "industry": industry,
        "raw_inventory": raw_inventory,
        "blockers": blockers,
        "observer_only": True,
        "production_eligible": False,
        "activation_authorized": False,
        "applied": False,
    }
    return {**semantic, "audit_hash": canonical_hash(semantic)}


def _render_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# CN Macro Coverage Audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- As of: `{payload['as_of']}`",
        f"- Audit hash: `{payload['audit_hash']}`",
        (
            "- National PIT signal coverage: "
            f"`{payload['national_pit_signal_ready_count']}/"
            f"{payload['national_indicator_count']}`"
        ),
        (
            "- Industry PIT signal coverage: "
            f"`{payload['industry_pit_signal_ready_count']}/"
            f"{payload['industry_component_count']}`"
        ),
        (
            "- Industry chains >=70%: "
            f"`{payload['industry_chain_70pct_ready_count']}/12`"
        ),
        "- Production eligible: `false`",
        "- Applied: `false`",
        "",
        "## National indicators",
        "",
        (
            "| Indicator | Domain | Status | Latest period | Raw table | "
            "Blockers |"
        ),
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["national"]:
        lines.append(
            f"| {row['indicator_id']} | {row['domain']} | {row['status']} | "
            f"{row['latest_period_end'] or ''} | "
            f"{str(row['raw_table_present']).lower()} | "
            f"{', '.join(row['blockers'])} |"
        )
    lines.extend(
        [
            "",
            "## Industry chains",
            "",
            "| Chain | Ready components | Coverage |",
            "| --- | ---: | ---: |",
        ]
    )
    for chain in INDUSTRY_CHAINS:
        chain_rows = [
            row
            for row in payload["industry"]
            if row["industry_chain"] == chain
        ]
        ready = sum(row["status"] == "pit_signal_ready" for row in chain_rows)
        lines.append(f"| {chain} | {ready}/8 | {ready / 8:.1%} |")
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{item}`" for item in payload["blockers"])
    return "\n".join(lines) + "\n"


def _artifact_payloads(payload: Mapping[str, Any]) -> dict[str, bytes]:
    national_frame = pd.DataFrame(payload["national"])
    industry_frame = pd.DataFrame(payload["industry"])
    national_frame = national_frame.reindex(
        sorted(national_frame.columns), axis=1
    )
    industry_frame = industry_frame.reindex(
        sorted(industry_frame.columns), axis=1
    )
    return {
        "coverage_audit.json": json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2
        ).encode("utf-8"),
        "national_coverage.csv": national_frame.to_csv(index=False).encode(
            "utf-8"
        ),
        "industry_coverage.csv": industry_frame.to_csv(index=False).encode(
            "utf-8"
        ),
        "coverage_report.md": _render_report(payload).encode("utf-8"),
    }


def persist_macro_coverage_audit(
    payload: Mapping[str, Any],
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    root = Path(output_root).expanduser()
    if _unsafe_path(root):
        raise MacroCoverageError("macro_coverage_output_root_symlink_rejected")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    market_root = root / "CN"
    if _unsafe_path(market_root):
        raise MacroCoverageError("macro_coverage_market_root_symlink_rejected")
    market_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(market_root, 0o700)
    as_of_slug = str(payload.get("as_of") or "").split("T", 1)[0]
    audit_hash = str(payload.get("audit_hash") or "")
    if not _SAFE_SLUG.fullmatch(as_of_slug) or not re.fullmatch(
        r"[0-9a-f]{64}", audit_hash
    ):
        raise MacroCoverageError("macro_coverage_output_slug_invalid")
    date_root = market_root / as_of_slug
    if _unsafe_path(date_root):
        raise MacroCoverageError("macro_coverage_date_root_symlink_rejected")
    date_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(date_root, 0o700)
    final = date_root / audit_hash
    if final.exists():
        manifest_path = final / "manifest.json"
        if (
            _unsafe_path(final)
            or final.stat().st_mode & 0o077
            or not manifest_path.is_file()
            or _unsafe_path(manifest_path)
            or manifest_path.stat().st_mode & 0o077
        ):
            raise MacroCoverageError("macro_coverage_existing_output_unsafe")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, Mapping):
            raise MacroCoverageError("macro_coverage_existing_output_mismatch")
        hashes = manifest.get("artifact_sha256")
        if (
            manifest.get("schema_version") != COVERAGE_MANIFEST_SCHEMA
            or manifest.get("audit_hash") != audit_hash
            or not isinstance(hashes, Mapping)
        ):
            raise MacroCoverageError("macro_coverage_existing_output_mismatch")
        try:
            persisted_audit = json.loads(
                (final / "coverage_audit.json").read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise MacroCoverageError(
                "macro_coverage_existing_audit_invalid"
            ) from exc
        if not isinstance(persisted_audit, Mapping):
            raise MacroCoverageError("macro_coverage_existing_audit_invalid")
        semantic = dict(persisted_audit)
        persisted_hash = str(semantic.pop("audit_hash", ""))
        if (
            persisted_hash != audit_hash
            or canonical_hash(semantic) != audit_hash
        ):
            raise MacroCoverageError(
                "macro_coverage_existing_audit_hash_mismatch"
            )
        expected_artifacts = _artifact_payloads(persisted_audit)
        if set(hashes) != set(expected_artifacts):
            raise MacroCoverageError("macro_coverage_existing_output_mismatch")
        for name, expected_bytes in expected_artifacts.items():
            artifact = final / name
            if (
                not artifact.is_file()
                or _unsafe_path(artifact)
                or artifact.stat().st_mode & 0o077
                or artifact.read_bytes() != expected_bytes
                or _sha256(expected_bytes) != hashes[name]
            ):
                raise MacroCoverageError(
                    "macro_coverage_existing_artifact_mismatch"
                )
        return {
            **dict(payload),
            "output_dir": str(final),
            "promoted": False,
            "idempotent": True,
        }
    staging = Path(
        tempfile.mkdtemp(prefix=f".{audit_hash[:12]}.", dir=date_root)
    )
    os.chmod(staging, 0o700)
    try:
        artifacts = _artifact_payloads(payload)
        for name, content in artifacts.items():
            _atomic_bytes(staging / name, content)
        manifest = {
            "schema_version": COVERAGE_MANIFEST_SCHEMA,
            "audit_hash": audit_hash,
            "generated_at": datetime.now(_UTC).isoformat(),
            "artifact_sha256": {
                name: _sha256(content) for name, content in artifacts.items()
            },
            "observer_only": True,
            "production_eligible": False,
            "applied": False,
        }
        _atomic_bytes(
            staging / "manifest.json",
            json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, indent=2
            ).encode("utf-8"),
        )
        _fsync_dir(staging)
        os.replace(staging, final)
        _fsync_dir(date_root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        **dict(payload),
        "output_dir": str(final),
        "promoted": False,
        "idempotent": False,
    }


def run_macro_coverage_audit(**kwargs: Any) -> dict[str, Any]:
    output_root = kwargs.pop("output_root", DEFAULT_OUTPUT_ROOT)
    payload = build_macro_coverage_audit(**kwargs)
    return persist_macro_coverage_audit(payload, output_root=output_root)


__all__ = [
    "COVERAGE_AUDIT_SCHEMA",
    "MacroCoverageError",
    "build_macro_coverage_audit",
    "persist_macro_coverage_audit",
    "run_macro_coverage_audit",
]
