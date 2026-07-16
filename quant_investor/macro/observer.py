"""Measurement-only macro observer orchestration and atomic reporting."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_investor.macro.contracts import (
    MacroObservation,
    MacroSnapshot,
    canonical_hash,
)
from quant_investor.macro.snapshot import build_macro_snapshot


STANDALONE_STAGING_SCHEMA = "macro-standalone-staging.v15"
DEFAULT_STANDALONE_STAGING_ROOT = Path(
    "results/v15/macro_observation_staging"
)
_SAFE_SLUG = re.compile(r"^[A-Za-z0-9_.-]+$")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise ValueError("macro_observer_output_symlink_rejected")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        os.chmod(path, 0o600)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_staging_slug(value: Any, *, blocker: str) -> str:
    slug = str(value or "").strip()
    if not slug or slug in {".", ".."} or _SAFE_SLUG.fullmatch(slug) is None:
        raise ValueError(blocker)
    return slug


def stage_standalone_macro_observations(
    observations: Iterable[Mapping[str, Any] | MacroObservation],
    *,
    market: str,
    as_of: str,
    run_id: str,
    output_root: str | Path = DEFAULT_STANDALONE_STAGING_ROOT,
) -> dict[str, Any]:
    """Persist only sanitized, non-canonical evidence for local input."""

    market_slug = _safe_staging_slug(
        str(market).upper(),
        blocker="macro_standalone_market_unsafe",
    )
    run_slug = _safe_staging_slug(
        run_id,
        blocker="macro_standalone_run_id_unsafe",
    )
    parsed_as_of = pd.to_datetime(str(as_of or ""), errors="coerce")
    if pd.isna(parsed_as_of):
        raise ValueError("macro_standalone_as_of_invalid")
    as_of_slug = pd.Timestamp(parsed_as_of).date().isoformat()

    rows_by_hash: dict[str, MacroObservation] = {}
    for value in observations:
        payload = value.to_dict() if isinstance(value, MacroObservation) else value
        item = MacroObservation.from_mapping(payload)
        rows_by_hash[item.content_hash] = item
    rows = [rows_by_hash[key] for key in sorted(rows_by_hash)]
    if not rows:
        return {
            "schema_version": STANDALONE_STAGING_SCHEMA,
            "status": "no_update",
            "promoted": False,
            "observer_only": True,
            "production_eligible": False,
            "applied": False,
            "reason": "empty_input",
        }

    snapshot = build_macro_snapshot(
        rows,
        market=market_slug,
        as_of=as_of_slug,
    )
    manifest = {
        "schema_version": STANDALONE_STAGING_SCHEMA,
        "status": "staged",
        "promoted": False,
        "observer_only": True,
        "production_eligible": False,
        "applied": False,
        "input_provenance": "manual_offline_snapshot",
        "provider_provenance_retained": False,
        "market": market_slug,
        "as_of": as_of_slug,
        "run_id": run_slug,
        "row_count": len(rows),
        "content_set_hash": canonical_hash(
            {"hashes": sorted(rows_by_hash)}
        ),
        "snapshot_hash": snapshot.snapshot_hash,
        "readiness_status": snapshot.readiness_status,
        "coverage": dict(snapshot.coverage),
        "freshness": dict(snapshot.freshness),
        "blockers": list(snapshot.blockers),
        "macro_score": snapshot.macro_score,
        "confidence": snapshot.confidence,
    }
    manifest_bytes = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )

    root = Path(output_root).expanduser()
    if root.exists() and (root.is_symlink() or not root.is_dir()):
        raise ValueError("macro_standalone_staging_root_unsafe")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    root = root.resolve()
    cursor = root
    for part in (market_slug, as_of_slug):
        cursor = cursor / part
        if cursor.exists() and (cursor.is_symlink() or not cursor.is_dir()):
            raise ValueError("macro_standalone_staging_descendant_unsafe")
        cursor.mkdir(exist_ok=True, mode=0o700)
        os.chmod(cursor, 0o700)
    final = cursor / run_slug
    if final.exists() or final.is_symlink():
        manifest_path = final / "manifest.json"
        if (
            final.is_symlink()
            or not final.is_dir()
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
            or manifest_path.read_bytes() != manifest_bytes
        ):
            raise ValueError("macro_standalone_staging_generation_exists")
        return {
            **manifest,
            "idempotent": True,
            "artifacts": {"manifest": str(manifest_path)},
        }

    staging = Path(
        tempfile.mkdtemp(prefix=f".{run_slug}.", dir=cursor)
    )
    os.chmod(staging, 0o700)
    try:
        manifest_path = staging / "manifest.json"
        _atomic_write(manifest_path, manifest_bytes)
        _fsync_directory(staging)
        os.replace(staging, final)
        _fsync_directory(cursor)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {
        **manifest,
        "idempotent": False,
        "artifacts": {"manifest": str(final / "manifest.json")},
    }


def render_macro_report(snapshot: MacroSnapshot) -> str:
    lines = [
        "# CN Macro v2 Observer",
        "",
        f"- Snapshot hash: `{snapshot.snapshot_hash}`",
        f"- As of: `{snapshot.as_of}`",
        f"- Published cutoff: `{snapshot.published_cutoff}`",
        f"- Readiness: `{snapshot.readiness_status}`",
        f"- Coverage: `{snapshot.coverage.get('national', 0.0):.1%}`",
        f"- Macro score: `{snapshot.macro_score:+.4f}`",
        f"- Confidence: `{snapshot.confidence:.1%}`",
        "- Production eligible: `false`",
        "- Production applied: `false`",
        "",
        "## National states",
        "",
        "| Domain | State |",
        "| --- | ---: |",
    ]
    for domain, value in sorted(snapshot.national_states.items()):
        lines.append(f"| {domain} | {float(value):+.4f} |")
    lines.extend(["", "## Industry chains", "", "| Chain | State | Coverage | Shadow delta | Applied |", "| --- | ---: | ---: | ---: | --- |"])
    chain_coverage = dict(snapshot.coverage.get("industry_chains", {}) or {})
    for chain, value in sorted(snapshot.industry_chain_states.items()):
        overlay = dict(snapshot.shadow_overlays.get(chain, {}) or {})
        lines.append(
            f"| {chain} | {float(value):+.4f} | {float(chain_coverage.get(chain, 0.0)):.1%} | "
            f"{float(overlay.get('delta_points', 0.0)):+.3f} | false |"
        )
    lines.extend(["", "## Blockers", ""])
    lines.extend(f"- `{item}`" for item in snapshot.blockers)
    if not snapshot.blockers:
        lines.append("- None")
    lines.extend(["", "## Selected observations", "", "| Indicator | Period | Release | Available | Vintage | Source | Fallback |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for indicator_id, payload in sorted(snapshot.source_lineage.items()):
        row = dict(payload)
        lines.append(
            f"| {indicator_id} | {row.get('period_end', '')} | {row.get('release_at', '')} | "
            f"{row.get('available_at', '')} | {row.get('vintage_id', '')} | "
            f"{row.get('source_system', '')} | {str(bool(row.get('fallback'))).lower()} |"
        )
    return "\n".join(lines) + "\n"


def persist_macro_observer(
    snapshot: MacroSnapshot,
    *,
    output_root: str | Path = "results/macro_observer",
    production_enabled: bool = False,
    production_kill_switch: bool = True,
    generation_provenance: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    root = Path(output_root).expanduser()
    if root.exists() and root.is_symlink():
        raise ValueError("macro_observer_root_symlink_rejected")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    market_slug = str(snapshot.market).upper()
    if not re.fullmatch(r"[A-Z0-9_-]+", market_slug):
        raise ValueError("macro_observer_market_unsafe")
    as_of_slug = str(snapshot.as_of).split("T", 1)[0]
    if not re.fullmatch(r"[0-9-]+", as_of_slug):
        raise ValueError("macro_observer_as_of_unsafe")
    generation_slug = str((generation_provenance or {}).get("generation_id") or "")
    if generation_slug and (
        generation_slug in {".", ".."}
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", generation_slug)
    ):
        raise ValueError("macro_observer_generation_id_unsafe")
    relative_parts = [market_slug, as_of_slug, snapshot.snapshot_hash]
    if generation_slug:
        relative_parts.append(generation_slug)
    out = root.joinpath(*relative_parts)
    cursor = root
    for part in relative_parts:
        cursor = cursor / part
        if cursor.exists() and cursor.is_symlink():
            raise ValueError("macro_observer_descendant_symlink_rejected")
    resolved_root = root.resolve()
    resolved_out = out.resolve()
    if resolved_root not in resolved_out.parents:
        raise ValueError("macro_observer_output_escape_rejected")
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    payload = {
        **snapshot.to_dict(),
        "observer_only": True,
        "production_eligible": False,
        "applied": False,
    }
    readiness = {
        "schema_version": "macro-readiness.v2",
        "snapshot_hash": snapshot.snapshot_hash,
        "status": snapshot.readiness_status,
        "coverage": dict(snapshot.coverage),
        "freshness": dict(snapshot.freshness),
        "blockers": list(snapshot.blockers),
        "production_enabled": bool(production_enabled),
        "production_kill_switch": bool(production_kill_switch),
        "observer_only": True,
        "production_eligible": False,
        "applied": False,
    }
    manifest = {
        "schema_version": "macro-observation-manifest.v2",
        "snapshot_hash": snapshot.snapshot_hash,
        "selected_observation_hashes": list(snapshot.selected_observation_hashes),
        "registry_version": snapshot.registry_version,
        "score_model_version": snapshot.score_model_version,
        "observation_generation": dict(generation_provenance or {}),
        "observer_only": True,
        "production_eligible": False,
        "applied": False,
    }
    files = {
        "snapshot": out / "macro_snapshot.json",
        "readiness": out / "macro_readiness.json",
        "report": out / "macro_report.md",
        "manifest": out / "macro_observations_manifest.json",
    }
    _atomic_write(files["snapshot"], json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
    _atomic_write(files["readiness"], json.dumps(readiness, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
    _atomic_write(files["manifest"], json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
    _atomic_write(files["report"], render_macro_report(snapshot).encode("utf-8"))
    return {key: str(path) for key, path in files.items()}


def build_macro_observer(
    observations: Iterable[Mapping[str, Any] | MacroObservation],
    *,
    market: str = "CN",
    as_of: str,
    enabled: bool = False,
    kill_switch: bool = True,
    persist: bool = False,
    output_root: str | Path = "results/macro_observer",
    production_enabled: bool = False,
    production_kill_switch: bool = True,
    generation_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    active = bool(enabled) and not bool(kill_switch)
    if not active:
        return {
            "schema_version": "macro-observer-runtime.v2",
            "enabled": bool(enabled),
            "kill_switch": bool(kill_switch),
            "active": False,
            "production_enabled": bool(production_enabled),
            "production_kill_switch": bool(production_kill_switch),
            "observer_only": True,
            "production_eligible": False,
            "applied": False,
            "reason": "kill_switch_active" if kill_switch else "observer_disabled",
        }
    snapshot = build_macro_snapshot(observations, market=market, as_of=as_of)
    artifacts = (
        persist_macro_observer(
            snapshot,
            output_root=output_root,
            production_enabled=production_enabled,
            production_kill_switch=production_kill_switch,
            generation_provenance=generation_provenance,
        )
        if persist
        else {}
    )
    return {
        "schema_version": "macro-observer-runtime.v2",
        "enabled": True,
        "kill_switch": False,
        "active": True,
        "production_enabled": bool(production_enabled),
        "production_kill_switch": bool(production_kill_switch),
        "observer_only": True,
        "production_eligible": False,
        "applied": False,
        "snapshot": snapshot.to_dict(),
        "snapshot_hash": snapshot.snapshot_hash,
        "observation_generation": dict(generation_provenance or {}),
        "artifacts": artifacts,
    }


def load_macro_observations(
    path: str | Path,
    *,
    allow_standalone_offline: bool = False,
) -> list[dict[str, Any]]:
    rows, _ = load_macro_observation_generation(
        path,
        allow_standalone_offline=allow_standalone_offline,
    )
    return rows


def load_macro_observation_generation(
    path: str | Path,
    *,
    allow_standalone_offline: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = Path(path).expanduser()
    if source.exists() and source.is_dir() and not source.is_symlink():
        from quant_investor.macro.store import load_observations

        generation_rows, generation = load_observations(source)
        if not generation:
            raise ValueError("macro_observation_generation_missing")
        return generation_rows, generation
    if not allow_standalone_offline:
        raise ValueError("macro_standalone_observations_disabled")
    if not source.exists() or source.is_symlink() or not source.is_file():
        raise ValueError("macro_observations_file_missing_or_unsafe")
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return [dict(item) for item in pd.read_parquet(source).to_dict(orient="records")], {}
    if suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError("macro_observation_jsonl_row_not_object")
            rows.append(dict(payload))
        return rows, {}
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get("observations", [])
        if not isinstance(payload, list) or not all(isinstance(item, Mapping) for item in payload):
            raise ValueError("macro_observations_json_invalid")
        return [dict(item) for item in payload], {}
    raise ValueError("macro_observations_format_unsupported")


__all__ = [
    "DEFAULT_STANDALONE_STAGING_ROOT",
    "STANDALONE_STAGING_SCHEMA",
    "build_macro_observer",
    "persist_macro_observer",
    "render_macro_report",
    "load_macro_observations",
    "load_macro_observation_generation",
    "stage_standalone_macro_observations",
]
