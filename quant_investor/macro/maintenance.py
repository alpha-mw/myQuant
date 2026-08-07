"""Registered fail-closed CN Macro and release-calendar maintenance path."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from quant_investor.macro.contracts import normalize_source_url
from quant_investor.macro.production_observation_bundle import (
    publish_local_market_breadth_roll,
)
from quant_investor.macro.release_calendar import (
    load_release_calendar,
    publish_release_calendar,
)
from quant_investor.macro.store import pointer_sha256 as observation_pointer_sha256


NBS_COVERAGE_URL = "https://www.stats.gov.cn/sj/zxfbhjd/"
PBC_COVERAGE_URL = "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/index.html"
MAX_COVERAGE_RESPONSE_BYTES = 8 * 1024 * 1024


class MacroMaintenanceError(RuntimeError):
    """Raised before a governed pointer write when maintenance cannot close."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_bytes(path: str | Path, expected_sha256: str, blocker: str) -> bytes:
    source = Path(path).expanduser()
    if source.is_symlink():
        raise MacroMaintenanceError(blocker)
    resolved = source.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_nlink != 1:
        raise MacroMaintenanceError(blocker)
    raw = resolved.read_bytes()
    if _sha256_bytes(raw) != str(expected_sha256).strip().lower():
        raise MacroMaintenanceError(f"{blocker}_sha256_mismatch")
    return raw


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _default_fetch(url: str, issuer: str) -> tuple[bytes, str]:
    normalized = normalize_source_url(url, source_system=issuer)
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(
            normalized,
            allow_redirects=False,
            timeout=(5.0, 30.0),
            headers={"User-Agent": "QuantInvestor/17 official-coverage-capture"},
        )
        if response.status_code != 200:
            raise MacroMaintenanceError(
                f"macro_coverage_http_status_invalid:{issuer}:{response.status_code}"
            )
        body = bytes(response.content)
    finally:
        session.close()
    if not 1 <= len(body) <= MAX_COVERAGE_RESPONSE_BYTES:
        raise MacroMaintenanceError(f"macro_coverage_response_size_invalid:{issuer}")
    completed_at = datetime.now(timezone.utc).isoformat()
    return body, completed_at


def run_cn_macro_maintenance(
    *,
    market: str,
    target_date: str,
    snapshot_manifest_path: str | Path,
    expected_snapshot_manifest_sha256: str,
    coverage_manifest_path: str | Path,
    expected_coverage_manifest_sha256: str,
    scope_artifact_path: str | Path,
    expected_scope_artifact_sha256: str,
    release_root: str | Path,
    expected_release_pointer_sha256: str,
    observations_root: str | Path,
    expected_observations_pointer_sha256: str,
    release_run_id: str,
    observations_run_id: str,
    allow_live: bool = False,
    commit: bool = False,
    fetcher: Callable[[str, str], tuple[bytes, str]] | None = None,
) -> dict[str, Any]:
    """Extend issuer coverage, then roll the exact local breadth observation.

    The unified cutoff is frozen after both official response entities have
    been captured and before either canonical pointer is written.
    """

    if str(market).upper() != "CN":
        raise MacroMaintenanceError("macro_maintenance_market_unsupported")
    target = str(target_date).replace("-", "")
    if len(target) != 8 or not target.isdigit():
        raise MacroMaintenanceError("macro_maintenance_target_date_invalid")
    for path, digest, blocker in (
        (snapshot_manifest_path, expected_snapshot_manifest_sha256, "macro_snapshot_manifest_invalid"),
        (coverage_manifest_path, expected_coverage_manifest_sha256, "macro_coverage_manifest_invalid"),
        (scope_artifact_path, expected_scope_artifact_sha256, "macro_scope_artifact_invalid"),
    ):
        _file_bytes(path, digest, blocker)
    release_root = Path(release_root).resolve(strict=True)
    observations_root = Path(observations_root).resolve(strict=True)
    release = load_release_calendar(
        canonical_root=release_root,
        expected_pointer_sha256=expected_release_pointer_sha256,
    )
    if observation_pointer_sha256(observations_root) != expected_observations_pointer_sha256:
        raise MacroMaintenanceError("macro_observations_pointer_cas_mismatch")
    if not commit:
        return {
            "schema_version": "cn-macro-maintenance-receipt.v1",
            "status": "DRY_RUN_OK",
            "promoted": False,
            "target_date": target,
            "release_parent": asdict(release.identity),
            "expected_observations_pointer_sha256": expected_observations_pointer_sha256,
            "writes": [],
        }
    if not allow_live:
        raise MacroMaintenanceError("macro_maintenance_allow_live_required")

    acquire = fetcher or _default_fetch
    captures: list[tuple[str, str, bytes, str]] = []
    for issuer, url in (
        ("nbs_official", NBS_COVERAGE_URL),
        ("pbc_official", PBC_COVERAGE_URL),
    ):
        body, completed_at = acquire(url, issuer)
        if not isinstance(body, bytes) or not 1 <= len(body) <= MAX_COVERAGE_RESPONSE_BYTES:
            raise MacroMaintenanceError(
                f"macro_coverage_response_size_invalid:{issuer}"
            )
        try:
            completed = datetime.fromisoformat(
                str(completed_at).strip().replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise MacroMaintenanceError(
                f"macro_coverage_completed_at_invalid:{issuer}"
            ) from exc
        if completed.tzinfo is None:
            raise MacroMaintenanceError(
                f"macro_coverage_completed_at_invalid:{issuer}"
            )
        normalized_completed_at = completed.astimezone(timezone.utc).isoformat()
        captures.append(
            (
                issuer,
                normalize_source_url(url, source_system=issuer),
                body,
                normalized_completed_at,
            )
        )
    cutoff_at = max(item[3] for item in captures)

    generation_root = Path(release.identity.generation_path)
    with tempfile.TemporaryDirectory(prefix="cn-macro-maintenance-") as temporary:
        stage = Path(temporary)
        plan_path = stage / "plan.json"
        open_days_path = stage / "market_open_days.json"
        capture_path = stage / "capture_manifest.json"
        raw_root = stage / "raw"
        shutil.copy2(generation_root / "plan.json", plan_path)
        shutil.copy2(generation_root / "market_open_days.json", open_days_path)
        shutil.copytree(generation_root / "raw", raw_root)
        capture_payload = json.loads(
            (generation_root / "capture_manifest.json").read_text(encoding="utf-8")
        )
        run_token = hashlib.sha256(release_run_id.encode()).hexdigest()[:10]
        for issuer, url, body, completed_at in captures:
            short = "nbs" if issuer == "nbs_official" else "pbc"
            response_id = f"coverage-response-{short}-{target}-{run_token}"
            receipt_id = f"coverage-{short}-{target}-{run_token}"
            response_relative = f"coverage_responses/{short}_{target}_{run_token}.html"
            receipt_relative = f"coverage/{short}_{target}_{run_token}.json"
            response_path = raw_root / response_relative
            receipt_path = raw_root / receipt_relative
            response_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_bytes(body)
            response_sha = _sha256_bytes(body)
            receipt_raw = _json_bytes({
                "schema_version": "macro-release-issuer-coverage.v2",
                "issuer": issuer,
                "through": cutoff_at,
                "response_source_id": response_id,
                "response_sha256": response_sha,
                "response_size_bytes": len(body),
            })
            receipt_path.write_bytes(receipt_raw)
            capture_payload["sources"].extend([
                {
                    "source_id": response_id, "issuer": issuer,
                    "artifact_kind": "coverage_response", "source_url": url,
                    "http_status": 200, "captured_at": cutoff_at,
                    "raw_path": response_relative, "raw_sha256": response_sha,
                    "size_bytes": len(body), "content_sha256": response_sha,
                },
                {
                    "source_id": receipt_id, "issuer": issuer,
                    "artifact_kind": "coverage_receipt", "source_url": url,
                    "http_status": 200, "captured_at": cutoff_at,
                    "raw_path": receipt_relative,
                    "raw_sha256": _sha256_bytes(receipt_raw),
                    "size_bytes": len(receipt_raw),
                    "content_sha256": _sha256_bytes(receipt_raw),
                },
            ])
            coverage = next(item for item in capture_payload["issuer_coverage"] if item["issuer"] == issuer)
            coverage["through"] = cutoff_at
            coverage["source_ids"].append(receipt_id)
        capture_payload["captured_at"] = cutoff_at
        capture_raw = _json_bytes(capture_payload)
        capture_path.write_bytes(capture_raw)
        plan_sha = _sha256_bytes(plan_path.read_bytes())
        open_days_sha = _sha256_bytes(open_days_path.read_bytes())
        release_result = publish_release_calendar(
            plan_path=plan_path,
            expected_plan_sha256=plan_sha,
            capture_manifest_path=capture_path,
            expected_capture_manifest_sha256=_sha256_bytes(capture_raw),
            raw_root=raw_root,
            market_open_days_path=open_days_path,
            expected_market_open_days_sha256=open_days_sha,
            canonical_root=release_root,
            run_id=release_run_id,
            expected_pointer_sha256=expected_release_pointer_sha256,
        )

    try:
        release_open_days_path = (
            Path(release_result.identity.generation_path) / "market_open_days.json"
        )
        pinned_open_dates = tuple(
            json.loads(release_open_days_path.read_text(encoding="utf-8"))["open_dates"]
        )
        observation_result = publish_local_market_breadth_roll(
            snapshot_manifest_path=snapshot_manifest_path,
            expected_snapshot_manifest_sha256=expected_snapshot_manifest_sha256,
            coverage_manifest_path=coverage_manifest_path,
            expected_coverage_manifest_sha256=expected_coverage_manifest_sha256,
            target_trade_date=target,
            scope_artifact_path=scope_artifact_path,
            expected_scope_artifact_sha256=expected_scope_artifact_sha256,
            target_as_of=target,
            decision_cutoff_at=cutoff_at,
            pinned_open_dates=pinned_open_dates,
            market_open_days_path=release_open_days_path,
            expected_market_open_days_sha256=release_result.evidence.market_open_days_sha256,
            canonical_observations_root=observations_root,
            run_id=observations_run_id,
            expected_pointer_sha256=expected_observations_pointer_sha256,
        )
    except Exception as exc:
        return {
            "schema_version": "cn-macro-maintenance-receipt.v1",
            "status": "PARTIAL",
            "promoted": False,
            "target_date": target,
            "cutoff_at": cutoff_at,
            "release": asdict(release_result.identity),
            "blockers": [f"macro_observation_roll_failed:{type(exc).__name__}:{exc}"],
        }
    return {
        "schema_version": "cn-macro-maintenance-receipt.v1",
        "status": "OK",
        "promoted": True,
        "target_date": target,
        "cutoff_at": cutoff_at,
        "release": asdict(release_result.identity),
        "observations": observation_result,
        "provider_calls": [
            {"issuer": issuer, "url": url, "response_sha256": _sha256_bytes(body), "size_bytes": len(body)}
            for issuer, url, body, _completed in captures
        ],
        "blockers": [],
    }


__all__ = ["MacroMaintenanceError", "run_cn_macro_maintenance"]
