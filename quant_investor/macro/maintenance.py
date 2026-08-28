"""Registered fail-closed CN Macro and release-calendar maintenance path."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import shutil
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from quant_investor.macro.contracts import normalize_source_url
from quant_investor.macro.maintenance_transaction import (
    MacroMaintenanceTransactionError,
    commit_prepared_macro_transaction,
    recover_macro_transaction,
    rollback_macro_transaction,
    seal_prepared_macro_transaction,
)
from quant_investor.macro.production_observation_bundle import (
    publish_local_market_breadth_roll,
)
from quant_investor.macro.release_calendar import (
    load_release_calendar,
    publish_release_calendar,
)
from quant_investor.macro.store import (
    load_observations,
    pointer_sha256 as observation_pointer_sha256,
)

NBS_COVERAGE_URL = "https://www.stats.gov.cn/sj/zxfbhjd/"
PBC_COVERAGE_URL = "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/index.html"
MAX_COVERAGE_RESPONSE_BYTES = 8 * 1024 * 1024


class MacroMaintenanceError(RuntimeError):
    """Raised before a governed pointer write when maintenance cannot close."""


def _private_preparation_root(path: str | Path) -> Path:
    unresolved = Path(path).expanduser()
    if not unresolved.is_absolute():
        raise MacroMaintenanceError("macro_preparation_root_must_be_absolute")
    try:
        current = os.lstat(unresolved)
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise MacroMaintenanceError("macro_preparation_root_invalid") from exc
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISDIR(current.st_mode)
        or stat.S_IMODE(current.st_mode) & 0o077
    ):
        raise MacroMaintenanceError("macro_preparation_root_not_private")
    return resolved


def _copy_canonical_parent(source: Path, destination: Path) -> None:
    """Copy only canonical generations and the exact pointer to a private root."""

    destination.mkdir(mode=0o700)
    generations = source / "_generations"
    if generations.is_symlink() or not generations.is_dir():
        raise MacroMaintenanceError("macro_canonical_generations_root_invalid")
    for item in generations.rglob("*"):
        current = os.lstat(item)
        if stat.S_ISLNK(current.st_mode) or not (
            stat.S_ISDIR(current.st_mode)
            or (stat.S_ISREG(current.st_mode) and current.st_nlink == 1)
        ):
            raise MacroMaintenanceError("macro_canonical_generation_tree_unsafe")
    shutil.copytree(generations, destination / "_generations", symlinks=False)
    pointer = source / "_latest.json"
    try:
        pointer_stat = os.lstat(pointer)
    except OSError as exc:
        raise MacroMaintenanceError("macro_canonical_pointer_invalid") from exc
    if (
        stat.S_ISLNK(pointer_stat.st_mode)
        or not stat.S_ISREG(pointer_stat.st_mode)
        or pointer_stat.st_nlink != 1
    ):
        raise MacroMaintenanceError("macro_canonical_pointer_invalid")
    shutil.copy2(pointer, destination / "_latest.json")
    for item in destination.rglob("*"):
        os.chmod(item, 0o700 if item.is_dir() else 0o600)
    os.chmod(destination, 0o700)


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


def _expected_retrospective_coverage_targets(
    *,
    release_root: Path,
    expected_release_pointer_sha256: str,
    observations_root: Path,
    expected_observations_pointer_sha256: str,
    target_date: str,
) -> list[str]:
    """Derive exact historical catch-up dates from frozen canonical parents."""

    release = load_release_calendar(
        canonical_root=release_root,
        expected_pointer_sha256=expected_release_pointer_sha256,
    )
    if observation_pointer_sha256(observations_root) != expected_observations_pointer_sha256:
        raise MacroMaintenanceError("macro_observations_pointer_cas_mismatch")
    _rows, projection = load_observations(observations_root)
    manifest = dict(projection.get("generation_manifest") or {})
    metadata = dict(manifest.get("metadata") or projection.get("metadata") or {})
    parent_target = str(metadata.get("local_target_trade_date") or "")
    target = str(target_date).replace("-", "")
    open_days_path = Path(release.identity.generation_path) / "market_open_days.json"
    try:
        pinned_open_dates = tuple(
            json.loads(open_days_path.read_text(encoding="utf-8"))["open_dates"]
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise MacroMaintenanceError("macro_release_open_days_invalid") from exc
    catch_up_targets = [value for value in pinned_open_dates if parent_target < value <= target]
    if (
        len(parent_target) != 8
        or not parent_target.isdigit()
        or not catch_up_targets
        or len(catch_up_targets) > 5
        or catch_up_targets[-1] != target
    ):
        raise MacroMaintenanceError("macro_observation_catch_up_window_invalid")
    return catch_up_targets[:-1]


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
    retrospective_coverage_by_target: Mapping[str, Mapping[str, str]] | None = None,
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
        (
            snapshot_manifest_path,
            expected_snapshot_manifest_sha256,
            "macro_snapshot_manifest_invalid",
        ),
        (
            coverage_manifest_path,
            expected_coverage_manifest_sha256,
            "macro_coverage_manifest_invalid",
        ),
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
            raise MacroMaintenanceError(f"macro_coverage_response_size_invalid:{issuer}")
        try:
            completed = datetime.fromisoformat(str(completed_at).strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise MacroMaintenanceError(f"macro_coverage_completed_at_invalid:{issuer}") from exc
        if completed.tzinfo is None:
            raise MacroMaintenanceError(f"macro_coverage_completed_at_invalid:{issuer}")
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
        stage = Path(temporary).resolve(strict=True)
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
            receipt_raw = _json_bytes(
                {
                    "schema_version": "macro-release-issuer-coverage.v2",
                    "issuer": issuer,
                    "through": cutoff_at,
                    "response_source_id": response_id,
                    "response_sha256": response_sha,
                    "response_size_bytes": len(body),
                }
            )
            receipt_path.write_bytes(receipt_raw)
            capture_payload["sources"].extend(
                [
                    {
                        "source_id": response_id,
                        "issuer": issuer,
                        "artifact_kind": "coverage_response",
                        "source_url": url,
                        "http_status": 200,
                        "captured_at": cutoff_at,
                        "raw_path": response_relative,
                        "raw_sha256": response_sha,
                        "size_bytes": len(body),
                        "content_sha256": response_sha,
                    },
                    {
                        "source_id": receipt_id,
                        "issuer": issuer,
                        "artifact_kind": "coverage_receipt",
                        "source_url": url,
                        "http_status": 200,
                        "captured_at": cutoff_at,
                        "raw_path": receipt_relative,
                        "raw_sha256": _sha256_bytes(receipt_raw),
                        "size_bytes": len(receipt_raw),
                        "content_sha256": _sha256_bytes(receipt_raw),
                    },
                ]
            )
            coverage = next(
                item for item in capture_payload["issuer_coverage"] if item["issuer"] == issuer
            )
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
        _existing_rows, existing_projection = load_observations(observations_root)
        existing_manifest = dict(existing_projection.get("generation_manifest") or {})
        existing_metadata = dict(
            existing_manifest.get("metadata") or existing_projection.get("metadata") or {}
        )
        parent_target = str(existing_metadata.get("local_target_trade_date") or "")
        catch_up_targets = [value for value in pinned_open_dates if parent_target < value <= target]
        if not catch_up_targets or len(catch_up_targets) > 5:
            raise MacroMaintenanceError("macro_observation_catch_up_window_invalid")
        observation_results: list[dict[str, Any]] = []
        expected_observation_pointer = expected_observations_pointer_sha256
        for catch_up_target in catch_up_targets:
            target_coverage_path: str | Path
            retrospective = (
                retrospective_coverage_by_target.get(catch_up_target)
                if retrospective_coverage_by_target is not None
                else None
            )
            if retrospective is not None:
                if set(retrospective) != {"path", "sha256"}:
                    raise MacroMaintenanceError("macro_retrospective_coverage_binding_invalid")
                target_coverage_path = retrospective["path"]
                target_coverage_sha256 = retrospective["sha256"]
            else:
                target_coverage_path = coverage_manifest_path
                target_coverage_sha256 = expected_coverage_manifest_sha256
            observation_result = publish_local_market_breadth_roll(
                snapshot_manifest_path=snapshot_manifest_path,
                expected_snapshot_manifest_sha256=expected_snapshot_manifest_sha256,
                coverage_manifest_path=target_coverage_path,
                expected_coverage_manifest_sha256=target_coverage_sha256,
                target_trade_date=catch_up_target,
                scope_artifact_path=scope_artifact_path,
                expected_scope_artifact_sha256=expected_scope_artifact_sha256,
                target_as_of=catch_up_target,
                decision_cutoff_at=cutoff_at,
                pinned_open_dates=pinned_open_dates,
                market_open_days_path=release_open_days_path,
                expected_market_open_days_sha256=(release_result.evidence.market_open_days_sha256),
                canonical_observations_root=observations_root,
                run_id=f"{observations_run_id}-{catch_up_target}",
                expected_pointer_sha256=expected_observation_pointer,
            )
            observation_results.append(observation_result)
            expected_observation_pointer = observation_pointer_sha256(observations_root)
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
        "observation_catch_up_targets": catch_up_targets,
        "observation_results": observation_results,
        "provider_calls": [
            {
                "issuer": issuer,
                "url": url,
                "response_sha256": _sha256_bytes(body),
                "size_bytes": len(body),
            }
            for issuer, url, body, _completed in captures
        ],
        "blockers": [],
    }


def prepare_cn_macro_maintenance_transaction(
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
    market_pointer_path: str | Path,
    expected_market_pointer_sha256: str,
    pit_pointer_path: str | Path,
    expected_pit_pointer_sha256: str,
    authority_mode: str,
    release_run_id: str,
    observations_run_id: str,
    private_run_root: str | Path,
    transaction_run_id: str,
    allow_live: bool = False,
    fetcher: Callable[[str, str], tuple[bytes, str]] | None = None,
    retrospective_recovery_contract_path: str | Path | None = None,
    expected_retrospective_recovery_contract_sha256: str | None = None,
) -> dict[str, Any]:
    """Prepare both component-owned candidates without canonical pointer writes.

    The existing publishers run against isolated copies of the two canonical
    parents.  Their private pointers are candidate artifacts only; canonical
    generation installation and pointer writes are reserved for the journaled
    commit API.  ``authority_mode='candidate'`` is shadow-only; only a prepared
    transaction sealed against exact canonical Market and PIT pointers is
    executable.
    """

    if not allow_live:
        raise MacroMaintenanceError("macro_maintenance_allow_live_required")
    parent = _private_preparation_root(private_run_root)
    token = str(transaction_run_id or "")
    if (
        not token
        or len(token) > 80
        or not token[0].isalnum()
        or any(
            character
            not in ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ" "0123456789_.-")
            for character in token
        )
    ):
        raise MacroMaintenanceError("macro_transaction_run_id_invalid")
    run_root = parent / token
    try:
        run_root.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise MacroMaintenanceError("macro_transaction_prepare_no_clobber") from exc

    release_canonical = Path(release_root).expanduser().resolve(strict=True)
    observations_canonical = Path(observations_root).expanduser().resolve(strict=True)
    release_candidate = run_root / "release_candidate"
    observations_candidate = run_root / "observations_candidate"
    prepared_root = run_root / "prepared"
    try:
        retrospective_contract: dict[str, Any] | None = None
        retrospective_coverage_by_target: dict[str, dict[str, str]] | None = None
        retrospective_input_bindings: dict[str, dict[str, str]] = {}
        if retrospective_recovery_contract_path is not None:
            if expected_retrospective_recovery_contract_sha256 is None:
                raise MacroMaintenanceError("macro_retrospective_contract_sha_required")
            contract_raw = _file_bytes(
                retrospective_recovery_contract_path,
                expected_retrospective_recovery_contract_sha256,
                "macro_retrospective_contract_invalid",
            )
            try:
                retrospective_contract = json.loads(contract_raw)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise MacroMaintenanceError("macro_retrospective_contract_invalid") from exc
            if (
                not isinstance(retrospective_contract, dict)
                or set(retrospective_contract)
                != {
                    "schema_version",
                    "status",
                    "classification",
                    "target_date",
                    "attempt_receipt",
                    "candidate_manifest",
                    "source_market_manifest",
                    "canonical_market_pointer_sha256",
                    "canonical_pit_pointer_sha256",
                    "expected_macro_observations_pointer_sha256",
                    "expected_macro_release_pointer_sha256",
                    "macro_veto",
                    "retrospective_coverage_manifests",
                    "canonical_final_coverage_manifest",
                    "content_sha256",
                }
                or retrospective_contract.get("schema_version")
                != "macro-retrospective-canonical-transaction.v1"
                or retrospective_contract.get("classification")
                != "MIXED_RETROSPECTIVE_AND_CANONICAL"
                or retrospective_contract.get("target_date") != str(target_date).replace("-", "")
                or retrospective_contract.get("status") != "PREPARED"
            ):
                raise MacroMaintenanceError("macro_retrospective_contract_invalid")
            contract_body = dict(retrospective_contract)
            contract_content_sha = contract_body.pop("content_sha256")
            if contract_content_sha != _sha256_bytes(
                json.dumps(
                    contract_body,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ):
                raise MacroMaintenanceError("macro_retrospective_contract_hash_invalid")
            if (
                retrospective_contract["canonical_market_pointer_sha256"]
                != expected_market_pointer_sha256
                or retrospective_contract["canonical_pit_pointer_sha256"]
                != expected_pit_pointer_sha256
                or retrospective_contract["expected_macro_observations_pointer_sha256"]
                != expected_observations_pointer_sha256
                or retrospective_contract["expected_macro_release_pointer_sha256"]
                != expected_release_pointer_sha256
            ):
                raise MacroMaintenanceError("macro_retrospective_contract_preimage_mismatch")
            rows = retrospective_contract.get("retrospective_coverage_manifests")
            expected_historical_targets = _expected_retrospective_coverage_targets(
                release_root=release_canonical,
                expected_release_pointer_sha256=expected_release_pointer_sha256,
                observations_root=observations_canonical,
                expected_observations_pointer_sha256=expected_observations_pointer_sha256,
                target_date=target_date,
            )
            if (
                not isinstance(rows, list)
                or [row.get("target_trade_date") for row in rows] != expected_historical_targets
            ):
                raise MacroMaintenanceError("macro_retrospective_contract_targets_invalid")
            retrospective_coverage_by_target = {
                row["target_trade_date"]: {"path": row["path"], "sha256": row["sha256"]}
                for row in rows
            }
            for name in (
                "attempt_receipt",
                "candidate_manifest",
                "source_market_manifest",
                "macro_veto",
                "canonical_final_coverage_manifest",
            ):
                ref = retrospective_contract[name]
                if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
                    raise MacroMaintenanceError("macro_retrospective_contract_ref_invalid")
                _file_bytes(ref["path"], ref["sha256"], "macro_retrospective_contract_ref_invalid")
                retrospective_input_bindings[name] = dict(ref)
            if retrospective_contract["source_market_manifest"] != {
                "path": str(Path(snapshot_manifest_path).expanduser().resolve()),
                "sha256": expected_snapshot_manifest_sha256,
            } or retrospective_contract["canonical_final_coverage_manifest"] != {
                "path": str(Path(coverage_manifest_path).expanduser().resolve()),
                "sha256": expected_coverage_manifest_sha256,
            }:
                raise MacroMaintenanceError("macro_retrospective_contract_market_binding_mismatch")
            for row in rows:
                _file_bytes(row["path"], row["sha256"], "macro_retrospective_coverage_invalid")
                retrospective_input_bindings[
                    "retrospective_coverage_" + row["target_trade_date"]
                ] = {"path": row["path"], "sha256": row["sha256"]}
        market_authority_raw = _file_bytes(
            market_pointer_path,
            expected_market_pointer_sha256,
            "macro_market_authority_pointer_invalid",
        )
        pit_authority_raw = _file_bytes(
            pit_pointer_path,
            expected_pit_pointer_sha256,
            "macro_pit_authority_pointer_invalid",
        )
        _copy_canonical_parent(release_canonical, release_candidate)
        _copy_canonical_parent(observations_canonical, observations_candidate)
        prepared_root.mkdir(mode=0o700)
        legacy_result = run_cn_macro_maintenance(
            market=market,
            target_date=target_date,
            snapshot_manifest_path=snapshot_manifest_path,
            expected_snapshot_manifest_sha256=expected_snapshot_manifest_sha256,
            coverage_manifest_path=coverage_manifest_path,
            expected_coverage_manifest_sha256=expected_coverage_manifest_sha256,
            scope_artifact_path=scope_artifact_path,
            expected_scope_artifact_sha256=expected_scope_artifact_sha256,
            release_root=release_candidate,
            expected_release_pointer_sha256=expected_release_pointer_sha256,
            observations_root=observations_candidate,
            expected_observations_pointer_sha256=expected_observations_pointer_sha256,
            release_run_id=release_run_id,
            observations_run_id=observations_run_id,
            allow_live=True,
            commit=True,
            fetcher=fetcher,
            retrospective_coverage_by_target=retrospective_coverage_by_target,
        )
        if legacy_result.get("status") != "OK":
            raise MacroMaintenanceError(
                "macro_transaction_candidate_preparation_incomplete:"
                + ":".join(str(item) for item in legacy_result.get("blockers", ()))
            )
        if (
            _file_bytes(
                market_pointer_path,
                expected_market_pointer_sha256,
                "macro_market_authority_pointer_invalid",
            )
            != market_authority_raw
        ):
            raise MacroMaintenanceError("macro_market_authority_pointer_drift")
        if (
            _file_bytes(
                pit_pointer_path,
                expected_pit_pointer_sha256,
                "macro_pit_authority_pointer_invalid",
            )
            != pit_authority_raw
        ):
            raise MacroMaintenanceError("macro_pit_authority_pointer_drift")
        sealed = seal_prepared_macro_transaction(
            prepared_root=prepared_root,
            release_candidate_root=release_candidate,
            observations_candidate_root=observations_candidate,
            release_canonical_root=release_canonical,
            observations_canonical_root=observations_canonical,
            expected_release_pointer_sha256=expected_release_pointer_sha256,
            expected_observations_pointer_sha256=expected_observations_pointer_sha256,
            market_pointer_path=market_pointer_path,
            expected_market_pointer_sha256=expected_market_pointer_sha256,
            pit_pointer_path=pit_pointer_path,
            expected_pit_pointer_sha256=expected_pit_pointer_sha256,
            authority_mode=authority_mode,
            target_date=target_date,
            input_bindings={
                "snapshot_manifest": {
                    "path": str(Path(snapshot_manifest_path).expanduser().resolve()),
                    "sha256": expected_snapshot_manifest_sha256,
                },
                "coverage_manifest": {
                    "path": str(Path(coverage_manifest_path).expanduser().resolve()),
                    "sha256": expected_coverage_manifest_sha256,
                },
                "scope_artifact": {
                    "path": str(Path(scope_artifact_path).expanduser().resolve()),
                    "sha256": expected_scope_artifact_sha256,
                },
                **(
                    {
                        "retrospective_recovery_contract": {
                            "path": str(
                                Path(retrospective_recovery_contract_path).expanduser().resolve()
                            ),
                            "sha256": expected_retrospective_recovery_contract_sha256,
                        }
                    }
                    if retrospective_contract is not None
                    else {}
                ),
                **retrospective_input_bindings,
            },
        )
    except Exception:
        # Evidence is intentionally retained for diagnosis.  It is isolated,
        # private and never a source of canonical fallback.
        raise
    return {
        "schema_version": "cn-macro-maintenance-preparation-receipt.v1",
        "status": "PREPARED",
        "promoted": False,
        "target_date": str(target_date).replace("-", ""),
        "prepared_path": sealed["prepared_path"],
        "prepared_sha256": sealed["prepared_sha256"],
        "release_candidate_generation_id": sealed["release"]["generation_id"],
        "observations_candidate_generation_id": sealed["observations"]["generation_id"],
        "market_pointer_path": str(Path(market_pointer_path).resolve()),
        "market_pointer_sha256": expected_market_pointer_sha256,
        "pit_pointer_path": str(Path(pit_pointer_path).resolve()),
        "pit_pointer_sha256": expected_pit_pointer_sha256,
        "authority_mode": authority_mode,
        "provider_calls": legacy_result.get("provider_calls", []),
        "canonical_writes": [],
    }


__all__ = [
    "MacroMaintenanceError",
    "MacroMaintenanceTransactionError",
    "commit_prepared_macro_transaction",
    "prepare_cn_macro_maintenance_transaction",
    "recover_macro_transaction",
    "rollback_macro_transaction",
    "run_cn_macro_maintenance",
]
