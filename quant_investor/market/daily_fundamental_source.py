"""Retain the health-checked native Fundamental source in its maintenance attempt."""

import hashlib
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import utc_stamp
from . import fundamental_generation as native


def retain_fundamental_source(context, *, expected_binding: dict) -> dict:
    """Publish evidence only; never promote or alter a Fundamental generation."""
    from .daily_maintenance import DailyMaintenanceError, _write_once

    if context.mode != "execute":
        raise DailyMaintenanceError("FUNDAMENTAL_SOURCE_EXECUTE_REQUIRED")
    workspace = context.workspace_root.resolve(strict=True)
    attempt = context.attempt_root.resolve(strict=True)
    attempt.relative_to(workspace)
    root = native._read_data_root(workspace / "data/parquet/cn")
    pointer_path = root / native.FUNDAMENTAL_POINTER_FILENAME
    raw, _ = native._stable_file_bytes(pointer_path)
    sha = hashlib.sha256(raw).hexdigest()
    verified = native.inspect_fundamental_pointer_bytes(
        root, pointer_bytes=raw, expected_pointer_sha256=sha
    )["pointer"]
    binding = verified.get("derivation_binding")
    if binding != expected_binding or binding.get("binding_aware_research_ready") is not True:
        raise DailyMaintenanceError("FUNDAMENTAL_SOURCE_BINDING_CHANGED")
    # Use declared native derivation time, never cutoff, mtime or publication clock.
    manifest_metadata = verified["manifest"]["metadata"]
    derivations = [
        manifest_metadata.get("provider_manifest", {}).get("derivation", {}),
        verified.get("metadata", {}).get("derivation", {}),
    ]
    stamps = [d["derivation_timestamp"] for d in derivations if "derivation_timestamp" in d]
    if not stamps:
        raise DailyMaintenanceError("FUNDAMENTAL_SOURCE_TIME_UNAVAILABLE")
    available = max(stamps, key=utc_stamp)
    daily = root / verified["tables"]["fundamental_daily"]
    daily_raw, _ = native._stable_file_bytes(daily)
    daily_sha = hashlib.sha256(daily_raw).hexdigest()
    if daily_sha != verified["manifest"]["tables"]["fundamental_daily"]["sha256"]:
        raise DailyMaintenanceError("FUNDAMENTAL_SOURCE_TABLE_CHANGED")
    if native._stable_file_bytes(pointer_path)[0] != raw:
        raise DailyMaintenanceError("FUNDAMENTAL_SOURCE_POINTER_CHANGED")
    retained = attempt / f"fundamental-pointer-{sha}.json"
    _write_once(retained, raw)
    descriptor = {
        "available_at": available,
        "pointer": {"path": str(retained.relative_to(workspace)), "sha256": sha},
        "daily_parquet": {"path": str(daily.relative_to(workspace)), "sha256": daily_sha},
    }
    # Revalidate original native provenance and sources before committing the descriptor.
    checked = native.inspect_fundamental_pointer_bytes(
        root, pointer_bytes=raw, expected_pointer_sha256=sha
    )["pointer"]
    if checked != verified or native._stable_file_bytes(pointer_path)[0] != raw:
        raise DailyMaintenanceError("FUNDAMENTAL_SOURCE_CHANGED_BEFORE_SEAL")
    path = attempt / "fundamental-research-source.json"
    descriptor_sha = _write_once(path, canonical_json_bytes(descriptor))
    return {"path": str(path.relative_to(workspace)), "sha256": descriptor_sha}
