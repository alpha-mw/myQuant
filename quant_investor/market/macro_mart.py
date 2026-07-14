"""Fail-closed CN macro compatibility mart persistence.

The v1 four-field row remains the public compatibility view.  Storage uses
immutable generations and one atomically replaced pointer so an interrupted
write cannot replace the last-good mart with an empty or partial table.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd

from quant_investor.market.branch_readiness import SOURCE_TUSHARE

DEFAULT_MACRO_ROOT = Path("data/parquet/cn/macro_daily")
DEFAULT_RAW_SNAPSHOT_ROOT = Path("data/cn_market_full/_snapshots/macro")
MACRO_FIELDS = ("macro_score", "liquidity_score", "volatility_percentile", "policy_signal")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class MacroMartPromotionError(RuntimeError):
    """Raised when a candidate generation is unsafe to promote."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(str(value or "").strip(), errors="coerce")
    if pd.isna(parsed):
        raise MacroMartPromotionError("macro_as_of_missing_or_invalid")
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_safe_root(root: Path) -> Path:
    if root.exists() and root.is_symlink():
        raise MacroMartPromotionError("macro_root_symlink_rejected")
    root.mkdir(parents=True, exist_ok=True)
    resolved = root.resolve()
    if not resolved.is_dir():
        raise MacroMartPromotionError("macro_root_not_directory")
    return resolved


def _safe_run_id(run_id: str) -> str:
    value = str(run_id or "").strip()
    if not value or not _SAFE_RUN_ID.fullmatch(value) or value in {".", ".."}:
        raise MacroMartPromotionError("macro_run_id_unsafe")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.is_symlink():
        raise MacroMartPromotionError("macro_pointer_symlink_rejected")
    descriptor, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(path.parent)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@contextmanager
def _promotion_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".promotion.lock"
    if lock_path.exists() and lock_path.is_symlink():
        raise MacroMartPromotionError("macro_lock_symlink_rejected")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _read_pointer(root: Path) -> dict[str, Any]:
    pointer = root / "latest_manifest.json"
    if not pointer.exists() or pointer.is_symlink() or not pointer.is_file():
        return {}
    try:
        payload = json.loads(pointer.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _generation_table(root: Path, pointer: Mapping[str, Any]) -> Path:
    relative = Path(str(pointer.get("table_path") or ""))
    if not relative or relative.is_absolute() or ".." in relative.parts:
        raise MacroMartPromotionError("macro_pointer_table_path_unsafe")
    table = (root / relative).resolve()
    if root not in table.parents or table.is_symlink() or not table.is_file():
        raise MacroMartPromotionError("macro_pointer_table_missing_or_unsafe")
    declared = str(pointer.get("parquet_sha256") or "")
    if not declared or _sha256(table) != declared:
        raise MacroMartPromotionError("macro_pointer_hash_mismatch")
    return table


def read_macro_mart(*, data_root: str | Path = DEFAULT_MACRO_ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read and hash-validate the pointed generation, with v1 layout support."""

    root = Path(data_root).expanduser()
    pointer = _read_pointer(root)
    pointer_path = root / "latest_manifest.json"
    if (pointer_path.exists() or pointer_path.is_symlink()) and not pointer:
        raise MacroMartPromotionError("macro_existing_pointer_invalid")
    if pointer.get("generation_id"):
        table = _generation_table(root.resolve(), pointer)
        return pd.read_parquet(table), pointer
    legacy = root / "part.parquet"
    if legacy.exists() and legacy.is_file() and not legacy.is_symlink():
        return pd.read_parquet(legacy), pointer
    return pd.DataFrame(), pointer


def _validated_frame(
    indicators: Mapping[str, Any],
    *,
    as_of: str,
    provider_status: str,
    source_priority: str,
) -> pd.DataFrame:
    row = dict(indicators)
    if not row:
        raise MacroMartPromotionError("macro_empty_candidate")
    missing = [field for field in MACRO_FIELDS if field not in row or pd.isna(row[field])]
    if missing:
        raise MacroMartPromotionError(f"macro_required_fields_missing:{','.join(missing)}")
    row.setdefault("trade_date", _date_text(as_of or row.get("trade_date")))
    row["trade_date"] = _date_text(row["trade_date"])
    if as_of and row["trade_date"] != _date_text(as_of):
        raise MacroMartPromotionError("macro_trade_date_as_of_mismatch")
    row.setdefault("source", provider_status)
    row.setdefault("source_priority", source_priority)
    row.setdefault("pit_status", "market_point_in_time")
    row.setdefault("fetched_at", _now_utc())
    frame = pd.DataFrame([row])
    for field in ("macro_score", "liquidity_score", "volatility_percentile"):
        frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if frame[list(MACRO_FIELDS)].isna().any().any():
        raise MacroMartPromotionError("macro_required_fields_invalid")
    policy_signal = str(frame.iloc[0]["policy_signal"] or "").strip()
    if not policy_signal:
        raise MacroMartPromotionError("macro_policy_signal_empty")
    frame.loc[:, "policy_signal"] = policy_signal
    if not frame["macro_score"].between(-1.0, 1.0).all():
        raise MacroMartPromotionError("macro_score_out_of_range")
    if not frame["liquidity_score"].between(-1.0, 1.0).all():
        raise MacroMartPromotionError("macro_liquidity_score_out_of_range")
    if not frame["volatility_percentile"].between(0.0, 100.0).all():
        raise MacroMartPromotionError("macro_volatility_percentile_out_of_range")
    return frame


def _compatibility_fingerprint(frame: pd.DataFrame) -> str:
    row = frame.iloc[0]
    payload = {
        "trade_date": str(row.get("trade_date") or ""),
        "macro_score": float(row.get("macro_score")),
        "liquidity_score": float(row.get("liquidity_score")),
        "volatility_percentile": float(row.get("volatility_percentile")),
        "policy_signal": str(row.get("policy_signal") or "").strip(),
        "source": str(row.get("source") or "").strip(),
        "source_priority": str(row.get("source_priority") or "").strip(),
        "pit_status": str(row.get("pit_status") or "").strip(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_macro_mart(
    indicators: Mapping[str, Any] | None = None,
    *,
    as_of: str = "",
    data_root: str | Path = DEFAULT_MACRO_ROOT,
    raw_snapshot_root: str | Path = DEFAULT_RAW_SNAPSHOT_ROOT,
    run_id: str = "",
    provider_status: str = "offline_input",
    source_priority: str = SOURCE_TUSHARE,
    provider_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and atomically promote one non-empty compatibility row."""

    run_id = _safe_run_id(run_id or f"cn_macro_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    root = _assert_safe_root(Path(data_root).expanduser())
    raw_root = _assert_safe_root(Path(raw_snapshot_root).expanduser())
    frame = _validated_frame(
        dict(indicators or {}),
        as_of=as_of,
        provider_status=provider_status,
        source_priority=source_priority,
    )
    candidate_as_of = str(frame.iloc[0]["trade_date"])
    with _promotion_lock(root):
        current = _read_pointer(root)
        pointer_path = root / "latest_manifest.json"
        if pointer_path.exists() and not current:
            raise MacroMartPromotionError("macro_existing_pointer_invalid")
        if current.get("generation_id"):
            _generation_table(root, current)
        current_as_of = str(current.get("as_of") or "")
        if current_as_of and candidate_as_of < current_as_of:
            raise MacroMartPromotionError("macro_older_generation_rejected")
        if current_as_of and candidate_as_of == current_as_of:
            current_table = _generation_table(root, current)
            current_frame = pd.read_parquet(current_table)
            if _compatibility_fingerprint(current_frame) == _compatibility_fingerprint(frame):
                return {**current, "idempotent": True, "promoted": False}
            raise MacroMartPromotionError("macro_same_as_of_conflict_rejected")

        raw_path = raw_root / f"{run_id}.csv"
        if raw_path.exists():
            raise MacroMartPromotionError("macro_raw_snapshot_already_exists")
        raw_bytes = frame.to_csv(index=False).encode("utf-8")
        _atomic_write_bytes(raw_path, raw_bytes)
        raw_sha = _sha256(raw_path)

        generations = root / "_generations"
        if generations.exists() and (generations.is_symlink() or not generations.is_dir()):
            raise MacroMartPromotionError("macro_generations_root_unsafe")
        generations.mkdir(parents=True, exist_ok=True)
        generation = generations / run_id
        if generation.exists():
            raise MacroMartPromotionError("macro_generation_already_exists")
        generation.mkdir(mode=0o700)
        table = generation / "part.parquet"
        try:
            frame.to_parquet(table, index=False)
            os.chmod(table, 0o600)
            readback = pd.read_parquet(table)
            if len(readback) != 1 or tuple(field for field in MACRO_FIELDS if field not in readback.columns):
                raise MacroMartPromotionError("macro_generation_readback_failed")
            parquet_sha = _sha256(table)
            manifest = {
                "run_id": run_id,
                "generation_id": run_id,
                "schema_version": "cn-macro-mart.v2",
                "provider_status": provider_status,
                "source_priority": source_priority,
                "daily_rows": 1,
                "field_set": list(MACRO_FIELDS),
                "coverage_rate": 1.0,
                "storage_backend": "parquet_canonical_generation",
                "table": "macro_daily",
                "table_path": str(table.relative_to(root)),
                "parquet_sha256": parquet_sha,
                "as_of": candidate_as_of,
                "promoted_at": _now_utc(),
                "raw_snapshot": str(raw_path),
                "raw_snapshot_sha256": raw_sha,
                "provider_manifest": dict(provider_manifest or {}),
            }
            generation_manifest = generation / "manifest.json"
            _atomic_write_bytes(
                generation_manifest,
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"),
            )
            manifest["generation_manifest_sha256"] = _sha256(generation_manifest)
            _fsync_directory(generation)
            _atomic_write_bytes(
                root / "latest_manifest.json",
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"),
            )
        except Exception:
            # An unpointed immutable generation is harmless audit evidence.  It
            # is intentionally not promoted or silently reused.
            raise

    return manifest


def run_cn_macro_maintenance(
    *,
    indicators: Mapping[str, Any] | None = None,
    as_of: str = "",
    data_root: str | Path = DEFAULT_MACRO_ROOT,
    raw_snapshot_root: str | Path = DEFAULT_RAW_SNAPSHOT_ROOT,
    allow_live: bool = False,
    allow_public_fallback: bool = False,
    run_id: str = "",
) -> dict[str, Any]:
    """Maintain the mart without pretending an unimplemented provider worked."""

    if not indicators:
        if allow_live:
            provider_status = "blocked_live_provider_not_implemented"
        elif allow_public_fallback:
            provider_status = "blocked_public_fallback_not_implemented"
        else:
            provider_status = "no_update_no_input"
        return {
            "run_id": run_id or "",
            "provider_status": provider_status,
            "status": "blocked",
            "promoted": False,
            "manifest": _read_pointer(Path(data_root).expanduser()),
        }

    manifest = write_macro_mart(
        indicators,
        as_of=as_of,
        data_root=data_root,
        raw_snapshot_root=raw_snapshot_root,
        run_id=run_id,
        provider_status="offline_input",
        source_priority=str(indicators.get("source_priority") or SOURCE_TUSHARE),
        provider_manifest={
            "allow_live": bool(allow_live),
            "allow_public_fallback": bool(allow_public_fallback),
        },
    )
    promoted = not bool(manifest.get("idempotent"))
    return {
        "run_id": manifest["run_id"],
        "provider_status": "offline_input",
        "status": "promoted" if promoted else "no_update_idempotent",
        "promoted": promoted,
        "manifest": manifest,
    }


__all__ = [
    "DEFAULT_MACRO_ROOT",
    "DEFAULT_RAW_SNAPSHOT_ROOT",
    "MACRO_FIELDS",
    "MacroMartPromotionError",
    "read_macro_mart",
    "run_cn_macro_maintenance",
    "write_macro_mart",
]
