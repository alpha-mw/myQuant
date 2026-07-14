"""Point-in-time, observer-only macro replay over a strict local calendar."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from quant_investor.macro.registry import REGISTRY_VERSION, SCORE_MODEL_VERSION
from quant_investor.macro.snapshot import build_macro_snapshot
from quant_investor.macro.store import DEFAULT_OBSERVATIONS_ROOT, load_observations

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class MacroReplayError(RuntimeError):
    """Raised when replay input or output cannot be proven safe."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists() and path.is_symlink():
        raise MacroReplayError("macro_replay_output_symlink_rejected")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        os.chmod(path, 0o600)
    finally:
        if tmp.exists():
            tmp.unlink()


def _calendar_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.exists() or path.is_symlink() or not path.is_file() or path.suffix.lower() != ".parquet":
        raise MacroReplayError("macro_replay_calendar_missing_or_unsafe")
    return path.resolve()


def _open_dates(calendar_path: Path, *, start_date: str, end_date: str) -> list[str]:
    frame = pd.read_parquet(calendar_path)
    if not {"cal_date", "is_open"}.issubset(frame.columns):
        raise MacroReplayError("macro_replay_calendar_schema_invalid")
    dates = pd.to_datetime(frame["cal_date"].astype(str), errors="coerce")
    is_open = pd.to_numeric(frame["is_open"], errors="coerce")
    if dates.isna().any() or is_open.isna().any() or not set(is_open.unique()).issubset({0.0, 1.0}):
        raise MacroReplayError("macro_replay_calendar_values_invalid")
    normalized = pd.DataFrame({"cal_date": dates.dt.normalize(), "is_open": is_open.astype(int)})
    if normalized.groupby("cal_date")["is_open"].nunique().gt(1).any():
        raise MacroReplayError("macro_replay_calendar_date_conflict")
    normalized = normalized.drop_duplicates(subset=["cal_date", "is_open"])
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise MacroReplayError("macro_replay_date_range_invalid")
    mask = normalized["is_open"].eq(1) & normalized["cal_date"].between(start, end)
    values = sorted(item.date().isoformat() for item in normalized.loc[mask, "cal_date"])
    if not values:
        raise MacroReplayError("macro_replay_no_open_dates")
    return values


def run_macro_replay(
    *,
    market: str = "CN",
    start_date: str,
    end_date: str,
    observations_root: str | Path = DEFAULT_OBSERVATIONS_ROOT,
    calendar_path: str | Path,
    output_root: str | Path = "results/macro_replay",
    run_id: str = "",
) -> dict[str, Any]:
    """Replay a single validated observation generation without rereading latest."""

    if str(market).upper() != "CN":
        raise MacroReplayError("macro_replay_market_unsupported")
    rows, pinned = load_observations(observations_root)
    generation_id = str(pinned.get("generation_id") or "")
    if not generation_id or not rows:
        raise MacroReplayError("macro_replay_observation_generation_missing")
    pinned_pointer_hash = str(pinned.get("pointer_sha256") or "")
    if not pinned_pointer_hash:
        raise MacroReplayError("macro_replay_pointer_hash_missing")
    calendar = _calendar_path(calendar_path)
    open_dates = _open_dates(calendar, start_date=start_date, end_date=end_date)
    resolved_run_id = run_id or f"{open_dates[0]}_{open_dates[-1]}_{generation_id[:12]}"
    if not _SAFE_RUN_ID.fullmatch(resolved_run_id) or resolved_run_id in {".", ".."}:
        raise MacroReplayError("macro_replay_run_id_unsafe")

    root = Path(output_root).expanduser()
    if root.exists() and root.is_symlink():
        raise MacroReplayError("macro_replay_root_symlink_rejected")
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    market_root = root / "CN"
    if market_root.exists() and market_root.is_symlink():
        raise MacroReplayError("macro_replay_market_root_symlink_rejected")
    market_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    out = market_root / resolved_run_id
    if out.exists():
        raise MacroReplayError("macro_replay_output_exists")
    staging = Path(tempfile.mkdtemp(prefix=f".{resolved_run_id}.", dir=market_root))
    os.chmod(staging, 0o700)

    table = staging / "daily_snapshots.parquet"
    temporary = staging / ".daily_snapshots.parquet.tmp"
    try:
        replay_rows: list[dict[str, Any]] = []
        for day in open_dates:
            snapshot = build_macro_snapshot(rows, market="CN", as_of=day)
            replay_rows.append(
                {
                    "as_of": day,
                    "published_cutoff": snapshot.published_cutoff,
                    "snapshot_hash": snapshot.snapshot_hash,
                    "readiness_status": snapshot.readiness_status,
                    "macro_score": snapshot.macro_score,
                    "confidence": snapshot.confidence,
                    "snapshot_json": json.dumps(
                        snapshot.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "observation_generation_id": generation_id,
                    "observer_only": True,
                    "production_eligible": False,
                    "applied": False,
                }
            )
        pd.DataFrame(replay_rows).to_parquet(temporary, index=False)
        os.chmod(temporary, 0o600)
        readback = pd.read_parquet(temporary)
        if len(readback) != len(replay_rows) or not (readback["applied"] == False).all():  # noqa: E712
            raise MacroReplayError("macro_replay_readback_failed")
        os.replace(temporary, table)
        os.chmod(table, 0o600)
        table_hash = _sha256(table)
        manifest: dict[str, Any] = {
            "schema_version": "macro-replay-manifest.v1",
            "status": "OK",
            "market": "CN",
            "start_date": open_dates[0],
            "end_date": open_dates[-1],
            "open_date_count": len(open_dates),
            "observation_generation_id": generation_id,
            "observation_content_set_hash": pinned.get("content_set_hash"),
            "pinned_pointer_sha256": pinned_pointer_hash,
            "calendar_path": str(calendar),
            "calendar_sha256": _sha256(calendar),
            "registry_version": REGISTRY_VERSION,
            "score_model_version": SCORE_MODEL_VERSION,
            "daily_snapshots_sha256": table_hash,
            "observer_only": True,
            "production_eligible": False,
            "applied": False,
        }
        report = "\n".join(
            [
                "# CN Macro PIT Replay",
                "",
                f"- Observation generation: `{generation_id}`",
                f"- Date range: `{open_dates[0]}` to `{open_dates[-1]}`",
                f"- Open dates: `{len(open_dates)}`",
                f"- Daily snapshots SHA-256: `{table_hash}`",
                "- Observer only: `true`",
                "- Production eligible: `false`",
                "- Applied: `false`",
                "",
            ]
        )
        _atomic_bytes(staging / "replay_manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8"))
        _atomic_bytes(staging / "replay_report.md", report.encode("utf-8"))
        os.replace(staging, out)
        return {**manifest, "output_dir": str(out), "promoted": False}
    except Exception:
        if temporary.exists():
            temporary.unlink()
        if staging.exists():
            shutil.rmtree(staging)
        raise


__all__ = ["MacroReplayError", "run_macro_replay"]
