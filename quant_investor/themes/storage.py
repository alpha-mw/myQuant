from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Mapping


SNAPSHOT_SCHEMA_VERSION = "theme_snapshot.v1"
_SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_STORAGE_METADATA = {
    "deterministic": True,
    "no_llm": True,
    "no_network": True,
    "storage": "json",
}


class ThemeSnapshotStore:
    def __init__(self, root_dir: str | Path = "results/theme_snapshots") -> None:
        self.root_dir = Path(root_dir)

    def save(
        self,
        theme_rotation: Mapping[str, Any],
        *,
        market: str = "CN",
        universe_key: str = "",
        as_of: str = "",
        run_id: str = "",
    ) -> Path:
        safe_market = _safe_component(market, "unknown_market")
        raw_universe_key = str(universe_key or "")
        raw_as_of = str(as_of or "")
        resolved_run_id = str(run_id or _auto_run_id(theme_rotation))
        payload_universe_key = raw_universe_key or "unknown_universe"
        payload_as_of = raw_as_of or "unknown_date"
        safe_universe_key = _safe_component(payload_universe_key, "unknown_universe")
        safe_as_of = _safe_component(payload_as_of, "unknown_date")
        safe_run_id = _safe_component(resolved_run_id, "snapshot")
        date_segment = _date_segment(raw_as_of)

        parent = self.root_dir / safe_market / date_segment
        filename = (
            f"{safe_universe_key}_{safe_as_of}_{safe_run_id}_theme_rotation.json"
        )
        target_path = parent / filename
        payload = {
            "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
            "market": str(market or ""),
            "universe_key": payload_universe_key,
            "as_of": payload_as_of,
            "run_id": resolved_run_id,
            "theme_rotation": copy.deepcopy(dict(theme_rotation)),
            "metadata": dict(_STORAGE_METADATA),
        }

        parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(f".{target_path.name}.tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            tmp_path.replace(target_path)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        return target_path

    def list_snapshots(
        self,
        *,
        market: str = "CN",
        universe_key: str | None = None,
    ) -> list[Path]:
        market_dir = self.root_dir / _safe_component(market, "unknown_market")
        if not market_dir.exists():
            return []
        paths = [
            path
            for path in market_dir.rglob("*_theme_rotation.json")
            if path.is_file()
        ]
        if universe_key is not None:
            safe_universe_key = _safe_component(universe_key, "unknown_universe")
            prefix = f"{safe_universe_key}_"
            paths = [path for path in paths if path.name.startswith(prefix)]
        return sorted(paths, key=lambda path: str(path))

    def load_latest(
        self,
        *,
        market: str = "CN",
        universe_key: str | None = None,
    ) -> dict[str, Any] | None:
        latest = self.load_latest_with_path(
            market=market,
            universe_key=universe_key,
        )
        if latest is None:
            return None
        return latest[1]

    def load_latest_with_path(
        self,
        *,
        market: str = "CN",
        universe_key: str | None = None,
        as_of: str | None = None,
    ) -> tuple[Path, dict[str, Any]] | None:
        target_date_segment = _date_segment(as_of or "") if as_of else ""
        latest_any: dict[str, Any] | None = None
        latest_any_path: Path | None = None
        for path in reversed(
            self.list_snapshots(market=market, universe_key=universe_key)
        ):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict):
                if latest_any is None:
                    latest_any = payload
                    latest_any_path = path
                if _is_load_latest_candidate(payload, universe_key=universe_key):
                    if target_date_segment:
                        rotation = payload.get("theme_rotation")
                        rotation_as_of = (
                            rotation.get("as_of")
                            if isinstance(rotation, Mapping)
                            else ""
                        )
                        payload_as_of = str(
                            payload.get("as_of")
                            or rotation_as_of
                            or ""
                        )
                        if _date_segment(payload_as_of) != target_date_segment:
                            continue
                    return path, payload
        if target_date_segment:
            return None
        if latest_any_path is None or latest_any is None:
            return None
        return latest_any_path, latest_any

    def load_recent(
        self,
        *,
        market: str = "CN",
        universe_key: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        max_items = max(int(limit or 0), 0)
        if max_items <= 0:
            return []
        payloads: list[dict[str, Any]] = []
        for path in reversed(
            self.list_snapshots(market=market, universe_key=universe_key)
        ):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            if not _is_load_latest_candidate(payload, universe_key=universe_key):
                continue
            payloads.append(payload)
            if len(payloads) >= max_items:
                break
        return list(reversed(payloads))


def _safe_component(value: Any, default: str) -> str:
    text = str(value or "").strip()
    if not text:
        text = default
    safe = _SAFE_COMPONENT_RE.sub("_", text)
    safe = safe.strip("_") or default
    if safe in {".", ".."}:
        return default
    return safe


def _date_segment(as_of: str) -> str:
    text = str(as_of or "").strip()
    if not text:
        return "unknown_date"
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return _safe_component(text, "unknown_date")


def _auto_run_id(theme_rotation: Mapping[str, Any]) -> str:
    metadata = {}
    if isinstance(theme_rotation, Mapping):
        raw_metadata = theme_rotation.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    for key in (
        "snapshot_id",
        "run_id",
        "universe_hash",
        "content_hash",
        "theme_hash",
    ):
        value = metadata.get(key)
        if value:
            return str(value)
    return "snapshot"


def _is_load_latest_candidate(
    payload: Mapping[str, Any],
    *,
    universe_key: str | None,
) -> bool:
    if universe_key != "full_a":
        return True
    rotation = payload.get("theme_rotation")
    if not isinstance(rotation, Mapping):
        return True
    metadata = rotation.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    input_scope = str(metadata.get("input_scope") or "").strip()
    if input_scope and input_scope != "full_market":
        return False
    scanned = _safe_int(metadata.get("scanned_symbol_count"))
    member_min = _safe_int(metadata.get("member_count_min"))
    if member_min > 0 and 0 < scanned < member_min:
        return False
    return True


def _safe_int(value: Any) -> int:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return 0
