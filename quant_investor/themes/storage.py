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
        for path in reversed(
            self.list_snapshots(market=market, universe_key=universe_key)
        ):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, TypeError, json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(payload, dict):
                return payload
        return None


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
