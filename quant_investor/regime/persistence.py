from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def load_regime_history(
    path: str | Path,
    market: str,
    universe_key: str,
    before_or_equal_as_of: str | None = None,
    limit: int = 252,
) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    market_text = str(market or "")
    universe_text = str(universe_key or "")
    cutoff = str(before_or_equal_as_of or "").strip()
    records: list[dict[str, Any]] = []
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                if str(payload.get("market") or "") != market_text:
                    continue
                if str(payload.get("universe_key") or "") != universe_text:
                    continue
                as_of = str(payload.get("as_of") or "")
                if cutoff and as_of and as_of > cutoff:
                    continue
                records.append(payload)
    except OSError:
        return []
    max_records = max(int(limit or 0), 0)
    if max_records <= 0:
        return []
    return records[-max_records:]


def append_regime_signal(path: str | Path, signal: Any) -> list[str]:
    file_path = Path(path)
    try:
        payload = signal.to_dict() if hasattr(signal, "to_dict") else signal
        if not isinstance(payload, Mapping):
            return ["regime_persistence_payload_not_mapping"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    except OSError as exc:
        return [f"regime_persistence_write_failed:{exc}"]
    except TypeError as exc:
        return [f"regime_persistence_serialize_failed:{exc}"]
    return []
