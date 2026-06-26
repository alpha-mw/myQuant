from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass
class RegimeHistoryLoadResult:
    records: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _record_scope_matches(
    payload: Mapping[str, Any],
    *,
    universe_key: str,
    scope_key: str | None,
    source_universe_key: str | None,
    diagnostics: list[str],
) -> bool:
    if not scope_key and not source_universe_key:
        return _text(payload.get("universe_key")) == _text(universe_key)

    record_scope = _text(payload.get("scope_key"))
    record_source = _text(payload.get("source_universe_key"))
    if not record_scope:
        if "legacy_ambiguous_regime_history_ignored" not in diagnostics:
            diagnostics.append("legacy_ambiguous_regime_history_ignored")
        return False
    if scope_key and record_scope != _text(scope_key):
        return False
    if source_universe_key and record_source != _text(source_universe_key):
        return False
    return True


def load_regime_history_result(
    path: str | Path,
    market: str,
    universe_key: str,
    before_or_equal_as_of: str | None = None,
    limit: int = 252,
    *,
    scope_key: str | None = None,
    source_universe_key: str | None = None,
) -> RegimeHistoryLoadResult:
    file_path = Path(path)
    diagnostics: list[str] = []
    if not file_path.exists():
        return RegimeHistoryLoadResult(records=[], diagnostics=["regime_history_missing"])
    market_text = _text(market)
    cutoff = _text(before_or_equal_as_of)
    records: list[dict[str, Any]] = []
    try:
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    diagnostics.append(f"regime_history_malformed_line_ignored:{line_number}")
                    continue
                if not isinstance(payload, dict):
                    diagnostics.append(f"regime_history_non_mapping_line_ignored:{line_number}")
                    continue
                if _text(payload.get("market")) != market_text:
                    continue
                as_of = _text(payload.get("as_of"))
                if cutoff and as_of and as_of > cutoff:
                    continue
                if not _record_scope_matches(
                    payload,
                    universe_key=universe_key,
                    scope_key=scope_key,
                    source_universe_key=source_universe_key,
                    diagnostics=diagnostics,
                ):
                    continue
                records.append(payload)
    except OSError as exc:
        return RegimeHistoryLoadResult(
            records=[],
            diagnostics=[f"regime_history_read_failed:{exc}"],
        )
    max_records = max(int(limit or 0), 0)
    if max_records <= 0:
        return RegimeHistoryLoadResult(records=[], diagnostics=diagnostics)
    return RegimeHistoryLoadResult(records=records[-max_records:], diagnostics=diagnostics)


def load_regime_history(
    path: str | Path,
    market: str,
    universe_key: str,
    before_or_equal_as_of: str | None = None,
    limit: int = 252,
) -> list[dict[str, Any]]:
    return load_regime_history_result(
        path,
        market=market,
        universe_key=universe_key,
        before_or_equal_as_of=before_or_equal_as_of,
        limit=limit,
    ).records


def _same_persistence_key(existing: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    if _text(existing.get("market")) != _text(payload.get("market")):
        return False
    if _text(existing.get("as_of")) != _text(payload.get("as_of")):
        return False
    payload_scope = _text(payload.get("scope_key"))
    if payload_scope:
        return (
            _text(existing.get("scope_key")) == payload_scope
            and _text(existing.get("source_universe_key"))
            == _text(payload.get("source_universe_key"))
        )
    return _text(existing.get("universe_key")) == _text(payload.get("universe_key"))


def append_regime_signal(path: str | Path, signal: Any) -> list[str]:
    file_path = Path(path)
    try:
        payload = signal.to_dict() if hasattr(signal, "to_dict") else signal
        if not isinstance(payload, Mapping):
            return ["regime_persistence_payload_not_mapping"]
        payload_text = json.dumps(dict(payload), ensure_ascii=False, sort_keys=True)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        raw_lines: list[str] = []
        replaced = False
        duplicate_count = 0
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.rstrip("\n")
                    if not text.strip():
                        continue
                    try:
                        existing = json.loads(text)
                    except json.JSONDecodeError:
                        raw_lines.append(text)
                        continue
                    if isinstance(existing, Mapping) and _same_persistence_key(existing, payload):
                        if not replaced:
                            raw_lines.append(payload_text)
                            replaced = True
                        else:
                            duplicate_count += 1
                        continue
                    raw_lines.append(text)
        if not replaced:
            raw_lines.append(payload_text)
        tmp_path = file_path.with_name(f"{file_path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for line in raw_lines:
                handle.write(line)
                handle.write("\n")
        tmp_path.replace(file_path)
    except OSError as exc:
        return [f"regime_persistence_write_failed:{exc}"]
    except TypeError as exc:
        return [f"regime_persistence_serialize_failed:{exc}"]
    notes: list[str] = []
    if replaced:
        notes.append("regime_persistence_replaced_existing_record")
    if duplicate_count:
        notes.append(f"regime_persistence_duplicate_records_collapsed:{duplicate_count}")
    return notes
