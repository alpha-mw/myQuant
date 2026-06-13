"""Shared local artifact store helpers for factor governance files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): json_safe(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value, key=str)]
    if isinstance(value, Path):
        return str(value)
    return value


class JsonArtifactStoreMixin:
    def _append_jsonl(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    json_safe(payload),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
            handle.write("\n")

    def _read_jsonl_payloads(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSON in {path} line {line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(payload, Mapping):
                    raise ValueError(
                        f"Expected JSON object in {path} line {line_number}."
                    )
                rows.append(dict(payload))
        return rows

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON in {path}: {exc.msg}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"Expected JSON object in {path}.")
        return dict(payload)

    def _write_text(self, path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> Path:
        return self._write_text(
            path,
            json.dumps(
                json_safe(payload),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
        )


__all__ = [
    "JsonArtifactStoreMixin",
    "json_safe",
]
