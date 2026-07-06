#!/usr/bin/env python3
"""Check that requirements.txt mirrors pyproject runtime + dev dependencies."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Any, Sequence


def _load_pyproject(path: Path) -> dict[str, Any]:
    try:
        import tomllib  # type: ignore[attr-defined]
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        return _load_pyproject_fallback(path)
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _load_pyproject_fallback(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    project_payload: dict[str, Any] = {}
    optional_payload: dict[str, list[str]] = {}
    current_section = ""
    current_key = ""
    array_lines: list[str] = []

    def flush_array() -> None:
        nonlocal current_key, array_lines
        if not current_key:
            return
        payload = "\n".join(array_lines)
        values = ast.literal_eval(payload)
        if current_section == "project":
            project_payload[current_key] = values
        elif current_section == "project.optional-dependencies":
            optional_payload[current_key] = values
        current_key = ""
        array_lines = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            flush_array()
            current_section = stripped.strip("[]")
            continue
        if current_key:
            array_lines.append(stripped)
            if stripped.endswith("]"):
                flush_array()
            continue
        if current_section not in {"project", "project.optional-dependencies"}:
            continue
        if "=" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split("=", 1))
        if value.startswith("["):
            current_key = key
            array_lines = [value]
            if value.endswith("]"):
                flush_array()
    flush_array()
    return {"project": {**project_payload, "optional-dependencies": optional_payload}}


def _normalize_requirement(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"\s+", "", text)


def _requirements_from_pyproject(path: Path, extras: Sequence[str]) -> set[str]:
    payload = _load_pyproject(path)
    project = payload.get("project", {})
    requirements = {
        _normalize_requirement(item)
        for item in list(project.get("dependencies", []) or [])
    }
    optional = project.get("optional-dependencies", {}) or {}
    for extra in extras:
        requirements.update(
            _normalize_requirement(item)
            for item in list(optional.get(extra, []) or [])
        )
    return {item for item in requirements if item}


def _requirements_from_file(path: Path) -> set[str]:
    requirements: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        normalized = _normalize_requirement(line)
        if normalized:
            requirements.add(normalized)
    return requirements


def check_requirements_sync(
    *,
    pyproject_path: Path,
    requirements_path: Path,
    extras: Sequence[str],
) -> tuple[bool, dict[str, list[str]]]:
    expected = _requirements_from_pyproject(pyproject_path, extras)
    actual = _requirements_from_file(requirements_path)
    missing = sorted(expected - actual, key=str.lower)
    extra = sorted(actual - expected, key=str.lower)
    return not missing and not extra, {"missing": missing, "extra": extra}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", default="pyproject.toml")
    parser.add_argument("--requirements", default="requirements.txt")
    parser.add_argument(
        "--extra",
        action="append",
        default=["dev"],
        help="Optional dependency group expected in requirements.txt. Repeatable.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    ok, diff = check_requirements_sync(
        pyproject_path=Path(args.pyproject),
        requirements_path=Path(args.requirements),
        extras=list(args.extra or []),
    )
    if ok:
        print("requirements.txt is in sync with pyproject.toml")
        return 0
    print("requirements.txt is out of sync with pyproject.toml", file=sys.stderr)
    if diff["missing"]:
        print("Missing from requirements.txt:", file=sys.stderr)
        for item in diff["missing"]:
            print(f"  {item}", file=sys.stderr)
    if diff["extra"]:
        print("Extra in requirements.txt:", file=sys.stderr)
        for item in diff["extra"]:
            print(f"  {item}", file=sys.stderr)
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
