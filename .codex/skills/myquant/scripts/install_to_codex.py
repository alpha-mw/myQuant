#!/usr/bin/env python3
"""
Install or update the repo-local myquant skill into Codex's discovery directory.

Examples:
    python3 scripts/install_to_codex.py --dry-run
    python3 scripts/install_to_codex.py --dest-root /tmp/codex-skills
    python3 scripts/install_to_codex.py --replace
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys


SKIP_DIR_NAMES = {"__pycache__"}
SKIP_FILE_SUFFIXES = {".pyc"}
SKIP_FILE_NAMES = {".DS_Store"}


def source_skill_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dest_root() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home).expanduser() / "skills"
    return Path.home() / ".codex" / "skills"


def should_skip(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts) or path.suffix in SKIP_FILE_SUFFIXES or path.name in SKIP_FILE_NAMES


def iter_source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if should_skip(relative):
            continue
        files.append(path)
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the repo-local myquant skill into Codex's skill directory.",
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=default_dest_root(),
        help="Destination skills root. Defaults to $CODEX_HOME/skills or ~/.codex/skills.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace the destination skill directory if it already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned install without writing files.",
    )
    return parser.parse_args()


def validate_source(root: Path) -> None:
    required_paths = [
        root / "SKILL.md",
        root / "agents" / "openai.yaml",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required skill files: {', '.join(missing)}")


def install_skill(source: Path, dest_root: Path, *, replace: bool, dry_run: bool) -> Path:
    destination = dest_root / source.name
    files = iter_source_files(source)

    if destination.exists():
        if not replace:
            raise FileExistsError(
                f"destination already exists: {destination}. Re-run with --replace to overwrite."
            )
        if dry_run:
            print(f"[dry-run] would remove existing destination: {destination}")
        else:
            shutil.rmtree(destination)

    if dry_run:
        print(f"[dry-run] would install {len(files)} files from {source} to {destination}")
        for path in files:
            print(f"[dry-run] copy {path.relative_to(source)}")
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    for source_path in files:
        relative = source_path.relative_to(source)
        target_path = destination / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    return destination


def main() -> int:
    args = parse_args()
    source = source_skill_dir()
    validate_source(source)
    destination = install_skill(
        source,
        args.dest_root.expanduser(),
        replace=args.replace,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Installed skill to {destination}")
        print("Restart Codex to pick up new skills.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
