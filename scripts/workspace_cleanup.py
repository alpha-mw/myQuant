"""Clean safe workspace caches and provision runtime tmp directories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.workspace_layout import (
    describe_environment_roles,
    ensure_runtime_tmp_dirs,
    get_repo_root,
    iter_cleanup_targets,
    remove_cleanup_targets,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean safe local caches and prepare runtime tmp directories."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete collected cache directories. Without this flag the script performs a dry run.",
    )
    parser.add_argument(
        "--skip-runtime-dirs",
        action="store_true",
        help="Do not create results/tmp and reports/tmp.",
    )
    parser.add_argument(
        "--show-envs",
        action="store_true",
        help="Print the current Python environment directory roles.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def _format_paths(root: Path, paths: Sequence[Path]) -> list[str]:
    return [path.relative_to(root).as_posix() for path in paths]


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    repo_root = get_repo_root(args.root)
    cleanup_targets = iter_cleanup_targets(repo_root)
    runtime_dirs: tuple[Path, ...] = ()

    if not args.skip_runtime_dirs:
        runtime_dirs = ensure_runtime_tmp_dirs(repo_root)

    mode = "apply" if args.apply else "dry-run"
    print(f"workspace cleanup mode: {mode}")
    print(f"workspace root: {repo_root}")

    if runtime_dirs:
        print("runtime tmp dirs:")
        for relative_path in _format_paths(repo_root, runtime_dirs):
            print(f"  - {relative_path}")

    if args.show_envs:
        print("environment roles:")
        for item in describe_environment_roles(repo_root):
            status = "present" if item["exists"] else "missing"
            print(f"  - {item['relative_path']}: {status} | {item['role']}")

    print("cleanup targets:")
    for relative_path in _format_paths(repo_root, cleanup_targets):
        print(f"  - {relative_path}")

    if args.apply:
        removed = remove_cleanup_targets(cleanup_targets)
        print(f"removed {len(removed)} directories")
    else:
        print(f"would remove {len(cleanup_targets)} directories")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
