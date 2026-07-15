"""Immutable evidence paths retained after the v14 Intelligence retirement."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

PROTECTED_RETIREMENT_EVIDENCE_ROOTS: tuple[str, ...] = (
    "data/parquet/cn/intelligence_daily",
    "reports/branch_readiness",
    "reports/branch_readiness_clean_parquet_smoke",
    "reports/daily",
    "reports/holdings_dag_review",
    "reports/intelligence_retirement",
)


class UnsafeRepositoryPath(ValueError):
    """Raised when a cleanup path is not contained by the repository root."""


def resolve_repo_relative_path(repo_root: Path, path: str) -> tuple[Path, str]:
    """Resolve a non-empty relative path and prove it stays below ``repo_root``.

    ``Path.resolve`` intentionally follows existing symlinks so a repo-local
    symlink targeting an external path cannot bypass cleanup protections.
    """

    raw_text = str(path).strip().replace("\\", "/")
    raw_path = Path(raw_text)
    if not raw_text or raw_path.is_absolute():
        raise UnsafeRepositoryPath(f"path must be repo-relative: {path}")

    try:
        root = repo_root.resolve(strict=True)
        resolved = (root / raw_path).resolve(strict=False)
        relative = resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise UnsafeRepositoryPath(f"path escapes repository root: {path}") from exc

    relative_text = relative.as_posix()
    if relative_text in {"", "."}:
        raise UnsafeRepositoryPath("repository root is not a valid cleanup path")
    return resolved, relative_text


def is_protected_retirement_evidence_path(path: str) -> bool:
    """Return whether a path intersects immutable retirement evidence.

    Cleanup callers must reject both descendants of an evidence root and a
    broader parent path whose removal would implicitly contain that evidence.
    """

    normalized = PurePosixPath(str(path).strip().replace("\\", "/")).as_posix().strip("/")
    return any(
        normalized == root or normalized.startswith(f"{root}/") or root.startswith(f"{normalized}/")
        for root in PROTECTED_RETIREMENT_EVIDENCE_ROOTS
    )
