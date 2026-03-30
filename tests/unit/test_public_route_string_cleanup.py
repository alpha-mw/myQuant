from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = [
    ROOT / "quant_investor",
    ROOT / "tests",
    ROOT / "docs",
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "pyproject.toml",
]
SCAN_SUFFIXES = {".py", ".md", ".toml"}


def _removed_public_tokens() -> list[str]:
    tokens = [f"QuantInvestor{suffix}" for suffix in ("V8", "V9", "V10", "V11", "Current", "Latest")]
    tokens.extend(["Current" + "PipelineResult", "--" + "architecture"])
    tokens.extend(
        [
            "quant_investor_" + f"v{suffix}"
            for suffix in ("8", "9", "10", "11")
        ]
    )
    tokens.append("legacy" + "_v8_pipeline")
    return tokens


def _iter_scan_files():
    for target in SCAN_ROOTS:
        if target.is_file():
            yield target
            continue
        for path in target.rglob("*"):
            if path.is_file() and path.suffix in SCAN_SUFFIXES:
                yield path


def test_removed_public_route_strings_do_not_exist_in_current_tree():
    removed_tokens = _removed_public_tokens()
    offenders: list[str] = []

    for path in _iter_scan_files():
        text = path.read_text(encoding="utf-8")
        for token in removed_tokens:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT)}::{token}")

    assert offenders == []
