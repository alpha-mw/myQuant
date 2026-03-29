from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]

CURRENT_DOCS = [
    ROOT / "README.md",
    ROOT / "AGENTS.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "architecture" / "entrypoints_and_versioning.md",
    ROOT / "docs" / "architecture" / "research_pipeline_and_protocols.md",
    ROOT / "docs" / "modules" / "module_map.md",
    ROOT / "docs" / "modules" / "macro_risk_reference.md",
]
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
AGENTS_TEST_RE = re.compile(r"`pytest ([^`]+?\.py) -v`")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _forbidden_public_tokens() -> list[str]:
    tokens = [f"QuantInvestor{suffix}" for suffix in ("V8", "V9", "V10", "V11", "Current", "Latest")]
    tokens.extend(["Current" + "PipelineResult", "--" + "architecture"])
    return tokens


def test_current_documentation_surface_exists():
    for path in CURRENT_DOCS:
        assert path.exists(), f"missing current doc: {path.relative_to(ROOT)}"


def test_history_docs_are_removed_from_current_tree():
    assert not (ROOT / "docs" / "history").exists()


def test_current_docs_use_only_relative_or_external_markdown_links():
    for path in CURRENT_DOCS:
        text = _read(path)
        assert "](/Users/" not in text, f"{path.relative_to(ROOT)} contains an absolute local markdown link"
        for _label, target in MARKDOWN_LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), f"{path.relative_to(ROOT)} links to missing target: {target}"


def test_current_docs_do_not_reference_removed_public_routes():
    forbidden_tokens = _forbidden_public_tokens()
    for path in CURRENT_DOCS:
        text = _read(path)
        for token in forbidden_tokens:
            assert token not in text, f"{path.relative_to(ROOT)} still references removed route: {token}"


def test_root_navigation_does_not_link_to_history_docs():
    for path in [ROOT / "README.md", ROOT / "docs" / "README.md"]:
        for _label, target in MARKDOWN_LINK_RE.findall(_read(path)):
            assert "docs/history/" not in target
            assert not target.startswith("history/")


def test_agents_recommended_tests_exist():
    text = _read(ROOT / "AGENTS.md")
    test_paths = AGENTS_TEST_RE.findall(text)
    assert test_paths, "expected recommended pytest targets in AGENTS.md"
    for relative in test_paths:
        assert (ROOT / relative).exists(), f"AGENTS.md references missing test: {relative}"


def test_workspace_docs_point_to_live_workspace_modules_and_launcher():
    readme = _read(ROOT / "README.md")
    docs_index = _read(ROOT / "docs" / "README.md")
    entrypoints = _read(ROOT / "docs" / "architecture" / "entrypoints_and_versioning.md")
    module_map = _read(ROOT / "docs" / "modules" / "module_map.md")

    assert "./run_web.sh" in readme
    assert "./run_web.sh" in docs_index
    assert "./run_web.sh" in entrypoints
    assert "placeholder" not in readme
    assert "placeholder" not in module_map
    assert "FastAPI research workspace backend" in readme
    assert "React/Vite research workspace frontend" in readme
