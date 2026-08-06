from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]

CURRENT_DOCS = [
    ROOT / "README.md",
    ROOT / "README.zh-CN.md",
    ROOT / "AGENTS.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "architecture" / "entrypoints_and_versioning.md",
    ROOT / "docs" / "architecture" / "research_pipeline_and_protocols.md",
    ROOT / "docs" / "architecture" / "v17_i0_investment_intelligence.md",
    ROOT / "docs" / "architecture" / "v17_r22_forward_research_evaluator.md",
    ROOT / "docs" / "architecture" / "v17_v4_production_research_contract.md",
    ROOT / "docs" / "modules" / "module_map.md",
    ROOT / "docs" / "runbooks" / "v17_legacy_configuration_cleanup.md",
    ROOT / "docs" / "runbooks" / "v17_v4_operations.md",
]
REPO_SKILL_DOCS = [
    ROOT / "skill" / "myquant-backend-ops" / "SKILL.md",
    ROOT / "skill" / "myquant-backend-ops" / "references" / "entrypoints-and-commands.md",
    ROOT / "skill" / "myquant-backend-ops" / "references" / "runtime-paths-and-artifacts.md",
    ROOT / "skill" / "myquant-backend-ops" / "references" / "workspace-api-surface.md",
]
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
AGENTS_TEST_RE = re.compile(r"`pytest ([^`]+?\.py) -v`")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _forbidden_public_tokens() -> list[str]:
    tokens = [
        f"QuantInvestor{suffix}" for suffix in ("V8", "V9", "V10", "V11", "Current", "Latest")
    ]
    tokens.extend(
        [
            "Current" + "PipelineResult",
            "QuantInvestor" + "PipelineResult",
            "execute_market_" + "dag",
            "quant_investor.market.analyze." + "run_market_analysis",
            "--" + "architecture",
        ]
    )
    return tokens


def test_current_documentation_surface_exists():
    for path in [*CURRENT_DOCS, *REPO_SKILL_DOCS]:
        assert path.exists(), f"missing current doc: {path.relative_to(ROOT)}"


def test_history_docs_are_removed_from_current_tree():
    assert not (ROOT / "docs" / "history").exists()


def test_current_docs_use_only_relative_or_external_markdown_links():
    for path in CURRENT_DOCS:
        text = _read(path)
        assert (
            "](/Users/" not in text
        ), f"{path.relative_to(ROOT)} contains an absolute local markdown link"
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
            assert (
                token not in text
            ), f"{path.relative_to(ROOT)} still references removed route: {token}"


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
    readme_cn = _read(ROOT / "README.zh-CN.md")
    docs_index = _read(ROOT / "docs" / "README.md")
    entrypoints = _read(ROOT / "docs" / "architecture" / "entrypoints_and_versioning.md")
    module_map = _read(ROOT / "docs" / "modules" / "module_map.md")
    pipeline = _read(ROOT / "docs" / "architecture" / "research_pipeline_and_protocols.md")
    i0 = _read(ROOT / "docs" / "architecture" / "v17_i0_investment_intelligence.md")
    r22 = _read(ROOT / "docs" / "architecture" / "v17_r22_forward_research_evaluator.md")
    mainline_contract = _read(
        ROOT / "docs" / "architecture" / "v17_v4_production_research_contract.md"
    )

    assert "quant-investor research run" in readme
    assert "research-evaluate" in readme
    assert "research-evaluate" in readme_cn
    assert "V17 v4 mainline" in docs_index
    assert "myquant.v17.v4" in entrypoints
    assert "v17_mainline" in module_map
    assert "placeholder" not in readme
    assert "placeholder" not in module_map
    assert "read-only public run" in readme
    assert "portfolio_dashboard" in module_map
    assert "read-only" in pipeline
    assert "no public production publisher" in pipeline
    assert "exactly one forward Markov filter step" in i0
    assert "recursively authorized V4 source closure" in r22
    assert "Both binding scope fields" in r22
    assert "low-level exact-once and compare-and-swap storage" in mainline_contract
    assert "do not constitute a governed operator workflow" in mainline_contract


def test_macro_reference_is_explicitly_non_authoritative():
    macro_reference = _read(ROOT / "docs" / "modules" / "macro_risk_reference.md")

    assert "独立旧版/手工参考" in macro_reference
    assert "不是当前 V17 mainline" in macro_reference
    assert "不能作为缺失正式证据时的" in macro_reference


def test_repository_backend_skill_matches_the_current_public_surface():
    skill_text = "\n".join(_read(path) for path in REPO_SKILL_DOCS)

    assert "--strategy-id" in skill_text
    assert "GET /api/research/{strategy_id}" in skill_text
    assert "expected_pointer_sha256" in skill_text
    assert "research-evaluate" in skill_text
    assert "stdout-only" in skill_text
    assert "没有 POST run" in skill_text
    assert "没有公开 production publisher" in skill_text
    assert "standalone legacy automation" in skill_text
