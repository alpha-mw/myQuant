from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]

CURRENT_DOCS = (
    ROOT / "README.md",
    ROOT / "README.zh-CN.md",
    ROOT / "AGENTS.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "factor_governance.md",
    ROOT / "docs" / "modules" / "module_map.md",
    ROOT / "docs" / "modules" / "macro_risk_reference.md",
    ROOT / "docs" / "trading_discipline.md",
    ROOT / "docs" / "migrations" / "unified-cutover" / "cli-mapping.md",
    ROOT / "docs" / "migrations" / "unified-cutover" / "README.md",
    ROOT / "docs" / "migrations" / "unified-cutover" / "replacement-test-map.md",
)

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_current_documentation_surface_exists() -> None:
    for path in CURRENT_DOCS:
        assert path.is_file(), f"missing current doc: {path.relative_to(ROOT)}"


def test_current_docs_use_only_resolvable_relative_or_external_links() -> None:
    for path in CURRENT_DOCS:
        text = _read(path)
        assert "](/Users/" not in text
        for _label, target in MARKDOWN_LINK_RE.findall(text):
            if target.startswith(("http://", "https://", "#")):
                continue
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            resolved = (path.parent / target_path).resolve()
            assert resolved.exists(), (
                f"{path.relative_to(ROOT)} links to missing target: {target}"
            )


def test_public_docs_describe_the_stable_runtime_and_single_pointer() -> None:
    joined = "\n".join(_read(path) for path in CURRENT_DOCS)
    assert "quant-investor system status" in joined
    assert "quant-investor factor status" in joined
    assert "quant-investor research run" in joined
    assert "results/system/_active.json" in joined
    assert "contract_sha256" in joined
    assert "BOOTSTRAP_EXCEPTION" in joined
    assert "SYSTEM_SUSPENDED" in joined
    assert "latest run" in joined.lower()
    assert "never scan" in joined.lower()


def test_module_map_preserves_independent_ownership_boundaries() -> None:
    module_map = _read(ROOT / "docs" / "modules" / "module_map.md")
    for name in (
        "quant_investor.contracts",
        "quant_investor.system",
        "quant_investor.factors.governance",
        "quant_investor.intelligence",
        "quant_investor.mainline",
        "Strategy Record Store",
        "Portfolio Cycle",
        "Dashboard",
    ):
        assert name in module_map
    assert "does not\nadvance or rewrite their independent current pointers" in module_map


def test_macro_reference_is_explicitly_non_authoritative() -> None:
    text = _read(ROOT / "docs" / "modules" / "macro_risk_reference.md")
    assert "独立旧版/手工参考" in text
    assert "不属于统一 Mainline" in text
    assert "不能作为缺失正式" in text


def test_cutover_docs_state_hard_stop_and_no_fallback() -> None:
    runbook = _read(
        ROOT / "docs" / "migrations" / "unified-cutover" / "README.md"
    )
    mapping = _read(
        ROOT / "docs" / "migrations" / "unified-cutover" / "cli-mapping.md"
    )
    assert "clean integration commit" in runbook
    assert "CAS" in runbook
    assert "SYSTEM_SUSPENDED" in runbook
    assert "fallback" in runbook.lower()
    assert "system verify" in mapping
    assert "research compile-evidence" in mapping
