from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = ROOT / "quant_investor"


def test_quant_investor_pyc_files_have_source_peers():
    missing_sources: list[str] = []

    for pyc_path in PACKAGE_ROOT.rglob("*.pyc"):
        if "__pycache__" not in pyc_path.parts:
            continue
        source_stem = pyc_path.name.split(".cpython", 1)[0]
        source_path = pyc_path.parent.parent / f"{source_stem}.py"
        if not source_path.exists():
            missing_sources.append(str(pyc_path.relative_to(ROOT)))

    assert not missing_sources


def test_promoted_mainline_paths_do_not_use_shadowed_pyc_wrappers():
    mainline_paths = [
        PACKAGE_ROOT / "agent_protocol.py",
        PACKAGE_ROOT / "branch_contracts.py",
        PACKAGE_ROOT / "pipeline" / "parallel_research_pipeline.py",
        PACKAGE_ROOT / "pipeline" / "mainline.py",
    ]

    for path in mainline_paths:
        text = path.read_text(encoding="utf-8")
        assert "_pyc_backup" not in text, f"{path} should not depend on _pyc_backup"
        assert "load_shadowed_module" not in text, f"{path} should not load sourceless shadow modules"


def test_removed_legacy_entrypoints_are_absent():
    removed_paths = [
        PACKAGE_ROOT / "pipeline" / ("legacy" + "_v8_pipeline.py"),
        PACKAGE_ROOT / "pipeline" / ("current" + ".py"),
        PACKAGE_ROOT / ("contracts" + ".py"),
    ]
    removed_paths.extend(
        PACKAGE_ROOT / "pipeline" / f"quant_investor_v{suffix}.py"
        for suffix in ("8", "9", "10", "11")
    )

    for path in removed_paths:
        assert not path.exists(), f"{path} should be removed from the single-mainline surface"
