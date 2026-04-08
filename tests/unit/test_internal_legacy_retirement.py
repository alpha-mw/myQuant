from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_production_paths_no_longer_import_parallel_research_pipeline_directly():
    targets = [
        ROOT / "quant_investor" / "pipeline" / "mainline.py",
        ROOT / "quant_investor" / "agents" / "quant_agent.py",
        ROOT / "quant_investor" / "agents" / "intelligence_agent.py",
    ]

    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "ParallelResearchPipeline" not in text, f"{path} still depends on legacy batch pipeline"


def test_tests_no_longer_depend_on_legacy_batch_pipeline_or_mainline_helpers():
    forbidden_markers = [
        "from quant_investor.pipeline.parallel_research_pipeline import",
        "import quant_investor.pipeline.parallel_research_pipeline",
        "ParallelResearchPipeline(",
        "._run_review_layer(",
        "._run_unified_control_chain(",
    ]
    allowed = {
        ROOT / "tests" / "unit" / "test_internal_legacy_retirement.py",
    }

    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for marker in forbidden_markers:
            assert marker not in text, f"{path} still contains legacy dependency marker: {marker}"
