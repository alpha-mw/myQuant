from pathlib import Path


RUNTIME_FILES = [
    "daily_runner.py",
    "quant_investor/pipeline/mainline.py",
    "quant_investor/monitoring/cn_aggressive_daily_review.py",
    "quant_investor/monitoring/cn_aggressive_portfolio_tracker.py",
    "quant_investor/portfolio/constructor.py",
    "quant_investor/risk/risk_guard.py",
]


def test_factor_evidence_not_imported_by_runtime_selection_surfaces() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    existing_files = [repo_root / relative for relative in RUNTIME_FILES if (repo_root / relative).exists()]
    assert existing_files
    for path in existing_files:
        text = path.read_text(encoding="utf-8")
        assert "factors.evidence" not in text
        assert "FactorEvidence" not in text
        assert "build_multi_date_factor_evidence_report" not in text


def test_factor_evidence_does_not_reference_orders_or_action_paths() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    evidence_text = (repo_root / "quant_investor/factors/evidence.py").read_text(encoding="utf-8")
    assert "orders.csv" not in evidence_text
    assert "action_taken_today" not in evidence_text
    assert "target_weights" not in evidence_text
