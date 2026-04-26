from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from quant_investor.agent_protocol import (
    BayesianDecisionRecord,
    BranchVerdict,
    GlobalContext,
    ICDecision,
    PortfolioDecision,
    PortfolioPlan,
    RiskDecision,
    SymbolResearchPacket,
)
from quant_investor.bayesian.calibration import CalibrationStore
from quant_investor.branch_config import (
    CANONICAL_BRANCH_ORDER,
    DEFAULT_BRANCH_WEIGHTS,
    get_default_branch_weights,
    validate_branch_weights,
)


ROOT = Path(__file__).resolve().parents[2]


def test_default_branch_weights_validate_and_sum_to_one() -> None:
    validate_branch_weights(DEFAULT_BRANCH_WEIGHTS)

    copied_weights = get_default_branch_weights()
    assert copied_weights == DEFAULT_BRANCH_WEIGHTS
    assert copied_weights is not DEFAULT_BRANCH_WEIGHTS
    assert math.isclose(sum(copied_weights.values()), 1.0, abs_tol=1e-9)


@pytest.mark.parametrize(
    "weights, message",
    [
        ({"quant": 0.28, "kline": 0.22, "intelligence": 0.20, "fundamental": 0.15}, "missing"),
        (
            {
                "quant": 0.28,
                "kline": 0.22,
                "intelligence": 0.20,
                "fundamental": 0.15,
                "macro": 0.15,
                "sentiment": 0.0,
            },
            "extra",
        ),
        (
            {
                "quant": -0.28,
                "kline": 0.22,
                "intelligence": 0.20,
                "fundamental": 0.15,
                "macro": 0.71,
            },
            "non-negative",
        ),
        (
            {
                "quant": float("nan"),
                "kline": 0.22,
                "intelligence": 0.20,
                "fundamental": 0.15,
                "macro": 0.43,
            },
            "finite",
        ),
        (
            {
                "quant": float("inf"),
                "kline": 0.22,
                "intelligence": 0.20,
                "fundamental": 0.15,
                "macro": 0.43,
            },
            "finite",
        ),
    ],
)
def test_invalid_branch_weights_raise_clear_value_error(
    weights: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_branch_weights(weights)


def _read_branch_weight_table() -> dict[str, str]:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    section_match = re.search(r"\*\*分支权重\*\*(?P<section>.*?)(?:\n---|\Z)", readme, re.S)
    assert section_match is not None
    rows: list[list[str]] = []
    for line in section_match.group("section").splitlines():
        if "|" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or all(re.fullmatch(r":?-{2,}:?", cell) for cell in cells):
            continue
        rows.append(cells)

    header = next(row for row in rows if row and row[0] == "分支")
    weights = next(row for row in rows if row and row[0] == "权重")
    assert len(header) == len(weights)
    return dict(zip(header[1:], weights[1:], strict=True))


def test_readme_branch_percentages_match_default_branch_weights() -> None:
    readme_weights = _read_branch_weight_table()
    expected_labels = {
        "Quant Factor": "quant",
        "K-Line": "kline",
        "Intelligence": "intelligence",
        "Fundamental": "fundamental",
        "Macro": "macro",
    }

    for label, branch_name in expected_labels.items():
        assert label in readme_weights
        displayed_percent = int(re.search(r"(\d+)\s*%", readme_weights[label]).group(1))
        assert displayed_percent == round(DEFAULT_BRANCH_WEIGHTS[branch_name] * 100)


@pytest.mark.parametrize(
    "protocol_type, critical_fields",
    [
        (GlobalContext, ["market", "universe_key", "universe_hash", "data_quality_quarantine", "metadata"]),
        (SymbolResearchPacket, ["symbol", "branch_verdicts", "branch_scores", "metadata"]),
        (BranchVerdict, ["agent_name", "final_score", "final_confidence", "evidence", "metadata"]),
        (RiskDecision, ["hard_veto", "max_weight", "gross_exposure_cap", "blocked_symbols", "reasons"]),
        (ICDecision, ["final_score", "final_confidence", "selected_symbols", "rejected_symbols"]),
        (PortfolioPlan, ["target_weights", "blocked_symbols", "turnover_estimate", "metadata"]),
        (PortfolioDecision, ["shortlist", "target_weights", "risk_constraints", "metadata"]),
        (
            BayesianDecisionRecord,
            [
                "posterior_win_rate",
                "posterior_expected_alpha",
                "posterior_edge_after_costs",
                "action_threshold_used",
                "metadata",
            ],
        ),
    ],
)
def test_core_protocol_surface_default_construction(
    protocol_type: type[Any],
    critical_fields: list[str],
) -> None:
    instance = protocol_type()
    payload = instance.to_dict()

    assert isinstance(payload, dict)
    for field_name in critical_fields:
        assert hasattr(instance, field_name)
        assert field_name in payload


def test_calibration_v1_anchor_records_outcome_jsonl(tmp_path: Path) -> None:
    store_path = tmp_path / "bayesian_calibration.json"
    store = CalibrationStore(str(store_path))

    for branch_name in CANONICAL_BRANCH_ORDER:
        for score in (-0.60, 0.0, 0.60):
            probability = store.calibrated_probability(branch_name, score)
            assert 0.0 <= probability <= 1.0

    store.record_outcome(
        symbol="000001.SZ",
        branch_name="quant",
        predicted_score=0.25,
        realized_return=0.03,
        run_date="2026-04-25",
    )

    outcomes_path = tmp_path / "bayesian_outcomes.jsonl"
    assert outcomes_path.exists()
    rows = outcomes_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    payload = json.loads(rows[0])
    assert {
        "symbol",
        "branch",
        "score",
        "bucket",
        "realized_return",
        "run_date",
    }.issubset(payload)
    assert payload["symbol"] == "000001.SZ"
    assert payload["branch"] == "quant"
    assert payload["score"] == 0.25
    assert payload["bucket"] == "positive"
    assert payload["realized_return"] == 0.03
    assert payload["run_date"] == "2026-04-25"
