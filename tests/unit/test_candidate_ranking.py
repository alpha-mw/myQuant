"""The miner must rank on incremental value, not standalone IC.

Ranking on standalone ICIR is what let the 2026-08-01 run put nine
near-duplicates of the existing pool at the top of its list.
"""

from __future__ import annotations

from typing import Any

from scripts.mine_quant_branch_factors import rank_candidates


def _candidate(
    name: str,
    *,
    gates: int = 5,
    icir: float = 0.5,
    pool_residual_icir: float = 0.5,
    master_return_delta: float = 0.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "gates_passed": gates,
        "metrics": {
            "icir": icir,
            "pool_residual_icir": pool_residual_icir,
            "master_return_delta": master_return_delta,
        },
    }


def test_incremental_alpha_outranks_a_stronger_standalone_duplicate() -> None:
    duplicate = _candidate("duplicate", icir=0.90, pool_residual_icir=0.02)
    incremental = _candidate("incremental", icir=0.40, pool_residual_icir=0.38)

    ranked = rank_candidates([duplicate, incremental])

    assert [item["name"] for item in ranked] == ["incremental", "duplicate"]


def test_gate_score_still_dominates_incremental_alpha() -> None:
    weak_gates = _candidate("weak_gates", gates=3, pool_residual_icir=0.90)
    strong_gates = _candidate("strong_gates", gates=6, pool_residual_icir=0.10)

    ranked = rank_candidates([weak_gates, strong_gates])

    assert [item["name"] for item in ranked] == ["strong_gates", "weak_gates"]


def test_standalone_icir_still_breaks_ties() -> None:
    lower = _candidate("lower", icir=0.20, pool_residual_icir=0.30)
    higher = _candidate("higher", icir=0.80, pool_residual_icir=0.30)

    ranked = rank_candidates([lower, higher])

    assert [item["name"] for item in ranked] == ["higher", "lower"]


def test_ranking_does_not_mutate_its_input() -> None:
    items = [
        _candidate("a", pool_residual_icir=0.10),
        _candidate("b", pool_residual_icir=0.90),
    ]

    ranked = rank_candidates(items)

    assert [item["name"] for item in items] == ["a", "b"]
    assert [item["name"] for item in ranked] == ["b", "a"]


def test_missing_incremental_evidence_sorts_last() -> None:
    absent = {"name": "absent", "gates_passed": 5, "metrics": {"icir": 0.9}}
    present = _candidate("present", icir=0.1, pool_residual_icir=0.5)

    ranked = rank_candidates([absent, present])

    assert [item["name"] for item in ranked] == ["present", "absent"]
