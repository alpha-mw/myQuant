"""Real strict-Parquet calculation and rank; generation/activation is not mocked."""

import json
import pytest
from _native_daily_factor_fixture import NativeFactorInputs
from quant_investor.factors.governance.production_authority import (
    recompute_factor_production_signals,
)
from quant_investor.intelligence.daily import rank_factor_signals
from quant_investor.intelligence.storage import approved_theme_policy_v2


def test_five_large_cohort_native_factor_calculations(tmp_path):
    fixture = NativeFactorInputs(tmp_path)
    proofs = []
    for offset in range(5):
        arguments = fixture.day(offset)
        result = recompute_factor_production_signals(**arguments)
        rank = rank_factor_signals(
            signal_values=result["signal_values"], policy=approved_theme_policy_v2()
        )
        assert all(len(values) == 3000 for values in result["signal_values"].values())
        assert rank["common_symbol_count"] == 3000
        assert len(rank["pool_rows"]) == 100
        assert len({r["symbol"] for r in rank["pool_rows"]}) == 100
        assert all(row["distinct_finite_count"] > 1 for row in result["signal_statistics"])
        proofs.append(
            {
                "trade_date": arguments["as_of"],
                "exact_replay_sha256": result["exact_replay_sha256"],
                "low_sha": result["low_signal_sha256"],
                "w80_sha": result["w80_signal_sha256"],
                "cohort": rank["common_symbol_count"],
                "top100": rank["pool_rows"],
            }
        )
        print("native-factor-day", arguments["as_of"], "cohort=3000", flush=True)
    replay = recompute_factor_production_signals(**arguments)
    assert replay == result
    deficient = {factor: dict(values) for factor, values in result["signal_values"].items()}
    for values in deficient.values():
        values.pop(fixture.symbols[-1])
    from quant_investor.intelligence import IntelligenceError

    with pytest.raises(IntelligenceError, match="below policy minimum"):
        rank_factor_signals(signal_values=deficient, policy=approved_theme_policy_v2())
    from quant_investor.factors.governance.errors import FactorGovernanceError

    with pytest.raises(FactorGovernanceError):
        recompute_factor_production_signals(**{**arguments, "market_history_sha256": "0" * 64})
    print("native-factor-proof", tmp_path / "native-factor-five-day.json", flush=True)
    (tmp_path / "native-factor-five-day.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "full_dag_proof": False,
                "generation_authority_verified": False,
                "scope": "NATIVE_FACTOR_RECOMPUTATION_AND_RANK",
                "days": proofs,
            },
            indent=2,
        )
    )
