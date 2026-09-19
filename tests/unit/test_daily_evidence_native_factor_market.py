"""No fake Market reader: native strict PIT/snapshot -> native Factor source extraction."""

import hashlib
from _native_daily_factor_fixture import (
    NativeFactorInputs,
    strict_market_from_factor_inputs,
    put_table,
)
from quant_investor.factors.governance.factor_production_prepare import (
    _factor_market_rows,
    _resolve_market_bound_pit_pointer,
    _canonical_market_scope,
)
from quant_investor.factors.governance.production_authority import (
    recompute_factor_production_signals,
)
from pathlib import Path


def test_native_large_market_reader_and_factor_prepare_extraction(tmp_path):
    fixture = NativeFactorInputs(tmp_path / "factor-inputs")
    args = fixture.day(0)
    reader, pit = strict_market_from_factor_inputs(tmp_path, args)
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (tmp_path / "data").rglob("*")
        if p.is_file()
    }
    gate = reader.clean_snapshot_gate(refresh=True)
    assert gate["healthy"] is True
    binding = reader.coverage_bound_pit(refresh=True)
    assert binding["status"] == "passed"
    assert len(binding["records"]) == 3000
    symbols, _ = _canonical_market_scope(tmp_path / "data")
    assert symbols == fixture.symbols
    _, _, discovery = _resolve_market_bound_pit_pointer(
        pit_manifest_path=Path(pit["generation_manifest_path"]),
        pit_generation_id=pit["generation_id"],
        pit_manifest_sha256=pit["generation_manifest_sha256"],
        pit_canonical_sha256=pit["canonical_sha256"],
    )
    assert discovery["generation_id"] == pit["generation_id"]
    market_rows, pit_rows = _factor_market_rows(
        reader, records=binding["records"], sessions=fixture.sessions[:91], as_of=args["as_of"]
    )
    assert len(market_rows) == 3000 * 91
    assert sum(row["tradable"] for row in pit_rows) == 3000
    output = tmp_path / "extracted"
    market_sha = put_table(output / "market.parquet", market_rows, "market_history")
    pit_sha = put_table(output / "pit.parquet", pit_rows, "pit_universe")
    extracted = recompute_factor_production_signals(
        **{
            **args,
            "market_history_path": output / "market.parquet",
            "market_history_sha256": market_sha,
            "pit_universe_path": output / "pit.parquet",
            "pit_universe_sha256": pit_sha,
        }
    )
    direct = recompute_factor_production_signals(**args)
    assert extracted["signal_values"] == direct["signal_values"]
    assert extracted["low_signal_sha256"] == direct["low_signal_sha256"]
    assert extracted["w80_signal_sha256"] == direct["w80_signal_sha256"]

    assert before == {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (tmp_path / "data").rglob("*")
        if p.is_file()
    }
    membership = Path(pit["canonical_path"])
    membership.write_bytes(membership.read_bytes() + b" ")
    assert reader.coverage_bound_pit(refresh=True)["status"] != "passed"
