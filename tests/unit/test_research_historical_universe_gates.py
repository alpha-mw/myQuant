"""Gates that keep a research historical universe honest.

The research universe exists so that identities which delisted inside a backtest
window can be built and consumed at all. It must never become a way to run the
production scope under a different name, so every gate here is about refusing
something, not about enabling it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.market import fundamental_mart


@pytest.fixture(autouse=True)
def canonical_roots(tmp_path: Path, monkeypatch):
    """Refusal tests prove isolation against existing synthetic canonical roots."""
    root = tmp_path / "canonical"
    fundamental = root / "parquet/cn"
    fundamental.mkdir(parents=True)
    monkeypatch.setattr(fundamental_mart, "DEFAULT_FUNDAMENTAL_ROOT", fundamental)
    monkeypatch.setattr(fundamental_mart, "DEFAULT_MARKET_DATA_ROOT", root)


def test_research_universe_key_is_not_a_full_a_alias() -> None:
    """The named key must not inherit full_a's serving-set special case."""
    assert (
        fundamental_mart.RESEARCH_HISTORICAL_UNIVERSE_KEY
        not in fundamental_mart.FULL_A_UNIVERSE_KEYS
    )


def test_unknown_universe_is_still_refused_for_authoritative_rebuild(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires universes=full_a"):
        fundamental_mart.run_cn_fundamental_maintenance(
            universes=["zz500"],
            as_of="20260904",
            allow_live=True,
            authoritative_full_rebuild=True,
            data_root=tmp_path / "staging",
            canonical_scope_path=tmp_path / "scope.json",
            canonical_market_pointer_path=tmp_path / "pointer.json",
            canonical_membership_path=tmp_path / "membership.parquet",
            checkpoint_root=tmp_path / "checkpoint",
        )


def test_research_universe_refused_on_the_production_market_root(tmp_path: Path) -> None:
    """The research key is inadmissible while pointed at the production market root.

    Without this the named key would resolve its symbols from production's own
    components file — the scope would be production's, only the label would differ.
    """
    with pytest.raises(ValueError, match="isolated market data root"):
        fundamental_mart.run_cn_fundamental_maintenance(
            universes=[fundamental_mart.RESEARCH_HISTORICAL_UNIVERSE_KEY],
            as_of="20260904",
            allow_live=True,
            authoritative_full_rebuild=True,
            data_root=tmp_path / "staging",
            market_data_root=fundamental_mart.DEFAULT_MARKET_DATA_ROOT,
            canonical_scope_path=tmp_path / "scope.json",
            canonical_market_pointer_path=tmp_path / "pointer.json",
            canonical_membership_path=tmp_path / "membership.parquet",
            checkpoint_root=tmp_path / "checkpoint",
        )


def test_staging_root_isolation_is_checked_before_the_universe_key(tmp_path: Path) -> None:
    """Isolation evidence comes first; the named key is only admissible after it.

    Ordering matters: the universe gate used to run before the staging root was
    proven distinct, which would reject a named research key before the proof it
    depends on existed.
    """
    with pytest.raises(ValueError, match="isolated staging data root"):
        fundamental_mart.run_cn_fundamental_maintenance(
            universes=[fundamental_mart.RESEARCH_HISTORICAL_UNIVERSE_KEY],
            as_of="20260904",
            allow_live=True,
            authoritative_full_rebuild=True,
            data_root=fundamental_mart.DEFAULT_FUNDAMENTAL_ROOT,
            market_data_root=tmp_path / "market",
            canonical_scope_path=tmp_path / "scope.json",
            canonical_market_pointer_path=tmp_path / "pointer.json",
            canonical_membership_path=tmp_path / "membership.parquet",
            checkpoint_root=tmp_path / "checkpoint",
        )


def test_empty_intersection_raises_instead_of_serving_the_whole_universe(
    tmp_path: Path,
) -> None:
    """An empty components-serving intersection is a scope error, not a wildcard.

    This used to fall back to ``sorted(serving_symbols)`` — every served symbol —
    turning a mis-scoped run into a silent full-scope one.
    """
    root = tmp_path / "market"
    (root / "cn_universe").mkdir(parents=True)
    (root / "cn_universe" / "cn_index_components.json").write_text(
        '{"full_a": ["999999.SZ"]}', encoding="utf-8"
    )

    class _Reader:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def list_symbols(self, *, universe_key: str) -> list[str]:
            return ["000001.SZ", "600000.SH"]

    original = fundamental_mart.MarketDataReader
    fundamental_mart.MarketDataReader = _Reader  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="do not intersect"):
            fundamental_mart._resolve_symbols_from_parquet_universe(root, ["full_a"])
    finally:
        fundamental_mart.MarketDataReader = original  # type: ignore[assignment]


def test_coverage_boundary_path_falls_back_to_the_module_global(
    monkeypatch, tmp_path: Path
) -> None:
    """The fallback must be read at call time so tests can monkeypatch the module.

    Capturing it as a default argument would freeze the production path into the
    signature and silently ignore every existing monkeypatch of the attribute.
    """
    monkeypatch.setattr(
        fundamental_mart, "DAILY_BASIC_COVERAGE_BOUNDARY_PATH", tmp_path / "declared.json"
    )

    result = fundamental_mart._declared_coverage_intervals(
        set(),
        listing_identities={},
        listing_dates={},
        history_end_dates={},
        membership_sha256="",
        cutoff="20260904",
    )

    assert list(result["daily_history_coverage_intervals"]) == []
    assert result["daily_history_coverage_interval_path"] == ""


def test_isolated_root_does_not_inherit_the_production_declaration(
    monkeypatch, tmp_path: Path
) -> None:
    """An isolated root must not stamp the production declaration into its evidence.

    This is the failure the parameter exists to prevent: with a single global path,
    an isolated run whose symbols appear nowhere in the production declaration gets
    an empty interval set that raises nothing, while the production file's absolute
    path and SHA256 are recorded as that run's coverage provenance.
    """
    production = tmp_path / "production_declaration.json"
    production.write_text(
        '{"schema_version": "daily-basic-coverage-intervals.v2"}', encoding="utf-8"
    )
    monkeypatch.setattr(fundamental_mart, "DAILY_BASIC_COVERAGE_BOUNDARY_PATH", production)

    result = fundamental_mart._declared_coverage_intervals(
        set(),
        listing_identities={},
        listing_dates={},
        history_end_dates={},
        membership_sha256="",
        cutoff="20260904",
        boundary_path=tmp_path / "isolated_declaration.json",
    )

    assert result["daily_history_coverage_interval_path"] != str(production)
    assert result["daily_history_coverage_interval_source_sha256"] == ""
