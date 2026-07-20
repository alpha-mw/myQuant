from __future__ import annotations

import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.retest_aquant_alpha_mix_8gate import RetestContext


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "build_factor_v4_1_same_snapshot_screening.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_factor_v4_1_same_snapshot_screening_under_test", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _context() -> RetestContext:
    dates = pd.date_range("2024-01-02", periods=8, freq="B")
    columns = ["000001.SZ", "000002.SZ"]
    price = pd.DataFrame(
        np.arange(16, dtype=float).reshape(8, 2) + 10.0,
        index=dates,
        columns=columns,
    )
    return RetestContext(
        frames={},
        universe_by_symbol={column: "full_a" for column in columns},
        adj_close=price,
        volume=price + 100.0,
        amount=price + 1000.0,
        forward_return=price.pct_change().shift(-1),
        rebalance_dates=list(dates),
        biweekly_dates=[],
        existing_composite=None,
    )


def test_blank_signal_context_does_not_expose_or_mutate_forward_returns() -> None:
    source = _context()
    before = source.forward_return.copy(deep=True)

    blank = runner._blank_signal_context(source)

    assert blank is not source
    assert blank.forward_return.isna().all().all()
    pd.testing.assert_frame_equal(source.forward_return, before)
    pd.testing.assert_frame_equal(blank.adj_close, source.adj_close)


def test_masked_forward_returns_use_fixed_session_shift_and_pit_mask() -> None:
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    prices = pd.DataFrame(
        {"A": [10.0, 11.0, 12.0, 13.0, 14.0], "B": [20.0] * 5},
        index=dates,
    )
    mask = pd.DataFrame(True, index=dates, columns=prices.columns)
    mask.loc[dates[0], "A"] = False

    observed = runner._masked_forward_returns(prices, mask, horizon=2)

    assert np.isnan(observed.loc[dates[0], "A"])
    assert observed.loc[dates[1], "A"] == pytest.approx(13.0 / 12.0 - 1.0)
    assert observed.loc[dates[2], "A"] == pytest.approx(14.0 / 13.0 - 1.0)
    assert observed.loc[dates[0], "B"] == 0.0
    assert observed.iloc[-2:].isna().all().all()


def test_monthly_dates_apply_warmup_then_choose_last_open_session() -> None:
    dates = pd.date_range("2024-01-02", "2024-04-30", freq="B")

    observed = runner._monthly_rebalance_dates(dates, warmup=10, horizon=5)

    full_month_ends = (
        pd.Series(dates, index=dates).groupby(dates.to_period("M")).max()
    )
    expected = full_month_ends[
        (full_month_ends >= dates[10]) & (full_month_ends <= dates[-6])
    ].tolist()
    assert observed == [pd.Timestamp(value) for value in expected]


def test_monthly_dates_exclude_partial_final_natural_month() -> None:
    dates = pd.date_range("2025-01-02", "2026-07-17", freq="B")

    observed = runner._monthly_rebalance_dates(dates, warmup=20, horizon=30)

    assert pd.Timestamp("2026-05-29") == observed[-1]
    assert all(date.month != 6 or date.year != 2026 for date in observed)


def test_rank_ic_and_raw_p_are_monthly_cross_sectional_only() -> None:
    dates = pd.date_range("2024-01-31", periods=4, freq="ME")
    columns = [f"S{index:02d}" for index in range(25)]
    signal = pd.DataFrame(
        [np.arange(25, dtype=float) + offset for offset in range(4)],
        index=dates,
        columns=columns,
    )
    forward = signal.copy()
    forward.iloc[3] = -forward.iloc[3]

    rank_ic = runner._rank_ic_series(signal, forward, list(dates))
    raw_p = runner._raw_p_value(rank_ic)

    assert rank_ic.tolist() == pytest.approx([1.0, 1.0, 1.0, -1.0])
    assert 0.0 <= raw_p <= 1.0


def test_signal_summary_hashes_full_masked_matrix_and_keeps_compact_months() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    columns = [f"S{index:02d}" for index in range(20)]
    signal = pd.DataFrame(
        np.tile(np.arange(20, dtype=float), (4, 1)),
        index=dates,
        columns=columns,
    )
    forward = signal.copy()
    forward.iloc[1] = signal.iloc[1] * 2.0
    forward.iloc[3] = -signal.iloc[3]
    mask = pd.DataFrame(True, index=dates, columns=columns)
    mask.iloc[0, 0] = False

    summary, monthly = runner._signal_summary(
        signal,
        mask,
        forward,
        list(dates),
    )

    assert len(summary["signal_sha256"]) == 64
    assert summary["finite_ratio"] == pytest.approx(1.0)
    assert 0.0 <= summary["raw_p_value"] <= 1.0
    assert monthly.dtype == np.float32
    assert monthly.shape == (4, 20)
    assert np.isnan(monthly[0, 0])


def test_turnover_signal_summary_requires_three_unsynthesized_rank_ic_months() -> None:
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    columns = [f"S{index:02d}" for index in range(20)]
    signal = pd.DataFrame(
        np.tile(np.arange(20, dtype=float), (3, 1)),
        index=dates,
        columns=columns,
    )
    signal.iloc[2] = np.nan
    forward = signal.copy()
    forward.iloc[1] = -np.arange(20, dtype=float)
    mask = pd.DataFrame(True, index=dates, columns=columns)

    summary, _monthly = runner._signal_summary(
        signal,
        mask,
        forward,
        list(dates),
    )
    assert summary["raw_p_value"] == pytest.approx(1.0)
    with pytest.raises(
        runner.FactorV4_1SameSnapshotRunnerError,
        match="observed=2;required=3",
    ):
        runner._signal_summary(
            signal,
            mask,
            forward,
            list(dates),
            minimum_valid_rank_ic_months=runner.MIN_VALID_MONTHS,
        )


def test_bound_ideas_must_match_profile_and_diagnostic_in_exact_order() -> None:
    ideas: list[dict[str, object]] = []
    classifications: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for index in range(runner.EXPECTED_NEW_COUNT):
        idea = {
            "candidate_id": f"id-{index}",
            "name": f"new-{index:02d}",
            "source_definition_sha256": _sha(f"source-{index}"),
            "full_candidate_normalized_ast_sha256": _sha(f"ast-{index}"),
            "catalog_definition_sha256": _sha(f"catalog-{index}"),
            "mapping_semantic_sha256": _sha(f"mapping-{index}"),
            "input_fields": ["close"],
        }
        mapped = {
            "candidate_id": idea["candidate_id"],
            "name": idea["name"],
            "source_definition_sha256": idea["source_definition_sha256"],
            "normalized_ast_sha256": idea[
                "full_candidate_normalized_ast_sha256"
            ],
            "catalog_definition_sha256": idea["catalog_definition_sha256"],
            "mapping_semantic_sha256": idea["mapping_semantic_sha256"],
            "input_fields": ["close"],
        }
        ideas.append(idea)
        classifications.append(copy.deepcopy(mapped))
        rows.append(copy.deepcopy(mapped))

    runner._cross_validate_bound_ideas(
        ideas,
        {"candidate_classifications": classifications},
        {"rows": rows},
    )
    rows[1], rows[2] = rows[2], rows[1]
    with pytest.raises(
        runner.FactorV4_1SameSnapshotRunnerError,
        match="index 1",
    ):
        runner._cross_validate_bound_ideas(
            ideas,
            {"candidate_classifications": classifications},
            {"rows": rows},
        )


def test_cli_has_no_authorizing_or_external_side_effect_parameters() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden_options = (
        "--apply",
        "--registry",
        "--wal",
        "--receipt",
        "--replay",
        "--transaction",
        "--provider",
        "--llm",
        "--broker",
        "--order",
        "--trade",
    )
    assert all(option not in source.lower() for option in forbidden_options)
    assert "--horizon" not in source
    assert "--warmup" not in source


def test_cli_parser_constructs_without_duplicate_options(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        runner.parse_args(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--discovery-bundle-path" in help_text
    assert "--resolve-predeclared-input-fields" in help_text
    assert "--input-resolution-bundle-path" in help_text


def test_input_resolution_bundle_is_default_off_and_explicitly_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert (
        runner._read_input_resolution_bundle(
            SimpleNamespace(resolve_predeclared_input_fields=False)
        )
        is None
    )
    with pytest.raises(
        runner.FactorV4_1SameSnapshotRunnerError,
        match="requires an explicit input-resolution bundle",
    ):
        runner._read_input_resolution_bundle(
            SimpleNamespace(resolve_predeclared_input_fields=True)
        )

    bundle = tmp_path / "resolution-bundle"
    bundle.mkdir()
    bundle.chmod(0o700)
    artifact = bundle / "aquant_input_resolution.v4_1.json"
    artifact.write_bytes(b"proof\n")
    artifact.chmod(0o600)
    artifact_sha = hashlib.sha256(b"proof\n").hexdigest()
    semantic_sha = _sha("resolution-semantic")
    observed: dict[str, object] = {}

    def validate_bundle(**kwargs: object) -> dict[str, object]:
        observed.update(kwargs)
        return {"artifact": {"resolution_semantic_sha256": semantic_sha}}

    monkeypatch.setattr(
        runner,
        "_input_resolution_module",
        lambda: SimpleNamespace(
            ARTIFACT_FILENAME=artifact.name,
            validate_input_resolution_bundle_v4_1=validate_bundle,
        ),
    )
    result = runner._read_input_resolution_bundle(
        SimpleNamespace(
            resolve_predeclared_input_fields=True,
            input_resolution_bundle_path=str(bundle),
            expected_input_resolution_artifact_sha256=artifact_sha,
            expected_input_resolution_semantic_sha256=semantic_sha,
        )
    )

    assert observed == {
        "artifact_path": artifact,
        "expected_artifact_sha256": artifact_sha,
        "expected_semantic_sha256": semantic_sha,
    }
    assert result is not None
    assert result["artifact_path"] == str(artifact)
    assert result["artifact_sha256"] == artifact_sha
    assert result["semantic_sha256"] == semantic_sha


def test_inventory_identity_projection_accepts_equivalent_digest_schemas() -> None:
    digest = _sha("inventory-member")
    artifact_inventory = [
        {
            "relative_path": "year=2024/month=01/part.parquet",
            "size_bytes": 128,
            "byte_sha256": digest,
            "dataset_member": True,
        }
    ]
    governed_inventory = [
        {
            "relative_path": "year=2024/month=01/part.parquet",
            "size_bytes": 128,
            "sha256": digest,
            "hard_link_count": 1,
            "dataset_member": True,
        }
    ]

    runner._assert_inventory_identity_equivalent(
        artifact_inventory,
        governed_inventory,
        label="table inventory",
    )


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    [
        ("relative_path", "year=2024/month=01/other.parquet"),
        ("size_bytes", 129),
        ("sha256", _sha("different-inventory-member")),
        ("dataset_member", False),
    ],
)
def test_inventory_identity_projection_rejects_member_drift(
    field: str,
    drifted_value: object,
) -> None:
    digest = _sha("inventory-member")
    artifact_inventory = [
        {
            "relative_path": "year=2024/month=01/part.parquet",
            "size_bytes": 128,
            "byte_sha256": digest,
            "dataset_member": True,
        }
    ]
    governed_inventory = [
        {
            "relative_path": "year=2024/month=01/part.parquet",
            "size_bytes": 128,
            "sha256": digest,
            "hard_link_count": 1,
            "dataset_member": True,
        }
    ]
    governed_inventory[0][field] = drifted_value

    with pytest.raises(
        runner.FactorV4_1SameSnapshotRunnerError,
        match="identity differs",
    ):
        runner._assert_inventory_identity_equivalent(
            artifact_inventory,
            governed_inventory,
            label="table inventory",
        )


def test_inventory_identity_projection_retains_governed_hard_link_contract() -> None:
    digest = _sha("inventory-member")
    artifact_inventory = [
        {
            "relative_path": "part.parquet",
            "size_bytes": 128,
            "byte_sha256": digest,
            "dataset_member": True,
        }
    ]
    governed_inventory = [
        {
            "relative_path": "part.parquet",
            "size_bytes": 128,
            "sha256": digest,
            "hard_link_count": 0,
            "dataset_member": True,
        }
    ]

    with pytest.raises(
        runner.FactorV4_1SameSnapshotRunnerError,
        match="hard_link_count must be positive",
    ):
        runner._assert_inventory_identity_equivalent(
            artifact_inventory,
            governed_inventory,
            label="table inventory",
        )


def test_market_loader_uses_exact_float64_projection_and_never_turnover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "table"
    target = root / "year=2024" / "month=01"
    target.mkdir(parents=True)
    values = {
        "ts_code": ["A", "B"],
        "trade_date": ["20240102", "20240103"],
        "open": [10.0, 20.0],
        "high": [11.0, 21.0],
        "low": [9.0, 19.0],
        "close": [10.5, 20.5],
        "adj_close": [10.25, 20.25],
        "vol": [100.0, 200.0],
        "amount": [1000.0, 2000.0],
        "turnover_rate": [1.0, 2.0],
    }
    pq.write_table(pa.table(values), target / "part.parquet")
    inventory, _digest = runner.predecessor_reader._inventory_table(root)
    dates = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    mask = pd.DataFrame(True, index=dates, columns=["A", "B"])
    real_dataset = runner.ds.dataset
    observed: dict[str, object] = {}

    class Probe:
        def __init__(self, wrapped: object) -> None:
            self._wrapped = wrapped
            self.schema = wrapped.schema

        def to_table(self, *args: object, **kwargs: object) -> object:
            observed["columns"] = kwargs.get("columns")
            return self._wrapped.to_table(*args, **kwargs)

    monkeypatch.setattr(
        runner.ds,
        "dataset",
        lambda *args, **kwargs: Probe(real_dataset(*args, **kwargs)),
    )

    matrices = runner._load_market_matrices(root, inventory, mask)

    assert observed["columns"] == list(runner.MARKET_COLUMNS)
    assert "turnover_rate" not in matrices
    assert matrices["close"].to_numpy().dtype == np.float64
    assert matrices["vwap"].to_numpy().dtype == np.float64
    assert matrices["vwap"].loc[dates[0], "A"] == 100.0


def test_resolved_market_loader_projects_turnover_only_from_bound_serving_root(
    tmp_path: Path,
) -> None:
    table_root = tmp_path / "table"
    table_target = table_root / "year=2024" / "month=01"
    table_target.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "ts_code": ["A", "B"],
                "trade_date": ["20240102", "20240103"],
                "open": [10.0, 20.0],
                "high": [11.0, 21.0],
                "low": [9.0, 19.0],
                "close": [10.5, 20.5],
                "adj_close": [10.25, 20.25],
                "vol": [100.0, 200.0],
                "amount": [1000.0, 2000.0],
            }
        ),
        table_target / "part.parquet",
    )
    serving_root = tmp_path / "serving"
    serving_target = serving_root / "year=2024" / "month=01"
    serving_target.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "ts_code": ["A", "A", "B", "B"],
                "trade_date": ["20240102", "20240103", "20240102", "20240103"],
                "turnover_rate": [1.25, 1.5, 2.25, 2.5],
            }
        ),
        serving_target / "part.parquet",
    )
    table_inventory, _ = runner.predecessor_reader._inventory_table(table_root)
    serving_inventory, _ = runner.predecessor_reader._inventory_table(serving_root)
    dates = pd.DatetimeIndex(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    mask = pd.DataFrame(True, index=dates, columns=["A", "B"])
    mask.loc[dates[1], "B"] = False

    matrices = runner._load_market_matrices(
        table_root,
        table_inventory,
        mask,
        serving_root=serving_root,
        serving_inventory=serving_inventory,
        include_turnover_rate=True,
    )

    turnover = matrices["turnover_rate"]
    assert turnover.to_numpy().dtype == np.float64
    assert turnover.index.equals(mask.index)
    assert turnover.columns.equals(mask.columns)
    assert turnover.loc[dates[0], "A"] == pytest.approx(1.25)
    assert np.isnan(turnover.loc[dates[1], "B"])


def test_runtime_inputs_preserve_float64_until_monthly_compaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dates = pd.date_range("2024-01-02", periods=320, freq="B")
    symbols = ["A", "B"]
    eligibility = pd.DataFrame(True, index=dates, columns=symbols)
    matrices = {
        name: pd.DataFrame(1.0, index=dates, columns=symbols, dtype=float)
        for name in (
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "vwap",
            "volume",
            "amount",
        )
    }
    fundamental = {
        name: pd.DataFrame(
            np.float32(1.0), index=dates, columns=symbols, dtype=np.float32
        )
        for name in runner.FUNDAMENTAL_METRICS
    }
    existing = pd.DataFrame(
        np.float32(0.5), index=dates, columns=symbols, dtype=np.float32
    )
    observed_fundamental: dict[str, object] = {}

    def load_fundamental(*_args: object, **kwargs: object) -> tuple[dict, dict]:
        observed_fundamental.update(kwargs)
        return fundamental, {"legacy_fallback_allowed": False}

    monkeypatch.setattr(
        runner,
        "build_fundamental_metric_matrices",
        load_fundamental,
    )
    monkeypatch.setattr(
        runner.MinedFactorRegistry,
        "from_dict",
        lambda _payload: object(),
    )
    monkeypatch.setattr(
        runner,
        "compute_existing_composite",
        lambda *_args, **_kwargs: (existing, ""),
    )
    monkeypatch.setattr(runner, "_formulaic_primitives", lambda *_args: {})

    runtime = runner._runtime_inputs(
        {
            "eligibility_mask": eligibility,
            "fundamental_root": tmp_path,
            "registry_payload": {},
        },
        matrices,
    )

    expression = runtime["expression_inputs"]
    statistical = runtime["statistical_context"]
    assert expression.fin_roe.to_numpy().dtype == np.float64
    assert expression.fin_fcf_to_profit.to_numpy().dtype == np.float64
    assert expression.turnover_rate.to_numpy().dtype == np.float64
    assert set(runner.FUNDAMENTAL_METRICS).issubset(runtime["aquant_matrices"])
    assert observed_fundamental["mart_root"] == tmp_path
    assert observed_fundamental["allow_legacy_fallback"] is False
    assert statistical.existing_composite.to_numpy().dtype == np.float64
    assert statistical.forward_return.to_numpy().dtype == np.float64


def test_monthly_correlation_rows_cover_only_pairs_with_a_new_candidate() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    base = np.tile(np.arange(20, dtype=np.float32), (3, 1))
    new_a = base.copy()
    new_b = np.flip(base, axis=1).copy()

    rows = runner._monthly_correlation_rows(
        {"base": base, "new_a": new_a, "new_b": new_b},
        list(dates),
        {"new_a", "new_b"},
    )

    assert len(rows) == 9
    assert {(row["left_name"], row["right_name"]) for row in rows} == {
        ("base", "new_a"),
        ("base", "new_b"),
        ("new_a", "new_b"),
    }
    assert all(row["valid_common_symbol_count"] == 20 for row in rows)
    assert all(row["left_name"] < row["right_name"] for row in rows)


def test_compact_correlation_evidence_uses_shared_axes_and_pair_arrays() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    base = np.tile(np.arange(20, dtype=np.float32), (3, 1))
    new_a = base.copy()
    new_b = np.flip(base, axis=1).copy()

    compact = runner._compact_correlation_evidence(
        {"base": base, "new_a": new_a, "new_b": new_b},
        list(dates),
        {"new_a", "new_b"},
    )

    assert compact["candidate_names"] == ["base", "new_a", "new_b"]
    assert compact["month_end_dates"] == [
        "2024-01-31",
        "2024-02-29",
        "2024-03-29",
    ]
    assert compact["new_candidate_names"] == ["new_a", "new_b"]
    assert compact["expected_pair_count"] == 3
    assert compact["observed_pair_count"] == 3
    assert len(compact["pair_rows"]) == 3
    assert all(
        set(row) == {
            "left_index",
            "right_index",
            "valid_month_indices",
            "abs_spearman",
            "valid_common_symbol_count",
        }
        for row in compact["pair_rows"]
    )
    assert all(
        row["valid_month_indices"] == [0, 1, 2]
        and row["valid_common_symbol_count"] == [20, 20, 20]
        for row in compact["pair_rows"]
    )


def test_compact_correlation_evidence_requires_three_valid_months() -> None:
    dates = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29"])
    base = np.tile(np.arange(20, dtype=np.float32), (3, 1))
    new_a = base.copy()
    new_a[1:, :] = np.nan

    with pytest.raises(
        runner.FactorV4_1SameSnapshotRunnerError,
        match="lack three valid months",
    ):
        runner._compact_correlation_evidence(
            {"base": base, "new_a": new_a},
            list(dates),
            {"new_a"},
        )


def test_build_correlation_diagnostic_prefers_compact_contract_argument() -> None:
    compact = {
        "schema_version": "factor-v4.1-compact-monthly-correlation-input.v1",
        "candidate_names": ["base", "new"],
        "month_end_dates": ["2024-01-31", "2024-02-29", "2024-03-29"],
        "new_candidate_names": ["new"],
        "expected_pair_count": 1,
        "observed_pair_count": 1,
        "minimum_valid_symbol_count_per_month": 20,
        "minimum_valid_month_count": 3,
        "pair_rows": [
            {
                "left_index": 0,
                "right_index": 1,
                "valid_month_indices": [0, 1, 2],
                "abs_spearman": [1.0, 1.0, 1.0],
                "valid_common_symbol_count": [20, 20, 20],
            }
        ],
    }
    observed: dict[str, object] = {}

    class CompactContract:
        @staticmethod
        def build_correlation_diagnostic_v4_1(
            *,
            cycle_id: str,
            base_ontology: dict,
            formal_ontology: dict,
            base_catalog: dict,
            formal_catalog: dict,
            screening: dict,
            compact_correlation: dict,
        ) -> dict[str, str]:
            observed.update(
                {
                    "cycle_id": cycle_id,
                    "base_ontology": base_ontology,
                    "formal_ontology": formal_ontology,
                    "base_catalog": base_catalog,
                    "formal_catalog": formal_catalog,
                    "screening": screening,
                    "compact_correlation": compact_correlation,
                }
            )
            return {"mode": "compact"}

    result = runner._build_correlation_diagnostic(
        CompactContract,
        cycle_id="cycle",
        base_ontology={},
        formal_ontology={},
        base_catalog={},
        formal_catalog={},
        screening={},
        compact_correlation=compact,
    )

    assert result == {"mode": "compact"}
    assert observed["compact_correlation"] is compact
    assert "monthly_rows" not in observed


def test_build_correlation_diagnostic_expands_for_legacy_monthly_rows() -> None:
    compact = {
        "schema_version": "factor-v4.1-compact-monthly-correlation-input.v1",
        "candidate_names": ["base", "new"],
        "month_end_dates": ["2024-01-31", "2024-02-29", "2024-03-29"],
        "new_candidate_names": ["new"],
        "expected_pair_count": 1,
        "observed_pair_count": 1,
        "minimum_valid_symbol_count_per_month": 20,
        "minimum_valid_month_count": 3,
        "pair_rows": [
            {
                "left_index": 0,
                "right_index": 1,
                "valid_month_indices": [0, 1, 2],
                "abs_spearman": [1.0, 0.5, 0.25],
                "valid_common_symbol_count": [20, 21, 22],
            }
        ],
    }
    observed: dict[str, object] = {}

    class LegacyContract:
        @staticmethod
        def build_correlation_diagnostic_v4_1(
            *,
            cycle_id: str,
            base_ontology: dict,
            formal_ontology: dict,
            base_catalog: dict,
            formal_catalog: dict,
            screening: dict,
            monthly_rows: list,
        ) -> dict[str, str]:
            observed.update(
                {
                    "cycle_id": cycle_id,
                    "base_ontology": base_ontology,
                    "formal_ontology": formal_ontology,
                    "base_catalog": base_catalog,
                    "formal_catalog": formal_catalog,
                    "screening": screening,
                    "monthly_rows": monthly_rows,
                }
            )
            return {"mode": "legacy"}

    result = runner._build_correlation_diagnostic(
        LegacyContract,
        cycle_id="cycle",
        base_ontology={},
        formal_ontology={},
        base_catalog={},
        formal_catalog={},
        screening={},
        compact_correlation=compact,
    )

    assert result == {"mode": "legacy"}
    assert "compact_correlation" not in observed
    monthly_rows = observed["monthly_rows"]
    assert isinstance(monthly_rows, list)
    assert [row["month_end"] for row in monthly_rows] == compact["month_end_dates"]
    assert [row["valid_common_symbol_count"] for row in monthly_rows] == [20, 21, 22]


def test_streaming_evaluation_accounts_for_full_267_and_keeps_only_month_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = runner._contract_module()
    base_names = [f"base_{index:03d}" for index in range(230)]
    diagnostic_names = [f"new_diag_{index:02d}" for index in range(27)]
    turnover_names = sorted(
        contract.PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES
    )
    fundamental_names = sorted(
        contract.PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES
    )
    new_names = [*diagnostic_names, *turnover_names, *fundamental_names]
    diagnostic_rows = []
    for name in new_names:
        status = (
            runner.no_label_contract.STATUS_SIGNAL_DIAGNOSTIC
            if name in diagnostic_names
            else runner.no_label_contract.STATUS_TURNOVER_BLOCKED
            if name in turnover_names
            else runner.no_label_contract.STATUS_FUNDAMENTAL_BLOCKED
        )
        diagnostic_rows.append({"name": name, "status": status})
    preflight = {
        "base_candidates": [SimpleNamespace(name=name) for name in base_names],
        "base_catalog": {"candidates": [{"name": name} for name in base_names]},
        "formal_catalog": {
            "candidates": [
                {"name": name} for name in [*base_names, *new_names]
            ]
        },
        "bound_ideas": [
            {
                "name": name,
                "input_fields": (
                    ["turnover_rate"]
                    if name in turnover_names
                    else ["fin_roa"]
                    if name in fundamental_names
                    else ["close"]
                ),
            }
            for name in new_names
        ],
        "diagnostic": {"rows": diagnostic_rows},
    }
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    columns = [f"S{index:02d}" for index in range(20)]
    matrix = pd.DataFrame(1.0, index=dates, columns=columns)
    governed = {"eligibility_mask": matrix.astype(bool)}
    runtime = {
        "signal_context": object(),
        "statistical_context": SimpleNamespace(forward_return=matrix),
        "expression_inputs": object(),
        "formulaic_primitives": {},
        "monthly_dates": list(dates),
    }
    counter = {"value": 0}

    monkeypatch.setattr(
        runner,
        "compute_candidate_signal",
        lambda *_args, **_kwargs: matrix,
    )
    monkeypatch.setattr(
        runner.aquant_eval,
        "evaluate_pinned_idea_v4_1",
        lambda **_kwargs: matrix,
    )

    def compact_summary(*_args: object, **_kwargs: object) -> tuple[dict, np.ndarray]:
        counter["value"] += 1
        return (
            {
                "signal_sha256": _sha(f"signal-{counter['value']}"),
                "finite_ratio": 1.0,
                "raw_p_value": 0.5,
            },
            np.ones((3, 20), dtype=np.float32),
        )

    monkeypatch.setattr(runner, "_signal_summary", compact_summary)

    evaluations, monthly = runner._evaluate_candidates(
        preflight, governed, {"close": matrix}, runtime
    )
    assert len(evaluations) == 267
    assert sum(row["status"] == contract.STATUS_EVALUATED for row in evaluations) == 257
    assert sum(row["status"] == contract.STATUS_TURNOVER_BLOCKED for row in evaluations) == 2
    assert sum(row["status"] == contract.STATUS_FUNDAMENTAL_BLOCKED for row in evaluations) == 8
    assert len(monthly) == 257
    assert all(value.shape == (3, 20) and value.dtype == np.float32 for value in monthly.values())
    blocked = [row for row in evaluations if row["status"] != contract.STATUS_EVALUATED]
    assert all(
        row["signal_sha256"] is None
        and row["finite_ratio"] is None
        and row["raw_p_value"] is None
        for row in blocked
    )


@pytest.mark.parametrize(
    ("full_turnover_coverage", "expected_profile", "expected_evaluated"),
    [
        (False, "fundamental_resolved", 265),
        (True, "fully_resolved", 267),
    ],
)
def test_resolved_streaming_evaluation_is_atomic_and_exact_profiled(
    monkeypatch: pytest.MonkeyPatch,
    full_turnover_coverage: bool,
    expected_profile: str,
    expected_evaluated: int,
) -> None:
    contract = runner._contract_module()
    base_names = [f"base_{index:03d}" for index in range(230)]
    diagnostic_names = [f"new_diag_{index:02d}" for index in range(27)]
    turnover_names = sorted(
        contract.PREDECLARED_TURNOVER_BLOCKED_CANDIDATE_NAMES
    )
    fundamental_names = sorted(
        contract.PREDECLARED_FUNDAMENTAL_BLOCKED_CANDIDATE_NAMES
    )
    new_names = [*diagnostic_names, *turnover_names, *fundamental_names]
    diagnostic_rows = [
        {
            "name": name,
            "status": (
                runner.no_label_contract.STATUS_SIGNAL_DIAGNOSTIC
                if name in diagnostic_names
                else runner.no_label_contract.STATUS_TURNOVER_BLOCKED
                if name in turnover_names
                else runner.no_label_contract.STATUS_FUNDAMENTAL_BLOCKED
            ),
        }
        for name in new_names
    ]
    bound_ideas = [
        {
            "name": name,
            "input_fields": (
                ["turnover_rate"]
                if name in turnover_names
                else ["fin_roa"]
                if name in fundamental_names
                else ["close"]
            ),
        }
        for name in new_names
    ]
    preflight = {
        "base_candidates": [SimpleNamespace(name=name) for name in base_names],
        "base_catalog": {"candidates": [{"name": name} for name in base_names]},
        "formal_catalog": {
            "candidates": [
                {"name": name} for name in [*base_names, *new_names]
            ]
        },
        "bound_ideas": bound_ideas,
        "diagnostic": {"rows": diagnostic_rows},
    }
    dates = pd.date_range("2024-01-31", periods=3, freq="ME")
    columns = [f"S{index:02d}" for index in range(20)]
    full_signal = pd.DataFrame(
        np.tile(np.arange(20, dtype=float), (3, 1)),
        index=dates,
        columns=columns,
    )
    sparse_turnover = full_signal.copy()
    sparse_turnover.iloc[2] = np.nan
    forward = full_signal.copy()
    forward.iloc[2] = -np.arange(20, dtype=float)
    mask = pd.DataFrame(True, index=dates, columns=columns)

    def evaluate_idea(**kwargs: object) -> pd.DataFrame:
        idea = kwargs["idea"]
        assert isinstance(idea, dict)
        if (
            idea["name"] == "alpha_turnover_low_60d"
            and not full_turnover_coverage
        ):
            return sparse_turnover
        return full_signal

    monkeypatch.setattr(
        runner,
        "compute_candidate_signal",
        lambda *_args, **_kwargs: full_signal,
    )
    monkeypatch.setattr(
        runner.aquant_eval,
        "evaluate_pinned_idea_v4_1",
        evaluate_idea,
    )
    expected_descriptors = {
        name: runner.aquant_eval.matrix_hash_descriptor_v4_1(
            sparse_turnover
            if name == "alpha_turnover_low_60d" and not full_turnover_coverage
            else full_signal
        )
        for name in contract.PREDECLARED_BLOCKED_CANDIDATE_NAMES
    }
    runtime = {
        "signal_context": object(),
        "statistical_context": SimpleNamespace(forward_return=forward),
        "expression_inputs": object(),
        "formulaic_primitives": {},
        "monthly_dates": list(dates),
        "aquant_matrices": {
            "close": full_signal,
            "fin_roa": full_signal,
            "turnover_rate": full_signal,
        },
        "candidate_signal_descriptors": expected_descriptors,
    }

    evaluations, monthly = runner._evaluate_candidates(
        preflight,
        {"eligibility_mask": mask},
        {"close": full_signal},
        runtime,
        resolve_predeclared_input_fields=True,
    )
    profile = runner._validate_evaluation_profile(
        evaluations,
        base_names=set(base_names),
        contract=contract,
    )

    assert profile == expected_profile
    assert sum(
        row["status"] == contract.STATUS_EVALUATED for row in evaluations
    ) == expected_evaluated
    assert len(monthly) == expected_evaluated
    assert sum(
        row["status"] == contract.STATUS_TURNOVER_BLOCKED
        for row in evaluations
    ) == (0 if full_turnover_coverage else 2)
    assert sum(
        row["status"] == contract.STATUS_FUNDAMENTAL_BLOCKED
        for row in evaluations
    ) == 0


def test_runner_delegates_full_family_bh_exactly_once() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.count("contract.build_same_snapshot_screening_v4_1(") == 1
    assert "screening_v4.build_screening_evidence_v4(" not in source


def test_resolved_run_config_binds_mode_path_hashes_and_exact_source_semantics(
    tmp_path: Path,
) -> None:
    contract = runner._contract_module()
    base_names = [f"base_{index:03d}" for index in range(230)]
    new_names = [
        *[f"new_diag_{index:02d}" for index in range(27)],
        *sorted(contract.PREDECLARED_BLOCKED_CANDIDATE_NAMES),
    ]
    preflight = {
        name: {"report_semantic_sha256": _sha(name)}
        for name in ("discovery", "formal", "no_label", "cutoff")
    }
    preflight["base_catalog"] = {
        "candidates": [{"name": name} for name in base_names]
    }
    preflight["formal_catalog"] = {
        "candidates": [{"name": name} for name in [*base_names, *new_names]]
    }
    artifact_path = tmp_path / "aquant_input_resolution.v4_1.json"

    legacy = runner._run_config_semantic_sha(preflight)
    resolved = runner._run_config_semantic_sha(
        preflight,
        resolve_predeclared_input_fields=True,
        input_resolution_artifact_path=str(artifact_path),
        input_resolution_artifact_sha256=_sha("artifact"),
        input_resolution_semantic_sha256=_sha("semantic"),
    )
    changed_proof = runner._run_config_semantic_sha(
        preflight,
        resolve_predeclared_input_fields=True,
        input_resolution_artifact_path=str(artifact_path),
        input_resolution_artifact_sha256=_sha("artifact-other"),
        input_resolution_semantic_sha256=_sha("semantic"),
    )

    assert legacy != resolved
    assert resolved != changed_proof


def test_source_bindings_use_active_fundamental_generation_manifest() -> None:
    args = type(
        "Args",
        (),
        {
            "expected_snapshot_manifest_sha256": _sha("snapshot"),
            "expected_pit_membership_sha256": _sha("pit"),
        },
    )()
    preflight = {
        name: {"report_semantic_sha256": _sha(name)}
        for name in ("discovery", "formal", "no_label", "cutoff")
    }
    controls = {"code_sha256": _sha("code")}
    governed = {
        "registry_row": {"byte_sha256": _sha("registry")},
        "latest_row": {"byte_sha256": _sha("latest")},
        "market_data_input_sha256": _sha("market"),
        "fundamental_manifest_row": {"byte_sha256": _sha("legacy-latest-manifest")},
        "fundamental_generation_manifest_sha256": _sha("active-generation-manifest"),
    }
    matrix_context = {"calendar_sha256": _sha("calendar")}

    observed = runner._source_bindings(
        args,
        preflight,
        controls,
        governed,
        matrix_context,
    )

    assert observed["fundamental_manifest_sha256"] == _sha(
        "active-generation-manifest"
    )
    assert observed["fundamental_manifest_sha256"] != governed[
        "fundamental_manifest_row"
    ]["byte_sha256"]
