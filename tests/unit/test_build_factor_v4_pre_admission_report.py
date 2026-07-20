from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import scripts.build_factor_v4_pre_admission_report as runner
from scripts.retest_aquant_alpha_mix_8gate import RetestContext


def _synthetic_market_inventory() -> dict[str, object]:
    return {
        "schema_version": "factor-v4-market-data-input-inventory.v1",
        "snapshot_id": "fixture-snapshot",
        "latest_pointer": {
            "relative_path": "_latest.json",
            "size_bytes": 1,
            "sha256": "3" * 64,
        },
        "snapshot_manifest": {
            "relative_path": "_snapshots/fixture-snapshot.json",
            "size_bytes": 1,
            "sha256": "4" * 64,
        },
        "pit_generation_id": "fixture-pit",
        "pit_generation_manifest": {
            "relative_path": "reference/_generations/fixture-pit/manifest.json",
            "size_bytes": 1,
            "sha256": "9" * 64,
        },
        "pit_membership": {
            "relative_path": (
                "reference/_generations/fixture-pit/"
                "stock_basic_membership.parquet"
            ),
            "size_bytes": 1,
            "sha256": "5" * 64,
        },
        "expected_scope": {"count": 2, "sha256": "a" * 64},
        "table_root": "_snapshots/fixture-snapshot/table/bars",
        "serving_root": "_snapshots/fixture-snapshot/serving/bars",
        "table_parquet_inventory": [
            {
                "relative_path": "year=2024/month=01/part.parquet",
                "size_bytes": 1,
                "sha256": "b" * 64,
                "hard_link_count": 1,
            }
        ],
        "serving_parquet_inventory": [
            {
                "relative_path": "symbol=000001.SZ/bars.parquet",
                "size_bytes": 1,
                "sha256": "c" * 64,
                "hard_link_count": 1,
            }
        ],
    }


def _write_strict_market_fixture(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "data"
    market_root = data_root / "parquet" / "cn"
    snapshot_id = "fixture-snapshot-v4"
    snapshot_root = market_root / "_snapshots" / snapshot_id
    table_root = snapshot_root / "table" / "bars"
    serving_root = snapshot_root / "serving" / "bars"
    table_parts = [
        table_root / "year=2024" / "month=02" / "part.parquet",
        table_root / "year=2024" / "month=01" / "part.parquet",
    ]
    serving_parts = [
        serving_root / "symbol=000002.SZ" / "bars.parquet",
        serving_root / "symbol=000001.SZ" / "bars.parquet",
    ]
    for index, path in enumerate([*table_parts, *serving_parts], start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"dummy-parquet-{index}".encode("ascii"))

    pit_generation_id = "pit-fixture-v4"
    pit_root = market_root / "reference" / "_generations" / pit_generation_id
    pit_root.mkdir(parents=True)
    pit_path = pit_root / "stock_basic_membership.parquet"
    pit_path.write_bytes(b"dummy-pit-parquet")
    pit_sha = hashlib.sha256(pit_path.read_bytes()).hexdigest()
    pit_manifest_path = pit_root / "manifest.json"
    pit_manifest = {
        "schema_version": "cn_pit_universe_manifest.v1",
        "generation_id": pit_generation_id,
        "canonical_path": str(pit_path),
        "canonical_sha256": pit_sha,
    }
    pit_manifest_path.write_bytes(runner.canonical_json_bytes(pit_manifest))
    pit_manifest_sha = hashlib.sha256(pit_manifest_path.read_bytes()).hexdigest()

    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "complete": True,
        "categories_checked": ["full_a"],
        "expected_scope_count": 2,
        "expected_scope_sha256": "d" * 64,
        "pit_generation_id": pit_generation_id,
        "pit_generation_manifest_path": str(pit_manifest_path),
        "pit_generation_manifest_sha256": pit_manifest_sha,
        "pit_membership_path": str(pit_path),
        "pit_membership_sha256": pit_sha,
    }
    manifest_path = market_root / "_snapshots" / f"{snapshot_id}.json"
    manifest = {
        "snapshot_id": snapshot_id,
        "market": "CN",
        "status": "OK",
        "readback_validated": True,
        "blockers": [],
        "manifest_path": str(manifest_path),
        "table_root": str(table_root),
        "derived_serving_root": str(serving_root),
        "coverage": coverage,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(runner.canonical_json_bytes(manifest))
    pointer_path = market_root / "_latest.json"
    pointer = {
        "snapshot_id": snapshot_id,
        "status": "OK",
        "blockers": [],
        "manifest_path": str(manifest_path),
        "table_root": str(table_root),
        "derived_serving_root": str(serving_root),
        "coverage": coverage,
    }
    pointer_path.write_bytes(runner.canonical_json_bytes(pointer))
    return {
        "data_root": data_root,
        "market_root": market_root,
        "pointer": pointer_path,
        "manifest": manifest_path,
        "pit": pit_path,
        "table_root": table_root,
        "serving_root": serving_root,
        "table_part": table_parts[0],
    }


def _context() -> RetestContext:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    symbols = ["000001.SZ", "000002.SZ"]
    matrix = pd.DataFrame(1.0, index=dates, columns=symbols)
    sizes = pd.DataFrame(
        [["large", "small"]] * len(dates),
        index=dates,
        columns=symbols,
    )
    return RetestContext(
        frames={},
        universe_by_symbol={symbol: "full_a" for symbol in symbols},
        adj_close=matrix,
        volume=matrix,
        amount=matrix,
        forward_return=matrix,
        rebalance_dates=list(dates),
        biweekly_dates=list(dates),
        existing_composite=None,
        sector_by_symbol={"000001.SZ": "bank", "000002.SZ": "health"},
        size_bucket_by_symbol={"000001.SZ": "large", "000002.SZ": "small"},
        size_bucket_by_date=sizes,
        exposure_metadata={"status": "ready", "source": "fixture"},
    )


def _freeze(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    private_root = tmp_path / "private"
    run_id = "freeze-catalog-fixture"
    catalog_path = private_root / run_id / "candidate_catalog.v4.json"
    args = runner.parse_args(
        [
            "freeze-catalog",
            "--private-root",
            str(private_root),
            "--run-id",
            run_id,
            "--catalog-path",
            str(catalog_path),
        ]
    )
    return catalog_path, runner.freeze_catalog(args)


def test_freeze_catalog_is_complete_canonical_private_and_exact_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "build_context",
        lambda **_kwargs: pytest.fail("freeze-catalog loaded market data"),
    )

    catalog_path, result = _freeze(tmp_path)
    ontology_path = catalog_path.with_name(runner.ONTOLOGY_FILENAME)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    ontology = json.loads(ontology_path.read_text(encoding="utf-8"))

    assert result["candidate_count"] == 230
    assert len(catalog["candidates"]) == 230
    assert len({item["name"] for item in catalog["candidates"]}) == 230
    assert catalog_path.read_bytes() == runner.canonical_json_bytes(catalog)
    assert ontology_path.read_bytes() == runner.canonical_json_bytes(ontology)
    assert stat.S_IMODE(catalog_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(ontology_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(catalog_path.parent.stat().st_mode) == 0o700
    assert result["catalog_sha256"] == hashlib.sha256(
        catalog_path.read_bytes()
    ).hexdigest()
    by_name = {item["name"]: item for item in catalog["candidates"]}
    assert by_name["builtin_short_term_return_20d"]["primitive_ids"] == [
        "close_return"
    ]
    assert by_name["builtin_volatility_penalty_60d"]["primitive_ids"] == [
        "close_return"
    ]
    assert by_name["builtin_short_term_return_20d"]["params"][
        "_runtime_family"
    ] == "short_term_return"
    assert all(
        set(item)
        == {
            "name",
            "implementation",
            "expression",
            "direction",
            "params",
            "lookback",
            "slot",
            "input_fields",
            "primitive_ids",
            "family",
            "definition_sha256",
        }
        for item in catalog["candidates"]
    )

    with pytest.raises(
        runner.FactorV4PreAdmissionRunnerError,
        match="exact-once outputs",
    ):
        _freeze(tmp_path)


def test_screen_rejects_catalog_sha_before_any_data_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path, _result = _freeze(tmp_path)
    loaded = False

    def forbidden_load(**_kwargs: object) -> RetestContext:
        nonlocal loaded
        loaded = True
        raise AssertionError("data load must follow catalog readback")

    monkeypatch.setattr(runner, "build_context", forbidden_load)
    args = runner.parse_args(
        [
            "screen",
            "--catalog-path",
            str(catalog_path),
            "--expected-catalog-sha256",
            "f" * 64,
            "--private-root",
            str(tmp_path / "screen-private"),
            "--run-id",
            "screen-bad-sha",
        ]
    )

    with pytest.raises(
        runner.FactorV4PreAdmissionRunnerError,
        match="catalog SHA-256 mismatch",
    ):
        runner.screen(args)
    assert loaded is False
    assert not (tmp_path / "screen-private").exists()


def test_restore_exposures_uses_replace_and_preserves_maps() -> None:
    source = _context()
    last_two = source.adj_close.index[-2:]
    dropped = replace(
        source,
        adj_close=source.adj_close.reindex(last_two),
        volume=source.volume.reindex(last_two),
        amount=source.amount.reindex(last_two),
        forward_return=source.forward_return.reindex(last_two),
        rebalance_dates=list(last_two),
        biweekly_dates=list(last_two),
        sector_by_symbol={},
        size_bucket_by_symbol={},
        size_bucket_by_date=pd.DataFrame(),
        exposure_metadata={},
    )

    restored = runner._restore_exposures(dropped, source)

    assert restored.sector_by_symbol == source.sector_by_symbol
    assert restored.size_bucket_by_symbol == source.size_bucket_by_symbol
    assert restored.exposure_metadata == source.exposure_metadata
    assert list(restored.size_bucket_by_date.index) == list(last_two)


def test_code_binding_covers_direct_signal_and_input_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: set[str] = set()

    def fake_sha(path: Path, *, private: bool = False) -> str:
        assert private is False
        observed.add(Path(path).name)
        return "a" * 64

    monkeypatch.setattr(runner, "_sha256_file", fake_sha)

    digest = runner._code_binding_sha256()

    assert len(digest) == 64
    assert {
        "build_factor_v4_pre_admission_report.py",
        "mine_quant_branch_factors.py",
        "retest_aquant_alpha_mix_8gate.py",
        "governance_screening_v4.py",
        "governance_pre_admission_artifact_v4.py",
        "aquant_expression.py",
        "pit_fundamentals.py",
        "price_volume.py",
        "runtime.py",
        "market_data_reader.py",
    }.issubset(observed)


def test_market_data_inventory_binds_exact_v4_roots_and_sorted_parquet_files(
    tmp_path: Path,
) -> None:
    market = _write_strict_market_fixture(tmp_path)

    inventory = runner._market_data_input_inventory(market["data_root"])

    assert inventory["schema_version"] == (
        "factor-v4-market-data-input-inventory.v1"
    )
    assert inventory["snapshot_id"] == "fixture-snapshot-v4"
    assert inventory["latest_pointer"]["relative_path"] == "_latest.json"
    assert inventory["snapshot_manifest"]["relative_path"] == (
        "_snapshots/fixture-snapshot-v4.json"
    )
    assert inventory["pit_membership"]["sha256"] == hashlib.sha256(
        market["pit"].read_bytes()
    ).hexdigest()
    assert inventory["expected_scope"] == {"count": 2, "sha256": "d" * 64}
    assert inventory["table_root"] == (
        "_snapshots/fixture-snapshot-v4/table/bars"
    )
    assert inventory["serving_root"] == (
        "_snapshots/fixture-snapshot-v4/serving/bars"
    )
    for key in ("table_parquet_inventory", "serving_parquet_inventory"):
        paths = [str(item["relative_path"]) for item in inventory[key]]
        assert paths == sorted(paths)
        assert all(int(item["size_bytes"]) > 0 for item in inventory[key])
        assert all(len(str(item["sha256"])) == 64 for item in inventory[key])
        assert all(int(item["hard_link_count"]) == 1 for item in inventory[key])


def test_market_data_inventory_accepts_stable_canonical_hardlinks_and_binds_count(
    tmp_path: Path,
) -> None:
    market = _write_strict_market_fixture(tmp_path)
    source = market["table_part"]
    linked = source.with_name("linked.parquet")
    os.link(source, linked)

    inventory = runner._market_data_input_inventory(market["data_root"])

    by_path = {
        str(item["relative_path"]): item
        for item in inventory["table_parquet_inventory"]
    }
    source_relative = source.relative_to(market["table_root"]).as_posix()
    linked_relative = linked.relative_to(market["table_root"]).as_posix()
    assert by_path[source_relative]["hard_link_count"] == 2
    assert by_path[linked_relative]["hard_link_count"] == 2
    assert by_path[source_relative]["sha256"] == by_path[linked_relative]["sha256"]


def test_market_data_inventory_rejects_symlinked_parquet_entry(
    tmp_path: Path,
) -> None:
    market = _write_strict_market_fixture(tmp_path)
    link = market["serving_root"] / "symbol=LINK" / "bars.parquet"
    link.parent.mkdir()
    link.symlink_to(market["table_part"])

    with pytest.raises(
        runner.FactorV4PreAdmissionRunnerError,
        match="inventory symlink rejected",
    ):
        runner._market_data_input_inventory(market["data_root"])


def test_market_data_inventory_rejects_non_owner_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _write_strict_market_fixture(tmp_path)
    actual_uid = os.getuid()
    monkeypatch.setattr(runner.os, "getuid", lambda: actual_uid + 1)

    with pytest.raises(
        runner.FactorV4PreAdmissionRunnerError,
        match="owner mismatch",
    ):
        runner._market_data_input_inventory(market["data_root"])


def test_market_data_inventory_rejects_hardlinked_governed_pit_file(
    tmp_path: Path,
) -> None:
    market = _write_strict_market_fixture(tmp_path)
    os.link(market["pit"], market["pit"].with_name("membership-alias.parquet"))

    with pytest.raises(
        runner.FactorV4PreAdmissionRunnerError,
        match="hard-link count must be one",
    ):
        runner._market_data_input_inventory(market["data_root"])


def test_market_data_inventory_rejects_file_set_drift_during_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    market = _write_strict_market_fixture(tmp_path)
    original = runner._stable_file_sha256_size
    injected = False

    def mutate_file_set(path: Path, **kwargs: object) -> tuple[str, int, int]:
        nonlocal injected
        result = original(path, **kwargs)
        if not injected and Path(path).is_relative_to(market["table_root"]):
            injected = True
            (market["table_root"] / "injected.parquet").write_bytes(
                b"dummy-injected-parquet"
            )
        return result

    monkeypatch.setattr(runner, "_stable_file_sha256_size", mutate_file_set)

    with pytest.raises(
        runner.FactorV4PreAdmissionRunnerError,
        match="directory or file set changed",
    ):
        runner._market_data_input_inventory(market["data_root"])


def test_static_source_binding_uses_canonical_market_inventory_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = _synthetic_market_inventory()
    monkeypatch.setattr(runner, "_code_binding_sha256", lambda: "1" * 64)
    monkeypatch.setattr(runner, "_sha256_file", lambda *_args, **_kwargs: "2" * 64)
    monkeypatch.setattr(
        runner,
        "_fundamental_binding_sha256",
        lambda *_args, **_kwargs: "6" * 64,
    )

    bindings = runner._static_source_bindings(
        data_root=Path("unused"),
        fundamental_root=Path("unused"),
        run_config_sha256="7" * 64,
        market_data_input=inventory,
    )

    assert bindings["market_data_input_sha256"] == (
        runner.canonical_semantic_sha256(inventory)
    )
    assert set(bindings) == runner.SOURCE_BINDING_KEYS - {"calendar_sha256"}


def test_screen_recomputes_every_candidate_keeps_failures_in_bh_and_defaults_codex(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path, freeze_result = _freeze(tmp_path)
    context = _context()
    computed: list[str] = []
    metric_names: list[str] = []
    failed_name = "builtin_short_term_return_20d"

    monkeypatch.setattr(runner, "build_context", lambda **_kwargs: context)
    monkeypatch.setattr(
        runner,
        "_analysis_context",
        lambda full_context, **_kwargs: (full_context, "2024-01-02"),
    )
    monkeypatch.setattr(
        runner,
        "build_aquant_expression_inputs",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        runner,
        "_formulaic_primitives",
        lambda *_args, **_kwargs: {},
    )

    def fake_signal(candidate: object, **_kwargs: object) -> str:
        name = str(getattr(candidate, "name"))
        computed.append(name)
        if name == failed_name:
            raise RuntimeError("synthetic compute failure")
        return name

    def fake_metrics(*, signal: object, **_kwargs: object) -> dict[str, float]:
        name = str(signal)
        metric_names.append(name)
        return {"rank_ic_p_value": 0.0001 if name.endswith("_5d") else 0.5}

    monkeypatch.setattr(runner, "compute_candidate_signal", fake_signal)
    monkeypatch.setattr(runner, "candidate_metrics", fake_metrics)
    market_inventory = _synthetic_market_inventory()
    static_bindings = {
        "code_sha256": "1" * 64,
        "registry_file_sha256": "2" * 64,
        "latest_pointer_sha256": "3" * 64,
        "manifest_sha256": "4" * 64,
        "market_data_input_sha256": runner.canonical_semantic_sha256(
            market_inventory
        ),
        "pit_sha256": "5" * 64,
        "fundamental_manifest_sha256": "6" * 64,
        "run_config_sha256": "7" * 64,
    }
    monkeypatch.setattr(
        runner,
        "_market_data_input_inventory",
        lambda *_args, **_kwargs: dict(market_inventory),
    )
    monkeypatch.setattr(
        runner,
        "_static_source_bindings",
        lambda **_kwargs: dict(static_bindings),
    )

    private_root = catalog_path.parent.parent
    run_id = catalog_path.parent.name
    args = runner.parse_args(
        [
            "screen",
            "--catalog-path",
            str(catalog_path),
            "--expected-catalog-sha256",
            str(freeze_result["catalog_sha256"]),
            "--private-root",
            str(private_root),
            "--run-id",
            run_id,
            "--no-candidate-maturity-start",
        ]
    )

    result = runner.screen(args)

    assert result["candidate_count"] == 230
    assert Path(str(result["run_directory"])) == catalog_path.parent
    assert result["evaluated_count"] == 229
    assert result["compute_failed_count"] == 1
    assert len(computed) == 230
    assert len(metric_names) == 229
    screening_path = Path(str(result["screening_path"]))
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    inventory_path = Path(str(result["market_data_input_path"]))
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    assert inventory_path.read_bytes() == runner.canonical_json_bytes(inventory)
    assert stat.S_IMODE(inventory_path.stat().st_mode) == 0o600
    assert result["market_data_input_sha256"] == hashlib.sha256(
        inventory_path.read_bytes()
    ).hexdigest()
    assert screening["source_bindings"]["market_data_input_sha256"] == (
        result["market_data_input_sha256"]
    )
    rows = {row["name"]: row for row in screening["rows"]}
    failed = rows[failed_name]
    assert failed["evaluation_status"] == "compute_failed"
    assert failed["raw_p_value"] is None
    assert failed["bh_input_p_value"] == 1.0
    assert failed["family_hypothesis_count"] >= 1
    assert "RuntimeError:synthetic compute failure" == failed["failure_reason"]
    assert set(screening["source_bindings"]) == runner.SOURCE_BINDING_KEYS
    assert screening_path.read_bytes() == runner.canonical_json_bytes(screening)
    assert stat.S_IMODE(screening_path.stat().st_mode) == 0o600

    pre_admission = json.loads(
        Path(str(result["pre_admission_path"])).read_text(encoding="utf-8")
    )
    assert pre_admission["status"] == "pending_codex"
    assert pre_admission["proposals"] == []
    assert pre_admission["registry_write_enabled"] is False
    assert pre_admission["production_apply_enabled"] is False


def test_screen_blocks_in_place_parquet_mutation_before_any_screen_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path, freeze_result = _freeze(tmp_path)
    market = _write_strict_market_fixture(tmp_path)
    context = _context()
    pointer_before = market["pointer"].read_bytes()
    manifest_before = market["manifest"].read_bytes()
    parquet_before = market["table_part"].read_bytes()
    mutated = False

    def mutate_during_data_load(**_kwargs: object) -> RetestContext:
        nonlocal mutated
        assert not mutated
        mutated = True
        replacement = parquet_before[:-1] + bytes([parquet_before[-1] ^ 1])
        assert len(replacement) == len(parquet_before)
        market["table_part"].write_bytes(replacement)
        return context

    monkeypatch.setattr(runner, "build_context", mutate_during_data_load)
    monkeypatch.setattr(
        runner,
        "_analysis_context",
        lambda full_context, **_kwargs: (full_context, "2024-01-02"),
    )
    monkeypatch.setattr(
        runner,
        "build_aquant_expression_inputs",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        runner,
        "_formulaic_primitives",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        runner,
        "compute_candidate_signal",
        lambda candidate, **_kwargs: str(candidate.name),
    )
    monkeypatch.setattr(
        runner,
        "candidate_metrics",
        lambda **_kwargs: {"rank_ic_p_value": 0.5},
    )
    monkeypatch.setattr(runner, "_code_binding_sha256", lambda: "1" * 64)
    monkeypatch.setattr(runner, "_sha256_file", lambda *_args, **_kwargs: "2" * 64)
    monkeypatch.setattr(
        runner,
        "_fundamental_binding_sha256",
        lambda *_args, **_kwargs: "6" * 64,
    )

    private_root = catalog_path.parent.parent
    run_id = catalog_path.parent.name
    args = runner.parse_args(
        [
            "screen",
            "--catalog-path",
            str(catalog_path),
            "--expected-catalog-sha256",
            str(freeze_result["catalog_sha256"]),
            "--private-root",
            str(private_root),
            "--run-id",
            run_id,
            "--data-root",
            str(market["data_root"]),
            "--no-candidate-maturity-start",
        ]
    )

    with pytest.raises(
        runner.FactorV4PreAdmissionRunnerError,
        match="source bindings changed during recomputation",
    ):
        runner.screen(args)

    assert mutated is True
    assert market["pointer"].read_bytes() == pointer_before
    assert market["manifest"].read_bytes() == manifest_before
    assert market["table_part"].read_bytes() != parquet_before
    for filename in (
        runner.MARKET_DATA_INPUT_INVENTORY_FILENAME,
        runner.RUN_CONFIG_FILENAME,
        runner.SCREENING_FILENAME,
        runner.PRE_ADMISSION_REPORT_FILENAME,
    ):
        assert not (catalog_path.parent / filename).exists()


def test_screen_requires_explicit_strict_parquet_environment_before_data_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path, freeze_result = _freeze(tmp_path)
    monkeypatch.setenv("MYQUANT_MARKET_DATA_BACKEND", "parquet")
    monkeypatch.setenv("MYQUANT_MARKET_DATA_MODE_POLICY", "permissive")
    monkeypatch.setattr(
        runner,
        "build_context",
        lambda **_kwargs: pytest.fail("non-strict mode reached data loading"),
    )
    args = runner.parse_args(
        [
            "screen",
            "--catalog-path",
            str(catalog_path),
            "--expected-catalog-sha256",
            str(freeze_result["catalog_sha256"]),
            "--private-root",
            str(tmp_path / "strict-private"),
            "--run-id",
            "strict-mode-fixture",
        ]
    )

    with pytest.raises(
        runner.FactorV4PreAdmissionRunnerError,
        match="MODE_POLICY must be strict",
    ):
        runner.screen(args)
    assert not (tmp_path / "strict-private").exists()


def test_shared_fundamental_setup_failure_is_blocking_and_writes_no_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path, freeze_result = _freeze(tmp_path)
    context = _context()
    monkeypatch.setattr(runner, "build_context", lambda **_kwargs: context)
    monkeypatch.setattr(
        runner,
        "_analysis_context",
        lambda full_context, **_kwargs: (full_context, "2024-01-02"),
    )
    monkeypatch.setattr(
        runner,
        "build_aquant_expression_inputs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("synthetic shared setup failure")
        ),
    )
    market_inventory = _synthetic_market_inventory()
    monkeypatch.setattr(
        runner,
        "_market_data_input_inventory",
        lambda *_args, **_kwargs: dict(market_inventory),
    )
    monkeypatch.setattr(
        runner,
        "_static_source_bindings",
        lambda **_kwargs: {
            "code_sha256": "1" * 64,
            "registry_file_sha256": "2" * 64,
            "latest_pointer_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "market_data_input_sha256": runner.canonical_semantic_sha256(
                market_inventory
            ),
            "pit_sha256": "5" * 64,
            "fundamental_manifest_sha256": "6" * 64,
            "run_config_sha256": "7" * 64,
        },
    )
    private_root = tmp_path / "blocked-private"
    run_id = "shared-setup-blocked"
    args = runner.parse_args(
        [
            "screen",
            "--catalog-path",
            str(catalog_path),
            "--expected-catalog-sha256",
            str(freeze_result["catalog_sha256"]),
            "--private-root",
            str(private_root),
            "--run-id",
            run_id,
        ]
    )

    with pytest.raises(
        runner.FactorV4PreAdmissionRunnerError,
        match="Fundamental expression input setup failed",
    ):
        runner.screen(args)
    assert not (private_root / run_id).exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["freeze-catalog", "--catalog-path", "/tmp/catalog.json"],
        [
            "freeze-catalog",
            "--catalog-path",
            "/tmp/catalog.json",
            "--private-root",
            "/tmp/private",
        ],
    ],
)
def test_freeze_catalog_requires_explicit_private_root_and_run_id(
    argv: list[str],
) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(argv)


@pytest.mark.parametrize(
    "forbidden",
    ["--registry-path", "--apply", "--wal-path", "--receipt-path"],
)
def test_screen_cli_has_no_mutation_or_receipt_interface(
    forbidden: str,
) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "screen",
                "--catalog-path",
                "/tmp/catalog.json",
                "--expected-catalog-sha256",
                "1" * 64,
                "--private-root",
                "/tmp/private",
                "--run-id",
                "fixture",
                forbidden,
                "value",
            ]
        )


def test_runner_never_calls_legacy_run_mining() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "run_mining(" not in source
