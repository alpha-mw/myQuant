from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "build_factor_v4_1_no_label_diagnostic.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_factor_v4_1_no_label_diagnostic_under_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _canonical_file(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _write_private(path: Path, raw: bytes) -> None:
    path.write_bytes(raw)
    path.chmod(0o600)


def _cutoff_bundle(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    bundle = tmp_path / "cutoff"
    bundle.mkdir(mode=0o700)
    bound: dict[str, str] = {}
    for filename in runner.CUTOFF_FILENAMES:
        if filename == runner.CUTOFF_REPORT:
            continue
        raw = _canonical_file({"artifact": filename})
        _write_private(bundle / filename, raw)
        bound[filename] = hashlib.sha256(raw).hexdigest()
    report = {
        "artifacts": {filename: {"sha256": digest} for filename, digest in sorted(bound.items())}
    }
    report_raw = _canonical_file(report)
    _write_private(bundle / runner.CUTOFF_REPORT, report_raw)
    _write_private(bundle / ".lock", b"")
    return bundle, {
        "bundle_path": bundle,
        "filenames": runner.CUTOFF_FILENAMES,
        "report_filename": runner.CUTOFF_REPORT,
        "expected_report_sha256": hashlib.sha256(report_raw).hexdigest(),
        "expected_report_semantic_sha256": runner._semantic_sha(report),
        "report_semantic_field": None,
        "allow_lock": True,
    }


def _write_market_table(root: Path, *, include_vol: bool = True) -> None:
    target = root / "year=2021" / "month=06"
    target.mkdir(parents=True)
    values: dict[str, object] = {
        "ts_code": ["000001.SZ", "000002.SZ"],
        "trade_date": ["20210625", "20260717"],
        "open": [10.0, 20.0],
        "high": [11.0, 21.0],
        "low": [9.0, 19.0],
        "close": [10.5, 20.5],
        "amount": [1000.0, 2000.0],
        # This physical column is deliberately unavailable to the authorized
        # eight-column projection and must never become an evaluator matrix.
        "turnover_rate": [1.0, 2.0],
    }
    if include_vol:
        values["vol"] = [100.0, 200.0]
    else:
        values["volume"] = [100.0, 200.0]
    pq.write_table(pa.table(values), target / "part.parquet")


def _eligibility_mask() -> pd.DataFrame:
    return pd.DataFrame(
        [[True, False], [False, True]],
        index=pd.DatetimeIndex(
            pd.to_datetime(["2021-06-25", "2026-07-17"]),
            name="trade_date",
        ),
        columns=["000001.SZ", "000002.SZ"],
        dtype=bool,
    )


def test_inventory_hash_uses_canonical_json_with_trailing_newline(tmp_path: Path) -> None:
    root = tmp_path / "table"
    member = root / "year=2021" / "month=06" / "part.parquet"
    member.parent.mkdir(parents=True)
    member.write_bytes(b"member")
    temporary = root / "_temporary" / "part.parquet"
    temporary.parent.mkdir()
    temporary.write_bytes(b"temporary")

    inventory, observed = runner._inventory_table(root)

    canonical = json.dumps(
        inventory,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert observed == hashlib.sha256(canonical + b"\n").hexdigest()
    assert len(inventory) == 2
    assert sum(row["dataset_member"] is True for row in inventory) == 1


def test_accepted_inventory_regression_uses_frozen_synthetic_fixture(
    tmp_path: Path,
) -> None:
    root = tmp_path / "frozen" / "table" / "bars"
    member_a = root / "year=2021" / "month=06" / "part.parquet"
    member_a.parent.mkdir(parents=True)
    member_a.write_bytes(b"alpha")
    member_b = root / "year=2021" / "month=07" / "part.parquet"
    member_b.parent.mkdir(parents=True)
    member_b.write_bytes(b"beta")
    temporary = root / "_temporary" / "part.parquet"
    temporary.parent.mkdir()
    temporary.write_bytes(b"scratch")

    inventory, observed = runner._inventory_table(root)

    assert observed == "a01ca55d0d3ebf42cf7837fad7a8b1ee6704d15dff540d6827b9775ade734ae5"
    assert len(inventory) == 3
    assert sum(row["dataset_member"] is True for row in inventory) == 2
    assert [row["relative_path"] for row in inventory] == [
        "_temporary/part.parquet",
        "year=2021/month=06/part.parquet",
        "year=2021/month=07/part.parquet",
    ]


def test_cutoff_bundle_requires_exact_empty_private_lock_and_detects_drift(
    tmp_path: Path,
) -> None:
    bundle, kwargs = _cutoff_bundle(tmp_path)

    accepted = runner._read_bundle(**kwargs)
    assert accepted["lock_descriptor"] == {
        "absolute_path": str(bundle / ".lock"),
        "byte_sha256": hashlib.sha256(b"").hexdigest(),
        "size_bytes": 0,
    }

    changed = bundle / runner.CUTOFF_FILENAMES[0]
    _write_private(changed, _canonical_file({"artifact": "drifted"}))
    with pytest.raises(runner.FactorV4_1SignalDiagnosticRunnerError, match="SHA mismatch"):
        runner._read_bundle(**kwargs)

    original = _canonical_file({"artifact": runner.CUTOFF_FILENAMES[0]})
    _write_private(changed, original)
    _write_private(bundle / ".lock", b"not-empty")
    with pytest.raises(runner.FactorV4_1SignalDiagnosticRunnerError, match="lock"):
        runner._read_bundle(**kwargs)


def test_market_loader_projects_only_eight_authorized_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "table"
    _write_market_table(root)
    inventory, _ = runner._inventory_table(root)
    real_dataset = runner.ds.dataset
    observed: dict[str, object] = {}

    class DatasetProbe:
        def __init__(self, wrapped: object) -> None:
            self._wrapped = wrapped
            self.schema = wrapped.schema

        def to_table(self, *args: object, **kwargs: object) -> object:
            observed["columns"] = kwargs.get("columns")
            return self._wrapped.to_table(*args, **kwargs)

    def probed_dataset(*args: object, **kwargs: object) -> DatasetProbe:
        return DatasetProbe(real_dataset(*args, **kwargs))

    monkeypatch.setattr(runner.ds, "dataset", probed_dataset)
    matrices = runner._load_market_matrices(
        table_root=root,
        inventory=inventory,
        eligibility_mask=_eligibility_mask(),
    )

    assert observed["columns"] == list(runner.MARKET_COLUMNS)
    assert set(matrices) == {"open", "high", "low", "close", "volume", "amount", "vwap"}
    assert "turnover_rate" not in matrices
    assert matrices["vwap"].loc[pd.Timestamp("2021-06-25"), "000001.SZ"] == 100.0
    assert matrices["vwap"].loc[pd.Timestamp("2026-07-17"), "000002.SZ"] == 100.0
    assert np.isnan(matrices["close"].loc[pd.Timestamp("2021-06-25"), "000002.SZ"])


def test_market_loader_fails_closed_when_exact_vol_column_is_missing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "table"
    _write_market_table(root, include_vol=False)
    inventory, _ = runner._inventory_table(root)

    with pytest.raises(
        runner.FactorV4_1SignalDiagnosticRunnerError,
        match="missing one or more exact market columns",
    ):
        runner._load_market_matrices(
            table_root=root,
            inventory=inventory,
            eligibility_mask=_eligibility_mask(),
        )


def test_prepublish_revalidation_propagates_predecessor_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table_root = tmp_path / "table"
    member = table_root / "part.parquet"
    table_root.mkdir()
    member.write_bytes(b"inventory")
    inventory, inventory_sha = runner._inventory_table(table_root)
    bound = tmp_path / "bound.py"
    bound.write_bytes(b"stable")
    digest = hashlib.sha256(b"stable").hexdigest()
    binding = {
        "binding_id": "bound",
        "absolute_path": str(bound),
        "byte_sha256": digest,
    }
    before = runner._binding_snapshot([binding])
    calls: list[str] = []

    def drift_on_cutoff(**kwargs: object) -> None:
        calls.append(str(kwargs["report_filename"]))
        if len(calls) == 3:
            raise runner.FactorV4_1SignalDiagnosticRunnerError("predecessor drift")

    monkeypatch.setattr(runner, "_read_bundle", drift_on_cutoff)
    specs = [
        {"report_filename": "discovery"},
        {"report_filename": "formal"},
        {"report_filename": "cutoff"},
    ]

    with pytest.raises(
        runner.FactorV4_1SignalDiagnosticRunnerError,
        match="predecessor drift",
    ):
        runner._revalidate_prepublication_inputs(
            precompute_paths=[binding],
            before_state=before,
            table_root=table_root,
            inventory=inventory,
            inventory_sha=inventory_sha,
            source_bindings=[binding, binding, binding],
            bundle_specs=specs,
        )
    assert calls == ["discovery", "formal", "cutoff"]
