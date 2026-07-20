from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
SCRIPT_PATH = REPOSITORY_ROOT / "scripts/build_factor_v4_1_signal_computability.py"
SPEC = importlib.util.spec_from_file_location(
    "build_factor_v4_1_signal_computability_under_test", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _matrix_source() -> bytes:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(WORKSPACE_ROOT),
            "show",
            f"{runner.contract.PINNED_COMMIT}:{runner.MATRIX_DATASET_PATH}",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("pinned A_quant source object is unavailable")
    return result.stdout


def _mask() -> pd.DataFrame:
    return pd.DataFrame(
        [[True, False], [True, True]],
        index=pd.DatetimeIndex(["2021-06-25", "2021-06-28"], name="trade_date"),
        columns=["000001.SZ", "000002.SZ"],
    )


def test_exact_pinned_transform_ast_is_closed() -> None:
    _, rows = runner._extract_transform_code(_matrix_source())

    assert runner._semantic_sha(rows) == (
        runner.contract.EXPECTED_TRANSFORMATION_AST_MANIFEST_SHA256
    )
    imports = runner._local_imports(SCRIPT_PATH.read_bytes(), SCRIPT_PATH)
    assert imports == runner.EXPECTED_LOCAL_IMPORTS
    assert not imports & runner.FORBIDDEN_CONTEXT_MODULES


def test_restricted_bar_child_matches_independent_parent() -> None:
    mask = _mask()
    table = pa.Table.from_pydict(
        {
            "symbol": ["000001.SZ", "000001.SZ", "000002.SZ"],
            "trade_date": [
                pd.Timestamp("2021-06-25").date(),
                pd.Timestamp("2021-06-28").date(),
                pd.Timestamp("2021-06-28").date(),
            ],
            "turnover_rate": [1.0, 2.0, 3.0],
            "total_mv": [10.0, 11.0, 12.0],
        }
    )
    raw = table.to_pandas()
    child = runner._run_restricted_child(
        operation="bars",
        matrix_source=_matrix_source(),
        mask=mask,
        payloads=[runner._arrow_stream_bytes(table)],
    )
    parent = runner._parent_bar_matrices(raw, mask)

    runner._compare_child_parent(
        child_rows=child["rows"], parent_matrices=parent, mask=mask
    )
    assert child["exec_event_count"] == 1


def test_descriptor_contract_detects_axis_nan_inf_and_signed_zero() -> None:
    index = pd.DatetimeIndex(["2021-01-01", "2021-01-02"])
    base = pd.DataFrame([[0.0, np.nan], [np.inf, -np.inf]], index=index, columns=["A", "B"])
    assert runner._worker_matrix_descriptor(base) == (
        runner.evaluator.matrix_hash_descriptor_v4_1(base)
    )
    mutations = []
    reversed_axis = base.iloc[:, ::-1]
    mutations.append(reversed_axis)
    signed_zero = base.copy()
    signed_zero.iloc[0, 0] = -0.0
    mutations.append(signed_zero)
    finite = base.copy()
    finite.iloc[0, 1] = 1.0
    mutations.append(finite)
    for value in mutations:
        assert runner._worker_matrix_descriptor(value)["matrix_sha256"] != (
            runner._worker_matrix_descriptor(base)["matrix_sha256"]
        )


def test_worker_message_enforces_actual_aggregate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "MAX_MESSAGE_BYTES", 32)

    with pytest.raises(
        runner.FactorV4_1SignalComputabilityRunnerError,
        match="aggregate message exceeds limit",
    ):
        runner._encode_worker_message({"operation": "bars"}, [b"x" * 40])


def test_malformed_git_batch_inventory_is_rejected() -> None:
    with pytest.raises(
        runner.FactorV4_1SignalComputabilityRunnerError,
        match="malformed NUL-delimited",
    ):
        runner._parse_ls_tree(b"not-a-valid-record\0")


def test_exact_coverage_deficiencies_are_nonblocking_but_drift_blocks() -> None:
    exact = copy.deepcopy(runner.contract.EXPECTED_CALENDAR_ACCOUNTING)
    runner._assert_bounded_calendar(exact)

    changed = copy.deepcopy(exact)
    changed["missing_myquant_through_max_observed_count"] += 1
    with pytest.raises(runner.TrustedComputabilityBlocker, match="bounded_calendar"):
        runner._assert_bounded_calendar(changed)


def test_runtime_resource_and_protected_context_contracts_are_exact() -> None:
    assert runner._resource_limits() == runner.contract.EXPECTED_RESOURCE_LIMITS
    assert runner.contract.EXPECTED_PROTECTED_CONTEXTS == {
        "aquant_input_resolution_lane": "forbidden_context_only",
        "same_snapshot_screening_bundle": (
            "protected_context_only_not_input_or_oracle"
        ),
        "provider_settings_sources": (
            "co_committed_context_only_not_producer_lineage"
        ),
    }


def test_already_dirty_byte_mutation_is_detected_without_status_change(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=tmp_path, check=True)
    tracked.write_text("first dirty\n", encoding="utf-8")
    status = runner._parse_porcelain(tmp_path)
    row = runner._dirty_row(tmp_path, "tracked.txt", status["tracked.txt"])
    baseline = {
        "schema_version": "factor-governance-v4.1-worktree-content-baseline.v1",
        "dirty_paths": [row],
        "dirty_path_count": 1,
        "dirty_paths_semantic_sha256": runner._semantic_sha([row]),
        "permitted_delta_paths": [],
    }
    runner._validate_worktree_baseline(root=tmp_path, baseline=baseline)
    tracked.write_text("second dirty\n", encoding="utf-8")
    assert runner._parse_porcelain(tmp_path)["tracked.txt"] == " M"

    with pytest.raises(
        runner.FactorV4_1SignalComputabilityRunnerError,
        match="byte/diff drift",
    ):
        runner._validate_worktree_baseline(root=tmp_path, baseline=baseline)


def test_unattested_review_metadata_has_no_runner_or_contract_input() -> None:
    text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "review_receipt" not in text
    assert "reviewed_exact_ten" not in text
