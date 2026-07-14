from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from scripts import run_quant_runtime_performance_audit as audit


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_quant_runtime_performance_audit.py"
REGISTRY = REPO_ROOT / "quant_investor" / "factor_registry" / "mined_factors.json"


def _registry_snapshot() -> tuple[str, int, int]:
    stat_result = REGISTRY.stat()
    return (
        hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
        stat_result.st_mode & 0o777,
        stat_result.st_nlink,
    )


def _run(*extra: str) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    command = [
        sys.executable,
        str(SCRIPT),
        "--symbols",
        "24",
        "--validation-rows",
        "91",
        "--runtime-rows",
        "91",
        "--warmup",
        "1",
        "--samples",
        "3",
        "--budget-scale",
        "1000",
        "--validation-throughput-min",
        "1",
        "--factor-throughput-min",
        "1",
        "--scaling-max",
        "100",
        "--native-incremental-max-mib",
        "1024",
        *extra,
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    stdout_lines = completed.stdout.splitlines()
    assert len(stdout_lines) == 1, completed.stdout
    return completed, json.loads(stdout_lines[0])


def test_small_offline_profile_uses_real_orchestration_and_preserves_registry() -> None:
    before = _registry_snapshot()

    completed, payload = _run()

    assert completed.returncode == 0, completed.stderr
    assert payload["schema_version"] == "myquant.quant-runtime-performance-audit.v1"
    assert payload["status"] == "pass"
    assert payload["profile"] == {
        "factor_counts": [5, 14],
        "runtime_rows": 91,
        "samples": 3,
        "seed": 20260714,
        "symbols": 24,
        "validation_rows": 91,
        "timed_warmup_per_operation": 1,
        "untimed_pre_timing_operations": [
            "fourteen_factor_runtime_input_digest",
            "production_frame_validation",
        ],
    }
    assert payload["reference_profile"] is False
    assert payload["reference_acceptance_eligible"] is False
    assert payload["orchestration"]["observed_calls"] == {
        "production_runtime_input_sha256": 2,
        "validate_production_frames": 1,
    }
    assert payload["orchestration"]["invariants_passed"] is True
    assert payload["safety"]["guarded_surface_calls"] == {
        "protocol.apply_governed_transition": 0,
        "socket.create_connection": 0,
        "subprocess.Popen": 0,
        "urllib.request.urlopen": 0,
    }
    assert payload["safety"]["guard_scope_exhaustive"] is False
    assert payload["registry_readback"]["unchanged"] is True
    assert payload["registry_readback"]["before"] == payload["registry_readback"]["after"]
    assert payload["memory"]["native_incremental_passed"] is True
    memory = payload["memory"]
    assert memory["native_incremental_bytes"] == (
        memory["native_runtime_and_validation_probe"]["incremental_bytes"]
    )
    assert memory["native_measurement"] == (
        "sampled_current_rss_common_baseline_with_both_inputs_resident"
    )
    assert set(payload["benchmarks"]["factors"]) == {"5", "14"}
    assert before == _registry_snapshot()
    assert "Traceback" not in completed.stderr


def test_combined_budget_does_not_reuse_single_operation_throughput_floor() -> None:
    def factor_result(seconds: float) -> dict[str, object]:
        operation = {"median_seconds": seconds}
        return {
            "digest": dict(operation),
            "score": dict(operation),
            "plan": dict(operation),
            "combined_median_seconds": seconds * 2.0,
            "factor_symbol_rows_per_second": {
                "digest": 100.0,
                "score": 100.0,
                "plan": 100.0,
                "combined": 50.0,
            },
        }

    budgets, blockers = audit._budgets(
        validation={"median_seconds": 1.0, "symbol_rows": 100},
        factors={"5": factor_result(1.0), "14": factor_result(1.0)},
        budget_scale=1.0,
        validation_throughput_min=100.0,
        factor_throughput_min=80.0,
        scaling_max=3.5,
        native_incremental_bytes=0,
        native_incremental_max_mib=128.0,
    )

    assert blockers == []
    assert budgets["factors"]["5"]["operation_passed"]["combined"] is True
    assert budgets["factors"]["14"]["operation_passed"]["combined"] is True


def test_native_rss_peak_update_cannot_be_overwritten_by_stale_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_ident = threading.get_ident()
    first_sampler_enter = threading.Event()
    release_sampler = threading.Event()
    sampler_second = threading.Event()
    main_calls = 0
    sampler_calls = 0

    def fake_current_rss() -> int:
        nonlocal main_calls, sampler_calls
        if threading.get_ident() == main_ident:
            main_calls += 1
            if main_calls == 1:
                return 100
            release_sampler.set()
            assert sampler_second.wait(2.0)
            return 100
        sampler_calls += 1
        if sampler_calls == 1:
            first_sampler_enter.set()
            assert release_sampler.wait(2.0)
            return 1000
        sampler_second.set()
        return 1000

    monkeypatch.setattr(audit, "_current_rss_bytes", fake_current_rss)

    def operation() -> None:
        assert first_sampler_enter.wait(2.0)

    assert audit._native_rss_probe(operation) == {
        "baseline_bytes": 100,
        "peak_bytes": 1000,
        "incremental_bytes": 900,
    }


def test_native_rss_sampler_error_propagates_and_thread_is_joined(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    main_ident = threading.get_ident()
    sampler_entered = threading.Event()

    def fake_current_rss() -> int:
        if threading.get_ident() == main_ident:
            return 100
        sampler_entered.set()
        raise OSError("injected sampler failure")

    monkeypatch.setattr(audit, "_current_rss_bytes", fake_current_rss)

    def operation() -> None:
        assert sampler_entered.wait(2.0)

    with pytest.raises(RuntimeError, match="native RSS sampler failed"):
        audit._native_rss_probe(operation)
    assert not any(
        thread.name == "quant-runtime-rss-sampler" and thread.is_alive()
        for thread in threading.enumerate()
    )


def test_reference_profile_is_exact_and_non_finite_budgets_are_rejected() -> None:
    assert audit._is_reference_profile(audit.parse_args([])) is True
    assert (
        audit._is_reference_profile(audit.parse_args(["--symbols", "24"]))
        is False
    )
    for value in ("nan", "inf", "-inf", "1e308"):
        with pytest.raises(SystemExit):
            audit.parse_args(["--budget-scale", value])


def test_registry_snapshot_rejects_symlink_and_tracks_file_identity(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(b"{}\n")
    target.chmod(0o644)
    snapshot = audit._registry_snapshot(target)
    assert snapshot.sha256 == hashlib.sha256(b"{}\n").hexdigest()
    assert snapshot.mode == 0o644
    assert snapshot.nlink == 1
    assert snapshot.uid == os.getuid()
    assert snapshot.size == 3

    target.chmod(0o1644)
    assert audit._registry_snapshot(target).mode == 0o1644
    target.chmod(0o644)

    link = tmp_path / "registry.json"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="regular non-symlink"):
        audit._registry_snapshot(link)


def test_budget_failure_is_one_json_document_and_nonzero() -> None:
    before = _registry_snapshot()

    completed, payload = _run(
        "--budget-scale",
        "0",
        "--native-incremental-max-mib",
        "0",
    )

    assert completed.returncode != 0
    assert payload["status"] == "fail"
    assert payload["reference_profile"] is False
    assert payload["reference_acceptance_eligible"] is False
    assert payload["blockers"]
    assert payload["registry_readback"]["unchanged"] is True
    assert before == _registry_snapshot()
    assert "Traceback" not in completed.stderr
