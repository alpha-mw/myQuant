"""Native capture/store wire fixtures; no real provider or portfolio activity."""

import json
import os
import pickle
import threading

import pytest

from _daily_preparation_fixture import put, snapshot
from _native_source_producer_fixture import BenchmarkClient, NOW
from quant_investor.market import cn_benchmark_capture as module
from quant_investor.market import cn_benchmark_store as native


def seed(root):
    capture = put(root, "fixtures/seed-capture.json", {"synthetic": True})
    return native.publish_generation(
        root / "data/parquet/cn/benchmarks",
        rows=[
            {
                "date": "2026-08-21",
                "ts_code": code,
                "close": 1100.0,
                "source_system": "tushare.index_daily",
                "coverage": "exact_close",
                "value_date": "2026-08-21",
            }
            for code in native.REQUIRED_CODES
        ],
        generation_id="synthetic-seed",
        captured_at="2026-08-21T12:00:00Z",
        expected_pointer_sha256=native.EMPTY_POINTER_SHA256,
        acquisition_receipt_ref=capture,
    )


def arguments(root, monkeypatch):
    parent = seed(root)
    monkeypatch.setattr(module, "TUSHARE_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(module, "_now", lambda: NOW)
    return dict(
        workspace_root=root,
        start_date="2026-08-24",
        end_date="2026-08-25",
        generation_id="synthetic-capture",
        expected_pointer_sha256=parent["pointer_sha256"],
        required_dates=["2026-08-24", "2026-08-25"],
    )


def forbidden(*args, **kwargs):
    raise AssertionError("provider/token/sleep must not run")


def test_native_wire_capture_and_readonly_repeat(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    client = BenchmarkClient()
    before_token = dict(os.environ)
    result = module.capture_benchmark_close(**args, client=client)
    assert result["status"] == "PUBLISHED" and len(client.calls) == 3
    assert dict(os.environ) == before_token
    receipt = json.loads((tmp_path / result["receipt_path"]).read_bytes())
    assert receipt["parent_pointer_ref"]["sha256"] == args["expected_pointer_sha256"]
    assert receipt["row_count"] == 6 and len(receipt["requests"]) == 3
    assert len(native.load_generation(tmp_path / "data/parquet/cn/benchmarks")["rows"]) == 9
    before = snapshot(tmp_path)
    monkeypatch.setattr(module, "OfficialTushareHttpsClient", forbidden)
    monkeypatch.setattr(module.time, "sleep", forbidden)
    again = module.capture_benchmark_close(**args)
    assert again["status"] == "NO_ACTION" and again["receipt_sha256"] == result["receipt_sha256"]
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("fault", ["date", "code", "duplicate", "negative", "infinite", "missing"])
def test_bad_wire_cannot_publish(tmp_path, monkeypatch, fault):
    args = arguments(tmp_path, monkeypatch)

    def mutate(items, params):
        if fault == "date":
            items[0][1] = "20260823"
        elif fault == "code":
            items[0][0] = "999999.SH"
        elif fault == "duplicate":
            items.append(items[0])
        elif fault == "negative":
            items[0][2] = -1
        elif fault == "infinite":
            items[0][2] = "Infinity"
        elif fault == "missing":
            items = items[:-1]
        return items

    with pytest.raises(Exception):
        module.capture_benchmark_close(**args, client=BenchmarkClient(fault=mutate))
    assert (
        native.pointer_sha256(tmp_path / "data/parquet/cn/benchmarks")
        == args["expected_pointer_sha256"]
    )
    assert not (
        tmp_path / "data/private/cn_benchmark_close/synthetic-capture/capture.v2.json"
    ).exists()


def test_overlapping_history_is_not_replaced(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    args.update(start_date="2026-08-21", required_dates=["2026-08-21", "2026-08-24", "2026-08-25"])
    with pytest.raises(module.BenchmarkCaptureError, match="EXISTING_ROW_CONFLICT"):
        module.capture_benchmark_close(
            **args, client=BenchmarkClient(days=("20260821", "20260824", "20260825"))
        )
    assert (
        native.pointer_sha256(tmp_path / "data/parquet/cn/benchmarks")
        == args["expected_pointer_sha256"]
    )


def test_capture_commitment_recovers_without_current_parent_inference(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    original = native._write_compatibility
    monkeypatch.setattr(
        native,
        "_write_compatibility",
        lambda *a: (_ for _ in ()).throw(OSError("synthetic post-CAS failure")),
    )
    with pytest.raises(OSError, match="post-CAS"):
        module.capture_benchmark_close(**args, client=BenchmarkClient())
    pointer = native.pointer_sha256(tmp_path / "data/parquet/cn/benchmarks")
    assert pointer != args["expected_pointer_sha256"]
    monkeypatch.setattr(native, "_write_compatibility", original)
    monkeypatch.setattr(module, "OfficialTushareHttpsClient", forbidden)
    monkeypatch.setattr(module.time, "sleep", forbidden)
    result = module.capture_benchmark_close(**args)
    assert result["pointer_sha256"] == pointer and result["compatibility_repaired"] is True
    parent = json.loads(
        (
            tmp_path / "data/private/cn_benchmark_close/synthetic-capture/parent-pointer.json"
        ).read_bytes()
    )
    assert parent["generation_id"] == "synthetic-seed"


@pytest.mark.parametrize("fault", ["parent", "raw", "arguments", "v1"])
def test_committed_capture_conflict_never_recaptures(tmp_path, monkeypatch, fault):
    args = arguments(tmp_path, monkeypatch)
    result = module.capture_benchmark_close(**args, client=BenchmarkClient())
    receipt = json.loads((tmp_path / result["receipt_path"]).read_bytes())
    if fault == "parent":
        (tmp_path / receipt["parent_pointer_ref"]["path"]).write_bytes(b"{}")
    elif fault == "raw":
        (tmp_path / receipt["requests"][0]["raw_ref"]["path"]).write_bytes(b"{}")
    elif fault == "arguments":
        args["required_dates"] = ["2026-08-25"]
    else:
        put(
            tmp_path,
            "data/private/cn_benchmark_close/synthetic-capture/capture.v1.json",
            {"legacy": True},
        )
    before = snapshot(tmp_path)
    monkeypatch.setattr(module, "OfficialTushareHttpsClient", forbidden)
    monkeypatch.setattr(module.time, "sleep", forbidden)
    with pytest.raises(Exception):
        module.capture_benchmark_close(**args)
    assert snapshot(tmp_path) == before


def test_foreign_successor_cannot_restore_older_compatibility(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    module.capture_benchmark_close(**args, client=BenchmarkClient())
    root = tmp_path / "data/parquet/cn/benchmarks"
    current = native.load_generation(root)
    native.publish_generation(
        root,
        rows=current["rows"],
        generation_id="foreign-successor",
        captured_at="2026-08-25T12:21:00Z",
        expected_pointer_sha256=current["pointer_sha256"],
        acquisition_receipt_ref=put(tmp_path, "fixtures/foreign.json", {"synthetic": True}),
    )
    before = snapshot(tmp_path)
    monkeypatch.setattr(module, "OfficialTushareHttpsClient", forbidden)
    with pytest.raises(native.CNBenchmarkCASMismatch):
        module.capture_benchmark_close(**args)
    assert snapshot(tmp_path) == before


def test_existing_publishers_wait_for_candidate_projection(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    real = native._write_compatibility
    entered, done, threads, errors = threading.Event(), threading.Event(), [], []

    def projection(scope, workspace, rows):
        current = native.load_generation(scope.root)

        def successor():
            entered.set()
            try:
                native.publish_generation(
                    scope.root,
                    rows=rows,
                    generation_id="concurrent-successor",
                    captured_at="2026-08-25T12:21:00Z",
                    expected_pointer_sha256=current["pointer_sha256"],
                    acquisition_receipt_ref={"path": "fixtures/successor.json", "sha256": "b" * 64},
                )
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=successor)
        threads.append(thread)
        thread.start()
        assert entered.wait(2)
        assert not done.wait(0.05)
        result = real(scope, workspace, rows)
        assert native.pointer_sha256(scope.root) == current["pointer_sha256"]
        assert (
            workspace / "portfolio_dashboard/inputs/cn_index_benchmark.csv"
        ).read_bytes() == native.compatibility_csv_bytes(rows)
        return result

    monkeypatch.setattr(native, "_write_compatibility", projection)
    module.capture_benchmark_close(**args, client=BenchmarkClient())
    for thread in threads:
        thread.join(5)
    assert done.is_set() and not errors


def test_publication_scope_is_revoked_and_not_serializable(tmp_path):
    with native.publication_scope(tmp_path) as scope:
        with pytest.raises(TypeError):
            pickle.dumps(scope)
    with pytest.raises(native.CNBenchmarkStoreError):
        scope.require(tmp_path)


@pytest.mark.parametrize("old", [None, "inherited-test-value"])
def test_cli_credential_scope_restores_environment(tmp_path, monkeypatch, old):
    if old is None:
        monkeypatch.delenv("TUSHARE_TOKEN", raising=False)
    else:
        monkeypatch.setenv("TUSHARE_TOKEN", old)
    monkeypatch.setattr(module, "read_project_env_token", lambda _: "synthetic-project-value")
    with pytest.raises(RuntimeError):
        with module._project_credentials(tmp_path):
            assert os.environ["TUSHARE_TOKEN"] == "synthetic-project-value"
            raise RuntimeError("synthetic")
    assert os.environ.get("TUSHARE_TOKEN") == old


def test_cli_plan_and_warm_capture_do_not_access_credentials(tmp_path, monkeypatch, capsys):
    args = arguments(tmp_path, monkeypatch)
    argv = []
    for name in (
        "workspace_root",
        "start_date",
        "end_date",
        "generation_id",
        "expected_pointer_sha256",
    ):
        argv.extend(["--" + name.replace("_", "-"), str(args[name])])
    monkeypatch.setattr(module, "read_project_env_token", forbidden)
    before = snapshot(tmp_path)
    assert module.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_ONLY"
    assert snapshot(tmp_path) == before
    args["required_dates"] = None  # CLI does not itself claim Calendar completeness.
    module.capture_benchmark_close(**args, client=BenchmarkClient())
    monkeypatch.setattr(module, "OfficialTushareHttpsClient", forbidden)
    before = snapshot(tmp_path)
    assert module.main(argv + ["--execute"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "NO_ACTION"
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("fault", ["pagination", "count"])
def test_wire_envelope_cannot_claim_complete_when_truncated(tmp_path, monkeypatch, fault):
    args = arguments(tmp_path, monkeypatch)
    from types import SimpleNamespace

    class Client(BenchmarkClient):
        def request(self, **kwargs):
            response = super().request(**kwargs)
            value = json.loads(response.raw_body)
            if fault == "pagination":
                value["data"]["has_more"] = True
            else:
                value["data"]["count"] = 999
            return SimpleNamespace(raw_body=json.dumps(value).encode())

    with pytest.raises(Exception):
        module.capture_benchmark_close(**args, client=Client())
    assert (
        native.pointer_sha256(tmp_path / "data/parquet/cn/benchmarks")
        == args["expected_pointer_sha256"]
    )


def test_compatibility_only_repair_and_equal_overlap(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    first = module.capture_benchmark_close(**args, client=BenchmarkClient())
    (tmp_path / "portfolio_dashboard/inputs/cn_index_benchmark.csv").write_bytes(b"stale")
    monkeypatch.setattr(module, "OfficialTushareHttpsClient", forbidden)
    repaired = module.capture_benchmark_close(**args)
    assert (
        repaired["compatibility_repaired"] is True
        and repaired["pointer_sha256"] == first["pointer_sha256"]
    )
    args.update(
        generation_id="synthetic-equal-overlap", expected_pointer_sha256=first["pointer_sha256"]
    )
    second = module.capture_benchmark_close(**args, client=BenchmarkClient())
    assert second["row_count"] == 9  # Existing rows are not duplicated.
