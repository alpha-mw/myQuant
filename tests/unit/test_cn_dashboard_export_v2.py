from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import export_cn_aggressive_dashboard_data as exporter  # noqa: E402
import check_cn_dashboard_export as checker  # noqa: E402
from cn_dashboard_v2_selector import build_selector, publish_selector  # noqa: E402


def _output_args(paths: tuple[Path, Path, Path, Path, Path, Path]) -> argparse.Namespace:
    return argparse.Namespace(
        json_output=paths[0],
        js_output=paths[1],
        v2_json_output=paths[2],
        v2_js_output=paths[3],
        selector_json_output=paths[4],
        selector_js_output=paths[5],
    )


@pytest.mark.parametrize("failure_index", [1, 2, 3, 4])
def test_pair_publication_restores_all_four_outputs_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = list(exporter._expected_output_paths(project_root))[:4]
    paths[0].parent.mkdir(parents=True)
    for index, path in enumerate(paths):
        path.write_bytes(f"old-{index}".encode())

    monkeypatch.setattr(exporter, "validate_bundle_shape", lambda _value: [])
    monkeypatch.setattr(exporter, "verify_source_refs", lambda *_args: [])
    monkeypatch.setattr(exporter, "validate_v2_shape", lambda _value: [])
    monkeypatch.setattr(exporter, "verify_v2_source_refs", lambda *_args, **_kwargs: [])
    original_replace = os.replace
    count = 0

    def failing_replace(source, destination):
        nonlocal count
        if Path(destination) in paths:
            count += 1
            if count == failure_index:
                raise OSError(f"injected-replace-failure-{failure_index}")
        return original_replace(source, destination)

    monkeypatch.setattr(exporter.os, "replace", failing_replace)
    with pytest.raises(OSError, match="injected-replace-failure"):
        exporter.publish_bundle_pair(
            v1_bundle={"version": 1},
            v2_bundle={"version": 2},
            v1_json_path=paths[0],
            v1_js_path=paths[1],
            v2_json_path=paths[2],
            v2_js_path=paths[3],
            project_root=project_root,
        )
    assert [path.read_bytes() for path in paths] == [
        b"old-0",
        b"old-1",
        b"old-2",
        b"old-3",
    ]


def test_default_output_paths_are_fixed_to_private_generated_root(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    args = argparse.Namespace(
        json_output=None,
        js_output=None,
        v2_json_output=None,
        v2_js_output=None,
        selector_json_output=None,
        selector_js_output=None,
    )
    assert exporter._resolve_output_paths(args, project_root) == exporter._expected_output_paths(
        project_root
    )


def test_partial_or_custom_output_paths_are_rejected(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    paths = list(exporter._expected_output_paths(project_root))
    partial_args = _output_args(tuple(paths))
    partial_args.v2_json_output = None
    with pytest.raises(
        exporter.DashboardInputError,
        match="dashboard_output_paths_must_be_complete",
    ):
        exporter._resolve_output_paths(partial_args, project_root)
    paths[0] = project_root / "public" / "cn_aggressive_dashboard.v1.json"
    with pytest.raises(
        exporter.DashboardInputError,
        match="dashboard_output_path_forbidden",
    ):
        exporter._resolve_output_paths(_output_args(tuple(paths)), project_root)


@pytest.mark.parametrize(
    "protected_relative_path",
    [
        "results/system/_active.json",
        "results/system/_migration_complete.json",
        "results/strategy_records/CN/aggressive_tech_manufacturing/_record_store/current.v1.json",
        "results/strategy_records/CN/aggressive_tech_manufacturing/catalogs/catalog.v3.json",
        "data/parquet/cn/_latest.json",
        "portfolio_dashboard/public/generated/cn_aggressive_dashboard.v1.json",
    ],
)
def test_exporter_rejects_protected_output_before_selector_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    protected_relative_path: str,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    protected = project_root / protected_relative_path
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"protected-sentinel")
    paths = list(exporter._expected_output_paths(project_root))
    paths[0] = protected
    args = argparse.Namespace(
        project_root=project_root,
        record_root=project_root / "results" / "strategy_records",
        benchmark=project_root / "benchmark.csv",
        risk_free=project_root / "risk_free.csv",
        history_integrity=project_root / "history.json",
        generated_at="2026-08-18T09:30:00+08:00",
        today="2026-08-18",
        attempt_id="dashboard-v2-protected-output-test",
        **vars(_output_args(tuple(paths))),
    )
    monkeypatch.setattr(exporter, "parse_args", lambda: args)

    assert exporter.main() == 2
    assert protected.read_bytes() == b"protected-sentinel"
    assert not exporter._expected_output_paths(project_root)[4].exists()


def test_output_paths_reject_symlinked_root_and_final_file(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    system_root = project_root / "results" / "system"
    system_root.mkdir(parents=True)
    sentinel = system_root / "_active.json"
    sentinel.write_bytes(b"active-sentinel")
    dashboard_root = project_root / "portfolio_dashboard"
    dashboard_root.mkdir(parents=True)
    (dashboard_root / "private").symlink_to(system_root, target_is_directory=True)
    default_args = argparse.Namespace(
        json_output=None,
        js_output=None,
        v2_json_output=None,
        v2_js_output=None,
        selector_json_output=None,
        selector_js_output=None,
    )
    with pytest.raises(
        exporter.DashboardInputError,
        match="dashboard_private_output_root_symlink_forbidden",
    ):
        exporter._resolve_output_paths(default_args, project_root)
    assert sentinel.read_bytes() == b"active-sentinel"

    private_root = project_root / "portfolio_dashboard" / "private"
    private_root.unlink()
    generated = private_root / "generated"
    generated.mkdir(parents=True)
    final_alias = generated / "cn_aggressive_dashboard.v1.json"
    final_alias.symlink_to(sentinel)
    with pytest.raises(
        exporter.DashboardInputError,
        match="dashboard_output_path_forbidden",
    ):
        exporter._resolve_output_paths(default_args, project_root)
    assert sentinel.read_bytes() == b"active-sentinel"


def test_write_primitives_reject_direct_protected_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    protected = project_root / "results" / "system" / "_active.json"
    protected.parent.mkdir(parents=True)
    protected.write_bytes(b"active-sentinel")
    paths = exporter._expected_output_paths(project_root)
    monkeypatch.setattr(exporter, "validate_bundle_shape", lambda _value: [])
    monkeypatch.setattr(exporter, "verify_source_refs", lambda *_args: [])
    monkeypatch.setattr(exporter, "validate_v2_shape", lambda _value: [])
    monkeypatch.setattr(exporter, "verify_v2_source_refs", lambda *_args, **_kwargs: [])

    with pytest.raises(exporter.DashboardInputError, match="dashboard_output_path_forbidden"):
        exporter.publish_bundle({"version": 1}, protected, paths[1], project_root)
    with pytest.raises(exporter.DashboardInputError, match="dashboard_output_path_forbidden"):
        exporter.publish_bundle_pair(
            v1_bundle={"version": 1},
            v2_bundle={"version": 2},
            v1_json_path=protected,
            v1_js_path=paths[1],
            v2_json_path=paths[2],
            v2_js_path=paths[3],
            project_root=project_root,
        )
    selector = build_selector(
        attempt_id="dashboard-v2-direct-writer-test",
        status="REFRESHING",
        updated_at="2026-08-18T09:30:00+08:00",
        reason="test",
    )
    with pytest.raises(ValueError, match="dashboard_output_path_forbidden"):
        publish_selector(
            selector,
            json_path=protected,
            js_path=paths[5],
            project_root=project_root,
            js_first=True,
        )
    assert protected.read_bytes() == b"active-sentinel"


def test_checker_custom_bundle_requires_complete_v1_v2_selector_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle_path = tmp_path / "custom-v1.json"
    bundle_path.write_text(json.dumps({}), encoding="utf-8")
    args = argparse.Namespace(
        project_root=tmp_path,
        bundle=bundle_path,
        js_bundle=None,
        v2_bundle=None,
        v2_js_bundle=None,
        selector=None,
        selector_js=None,
    )
    monkeypatch.setattr(checker, "parse_args", lambda: args)

    assert checker.main() == 2
    result = json.loads(capsys.readouterr().out)
    assert "custom_bundle_requires_complete_v1_v2_selector_set" in result["errors"]


def test_checker_reopens_exact_js_wrapper_payload(tmp_path: Path) -> None:
    path = tmp_path / "bundle.js"
    path.write_text('window.TestDashboard = {"value":1};\n', encoding="utf-8")
    assert checker._read_js_wrapper(path, "TestDashboard") == {"value": 1}

    path.write_text('window.TestDashboard = {"value":2};\n', encoding="utf-8")
    assert checker._read_js_wrapper(path, "TestDashboard") != {"value": 1}

    path.write_text('window.OtherDashboard = {"value":1};\n', encoding="utf-8")
    with pytest.raises(ValueError, match="js_wrapper_invalid"):
        checker._read_js_wrapper(path, "TestDashboard")


def test_checker_reports_typed_official_valuation_requirement() -> None:
    v1 = {
        "portfolio": {"performance_end_date": "2026-08-19"},
        "current_evidence": {"official_valuation": False},
    }
    v2 = {
        "freshness": {"status": "UPDATED", "mark_as_of": "2026-08-21"},
        "continuity_authority": {"status": "NO_ACTION_BOUND"},
    }
    assert checker.official_valuation_publication_requirement(v1, v2) == (
        checker.OFFICIAL_VALUATION_PUBLICATION_REQUIRED
    )


def test_checker_omits_requirement_when_canonical_date_is_current() -> None:
    v1 = {
        "portfolio": {"performance_end_date": "2026-08-21"},
        "current_evidence": {"official_valuation": True},
    }
    v2 = {
        "freshness": {"status": "UPDATED", "mark_as_of": "2026-08-21"},
        "continuity_authority": {"status": "FINANCIAL_STATE_PUBLICATION"},
    }
    assert checker.official_valuation_publication_requirement(v1, v2) is None


def test_latest_required_close_is_independent_of_event_closure(tmp_path: Path) -> None:
    from quant_investor.market.cn_benchmark_store import (
        EMPTY_POINTER_SHA256,
        REQUIRED_CODES,
        publish_generation,
    )

    market = tmp_path / "data/parquet/cn/_latest.json"
    market.parent.mkdir(parents=True)
    market.write_text(json.dumps({"latest_complete_trade_date": "20260828"}), encoding="utf-8")
    rows = [
        {
            "date": day,
            "ts_code": code,
            "close": 1000.0 + index,
            "source_system": "fixture.index_daily",
            "coverage": "exact_close",
            "value_date": day,
        }
        for day in ("2026-08-24", "2026-08-28")
        for index, code in enumerate(REQUIRED_CODES)
    ]
    publish_generation(
        tmp_path / "data/parquet/cn/benchmarks",
        rows=rows,
        generation_id="benchmark-required-date-test",
        captured_at="2026-08-28T10:00:00Z",
        expected_pointer_sha256=EMPTY_POINTER_SHA256,
        acquisition_receipt_ref={"path": "capture.json", "sha256": "a" * 64},
    )

    assert checker.latest_required_close_date(tmp_path) == "2026-08-28"
    assert checker.latest_market_close_date(tmp_path) == "2026-08-28"
    assert checker.latest_benchmark_close_date(tmp_path) == "2026-08-28"


def test_attempt_receipt_is_immutable_and_does_not_touch_selector(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    selector = exporter._expected_output_paths(project_root)[4]
    selector.parent.mkdir(parents=True)
    selector.write_bytes(b"last-good-selector")
    path = exporter._publish_attempt_receipt(
        project_root=project_root,
        attempt_id="attempt-1",
        status="BLOCKED",
        updated_at="2026-08-21T20:20:00+08:00",
        reason="benchmark tail unavailable",
        selector_sha256=exporter.hashlib.sha256(selector.read_bytes()).hexdigest(),
    )
    first = path.read_bytes()
    assert selector.read_bytes() == b"last-good-selector"
    assert json.loads(first)["status"] == "BLOCKED"
    assert path.stat().st_mode & 0o777 == 0o600
    assert (
        exporter._publish_attempt_receipt(
            project_root=project_root,
            attempt_id="attempt-1",
            status="BLOCKED",
            updated_at="2026-08-21T20:20:00+08:00",
            reason="benchmark tail unavailable",
            selector_sha256=exporter.hashlib.sha256(selector.read_bytes()).hexdigest(),
        ).read_bytes()
        == first
    )
