from __future__ import annotations

import json
from pathlib import Path
import shutil
import stat

from quant_investor.v17.latest import publish_terminal_latest
from quant_investor.v17.semantic import seal_semantic
from quant_investor.v17.state_machine import (
    EMPTY_SHA,
    TERMINAL_OUTPUT_VERSION,
    advance_run_state,
    initialize_run,
)
from scripts.export_v17_shadow_dashboard import export_v17_shadow_dashboard

ROOT = Path(__file__).resolve().parents[2]
TIMES = (
    "2026-07-22T07:00:00Z",
    "2026-07-22T07:01:00Z",
    "2026-07-22T07:02:00Z",
    "2026-07-22T07:03:00Z",
    "2026-07-22T07:04:00Z",
)


def _install_schema(repo: Path) -> None:
    destination = repo / "portfolio_dashboard/schema/dashboard_contract.v17-shadow.schema.json"
    destination.parent.mkdir(parents=True)
    shutil.copy2(
        ROOT / "portfolio_dashboard/schema/dashboard_contract.v17-shadow.schema.json",
        destination,
    )


def _artifact(version: str) -> dict[str, object]:
    return seal_semantic(
        {
            "version": version,
            "value": "synthetic",
            "authority": False,
        }
    )


def _publish_terminal(repo: Path) -> tuple[dict[str, object], str]:
    ledger, ledger_sha = initialize_run(
        repo,
        run_id="cn-v17-dashboard-export",
        strategy_id="cn-shadow",
        cutoff=TIMES[0],
        prepared_at=TIMES[0],
        input_bindings={
            "source_manifest_sha256": "a" * 64,
            "source_manifest_path": ("data/private/v17_sources/manifests/dashboard-test.json"),
        },
        expected_ledger_sha256=EMPTY_SHA,
    )
    for state, role, timestamp in (
        ("DETERMINISTIC_COMPLETE", "deterministic_result", TIMES[1]),
        ("DEEP_REQUEST_READY", "deep_request", TIMES[2]),
        ("DEEP_RESPONSE_RECEIVED", "deep_response", TIMES[3]),
    ):
        ledger, ledger_sha = advance_run_state(
            repo,
            run_id=ledger["run_id"],
            expected_ledger_sha256=ledger_sha,
            next_state=state,
            transitioned_at=timestamp,
            artifacts={role: _artifact(f"test.{role}.v1")},
        )
    terminal = seal_semantic(
        {
            "version": TERMINAL_OUTPUT_VERSION,
            "run_id": ledger["run_id"],
            "strategy_id": ledger["strategy_id"],
            "market": "CN",
            "cutoff": ledger["cutoff"],
            "terminal_state": "SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
            "rank_output": {"ranked_symbols": ["000001.SZ"]},
            "portfolio_output": None,
            "blockers": ["source_unavailable:risk_policy"],
            "source_manifest_sha256": "a" * 64,
            "ledger_predecessor_sha256": ledger_sha,
            "generated_at": TIMES[4],
            "authority": False,
        }
    )
    ledger, ledger_sha = advance_run_state(
        repo,
        run_id=ledger["run_id"],
        expected_ledger_sha256=ledger_sha,
        next_state="SHADOW_RANK_COMPLETE_NO_PORTFOLIO",
        transitioned_at=TIMES[4],
        terminal_output=terminal,
    )
    pointer, pointer_sha = publish_terminal_latest(
        repo,
        run_id=ledger["run_id"],
        expected_ledger_sha256=ledger_sha,
        expected_latest_sha256=EMPTY_SHA,
        published_at=TIMES[4],
    )
    return pointer, pointer_sha


def _loader_contract(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    marker = "window.V17ShadowLatest = "
    payload = text[text.index(marker) + len(marker) :].strip().removesuffix(";")
    parsed = json.loads(payload)
    assert isinstance(parsed, dict)
    return parsed


def test_missing_latest_exports_unavailable_without_fallback(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _install_schema(repo)

    result = export_v17_shadow_dashboard(
        repo_root=repo,
        generated_at=TIMES[4],
    )

    assert result["availability"] == "UNAVAILABLE"
    assert result["reason"] == "v17_latest_pointer_missing"
    loader = repo / "portfolio_dashboard/generated/v17_shadow_latest.js"
    assert stat.S_IMODE(loader.stat().st_mode) == 0o600
    contract = _loader_contract(loader)
    assert contract["availability"] == "UNAVAILABLE"
    assert contract["latest_pointer"] is None
    assert contract["terminal_output"] is None
    assert contract["source"]["fallback_used"] is False


def test_exact_terminal_latest_exports_hash_bound_available_loader(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _install_schema(repo)
    pointer, pointer_sha = _publish_terminal(repo)

    result = export_v17_shadow_dashboard(
        repo_root=repo,
        generated_at=TIMES[4],
    )

    assert result["availability"] == "AVAILABLE"
    loader = repo / "portfolio_dashboard/generated/v17_shadow_latest.js"
    contract = _loader_contract(loader)
    assert contract["availability"] == "AVAILABLE"
    assert contract["latest_pointer"] == pointer
    assert contract["terminal_output"]["run_id"] == pointer["run_id"]
    assert contract["source"] == {
        "path": "results/v17_shadow/_latest/shadow.json",
        "latest_pointer_sha256": pointer_sha,
        "ledger_sha256": pointer["ledger_sha256"],
        "output_sha256": pointer["output_sha256"],
        "readback_verified": True,
        "fallback_used": False,
    }
    assert contract["authority"] is False
