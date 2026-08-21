from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from quant_investor.cli import main as cli_main
from quant_investor.market.daily_components import (
    DailyComponentAPIs,
    _open_session_window,
    _sealed_nontrading_evidence,
    build_default_components,
)
from quant_investor.market.daily_maintenance import MaintenanceContext


def _json(path: Path, payload: dict) -> str:
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _context(tmp_path: Path, *, mode: str, prior=()) -> MaintenanceContext:
    receipt = tmp_path / "close.json"
    receipt_sha = _json(receipt, {"target_trade_date": "20260819"})
    attempt = tmp_path / "attempt"
    attempt.mkdir(exist_ok=True)
    return MaintenanceContext(
        workspace_root=tmp_path,
        run_root=tmp_path,
        attempt_root=attempt,
        target_date="20260819",
        attempt_slot="1620",
        mode=mode,
        close_session_receipt={"target_trade_date": "20260819"},
        close_session_receipt_path=receipt,
        close_session_receipt_sha256=receipt_sha,
        prior_stage_results=tuple(prior),
    )


def _apis(**overrides) -> DailyComponentAPIs:
    default = lambda *_args, **_kwargs: {}
    values = {
        "provider_factory": lambda: SimpleNamespace(),
        "pit_store_factory": lambda _workspace: SimpleNamespace(),
        "pit_acquire": default,
        "pit_validate": default,
        "pit_publish": default,
        "market_capture": default,
        "market_replay": default,
        "market_publish": default,
        "market_shadow": default,
        "history_audit": default,
        "macro_prepare": default,
        "macro_commit": default,
        "macro_recover": default,
        "macro_load_release": lambda **_kwargs: SimpleNamespace(
            open_dates=("20260819",),
            captured_at="2026-08-19T16:30:00+08:00",
            identity=SimpleNamespace(
                manifest_sha256="a" * 64,
                semantic_sha256="b" * 64,
            ),
        ),
        "macro_load_observations": lambda _root: (
            [],
            {
                "manifest_sha256": "c" * 64,
                "generation_manifest": {"metadata": {"local_target_trade_date": "20260818"}},
            },
        ),
        "suspension_loader": lambda _provider, _target, root: (set(), root / "none"),
    }
    values.update(overrides)
    return DailyComponentAPIs(**values)


def test_shadow_pit_fails_closed_without_private_generation_binding(tmp_path):
    scope = tmp_path / "data/cn_universe/cn_index_components.json"
    _json(scope, {"full_a": ["000001.SZ"]})
    capture = tmp_path / "capture.json"
    capture.write_bytes(b"capture")
    store = SimpleNamespace(load_generation_binding=lambda: {"discovery_pointer_sha256": "a" * 64})
    apis = _apis(
        pit_store_factory=lambda _workspace: store,
        pit_acquire=lambda *_args, **_kwargs: {
            "capture_receipt_path": str(capture),
            "capture_receipt_sha256": hashlib.sha256(b"capture").hexdigest(),
            "provider_call_count": 3,
        },
        pit_publish=lambda *_args, **_kwargs: {
            "shadow_candidate": {
                "candidate_manifest_path": "/private/candidate.json",
                "candidate_manifest_sha256": "b" * 64,
                "candidate_path": "/private/candidate.parquet",
                "candidate_sha256": "c" * 64,
            }
        },
    )
    components = build_default_components(workspace_root=tmp_path, apis=apis)

    result = components.pit(_context(tmp_path, mode="shadow"))

    assert result["status"] == "BLOCKED"
    assert result["blockers"] == ["PIT_SHADOW_GENERATION_BINDING_UNAVAILABLE"]
    assert result["write_performed"] is False


def test_open_session_window_is_ordered_parent_exclusive_target_inclusive():
    receipt = {
        "ordered_open_dates": [
            "20260814",
            "20260817",
            "20260818",
            "20260819",
        ]
    }
    assert _open_session_window(
        close_receipt=receipt,
        parent_date="20260814",
        target_date="20260819",
    ) == ["20260817", "20260818", "20260819"]


def test_open_session_window_rejects_more_than_five_sessions():
    dates = [f"202608{day:02d}" for day in range(10, 17)]
    with pytest.raises(RuntimeError, match="MARKET_SESSION_WINDOW_EXCEEDS_BOUND"):
        _open_session_window(
            close_receipt={"ordered_open_dates": dates},
            parent_date=dates[0],
            target_date=dates[-1],
        )


def test_bak_daily_nontrading_evidence_is_bounded_sealed_and_cached(tmp_path):
    import pandas as pd

    class Provider:
        def __init__(self):
            self.daily_calls = 0
            self.bak_calls = 0

        def daily(self, **_kwargs):
            self.daily_calls += 1
            return pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20260819"}])

        def bak_daily(self, **_kwargs):
            self.bak_calls += 1
            return pd.DataFrame(
                [
                    {
                        "ts_code": "000002.SZ",
                        "trade_date": "20260819",
                        "open": 0,
                        "high": 0,
                        "low": 0,
                        "close": 10,
                        "pre_close": 10,
                        "vol": 0,
                        "amount": 0,
                    }
                ]
            )

    provider = Provider()
    wrapped, symbols, reference = _sealed_nontrading_evidence(
        provider=provider,
        trade_date="20260819",
        scope_symbols=["000001.SZ", "000002.SZ"],
        excluded_symbols=set(),
        pit_binding={
            "canonical_path": "/private/pit.parquet",
            "canonical_sha256": "a" * 64,
        },
        evidence_root=tmp_path,
    )

    assert symbols == ["000002.SZ"]
    assert reference is not None
    assert Path(reference["path"]).is_file()
    assert hashlib.sha256(Path(reference["path"]).read_bytes()).hexdigest() == (reference["sha256"])
    assert provider.daily_calls == 1
    assert provider.bak_calls == 1
    wrapped.daily(trade_date="20260819")
    assert provider.daily_calls == 1


def test_macro_recovery_is_forward_only_and_deterministic_by_target(tmp_path):
    _json(
        tmp_path / "data/parquet/cn/macro_release_calendar/_latest.json",
        {"generation_id": "release-20260818"},
    )
    _json(
        tmp_path / "data/parquet/cn/macro_observations/_latest.json",
        {"metadata": {"local_target_trade_date": "20260818"}},
    )
    journal = tmp_path / "journals/macro/20260819/macro-20260819"
    journal.mkdir(parents=True)
    calls = []

    def recover(**kwargs):
        calls.append(dict(kwargs))
        if kwargs["execute_forward"]:
            return {"status": "SUCCESS", "terminal": True}
        return {
            "status": "CAN_EXECUTE_FORWARD",
            "terminal": False,
            "execute_forward_eligible": True,
        }

    components = build_default_components(
        workspace_root=tmp_path,
        apis=_apis(macro_recover=recover),
    )
    prior = (
        {
            "stage": "PIT",
            "evidence": {
                "scope_path": "/scope",
                "scope_sha256": "a" * 64,
                "pit_binding": {
                    "discovery_pointer_path": "/pit-pointer",
                    "discovery_pointer_sha256": "d" * 64,
                },
            },
        },
        {
            "stage": "MARKET",
            "evidence": {
                "snapshot_manifest_path": "/snapshot",
                "snapshot_manifest_sha256": "b" * 64,
                "pointer_path": "/market-pointer",
                "pointer_sha256": "e" * 64,
            },
        },
        {"stage": "HISTORY", "evidence": {"history_audit_status": "passed"}},
    )

    result = components.macro_release(_context(tmp_path, mode="execute", prior=prior))

    assert result["status"] == "READY"
    assert result["write_performed"] is True
    assert [call["execute_forward"] for call in calls] == [False, True]
    assert {call["journal_run_id"] for call in calls} == {"macro-20260819"}
    assert all("execute_rollback" not in call for call in calls)
    assert {call["market_pointer_path"] for call in calls} == {"/market-pointer"}
    assert {call["expected_market_pointer_sha256"] for call in calls} == {"e" * 64}
    assert {call["pit_pointer_path"] for call in calls} == {"/pit-pointer"}
    assert {call["expected_pit_pointer_sha256"] for call in calls} == {"d" * 64}


def test_same_target_market_rebind_recaptures_under_exact_cas(tmp_path):
    import pandas as pd

    scope = tmp_path / "data/cn_universe/cn_index_components.json"
    scope_sha = _json(scope, {"full_a": ["000001.SZ"]})
    manifest = tmp_path / "data/parquet/cn/_snapshots/old.json"
    _json(manifest, {"snapshot_id": "old"})
    pointer = tmp_path / "data/parquet/cn/_latest.json"
    old_pointer_sha = _json(
        pointer,
        {
            "latest_complete_trade_date": "20260819",
            "manifest_path": str(manifest),
            "coverage": {
                "pit_generation_id": "old-pit",
                "pit_generation_manifest_sha256": "1" * 64,
                "pit_membership_sha256": "2" * 64,
            },
        },
    )
    observed = {}

    class Provider:
        def daily(self, **_kwargs):
            return pd.DataFrame([{"ts_code": "000001.SZ", "trade_date": "20260819"}])

    capture_manifest = tmp_path / "capture.json"

    def capture(**kwargs):
        observed.update(kwargs)
        _json(capture_manifest, {"status": "CAPTURED"})
        return {
            "manifest_path": str(capture_manifest),
            "manifest_sha256": hashlib.sha256(capture_manifest.read_bytes()).hexdigest(),
        }

    def publish(**_kwargs):
        new_manifest = tmp_path / "data/parquet/cn/_snapshots/new.json"
        _json(new_manifest, {"snapshot_id": "new"})
        _json(
            pointer,
            {
                "latest_complete_trade_date": "20260819",
                "manifest_path": str(new_manifest),
                "coverage": {
                    "pit_generation_id": "new-pit",
                    "pit_generation_manifest_sha256": "3" * 64,
                    "pit_membership_sha256": "4" * 64,
                },
            },
        )
        return {"classification": {"status": "PASSED"}}

    components = build_default_components(
        workspace_root=tmp_path,
        apis=_apis(
            provider_factory=Provider,
            market_capture=capture,
            market_replay=lambda **_kwargs: {"status": "REPLAYED"},
            market_publish=publish,
        ),
    )
    pit_binding = {
        "generation_id": "new-pit",
        "generation_manifest_path": "/new-pit-manifest",
        "generation_manifest_sha256": "3" * 64,
        "canonical_path": "/new-pit",
        "canonical_sha256": "4" * 64,
        "discovery_pointer_path": "/new-pit-pointer",
        "discovery_pointer_sha256": "5" * 64,
    }
    context = _context(
        tmp_path,
        mode="execute",
        prior=(
            {
                "stage": "PIT",
                "evidence": {
                    "scope_path": str(scope),
                    "scope_sha256": scope_sha,
                    "pit_binding": pit_binding,
                    "reason_sets": {
                        "suspended": [],
                        "non_trading": [],
                        "delisted": [],
                        "prelisting": [],
                        "inactive": [],
                    },
                    "classification_evidence": {},
                },
            },
        ),
    )
    context = MaintenanceContext(
        **{
            **context.__dict__,
            "close_session_receipt": {
                "target_trade_date": "20260819",
                "ordered_open_dates": ["20260819"],
            },
        }
    )

    result = components.market(context)

    assert result["status"] == "READY"
    assert result["evidence"]["same_target_pit_rebind"] is True
    assert observed["target_trade_dates"] == ["20260819"]
    assert observed["parent_latest_complete_trade_date"] == "20260819"
    assert observed["same_target_rebind"] is True
    assert observed["expected_market_pointer_sha256"] == old_pointer_sha
    current = json.loads(pointer.read_text())
    assert current["coverage"]["pit_generation_id"] == "new-pit"


@pytest.mark.parametrize("release_open_date", ["20260819", "2026-08-19"])
def test_macro_no_action_requires_semantic_market_manifest_binding(
    tmp_path, release_open_date
):
    _json(
        tmp_path / "data/parquet/cn/macro_release_calendar/_latest.json",
        {"generation_id": "opaque-release-id"},
    )
    _json(
        tmp_path / "data/parquet/cn/macro_observations/_latest.json",
        {"metadata": {"local_target_trade_date": "20260819"}},
    )
    market_sha = "b" * 64
    scope_sha = "a" * 64
    apis = _apis(
        macro_prepare=lambda **_kwargs: pytest.fail("no-action must not prepare"),
        macro_load_release=lambda **_kwargs: SimpleNamespace(
            open_dates=(release_open_date,),
            captured_at="2026-08-19T16:30:00+08:00",
            identity=SimpleNamespace(
                manifest_sha256="a" * 64,
                semantic_sha256="b" * 64,
            ),
        ),
        macro_load_observations=lambda _root: (
            [],
            {
                "manifest_sha256": "c" * 64,
                "generation_manifest": {
                    "metadata": {
                        "local_target_trade_date": "20260819",
                        "local_snapshot_manifest_sha256": market_sha,
                        "local_coverage_manifest_sha256": market_sha,
                        "local_scope_artifact_sha256": scope_sha,
                    }
                },
            },
        ),
    )
    components = build_default_components(workspace_root=tmp_path, apis=apis)
    prior = (
        {
            "stage": "PIT",
            "evidence": {"scope_path": "/scope", "scope_sha256": scope_sha},
        },
        {
            "stage": "MARKET",
            "evidence": {
                "snapshot_manifest_path": "/snapshot",
                "snapshot_manifest_sha256": market_sha,
            },
        },
        {"stage": "HISTORY", "evidence": {"history_audit_status": "passed"}},
    )

    result = components.macro_release(_context(tmp_path, mode="execute", prior=prior))

    assert result["status"] == "NO_ACTION"
    assert result["write_performed"] is False
    assert result["evidence"]["market_manifest_sha256"] == market_sha


@pytest.mark.parametrize(
    ("mode", "expected_authority"),
    [("shadow", "candidate"), ("execute", "canonical")],
)
def test_daily_macro_prepare_and_commit_receive_exact_authority(tmp_path, mode, expected_authority):
    _json(
        tmp_path / "data/parquet/cn/macro_release_calendar/_latest.json",
        {"generation_id": "release-20260818"},
    )
    _json(
        tmp_path / "data/parquet/cn/macro_observations/_latest.json",
        {"metadata": {"local_target_trade_date": "20260818"}},
    )
    prepared_calls = []
    commit_calls = []

    def prepare(**kwargs):
        prepared_calls.append(dict(kwargs))
        return {
            "status": "PREPARED",
            "prepared_path": "/private/prepared.json",
            "prepared_sha256": "f" * 64,
        }

    def commit(**kwargs):
        commit_calls.append(dict(kwargs))
        return {"status": "SUCCESS"}

    components = build_default_components(
        workspace_root=tmp_path,
        apis=_apis(macro_prepare=prepare, macro_commit=commit),
    )
    prior = (
        {
            "stage": "PIT",
            "evidence": {
                "scope_path": "/scope",
                "scope_sha256": "a" * 64,
                "pit_binding": {
                    "discovery_pointer_path": "/pit-pointer",
                    "discovery_pointer_sha256": "b" * 64,
                },
            },
        },
        {
            "stage": "MARKET",
            "evidence": {
                "snapshot_manifest_path": "/snapshot",
                "snapshot_manifest_sha256": "c" * 64,
                "pointer_path": "/market-pointer",
                "pointer_sha256": "d" * 64,
            },
        },
        {"stage": "HISTORY", "evidence": {"history_audit_status": "passed"}},
    )

    result = components.macro_release(_context(tmp_path, mode=mode, prior=prior))

    assert result["status"] == "READY"
    assert prepared_calls[0]["authority_mode"] == expected_authority
    assert prepared_calls[0]["market_pointer_path"] == "/market-pointer"
    assert prepared_calls[0]["pit_pointer_path"] == "/pit-pointer"
    if mode == "execute":
        assert commit_calls[0]["market_pointer_path"] == "/market-pointer"
        assert commit_calls[0]["expected_market_pointer_sha256"] == "d" * 64
        assert commit_calls[0]["pit_pointer_path"] == "/pit-pointer"
        assert commit_calls[0]["expected_pit_pointer_sha256"] == "b" * 64
    else:
        assert commit_calls == []


def test_macro_cli_recovery_matrix_dispatches_forward_only(monkeypatch, capsys):
    observed = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {"status": "SUCCESS"}

    monkeypatch.setattr(cli_main, "run_macro_maintenance", fake_run)
    cli_main.main(
        [
            "market",
            "macro-maintain",
            "--market",
            "CN",
            "--recover",
            "--execute-forward",
            "--journal-root",
            "/private/tmp/macro-journal",
            "--journal-run-id",
            "macro-20260819",
            "--market-pointer-path",
            "/private/tmp/market-pointer.json",
            "--expected-market-pointer-sha256",
            "a" * 64,
            "--pit-pointer-path",
            "/private/tmp/pit-pointer.json",
            "--expected-pit-pointer-sha256",
            "b" * 64,
        ]
    )

    assert json.loads(capsys.readouterr().out)["status"] == "SUCCESS"
    assert observed["recover"] is True
    assert observed["execute_forward"] is True
    assert observed["execute_rollback"] is False
    assert observed["journal_run_id"] == "macro-20260819"
    assert observed["market_pointer_path"] == "/private/tmp/market-pointer.json"
    assert observed["pit_pointer_path"] == "/private/tmp/pit-pointer.json"


def test_macro_wrapper_passes_authority_to_prepare(monkeypatch, tmp_path):
    from quant_investor.macro import maintenance as macro_maintenance

    observed = {}

    def fake_prepare(**kwargs):
        observed.update(kwargs)
        return {"status": "PREPARED"}

    monkeypatch.setattr(
        macro_maintenance,
        "prepare_cn_macro_maintenance_transaction",
        fake_prepare,
    )
    result = cli_main.run_macro_maintenance(
        prepare_transaction=True,
        commit=False,
        authority_mode="candidate",
        journal_root=str(tmp_path / "journal"),
        journal_run_id="macro-20260819",
        market_pointer_path="/private/market.json",
        expected_market_pointer_sha256="a" * 64,
        pit_pointer_path="/private/pit.json",
        expected_pit_pointer_sha256="b" * 64,
        market="CN",
        target_date="20260819",
        allow_live=True,
    )

    assert result["status"] == "PREPARED"
    assert observed["authority_mode"] == "candidate"
    assert observed["market_pointer_path"] == "/private/market.json"
    assert observed["expected_market_pointer_sha256"] == "a" * 64
    assert observed["pit_pointer_path"] == "/private/pit.json"
    assert observed["expected_pit_pointer_sha256"] == "b" * 64


@pytest.mark.parametrize(
    ("mode", "function_name"),
    [
        ("commit", "commit_prepared_macro_transaction"),
        ("recover", "recover_macro_transaction"),
        ("rollback", "rollback_macro_transaction"),
    ],
)
def test_macro_wrapper_passes_authority_to_terminal_paths(
    monkeypatch, tmp_path, mode, function_name
):
    from quant_investor.macro import maintenance as macro_maintenance

    observed = {}

    def fake_call(**kwargs):
        observed.update(kwargs)
        return {"status": "SUCCESS"}

    monkeypatch.setattr(macro_maintenance, function_name, fake_call)
    common = {
        "commit": False,
        "journal_root": str(tmp_path / "journal"),
        "journal_run_id": "macro-20260819",
        "market_pointer_path": "/private/market.json",
        "expected_market_pointer_sha256": "a" * 64,
        "pit_pointer_path": "/private/pit.json",
        "expected_pit_pointer_sha256": "b" * 64,
    }
    if mode == "commit":
        common.update(
            {
                "commit_prepared": True,
                "prepared_path": "/private/prepared.json",
                "expected_prepared_sha256": "c" * 64,
            }
        )
    elif mode == "recover":
        common.update({"recover": True, "execute_forward": True})
    else:
        common.update(
            {
                "recover": True,
                "execute_rollback": True,
                "old_release_pointer_sha256": "d" * 64,
                "new_release_pointer_sha256": "e" * 64,
                "old_observations_pointer_sha256": "f" * 64,
                "new_observations_pointer_sha256": "1" * 64,
            }
        )

    result = cli_main.run_macro_maintenance(**common)

    assert result["status"] == "SUCCESS"
    assert observed["market_pointer_path"] == "/private/market.json"
    assert observed["expected_market_pointer_sha256"] == "a" * 64
    assert observed["pit_pointer_path"] == "/private/pit.json"
    assert observed["expected_pit_pointer_sha256"] == "b" * 64
