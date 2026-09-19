"""Adapter boundary tests with real pool/journal writers, synthetic Factor seam.

These deliberately do not count as full-DAG or full Factor verification tests.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.intelligence.storage import (
    DailyResearchPoolStore,
    approved_theme_policy_v2,
    publish_theme_policy_v2,
)
from quant_investor.operations import core_pool
from quant_investor.operations.daily_journal import DailyJournal
from test_unified_daily_intelligence_storage import _pool_rank


def fixture(tmp_path, monkeypatch):
    publish_theme_policy_v2(tmp_path)
    raw = canonical_json_bytes({"synthetic_pointer": True})
    sha = hashlib.sha256(raw).hexdigest()
    selected = {"factor_generation": {"created_at": "2026-09-06T03:29:00Z"}}

    class FakeFactor:
        def read(self, path):
            return SimpleNamespace(data=raw, byte_sha256=sha)

        def read_historical_research_inputs(self, **kwargs):
            assert kwargs == {"expected_pointer_sha256": sha, "expected_trade_date": "20260904"}
            return selected

    monkeypatch.setattr(core_pool, "FactorProductionStore", lambda _: FakeFactor())
    rank = _pool_rank(tmp_path, approved_theme_policy_v2(), signal_date="20260904", pointer_sha=sha)
    monkeypatch.setattr(core_pool, "build_factor_research_rank", lambda **kwargs: rank)
    release = tmp_path / "release.json"
    release.write_bytes(canonical_json_bytes({"synthetic_release": True}))
    release.chmod(0o600)
    # Core-source validation is the fixture boundary; pool and journal writers
    # remain real. A separate test covers the native observation binding checks.
    monkeypatch.setattr(
        core_pool.CoreContext,
        "core_outputs",
        lambda self, node: {"fixture_core_source": self.source("release.json")[1]},
    )

    def observation(self, alias, snapshot):
        return self.source(f"results/factors/observations/2026/09/04/{alias}.json")

    monkeypatch.setattr(core_pool.CoreContext, "observation", observation)
    return {
        "workspace": str(tmp_path),
        "trade_date": "20260904",
        "factor_pointer_sha256": sha,
        "release_ref": {
            "path": "release.json",
            "sha256": hashlib.sha256(release.read_bytes()).hexdigest(),
        },
    }


def test_core_adapter_publishes_pool_and_identical_replay_has_zero_writer_calls(
    tmp_path, monkeypatch
):
    arguments = fixture(tmp_path, monkeypatch)
    real_publish = DailyResearchPoolStore.publish
    calls = []

    def publish(self, **kwargs):
        calls.append(1)
        return real_publish(self, **kwargs)

    monkeypatch.setattr(DailyResearchPoolStore, "publish", publish)
    first = core_pool.publish_core_pool(**arguments)
    assert first["command_status"] == "PUBLISHED" and first["state"] == "SUCCEEDED"
    assert set(first["output_refs"]) == {
        "factor_research_rank.json",
        "manifest.json",
        "publish_receipt.json",
        "selected_symbols.json",
        "top100.parquet",
    }
    second = core_pool.publish_core_pool(**arguments)
    assert second["command_status"] == "NO_ACTION"
    assert second["terminal_ref"] == first["terminal_ref"]
    assert len(calls) == 1
    handoff = json.loads((tmp_path / first["core_handoff_ref"]["path"]).read_bytes())
    assert set(handoff["node_refs"]) == set(core_pool.CORE_NODES)
    assert second["core_handoff_ref"] == first["core_handoff_ref"]
    for ref in handoff["node_refs"].values():
        raw = (tmp_path / ref["path"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ref["sha256"]
        assert json.loads(raw)["state"] == "SUCCEEDED"


def test_crash_after_native_publish_adopts_without_repeating_business_write(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    real_finish = DailyJournal.finish

    def crash(self, *args, **kwargs):
        if args[0]["node_id"] == "top100":
            raise RuntimeError("process interrupted before terminal receipt")
        return real_finish(self, *args, **kwargs)

    monkeypatch.setattr(DailyJournal, "finish", crash)
    with pytest.raises(core_pool.ContractError, match="CORE_HANDOFF_INCOMPLETE"):
        core_pool.publish_core_pool(**arguments)
    monkeypatch.setattr(DailyJournal, "finish", real_finish)

    def no_write(*args, **kwargs):
        pytest.fail("native pool writer repeated after committed output")

    monkeypatch.setattr(DailyResearchPoolStore, "publish", no_write)
    result = core_pool.publish_core_pool(**arguments)
    assert result["command_status"] == "ADOPTED"
    assert result["terminal"]["recovered"] is True
    assert result["attempt"] == 1


def test_existing_pool_tamper_fails_before_publisher(tmp_path, monkeypatch):
    arguments = fixture(tmp_path, monkeypatch)
    result = core_pool.publish_core_pool(**arguments)
    ref = result["terminal"]["output_refs"]["selected_symbols.json"]
    (tmp_path / ref["path"]).write_bytes(b"{}")
    monkeypatch.setattr(DailyResearchPoolStore, "publish", lambda **_: pytest.fail("writer called"))
    with pytest.raises(Exception):
        core_pool.publish_core_pool(**arguments)


def test_registered_core_hook_runs_pool_before_settlement_and_auxiliary_return(
    tmp_path, monkeypatch
):
    from quant_investor.market import daily_factor_loop as module

    loop = object.__new__(module.DailyFactorLoop)
    loop.workspace = tmp_path
    loop.run_root = tmp_path
    loop.context = {"release_install_input_ref": {"path": "release.json", "sha256": "a" * 64}}
    release_ref = {"path": "release-artifact.json", "sha256": "f" * 64}
    loop._core_release_ref = lambda: release_ref
    loop.store = SimpleNamespace(read=lambda _: SimpleNamespace(byte_sha256="b" * 64))
    loop.stages = {}
    monkeypatch.setattr(
        module,
        "validate_daily_maintenance_receipt",
        lambda **_: {
            "target_date": "20260904",
            "close_session_receipt_ref": {"path": "close.json", "sha256": "c" * 64},
        },
    )
    monkeypatch.setattr(
        module,
        "register_factor_production_observations",
        lambda *args, **kwargs: {
            "command_status": "REGISTERED",
            "observations": [
                {
                    "factor_alias": alias,
                    "observation_path": f"observations/{alias}.json",
                    "observation_sha256": "e" * 64,
                }
                for alias in ("LOW", "W80")
            ],
        },
    )
    order = []
    attempt = loop._attempt

    def selected_attempt(stage, callback):
        if stage == "factor_rollover":
            loop.stages[stage] = {"status": "ROLLOVER_ACTIVATED"}
            return loop.stages[stage]
        return attempt(stage, callback)

    loop._attempt = selected_attempt

    def pool(**kwargs):
        assert kwargs["trade_date"] == "20260904"
        assert kwargs["release_ref"] == release_ref
        order.append("pool")
        return {"status": "SUCCEEDED", "command_status": "PUBLISHED"}

    monkeypatch.setattr(core_pool, "publish_core_pool", pool)
    loop._settle = lambda _: order.append("settlement") or {}
    saved = []
    loop._save_state = lambda value: saved.append(dict(value))
    loop.report = lambda **_: order.append("return_to_auxiliaries") or loop.stages
    result = loop.core_completed({"path": "core.json", "sha256": "d" * 64})
    assert order == ["pool", "settlement", "return_to_auxiliaries"]
    assert result["top100_publication"]["status"] == "SUCCEEDED"
    assert set(saved[0]["core_observation_refs"]) == {"LOW", "W80"}
