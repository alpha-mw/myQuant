from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_module():
    spec = importlib.util.spec_from_file_location("log_decision", ROOT / "scripts" / "log_decision.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_append_advisory_event_and_readback(tmp_path):
    mod = _load_module()
    path = tmp_path / "decision_log.jsonl"
    event = mod.append_event(
        path,
        {
            "event_type": "advisory",
            "trade_date": "2026-07-07",
            "channel": "codex",
            "question": "该卖出了吗",
            "answer_summary": "先运行管线状态快照，只翻译系统状态。",
            "answer_source": "codex_thread",
        },
    )

    rows = mod.read_events(path)
    assert rows == [event]
    assert rows[0]["event_id"].startswith("decision-")
    assert json.loads(path.read_text(encoding="utf-8"))["event_type"] == "advisory"


def test_rejects_invalid_advisory_source(tmp_path):
    mod = _load_module()
    try:
        mod.append_event(
            tmp_path / "decision_log.jsonl",
            {
                "event_type": "advisory",
                "question": "buy?",
                "answer_summary": "bad source",
                "answer_source": "broker",
            },
        )
    except ValueError as exc:
        assert "answer_source" in str(exc)
    else:
        raise AssertionError("expected invalid source rejection")


def test_human_action_requires_action(tmp_path):
    mod = _load_module()
    try:
        mod.append_event(tmp_path / "decision_log.jsonl", {"event_type": "human_action"})
    except ValueError as exc:
        assert "requires action" in str(exc)
    else:
        raise AssertionError("expected missing action rejection")


def _envelope(**overrides):
    value = {
        "schema_version": "decision_log.v2",
        "event_type": "advisory_envelope",
        "report_group_id": "myquant-cn:2026-W33",
        "idempotency_key": (
            "myquant-cn:2026-W33:cn-aggressive-tech-manufacturing:" + "a" * 64
        ),
        "report_week": "2026-W33",
        "scheduled_at": "2026-08-16T10:00:00Z",
        "canonical_strategy_id": "cn-aggressive-tech-manufacturing",
        "identity_sha256": "b" * 64,
        "v17_active_run_sha256": "a" * 64,
        "v17_active_pointer_sha256": "c" * 64,
        "store_pointer_sha256": "d" * 64,
        "catalog_sha256": "e" * 64,
        "performance_manifest_sha256": "f" * 64,
        "financial_state_sha256": "1" * 64,
        "executable": False,
        "formal_outcome": "ADVISORY",
        "actions": [
            {
                "symbol": "000001.SZ",
                "company_name": "平安银行",
                "action": "BUY",
                "shares_delta": 100,
                "validity": "2026-08-17 only",
                "invalidation": "risk veto changes",
                "evidence_refs": [
                    {"path": "results/v17/run.json", "sha256": "a" * 64}
                ],
            }
        ],
        "supersedes_event_id": None,
    }
    value.update(overrides)
    return value


def test_v2_envelope_is_idempotent_and_conflicts_fail_closed(tmp_path):
    mod = _load_module()
    path = tmp_path / "decision_log.jsonl"
    first = mod.append_event(path, _envelope())
    replay = mod.append_event(path, _envelope())

    assert first["already_recorded"] is False
    assert replay["already_recorded"] is True
    assert first["event_id"] == replay["event_id"]
    assert len(mod.read_events(path)) == 1
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert oct(path.with_name(path.name + ".lock").stat().st_mode & 0o777) == "0o600"

    changed = _envelope()
    changed["actions"][0]["invalidation"] = "different body"
    with pytest.raises(mod.DecisionLogError, match="IDEMPOTENCY_CONFLICT"):
        mod.append_event(path, changed)


def test_v2_rejects_malformed_tail_and_unsafe_mode(tmp_path):
    mod = _load_module()
    path = tmp_path / "decision_log.jsonl"
    path.write_bytes(b'{"schema_version":"decision_log.v1"}')
    path.chmod(0o600)
    with pytest.raises(mod.DecisionLogError, match="CORRUPT_TAIL"):
        mod.read_events(path)

    path.write_bytes(b"")
    path.chmod(0o644)
    with pytest.raises(mod.DecisionLogError, match="0600"):
        mod.append_event(path, _envelope())


def test_permission_hardening_preserves_exact_content(tmp_path):
    mod = _load_module()
    path = tmp_path / "decision_log.jsonl"
    legacy = {
        "event_id": "decision-legacy",
        "event_type": "human_action",
        "schema_version": "decision_log.v1",
    }
    original = (json.dumps(legacy, ensure_ascii=False, sort_keys=True) + "\n").encode()
    path.write_bytes(original)
    path.chmod(0o644)

    receipt = mod.harden_log_permissions(path)

    assert receipt["content_unchanged"] is True
    assert receipt["content_sha256_before"] == receipt["content_sha256_after"]
    assert path.read_bytes() == original
    assert oct(path.stat().st_mode & 0o777) == "0o600"


def test_v2_rejects_symlink_and_hardlink_storage(tmp_path):
    mod = _load_module()
    source = tmp_path / "source.jsonl"
    source.write_bytes(b"")
    source.chmod(0o600)
    symlink = tmp_path / "symlink.jsonl"
    symlink.symlink_to(source)
    with pytest.raises((mod.DecisionLogError, OSError)):
        mod.append_event(symlink, _envelope())

    hardlink = tmp_path / "hardlink.jsonl"
    os.link(source, hardlink)
    with pytest.raises(mod.DecisionLogError, match="single-link"):
        mod.append_event(source, _envelope())


def test_v2_concurrent_same_envelope_appends_once(tmp_path):
    mod = _load_module()
    path = tmp_path / "decision_log.jsonl"
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(lambda _index: mod.append_event(path, _envelope()), range(2))
        )

    assert sorted(result["already_recorded"] for result in results) == [False, True]
    assert len(mod.read_events(path)) == 1


def test_v2_same_week_new_active_run_requires_explicit_supersession(tmp_path):
    mod = _load_module()
    path = tmp_path / "decision_log.jsonl"
    first = mod.append_event(path, _envelope())
    replacement = _envelope(
        idempotency_key=(
            "myquant-cn:2026-W33:cn-aggressive-tech-manufacturing:" + "2" * 64
        ),
        v17_active_run_sha256="2" * 64,
        v17_active_pointer_sha256="3" * 64,
    )
    with pytest.raises(mod.DecisionLogError, match="explicit supersession"):
        mod.append_event(path, replacement)

    replacement["supersedes_event_id"] = first["event_id"]
    second = mod.append_event(path, replacement)
    assert second["supersedes_event_id"] == first["event_id"]
    assert len(mod.read_events(path)) == 2
