from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


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
