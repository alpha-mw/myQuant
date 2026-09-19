"""Current v3 consumer/CLI with native risk sources and explicit outer EOD/Calendar seams."""

from contextlib import contextmanager
from copy import deepcopy

import pytest

from test_morning_threshold_review import fixture as risk_fixture, inventory
from _native_corporate_fixture import put
from scripts import daily_morning_consumer as consumer
from quant_investor.operations.daily_contract import EOD_NODE_IDS, ContractError
from quant_investor.operations.daily_journal import FALSE_AUTHORITY, DailyJournal
from quant_investor.operations.morning_report import validate_morning_report
from quant_investor.cli import morning_v2


def setup(root, monkeypatch):
    f, read_sources, projection = risk_fixture(root)
    recorded = read_sources().recorded
    decision = {
        "as_of": "2026-08-27T13:30:00Z",
        "decisions": [{"company_code": "002463.SZ", "state": "WATCHLIST"}],
        "synthetic_decision_admission_seam": True,
    }
    decision_ref = put(root, "fixtures/morning/decision-seam.json", decision)
    decision_terminal = put(
        root,
        "fixtures/morning/decision-terminal.json",
        {
            "state": "SUCCEEDED",
            "finished_at": "2026-08-27T13:30:01Z",
            "output_refs": {"result": decision_ref},
        },
    )
    future = put(
        root, "fixtures/morning/future-seam.json", {"synthetic_calendar_admission_seam": True}
    )
    calendar = put(
        root,
        "fixtures/morning/calendar-terminal.json",
        {"output_refs": {"next_session_calendar_proof": future}},
    )
    recorded["node_terminal_refs"].update(decision=decision_terminal, calendar=calendar)
    completion_ref = put(
        root, "results/operations/daily_production/CN/20260827/completion.v1.json", recorded
    )
    monkeypatch.setattr(
        consumer, "inspect_recorded_completion", lambda **kwargs: {"recorded_completion": recorded}
    )
    monkeypatch.setattr(
        consumer,
        "read_next_session_proof",
        lambda **kwargs: {
            "proof": {
                "next_open_session": "20260828",
                "schema_version": "cn-next-session-calendar-proof.v1",
            },
            "proof_sealed_at": "2026-08-27T13:00:00Z",
            "live_eligible": False,
            "synthetic": True,
            "projection": [{"date": d, "sse_is_open": 1} for d in ("20260827", "20260828")],
        },
    )
    monkeypatch.setattr(
        consumer,
        "replay_native_completion",
        lambda **kwargs: {
            "native_replay_validated": True,
            "validated_nodes": sorted(EOD_NODE_IDS),
            "completion_ref": completion_ref,
            "trade_date": "20260827",
            "decision": decision,
            "synthetic": True,
        },
    )
    policy = put(
        root,
        "fixtures/morning/quote-scope.json",
        {
            "schema_version": "morning-quote-policy.v1",
            "strategy_id": "aggressive_tech_manufacturing",
            "market": "CN",
            "effective_from": "20260828",
            "effective_through": "20260828",
            "revoked_at": None,
            "additional_symbols": [],
            "authority": FALSE_AUTHORITY,
        },
    )
    request = {
        "schema_version": "morning-strategy-request.v3",
        "action": "REPLAY",
        "run_date": "20260828",
        "previous_completion_ref": completion_ref,
        "quote_capture_ref": projection["quote_capture_ref"],
        "quote_raw_ref": projection["quote_raw_ref"],
        "owner_policy_ref": policy,
        "threshold_policy_refs": f["policy_refs"],
        "output_ref": None,
    }
    return request


def test_v3_native_consumer_builds_exact_report_without_writers(tmp_path, monkeypatch):
    request = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(
        DailyJournal, "locked", lambda *a, **kw: pytest.fail("Morning acquired writer lock")
    )
    before = inventory(tmp_path)
    result = consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
    assert result["schema_version"] == "morning-strategy-replay.v3"
    assert result["threshold_review"]["summary_state"] == "COMPLETE_RESEARCH_REVIEW"
    assert "WATCHLIST 1" in result["report_markdown"]
    assert "002463.SZ" in result["report_markdown"]
    assert result["threshold_review"]["evidence_mode"] == "REPLAY_ONLY"
    validate_morning_report(result["report_markdown"].encode(), result)
    assert inventory(tmp_path) == before
    again = consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
    assert again["report_markdown"] == result["report_markdown"]
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("fault", ["table", "declaration", "payload"])
def test_v3_report_requires_native_text_not_just_declarations(tmp_path, monkeypatch, fault):
    request = setup(tmp_path, monkeypatch)
    result = consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
    report = result["report_markdown"]
    if fault == "table":
        report = report.replace("22.00", "999.00", 1)
    elif fault == "declaration":
        report = report.replace("broker=false", "broker=true")
    else:
        report += "forged trailing text\n"
    with pytest.raises(ContractError, match="NATIVE_CONTENT_DIFFERS"):
        validate_morning_report(report.encode(), result)


def test_public_v3_route_validates_current_native_result(tmp_path, monkeypatch):
    request = setup(tmp_path, monkeypatch)
    install = put(
        tmp_path, "fixtures/morning/install-seam.json", {"synthetic_install_admission_seam": True}
    )
    # Native policies may be 0644; preserve their owner's existing safe reader contract.
    (tmp_path / request["threshold_policy_refs"]["trailing"]["path"]).chmod(0o644)

    @contextmanager
    def bridge(**kwargs):
        yield {"morning": consumer.prepare_morning_consumer}

    monkeypatch.setattr(morning_v2, "verified_native_context", bridge)
    before = inventory(tmp_path)
    result = morning_v2.run_morning_v2(
        workspace=str(tmp_path),
        request=request,
        release_repository_root=str(tmp_path),
        release_install_input_path=install["path"],
        expected_release_install_input_sha256=install["sha256"],
    )
    assert result["schema_version"] == "morning-strategy-replay.v3"
    assert inventory(tmp_path) == before
    changed = deepcopy(result)
    changed["threshold_policy_refs"]["trailing"]["sha256"] = "f" * 64
    with pytest.raises(ValueError):
        morning_v2._validate_result(changed, request, str(tmp_path))


def test_missing_eod_is_explicit_upstream_breakpoint_without_resume(tmp_path, monkeypatch):
    request = setup(tmp_path, monkeypatch)

    def missing(**kwargs):
        raise ContractError("EOD_COMPLETION_MISSING")

    monkeypatch.setattr(consumer, "inspect_recorded_completion", missing)
    from quant_investor.operations import daily_status

    monkeypatch.setattr(
        daily_status,
        "read_daily_status",
        lambda *a: {"nodes": {"store": {"state": "BLOCKED"}, "decision": {"state": "SUCCEEDED"}}},
    )
    before = inventory(tmp_path)
    with pytest.raises(ContractError, match="MORNING_UPSTREAM_DAG_INCOMPLETE") as caught:
        consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
    assert caught.value.failed_node_ids == ["store"]
    from quant_investor.cli.output import CommandError

    with pytest.raises(CommandError) as public:
        morning_v2._invoke_morning(
            {"morning": consumer.prepare_morning_consumer}, str(tmp_path), request, None
        )
    assert public.value.to_dict()["failed_node_ids"] == ["store"]
    assert public.value.to_dict()["previous_trade_date"] == "20260827"
    assert inventory(tmp_path) == before


def test_synthetic_v3_evidence_cannot_pass_live_or_seal(tmp_path, monkeypatch):
    request = setup(tmp_path, monkeypatch)
    request["action"] = "PREFLIGHT"
    before = inventory(tmp_path)
    with pytest.raises(ContractError, match="LIVE_PROVENANCE_REQUIRED"):
        consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
    assert inventory(tmp_path) == before


def test_quote_only_symbol_does_not_acquire_holdings_or_thresholds(tmp_path, monkeypatch):
    import json
    import hashlib
    from quant_investor.intelligence.sina_quotes import parse_sina_quote_response

    request = setup(tmp_path, monkeypatch)
    policy = json.loads((tmp_path / request["owner_policy_ref"]["path"]).read_bytes())
    policy["additional_symbols"] = ["000001.SZ"]
    request["owner_policy_ref"] = put(tmp_path, "fixtures/morning/extra-scope.json", policy)
    capture = json.loads((tmp_path / request["quote_capture_ref"]["path"]).read_bytes())
    raw = (tmp_path / request["quote_raw_ref"]["path"]).read_bytes()
    raw = raw.replace(b"sz002463", b"sz000001") + raw
    raw_path = tmp_path / "fixtures/morning/extra-quote-raw.txt"
    raw_path.write_bytes(raw)
    raw_path.chmod(0o600)
    raw_ref = {
        "path": str(raw_path.relative_to(tmp_path)),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    capture["symbol_mapping"].insert(0, {"provider_symbol": "sz000001", "symbol": "000001.SZ"})
    capture["quote_rows"] = parse_sina_quote_response(raw, capture["symbol_mapping"])
    capture["raw_ref"] = {**raw_ref, "size": len(raw)}
    request["quote_raw_ref"] = raw_ref
    request["quote_capture_ref"] = put(tmp_path, "fixtures/morning/extra-capture.json", capture)
    result = consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
    assert result["expected_symbols"] == ["000001.SZ", "002463.SZ"]
    assert result["threshold_review"]["quote_only_symbols"] == ["000001.SZ"]
    assert [r["symbol"] for r in result["threshold_review"]["rows"]] == ["002463.SZ"]


def test_v3_rejects_decision_payload_not_equal_to_bound_result(tmp_path, monkeypatch):
    request = setup(tmp_path, monkeypatch)
    original = consumer.replay_native_completion

    def bad(**kw):
        value = deepcopy(original(**kw))
        value["decision"]["decisions"][0]["state"] = "PAPER_CANDIDATE"
        return value

    monkeypatch.setattr(consumer, "replay_native_completion", bad)
    with pytest.raises(ContractError, match="DECISION_SOURCE_DIFFERS"):
        consumer.prepare_morning_consumer(workspace=str(tmp_path), request=request)
