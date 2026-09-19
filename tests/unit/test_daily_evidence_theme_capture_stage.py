"""Native capture/projection with only external transport and core binding controlled."""

from datetime import datetime, timezone
import hashlib
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations import theme_capture_stage as stage, theme_acquisition as claims
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.daily_contract import ContractError
from quant_investor.intelligence.theme_governance import TECHNOLOGY_THEME_IDS
from test_tushare_theme_capture_stable import FakeClient
from test_daily_evidence_theme_claim import identity


@pytest.mark.parametrize(
    "fallback,expired,read_fault,interruption",
    [
        (False, False, None, None),
        (True, False, None, None),
        (False, True, None, None),
        (False, False, "partition", None),
        (True, False, "tdx_plan", None),
        (False, False, "descriptor", None),
        (False, False, "claim", None),
        (False, False, None, ("dc", 0)),
        (False, False, None, ("dc", 1)),
        (True, False, None, ("tdx", 0)),
        (True, False, None, ("tdx", 1)),
        (True, False, "terminal_dc", None),
        (True, False, "terminal_both", None),
    ],
)
def test_native_capture_reuses_complete_partitions_after_cutoff(
    tmp_path, monkeypatch, fallback, expired, read_fault, interruption
):
    companies = ["000001.SZ"]
    day = "20260828"
    bound = identity()
    bound["company_set_sha256"] = hashlib.sha256(canonical_json_bytes(companies)).hexdigest()
    binding = {
        "trade_date": day,
        "company_keyset": companies,
        "research_cutoff": "2026-08-28T13:30:00Z",
        "core_completed_at": "2026-08-28T12:00:00Z",
        "identity": bound,
    }
    monkeypatch.setattr(stage, "bind_theme_acquisition", lambda **kwargs: binding)

    class Clock(datetime):
        hour = 13

        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 8, 28, cls.hour, tzinfo=timezone.utc)

    monkeypatch.setattr(stage, "datetime", Clock)
    monkeypatch.setattr(claims, "datetime", Clock)
    if expired:
        Clock.hour = 14
    theme = TECHNOLOGY_THEME_IDS[0].split(":", 1)[1]
    tdx = next(
        (name.split(":", 1)[1] for name in TECHNOLOGY_THEME_IDS if name.startswith("TUSHARE_TDX:")),
        "880001.TDX",
    )
    client = FakeClient(
        {
            ("dc_index", "ALL"): [(theme, day, "fixture theme", "概念板块", "1")],
            ("dc_member", companies[0]): [
                (day, "BK9999.DC" if fallback else theme, companies[0], "fixture company")
            ],
            ("tdx_index", "ALL"): [(tdx, day, "fixture TDX theme", "概念板块", 20)],
            ("tdx_member", companies[0]): [(tdx, day, companies[0], "fixture company")],
        }
    )
    terminal_failure = read_fault in {"terminal_dc", "terminal_both"}
    both_failed = read_fault == "terminal_both"
    if terminal_failure:
        original_request = client.request

        def fail_member(*, api_name, params, expected_fields):
            if api_name == "dc_member" or (both_failed and api_name == "tdx_member"):
                client.calls.append((api_name, dict(params)))
                raise RuntimeError("synthetic transport failure")
            return original_request(
                api_name=api_name, params=params, expected_fields=expected_fields
            )

        monkeypatch.setattr(client, "request", fail_member)
        read_fault = None
    monkeypatch.setattr(stage.native, "OfficialTushareHttpsClient", lambda **kwargs: client)
    journal = DailyJournal(str(tmp_path), day)
    if expired:
        with journal.locked():
            with pytest.raises(ContractError, match="CUTOFF_EXPIRED"):
                stage.capture_bound_theme(
                    journal=journal,
                    request_ref=bound["request_ref"],
                    core_handoff_ref=bound["core_handoff_ref"],
                )
        assert client.calls == []
        assert journal.storage.read(str(journal.root / "theme-acquisition.v1.json")) is None
        return
    if both_failed:
        for attempt in range(2):
            with journal.locked(), pytest.raises(ContractError, match="NATIVE_SOURCE_INCOMPLETE"):
                stage.capture_bound_theme(
                    journal=journal,
                    request_ref=bound["request_ref"],
                    core_handoff_ref=bound["core_handoff_ref"],
                )
            inventory = {
                str(p): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in tmp_path.rglob("*")
                if p.is_file()
            }
            assert len(client.calls) == 4
            if attempt == 0:
                retained_failed = inventory
            else:
                assert inventory == retained_failed
        return
    retained = {}
    if interruption is not None:
        original_write = stage.native.write_exact

        class CaptureInterrupted(BaseException):
            pass

        def interrupted_write(path, value):
            original_write(path, value)
            if (
                path.parent.name == "partitions"
                and (path.parent.parent.name, int(path.stem)) == interruption
            ):
                retained[str(path)] = (path.read_bytes(), path.stat().st_mtime_ns)
                raise CaptureInterrupted()

        monkeypatch.setattr(stage.native, "write_exact", interrupted_write)
        with journal.locked(), pytest.raises(CaptureInterrupted):
            stage.capture_bound_theme(
                journal=journal,
                request_ref=bound["request_ref"],
                core_handoff_ref=bound["core_handoff_ref"],
            )
        assert retained
        claim_path = tmp_path / journal.root / "theme-acquisition.v1.json"
        retained[str(claim_path)] = (claim_path.read_bytes(), claim_path.stat().st_mtime_ns)
        monkeypatch.setattr(stage.native, "write_exact", original_write)
    with journal.locked():
        result = stage.capture_bound_theme(
            journal=journal,
            request_ref=bound["request_ref"],
            core_handoff_ref=bound["core_handoff_ref"],
        )
    for path, original in retained.items():
        from pathlib import Path

        assert (Path(path).read_bytes(), Path(path).stat().st_mtime_ns) == original
    expected_calls = 4 if fallback else 2
    assert len(client.calls) == expected_calls
    assert result["fallback_company_keyset"] == (companies if fallback else [])
    assert result["projection"]["payload"]["blocker_codes"] == []
    if terminal_failure:
        import json

        partition = json.loads(
            (tmp_path / result["descriptor"]["dc_partitions"][1]["path"]).read_bytes()
        )
        assert partition["status"] == "INCOMPLETE"
        assert partition["blocker_codes"] == ["TRANSPORT_ERROR"]
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    Clock.hour = 14
    with journal.locked():
        replay = stage.capture_bound_theme(
            journal=journal,
            request_ref=bound["request_ref"],
            core_handoff_ref=bound["core_handoff_ref"],
        )
    assert replay == result and len(client.calls) == expected_calls
    assert {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    } == before

    from quant_investor.operations import theme_handoff_readback as reader
    from quant_investor.operations.theme_handoff_publish import publish_theme_handoff

    monkeypatch.setattr(reader, "bind_theme_acquisition", lambda **kwargs: binding)
    monkeypatch.setattr(
        "quant_investor.operations.theme_handoff_publish.bind_theme_acquisition",
        lambda **kwargs: binding,
    )
    with journal.locked():
        handoff_ref = publish_theme_handoff(
            journal=journal,
            request_ref=bound["request_ref"],
            core_handoff_ref=bound["core_handoff_ref"],
        )
    raw_handoff = (tmp_path / handoff_ref["path"]).read_bytes()
    with journal.locked():
        assert (
            publish_theme_handoff(
                journal=journal,
                request_ref=bound["request_ref"],
                core_handoff_ref=bound["core_handoff_ref"],
            )
            == handoff_ref
        )
    assert (tmp_path / handoff_ref["path"]).read_bytes() == raw_handoff

    def forbidden(*args, **kwargs):
        pytest.fail("Theme readback attempted provider/write/lock")

    monkeypatch.setattr(stage.native, "capture_theme_plan", forbidden)
    monkeypatch.setattr(journal.storage, "write", forbidden)
    monkeypatch.setattr(journal, "locked", forbidden)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    arguments = dict(
        journal=journal,
        request_ref=bound["request_ref"],
        core_handoff_ref=bound["core_handoff_ref"],
        handoff_ref=handoff_ref,
    )
    if read_fault:
        import json

        targets = {
            "partition": result["descriptor"]["dc_partitions"][1],
            "tdx_plan": result["descriptor"]["tdx_plan"],
            "claim": result["claim_ref"],
            "descriptor": json.loads(raw_handoff)["source_descriptor_ref"],
        }
        target = tmp_path / targets[read_fault]["path"]
        target.write_bytes(target.read_bytes() + b" ")
        with pytest.raises(ContractError, match="THEME_HANDOFF_SOURCE_SHA_MISMATCH"):
            reader.read_theme_handoff(**arguments)
        return
    verified = reader.read_theme_handoff(**arguments)
    assert verified["descriptor"] == result["descriptor"]
    assert {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    } == before
