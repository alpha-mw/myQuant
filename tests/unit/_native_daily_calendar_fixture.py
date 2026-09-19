"""Installed native Calendar capture with only external transport simulated.

Run using the isolated release Python. Provider/documentation bytes are explicitly
synthetic fixtures, never real exchange or contemporaneous provider evidence.
"""

import ast
from contextlib import contextmanager
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch
import sys

from quant_investor.market import tushare_calendar_authority as native
from quant_investor.market.tushare_transport import replay_tushare_response_bytes


@contextmanager
def synthetic_calendar_transport(*, fixture_source: Path, cutoff: str):
    """Only replace external provider/documentation transport for native callbacks."""
    # Reuse precisely the existing synthetic wire-document examples, without
    # importing their mocked release, clock, reader or pytest fixture machinery.
    text = fixture_source.read_text()
    tree = ast.parse(text)
    functions = "\n\n".join(
        ast.get_source_segment(text, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_docs", "_provider_raw"}
    )
    scope = {
        "date": date,
        "timedelta": timedelta,
        "Any": Any,
        "json": json,
        "CAPTURE_START": date(2023, 12, 1),
        "CUTOFF": date.fromisoformat(cutoff),
        "EXPECTED_FIELDS": native.EXPECTED_FIELDS,
    }
    exec(compile(functions, str(fixture_source), "exec"), scope)
    calls = []

    class SyntheticTransport:
        def __init__(self, **kwargs):
            pass

        def request(self, *, api_name, params, **kwargs):
            calls.append({"api_name": api_name, "params": params})
            return replay_tushare_response_bytes(
                scope["_provider_raw"](params["exchange"]),
                api_name=api_name,
                expected_fields=native.EXPECTED_FIELDS,
                strict_decimal_decode=True,
            )

    with (
        patch.object(native, "OfficialTushareHttpsClient", SyntheticTransport),
        patch.object(
            native,
            "_official_documentation_fetch",
            lambda: (scope["_docs"](), 200, {"content-type": "text/html; charset=utf-8"}, True, []),
        ),
    ):
        yield calls


def capture_synthetic_calendar(
    *, release_root: Path, fixture_source: Path, cutoff: str, capture_parent: Path | None = None
) -> dict:
    parent = capture_parent if capture_parent is not None else release_root / ("calendar-" + cutoff)
    parent.mkdir(mode=0o700)
    raw = (release_root / "release-input.json").read_bytes()
    with synthetic_calendar_transport(fixture_source=fixture_source, cutoff=cutoff) as calls:
        captured = native.capture_trusted_provider_calendar_evidence(
            capture_parent=parent,
            capture_root_name="synthetic-provider-capture",
            cutoff_date=cutoff,
            release_install_input_raw=raw,
            expected_release_install_input_sha256=hashlib.sha256(raw).hexdigest(),
            release_repository_root=release_root / "repository",
        )
    result = {
        "synthetic": True,
        "external_transport_simulated": True,
        "live_provider_calls": 0,
        "release_verifier_mocked": False,
        "clock_mocked": False,
        "capture": captured,
        "requests": calls,
        "full_dag_proof": False,
    }
    (parent / "fixture-result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(
        json.dumps(
            capture_synthetic_calendar(
                release_root=Path(sys.argv[1]), fixture_source=Path(sys.argv[2]), cutoff=sys.argv[3]
            ),
            indent=2,
        )
    )


def publish_daily_future_proof(*, release_root: Path, workspace: Path, trade_date: str) -> dict:
    """Synthetic native future proof published before the day's Calendar terminal."""
    from datetime import datetime, timedelta
    import os
    from quant_investor.operations.daily_journal import DailyJournal
    from quant_investor.market.next_session_proof import publish_synthetic_next_session_proof

    journal = DailyJournal(str(workspace), trade_date)
    fd, _, _ = journal.storage._parent(
        str(journal.root / "calendar-future/placeholder"), create=True
    )
    os.close(fd)
    cutoff = (datetime.strptime(trade_date, "%Y%m%d") + timedelta(days=21)).date().isoformat()
    capture = capture_synthetic_calendar(
        release_root=release_root,
        fixture_source=release_root / "repository/tests/unit/test_tushare_calendar_authority.py",
        cutoff=cutoff,
        capture_parent=workspace / journal.root / "calendar-future/captures",
    )["capture"]
    return publish_synthetic_next_session_proof(
        workspace=str(workspace),
        eod_trade_date=trade_date,
        execution=capture["capture_execution"],
        execution_ref=capture["capture_execution_file_ref"],
        success=capture["capture_success"],
        success_ref=capture["capture_success_file_ref"],
    )


@contextmanager
def configured_future_calendar_scope(*, root: Path, trade_date: str):
    """Pinned offline bytes for the configured Factor and future Calendar lanes."""
    from datetime import datetime
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.market._calendar_fixture_capability import _offline_calendar_fixture

    source = root / "repository/tests/unit/test_tushare_calendar_authority.py"
    source_raw = source.read_bytes()
    tree = ast.parse(source_raw.decode())
    functions = "\n\n".join(
        ast.get_source_segment(source_raw.decode(), node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_docs", "_provider_raw"}
    )

    def wires(cutoff):
        scope = {
            "date": date,
            "timedelta": timedelta,
            "Any": Any,
            "json": json,
            "CAPTURE_START": date(2023, 12, 1),
            "CUTOFF": cutoff,
            "EXPECTED_FIELDS": native.EXPECTED_FIELDS,
        }
        exec(compile(functions, str(source), "exec"), scope)
        return scope["_docs"](), {
            exchange: scope["_provider_raw"](exchange) for exchange in ("SSE", "SZSE", "BSE")
        }

    day = datetime.strptime(trade_date, "%Y%m%d").date()
    docs, current = wires(day)
    future_docs, future = wires(day + timedelta(days=21))
    if docs != future_docs:
        raise ValueError("fixture documentation differs by horizon")
    raw = canonical_json_bytes(json.loads((root / "release-input.json").read_bytes()))
    # Baseline Factor custody uses the native real clock. Only the future lane
    # shares the scenario business clock; do not backdate release/bootstrap closure.
    from quant_investor.market import future_calendar_producer
    from quant_investor.market import _calendar_fixture_capability as fixture_clock

    acquire = future_calendar_producer.capture_next_session_calendar

    class FutureCaptureClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixture_clock.datetime.now(tz)

    def future_capture(**kwargs):
        with patch.object(native, "datetime", FutureCaptureClock):
            return acquire(**kwargs)

    with (
        _offline_calendar_fixture(
            workspace=root / "factor-workspace",
            trade_date=trade_date,
            install_sha=hashlib.sha256(raw).hexdigest(),
            fixture_source_sha=hashlib.sha256(source_raw).hexdigest(),
            documentation_raw=docs,
            provider_raw=future,
            current_provider_raw=current,
        ),
        patch.object(future_calendar_producer, "capture_next_session_calendar", future_capture),
    ):
        yield
