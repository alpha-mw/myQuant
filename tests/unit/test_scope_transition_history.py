from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from quant_investor.market import daily_components, download, market_data_store
from quant_investor.market.scope_transition import encoded, reference
from quant_investor.market.scope_transition_history import repair_and_verify_added_history


@pytest.fixture
def history_env(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    pointer = root / "data/parquet/cn/_latest.json"
    table = root / "old-table"
    table.mkdir()
    original = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 2,
            "trade_date": ["20260817", "20260819"],
            "close": [10.0, 12.0],
        }
    )
    from quant_investor.market.scope_transition_history import REQUIRED_FIELDS, NULLABLE_FIELDS

    for col in (*REQUIRED_FIELDS, *NULLABLE_FIELDS):
        if col not in original:
            original[col] = 1.0
    original.to_parquet(table / "part.parquet", index=False)
    pointer.parent.mkdir(parents=True)
    manifest = pointer.parent / "manifest.json"
    manifest.write_text("{}")
    pointer.write_bytes(
        encoded(
            {
                "manifest_path": str(manifest),
                "table_root": str(table),
                "latest_complete_trade_date": "20260819",
            }
        )
    )
    listed = root / "listed.json"
    listed.write_bytes(encoded({"items": [{"ts_code": "000001.SZ", "list_date": "20260817"}]}))
    capture = root / "capture.json"
    capture.write_bytes(encoded({"partitions": [{"path": str(listed)}]}))
    q = {
        "workspace_root": str(root),
        "operation_id": "unit",
        "added": ["000001.SZ"],
        "effective_date": "20260819",
        "pit_capture_ref": reference(capture),
    }
    provider = SimpleNamespace(
        trade_cal=lambda **kw: pd.DataFrame(
            [
                {"exchange": "SSE", "cal_date": d, "is_open": 1}
                for d in ["20260817", "20260818", "20260819"]
            ]
        )
    )
    calls = []
    mutate = {"missing_basic": False}

    class Maintainer:
        def __init__(self, **kw):
            self.downloader = SimpleNamespace(pro=None)
            self.store = self

        def _fetch_endpoint(self, endpoint, date, fields):
            calls.append((endpoint, date))
            symbol = (
                "000002.SZ"
                if endpoint == "daily_basic" and mutate["missing_basic"]
                else "000001.SZ"
            )
            return pd.DataFrame({"ts_code": [symbol], "trade_date": [date], "close": [11.0]}), ""

        def _build_bars_frame(self, daily, adj, basic):
            for col in (*REQUIRED_FIELDS, *NULLABLE_FIELDS):
                if col not in daily:
                    daily[col] = 1.0
            return daily

        def upsert_bars(self, incoming, **kw):
            assert kw["expected_latest_pointer_sha256"] == reference(pointer)["sha256"]
            assert list(zip(incoming.ts_code, incoming.trade_date)) == [("000001.SZ", "20260818")]
            new = root / "new-table"
            new.mkdir()
            pd.concat([original, incoming]).to_parquet(new / "part.parquet", index=False)
            pointer.write_bytes(
                encoded(
                    {
                        "manifest_path": str(manifest),
                        "table_root": str(new),
                        "latest_complete_trade_date": "20260819",
                    }
                )
            )

    monkeypatch.setattr(daily_components, "_default_provider_factory", lambda: provider)
    monkeypatch.setattr(download, "CNParquetBatchMaintainer", Maintainer)
    monkeypatch.setattr(
        market_data_store,
        "MarketDataStore",
        lambda **kw: SimpleNamespace(validate_latest=lambda: {"status": "passed"}),
    )
    operation = root / "operation"
    operation.mkdir()
    return q, operation, pointer, calls, mutate


def test_history_fills_only_proven_missing_keys_and_replays_without_calls(history_env):
    q, operation, pointer, calls, mutate = history_env
    result = repair_and_verify_added_history(q, operation_root=operation)
    assert result["repaired_row_count"] == 1
    assert result["unresolved_gap_count"] == 0
    assert len(calls) == 3
    assert result["symbol_coverage"]["000001.SZ"]["count"] == 3
    assert repair_and_verify_added_history(q, operation_root=operation) == result
    assert len(calls) == 3


def test_history_auxiliary_gap_blocks_before_pointer_write(history_env):
    q, operation, pointer, calls, mutate = history_env
    before = reference(pointer)
    mutate["missing_basic"] = True
    with pytest.raises(RuntimeError, match="AUXILIARY_COVERAGE_MISSING"):
        repair_and_verify_added_history(q, operation_root=operation)
    assert reference(pointer) == before


def test_history_replay_refuses_pointer_drift(history_env):
    q, operation, pointer, calls, mutate = history_env
    repair_and_verify_added_history(q, operation_root=operation)
    pointer.write_text(json.dumps({"unrelated": True}))
    with pytest.raises(RuntimeError, match="SHA_MISMATCH"):
        repair_and_verify_added_history(q, operation_root=operation)


def test_existing_row_with_invalid_adjustment_is_not_counted_complete(history_env):
    q, operation, pointer, calls, mutate = history_env
    table = Path(json.loads(pointer.read_bytes())["table_root"]) / "part.parquet"
    frame = pd.read_parquet(table)
    frame.loc[0, "adj_factor"] = float("nan")
    frame.to_parquet(table, index=False)
    before = reference(pointer)
    with pytest.raises(RuntimeError, match="REQUIRED_FIELD_INVALID:adj_factor"):
        repair_and_verify_added_history(q, operation_root=operation)
    assert reference(pointer) == before


@pytest.mark.parametrize(
    "tamper", ["calendar", "incoming_path", "explanation", "request", "completion"]
)
def test_history_evidence_tamper_blocks_replay(history_env, tamper):
    q, operation, pointer, calls, mutate = history_env
    repair_and_verify_added_history(q, operation_root=operation)
    acquisition = operation / "history-acquisition.json"
    value = json.loads(acquisition.read_bytes())
    if tamper == "calendar":
        (operation / "history-calendar.json").write_bytes(encoded([]))
    elif tamper == "completion":
        p = operation / "history-repair.json"
        done = json.loads(p.read_bytes())
        done["unresolved_gap_count"] = 9
        p.write_bytes(encoded(done))
    else:
        if tamper == "incoming_path":
            value["incoming_ref"]["path"] = str(operation / "other.parquet")
        elif tamper == "explanation":
            value["explained_missing"] = [["000001.SZ", "20260818"]]
        else:
            value["request_sha256"] = "other-request"
        acquisition.write_bytes(encoded(value))
    with pytest.raises(RuntimeError):
        repair_and_verify_added_history(q, operation_root=operation)


def test_duplicate_existing_history_keys_are_rejected(history_env):
    q, operation, pointer, calls, mutate = history_env
    table = Path(json.loads(pointer.read_bytes())["table_root"]) / "part.parquet"
    frame = pd.read_parquet(table)
    pd.concat([frame, frame.iloc[[0]]]).to_parquet(table, index=False)
    with pytest.raises(RuntimeError, match="CANONICAL_DUPLICATE_KEYS"):
        repair_and_verify_added_history(q, operation_root=operation)
