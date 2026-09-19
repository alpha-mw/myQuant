"""Both Morning v3 EOD profiles replay real Store closure without mutable heads."""

import pytest
from _native_morning_threshold_fixture import build
from _native_corporate_fixture import put
from scripts import daily_completion_store as replay
from test_morning_threshold_review import inventory


@pytest.mark.parametrize("version", ["v4", "v5"])
def test_native_store_profile_uses_complete_frozen_proof(tmp_path, monkeypatch, version):
    f = build(tmp_path)
    inputs = put(
        tmp_path,
        "fixtures/morning/store-replay-inputs.json",
        {**f["native_inputs"], "schema_version": "cn-daily-native-inputs." + version},
    )
    recorded = {"native_inputs_ref": inputs, "node_terminal_refs": f["terminal_refs"]}
    ref = put(
        tmp_path, "results/operations/daily_production/CN/20260827/completion.v1.json", recorded
    )
    monkeypatch.setattr(
        replay, "inspect_recorded_completion", lambda **kw: {"recorded_completion": recorded}
    )
    before = inventory(tmp_path)
    result = replay.replay_completed_store(
        workspace=str(tmp_path), trade_date="20260827", completion_ref=ref
    )
    assert result["output_refs"] == f["store_outputs"]
    assert inventory(tmp_path) == before
    (f["book"].root / "_record_store/current.v1.json").unlink()
    (f["book"].root / "_event_store/current.v1.json").unlink()
    (tmp_path / "data/parquet/cn/_latest.json").unlink()
    before = inventory(tmp_path)
    assert (
        replay.replay_completed_store(
            workspace=str(tmp_path), trade_date="20260827", completion_ref=ref
        )
        == result
    )
    assert inventory(tmp_path) == before
