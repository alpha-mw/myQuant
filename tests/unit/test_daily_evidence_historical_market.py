"""Historical native Market capture and replay with in-memory provider data."""

import hashlib
import json
from pathlib import Path

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.market import market_daily_capture as market
from quant_investor.market.historical_session import (
    FILENAME,
    CALENDAR_FILENAME,
    RAW_FILENAME,
    build_historical_session,
)
from test_daily_evidence_requested_session import capture
from test_market_daily_capture import _Provider, _capture_args


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def setup(root):
    value = capture("2026-08-19T20:20:00+08:00")
    calendar = canonical_json_bytes(value.receipt)
    proof = build_historical_session(
        requested_trade_date="20260818",
        previous_trade_date="20260817",
        calendar_bytes=calendar,
        raw=value.raw_response_bytes,
    )
    for name, raw in (
        (CALENDAR_FILENAME, calendar),
        (RAW_FILENAME, value.raw_response_bytes),
        (FILENAME, canonical_json_bytes(proof)),
    ):
        (root / name).write_bytes(raw)
    return root / FILENAME, proof


def load(path):
    return market._load_target_authority(path, sha(path.read_bytes()))


def test_native_historical_capture_and_replay_preserve_proof(tmp_path):
    provider = _Provider()
    args = _capture_args(tmp_path, provider)
    path, proof = setup(tmp_path)
    original = {
        name: (tmp_path / name).read_bytes() for name in (FILENAME, CALENDAR_FILENAME, RAW_FILENAME)
    }
    args.update(
        target_authority_path=path,
        expected_target_authority_sha256=sha(path.read_bytes()),
        target_trade_dates=["20260818"],
        parent_latest_complete_trade_date="20260817",
    )
    result = market.capture_market_daily(**args)
    calls = list(provider.calls)
    manifest_path = Path(result["manifest_path"])
    manifest = json.loads(manifest_path.read_bytes())
    assert manifest["target_trade_date"] == "20260818"
    assert manifest["target_authority"]["captured_bytes_sha256"] == sha(original[FILENAME])
    assert proof["calendar_ref"]["sha256"] == sha(original[CALENDAR_FILENAME])
    assert proof["raw_calendar_ref"]["sha256"] == sha(original[RAW_FILENAME])
    assert proof["observed_at"] == "2026-08-19T12:20:00Z"
    replay = market.replay_market_daily_capture(
        capture_manifest_path=result["manifest_path"],
        expected_capture_manifest_sha256=result["manifest_sha256"],
        **{
            key: args[key]
            for key in (
                "scope_path",
                "expected_scope_sha256",
                "pit_generation_binding",
                "expected_market_pointer_sha256",
            )
        },
    )
    assert replay["target_trade_date"] == "20260818"
    assert provider.calls == calls == ["daily", "daily_basic", "adj_factor"]
    assert original == {name: (tmp_path / name).read_bytes() for name in original}
    # Recomputed manifest SHA cannot turn a missing historical parent into a
    # valid standalone window during readback.
    manifest.pop("parent_latest_complete_trade_date")
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(
        market.MarketDailyCaptureBlocked, match="historical_authority_window_invalid"
    ):
        market.replay_market_daily_capture(
            capture_manifest_path=manifest_path,
            expected_capture_manifest_sha256=sha(manifest_path.read_bytes()),
            **{
                key: args[key]
                for key in (
                    "scope_path",
                    "expected_scope_sha256",
                    "pit_generation_binding",
                    "expected_market_pointer_sha256",
                )
            },
        )


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "calendar_ref",
        "raw_calendar_ref",
        "requested_trade_date",
        "previous_trade_date",
        "authorized_close_trade_date",
        "observed_at",
        "classification",
        "evidence_classification",
        "prospective",
        "execution_authorized",
        "open_trade_dates",
        "extra",
    ],
)
def test_any_proof_field_drift_rejects(tmp_path, field):
    path, proof = setup(tmp_path)
    proof[field] = "changed"
    path.write_bytes(canonical_json_bytes(proof))
    with pytest.raises(market.MarketDailyCaptureBlocked):
        load(path)


@pytest.mark.parametrize("name", [CALENDAR_FILENAME, RAW_FILENAME])
def test_source_drift_rejects(tmp_path, name):
    path, _ = setup(tmp_path)
    source = tmp_path / name
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(market.MarketDailyCaptureBlocked):
        load(path)


@pytest.mark.parametrize(
    "window,parent,rebind",
    [
        (None, "20260817", False),
        (["20260818"], "", False),
        (["20260818"], "20260814", False),
        (["20260817", "20260818"], "20260817", False),
        (["20260818"], "20260817", True),
        (["20260818"], "20260817", 0),
    ],
)
def test_window_drift_blocks_before_provider(tmp_path, window, parent, rebind):
    provider = _Provider()
    args = _capture_args(tmp_path, provider)
    path, _ = setup(tmp_path)
    args.update(
        target_authority_path=path,
        expected_target_authority_sha256=sha(path.read_bytes()),
        target_trade_dates=window,
        parent_latest_complete_trade_date=parent,
        same_target_rebind=rebind,
    )
    with pytest.raises(
        market.MarketDailyCaptureBlocked, match="historical_authority_window_invalid"
    ):
        market.capture_market_daily(**args)
    assert provider.calls == []


def test_three_file_set_is_rechecked(tmp_path, monkeypatch):
    path, _ = setup(tmp_path)
    original = market._stable_regular_bytes
    changed = False

    def read(selected, *, label):
        nonlocal changed
        raw = original(selected, label=label)
        if label == "historical_calendar_raw" and not changed:
            changed = True
            source = tmp_path / CALENDAR_FILENAME
            source.write_bytes(source.read_bytes() + b" ")
        return raw

    monkeypatch.setattr(market, "_stable_regular_bytes", read)
    with pytest.raises(market.MarketDailyCaptureBlocked, match="historical_authority_changed"):
        load(path)


def test_symlink_and_alternate_proof_paths_reject(tmp_path):
    path, _ = setup(tmp_path)
    other = tmp_path / "alternate.json"
    other.write_bytes(path.read_bytes())
    with pytest.raises(market.MarketDailyCaptureBlocked):
        load(other)
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(market.MarketDailyCaptureBlocked):
        load(alias / FILENAME)


@pytest.mark.parametrize(
    "bad_path", ["../close-session-receipt.json", "/close-session-receipt.json"]
)
def test_embedded_source_paths_cannot_redirect_reads(tmp_path, bad_path):
    path, proof = setup(tmp_path)
    proof["calendar_ref"]["path"] = bad_path
    path.write_bytes(canonical_json_bytes(proof))
    with pytest.raises(market.MarketDailyCaptureBlocked):
        load(path)
