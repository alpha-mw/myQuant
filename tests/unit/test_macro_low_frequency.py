"""Freshness uses native vintage selection and registered frequency limits."""

from copy import deepcopy
from decimal import Decimal
import hashlib

import pytest

from quant_investor.macro.contracts import MacroObservation
from quant_investor.intelligence.macro_freshness import macro_freshness, macro_subjects
from test_macro_observation_evidence_store import _row

AS_OF = "2026-08-28T13:30:00Z"


def report(rows, as_of=AS_OF):
    return macro_freshness(
        observations=[MacroObservation.from_mapping(row).to_dict() for row in rows],
        as_of=as_of,
        refs=[],
    )["payload"]


@pytest.mark.parametrize(
    "period,available,state",
    [
        ("2026-08-28", "2026-08-28T01:00:00Z", "FRESH"),
        ("2026-07-31", "2026-08-01T01:00:00Z", "ACCEPTABLE_LAG"),
        ("2026-04-30", "2026-08-01T01:00:00Z", "STALE_WARNING"),
        ("2026-06-30", "2026-07-01T01:00:00Z", "STALE_WARNING"),
        ("2026-08-28", "2026-08-29T01:00:00Z", "MISSING"),
    ],
)
def test_two_ages_and_noncritical_gaps(period, available, state):
    body = report([_row(period=period, available=available)])
    assert body["freshness_state"] == state
    assert len(body["entries"]) == len(macro_subjects())
    assert bool(body["critical_missing_codes"]) == (state == "MISSING")
    row = next(row for row in body["entries"] if row["subject_id"] == "cn.pmi_manufacturing")
    if state != "MISSING":
        assert row["availability_limit_days"] == 50 and row["period_lag_limit_days"] == 75
        assert (
            row["known_at"]
            == MacroObservation.from_mapping(_row(period=period, available=available)).available_at
        )
        assert row["critical_missing_codes"] == []
        assert "MACRO_NATIVE_INSUFFICIENT_HISTORY" in row["warning_codes"]
        assert row["freshness_state"] == state
    assert body["warning_codes"]  # Other individual indicators are not critical gates.


def test_later_vintage_only_wins_when_pit_admissible_and_uses_native_priority():
    first = _row(period="2026-07-31", available="2026-08-01T01:00:00Z")
    revised = _row(
        period="2026-07-31", available="2026-08-20T01:00:00Z", vintage="revision", value=49
    )
    future = _row(period="2026-07-31", available="2026-08-29T01:00:00Z", vintage="future", value=48)
    body = report([first, revised, future])
    assert body == report([revised, first])
    chosen = next(row for row in body["entries"] if row["subject_id"] == "cn.pmi_manufacturing")
    assert chosen["known_at"] == MacroObservation.from_mapping(revised).available_at
    lower = {
        **revised,
        "source_system": "tushare",
        "available_at": "2026-08-21T01:00:00Z",
        "fetched_at": "2026-08-21T01:00:00Z",
        "source_url": "https://tushare.pro/fixture/pmi",
    }
    assert report([first, revised, lower]) == report([first, revised])


def test_native_conflicting_vintage_rejects_and_frequency_has_both_limits():
    row = _row(period="2026-07-31", available="2026-08-01T01:00:00Z")
    with pytest.raises(ValueError, match="conflicting_vintage"):
        report([row, {**row, "value": 10}])
    quarterly = {**row, "indicator_id": "cn.gdp_yoy", "frequency": "quarterly", "unit": "%"}
    body = report([quarterly])
    entry = next(row for row in body["entries"] if row["subject_id"] == "cn.gdp_yoy")
    assert (entry["availability_limit_days"], entry["period_lag_limit_days"]) == (140, 180)


def test_subday_age_is_not_floored_at_warning_boundary():
    row = _row(period="2026-06-30", available="2026-07-09T13:29:59.999999Z")
    body = report([row])
    chosen = next(row for row in body["entries"] if row["subject_id"] == "cn.pmi_manufacturing")
    assert Decimal(chosen["availability_age_days"]) > 50
    assert "MACRO_STALE_AVAILABILITY" in chosen["warning_codes"]


@pytest.mark.parametrize("validated", [False, True])
def test_closure_reader_uses_native_frozen_generation(tmp_path, monkeypatch, validated):
    from quant_investor.intelligence import macro_freshness as module
    from quant_investor.macro import store
    from test_macro_observation_evidence_store import _publish_with_evidence
    from test_daily_evidence_research_sources import put
    from quant_investor.cli.unified import _daily_source_file
    from functools import partial

    root = tmp_path / "data/parquet/cn/macro_observations"
    _publish_with_evidence(
        root, _row(period="2026-07-31", available="2026-08-01T01:00:00Z"), run_id="g1"
    )
    current = root / "_latest.json"
    raw = current.read_bytes()
    frozen = tmp_path / "retained/pointer.json"
    frozen.parent.mkdir()
    frozen.write_bytes(raw)
    frozen.chmod(0o600)
    closure = {
        "target_date": "20260828",
        "available_at": "2026-08-27T00:00:00Z",
        "frozen_pointers": {
            "observations": {
                "current_path": str(current.relative_to(tmp_path)),
                "generation_id": "g1",
                "frozen_ref": {
                    "path": str(frozen.relative_to(tmp_path)),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
            }
        },
    }
    closure_ref = put(tmp_path, "closure.json", closure)
    current.unlink()

    def forbidden(*args, **kwargs):
        pytest.fail("archived reader touched current head")

    monkeypatch.setattr(store, "_optional_pointer_bytes", forbidden)
    monkeypatch.setattr(
        module, "validate_macro_readiness_closure", lambda **kwargs: kwargs["closure"]
    )
    # The closure admission seam is explicit; observation generation/evidence decoding is real.
    body = module.macro_freshness_from_closure(
        workspace=tmp_path,
        as_of=AS_OF,
        closure_ref=closure_ref,
        source_file=partial(_daily_source_file, tmp_path),
        closure=deepcopy(closure) if validated else None,
    )["payload"]
    assert body["freshness_state"] == "ACCEPTABLE_LAG"
    assert closure["frozen_pointers"]["observations"]["frozen_ref"] in body["source_refs"]
    assert not current.exists()
