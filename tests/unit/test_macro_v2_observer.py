from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from quant_investor.macro.observer import build_macro_observer, persist_macro_observer
from quant_investor.macro.registry import INDUSTRY_COMPONENT_WEIGHTS, NATIONAL_INDICATORS, definition_for
from quant_investor.macro.snapshot import build_macro_snapshot

UTC = ZoneInfo("UTC")


def _observation(
    indicator_id: str,
    *,
    period_end: str,
    available_at: str,
    value: float,
    vintage: str = "initial",
    source: str = "nbs_official",
) -> dict[str, object]:
    available = datetime.fromisoformat(available_at.replace("Z", "+00:00")).astimezone(UTC)
    source_url = (
        "https://www.pbc.gov.cn/fixture"
        if source == "pbc_official"
        else "https://www.stats.gov.cn/fixture"
    )
    return {
        "indicator_id": indicator_id,
        "dimension_type": "industry" if indicator_id.startswith("industry.") else "national",
        "industry_chain": indicator_id.split(".")[1] if indicator_id.startswith("industry.") else "",
        "period_end": period_end,
        "release_at": (available - timedelta(hours=1)).isoformat(),
        "available_at": available.isoformat(),
        "vintage_id": vintage,
        "value": value,
        "unit": (definition_for(indicator_id) or definition_for(indicator_id, "monthly")).unit or "%",
        "frequency": (definition_for(indicator_id) or definition_for(indicator_id, "monthly")).frequency,
        "source_system": source,
        "source_record_id": f"{source}:{indicator_id}:{period_end}:{vintage}",
        "source_url": source_url,
        "fetched_at": (available + timedelta(minutes=5)).isoformat(),
        "quality_status": "pass",
    }


def _complete_fixture() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for definition in NATIONAL_INDICATORS:
        for index, day in enumerate(("2024-03-01", "2024-04-01", "2024-05-09")):
            rows.append(
                _observation(
                    definition.indicator_id,
                    period_end=day,
                    available_at=f"{day}T06:00:00Z",
                    value=10.0 + index,
                    source="pbc_official" if definition.domain == "credit_liquidity" else "nbs_official",
                )
            )
    for component in list(INDUSTRY_COMPONENT_WEIGHTS)[:6]:
        indicator_id = f"industry.semiconductor_electronics.{component}"
        for index, day in enumerate(("2024-03-01", "2024-04-01", "2024-05-09")):
            rows.append(
                _observation(
                    indicator_id,
                    period_end=day,
                    available_at=f"{day}T06:30:00Z",
                    value=5.0 + index,
                )
            )
    return rows


def test_snapshot_is_order_invariant_and_uses_one_hash():
    rows = _complete_fixture()
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)

    first = build_macro_snapshot(rows, as_of="2024-05-10")
    second = build_macro_snapshot(shuffled, as_of="2024-05-10")

    assert first.snapshot_hash == second.snapshot_hash
    assert first.to_dict() == second.to_dict()
    assert first.coverage["industry_chains"]["semiconductor_electronics"] == 0.75
    assert first.shadow_overlays["semiconductor_electronics"]["applied"] is False
    assert abs(first.shadow_overlays["semiconductor_electronics"]["delta_points"]) <= 5.0


def test_future_and_after_close_vintages_are_not_visible():
    rows = _complete_fixture()
    baseline = build_macro_snapshot(rows, as_of="2024-05-10")
    future = _observation(
        "cn.gdp_yoy",
        period_end="2024-05-09",
        available_at="2024-05-10T10:00:00Z",  # 18:00 Asia/Shanghai, after close.
        value=99.0,
        vintage="future_revision",
    )

    with_future = build_macro_snapshot([*rows, future], as_of="2024-05-10")

    assert with_future.snapshot_hash == baseline.snapshot_hash
    assert with_future.source_lineage["cn.gdp_yoy"]["vintage_id"] == "initial"


def test_future_period_is_not_visible_before_a_later_decision_cutoff():
    rows = _complete_fixture()
    baseline = build_macro_snapshot(
        rows,
        as_of="2024-05-10",
        decision_cutoff_at="2024-05-11T07:00:00Z",
    )
    future_period = _observation(
        "cn.gdp_yoy",
        period_end="2024-05-11",
        available_at="2024-05-10T06:30:00Z",
        value=99.0,
        vintage="future_period",
    )

    with_future_period = build_macro_snapshot(
        [*rows, future_period],
        as_of="2024-05-10",
        decision_cutoff_at="2024-05-11T07:00:00Z",
    )

    assert with_future_period.to_dict() == baseline.to_dict()
    assert with_future_period.source_lineage["cn.gdp_yoy"]["vintage_id"] == (
        "initial"
    )


def test_conflicting_vintage_fails_closed():
    rows = _complete_fixture()
    left = _observation(
        "cn.gdp_yoy",
        period_end="2024-05-09",
        available_at="2024-05-09T07:00:00Z",
        value=12.0,
        vintage="conflict",
    )
    right = dict(left, value=13.0, vintage="different_label")

    with pytest.raises(ValueError, match="conflicting_vintage"):
        build_macro_snapshot([*rows, left, right], as_of="2024-05-10")


def test_one_point_history_does_not_count_as_ready_coverage():
    latest_only = [
        row
        for row in _complete_fixture()
        if row["period_end"] == "2024-05-09"
    ]

    snapshot = build_macro_snapshot(latest_only, as_of="2024-05-10")

    assert snapshot.readiness_status != "pass"
    assert snapshot.coverage["national"] == 0.0
    assert any(item.startswith("insufficient_history:") for item in snapshot.blockers)
    assert snapshot.coverage["industry_chains"]["semiconductor_electronics"] == 0.0


def test_recent_revision_of_ancient_period_is_period_stale():
    rows = [row for row in _complete_fixture() if row["indicator_id"] != "cn.gdp_yoy"]
    for index, period in enumerate(("2020-03-31", "2020-06-30", "2020-09-30")):
        rows.append(
            _observation(
                "cn.gdp_yoy",
                period_end=period,
                available_at=f"2024-05-0{7 + index}T06:00:00Z",
                value=4.0 + index,
                vintage=f"revision-{index}",
            )
        )

    snapshot = build_macro_snapshot(rows, as_of="2024-05-10")

    assert "cn.gdp_yoy" in snapshot.freshness["stale_periods"]
    assert "stale_period:cn.gdp_yoy" in snapshot.blockers


@pytest.mark.parametrize(
    ("enabled", "kill_switch", "active"),
    [(False, True, False), (False, False, False), (True, True, False), (True, False, True)],
)
def test_observer_double_gate(enabled, kill_switch, active):
    result = build_macro_observer(
        _complete_fixture(),
        as_of="2024-05-10",
        enabled=enabled,
        kill_switch=kill_switch,
    )
    assert result["active"] is active
    assert result["observer_only"] is True
    assert result["applied"] is False
    assert result["production_eligible"] is False
    assert result["production_enabled"] is False
    assert ("snapshot_hash" in result) is active


def test_observer_reports_are_private_and_readback_safe(tmp_path):
    snapshot = build_macro_snapshot(_complete_fixture(), as_of="2024-05-10")
    artifacts = persist_macro_observer(
        snapshot,
        output_root=tmp_path / "observer",
        production_enabled=True,
        production_kill_switch=False,
    )

    payload = json.loads(open(artifacts["snapshot"], encoding="utf-8").read())
    readiness = json.loads(open(artifacts["readiness"], encoding="utf-8").read())
    assert payload["snapshot_hash"] == snapshot.snapshot_hash
    assert payload["observer_only"] is True
    assert readiness["observer_only"] is True
    assert readiness["applied"] is False
    assert readiness["production_enabled"] is True
    assert readiness["production_kill_switch"] is False
    assert readiness["production_eligible"] is False
    for path in artifacts.values():
        assert int(oct(__import__("os").stat(path).st_mode & 0o777), 8) == 0o600


@pytest.mark.parametrize("generation_id", [".", ".."])
def test_observer_persistence_rejects_dot_generation_ids(
    tmp_path,
    generation_id,
):
    snapshot = build_macro_snapshot(
        _complete_fixture(),
        as_of="2024-05-10",
    )

    with pytest.raises(ValueError, match="generation_id_unsafe"):
        persist_macro_observer(
            snapshot,
            output_root=tmp_path / "observer",
            generation_provenance={"generation_id": generation_id},
        )
