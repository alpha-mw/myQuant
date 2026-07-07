from __future__ import annotations

import json
from pathlib import Path

from quant_investor.themes.membership import (
    ThemeMembership,
    ThemeMembershipStore,
    active_memberships_by_symbol,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_theme_membership_store_loads_pit_jsonl(tmp_path: Path):
    path = tmp_path / "theme_membership.jsonl"
    _write_jsonl(
        path,
        [
            {
                "schema_version": "theme_membership.v1",
                "membership_id": "low-altitude-000001-20260101",
                "theme_id": "concept::low-altitude-economy",
                "theme_name": "Low-altitude Economy",
                "theme_type": "concept",
                "symbol": "000001.SZ",
                "symbol_name": "Example Co",
                "effective_from": "2026-01-01",
                "effective_to": "",
                "confidence": 0.8,
                "source_type": "manual_review",
                "source_ref": "policy_event:low-altitude-plan",
                "evidence_text": "Reviewed PIT source.",
                "updated_at": "2026-01-02T00:00:00Z",
            }
        ],
    )

    result = ThemeMembershipStore(path).load()

    assert result.status == "success"
    assert result.diagnostic_notes == ["theme_membership_count=1"]
    assert result.memberships == [
        ThemeMembership(
            membership_id="low-altitude-000001-20260101",
            theme_id="concept::low-altitude-economy",
            theme_name="Low-altitude Economy",
            theme_type="concept",
            symbol="000001.SZ",
            symbol_name="Example Co",
            effective_from="2026-01-01",
            effective_to="",
            membership_status="active",
            confidence=0.8,
            source_type="manual_review",
            source_ref="policy_event:low-altitude-plan",
            evidence_text="Reviewed PIT source.",
            updated_at="2026-01-02T00:00:00Z",
        )
    ]


def test_active_memberships_respect_effective_dates_and_dedupe():
    memberships = [
        ThemeMembership(
            membership_id="old",
            theme_id="concept::robotics",
            theme_name="Robotics",
            symbol="000001.SZ",
            effective_from="2026-01-01",
            effective_to="2026-01-10",
            updated_at="2026-01-01T00:00:00Z",
        ),
        ThemeMembership(
            membership_id="newer",
            theme_id="concept::robotics",
            theme_name="Robotics",
            symbol="000001.SZ",
            effective_from="2026-01-01",
            effective_to="",
            updated_at="2026-01-03T00:00:00Z",
        ),
        ThemeMembership(
            membership_id="future",
            theme_id="concept::low-altitude-economy",
            theme_name="Low-altitude Economy",
            symbol="000002.SZ",
            effective_from="2026-02-01",
        ),
    ]

    active = active_memberships_by_symbol(memberships, as_of="20260115")

    assert list(active) == ["000001.SZ"]
    assert [item.membership_id for item in active["000001.SZ"]] == ["newer"]


def test_theme_membership_store_missing_file_is_fail_open(tmp_path: Path):
    result = ThemeMembershipStore(tmp_path / "missing.jsonl").load()

    assert result.status == "missing"
    assert result.memberships == []
    assert "theme_membership_file_missing" in result.diagnostic_notes


def test_theme_membership_rejects_non_concept_records(tmp_path: Path):
    path = tmp_path / "theme_membership.jsonl"
    _write_jsonl(
        path,
        [
            {
                "schema_version": "theme_membership.v1",
                "membership_id": "bad-industry-record",
                "theme_id": "industry::machinery",
                "theme_name": "Machinery",
                "theme_type": "industry",
                "symbol": "000001.SZ",
                "effective_from": "2026-01-01",
            }
        ],
    )

    result = ThemeMembershipStore(path).load()

    assert result.status == "error"
    assert result.memberships == []
    assert any("theme_id must start with concept::" in note for note in result.diagnostic_notes)


def test_theme_membership_rejects_non_concept_theme_type(tmp_path: Path):
    path = tmp_path / "theme_membership.jsonl"
    _write_jsonl(
        path,
        [
            {
                "schema_version": "theme_membership.v1",
                "membership_id": "bad-type-record",
                "theme_id": "concept::machinery",
                "theme_name": "Machinery",
                "theme_type": "industry",
                "symbol": "000001.SZ",
                "effective_from": "2026-01-01",
            }
        ],
    )

    result = ThemeMembershipStore(path).load()

    assert result.status == "error"
    assert result.memberships == []
    assert any("theme_type must be concept" in note for note in result.diagnostic_notes)
