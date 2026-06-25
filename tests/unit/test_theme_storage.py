from __future__ import annotations

import copy
import json

from quant_investor.themes.storage import ThemeSnapshotStore


def test_theme_snapshot_store_save_and_load_latest(tmp_path):
    store = ThemeSnapshotStore(tmp_path)
    theme_rotation = {
        "status": "success",
        "metadata": {"run_id": "scan-001"},
        "theme_scores": {"industry::ai": {"score": 78.5}},
    }

    path = store.save(
        theme_rotation,
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        run_id="run-a",
    )

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["snapshot_schema_version"] == "theme_snapshot.v1"
    assert payload["theme_rotation"]["status"] == "success"

    latest = store.load_latest(market="CN", universe_key="full_a")
    assert latest is not None
    assert latest["snapshot_schema_version"] == "theme_snapshot.v1"
    assert latest["theme_rotation"]["status"] == "success"


def test_theme_snapshot_store_sanitizes_path(tmp_path):
    store = ThemeSnapshotStore(tmp_path)

    path = store.save(
        {"status": "success"},
        market="CN/unsafe market",
        universe_key="full/a weird key",
        as_of="2026/06/18",
        run_id="run id:1?",
    )

    assert path.exists()
    relative_parts = path.relative_to(tmp_path).parts
    for part in relative_parts:
        assert "/" not in part
        assert " " not in part
        assert ":" not in part
        assert "?" not in part


def test_theme_snapshot_store_list_snapshots_sorted(tmp_path):
    store = ThemeSnapshotStore(tmp_path)

    first = store.save(
        {"status": "success"},
        market="CN",
        universe_key="full_a",
        as_of="20260617",
        run_id="b",
    )
    second = store.save(
        {"status": "success"},
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        run_id="a",
    )

    assert store.list_snapshots(market="CN", universe_key="full_a") == sorted(
        [first, second]
    )


def test_theme_snapshot_store_load_latest_empty_returns_none(tmp_path):
    store = ThemeSnapshotStore(tmp_path)

    assert store.load_latest(market="CN", universe_key="full_a") is None


def test_theme_snapshot_store_does_not_mutate_input(tmp_path):
    store = ThemeSnapshotStore(tmp_path)
    theme_rotation = {
        "status": "success",
        "metadata": {"run_id": "scan-001"},
        "top_themes": [{"theme_id": "industry::ai", "score": 78.5}],
    }
    original = copy.deepcopy(theme_rotation)

    store.save(
        theme_rotation,
        market="CN",
        universe_key="full_a",
        as_of="20260618",
    )

    assert theme_rotation == original


def test_theme_snapshot_store_load_latest_skips_single_symbol_full_a(tmp_path):
    store = ThemeSnapshotStore(tmp_path)
    full_path = store.save(
        {
            "status": "success",
            "metadata": {
                "scanned_symbol_count": 78,
                "member_count_min": 5,
                "theme_count": 1,
            },
            "top_themes": [{"theme_id": "industry::auto-parts"}],
        },
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        run_id="a-full",
    )
    single_path = store.save(
        {
            "status": "success",
            "metadata": {
                "scanned_symbol_count": 1,
                "member_count_min": 5,
                "theme_count": 0,
            },
            "top_themes": [],
        },
        market="CN",
        universe_key="full_a",
        as_of="20260618",
        run_id="z-single",
    )

    assert full_path.exists()
    assert single_path.exists()
    latest = store.load_latest(market="CN", universe_key="full_a")

    assert latest is not None
    assert latest["run_id"] == "a-full"


def test_theme_snapshot_store_load_recent_returns_bounded_valid_history(tmp_path):
    store = ThemeSnapshotStore(tmp_path)
    for day, score in (("20260616", 54), ("20260617", 57), ("20260618", 60)):
        store.save(
            {
                "status": "success",
                "metadata": {
                    "scanned_symbol_count": 78,
                    "member_count_min": 5,
                },
                "theme_scores": {"industry::ai": {"score": score}},
            },
            market="CN",
            universe_key="full_a",
            as_of=day,
            run_id=f"run-{day}",
        )
    store.save(
        {
            "status": "success",
            "metadata": {
                "scanned_symbol_count": 1,
                "member_count_min": 5,
            },
            "theme_scores": {},
        },
        market="CN",
        universe_key="full_a",
        as_of="20260619",
        run_id="single-symbol",
    )

    recent = store.load_recent(market="CN", universe_key="full_a", limit=2)

    assert [payload["as_of"] for payload in recent] == ["20260617", "20260618"]
    assert [payload["theme_rotation"]["theme_scores"]["industry::ai"]["score"] for payload in recent] == [57, 60]
