from __future__ import annotations

import json
import stat
from pathlib import Path

import pandas as pd
import pytest

import quant_investor.config as config_module
from quant_investor.market.dag import theme_context
from quant_investor.market.dag.theme_context import build_theme_rotation_metadata
from quant_investor.themes.membership_migration import (
    approve_membership_v2_draft,
    build_membership_v2_draft,
    validate_membership_v2_store,
)


def _row(symbol: str = "000001.SZ") -> dict[str, object]:
    return {
        "schema_version": "theme_membership.v2",
        "membership_id": f"ai-{symbol}",
        "theme_id": "tech::ai",
        "theme_name": "Artificial Intelligence",
        "theme_type": "technology",
        "symbol": symbol,
        "taxonomy_node_id": "tech::ai",
        "supply_chain_role": "application",
        "revenue_exposure": None,
        "effective_from": "2026-01-01",
        "effective_to": "",
        "available_at": "2026-07-01",
        "updated_at": "2026-07-01T00:00:00Z",
        "confidence": 0.9,
        "source_type": "local_diligence",
        "source_ref": "local://explicit-membership-source",
    }


def _source(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "membership.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_missing_trusted_symbol_master_is_coverage_blocked_and_cannot_approve(
    tmp_path: Path,
) -> None:
    result = build_membership_v2_draft(
        _source(tmp_path, [_row()]),
        draft_dir=tmp_path / "drafts",
    )

    assert result["status"] == "coverage_blocked"
    assert result["formal_activation_ready"] is False
    assert result["mapping_inferred"] is False
    assert "trusted_symbol_master_missing" in result["coverage_blockers"]
    with pytest.raises(ValueError, match="coverage is blocked"):
        approve_membership_v2_draft(
            result["draft_path"],
            expected_draft_hash=result["draft_hash"],
            canonical_path=tmp_path / "canonical.jsonl",
        )


def test_canonical_v2_draft_requires_valid_updated_at(tmp_path: Path) -> None:
    row = _row()
    row.pop("updated_at")
    master = tmp_path / "symbols.json"
    master.write_text(json.dumps({"symbols": ["000001.SZ"]}), encoding="utf-8")

    result = build_membership_v2_draft(
        _source(tmp_path, [row]),
        symbol_master_path=master,
        draft_dir=tmp_path / "drafts",
    )

    assert result["status"] == "validation_blocked"
    assert any("updated_at is required" in item for item in result["validation_errors"])


def test_explicit_local_source_and_trusted_master_can_be_approved_atomically(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, [_row()])
    master = tmp_path / "symbols.json"
    master.write_text(json.dumps({"symbols": ["000001.SZ"]}), encoding="utf-8")
    result = build_membership_v2_draft(
        source,
        symbol_master_path=master,
        draft_dir=tmp_path / "drafts",
    )
    canonical = tmp_path / "canonical.jsonl"

    assert result["status"] == "ready_for_approval"
    assert result["formal_activation_ready"] is True
    approved = approve_membership_v2_draft(
        result["draft_path"],
        expected_draft_hash=result["draft_hash"],
        canonical_path=canonical,
    )

    assert approved["status"] == "approved"
    assert approved["record_count"] == 1
    assert approved["mapping_inferred"] is False
    assert stat.S_IMODE(canonical.stat().st_mode) == 0o600
    validation = validate_membership_v2_store(
        canonical,
        symbol_master_path=master,
        as_of="2026-07-10",
    )
    assert validation["status"] == "success"
    assert validation["formal_activation_ready"] is True


def test_symbol_not_in_trusted_master_is_coverage_blocked_without_inference(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, [_row("999999.SZ")])
    master = tmp_path / "symbols.json"
    master.write_text(json.dumps({"symbols": ["000001.SZ"]}), encoding="utf-8")

    result = build_membership_v2_draft(
        source,
        symbol_master_path=master,
        draft_dir=tmp_path / "drafts",
    )

    assert result["status"] == "coverage_blocked"
    assert result["mapping_inferred"] is False
    assert result["coverage_blockers"] == [
        "symbols_missing_from_trusted_master=999999.SZ"
    ]


def test_approved_canonical_store_wires_to_scanner_and_protocol_observer(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path, [_row()])
    master = tmp_path / "symbols.json"
    master.write_text(json.dumps({"symbols": ["000001.SZ"]}), encoding="utf-8")
    draft = build_membership_v2_draft(
        source,
        symbol_master_path=master,
        draft_dir=tmp_path / "drafts",
    )
    canonical = tmp_path / "private" / "theme_membership.v2.jsonl"
    approved = approve_membership_v2_draft(
        draft["draft_path"],
        expected_draft_hash=draft["draft_hash"],
        canonical_path=canonical,
    )
    frame = pd.DataFrame(
        {
            "trade_date": pd.bdate_range(end="2026-07-10", periods=130),
            "close": [10.0 + index * 0.02 for index in range(130)],
            "vol": [1000.0 + index * 5.0 for index in range(130)],
            "amount": [100000.0 + index * 1000.0 for index in range(130)],
        }
    )

    observer = build_theme_rotation_metadata(
        frames={"000001.SZ": frame},
        industry_map={},
        symbol_market_state={},
        market="CN",
        universe_key="full_a",
        as_of="2026-07-10",
        min_member_count=1,
        membership_v2_enabled=True,
        membership_v2_path=canonical,
        membership_v2_required=False,
        membership_v2_expected_sha256=approved["canonical_hash"],
        concept_membership_enabled=False,
        formal_v2_enabled=False,
    )

    assert observer["status"] == "success"
    assert "tech::ai" in observer["theme_scores"]
    assert observer["theme_scores"]["tech::ai"]["membership_source"] == (
        "canonical_theme_membership.v2"
    )
    detail = observer["symbol_theme_membership_details"]["000001.SZ"][0]
    assert detail["canonical_membership_v2"] is True
    assert observer["membership_v2"]["membership_v2_status"] == "success"
    assert observer["membership_v2"]["membership_v2_hash_verified"] is True
    assert observer["membership_v2"]["membership_v2_pit_status"] == "success"
    protocol = observer["protocol_v2"]
    assert protocol["status"] == "observer"
    assert protocol["pit_membership_count"] == 1
    assert protocol["formal_pool"] == []

    formal_blocked = build_theme_rotation_metadata(
        frames={"000001.SZ": frame},
        industry_map={},
        symbol_market_state={},
        market="CN",
        universe_key="full_a",
        as_of="2026-07-10",
        min_member_count=1,
        membership_v2_enabled=True,
        membership_v2_path=canonical,
        membership_v2_required=False,
        membership_v2_expected_sha256=approved["canonical_hash"],
        concept_membership_enabled=False,
        formal_v2_enabled=True,
        formal_v2_kill_switch=False,
    )["protocol_v2"]
    assert formal_blocked["status"] == "blocked"
    assert formal_blocked["prequalified_pool"] == []
    assert "theme_membership_v2_not_required" in formal_blocked[
        "formal_activation_blockers"
    ]

    formal_prerequisites_ready = build_theme_rotation_metadata(
        frames={"000001.SZ": frame},
        industry_map={},
        symbol_market_state={},
        market="CN",
        universe_key="full_a",
        as_of="2026-07-10",
        min_member_count=1,
        membership_v2_enabled=True,
        membership_v2_path=canonical,
        membership_v2_required=True,
        membership_v2_expected_sha256=approved["canonical_hash"],
        concept_membership_enabled=False,
        formal_v2_enabled=True,
        formal_v2_kill_switch=False,
    )["protocol_v2"]
    assert formal_prerequisites_ready["formal_activation_blockers"] == [
        "canonical_joint_replay_producer_not_implemented",
        "theme_live_shadow_20_distinct_days_unverified",
        "formal_reconciliation_persistence_disabled",
    ]
    assert formal_prerequisites_ready["formal_activation_ready"] is False


def test_live_shadow_blocker_is_independent_from_canonical_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, [_row()])
    master = tmp_path / "symbols.json"
    master.write_text(json.dumps({"symbols": ["000001.SZ"]}), encoding="utf-8")
    draft = build_membership_v2_draft(
        source,
        symbol_master_path=master,
        draft_dir=tmp_path / "drafts",
    )
    canonical = tmp_path / "private" / "theme_membership.v2.jsonl"
    approved = approve_membership_v2_draft(
        draft["draft_path"],
        expected_draft_hash=draft["draft_hash"],
        canonical_path=canonical,
    )
    frame = pd.DataFrame(
        {
            "trade_date": pd.bdate_range(end="2026-07-10", periods=130),
            "close": [10.0 + index * 0.02 for index in range(130)],
            "vol": [1000.0 + index * 5.0 for index in range(130)],
            "amount": [100000.0 + index * 1000.0 for index in range(130)],
        }
    )
    monkeypatch.setattr(
        theme_context.replay_v13_1,
        "CANONICAL_JOINT_REPLAY_PRODUCER_AVAILABLE",
        True,
    )

    protocol = build_theme_rotation_metadata(
        frames={"000001.SZ": frame},
        industry_map={},
        symbol_market_state={},
        market="CN",
        universe_key="full_a",
        as_of="2026-07-10",
        min_member_count=1,
        membership_v2_enabled=True,
        membership_v2_path=canonical,
        membership_v2_required=True,
        membership_v2_expected_sha256=approved["canonical_hash"],
        concept_membership_enabled=False,
        formal_v2_enabled=True,
        formal_v2_kill_switch=False,
    )["protocol_v2"]

    assert "canonical_joint_replay_producer_not_implemented" not in protocol[
        "formal_activation_blockers"
    ]
    assert "theme_live_shadow_20_distinct_days_unverified" in protocol[
        "formal_activation_blockers"
    ]


def test_live_shadow_blocker_clears_only_after_verified_20_distinct_days(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path, [_row()])
    master = tmp_path / "symbols.json"
    master.write_text(json.dumps({"symbols": ["000001.SZ"]}), encoding="utf-8")
    draft = build_membership_v2_draft(
        source,
        symbol_master_path=master,
        draft_dir=tmp_path / "drafts",
    )
    canonical = tmp_path / "private" / "theme_membership.v2.jsonl"
    approved = approve_membership_v2_draft(
        draft["draft_path"],
        expected_draft_hash=draft["draft_hash"],
        canonical_path=canonical,
    )
    frame = pd.DataFrame(
        {
            "trade_date": pd.bdate_range(end="2026-07-10", periods=130),
            "close": [10.0 + index * 0.02 for index in range(130)],
            "vol": [1000.0 + index * 5.0 for index in range(130)],
            "amount": [100000.0 + index * 1000.0 for index in range(130)],
        }
    )
    manifest_sha = "a" * 64
    shadow_dates = [
        value.date().isoformat()
        for value in pd.bdate_range(end="2026-07-10", periods=20)
    ]
    monkeypatch.setattr(
        theme_context.replay_v13_1,
        "CANONICAL_JOINT_REPLAY_PRODUCER_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        config_module.Config,
        "THEME_V2_JOINT_MANIFEST_PATH",
        "verified.json",
    )
    monkeypatch.setattr(
        config_module.Config,
        "THEME_V2_EXPECTED_JOINT_MANIFEST_SHA256",
        manifest_sha,
    )

    def _verified_manifest(
        path: str,
        *,
        expected_artifact_sha256: str,
        expected_theme_protocol_hash: str,
    ) -> dict[str, object]:
        assert path == "verified.json"
        assert expected_artifact_sha256 == manifest_sha
        assert len(expected_theme_protocol_hash) == 64
        return {
            "ready": True,
            "readback_verified": True,
            "artifact_sha256": manifest_sha,
            "manifest": {
                "theme_live_shadow": {
                    "passed": True,
                    "distinct_trade_day_count": 20,
                    "dates": shadow_dates,
                }
            },
        }

    monkeypatch.setattr(
        theme_context.replay_v13_1,
        "verify_joint_replay_manifest",
        _verified_manifest,
    )

    protocol = build_theme_rotation_metadata(
        frames={"000001.SZ": frame},
        industry_map={},
        symbol_market_state={},
        market="CN",
        universe_key="full_a",
        as_of="2026-07-10",
        min_member_count=1,
        membership_v2_enabled=True,
        membership_v2_path=canonical,
        membership_v2_required=True,
        membership_v2_expected_sha256=approved["canonical_hash"],
        concept_membership_enabled=False,
        formal_v2_enabled=True,
        formal_v2_kill_switch=False,
    )["protocol_v2"]

    assert "theme_live_shadow_20_distinct_days_unverified" not in protocol[
        "formal_activation_blockers"
    ]
