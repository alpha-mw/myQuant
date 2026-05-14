from __future__ import annotations

import shutil
from pathlib import Path

from quant_investor.factors.report import (
    FACTOR_LIBRARY_SHADOW_NON_RUNTIME_IMPACT_NOTE,
    load_factor_library_shadow_status,
    render_factor_library_shadow_markdown,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_FIXTURE_ROOT = ROOT / "tests" / "fixtures" / "factor_library_shadow"

EXPECTED_PRODUCTION_FACTOR_IDS = [
    "factor_allowed_momentum_v1",
    "factor_expired_value_v1",
]
EXPECTED_BLOCKED_FACTOR_IDS = ["factor_expired_value_v1"]
EXPECTED_SHADOW_ONLY_FACTOR_IDS = ["factor_shadow_quality_v1"]


def _copy_canonical_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "factor_library_shadow"
    shutil.copytree(CANONICAL_FIXTURE_ROOT, target)
    return target


def test_canonical_factor_shadow_fixture_loads_successfully() -> None:
    status = load_factor_library_shadow_status(
        root_dir=CANONICAL_FIXTURE_ROOT,
        as_of="2026-05-01",
    )

    assert status["library_exists"] is True
    assert status["audit_exists"] is True
    assert status["production_factor_count"] == 2
    assert status["blocked_factor_count"] == 1
    assert status["shadow_only_factor_count"] == 1
    assert status["expired_factor_count"] == 1
    assert status["warning_count"] == 1
    assert status["blocker_count"] == 1
    assert status["verdict"] == "fail"
    assert status["production_factor_ids"] == EXPECTED_PRODUCTION_FACTOR_IDS
    assert status["production_factor_ids"] == sorted(status["production_factor_ids"])
    assert status["blocked_factor_ids"] == EXPECTED_BLOCKED_FACTOR_IDS
    assert status["blocked_factor_ids"] == sorted(status["blocked_factor_ids"])
    assert status["shadow_only_factor_ids"] == EXPECTED_SHADOW_ONLY_FACTOR_IDS


def test_shadow_markdown_renders_canonical_fixture_deterministically() -> None:
    status = load_factor_library_shadow_status(
        root_dir=CANONICAL_FIXTURE_ROOT,
        as_of="2026-05-01",
    )

    markdown = render_factor_library_shadow_markdown(status)
    markdown_again = render_factor_library_shadow_markdown(status)

    assert markdown == markdown_again
    assert "| `production_factor_count` | 2 |" in markdown
    assert "| Verdict | `fail` |" in markdown
    assert "factor_expired_value_v1" in markdown
    assert FACTOR_LIBRARY_SHADOW_NON_RUNTIME_IMPACT_NOTE in markdown


def test_malformed_production_fixture_returns_fail_status_without_crashing(tmp_path) -> None:
    fixture_root = _copy_canonical_fixture(tmp_path)
    (fixture_root / "production_factors.json").write_text("{bad json}", encoding="utf-8")

    status = load_factor_library_shadow_status(root_dir=fixture_root, as_of="2026-05-01")
    markdown = render_factor_library_shadow_markdown(status)

    assert status["library_exists"] is True
    assert status["verdict"] == "fail"
    assert "Malformed JSON" in " ".join(status["warnings"])
    assert "Malformed JSON" in markdown


def test_malformed_audit_fixture_returns_fail_status_without_crashing(tmp_path) -> None:
    fixture_root = _copy_canonical_fixture(tmp_path)
    audit_path = fixture_root / "audit" / "factor_library_audit_reports.jsonl"
    audit_path.write_text("{bad json}\n", encoding="utf-8")

    status = load_factor_library_shadow_status(root_dir=fixture_root, as_of="2026-05-01")
    markdown = render_factor_library_shadow_markdown(status)

    assert status["library_exists"] is True
    assert status["audit_exists"] is True
    assert status["verdict"] == "fail"
    assert "Malformed JSONL" in " ".join(status["warnings"])
    assert "Malformed JSONL" in markdown


def test_missing_factor_shadow_fixture_warns_without_raising(tmp_path) -> None:
    status = load_factor_library_shadow_status(
        root_dir=tmp_path / "missing_factor_library",
        as_of="2026-05-01",
    )

    assert status["library_exists"] is False
    assert status["audit_exists"] is False
    assert status["verdict"] == "warn"
    assert status["production_factor_count"] == 0
    assert status["blocked_factor_count"] == 0
    assert status["shadow_only_factor_count"] == 0
    assert status["warnings"]
