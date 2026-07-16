from __future__ import annotations

from pathlib import Path

import pytest

from quant_investor.market.macro_mart import (
    MacroMartPromotionError,
    _copy_staged_generation,
)
from quant_investor.macro.registry import NATIONAL_DOMAIN_WEIGHTS
from quant_investor.macro.v15_controls import (
    V15MacroControlError,
    build_v15_macro_controls,
)


def _snapshot() -> dict[str, object]:
    states = {domain: 0.1 for domain in NATIONAL_DOMAIN_WEIGHTS}
    states["credit_liquidity"] = 0.4
    states["policy_fiscal"] = 0.25
    return {
        "readiness_status": "pass",
        "national_states": states,
        "coverage": {"national": 0.875},
        "snapshot_hash": "a" * 64,
    }


def test_v15_controls_use_domain_score_liquidity_and_fiscal_policy() -> None:
    controls = build_v15_macro_controls(
        _snapshot(), volatility_percentile=72.5
    )

    assert controls["liquidity_score"] == 0.4
    assert controls["policy_signal"] == "supportive"
    assert controls["macro_score_100"] == pytest.approx(
        50.0 * (controls["macro_score"] + 1.0)
    )
    assert controls["read_only_projection"] is True
    assert len(controls["semantic_sha256"]) == 64


@pytest.mark.parametrize(
    ("mutator", "blocker"),
    [
        (
            lambda payload: payload.update(readiness_status="blocked"),
            "macro_v15_snapshot_not_ready",
        ),
        (
            lambda payload: payload["national_states"].pop("external"),
            "macro_v15_domain_missing:external",
        ),
        (
            lambda payload: payload.update(coverage={"national": 0.79}),
            "macro_v15_national_coverage_below_80pct",
        ),
    ],
)
def test_v15_controls_fail_closed(mutator, blocker: str) -> None:
    payload = _snapshot()
    mutator(payload)
    with pytest.raises(V15MacroControlError, match=blocker):
        build_v15_macro_controls(payload, volatility_percentile=50.0)


def test_v15_policy_thresholds_are_exact() -> None:
    payload = _snapshot()
    payload["national_states"]["policy_fiscal"] = -0.25
    assert build_v15_macro_controls(
        payload, volatility_percentile=50.0
    )["policy_signal"] == "restrictive"
    payload["national_states"]["policy_fiscal"] = 0.249
    assert build_v15_macro_controls(
        payload, volatility_percentile=50.0
    )["policy_signal"] == "neutral"


def test_macro_promotion_copy_rejects_staging_symlink(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (source / "manifest.json").symlink_to(outside)

    with pytest.raises(
        MacroMartPromotionError, match="macro_staging_symlink_rejected"
    ):
        _copy_staged_generation(
            source=source,
            destination=tmp_path / "canonical" / "run-1",
        )


def test_macro_promotion_copy_is_immutable_and_exact(tmp_path: Path) -> None:
    source = tmp_path / "staging"
    source.mkdir()
    (source / "manifest.json").write_text('{"ok":true}\n', encoding="utf-8")
    destination = tmp_path / "canonical" / "run-1"
    destination.parent.mkdir()

    _copy_staged_generation(source=source, destination=destination)

    assert (destination / "manifest.json").read_bytes() == (
        source / "manifest.json"
    ).read_bytes()
    with pytest.raises(MacroMartPromotionError, match="macro_generation_exists"):
        _copy_staged_generation(source=source, destination=destination)
