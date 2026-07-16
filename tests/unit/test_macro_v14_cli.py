from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_investor.cli import main as cli_main
from quant_investor.macro.providers import (
    MacroFetchRequest,
    OfficialMacroProvider,
    maintain_macro_observations,
)
from quant_investor.macro.store import pointer_sha256, publish_observations


def _observation(
    *,
    source: str = "nbs_official",
    period: str = "2024-04-30",
    available: str = "2024-05-01T06:00:00+00:00",
) -> dict[str, object]:
    return {
        "indicator_id": "cn.pmi_manufacturing",
        "dimension_type": "national",
        "industry_chain": "",
        "period_end": period,
        "release_at": available,
        "available_at": available,
        "vintage_id": "initial",
        "value": 10.0,
        "unit": "index",
        "frequency": "monthly",
        "source_system": source,
        "source_record_id": f"{source}:{period}",
        "source_url": (
            "https://www.stats.gov.cn/fixture"
            if source == "nbs_official"
            else "https://tushare.pro/document/fixture"
        ),
        "fetched_at": available,
        "quality_status": "pass",
    }


def test_macro_maintain_rejects_ambiguous_input_modes(
    tmp_path: Path,
) -> None:
    row = tmp_path / "row.json"
    observations = tmp_path / "observations.json"
    row.write_text("{}", encoding="utf-8")
    observations.write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        cli_main.main(
            [
                "market",
                "macro-maintain",
                "--market",
                "CN",
                "--as-of",
                "2024-05-10",
                "--input-json",
                str(row),
                "--input-observations",
                str(observations),
            ]
        )

    assert exc.value.code == 2


def test_macro_compatibility_staging_preserves_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "row.json"
    path.write_text(json.dumps({"macro_score": 0.2}), encoding="utf-8")
    captured: dict[str, object] = {}

    def _fake(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "staged", "promoted": False}

    monkeypatch.setattr(cli_main, "run_macro_maintenance", _fake)
    cli_main.main(
        [
            "market",
            "macro-maintain",
            "--market",
            "CN",
            "--as-of",
            "2024-05-10",
            "--input-json",
            str(path),
            "--run-id",
            "fixture-run",
        ]
    )

    assert captured["run_id"] == "fixture-run"
    assert captured["indicators"] == {"macro_score": 0.2}
    assert '"promoted": false' in capsys.readouterr().out


def test_macro_observation_maintenance_preserves_run_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "observations.json"
    path.write_text("[]", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_observation_maintenance",
        lambda **kwargs: captured.update(kwargs) or {"promoted": False},
    )

    cli_main.main(
        [
            "market",
            "macro-maintain",
            "--market",
            "CN",
            "--as-of",
            "2024-05-10",
            "--input-observations",
            str(path),
            "--run-id",
            "observation-run",
        ]
    )

    assert captured["run_id"] == "observation-run"
    assert captured["local_observations"] == []
    assert captured["staging_root"] == "results/v15/macro_observation_staging"
    assert '"promoted": false' in capsys.readouterr().out


def test_macro_standalone_input_stages_without_advancing_canonical_pointer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observations_root = tmp_path / "canonical"
    publish_observations(
        [_observation()],
        root=observations_root,
        run_id="canonical-g1",
    )
    pointer_before = (observations_root / "_latest.json").read_bytes()
    input_path = tmp_path / "standalone.json"
    input_path.write_text(
        json.dumps([_observation(source="tushare_fallback")]),
        encoding="utf-8",
    )
    staging_root = tmp_path / "staging"

    cli_main.main(
        [
            "market",
            "macro-maintain",
            "--market",
            "CN",
            "--as-of",
            "2024-05-10",
            "--input-observations",
            str(input_path),
            "--observations-root",
            str(observations_root),
            "--staging-root",
            str(staging_root),
            "--run-id",
            "standalone-run",
        ]
    )

    assert (observations_root / "_latest.json").read_bytes() == pointer_before
    manifest_path = (
        staging_root
        / "CN"
        / "2024-05-10"
        / "standalone-run"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "staged"
    assert manifest["promoted"] is False
    assert manifest["observer_only"] is True
    assert manifest["production_eligible"] is False
    assert manifest["applied"] is False
    assert manifest["input_provenance"] == "manual_offline_snapshot"
    assert manifest["provider_provenance_retained"] is False
    assert set(manifest_path.parent.iterdir()) == {manifest_path}
    serialized = json.dumps(manifest, sort_keys=True).lower()
    assert "tushare" not in serialized
    assert "source_system" not in serialized
    assert "source_url" not in serialized
    assert "source_record_id" not in serialized
    assert '"promoted": false' in capsys.readouterr().out


@pytest.mark.parametrize(
    "live_flags",
    [
        ["--allow-live"],
        ["--allow-live", "--allow-tushare-fallback"],
    ],
)
def test_macro_standalone_input_cannot_mix_with_live_flags(
    tmp_path: Path,
    live_flags: list[str],
) -> None:
    input_path = tmp_path / "standalone.json"
    input_path.write_text(json.dumps([_observation()]), encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        cli_main.main(
            [
                "market",
                "macro-maintain",
                "--market",
                "CN",
                "--as-of",
                "2024-05-10",
                "--input-observations",
                str(input_path),
                *live_flags,
            ]
        )

    assert exc.value.code == 2


def test_macro_standalone_api_rejects_live_before_provider_fetch(
    tmp_path: Path,
) -> None:
    called = False

    def transport(_indicator_id: str, _request: MacroFetchRequest):
        nonlocal called
        called = True
        return [_observation()]

    result = maintain_macro_observations(
        local_observations=[_observation()],
        market="CN",
        as_of="2024-05-10",
        indicator_ids=["cn.pmi_manufacturing"],
        root=str(tmp_path / "canonical"),
        staging_root=str(tmp_path / "staging"),
        run_id="mixed",
        allow_live=True,
        official_provider=OfficialMacroProvider(transport),
    )

    assert result["status"] == "blocked"
    assert result["promoted"] is False
    assert result["reason"] == (
        "standalone_and_live_modes_are_mutually_exclusive"
    )
    assert called is False
    assert not (tmp_path / "canonical" / "_latest.json").exists()
    assert not (tmp_path / "staging").exists()


def test_macro_injected_official_provider_remains_only_canonical_publish_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "canonical"

    def transport(_indicator_id: str, _request: MacroFetchRequest):
        return [_observation()]

    result = maintain_macro_observations(
        market="CN",
        as_of="2024-05-10",
        indicator_ids=["cn.pmi_manufacturing"],
        root=str(root),
        staging_root=str(tmp_path / "staging"),
        run_id="provider-g1",
        allow_live=True,
        official_provider=OfficialMacroProvider(transport),
    )

    assert result["promoted"] is True
    assert pointer_sha256(root)
    assert not (tmp_path / "staging").exists()


def test_macro_fallback_requires_explicit_live_mode() -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main.main(
            [
                "market",
                "macro-maintain",
                "--market",
                "CN",
                "--as-of",
                "2024-05-10",
                "--allow-tushare-fallback",
            ]
        )

    assert exc.value.code == 2


def test_macro_analyze_dispatches_only_explicit_local_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "observations.json"
    path.write_text("[]", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli_main,
        "run_macro_analysis",
        lambda **kwargs: captured.update(kwargs)
        or {"production_eligible": False, "applied": False},
    )

    cli_main.main(
        [
            "market",
            "macro-analyze",
            "--market",
            "CN",
            "--as-of",
            "2024-05-10",
            "--observations",
            str(path),
        ]
    )

    assert captured["observations_path"] == str(path)
    assert '"applied": false' in capsys.readouterr().out
