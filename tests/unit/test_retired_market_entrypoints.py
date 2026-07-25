from __future__ import annotations

from pathlib import Path

import pytest

import quant_investor.cli.main as cli_main
import quant_investor.market.analyze as market_analyze
import quant_investor.market.run_pipeline as market_pipeline

RETIRED_PROTOCOL = "v" + str(4 * 4)


@pytest.mark.parametrize(
    "command",
    [
        *(
            f"{RETIRED_PROTOCOL}-advisory-{suffix}"
            for suffix in (
                "prepare",
                "receive",
                "finalize",
                "run",
                "provider-resume",
                "status",
                "decision-record",
            )
        ),
        *(
            f"codex-review-{suffix}"
            for suffix in (
                "export",
                "receive",
                "validate",
                "resume",
                "status",
            )
        ),
    ],
)
def test_retired_market_commands_exit_two_without_writes(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(["market", command])

    assert exc_info.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("command", ["analyze", "run"])
def test_retired_decision_protocol_literal_exits_two_without_writes(
    command: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as exc_info:
        cli_main.main(
            [
                "market",
                command,
                "--market",
                "CN",
                "--decision-protocol",
                RETIRED_PROTOCOL,
            ]
        )

    assert exc_info.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_direct_market_analysis_rejects_retired_protocol_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached_resolution = False

    def _unexpected_resolution(*args: object, **kwargs: object) -> object:
        nonlocal reached_resolution
        reached_resolution = True
        raise AssertionError("retired protocol reached market resolution")

    monkeypatch.setattr(market_analyze, "get_market_settings", _unexpected_resolution)

    with pytest.raises(ValueError, match="supports v15 only"):
        market_analyze.run_market_analysis(
            market="CN",
            decision_protocol=RETIRED_PROTOCOL,
        )

    assert reached_resolution is False


def test_direct_market_pipeline_rejects_retired_protocol_before_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reached_resolution = False

    def _unexpected_resolution(*args: object, **kwargs: object) -> object:
        nonlocal reached_resolution
        reached_resolution = True
        raise AssertionError("retired protocol reached market resolution")

    monkeypatch.setattr(market_pipeline, "get_market_settings", _unexpected_resolution)

    with pytest.raises(ValueError, match="supports v15 only"):
        market_pipeline.run_unified_pipeline(
            market="CN",
            decision_protocol=RETIRED_PROTOCOL,
        )

    assert reached_resolution is False


@pytest.mark.parametrize(
    ("runner", "argument_name"),
    [
        (cli_main.run_market_analysis, f"{RETIRED_PROTOCOL}_state_path"),
        (cli_main.run_market_pipeline, f"{RETIRED_PROTOCOL}_state_path"),
    ],
)
def test_cli_wrappers_reject_retired_keyword_surface_before_import(
    runner: object,
    argument_name: str,
) -> None:
    with pytest.raises(ValueError, match="retired"):
        runner(market="CN", **{argument_name: "/tmp/not-read"})
