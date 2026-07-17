from __future__ import annotations

from quant_investor.cli import main as cli


def test_codex_review_public_commands_parse_strict_cas_inputs() -> None:
    parser = cli._build_parser()
    exported = parser.parse_args(
        [
            "market",
            "codex-review-export",
            "--run-id",
            "run-1",
            "--expected-state-sha256",
            "a" * 64,
        ]
    )
    assert exported.root == "results/v16/codex_review"
    assert exported.run_id == "run-1"
    assert exported.market_command == "codex-review-export"

    received = parser.parse_args(
        [
            "market",
            "codex-review-receive",
            "--run-id",
            "run-1",
            "--response",
            "response.json",
            "--expected-state-sha256",
            "b" * 64,
        ]
    )
    assert received.response == "response.json"

    resumed = parser.parse_args(
        [
            "market",
            "codex-review-resume",
            "--run-id",
            "run-1",
            "--expected-state-sha256",
            "c" * 64,
            "--total-capital",
            "1000000",
        ]
    )
    assert resumed.total_capital == 1_000_000.0


def test_codex_review_cli_dispatches_without_resolving_llm_models(
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_codex_review_validate",
        lambda **kwargs: captured.update(kwargs) or {"state": "S1_VALIDATED"},
    )
    monkeypatch.setattr(
        cli.ResolvedReviewModels,
        "from_mapping",
        classmethod(lambda cls, mapping: (_ for _ in ()).throw(AssertionError("LLM resolution"))),
    )
    cli.main(
        [
            "market",
            "codex-review-validate",
            "--root",
            "/tmp/review",
            "--run-id",
            "run-1",
            "--expected-state-sha256",
            "d" * 64,
        ]
    )
    assert captured == {
        "root": "/tmp/review",
        "run_id": "run-1",
        "expected_state_sha256": "d" * 64,
    }
    assert '"state": "S1_VALIDATED"' in capsys.readouterr().out


def test_market_codex_review_cli_is_the_formal_dispatch_path(
    monkeypatch,
    capsys,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_codex_review_status",
        lambda **kwargs: captured.update(kwargs) or {"state": "S1_EXPORTED"},
    )
    cli.main(
        [
            "market",
            "codex-review-status",
            "--root",
            "/tmp/review",
            "--run-id",
            "run-2",
        ]
    )
    assert captured == {"root": "/tmp/review", "run_id": "run-2"}
    assert '"state": "S1_EXPORTED"' in capsys.readouterr().out


def test_market_decision_protocol_defaults_v15_and_explicit_v16_is_forwarded(
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "run_market_analysis",
        lambda **kwargs: captured.append(kwargs),
    )
    cli.main(["market", "analyze", "--market", "CN"])
    assert captured[-1]["decision_protocol"] == "v15"
    assert captured[-1]["enable_agent_layer"] is True

    cli.main(
        [
            "market",
            "analyze",
            "--market",
            "CN",
            "--decision-protocol",
            "v16",
            "--no-agent-layer",
        ]
    )
    assert captured[-1]["decision_protocol"] == "v16"
    assert captured[-1]["enable_agent_layer"] is False


def test_market_run_forwards_v16_protocol_to_pipeline(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_market_pipeline",
        lambda **kwargs: captured.update(kwargs),
    )
    cli.main(
        [
            "market",
            "run",
            "--market",
            "CN",
            "--decision-protocol",
            "v16",
        ]
    )
    assert captured["decision_protocol"] == "v16"
    assert captured["enable_agent_layer"] is True
