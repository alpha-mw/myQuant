from __future__ import annotations

import hashlib
import json

import pytest

from quant_investor.v17_v3_runtime import cli
from quant_investor.v17_v3_runtime.authority import DELIVERY_STATUS, authority_envelope


def test_help_lists_closed_commands_and_no_execution_surface(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    output = capsys.readouterr()
    for command in (
        "verify",
        "admit-sources",
        "calibrate-fusion",
        "build-initial-pool",
        "activate-formal-research",
        "analyze",
        "revoke-formal-research",
        "status",
    ):
        assert command in output.out
    assert "execute-order" not in output.out
    assert "broker-connect" not in output.out


def test_verify_phase_a_status_and_authority(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    class Readiness:
        def to_public_wire(self):
            return {
                "status": DELIVERY_STATUS,
                "provider_calls": False,
                **authority_envelope(),
            }

    monkeypatch.setattr(cli, "verify_runtime", lambda: Readiness())
    assert cli.main(["verify"]) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == DELIVERY_STATUS
    assert payload["provider_calls"] is False
    assert payload["formal_research_publication_authority"] is False
    assert payload["execution_authority"] is False
    assert payload["broker_authority"] is False
    assert payload["order_authority"] is False
    assert payload["trade_authority"] is False
    assert captured.err == ""


def test_cli_runtime_error_never_echoes_private_values(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    secret = "600000.SH NAV=123 cash=45 quantity=6 price=7"

    def fail(**kwargs):
        raise ValueError(secret)

    monkeypatch.setattr(cli, "admitted_sources", fail)
    code = cli.main(
        [
            "admit-sources",
            "--source-locator",
            "data/private/v17_v3_sources/locators/x.json",
            "--expected-source-locator-sha256",
            "0" * 64,
            "--workspace-root",
            "/private/tmp/not-used",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert secret not in captured.out
    assert "600000.SH" not in captured.out
    assert "123" not in captured.out
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "BLOCKED"
    assert payload["execution_authority"] is False


def test_argparse_rejects_unknown_mode_without_running_analysis(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    called = False

    def unexpected(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "analyze", unexpected)
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "analyze",
                "--mode",
                "execution",
                "--source-locator",
                "data/private/v17_v3_sources/locators/x.json",
                "--expected-source-locator-sha256",
                "0" * 64,
            ]
        )
    assert exc.value.code == 2
    assert called is False
    captured = capsys.readouterr()
    assert "invalid choice" in captured.err


def test_blocked_activation_returns_exit_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    promotion = b'{"cutoff":"2026-07-25T07:00:00Z",' b'"strategy_id":"cn-research"}\n'
    promotion_sha256 = hashlib.sha256(promotion).hexdigest()

    class Store:
        def read_path(self, path, expected_sha256):
            assert path == "promotion.json"
            assert expected_sha256 == promotion_sha256
            return promotion

    class Outcome:
        status = "ACTIVATION_REJECTED"

        def to_public_wire(self):
            return {
                "status": self.status,
                **authority_envelope(),
            }

    class Publisher:
        def __init__(self, store):
            assert isinstance(store, Store)

        def activate(self, **kwargs):
            assert kwargs["strategy_id"] == "cn-research"
            return Outcome()

    monkeypatch.setattr(cli, "SecureStore", lambda _root: Store())
    monkeypatch.setattr(cli, "ActivationPublisher", Publisher)
    code = cli.main(
        [
            "activate-formal-research",
            "--promotion-receipt",
            "promotion.json",
            "--expected-promotion-receipt-sha256",
            promotion_sha256,
            "--workspace-root",
            "/private/tmp/not-used",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["status"] == "ACTIVATION_REJECTED"
    assert payload["formal_research_publication_authority"] is False
