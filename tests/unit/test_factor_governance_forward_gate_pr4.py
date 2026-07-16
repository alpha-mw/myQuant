from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

import quant_investor.factors.governance_protocol_v2 as protocol
import scripts.daily_factor_mining_automation as daily
import scripts.factor_health_automation as health
import scripts.mine_quant_branch_factors as mining

BLOCKER = "forward_factor_apply_not_authorized_pr4"


class Poison:
    def __getattribute__(self, name: str):
        raise AssertionError(f"forward gate touched poison input: {name}")

    def __fspath__(self):
        raise AssertionError("forward gate coerced poison path")

    def __iter__(self):
        raise AssertionError("forward gate iterated poison input")


def _explode(*_args, **_kwargs):
    raise AssertionError("forward gate crossed into downstream work")


def _first_statements(function):
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    function_node = tree.body[0]
    assert isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef))
    statements = list(function_node.body)
    if (
        statements
        and isinstance(statements[0], ast.Expr)
        and isinstance(statements[0].value, ast.Constant)
        and isinstance(statements[0].value.value, str)
    ):
        statements.pop(0)
    return statements


def _valid_daily_apply_argv() -> list[str]:
    return [
        "--apply-governed-transitions",
        "--protocol-version",
        "v2",
        "--expected-protocol-hash",
        protocol.protocol_hash(),
        "--governed-evidence-json",
        "poison-evidence.json",
    ]


def test_forward_apply_static_contract_and_hash_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert protocol.FORWARD_PRODUCTION_APPLY_ENABLED is False
    assert protocol.FORWARD_PRODUCTION_APPLY_BLOCKER == BLOCKER

    contract = protocol.canonical_replay_producer_contract()
    policy = protocol.protocol_policy()
    assert policy["canonical_replay_producer_contract"] == contract
    assert "canonical_replay_producer_control" not in policy
    assert policy["forward_production_apply"] == {
        "enabled": False,
        "blocker": BLOCKER,
    }

    control = protocol.canonical_replay_producer_control()
    assert control["producer_implemented"] is True
    for field_name in (
        "local_bytes_readback_verified",
        "canonical_producer_authenticated",
        "production_apply_authorized",
        "production_apply_eligible",
    ):
        assert control[field_name] is False
    assert control["blocker"] == BLOCKER

    before = protocol.protocol_hash()
    monkeypatch.setattr(
        protocol,
        "canonical_replay_producer_control",
        lambda: {
            "producer_implemented": True,
            "local_bytes_readback_verified": True,
            "canonical_producer_authenticated": True,
            "production_apply_authorized": True,
            "production_apply_eligible": True,
            "blocker": "",
        },
    )
    assert protocol.protocol_hash() == before


def test_apply_write_true_returns_static_block_before_any_input_or_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(protocol, "Path", _explode)
    monkeypatch.setattr(protocol, "load_registry_snapshot_strict", _explode)
    monkeypatch.setattr(protocol, "protocol_hash", _explode)
    monkeypatch.setattr(
        protocol,
        "canonical_replay_producer_control",
        lambda: {
            "producer_implemented": True,
            "local_bytes_readback_verified": True,
            "canonical_producer_authenticated": True,
            "production_apply_authorized": True,
            "production_apply_eligible": True,
            "blocker": "",
        },
    )
    monkeypatch.setattr(protocol, "apply_factor_record_patch", _explode)
    monkeypatch.setattr(protocol, "reserve_monthly_mutation_budget", _explode)

    result = protocol.apply_governed_transition(
        Poison(),
        Poison(),
        expected_protocol_hash=Poison(),
        valid_trading_days=Poison(),
        write=True,
    )

    assert result["status"] == "blocked"
    assert result["apply_requested"] is True
    assert result["blockers"] == [BLOCKER]
    assert result["changed_record_names"] == []


def test_apply_function_has_literal_write_gate_as_first_effective_statement() -> None:
    statements = _first_statements(protocol.apply_governed_transition)
    assert isinstance(statements[0], ast.If)
    assert isinstance(statements[0].test, ast.Name)
    assert statements[0].test.id == "write"
    assert isinstance(statements[0].body[0], ast.Return)


def test_daily_direct_apply_run_blocks_before_mining_reports_or_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = daily.parse_args(_valid_daily_apply_argv())
    monkeypatch.setattr(daily, "_now_shanghai", _explode)
    monkeypatch.setattr(daily, "run_mining", _explode)
    monkeypatch.setattr(daily, "latest_download_report", _explode)
    monkeypatch.setattr(daily, "load_registry_snapshot_strict", _explode)
    monkeypatch.setattr(
        daily,
        "canonical_replay_producer_control",
        lambda: {"production_apply_eligible": True, "blocker": ""},
    )

    result = daily.run_daily_automation(args)

    assert result["factor_protocol"]["status"] == "blocked"
    assert result["factor_protocol"]["schema_version"] == (
        "factor-governance-protocol.v3"
    )
    assert result["factor_protocol"]["blockers"] == [BLOCKER]
    assert result["registry_write"] is False


def test_daily_apply_cli_exits_after_parse_without_running_automation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(daily, "run_daily_automation", _explode)

    assert daily.main(_valid_daily_apply_argv()) == 2
    assert BLOCKER in capsys.readouterr().err


def test_factor_health_apply_blocks_before_semantics_directories_or_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "must-not-exist"
    monkeypatch.setattr(health, "load_registry_snapshot_strict", _explode)
    monkeypatch.setattr(health, "build_runtime_smoke", _explode)

    assert (
        health.main(
            [
                "--apply-registry-actions",
                "--horizon-days",
                "0",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 2
    )
    assert BLOCKER in capsys.readouterr().err
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "helper_name",
    [
        "apply_production_candidate_registry_updates",
        "apply_production_family_governance",
    ],
)
def test_mining_direct_write_helpers_block_before_path_or_registry_read(
    helper_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mining, "Path", _explode)
    monkeypatch.setattr(mining, "load_registry_snapshot_strict", _explode)
    helper = getattr(mining, helper_name)
    if helper_name == "apply_production_candidate_registry_updates":
        result = helper(
            registry_path=Poison(),
            qualified_results=Poison(),
            run_timestamp=Poison(),
            run_id=Poison(),
            report_path=Poison(),
            owner=Poison(),
            source_notes=Poison(),
            journal_path=Poison(),
            write=True,
        )
    else:
        result = helper(
            registry_path=Poison(),
            results=Poison(),
            run_timestamp=Poison(),
            run_id=Poison(),
            report_path=Poison(),
            journal_path=Poison(),
            write=True,
        )

    assert result["status"] == "blocked"
    assert result["fail_closed_reason"] == BLOCKER
    assert result["changed_record_names"] == []


def test_mining_apply_cli_uses_forward_gate_blocker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(mining, "run_mining", _explode)

    assert mining.main(["--write-production-candidates"]) == 2
    assert BLOCKER in capsys.readouterr().err


def test_apply_entrypoint_statement_order_is_fail_closed() -> None:
    daily_run = _first_statements(daily.run_daily_automation)
    assert isinstance(daily_run[0], ast.If)

    daily_main = _first_statements(daily.main)
    assert isinstance(daily_main[0], ast.Assign)
    assert isinstance(daily_main[1], ast.If)

    health_main = _first_statements(health.main)
    assert isinstance(health_main[0], ast.Assign)
    assert isinstance(health_main[1], ast.If)

    mining_helper = _first_statements(mining.apply_production_candidate_registry_updates)
    assert isinstance(mining_helper[0], ast.If)


def test_report_only_defaults_and_public_signatures_remain() -> None:
    assert (
        inspect.signature(protocol.apply_governed_transition).parameters["write"].default is False
    )
    assert daily.parse_args([]).apply_governed_transitions is False
    assert health.parse_args([]).apply_registry_actions is False
    assert mining.parse_args([]).write_production_candidates is False
