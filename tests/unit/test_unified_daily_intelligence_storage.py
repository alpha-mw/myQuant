from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

import quant_investor.intelligence.daily as daily
from quant_investor.contracts import canonical_json_bytes
from quant_investor.cli.main import main
from quant_investor.intelligence import IntelligenceError
from quant_investor.intelligence._common import (
    artifact_ref,
    build_artifact,
    business_identity,
    decimal_text,
)
from quant_investor.intelligence.storage import (
    DailyResearchPoolStore,
    PHASE_A_POLICY_RELATIVE_PATH,
    POOL_ROOT_RELATIVE_PATH,
    _publish_phase_a_policy,
    approved_phase_a_policy,
    publish_phase_a_policy,
)
from quant_investor.intelligence import (
    compile_daily_intelligence,
    project_tushare_theme_source,
)


def _observation_ref(alias: str, character: str) -> dict[str, str]:
    return {
        "artifact_id": f"observation-{alias}",
        "byte_sha256": character * 64,
        "contract_sha256": "c" * 64,
        "kind": "factor.production_observation",
        "semantic_sha256": "d" * 64,
    }


def _write_request(path: Path, document: dict) -> str:
    raw = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def _rank(policy: dict, *, signal_date: str = "20260824") -> dict:
    created_at = f"{signal_date[0:4]}-{signal_date[4:6]}-{signal_date[6:8]}T13:30:00Z"
    rows = []
    for ordinal in range(100):
        symbol = f"{ordinal + 1:06d}.SZ"
        percentile = Decimal(100 - ordinal) / Decimal(100)
        text = decimal_text(percentile)
        rows.append(
            {
                "combined_percentile": text,
                "factor_percentiles": {"LOW": text, "W80": text},
                "symbol": symbol,
            }
        )
    return build_artifact(
        kind="factor_research_rank",
        identity_field="rank_id",
        identity=business_identity(
            kind="factor_research_rank",
            identity_inputs={
                "factor_pointer_sha256": "a" * 64,
                "policy_id": policy["artifact_id"],
            },
        ),
        created_at=created_at,
        fields={
            "as_of": created_at,
            "blocker_codes": [],
            "common_symbol_count": 3000,
            "common_symbol_set_sha256": "b" * 64,
            "factor_generation_ref": {
                "artifact_id": "generation-a",
                "byte_sha256": "e" * 64,
                "contract_sha256": "f" * 64,
                "kind": "factor.production_generation",
                "semantic_sha256": "1" * 64,
            },
            "factor_pointer_sha256": "a" * 64,
            "observation_refs": [
                _observation_ref("low", "2"),
                _observation_ref("w80", "3"),
            ],
            "policy_ref": artifact_ref(policy),
            "pool_rows": rows,
            "signal_date": signal_date,
            "status": "READY",
            "strategy_id": "aggressive_tech_manufacturing",
        },
    )


def test_approved_policy_is_exact_prospective_and_idempotent(tmp_path: Path) -> None:
    policy = approved_phase_a_policy()
    payload = policy["payload"]
    assert payload["strategy_id"] == "aggressive_tech_manufacturing"
    assert payload["effective_signal_date"] == "20260822"
    assert payload["technology_policy_state"] == "UNCONFIGURED"
    assert payload["technology_theme_ids"] == []
    assert payload["pool_policy"]["pool_size"] == 100
    assert payload["pool_policy"]["minimum_cohort"] == 3000
    assert payload["decision_thresholds"] == {
        "paper_candidate": "0.900000000000",
        "research_approved": "0.800000000000",
    }

    first = publish_phase_a_policy(tmp_path)
    second = publish_phase_a_policy(tmp_path)
    assert first["command_status"] == "PUBLISHED"
    assert second["command_status"] == "NO_ACTION"
    assert first["policy_sha256"] == second["policy_sha256"]
    path = tmp_path / PHASE_A_POLICY_RELATIVE_PATH
    assert path.stat().st_mode & 0o777 == 0o600
    assert hashlib.sha256(path.read_bytes()).hexdigest() == first["policy_sha256"]


def test_policy_publish_cli_uses_only_code_owned_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["research", "policy-publish", "--workspace-root", str(tmp_path)])
    first = json.loads(capsys.readouterr().out)
    main(["research", "policy-publish", "--workspace-root", str(tmp_path)])
    second = json.loads(capsys.readouterr().out)
    assert first["command_status"] == "PUBLISHED"
    assert second["command_status"] == "NO_ACTION"
    assert first["policy_path"] == PHASE_A_POLICY_RELATIVE_PATH


def test_ineligible_20260821_signal_fails_before_store_creation(tmp_path: Path) -> None:
    policy = approved_phase_a_policy()
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    with pytest.raises(IntelligenceError, match="predates"):
        daily._require_policy_signal_date(policy["payload"], signal_date="20260821")
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert before == after == []
    assert not (tmp_path / POOL_ROOT_RELATIVE_PATH).exists()


def test_unconfigured_policy_is_pool_only() -> None:
    policy = approved_phase_a_policy()
    rank = _rank(policy)
    with pytest.raises(IntelligenceError, match="technology policy is not configured"):
        compile_daily_intelligence(
            as_of=rank["payload"]["as_of"],
            strategy_id="aggressive_tech_manufacturing",
            rank=rank,
            policy=policy,
            industry_projection=None,
            theme_projection=None,
        )
    with pytest.raises(IntelligenceError, match="technology policy is not configured"):
        project_tushare_theme_source(
            dc_plan={},
            dc_capture={},
            dc_partitions=[],
            tdx_plan=None,
            tdx_capture=None,
            tdx_partitions=[],
            policy=policy,
            as_of=rank["payload"]["as_of"],
        )


def test_pool_publication_is_atomic_idempotent_and_conflicting(tmp_path: Path) -> None:
    policy_result = publish_phase_a_policy(tmp_path)
    policy = approved_phase_a_policy()
    rank = _rank(policy)
    calls = []
    store = DailyResearchPoolStore(tmp_path)
    first = store.publish(
        rank=rank,
        expected_policy_sha256=policy_result["policy_sha256"],
        before_publish=lambda: calls.append("checked"),
    )
    second = store.publish(
        rank=rank,
        expected_policy_sha256=policy_result["policy_sha256"],
        before_publish=lambda: calls.append("unexpected"),
    )
    assert first["command_status"] == "PUBLISHED"
    assert second["command_status"] == "NO_ACTION"
    assert calls == ["checked"]
    root = tmp_path / first["pool_root"]
    assert root.stat().st_mode & 0o777 == 0o700
    assert sorted(path.name for path in root.iterdir()) == [
        "factor_research_rank.json",
        "manifest.json",
        "publish_receipt.json",
        "selected_symbols.json",
    ]
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in root.iterdir())
    manifest = json.loads((root / "manifest.json").read_bytes())
    assert manifest["payload"]["factor_admission_route"] == "BOOTSTRAP_EXCEPTION"
    assert manifest["payload"]["prospective_admission_state"] == "NOT_CLAIMED"
    assert manifest["payload"]["run_state"] == "INACTIVE"
    assert manifest["payload"]["authority"] == {
        "broker": False,
        "execution": False,
        "factor_governance_write": False,
        "llm_control": False,
        "mainline_activation": False,
        "order": False,
        "portfolio_activation": False,
        "provider": False,
        "selector_write": False,
        "trade": False,
    }

    (root / "manifest.json").write_bytes(b"{}")
    with pytest.raises(IntelligenceError, match="leaf validation"):
        store.publish(
            rank=rank,
            expected_policy_sha256=policy_result["policy_sha256"],
            before_publish=lambda: None,
        )


def test_pre_rename_failure_leaves_no_final_pool(tmp_path: Path) -> None:
    policy_result = publish_phase_a_policy(tmp_path)
    policy = approved_phase_a_policy()
    store = DailyResearchPoolStore(tmp_path)

    def fail() -> None:
        raise RuntimeError("fault-before-rename")

    with pytest.raises(RuntimeError, match="fault-before-rename"):
        store.publish(
            rank=_rank(policy, signal_date="20260825"),
            expected_policy_sha256=policy_result["policy_sha256"],
            before_publish=fail,
        )
    final = tmp_path / POOL_ROOT_RELATIVE_PATH / "aggressive_tech_manufacturing" / "2026-08-25"
    assert not final.exists()
    staging = list(final.parent.glob(".2026-08-25.staging-*"))
    assert len(staging) == 1
    assert sorted(path.name for path in staging[0].iterdir()) == [
        "factor_research_rank.json",
        "manifest.json",
        "publish_receipt.json",
        "selected_symbols.json",
    ]


def test_policy_conflict_is_not_overwritten(tmp_path: Path) -> None:
    publish_phase_a_policy(tmp_path)
    path = tmp_path / PHASE_A_POLICY_RELATIVE_PATH
    original = path.read_bytes()
    path.write_bytes(b"{}")
    with pytest.raises(IntelligenceError, match="immutable conflict"):
        publish_phase_a_policy(tmp_path)
    assert path.read_bytes() == b"{}"
    assert original != b"{}"


def test_policy_publish_is_atomic_before_final_rename(tmp_path: Path) -> None:
    def fail() -> None:
        raise RuntimeError("fault-before-policy-rename")

    with pytest.raises(RuntimeError, match="fault-before-policy-rename"):
        _publish_phase_a_policy(tmp_path, before_publish=fail)
    final = tmp_path / PHASE_A_POLICY_RELATIVE_PATH
    assert not final.exists()
    temporary = list(final.parent.glob(".v1.json.publish-*"))
    assert len(temporary) == 1
    assert temporary[0].stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("unsafe", ["mode", "hardlink", "symlink", "partial"])
def test_policy_replay_rejects_unsafe_existing_leaf(tmp_path: Path, unsafe: str) -> None:
    result = publish_phase_a_policy(tmp_path)
    path = tmp_path / PHASE_A_POLICY_RELATIVE_PATH
    raw = path.read_bytes()
    if unsafe == "mode":
        path.chmod(0o644)
    elif unsafe == "hardlink":
        (path.parent / "v1-linked.json").hardlink_to(path)
    elif unsafe == "symlink":
        path.unlink()
        target = path.parent / "policy-target.json"
        target.write_bytes(raw)
        target.chmod(0o600)
        path.symlink_to(target.name)
    else:
        path.write_bytes(raw[: max(1, len(raw) // 2)])
    with pytest.raises(IntelligenceError, match="immutable conflict"):
        publish_phase_a_policy(tmp_path)
    assert result["policy_sha256"] == hashlib.sha256(raw).hexdigest()


def test_store_rejects_predate_rank_before_pool_store_creation(tmp_path: Path) -> None:
    policy_result = publish_phase_a_policy(tmp_path)
    policy = approved_phase_a_policy()
    store = DailyResearchPoolStore(tmp_path)
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    with pytest.raises(IntelligenceError, match="predates"):
        store.publish(
            rank=_rank(policy, signal_date="20260821"),
            expected_policy_sha256=policy_result["policy_sha256"],
            before_publish=lambda: pytest.fail("must not reach publication callback"),
        )
    after = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    assert before == after
    assert not (tmp_path / POOL_ROOT_RELATIVE_PATH).exists()
