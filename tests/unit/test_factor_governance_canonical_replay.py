from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

import quant_investor.factors.governance_canonical_replay as canonical_module
from quant_investor.factors.governance_canonical_replay import (
    ARM_NAMES,
    BUNDLE_SCHEMA_VERSION,
    CODE_CONFIG_ROLES,
    CONTROL_CHAIN_STAGES,
    CanonicalReplayError,
    SafeReadSession,
    canonical_json_bytes,
    producer_contract_sha256,
    produce_canonical_replay,
    publish_immutable_json,
    strict_json_loads,
    verify_canonical_replay,
)
from scripts.build_factor_governance_canonical_replay import main as replay_main


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _write_json(path: Path, payload: Any, *, mode: int = 0o600) -> str:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw = _json_bytes(payload)
    path.write_bytes(raw)
    path.chmod(mode)
    return _sha(raw)


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha(path.read_bytes())}


def _publish_reserved_paths(
    parent: Path, leaf: str, payload: dict[str, Any]
) -> tuple[Path, Path]:
    raw = _json_bytes(payload)
    leaf_sha = _sha(leaf.encode("utf-8"))
    lock = parent / f".canonical-publish.{leaf_sha}.lock"
    temp = parent / f".canonical-publish.{leaf_sha}.{_sha(raw)}.tmp"
    return lock, temp


def _factor_record(
    name: str, registry_records: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    registry_record = registry_records[name]
    return {
        "name": name,
        "family": "value",
        "slot": "value::primary",
        "registry_state": registry_record["state"],
        "registry_record_sha256": _sha(
            canonical_json_bytes(registry_record)
        ),
        "p_value": 0.01,
        "health_failure_windows": [],
        "month_end_rankic_dates": [
            f"2025-{month:02d}-28" for month in range(1, 13)
        ],
        "forward_cohorts": [],
        "walk_forward": {
            "purged": True,
            "purge_days": 30,
            "embargo_days": 30,
            "folds": [
                {
                    "train_end": "2025-01-28",
                    "validation_start": "2025-03-28",
                    "validation_end": "2025-04-28",
                    "evidence_hash": "1" * 64,
                }
            ],
        },
    }


def _stage_output(
    stage: str,
    selected_factors: list[str],
    registry_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if stage == "quant":
        return {
            "schema_version": "factor-governance-quant-stage-output.v1",
            "selected_factors": selected_factors,
            "factor_records": [
                _factor_record(name, registry_records)
                for name in selected_factors
            ],
        }
    if stage == "theme":
        return {
            "schema_version": "factor-governance-theme-stage-output.v1",
            "eligible_symbols": ["AAA"],
        }
    if stage == "bayesian":
        return {
            "schema_version": "factor-governance-bayesian-stage-output.v1",
            "posterior_scores": {"AAA": 0.7},
        }
    if stage == "risk_guard":
        return {
            "schema_version": "factor-governance-risk-stage-output.v1",
            "dates": ["2026-01-05", "2026-01-06", "2026-01-07"],
            "adjusted_returns": {"AAA": [0.01, -0.005, 0.02]},
        }
    return {
        "schema_version": "factor-governance-portfolio-stage-output.v1",
        "dates": ["2026-01-05", "2026-01-06", "2026-01-07"],
        "weights": [{"AAA": 0.5}, {"AAA": 0.5}, {"AAA": 0.5}],
        "costs": [0.001, 0.001, 0.001],
        "after_cost_returns": [0.004, -0.0035, 0.009],
        "turnover": 0.25,
        "slippage": 0.003,
        "tail_risk": 0.0035,
    }


@pytest.fixture()
def replay_fixture(tmp_path: Path) -> dict[str, Any]:
    repo = tmp_path / "repo"
    private = tmp_path / "private"
    inputs = tmp_path / "inputs"
    repo.mkdir(mode=0o755)
    repo.chmod(0o755)
    private.mkdir(mode=0o700)
    inputs.mkdir(mode=0o700)
    stage_root = private / "stages"
    stage_root.mkdir(mode=0o700)
    stage_root.chmod(0o700)

    registry_path = repo / "mined_factors.json"
    registry_records = {
        "incumbent": {
            "name": "incumbent",
            "state": "production_factor",
            "metadata": {
                "factor_family": "value",
                "dominant_primitive_cluster": "primary",
            },
        },
        "challenger": {
            "name": "challenger",
            "state": "mature_candidate",
            "metadata": {
                "factor_family": "value",
                "dominant_primitive_cluster": "primary",
            },
        },
    }
    _write_json(
        registry_path,
        {
            "schema_version": "mined-factor-registry.v1",
            "metadata": {},
            "factors": list(registry_records.values()),
        },
        mode=0o644,
    )

    calendar_path = inputs / "calendar.json"
    calendar_sha = _write_json(
        calendar_path,
        {
            "schema_version": "independent-open-day-calendar.v1",
            "market": "CN",
            "open_days": [
                *[f"2025-{month:02d}-28" for month in range(1, 13)],
                "2026-01-05",
                "2026-01-06",
                "2026-01-07",
            ],
        },
    )
    pit_canonical_path = inputs / "pit-canonical.json"
    pit_canonical_sha = _write_json(
        pit_canonical_path,
        {
            "schema_version": "cn-pit-canonical.v1",
            "as_of": "2026-01-07",
            "symbols": ["AAA"],
        },
    )
    pit_manifest_path = inputs / "pit-manifest.json"
    pit_manifest_sha = _write_json(
        pit_manifest_path,
        {
            "schema_version": "cn-pit-manifest.v1",
            "as_of": "2026-01-07",
            "canonical_path": str(pit_canonical_path),
            "canonical_sha256": pit_canonical_sha,
            "symbols": ["AAA"],
        },
    )
    market_path = inputs / "market.json"
    market_sha = _write_json(
        market_path,
        {
            "schema_version": "factor-governance-market-data.v1",
            "dates": ["2026-01-05", "2026-01-06", "2026-01-07"],
            "returns": {"AAA": [0.01, -0.005, 0.02]},
        },
    )
    snapshot_manifest_path = inputs / "snapshot-manifest.json"
    snapshot_manifest_sha = _write_json(
        snapshot_manifest_path,
        {
            "schema_version": "strict-parquet-snapshot-manifest.v1",
            "snapshot_id": "snapshot-20260107",
            "latest_complete_trade_date": "2026-01-07",
            "calendar_path": str(calendar_path),
            "calendar_sha256": calendar_sha,
            "pit_manifest_path": str(pit_manifest_path),
            "pit_manifest_sha256": pit_manifest_sha,
            "pit_canonical_path": str(pit_canonical_path),
            "pit_canonical_sha256": pit_canonical_sha,
            "market_data_path": str(market_path),
            "market_data_sha256": market_sha,
        },
    )
    snapshot_pointer_path = inputs / "snapshot-pointer.json"
    _write_json(
        snapshot_pointer_path,
        {
            "schema_version": "strict-parquet-snapshot-pointer.v1",
            "snapshot_id": "snapshot-20260107",
            "manifest_path": str(snapshot_manifest_path),
            "manifest_sha256": snapshot_manifest_sha,
        },
    )

    code_entries: list[dict[str, str]] = []
    for role in CODE_CONFIG_ROLES:
        source_path = inputs / "code" / f"{role}.txt"
        source_path.parent.mkdir(mode=0o700, exist_ok=True)
        source_path.write_text(f"{role}\n", encoding="utf-8")
        source_path.chmod(0o600)
        code_entries.append(
            {"role": role, "path": str(source_path), "sha256": _sha(source_path.read_bytes())}
        )
    code_manifest_path = inputs / "code-config-manifest.json"
    _write_json(
        code_manifest_path,
        {
            "schema_version": "factor-governance-code-config-manifest.v1",
            "files": code_entries,
        },
    )

    refs = {
        "registry": _source(registry_path),
        "snapshot_pointer": _source(snapshot_pointer_path),
        "snapshot_manifest": _source(snapshot_manifest_path),
        "calendar": _source(calendar_path),
        "pit_manifest": _source(pit_manifest_path),
        "pit_canonical": _source(pit_canonical_path),
        "market_data": _source(market_path),
        "code_config_manifest": _source(code_manifest_path),
    }
    factor_set = ["incumbent"]
    factor_set_sha = _sha(canonical_json_bytes(factor_set))
    context = {
        "registry_sha256": refs["registry"]["sha256"],
        "factor_set_sha256": factor_set_sha,
        "snapshot_pointer_sha256": refs["snapshot_pointer"]["sha256"],
        "snapshot_manifest_sha256": refs["snapshot_manifest"]["sha256"],
        "calendar_sha256": refs["calendar"]["sha256"],
        "pit_manifest_sha256": refs["pit_manifest"]["sha256"],
        "pit_canonical_sha256": refs["pit_canonical"]["sha256"],
        "market_data_sha256": refs["market_data"]["sha256"],
        "code_config_manifest_sha256": refs["code_config_manifest"]["sha256"],
    }
    arm_factors = {
        "A": ["incumbent"],
        "B": [],
        "C": ["challenger"],
        "D": ["challenger"],
    }
    stage_refs: list[dict[str, str]] = []
    for arm in ARM_NAMES:
        predecessor = {
            "kind": "genesis",
            "byte_sha256": "0" * 64,
            "semantic_sha256": "0" * 64,
        }
        for stage in CONTROL_CHAIN_STAGES:
            output = _stage_output(
                stage,
                arm_factors[arm],
                registry_records,
            )
            stage_payload = {
                "schema_version": "factor-governance-canonical-stage.v1",
                "arm": arm,
                "stage": stage,
                "run_id": "canonical-replay-1",
                "as_of": "2026-01-07",
                "window_start": "2026-01-05",
                "window_end": "2026-01-07",
                "context": context,
                "predecessor": predecessor,
                "output": output,
            }
            stage_path = stage_root / arm / f"{stage}.json"
            stage_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            stage_path.parent.chmod(0o700)
            stage_sha = _write_json(stage_path, stage_payload)
            semantic_sha = _sha(canonical_json_bytes(output))
            stage_refs.append(
                {
                    "arm": arm,
                    "stage": stage,
                    "path": str(stage_path),
                    "sha256": stage_sha,
                    "semantic_sha256": semantic_sha,
                }
            )
            predecessor = {
                "kind": "stage",
                "byte_sha256": stage_sha,
                "semantic_sha256": semantic_sha,
            }

    draft = {
        "schema_version": "factor-governance-canonical-replay-draft.v1",
        "evidence_id": "evidence-20260107-1",
        "run_id": "canonical-replay-1",
        "as_of": "2026-01-07",
        "window_start": "2026-01-05",
        "window_end": "2026-01-07",
        "producer_contract_sha256": producer_contract_sha256(),
        **refs,
        "factor_set": factor_set,
        "comparison": {
            "incumbent": "incumbent",
            "challenger": "challenger",
            "slot": "value::primary",
        },
        "stages": stage_refs,
    }
    draft_path = tmp_path / "draft.json"
    _write_json(draft_path, draft)
    return {
        "private": private,
        "registry": registry_path,
        "draft": draft_path,
        "draft_payload": draft,
        "inputs": inputs,
        "stage_root": stage_root,
    }


def _replace_registry_and_rebind_graph(
    replay_fixture: dict[str, Any], registry_payload: dict[str, Any]
) -> None:
    registry_sha = _write_json(
        replay_fixture["registry"],
        registry_payload,
        mode=0o644,
    )
    draft = replay_fixture["draft_payload"]
    draft["registry"]["sha256"] = registry_sha
    context = {
        "registry_sha256": registry_sha,
        "factor_set_sha256": _sha(canonical_json_bytes(draft["factor_set"])),
        "snapshot_pointer_sha256": draft["snapshot_pointer"]["sha256"],
        "snapshot_manifest_sha256": draft["snapshot_manifest"]["sha256"],
        "calendar_sha256": draft["calendar"]["sha256"],
        "pit_manifest_sha256": draft["pit_manifest"]["sha256"],
        "pit_canonical_sha256": draft["pit_canonical"]["sha256"],
        "market_data_sha256": draft["market_data"]["sha256"],
        "code_config_manifest_sha256": draft["code_config_manifest"]["sha256"],
    }
    for arm in ARM_NAMES:
        predecessor = {
            "kind": "genesis",
            "byte_sha256": "0" * 64,
            "semantic_sha256": "0" * 64,
        }
        for stage in CONTROL_CHAIN_STAGES:
            ref = next(
                item
                for item in draft["stages"]
                if item["arm"] == arm and item["stage"] == stage
            )
            stage_path = Path(ref["path"])
            stage_payload = json.loads(stage_path.read_bytes())
            stage_payload["context"] = context
            stage_payload["predecessor"] = predecessor
            stage_sha = _write_json(stage_path, stage_payload)
            semantic_sha = _sha(canonical_json_bytes(stage_payload["output"]))
            ref["sha256"] = stage_sha
            ref["semantic_sha256"] = semantic_sha
            predecessor = {
                "kind": "stage",
                "byte_sha256": stage_sha,
                "semantic_sha256": semantic_sha,
            }
    _write_json(replay_fixture["draft"], draft)


def _replace_calendar_and_rebind_graph(
    replay_fixture: dict[str, Any],
    open_days: list[str],
    *,
    quant_mutator: Any | None = None,
) -> None:
    draft = replay_fixture["draft_payload"]
    calendar_path = Path(draft["calendar"]["path"])
    calendar_sha = _write_json(
        calendar_path,
        {
            "schema_version": "independent-open-day-calendar.v1",
            "market": "CN",
            "open_days": sorted(set(open_days)),
        },
    )
    draft["calendar"]["sha256"] = calendar_sha

    manifest_path = Path(draft["snapshot_manifest"]["path"])
    manifest = json.loads(manifest_path.read_bytes())
    manifest["calendar_sha256"] = calendar_sha
    manifest_sha = _write_json(manifest_path, manifest)
    draft["snapshot_manifest"]["sha256"] = manifest_sha

    pointer_path = Path(draft["snapshot_pointer"]["path"])
    pointer = json.loads(pointer_path.read_bytes())
    pointer["manifest_sha256"] = manifest_sha
    pointer_sha = _write_json(pointer_path, pointer)
    draft["snapshot_pointer"]["sha256"] = pointer_sha

    context = {
        "registry_sha256": draft["registry"]["sha256"],
        "factor_set_sha256": _sha(canonical_json_bytes(draft["factor_set"])),
        "snapshot_pointer_sha256": pointer_sha,
        "snapshot_manifest_sha256": manifest_sha,
        "calendar_sha256": calendar_sha,
        "pit_manifest_sha256": draft["pit_manifest"]["sha256"],
        "pit_canonical_sha256": draft["pit_canonical"]["sha256"],
        "market_data_sha256": draft["market_data"]["sha256"],
        "code_config_manifest_sha256": draft[
            "code_config_manifest"
        ]["sha256"],
    }
    for arm in ARM_NAMES:
        predecessor = {
            "kind": "genesis",
            "byte_sha256": "0" * 64,
            "semantic_sha256": "0" * 64,
        }
        for stage in CONTROL_CHAIN_STAGES:
            ref = next(
                item
                for item in draft["stages"]
                if item["arm"] == arm and item["stage"] == stage
            )
            stage_path = Path(ref["path"])
            stage_payload = json.loads(stage_path.read_bytes())
            stage_payload["context"] = context
            stage_payload["predecessor"] = predecessor
            if stage == "quant" and quant_mutator is not None:
                for record in stage_payload["output"]["factor_records"]:
                    quant_mutator(record)
            stage_sha = _write_json(stage_path, stage_payload)
            semantic_sha = _sha(canonical_json_bytes(stage_payload["output"]))
            ref["sha256"] = stage_sha
            ref["semantic_sha256"] = semantic_sha
            predecessor = {
                "kind": "stage",
                "byte_sha256": stage_sha,
                "semantic_sha256": semantic_sha,
            }
    _write_json(replay_fixture["draft"], draft)


def test_strict_json_rejects_duplicate_nonfinite_unknown_depth_and_nodes() -> None:
    with pytest.raises(CanonicalReplayError, match="duplicate"):
        strict_json_loads(b'{"a":1,"a":2}', expected_fields={"a"})
    with pytest.raises(CanonicalReplayError, match="non-finite"):
        strict_json_loads(b'{"a":NaN}', expected_fields={"a"})
    with pytest.raises(CanonicalReplayError, match="non-finite"):
        strict_json_loads(b'{"a":1e999}', expected_fields={"a"})
    with pytest.raises(CanonicalReplayError, match="unknown"):
        strict_json_loads(b'{"a":1,"b":2}', expected_fields={"a"})
    with pytest.raises(CanonicalReplayError, match="depth"):
        strict_json_loads(b'{"a":{"b":{"c":1}}}', max_depth=2)
    with pytest.raises(CanonicalReplayError, match="node"):
        strict_json_loads(b'{"a":[1,2,3]}', max_nodes=3)


def test_top_level_date_rejects_iso_week_date_spelling(
    replay_fixture: dict[str, Any],
) -> None:
    replay_fixture["draft_payload"]["window_start"] = "2026-W02-1"
    _write_json(replay_fixture["draft"], replay_fixture["draft_payload"])
    with pytest.raises(CanonicalReplayError, match="exact ISO"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_walk_forward_date_rejects_iso_week_date_after_full_rebind(
    replay_fixture: dict[str, Any],
) -> None:
    calendar = json.loads(
        Path(replay_fixture["draft_payload"]["calendar"]["path"]).read_bytes()
    )

    def mutate(record: dict[str, Any]) -> None:
        record["walk_forward"]["folds"][0]["train_end"] = "2025-W05-2"

    _replace_calendar_and_rebind_graph(
        replay_fixture,
        calendar["open_days"],
        quant_mutator=mutate,
    )
    with pytest.raises(CanonicalReplayError, match="exact ISO"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_safe_reader_rejects_symlink_fifo_hardlink_and_wrong_mode(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    good = private / "good.json"
    good.write_text("{}\n", encoding="utf-8")
    good.chmod(0o600)
    session = SafeReadSession(private)
    assert session.read_bytes(good) == b"{}\n"

    symlink = private / "symlink.json"
    symlink.symlink_to(good)
    with pytest.raises(CanonicalReplayError):
        session.read_bytes(symlink)

    hardlink = private / "hardlink.json"
    os.link(good, hardlink)
    with pytest.raises(CanonicalReplayError, match="link"):
        session.read_bytes(good)
    hardlink.unlink()

    good.chmod(0o640)
    with pytest.raises(CanonicalReplayError, match="mode"):
        session.read_bytes(good)

    fifo = private / "fifo"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(CanonicalReplayError, match="regular"):
        session.read_bytes(fifo)


def test_safe_reader_rejects_symlink_ancestor_and_size_budget(tmp_path: Path) -> None:
    private = tmp_path / "private"
    real = tmp_path / "real"
    private.mkdir(mode=0o700)
    real.mkdir(mode=0o700)
    target = real / "value"
    target.write_bytes(b"12345")
    target.chmod(0o600)
    (private / "linked").symlink_to(real, target_is_directory=True)
    session = SafeReadSession(private, max_file_bytes=4, max_total_bytes=8)
    with pytest.raises(CanonicalReplayError):
        session.read_bytes(private / "linked" / "value")
    direct = private / "direct"
    direct.write_bytes(b"12345")
    direct.chmod(0o600)
    with pytest.raises(CanonicalReplayError, match="size"):
        session.read_bytes(direct)


def test_safe_reader_rejects_ancestor_directory_rename_and_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    sub = private / "sub"
    private.mkdir(mode=0o700)
    sub.mkdir(mode=0o700)
    value = sub / "value"
    value.write_bytes(b"OLD")
    value.chmod(0o600)
    session = SafeReadSession(private)
    original_open_leaf = session._open_leaf

    def replace_after_open(path: str) -> tuple[Any, ...]:
        opened = original_open_leaf(path)
        sub.rename(private / "moved")
        sub.mkdir(mode=0o700)
        replacement = sub / "value"
        replacement.write_bytes(b"NEW")
        replacement.chmod(0o600)
        return opened

    monkeypatch.setattr(session, "_open_leaf", replace_after_open)
    with pytest.raises(CanonicalReplayError, match="directory|ancestor|path"):
        session.read_bytes(value)


def test_safe_reader_accepts_owned_external_0644_but_not_external_writable(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    repo = tmp_path / "repo"
    private.mkdir(mode=0o700)
    repo.mkdir(mode=0o755)
    external = repo / "registry.json"
    external.write_bytes(b"{}\n")
    external.chmod(0o644)
    session = SafeReadSession(private)
    assert session.read_bytes(external) == b"{}\n"
    external.chmod(0o664)
    with pytest.raises(CanonicalReplayError, match="group/world writable"):
        session.read_bytes(external)


@pytest.mark.parametrize("bad_path", [True, 7])
def test_registry_source_and_stage_paths_reject_non_strings(
    replay_fixture: dict[str, Any], bad_path: Any
) -> None:
    with pytest.raises(CanonicalReplayError, match="path"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=bad_path,
            draft_path=replay_fixture["draft"],
        )

    draft = replay_fixture["draft_payload"]
    draft["calendar"]["path"] = bad_path
    _write_json(replay_fixture["draft"], draft)
    with pytest.raises(CanonicalReplayError, match="path"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )

    draft["calendar"]["path"] = str(replay_fixture["inputs"] / "calendar.json")
    draft["stages"][0]["path"] = bad_path
    _write_json(replay_fixture["draft"], draft)
    with pytest.raises(CanonicalReplayError, match="path"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_cli_non_string_source_path_is_blocked_without_traceback(
    replay_fixture: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    draft = replay_fixture["draft_payload"]
    draft["calendar"]["path"] = True
    _write_json(replay_fixture["draft"], draft)
    assert replay_main(
        [
            "--private-root",
            str(replay_fixture["private"]),
            "--registry-path",
            str(replay_fixture["registry"]),
            "--draft-path",
            str(replay_fixture["draft"]),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_constructible_replay_uses_exact_receipt_and_discloses_no_authority(
    replay_fixture: dict[str, Any],
) -> None:
    assert replay_fixture["registry"].parent.stat().st_mode & 0o777 == 0o755
    assert replay_fixture["registry"].stat().st_mode & 0o777 == 0o644
    result = produce_canonical_replay(
        private_root=replay_fixture["private"],
        registry_path=replay_fixture["registry"],
        draft_path=replay_fixture["draft"],
    )
    assert result["producer_implemented"] is True
    assert result["local_bytes_readback_verified"] is True
    assert result["canonical_producer_authenticated"] is False
    assert result["production_apply_authorized"] is False
    assert result["production_apply_eligible"] is False
    assert "path" not in result

    verified = verify_canonical_replay(
        private_root=replay_fixture["private"],
        registry_path=replay_fixture["registry"],
    )
    assert verified == result
    receipt = replay_fixture["private"] / "receipts" / f'{result["registry_sha256"]}.json'
    bundle = replay_fixture["private"] / "bundles" / "evidence-20260107-1.json"
    assert receipt.is_file() and bundle.is_file()
    assert receipt.stat().st_mode & 0o777 == 0o600
    assert bundle.stat().st_mode & 0o777 == 0o600
    assert json.loads(bundle.read_bytes())["schema_version"] == BUNDLE_SCHEMA_VERSION


def test_producer_contract_binds_registry_identity_algorithm_and_states() -> None:
    identity = canonical_module.canonical_replay_producer_contract()[
        "registry_record_identity"
    ]
    assert identity["algorithm"] == (
        "sha256_of_canonical_compact_sorted_json_registry_record"
    )
    assert identity["allowed_challenger_states"] == [
        "mature_candidate",
        "production_candidate",
        "shadow",
    ]
    assert identity["production_incumbent_state"] == "production_factor"
    assert identity["family_fallback_fields"] == [
        "metadata.factor_family",
        "metadata.governance_family",
        "category",
    ]
    assert identity["cluster_fallback_fields"] == [
        "metadata.dominant_primitive_cluster",
        "metadata.dominant_primitives_sorted_join_plus",
    ]
    assert identity["slot_format"] == (
        "{family}::{dominant_primitive_cluster}"
    )


def test_producer_contract_binds_temporal_and_crash_recovery_rules() -> None:
    contract = canonical_module.canonical_replay_producer_contract()
    temporal = contract["temporal_evidence"]
    assert temporal["cutoff"] == "window_end"
    assert temporal["single_pit_snapshot_rule"] == "window_end_equals_as_of"
    assert temporal["calendar_membership_required"] == [
        "window_start",
        "window_end",
        "as_of",
    ]
    publish = contract["publish"]
    assert publish["recovery_selection"].endswith("no_scan")
    assert publish["exact_verification_required_after_ambiguous_failure"] is True
    assert "file0600_and_directory0700" in publish[
        "supported_umask_preflight"
    ]
    assert publish["unsupported_umask"].startswith("reject_before")
    assert "never_unlink_final" in publish["crash_recovery"]


def test_registry_byte_drift_selects_no_fallback_receipt(
    replay_fixture: dict[str, Any],
) -> None:
    produce_canonical_replay(
        private_root=replay_fixture["private"],
        registry_path=replay_fixture["registry"],
        draft_path=replay_fixture["draft"],
    )
    replay_fixture["registry"].write_bytes(replay_fixture["registry"].read_bytes() + b" ")
    with pytest.raises(CanonicalReplayError, match="receipt"):
        verify_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
        )


def test_source_byte_drift_and_bundle_byte_drift_fail_closed(
    replay_fixture: dict[str, Any],
) -> None:
    result = produce_canonical_replay(
        private_root=replay_fixture["private"],
        registry_path=replay_fixture["registry"],
        draft_path=replay_fixture["draft"],
    )
    stage = replay_fixture["stage_root"] / "A" / "quant.json"
    stage.write_bytes(stage.read_bytes() + b" ")
    with pytest.raises(CanonicalReplayError, match="SHA"):
        verify_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
        )
    stage.write_bytes(stage.read_bytes()[:-1])
    bundle = replay_fixture["private"] / "bundles" / "evidence-20260107-1.json"
    bundle.write_bytes(bundle.read_bytes() + b" ")
    with pytest.raises(CanonicalReplayError, match="bundle"):
        verify_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
        )
    assert result["production_apply_eligible"] is False


def test_verify_rejects_noncanonical_receipt_bytes(
    replay_fixture: dict[str, Any],
) -> None:
    result = produce_canonical_replay(
        private_root=replay_fixture["private"],
        registry_path=replay_fixture["registry"],
        draft_path=replay_fixture["draft"],
    )
    receipt = (
        replay_fixture["private"]
        / "receipts"
        / f'{result["registry_sha256"]}.json'
    )
    payload = json.loads(receipt.read_bytes())
    receipt.write_bytes(
        json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    )
    with pytest.raises(CanonicalReplayError, match="canonical receipt bytes"):
        verify_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
        )


def test_verify_rejects_noncanonical_bundle_even_with_rebound_receipt_sha(
    replay_fixture: dict[str, Any],
) -> None:
    result = produce_canonical_replay(
        private_root=replay_fixture["private"],
        registry_path=replay_fixture["registry"],
        draft_path=replay_fixture["draft"],
    )
    bundle = (
        replay_fixture["private"]
        / "bundles"
        / "evidence-20260107-1.json"
    )
    bundle_payload = json.loads(bundle.read_bytes())
    rebound_bundle = (
        json.dumps(bundle_payload, indent=2, ensure_ascii=False).encode("utf-8")
        + b"\n"
    )
    bundle.write_bytes(rebound_bundle)
    receipt = (
        replay_fixture["private"]
        / "receipts"
        / f'{result["registry_sha256"]}.json'
    )
    receipt_payload = json.loads(receipt.read_bytes())
    receipt_payload["bundle_sha256"] = _sha(rebound_bundle)
    receipt.write_bytes(_json_bytes(receipt_payload))
    with pytest.raises(CanonicalReplayError, match="bundle bytes"):
        verify_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
        )


def test_stage_graph_requires_exactly_20_unique_paths_and_predecessors(
    replay_fixture: dict[str, Any],
) -> None:
    payload = replay_fixture["draft_payload"]
    payload["stages"][1]["path"] = payload["stages"][0]["path"]
    payload["stages"][1]["sha256"] = payload["stages"][0]["sha256"]
    payload["stages"][1]["semantic_sha256"] = payload["stages"][0]["semantic_sha256"]
    _write_json(replay_fixture["draft"], payload)
    with pytest.raises(CanonicalReplayError, match="unique"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_stage_graph_rejects_predecessor_and_context_laundering(
    replay_fixture: dict[str, Any],
) -> None:
    stage = replay_fixture["stage_root"] / "A" / "theme.json"
    payload = json.loads(stage.read_bytes())
    payload["predecessor"]["byte_sha256"] = "f" * 64
    new_sha = _write_json(stage, payload)
    for ref in replay_fixture["draft_payload"]["stages"]:
        if ref["arm"] == "A" and ref["stage"] == "theme":
            ref["sha256"] = new_sha
    _write_json(replay_fixture["draft"], replay_fixture["draft_payload"])
    with pytest.raises(CanonicalReplayError, match="predecessor"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_market_data_requires_every_independent_open_day_in_window(
    replay_fixture: dict[str, Any],
) -> None:
    draft = replay_fixture["draft_payload"]
    market_path = Path(draft["market_data"]["path"])
    market = json.loads(market_path.read_bytes())
    market["dates"] = ["2026-01-05", "2026-01-07"]
    market["returns"]["AAA"] = [0.01, 0.02]
    market_sha = _write_json(market_path, market)
    draft["market_data"]["sha256"] = market_sha

    manifest_path = Path(draft["snapshot_manifest"]["path"])
    manifest = json.loads(manifest_path.read_bytes())
    manifest["market_data_sha256"] = market_sha
    manifest_sha = _write_json(manifest_path, manifest)
    draft["snapshot_manifest"]["sha256"] = manifest_sha

    pointer_path = Path(draft["snapshot_pointer"]["path"])
    pointer = json.loads(pointer_path.read_bytes())
    pointer["manifest_sha256"] = manifest_sha
    pointer_sha = _write_json(pointer_path, pointer)
    draft["snapshot_pointer"]["sha256"] = pointer_sha

    context = {
        "registry_sha256": draft["registry"]["sha256"],
        "factor_set_sha256": _sha(canonical_json_bytes(draft["factor_set"])),
        "snapshot_pointer_sha256": pointer_sha,
        "snapshot_manifest_sha256": manifest_sha,
        "calendar_sha256": draft["calendar"]["sha256"],
        "pit_manifest_sha256": draft["pit_manifest"]["sha256"],
        "pit_canonical_sha256": draft["pit_canonical"]["sha256"],
        "market_data_sha256": market_sha,
        "code_config_manifest_sha256": draft["code_config_manifest"]["sha256"],
    }
    for arm in ARM_NAMES:
        predecessor = {
            "kind": "genesis",
            "byte_sha256": "0" * 64,
            "semantic_sha256": "0" * 64,
        }
        for stage in CONTROL_CHAIN_STAGES:
            ref = next(
                item
                for item in draft["stages"]
                if item["arm"] == arm and item["stage"] == stage
            )
            stage_path = Path(ref["path"])
            stage_payload = json.loads(stage_path.read_bytes())
            stage_payload["context"] = context
            stage_payload["predecessor"] = predecessor
            if stage == "risk_guard":
                stage_payload["output"]["dates"] = [
                    "2026-01-05",
                    "2026-01-07",
                ]
                stage_payload["output"]["adjusted_returns"]["AAA"] = [
                    0.01,
                    0.02,
                ]
            elif stage == "portfolio_constructor":
                stage_payload["output"].update(
                    {
                        "dates": ["2026-01-05", "2026-01-07"],
                        "weights": [{"AAA": 0.5}, {"AAA": 0.5}],
                        "costs": [0.001, 0.001],
                        "after_cost_returns": [0.004, 0.009],
                        "turnover": 0.25,
                        "slippage": 0.002,
                        "tail_risk": 0.0,
                    }
                )
            stage_sha = _write_json(stage_path, stage_payload)
            semantic_sha = _sha(canonical_json_bytes(stage_payload["output"]))
            ref["sha256"] = stage_sha
            ref["semantic_sha256"] = semantic_sha
            predecessor = {
                "kind": "stage",
                "byte_sha256": stage_sha,
                "semantic_sha256": semantic_sha,
            }
    _write_json(replay_fixture["draft"], draft)
    with pytest.raises(CanonicalReplayError, match="open-day|complete|exact"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_calendar_must_include_window_start_end_and_as_of(
    replay_fixture: dict[str, Any],
) -> None:
    calendar = json.loads(
        Path(replay_fixture["draft_payload"]["calendar"]["path"]).read_bytes()
    )
    open_days = [
        item for item in calendar["open_days"] if item != "2026-01-07"
    ]
    _replace_calendar_and_rebind_graph(replay_fixture, open_days)
    with pytest.raises(CanonicalReplayError, match="window_end|as_of|cut point"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


@pytest.mark.parametrize("attack", ["month_end", "cohort", "walk_forward"])
def test_future_quant_evidence_is_rejected_after_full_graph_rebind(
    replay_fixture: dict[str, Any], attack: str
) -> None:
    calendar_path = Path(
        replay_fixture["draft_payload"]["calendar"]["path"]
    )
    calendar = json.loads(calendar_path.read_bytes())
    if attack == "month_end":
        future_days = [
            *[f"2026-{month:02d}-28" for month in range(2, 13)],
            "2027-01-28",
        ]
    elif attack == "cohort":
        future_days = [
            *[f"2026-02-{day:02d}" for day in range(1, 29)],
            "2026-03-01",
            "2026-03-02",
        ]
    else:
        future_days = ["2026-02-01", "2026-03-03", "2026-04-03"]

    def mutate(record: dict[str, Any]) -> None:
        if attack == "month_end":
            record["month_end_rankic_dates"] = future_days
        elif attack == "cohort":
            record["forward_cohorts"] = [
                {
                    "cohort_id": "future-cohort",
                    "start": future_days[0],
                    "end": future_days[-1],
                    "horizon_days": 30,
                }
            ]
        else:
            record["walk_forward"]["folds"] = [
                {
                    "train_end": future_days[0],
                    "validation_start": future_days[1],
                    "validation_end": future_days[2],
                    "evidence_hash": "2" * 64,
                }
            ]

    _replace_calendar_and_rebind_graph(
        replay_fixture,
        [*calendar["open_days"], *future_days],
        quant_mutator=mutate,
    )
    with pytest.raises(CanonicalReplayError, match="cutoff|after"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_later_as_of_pit_snapshot_cannot_launder_earlier_replay_window(
    replay_fixture: dict[str, Any],
) -> None:
    draft = replay_fixture["draft_payload"]
    later_as_of = "2026-01-08"

    calendar_path = Path(draft["calendar"]["path"])
    calendar = json.loads(calendar_path.read_bytes())
    calendar["open_days"].append(later_as_of)
    calendar_sha = _write_json(calendar_path, calendar)
    draft["calendar"]["sha256"] = calendar_sha

    pit_canonical_path = Path(draft["pit_canonical"]["path"])
    pit_canonical = json.loads(pit_canonical_path.read_bytes())
    pit_canonical["as_of"] = later_as_of
    pit_canonical_sha = _write_json(pit_canonical_path, pit_canonical)
    draft["pit_canonical"]["sha256"] = pit_canonical_sha

    pit_manifest_path = Path(draft["pit_manifest"]["path"])
    pit_manifest = json.loads(pit_manifest_path.read_bytes())
    pit_manifest["as_of"] = later_as_of
    pit_manifest["canonical_sha256"] = pit_canonical_sha
    pit_manifest_sha = _write_json(pit_manifest_path, pit_manifest)
    draft["pit_manifest"]["sha256"] = pit_manifest_sha

    snapshot_manifest_path = Path(draft["snapshot_manifest"]["path"])
    snapshot_manifest = json.loads(snapshot_manifest_path.read_bytes())
    snapshot_manifest["latest_complete_trade_date"] = later_as_of
    snapshot_manifest["calendar_sha256"] = calendar_sha
    snapshot_manifest["pit_manifest_sha256"] = pit_manifest_sha
    snapshot_manifest["pit_canonical_sha256"] = pit_canonical_sha
    snapshot_manifest_sha = _write_json(
        snapshot_manifest_path, snapshot_manifest
    )
    draft["snapshot_manifest"]["sha256"] = snapshot_manifest_sha

    pointer_path = Path(draft["snapshot_pointer"]["path"])
    pointer = json.loads(pointer_path.read_bytes())
    pointer["manifest_sha256"] = snapshot_manifest_sha
    pointer_sha = _write_json(pointer_path, pointer)
    draft["snapshot_pointer"]["sha256"] = pointer_sha
    draft["as_of"] = later_as_of

    context = {
        "registry_sha256": draft["registry"]["sha256"],
        "factor_set_sha256": _sha(canonical_json_bytes(draft["factor_set"])),
        "snapshot_pointer_sha256": pointer_sha,
        "snapshot_manifest_sha256": snapshot_manifest_sha,
        "calendar_sha256": calendar_sha,
        "pit_manifest_sha256": pit_manifest_sha,
        "pit_canonical_sha256": pit_canonical_sha,
        "market_data_sha256": draft["market_data"]["sha256"],
        "code_config_manifest_sha256": draft[
            "code_config_manifest"
        ]["sha256"],
    }
    for arm in ARM_NAMES:
        predecessor = {
            "kind": "genesis",
            "byte_sha256": "0" * 64,
            "semantic_sha256": "0" * 64,
        }
        for stage in CONTROL_CHAIN_STAGES:
            ref = next(
                item
                for item in draft["stages"]
                if item["arm"] == arm and item["stage"] == stage
            )
            stage_path = Path(ref["path"])
            stage_payload = json.loads(stage_path.read_bytes())
            stage_payload["as_of"] = later_as_of
            stage_payload["context"] = context
            stage_payload["predecessor"] = predecessor
            stage_sha = _write_json(stage_path, stage_payload)
            semantic_sha = _sha(canonical_json_bytes(stage_payload["output"]))
            ref["sha256"] = stage_sha
            ref["semantic_sha256"] = semantic_sha
            predecessor = {
                "kind": "stage",
                "byte_sha256": stage_sha,
                "semantic_sha256": semantic_sha,
            }
    _write_json(replay_fixture["draft"], draft)
    with pytest.raises(CanonicalReplayError, match="chronology|window"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_abcd_structure_requires_one_slot_replacement(
    replay_fixture: dict[str, Any],
) -> None:
    replay_fixture["draft_payload"]["comparison"]["challenger"] = "incumbent"
    _write_json(replay_fixture["draft"], replay_fixture["draft_payload"])
    with pytest.raises(CanonicalReplayError, match="one-slot"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_challenger_must_exist_in_complete_registry(
    replay_fixture: dict[str, Any],
) -> None:
    registry = json.loads(replay_fixture["registry"].read_bytes())
    registry["factors"] = [
        item for item in registry["factors"] if item["name"] != "challenger"
    ]
    _replace_registry_and_rebind_graph(replay_fixture, registry)
    with pytest.raises(CanonicalReplayError, match="challenger|registry"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_duplicate_registry_factor_name_fails_closed(
    replay_fixture: dict[str, Any],
) -> None:
    registry = json.loads(replay_fixture["registry"].read_bytes())
    registry["factors"].append(dict(registry["factors"][0]))
    _replace_registry_and_rebind_graph(replay_fixture, registry)
    with pytest.raises(CanonicalReplayError, match="distinct|duplicate"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


@pytest.mark.parametrize("substitution", ["invalid_state", "raw_record"])
def test_challenger_registry_state_or_record_substitution_fails_closed(
    replay_fixture: dict[str, Any], substitution: str
) -> None:
    registry = json.loads(replay_fixture["registry"].read_bytes())
    challenger = next(
        item for item in registry["factors"] if item["name"] == "challenger"
    )
    if substitution == "invalid_state":
        challenger["state"] = "deprecated"
    else:
        challenger["version"] = "substituted-registry-record"
    _replace_registry_and_rebind_graph(replay_fixture, registry)
    with pytest.raises(CanonicalReplayError, match="challenger|record|state"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_stage_family_and_slot_must_match_registry_identity(
    replay_fixture: dict[str, Any],
) -> None:
    draft = replay_fixture["draft_payload"]
    draft["comparison"]["slot"] = "forged::cluster"
    for arm in ARM_NAMES:
        predecessor = {
            "kind": "genesis",
            "byte_sha256": "0" * 64,
            "semantic_sha256": "0" * 64,
        }
        for stage in CONTROL_CHAIN_STAGES:
            ref = next(
                item
                for item in draft["stages"]
                if item["arm"] == arm and item["stage"] == stage
            )
            stage_path = Path(ref["path"])
            stage_payload = json.loads(stage_path.read_bytes())
            stage_payload["predecessor"] = predecessor
            if stage == "quant":
                for record in stage_payload["output"]["factor_records"]:
                    record["family"] = "forged"
                    record["slot"] = "forged::cluster"
            stage_sha = _write_json(stage_path, stage_payload)
            semantic_sha = _sha(canonical_json_bytes(stage_payload["output"]))
            ref["sha256"] = stage_sha
            ref["semantic_sha256"] = semantic_sha
            predecessor = {
                "kind": "stage",
                "byte_sha256": stage_sha,
                "semantic_sha256": semantic_sha,
            }
    _write_json(replay_fixture["draft"], draft)
    with pytest.raises(CanonicalReplayError, match="family|slot|registry"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("family", True),
        ("cluster", 7),
        ("dominant", ["same", "same"]),
        ("missing", None),
    ],
)
def test_selected_registry_slot_identity_must_be_exact_and_complete(
    replay_fixture: dict[str, Any], mutation: str, value: Any
) -> None:
    registry = json.loads(replay_fixture["registry"].read_bytes())
    incumbent = next(
        item for item in registry["factors"] if item["name"] == "incumbent"
    )
    if mutation == "family":
        incumbent["metadata"]["factor_family"] = value
    elif mutation == "cluster":
        incumbent["metadata"]["dominant_primitive_cluster"] = value
    elif mutation == "dominant":
        incumbent["metadata"].pop("dominant_primitive_cluster")
        incumbent["metadata"]["dominant_primitives"] = value
    else:
        incumbent["metadata"] = {}
        incumbent.pop("category", None)
    _replace_registry_and_rebind_graph(replay_fixture, registry)
    with pytest.raises(CanonicalReplayError, match="registry|family|cluster|slot"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_recomputes_weights_returns_cost_turnover_slippage_and_tail_risk(
    replay_fixture: dict[str, Any],
) -> None:
    stage = replay_fixture["stage_root"] / "D" / "portfolio_constructor.json"
    payload = json.loads(stage.read_bytes())
    payload["output"]["after_cost_returns"][0] = 0.5
    new_sha = _write_json(stage, payload)
    semantic_sha = _sha(canonical_json_bytes(payload["output"]))
    for ref in replay_fixture["draft_payload"]["stages"]:
        if ref["arm"] == "D" and ref["stage"] == "portfolio_constructor":
            ref["sha256"] = new_sha
            ref["semantic_sha256"] = semantic_sha
    _write_json(replay_fixture["draft"], replay_fixture["draft_payload"])
    with pytest.raises(CanonicalReplayError, match="after-cost"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_immutable_publish_is_idempotent_no_clobber_and_rejects_nlink2(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    first = publish_immutable_json(private, "bundles/value.json", {"value": 1})
    again = publish_immutable_json(private, "bundles/value.json", {"value": 1})
    assert first == again
    with pytest.raises(CanonicalReplayError, match="different bytes"):
        publish_immutable_json(private, "bundles/value.json", {"value": 2})
    destination = private / "bundles" / "value.json"
    crash_link = private / "bundles" / "crash-temp"
    os.link(destination, crash_link)
    with pytest.raises(CanonicalReplayError, match="link"):
        SafeReadSession(private).read_bytes(destination)


def test_immutable_publish_retains_recoverable_temp_after_pre_link_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    bundle_dir = private / "bundles"
    bundle_dir.mkdir(mode=0o700)
    real_fsync = os.fsync
    failed = False

    def fail_first_fsync(fd: int) -> None:
        nonlocal failed
        value = os.fstat(fd)
        if stat.S_ISREG(value.st_mode) and value.st_size > 0 and not failed:
            failed = True
            raise OSError("injected fsync failure")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="injected"):
        publish_immutable_json(private, "bundles/value.json", {"value": 1})
    lock, temp = _publish_reserved_paths(bundle_dir, "value.json", {"value": 1})
    assert failed is True
    assert lock.is_file()
    assert temp.read_bytes() == b'{"value":1}\n'
    assert temp.stat().st_nlink == 1
    assert not (bundle_dir / "value.json").exists()
    monkeypatch.setattr(os, "fsync", real_fsync)
    assert publish_immutable_json(
        private, "bundles/value.json", {"value": 1}
    )["sha256"] == _sha(b'{"value":1}\n')
    assert not temp.exists()


def test_immutable_publish_rejects_oversize_before_destination_or_temp(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    payload = {"value": "x" * (16 * 1024 * 1024 + 1)}
    with pytest.raises(CanonicalReplayError, match="size|byte"):
        publish_immutable_json(private, "bundles/oversize.json", payload)
    bundle_dir = private / "bundles"
    assert not bundle_dir.exists() or list(bundle_dir.iterdir()) == []


def test_immutable_publish_wraps_recursive_payload_before_destination(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    nested: list[Any] = []
    for _ in range(2_000):
        nested = [nested]

    with pytest.raises(CanonicalReplayError, match="canonical JSON"):
        publish_immutable_json(
            private,
            "bundles/recursive.json",
            {"value": nested},
        )

    assert not (private / "bundles" / "recursive.json").exists()


def test_publish_rejects_unsupported_umask_before_named_namespace_then_retries(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    previous_umask = os.umask(0o777)
    try:
        with pytest.raises(CanonicalReplayError, match="umask"):
            publish_immutable_json(
                private, "bundles/value.json", {"value": 1}
            )
    finally:
        os.umask(previous_umask)
    assert not (private / "bundles").exists()
    assert list(private.iterdir()) == []

    first = publish_immutable_json(
        private, "bundles/value.json", {"value": 1}
    )
    bundles = private / "bundles"
    destination = bundles / "value.json"
    lock, temp = _publish_reserved_paths(bundles, "value.json", {"value": 1})
    assert bundles.stat().st_mode & 0o777 == 0o700
    assert lock.stat().st_mode & 0o777 == 0o600
    assert destination.stat().st_mode & 0o777 == 0o600
    assert not temp.exists()
    assert publish_immutable_json(
        private, "bundles/value.json", {"value": 1}
    ) == first


def test_persistent_pre_link_temp_fsync_failure_never_creates_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    bundles = private / "bundles"
    bundles.mkdir(mode=0o700)
    _, temp = _publish_reserved_paths(bundles, "value.json", {"value": 1})
    temp.write_bytes(b'{"value":1}\n')
    temp.chmod(0o600)
    temp_identity = (temp.stat().st_dev, temp.stat().st_ino)
    real_fsync = os.fsync
    link_called = False
    real_link = os.link

    def fail_temp_fsync(fd: int) -> None:
        value = os.fstat(fd)
        if (value.st_dev, value.st_ino) == temp_identity:
            raise OSError("persistent prepared-temp fsync fault")
        real_fsync(fd)

    def observe_link(*args: Any, **kwargs: Any) -> None:
        nonlocal link_called
        link_called = True
        real_link(*args, **kwargs)

    monkeypatch.setattr(os, "fsync", fail_temp_fsync)
    monkeypatch.setattr(os, "link", observe_link)
    with pytest.raises(OSError, match="prepared-temp"):
        publish_immutable_json(private, "bundles/value.json", {"value": 1})
    assert link_called is False
    assert not (bundles / "value.json").exists()
    assert temp.stat().st_nlink == 1

    monkeypatch.setattr(os, "fsync", real_fsync)
    monkeypatch.setattr(os, "link", real_link)
    publish_immutable_json(private, "bundles/value.json", {"value": 1})
    assert (bundles / "value.json").stat().st_nlink == 1
    assert not temp.exists()


def test_publish_parent_mode_failure_does_not_leak_child_fds(tmp_path: Path) -> None:
    fd_root = Path("/dev/fd")
    if not fd_root.is_dir():
        pytest.skip("platform does not expose /dev/fd")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    bad_child = private / "bundles"
    bad_child.mkdir(mode=0o755)
    before = len(list(fd_root.iterdir()))
    for _ in range(25):
        with pytest.raises(CanonicalReplayError, match="mode"):
            publish_immutable_json(private, "bundles/value.json", {"value": 1})
    after = len(list(fd_root.iterdir()))
    assert after == before


def test_publish_parent_rename_attack_leaves_no_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    bundles = private / "bundles"
    bundles.mkdir(mode=0o700)
    moved = private / "moved"
    original_open_parent = canonical_module._open_publish_parent

    def rename_after_open(
        private_root: str | os.PathLike[str], relative_path: str
    ) -> tuple[Any, ...]:
        opened = original_open_parent(private_root, relative_path)
        bundles.rename(moved)
        bundles.mkdir(mode=0o700)
        return opened

    monkeypatch.setattr(
        canonical_module,
        "_open_publish_parent",
        rename_after_open,
    )
    with pytest.raises(CanonicalReplayError, match="directory|ancestor|path|publish"):
        publish_immutable_json(private, "bundles/value.json", {"value": 1})
    assert list(bundles.iterdir()) == []
    moved_entries = list(moved.iterdir())
    assert all(".tmp." not in item.name for item in moved_entries)
    if moved_entries:
        assert [item.name for item in moved_entries] == ["value.json"]
        orphan = moved_entries[0]
        assert orphan.read_bytes() == b'{"value":1}\n'
        assert orphan.stat().st_mode & 0o777 == 0o600
        assert orphan.stat().st_nlink == 1


def test_publish_fault_cleanup_never_unlinks_unrelated_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    bundles = private / "bundles"
    bundles.mkdir(mode=0o700)
    leaf = bundles / "value.json"
    real_assert_chain = SafeReadSession._assert_directory_chain
    real_stat = os.stat
    replaced = False

    def replace_after_link(
        self: SafeReadSession,
        path: str,
        expected: tuple[Any, ...],
        *,
        private: bool,
    ) -> None:
        nonlocal replaced
        if leaf.exists() and not replaced:
            replaced = True
            published_stat = real_stat(leaf)
            leaf.unlink()
            leaf.write_bytes(b"UNRELATED")
            leaf.chmod(0o600)

            def stale_leaf_stat(
                path_value: Any, *args: Any, **kwargs: Any
            ) -> os.stat_result:
                if path_value == "value.json" and kwargs.get("dir_fd") is not None:
                    return published_stat
                return real_stat(path_value, *args, **kwargs)

            monkeypatch.setattr(os, "stat", stale_leaf_stat)
            raise CanonicalReplayError("injected post-link identity fault")
        real_assert_chain(
            self,
            path,
            expected,
            private=private,
        )

    monkeypatch.setattr(
        SafeReadSession,
        "_assert_directory_chain",
        replace_after_link,
    )
    with pytest.raises(CanonicalReplayError, match="injected"):
        publish_immutable_json(private, "bundles/value.json", {"value": 1})
    assert leaf.read_bytes() == b"UNRELATED"
    assert leaf.stat().st_mode & 0o777 == 0o600


def test_post_link_one_shot_fault_reconciles_exact_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    bundles = private / "bundles"
    bundles.mkdir(mode=0o700)
    real_fsync = os.fsync
    real_link = os.link
    linked = False
    failed = False

    def observe_link(*args: Any, **kwargs: Any) -> None:
        nonlocal linked
        real_link(*args, **kwargs)
        linked = True

    def fail_once_after_link(fd: int) -> None:
        nonlocal failed
        if linked and not failed:
            failed = True
            raise OSError("injected one-shot post-link fault")
        real_fsync(fd)

    monkeypatch.setattr(os, "link", observe_link)
    monkeypatch.setattr(os, "fsync", fail_once_after_link)
    result = publish_immutable_json(
        private, "bundles/value.json", {"value": 1}
    )
    destination = bundles / "value.json"
    _, temp = _publish_reserved_paths(bundles, "value.json", {"value": 1})
    assert failed is True
    assert result["sha256"] == _sha(b'{"value":1}\n')
    assert destination.read_bytes() == b'{"value":1}\n'
    assert destination.stat().st_nlink == 1
    assert not temp.exists()


@pytest.mark.parametrize("crash_point", ["prepared", "linked"])
def test_subprocess_crash_state_is_recovered_without_scan(
    tmp_path: Path, crash_point: str
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    bundles = private / "bundles"
    bundles.mkdir(mode=0o700)
    script = """
import os
import stat
import sys
from pathlib import Path
from quant_investor.factors.governance_canonical_replay import publish_immutable_json

private = Path(sys.argv[1])
crash_point = sys.argv[2]
if crash_point == "prepared":
    real_fsync = os.fsync
    def crash_after_temp_fsync(fd):
        real_fsync(fd)
        value = os.fstat(fd)
        if stat.S_ISREG(value.st_mode) and value.st_size > 0:
            os._exit(78)
    os.fsync = crash_after_temp_fsync
else:
    real_link = os.link
    def crash_after_link(*args, **kwargs):
        real_link(*args, **kwargs)
        os._exit(77)
    os.link = crash_after_link
publish_immutable_json(private, "bundles/value.json", {"value": 1})
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(private), crash_point],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == (78 if crash_point == "prepared" else 77)
    destination = bundles / "value.json"
    _, temp = _publish_reserved_paths(bundles, "value.json", {"value": 1})
    if crash_point == "prepared":
        assert not destination.exists()
        assert temp.stat().st_nlink == 1
    else:
        assert destination.stat().st_nlink == 2
        assert temp.stat().st_nlink == 2
        assert os.path.samefile(destination, temp)

    publish_immutable_json(private, "bundles/value.json", {"value": 1})
    assert destination.read_bytes() == b'{"value":1}\n'
    assert destination.stat().st_nlink == 1
    assert not temp.exists()


def test_normal_umask_lock_crash_immediately_after_exact_create_is_retryable(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    bundles = private / "bundles"
    bundles.mkdir(mode=0o700)
    script = """
import os
import sys
from pathlib import Path
from quant_investor.factors.governance_canonical_replay import publish_immutable_json
private = Path(sys.argv[1])
real_open = os.open
def crash_after_exact_lock_create(path, flags, mode=0o777, *, dir_fd=None):
    fd = real_open(path, flags, mode, dir_fd=dir_fd)
    if isinstance(path, str) and path.endswith(".lock") and flags & os.O_EXCL:
        os._exit(79)
    return fd
os.open = crash_after_exact_lock_create
publish_immutable_json(private, "bundles/value.json", {"value": 1})
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(private)],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        timeout=10,
    )
    assert completed.returncode == 79
    lock, temp = _publish_reserved_paths(bundles, "value.json", {"value": 1})
    assert lock.is_file()
    assert lock.stat().st_mode & 0o777 == 0o600
    assert lock.stat().st_nlink == 1
    assert not temp.exists()
    assert not (bundles / "value.json").exists()

    publish_immutable_json(private, "bundles/value.json", {"value": 1})
    destination = bundles / "value.json"
    assert destination.read_bytes() == b'{"value":1}\n'
    assert destination.stat().st_nlink == 1


def test_different_byte_final_is_never_clobbered_during_temp_recovery(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    bundles = private / "bundles"
    bundles.mkdir(mode=0o700)
    destination = bundles / "value.json"
    destination.write_bytes(b'{"value":2}\n')
    destination.chmod(0o600)
    _, temp = _publish_reserved_paths(bundles, "value.json", {"value": 1})
    temp.write_bytes(b'{"value":1}\n')
    temp.chmod(0o600)

    with pytest.raises(CanonicalReplayError, match="different bytes"):
        publish_immutable_json(private, "bundles/value.json", {"value": 1})
    assert destination.read_bytes() == b'{"value":2}\n'
    assert destination.stat().st_nlink == 1
    assert not temp.exists()


def test_persistent_receipt_post_link_fault_blocks_then_exact_retry_recovers(
    replay_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    receipts = replay_fixture["private"] / "receipts"
    receipts.mkdir(mode=0o700)
    unrelated = receipts / "unrelated.json"
    unrelated.write_bytes(b"UNRELATED")
    unrelated.chmod(0o600)
    receipt_leaf = f'{replay_fixture["draft_payload"]["registry"]["sha256"]}.json'
    real_link = os.link
    real_fsync = os.fsync
    receipt_linked = False

    def observe_receipt_link(*args: Any, **kwargs: Any) -> None:
        nonlocal receipt_linked
        real_link(*args, **kwargs)
        if len(args) >= 2 and args[1] == receipt_leaf:
            receipt_linked = True

    def fail_persistently_after_receipt_link(fd: int) -> None:
        if receipt_linked:
            raise OSError("persistent receipt fsync fault")
        real_fsync(fd)

    monkeypatch.setattr(os, "link", observe_receipt_link)
    monkeypatch.setattr(os, "fsync", fail_persistently_after_receipt_link)
    with pytest.raises(OSError, match="persistent receipt"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )
    receipt = receipts / receipt_leaf
    assert receipt.is_file()
    assert receipt.stat().st_nlink == 2
    assert unrelated.read_bytes() == b"UNRELATED"
    with pytest.raises(CanonicalReplayError, match="receipt|unavailable"):
        verify_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
        )

    monkeypatch.setattr(os, "link", real_link)
    monkeypatch.setattr(os, "fsync", real_fsync)
    result = produce_canonical_replay(
        private_root=replay_fixture["private"],
        registry_path=replay_fixture["registry"],
        draft_path=replay_fixture["draft"],
    )
    assert receipt.stat().st_nlink == 1
    assert verify_canonical_replay(
        private_root=replay_fixture["private"],
        registry_path=replay_fixture["registry"],
    ) == result
    assert unrelated.read_bytes() == b"UNRELATED"


@pytest.mark.parametrize("attack", ["lock_fifo", "temp_fifo_race"])
def test_publish_reserved_fifo_fails_without_blocking(
    tmp_path: Path, attack: str
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    bundles = private / "bundles"
    bundles.mkdir(mode=0o700)
    lock, temp = _publish_reserved_paths(bundles, "value.json", {"value": 1})
    if attack == "lock_fifo":
        os.mkfifo(lock, 0o600)
        script = """
import sys
from pathlib import Path
from quant_investor.factors.governance_canonical_replay import publish_immutable_json
publish_immutable_json(Path(sys.argv[1]), "bundles/value.json", {"value": 1})
"""
    else:
        temp.write_bytes(b'{"value":1}\n')
        temp.chmod(0o600)
        script = """
import os
import sys
from pathlib import Path
from quant_investor.factors.governance_canonical_replay import publish_immutable_json
private = Path(sys.argv[1])
temp_name = sys.argv[2]
real_open = os.open
raced = False
def race_to_fifo(path, flags, mode=0o777, *, dir_fd=None):
    global raced
    if (
        path == temp_name
        and not raced
        and flags & os.O_ACCMODE == os.O_RDONLY
    ):
        raced = True
        os.unlink(path, dir_fd=dir_fd)
        os.mkfifo(path, 0o600, dir_fd=dir_fd)
    return real_open(path, flags, mode, dir_fd=dir_fd)
os.open = race_to_fifo
publish_immutable_json(private, "bundles/value.json", {"value": 1})
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(private), temp.name],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        timeout=5,
    )
    assert completed.returncode != 0
    assert not (bundles / "value.json").exists()
    with pytest.raises(CanonicalReplayError, match="regular file"):
        publish_immutable_json(private, "bundles/value.json", {"value": 1})


def test_publish_subdirectory_creation_race_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    real_mkdir = os.mkdir
    barrier = threading.Barrier(2)

    def synchronized_mkdir(
        path: Any, mode: int = 0o777, *, dir_fd: int | None = None
    ) -> None:
        if path == "bundles" and dir_fd is not None:
            barrier.wait(timeout=5)
        real_mkdir(path, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", synchronized_mkdir)
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            results.append(
                publish_immutable_json(
                    private, "bundles/value.json", {"value": 1}
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(results) == 2
    assert results[0] == results[1]
    destination = private / "bundles" / "value.json"
    assert destination.read_bytes() == b'{"value":1}\n'
    assert destination.stat().st_nlink == 1


def test_publish_root_rename_guard_failure_does_not_leak_parent_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fd_root = Path("/dev/fd")
    if not fd_root.is_dir():
        pytest.skip("platform does not expose /dev/fd")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    moved = tmp_path / "moved-private"
    original_open_parent = canonical_module._open_publish_parent

    def rename_root_after_open(
        private_root: str | os.PathLike[str], relative_path: str
    ) -> tuple[Any, ...]:
        opened = original_open_parent(private_root, relative_path)
        private.rename(moved)
        return opened

    monkeypatch.setattr(
        canonical_module,
        "_open_publish_parent",
        rename_root_after_open,
    )
    before = len(list(fd_root.iterdir()))
    try:
        with pytest.raises(CanonicalReplayError):
            publish_immutable_json(private, "value.json", {"value": 1})
    finally:
        if moved.exists() and not private.exists():
            moved.rename(private)
    after = len(list(fd_root.iterdir()))
    assert after == before


def test_publish_surrogate_leaf_fails_without_fd_leak(tmp_path: Path) -> None:
    fd_root = Path("/dev/fd")
    if not fd_root.is_dir():
        pytest.skip("platform does not expose /dev/fd")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    before = len(list(fd_root.iterdir()))
    for _ in range(25):
        with pytest.raises(CanonicalReplayError, match="UTF-8"):
            publish_immutable_json(
                private, "bundles/\udcff.json", {"value": 1}
            )
    after = len(list(fd_root.iterdir()))
    assert after == before


def test_root_capture_fstat_failure_does_not_leak_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fd_root = Path("/dev/fd")
    if not fd_root.is_dir():
        pytest.skip("platform does not expose /dev/fd")
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    session = SafeReadSession(private)
    real_fstat = os.fstat
    before = len(list(fd_root.iterdir()))

    def fail_fstat(fd: int) -> os.stat_result:
        raise OSError("injected root capture fstat failure")

    monkeypatch.setattr(os, "fstat", fail_fstat)
    for _ in range(25):
        with pytest.raises(OSError, match="root capture"):
            session._open_directory_path(
                str(private), require_private=True, capture=[]
            )
    monkeypatch.setattr(os, "fstat", real_fstat)
    after = len(list(fd_root.iterdir()))
    assert after == before


def test_receipt_unknown_fields_fail_closed(
    replay_fixture: dict[str, Any],
) -> None:
    result = produce_canonical_replay(
        private_root=replay_fixture["private"],
        registry_path=replay_fixture["registry"],
        draft_path=replay_fixture["draft"],
    )
    receipt = replay_fixture["private"] / "receipts" / f'{result["registry_sha256"]}.json'
    payload = json.loads(receipt.read_bytes())
    payload["unknown"] = True
    receipt.write_bytes(_json_bytes(payload))
    with pytest.raises(CanonicalReplayError, match="unknown"):
        verify_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
        )


def test_strict_json_wraps_huge_integer_conversion_limit() -> None:
    raw = b'{"a":' + (b"9" * 5000) + b"}"
    with pytest.raises(CanonicalReplayError, match="invalid JSON"):
        strict_json_loads(raw, expected_fields={"a"})


def test_huge_numeric_conversion_and_cli_fail_closed_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(CanonicalReplayError, match="finite"):
        canonical_module._finite(10**4000, "huge")

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    registry = tmp_path / "registry.json"
    _write_json(
        registry,
        {
            "schema_version": "mined-factor-registry.v1",
            "metadata": {},
            "factors": [],
        },
        mode=0o644,
    )
    draft = tmp_path / "huge.json"
    draft.write_bytes(b'{"a":' + (b"9" * 5000) + b"}")
    draft.chmod(0o600)
    assert replay_main(
        [
            "--private-root",
            str(private),
            "--registry-path",
            str(registry),
            "--draft-path",
            str(draft),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err


def test_walk_forward_purged_requires_exact_boolean() -> None:
    evidence = {
        "purged": 1,
        "purge_days": 30,
        "embargo_days": 30,
        "folds": [],
    }
    with pytest.raises(CanonicalReplayError, match="boolean"):
        canonical_module._validate_walk_forward(
            evidence,
            "walk_forward",
            open_days=[],
            cutoff="2025-04-28",
        )


@pytest.mark.parametrize(
    "evidence",
    [
        {
            "purged": False,
            "purge_days": True,
            "embargo_days": 30,
            "folds": [],
        },
        {
            "purged": False,
            "purge_days": "30",
            "embargo_days": 30,
            "folds": [],
        },
        {
            "purged": False,
            "purge_days": 30,
            "embargo_days": 30,
            "folds": {},
        },
        {
            "purged": True,
            "purge_days": 1,
            "embargo_days": 1,
            "folds": [{"unexpected": True}],
        },
        {
            "purged": False,
            "purge_days": 30,
            "embargo_days": 30,
            "folds": [
                {
                    "train_end": "2025-01-28",
                    "validation_start": "2025-03-28",
                    "validation_end": "2025-04-28",
                    "evidence_hash": True,
                }
            ],
        },
    ],
)
def test_walk_forward_validates_all_fields_even_when_semantically_ineligible(
    evidence: dict[str, Any],
) -> None:
    with pytest.raises(CanonicalReplayError):
        canonical_module._validate_walk_forward(
            evidence,
            "walk_forward",
            open_days=["2025-01-28", "2025-03-28", "2025-04-28"],
            cutoff="2025-04-28",
        )


def test_unselected_factor_malformed_walk_forward_blocks_full_graph(
    replay_fixture: dict[str, Any],
) -> None:
    registry = json.loads(replay_fixture["registry"].read_bytes())
    unselected_registry_record = {
        "name": "unselected",
        "state": "shadow",
        "metadata": {
            "factor_family": "value",
            "dominant_primitive_cluster": "primary",
        },
    }
    registry["factors"].append(unselected_registry_record)
    _replace_registry_and_rebind_graph(replay_fixture, registry)
    draft = replay_fixture["draft_payload"]
    unselected = _factor_record(
        "unselected",
        {"unselected": unselected_registry_record},
    )
    unselected["p_value"] = 0.9
    unselected["walk_forward"] = {
        "purged": False,
        "purge_days": True,
        "embargo_days": 30,
        "folds": {},
    }
    predecessor = {
        "kind": "genesis",
        "byte_sha256": "0" * 64,
        "semantic_sha256": "0" * 64,
    }
    for stage in CONTROL_CHAIN_STAGES:
        ref = next(
            item
            for item in draft["stages"]
            if item["arm"] == "A" and item["stage"] == stage
        )
        stage_path = Path(ref["path"])
        stage_payload = json.loads(stage_path.read_bytes())
        stage_payload["predecessor"] = predecessor
        if stage == "quant":
            stage_payload["output"]["factor_records"].append(unselected)
        stage_sha = _write_json(stage_path, stage_payload)
        semantic_sha = _sha(canonical_json_bytes(stage_payload["output"]))
        ref["sha256"] = stage_sha
        ref["semantic_sha256"] = semantic_sha
        predecessor = {
            "kind": "stage",
            "byte_sha256": stage_sha,
            "semantic_sha256": semantic_sha,
        }
    _write_json(replay_fixture["draft"], draft)
    with pytest.raises(CanonicalReplayError, match="walk|integer|fold"):
        produce_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
            draft_path=replay_fixture["draft"],
        )


def test_bundle_recomputed_boolean_cannot_equal_numeric_value(
    replay_fixture: dict[str, Any],
) -> None:
    result = produce_canonical_replay(
        private_root=replay_fixture["private"],
        registry_path=replay_fixture["registry"],
        draft_path=replay_fixture["draft"],
    )
    bundle = replay_fixture["private"] / "bundles" / "evidence-20260107-1.json"
    bundle_payload = json.loads(bundle.read_bytes())
    bundle_payload["recomputed"]["arms"]["A"]["turnover"] = True
    bundle_raw = _json_bytes(bundle_payload)
    bundle.write_bytes(bundle_raw)
    receipt = replay_fixture["private"] / "receipts" / f'{result["registry_sha256"]}.json'
    receipt_payload = json.loads(receipt.read_bytes())
    receipt_payload["bundle_sha256"] = _sha(bundle_raw)
    receipt.write_bytes(_json_bytes(receipt_payload))
    with pytest.raises(CanonicalReplayError, match="recomputed"):
        verify_canonical_replay(
            private_root=replay_fixture["private"],
            registry_path=replay_fixture["registry"],
        )


def test_retired_v1_cli_rejects_without_logical_control_output(
    replay_fixture: dict[str, Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert replay_main(
        [
            "--private-root",
            str(replay_fixture["private"]),
            "--registry-path",
            str(replay_fixture["registry"]),
            "--draft-path",
            str(replay_fixture["draft"]),
        ]
    ) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "legacy_canonical_replay_v1_retired" in captured.err
    assert str(replay_fixture["registry"]) not in captured.err
