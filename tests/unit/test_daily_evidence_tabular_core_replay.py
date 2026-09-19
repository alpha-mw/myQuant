"""Exercise binary/legacy pool replay with controlled outer Factor/EOD admission.

The pool, native observations, byte readers, request identities and timing checks
are real. This isolates the new binary integration, not full EOD authority.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest

from quant_investor.intelligence._common import artifact_ref
from quant_investor.intelligence import storage
from quant_investor.operations import completion_core, core_pool
from quant_investor.operations.daily_journal import request_identity
from test_daily_evidence_dag_journal import request
from test_daily_evidence_tabular_pool import inventory, setup, fail
from test_unified_daily_intelligence_storage import _pool_rank, _write_request


@pytest.mark.parametrize("legacy", [False, True])
def test_completed_core_replays_exact_pool_format_without_writes(tmp_path, monkeypatch, legacy):
    store, args, policy = setup(tmp_path, monkeypatch)

    def put(path, value):
        return {"path": path, "sha256": _write_request(tmp_path / path, value)}

    pointer = put("pointer.json", {"activated_at": "2026-08-28T13:00:00Z"})
    rank = _pool_rank(tmp_path, policy, signal_date="20260828", pointer_sha=pointer["sha256"])
    args["rank"] = rank
    observations = store._observations(rank, None)
    policy_ref = {"path": args["policy_path"], "sha256": args["expected_policy_sha256"]}
    obj = artifact_ref(policy)
    obj_file = put(f"results/factors/objects/{obj['kind']}/{obj['byte_sha256']}.json", policy)
    payload = observations[0]["payload"]
    generation = {
        "created_at": rank["created_at"],
        "payload": {
            key: obj
            for key in (
                "deployed_release_ref",
                "calendar_compilation_ref",
                "calendar_capture_custody_attestation_ref",
                "market_pit_selection_ref",
                "market_input_ref",
            )
        },
    }
    generation_ref = put(
        f"results/factors/generations/{payload['factor_generation_id']}/generation.json", generation
    )
    snapshot = {
        **payload,
        "factor_generation": generation,
        "factor_rows": [value["payload"] for value in observations],
        "factor_generation_sha256": generation_ref["sha256"],
    }
    # The external generation admission is controlled. Observation source binding
    # remains real and is checked against its original native generation identity.
    original_validate = completion_core.validate_core_observation

    def original_generation_binding(document, *, alias, snapshot):
        original_validate(
            document,
            alias=alias,
            snapshot={
                **snapshot,
                "factor_generation_sha256": payload["factor_generation_sha256"],
            },
        )

    monkeypatch.setattr(completion_core, "validate_core_observation", original_generation_binding)
    monkeypatch.setattr(
        completion_core,
        "FactorProductionStore",
        lambda _: SimpleNamespace(
            inspect_recorded_research_inputs=lambda **kwargs: snapshot,
        ),
    )
    monkeypatch.setattr(completion_core, "build_factor_research_rank", lambda **kwargs: rank)
    monkeypatch.setattr(
        "quant_investor.operations.native_input_contract.validate_native_input_shape",
        lambda _: None,
    )
    monkeypatch.setattr(
        "quant_investor.operations.future_calendar_binding.future_calendar_outputs",
        lambda **kwargs: {},
    )

    if legacy:
        docs = storage._pool_documents(
            rank=rank,
            policy=policy,
            policy_path=args["policy_path"],
            policy_sha256=args["expected_policy_sha256"],
        )
        root = tmp_path / storage.POOL_ROOT_RELATIVE_PATH / storage.POOL_STRATEGY_ID / "2026-08-28"
        root.mkdir(parents=True, mode=0o700)
        for name, value in docs.items():
            _write_request(root / name, value)
    else:
        store.publish(**args, before_publish=lambda: None)
    pool_refs = store.verify(**args)
    node_outputs = {
        "calendar": {
            key: obj_file
            for key in ("calendar_compilation_ref", "calendar_capture_custody_attestation_ref")
        },
        "pit": {"market_pit_selection_ref": obj_file},
        "market": {"market_input_ref": obj_file},
        "factor": {"generation": generation_ref},
        "top100": pool_refs,
    }
    for node, alias in [("low_observation", "LOW"), ("w80_observation", "W80")]:
        path = f"results/factors/observations/2026/08/28/{alias}.json"
        node_outputs[node] = {
            alias: {
                "path": path,
                "sha256": hashlib.sha256((tmp_path / path).read_bytes()).hexdigest(),
            }
        }
    terminals = {}
    for node in core_pool.CORE_NODES:
        req = {
            **request(),
            "trade_date": "20260828",
            "node_id": node,
            "input_refs": {"factor_pointer": pointer},
            "release_ref": policy_ref,
            "policy_refs": {"research": policy_ref} if node == "top100" else {},
        }
        put(f"journals/{node}/request.json", req)
        terminals[node] = put(
            f"journals/{node}/attempt-0001/terminal.json",
            {
                "request_key": request_identity(req)[1],
                "finished_at": "2026-08-28T14:01:00Z",
                "output_refs": node_outputs[node],
                "recovered": False,
            },
        )
    recorded = {
        "node_terminal_refs": terminals,
        "release_ref": policy_ref,
        "native_inputs_ref": put(
            "native-inputs.json", {"factor_pointer_sha256": pointer["sha256"]}
        ),
    }
    monkeypatch.setattr(
        completion_core,
        "inspect_recorded_completion",
        lambda **kwargs: {"recorded_completion": recorded},
    )
    monkeypatch.setattr(storage.DailyResearchPoolStore, "publish", fail)
    monkeypatch.setattr(storage, "_pool_generated_at", fail)
    before = inventory(tmp_path)
    result = completion_core.replay_completed_core(
        workspace=str(tmp_path),
        trade_date="20260828",
        completion_ref=policy_ref,
    )
    assert result["validation_scope"] == "COMPLETED_CORE_NATIVE_REPLAY"
    assert result["consumer_admission"] is False
    assert before == inventory(tmp_path)
    if not legacy:
        table = tmp_path / pool_refs["top100.parquet"]["path"]
        table.write_bytes(b"not a table")
        with pytest.raises(storage.ResearchPoolConflict):
            completion_core.replay_completed_core(
                workspace=str(tmp_path),
                trade_date="20260828",
                completion_ref=policy_ref,
            )


def test_active_pool_probe_rejects_legacy_format_without_publishing(tmp_path, monkeypatch):
    store, args, policy = setup(tmp_path, monkeypatch)
    docs = storage._pool_documents(
        rank=args["rank"],
        policy=policy,
        policy_path=args["policy_path"],
        policy_sha256=args["expected_policy_sha256"],
    )
    root = tmp_path / storage.POOL_ROOT_RELATIVE_PATH / storage.POOL_STRATEGY_ID / "2026-08-28"
    root.mkdir(parents=True, mode=0o700)
    for name, value in docs.items():
        _write_request(root / name, value)
    adapter = core_pool.NativePoolAdapter(
        SimpleNamespace(
            pool=store,
            pool_arguments=lambda req: args,
            recheck=fail,
        )
    )
    before = inventory(tmp_path)
    monkeypatch.setattr(storage.DailyResearchPoolStore, "publish", fail)
    with pytest.raises(storage.ResearchPoolConflict):
        adapter.probe({})
    assert before == inventory(tmp_path)


def test_weekly_actual_publication_time_remains_late(tmp_path, monkeypatch):
    from scripts.cn_weekly_review_v2 import assess_production_days

    store, args, _ = setup(tmp_path, monkeypatch)
    monkeypatch.setattr(storage, "_pool_generated_at", lambda: "2026-08-31T07:00:00Z")
    result = store.publish(**args, before_publish=lambda: None)
    manifest = json.loads((tmp_path / result["manifest_path"]).read_bytes())
    assessment = assess_production_days(
        [
            {
                "trade_date": "2026-08-28",
                "maintenance_verified": True,
                "observations_verified": True,
                "top100_verified": True,
                "artifact_completion_at": manifest["created_at"],
            }
        ],
        ["2026-08-28"],
    )
    assert assessment["complete_same_day_dates"] == []
    assert assessment["late_recovery_dates"] == ["2026-08-28"]
