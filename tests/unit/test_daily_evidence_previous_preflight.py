"""Previous-EOD gate composition; native replay is an explicit controlled seam."""

import hashlib
from types import SimpleNamespace
import pytest
from test_daily_evidence_store_materialization import context as _script_context  # noqa: F401
from scripts import daily_completion_replay as module
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import EOD_NODE_IDS, ContractError
from quant_investor.factors import production_authority as factor


@pytest.mark.parametrize("fault", [None, "replay", "factor", "store"])
@pytest.mark.parametrize("registered", [False, True])
def test_previous_anchor_requires_full_replay_and_matching_baselines(
    tmp_path, monkeypatch, fault, registered
):
    raw = canonical_json_bytes({"factor_pointer_sha256": "a" * 64})
    path = tmp_path / "inputs.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    input_ref = {"path": "inputs.json", "sha256": hashlib.sha256(raw).hexdigest()}
    recipe = {
        "previous_completion_ref": {
            "path": "results/operations/daily_production/CN/20260827/completion.v1.json",
            "sha256": "b" * 64,
        },
        "target_trade_date": "20260828",
        "bootstrap_ref": None,
        "store_preimages": {"store_pointer_ref": {"path": "store.json", "sha256": "c" * 64}},
    }
    if registered:
        from scripts import registered_daily_event_sources

        recipe["schema_version"] = "cn-daily-execute-recipe.v6"
        recipe["store_preimages"]["store_pointer_ref"]["sha256"] = "e" * 64

        def registered_source(**kwargs):
            assert kwargs["recipe"] is recipe
            return {
                "declaration": {
                    "baseline_store_pointer_ref": {
                        "path": "synthetic-prior-committed-pointer.json",
                        "sha256": "c" * 64,
                    }
                }
            }

        monkeypatch.setattr(registered_daily_event_sources, "read_recipe_source", registered_source)
    calls = []

    def replay(**kwargs):
        calls.append("replay")
        assert kwargs["trade_date"] == "20260827"
        return {
            "native_replay_validated": True,
            "validated_nodes": [] if fault == "replay" else sorted(EOD_NODE_IDS),
        }

    monkeypatch.setattr(module, "replay_native_completion", replay)
    monkeypatch.setattr(
        module,
        "inspect_recorded_completion",
        lambda **kw: {"recorded_completion": {"native_inputs_ref": input_ref}},
    )
    pointer = SimpleNamespace(byte_sha256="d" * 64 if fault == "factor" else "a" * 64)
    marker = object()

    def make_store(*args):
        calls.append("factor")
        return SimpleNamespace(
            read=lambda path: pointer if path == factor.FACTOR_ACTIVE_POINTER_PATH else marker,
            verify_active=lambda: {
                "factor_authority": factor.FACTOR_AUTHORITY_ACTIVE,
                "as_of": "20260827",
                "factor_pointer_byte_sha256": pointer.byte_sha256,
            },
        )

    monkeypatch.setattr(factor, "FactorProductionStore", make_store)
    monkeypatch.setattr(
        "quant_investor.operations.daily_status.read_daily_status",
        lambda *a: {
            "nodes": {
                "store": {
                    "terminal": {
                        "output_refs": {
                            "pointer": {
                                "sha256": "d" * 64 if fault == "store" else "c" * 64,
                                **(
                                    {"path": "synthetic-prior-committed-pointer.json"}
                                    if registered
                                    else {}
                                ),
                            }
                        }
                    }
                }
            }
        },
    )
    if fault:
        with pytest.raises(ContractError, match="EXECUTION_PREVIOUS"):
            module.verify_initial_previous_completion(workspace=str(tmp_path), recipe=recipe)
    else:
        assert (
            module.verify_initial_previous_completion(workspace=str(tmp_path), recipe=recipe)[
                "previous_trade_date"
            ]
            == "20260827"
        )
    assert calls == (["replay"] if fault == "replay" else ["replay", "factor"])
