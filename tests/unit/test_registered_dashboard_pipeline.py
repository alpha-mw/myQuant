"""Native financial/corporate Dashboard; five-domain and full EOD admission are seams."""

from datetime import datetime, timezone
from types import SimpleNamespace
import json
from copy import deepcopy

import pytest

from test_registered_dashboard_sources import case
from test_registered_close_native import inventory
from _native_corporate_fixture import put
from quant_investor.intelligence._common import build_artifact, business_identity
from quant_investor.intelligence.investment_decision import DECISION_STATES
from quant_investor.operations.dashboard_evidence import KIND, DOMAINS
from quant_investor.operations.dashboard_serving_contract import build_head, digest, EVIDENCE_JSON
from quant_investor.operations.registered_dashboard import RegisteredDashboardSources
from quant_investor.contracts import canonical_json_bytes
from quant_investor.contracts.core import ArtifactValidationError
from scripts.daily_dashboard_sealed import SealedDashboardAdapter
from scripts import daily_dashboard_publication as publication
from quant_investor.operations.daily_contract import ContractError


def fixture(root, monkeypatch, *, symbol="002463.SZ", current=False):
    book, corporate, kwargs, cout, sout = case(root, monkeypatch, symbol=symbol)
    synthetic = put(
        root,
        "fixtures/controlled-five-domain-source.json",
        {"synthetic": True, "native_five_domain_admission": False},
    )
    authority_refs = {
        k: kwargs["store_terminal_ref"] if k == "store" else synthetic for k in DOMAINS
    }
    store_request_path = kwargs["store_terminal_ref"]["path"].rsplit("/", 2)[0] + "/request.json"
    store_request = json.loads((root / store_request_path).read_bytes())
    store_request_ref = {
        "path": store_request_path,
        "sha256": digest(canonical_json_bytes(store_request)),
    }
    counts = dict.fromkeys(DECISION_STATES, 0)
    counts["WATCHLIST"] = 100

    def evidence(stamp):
        fields = {
            "trade_date": corporate.journal.trade_date,
            "strategy_id": "aggressive_tech_manufacturing",
            "source_bindings": {
                k: {
                    "node_id": k,
                    "request_ref": store_request_ref if k == "store" else synthetic,
                    "terminal_ref": authority_refs[k],
                    "output_refs": (
                        sout
                        if k == "store"
                        else (
                            {"decision.v2.json": synthetic}
                            if k == "decision"
                            else {"synthetic": synthetic}
                        )
                    ),
                    "trade_date": corporate.journal.trade_date,
                    "state": "SUCCEEDED",
                }
                for k in DOMAINS
            },
            "source_refs": [synthetic],
            "top100_count": 100,
            "decision_state_counts": counts,
            "research_state": "EVIDENCE_BOUND",
        }
        return build_artifact(
            kind=KIND,
            identity_field="dashboard_evidence_id",
            identity=business_identity(kind=KIND, identity_inputs=fields),
            created_at=stamp,
            fields=fields,
        )

    monkeypatch.setattr(
        SealedDashboardAdapter, "sources", lambda self: SimpleNamespace(build=evidence)
    )
    node = SealedDashboardAdapter(
        workspace=str(root),
        journal=corporate.journal,
        release_ref=corporate.release_ref,
        plan_ref=kwargs["store_plan_ref"],
        market_ref=corporate.market_snapshot_ref,
        benchmark_ref={
            "path": "portfolio_dashboard/inputs/cn_index_benchmark.csv",
            "sha256": digest(
                (root / "portfolio_dashboard/inputs/cn_index_benchmark.csv").read_bytes()
            ),
        },
        risk_free_ref={
            "path": "portfolio_dashboard/inputs/cn_govt_bond_yield.csv",
            "sha256": digest(
                (root / "portfolio_dashboard/inputs/cn_govt_bond_yield.csv").read_bytes()
            ),
        },
        authority_terminal_refs=authority_refs,
        publish_current_dashboard=current,
        corporate_terminal_ref=kwargs["corporate_terminal_ref"],
        registered_event_declaration_ref=kwargs["registered_event_declaration_ref"],
    )
    with node.journal.locked():
        node.prepare()
        request = node.template()
        node.journal.begin(request)
        node.execute(request)
        outcome = node.probe(request).outcome
        terminal = node.journal.finish(
            request, state=outcome.state, output_refs=outcome.output_refs
        )
    registered = RegisteredDashboardSources(**kwargs).result(node.evidence_recipe["created_at"])
    outputs = outcome.output_refs
    completion = put(
        root, str(node.journal.root / "completion.v1.json"), {"synthetic_full_eod_admission": False}
    )
    proof = SimpleNamespace(
        inputs={"schema_version": "cn-daily-native-inputs.v7"},
        registered=registered,
        day=node.journal.trade_date,
        cutoff=registered["registered_transition"]["payload"]["as_of"],
        completion_ref=completion,
        output_refs=outputs,
        evidence=json.loads((root / outputs["daily_evidence"]["path"]).read_bytes()),
        v1=(root / outputs["v1"]["path"]).read_bytes(),
        v2=(root / outputs["v2"]["path"]).read_bytes(),
    )
    return node, request, terminal, proof, cout


@pytest.mark.parametrize("symbol,current", [("002463.SZ", False), ("300308.SZ", True)])
def test_native_registered_dashboard_forwards_same_report_and_serving_descriptor(
    tmp_path, monkeypatch, symbol, current
):
    node, request, terminal, proof, cout = fixture(
        tmp_path, monkeypatch, symbol=symbol, current=current
    )
    assert node.evidence_recipe["schema_version"] == "cn-daily-dashboard-evidence-recipe.v2"
    assert set(terminal["terminal"]["output_refs"]) == {
        "capture",
        "v1",
        "v2",
        "daily_evidence",
        "registered_transition",
    }
    assert proof.output_refs["registered_transition"] == cout["registered_transition"]
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = build_head(
        day=proof.day, ref=proof.completion_ref, previous_sha=None, registered_at=stamp
    )
    head_ref = put(tmp_path, "fixtures/serving-head.json", head)
    files = publication._serving_data_files(proof, head_ref)
    view = json.loads(files[EVIDENCE_JSON])
    assert view["schema_version"] == "cn-daily-dashboard-serving.v2"
    assert view["registered_transition_ref"] == cout["registered_transition"]
    assert view["registered_close_summary"]["official_valuation"] is True
    assert (
        view["registered_close_summary"]["final_record_id"]
        == json.loads(proof.v1)["latest_valid_record"]
    )
    before = inventory(tmp_path)
    with node.journal.locked():
        node.prepare()
        node.execute(request)
    assert inventory(tmp_path) == before
    if current:
        from quant_investor.operations.dashboard_serving_contract import SELECTOR_JSON

        head_raw = canonical_json_bytes(head)
        selectors = publication._selector_files(proof, head, head_raw, files, stamp)
        raw_fixture = {
            "synthetic": True,
            "native_financial_and_registered_sources": True,
            "five_domain_and_full_eod_admission": False,
            "raw": {
                "head": head_raw.decode(),
                "selector": selectors[SELECTOR_JSON].decode(),
                "evidence": files[EVIDENCE_JSON].decode(),
                "v1": proof.v1.decode(),
                "v2": proof.v2.decode(),
            },
        }
        # A test wrapper contains exact multiline native bytes, not a DAG artifact.
        (tmp_path / "registered-serving-raw.json").write_text(
            json.dumps(raw_fixture, ensure_ascii=False, indent=2) + "\n"
        )


def test_registered_completed_dashboard_replays_frozen_sources(tmp_path, monkeypatch):
    from scripts import daily_completion_dashboard as dashboard_replay
    from scripts import daily_completion_store as store_replay
    from quant_investor.operations import completion_corporate, dashboard_evidence
    from quant_investor.operations.registered_dashboard import recorded_registered_view

    node, request, terminal, proof, _ = fixture(
        tmp_path, monkeypatch, symbol="300308.SZ", current=True
    )
    corporate_request = json.loads(
        (tmp_path / node.registered_binding["corporate_terminal_ref"]["path"])
        .parent.parent.joinpath("request.json")
        .read_bytes()
    )
    corporate_recipe = json.loads(
        (tmp_path / corporate_request["input_refs"]["recipe"]["path"]).read_bytes()
    )
    inputs = {
        "schema_version": "cn-daily-native-inputs.v7",
        "trade_date": node.journal.trade_date,
        "release_ref": node.release_ref,
        "store_plan_ref": node.refs["store_plan"],
        "market_snapshot_ref": node.refs["market"],
        "benchmark_ref": node.refs["benchmark"],
        "risk_free_ref": node.refs["risk_free"],
        "publish_current_dashboard": True,
        "dashboard_publication_policy": publication.POLICY,
        **{
            k: corporate_recipe[k]
            for k in (
                "previous_trade_date",
                "calendar_ref",
                "corporate_action_context_ref",
                "decision_recipe_ref",
                "research_request_ref",
                "registered_event_declaration_ref",
            )
        },
        "adjustment_market_refs": corporate_recipe["market_refs"],
    }
    recorded = {
        "release_ref": node.release_ref,
        "native_inputs_ref": put(tmp_path, "fixtures/controlled-completion-inputs.json", inputs),
        "node_terminal_refs": {
            **node.authority_refs,
            "corporate_action_recon": node.registered_binding["corporate_terminal_ref"],
            "dashboard": terminal["terminal_ref"],
        },
    }
    completion_ref = put(
        tmp_path,
        "fixtures/controlled-eod-reference.json",
        {"synthetic": True, "full_eod_admission": False},
    )
    for module in (dashboard_replay, store_replay, completion_corporate):
        monkeypatch.setattr(
            module,
            "inspect_recorded_completion",
            lambda **kwargs: {"recorded_completion": recorded},
        )

    def five_domain(**kwargs):
        def build(stamp):
            assert stamp == proof.evidence["created_at"]
            return proof.evidence

        return SimpleNamespace(build=build)

    monkeypatch.setattr(dashboard_evidence, "DashboardEvidenceSources", five_domain)
    for path in (
        "results/strategy_records/CN/aggressive_tech_manufacturing/_record_store/current.v1.json",
        "results/strategy_records/CN/aggressive_tech_manufacturing/_event_store/current.v1.json",
        "data/parquet/cn/_latest.json",
        "data/parquet/cn/benchmarks/_latest.json",
    ):
        (tmp_path / path).unlink()
    before = inventory(tmp_path)
    result = dashboard_replay.replay_completed_dashboard(
        workspace=str(tmp_path), trade_date=node.journal.trade_date, completion_ref=completion_ref
    )
    assert result["registered_transition_ref"] == proof.registered["registered_transition_ref"]
    assert result["registered_close_summary"] == proof.registered["registered_close_summary"]
    observed = recorded_registered_view(workspace=tmp_path, completion=recorded, inputs=inputs)
    assert observed == proof.registered
    assert inventory(tmp_path) == before


def test_registered_serving_rejects_changed_summary_and_profile(tmp_path, monkeypatch):
    _, _, _, proof, _ = fixture(tmp_path, monkeypatch, current=True)
    reference = {"path": "fixtures/head.json", "sha256": "a" * 64}
    for fault in ("counts", "pointer", "record", "output", "old_profile"):
        changed = deepcopy(proof)
        if fault == "counts":
            changed.registered["registered_close_summary"]["close_writer_fill_count"] = 1
        elif fault == "pointer":
            changed.registered["registered_close_summary"]["writer_pointer_ref"]["sha256"] = (
                "f" * 64
            )
        elif fault == "record":
            changed.registered["registered_close_summary"]["final_record_id"] = "wrong-record"
        elif fault == "output":
            changed.output_refs["registered_transition"] = reference
        else:
            changed.inputs["schema_version"] = "cn-daily-native-inputs.v6"
        with pytest.raises((ContractError, ArtifactValidationError)):
            publication._serving_data_files(changed, reference)


def test_registered_publication_and_observation_bind_all_five_outputs(tmp_path, monkeypatch):
    from dataclasses import replace
    from quant_investor.operations.dashboard_serving_contract import (
        PREFIX,
        HEAD_JSON,
        SELECTOR_JSON,
    )

    node, _, terminal, source, _ = fixture(tmp_path, monkeypatch, current=True)
    inputs = {
        "schema_version": "cn-daily-native-inputs.v7",
        "trade_date": source.day,
        "publish_current_dashboard": True,
        "dashboard_publication_policy": publication.POLICY,
        "store_plan_ref": node.refs["store_plan"],
        "registered_event_declaration_ref": node.registered_binding[
            "registered_event_declaration_ref"
        ],
    }
    completion = {
        "native_inputs_ref": put(tmp_path, "fixtures/publication-inputs.json", inputs),
        "release_ref": node.release_ref,
        "native_validation_completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "node_terminal_refs": {
            **node.authority_refs,
            "corporate_action_recon": node.registered_binding["corporate_terminal_ref"],
            "dashboard": terminal["terminal_ref"],
        },
        "synthetic_full_eod_admission": False,
    }
    reference = put(tmp_path, str(node.journal.root / "completion.v1.json"), completion)
    proof = publication.VerifiedDashboard(
        workspace=tmp_path,
        day=source.day,
        completion_ref=reference,
        completion=completion,
        inputs=inputs,
        terminal_ref=terminal["terminal_ref"],
        output_refs=source.output_refs,
        v1=source.v1,
        v2=source.v2,
        evidence=source.evidence,
        cutoff=source.cutoff,
        head_preimage=None,
        predecessor_refs=(),
        anchor_ref=node.refs["store_plan"],
        registered=source.registered,
        _key=publication._PROOF_KEY,  # Explicit five-domain/full-EOD admission seam.
    )
    monkeypatch.setattr(
        publication,
        "verify_dashboard_for_publication",
        lambda *a: replace(
            proof,
            head_preimage=publication._optional(tmp_path, PREFIX + "/" + HEAD_JSON),
            _key=publication._PROOF_KEY,
        ),
    )
    real_validate = publication.validate_artifact

    def validate(value, *, expected_kind):
        if expected_kind == "daily_research_decision_report":
            assert json.loads(value)["synthetic"] is True
            return {"payload": {"as_of": proof.cutoff}}
        return real_validate(value, expected_kind=expected_kind)

    monkeypatch.setattr(publication, "validate_artifact", validate)
    result = publication.publish_completed_dashboard(
        workspace=str(tmp_path), completion_ref=reference
    )
    assert result["status"] == "PUBLISHED"
    descriptor_raw = (tmp_path / PREFIX / EVIDENCE_JSON).read_bytes()
    descriptor = json.loads(descriptor_raw)
    selector = json.loads((tmp_path / PREFIX / SELECTOR_JSON).read_bytes())
    assert selector["daily_evidence_sha256"] == digest(descriptor_raw)
    assert descriptor["registered_transition_ref"] == source.output_refs["registered_transition"]
    assert descriptor["registered_transition"] == source.registered["registered_transition"]
    before = inventory(tmp_path)
    assert (
        publication.publish_completed_dashboard(workspace=str(tmp_path), completion_ref=reference)
        == result
    )
    observed = publication.observed_serving_status(str(tmp_path), source.day)
    assert observed["publication_state"] == "RECORDED_EOD_PUBLICATION"
    assert inventory(tmp_path) == before
