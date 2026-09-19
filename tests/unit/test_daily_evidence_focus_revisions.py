"""Focus-only research output may revise missing inputs without broadening authority."""

from copy import deepcopy
from functools import partial
import hashlib

import pytest

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.cli import unified
from quant_investor.intelligence.storage import approved_theme_policy_v2
from quant_investor.intelligence.pcb_ai_hardware import build_focus_membership, build_focus_evidence
from quant_investor.operations.daily_contract import NodeState, ContractError
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.journal_revisions import append_revision
from test_daily_evidence_dag_journal import request
from test_daily_evidence_focus_sources import theme_source, pit_context, AS_OF, DAY
from test_daily_evidence_research_sources import put
from quant_investor.intelligence.theme_sources import descriptor_refs


@pytest.mark.parametrize("authority_value", [False, True, 0])
def test_focus_input_revision_requires_exact_false_authority(tmp_path, authority_value):
    policy = approved_theme_policy_v2()
    source = theme_source(tmp_path, ["000001.SZ"], "pool")
    native = unified._daily_theme_projection(
        {"as_of": AS_OF, "policy": policy, "theme_source": source},
        ["000001.SZ"],
        partial(unified._daily_source_document, str(tmp_path)),
    )
    pit = pit_context(tmp_path)
    pool_ref = put(tmp_path, "pool.json", {"synthetic": True})
    member = build_focus_membership(
        as_of=AS_OF,
        pool_manifest_ref=pool_ref,
        pit=pit,
        pool_theme=native,
        focus_theme=None,
        pool_source_refs=descriptor_refs(source),
        focus_source_refs=[],
    )
    report = build_focus_evidence(
        membership=member,
        pit=pit,
        focus_theme=None,
        industry=None,
        evidence=[],
        industry_source_refs=[],
        daily_policy=policy,
    )
    report["payload"]["authority"]["trade"] = authority_value
    report = seal_artifact(report["kind"], report["payload"], created_at=report["created_at"])
    journal = DailyJournal(str(tmp_path), DAY)
    raw = canonical_json_bytes(report)
    output = {
        "path": str(
            journal.root / "research/artifacts" / (hashlib.sha256(raw).hexdigest() + ".json")
        ),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    req = {**request(), "trade_date": DAY, "node_id": "exposure"}
    with journal.locked():
        journal.storage.write(output["path"], raw)
        journal.begin(req)
        prior = journal.finish(
            req,
            state=NodeState.PARTIAL,
            output_refs={"pcb_ai_hardware_evidence": output},
            failure_code="INPUT_MISSING",
        )
        old_terminal = journal.storage.read(prior["terminal_ref"]["path"])
        new = deepcopy(req)
        new["input_refs"]["factor"] = {"path": "arrived.json", "sha256": "d" * 64}
        if authority_value is False:
            append_revision(journal, new, expected_request_key=prior["request_key"])
        else:
            with pytest.raises(ContractError, match="NOT_RETRYABLE"):
                append_revision(journal, new, expected_request_key=prior["request_key"])
        assert journal.storage.read(prior["terminal_ref"]["path"]) == old_terminal
        assert journal.storage.read(output["path"]).data == raw
