from copy import deepcopy

import pytest

from quant_investor.operations.daily_contract import ContractError, NodeState
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.journal_revisions import append_revision
from test_daily_evidence_dag_journal import request, output


def successor():
    value = deepcopy(request())
    value["input_refs"]["factor"]["sha256"] = "f" * 64
    return value


def test_arriving_input_creates_linked_revision_preserving_failed_attempt(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    with journal.locked():
        journal.begin(request())
        old = journal.finish(
            request(), state=NodeState.BLOCKED, output_refs={}, failure_code="INPUT_MISSING"
        )
        old_bytes = (tmp_path / old["terminal_ref"]["path"]).read_bytes()
        new_key = append_revision(journal, successor(), expected_request_key=old["request_key"])
        current = journal.begin(successor())
        assert current["request_key"] == new_key and current["attempt"] == 1
        journal.finish(successor(), state=NodeState.SUCCEEDED, output_refs=output())
        assert (tmp_path / old["terminal_ref"]["path"]).read_bytes() == old_bytes
        with pytest.raises(ContractError, match="NODE_INPUT_CONFLICT"):
            journal.inspect(request())


@pytest.mark.parametrize(
    "state,code,refs",
    [
        (NodeState.SUCCEEDED, None, output()),
        (NodeState.FAILED, "POST_WRITE_IN_DOUBT", {}),
        (NodeState.BLOCKED, "AUTHORIZATION_BLOCKED", {}),
        (NodeState.BLOCKED, "INPUT_MISSING", output()),
    ],
)
def test_input_rebinding_never_bypasses_success_authority_or_indeterminate_write(
    tmp_path, state, code, refs
):
    journal = DailyJournal(str(tmp_path), "20260904")
    with journal.locked():
        old = journal.begin(request())
        journal.finish(request(), state=state, output_refs=refs, failure_code=code)
        with pytest.raises(ContractError):
            append_revision(journal, successor(), expected_request_key=old["request_key"])


def test_release_or_adapter_change_requires_separate_migration(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    with journal.locked():
        old = journal.begin(request())
        journal.finish(
            request(), state=NodeState.BLOCKED, output_refs={}, failure_code="INPUT_STALE"
        )
        new = successor()
        new["adapter_sha256"] = "c" * 64
        with pytest.raises(ContractError, match="REVISION_SCOPE_INVALID"):
            append_revision(journal, new, expected_request_key=old["request_key"])


def test_unfinished_attempt_cannot_be_rebound(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260904")
    with journal.locked():
        old = journal.begin(request())
        with pytest.raises(ContractError, match="RECONCILIATION_REQUIRED"):
            append_revision(journal, successor(), expected_request_key=old["request_key"])
