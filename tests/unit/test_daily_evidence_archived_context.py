"""Archived claim boundary; deep installation is an explicit controlled seam here."""

import hashlib
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.market.maintenance_journal import logical_task_claim
from quant_investor.operations.completed_handoff_snapshot import _mint_snapshot
from quant_investor.operations import archived_handoff_context as module
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_contract import GRAPH_SHA256
from quant_investor.operations.daily_journal import FALSE_AUTHORITY
from quant_investor.operations.maintenance_handoff_contract import (
    SCHEMA,
    HISTORICAL_SCHEMA,
    HANDOFF_V2_FIELDS,
)


def fixture(root, monkeypatch, *, budget=2, historical=False):
    docs = {}

    def put(role, path, value):
        p = root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        q = p.parent
        while q != root:
            q.chmod(0o700)
            q = q.parent
        raw = canonical_json_bytes(value)
        p.write_bytes(raw)
        p.chmod(0o600)
        ref = {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
        if role:
            docs[role] = (role, path, ref["sha256"], raw)
        return ref

    native_module = root / "install/quant_investor/market/maintenance_journal.py"
    native_module.parent.mkdir(parents=True)
    native_module.write_bytes(b"original native source")
    native_module.chmod(0o644)
    install = put(
        "release_install_input",
        "install-input.json",
        {"release_install_evidence": {"payload": {"python_executable": "/verified/python"}}},
    )
    started = put(
        None,
        "started.json",
        {"state": "STARTED", "mode": "execute", "started_at": "2026-09-08T12:20:00Z"},
    )
    put(
        "loop_context",
        "context.json",
        {
            "schema_version": "cn-daily-factor-loop.v1",
            "release_install_input_ref": install,
            "release_repository_root": "/verified/repository",
        },
    )
    day = "20260907" if historical else "20260908"
    claim = logical_task_claim(logical_key=day + "-2020-execute", slot="2020", mode="execute")
    claim["attempt_budget"] = budget
    claim["installation"] = {
        "python": "/verified/python",
        "module": str(native_module.resolve()),
        "implementation_sha256": hashlib.sha256(native_module.read_bytes()).hexdigest(),
    }
    claim_ref = put("logical_claim", f"run/logical_tasks/{day}-2020-execute/claim.json", claim)
    handoff = {
        key: {"path": "unused.json", "sha256": "a" * 64}
        for key in HANDOFF_V2_FIELDS
        if key.endswith("_ref")
    }
    handoff.update(
        schema_version=HISTORICAL_SCHEMA if historical else SCHEMA,
        trade_date=day,
        graph_sha256=GRAPH_SHA256,
        authority=FALSE_AUTHORITY,
        sealed_at="2026-09-08T13:00:00Z",
        release_ref={"path": "release.json", "sha256": "e" * 64},
        maintenance_started_ref=started,
        logical_claim_ref=claim_ref,
        prospective_policy_ref=None,
    )
    if historical:
        handoff["historical_session_ref"] = {
            "path": "historical-session.v1.json",
            "sha256": "b" * 64,
        }
        handoff["catchup_binding_ref"] = {"path": "binding.json", "sha256": "c" * 64}
        from types import SimpleNamespace

        # This fixture isolates archived install/claim identity. Real binding and
        # original-source replay are exercised by the historical handoff tests.
        monkeypatch.setattr(
            "quant_investor.operations.catchup_binding.read_catchup_binding",
            lambda **kwargs: {
                "binding": {"execution_request_ref": handoff["request_ref"]},
                "recipe": {"role": "recipe"},
                "sources": SimpleNamespace(recheck=lambda: None),
            },
        )
    put("handoff", "handoff.json", handoff)
    for role in ("completion", "ledger", "materialization", "recipe"):
        put(role, role + ".json", {"role": role})
    calls = []

    def deep(raw, **kwargs):
        calls.append(kwargs)
        return {
            "state": "PASS",
            "release_ref": {"byte_sha256": "e" * 64},
            "import_origin": str(root / "install/quant_investor/__init__.py"),
        }

    monkeypatch.setattr(module, "verify_archived_release_install_input", deep)
    return (
        _mint_snapshot(workspace=str(root), trade_date=day, documents=tuple(docs.values())),
        native_module,
        claim_ref,
        calls,
        deep,
    )


def test_archived_claim_keeps_original_budget_and_no_authority(tmp_path, monkeypatch):
    snapshot, _, _, calls, _ = fixture(tmp_path, monkeypatch)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = module.verify_archived_handoff_context(snapshot)
    assert len(calls) == 1 and calls[0]["repository_root"] == "/verified/repository"
    assert result["claim"]["attempt_budget"] == result["claim"]["close_request_budget"] == 2
    assert (
        result["claim"]["new_budget_granted"] is False
        and result["claim"]["execution_authorized"] is False
    )
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_archived_historical_claim_uses_target_day_not_runtime_day(tmp_path, monkeypatch):
    snapshot, _, _, _, _ = fixture(tmp_path, monkeypatch, historical=True)
    result = module.verify_archived_handoff_context(snapshot)
    assert result["claim"]["run_date"] == "20260907"
    assert result["claim"]["logical_key"] == "20260907-2020-execute"
    assert result["claim"]["new_budget_granted"] is False


@pytest.mark.parametrize("fault", ["budget", "module", "claim_race", "module_race", "deep_failure"])
def test_bad_or_changed_archived_identity_never_falls_back(tmp_path, monkeypatch, fault):
    snapshot, native_module, claim_ref, calls, deep = fixture(
        tmp_path, monkeypatch, budget=3 if fault == "budget" else 2
    )
    if fault == "module":
        native_module.write_bytes(b"changed module")
    elif fault == "claim_race":

        def changed(*a, **kw):
            result = deep(*a, **kw)
            (tmp_path / claim_ref["path"]).write_bytes(b"changed claim")
            return result

        monkeypatch.setattr(module, "verify_archived_release_install_input", changed)
    elif fault == "module_race":
        original = module.read_stable_regular_file
        count = []

        def race(path, **kw):
            raw = original(path, **kw)
            count.append(1)
            if len(count) == 1:
                native_module.write_bytes(b"changed during read")
            return raw

        monkeypatch.setattr(module, "read_stable_regular_file", race)
    elif fault == "deep_failure":

        def failed(*a, **kw):
            raise ContractError("DEEP_INSTALL_REJECTED")

        monkeypatch.setattr(module, "verify_archived_release_install_input", failed)
    with pytest.raises(ContractError):
        module.verify_archived_handoff_context(snapshot)
