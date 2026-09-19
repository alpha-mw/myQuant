"""Exact catalog receipt resolution without selecting current Store state."""

from argparse import Namespace
from copy import deepcopy
import hashlib
import json

import pytest

from quant_investor.strategy_records import store
from quant_investor.strategy_records.event_store import build_empty_closure, StrategyEventStoreError
from quant_investor.strategy_records.event_receipts import (
    resolve_catalog_event_receipt,
    RECORD_ROOT,
)
from quant_investor.strategy_records.event_contracts import source_ref
from quant_investor.system.storage import SecureSystemStorage
from test_strategy_record_store import _bootstrap
from scripts.manage_cn_strategy_records import command_no_action


def put(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = store.canonical_json_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o644)  # Repository policy sources use their native public-readable mode.
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def fixture(root):
    record_root = root / RECORD_ROOT
    record_root.mkdir(parents=True)
    seeded = _bootstrap(record_root)
    result = command_no_action(
        Namespace(
            record_root=str(record_root),
            expected_pointer_sha=seeded["pointer_sha256"],
            receipt_id="receipt-20260824",
            reason="fixture-no-action",
            published_at="2026-08-24T08:00:00Z",
            generation_id="with-receipt",
        )
    )
    receipt = result["catalog"]["receipts"][-1]
    policy = put(
        root,
        "operations/policy.json",
        {
            "schema_id": "myquant.cn_daily_official_close_policy.v1",
            "policy_id": "test-policy",
            "revoked_at": None,
            "broker_order_trade_authority": False,
            "actual_holdings_mutation_authority": False,
        },
    )
    from quant_investor.strategy_records.event_store import EVENT_DIMENSIONS

    declaration = put(
        root,
        "operations/owner.json",
        {
            "schema_id": "myquant.cn_official_close_retrospective_owner_declaration.v1",
            "policy_id": "test-policy",
            "owner": "Maxwell",
            "strategy_label": "aggressive_tech_manufacturing",
            "authorized_at": "2026-08-24T09:00:00Z",
            "retrospective_empty_event_closure_authorized": True,
            "actual_holdings_mutation_authority": False,
            "cash_mutation_authority": False,
            "broker_order_trade_authority": False,
            "dates": [
                {
                    "trade_date": "2026-08-24",
                    "source_receipt_id": receipt["receipt_id"],
                    **{d: [] for d in EVENT_DIMENSIONS},
                }
            ],
        },
    )
    closure = build_empty_closure(
        trade_date="2026-08-24",
        sealed_at="2026-08-24T10:00:00Z",
        cutoff_at="2026-08-24T07:30:00Z",
        policy_ref=policy,
        owner_declaration_ref=declaration,
        source_receipt_ref={
            "path": "catalog:with-receipt#receipt:" + receipt["receipt_id"],
            "sha256": receipt["content_sha256"],
        },
    )
    return closure, record_root / result["pointer"]["catalog_path"]


def test_native_no_action_receipt_resolves_without_current_pointer(tmp_path, monkeypatch):
    closure, catalog = fixture(tmp_path)
    (tmp_path / RECORD_ROOT / "_record_store/current.v1.json").unlink()
    original = SecureSystemStorage.read_workspace_file_bytes

    def no_current(self, path, **kwargs):
        assert not str(path).endswith("current.v1.json")
        return original(self, path, **kwargs)

    monkeypatch.setattr(SecureSystemStorage, "read_workspace_file_bytes", no_current)
    before = {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }
    result = resolve_catalog_event_receipt(workspace=tmp_path, closure=closure)
    assert result["receipt"]["receipt_id"] == "receipt-20260824"
    assert result["source_receipt_ref"] == closure["source_receipt_ref"]
    assert result["catalog_ref"]["path"] == str(catalog.relative_to(tmp_path))
    assert result["mutation_authority"] is False
    assert before == {
        str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "case",
    [
        "unknown_schema",
        "wrong_date",
        "wrong_checkpoint",
        "authority",
        "wrong_owner_receipt",
        "nonempty_owner",
        "ambiguous_catalog",
    ],
)
def test_rehashed_provenance_cannot_waive_native_meaning(tmp_path, case):
    closure, path = fixture(tmp_path)
    if case == "ambiguous_catalog":
        path.with_name("catalog.v2.json").write_bytes(path.read_bytes())
    elif case in {"wrong_owner_receipt", "nonempty_owner"}:
        owner_ref = closure["owner_declaration_ref"]
        owner = json.loads((tmp_path / owner_ref["path"]).read_bytes())
        if case == "wrong_owner_receipt":
            owner["dates"][0]["source_receipt_id"] = "other"
        else:
            owner["dates"][0]["fills"] = [{"event_id": "nonempty"}]
        closure["owner_declaration_ref"] = put(tmp_path, owner_ref["path"], owner)
    else:
        catalog = json.loads(path.read_bytes())
        receipt = deepcopy(catalog["receipts"][-1])
        if case == "unknown_schema":
            receipt["schema_id"] = "unknown-receipt"
        elif case == "wrong_date":
            receipt["created_at"] = "2026-08-25T08:00:00Z"
        elif case == "wrong_checkpoint":
            receipt["active_checkpoint"]["record_id"] = "wrong"
        else:
            receipt["broker_order_trade_authority"] = True
        receipt["content_sha256"] = store.content_sha256(receipt)
        catalog["receipts"][-1] = receipt
        catalog["content_sha256"] = store.content_sha256(catalog)
        path.write_bytes(store.canonical_json_bytes(catalog))
        closure["source_receipt_ref"]["sha256"] = receipt["content_sha256"]
    with pytest.raises((StrategyEventStoreError, store.StrategyRecordStoreError)):
        resolve_catalog_event_receipt(workspace=tmp_path, closure=closure)


@pytest.mark.parametrize(
    "path",
    [
        "catalog:#receipt:r",
        "catalog:g#receipt:",
        "catalog:../g#receipt:r",
        "catalog:g#receipt:r#extra",
        "catalog:g#receipt:r/s",
        "/absolute",
        "a//b",
        "a/../b",
    ],
)
def test_symbolic_and_physical_grammars_are_distinct(path):
    with pytest.raises(StrategyEventStoreError):
        source_ref({"path": path, "sha256": "a" * 64}, symbolic=True)
