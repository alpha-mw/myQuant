"""Read-only resolution of native catalog no-action receipt references."""

from pathlib import Path
import hashlib
import os
import stat

from . import store
from .event_contracts import SYMBOLIC_RECEIPT, source_ref, instant, event_date
from .event_store import EVENT_DIMENSIONS, StrategyEventStoreError
from .receipts import validate_no_action_receipt

RECORD_ROOT = "results/strategy_records/CN/aggressive_tech_manufacturing"


def read_event_source(workspace, reference):
    """Native repository policies may be 0644; refuse aliases and writable sources."""
    ref = source_ref(reference)
    root = Path(workspace).resolve(strict=True)
    path = root / ref["path"]
    if path.resolve(strict=True) != path:
        raise StrategyEventStoreError("event source symlink/path alias rejected")
    mode = path.stat()
    if mode.st_uid != os.geteuid() or stat.S_IMODE(mode.st_mode) not in {0o400, 0o600, 0o644}:
        raise StrategyEventStoreError("event source ownership/mode invalid")
    raw, _ = store._read_regular(path, max_bytes=64 * 1024 * 1024, label="event source")
    if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
        raise StrategyEventStoreError("event source SHA mismatch")
    return raw


def _owner_row(closure, policy, declaration, receipt_id):
    if (
        policy.get("schema_id") != "myquant.cn_daily_official_close_policy.v1"
        or policy.get("revoked_at") is not None
        or type(policy.get("policy_id")) is not str
        or not policy["policy_id"]
        or policy.get("broker_order_trade_authority") is not False
        or policy.get("actual_holdings_mutation_authority") is not False
        or declaration.get("schema_id")
        != "myquant.cn_official_close_retrospective_owner_declaration.v1"
        or declaration.get("policy_id") != policy.get("policy_id")
        or declaration.get("strategy_label") != "aggressive_tech_manufacturing"
        or type(declaration.get("owner")) is not str
        or not declaration["owner"].strip()
        or declaration.get("retrospective_empty_event_closure_authorized") is not True
        or any(
            declaration.get(key) is not False
            for key in (
                "actual_holdings_mutation_authority",
                "cash_mutation_authority",
                "broker_order_trade_authority",
            )
        )
    ):
        raise StrategyEventStoreError("event retrospective owner declaration invalid")
    authorized = instant(declaration.get("authorized_at"), label="owner authorization")
    if (
        not instant(closure["cutoff_at"], label="cutoff")
        <= authorized
        <= instant(closure["sealed_at"], label="seal")
    ):
        raise StrategyEventStoreError("event owner authorization chronology invalid")
    rows = declaration.get("dates")
    if type(rows) is not list or not rows:
        raise StrategyEventStoreError("event owner date rows missing")
    dates = []
    for row in rows:
        if type(row) is not dict or any(row.get(name) != [] for name in EVENT_DIMENSIONS):
            raise StrategyEventStoreError("event owner dimensions are not closed empty")
        dates.append(event_date(row.get("trade_date")).isoformat())
    if len(dates) != len(set(dates)):
        raise StrategyEventStoreError("event owner date duplicated")
    matches = [row for row in rows if row["trade_date"] == closure["trade_date"]]
    if len(matches) != 1 or matches[0].get("source_receipt_id") != receipt_id:
        raise StrategyEventStoreError("event owner receipt/date binding invalid")


def resolve_catalog_event_receipt(*, workspace, closure):
    from quant_investor.system.storage import SecureSystemStorage
    import json

    ref = source_ref(closure["source_receipt_ref"], symbolic=True)
    match = SYMBOLIC_RECEIPT.fullmatch(ref["path"])
    if match is None:
        raise StrategyEventStoreError("event catalog receipt reference invalid")
    generation, receipt_id = match.groups()
    root = Path(workspace).resolve(strict=True) / RECORD_ROOT
    candidates = [
        (version, f"_record_store/catalogs/{generation}/catalog.v{version}.json")
        for version in (1, 2, 3)
        if os.path.lexists(root / f"_record_store/catalogs/{generation}/catalog.v{version}.json")
    ]
    if len(candidates) != 1:
        raise StrategyEventStoreError("event catalog receipt generation absent or ambiguous")
    version, relative = candidates[0]
    reader = SecureSystemStorage(root)
    stored = reader.read_workspace_file_bytes(relative, maximum_bytes=store.CATALOG_MAX_BYTES)
    catalog = store._parse_canonical(stored.data, label="event receipt catalog")
    if catalog["schema_id"] != f"myquant.strategy_record_catalog.v{version}":
        raise StrategyEventStoreError("event receipt catalog version mismatch")
    store._validate_catalog(catalog, generation_id=generation)
    store._validate_external_catalog_bindings(root, catalog)
    rows = [
        row
        for row in catalog["receipts"]
        if isinstance(row, dict) and row.get("receipt_id") == receipt_id
    ]
    if len(rows) != 1:
        raise StrategyEventStoreError("event catalog receipt is not unique")
    receipt = rows[0]
    record_id = receipt.get("active_record_id")
    checkpoint = store._active_closure(catalog["records"], record_id)
    validated = validate_no_action_receipt(
        receipt,
        receipt_id=receipt_id,
        expected_sha=ref["sha256"],
        record_id=record_id,
        checkpoint=checkpoint,
        trade_date=closure["trade_date"],
    )
    policy_raw = read_event_source(workspace, closure["policy_ref"])
    owner_raw = read_event_source(workspace, closure["owner_declaration_ref"])
    _owner_row(closure, json.loads(policy_raw), json.loads(owner_raw), receipt_id)
    after = reader.read_workspace_file_bytes(relative, maximum_bytes=store.CATALOG_MAX_BYTES)
    if (after.data, after.stat_identity) != (stored.data, stored.stat_identity):
        raise StrategyEventStoreError("event receipt catalog changed during read")
    if (
        read_event_source(workspace, closure["policy_ref"]) != policy_raw
        or read_event_source(workspace, closure["owner_declaration_ref"]) != owner_raw
    ):
        raise StrategyEventStoreError("event owner sources changed during read")
    return {
        "receipt": validated,
        "source_receipt_ref": ref,
        "catalog_ref": {"path": f"{RECORD_ROOT}/{relative}", "sha256": stored.byte_sha256},
        "mutation_authority": False,
    }
