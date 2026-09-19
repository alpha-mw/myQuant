"""Freeze the native pre-close book before Store advancement; replay exact custody."""

from datetime import datetime, timezone
from pathlib import Path

from quant_investor.contracts import (
    canonical_json_bytes,
    validate_artifact,
)
from quant_investor.migration.canonical import parse_json_bytes
from quant_investor.strategy_records.store import (
    CATALOG_SCHEMA_V3,
    POINTER_MAX_BYTES,
    content_sha256,
    load_catalog_snapshot_bytes,
)
from quant_investor.strategy_records.holdings import load_holdings_identity
from quant_investor.intelligence.portfolio_state import build_portfolio_state, PORTFOLIO_STATE_KIND
from quant_investor.intelligence.fundamental_time import session_date
from quant_investor.system.storage import SecureSystemStorage
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import DailyJournal
from .research_file_readback import ResearchFileReadback
from quant_investor.strategy_records.close_plan_contracts import validate_plan, portfolio_identity

RECORD_ROOT = "results/strategy_records/CN/aggressive_tech_manufacturing"


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class NativePortfolioSource:
    """Exact native book reader, independent of a not-yet-selected Decision cutoff."""

    def __init__(self, *, workspace, trade_date, store_plan_ref):
        self.workspace = Path(workspace).resolve(strict=True)
        self.root = self.workspace / RECORD_ROOT
        self.files = ResearchFileReadback(str(self.workspace))
        self.plan_ref = validate_ref(store_plan_ref)
        self.plan = parse_json_bytes(
            self.read(self.plan_ref),
            label="portfolio native Store plan",
            require_canonical=False,
        )
        plan = self.plan
        self.plan_version = validate_plan(plan, path=self.plan_ref["path"])
        self.source_pointer_sha, self.source_catalog_sha, self.source_record_id = (
            portfolio_identity(plan)
        )
        expected = (
            f"{RECORD_ROOT}/_record_store/daily_close_transactions/"
            f"{plan.get('transaction_id')}/plan.v{self.plan_version}.json"
        )
        if (
            self.plan_ref["path"] != expected
            or plan.get("schema_id") != f"myquant.cn_official_close_batch_plan.v{self.plan_version}"
            or plan.get("content_sha256") != content_sha256(plan)
            or plan.get("all_or_nothing") is not True
            or plan.get("broker_order_trade_authority") is not False
            or session_date(trade_date).isoformat() not in plan.get("missing_dates", [])
        ):
            raise ContractError("PORTFOLIO_NATIVE_PLAN_BINDING_INVALID")
        if self.plan_version == 2:
            # Fixed helper is already part of the verified native composition.
            # There is no dynamic path selection or unverified import fallback.
            from scripts.cn_official_close_batch import _registered_plan_proof

            _registered_plan_proof(self.root, plan, retained=True)
        journal = DailyJournal(str(self.workspace), trade_date)
        self.prefix = journal.root / "inputs" / "portfolio" / self.plan_ref["sha256"]
        self.pointer_path = str(self.prefix / "pointer.v1.json")
        self.state_path = str(self.prefix / "state.v1.json")

    def read(self, ref):
        ref = validate_ref(ref)
        path, raw, _ = self.files.source_file(ref, code="PORTFOLIO_SOURCE_REF_INVALID")
        if path != self.workspace / ref["path"]:
            raise ContractError("PORTFOLIO_SOURCE_PATH_ALIAS_REJECTED")
        return raw

    def source_fields(self, pointer_ref):
        pointer_ref = validate_ref(pointer_ref)
        if pointer_ref != {"path": self.pointer_path, "sha256": self.source_pointer_sha}:
            raise ContractError("PORTFOLIO_FROZEN_PREIMAGE_MISMATCH")
        pointer, catalog = load_catalog_snapshot_bytes(
            self.root,
            pointer_bytes=self.read(pointer_ref),
            expected_pointer_sha256=self.source_pointer_sha,
        )
        if (
            catalog["schema_id"] != CATALOG_SCHEMA_V3
            or pointer["catalog_sha256"] != self.source_catalog_sha
            or pointer["active_record_id"] != self.source_record_id
        ):
            raise ContractError("PORTFOLIO_NATIVE_CATALOG_PLAN_MISMATCH")
        catalog_ref = {
            "path": f"{RECORD_ROOT}/{pointer['catalog_path']}",
            "sha256": pointer["catalog_sha256"],
        }
        self.read(catalog_ref)
        records = [
            row for row in catalog["records"] if row["record_id"] == pointer["active_record_id"]
        ]
        if len(records) != 1:
            raise ContractError("PORTFOLIO_ACTIVE_RECORD_MISSING")
        record = records[0]
        ledger_ref = {
            "path": f"{RECORD_ROOT}/{record['ledger_path']}",
            "sha256": record["ledger_sha256"],
        }
        manual_ref = {
            "path": f"{RECORD_ROOT}/{record['manual_manifest_path']}",
            "sha256": record["manual_manifest_sha256"],
        }
        self.read(ledger_ref)
        manual = parse_json_bytes(
            self.read(manual_ref), label="portfolio native manual", require_canonical=False
        )
        effective = session_date(manual["valuation_trade_date"])
        if effective != session_date(self.plan["last_official_date"]):
            raise ContractError("PORTFOLIO_NATIVE_EFFECTIVE_DATE_MISMATCH")
        ledger, cash = load_holdings_identity(self.root, record)
        self.files.recheck()
        return dict(
            store_plan_ref=self.plan_ref,
            frozen_pointer_ref=pointer_ref,
            catalog_ref=catalog_ref,
            source_record_id=record["record_id"],
            source_effective_trade_date=effective.strftime("%Y%m%d"),
            source_sealed_at=record["sealed_at"],
            pointer_published_at=pointer["published_at"],
            source_refs=[ledger_ref, manual_ref],
            positions=ledger.to_dict("records"),
            cash=cash,
        )


class PortfolioSource(NativePortfolioSource):
    """Fixed-cutoff native replay; historical and legacy semantics stay unchanged."""

    def __init__(self, *, workspace, trade_date, as_of, store_plan_ref):
        super().__init__(workspace=workspace, trade_date=trade_date, store_plan_ref=store_plan_ref)
        if session_date(as_of[:10]) != session_date(trade_date):
            raise ContractError("PORTFOLIO_NATIVE_PLAN_BINDING_INVALID")
        self.as_of = as_of

    def source_fields(self, pointer_ref):
        return dict(as_of=self.as_of, **super().source_fields(pointer_ref))

    def replay(self, state_ref):
        if validate_ref(state_ref)["path"] != self.state_path:
            raise ContractError("PORTFOLIO_STATE_PATH_MISMATCH")
        state = validate_artifact(self.read(state_ref), expected_kind=PORTFOLIO_STATE_KIND)
        if utc_stamp(state["created_at"]) < utc_stamp(self.plan.get("transaction_planned_at")):
            raise ContractError("PORTFOLIO_CUSTODY_BEFORE_NATIVE_PLAN")
        fields = self.source_fields(state["payload"]["frozen_pointer_ref"])
        rebuilt = build_portfolio_state(created_at=state["created_at"], **fields)
        if rebuilt != state:
            raise ContractError("PORTFOLIO_STATE_REPLAY_MISMATCH")
        self.files.recheck()
        return state


def freeze_portfolio_state(*, journal, store_plan_ref, as_of, retained_pointer_ref=None):
    """Called immediately after native Store planning under the existing day lock."""
    journal._require_lock()
    workspace = journal.storage._io.workspace_root
    source = PortfolioSource(
        workspace=workspace,
        trade_date=journal.trade_date,
        as_of=as_of,
        store_plan_ref=store_plan_ref,
    )
    existing = journal.storage.read(source.state_path)
    if existing is not None:
        ref = {"path": source.state_path, "sha256": existing.byte_sha256}
        return {"state_ref": ref, "state": source.replay(ref)}
    fields = retain_portfolio_source(
        journal=journal, source=source, retained_pointer_ref=retained_pointer_ref
    )
    created_at = _now()
    if utc_stamp(created_at) < utc_stamp(source.plan.get("transaction_planned_at")):
        raise ContractError("PORTFOLIO_CUSTODY_BEFORE_NATIVE_PLAN")
    state = build_portfolio_state(created_at=created_at, **fields)
    source.files.recheck()
    stored = journal.storage.write(source.state_path, canonical_json_bytes(state))
    ref = {"path": source.state_path, "sha256": stored.byte_sha256}
    return {"state_ref": ref, "state": source.replay(ref)}


def retain_portfolio_source(*, journal, source, retained_pointer_ref=None):
    """Retain native preimage custody without selecting a cutoff or writing a state."""
    journal._require_lock()
    workspace = journal.storage._io.workspace_root
    if source.workspace != Path(workspace).resolve(strict=True) or source.prefix != (
        journal.root / "inputs" / "portfolio" / source.plan_ref["sha256"]
    ):
        raise ContractError("PORTFOLIO_CUSTODY_JOURNAL_MISMATCH")
    expected = source.source_pointer_sha
    if source.plan_version == 2:
        required = {
            "path": (
                f"{RECORD_ROOT}/_record_store/daily_close_transactions/"
                f"{source.plan['transaction_id']}/decision-source-pointer.v1.json"
            ),
            "sha256": expected,
        }
        if retained_pointer_ref is not None and validate_ref(retained_pointer_ref) != required:
            raise ContractError("PORTFOLIO_REGISTERED_BASELINE_OVERRIDE_FORBIDDEN")
        retained_pointer_ref = required
    retained = journal.storage.read(source.pointer_path)
    secure = SecureSystemStorage(workspace)
    current_path = RECORD_ROOT + "/_record_store/current.v1.json"
    first = None
    if retained is None:
        if retained_pointer_ref is None:
            first = secure.read_workspace_file_bytes(current_path, maximum_bytes=POINTER_MAX_BYTES)
            if first.byte_sha256 != expected:
                raise ContractError("PORTFOLIO_CURRENT_PREIMAGE_MISMATCH")
            pointer_bytes = first.data
        else:
            required = {
                "path": (
                    f"{RECORD_ROOT}/_record_store/daily_close_transactions/"
                    f"{source.plan['transaction_id']}/"
                    + (
                        "source-pointer.v1.json"
                        if source.plan_version == 1
                        else "decision-source-pointer.v1.json"
                    )
                ),
                "sha256": expected,
            }
            if validate_ref(retained_pointer_ref) != required:
                raise ContractError("PORTFOLIO_ADOPTED_SOURCE_REF_MISMATCH")
            pointer_bytes = source.read(required)
        journal.storage.write(source.pointer_path, pointer_bytes)
    elif retained.byte_sha256 != expected:
        raise ContractError("PORTFOLIO_RETAINED_PREIMAGE_MISMATCH")
    pointer_ref = {"path": source.pointer_path, "sha256": expected}
    fields = source.source_fields(pointer_ref)
    if first is not None:
        last = secure.read_workspace_file_bytes(current_path, maximum_bytes=POINTER_MAX_BYTES)
        if (last.data, last.stat_identity) != (first.data, first.stat_identity):
            raise ContractError("PORTFOLIO_PREIMAGE_CHANGED_DURING_CAPTURE")
    return fields
