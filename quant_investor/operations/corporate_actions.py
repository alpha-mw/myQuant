"""Financial event closure and non-executable adjustment warnings; no position writer."""

from pathlib import Path
import hashlib
import json
import math
from typing import Mapping

import pandas as pd

from quant_investor.contracts import canonical_json_bytes
from quant_investor.strategy_records.event_store import (
    load_historical_generation,
    load_frozen_generation,
)
from quant_investor.strategy_records.event_receipts import (
    read_event_source,
    resolve_catalog_event_receipt,
)
from quant_investor.strategy_records.event_contracts import SYMBOLIC_RECEIPT
from quant_investor.system.storage import SecureSystemStorage
from .daily_contract import ContractError, GRAPH_SHA256, NodeState, validate_ref
from .daily_journal import DailyJournal, FALSE_AUTHORITY, _validate_day
from .daily_runner import NativeOutcome, Probe

EVENT_ROOT = "results/strategy_records/CN/aggressive_tech_manufacturing/_event_store"


def retain_event_pointer(*, journal, event_pointer_sha256):
    """Use the existing exact native historical lookup and custody path; no event write."""
    journal._require_lock()
    events = load_historical_generation(
        journal.storage._io.workspace_root / EVENT_ROOT,
        expected_pointer_sha256=event_pointer_sha256,
    )
    path = str(journal.root / "inputs" / f"event-pointer-{event_pointer_sha256}.json")
    journal.storage.write(path, events["pointer_bytes"])
    return {"path": path, "sha256": event_pointer_sha256}


class CorporateActionEvidence:
    """Read-only native event and frozen adjustment projection; no journal writer."""

    def __init__(self, *, workspace: str, trade_date: str, recipe: dict):
        self.workspace = Path(workspace)
        self.trade_date = trade_date
        self.event_sha = recipe["event_pointer_ref"]["sha256"]
        self.event_pointer_ref = recipe["event_pointer_ref"]
        self.previous = recipe["previous_trade_date"]
        self.market_refs = recipe["market_refs"]
        self.calendar_ref = recipe["calendar_ref"]
        self.market_snapshot_ref = recipe["market_snapshot_ref"]
        self.reader = SecureSystemStorage(workspace)
        self._bytes(recipe["event_pointer_ref"])

    def _bytes(self, ref: Mapping[str, str]) -> bytes:
        return read_event_source(self.workspace, ref)

    def _events(self) -> dict:
        return load_frozen_generation(
            self.workspace / EVENT_ROOT,
            pointer_bytes=self._bytes(self.event_pointer_ref),
            expected_pointer_sha256=self.event_sha,
        )

    def _adjustments(self) -> list[dict]:
        valid_window = self._calendar_window()
        reader = self._market_reader()
        rows = []
        for symbol, ref in sorted(self.market_refs.items()):
            validate_ref(ref)
            frame = pd.DataFrame()
            state = "NON_EXECUTABLE_MISSING_ADJUSTMENT_EVIDENCE"
            if reader is not None:
                selected = reader.resolve_symbol_path(symbol)
                if selected is None or selected != self.workspace / ref["path"]:
                    raise ContractError("CORPORATE_MARKET_FRAME_PATH_MISMATCH")
                reader._assert_path_has_no_symlink(
                    selected,
                    boundary=reader.data_root,
                    label="corporate Market frame",
                    require_exists=True,
                )
                frame = reader._read_strict_catalog_parquet(
                    selected,
                    table_meta={"sha256": ref["sha256"]},
                    logical_table="corporate adjustment",
                )
            if valid_window and reader is not None and {"trade_date", "adj_factor"} <= set(frame):
                dates = frame["trade_date"].astype(str).str.replace("-", "", regex=False)
                before = frame.loc[dates == self.previous, "adj_factor"]
                after = frame.loc[dates == self.trade_date, "adj_factor"]
                if (
                    len(before) == len(after) == 1
                    and pd.notna(before.iloc[0])
                    and pd.notna(after.iloc[0])
                    and all(
                        math.isfinite(float(v)) and float(v) > 0
                        for v in (before.iloc[0], after.iloc[0])
                    )
                ):
                    state = (
                        "NO_ADJUSTMENT_FACTOR_CHANGE"
                        if before.iloc[0] == after.iloc[0]
                        else "NON_EXECUTABLE_CORPORATE_ACTION_UNRECONCILED"
                    )
            rows.append({"symbol": symbol, "threshold_state": state, "evidence_ref": ref})
        return rows

    def _market_reader(self):
        from quant_investor.market.market_data_reader import MarketDataReader

        if self.market_snapshot_ref is None:
            return None
        self._bytes(self.market_snapshot_ref)
        ref = {
            "path": Path(self.market_snapshot_ref["path"]).relative_to("data").as_posix(),
            "sha256": self.market_snapshot_ref["sha256"],
        }
        reader = MarketDataReader(
            market="CN",
            data_root=self.workspace / "data",
            mode_policy="strict",
            frozen_snapshot_ref=ref,
        )
        snapshot = reader.snapshot()
        if (
            snapshot.get("healthy") is not True
            or snapshot.get("latest_complete_trade_date") != self.trade_date
        ):
            raise ContractError("CORPORATE_MARKET_SNAPSHOT_INVALID")
        return reader

    def _calendar_window(self) -> bool:
        if self.calendar_ref is None or self.previous is None:
            return False
        value = json.loads(self._bytes(self.calendar_ref))
        days = value.get("ordered_open_dates")
        if type(days) is not list:
            raise ContractError("CORPORATE_CALENDAR_BINDING_INVALID")
        for day in days:
            _validate_day(day)
        if (
            value.get("schema_version") != "cn-close-session-receipt.v1"
            or value.get("status") != "TARGET_AUTHORIZED"
            or type(days) is not list
            or days != sorted(set(days))
            or self.trade_date not in days
        ):
            raise ContractError("CORPORATE_CALENDAR_BINDING_INVALID")
        prior = [d for d in days if d < self.trade_date]
        if not prior or prior[-1] != self.previous:
            raise ContractError("CORPORATE_PREVIOUS_SESSION_MISMATCH")
        return True

    def project(self) -> dict | None:
        events = self._events()
        day = self.trade_date
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        matching = [row for row in events["closures"] if row["trade_date"] == iso]
        if not matching:
            return None
        if len(matching) != 1:
            raise ContractError("CORPORATE_EVENT_DATE_DUPLICATE")
        closure = matching[0]
        for ref in (
            closure["policy_ref"],
            closure["owner_declaration_ref"],
        ):
            if ref is not None:
                self._bytes(ref)
        receipt = closure["source_receipt_ref"]
        if receipt is not None:
            if SYMBOLIC_RECEIPT.fullmatch(receipt["path"]):
                resolve_catalog_event_receipt(workspace=self.workspace, closure=closure)
            else:
                self._bytes(receipt)
                from quant_investor.strategy_records.daily_event_source import (
                    validate_daily_closure_source,
                )

                validate_daily_closure_source(workspace=self.workspace, closure=closure)
        return {
            "trade_date": day,
            "event_pointer_sha256": self.event_sha,
            "event_closure": closure,
            "event_generation_ref": {
                "path": EVENT_ROOT + "/" + events["pointer"]["generation"]["path"],
                "sha256": events["generation_sha256"],
            },
            "financial_event_state": "VERIFIED_CLOSED_EMPTY",
            "adjustment_checks": self._adjustments(),
            "threshold_state": "NOT_CONFIGURED" if not self.market_refs else "SEE_PER_SYMBOL",
            "threshold_anchor_mutation": False,
            "authority": FALSE_AUTHORITY,
        }


class CorporateActionAdapter(CorporateActionEvidence):
    resume_safe = True

    def __init__(
        self,
        *,
        workspace: str,
        journal: DailyJournal,
        event_pointer_sha256: str,
        release_ref: Mapping[str, str],
        previous_trade_date: str | None = None,
        market_refs: Mapping[str, dict] | None = None,
        calendar_ref: Mapping[str, str] | None = None,
        market_snapshot_ref: Mapping[str, str] | None = None,
    ):
        self.workspace = Path(workspace)
        self.journal = journal
        self.trade_date = journal.trade_date
        self.event_sha = event_pointer_sha256
        self.release_ref = validate_ref(release_ref)
        self.previous = previous_trade_date
        self.calendar_ref = None if calendar_ref is None else validate_ref(calendar_ref)
        self.market_snapshot_ref = (
            None if market_snapshot_ref is None else validate_ref(market_snapshot_ref)
        )
        self.market_refs = dict(market_refs or {})
        self.reader = SecureSystemStorage(workspace)
        self.recipe_ref: dict[str, str] | None = None

    def prepare(self) -> None:
        self.journal._require_lock()
        self.event_pointer_ref = retain_event_pointer(
            journal=self.journal, event_pointer_sha256=self.event_sha
        )
        recipe = {
            "event_pointer_ref": self.event_pointer_ref,
            "previous_trade_date": self.previous,
            "market_refs": self.market_refs,
            "calendar_ref": self.calendar_ref,
            "market_snapshot_ref": self.market_snapshot_ref,
        }
        raw = canonical_json_bytes(recipe)
        sha = hashlib.sha256(raw).hexdigest()
        path = str(self.journal.root / "inputs" / (sha + ".json"))
        self.journal.storage.write(path, raw)
        self.recipe_ref = {"path": path, "sha256": sha}

    def template(self) -> dict:
        if self.recipe_ref is None:
            raise ContractError("CORPORATE_CONTEXT_NOT_PREPARED")
        return {
            "schema_version": "cn-daily-node-request.v1",
            "trade_date": self.journal.trade_date,
            "node_id": "corporate_action_recon",
            "graph_sha256": GRAPH_SHA256,
            "release_ref": self.release_ref,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "policy_refs": {},
            "input_refs": {"recipe": self.recipe_ref},
        }

    def _expected(self, request: dict) -> dict | None:
        observed = {
            **request,
            "input_refs": {
                k: v for k, v in request["input_refs"].items() if not k.startswith("upstream.")
            },
        }
        if observed != self.template():
            raise ContractError("CORPORATE_REQUEST_MISMATCH")
        if self.recipe_ref is None:
            raise ContractError("CORPORATE_CONTEXT_NOT_PREPARED")
        recipe = json.loads(self._bytes(self.recipe_ref))
        if (
            any(
                recipe.get(key) != value
                for key, value in {
                    "previous_trade_date": self.previous,
                    "market_refs": self.market_refs,
                    "calendar_ref": self.calendar_ref,
                    "market_snapshot_ref": self.market_snapshot_ref,
                }.items()
            )
            or recipe["event_pointer_ref"]["sha256"] != self.event_sha
        ):
            raise ContractError("CORPORATE_RECIPE_BINDING_MISMATCH")
        self._bytes(recipe["event_pointer_ref"])
        self._bytes(self.release_ref)
        return self.project()

    def probe(self, request: dict) -> Probe:
        expected = self._expected(request)
        if expected is None:
            return Probe(NativeOutcome(NodeState.BLOCKED, {}, "INPUT_MISSING"))
        raw = canonical_json_bytes(expected)
        sha = hashlib.sha256(raw).hexdigest()
        path = str(self.journal.root / "corporate-actions" / (sha + ".json"))
        stored = self.journal.storage.read(path)
        if stored is None:
            return Probe(None, safe_to_execute=True)
        if stored.data != raw:
            raise ContractError("CORPORATE_CHECK_IMMUTABLE_CONFLICT")
        return Probe(
            NativeOutcome(
                NodeState.SUCCEEDED,
                {
                    "financial_events": {"path": path, "sha256": sha},
                    "event_generation": expected["event_generation_ref"],
                },
            )
        )

    def execute(self, request: dict) -> None:
        self.journal._require_lock()
        expected = self._expected(request)
        if expected is None:
            raise ContractError("CORPORATE_EVENT_CLOSURE_MISSING")
        raw = canonical_json_bytes(expected)
        sha = hashlib.sha256(raw).hexdigest()
        self.journal.storage.write(
            str(self.journal.root / "corporate-actions" / (sha + ".json")), raw
        )
