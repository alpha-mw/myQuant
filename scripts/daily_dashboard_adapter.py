"""Native historical Dashboard DAG adapter with immutable render intent and custody.

The current-page publication step is separate: this adapter never rolls back its
selector while catching up older trade dates.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo

from cn_dashboard_common import stable_read, build_bundle
from cn_dashboard_v2 import build_v2_bundle
from export_cn_aggressive_dashboard_data import _render_json, _expected_output_paths
from scripts.daily_dashboard_capture import DailyDashboardCapture
from scripts.daily_dashboard_history import committed_dashboard_store
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.operations.daily_contract import ContractError, GRAPH_SHA256, NodeState
from quant_investor.operations.daily_contract import validate_ref, utc_stamp
from quant_investor.operations.daily_journal import DailyJournal, request_identity
from quant_investor.operations.daily_runner import NativeOutcome, Probe

RECORD_ROOT = "results/strategy_records/CN/aggressive_tech_manufacturing"


def dashboard_publication_bytes(
    *, workspace: Path, v1: dict, v2: dict, raw_v1: bytes, raw_v2: bytes
) -> dict[Path, bytes]:
    """Render the native publication set from an already validated retained pair."""
    from cn_dashboard_v2_selector import build_selector, render_json, render_js
    from export_cn_aggressive_dashboard_data import _render_js, _render_v2_js

    selector = build_selector(
        attempt_id=v2["publication_attempt_id"],
        status="UPDATED",
        updated_at=v2["generated_at"],
        reason="refresh_completed",
        v2_content_sha256=v2["content_sha256"],
    )
    paths = _expected_output_paths(workspace)
    return dict(
        zip(
            paths,
            [
                raw_v1,
                _render_js(v1),
                raw_v2,
                _render_v2_js(v2),
                render_json(selector),
                render_js(selector),
            ],
        )
    )


class HistoricalDashboardAdapter:
    resume_safe = True
    historical_mode = True

    def __init__(
        self,
        *,
        workspace: str,
        journal: DailyJournal,
        release_ref: Mapping[str, str],
        plan_ref: Mapping[str, str],
        market_ref: Mapping[str, str],
        benchmark_ref: Mapping[str, str],
        risk_free_ref: Mapping[str, str],
    ):
        self.workspace = Path(workspace)
        self.journal = journal
        self.release_ref = validate_ref(release_ref)
        self.refs = {
            name: validate_ref(ref)
            for name, ref in {
                "store_plan": plan_ref,
                "market": market_ref,
                "benchmark": benchmark_ref,
                "risk_free": risk_free_ref,
            }.items()
        }
        self.capture = DailyDashboardCapture(workspace, journal)
        self.recipe_ref: dict[str, str] | None = None
        self.recipe: dict | None = None

    def _source(self, ref: Mapping[str, str]) -> bytes:
        validate_ref(ref)
        path = self.workspace / ref["path"]
        if path.resolve(strict=True) != path.absolute():
            raise ContractError("DASHBOARD_ADAPTER_SOURCE_SYMLINK")
        value = stable_read(path, self.workspace)
        if value.sha256 != ref["sha256"]:
            raise ContractError("DASHBOARD_ADAPTER_SOURCE_SHA_MISMATCH")
        return value.data

    def prepare(self) -> None:
        self.journal._require_lock()
        identity: dict = {
            "release_ref": self.release_ref,
            "sources": self.refs,
            "trade_date": self.journal.trade_date,
            "historical_mode": self.historical_mode,
        }
        key = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
        path = str(self.journal.root / "inputs" / ("dashboard-intent-" + key + ".json"))
        stored = self.journal.storage.read(path)
        if stored is None:
            value = {
                **identity,
                "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            raw = canonical_json_bytes(value)
            self.journal.storage.write(path, raw)
        else:
            raw = stored.data
            value = parse_canonical_json_bytes(raw)
            if set(value) != {*identity, "generated_at"} or any(
                value[k] != v for k, v in identity.items()
            ):
                raise ContractError("DASHBOARD_ADAPTER_INTENT_MISMATCH")
        utc_stamp(value["generated_at"])
        self.recipe, self.recipe_ref = value, {
            "path": path,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }

    def template(self) -> dict:
        if self.recipe_ref is None:
            raise ContractError("DASHBOARD_ADAPTER_NOT_PREPARED")
        return {
            "schema_version": "cn-daily-node-request.v1",
            "trade_date": self.journal.trade_date,
            "node_id": "dashboard",
            "graph_sha256": GRAPH_SHA256,
            "release_ref": self.release_ref,
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "policy_refs": {},
            "input_refs": {**self.refs, "render_intent": self.recipe_ref},
        }

    def _request(self, request: dict) -> tuple[dict, dict[str, str]]:
        document, key = request_identity(request)
        observed = {
            **document,
            "input_refs": {
                k: v for k, v in document["input_refs"].items() if not k.startswith("upstream.")
            },
        }
        if observed != self.template():
            raise ContractError("DASHBOARD_ADAPTER_REQUEST_MISMATCH")
        if self.recipe_ref is None or self.recipe is None:
            raise ContractError("DASHBOARD_ADAPTER_NOT_PREPARED")
        if self._source(self.recipe_ref) != canonical_json_bytes(self.recipe):
            raise ContractError("DASHBOARD_ADAPTER_INTENT_MISMATCH")
        self._source(self.release_ref)
        return document, {
            "path": str(self.journal.root / "inputs" / ("dashboard-request-" + key + ".json")),
            "sha256": key,
        }

    def probe(self, request: dict) -> Probe:
        _, ref = self._request(request)
        value = self.capture.read(request_ref=ref)
        if value is None:
            for source in self.refs.values():
                self._source(source)
            return Probe(None, safe_to_execute=True)
        stored = self.journal.storage.read(self.capture.receipt_path)
        if stored is None:
            raise ContractError("DASHBOARD_ADAPTER_CAPTURE_DISAPPEARED")
        return Probe(
            NativeOutcome(
                NodeState.SUCCEEDED,
                {
                    "capture": {"path": self.capture.receipt_path, "sha256": stored.byte_sha256},
                    "v1": value["v1_ref"],
                    "v2": value["v2_ref"],
                },
            )
        )

    def execute(self, request: dict) -> None:
        self.journal._require_lock()
        document, request_ref = self._request(request)
        if self.capture.read(request_ref=request_ref) is not None:
            return
        for source in self.refs.values():
            self._source(source)
        self.journal.storage.write(request_ref["path"], canonical_json_bytes(document))
        day = self.journal.trade_date
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        records = self.workspace / RECORD_ROOT
        selected, _ = committed_dashboard_store(
            project_root=self.workspace,
            record_root=records,
            plan_ref=self.refs["store_plan"],
            valuation_date=iso,
        )
        market = self.refs["market"]
        reader = MarketDataReader(
            data_root=self.workspace / "data",
            frozen_snapshot_ref={
                "path": Path(market["path"]).relative_to("data").as_posix(),
                "sha256": market["sha256"],
            },
        )
        if not self.historical_mode:
            from quant_investor.strategy_records.store import load_registered_catalog

            if load_registered_catalog(records) != selected:
                raise ContractError("DASHBOARD_CURRENT_STORE_NOT_SELECTED_COMMIT")
            current_reader = MarketDataReader(data_root=self.workspace / "data")
            if current_reader.snapshot().get("snapshot_id") != reader.snapshot().get("snapshot_id"):
                raise ContractError("DASHBOARD_CURRENT_MARKET_NOT_SELECTED_SNAPSHOT")
            reader = current_reader
        if self.recipe is None:
            raise ContractError("DASHBOARD_ADAPTER_NOT_PREPARED")
        now = utc_stamp(self.recipe["generated_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
        v1 = build_bundle(
            project_root=self.workspace,
            record_root=records,
            benchmark_path=self.workspace / self.refs["benchmark"]["path"],
            risk_free_path=self.workspace / self.refs["risk_free"]["path"],
            generated_at=now.isoformat(timespec="seconds"),
            today=now.date(),
            historical_close_plan_ref=self.refs["store_plan"] if self.historical_mode else None,
            historical_valuation_date=iso if self.historical_mode else None,
        )
        v2 = build_v2_bundle(
            project_root=self.workspace,
            v1_bundle=v1,
            v1_json_path=_expected_output_paths(self.workspace)[0],
            record_root=records,
            generation_local_date=now.date(),
            generated_at=now.isoformat(timespec="seconds"),
            publication_attempt_id="dashboard-v2-dag-" + day + "-" + request_ref["sha256"][:16],
            market_reader=reader,
            v1_json_bytes_override=_render_json(v1),
            historical_close_plan_ref=self.refs["store_plan"] if self.historical_mode else None,
            historical_valuation_date=iso if self.historical_mode else None,
        )
        for source in self.refs.values():
            self._source(source)
        self.capture.capture(v1=v1, v2=v2, request_ref=request_ref)


class CurrentDashboardAdapter(HistoricalDashboardAdapter):
    """Publish the current native pair/selector, then seal exact readback evidence."""

    historical_mode = False

    @property
    def publication_path(self) -> str:
        return str(self.journal.root / "dashboard/publication.v1.json")

    def _publication_bytes(self, request: dict) -> dict[Path, bytes]:
        import json

        _, ref = self._request(request)
        captured = self.capture.read(request_ref=ref)
        if captured is None:
            raise ContractError("DASHBOARD_PUBLICATION_CAPTURE_MISSING")
        raw_v1, raw_v2 = self.capture._read(captured["v1_ref"]), self.capture._read(
            captured["v2_ref"]
        )
        v1, v2 = json.loads(raw_v1), json.loads(raw_v2)
        if v2["schema_version"] != "cn_aggressive_dashboard.v2":
            raise ContractError("DASHBOARD_CURRENT_SCHEMA_REQUIRED")
        return dashboard_publication_bytes(
            workspace=self.workspace, v1=v1, v2=v2, raw_v1=raw_v1, raw_v2=raw_v2
        )

    def probe(self, request: dict) -> Probe:
        base = super().probe(request)
        stored = self.journal.storage.read(self.publication_path)
        if stored is None:
            if base.outcome is None:
                return base
            return Probe(None, recovery_only=True)
        _, ref = self._request(request)
        value = parse_canonical_json_bytes(stored.data)
        if (
            set(value) != {"schema_version", "request_ref", "published_at", "files"}
            or value["schema_version"] != "cn-daily-dashboard-publication.v1"
            or value["request_ref"] != ref
        ):
            raise ContractError("DASHBOARD_PUBLICATION_RECEIPT_INVALID")
        if utc_stamp(value["published_at"]) > datetime.now(timezone.utc):
            raise ContractError("DASHBOARD_PUBLICATION_TIMESTAMP_INVALID")
        expected = self._publication_bytes(request)
        rows = value["files"]
        if type(rows) is not list or len(rows) != len(expected):
            raise ContractError("DASHBOARD_PUBLICATION_FILE_SET_INVALID")
        for row, (path, raw) in zip(rows, expected.items()):
            if set(row) != {"path", "retained_ref"} or row["path"] != str(
                path.relative_to(self.workspace)
            ):
                raise ContractError("DASHBOARD_PUBLICATION_FILE_SET_INVALID")
            if self.capture._read(row["retained_ref"]) != raw:
                raise ContractError("DASHBOARD_PUBLICATION_READBACK_INVALID")
        if base.outcome is None:
            raise ContractError("DASHBOARD_PUBLICATION_WITHOUT_CAPTURE")
        return Probe(
            NativeOutcome(
                NodeState.SUCCEEDED,
                {
                    **base.outcome.output_refs,
                    "publication": {"path": self.publication_path, "sha256": stored.byte_sha256},
                },
            )
        )

    def execute(self, request: dict) -> None:
        import json
        from cn_dashboard_v2_selector import publish_selector
        from export_cn_aggressive_dashboard_data import (
            publish_bundle_pair,
            require_expected_close_date,
        )
        from cn_dashboard_common import verify_source_refs
        from cn_dashboard_v2 import verify_v2_source_refs

        self.journal._require_lock()
        if self.journal.storage.read(self.publication_path) is not None:
            self.probe(request)
            return
        super().execute(request)
        expected = self._publication_bytes(request)
        paths = list(expected)
        v1, v2, selector = [json.loads(expected[paths[i]]) for i in (0, 2, 4)]
        # Never roll current UI backward after source heads have advanced.
        for ref in self.refs.values():
            self._source(ref)
        errors = verify_source_refs(v1, self.workspace) + verify_v2_source_refs(
            v2, self.workspace, v1_bytes_override=expected[paths[0]]
        )
        if errors:
            raise ContractError("DASHBOARD_PUBLICATION_SOURCE_DRIFT:" + ";".join(errors))
        require_expected_close_date(self.workspace, v1, v2, self.journal.trade_date)
        publish_bundle_pair(
            v1_bundle=v1,
            v2_bundle=v2,
            v1_json_path=paths[0],
            v1_js_path=paths[1],
            v2_json_path=paths[2],
            v2_js_path=paths[3],
            project_root=self.workspace,
        )
        publish_selector(
            selector,
            json_path=paths[4],
            js_path=paths[5],
            project_root=self.workspace,
            js_first=False,
        )
        rows = []
        for path, raw in expected.items():
            if stable_read(path, self.workspace).data != raw:
                raise ContractError("DASHBOARD_PUBLICATION_READBACK_INVALID")
            rows.append(
                {
                    "path": str(path.relative_to(self.workspace)),
                    "retained_ref": self.capture._retain(raw),
                }
            )
        _, ref = self._request(request)
        value = {
            "schema_version": "cn-daily-dashboard-publication.v1",
            "request_ref": ref,
            "published_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": rows,
        }
        self.journal.storage.write(self.publication_path, canonical_json_bytes(value))
