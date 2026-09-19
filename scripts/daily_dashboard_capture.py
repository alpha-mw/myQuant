"""Immutable custody for verified native Dashboard bundles and all physical sources.

This is a historical evidence reader, not a current Dashboard selector writer.
Native aliases remain logical provenance; retained bytes are addressed separately.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

from cn_dashboard_common import stable_read, validate_bundle_shape, verify_source_refs
from cn_dashboard_v2 import validate_v2_shape, verify_v2_source_refs
from export_cn_aggressive_dashboard_data import _render_json, require_expected_close_date
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, validate_ref, utc_stamp
from quant_investor.operations.daily_journal import (
    DailyJournal,
    FALSE_AUTHORITY,
    request_identity,
    _false_authority,
)


class DailyDashboardCapture:
    def __init__(self, workspace: str, journal: DailyJournal):
        self.workspace = Path(workspace)
        self.journal = journal
        self.root = journal.root / "dashboard"
        self.receipt_path = str(self.root / "capture.v1.json")

    def _retain(self, raw: bytes) -> dict[str, str]:
        digest = hashlib.sha256(raw).hexdigest()
        path = str(self.root / "objects" / (digest + ".bin"))
        self.journal.storage.write(path, raw)
        return {"path": path, "sha256": digest}

    def _read(self, ref: Mapping[str, str]) -> bytes:
        validate_ref(ref)
        expected = str(self.root / "objects" / (ref["sha256"] + ".bin"))
        if ref["path"] != expected:
            raise ContractError("DASHBOARD_CAPTURE_OBJECT_PATH_INVALID")
        stored = self.journal.storage.read(expected)
        if stored is None or stored.byte_sha256 != ref["sha256"]:
            raise ContractError("DASHBOARD_CAPTURE_OBJECT_MISSING_OR_CHANGED")
        return stored.data

    def _request(self, ref: Mapping[str, str]) -> dict:
        validate_ref(ref)
        stored = self.journal.storage.read(ref["path"])
        if stored is None or stored.byte_sha256 != ref["sha256"]:
            raise ContractError("DASHBOARD_CAPTURE_REQUEST_MISSING_OR_CHANGED")
        document, _ = request_identity(parse_canonical_json_bytes(stored.data))
        if document["node_id"] != "dashboard" or document["trade_date"] != self.journal.trade_date:
            raise ContractError("DASHBOARD_CAPTURE_REQUEST_NODE_OR_DATE_MISMATCH")
        return document

    @staticmethod
    def _sources(v1: dict, v2: dict) -> list[dict]:
        refs = {}
        for raw in v1["source_refs"] + v2["source_refs"]:
            ref = validate_ref(raw)
            previous = refs.setdefault(ref["path"], ref)
            if previous != ref:
                raise ContractError("DASHBOARD_CAPTURE_SOURCE_CONFLICT")
        return [refs[key] for key in sorted(refs)]

    def _date_gate(self, v1: dict, v2: dict) -> None:
        if v2.get("schema_version") != "cn_aggressive_dashboard_history.v1":
            require_expected_close_date(self.workspace, v1, v2, self.journal.trade_date)
            return
        day = self.journal.trade_date
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        ref = v2["historical_binding"]["market_snapshot_ref"]
        manifest = stable_read(self.workspace / ref["path"], self.workspace)
        if (
            manifest.sha256 != ref["sha256"]
            or json.loads(manifest.data).get("latest_complete_trade_date") != day
            or any(
                d != iso
                for d in (
                    v1["latest_data_date"],
                    v1["portfolio"]["performance_end_date"],
                    v2["freshness"]["mark_as_of"],
                )
            )
        ):
            raise ContractError("DASHBOARD_HISTORY_DATE_MISMATCH")

    def capture(self, *, v1: dict, v2: dict, request_ref: Mapping[str, str]) -> dict:
        """Seal only after native validation against the exact current source bytes."""
        self.journal._require_lock()
        request_ref = validate_ref(request_ref)
        existing = self.read(request_ref=request_ref)
        if existing is not None:
            if self._read(existing["v1_ref"]) != _render_json(v1) or self._read(
                existing["v2_ref"]
            ) != _render_json(v2):
                raise ContractError("DASHBOARD_CAPTURE_REPLAY_BODY_MISMATCH")
            return existing
        self._request(request_ref)
        raw_v1, raw_v2 = _render_json(v1), _render_json(v2)
        errors = (
            validate_bundle_shape(v1)
            + validate_v2_shape(v2)
            + verify_source_refs(v1, self.workspace)
            + verify_v2_source_refs(v2, self.workspace, v1_bytes_override=raw_v1)
        )
        if errors:
            raise ContractError("DASHBOARD_NATIVE_VALIDATION_FAILED:" + ";".join(errors))
        self._date_gate(v1, v2)
        retained = []
        for ref in self._sources(v1, v2):
            if ref == v2["canonical_v1_ref"]:
                raw = raw_v1
            else:
                target = self.workspace / ref["path"]
                if target.resolve(strict=True) != target.absolute():
                    raise ContractError("DASHBOARD_CAPTURE_SOURCE_SYMLINK")
                raw = stable_read(target, self.workspace).data
            if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                raise ContractError("DASHBOARD_CAPTURE_SOURCE_CHANGED")
            retained.append({"source_ref": ref, "retained_ref": self._retain(raw)})
        # A moving current alias cannot silently produce a mixed-day archive.
        errors = verify_source_refs(v1, self.workspace) + verify_v2_source_refs(
            v2, self.workspace, v1_bytes_override=raw_v1
        )
        if errors:
            raise ContractError("DASHBOARD_CAPTURE_SOURCE_DRIFT:" + ";".join(errors))
        self._date_gate(v1, v2)
        value = {
            "schema_version": "cn-daily-dashboard-capture.v1",
            "trade_date": self.journal.trade_date,
            "request_ref": request_ref,
            "v1_ref": self._retain(raw_v1),
            "v2_ref": self._retain(raw_v2),
            "sources": retained,
            "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "authority": FALSE_AUTHORITY,
        }
        self.journal.storage.write(self.receipt_path, canonical_json_bytes(value))
        return self.read(request_ref=request_ref)

    def read(self, *, request_ref: Mapping[str, str]) -> dict | None:
        """Validate retained history without consulting mutable current aliases."""
        stored = self.journal.storage.read(self.receipt_path)
        if stored is None:
            return None
        value = parse_canonical_json_bytes(stored.data, label="Dashboard capture")
        if (
            set(value)
            != {
                "schema_version",
                "trade_date",
                "request_ref",
                "v1_ref",
                "v2_ref",
                "sources",
                "captured_at",
                "authority",
            }
            or value["schema_version"] != "cn-daily-dashboard-capture.v1"
            or value["trade_date"] != self.journal.trade_date
            or value["request_ref"] != validate_ref(request_ref)
            or not _false_authority(value["authority"])
        ):
            raise ContractError("DASHBOARD_CAPTURE_CONTRACT_INVALID")
        self._request(request_ref)
        captured = utc_stamp(value["captured_at"])
        if captured > datetime.now(timezone.utc):
            raise ContractError("DASHBOARD_CAPTURE_FUTURE_TIMESTAMP")
        raw_v1, raw_v2 = self._read(value["v1_ref"]), self._read(value["v2_ref"])
        v1, v2 = json.loads(raw_v1), json.loads(raw_v2)
        errors = validate_bundle_shape(v1) + validate_v2_shape(v2)
        if errors or v2["canonical_v1"] != v1:
            raise ContractError("DASHBOARD_CAPTURE_NATIVE_BODY_INVALID")
        day = self.journal.trade_date
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
        if any(
            d != iso
            for d in (
                v1["latest_data_date"],
                v1["portfolio"]["performance_end_date"],
                v2["freshness"]["mark_as_of"],
            )
        ):
            raise ContractError("DASHBOARD_CAPTURE_DATE_MISMATCH")
        expected = self._sources(v1, v2)
        if type(value["sources"]) is not list or len(value["sources"]) != len(expected):
            raise ContractError("DASHBOARD_CAPTURE_SOURCE_SET_MISMATCH")
        for row, ref in zip(value["sources"], expected):
            if set(row) != {"source_ref", "retained_ref"} or row["source_ref"] != ref:
                raise ContractError("DASHBOARD_CAPTURE_SOURCE_SET_MISMATCH")
            raw = self._read(row["retained_ref"])
            if hashlib.sha256(raw).hexdigest() != ref["sha256"]:
                raise ContractError("DASHBOARD_CAPTURE_SOURCE_SHA_MISMATCH")
            if ref == v2["canonical_v1_ref"] and raw != raw_v1:
                raise ContractError("DASHBOARD_CAPTURE_CANONICAL_V1_MISMATCH")
        return value
