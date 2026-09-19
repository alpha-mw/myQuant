"""Immutable capture of native inactive research compilation, with real custody time.

The native artifact created_at/as_of fields retain research cutoff semantics.
They are never substituted for this capture's actual publication timestamp.
"""

from datetime import datetime, timezone
import hashlib
from typing import Any, Mapping

from quant_investor.contracts import (
    canonical_json_bytes,
    parse_canonical_json_bytes,
    validate_artifact,
)
from quant_investor.intelligence._common import artifact_ref, NO_AUTHORITY
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.intelligence.runtime import compile_evidence
from .daily_contract import ContractError, validate_ref, utc_stamp
from .daily_journal import DailyJournal, FALSE_AUTHORITY, _false_authority

ALLOWED_KINDS = frozenset(
    {
        "daily_research_policy",
        "factor_research_rank",
        "industry_source_projection",
        "industry_assessment",
        "theme_membership_projection",
        "theme_governance_policy",
        "theme_economic_exposure_projection",
        "theme_assessment",
        "company_source_evidence",
        "fundamental_assessment",
        "market_risk_evidence",
        "investment_hypothesis",
        "decision_context",
        "investment_decision",
        "research_portfolio",
        "research_request",
        "research_evaluation",
        "evidence_bundle",
    }
)


def _reference_key(ref: Mapping[str, str]) -> bytes:
    return canonical_json_bytes(dict(ref))


def _artifact_index(artifacts: Any) -> dict:
    if type(artifacts) is not list or not artifacts:
        raise ContractError("RESEARCH_COMPILATION_EMPTY")
    by_ref = {}
    for raw in artifacts:
        artifact = validate_artifact(raw)
        if artifact["kind"] not in ALLOWED_KINDS:
            raise ContractError("RESEARCH_COMPILATION_KIND_FORBIDDEN")
        key = _reference_key(artifact_ref(artifact))
        if key in by_ref:
            raise ContractError("RESEARCH_COMPILATION_DUPLICATE")
        by_ref[key] = artifact
    return by_ref


def _validate_decisions(value: Mapping[str, Any], by_ref: dict) -> None:
    expected_decisions = {
        key for key, artifact in by_ref.items() if artifact["kind"] == "investment_decision"
    }
    observed_decisions = [_reference_key(row["decision_ref"]) for row in value["decisions"]]
    if (
        len(observed_decisions) != len(expected_decisions)
        or set(observed_decisions) != expected_decisions
    ):
        raise ContractError("RESEARCH_COMPILATION_DECISION_SET_MISMATCH")
    for row in value["decisions"]:
        artifact = by_ref.get(_reference_key(row["decision_ref"]))
        if (
            artifact is None
            or artifact["kind"] != "investment_decision"
            or artifact["payload"]["company_code"] != row["company_code"]
            or artifact["payload"]["state"] != row["state"]
        ):
            raise ContractError("RESEARCH_COMPILATION_DECISION_MISMATCH")


def _validate_authority(authority: Any) -> None:
    if (
        type(authority) is not dict
        or set(authority) != set(NO_AUTHORITY)
        or any(v is not False for v in authority.values())
    ):
        raise ContractError("RESEARCH_COMPILATION_AUTHORITY_INVALID")


def validate_compilation(value: Mapping[str, Any], *, trade_date: str) -> dict:
    if (
        value.get("research_only") is not True
        or value.get("production") is not False
        or value.get("run_state") != "INACTIVE"
        or value.get("status") not in {"COMPLETE", "PARTIAL"}
    ):
        raise ContractError("RESEARCH_COMPILATION_AUTHORITY_INVALID")
    _validate_authority(value.get("authority"))
    if utc_stamp(value["as_of"]).strftime("%Y%m%d") != trade_date:
        raise ContractError("RESEARCH_COMPILATION_DATE_MISMATCH")
    by_ref = _artifact_index(value.get("artifacts"))
    bundle = validate_artifact(value["evidence_bundle"])
    evaluation = validate_artifact(value["evaluation"])
    for artifact in (bundle, evaluation):
        if _reference_key(artifact_ref(artifact)) not in by_ref:
            raise ContractError("RESEARCH_COMPILATION_CLOSURE_MISSING")
    try:
        evidence = [by_ref[_reference_key(ref)] for ref in bundle["payload"]["evidence_refs"]]
        rebuilt = compile_evidence(evaluation, evidence=evidence, compiled_at=value["as_of"])
    except (KeyError, TypeError) as exc:
        raise ContractError("RESEARCH_COMPILATION_CLOSURE_MISSING") from exc
    if rebuilt != bundle:
        raise ContractError("RESEARCH_COMPILATION_BUNDLE_MISMATCH")
    if value["strategy_id"] != evaluation["payload"]["strategy_id"]:
        raise ContractError("RESEARCH_COMPILATION_STRATEGY_MISMATCH")
    if value["status"] == "COMPLETE" and bundle["payload"]["blocker_codes"]:
        raise ContractError("RESEARCH_COMPILATION_FALSE_COMPLETE")
    _validate_decisions(value, by_ref)
    return dict(value)


class ResearchCapture:
    def __init__(self, journal: DailyJournal):
        self.journal = journal
        self.reader = SecureSystemStorage(journal.storage._io.workspace_root)

    def _request_data(self, ref: Mapping[str, str]) -> dict:
        validate_ref(ref)
        value = self.reader.read_workspace_file_bytes(ref["path"], maximum_bytes=8 * 1024 * 1024)
        if value.byte_sha256 != ref["sha256"]:
            raise ContractError("RESEARCH_CAPTURE_REQUEST_SHA_MISMATCH")
        document = parse_canonical_json_bytes(value.data, label="native compilation request")
        if document.get("schema_version") in {
            "cn-daily-research-request.v2",
            "cn-daily-research-request.v3",
        }:
            from .research_request import load_research_request

            return load_research_request(
                workspace=self.journal.storage._io.workspace_root, reference=dict(ref)
            )["document"]
        return document

    def _root(self, request_ref: Mapping[str, str]) -> str:
        validate_ref(request_ref)
        return str(self.journal.root / "research" / request_ref["sha256"])

    def read(self, request_ref: Mapping[str, str]) -> tuple[dict, dict] | None:
        self._request_data(request_ref)
        prefix = self._root(request_ref)
        captured = self.journal.storage.read(prefix + "/capture.v1.json")
        if captured is None:
            return None
        manifest = parse_canonical_json_bytes(captured.data, label="research capture")
        if (
            set(manifest)
            != {
                "schema_version",
                "trade_date",
                "request_ref",
                "result_ref",
                "artifact_refs",
                "captured_at",
                "research_cutoff",
                "research_status",
                "recovered_custody",
                "native_timestamps_are_publication_proof",
                "authority",
            }
            or manifest.get("schema_version") != "cn-daily-research-capture.v1"
            or manifest.get("request_ref") != dict(request_ref)
            or manifest.get("trade_date") != self.journal.trade_date
            or not _false_authority(manifest.get("authority"))
            or manifest.get("native_timestamps_are_publication_proof") is not False
            or type(manifest.get("recovered_custody")) is not bool
        ):
            raise ContractError("RESEARCH_CAPTURE_BINDING_INVALID")
        utc_stamp(manifest["captured_at"])
        ref = manifest["result_ref"]
        if ref["path"] != prefix + "/result.json":
            raise ContractError("RESEARCH_CAPTURE_RESULT_PATH_INVALID")
        stored = self.journal.storage.read(ref["path"])
        if stored is None or stored.byte_sha256 != ref["sha256"]:
            raise ContractError("RESEARCH_CAPTURE_RESULT_SHA_MISMATCH")
        value = validate_compilation(
            parse_canonical_json_bytes(stored.data, label="research compilation"),
            trade_date=self.journal.trade_date,
        )
        expected = self._artifacts(value, prefix, write=False)
        if manifest["artifact_refs"] != expected:
            raise ContractError("RESEARCH_CAPTURE_ARTIFACT_SET_MISMATCH")
        if (
            manifest["research_cutoff"] != value["as_of"]
            or manifest["research_status"] != value["status"]
            or utc_stamp(manifest["captured_at"]) < utc_stamp(value["as_of"])
        ):
            raise ContractError("RESEARCH_CAPTURE_TIMING_OR_STATUS_INVALID")
        return manifest, value

    def _artifacts(self, value: dict, prefix: str, *, write: bool) -> list[dict]:
        refs = []
        for artifact in value["artifacts"]:
            raw = canonical_json_bytes(artifact)
            sha = hashlib.sha256(raw).hexdigest()
            path = prefix + "/artifacts/" + sha + ".json"
            if write:
                self.journal.storage.write(path, raw)
            else:
                stored = self.journal.storage.read(path)
                if stored is None or stored.data != raw:
                    raise ContractError("RESEARCH_CAPTURE_ARTIFACT_DRIFT")
            refs.append(
                {
                    "kind": artifact["kind"],
                    "artifact_id": artifact["artifact_id"],
                    "path": path,
                    "sha256": sha,
                }
            )
        return sorted(refs, key=lambda row: (row["kind"], row["artifact_id"]))

    def publish(self, request_ref: Mapping[str, str], result: Mapping[str, Any]) -> dict:
        self.journal._require_lock()
        value = validate_compilation(result, trade_date=self.journal.trade_date)
        request = self._request_data(request_ref)
        if (
            request.get("as_of") != value["as_of"]
            or request.get("strategy_id") != value["strategy_id"]
        ):
            raise ContractError("RESEARCH_CAPTURE_REQUEST_RESULT_MISMATCH")
        if datetime.now(timezone.utc) < utc_stamp(value["as_of"]):
            raise ContractError("RESEARCH_CAPTURE_CUTOFF_IN_FUTURE")
        prefix = self._root(request_ref)
        previous = self.read(request_ref)
        if previous is not None:
            if canonical_json_bytes(previous[1]) != canonical_json_bytes(value):
                raise ContractError("RESEARCH_CAPTURE_IMMUTABLE_CONFLICT")
            return previous[0]
        result_path = prefix + "/result.json"
        # A prior result without a capture is crash custody, never an earlier
        # publication time. Exact bytes can be adopted with the current clock.
        recovered = self.journal.storage.read(result_path) is not None
        stored = self.journal.storage.write(result_path, canonical_json_bytes(value))
        refs = self._artifacts(value, prefix, write=True)
        manifest = {
            "schema_version": "cn-daily-research-capture.v1",
            "trade_date": self.journal.trade_date,
            "request_ref": dict(request_ref),
            "result_ref": {"path": result_path, "sha256": stored.byte_sha256},
            "artifact_refs": refs,
            "captured_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "research_cutoff": value["as_of"],
            "research_status": value["status"],
            "recovered_custody": recovered,
            "native_timestamps_are_publication_proof": False,
            "authority": FALSE_AUTHORITY,
        }
        self.journal.storage.write(prefix + "/capture.v1.json", canonical_json_bytes(manifest))
        readback = self.read(request_ref)
        if readback is None:
            raise ContractError("RESEARCH_CAPTURE_READBACK_MISSING")
        return readback[0]
