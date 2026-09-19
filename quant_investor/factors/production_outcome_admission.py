"""Daily-ledger admission for bounded raw-close diagnostics, never trading authority."""

from collections import Counter
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
from statistics import fmean

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.prospective_storage import ledger_path
from quant_investor.operations.daily_contract import validate_ref
from quant_investor.system.errors import SystemNotFound, SystemStorageError
from quant_investor.system.storage import SecureSystemStorage
from .governance.errors import FactorGovernanceError
from .production_observation import validate_factor_production_observation
from .production_outcome_sources import persist_source

ADMISSION_SCHEMA = "factor-production-daily-admission.v1"
OOS_SCHEMA = "factor-production-oos-summary.v1"
SCOPE = "LOCAL_COORDINATOR_AVAILABILITY"
HORIZONS = (1, 5, 20, 60)
METRICS = ("ic", "rank_ic", "long_short_spread")
REASON_STATES = {
    "DAILY_COMPLETION_MISSING": "UNCONFIRMED",
    "UNKNOWN_LEGACY": "INELIGIBLE",
    "SYNTHETIC_EVIDENCE": "INELIGIBLE",
    "RETROSPECTIVE_RECOMPUTE": "INELIGIBLE",
    "LATE_REGISTERED": "INELIGIBLE",
    "DAILY_NATIVE_EVIDENCE_INVALID": "INVALID",
    "DAILY_OBSERVATION_BINDING_MISMATCH": "INVALID",
    "DAILY_EVIDENCE_CHANGED": "INVALID",
}
ADMISSION_FIELDS = frozenset(
    {
        "schema_version",
        "authority",
        "factor_admission",
        "eligibility_scope",
        "native_replay_required",
        "observation_ref",
        "factor_id",
        "signal_date",
        "factor_pointer_sha256",
        "factor_generation_sha256",
        "completion_ref",
        "ledger_ref",
        "state",
        "reason_codes",
        "prospective",
        "classification",
        "synthetic",
        "recomputed",
        "implementation",
        "evidence_validation_scope",
    }
)


def _sha(value):
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _source_ref(value):
    sha = _sha(value)
    return {"path": f"results/factors/outcome_sources/{sha}.json", "sha256": sha}


def _implementation():
    from quant_investor.operations import (
        outcome_native_replay,
        native_bridge,
        completed_handoff_snapshot,
        archived_handoff_context,
    )
    from . import production_observation

    paths = {
        "factors/production_outcome_admission.py": Path(__file__),
        "factors/production_observation.py": Path(production_observation.__file__),
        "operations/outcome_native_replay.py": Path(outcome_native_replay.__file__),
        "operations/native_bridge.py": Path(native_bridge.__file__),
        "operations/completed_handoff_snapshot.py": Path(completed_handoff_snapshot.__file__),
        "operations/archived_handoff_context.py": Path(archived_handoff_context.__file__),
    }
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in paths.items()}
    value = {
        "evaluator": hashes,
        "runner_protocol_sha256": outcome_native_replay.runner_protocol_sha256(),
        "native_replay": None,
    }
    return {**value, "sha256": _sha(value)}


def _bind_implementation(value, native_identity):
    body = {key: deepcopy(value[key]) for key in ("evaluator", "runner_protocol_sha256")}
    body["native_replay"] = deepcopy(native_identity)
    return {**body, "sha256": _sha(body)}


def _read_optional_source(reader, path):
    try:
        return reader.read_workspace_file_bytes(path, maximum_bytes=32 * 1024 * 1024).data
    except FileNotFoundError:
        return None
    except SystemStorageError as exc:
        if type(exc) is SystemNotFound or (
            type(exc) is SystemStorageError and isinstance(exc.__cause__, FileNotFoundError)
        ):
            return None
        raise


class _EvidenceChanged(FactorGovernanceError):
    pass


def _with_reason(value, reason):
    value = dict(value)
    if reason == "DAILY_COMPLETION_MISSING":
        value["evidence_validation_scope"] = "NO_DAILY_COMPLETION"
    elif reason in {"DAILY_NATIVE_EVIDENCE_INVALID", "DAILY_EVIDENCE_CHANGED"}:
        value["evidence_validation_scope"] = "UNCONFIRMED_NATIVE_EVIDENCE"
    return {
        **value,
        "state": "ELIGIBLE" if reason is None else REASON_STATES[reason],
        "reason_codes": [] if reason is None else [reason],
        "prospective": reason is None,
    }


def _base(job, implementation):
    observation = job["observation"]
    return {
        "schema_version": ADMISSION_SCHEMA,
        "authority": "NON_AUTHORIZING",
        "factor_admission": False,
        "eligibility_scope": SCOPE,
        "native_replay_required": True,
        "observation_ref": validate_ref(job["observation_ref"]),
        **{
            key: observation[key]
            for key in (
                "factor_id",
                "signal_date",
                "factor_pointer_sha256",
                "factor_generation_sha256",
            )
        },
        "completion_ref": None,
        "ledger_ref": None,
        "classification": None,
        "synthetic": None,
        "recomputed": None,
        "implementation": implementation,
        "evidence_validation_scope": "UNCONFIRMED_NATIVE_EVIDENCE",
    }


def _bound_observation(day, job):
    observation, ledger = job["observation"], day["ledger"]
    alias = observation["factor_alias"]
    node = {"LOW": "low_observation", "W80": "w80_observation"}[alias]
    core = ledger["core_timing"]
    full = ledger["node_custody"]["nodes"][node]
    return (
        day["completion"]["trade_date"]
        == observation["signal_date"]
        == ledger["trade_date"]
        == core["effective_trade_date"]
        and core["factor_pointer_ref"]["sha256"] == observation["factor_pointer_sha256"]
        and core["observation_registration"][alias]
        == {
            "registered_at": observation["registered_at"],
            "observation_ref": job["observation_ref"],
        }
        and core["node_custody"][node]["terminal_ref"]
        == full["terminal_ref"]
        == day["completion"]["node_terminal_refs"][node]
        and full["output_refs"].get(alias) == job["observation_ref"]
    )


def _classification_reason(available):
    if available["ledger_ref"] is None:
        return "UNKNOWN_LEGACY"
    if available["synthetic"]:
        return "SYNTHETIC_EVIDENCE"
    if available["recomputed"] or available["classification"] == "RETROSPECTIVE_RECOMPUTE":
        return "RETROSPECTIVE_RECOMPUTE"
    if available["classification"] == "LATE_REGISTERED":
        return "LATE_REGISTERED"
    if available["classification"] == "UNKNOWN_LEGACY":
        return "UNKNOWN_LEGACY"
    if available["prospective"] is not True:
        raise FactorGovernanceError("OUTCOME_DAILY_AVAILABILITY_INVALID")
    return None


class AdmissionBatch:
    """One bounded invocation. Pinned absence and invalidations never reset here."""

    def __init__(self, store, jobs):
        self.store = store
        self.workspace = str(store.workspace_root)
        self.reader = SecureSystemStorage(self.workspace)
        self.jobs = {job["observation_ref"]["path"]: job for job in jobs}
        if len(self.jobs) != len(jobs):
            raise FactorGovernanceError("OUTCOME_OOS_DUPLICATE_OBSERVATION")
        self.implementation = _implementation()
        self.days = {}
        self.pins = {}
        self.invalid_dates = set()
        self.invalid_observations = set()
        self.invalid_native_observations = set()
        self.revision = 0
        self.projections = {}
        for key, job in sorted(self.jobs.items()):
            self.projections[key] = self._observe(job)
        self.recheck()

    def _read(self, path, *, kind, day):
        if kind == "completion":
            stored = DailyJournal(self.workspace, day).storage.read(path)
            return None if stored is None else stored.data
        return _read_optional_source(self.reader, path)

    def _invalidate(self, *, kind, day, observation):
        if kind == "observation":
            self.invalid_native_observations.add(observation)
        target, key = (
            (self.invalid_observations, observation)
            if kind in {"observation", "outcome"}
            else (self.invalid_dates, day)
        )
        if key not in target:
            target.add(key)
            self.revision += 1

    def _pin(self, path, *, kind, day, observation=None, reference=None):
        raw = self._read(path, kind=kind, day=day)
        prior = self.pins.get(path)
        if prior is None:
            self.pins[path] = {"raw": raw, "kind": kind, "day": day, "observation": observation}
        elif raw != prior["raw"]:
            self._invalidate(kind=kind, day=day, observation=observation)
            raise _EvidenceChanged("OUTCOME_DAILY_EVIDENCE_CHANGED")
        if reference is not None and (
            raw is None or hashlib.sha256(raw).hexdigest() != validate_ref(reference)["sha256"]
        ):
            self._invalidate(kind=kind, day=day, observation=observation)
            raise _EvidenceChanged("OUTCOME_DAILY_EVIDENCE_CHANGED")
        return raw

    def _load_day(self, day):
        if day in self.days:
            return self.days[day]
        value = {"completion_ref": None, "ledger_ref": None, "reason": None}
        self.days[day] = value
        path = str(DailyJournal(self.workspace, day).root / "completion.v1.json")
        try:
            raw = self._pin(path, kind="completion", day=day)
            if raw is None:
                value["reason"] = "DAILY_COMPLETION_MISSING"
                return value
            ref = {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
            value["completion_ref"] = ref
            value["completion"] = parse_canonical_json_bytes(raw)
            from quant_investor.operations.outcome_native_replay import (
                replay_outcome_daily_evidence,
            )

            available = replay_outcome_daily_evidence(
                workspace=self.workspace, trade_date=day, completion_ref=ref
            )
            self._load_ledger(value, available, day=day)
            self.recheck()
        except _EvidenceChanged:
            value["reason"] = "DAILY_EVIDENCE_CHANGED"
        except Exception:
            value["reason"] = "DAILY_NATIVE_EVIDENCE_INVALID"
        return value

    def _load_ledger(self, value, available, *, day):
        if (
            available["completion_ref"] != value["completion_ref"]
            or available["trade_date"] != day
            or available["validation_scope"]
            not in {"FULL_NATIVE_EOD_AVAILABILITY", "RECORDED_EOD_WITHOUT_NATIVE_RELEASE"}
        ):
            raise FactorGovernanceError("OUTCOME_DAILY_REPLAY_BINDING_INVALID")
        value["available"] = available
        value["snapshot"] = available["snapshot"]
        value["ledger_ref"] = available["ledger_ref"]
        ref = available["ledger_ref"]
        if ref is None:
            if value["completion"]["schema_version"] != "cn-daily-eod-completion.v1":
                raise FactorGovernanceError("OUTCOME_DAILY_LEDGER_MISSING")
            return
        if ref["path"] != ledger_path(day) or value["completion"]["prospective_ledger_ref"] != ref:
            raise FactorGovernanceError("OUTCOME_DAILY_LEDGER_BINDING_INVALID")
        raw = self._pin(ref["path"], kind="ledger", day=day, reference=ref)
        ledger = parse_canonical_json_bytes(raw)
        for name in ("classification", "prospective", "synthetic", "recomputed"):
            if canonical_json_bytes(ledger[name]) != canonical_json_bytes(available[name]):
                raise FactorGovernanceError("OUTCOME_DAILY_LEDGER_RESULT_MISMATCH")
        value["ledger"] = ledger

    def _observe(self, job):
        base = _base(job, self.implementation)
        observation, ref = job["observation"], job["observation_ref"]
        try:
            raw = self._pin(
                ref["path"],
                kind="observation",
                day=observation["signal_date"],
                observation=ref["path"],
                reference=ref,
            )
            if validate_factor_production_observation(raw)["payload"] != observation:
                raise FactorGovernanceError("OUTCOME_ORIGINAL_OBSERVATION_CHANGED")
            day = self._load_day(observation["signal_date"])
            base.update({name: day[name] for name in ("completion_ref", "ledger_ref")})
            if day["reason"] is not None:
                return _with_reason(base, day["reason"])
            available = day["available"]
            base["implementation"] = _bind_implementation(
                self.implementation, available["native_identity"]
            )
            base["evidence_validation_scope"] = available["validation_scope"]
            base.update(
                {name: available[name] for name in ("classification", "synthetic", "recomputed")}
            )
            if available["ledger_ref"] is not None and not _bound_observation(day, job):
                return _with_reason(base, "DAILY_OBSERVATION_BINDING_MISMATCH")
            return _with_reason(base, _classification_reason(available))
        except _EvidenceChanged:
            return _with_reason(base, "DAILY_EVIDENCE_CHANGED")
        except Exception:
            return _with_reason(base, "DAILY_NATIVE_EVIDENCE_INVALID")

    def recheck(self):
        for day, value in self.days.items():
            ledger = value.get("ledger")
            if ledger is not None and any(
                row["observation_ref"]["path"] in self.invalid_native_observations
                for row in ledger["core_timing"]["observation_registration"].values()
            ):
                self._invalidate(kind="ledger", day=day, observation=None)
            snapshot = value.get("snapshot")
            if snapshot is not None:
                try:
                    snapshot.recheck()
                except Exception:
                    self._invalidate(kind="snapshot", day=day, observation=None)
        for path, pin in self.pins.items():
            try:
                raw = self._read(path, kind=pin["kind"], day=pin["day"])
                changed = raw != pin["raw"]
            except Exception:
                changed = True
            if changed:
                self._invalidate(**{key: pin[key] for key in ("kind", "day", "observation")})
                if pin["kind"] == "observation":
                    ledger = self.days.get(pin["day"], {}).get("ledger")
                    if ledger is not None and any(
                        row["observation_ref"]["path"] == path
                        for row in ledger["core_timing"]["observation_registration"].values()
                    ):
                        self._invalidate(kind="ledger", day=pin["day"], observation=None)
        if _implementation() != self.implementation:
            for key in self.jobs:
                self._invalidate(kind="observation", day=None, observation=key)

    def projection(self, key):
        value = deepcopy(self.projections[key])
        if key in self.invalid_observations or value["signal_date"] in self.invalid_dates:
            value = _with_reason(value, "DAILY_EVIDENCE_CHANGED")
        return value

    def evidence(self, outcomes, *, publish):
        self.recheck()
        revision = self.revision
        projections = {key: self.projection(key) for key in sorted(self.jobs)}
        rows = []
        for value in projections.values():
            self.recheck()
            ref = persist_source(self.store, value) if publish else _source_ref(value)
            rows.append(
                {
                    **{
                        name: value[name]
                        for name in (
                            "observation_ref",
                            "factor_id",
                            "signal_date",
                            "state",
                            "reason_codes",
                        )
                    },
                    "admission_ref": ref,
                }
            )
        grouped = _aggregate(self, outcomes, projections)
        self.recheck()
        if revision != self.revision:
            raise _EvidenceChanged("OUTCOME_DAILY_EVIDENCE_CHANGED")
        eligible = [value for value in projections.values() if value["state"] == "ELIGIBLE"]
        return {
            "schema_version": OOS_SCHEMA,
            "authority": "NON_AUTHORIZING",
            "factor_admission": False,
            "eligibility_scope": SCOPE,
            "native_replay_required": True,
            "batch_scope": "CURRENT_CURSOR_BATCH",
            "admissions": rows,
            "eligible_observation_count": len(eligible),
            "eligible_signal_day_count": len({value["signal_date"] for value in eligible}),
            "exclusions_by_reason": dict(
                sorted(
                    Counter(
                        reason for value in projections.values() for reason in value["reason_codes"]
                    ).items()
                )
            ),
            "by_factor": grouped,
        }


def _metric(values):
    if not values:
        return {"state": "UNAVAILABLE", "sample_count": 0, "mean": None}
    value = fmean(values)
    if not math.isfinite(value):
        raise FactorGovernanceError("OUTCOME_OOS_METRIC_INVALID")
    return {"state": "AVAILABLE", "sample_count": len(values), "mean": format(value, ".17g")}


def _metric_value(value):
    if type(value) is not dict or value.get("state") not in {"AVAILABLE", "UNAVAILABLE"}:
        raise FactorGovernanceError("OUTCOME_OOS_METRIC_INVALID")
    if value["state"] == "UNAVAILABLE":
        return None
    raw = value.get("value")
    if type(raw) is not str:
        raise FactorGovernanceError("OUTCOME_OOS_METRIC_INVALID")
    try:
        result = float(raw)
    except ValueError as exc:
        raise FactorGovernanceError("OUTCOME_OOS_METRIC_INVALID") from exc
    if not math.isfinite(result):
        raise FactorGovernanceError("OUTCOME_OOS_METRIC_INVALID")
    return result


def _outcome_payload(batch, row, admission):
    from .production_outcomes import _read_outcome

    ref = validate_ref(row["evaluation_ref"])
    artifact = _read_outcome(batch.store, ref)
    payload = artifact["payload"]
    observation = batch.jobs[admission["observation_ref"]["path"]]["observation"]
    if (
        payload["observation_ref"] != admission["observation_ref"]
        or row["observation_ref"] != admission["observation_ref"]
        or payload["origin_session"] != admission["signal_date"]
        or row["origin_session"] != admission["signal_date"]
        or payload["horizon"] != row["horizon"]
        or row["factor_id"] != observation["factor_id"]
        or payload["signal_evidence"]["factor_pointer_sha256"]
        != observation["factor_pointer_sha256"]
        or payload["signal_evidence"]["factor_generation_sha256"]
        != observation["factor_generation_sha256"]
    ):
        raise FactorGovernanceError("OUTCOME_OOS_AGGREGATE_BINDING_INVALID")
    batch._pin(
        ref["path"],
        kind="outcome",
        day=admission["signal_date"],
        observation=admission["observation_ref"]["path"],
        reference=ref,
    )
    return payload


def _aggregate(batch, outcomes, admissions):
    groups = {
        factor: {
            str(h): {"refs": [], "origins": set(), "values": {m: [] for m in METRICS}}
            for h in HORIZONS
        }
        for factor in sorted({job["observation"]["factor_id"] for job in batch.jobs.values()})
    }
    seen, seen_origins = set(), set()
    for row in outcomes:
        admission = admissions.get(row["observation_ref"]["path"])
        if row["state"] != "EVALUATED" or admission is None or admission["state"] != "ELIGIBLE":
            continue
        ref = validate_ref(row["evaluation_ref"])
        key = ref["path"], ref["sha256"]
        origin = row["factor_id"], row["origin_session"], row["horizon"]
        if key in seen or origin in seen_origins or row["horizon"] not in HORIZONS:
            raise FactorGovernanceError("OUTCOME_OOS_DUPLICATE_OR_INVALID_HORIZON")
        seen.add(key)
        seen_origins.add(origin)
        payload = _outcome_payload(batch, row, admission)
        group = groups[row["factor_id"]][str(row["horizon"])]
        group["refs"].append(ref)
        group["origins"].add(payload["origin_session"])
        for metric in METRICS:
            number = _metric_value(payload["diagnostics"][metric])
            if number is not None:
                group["values"][metric].append(number)
    return {
        factor: {
            horizon: {
                "eligible_outcome_refs": sorted(
                    group["refs"], key=lambda ref: (ref["path"], ref["sha256"])
                ),
                "eligible_origin_count": len(group["origins"]),
                "metrics": {metric: _metric(group["values"][metric]) for metric in METRICS},
            }
            for horizon, group in horizons.items()
        }
        for factor, horizons in groups.items()
    }


def publish_diagnostics_with_admission(store, summary, jobs):
    """One correction only; caller advances the processing cursor after success."""
    batch = AdmissionBatch(store, jobs)
    for attempt in range(2):
        try:
            oos = batch.evidence(summary["outcomes"], publish=True)
            revision = batch.revision
            value = {**summary, "oos_evidence": oos}
            ref = persist_source(store, value)
            batch.recheck()
            if revision == batch.revision:
                return value, ref
        except _EvidenceChanged:
            pass
    raise FactorGovernanceError("OUTCOME_DAILY_EVIDENCE_UNSTABLE")


def _read_admission_source(store, reference):
    ref = validate_ref(reference)
    stored = store.read(ref["path"])
    if stored is None or stored.byte_sha256 != ref["sha256"]:
        raise FactorGovernanceError("OUTCOME_OOS_SOURCE_SHA_MISMATCH")
    value = parse_canonical_json_bytes(stored.data)
    if _source_ref(value) != ref:
        raise FactorGovernanceError("OUTCOME_OOS_SOURCE_PATH_INVALID")
    return value


def _read_projection_job(store, reference):
    value = _read_admission_source(store, reference)
    if (
        type(value) is not dict
        or set(value) != ADMISSION_FIELDS
        or value["schema_version"] != ADMISSION_SCHEMA
        or not _valid_implementation(value["implementation"])
    ):
        raise FactorGovernanceError("OUTCOME_OOS_ADMISSION_CONTRACT_INVALID")
    ref = validate_ref(value["observation_ref"])
    stored = store.read(ref["path"])
    if stored is None or stored.byte_sha256 != ref["sha256"]:
        raise FactorGovernanceError("OUTCOME_OOS_OBSERVATION_CHANGED")
    observation = validate_factor_production_observation(stored.data)["payload"]
    return value, {"observation": observation, "observation_ref": ref}


def _valid_implementation(value):
    if type(value) is not dict or set(value) != {
        "evaluator",
        "runner_protocol_sha256",
        "native_replay",
        "sha256",
    }:
        return False
    current = _implementation()
    if (
        value["evaluator"] != current["evaluator"]
        or value["runner_protocol_sha256"] != current["runner_protocol_sha256"]
    ):
        return False
    native = value["native_replay"]
    if native is not None and not _valid_native_identity(native):
        return False
    return value == _bind_implementation(current, value["native_replay"])


def _valid_native_identity(value):
    if type(value) is not dict or set(value) != {
        "release_install_ref",
        "final_commit",
        "final_tree",
        "installed_code_manifest_sha256",
        "repository_root",
        "operation",
    }:
        return False
    try:
        validate_ref(value["release_install_ref"])
        validate_ref({"path": "manifest.json", "sha256": value["installed_code_manifest_sha256"]})
        return (
            value["operation"] == "completion_replay"
            and all(
                type(value[key]) is str and re.fullmatch(r"[0-9a-f]{40}", value[key])
                for key in ("final_commit", "final_tree")
            )
            and type(value["repository_root"]) is str
            and Path(value["repository_root"]).is_absolute()
        )
    except (ValueError, TypeError):
        return False


def read_daily_admission(*, store, admission_ref):
    """Rebuild one saved projection through native evidence; never trust its flag."""
    value, job = _read_projection_job(store, admission_ref)
    batch = AdmissionBatch(store, [job])
    batch.recheck()
    expected = batch.projection(job["observation_ref"]["path"])
    if (
        canonical_json_bytes(value) != canonical_json_bytes(expected)
        or _read_admission_source(store, admission_ref) != value
    ):
        raise FactorGovernanceError("OUTCOME_OOS_ADMISSION_STALE")
    return {"admission_ref": admission_ref, "projection": value, "native_readback_validated": True}


def read_oos_summary(*, store, summary_ref):
    """Reconstruct bounded OOS groups from native admissions and immutable outcomes."""
    summary = _read_admission_source(store, summary_ref)
    if summary.get("schema_version") != "factor-production-diagnostics.v2":
        raise FactorGovernanceError("OUTCOME_OOS_SUMMARY_V2_REQUIRED")
    jobs = []
    for row in summary["oos_evidence"]["admissions"]:
        _, job = _read_projection_job(store, row["admission_ref"])
        jobs.append(job)
    batch = AdmissionBatch(store, jobs)
    expected = batch.evidence(summary["outcomes"], publish=False)
    revision = batch.revision
    batch.recheck()
    if (
        revision != batch.revision
        or canonical_json_bytes(expected) != canonical_json_bytes(summary["oos_evidence"])
        or _read_admission_source(store, summary_ref) != summary
    ):
        raise FactorGovernanceError("OUTCOME_OOS_SUMMARY_STALE")
    return {"summary_ref": summary_ref, "oos_evidence": expected, "native_readback_validated": True}
