"""Immutable, non-authorizing production observation outcome diagnostics.

Raw-close reference returns are not total returns or executable trading returns.
Original signals and observation bytes are never rewritten.
"""

from __future__ import annotations

from collections import Counter
from bisect import bisect_left
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
from statistics import fmean
from typing import Any, Mapping

from quant_investor.contracts import canonical_json_bytes, seal_artifact, validate_artifact
from . import forward_evaluator
from .forward_evaluator import pearson_ic, spearman_rank_ic
from .governance.errors import FactorGovernanceError
from .production_authority import FactorProductionStore
from .production_observation import _matches_inputs, validate_factor_production_observation
from .production_outcome_sources import (
    close_calendar,
    signal_calendar,
    merged_sessions,
    load_market_slice,
    persist_source,
    _file_hash,
)

KIND = "factor.production_observation_outcome"
ROOT = PurePosixPath("results/factors/outcomes")
LABEL_POLICY = {
    "id": "production-observation-raw-close-reference",
    "formula": "close(T+h)/close(T)-1",
    "price_field": "close",
    "adjustment": "RAW_PRICE_RETURN",
    "missing_policy": "KEEP_ORIGINAL_DENOMINATOR_COMPLETE_PAIR_METRICS",
    "execution_policy": "UNAVAILABLE_NOT_BOUND",
    "cost_policy": "UNAVAILABLE_NOT_BOUND",
    "corporate_actions_policy": "ECONOMIC_RETURN_UNAVAILABLE_WITHOUT_EVIDENCE",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    return number if math.isfinite(number) else None


def _text(value: float) -> str:
    return format(value, ".17g")


def decode_generation_signals(values: Mapping[str, Any]) -> dict[str, float]:
    """Decode the actual production generation contract, preserving binary64 values."""
    decoded = {}
    for symbol, encoded in values.items():
        if not isinstance(encoded, str):
            raise FactorGovernanceError("PRODUCTION_SIGNAL_ENCODING_INVALID")
        try:
            value = float.fromhex(encoded)
        except ValueError as exc:
            raise FactorGovernanceError("PRODUCTION_SIGNAL_ENCODING_INVALID") from exc
        if not math.isfinite(value) or value.hex() != encoded:
            raise FactorGovernanceError("PRODUCTION_SIGNAL_ENCODING_INVALID")
        decoded[symbol] = value
    return decoded


def _unavailable(reason: str) -> dict[str, Any]:
    return {"state": "UNAVAILABLE", "value": None, "reasons": [reason]}


def _metric(result: Any) -> dict[str, Any]:
    return {
        "state": "AVAILABLE" if result.available else "UNAVAILABLE",
        "value": _text(result.value) if result.available else None,
        "reasons": list(result.blockers),
    }


def classify_seal(
    *,
    signal_date: str,
    seal_time_upper_bound: str | None,
    registered_at: str,
    next_session_open_utc: str | None = None,
    generation_created_at: str | None = None,
) -> dict[str, Any]:
    close = f"{signal_date[:4]}-{signal_date[4:6]}-{signal_date[6:]}T07:00:00Z"
    proven = seal_time_upper_bound is not None
    if proven:
        for stamp in (seal_time_upper_bound, registered_at):
            datetime.strptime(str(stamp), "%Y-%m-%dT%H:%M:%SZ")
    eligible = bool(proven and str(seal_time_upper_bound) <= close)
    historical = bool(
        next_session_open_utc
        and generation_created_at
        and generation_created_at >= next_session_open_utc
    )
    timely_next_open = bool(
        proven and next_session_open_utc and str(seal_time_upper_bound) < next_session_open_utc
    )
    return {
        "signal_date": signal_date,
        "seal_time_upper_bound": seal_time_upper_bound,
        "seal_time_evidence_semantics": "LOCAL_IMMUTABLE_POINTER_UPPER_BOUND_NOT_EXTERNAL_TIMESTAMP",
        "registration_time": registered_at,
        "signal_session_close_utc": close,
        "cohort": (
            "ORIGINAL_CLOSE_COHORT"
            if eligible
            else (
                "HISTORICAL_REPLAY"
                if historical
                else "POST_CLOSE_LATE" if proven else "SEAL_TIME_UNPROVEN"
            )
        ),
        "next_session_open_utc": next_session_open_utc,
        "sealed_before_next_open": timely_next_open,
        "recovered_registration": bool(
            timely_next_open and registered_at >= str(next_session_open_utc)
        ),
        "prospective_eligible": eligible,
        "registration_after_seal": bool(proven and registered_at > str(seal_time_upper_bound)),
        "reason": (
            "SEALED_BY_ORIGINAL_CLOSE"
            if eligible
            else "ORIGINAL_CLOSE_PRECEDES_PROVEN_SEAL" if proven else "NO_PROVABLE_SEAL_TIME"
        ),
        "executable_origin": None,
        "executable_origin_reason": "EXECUTION_POLICY_NOT_BOUND",
        "authority": "NON_AUTHORIZING",
    }


def evaluate_cross_section(signals: Mapping[str, Any], prices: Mapping[str, Any]) -> dict[str, Any]:
    securities: dict[str, Any] = {}
    joint_signals, joint_returns = [], []
    for symbol, signal in sorted(signals.items()):
        origin, end = prices.get(symbol, (None, None))
        a, b, factor = _number(origin), _number(end), _number(signal)
        reason = (
            "SIGNAL_INVALID"
            if factor is None
            else (
                "ORIGIN_CLOSE_MISSING"
                if origin is None
                else (
                    "ORIGIN_CLOSE_INVALID"
                    if a is None or a <= 0
                    else (
                        "END_CLOSE_MISSING"
                        if end is None
                        else "END_CLOSE_INVALID" if b is None or b <= 0 else None
                    )
                )
            )
        )
        value = None
        if reason is None:
            assert a is not None and b is not None
            value = b / a - 1
        securities[symbol] = {
            "signal": str(signal),
            "origin_close": str(origin) if origin is not None else None,
            "end_close": str(end) if end is not None else None,
            "raw_price_return": _text(value) if value is not None else None,
            "reason": reason,
        }
        if value is not None:
            joint_signals.append(factor)
            joint_returns.append(value)
    size = len(signals)
    signal_count = sum(_number(v) is not None for v in signals.values())
    label_count = sum(r["raw_price_return"] is not None for r in securities.values())
    groups = []
    # Group membership is fixed by the entire original signal cross-section.
    ordered = sorted(
        signals, key=lambda sym: (_number(signals[sym]) is None, _number(signals[sym]) or 0, sym)
    )
    if size >= 5:
        for group in range(5):
            members = ordered[group * size // 5 : (group + 1) * size // 5]
            valid = [
                float(securities[sym]["raw_price_return"])
                for sym in members
                if securities[sym]["raw_price_return"] is not None
            ]
            groups.append(
                {
                    "group": group + 1,
                    "denominator": len(members),
                    "label_count": len(valid),
                    "mean_raw_return": _text(fmean(valid)) if valid else None,
                }
            )
    spread = None
    if groups and all(groups[i]["label_count"] == groups[i]["denominator"] for i in (0, 4)):
        spread = _text(
            float(str(groups[4]["mean_raw_return"])) - float(str(groups[0]["mean_raw_return"]))
        )
    return {
        "denominator": size,
        "signal_count": signal_count,
        "label_count": label_count,
        "joint_count": len(joint_returns),
        "signal_coverage": _text(signal_count / size) if size else "0",
        "label_coverage": _text(label_count / size) if size else "0",
        "joint_coverage": _text(len(joint_returns) / size) if size else "0",
        "missing_reasons": dict(
            Counter(row["reason"] for row in securities.values() if row["reason"])
        ),
        "securities": securities,
        "ic": _metric(pearson_ic(joint_signals, joint_returns)),
        "rank_ic": _metric(spearman_rank_ic(joint_signals, joint_returns)),
        "groups": groups,
        "long_short_spread": {
            "state": "AVAILABLE" if spread is not None else "UNAVAILABLE",
            "value": spread,
            "reason": (
                None
                if spread is not None
                else "ORIGINAL_EXTREME_GROUP_COVERAGE_INCOMPLETE_OR_SMALL"
            ),
        },
        "economic_return": _unavailable("CORPORATE_ACTION_EVIDENCE_NOT_BOUND"),
        "executable_return": _unavailable("EXECUTION_POLICY_AND_TRADABILITY_NOT_BOUND"),
        "cost_result": _unavailable("COST_POLICY_NOT_BOUND_NO_COST_ASSUMED"),
        "turnover": _unavailable("EXECUTABLE_PORTFOLIO_POLICY_NOT_BOUND"),
        "limitations": [
            "DESCRIPTIVE_RAW_PRICE_ONLY",
            "MISSING_SECURITIES_RETAINED",
            "SAMPLE_SIZE_NOT_EFFECTIVENESS_PROOF",
            "NO_PROSPECTIVE_ADMISSION",
        ],
    }


def _registered_files(
    store: FactorProductionStore, root: PurePosixPath, *, maximum: int = 10000
) -> list[PurePosixPath]:
    base = store.workspace_root / str(root)
    if not base.exists():
        return []
    paths = []
    for directory, directories, files in os.walk(base, followlinks=False):
        for name in directories + files:
            if (Path(directory) / name).is_symlink():
                raise FactorGovernanceError("REGISTERED_INVENTORY_SYMLINK")
        for name in files:
            if name.endswith(".json"):
                paths.append(
                    PurePosixPath(
                        (Path(directory) / name).relative_to(store.workspace_root).as_posix()
                    )
                )
                if len(paths) > maximum:
                    raise FactorGovernanceError("REGISTERED_INVENTORY_BOUND")
    return sorted(paths)


def _ref(store: FactorProductionStore, path: PurePosixPath) -> dict[str, str]:
    return {"path": str(path), "sha256": store.read(path).byte_sha256}


def _read_outcome(store: FactorProductionStore, ref: Mapping[str, str]) -> dict[str, Any]:
    stored = store.read(ref["path"])
    if stored.byte_sha256 != ref["sha256"]:
        raise FactorGovernanceError("OUTCOME_REF_SHA_MISMATCH")
    artifact = validate_artifact(stored.data, expected_kind=KIND)
    payload = artifact["payload"]
    if payload["authority"] != "NON_AUTHORIZING" or payload["other_authority"] != "NONE":
        raise FactorGovernanceError("OUTCOME_AUTHORITY_INVALID")
    if (
        payload["state"] != "EVALUATED"
        or payload["horizon"] not in (1, 5, 20, 60)
        or payload["label_policy"] != LABEL_POLICY
    ):
        raise FactorGovernanceError("OUTCOME_POLICY_INVALID")
    body = dict(payload)
    identity = body.pop("outcome_id")
    if identity != "production-outcome-" + _sha(body):
        raise FactorGovernanceError("OUTCOME_IDENTITY_MISMATCH")
    for field in ("observation_ref", "classification_ref", "outcome_source_ref"):
        bound = payload[field]
        if store.read(bound["path"]).byte_sha256 != bound["sha256"]:
            raise FactorGovernanceError("OUTCOME_BOUND_EVIDENCE_CHANGED")
    return artifact


def _source_readback(
    store: FactorProductionStore, body: Mapping[str, Any]
) -> list[tuple[Path, tuple[int, ...]]]:
    source_ref = body["outcome_source_ref"]
    stored = store.read(source_ref["path"])
    if stored.byte_sha256 != source_ref["sha256"]:
        raise FactorGovernanceError("OUTCOME_SOURCE_SHA_MISMATCH")
    source = json.loads(stored.data)
    manifest = source.get("manifest_ref")
    if manifest:
        from .production_outcome_sources import exact_json

        exact_json(store.workspace_root, manifest)
    signatures: list[tuple[Path, tuple[int, ...]]] = []
    for ref in source.get("file_refs", []):
        path = Path(ref["path"])
        if _file_hash(path, store.workspace_root) != ref["sha256"]:
            raise FactorGovernanceError("OUTCOME_SOURCE_CHANGED_BEFORE_PUBLICATION")
        metadata = path.stat()
        signatures.append(
            (path, (metadata.st_ino, metadata.st_size, metadata.st_mtime_ns, metadata.st_ctime_ns))
        )
    return signatures


def _head(store: FactorProductionStore, series: str) -> dict[str, str] | None:
    refs = [_ref(store, path) for path in _registered_files(store, ROOT / series)]
    if not refs:
        return None
    by_sha = {ref["sha256"]: ref for ref in refs}
    predecessors = []
    roots = 0
    for ref in refs:
        payload = _read_outcome(store, ref)["payload"]
        if payload["series_id"] != series:
            raise FactorGovernanceError("OUTCOME_SERIES_MISMATCH")
        previous = payload["supersedes_ref"]
        if previous is None:
            roots += 1
        else:
            if by_sha.get(previous["sha256"]) != previous:
                raise FactorGovernanceError("OUTCOME_REVISION_PREDECESSOR_MISSING")
            predecessors.append(previous["sha256"])
    heads = [ref for ref in refs if ref["sha256"] not in predecessors]
    if roots != 1 or len(heads) != 1 or len(set(predecessors)) != len(predecessors):
        raise FactorGovernanceError("OUTCOME_REVISION_FORK_OR_CYCLE")
    seen = set()
    cursor = heads[0]
    while cursor is not None:
        if cursor["sha256"] in seen:
            raise FactorGovernanceError("OUTCOME_REVISION_CYCLE")
        seen.add(cursor["sha256"])
        cursor = _read_outcome(store, cursor)["payload"]["supersedes_ref"]
    if len(seen) != len(refs):
        raise FactorGovernanceError("OUTCOME_REVISION_DETACHED")
    return heads[0]


def _implementation() -> dict[str, str]:
    paths = [
        Path(__file__),
        Path(forward_evaluator.__file__),
        Path(__file__).with_name("production_outcome_sources.py"),
    ]
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {"sha256": _sha(files), **files}


def _processing_cursor(store: FactorProductionStore, observations: list[PurePosixPath]) -> int:
    stored = store.read_optional("results/factors/outcome-processing.json")
    if stored is None:
        return 0
    state = json.loads(stored.data)
    inventory = state["inventory_ref"]
    if store.read(inventory["path"]).byte_sha256 != inventory["sha256"]:
        raise FactorGovernanceError("OUTCOME_PROCESSING_INDEX_REF_INVALID")
    next_path = state.get("next_observation_path")
    return bisect_left([str(p) for p in observations], next_path) if next_path else 0


def _save_processing_cursor(
    store: FactorProductionStore, *, inventory_ref: Mapping[str, str], next_path: str | None
) -> None:
    raw = canonical_json_bytes(
        {
            "schema_version": "factor-outcome-processing.v1",
            "inventory_ref": dict(inventory_ref),
            "next_observation_path": next_path,
        }
    )
    with store._storage.exclusive_lock("results/factors/.outcomes.lock"):
        path = PurePosixPath("results/factors/outcome-processing.json")
        temporary = path.with_name(
            "outcome-processing-" + hashlib.sha256(raw).hexdigest() + ".pending"
        )
        store.write_exact_once(temporary, raw)
        parent, leaf, _ = store._storage._parent_leaf(path, create=True)
        try:
            os.replace(temporary.name, leaf, src_dir_fd=parent, dst_dir_fd=parent)
            os.fsync(parent)
        finally:
            os.close(parent)


def _publish_outcome(
    store: FactorProductionStore, body: dict[str, Any]
) -> tuple[dict[str, str], bool]:
    series = body["series_id"]
    signatures = _source_readback(store, body)
    # No Factor lock is acquired inside this dedicated, short publication lock.
    with store._storage.exclusive_lock("results/factors/.outcomes.lock"):
        for path, expected in signatures:
            metadata = path.stat()
            if (
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
                metadata.st_ctime_ns,
            ) != expected:
                raise FactorGovernanceError("OUTCOME_SOURCE_DRIFT_AT_PUBLICATION")
        head = _head(store, series)
        if head:
            prior = _read_outcome(store, head)["payload"]
            if prior["source_slice_sha256"] == body["source_slice_sha256"]:
                return head, False
        body = {
            **body,
            "supersedes_ref": head,
            "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        identity = "production-outcome-" + _sha(body)
        artifact = seal_artifact(
            KIND, {"outcome_id": identity, **body}, created_at=body["evaluated_at"]
        )
        stored = store.write_exact_once(
            ROOT / series / f"{identity}.json", canonical_json_bytes(artifact)
        )
        ref = {"path": str(ROOT / series / f"{identity}.json"), "sha256": stored.byte_sha256}
        _read_outcome(store, ref)
        return ref, True


def settle_production_observations(
    *,
    workspace_root: str,
    calendar_receipt: str,
    expected_calendar_sha256: str,
    as_of: str | None = None,
    limit: int = 32,
    cursor: int | None = None,
) -> dict[str, Any]:
    """Evaluate bounded independent horizons, even if today's signal production failed."""
    if not 1 <= limit <= 256 or (cursor is not None and cursor < 0):
        raise FactorGovernanceError("OUTCOME_BATCH_ARGUMENT_INVALID")
    store = FactorProductionStore(workspace_root)
    current_calendar_ref = {"path": calendar_receipt, "sha256": expected_calendar_sha256}
    current_calendar = close_calendar(store.workspace_root, current_calendar_ref)
    target = as_of or current_calendar["target_trade_date"]
    datetime.strptime(target, "%Y%m%d")
    if target > current_calendar["target_trade_date"]:
        raise FactorGovernanceError("OUTCOME_AS_OF_CALENDAR_NOT_COVERED")
    with store._active_lock():
        history = store.read_observation_history()
    by_pointer = {item["factor_pointer_sha256"]: item for item in history}
    observations = _registered_files(store, PurePosixPath("results/factors/observations"))
    cursor = _processing_cursor(store, observations) if cursor is None else cursor
    selected = observations[cursor : cursor + limit]
    jobs, errors = [], []
    calendars: dict[str, list[str]] = {}
    evaluation_prefix = signal_calendar(store, history[-1]) if history else []
    for path in selected:
        try:
            stored = store.read(path)
            observation = validate_factor_production_observation(stored.data)["payload"]
            inputs = by_pointer.get(observation["factor_pointer_sha256"])
            if inputs is None:
                raise FactorGovernanceError("OBSERVATION_POINTER_NOT_IN_VERIFIED_LINEAGE")
            row = next(
                r for r in inputs["factor_rows"] if r["factor_id"] == observation["factor_id"]
            )
            if not _matches_inputs({"payload": observation}, inputs, row):
                raise FactorGovernanceError("OBSERVATION_GENERATION_BINDING_MISMATCH")
            key = inputs["calendar_compilation_ref"]["byte_sha256"]
            if key not in calendars:
                original = signal_calendar(store, inputs)
                if not original or original != [
                    day for day in evaluation_prefix if day <= original[-1]
                ]:
                    raise FactorGovernanceError("SIGNAL_CALENDAR_SUCCESSOR_CONFLICT")
                calendars[key] = merged_sessions(evaluation_prefix, current_calendar)
            sessions = calendars[key]
            if observation["signal_date"] not in sessions:
                raise FactorGovernanceError("OBSERVATION_SESSION_NOT_IN_CALENDAR")
            classification = classify_seal(
                signal_date=observation["signal_date"],
                seal_time_upper_bound=inputs["seal_time_upper_bound"],
                registered_at=observation["registered_at"],
                next_session_open_utc=(
                    datetime.strptime(
                        sessions[sessions.index(observation["signal_date"]) + 1], "%Y%m%d"
                    ).strftime("%Y-%m-%dT01:30:00Z")
                    if sessions.index(observation["signal_date"]) + 1 < len(sessions)
                    else None
                ),
                generation_created_at=inputs.get("generation_created_at"),
            )
            classification.update(
                {
                    "pointer_sha256": inputs["factor_pointer_sha256"],
                    "generation_sha256": inputs["factor_generation_sha256"],
                    "signal_calendar_ref": inputs["calendar_compilation_ref"],
                }
            )
            classification_ref = persist_source(store, classification)
            decoded_signals = decode_generation_signals(
                inputs["signal_values"][observation["factor_id"]]
            )
            jobs.append(
                {
                    "observation": observation,
                    "observation_ref": {"path": str(path), "sha256": stored.byte_sha256},
                    "inputs": inputs,
                    "sessions": sessions,
                    "classification_ref": classification_ref,
                    "classification": classification,
                    "decoded_signals": decoded_signals,
                }
            )
        except Exception as exc:
            errors.append({"observation_path": str(path), "state": "INVALID", "reason": str(exc)})
    impl = _implementation()
    outcomes, new_count = [], 0
    source = None
    if jobs:
        symbols = sorted(
            {
                sym
                for job in jobs
                for sym in job["inputs"]["signal_values"][job["observation"]["factor_id"]]
            }
        )
        source = load_market_slice(
            store.workspace_root,
            symbols=symbols,
            start=min(j["observation"]["signal_date"] for j in jobs),
            end=target,
        )
    for job in jobs:
        observation, inputs, sessions = job["observation"], job["inputs"], job["sessions"]
        origin = observation["signal_date"]
        origin_index = sessions.index(origin)
        signals = inputs["signal_values"][observation["factor_id"]]
        for horizon in observation["planned_horizons"]:
            row = {
                "observation_ref": job["observation_ref"],
                "factor_id": observation["factor_id"],
                "origin_session": origin,
                "horizon": horizon,
            }
            try:
                endpoint = (
                    sessions[origin_index + horizon]
                    if origin_index + horizon < len(sessions)
                    else None
                )
                if endpoint is None or endpoint > target:
                    outcomes.append(
                        {
                            **row,
                            "state": "WAITING",
                            "end_session": endpoint,
                            "reason": "FUTURE_SESSIONS_NOT_ELAPSED",
                        }
                    )
                    continue
                if source is None or str(source["market_date"]) < endpoint:
                    outcomes.append(
                        {
                            **row,
                            "state": "DATA_PENDING",
                            "end_session": endpoint,
                            "reason": "MARKET_ENDPOINT_NOT_AVAILABLE",
                        }
                    )
                    continue
                pairs = {
                    sym: [source["prices"][sym].get(origin), source["prices"][sym].get(endpoint)]
                    for sym in signals
                }
                slice_sha = _sha(
                    {
                        "pairs": pairs,
                        "calendar_sessions": sessions[origin_index : origin_index + horizon + 1],
                    }
                )
                outcome_source = {k: v for k, v in source.items() if k != "prices"}
                outcome_source.update(
                    {
                        "calendar_ref": current_calendar_ref,
                        "evaluation_calendar_prefix_ref": history[-1]["calendar_compilation_ref"],
                        "signal_calendar_ref": inputs["calendar_compilation_ref"],
                        "origin_session": origin,
                        "end_session": endpoint,
                        "pairs": pairs,
                    }
                )
                source_ref = persist_source(store, outcome_source)
                series = _sha(
                    {
                        "observation_sha256": job["observation_ref"]["sha256"],
                        "horizon": horizon,
                        "label_policy": LABEL_POLICY,
                        "source_family": source["source_family"],
                        "implementation": impl["sha256"],
                    }
                )
                body = {
                    "series_id": series,
                    "state": "EVALUATED",
                    "authority": "NON_AUTHORIZING",
                    "other_authority": "NONE",
                    "observation_ref": job["observation_ref"],
                    "horizon": horizon,
                    "label_policy": LABEL_POLICY,
                    "origin_session": origin,
                    "end_session": endpoint,
                    "signal_evidence": {
                        k: inputs[k]
                        for k in (
                            "factor_generation_id",
                            "factor_generation_sha256",
                            "factor_pointer_sha256",
                            "active_factor_rows",
                            "factor_implementation_refs",
                            "factor_policy_ref",
                        )
                    },
                    "classification_ref": job["classification_ref"],
                    "outcome_source_ref": source_ref,
                    "source_slice_sha256": slice_sha,
                    "evaluation_implementation": impl,
                    "diagnostics": evaluate_cross_section(job["decoded_signals"], pairs),
                }
                ref, created = _publish_outcome(store, body)
                new_count += int(created)
                outcomes.append(
                    {
                        **row,
                        "state": "EVALUATED",
                        "end_session": endpoint,
                        "evaluation_ref": ref,
                        "new": created,
                        "metrics": {
                            k: body["diagnostics"][k]
                            for k in (
                                "denominator",
                                "joint_count",
                                "joint_coverage",
                                "ic",
                                "rank_ic",
                                "long_short_spread",
                            )
                        },
                    }
                )
            except Exception as exc:
                outcomes.append({**row, "state": "INVALID", "reason": str(exc)})
    counts = {
        str(h): dict(Counter(r["state"] for r in outcomes if r["horizon"] == h))
        for h in (1, 5, 20, 60)
    }
    summary = {
        "schema_version": "factor-production-diagnostics.v2",
        "authority": "NON_AUTHORIZING",
        "provider_calls": False,
        "as_of": target,
        "market_date": source["market_date"] if source else None,
        "observation_inventory_count": len(observations),
        "processed_observation_count": len(jobs),
        "next_cursor": cursor + limit if cursor + limit < len(observations) else None,
        "new_evaluation_count": new_count,
        "horizon_states": counts,
        "component_classification_scope": "SIGNAL_SEAL_TIMING_ONLY",
        "component_sealed_by_close_count": sum(
            job["classification"]["prospective_eligible"] for job in jobs
        ),
        "cohorts": dict(Counter(job["classification"]["cohort"] for job in jobs)),
        "origin_count": len({job["observation"]["signal_date"] for job in jobs}),
        "timely_sealed_signal_days": len(
            {
                job["observation"]["signal_date"]
                for job in jobs
                if job["classification"]["sealed_before_next_open"]
            }
        ),
        "recovered_registration_count": sum(
            job["classification"]["recovered_registration"] for job in jobs
        ),
        "outcomes": outcomes,
        "errors": errors,
        "paper_comparison": {
            "state": "UNAVAILABLE",
            "reason": "NO_VERIFIED_FACTOR_CONSUMPTION_BINDING",
            "writes": 0,
        },
        "effectiveness_state": "FACTOR_EFFECTIVENESS_INSUFFICIENT_EVIDENCE",
        "implementation": impl,
    }
    from .production_outcome_admission import publish_diagnostics_with_admission

    summary, summary_ref = publish_diagnostics_with_admission(store, summary, jobs)
    _save_processing_cursor(
        store,
        inventory_ref=summary_ref,
        next_path=str(observations[cursor + limit]) if cursor + limit < len(observations) else None,
    )
    return {**summary, "inventory_ref": summary_ref}
