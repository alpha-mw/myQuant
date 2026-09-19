"""Pure whole-DAG local availability classification after native evidence replay.

This module cannot load, publish, register or admit evidence. Its caller must replay
Calendar, handoff, native core and journal/source bytes before comparing times.
"""

import hashlib
from quant_investor.contracts import canonical_json_bytes
from .daily_contract import ContractError, EOD_NODE_IDS, availability, utc_stamp, validate_ref
from .prediction_policy import validate_prediction_policy
from quant_investor.intelligence.fundamental_time import availability_ceiling

SOURCE_NODES = frozenset({"industry", "theme", "exposure", "fundamental", "macro"})
CORE_NODES = frozenset(
    {"calendar", "pit", "market", "factor", "low_observation", "w80_observation", "top100"}
)


def classify_daily_evidence(
    *,
    trade_date: str,
    handoff: dict,
    recipe: dict,
    policy: dict | None,
    core_timing: dict,
    node_custody: dict,
    source_times: dict,
    synthetic: bool,
    portfolio_late: bool = False,
    corporate_late: bool = False,
) -> dict:
    """Return a comparison, not standalone evidence of prospective eligibility."""
    if any(type(v) is not bool for v in (synthetic, portfolio_late, corporate_late)):
        raise ContractError("PROSPECTIVE_PROVENANCE_INVALID")
    validate_ref(core_timing["factor_pointer_ref"])
    validate_ref(handoff["factor_pointer_ref"])
    if (
        handoff["trade_date"] != trade_date
        or core_timing["effective_trade_date"] != trade_date
        or node_custody["effective_trade_date"] != trade_date
        or set(node_custody["nodes"]) != EOD_NODE_IDS
        or set(core_timing["node_custody"]) != CORE_NODES
        or set(core_timing["observation_registration"]) != {"LOW", "W80"}
        or set(source_times) != SOURCE_NODES
        or core_timing["factor_pointer_ref"]["sha256"] != handoff["factor_pointer_ref"]["sha256"]
    ):
        raise ContractError("PROSPECTIVE_TIMING_BINDING_INVALID")
    bounds = {}
    recovered = False
    for node, row in node_custody["nodes"].items():
        finished = utc_stamp(row["completed_at"])
        if (
            row["first_verified_at"] != row["completed_at"]
            or utc_stamp(row["attempt_started_at"]) > finished
            or row["earlier_publication_proven"] is not False
            or row["recovered_at"] not in (None, row["completed_at"])
        ):
            raise ContractError("PROSPECTIVE_CUSTODY_INVALID")
        validate_ref(row["terminal_ref"])
        recovered |= row["recovered_at"] is not None
        bounds[node] = row["completed_at"]
        if node in CORE_NODES:
            original = core_timing["node_custody"][node]
            if (
                original["terminal_ref"] != row["terminal_ref"]
                or original["completed_at"] != row["completed_at"]
                or type(original["recovered"]) is not bool
                or original["recovered"] != (row["recovered_at"] is not None)
            ):
                raise ContractError("PROSPECTIVE_CORE_CUSTODY_MISMATCH")
    if (
        type(node_custody["recovered_unknown"]) is not bool
        or node_custody["recovered_unknown"] != recovered
    ):
        raise ContractError("PROSPECTIVE_RECOVERY_MISMATCH")

    def include(node, stamp):
        if utc_stamp(stamp) > utc_stamp(bounds[node]):
            bounds[node] = stamp

    include("factor", core_timing["generation_seal_upper_bound"])
    for alias, node in (("LOW", "low_observation"), ("W80", "w80_observation")):
        registration = core_timing["observation_registration"][alias]
        validate_ref(registration["observation_ref"])
        if node_custody["nodes"][node]["output_refs"].get(alias) != registration["observation_ref"]:
            raise ContractError("PROSPECTIVE_OBSERVATION_REF_MISMATCH")
        include(node, registration["registered_at"])
    for node, rows in source_times.items():
        if type(rows) is not list:
            raise ContractError("PROSPECTIVE_SOURCE_TIMES_INVALID")
        keys = []
        for row in rows:
            if type(row) is not dict or set(row) != {"source_ref", "declared_available_at"}:
                raise ContractError("PROSPECTIVE_SOURCE_TIMES_INVALID")
            ref = validate_ref(row["source_ref"])
            keys.append((ref["path"], ref["sha256"]))
            if row["declared_available_at"] is not None:
                include(node, availability_ceiling(row["declared_available_at"]))
        if keys != sorted(set(keys)):
            raise ContractError("PROSPECTIVE_SOURCE_ORDER_INVALID")
    selected = recipe["policy_refs"]["prospective"]
    retained = handoff.get("prospective_policy_ref")
    deadline = None
    if handoff["schema_version"] in {
        "cn-daily-maintenance-handoff.v2",
        "cn-daily-maintenance-handoff.v3",
        "cn-daily-maintenance-handoff.v4",
    }:
        if (selected is None) != (retained is None) or (retained is None) != (policy is None):
            raise ContractError("PROSPECTIVE_POLICY_BINDING_INVALID")
        if retained is not None:
            validate_ref(retained)
            validate_ref(selected)
            if (
                retained["sha256"] != selected["sha256"]
                or hashlib.sha256(canonical_json_bytes(policy)).hexdigest() != retained["sha256"]
            ):
                raise ContractError("PROSPECTIVE_POLICY_SHA_MISMATCH")
            validate_prediction_policy(policy, trade_date=trade_date)
            deadline = policy["prediction_deadline"]
            # Policy custody is a required bound for the complete DAG.
            for node in EOD_NODE_IDS:
                include(node, handoff["sealed_at"])
    elif handoff["schema_version"] != "cn-daily-maintenance-handoff.v1":
        raise ContractError("PROSPECTIVE_HANDOFF_VERSION_INVALID")
    retrospective = recipe["retrospective_ref"]
    if retrospective is not None:
        validate_ref(retrospective)
    result = availability(
        proven_available_at=bounds,
        required_nodes=EOD_NODE_IDS,
        deadline=deadline,
        deadline_ref=retained if deadline is not None else None,
        synthetic=synthetic,
        recovered_unknown=recovered,
        recomputed=synthetic
        or portfolio_late
        or corporate_late
        or retrospective is not None
        or handoff["schema_version"] == "cn-daily-maintenance-handoff.v3",
    )
    return {
        **result,
        "proven_available_at": bounds,
        "validation_scope": "LOCAL_COORDINATOR_AVAILABILITY_COMPARISON",
        "native_replay_required": True,
    }
