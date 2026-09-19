"""Preserve native core timing separately from effective dates and OOS admission."""

from .daily_contract import ContractError, utc_stamp, validate_ref


def recorded_core_timing(
    *,
    pointer: dict,
    pointer_ref: dict,
    generation: dict,
    observations: list[dict],
    observation_refs: dict,
    terminal_refs: dict,
    terminals: dict,
) -> dict:
    """Called after native core replay; no timestamp is inferred from filesystem age."""
    validate_ref(pointer_ref)
    seal = pointer["activated_at"]
    utc_stamp(seal)
    registration = {}
    for observation in observations:
        payload = observation["payload"]
        alias = payload["factor_alias"]
        if alias not in {"LOW", "W80"} or alias in registration:
            raise ContractError("CORE_TIMING_OBSERVATION_SET_INVALID")
        registered = payload["registered_at"]
        utc_stamp(registered)
        ref = observation_refs[alias]
        validate_ref(ref)
        node = "low_observation" if alias == "LOW" else "w80_observation"
        if utc_stamp(registered) > utc_stamp(terminals[node]["finished_at"]):
            raise ContractError("CORE_TIMING_REGISTRATION_AFTER_VALIDATION")
        registration[alias] = {"registered_at": registered, "observation_ref": ref}
    if set(registration) != {"LOW", "W80"}:
        raise ContractError("CORE_TIMING_OBSERVATION_SET_INVALID")
    if utc_stamp(seal) > utc_stamp(terminals["factor"]["finished_at"]):
        raise ContractError("CORE_TIMING_SEAL_AFTER_VALIDATION")
    completed = {}
    for node, terminal in terminals.items():
        utc_stamp(terminal["finished_at"])
        validate_ref(terminal_refs[node])
        completed[node] = {
            "completed_at": terminal["finished_at"],
            "recovered": terminal["recovered"],
            "terminal_ref": terminal_refs[node],
        }
    return {
        "schema_version": "cn-daily-core-timing.v1",
        "effective_trade_date": observations[0]["payload"]["signal_date"],
        "factor_pointer_ref": pointer_ref,
        "generation_seal_upper_bound": seal,
        "seal_semantics": "LOCAL_IMMUTABLE_POINTER_UPPER_BOUND",
        "generation_created_at": generation["created_at"],
        "generation_created_at_is_availability_proof": False,
        "observation_registration": registration,
        "node_custody": completed,
        "prospective_admission": False,
        "validation_scope": "NATIVE_CORE_TIMESTAMPS_ONLY",
    }
