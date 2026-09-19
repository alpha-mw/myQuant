"""Receipt-derived custody times; never effective-date or file-age availability."""

from .daily_contract import ContractError, EOD_NODE_IDS, utc_stamp, validate_ref
from .daily_journal import _validate_day


def recorded_daily_timing(
    *, trade_date: str, nodes: dict, terminal_refs: dict, verified_at: str
) -> dict:
    """Project already validated journal rows; source/native replay remains required.

    A start receipt proves an attempt began, not that its outputs were available.
    The terminal is the coordinator's first successful verification upper bound.
    Recovery cannot establish earlier publication from a successful later read.
    """
    _validate_day(trade_date)
    bound = utc_stamp(verified_at)
    if set(nodes) != EOD_NODE_IDS or set(terminal_refs) != EOD_NODE_IDS:
        raise ContractError("DAILY_TIMING_NODE_SET_INVALID")
    custody = {}
    for node in sorted(EOD_NODE_IDS):
        row = nodes[node]
        ref = validate_ref(terminal_refs[node])
        if row.get("state") != "SUCCEEDED" or row.get("terminal_ref") != ref:
            raise ContractError("DAILY_TIMING_TERMINAL_MISMATCH:" + node)
        start_ref = validate_ref(row["start_ref"])
        if start_ref["path"] != ref["path"].removesuffix("terminal.json") + "start.json":
            raise ContractError("DAILY_TIMING_ATTEMPT_BINDING_INVALID")
        terminal, start = row["terminal"], row["start"]
        began, finished = utc_stamp(start["started_at"]), utc_stamp(terminal["finished_at"])
        if began > finished or finished > bound:
            raise ContractError("DAILY_TIMING_CHRONOLOGY_INVALID:" + node)
        recovered = terminal["recovered"]
        if type(recovered) is not bool:
            raise ContractError("DAILY_TIMING_RECOVERY_INVALID")
        custody[node] = {
            "attempt_started_at": start["started_at"],
            "first_verified_at": terminal["finished_at"],
            "completed_at": terminal["finished_at"],
            "recovered_at": terminal["finished_at"] if recovered else None,
            "earlier_publication_proven": False,
            "start_ref": start_ref,
            "terminal_ref": ref,
            "output_refs": terminal["output_refs"],
        }
    return {
        "schema_version": "cn-daily-custody-timing.v1",
        "effective_trade_date": trade_date,
        "nodes": custody,
        "recovered_unknown": any(row["recovered_at"] is not None for row in custody.values()),
        "prospective_admission": False,
        "validation_scope": "RECORDED_COORDINATOR_CUSTODY_ONLY",
    }
