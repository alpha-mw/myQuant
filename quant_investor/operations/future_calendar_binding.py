"""Exact optional future Calendar capability binding; absent capability is valid EOD."""

from quant_investor.contracts import parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.market.next_session_proof import read_next_session_proof
from quant_investor.market.next_session_failure import read_next_session_failure
from .daily_contract import ContractError, validate_ref, utc_stamp


def future_calendar_outputs(
    *,
    workspace: str,
    trade_date: str,
    proof_ref=None,
    failure_ref=None,
    factor_snapshot=None,
    finished_at=None,
) -> dict:
    if proof_ref is not None and failure_ref is not None:
        raise ContractError("NEXT_SESSION_PROOF_FAILURE_CONFLICT")
    if proof_ref is not None:
        validate_ref(proof_ref)
        value = read_next_session_proof(
            workspace=workspace, eod_trade_date=trade_date, publication_ref=proof_ref
        )
        if finished_at is not None and utc_stamp(value["proof_sealed_at"]) > utc_stamp(finished_at):
            raise ContractError("NEXT_SESSION_PROOF_AFTER_CALENDAR_TERMINAL")
        if factor_snapshot is not None:
            generation = factor_snapshot["factor_generation"]["payload"]
            if value["proof"]["release_ref"] != generation["deployed_release_ref"]:
                raise ContractError("NEXT_SESSION_FACTOR_RELEASE_MISMATCH")
            ref = generation["calendar_compilation_ref"]
            raw = SecureSystemStorage(workspace).read_workspace_file_bytes(
                f"results/factors/objects/{ref['kind']}/{ref['byte_sha256']}.json",
                maximum_bytes=8 * 1024 * 1024,
            )
            if raw.byte_sha256 != ref["byte_sha256"]:
                raise ContractError("NEXT_SESSION_FACTOR_CALENDAR_SHA_MISMATCH")
            calendar = parse_canonical_json_bytes(raw.data)["payload"]
            rows = [
                row
                for row in calendar["runtime_projection"]
                if row["date"].replace("-", "") == trade_date
            ]
            policy = value["calendar_policy"]["payload"]
            if (
                len(rows) != 1
                or rows[0]["status"] != "OPEN"
                or any(
                    calendar[key] != policy[key]
                    for key in (
                        "timezone",
                        "processing_open_local",
                        "processing_close_local",
                        "time_semantics",
                        "envelope_source",
                    )
                )
            ):
                raise ContractError("NEXT_SESSION_FACTOR_CALENDAR_DISAGREEMENT")
        return {"next_session_calendar_proof": dict(proof_ref)}
    if failure_ref is not None:
        validate_ref(failure_ref)
        value = read_next_session_failure(
            workspace=workspace, eod_trade_date=trade_date, failure_ref=failure_ref
        )
        if finished_at is not None and utc_stamp(value["failure"]["failed_at"]) > utc_stamp(
            finished_at
        ):
            raise ContractError("NEXT_SESSION_FAILURE_AFTER_CALENDAR_TERMINAL")
        return {"next_session_calendar_failure": dict(failure_ref)}
    return {}
