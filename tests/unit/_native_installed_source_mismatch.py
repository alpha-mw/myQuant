"""Fail-closed full-EOD entry checks on copied native journal/source bytes.

The successful producer workspace is read only. A separate fixture contains its
exact node journals and output leaves; only one Market or PIT output gets an
explicit byte-corruption fault. Native status and EOD admission stay unchanged.
"""

from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
from unittest.mock import patch


def run(root):
    import quant_investor
    from quant_investor.operations.core_handoff import inspect_core_handoff
    from quant_investor.operations.daily_contract import ContractError
    from quant_investor.operations.daily_status import read_daily_status
    from quant_investor.operations.journal_storage import JournalStorage
    from quant_investor.strategy_records import store
    from scripts.daily_completion_replay import replay_native_completion

    receipt = json.loads((root / "fixture-receipt.json").read_bytes())
    origin = Path(quant_investor.__file__).resolve()
    if origin != Path(receipt["runtime_verification"]["import_origin"]).resolve():
        raise ValueError("verified installed runtime required")
    original = root / "factor-workspace"
    target = root / "source-mismatch-validation-workspace"
    target.mkdir(mode=0o700)
    day = "20260827"
    day_path = Path("results/operations/daily_production/CN") / day
    copied = {}

    def copy(path, expected=None):
        source = original / path
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("original native source SHA mismatch: " + str(path))
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        for parent in destination.parents:
            if parent == target:
                break
            parent.chmod(0o700)
        shutil.copy2(source, destination)
        if destination.read_bytes() != raw:
            raise ValueError("native fixture copy changed bytes")
        copied[str(path)] = (digest, source.stat().st_mtime_ns)
        return json.loads(raw) if source.suffix == ".json" else None

    completion = copy(day_path / "completion.v1.json")
    if completion["synthetic"] is not True or len(completion["node_terminal_refs"]) != 16:
        raise ValueError("current synthetic all-node native completion required")
    completion_ref = {
        "path": str(day_path / "completion.v1.json"),
        "sha256": copied[str(day_path / "completion.v1.json")][0],
    }
    for key in ("release_ref", "native_inputs_ref"):
        copy(completion[key]["path"], completion[key]["sha256"])
    for source in (original / day_path / "nodes").rglob("*"):
        if source.is_file():
            copy(source.relative_to(original))
    terminals = {}
    for node, ref in completion["node_terminal_refs"].items():
        terminal = copy(ref["path"], ref["sha256"])
        terminals[node] = terminal
        for output in terminal["output_refs"].values():
            copy(output["path"], output["sha256"])
    handoff = copy(day_path / "core-handoff.v1.json")
    handoff_ref = {
        "path": str(day_path / "core-handoff.v1.json"),
        "sha256": copied[str(day_path / "core-handoff.v1.json")][0],
    }

    def forbidden(*args, **kwargs):
        raise AssertionError("source mismatch reader invoked a writer or network")

    def inventory():
        return {
            str(p.relative_to(target)): (
                hashlib.sha256(p.read_bytes()).hexdigest(),
                p.stat().st_mode,
                p.stat().st_mtime_ns,
            )
            for p in target.rglob("*")
            if p.is_file()
        }

    cases = []
    with ExitStack() as stack:
        stack.enter_context(patch.object(JournalStorage, "write", forbidden))
        stack.enter_context(patch.object(store, "_cas_pointer", forbidden))
        stack.enter_context(patch("socket.socket.connect", forbidden))
        baseline = inventory()
        status = read_daily_status(str(target), day)
        if any(row["state"] != "SUCCEEDED" for row in status["nodes"].values()):
            raise ValueError("copied native journal/output baseline failed: " + repr(status))
        core = inspect_core_handoff(
            workspace=str(target),
            trade_date=day,
            handoff_ref=handoff_ref,
            release_ref=handoff["release_ref"],
        )
        if inventory() != baseline:
            raise ValueError("positive native baseline read changed fixture")
        for node, output_name in (
            ("market", "market_input_ref"),
            ("pit", "market_pit_selection_ref"),
        ):
            output = terminals[node]["output_refs"][output_name]
            leaf = target / output["path"]
            raw = leaf.read_bytes()
            leaf.write_bytes(raw + b" ")
            fault_inventory = inventory()
            failed = read_daily_status(str(target), day)["nodes"][node]
            if failed["state"] != "STALE" or "DAILY_STATUS_OUTPUT_SHA_MISMATCH" not in repr(failed):
                raise ValueError("native status did not identify corrupt " + node)
            try:
                replay_native_completion(
                    workspace=str(target), trade_date=day, completion_ref=completion_ref
                )
            except ContractError as exc:
                expected = "EOD_READBACK_TERMINAL_CHANGED:" + node
                if str(exc) != expected:
                    raise
                rejection = str(exc)
            else:
                raise ValueError("corrupt native source admitted to full EOD replay")
            if inventory() != fault_inventory:
                raise ValueError("negative full-EOD entry changed fixture")
            cases.append(
                {
                    "node": node,
                    "original_output_ref": output,
                    "corrupt_sha256": hashlib.sha256(leaf.read_bytes()).hexdigest(),
                    "status_state": failed["state"],
                    "status_reason": "DAILY_STATUS_OUTPUT_SHA_MISMATCH",
                    "full_eod_entry_rejection": rejection,
                    "entry_read_only": True,
                }
            )
            shutil.copy2(original / output["path"], leaf)
        if inventory() != baseline:
            raise ValueError("explicit faults were not confined to the copied fixture")
    for path, before in copied.items():
        source = original / path
        if (hashlib.sha256(source.read_bytes()).hexdigest(), source.stat().st_mtime_ns) != before:
            raise ValueError("original selected native evidence changed during validation")
    proof = {
        "case": 8,
        "synthetic": True,
        "producer_commit": receipt["commit"],
        "installed_import_origin": str(origin),
        "completion_ref": completion_ref,
        "fixture_workspace": str(target),
        "copied_native_file_count": len(copied),
        "positive_baseline": "ALL16_JOURNAL_OUTPUT_BYTES_AND_CORE_HANDOFF",
        "positive_core_scope": core["validation_scope"],
        "full_positive_native_replay_claimed": False,
        "fault": "ONE_PHYSICAL_OUTPUT_BYTE_SUFFIX_NO_REFERENCE_RESEALING",
        "cases": cases,
        "original_selected_evidence_unchanged": True,
        "production_deployed": False,
    }
    (root / "configured-source-mismatch-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print("PASS installed Market and PIT corrupt-source full-EOD entry rejection", flush=True)
    return proof
