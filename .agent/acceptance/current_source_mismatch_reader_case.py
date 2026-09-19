"""Case8 physical SHA faults with one explicitly scoped native-read routing seam."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
from unittest.mock import patch


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def route_physical_fault(original_root, relative_path, fault_root):
    from quant_investor.system.storage import SecureSystemStorage

    original_root = Path(original_root).resolve(strict=True)
    fault_reader = SecureSystemStorage(str(Path(fault_root).resolve(strict=True)))
    saved = SecureSystemStorage.read_workspace_file_bytes
    counts = {
        "wrapper_calls": 0,
        "target_redirects": 0,
        "successful_physical_reads": 0,
        "non_target_redirects": 0,
        "recursive_calls": 0,
        "restored": False,
    }
    active = False

    def routed(reader, source_relative_path, *, maximum_bytes):
        nonlocal active
        counts["wrapper_calls"] += 1
        if active:
            counts["recursive_calls"] += 1
            raise AssertionError("Case8 routing recursed")
        if reader.workspace_root == original_root and str(source_relative_path) == relative_path:
            counts["target_redirects"] += 1
            active = True
            try:
                value = saved(fault_reader, source_relative_path, maximum_bytes=maximum_bytes)
                counts["successful_physical_reads"] += 1
                return value
            finally:
                active = False
        return saved(reader, source_relative_path, maximum_bytes=maximum_bytes)

    try:
        with patch.object(SecureSystemStorage, "read_workspace_file_bytes", routed):
            yield counts
    finally:
        counts["restored"] = SecureSystemStorage.read_workspace_file_bytes is saved
        if not counts["restored"]:
            raise AssertionError("Case8 reader was not restored")


def file_identity(path):
    meta = path.lstat()
    if not stat.S_ISREG(meta.st_mode) or meta.st_uid != os.geteuid() or meta.st_nlink != 1:
        raise AssertionError("Case8 selected evidence is not owner-safe")
    raw = path.read_bytes()
    after = path.lstat()
    fields = lambda s: (
        s.st_dev,
        s.st_ino,
        s.st_mode,
        s.st_uid,
        s.st_nlink,
        s.st_size,
        s.st_mtime_ns,
        s.st_ctime_ns,
    )
    if fields(meta) != fields(after):
        raise AssertionError("Case8 selected evidence changed during inventory")
    return {"sha256": sha(raw), "stat": list(fields(meta))}


def directory_identity(path):
    meta = path.lstat()
    if not stat.S_ISDIR(meta.st_mode) or meta.st_uid != os.geteuid():
        raise AssertionError("Case8 capture directory is not owner-safe")
    return [
        meta.st_dev,
        meta.st_ino,
        meta.st_mode,
        meta.st_uid,
        meta.st_nlink,
        meta.st_mtime_ns,
        meta.st_ctime_ns,
    ]


def read_selected_bytes(reader, workspace, ref, *, financial=False):
    from quant_investor.operations.daily_contract import validate_ref

    validate_ref(ref)
    if financial:
        prefix = "results/strategy_records/CN/aggressive_tech_manufacturing/"
        if not ref["path"].startswith(prefix):
            raise AssertionError("Case8 financial reader scope invalid")
        path = Path(workspace) / ref["path"]
        if stat.S_IMODE(path.stat().st_mode) not in {0o600, 0o644}:
            raise AssertionError("Case8 financial source mode invalid")
        from quant_investor.strategy_records.event_receipts import read_event_source

        return read_event_source(str(workspace), ref)
    stored = reader.read_workspace_file_bytes(ref["path"], maximum_bytes=64 * 1024 * 1024)
    if stored.byte_sha256 != ref["sha256"]:
        raise AssertionError("Case8 baseline selected SHA mismatch")
    return stored.data


def run(root):
    import quant_investor
    from quant_investor.operations.daily_contract import ContractError, EOD_NODE_IDS, validate_ref
    from quant_investor.operations.daily_status import read_daily_status
    from quant_investor.operations.core_handoff import inspect_core_handoff
    from quant_investor.system.storage import SecureSystemStorage
    from scripts.daily_completion_replay import replay_native_completion
    from _verify_five_native_days import readonly_replay_guard

    root = Path(root).resolve(strict=True)
    receipt = json.loads((root / "fixture-receipt.json").read_bytes())
    if (
        Path(quant_investor.__file__).resolve()
        != Path(receipt["runtime_verification"]["import_origin"]).resolve()
    ):
        raise AssertionError("Case8 verified installed origin required")
    driver = json.loads((root / "configured-driver-status.json").read_bytes())
    original_proof = json.loads((root / "configured-native-proof.json").read_bytes())
    if (
        driver["state"] != "PASS"
        or driver["producer_commit"] != receipt["commit"]
        or original_proof["synthetic"] is not True
        or original_proof["replay"]["native_replay_validated"] is not True
        or set(original_proof["replay"]["validated_nodes"]) != EOD_NODE_IDS
    ):
        raise AssertionError("Case8 original complete native proof required")
    workspace = (root / "factor-workspace").resolve(strict=True)
    completion_ref = validate_ref(original_proof["completion_ref"])
    day = Path(completion_ref["path"]).parent.name
    output = root / "current-source-mismatch-case8-guard-fixed"
    output.mkdir(mode=0o700, exist_ok=False)
    reader = SecureSystemStorage(str(workspace))
    paths = set()

    def read(ref, *, financial=False):
        raw = read_selected_bytes(reader, workspace, ref, financial=financial)
        paths.add(ref["path"])
        return json.loads(raw) if ref["path"].endswith(".json") else None

    completion = read(completion_ref)
    if completion["synthetic"] is not True or set(completion["node_terminal_refs"]) != EOD_NODE_IDS:
        raise AssertionError("Case8 synthetic complete day required")
    for key in ("release_ref", "native_inputs_ref"):
        read(completion[key])
    terminals = {}
    for node, ref in completion["node_terminal_refs"].items():
        terminal = read(ref)
        terminals[node] = terminal
        request = str(Path(ref["path"]).parent.parent / "request.json")
        paths.add(request)
        for selected in terminal["output_refs"].values():
            read(selected, financial=(node == "store"))
    handoff_path = str(Path(completion_ref["path"]).parent / "core-handoff.v1.json")
    handoff_ref = {"path": handoff_path, "sha256": sha((workspace / handoff_path).read_bytes())}
    handoff = read(handoff_ref)
    future_ref = terminals["calendar"]["output_refs"].get("next_session_calendar_proof")
    if future_ref is None:
        raise AssertionError("Case8 current path-bound Calendar proof required")
    publication = read(future_ref)
    future = read(publication["proof_ref"])
    read(future["acquisition_provenance_ref"])
    if future.get("transport_evidence_ref") is not None:
        read(future["transport_evidence_ref"])
    capture_ref = future["capture_root_ref"]
    capture_parent = Path(capture_ref["capture_parent"])
    expected_parent = workspace / Path(completion_ref["path"]).parent / "calendar-future/captures"
    if capture_parent != expected_parent or capture_parent.resolve(strict=True) != capture_parent:
        raise AssertionError("Case8 capture parent differs")
    capture_root = capture_parent / capture_ref["capture_root_name"]
    capture_identity = directory_identity(capture_root)
    for p in capture_root.iterdir():
        if not p.is_file() or p.is_symlink():
            raise AssertionError("Case8 unexpected native capture topology")
        paths.add(str(p.relative_to(workspace)))
    before = {p: file_identity(workspace / p) for p in sorted(paths)}
    fault_refs = {}
    for node, name in (("market", "market_input_ref"), ("pit", "market_pit_selection_ref")):
        ref = terminals[node]["output_refs"][name]
        raw = (workspace / ref["path"]).read_bytes()
        if not raw or sha(raw) != ref["sha256"]:
            raise AssertionError("Case8 original output unavailable")
        faulty = bytes([raw[0] ^ 1]) + raw[1:]
        fault_root = output / (node + "-physical-fault")
        fault_root.mkdir(mode=0o700)
        leaf = fault_root / ref["path"]
        leaf.parent.mkdir(parents=True, mode=0o700)
        for parent in leaf.parents:
            if parent == fault_root:
                break
            parent.chmod(0o700)
        leaf.write_bytes(faulty)
        leaf.chmod(0o600)
        identity = file_identity(leaf)
        if (
            len(faulty) != len(raw)
            or identity["sha256"] == ref["sha256"]
            or stat.S_IMODE(leaf.stat().st_mode) != 0o600
        ):
            raise AssertionError("Case8 physical fault differs from reviewed shape")
        fault_refs[node] = (ref, fault_root, identity)
    cases = []
    with readonly_replay_guard():
        baseline = read_daily_status(str(workspace), day)
        if any(baseline["nodes"][n]["state"] != "SUCCEEDED" for n in EOD_NODE_IDS):
            raise AssertionError("Case8 positive native status failed")
        inspect_core_handoff(
            workspace=str(workspace),
            trade_date=day,
            handoff_ref=handoff_ref,
            release_ref=handoff["release_ref"],
        )
        positive = replay_native_completion(
            workspace=str(workspace), trade_date=day, completion_ref=completion_ref
        )
        if (
            positive["native_replay_validated"] is not True
            or set(positive["validated_nodes"]) != EOD_NODE_IDS
        ):
            raise AssertionError("Case8 fresh positive full replay failed")
        (output / "positive-native-replay.json").write_text(json.dumps(positive, indent=2) + "\n")
        for node in ("market", "pit"):
            ref, fault_root, identity = fault_refs[node]
            with route_physical_fault(workspace, ref["path"], fault_root) as counts:
                status = read_daily_status(str(workspace), day)
                failed = status["nodes"][node]
                if failed["state"] != "STALE" or "DAILY_STATUS_OUTPUT_SHA_MISMATCH" not in repr(
                    failed
                ):
                    raise AssertionError("Case8 native SHA diagnostic missing")
                if any(
                    status["nodes"][n]["state"] != "SUCCEEDED" for n in EOD_NODE_IDS if n != node
                ):
                    raise AssertionError("Case8 non-target node changed")
                try:
                    replay_native_completion(
                        workspace=str(workspace), trade_date=day, completion_ref=completion_ref
                    )
                except ContractError as exc:
                    rejection = str(exc)
                    if rejection != "EOD_READBACK_TERMINAL_CHANGED:" + node:
                        raise
                else:
                    raise AssertionError("Case8 corrupt physical source admitted")
            if (
                counts["target_redirects"] < 1
                or counts["successful_physical_reads"] != counts["target_redirects"]
                or counts["non_target_redirects"]
                or counts["recursive_calls"]
                or not counts["restored"]
            ):
                raise AssertionError("Case8 routing scope failed")
            cases.append(
                {
                    "node": node,
                    "original_output_ref": ref,
                    "physical_fault_root": str(fault_root),
                    "fault_identity": identity,
                    "read_counts": counts,
                    "rejection": rejection,
                }
            )
    after = {p: file_identity(workspace / p) for p in sorted(paths)}
    if before != after or directory_identity(capture_root) != capture_identity:
        raise AssertionError("Case8 selected original evidence changed")
    result = {
        "case": 8,
        "state": "PASS",
        "producer_commit": receipt["commit"],
        "synthetic": True,
        "completion_ref": completion_ref,
        "fresh_positive_native_replay": True,
        "cases": cases,
        "fault": "ONE_PHYSICAL_OUTPUT_CORRUPTION_WITH_EXACT_READ_ROUTING_TEST_SEAM",
        "selected_immutable_inventory": before,
        "capture_root": str(capture_root),
        "capture_root_identity": capture_identity,
        "selected_original_evidence_unchanged": True,
        "whole_concurrent_workspace_invariance_claimed": False,
        "production_deployed": False,
        "harness_sha256": sha(Path(__file__).read_bytes()),
    }
    (root / "configured-current-source-mismatch-proof-guard-fixed.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print("PASS current installed Case8 physical-source SHA mismatch rejection", flush=True)
    return result
