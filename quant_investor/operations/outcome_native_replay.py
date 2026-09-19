"""Read one exact daily proof through its recorded, verified installed release."""

import hashlib
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time

from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from .archived_handoff_context import verify_archived_handoff_context
from .completion_readback import inspect_recorded_completion
from .daily_contract import ContractError, EOD_NODE_IDS, validate_ref
from .native_bridge import _invoke_active_completion_replay

MAX_INPUT = 64 * 1024
MAX_OUTPUT = 1024 * 1024
MAX_ERROR = 256 * 1024
TIMEOUT_SECONDS = 300
ENVELOPE_SCHEMA = "factor-daily-native-replay.v1"
ENVELOPE_FIELDS = frozenset(
    {
        "schema_version",
        "native_schema_version",
        "completion_ref",
        "trade_date",
        "native_replay_validated",
        "validated_nodes",
        "synthetic",
        "ledger",
    }
)

# This protocol runs under old installed releases too. Never import new evaluator
# modules in the child or accept code/operation selectors from its input.
RUNNER_SOURCE = r"""
import contextlib, fcntl, hashlib, os, pathlib, stat, sys
from quant_investor.contracts import canonical_json_bytes, parse_canonical_json_bytes
from quant_investor.system.storage import SecureSystemStorage
from quant_investor.operations.native_bridge import verified_native_context
raw = sys.stdin.buffer.read(65537)
if len(raw) > 65536:
    raise ValueError("REPLAY_INPUT_BOUND")
request = parse_canonical_json_bytes(raw)
if type(request) is not dict or set(request) != {
    "workspace", "trade_date", "completion_ref", "release_install_ref", "repository_root"
}:
    raise ValueError("REPLAY_INPUT_FIELDS")
temporary = pathlib.Path(os.environ["TMPDIR"]).resolve(strict=True)
def allowed(path, directory_fd=None):
    if not isinstance(path, (str, bytes)):
        return False
    p = pathlib.Path(os.fsdecode(path))
    if p == pathlib.Path(os.devnull) and stat.S_ISCHR(p.stat().st_mode):
        return True
    if not p.is_absolute() and isinstance(directory_fd, int) and directory_fd >= 0:
        try:
            if hasattr(fcntl, "F_GETPATH"):
                base = fcntl.fcntl(directory_fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0]
                p = pathlib.Path(os.fsdecode(base)) / p
            else:
                p = pathlib.Path(os.readlink("/proc/self/fd/" + str(directory_fd))) / p
        except OSError:
            return False
    return p.is_absolute() and p.resolve().is_relative_to(temporary)
def guard(event, args):
    if event.startswith("socket."):
        raise RuntimeError("REPLAY_NETWORK_FORBIDDEN")
    if event == "open":
        mode = args[1] if len(args) > 1 else None
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writes = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        mode_write = isinstance(mode, str) and any(c in mode for c in "wax+")
        if (mode_write or flags & writes) and not allowed(args[0]):
            raise RuntimeError("REPLAY_SOURCE_WRITE_FORBIDDEN")
    paths = {"os.mkdir": 1, "os.remove": 1, "os.rmdir": 1, "os.rename": 2,
             "os.link": 2, "os.symlink": 2, "os.chmod": 1, "os.chown": 1, "os.utime": 1}
    if event in {"os.remove", "os.rmdir"}:
        if not allowed(args[0], args[1]):
            raise RuntimeError("REPLAY_SOURCE_MUTATION_FORBIDDEN")
        return
    if event in paths and not all(allowed(p) for p in args[:paths[event]]):
        raise RuntimeError("REPLAY_SOURCE_MUTATION_FORBIDDEN")
sys.addaudithook(guard)
reference = request["release_install_ref"]
stored = SecureSystemStorage(request["workspace"]).read_workspace_file_bytes(
    reference["path"], maximum_bytes=16777216
)
if stored.byte_sha256 != reference["sha256"]:
    raise ValueError("REPLAY_INSTALL_SHA_MISMATCH")
with contextlib.redirect_stdout(sys.stderr):
    with verified_native_context(
        release_input_raw=stored.data, expected_sha256=reference["sha256"],
        repository_root=request["repository_root"]
    ) as operations:
        result = operations["completion_replay"](
            workspace=request["workspace"], trade_date=request["trade_date"],
            completion_ref=request["completion_ref"]
        )
envelope = {
    "schema_version": "factor-daily-native-replay.v1",
    "native_schema_version": result["schema_version"],
    "completion_ref": result["completion_ref"], "trade_date": result["trade_date"],
    "native_replay_validated": result["native_replay_validated"],
    "validated_nodes": result["validated_nodes"], "synthetic": result["synthetic"],
    "ledger": result.get("ledger"),
}
sys.stdout.buffer.write(canonical_json_bytes(envelope))
"""


def runner_protocol_sha256():
    return hashlib.sha256(RUNNER_SOURCE.encode()).hexdigest()


def _stop_process(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5)


def _child_replay(*, python, request):
    raw = canonical_json_bytes(request)
    if len(raw) > MAX_INPUT:
        raise ContractError("OUTCOME_NATIVE_REQUEST_BOUND")
    with tempfile.TemporaryDirectory(prefix="myquant-outcome-replay-") as temporary:
        root = Path(temporary)
        source, output, errors = root / "request.json", root / "result.json", root / "errors.txt"
        source.write_bytes(raw)
        environment = {
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "TMPDIR": temporary,
            "HOME": temporary,
        }
        with source.open("rb") as stdin, output.open("wb") as stdout, errors.open("wb") as stderr:
            process = subprocess.Popen(
                [python, "-I", "-B", "-c", RUNNER_SOURCE],
                cwd=temporary,
                env=environment,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + TIMEOUT_SECONDS
                while process.poll() is None:
                    _check_process_bounds(output, errors, deadline)
                    time.sleep(0.05)
                _check_process_bounds(output, errors, deadline)
                if process.returncode != 0:
                    raise ContractError("OUTCOME_NATIVE_CHILD_FAILED")
            finally:
                _stop_process(process)
        return parse_canonical_json_bytes(output.read_bytes())


def _check_process_bounds(output, errors, deadline):
    if output.stat().st_size > MAX_OUTPUT or errors.stat().st_size > MAX_ERROR:
        raise ContractError("OUTCOME_NATIVE_OUTPUT_BOUND")
    if time.monotonic() > deadline:
        raise ContractError("OUTCOME_NATIVE_REPLAY_TIMEOUT")


def _envelope(result):
    return {
        "schema_version": ENVELOPE_SCHEMA,
        "native_schema_version": result["schema_version"],
        **{
            key: result[key]
            for key in (
                "completion_ref",
                "trade_date",
                "native_replay_validated",
                "validated_nodes",
                "synthetic",
            )
        },
        "ledger": result.get("ledger"),
    }


def _validate_result(result, *, trade_date, completion_ref, ledger, ledger_ref):
    if (
        type(result) is not dict
        or set(result) != ENVELOPE_FIELDS
        or result["schema_version"] != ENVELOPE_SCHEMA
        or result["native_schema_version"] != "cn-daily-eod-native-readback.v1"
        or result["completion_ref"] != completion_ref
        or result["trade_date"] != trade_date
        or result["native_replay_validated"] is not True
        or result["validated_nodes"] != sorted(EOD_NODE_IDS)
        or type(result["synthetic"]) is not bool
        or result["synthetic"] != ledger["synthetic"]
    ):
        raise ContractError("OUTCOME_NATIVE_REPLAY_RESULT_INVALID")
    proof = result["ledger"]
    required = {
        "ledger_ref",
        "classification",
        "prospective",
        "synthetic",
        "validation_scope",
        "native_business_replay_required",
    }
    if (
        type(proof) is not dict
        or set(proof) not in (required, required | {"recomputed"})
        or proof["ledger_ref"] != ledger_ref
        or proof["validation_scope"] != "NATIVE_LEDGER_DERIVATION"
        or proof["native_business_replay_required"] is not True
    ):
        raise ContractError("OUTCOME_NATIVE_LEDGER_RESULT_INVALID")
    for key in ("classification", "prospective", "synthetic", "recomputed"):
        if key in proof and canonical_json_bytes(proof[key]) != canonical_json_bytes(ledger[key]):
            raise ContractError("OUTCOME_NATIVE_LEDGER_RESULT_MISMATCH")
    _validate_ledger_flags(ledger)


def _validate_ledger_flags(ledger):
    classification = ledger["classification"]
    if (
        type(classification) is not str
        or classification
        not in {"CONTEMPORANEOUS", "LATE_REGISTERED", "UNKNOWN_LEGACY", "RETROSPECTIVE_RECOMPUTE"}
        or any(type(ledger[key]) is not bool for key in ("prospective", "synthetic", "recomputed"))
        or ledger["prospective"] is not (classification == "CONTEMPORANEOUS")
        or (
            (ledger["synthetic"] or ledger["recomputed"])
            != (classification == "RETROSPECTIVE_RECOMPUTE")
        )
    ):
        raise ContractError("OUTCOME_NATIVE_LEDGER_FLAGS_INVALID")


def _native_identity(snapshot):
    context = snapshot.document("loop_context")
    payload = snapshot.document("release_install_input")["release_install_evidence"]["payload"]
    identity = {
        "release_install_ref": snapshot.reference("release_install_input"),
        "final_commit": payload["final_commit"],
        "final_tree": payload["final_tree"],
        "installed_code_manifest_sha256": payload["installed_code_manifest_sha256"],
        "repository_root": str(Path(context["release_repository_root"]).resolve(strict=True)),
        "operation": "completion_replay",
    }
    return identity, payload["python_executable"]


def replay_outcome_daily_evidence(*, workspace, trade_date, completion_ref):
    """Return release-bound availability and its live, immutable snapshot capability."""
    ref = validate_ref(completion_ref)
    inspected = inspect_recorded_completion(
        workspace=workspace, trade_date=trade_date, completion_ref=ref
    )
    completion = inspected["recorded_completion"]
    if completion["schema_version"] == "cn-daily-eod-completion.v1":
        return {
            "completion_ref": ref,
            "trade_date": trade_date,
            "ledger_ref": None,
            "classification": "UNKNOWN_LEGACY",
            "prospective": False,
            "synthetic": completion["synthetic"],
            "recomputed": None,
            "validation_scope": "RECORDED_EOD_WITHOUT_NATIVE_RELEASE",
            "native_identity": None,
            "snapshot": None,
        }
    snapshot = inspected["completed_handoff_snapshot"]
    verify_archived_handoff_context(snapshot)
    identity, python = _native_identity(snapshot)
    matched, native = _invoke_active_completion_replay(
        release_input_sha256=identity["release_install_ref"]["sha256"],
        repository_root=identity["repository_root"],
        final_commit=identity["final_commit"],
        workspace=workspace,
        trade_date=trade_date,
        completion_ref=ref,
    )
    result = (
        _envelope(native)
        if matched
        else _child_replay(
            python=python,
            request={
                "workspace": workspace,
                "trade_date": trade_date,
                "completion_ref": ref,
                "release_install_ref": identity["release_install_ref"],
                "repository_root": identity["repository_root"],
            },
        )
    )
    snapshot.recheck()
    ledger = snapshot.document("ledger")
    ledger_ref = snapshot.reference("ledger")
    _validate_result(
        result, trade_date=trade_date, completion_ref=ref, ledger=ledger, ledger_ref=ledger_ref
    )
    return {
        "completion_ref": ref,
        "trade_date": trade_date,
        "ledger_ref": ledger_ref,
        **{
            key: ledger[key] for key in ("classification", "prospective", "synthetic", "recomputed")
        },
        "validation_scope": "FULL_NATIVE_EOD_AVAILABILITY",
        "native_identity": identity,
        "snapshot": snapshot,
    }
