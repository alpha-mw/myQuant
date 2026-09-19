"""Verify the installed synthetic five-day EOD chain and readonly native replay."""

from pathlib import Path
from contextlib import contextmanager, ExitStack
import os
import stat
import hashlib
import json
import sys
from unittest.mock import patch

if __name__ == "__main__":
    import quant_investor

    original = json.loads((Path(sys.argv[1]) / "fixture-receipt.json").read_bytes())
    if (
        Path(quant_investor.__file__).resolve()
        != Path(original["runtime_verification"]["import_origin"]).resolve()
    ):
        raise AssertionError("five-day verification requires the original installed package")
    pinned = Path(original["repository"]).resolve(strict=True)
    sys.path.extend([str(pinned), str(pinned / "scripts"), str(pinned / "tests/unit")])
    if len(sys.argv) > 3:
        sys.path.append(sys.argv[3])
from quant_investor.operations.completion_readback import inspect_recorded_completion
from quant_investor.operations.daily_status import read_daily_status
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.factors.production_authority import (
    FactorProductionStore,
    _FactorSecureStorage,
    FACTOR_ACTIVE_LOCK_PATH,
)
from quant_investor.operations.daily_contract import EOD_NODE_IDS
from quant_investor.operations.journal_storage import JournalStorage
from quant_investor.contracts import validate_artifact, canonical_json_bytes
from quant_investor.system.store import object_ref_for_artifact

DAYS = ["20260827", "20260828", "20260831", "20260901", "20260902"]


def inventory(workspace):
    rows = {}
    for path in workspace.rglob("*"):
        metadata = path.lstat()
        digest = None
        if path.is_file():
            h = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        rows[str(path.relative_to(workspace))] = (metadata.st_mode, metadata.st_mtime_ns, digest)
    return rows


@contextmanager
def readonly_replay_guard():
    """Keep native read synchronization while rejecting data publication."""
    native_lock = FactorProductionStore._active_lock

    def forbidden(*args, **kwargs):
        raise AssertionError("historical replay attempted data write or external access")

    @contextmanager
    def existing_native_lock(store):
        path = store.workspace_root / FACTOR_ACTIVE_LOCK_PATH
        before = path.lstat()
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.geteuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) != 0o600
        ):
            raise AssertionError("historical replay requires an existing safe Factor lock")
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
        )
        with native_lock(store):
            yield
        after = path.lstat()
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise AssertionError("historical replay changed the Factor lock")

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(FactorProductionStore, "_active_lock", existing_native_lock)
        )
        for name in (
            "write_exact_once",
            "_write_blob_exact_once",
            "_write_exact_once",
            "_write_reserved_atomic_no_replace",
            "write_initial_pointer_under_lock",
            "write_permanent_marker_under_lock",
            "replace_active_pointer_under_lock",
        ):
            stack.enter_context(patch.object(_FactorSecureStorage, name, forbidden))
        stack.enter_context(patch.object(DailyJournal, "locked", forbidden))
        stack.enter_context(patch.object(JournalStorage, "write", forbidden))
        stack.enter_context(patch("socket.socket.connect", forbidden))
        stack.enter_context(patch("socket.create_connection", forbidden))
        yield


def run(root, *, replay_native_completion, v2=False, completion_refs=None):
    receipt = json.loads((root / "fixture-receipt.json").read_text())
    assert receipt["synthetic_test_environment"] is True
    import quant_investor

    assert (
        Path(quant_investor.__file__).resolve()
        == Path(receipt["runtime_verification"]["import_origin"]).resolve()
    )
    pinned = Path(receipt["repository"]).resolve(strict=True)
    assert (
        Path(sys.modules["scripts.daily_completion_replay"].__file__)
        .resolve()
        .is_relative_to(pinned)
    )
    workspace = root / "factor-workspace"
    refs = (
        completion_refs
        if completion_refs is not None
        else {
            d: json.loads(
                (
                    root / ("full-completion-ref" + ("" if d == DAYS[0] else "-" + d) + ".json")
                ).read_text()
            )
            for d in DAYS
        }
    )
    if type(refs) is not dict or set(refs) != set(DAYS):
        raise ValueError("the exact five consecutive completion references are required")
    previous_factor = previous_store = release = None
    previous_release_bytes = previous_install_bytes = None
    release_refs = {}
    for day in DAYS:
        inspected = inspect_recorded_completion(
            workspace=str(workspace), trade_date=day, completion_ref=refs[day]
        )
        completion = inspected["recorded_completion"]
        release_ref = completion["release_ref"]
        release_raw = (workspace / release_ref["path"]).read_bytes()
        assert hashlib.sha256(release_raw).hexdigest() == release_ref["sha256"]
        release_identity = object_ref_for_artifact(validate_artifact(release_raw))
        release_refs[day] = release_ref
        assert completion["schema_version"] == (
            "cn-daily-eod-completion.v2" if v2 else "cn-daily-eod-completion.v1"
        )
        if v2:
            snapshot = inspected["completed_handoff_snapshot"]
            recipe = snapshot.document("recipe")
            index = DAYS.index(day)
            assert recipe["previous_completion_ref"] == (refs[DAYS[index - 1]] if index else None)
            assert (recipe["bootstrap_ref"] is not None) == (index == 0)
            assert snapshot.reference("materialization") == completion["materialization_ref"]
            assert snapshot.reference("ledger") == completion["prospective_ledger_ref"]
            install_bytes = canonical_json_bytes(snapshot.document("release_install_input"))
            if previous_install_bytes is not None:
                assert install_bytes == previous_install_bytes
            previous_install_bytes = install_bytes
        inputs = json.loads((workspace / completion["native_inputs_ref"]["path"]).read_text())
        status = read_daily_status(str(workspace), day)
        assert completion["synthetic"] is True
        assert all(
            row["state"] == "SUCCEEDED" and row["attempt"] == 1 for row in status["nodes"].values()
        )
        pointer = json.loads(
            (
                workspace
                / f"results/operations/daily_production/CN/{day}/inputs/factor-pointer-{inputs['factor_pointer_sha256']}.json"
            ).read_text()
        )
        plan = json.loads((workspace / inputs["store_plan_ref"]["path"]).read_text())
        if previous_factor is not None:
            assert pointer["previous_pointer_sha256"] == previous_factor
            assert plan["preimages"]["store_pointer_sha256"] == previous_store
            assert release_identity == release and release_raw == previous_release_bytes
        previous_factor = inputs["factor_pointer_sha256"]
        previous_store = status["nodes"]["store"]["terminal"]["output_refs"]["pointer"]["sha256"]
        release = release_identity
        previous_release_bytes = release_raw
    generation = json.loads(
        (
            workspace
            / f"results/factors/generations/{pointer['factor_generation_id']}/generation.json"
        ).read_text()
    )
    ref = generation["payload"]["calendar_compilation_ref"]
    calendar = json.loads(
        (workspace / f"results/factors/objects/{ref['kind']}/{ref['byte_sha256']}.json").read_text()
    )
    opened = [
        row["date"].replace("-", "")
        for row in calendar["payload"]["runtime_projection"]
        if row["status"] == "OPEN" and DAYS[0] <= row["date"].replace("-", "") <= DAYS[-1]
    ]
    assert opened == DAYS
    before = inventory(workspace)

    with readonly_replay_guard():
        for day in DAYS:
            print("START native replay", day, flush=True)
            value = replay_native_completion(
                workspace=str(workspace), trade_date=day, completion_ref=refs[day]
            )
            assert value["native_replay_validated"] is True
            assert set(value["validated_nodes"]) == set(EOD_NODE_IDS)
            if v2:
                ledger = value["ledger"]
                assert ledger["validation_scope"] == "NATIVE_LEDGER_DERIVATION"
                assert ledger["classification"] == "RETROSPECTIVE_RECOMPUTE"
                assert ledger["prospective"] is False and ledger["synthetic"] is True
            (root / f"five-day-native-readback-{day}.json").write_text(
                json.dumps(value, indent=2) + "\n"
            )
            print("PASS native replay", day, flush=True)
    assert inventory(workspace) == before
    proof = {
        "synthetic": True,
        "scope": "NATIVE_EOD_DAG_V2" if v2 else "PINNED_NATIVE_EOD_DAG_V1",
        "v2_ledger_and_materialization_chain": v2,
        "verification_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "producer_commit": receipt["commit"],
        "business_scripts_repository": str(pinned),
        "full_five_day_proof": True,
        "trade_dates": DAYS,
        "completion_refs": refs,
        "release_refs": release_refs,
        "same_native_release_artifact_bytes": True,
        "release_reference_paths_differ": len({ref["path"] for ref in release_refs.values()}) > 1,
        "single_attempt_all_nodes": True,
        "factor_and_store_parent_chain": True,
        "calendar_consecutive": True,
        "full_native_replay_all_days": True,
        "no_write_replay": True,
        "existing_native_factor_read_lock_allowed": True,
        "factor_data_writers_forbidden": True,
        "inventory_entries": len(before),
        "prospective": False,
        "morning_v2_complete": False,
        "runtime": receipt["runtime_verification"],
    }
    return proof


if __name__ == "__main__":
    from quant_investor.operations.native_bridge import verified_native_context

    root = Path(sys.argv[1])
    raw = (root / "release-input.json").read_bytes()
    with verified_native_context(
        release_input_raw=raw,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
        repository_root=original["repository"],
    ) as operations:
        proof = run(
            root,
            replay_native_completion=operations["completion_replay"],
            v2=len(sys.argv) > 2 and sys.argv[2] == "v2",
        )
    (root / "five-native-trading-day-proof.json").write_text(json.dumps(proof, indent=2) + "\n")
    print("PASS FIVE NATIVE DAYS AND READONLY REPLAY", flush=True)
