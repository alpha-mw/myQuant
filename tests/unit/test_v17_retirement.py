from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading

import pytest

import quant_investor.v17.retirement as retirement_module
from quant_investor.v17.retirement import (
    BLOCKED_STATE,
    PostPurgeBlockedError,
    REQUIRED_SCHEDULE_NAMES,
    RetirementConflictError,
    RetirementError,
    advance_gate,
    initialize_cutover,
    journal_sha256,
    load_journal,
    mark_resumed,
    purge,
    rollback_pre_purge,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


REPO_SHA = _sha("repo")
SKILL_SHA = _sha("skill")
SCHEDULE_SHAS = {name: _sha(f"schedule:{name}") for name in REQUIRED_SCHEDULE_NAMES}
FINAL_SCAN_SHA = _sha("final-scan")


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    for relative in ("results/v16", "results/v16_operator_advisory"):
        target = repo / relative
        target.mkdir(parents=True)
        (target / "payload.bin").write_bytes(relative.encode("utf-8"))
    return repo


def _initialize(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = _repo(tmp_path)
    journal = tmp_path / "private" / "journal.json"
    digest = initialize_cutover(
        repo_root=repo,
        journal_path=journal,
        cutover_id="cutover-1",
        repo_sha256=REPO_SHA,
        skill_sha256=SKILL_SHA,
        schedule_sha256=SCHEDULE_SHAS,
        nonce="abc123",
    )
    return repo, journal, digest


def _binding(repo: Path) -> dict[str, object]:
    return {
        "cutover_id": "cutover-1",
        "repo_root": repo,
        "repo_sha256": REPO_SHA,
        "skill_sha256": SKILL_SHA,
        "schedule_sha256": SCHEDULE_SHAS,
    }


def _eligible(tmp_path: Path) -> tuple[Path, Path, str]:
    repo, journal, digest = _initialize(tmp_path)
    for state in (
        "CODE_VERIFIED",
        "SKILL_SCHEDULES_VERIFIED",
        "PURGE_ELIGIBLE",
    ):
        digest = advance_gate(
            journal_path=journal,
            expected_journal_sha256=digest,
            next_state=state,
            **_binding(repo),
        )
    return repo, journal, digest


def _write_journal(path: Path, payload: dict[str, object]) -> str:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    path.write_bytes(encoded)
    path.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()


def test_initialize_binds_only_the_two_exact_roots_and_private_journal(
    tmp_path: Path,
) -> None:
    repo, journal, digest = _initialize(tmp_path)

    assert journal_sha256(journal) == digest
    assert journal.stat().st_mode & 0o777 == 0o600
    assert journal.parent.stat().st_mode & 0o777 == 0o700
    payload = load_journal(journal)
    assert payload["state"] == "QUIESCED"
    assert [item["relative_path"] for item in payload["roots"]] == [
        "results/v16",
        "results/v16_operator_advisory",
    ]
    assert all(item["source_st_ino"] > 0 for item in payload["roots"])
    assert payload["schedule_sha256"] == dict(sorted(SCHEDULE_SHAS.items()))
    assert (repo / "results/v16").is_dir()


@pytest.mark.parametrize("mutation", ["substitute-root", "remove-roots", "extra-key"])
def test_tampered_journal_shape_cannot_expand_or_bypass_purge_scope(
    tmp_path: Path,
    mutation: str,
) -> None:
    repo, journal, _ = _eligible(tmp_path)
    victim = tmp_path / "must-survive"
    victim.mkdir()
    (victim / "keep.txt").write_text("safe", encoding="utf-8")
    payload = load_journal(journal)
    if mutation == "substitute-root":
        payload["roots"][0]["source_path"] = str(victim)
        payload["roots"][0]["source_realpath"] = str(victim)
        payload["roots"][0]["source_st_dev"] = victim.stat().st_dev
        payload["roots"][0]["source_st_ino"] = victim.stat().st_ino
    elif mutation == "remove-roots":
        payload["roots"] = []
    else:
        payload["untrusted_override"] = True
    tampered_sha = _write_journal(journal, payload)

    with pytest.raises(RetirementError):
        load_journal(journal)
    with pytest.raises(RetirementError):
        purge(
            journal_path=journal,
            expected_journal_sha256=tampered_sha,
            **_binding(repo),
        )
    assert (victim / "keep.txt").read_text(encoding="utf-8") == "safe"
    assert (repo / "results/v16/payload.bin").is_file()


def test_gate_cas_is_ordered_and_conflicts_are_zero_write(tmp_path: Path) -> None:
    repo, journal, digest = _initialize(tmp_path)
    before = journal.read_bytes()

    with pytest.raises(RetirementConflictError, match="CAS mismatch"):
        advance_gate(
            journal_path=journal,
            expected_journal_sha256="0" * 64,
            next_state="CODE_VERIFIED",
            **_binding(repo),
        )
    assert journal.read_bytes() == before

    with pytest.raises(RetirementConflictError, match="gate transition"):
        advance_gate(
            journal_path=journal,
            expected_journal_sha256=digest,
            next_state="SKILL_SCHEDULES_VERIFIED",
            **_binding(repo),
        )
    assert journal.read_bytes() == before


def test_cutover_lock_serializes_the_entire_purge_operation(tmp_path: Path) -> None:
    repo, journal, digest = _eligible(tmp_path)
    entered = threading.Event()
    release = threading.Event()
    errors: list[BaseException] = []

    def pause(event: str, _root: object) -> None:
        if event == "RENAME_INTENT_COMMITTED" and not entered.is_set():
            entered.set()
            if not release.wait(timeout=5):
                raise AssertionError("test did not release purge thread")

    def run_first() -> None:
        try:
            purge(
                journal_path=journal,
                expected_journal_sha256=digest,
                _event_hook=pause,
                **_binding(repo),
            )
        except BaseException as exc:  # pragma: no cover - surfaced in main thread
            errors.append(exc)

    worker = threading.Thread(target=run_first)
    worker.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(RetirementConflictError, match="holds the cutover lock"):
            purge(
                journal_path=journal,
                expected_journal_sha256=digest,
                **_binding(repo),
            )
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert errors == []
    assert load_journal(journal)["state"] == "PURGED"


def test_pre_purge_rollback_is_terminal_and_preserves_sources(tmp_path: Path) -> None:
    repo, journal, digest = _initialize(tmp_path)
    digest = rollback_pre_purge(
        journal_path=journal,
        expected_journal_sha256=digest,
        reason="owner stopped cutover",
        **_binding(repo),
    )

    assert load_journal(journal)["state"] == "ROLLED_BACK_PRE_PURGE"
    assert journal_sha256(journal) == digest
    assert (repo / "results/v16/payload.bin").is_file()
    with pytest.raises(RetirementConflictError):
        purge(
            journal_path=journal,
            expected_journal_sha256=digest,
            **_binding(repo),
        )


def test_preflight_checks_both_roots_before_crossing_rename_boundary(tmp_path: Path) -> None:
    repo, journal, digest = _eligible(tmp_path)
    payload = load_journal(journal)
    rogue = Path(payload["roots"][1]["quarantine_path"])
    rogue.mkdir()
    before = journal.read_bytes()

    with pytest.raises(RetirementError, match="pre-purge root preflight failed"):
        purge(
            journal_path=journal,
            expected_journal_sha256=digest,
            **_binding(repo),
        )
    assert journal.read_bytes() == before
    assert load_journal(journal)["state"] == "PURGE_ELIGIBLE"
    assert (repo / "results/v16/payload.bin").is_file()

    rogue.rmdir()
    rollback_pre_purge(
        journal_path=journal,
        expected_journal_sha256=digest,
        reason="preflight stopped safely",
        **_binding(repo),
    )
    assert load_journal(journal)["state"] == "ROLLED_BACK_PRE_PURGE"


def test_full_purge_and_resume_receipt_do_not_preserve_deleted_bytes(
    tmp_path: Path,
) -> None:
    repo, journal, digest = _eligible(tmp_path)
    digest = purge(
        journal_path=journal,
        expected_journal_sha256=digest,
        **_binding(repo),
    )

    assert load_journal(journal)["state"] == "PURGED"
    assert not (repo / "results/v16").exists()
    assert not (repo / "results/v16_operator_advisory").exists()
    assert not list((repo / "results").glob(".*.v17-purge-*"))

    digest, receipt = mark_resumed(
        journal_path=journal,
        expected_journal_sha256=digest,
        active_schedules_restored=7,
        legacy_schedules_deleted=2,
        final_scan_clean=True,
        final_scan_sha256=FINAL_SCAN_SHA,
        **_binding(repo),
    )
    assert load_journal(journal)["state"] == "RESUMED"
    assert receipt["journal_sha256"] == digest
    assert receipt["secure_erasure"] is False
    assert receipt["deleted_file_bytes_preserved"] is False
    encoded = str(receipt).lower()
    assert "payload.bin" not in encoded
    assert "content_sha" not in encoded


@pytest.mark.parametrize(
    ("crash_event", "durable_step"),
    [
        ("RENAME_INTENT_COMMITTED", "RENAME_INTENT"),
        ("RENAMED_FILESYSTEM", "RENAME_INTENT"),
        ("RENAMED_COMMITTED", "RENAMED"),
        ("DELETE_INTENT_COMMITTED", "DELETE_INTENT"),
        ("DELETE_STARTED_COMMITTED", "DELETE_STARTED"),
        ("DELETED_FILESYSTEM", "DELETE_STARTED"),
        ("DELETED_COMMITTED", "DELETED"),
    ],
)
def test_crash_replay_continues_only_from_durable_step(
    tmp_path: Path,
    crash_event: str,
    durable_step: str,
) -> None:
    repo, journal, digest = _eligible(tmp_path)

    class SimulatedCrash(Exception):
        pass

    def crash(event: str, _root: object) -> None:
        if event == crash_event:
            raise SimulatedCrash(event)

    with pytest.raises(SimulatedCrash):
        purge(
            journal_path=journal,
            expected_journal_sha256=digest,
            _event_hook=crash,
            **_binding(repo),
        )

    crashed = load_journal(journal)
    assert crashed["roots"][0]["step"] == durable_step
    resumed_sha = journal_sha256(journal)
    final_sha = purge(
        journal_path=journal,
        expected_journal_sha256=resumed_sha,
        **_binding(repo),
    )
    assert journal_sha256(journal) == final_sha
    assert load_journal(journal)["state"] == "PURGED"
    assert not (repo / "results/v16").exists()


@pytest.mark.parametrize("injection_event", ["RENAMED_FILESYSTEM", "DELETED_FILESYSTEM"])
def test_broken_source_symlink_after_rename_is_durably_blocked(
    tmp_path: Path,
    injection_event: str,
) -> None:
    repo, journal, digest = _eligible(tmp_path)
    source = repo / "results/v16"

    def inject(event: str, _root: object) -> None:
        if event == injection_event and not source.is_symlink():
            source.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(PostPurgeBlockedError):
        purge(
            journal_path=journal,
            expected_journal_sha256=digest,
            _event_hook=inject,
            **_binding(repo),
        )
    blocked = load_journal(journal)
    assert blocked["state"] == BLOCKED_STATE
    assert source.is_symlink()


def test_recreated_source_after_rename_remains_across_irreversible_boundary(
    tmp_path: Path,
) -> None:
    repo, journal, digest = _eligible(tmp_path)
    source = repo / "results/v16"

    class SimulatedCrash(Exception):
        pass

    def recreate_then_crash(event: str, _root: object) -> None:
        if event == "RENAMED_FILESYSTEM":
            source.mkdir()
            raise SimulatedCrash

    with pytest.raises(SimulatedCrash):
        purge(
            journal_path=journal,
            expected_journal_sha256=digest,
            _event_hook=recreate_then_crash,
            **_binding(repo),
        )
    crashed_sha = journal_sha256(journal)
    assert load_journal(journal)["state"] == "PURGE_ELIGIBLE"

    with pytest.raises(PostPurgeBlockedError, match="both exist"):
        purge(
            journal_path=journal,
            expected_journal_sha256=crashed_sha,
            **_binding(repo),
        )
    assert load_journal(journal)["state"] == BLOCKED_STATE


def test_delete_operational_failure_after_rename_is_durably_blocked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, journal, digest = _eligible(tmp_path)
    original_rmtree = retirement_module.shutil.rmtree

    def fail_delete(_path: object) -> None:
        raise OSError("simulated unlink failure")

    monkeypatch.setattr(retirement_module.shutil, "rmtree", fail_delete)
    with pytest.raises(PostPurgeBlockedError, match="simulated unlink failure"):
        purge(
            journal_path=journal,
            expected_journal_sha256=digest,
            **_binding(repo),
        )
    blocked = load_journal(journal)
    assert blocked["state"] == BLOCKED_STATE
    assert blocked["roots"][0]["step"] == "DELETE_STARTED"

    monkeypatch.setattr(retirement_module.shutil, "rmtree", original_rmtree)
    final_sha = purge(
        journal_path=journal,
        expected_journal_sha256=journal_sha256(journal),
        repair_acknowledgement="filesystem deletion restored",
        **_binding(repo),
    )
    assert journal_sha256(journal) == final_sha
    assert load_journal(journal)["state"] == "PURGED"


def test_blocked_cutover_requires_repair_then_explicit_acknowledgement(
    tmp_path: Path,
) -> None:
    repo, journal, digest = _eligible(tmp_path)
    source = repo / "results/v16"

    def inject(event: str, _root: object) -> None:
        if event == "RENAMED_FILESYSTEM":
            source.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(PostPurgeBlockedError):
        purge(
            journal_path=journal,
            expected_journal_sha256=digest,
            _event_hook=inject,
            **_binding(repo),
        )

    with pytest.raises(PostPurgeBlockedError, match="acknowledgement"):
        purge(
            journal_path=journal,
            expected_journal_sha256=journal_sha256(journal),
            **_binding(repo),
        )

    source.unlink()
    final_sha = purge(
        journal_path=journal,
        expected_journal_sha256=journal_sha256(journal),
        repair_acknowledgement="removed unexpected broken source symlink",
        **_binding(repo),
    )
    assert journal_sha256(journal) == final_sha
    assert load_journal(journal)["state"] == "PURGED"


def test_resume_rejects_missing_schedule_or_scan_postconditions(tmp_path: Path) -> None:
    repo, journal, digest = _eligible(tmp_path)
    digest = purge(
        journal_path=journal,
        expected_journal_sha256=digest,
        **_binding(repo),
    )

    with pytest.raises(RetirementError, match="schedule postconditions"):
        mark_resumed(
            journal_path=journal,
            expected_journal_sha256=digest,
            active_schedules_restored=6,
            legacy_schedules_deleted=2,
            final_scan_clean=True,
            final_scan_sha256=FINAL_SCAN_SHA,
            **_binding(repo),
        )
    assert load_journal(journal)["state"] == "PURGED"

    with pytest.raises(RetirementError, match="lowercase SHA256"):
        mark_resumed(
            journal_path=journal,
            expected_journal_sha256=digest,
            active_schedules_restored=7,
            legacy_schedules_deleted=2,
            final_scan_clean=True,
            final_scan_sha256="not-a-sha",
            **_binding(repo),
        )
    assert load_journal(journal)["state"] == "PURGED"
