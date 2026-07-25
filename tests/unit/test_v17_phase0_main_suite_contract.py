from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HARNESS = _load_script(
    "v17_phase0_main_suite_harness",
    "scripts/v17_phase0_main_suite_harness.py",
)
WRAPPER = _load_script(
    "v17_phase0_main_suite_wrapper",
    "scripts/v17_phase0_main_suite_wrapper.py",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _attestation_frame(
    *,
    phase: int,
    nonce: bytes,
    challenge_sha: bytes,
    child_pid: int,
    parent_pid: int,
) -> bytes:
    payload: dict[str, object] = {
        "challenge_binding_sha256": challenge_sha.hex(),
        "frame": {
            1: "pre_import",
            2: "pre_collection",
            3: "terminal_complete",
        }[phase],
        "pid": child_pid,
        "ppid": parent_pid,
    }
    if phase == 3:
        payload.update(
            {
                "final_loaded_modules": {
                    "classification_counts": {"candidate": 1},
                    "count": 1,
                    "rows_sha256": "a" * 64,
                },
                "pytest_exit_code": 0,
            }
        )
    raw = _canonical(payload)
    return (
        HARNESS.ATTEST_HEADER.pack(
            HARNESS.ATTEST_MAGIC,
            HARNESS.PROTOCOL_VERSION,
            phase,
            0,
            len(raw),
            nonce,
            hashlib.sha256(raw).digest(),
        )
        + raw
    )


def test_policy_parser_is_canonical_but_leaves_key_semantics_to_schema() -> None:
    raw = b'{"extra":1}\n'
    assert HARNESS._parse_policy_bytes(raw) == {"extra": 1}
    with pytest.raises(HARNESS.MainSuiteHarnessError, match="unexpected keys"):
        HARNESS.validate_policy_bytes(raw)
    with pytest.raises(HARNESS.MainSuiteHarnessError, match="duplicate"):
        HARNESS._parse_policy_bytes(b'{"extra":1,"extra":2}\n')
    with pytest.raises(HARNESS.MainSuiteHarnessError, match="canonical"):
        HARNESS._parse_policy_bytes(b'{ "extra": 1 }\n')


def test_attestation_consumer_requires_three_ordered_bound_frames() -> None:
    nonce = b"n" * 32
    challenge_sha = b"c" * 32
    child_pid = 101
    parent_pid = 202
    raw = b"".join(
        _attestation_frame(
            phase=phase,
            nonce=nonce,
            challenge_sha=challenge_sha,
            child_pid=child_pid,
            parent_pid=parent_pid,
        )
        for phase in (1, 2, 3)
    )
    frames: list[dict[str, object]] = []
    offset = HARNESS._consume_frames(
        bytearray(raw),
        0,
        frames,
        nonce=nonce,
        challenge_sha=challenge_sha,
        child_pid=child_pid,
        parent_pid=parent_pid,
    )
    assert offset == len(raw)
    assert [frame["phase"] for frame in frames] == [1, 2, 3]
    terminal_payload = frames[2]["payload"]
    assert type(terminal_payload) is dict
    assert terminal_payload["pytest_exit_code"] == 0

    tampered = bytearray(raw)
    tampered[-1] ^= 1
    with pytest.raises(HARNESS.MainSuiteHarnessError, match="digest"):
        HARNESS._consume_frames(
            tampered,
            0,
            [],
            nonce=nonce,
            challenge_sha=challenge_sha,
            child_pid=child_pid,
            parent_pid=parent_pid,
        )


def test_attestation_consumer_rejects_phase_reordering_and_trailing_bytes() -> None:
    nonce = b"n" * 32
    challenge_sha = b"c" * 32
    reordered = _attestation_frame(
        phase=2,
        nonce=nonce,
        challenge_sha=challenge_sha,
        child_pid=101,
        parent_pid=202,
    )
    with pytest.raises(HARNESS.MainSuiteHarnessError, match="header"):
        HARNESS._consume_frames(
            bytearray(reordered),
            0,
            [],
            nonce=nonce,
            challenge_sha=challenge_sha,
            child_pid=101,
            parent_pid=202,
        )

    complete = b"".join(
        _attestation_frame(
            phase=phase,
            nonce=nonce,
            challenge_sha=challenge_sha,
            child_pid=101,
            parent_pid=202,
        )
        for phase in (1, 2, 3)
    )
    with pytest.raises(HARNESS.MainSuiteHarnessError, match="trailing"):
        HARNESS._consume_frames(
            bytearray(complete + b"x"),
            0,
            [],
            nonce=nonce,
            challenge_sha=challenge_sha,
            child_pid=101,
            parent_pid=202,
        )


def test_terminal_frame_cap_is_enforced_before_payload_decode() -> None:
    nonce = b"n" * 32
    challenge_sha = b"c" * 32
    oversized = HARNESS.ATTEST_HEADER.pack(
        HARNESS.ATTEST_MAGIC,
        HARNESS.PROTOCOL_VERSION,
        3,
        0,
        HARNESS.MAX_TERMINAL_FRAME_BYTES + 1,
        nonce,
        b"d" * 32,
    )
    prefix = b"".join(
        _attestation_frame(
            phase=phase,
            nonce=nonce,
            challenge_sha=challenge_sha,
            child_pid=101,
            parent_pid=202,
        )
        for phase in (1, 2)
    )
    with pytest.raises(HARNESS.MainSuiteHarnessError, match="header"):
        HARNESS._consume_frames(
            bytearray(prefix + oversized),
            0,
            [],
            nonce=nonce,
            challenge_sha=challenge_sha,
            child_pid=101,
            parent_pid=202,
        )


def test_project_import_bypasses_arbitrary_meta_path_finder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "candidate"
    package = candidate / "quant_investor"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    authority = tmp_path / "authority"
    authority.mkdir()
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()

    called = False

    class MaliciousFinder:
        def find_spec(self, fullname, path=None, target=None):
            nonlocal called
            called = True
            raise AssertionError(fullname)

    guard = object.__new__(WRAPPER._CandidateImportGuard)
    guard.policy = {
        "discovery_mode": True,
        "module_policy": {
            "allowed_namespace_modules": [],
            "allowed_no_origin_modules": [],
            "authority_root": str(authority),
            "candidate_content_binding": "OUTER_SOURCE_STATE",
            "candidate_module_source_paths": [],
            "candidate_root": str(candidate),
            "distribution_ownership": [],
            "runtime_roots": [],
            "site_packages_root": str(site_packages),
            "unowned_site_package_files": [],
        },
    }
    guard.owners = {}
    guard.unowned = {}
    monkeypatch.setattr(sys, "path", [str(candidate), *sys.path])
    monkeypatch.setattr(sys, "meta_path", [guard, MaliciousFinder(), *sys.meta_path])

    spec = guard.find_spec("quant_investor")
    assert spec is not None
    assert type(spec.loader) is importlib.machinery.SourceFileLoader
    assert Path(spec.origin).resolve() == (package / "__init__.py").resolve()
    assert called is False
    WRAPPER._assert_guard_head(guard)


def test_guard_head_detects_displacement(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = object.__new__(WRAPPER._CandidateImportGuard)
    monkeypatch.setattr(sys, "meta_path", [guard])
    WRAPPER._assert_guard_head(guard)
    monkeypatch.setattr(sys, "meta_path", [object(), guard])
    with pytest.raises(WRAPPER.MainSuiteWrapperError, match="not first"):
        WRAPPER._assert_guard_head(guard)


def test_only_pytest_assertion_hook_may_be_removed_ahead_of_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard = object.__new__(WRAPPER._CandidateImportGuard)
    assertion_hook_type = type(
        "AssertionRewritingHook",
        (),
        {"__module__": "_pytest.assertion.rewrite"},
    )
    assertion_hook = assertion_hook_type()
    monkeypatch.setattr(sys, "meta_path", [assertion_hook, guard])
    WRAPPER._restore_guard_head_after_pytest(guard)
    assert sys.meta_path == [guard, assertion_hook]

    monkeypatch.setattr(sys, "meta_path", [object(), guard])
    with pytest.raises(WRAPPER.MainSuiteWrapperError, match="unknown finder"):
        WRAPPER._restore_guard_head_after_pytest(guard)


def test_sourceless_finder_stays_behind_phase0_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quant_investor import _sourceless

    guard = object.__new__(WRAPPER._CandidateImportGuard)
    guard._myquant_phase0_candidate_guard = True
    monkeypatch.setattr(sys, "meta_path", [guard])
    _sourceless.install_sourceless_finder()
    assert sys.meta_path[0] is guard
    assert isinstance(sys.meta_path[1], _sourceless._QuantInvestorSourcelessFinder)


def test_bytecode_policy_requires_startup_prefix_and_empty_private_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pycache = tmp_path / "pycache"
    pycache.mkdir(mode=0o700)
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(pycache))
    monkeypatch.setattr(sys, "pycache_prefix", str(pycache))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    assert WRAPPER._bytecode_policy() == {
        "dont_write_bytecode": True,
        "pycache_prefix": str(pycache),
    }

    (pycache / "unexpected.pyc").write_bytes(b"x")
    with pytest.raises(WRAPPER.MainSuiteWrapperError, match="not empty"):
        WRAPPER._bytecode_policy()


def test_parent_pycache_binding_detects_directory_entry_churn(
    tmp_path: Path,
) -> None:
    pycache = tmp_path / "pycache"
    pycache.mkdir(mode=0o700)
    before = HARNESS._empty_private_directory_binding(
        pycache,
        label="PYTHONPYCACHEPREFIX",
    )
    transient = pycache / "transient.pyc"
    transient.write_bytes(b"x")
    transient.unlink()
    after = HARNESS._empty_private_directory_binding(
        pycache,
        label="PYTHONPYCACHEPREFIX",
    )
    assert before["st_ino"] == after["st_ino"]
    assert (
        before["st_mtime_ns"],
        before["st_ctime_ns"],
    ) != (
        after["st_mtime_ns"],
        after["st_ctime_ns"],
    )


def test_stable_file_binding_rejects_symlink_and_multiple_links(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"x = 1\n")
    symlink = tmp_path / "symlink.py"
    symlink.symlink_to(source)
    with pytest.raises(WRAPPER.MainSuiteWrapperError, match="cannot open"):
        WRAPPER._stable_file_binding(str(symlink))

    linked = tmp_path / "linked.py"
    linked.hardlink_to(source)
    with pytest.raises(WRAPPER.MainSuiteWrapperError, match="unsafe"):
        WRAPPER._stable_file_binding(str(source))
