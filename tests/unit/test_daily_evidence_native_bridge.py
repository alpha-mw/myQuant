"""Closed-import mechanics; fixture runtime verifier is explicitly substituted."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType, ModuleType
import pytest
from quant_investor.operations import native_bridge as bridge
from quant_investor.operations.daily_contract import ContractError


@pytest.fixture
def context(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/__init__.py").write_text('"""fixture package"""\n')
    (root / "scripts/entry.py").write_text(
        'def execute():\n    return "verified"\n'
        'def replay(**kwargs):\n    return {"replayed": True}\n'
    )
    for args in [
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ]:
        subprocess.run(["git", "-C", str(root), *args], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    raw = json.dumps(
        {"release_install_evidence": {"payload": {"final_commit": commit}}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    modules = {"scripts": "scripts/__init__.py", "scripts.entry": "scripts/entry.py"}
    monkeypatch.setattr(bridge, "_MODULES", MappingProxyType(modules))
    monkeypatch.setattr(
        bridge,
        "_OPERATIONS",
        MappingProxyType(
            {
                "execute": ("scripts.entry", "execute"),
                "completion_replay": ("scripts.entry", "replay"),
            }
        ),
    )
    calls = []
    monkeypatch.setattr(
        bridge,
        "verify_running_release_install_input",
        lambda *a, **k: (calls.append("verify") or {"state": "PASS"}),
    )
    # Other unit modules import native scripts during collection. Restore them
    # after each mechanical fixture; production never permits this removal.
    previous = {k: v for k, v in sys.modules.items() if k == "scripts" or k.startswith("scripts.")}
    for key in previous:
        del sys.modules[key]
    yield root, {
        "release_input_raw": raw,
        "expected_sha256": hashlib.sha256(raw).hexdigest(),
        "repository_root": str(root),
    }, calls
    for key in list(sys.modules):
        if key == "scripts" or key.startswith("scripts."):
            del sys.modules[key]
    sys.modules.update(previous)


def state():
    return list(sys.path), list(sys.meta_path), sys.pycache_prefix, dict(sys.path_importer_cache)


@pytest.mark.parametrize("failure", [None, "typed", "unexpected", "result"])
def test_two_verifications_and_cleanup_on_every_outcome(context, failure):
    _, args, calls = context
    before = state()
    error = ContractError if failure == "typed" else RuntimeError
    try:
        with bridge.verified_native_context(**args) as operations:
            assert calls == ["verify", "verify"]
            assert operations["execute"]() == "verified"
            if failure:
                raise error(failure)
    except (ContractError, RuntimeError):
        assert failure is not None
    assert state() == before
    assert "scripts" not in sys.modules and "scripts.entry" not in sys.modules
    with bridge.verified_native_context(**args) as operations:
        assert operations["execute"]() == "verified"
    assert calls == ["verify"] * 4
    assert state() == before


def test_wrong_sha_rejected_before_verification(context):
    _, args, calls = context
    with pytest.raises(ContractError, match="RELEASE_SHA_MISMATCH"):
        with bridge.verified_native_context(**{**args, "expected_sha256": "a" * 64}):
            pytest.fail("entered")
    assert calls == []


def test_active_completion_capability_is_identity_thread_and_scope_bound(context):
    from concurrent.futures import ThreadPoolExecutor
    from contextvars import copy_context

    root, args, _ = context
    request = {
        "release_input_sha256": args["expected_sha256"],
        "repository_root": str(root),
        "final_commit": json.loads(args["release_input_raw"])["release_install_evidence"][
            "payload"
        ]["final_commit"],
        "workspace": str(root),
        "trade_date": "20260820",
        "completion_ref": {"path": "completion.json", "sha256": "a" * 64},
    }
    assert bridge._invoke_active_completion_replay(**request) == (False, None)
    with bridge.verified_native_context(**args):
        assert bridge._invoke_active_completion_replay(**request) == (True, {"replayed": True})
        assert bridge._invoke_active_completion_replay(
            **{**request, "release_input_sha256": "f" * 64}
        ) == (False, None)
        copied = copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(
                copied.run, lambda: bridge._invoke_active_completion_replay(**request)
            ).result() == (False, None)
    assert bridge._invoke_active_completion_replay(**request) == (False, None)


def test_same_origin_preloaded_mutated_callable_rejected(context):
    root, args, calls = context
    module = ModuleType("scripts.entry")
    module.__file__ = str(root / "scripts/entry.py")
    module.execute = lambda: "mutated"
    sys.modules["scripts.entry"] = module
    with pytest.raises(ContractError, match="PRELOADED"):
        with bridge.verified_native_context(**args):
            pytest.fail("entered")
    assert calls == ["verify"]


def test_source_mutation_rejected_before_import(context):
    root, args, calls = context
    (root / "scripts/entry.py").write_text('raise AssertionError("must not execute")\n')
    with pytest.raises(ContractError, match="BLOB_MISMATCH"):
        with bridge.verified_native_context(**args):
            pytest.fail("entered")
    assert "scripts.entry" not in sys.modules
    assert calls == ["verify"]


def test_late_repository_import_rejected_even_with_package_path(context):
    _, args, _ = context
    with bridge.verified_native_context(**args):
        del sys.modules["scripts.entry"]
        with pytest.raises(ContractError, match="LATE_REPOSITORY_IMPORT"):
            __import__("scripts.entry")
        with pytest.raises(ContractError, match="MODULE_NOT_ALLOWED"):
            __import__("scripts.unlisted")


def test_ignored_pycache_cannot_replace_git_verified_source(context):
    root, args, _ = context
    cached = root / "scripts/__pycache__"
    cached.mkdir()
    (cached / f"entry.{sys.implementation.cache_tag}.pyc").write_bytes(b"untrusted bytecode")
    with bridge.verified_native_context(**args) as operations:
        assert operations["execute"]() == "verified"


def test_unverified_runtime_cannot_preload(context, monkeypatch):
    _, args, _ = context
    monkeypatch.setattr(
        bridge, "verify_running_release_install_input", lambda *a, **k: {"state": "BLOCKED"}
    )
    with pytest.raises(ContractError, match="RUNTIME_NOT_VERIFIED"):
        with bridge.verified_native_context(**args):
            pytest.fail("entered")
    assert "scripts.entry" not in sys.modules


def test_symlink_source_rejected(context):
    root, args, _ = context
    source = root / "scripts/entry.py"
    copy = root / "outside.py"
    copy.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(copy)
    with pytest.raises(ContractError, match="SOURCE_UNSAFE"):
        with bridge.verified_native_context(**args):
            pytest.fail("entered")
