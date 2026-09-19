"""Full real script preload with actual Git bytes; running-install verifier is controlled."""

import hashlib
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations import native_bridge as bridge
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.journal_storage import JournalStorage
from quant_investor.factors.production_authority import FactorProductionStore
from quant_investor.strategy_records import store as financial_store
from quant_investor.system.storage import SecureSystemStorage


def test_complete_real_closed_module_preload_is_pure_and_cleans_up(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2]
    root = tmp_path / "repository"
    for relative in sorted(set(bridge._MODULES.values())):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / relative).read_bytes())
    for args in (
        ["init", "-q"],
        ["add", "."],
        [
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "native import fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    raw = canonical_json_bytes({"release_install_evidence": {"payload": {"final_commit": commit}}})
    checks, forbidden = [], []
    monkeypatch.setattr(
        bridge,
        "verify_running_release_install_input",
        lambda *a, **k: checks.append(True) or {"state": "PASS"},
    )

    def denied(*args, **kwargs):
        forbidden.append(True)
        pytest.fail("native module import attempted network or canonical publication")

    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(JournalStorage, "write", denied)
    monkeypatch.setattr(FactorProductionStore, "write_exact_once", denied)
    monkeypatch.setattr(financial_store, "publish_catalog", denied)
    monkeypatch.setattr(SecureSystemStorage, "_write_temporary_file", denied)
    old_modules = {
        name: module
        for name, module in sys.modules.items()
        if name in bridge._MODULES or name == "scripts" or name.startswith("scripts.")
    }
    for name in old_modules:
        del sys.modules[name]
    state = list(sys.path), list(sys.meta_path), dict(sys.path_importer_cache), sys.pycache_prefix
    operations_before = dict(bridge._OPERATIONS)
    try:
        with bridge.verified_native_context(
            release_input_raw=raw,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
            repository_root=str(root),
        ) as operations:
            assert len(checks) == 2 and set(operations) == set(operations_before)
            for name, relative in bridge._MODULES.items():
                assert sys.modules[name].__spec__.origin == str(root / relative)
                assert (root / relative).read_bytes() == (source / relative).read_bytes()
            for name, (module_name, function) in operations_before.items():
                assert operations[name] is getattr(sys.modules[module_name], function)
            for name in ("cn_dashboard_v2_selector", "export_cn_aggressive_dashboard_data"):
                assert sys.modules[name].__file__ == sys.modules["scripts." + name].__file__
            with pytest.raises(ContractError, match="MODULE_NOT_ALLOWED|LATE_REPOSITORY_IMPORT"):
                __import__("scripts.unlisted_native_module")
        assert not forbidden
        assert not any(name in sys.modules for name in bridge._MODULES)
        assert (
            list(sys.path),
            list(sys.meta_path),
            dict(sys.path_importer_cache),
            sys.pycache_prefix,
        ) == state
        assert dict(bridge._OPERATIONS) == operations_before
    finally:
        for name in list(sys.modules):
            if name in bridge._MODULES or name == "scripts" or name.startswith("scripts."):
                del sys.modules[name]
        sys.modules.update(old_modules)
