"""Historical probe keeps native output shape while rejecting mutation callbacks."""

import json, subprocess, sys
from pathlib import Path
import pytest
from quant_investor.system import release_install as module


def command_fixture(root, monkeypatch):
    install = root / "install"
    install.mkdir()
    repository = root / "repository"
    repository.mkdir()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "import_origin": "fixture",
                    "installed_code_manifest_sha256": "a" * 64,
                    "contract_catalog_sha256": "b" * 64,
                }
            ).encode(),
            b"",
        )

    monkeypatch.setattr(
        module.tempfile,
        "TemporaryDirectory",
        lambda **kw: pytest.fail("read-only probe created directory"),
    )
    monkeypatch.setattr(module.subprocess, "run", run)
    module._probe_install(Path(sys.executable), install, repository, read_only=True)
    return calls[0], install


def test_readonly_probe_uses_existing_cwd_and_disables_cache_writes(tmp_path, monkeypatch):
    (command, kwargs), install = command_fixture(tmp_path, monkeypatch)
    assert command[1:4] == ["-I", "-B", "-X"]
    assert command[4].startswith("pycache_prefix=")
    assert not Path(command[4].split("=", 1)[1]).exists()
    assert kwargs["cwd"] == install
    assert list(install.iterdir()) == []


@pytest.mark.parametrize(
    "operation",
    [
        "open(TARGET,'w').write('bad')",
        "os.mkdir(TARGET)",
        "__import__('socket').socket().connect(('127.0.0.1',9))",
    ],
)
def test_probe_audit_guard_blocks_real_child_side_effects(tmp_path, monkeypatch, operation):
    real_run = subprocess.run
    (command, _), _ = command_fixture(tmp_path, monkeypatch)
    guard = command[-1].split("import json, pathlib, quant_investor;", 1)[0]
    target = tmp_path / "forbidden"
    program = guard + "\nTARGET=" + repr(str(target)) + "\n" + operation
    result = real_run([sys.executable, "-I", "-B", "-c", program], capture_output=True, text=True)
    assert result.returncode != 0 and "ARCHIVE_PROBE_" in result.stderr
    assert not target.exists()
