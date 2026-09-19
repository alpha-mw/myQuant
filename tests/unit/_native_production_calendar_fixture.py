"""Explicit offline HTTPS and install-admission seams for production Calendar tests.

No result from this fixture is real provider, deployed or unattended evidence.
"""

from datetime import datetime
import hashlib
from pathlib import Path
import json
import socket
import zipfile

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.system.store import object_ref_for_artifact
from quant_investor.system import release_install, release as release_module
from quant_investor.market import tushare_calendar_authority as native
from quant_investor.market import tushare_transport as transport
from quant_investor.market import next_session_acquisition as acquisition
from quant_investor.market import _calendar_production_transport as recorder
from test_tushare_calendar_authority import _docs, _provider_raw, CREATED_AT


def offline_https(monkeypatch):
    """Replace only sockets and the explicit guard seam, retaining real client/hooks."""
    calls = []

    class Connection:
        def __init__(self, host, port, **kwargs):
            assert host in {"tushare.pro", "api.tushare.pro"} and port == 443
            self.host, self.raw = host, None

        def request(self, method, path, body=None, headers=None):
            if self.host == "tushare.pro":
                assert method == "GET" and path == "/document/2?doc_id=26" and body is None
                self.raw = _docs()
                calls.append("DOCUMENTATION")
            else:
                assert method == "POST" and path == "/"
                request = json.loads(body)
                assert request["api_name"] == "trade_cal"
                params = request["params"]
                self.raw = _provider_raw(
                    params["exchange"],
                    cutoff=datetime.strptime(params["end_date"], "%Y%m%d").date(),
                )
                calls.append(params["exchange"])

        def getresponse(self):
            return self

        status = 200

        def read(self, count):
            return self.raw[:count]

        def getheaders(self):
            return [("Content-Type", "text/html; charset=utf-8")]

        def close(self):
            pass

    monkeypatch.setattr(transport, "_HTTPS_CONNECTION", Connection)
    monkeypatch.setattr(native.http.client, "HTTPSConnection", Connection)

    def forbidden(*args, **kwargs):
        raise AssertionError("OFFLINE_CALENDAR_TEST_ATTEMPTED_REAL_NETWORK")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    # Explicit test admission seam: the normal guard rejects this socket fixture.
    guard = lambda: None
    monkeypatch.setattr(recorder, "_guard_route", guard)
    monkeypatch.setattr(recorder, "_ORIGINAL_GUARD", guard)
    monkeypatch.setenv("TUSHARE_TOKEN", "N" * 40)
    return calls


def install_seam(root, monkeypatch):
    """Retain native artifact/wheel custody; only running installation admission is controlled."""
    wheel_path = root / "unit-release.whl"
    rows = []
    package_parent = Path(native.__file__).resolve().parents[2]
    with zipfile.ZipFile(wheel_path, "w") as wheel:
        for name in sorted(recorder._MODULE_PATHS):
            raw = (package_parent / name).read_bytes()
            wheel.writestr(name, raw)
            rows.append(
                {"path": name, "byte_sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
            )
    wheel_path.chmod(0o600)
    wheel_raw = wheel_path.read_bytes()
    wheel_sha = hashlib.sha256(wheel_raw).hexdigest()
    manifest = {"domain": release_module.INSTALLED_CODE_MANIFEST_DOMAIN, "files": rows}
    manifest_sha = hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    release = seal_artifact(
        "system.release",
        {
            "release_id": "offline-native-production-calendar",
            "state": "OPERATIONAL",
            "code_sha256": "1" * 64,
            "wheel_sha256": wheel_sha,
            "code_manifest_sha256": manifest_sha,
        },
        created_at=CREATED_AT,
    )
    release_ref = object_ref_for_artifact(release)
    evidence = release_install.build_release_install_evidence(
        final_commit="4" * 40,
        final_tree="5" * 40,
        code_tree_sha256_value="1" * 64,
        git_code_manifest_sha256_value=manifest_sha,
        release_ref=release_ref,
        source_archive={
            "path": str(root / "unit-source.tar.gz"),
            "byte_sha256": "6" * 64,
            "size": 1,
        },
        wheel={"path": str(wheel_path), "byte_sha256": wheel_sha, "size": len(wheel_raw)},
        install_root=str(root / "unit-installed"),
        python_executable=str(root / "unit-installed/bin/python"),
        python_executable_sha256="8" * 64,
        import_origin=str(root / "unit-installed/quant_investor/__init__.py"),
        installed_code_manifest_sha256=manifest_sha,
        contract_catalog_sha256_value="a" * 64,
        lockfile_sha256="b" * 64,
        created_at=CREATED_AT,
    )
    raw = canonical_json_bytes({"release_install_evidence": evidence, "deployed_release": release})
    verification = {
        "state": "PASS",
        "release_ref": release_ref,
        "wheel_sha256": wheel_sha,
        "installed_code_manifest_sha256": manifest_sha,
        "contract_catalog_sha256": "a" * 64,
        "import_origin": evidence["payload"]["import_origin"],
    }

    def components(given, **kwargs):
        assert given == raw
        return evidence, release, verification

    monkeypatch.setattr(native, "_release_install_components", components)
    monkeypatch.setattr(
        release_install, "verify_running_release_install_input", lambda *a, **k: verification
    )
    monkeypatch.setattr(
        acquisition, "verify_running_release_install_input", lambda *a, **k: verification
    )
    monkeypatch.setattr(release_module, "installed_code_manifest", lambda: manifest)
    path = root / "release.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    return {"path": path.name, "sha256": hashlib.sha256(raw).hexdigest()}
