from __future__ import annotations

import ast
import base64
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from typing import Any, Mapping, NoReturn, cast
import zipfile

import pytest

import quant_investor.v17_v2_contract.package_parity as package_parity_module
from quant_investor.v17_v2_contract.package_parity import (
    PackageParityError,
    collect_physical_source_superset,
    main as package_parity_main,
    validate_hatch_namespace_rows,
    verify_package_payload_parity,
)
from quant_investor.v17_v2_contract.resources import (
    LEDGER_IMPLEMENTATION_MODULES,
    PACKAGE_ASSET_SHA256S,
    PACKAGE_MANIFEST_PATH,
    PACKAGE_MANIFEST_SHA256,
    PackageResourceError,
    expected_ledger_contract_bindings,
    expected_ledger_implementation_bindings,
    load_package_manifest,
    read_packaged_asset,
    verify_packaged_assets,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = REPO_ROOT / "quant_investor" / "v17"
PACKAGE_ROOT = REPO_ROOT / "quant_investor" / "v17_v2_contract"
EXPECTED_PACKAGE_MANIFEST_SHA256 = (
    "0a9a43cfe0bcd9dd5036daf72fdea6187fbf290c1faf2d00ca2397bbd95a950c"
)

LEGACY_FROZEN_SHA256S = {
    "resources/deep_research_template.v1.json": (
        "434cf726270d5f65eb7a0d2e2f2569363b281be23c16ee08304acb6962cc6537"
    ),
    "resources/quant_factor_set.v1.json": (
        "670d18dd8f164f3390ee9838b626a7fd893f699042e58ab71d303c022eb47c56"
    ),
    "resources/retirement_scan_allowlist.json": (
        "54931a0a345f91da763fa1306953d9b8ad4a92bdfdfea5195cfef5b27d2f209d"
    ),
    "resources/shadow_policy.v1.json": (
        "c55331e1d7e67e1a958491a616ccd6e2413ba3413fb90d53ea93d61d5e979137"
    ),
    "resources/state_machine.v1.json": (
        "241b9c784cc3623a1d81a7a706b15abe44bb0157ccdbd3da36a15ef4ff6f60f4"
    ),
    "schemas/dataset_manifest.v1.schema.json": (
        "46372f960ea9baf424f55d3abc603c8c171e95933f72e64c06fb52ada315061e"
    ),
    "schemas/deep_research_request.v2.schema.json": (
        "e690e320aa1c9afaec5407a6beb94037609604339133a336721ad1e83de6a7ab"
    ),
    "schemas/deep_research_response.v2.schema.json": (
        "a33ce2171f0a2624d6fa861b26d7ed12282582606a97550b953d92de0c9d65cc"
    ),
    "schemas/execution_cost_policy.v1.schema.json": (
        "3e1cf5aa3e05fee0fc798c07844dcbd93ce40904f9cc0cf440485352be1c7678"
    ),
    "schemas/generation_catalog.v1.schema.json": (
        "85eaf777c72bc5ef867275e5a8f494a81ff84fc632cc1777516c1d9f0a9be4c8"
    ),
    "schemas/holdings_snapshot.v1.schema.json": (
        "3d2d8ad9a9c54ff2651c282cf5aef0c75a81a41fbbfbed3099aa85731e04a981"
    ),
    "schemas/observation_disposition.v1.schema.json": (
        "49f8d80550772f376e5beaa63e814657c5f4340e5418c81a8268e27370f73cbe"
    ),
    "schemas/portfolio_risk_policy_snapshot.v1.schema.json": (
        "d3441aa64f2b5a5fdd650fe499ebff0a6b9077b5139f2ccc07519c004ad21136"
    ),
    "schemas/pretrade_result.v1.schema.json": (
        "5bce9ff0937d259297bcee458df0cefc807886243c9197026159aa538ba3d350"
    ),
    "schemas/regime_portfolio_overlay.v1.schema.json": (
        "89a3d31ab0b1817a0fa81e34a85eac6e79190e35bc387141196d1152a28d8061"
    ),
    "schemas/shadow_output.v1.schema.json": (
        "ad5298f44b3feb998f4f7a7ae7b2c63a2240531f4bf1e847d6433a7d81a0ff61"
    ),
    "schemas/shadow_state.v1.schema.json": (
        "e724ab3bbf17616d7d1de72e05f47fd823ae030624d76c9845a5c26b728c18a5"
    ),
    "schemas/source_manifest.v2.schema.json": (
        "e11bad313af50e7453265912fadad472c3c092c03e4d1d02b69db7b9ac773b92"
    ),
    "schemas/trade_permission.v1.schema.json": (
        "f16cd9cae575f63535981752fc8d26a98835062e835fd60815dbe8016f325688"
    ),
}


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _load_canonical_json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    payload = json.loads(
        raw,
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    assert isinstance(payload, dict)
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert raw == canonical + b"\n"
    return payload


def test_frozen_legacy_prototypes_remain_exact() -> None:
    observed = {
        path.relative_to(LEGACY_ROOT).as_posix()
        for directory in ("resources", "schemas")
        for path in (LEGACY_ROOT / directory).glob("*.json")
    }
    assert observed == set(LEGACY_FROZEN_SHA256S)
    for relative_path, expected_sha256 in LEGACY_FROZEN_SHA256S.items():
        raw = (LEGACY_ROOT / relative_path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha256


def test_contract_import_isolated_from_legacy_v17_runtime() -> None:
    code = """
import json
import sys
import quant_investor.v17_v2_contract as contract
import quant_investor.v17_v2_contract.resources
legacy = sorted(
    name for name in sys.modules
    if name == "quant_investor.v17" or name.startswith("quant_investor.v17.")
)
print(json.dumps({
    "protocol": contract.PROTOCOL_VERSION,
    "runtime_authority": contract.RUNTIME_AUTHORITY,
    "legacy": legacy,
}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "legacy": [],
        "protocol": "myquant.v17.v2",
        "runtime_authority": False,
    }


def test_contract_static_import_closure_is_stdlib_or_sibling_only() -> None:
    stdlib = set(sys.stdlib_module_names) | {"__future__"}
    for path in PACKAGE_ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    continue
                module = node.module or ""
                top_level = module.partition(".")[0]
                assert top_level in stdlib or module.startswith("quant_investor.v17_v2_contract"), (
                    path.name,
                    module,
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    top_level = alias.name.partition(".")[0]
                    assert top_level in stdlib or alias.name.startswith(
                        "quant_investor.v17_v2_contract"
                    ), (path.name, alias.name)


def test_new_package_json_is_canonical_and_identifiers_are_unique() -> None:
    resource_paths = sorted((PACKAGE_ROOT / "resources").glob("*.json"))
    schema_paths = sorted((PACKAGE_ROOT / "schemas").glob("*.json"))
    paths = resource_paths + schema_paths
    assert resource_paths and schema_paths
    relative_paths = [path.relative_to(PACKAGE_ROOT).as_posix() for path in paths]
    assert len(relative_paths) == len({path.casefold() for path in relative_paths})

    resource_versions: list[str] = []
    for path in resource_paths:
        payload = _load_canonical_json(path)
        version = payload.get("version")
        assert isinstance(version, str)
        assert version.startswith("myquant.v17.v2.")
        resource_versions.append(version)

    schema_ids: list[str] = []
    artifact_versions: list[str] = []
    for path in schema_paths:
        payload = _load_canonical_json(path)
        schema_id = payload.get("$id")
        properties = payload.get("properties")
        assert isinstance(schema_id, str)
        assert isinstance(properties, dict)
        version_schema = properties.get("version")
        assert isinstance(version_schema, dict)
        artifact_version = version_schema.get("const")
        assert isinstance(artifact_version, str)
        assert schema_id == artifact_version.removesuffix(".v1") + ".schema.v1"
        schema_ids.append(schema_id)
        artifact_versions.append(artifact_version)

    resource_keys = [value.casefold() for value in resource_versions]
    schema_keys = [value.casefold() for value in schema_ids]
    artifact_keys = [value.casefold() for value in artifact_versions]
    assert len(resource_keys) == len(set(resource_keys))
    assert len(schema_keys) == len(set(schema_keys))
    assert len(artifact_keys) == len(set(artifact_keys))
    assert set(resource_keys).isdisjoint(schema_keys)
    assert set(schema_keys).isdisjoint(artifact_keys)


def test_complete_source_role_matrix_authorizes_shadow_runtime_only() -> None:
    payload = _load_canonical_json(PACKAGE_ROOT / "resources" / "source_role_matrix.v1.json")
    assert payload["authority"] is False
    assert payload["completeness"] == "COMPLETE"
    assert payload["runtime_usable"] is True
    assert payload["forbidden_role_suffixes"] == ["_verification_receipt"]
    pending_registry = payload["pending_registry"]
    assert pending_registry == []

    roles = payload["roles"]
    assert isinstance(roles, list)
    role_names = [item["role"] for item in roles]
    assert role_names == sorted(role_names, key=str.casefold)
    assert len(role_names) == len(set(name.casefold() for name in role_names))
    assert set(role_names) == {
        "H00300_total_return_dataset",
        "cn_open_day_calendar_dataset",
        "corporate_actions_dataset",
        "deep_evidence_dataset",
        "fundamental_generation_catalog",
        "fundamental_raw_tables_dataset",
        "macro_overlay",
        "market_bars_dataset",
        "market_pointer",
        "market_snapshot_manifest",
        "markov_overlay",
        "official_delisting_cash_dataset",
        "pit_generation_catalog",
        "portfolio_required_inputs",
        "risk_policy_snapshot",
    }
    pending_roles = {item["role"] for item in roles if item["schema_status"] == "PENDING"}
    assert pending_roles == set()


def test_package_manifest_and_loader_bind_the_exact_asset_inventory() -> None:
    manifest_path = PACKAGE_ROOT / PACKAGE_MANIFEST_PATH
    manifest_raw = manifest_path.read_bytes()
    assert hashlib.sha256(manifest_raw).hexdigest() == EXPECTED_PACKAGE_MANIFEST_SHA256
    assert PACKAGE_MANIFEST_SHA256 == EXPECTED_PACKAGE_MANIFEST_SHA256

    discovered = {
        path.relative_to(PACKAGE_ROOT).as_posix()
        for directory in ("resources", "schemas")
        for path in (PACKAGE_ROOT / directory).glob("*.json")
    }
    assert set(PACKAGE_ASSET_SHA256S) == discovered
    assert verify_packaged_assets() == dict(PACKAGE_ASSET_SHA256S)
    for relative_path, expected_sha256 in PACKAGE_ASSET_SHA256S.items():
        raw = read_packaged_asset(relative_path)
        assert hashlib.sha256(raw).hexdigest() == expected_sha256

    manifest = load_package_manifest()
    assert manifest["authority"] is False
    assert manifest["runtime_usable"] is True
    assert manifest["distribution"] == {
        "name": "quant-investor",
        "version": "17.0.0",
    }
    assert manifest["self_binding"] == {
        "byte_sha256_source": ("quant_investor.v17_v2_contract.resources.PACKAGE_MANIFEST_SHA256"),
        "relative_path": PACKAGE_MANIFEST_PATH,
    }

    listed_assets = {
        item["relative_path"]: item["byte_sha256"]
        for section in ("resources", "schemas")
        for item in manifest[section]
    }
    assert listed_assets == {
        relative_path: expected_sha256
        for relative_path, expected_sha256 in PACKAGE_ASSET_SHA256S.items()
        if relative_path != PACKAGE_MANIFEST_PATH
    }
    assert [item["relative_path"] for item in manifest["resources"]] == sorted(
        item["relative_path"] for item in manifest["resources"]
    )
    assert [item["relative_path"] for item in manifest["schemas"]] == sorted(
        item["relative_path"] for item in manifest["schemas"]
    )

    expected_legacy = {
        f"quant_investor/v17/{relative_path}": expected_sha256
        for relative_path, expected_sha256 in LEGACY_FROZEN_SHA256S.items()
    }
    observed_legacy = {
        item["relative_path"]: item["byte_sha256"] for item in manifest["frozen_legacy_prototypes"]
    }
    assert observed_legacy == expected_legacy


def test_ledger_binding_builders_are_exact_and_byte_backed() -> None:
    manifest = load_package_manifest()
    contract_bindings = expected_ledger_contract_bindings()
    assert contract_bindings["package_manifest_sha256"] == PACKAGE_MANIFEST_SHA256
    assert contract_bindings["resource_bindings"] == sorted(
        [
            {
                "binding_id": row["resource_version"],
                "relative_path": row["relative_path"],
                "byte_sha256": row["byte_sha256"],
            }
            for row in manifest["resources"]
        ],
        key=lambda row: (
            row["binding_id"],
            row["relative_path"],
            row["byte_sha256"],
        ),
    )
    assert contract_bindings["schema_bindings"] == sorted(
        [
            {
                "binding_id": row["schema_id"],
                "relative_path": row["relative_path"],
                "byte_sha256": row["byte_sha256"],
            }
            for row in manifest["schemas"]
        ],
        key=lambda row: (
            row["binding_id"],
            row["relative_path"],
            row["byte_sha256"],
        ),
    )

    implementation_bindings = expected_ledger_implementation_bindings()
    assert {row["module_id"] for row in implementation_bindings} == set(
        LEDGER_IMPLEMENTATION_MODULES
    )
    for row in implementation_bindings:
        path = PACKAGE_ROOT.parent / row["relative_path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["byte_sha256"]


def test_package_resource_loader_rejects_unlisted_paths_before_read() -> None:
    with pytest.raises(PackageResourceError, match="unknown"):
        read_packaged_asset("../v17/resources/shadow_policy.v1.json")
    with pytest.raises(PackageResourceError, match="unknown"):
        read_packaged_asset("resources/SHADOW_POLICY.v1.json")


def _record_digest(raw: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode(
        "ascii"
    )


def _record_rows(
    files: dict[str, bytes],
    *,
    record_path: str,
    omit_paths: set[str] | None = None,
    tamper_path: str | None = None,
    tamper_size_path: str | None = None,
    extra_rows: dict[str, bytes] | None = None,
    raw_extra_rows: list[str] | None = None,
) -> bytes:
    rows: list[str] = []
    omit_paths = omit_paths or set()
    for relative_path, raw in sorted(files.items()):
        if relative_path in omit_paths:
            continue
        if relative_path == record_path:
            rows.append(f"{relative_path},,\n")
            continue
        digest = "sha256=deadbeef" if relative_path == tamper_path else _record_digest(raw)
        size = str(len(raw) + 1) if relative_path == tamper_size_path else str(len(raw))
        rows.append(f"{relative_path},{digest},{size}\n")
    for relative_path, raw in sorted((extra_rows or {}).items()):
        rows.append(f"{relative_path},{_record_digest(raw)},{len(raw)}\n")
    rows.extend(raw_extra_rows or [])
    return "".join(rows).encode("utf-8")


def _write_files(root: Path, files: dict[str, bytes]) -> None:
    for relative_path, raw in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)


def _pip_25_2_posix_wrapper(environment: Path, target: str) -> bytes:
    module, callable_name = target.split(":", 1)
    import_name = callable_name.split(".", 1)[0]
    return (
        f"#!{environment / 'bin' / 'python'}\n"
        "import sys\n"
        f"from {module} import {import_name}\n"
        "if __name__ == '__main__':\n"
        "    if sys.argv[0].endswith('.exe'):\n"
        "        sys.argv[0] = sys.argv[0][:-4]\n"
        f"    sys.exit({callable_name}())\n"
    ).encode("utf-8")


def _write_synthetic_distribution(
    tmp_path: Path,
    *,
    installed_mutations: dict[str, bytes | None] | None = None,
    installed_dist_info_mutations: dict[str, bytes | None] | None = None,
    sdist_prefix: str = "quant_investor-17.0.0",
    sdist_pyproject_raw: bytes | None = None,
    omit_sdist_pyproject: bool = False,
    sdist_special_type: bytes | None = None,
    wheel_prefix: str = "",
    wheel_metadata_name: str = "quant-investor",
    wheel_metadata_version: str = "17.0.0",
    wheel_entry_points_raw: bytes | None = None,
    omit_wheel_entry_points: bool = False,
    wheel_dist_info_extra_files: dict[str, bytes] | None = None,
    wheel_special_directory_type: int | None = None,
    wheel_record_extra_rows: dict[str, bytes] | None = None,
    wheel_record_omit_paths: set[str] | None = None,
    wheel_record_tamper_path: str | None = None,
    wheel_extra_dist_info_root: bool = False,
    wheel_dist_info_name: str = "quant_investor-17.0.0.dist-info",
    wheel_malformed_record: bool = False,
    installed_record_extra_rows: dict[str, bytes] | None = None,
    installed_record_raw_extra_rows: list[str] | None = None,
    installed_record_omit_paths: set[str] | None = None,
    installed_record_tamper_path: str | None = None,
    installed_record_tamper_size_path: str | None = None,
    direct_url_payload: dict[str, object] | None = None,
    omit_direct_url: bool = False,
    installed_dist_info_name: str = "quant_investor-17.0.0.dist-info",
    metadata_name: str = "quant-investor",
    metadata_version: str = "17.0.0",
    installed_script_names: set[str] | None = None,
) -> dict[str, Path]:
    files = {
        "__init__.py": b'__version__ = "17.0.0"\n',
        "cli/main.py": b"def main():\n    return 0\n",
        "v17_v2_contract/__init__.py": b'PROTOCOL_VERSION = "myquant.v17.v2"\n',
        "v17_v2_contract/resources/main_suite_runtime_policy.v1.json": (b'{"authority":false}\n'),
        "v17_v2_contract/schemas/main_suite_runtime_policy.v1.schema.json": (
            b'{"type":"object"}\n'
        ),
    }
    source = tmp_path / "source" / "quant_investor"
    environment = tmp_path / "env"
    site_packages = environment / "lib" / "python" / "site-packages"
    installed = site_packages / "quant_investor"
    (tmp_path / "source").mkdir(parents=True)
    source_pyproject_raw = (
        b'[project]\nname = "quant-investor"\nversion = "17.0.0"\n'
        b"\n[project.scripts]\n"
        b'app = "web.main:app"\n'
        b'quant-investor = "quant_investor.cli.main:main"\n'
    )
    source_pyproject = tmp_path / "source" / "pyproject.toml"
    source_pyproject.write_bytes(source_pyproject_raw)
    _write_files(source, files)
    _write_files(installed, files)
    for relative_path, raw in (installed_mutations or {}).items():
        path = installed / relative_path
        if raw is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)

    sdist = tmp_path / "quant_investor-17.0.0.tar.gz"
    with tarfile.open(sdist, mode="w:gz") as archive:
        if not omit_sdist_pyproject:
            raw = source_pyproject_raw if sdist_pyproject_raw is None else sdist_pyproject_raw
            member = tarfile.TarInfo(f"{sdist_prefix}/pyproject.toml")
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
        for relative_path, raw in files.items():
            name = f"{sdist_prefix}/quant_investor/{relative_path}"
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
        if sdist_special_type is not None:
            member = tarfile.TarInfo(f"{sdist_prefix}/README.special")
            member.type = sdist_special_type
            if sdist_special_type in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                member.linkname = "pyproject.toml"
            archive.addfile(member)

    wheel = tmp_path / "quant_investor-17.0.0-py3-none-any.whl"
    dist_info_name = wheel_dist_info_name
    default_entry_points = (
        b"[console_scripts]\n"
        b"app = web.main:app\n"
        b"quant-investor = quant_investor.cli.main:main\n"
    )
    wheel_dist_info_files = {
        "METADATA": (
            f"Metadata-Version: 2.3\nName: {wheel_metadata_name}\n"
            f"Version: {wheel_metadata_version}\n"
        ).encode("utf-8"),
        "WHEEL": (
            "Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        ).encode("utf-8"),
        "RECORD": b"",
    }
    if not omit_wheel_entry_points:
        wheel_dist_info_files["entry_points.txt"] = (
            default_entry_points if wheel_entry_points_raw is None else wheel_entry_points_raw
        )
    wheel_dist_info_files.update(wheel_dist_info_extra_files or {})
    wheel_files = {
        **{
            f"{wheel_prefix}quant_investor/{relative_path}": raw
            for relative_path, raw in files.items()
        },
        **{
            f"{dist_info_name}/{relative_path}": raw
            for relative_path, raw in wheel_dist_info_files.items()
        },
    }
    wheel_files[f"{dist_info_name}/RECORD"] = _record_rows(
        wheel_files,
        record_path=f"{dist_info_name}/RECORD",
        omit_paths=wheel_record_omit_paths,
        tamper_path=wheel_record_tamper_path,
        extra_rows=wheel_record_extra_rows,
        raw_extra_rows=["not,enough\n"] if wheel_malformed_record else None,
    )
    with zipfile.ZipFile(wheel, mode="w") as archive:
        for relative_path, raw in wheel_files.items():
            archive.writestr(relative_path, raw)
        if wheel_extra_dist_info_root:
            archive.writestr("other-1.0.0.dist-info/METADATA", b"Name: other\nVersion: 1.0.0\n")
        if wheel_special_directory_type is not None:
            zip_member = zipfile.ZipInfo("quant_investor/special/")
            zip_member.create_system = 3
            zip_member.external_attr = (wheel_special_directory_type | 0o755) << 16
            archive.writestr(zip_member, b"")

    dist_info = site_packages / installed_dist_info_name
    dist_info.mkdir(parents=True)
    for relative_path, raw in wheel_dist_info_files.items():
        if relative_path == "RECORD" or relative_path in {
            "INSTALLER",
            "REQUESTED",
            "direct_url.json",
        }:
            continue
        path = dist_info / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.3\nName: {metadata_name}\nVersion: {metadata_version}\n",
        encoding="utf-8",
    )
    (dist_info / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: synthetic\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    if not omit_direct_url:
        payload = direct_url_payload or {
            "url": wheel.resolve().as_uri(),
            "archive_info": {"hash": "sha256=" + hashlib.sha256(wheel.read_bytes()).hexdigest()},
            "dir_info": {"editable": False},
        }
        (dist_info / "direct_url.json").write_text(
            json.dumps(payload, sort_keys=True), encoding="utf-8"
        )
    (dist_info / "INSTALLER").write_text("pip\n", encoding="utf-8")
    (dist_info / "REQUESTED").write_bytes(b"")
    for relative_path, raw in (installed_dist_info_mutations or {}).items():
        path = dist_info / relative_path
        if raw is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
    script_directory = environment / "bin"
    script_directory.mkdir(parents=True)
    script_targets = {
        "app": "web.main:app",
        "quant-investor": "quant_investor.cli.main:main",
    }
    names_to_install = (
        set(script_targets) if installed_script_names is None else installed_script_names
    )
    for script_name in sorted(names_to_install):
        script = script_directory / script_name
        script.write_bytes(_pip_25_2_posix_wrapper(environment, script_targets[script_name]))

    installed_record_files = {
        f"quant_investor/{relative_path}": (installed / relative_path).read_bytes()
        for relative_path in sorted(files)
        if (installed / relative_path).is_file()
    }
    for dist_file in sorted(dist_info.rglob("*")):
        if dist_file.name == "RECORD" or not dist_file.is_file():
            continue
        installed_record_files[
            f"{dist_info.name}/{dist_file.relative_to(dist_info).as_posix()}"
        ] = dist_file.read_bytes()
    for script in sorted(script_directory.iterdir()):
        if not script.is_file() or script.is_symlink():
            continue
        script_relative = PurePosixPath(os.path.relpath(script, site_packages)).as_posix()
        installed_record_files[script_relative] = script.read_bytes()
    installed_record_files[f"{dist_info.name}/RECORD"] = b""
    (dist_info / "RECORD").write_bytes(
        _record_rows(
            installed_record_files,
            record_path=f"{dist_info.name}/RECORD",
            omit_paths=installed_record_omit_paths,
            tamper_path=installed_record_tamper_path,
            tamper_size_path=installed_record_tamper_size_path,
            extra_rows=installed_record_extra_rows,
            raw_extra_rows=installed_record_raw_extra_rows,
        )
    )
    return {
        "source": source,
        "source_contract": source / "v17_v2_contract",
        "sdist": sdist,
        "wheel": wheel,
        "environment": environment,
        "site_packages": site_packages,
        "installed": installed,
        "installed_contract": installed / "v17_v2_contract",
        "dist_info": dist_info,
        "source_pyproject": source_pyproject,
        "app_script": script_directory / "app",
        "quant_script": script_directory / "quant-investor",
        "script_relative": Path(
            PurePosixPath(
                os.path.relpath(script_directory / "quant-investor", site_packages)
            ).as_posix()
        ),
    }


def _hatch_rows_from_physical(
    physical_superset: Mapping[str, object],
) -> list[dict[str, object]]:
    raw_rows = cast(list[Mapping[str, object]], physical_superset["rows"])
    files = {str(row["path"]): row for row in raw_rows if row["kind"] == "file"}
    package_sources = {
        path
        for path in files
        if path.startswith("quant_investor/")
        and "__pycache__" not in PurePosixPath(path).parts
        and not path.endswith(".pyc")
    }
    extra_sources = {path for path in files if not path.startswith("quant_investor/")}
    rows: list[dict[str, object]] = []
    for target, sources in (
        ("sdist", package_sources | extra_sources),
        ("wheel", package_sources),
    ):
        for source_path in sorted(sources):
            physical = files[source_path]
            rows.append(
                {
                    "target": target,
                    "source_path": source_path,
                    "distribution_path": source_path,
                    "sha256": physical["sha256"],
                    "size_bytes": physical["size_bytes"],
                    "mode": physical["mode"],
                }
            )
    return rows


def _refresh_installed_record(layout: dict[str, Path]) -> None:
    installed = layout["installed"]
    dist_info = layout["dist_info"]
    site_packages = layout["site_packages"]
    record_files = {
        path.relative_to(site_packages).as_posix(): path.read_bytes()
        for path in sorted(installed.rglob("*"))
        if path.is_file()
    }
    for dist_file in sorted(dist_info.rglob("*")):
        if dist_file.name == "RECORD" or not dist_file.is_file():
            continue
        record_files[f"{dist_info.name}/{dist_file.relative_to(dist_info).as_posix()}"] = (
            dist_file.read_bytes()
        )
    for script in sorted((layout["environment"] / "bin").iterdir()):
        if script.is_file() and not script.is_symlink():
            record_files[PurePosixPath(os.path.relpath(script, site_packages)).as_posix()] = (
                script.read_bytes()
            )
    record_files[f"{dist_info.name}/RECORD"] = b""
    (dist_info / "RECORD").write_bytes(
        _record_rows(record_files, record_path=f"{dist_info.name}/RECORD")
    )


def _append_installed_record_row(
    layout: dict[str, Path],
    *,
    relative_path: str,
    raw: bytes,
) -> None:
    record_path = layout["dist_info"] / "RECORD"
    record_path.write_bytes(
        record_path.read_bytes()
        + f"{relative_path},{_record_digest(raw)},{len(raw)}\n".encode("utf-8")
    )


def _rewrite_installed_record_row(
    layout: dict[str, Path],
    *,
    relative_path: str,
    raw: bytes,
) -> None:
    record_path = layout["dist_info"] / "RECORD"
    found = False
    output_rows: list[str] = []
    for line in record_path.read_text(encoding="utf-8").splitlines():
        row = line.split(",", 2)
        assert len(row) == 3
        if row[0] == relative_path:
            row = [relative_path, _record_digest(raw), str(len(raw))]
            found = True
        output_rows.append(",".join(row))
    assert found
    record_path.write_text("\n".join(output_rows) + "\n", encoding="utf-8")


def test_package_payload_parity_accepts_whole_quant_investor_surfaces(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    result = cast(
        dict[str, Any],
        verify_package_payload_parity(
            source_package_root=layout["source_contract"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed_contract"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        ),
    )
    package_inventory = cast(dict[str, Any], result["package_inventory"])
    installed_provenance = cast(dict[str, Any], result["installed_provenance"])
    installed_metadata = cast(dict[str, Any], installed_provenance["metadata"])
    installed_record = cast(dict[str, Any], installed_provenance["record"])
    installed_direct_url = cast(dict[str, Any], installed_provenance["direct_url"])
    wheel_provenance = cast(dict[str, Any], result["wheel_provenance"])
    wheel_metadata = cast(dict[str, Any], wheel_provenance["metadata"])
    wheel_record = cast(dict[str, Any], wheel_provenance["record"])
    assert result["package_file_count"] == 5
    assert package_inventory["file_count"] == 5
    assert result["source_equals_sdist_equals_wheel_equals_installed"] is True
    assert (layout["source_contract"] / "resources" / "main_suite_runtime_policy.v1.json").is_file()
    assert (
        layout["installed_contract"] / "schemas" / "main_suite_runtime_policy.v1.schema.json"
    ).is_file()
    assert installed_metadata == {
        "name": "quant-investor",
        "version": "17.0.0",
    }
    assert installed_record["package_file_count"] == 5
    assert installed_direct_url["present"] is True
    assert installed_provenance["non_editable_verified"] is True
    assert wheel_metadata == {
        "name": "quant-investor",
        "version": "17.0.0",
    }
    assert wheel_record["file_count"] == 9


def test_package_payload_parity_accepts_nested_installed_dist_info_files(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(
        tmp_path,
        wheel_dist_info_extra_files={
            "licenses/LICENSE.txt": b"synthetic license\n",
            "licenses/INSTALLER": b"nested immutable installer marker\n",
        },
    )
    result = cast(
        dict[str, Any],
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        ),
    )
    installed_provenance = cast(dict[str, Any], result["installed_provenance"])
    dist_info_hashes = cast(dict[str, str], installed_provenance["dist_info_file_sha256s"])
    assert "licenses/LICENSE.txt" in dist_info_hashes
    assert "licenses/INSTALLER" in dist_info_hashes


def test_package_payload_parity_cli_success_outputs_canonical_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    exit_code = package_parity_main(
        [
            "--source-package-root",
            str(layout["source_contract"]),
            "--sdist",
            str(layout["sdist"]),
            "--wheel",
            str(layout["wheel"]),
            "--installed-package-root",
            str(layout["installed_contract"]),
            "--installed-dist-info",
            str(layout["dist_info"]),
            "--installed-environment-root",
            str(layout["environment"]),
        ]
    )
    assert exit_code == 0
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["accepted"] is True
    assert stdout == (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def test_package_payload_parity_rejects_extra_installed_package_file(tmp_path: Path) -> None:
    layout = _write_synthetic_distribution(
        tmp_path,
        installed_mutations={"extra_runtime.py": b"EXTRA = True\n"},
    )
    with pytest.raises(PackageParityError, match="installed package payload differs"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_package_replacement_between_path_scan_and_record_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    target = layout["installed"] / "cli" / "main.py"
    replacement = b"def main():\n    return 99\n"
    _rewrite_installed_record_row(
        layout,
        relative_path="quant_investor/cli/main.py",
        raw=replacement,
    )
    original_scan = package_parity_module._collect_installed_package_paths

    def replace_after_scan(installed_root: Path) -> set[str]:
        paths = original_scan(installed_root)
        replacement_path = target.with_name("main.py.replacement")
        replacement_path.write_bytes(replacement)
        os.replace(replacement_path, target)
        return paths

    monkeypatch.setattr(
        package_parity_module,
        "_collect_installed_package_paths",
        replace_after_scan,
    )
    with pytest.raises(PackageParityError, match="installed package payload differs"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_package_namespace_addition_after_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    original_scan = package_parity_module._collect_installed_package_paths

    def add_after_scan(installed_root: Path) -> set[str]:
        paths = original_scan(installed_root)
        (installed_root / "late_extra.py").write_bytes(b"LATE = True\n")
        return paths

    monkeypatch.setattr(
        package_parity_module,
        "_collect_installed_package_paths",
        add_after_scan,
    )
    with pytest.raises(PackageParityError, match="installed package payload differs"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_dist_info_replacement_between_scan_and_record_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    target = layout["dist_info"] / "WHEEL"
    replacement = b"Wheel-Version: 1.0\nGenerator: replaced\nTag: py3-none-any\n"
    _rewrite_installed_record_row(
        layout,
        relative_path=f"{layout['dist_info'].name}/WHEEL",
        raw=replacement,
    )
    original_scan = package_parity_module._collect_installed_dist_info_paths

    def replace_after_scan(dist_info_path: Path) -> set[str]:
        paths = original_scan(dist_info_path)
        replacement_path = target.with_name("WHEEL.replacement")
        replacement_path.write_bytes(replacement)
        os.replace(replacement_path, target)
        return paths

    monkeypatch.setattr(
        package_parity_module,
        "_collect_installed_dist_info_paths",
        replace_after_scan,
    )
    with pytest.raises(
        PackageParityError,
        match="wheel-owned dist-info differs from wheel",
    ):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_dist_info_namespace_addition_after_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    original_scan = package_parity_module._collect_installed_dist_info_paths

    def add_after_scan(dist_info_path: Path) -> set[str]:
        paths = original_scan(dist_info_path)
        (dist_info_path / "late_extra.txt").write_bytes(b"late\n")
        return paths

    monkeypatch.setattr(
        package_parity_module,
        "_collect_installed_dist_info_paths",
        add_after_scan,
    )
    with pytest.raises(
        PackageParityError,
        match="wheel-owned dist-info differs from wheel",
    ):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_path_replacement_after_fd_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    target = layout["installed"] / "cli" / "main.py"
    original_raw = target.read_bytes()
    replacement = b"def main():\n    return 100\n"
    original_read = package_parity_module._read_fd_bytes
    replaced = False

    def replace_after_read(descriptor: int, *, max_bytes: int, label: str) -> bytes:
        nonlocal replaced
        raw = original_read(descriptor, max_bytes=max_bytes, label=label)
        if (
            not replaced
            and raw == original_raw
            and label == "installed RECORD file quant_investor/cli/main.py"
        ):
            replacement_path = target.with_name("main.py.replacement")
            replacement_path.write_bytes(replacement)
            os.replace(replacement_path, target)
            replaced = True
        return raw

    monkeypatch.setattr(package_parity_module, "_read_fd_bytes", replace_after_read)
    with pytest.raises(PackageParityError, match="changed while it was read"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )
    assert replaced is True


def test_package_payload_parity_rejects_sdist_replacement_after_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    artifact = layout["sdist"]
    artifact_raw = artifact.read_bytes()
    original_inspect = package_parity_module._collect_sdist_payload_from_bytes
    replaced = False

    def inspect_then_replace(
        sdist_raw: bytes,
        *,
        expected_name: str,
        expected_version: str,
        expected_pyproject_raw: bytes | None,
    ) -> dict[str, dict[str, object]]:
        nonlocal replaced
        inventory = original_inspect(
            sdist_raw,
            expected_name=expected_name,
            expected_version=expected_version,
            expected_pyproject_raw=expected_pyproject_raw,
        )
        replacement_path = artifact.with_name("replacement-sdist.tar.gz")
        replacement_path.write_bytes(artifact_raw)
        os.replace(replacement_path, artifact)
        replaced = True
        return inventory

    monkeypatch.setattr(
        package_parity_module,
        "_collect_sdist_payload_from_bytes",
        inspect_then_replace,
    )
    with pytest.raises(PackageParityError, match="sdist changed during verification"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=artifact,
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )
    assert replaced is True


def test_package_payload_parity_rejects_wheel_replacement_after_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    artifact = layout["wheel"]
    artifact_raw = artifact.read_bytes()
    original_inspect = package_parity_module._inspect_wheel_provenance_from_bytes
    replaced = False

    def inspect_then_replace(
        *,
        wheel_raw: bytes,
        expected_name: str,
        expected_version: str,
        expected_scripts: Mapping[str, str],
    ) -> Any:
        nonlocal replaced
        inspection = original_inspect(
            wheel_raw=wheel_raw,
            expected_name=expected_name,
            expected_version=expected_version,
            expected_scripts=expected_scripts,
        )
        replacement_path = artifact.with_name("replacement-wheel.whl")
        replacement_path.write_bytes(artifact_raw)
        os.replace(replacement_path, artifact)
        replaced = True
        return inspection

    monkeypatch.setattr(
        package_parity_module,
        "_inspect_wheel_provenance_from_bytes",
        inspect_then_replace,
    )
    with pytest.raises(PackageParityError, match="wheel changed during verification"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=artifact,
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )
    assert replaced is True


@pytest.mark.parametrize(
    "archive_kwargs",
    [
        {"wheel_prefix": "nested/"},
        {"sdist_prefix": "outer/quant_investor-17.0.0"},
    ],
)
def test_package_payload_parity_rejects_wrong_archive_root(
    tmp_path: Path,
    archive_kwargs: Any,
) -> None:
    layout = _write_synthetic_distribution(tmp_path, **archive_kwargs)
    with pytest.raises(
        PackageParityError,
        match="archive root mismatch|incorrectly nested package path|unexpected root",
    ):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize(
    "member_type",
    [
        tarfile.SYMTYPE,
        tarfile.LNKTYPE,
        tarfile.FIFOTYPE,
        tarfile.CHRTYPE,
    ],
)
def test_package_payload_parity_rejects_non_regular_sdist_members_globally(
    tmp_path: Path,
    member_type: bytes,
) -> None:
    layout = _write_synthetic_distribution(
        tmp_path,
        sdist_special_type=member_type,
    )
    with pytest.raises(PackageParityError, match="sdist contains a non-regular member"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"omit_sdist_pyproject": True}, "sdist pyproject.toml is missing"),
        (
            {"sdist_pyproject_raw": b'[project]\nname = "borrowed"\n'},
            "sdist pyproject.toml differs from source",
        ),
    ],
)
def test_package_payload_parity_requires_source_bound_sdist_pyproject(
    tmp_path: Path,
    kwargs: Any,
    message: str,
) -> None:
    layout = _write_synthetic_distribution(tmp_path, **kwargs)
    with pytest.raises(PackageParityError, match=message):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize("file_type", [stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR])
def test_package_payload_parity_rejects_special_wheel_directory_members(
    tmp_path: Path,
    file_type: int,
) -> None:
    layout = _write_synthetic_distribution(
        tmp_path,
        wheel_special_directory_type=file_type,
    )
    with pytest.raises(PackageParityError, match="wheel contains a non-regular member"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize(
    "archive_kwargs",
    [
        {"sdist_prefix": "quant_investor-17.0.0//nested"},
        {"wheel_prefix": "quant_investor//"},
        {"wheel_prefix": "quant_investor\\nested/"},
    ],
)
def test_package_payload_parity_rejects_noncanonical_archive_paths(
    tmp_path: Path,
    archive_kwargs: Any,
) -> None:
    layout = _write_synthetic_distribution(tmp_path, **archive_kwargs)
    with pytest.raises(PackageParityError, match="unsafe archive path"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_casefold_collision(tmp_path: Path) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    with zipfile.ZipFile(layout["wheel"], mode="w") as archive:
        archive.writestr("quant_investor/case.py", b"LOWER = True\n")
        archive.writestr("quant_investor/CASE.py", b"UPPER = True\n")
    with pytest.raises(PackageParityError, match="casefold-colliding path"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_wheel_file_directory_collision(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    with zipfile.ZipFile(layout["wheel"], mode="a") as archive:
        archive.writestr("quant_investor/cli", b"not a directory\n")
    with pytest.raises(PackageParityError, match="file-directory collision"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"wheel_metadata_version": "17.0.1"}, "wheel METADATA name/version mismatch"),
        (
            {"wheel_record_omit_paths": {"quant_investor/cli/main.py"}},
            "wheel RECORD paths mismatch",
        ),
        (
            {"wheel_record_extra_rows": {"quant_investor/missing.py": b"missing = True\n"}},
            "wheel RECORD paths mismatch",
        ),
        (
            {"wheel_record_tamper_path": "quant_investor/cli/main.py"},
            "wheel RECORD sha256 mismatch",
        ),
        (
            {"wheel_record_extra_rows": {"quant_investor//alias.py": b"alias\n"}},
            "unsafe RECORD path",
        ),
        (
            {"wheel_record_extra_rows": {"quant_investor/cli": b"not a directory\n"}},
            "file-directory collision",
        ),
        (
            {"wheel_dist_info_name": "quant-investor-17.0.0.dist-info"},
            "wheel dist-info directory name mismatch",
        ),
        ({"wheel_extra_dist_info_root": True}, "exactly one root dist-info"),
        ({"wheel_malformed_record": True}, "wheel RECORD contains a malformed row"),
    ],
)
def test_package_payload_parity_rejects_wheel_metadata_and_record_problems(
    tmp_path: Path,
    kwargs: Any,
    message: str,
) -> None:
    layout = _write_synthetic_distribution(tmp_path, **kwargs)
    with pytest.raises(PackageParityError, match=message):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"omit_wheel_entry_points": True},
            "missing required dist-info file",
        ),
        (
            {"wheel_entry_points_raw": b"not an ini file\n"},
            "entry_points.txt is malformed",
        ),
        (
            {
                "wheel_entry_points_raw": (
                    b"[console_scripts]\n" b"quant-investor = quant_investor.cli.main:main\n"
                )
            },
            "does not match source project scripts",
        ),
        (
            {
                "wheel_entry_points_raw": (
                    b"[console_scripts]\n"
                    b"app = borrowed.module:main\n"
                    b"quant-investor = quant_investor.cli.main:main\n"
                )
            },
            "does not match source project scripts",
        ),
        (
            {
                "wheel_entry_points_raw": (
                    b"[console_scripts]\n"
                    b"app = web.main:app\n"
                    b"extra = borrowed.module:main\n"
                    b"quant-investor = quant_investor.cli.main:main\n"
                )
            },
            "does not match source project scripts",
        ),
        (
            {
                "wheel_entry_points_raw": (
                    b"[console_scripts]\n"
                    b"app = web.main:app\n"
                    b"quant-investor = quant_investor.cli.main:main\n"
                    b"\n[other]\nplugin = borrowed.module:main\n"
                )
            },
            "unexpected entry-point groups",
        ),
    ],
)
def test_package_payload_parity_rejects_bad_wheel_entry_points(
    tmp_path: Path,
    kwargs: Any,
    message: str,
) -> None:
    layout = _write_synthetic_distribution(tmp_path, **kwargs)
    with pytest.raises(PackageParityError, match=message):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize("generated_name", ["INSTALLER", "REQUESTED", "direct_url.json"])
def test_package_payload_parity_rejects_installer_generated_files_in_wheel(
    tmp_path: Path,
    generated_name: str,
) -> None:
    layout = _write_synthetic_distribution(
        tmp_path,
        wheel_dist_info_extra_files={generated_name: b"forbidden in wheel\n"},
    )
    with pytest.raises(PackageParityError, match="installer-generated dist-info files"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"installed_record_tamper_path": "quant_investor/cli/main.py"},
            "installed RECORD sha256 mismatch",
        ),
        (
            {"installed_record_omit_paths": {"quant_investor/cli/main.py"}},
            "installed RECORD paths mismatch",
        ),
        (
            {"installed_record_extra_rows": {"other_package/__init__.py": b"missing = True\n"}},
            "installed RECORD path does not exist|installed RECORD paths mismatch",
        ),
        (
            {"installed_record_omit_paths": {"quant_investor-17.0.0.dist-info/WHEEL"}},
            "installed RECORD missing required dist-info rows",
        ),
        (
            {"installed_record_tamper_size_path": "quant_investor/cli/main.py"},
            "installed RECORD size mismatch",
        ),
        (
            {"installed_record_raw_extra_rows": ["/tmp/outside,sha256=deadbeef,1\n"]},
            "unsafe RECORD path",
        ),
        (
            {"installed_record_raw_extra_rows": ["not,enough\n"]},
            "installed RECORD contains a malformed row",
        ),
    ],
)
def test_package_payload_parity_rejects_installed_record_problems(
    tmp_path: Path,
    kwargs: Any,
    message: str,
) -> None:
    layout = _write_synthetic_distribution(tmp_path, **kwargs)
    with pytest.raises(PackageParityError, match=message):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_parity_strict_record_parser_accepts_quoted_rfc_fields() -> None:
    rows = package_parity_module._parse_record_bytes(
        (
            b'"quant_investor/data,quoted""name.py",sha256=abc,12\r\n'
            b"quant_investor-17.0.0.dist-info/RECORD,,\n"
        ),
        label="wheel",
    )
    assert rows == {
        'quant_investor/data,quoted"name.py': ("sha256=abc", "12"),
        "quant_investor-17.0.0.dist-info/RECORD": ("", ""),
    }


def test_package_parity_record_parser_rejects_oversized_record_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package_parity_module, "MAX_RECORD_BYTES", 3)
    with pytest.raises(PackageParityError, match="wheel RECORD exceeds byte limit"):
        package_parity_module._parse_record_bytes(b"a,b,c\n", label="wheel")


def test_package_parity_record_parser_rejects_oversized_row_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package_parity_module, "MAX_RECORD_ROWS", 1)
    with pytest.raises(PackageParityError, match="wheel RECORD exceeds row count limit"):
        package_parity_module._parse_record_bytes(
            b"quant_investor/a.py,sha256=abc,1\nquant_investor/b.py,sha256=def,1\n",
            label="wheel",
        )


def test_package_parity_record_parser_rejects_oversized_field_and_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(package_parity_module, "MAX_RECORD_FIELD_CHARS", 3)
    with pytest.raises(PackageParityError, match="wheel RECORD contains an oversized field"):
        package_parity_module._parse_record_bytes(b"abcd,sha256=abc,1\n", label="wheel")

    monkeypatch.setattr(package_parity_module, "MAX_RECORD_FIELD_CHARS", 64)
    monkeypatch.setattr(package_parity_module, "MAX_RECORD_PATH_CHARS", 3)
    with pytest.raises(PackageParityError, match="oversized RECORD path"):
        package_parity_module._parse_record_bytes(b"abcd,sha256=abc,1\n", label="wheel")


@pytest.mark.parametrize(
    "raw",
    [
        b"\n",
        b"quant_investor/module.py,sha256=abc\n",
        b"quant_investor/module.py,sha256=abc,1,extra\n",
        b'"quant_investor/module.py,sha256=abc,1\n',
        b'"quant_investor/module.py\n",sha256=abc,1\n',
        b"quant_investor/module.py,sha256=abc,1\rquant_investor/other.py,sha256=def,2\n",
        b'quant_investor/module.py,"sha256=abc"x,1\n',
        b'quant_investor/module.py,sha"256=abc,1\n',
    ],
)
def test_package_parity_strict_record_parser_rejects_malformed_rows(raw: bytes) -> None:
    with pytest.raises(PackageParityError, match="wheel RECORD .*malformed"):
        package_parity_module._parse_record_bytes(raw, label="wheel")


@pytest.mark.parametrize(
    "raw",
    [
        b",sha256=abc,1\n",
        b"/tmp/outside,sha256=abc,1\n",
        b"quant_investor//module.py,sha256=abc,1\n",
        b"quant_investor/../module.py,sha256=abc,1\n",
        b"quant_investor\\module.py,sha256=abc,1\n",
        b"quant_investor/module.py/,sha256=abc,1\n",
    ],
)
def test_package_parity_strict_record_parser_rejects_unsafe_paths(raw: bytes) -> None:
    with pytest.raises(PackageParityError, match="unsafe RECORD path"):
        package_parity_module._parse_record_bytes(raw, label="wheel")


def _zip_payload(files: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w") as archive:
        for relative_path, raw in files.items():
            archive.writestr(relative_path, raw)
    return output.getvalue()


def _zip_eocd(
    *,
    disk_number: int = 0,
    central_directory_disk: int = 0,
    entries_on_disk: int = 0,
    total_entries: int = 0,
    central_directory_size: int = 0,
    central_directory_offset: int = 0,
    comment: bytes = b"",
) -> bytes:
    return (
        b"PK\x05\x06"
        + disk_number.to_bytes(2, "little")
        + central_directory_disk.to_bytes(2, "little")
        + entries_on_disk.to_bytes(2, "little")
        + total_entries.to_bytes(2, "little")
        + central_directory_size.to_bytes(4, "little")
        + central_directory_offset.to_bytes(4, "little")
        + len(comment).to_bytes(2, "little")
        + comment
    )


def _zip_central_header(filename: bytes, *, extra: bytes = b"", comment: bytes = b"") -> bytes:
    return (
        b"PK\x01\x02"
        + b"\x14\x03"
        + b"\x14\x00"
        + b"\x00\x00"
        + b"\x00\x00"
        + b"\x00\x00"
        + b"\x00\x00"
        + b"\x00\x00\x00\x00"
        + b"\x00\x00\x00\x00"
        + b"\x00\x00\x00\x00"
        + len(filename).to_bytes(2, "little")
        + len(extra).to_bytes(2, "little")
        + len(comment).to_bytes(2, "little")
        + b"\x00\x00"
        + b"\x00\x00"
        + b"\x00\x00\x00\x00"
        + b"\x00\x00\x00\x00"
        + filename
        + extra
        + comment
    )


def _fail_if_zipfile_is_constructed(*_args: object, **_kwargs: object) -> NoReturn:
    raise AssertionError("ZipFile should not be constructed before raw preflight rejection")


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            _zip_eocd(disk_number=1),
            "wheel multi-disk ZIP archives are not supported",
        ),
        (
            _zip_eocd(entries_on_disk=0xFFFF, total_entries=0xFFFF),
            "wheel ZIP64 records are not supported",
        ),
        (
            (
                lambda central: central
                + _zip_eocd(
                    entries_on_disk=2,
                    total_entries=2,
                    central_directory_size=len(central),
                    central_directory_offset=0,
                )
            )(_zip_central_header(b"quant_investor/__init__.py")),
            "wheel central directory count mismatch",
        ),
    ],
)
def test_package_parity_wheel_preflight_rejects_raw_zip_before_zipfile(
    monkeypatch: pytest.MonkeyPatch,
    raw: bytes,
    message: str,
) -> None:
    monkeypatch.setattr(package_parity_module.zipfile, "ZipFile", _fail_if_zipfile_is_constructed)
    with pytest.raises(PackageParityError, match=message):
        package_parity_module._wheel_member_inventory(raw)


def _sdist_payload(files: Mapping[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        for relative_path, raw in files.items():
            member = tarfile.TarInfo(relative_path)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))
    return output.getvalue()


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "files", "message"),
    [
        (
            "MAX_MEMBER_BYTES",
            1,
            {
                "quant_investor/__init__.py": b"12",
                "quant_investor-17.0.0.dist-info/METADATA": b"m",
            },
            "wheel member quant_investor/__init__.py exceeds byte limit",
        ),
        (
            "MAX_MEMBER_COUNT",
            1,
            {
                "quant_investor/__init__.py": b"1",
                "quant_investor-17.0.0.dist-info/METADATA": b"m",
            },
            "wheel exceeds member count limit",
        ),
        (
            "MAX_TOTAL_PAYLOAD_BYTES",
            1,
            {
                "quant_investor/__init__.py": b"1",
                "quant_investor-17.0.0.dist-info/METADATA": b"m",
            },
            "wheel total payload exceeds byte limit",
        ),
    ],
)
def test_package_parity_wheel_rejects_artifact_limits_before_member_read(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    files: Mapping[str, bytes],
    message: str,
) -> None:
    monkeypatch.setattr(package_parity_module, limit_name, limit_value)
    with pytest.raises(PackageParityError, match=re.escape(message)):
        package_parity_module._wheel_member_inventory(_zip_payload(files))


@pytest.mark.parametrize(
    ("limit_name", "limit_value", "message"),
    [
        ("MAX_MEMBER_BYTES", 1, "sdist member quant_investor-17.0.0/pyproject.toml"),
        ("MAX_MEMBER_COUNT", 1, "sdist exceeds member count limit"),
        ("MAX_TOTAL_PAYLOAD_BYTES", 1, "sdist total payload exceeds byte limit"),
    ],
)
def test_package_parity_sdist_rejects_artifact_limits_before_member_read(
    monkeypatch: pytest.MonkeyPatch,
    limit_name: str,
    limit_value: int,
    message: str,
) -> None:
    monkeypatch.setattr(package_parity_module, limit_name, limit_value)
    raw = _sdist_payload(
        {
            "quant_investor-17.0.0/pyproject.toml": b"12",
            "quant_investor-17.0.0/quant_investor/__init__.py": b"1",
        }
    )
    with pytest.raises(PackageParityError, match=message):
        package_parity_module._collect_sdist_payload_from_bytes(
            raw,
            expected_name="quant-investor",
            expected_version="17.0.0",
            expected_pyproject_raw=None,
        )


def test_package_parity_source_stat_rejects_oversized_file_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = tmp_path / "quant_investor"
    package_root.mkdir()
    (package_root / "__init__.py").write_bytes(b"12")
    monkeypatch.setattr(package_parity_module, "MAX_MEMBER_BYTES", 1)

    with pytest.raises(PackageParityError, match="source file quant_investor/__init__.py"):
        package_parity_module.collect_directory_payload(package_root, label="source")


def test_package_parity_installed_stat_rejects_oversized_file_before_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = tmp_path / "env"
    site_packages = environment / "lib" / "python" / "site-packages"
    package_file = site_packages / "quant_investor" / "__init__.py"
    package_file.parent.mkdir(parents=True)
    package_file.write_bytes(b"12")
    snapshot = package_parity_module._InstalledRecordSnapshot(
        dist_info_parent=site_packages,
        installed_environment_root=environment,
    )
    monkeypatch.setattr(package_parity_module, "MAX_MEMBER_BYTES", 1)
    try:
        with pytest.raises(PackageParityError, match="installed RECORD file"):
            snapshot.read("quant_investor/__init__.py")
    finally:
        snapshot.close()


def test_package_parity_fd_reader_stops_before_second_read_after_growth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStat:
        def __init__(self, size: int) -> None:
            self.st_size = size

    read_sizes: list[int] = []

    def fake_fstat(_descriptor: int) -> FakeStat:
        return FakeStat(1 if not read_sizes else 2)

    def fake_read(_descriptor: int, size: int) -> bytes:
        read_sizes.append(size)
        return b"1"

    monkeypatch.setattr(package_parity_module.os, "fstat", fake_fstat)
    monkeypatch.setattr(package_parity_module.os, "read", fake_read)

    with pytest.raises(PackageParityError, match="fd-test exceeds byte limit"):
        package_parity_module._read_fd_bytes(99, max_bytes=1, label="fd-test")
    assert read_sizes == [1]


def test_package_parity_source_fd_reader_rejects_growth_before_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeStat:
        st_size = 2

    def fail_read(_descriptor: int, _size: int) -> bytes:
        raise AssertionError("read should not run after fstat exceeds cap")

    monkeypatch.setattr(package_parity_module.os, "fstat", lambda _descriptor: FakeStat())
    monkeypatch.setattr(package_parity_module.os, "read", fail_read)

    with pytest.raises(PackageParityError, match="source-fd exceeds byte limit"):
        package_parity_module._read_source_fd_bytes(99, max_bytes=1, label="source-fd")


@pytest.mark.parametrize(
    "raw_path",
    [
        "",
        "./alias.py",
        "quant_investor//alias.py",
        "quant_investor/../alias.py",
        "quant_investor\\alias.py",
        "quant_investor/\x00alias.py",
    ],
)
def test_package_payload_parity_rejects_noncanonical_installed_record_paths(
    tmp_path: Path,
    raw_path: str,
) -> None:
    layout = _write_synthetic_distribution(
        tmp_path,
        installed_record_raw_extra_rows=[
            f"{raw_path},sha256=deadbeef,1\n",
        ],
    )
    with pytest.raises(PackageParityError, match="unsafe RECORD path"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize(
    "installed_script_names",
    [{"quant-investor"}, set()],
)
def test_package_payload_parity_rejects_deleted_script_and_record_row(
    tmp_path: Path,
    installed_script_names: set[str],
) -> None:
    layout = _write_synthetic_distribution(
        tmp_path,
        installed_script_names=installed_script_names,
    )
    with pytest.raises(PackageParityError, match="installed generated script is missing"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize("script_key", ["quant_script", "app_script"])
def test_package_payload_parity_rejects_tampered_script_with_refreshed_record(
    tmp_path: Path,
    script_key: str,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    script = layout[script_key]
    raw = script.read_bytes()
    if script_key == "quant_script":
        tampered = raw.replace(
            b"from quant_investor.cli.main import main\n",
            b"from borrowed.module import main\n",
        )
        assert tampered != raw
    else:
        tampered = raw + b"print('unexpected statement')\n"
    script.write_bytes(tampered)
    _refresh_installed_record(layout)

    with pytest.raises(
        PackageParityError,
        match="installed generated script wrapper differs from pip 25.2 contract",
    ):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize("interpreter_name", ["python3", "python3.13", "python999"])
def test_package_payload_parity_rejects_script_shebang_drift_with_refreshed_record(
    tmp_path: Path,
    interpreter_name: str,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    script = layout["quant_script"]
    raw = script.read_bytes()
    expected_shebang = f"#!{layout['environment'] / 'bin' / 'python'}\n".encode("utf-8")
    tampered_shebang = f"#!{layout['environment'] / 'bin' / interpreter_name}\n".encode("utf-8")
    assert raw.startswith(expected_shebang)
    script.write_bytes(raw.replace(expected_shebang, tampered_shebang, 1))
    _refresh_installed_record(layout)

    with pytest.raises(
        PackageParityError,
        match="installed generated script wrapper differs from pip 25.2 contract",
    ):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_symlinked_installed_record_path(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    alias = layout["environment"] / "bin" / "alias"
    alias.symlink_to(layout["quant_script"])
    raw = layout["quant_script"].read_bytes()
    _append_installed_record_row(
        layout,
        relative_path="../../../bin/alias",
        raw=raw,
    )
    with pytest.raises(PackageParityError, match="regular non-symlink"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_record_paths_with_same_resolved_destination(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    raw = layout["app_script"].read_bytes()
    _append_installed_record_row(
        layout,
        relative_path="../../../../env/bin/app",
        raw=raw,
    )
    with pytest.raises(
        PackageParityError,
        match="same destination|identify the same file",
    ):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize(
    "installed_dist_info_mutations",
    [
        {"entry_points.txt": b"[console_scripts]\napp = borrowed.module:main\n"},
        {"entry_points.txt": None},
        {"unexpected.txt": b"installed only\n"},
    ],
)
def test_package_payload_parity_rejects_installed_immutable_dist_info_drift(
    tmp_path: Path,
    installed_dist_info_mutations: dict[str, bytes | None],
) -> None:
    layout = _write_synthetic_distribution(
        tmp_path,
        installed_dist_info_mutations=installed_dist_info_mutations,
    )
    with pytest.raises(PackageParityError, match="wheel-owned dist-info differs from wheel"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_nested_mutable_name_drift(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(
        tmp_path,
        wheel_dist_info_extra_files={
            "licenses/INSTALLER": b"wheel-owned nested bytes\n",
        },
        installed_dist_info_mutations={
            "licenses/INSTALLER": b"tampered nested bytes\n",
        },
    )
    with pytest.raises(PackageParityError, match="wheel-owned dist-info differs from wheel"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_allows_only_root_installer_generated_differences(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    (layout["dist_info"] / "INSTALLER").write_bytes(b"pip-from-ensurepip\n")
    (layout["dist_info"] / "REQUESTED").write_bytes(b"requested\n")
    _refresh_installed_record(layout)
    result = verify_package_payload_parity(
        source_package_root=layout["source"],
        sdist_path=layout["sdist"],
        wheel_path=layout["wheel"],
        installed_package_root=layout["installed"],
        installed_dist_info_path=layout["dist_info"],
        installed_environment_root=layout["environment"],
    )
    assert result["source_equals_sdist_equals_wheel_equals_installed"] is True


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "url": "",
                "archive_info": {"hash": "sha256=deadbeef"},
                "dir_info": {"editable": False},
            },
            "must point to a local wheel file",
        ),
        (
            {
                "url": "",
                "archive_info": {"hash": "sha256=deadbeef"},
                "dir_info": {"editable": True},
            },
            "editable",
        ),
    ],
)
def test_package_payload_parity_rejects_bad_direct_url_payloads(
    tmp_path: Path,
    payload: dict[str, object],
    message: str,
) -> None:
    layout = _write_synthetic_distribution(tmp_path, direct_url_payload=payload)
    with pytest.raises(PackageParityError, match=message):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_absent_direct_url(tmp_path: Path) -> None:
    layout = _write_synthetic_distribution(tmp_path, omit_direct_url=True)
    with pytest.raises(PackageParityError, match="direct_url.json is required"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_direct_url_wrong_wheel_and_hash(tmp_path: Path) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    other_wheel = tmp_path / "other.whl"
    other_wheel.write_bytes(b"not a wheel")
    (layout["dist_info"] / "direct_url.json").write_text(
        json.dumps(
            {
                "url": other_wheel.resolve().as_uri(),
                "archive_info": {
                    "hash": "sha256=" + hashlib.sha256(other_wheel.read_bytes()).hexdigest()
                },
                "dir_info": {"editable": False},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _refresh_installed_record(layout)
    with pytest.raises(PackageParityError, match="does not point at the verified wheel"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )

    (layout["dist_info"] / "direct_url.json").write_text(
        json.dumps(
            {
                "url": layout["wheel"].resolve().as_uri(),
                "archive_info": {"hash": "sha256=deadbeef"},
                "dir_info": {"editable": False},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _refresh_installed_record(layout)
    with pytest.raises(PackageParityError, match="wheel sha256 mismatch"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_conflicting_direct_url_hashes(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    wheel_sha256 = hashlib.sha256(layout["wheel"].read_bytes()).hexdigest()
    (layout["dist_info"] / "direct_url.json").write_text(
        json.dumps(
            {
                "url": layout["wheel"].resolve().as_uri(),
                "archive_info": {
                    "hash": "sha256=" + wheel_sha256,
                    "hashes": {"sha256": "deadbeef"},
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    _refresh_installed_record(layout)
    with pytest.raises(PackageParityError, match="wheel sha256 mismatch"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_editable_direct_url(tmp_path: Path) -> None:
    layout = _write_synthetic_distribution(
        tmp_path,
        direct_url_payload={
            "url": (tmp_path / "source").as_uri(),
            "dir_info": {"editable": True},
        },
    )
    with pytest.raises(PackageParityError, match="editable"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_invalid_direct_url_json(tmp_path: Path) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    (layout["dist_info"] / "direct_url.json").write_text("{", encoding="utf-8")
    _refresh_installed_record(layout)
    with pytest.raises(PackageParityError, match="invalid JSON"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_invalid_pyproject_toml(tmp_path: Path) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    (layout["source"].parent / "pyproject.toml").write_text("[project\n", encoding="utf-8")
    with pytest.raises(PackageParityError, match="invalid TOML"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


@pytest.mark.parametrize("pyproject_mode", ["missing", "symlink"])
def test_package_payload_parity_requires_regular_source_pyproject_even_with_expected_identity(
    tmp_path: Path,
    pyproject_mode: str,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    pyproject = layout["source_pyproject"]
    if pyproject_mode == "missing":
        pyproject.unlink()
    else:
        borrowed = tmp_path / "borrowed-pyproject.toml"
        borrowed.write_bytes(pyproject.read_bytes())
        pyproject.unlink()
        pyproject.symlink_to(borrowed)
    with pytest.raises(
        PackageParityError,
        match="source pyproject.toml is missing|regular non-symlink",
    ):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
            expected_name="quant-investor",
            expected_version="17.0.0",
        )


def test_package_payload_parity_requires_exact_source_project_scripts(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    layout["source_pyproject"].write_text(
        '[project]\nname = "quant-investor"\nversion = "17.0.0"\n'
        "\n[project.scripts]\n"
        'quant-investor = "quant_investor.cli.main:main"\n',
        encoding="utf-8",
    )
    with pytest.raises(PackageParityError, match="project scripts mismatch"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_root_normalization_uses_nearest_qualifying_ancestor(
    tmp_path: Path,
) -> None:
    outer_package = tmp_path / "quant_investor"
    outer_package.mkdir()
    (outer_package / "__init__.py").write_bytes(b"outer package marker\n")
    layout = _write_synthetic_distribution(outer_package)
    result = verify_package_payload_parity(
        source_package_root=layout["source_contract"],
        sdist_path=layout["sdist"],
        wheel_path=layout["wheel"],
        installed_package_root=layout["installed_contract"],
        installed_dist_info_path=layout["dist_info"],
        installed_environment_root=layout["environment"],
    )
    assert result["source_equals_sdist_equals_wheel_equals_installed"] is True


def test_package_payload_parity_cli_failure_outputs_canonical_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_synthetic_distribution(tmp_path, omit_direct_url=True)
    exit_code = package_parity_main(
        [
            "--source-package-root",
            str(layout["source_contract"]),
            "--sdist",
            str(layout["sdist"]),
            "--wheel",
            str(layout["wheel"]),
            "--installed-package-root",
            str(layout["installed_contract"]),
            "--installed-dist-info",
            str(layout["dist_info"]),
            "--installed-environment-root",
            str(layout["environment"]),
        ]
    )
    assert exit_code == 2
    stdout = capsys.readouterr().out
    payload = json.loads(stdout)
    assert payload["accepted"] is False
    assert stdout == (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def test_package_payload_parity_cli_parse_errors_are_canonical_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    valid_args = [
        "--source-package-root",
        str(layout["source_contract"]),
        "--sdist",
        str(layout["sdist"]),
        "--wheel",
        str(layout["wheel"]),
        "--installed-package-root",
        str(layout["installed_contract"]),
        "--installed-dist-info",
        str(layout["dist_info"]),
        "--installed-environment-root",
        str(layout["environment"]),
    ]
    for argv in ([], [*valid_args, "--unknown"], [*valid_args, "unexpected"]):
        exit_code = package_parity_main(argv)
        captured = capsys.readouterr()
        payload = json.loads(captured.out)
        assert exit_code == 2
        assert payload["accepted"] is False
        assert captured.err == ""
        assert captured.out == (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )


def test_package_payload_parity_rejects_metadata_mismatch(tmp_path: Path) -> None:
    layout = _write_synthetic_distribution(tmp_path, metadata_version="17.0.1")
    with pytest.raises(PackageParityError, match="METADATA name/version mismatch"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_borrowed_dist_info(tmp_path: Path) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    borrowed = tmp_path / "other-site-packages" / layout["dist_info"].name
    shutil.copytree(layout["dist_info"], borrowed)
    with pytest.raises(PackageParityError, match="must share the installed package"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=borrowed,
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_installed_dist_info_wrong_root_name(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(
        tmp_path,
        installed_dist_info_name="quant-investor-17.0.0.dist-info",
    )
    with pytest.raises(PackageParityError, match="installed dist-info directory name mismatch"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )


def test_package_payload_parity_rejects_expected_identity_override(tmp_path: Path) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    with pytest.raises(PackageParityError, match="does not match source pyproject"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
            expected_name="borrowed-quant-investor",
            expected_version="17.0.0",
        )
    with pytest.raises(PackageParityError, match="does not match source pyproject"):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
            expected_name="quant-investor",
            expected_version="17.0.1",
        )


def test_physical_source_superset_and_hatch_projection_bind_package_parity(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    ignored = layout["source"] / "__pycache__" / "ignored.pyc"
    ignored.parent.mkdir()
    ignored.write_bytes(b"ignored bytecode\n")

    physical = collect_physical_source_superset(
        layout["source"].parent,
        extra_paths=("pyproject.toml",),
    )
    rows = cast(list[dict[str, object]], physical["rows"])
    assert physical["row_count"] == len(rows)
    assert (
        physical["sha256"]
        == hashlib.sha256(
            json.dumps(
                rows,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert [row["path"] for row in rows] == sorted(str(row["path"]) for row in rows)
    assert {
        row["path"]: {
            "kind": row["kind"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
        }
        for row in rows
        if row["path"]
        in {"quant_investor", ignored.relative_to(layout["source"].parent).as_posix()}
    } == {
        "quant_investor": {
            "kind": "directory",
            "sha256": None,
            "size_bytes": 0,
        },
        "quant_investor/__pycache__/ignored.pyc": {
            "kind": "file",
            "sha256": hashlib.sha256(b"ignored bytecode\n").hexdigest(),
            "size_bytes": len(b"ignored bytecode\n"),
        },
    }

    hatch = validate_hatch_namespace_rows(
        _hatch_rows_from_physical(physical),
        physical_superset=physical,
    )
    parity = verify_package_payload_parity(
        source_package_root=layout["source"],
        sdist_path=layout["sdist"],
        wheel_path=layout["wheel"],
        installed_package_root=layout["installed"],
        installed_dist_info_path=layout["dist_info"],
        installed_environment_root=layout["environment"],
    )
    assert hatch["wheel_projection_sha256"] == parity["package_inventory_sha256"]
    assert hatch["row_count"] == len(cast(list[object], hatch["rows"]))


def test_hatch_namespace_rows_fail_closed_on_order_metadata_and_coverage(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    physical = collect_physical_source_superset(
        layout["source"].parent,
        extra_paths=("pyproject.toml",),
    )
    rows = _hatch_rows_from_physical(physical)

    with pytest.raises(PackageParityError, match="not canonically sorted"):
        validate_hatch_namespace_rows(
            list(reversed(rows)),
            physical_superset=physical,
        )

    changed = [dict(row) for row in rows]
    changed_mode = changed[0]["mode"]
    assert type(changed_mode) is int
    changed[0]["mode"] = changed_mode ^ 0o100
    with pytest.raises(PackageParityError, match="differs from physical source"):
        validate_hatch_namespace_rows(changed, physical_superset=physical)

    missing = [dict(row) for row in rows[1:]]
    with pytest.raises(PackageParityError, match="source coverage mismatch"):
        validate_hatch_namespace_rows(missing, physical_superset=physical)


def test_hatch_namespace_rejects_casefold_collision_in_physical_binding(
    tmp_path: Path,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    physical = collect_physical_source_superset(layout["source"].parent)
    rows = [dict(row) for row in cast(list[dict[str, object]], physical["rows"])]
    original = next(row for row in rows if row["path"] == "quant_investor/__init__.py")
    duplicate = dict(original)
    duplicate["path"] = "quant_investor/__INIT__.py"
    rows.append(duplicate)
    rows.sort(key=lambda row: str(row["path"]))
    tampered = {
        "rows": rows,
        "row_count": len(rows),
        "sha256": hashlib.sha256(
            json.dumps(
                rows,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    with pytest.raises(PackageParityError, match="casefold-colliding path"):
        validate_hatch_namespace_rows([], physical_superset=tampered)


@pytest.mark.parametrize("node_kind", ["symlink", "fifo"])
def test_physical_source_superset_rejects_symlink_and_nonregular_nodes(
    tmp_path: Path,
    node_kind: str,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    node = layout["source"] / "unsafe"
    if node_kind == "symlink":
        node.symlink_to(layout["source"] / "__init__.py")
        message = "contains a symlink"
    else:
        os.mkfifo(node)
        message = "contains a non-regular node"
    with pytest.raises(PackageParityError, match=message):
        collect_physical_source_superset(layout["source"].parent)


def test_physical_source_superset_rejects_hardlinks(tmp_path: Path) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    os.link(layout["source"] / "__init__.py", layout["source"] / "hardlink.py")
    with pytest.raises(PackageParityError, match="hardlinked|hardlink-colliding"):
        collect_physical_source_superset(layout["source"].parent)


def test_physical_source_superset_fails_closed_on_emfile_without_fd_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    original_open = os.open
    descriptor_count = len(os.listdir("/dev/fd"))

    def fail_package_file_open(
        path: str | bytes | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if path == "__init__.py":
            raise OSError(errno.EMFILE, "synthetic descriptor exhaustion")
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", fail_package_file_open)
    with pytest.raises(PackageParityError, match="cannot be opened"):
        collect_physical_source_superset(layout["source"].parent)
    assert len(os.listdir("/dev/fd")) == descriptor_count


def test_physical_source_superset_rejects_namespace_change_during_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    sentinel = layout["source"] / "sentinel.py"
    sentinel.write_bytes(b"SENTINEL = True\n")
    original_read = package_parity_module._read_source_fd_bytes
    mutated = False

    def read_then_add(descriptor: int, *, max_bytes: int, label: str) -> bytes:
        nonlocal mutated
        raw = original_read(descriptor, max_bytes=max_bytes, label=label)
        if raw == b"SENTINEL = True\n" and not mutated:
            (layout["source"] / "late.py").write_bytes(b"LATE = True\n")
            mutated = True
        return raw

    monkeypatch.setattr(package_parity_module, "_read_source_fd_bytes", read_then_add)
    with pytest.raises(
        PackageParityError,
        match="physical source .* changed during verification|namespace changed",
    ):
        collect_physical_source_superset(layout["source"].parent)
    assert mutated is True


@pytest.mark.parametrize(
    "mutation",
    [
        "add",
        "file_switch_restore",
        "same_inode_restore",
        "pyproject_same_inode_restore",
        "package_root_switch_restore",
        "repo_root_switch_restore",
    ],
)
def test_package_payload_parity_rejects_source_namespace_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    layout = _write_synthetic_distribution(tmp_path)
    original_inspect = package_parity_module._inspect_wheel_provenance_from_bytes

    def inspect_then_mutate(
        *,
        wheel_raw: bytes,
        expected_name: str,
        expected_version: str,
        expected_scripts: Mapping[str, str],
    ) -> Any:
        inspection = original_inspect(
            wheel_raw=wheel_raw,
            expected_name=expected_name,
            expected_version=expected_version,
            expected_scripts=expected_scripts,
        )
        target = layout["source"] / "cli" / "main.py"
        if mutation == "add":
            (layout["source"] / "late_addition.py").write_bytes(b"LATE = True\n")
        elif mutation == "file_switch_restore":
            held = target.with_name("main.py.held")
            os.replace(target, held)
            os.replace(held, target)
        elif mutation in {"same_inode_restore", "pyproject_same_inode_restore"}:
            if mutation == "pyproject_same_inode_restore":
                target = layout["source_pyproject"]
            original = target.read_bytes()
            target.write_bytes(b"X" * len(original))
            target.write_bytes(original)
        else:
            switched = (
                layout["source"]
                if mutation == "package_root_switch_restore"
                else layout["source"].parent
            )
            held = switched.with_name(switched.name + ".held")
            os.replace(switched, held)
            switched.mkdir()
            switched.rmdir()
            os.replace(held, switched)
        return inspection

    monkeypatch.setattr(
        package_parity_module,
        "_inspect_wheel_provenance_from_bytes",
        inspect_then_mutate,
    )
    with pytest.raises(
        PackageParityError,
        match="source .* changed during verification|source .* namespace changed",
    ):
        verify_package_payload_parity(
            source_package_root=layout["source"],
            sdist_path=layout["sdist"],
            wheel_path=layout["wheel"],
            installed_package_root=layout["installed"],
            installed_dist_info_path=layout["dist_info"],
            installed_environment_root=layout["environment"],
        )
