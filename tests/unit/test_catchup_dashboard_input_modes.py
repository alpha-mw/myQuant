"""Exact byte/security policy tests; CSV meaning stays with native consumers."""

from copy import deepcopy
import json
import os
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from _public_catchup_fixture import put, routed_collection
from quant_investor.operations.catchup_binding import BindingSources, check_template_sources
from quant_investor.operations.daily_contract import ContractError
from quant_investor.operations.daily_preparation import Sources
from quant_investor.operations.daily_preparation_contract import PreparationError
from quant_investor.system.errors import SystemSecurityError, SystemStorageError

PATHS = {
    "benchmark_ref": "portfolio_dashboard/inputs/cn_index_benchmark.csv",
    "risk_free_ref": "portfolio_dashboard/inputs/cn_govt_bond_yield.csv",
}
MIB = 1024 * 1024


def input_ref(root, role="benchmark_ref", *, mode=0o644, size=20):
    ref = put(root, PATHS[role], b"x" * size)
    (root / ref["path"]).chmod(mode)
    return ref


def identity(path):
    s = path.stat()
    return (
        s.st_dev,
        s.st_ino,
        s.st_mode,
        s.st_uid,
        s.st_gid,
        s.st_nlink,
        s.st_size,
        s.st_mtime_ns,
        s.st_ctime_ns,
    )


def inventory(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), identity(p))
        for p in root.rglob("*")
        if p.is_file()
    }


def template(root):
    _, request = routed_collection(root)
    value = json.loads((root / request["recipe_ref"]["path"]).read_bytes())["recipes"]["20260827"]
    value["dashboard_sources"] = {role: input_ref(root, role) for role in PATHS}
    return value


@pytest.mark.parametrize("mode", [0o400, 0o600, 0o644])
def test_exact_roles_read_recheck_without_writes(tmp_path, monkeypatch, mode):
    value = template(tmp_path)
    for ref in value["dashboard_sources"].values():
        (tmp_path / ref["path"]).chmod(mode)
    before = inventory(tmp_path)
    source = BindingSources(str(tmp_path))
    with monkeypatch.context() as patch:

        def forbidden(*args, **kwargs):
            pytest.fail("input reader attempted a file mutation")

        for name in ("chmod", "fchmod", "rename", "replace", "unlink", "mkdir"):
            patch.setattr(os, name, forbidden)
        check_template_sources(source, value, profile="FRESH")
        source.recheck()
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("role", list(PATHS))
@pytest.mark.parametrize("size", [8 * MIB, 8 * MIB + 1, 32 * MIB, 32 * MIB + 1])
def test_exact_budgets_keep_large_private_catchup_inputs(tmp_path, role, size):
    ref = input_ref(tmp_path, role, mode=0o600, size=size)
    source = BindingSources(str(tmp_path))
    if size <= 32 * MIB:
        assert len(source.dashboard_input(role, ref)) == size
        source.recheck()
    else:
        with pytest.raises(SystemSecurityError):
            source.dashboard_input(role, ref)
    prep = Sources(str(tmp_path))
    if size <= 8 * MIB:
        assert len(prep.raw(ref)) == size
        prep.recheck()
    else:
        with pytest.raises(SystemSecurityError):
            prep.raw(ref)


@pytest.mark.parametrize("fault", ["wrong", "swapped", "duplicate"])
@pytest.mark.parametrize("mode", [0o600, 0o644])
def test_nonmatching_paths_only_gain_strict_admission(tmp_path, fault, mode):
    value = template(tmp_path)
    refs = value["dashboard_sources"]
    if fault == "wrong":
        refs["benchmark_ref"] = put(tmp_path, "arbitrary.csv", b"data")
    elif fault == "swapped":
        refs["benchmark_ref"], refs["risk_free_ref"] = refs["risk_free_ref"], refs["benchmark_ref"]
    else:
        refs["risk_free_ref"] = refs["benchmark_ref"]
    for ref in refs.values():
        (tmp_path / ref["path"]).chmod(mode)
    source = BindingSources(str(tmp_path))
    if mode == 0o600:
        check_template_sources(source, value)
        source.recheck()
    else:
        with pytest.raises(SystemSecurityError):
            check_template_sources(source, value)
    bad_role = "risk_free_ref" if fault == "duplicate" else "benchmark_ref"
    with pytest.raises(ContractError, match="CATCHUP_DASHBOARD_INPUT_ROLE_PATH_INVALID"):
        source.dashboard_input(bad_role, refs[bad_role])


def test_dashboard_read_never_satisfies_later_strict_governance_read(tmp_path):
    ref = input_ref(tmp_path)
    source = BindingSources(str(tmp_path))
    source.dashboard_input("benchmark_ref", ref)
    with pytest.raises(SystemSecurityError):
        source.raw(ref)
    value = template(tmp_path)
    value["research_sources"]["injected"] = {
        "schema_version": "cn-daily-execute-recipe.v6",
        "dashboard_sources": {"benchmark_ref": value["dashboard_sources"]["benchmark_ref"]},
    }
    with pytest.raises(SystemSecurityError):
        check_template_sources(BindingSources(str(tmp_path)), value)


@pytest.mark.parametrize(
    "fault",
    [
        "symlink",
        "hardlink",
        "directory",
        "writable",
        "executable",
        "other_mode",
        "empty",
        "parent_symlink",
        "casefold",
    ],
)
@pytest.mark.parametrize("reader", ["preparation", "catchup"])
def test_unsafe_sources_have_security_exception(tmp_path, fault, reader):
    ref = input_ref(tmp_path)
    path = tmp_path / ref["path"]
    if fault == "symlink":
        original = path.with_name("original.csv")
        path.rename(original)
        path.symlink_to(original)
    elif fault == "hardlink":
        os.link(path, path.with_name("alias.csv"))
    elif fault in {"directory", "empty"}:
        path.unlink()
        path.mkdir() if fault == "directory" else path.touch(mode=0o600)
    elif fault == "parent_symlink":
        original = path.parent.with_name("original")
        path.parent.rename(original)
        path.parent.symlink_to(original, target_is_directory=True)
    elif fault == "casefold":
        path.rename(path.with_name(path.name.upper()))
    else:
        path.chmod({"writable": 0o666, "executable": 0o744, "other_mode": 0o640}[fault])
    with pytest.raises(SystemSecurityError) as caught:
        if reader == "preparation":
            Sources(str(tmp_path)).raw(ref)
        else:
            BindingSources(str(tmp_path)).dashboard_input("benchmark_ref", ref)
    assert caught.value.code == "SYSTEM_STORAGE_SECURITY"


def test_fifo_open_is_nonblocking_in_real_child(tmp_path):
    ref = input_ref(tmp_path)
    path = tmp_path / ref["path"]
    path.unlink()
    os.mkfifo(path, 0o600)
    program = """import sys,json
from quant_investor.operations.catchup_binding import BindingSources
from quant_investor.operations.daily_preparation import Sources
from quant_investor.system.errors import SystemSecurityError
r=json.loads(sys.argv[2])
for read in [lambda: BindingSources(sys.argv[1]).dashboard_input('benchmark_ref',r),lambda:Sources(sys.argv[1]).raw(r)]:
 try: read()
 except SystemSecurityError: pass
 else: raise AssertionError('FIFO admitted')
print('FIFO_REFUSED')
"""
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), json.dumps(ref)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "FIFO_REFUSED"


@pytest.mark.parametrize("reader", ["preparation", "catchup"])
@pytest.mark.parametrize("fault", ["bytes", "inode", "mode", "mtime"])
def test_recheck_rejects_identity_or_byte_drift(tmp_path, reader, fault):
    ref = input_ref(tmp_path)
    path = tmp_path / ref["path"]
    source = Sources(str(tmp_path)) if reader == "preparation" else BindingSources(str(tmp_path))
    source.raw(ref) if reader == "preparation" else source.dashboard_input("benchmark_ref", ref)
    if fault == "bytes":
        path.write_bytes(b"y" * 20)
    elif fault == "inode":
        replacement = path.with_name("replacement.csv")
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(0o644)
        replacement.replace(path)
    elif fault == "mode":
        path.chmod(0o600)
    else:
        s = path.stat()
        os.utime(path, ns=(s.st_atime_ns, s.st_mtime_ns + 1000000))
    error = PreparationError if reader == "preparation" else ContractError
    code = (
        "PREPARATION_SOURCE_CHANGED"
        if reader == "preparation"
        else "CATCHUP_BINDING_SOURCE_CHANGED"
    )
    with pytest.raises(error, match=code):
        source.recheck()


def test_sha_and_ordinary_io_errors_keep_exact_taxonomy(tmp_path):
    ref = input_ref(tmp_path)
    with pytest.raises(ContractError, match="CATCHUP_BINDING_SOURCE_SHA_MISMATCH"):
        BindingSources(str(tmp_path)).dashboard_input("benchmark_ref", {**ref, "sha256": "0" * 64})
    with pytest.raises(PreparationError, match="PREPARATION_SOURCE_SHA_MISMATCH"):
        Sources(str(tmp_path)).raw({**ref, "sha256": "0" * 64})
    (tmp_path / ref["path"]).unlink()
    with pytest.raises(SystemStorageError) as caught:
        BindingSources(str(tmp_path)).dashboard_input("benchmark_ref", ref)
    assert type(caught.value) is SystemStorageError
    assert isinstance(caught.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6])
def test_schema_gate_only_grants_modern_top_level_role_profile(tmp_path, version):
    from quant_investor.operations import execution_recipe as recipes

    value = template(tmp_path)
    fields = getattr(recipes, "FIELDS" if version == 1 else f"FIELDS_V{version}")
    leaf = put(tmp_path, "extra.json", b"{}")
    value = {key: value.get(key, deepcopy(leaf)) for key in fields}
    value["schema_version"] = f"cn-daily-execute-recipe.v{version}"
    source = BindingSources(str(tmp_path))
    if version < 4:
        with pytest.raises(SystemSecurityError):
            check_template_sources(source, value)
    else:
        check_template_sources(source, value)
        source.recheck()
    value["unexpected_top_level_field"] = None
    with pytest.raises(ContractError, match="CATCHUP_TEMPLATE_FIELDS_INVALID"):
        check_template_sources(BindingSources(str(tmp_path)), value)


def test_frozen_v6_checks_risk_free_without_editing_or_reading_mutable_fields(tmp_path):
    from quant_investor.operations.execution_recipe import FIELDS_V6

    value = template(tmp_path)
    leaf = put(tmp_path, "extra.json", b"{}")
    value = {key: value.get(key, deepcopy(leaf)) for key in FIELDS_V6}
    value["schema_version"] = "cn-daily-execute-recipe.v6"
    value["store_preimages"] = {"pointer": {"path": "absent.json", "sha256": "a" * 64}}
    (tmp_path / PATHS["benchmark_ref"]).unlink()
    before = deepcopy(value)
    source = BindingSources(str(tmp_path))
    check_template_sources(source, value, profile="FROZEN_REGISTERED_RECOVERY")
    source.recheck()
    assert value == before
    with pytest.raises(SystemStorageError):
        check_template_sources(BindingSources(str(tmp_path)), value, profile="FRESH")
    (tmp_path / PATHS["risk_free_ref"]).write_bytes(b"changed")
    with pytest.raises(ContractError, match="CATCHUP_BINDING_SOURCE_CHANGED"):
        source.recheck()


@pytest.mark.parametrize(
    "profile", ["UNKNOWN", None, {"schema_version": "FROZEN_REGISTERED_RECOVERY"}]
)
def test_caller_profile_cannot_be_inferred_from_data(tmp_path, profile):
    with pytest.raises(ContractError, match="CATCHUP_TEMPLATE_READ_PROFILE_INVALID"):
        check_template_sources(BindingSources(str(tmp_path)), template(tmp_path), profile=profile)


@pytest.mark.parametrize(
    "fault", ["owner", "gid_drift", "replaced_during_read", "named_replacement", "read_io"]
)
def test_descriptor_metadata_and_io_races_fail_closed(tmp_path, monkeypatch, fault):
    from quant_investor.operations import _source_bytes as reader

    ref = input_ref(tmp_path)
    original_fstat, original_read = os.fstat, os.read
    changed = False

    def metadata(fd):
        s = original_fstat(fd)
        if not stat.S_ISREG(s.st_mode):
            return s
        if fault in {"owner", "gid_drift"}:
            data = {name: getattr(s, name) for name in dir(s) if name.startswith("st_")}
            if fault == "owner":
                data["st_uid"] += 1
            elif changed:
                data["st_gid"] += 1
            return SimpleNamespace(**data)
        return s

    def read(fd, amount):
        nonlocal changed
        if fault == "read_io":
            raise OSError("private I/O detail")
        raw = original_read(fd, amount)
        if not changed and fault in {"replaced_during_read", "named_replacement"}:
            path = tmp_path / ref["path"]
            replacement = path.with_name("new.csv")
            replacement.write_bytes(path.read_bytes())
            replacement.chmod(0o644)
            if fault == "named_replacement":
                path.rename(path.with_name("retained-original.csv"))
            replacement.replace(path)
        changed = True
        return raw

    monkeypatch.setattr(reader.os, "fstat", metadata)
    monkeypatch.setattr(reader.os, "read", read)
    error = (
        SystemSecurityError
        if fault in {"owner", "replaced_during_read"}
        else SystemStorageError if fault == "read_io" else ContractError
    )
    with pytest.raises(error) as caught:
        BindingSources(str(tmp_path)).dashboard_input("benchmark_ref", ref)
    if error is ContractError:
        assert str(caught.value) == "CATCHUP_BINDING_SOURCE_CHANGED"
    if fault == "read_io":
        assert type(caught.value) is SystemStorageError
        assert isinstance(caught.value.__cause__, OSError)


@pytest.mark.parametrize(
    "path", ["../out.csv", "/out.csv", "a//b", "./a", "a\\b", "中文.csv", "bad\x00.csv"]
)
def test_private_helper_rejects_noncanonical_paths(tmp_path, path):
    from quant_investor.operations._source_bytes import read_source_bytes
    from quant_investor.system.storage import SecureSystemStorage

    with pytest.raises(SystemSecurityError):
        read_source_bytes(
            SecureSystemStorage(str(tmp_path)),
            path,
            allowed_modes=frozenset({0o400, 0o600, 0o644}),
            maximum_bytes=8 * MIB,
            require_owner=True,
            require_single_link=True,
            require_non_executable=True,
        )


def test_all_three_fresh_entrypoints_use_real_reader_with_public_csv(tmp_path, monkeypatch):
    from test_automatic_catchup_execution import execution_fixture, run
    from test_daily_launch_inspection import inspect
    from test_automatic_catchup_resolution import resolve
    from scripts import daily_launch_inspection

    request, _, _, calls, _ = execution_fixture(tmp_path, monkeypatch)
    declaration = json.loads((tmp_path / request["recipe_ref"]["path"]).read_bytes())
    refs = {role: input_ref(tmp_path, role) for role in PATHS}
    for recipe in declaration["recipes"].values():
        recipe["dashboard_sources"] = refs
    request["recipe_ref"] = put(tmp_path, request["recipe_ref"]["path"], declaration)
    request_ref = put(tmp_path, "auto.json", request)
    before = inventory(tmp_path)
    selected = resolve(tmp_path, request_ref, request)
    assert selected["missing_input_dates"] == []
    monkeypatch.setattr(daily_launch_inspection, "verify_recipe_static_controls", lambda **kw: None)
    assert inspect(tmp_path, request, request_ref)["mode"] == "PRODUCER_REQUIRED"
    assert inventory(tmp_path) == before and calls == []
    # Native Calendar/resolver/checker/coordinator, explicit existing EOD/producer seams.
    run(tmp_path, request, request_ref)
    assert [row[0] for row in calls] == ["20260827", "20260828"]


def test_cli_maps_real_unsafe_source_to_existing_input_failure(tmp_path, monkeypatch, capsys):
    from quant_investor.cli import daily_prepare as cli
    from test_daily_prepare_cli import arguments

    args = arguments(tmp_path)
    ref = input_ref(tmp_path, mode=0o666)
    before = inventory(tmp_path)
    monkeypatch.setattr(cli, "verify_running_release_install_input", lambda *a, **k: {})
    monkeypatch.setattr(cli, "prepare_daily_request", lambda **kw: Sources(str(tmp_path)).raw(ref))
    with pytest.raises(SystemExit) as caught:
        cli.main(args)
    output = capsys.readouterr()
    assert caught.value.code == 2
    assert output.out == '{"blocker_code":"PREPARATION_INPUT_REJECTED","status":"BLOCKED"}\n'
    assert str(tmp_path) not in output.out + output.err
    assert inventory(tmp_path) == before
