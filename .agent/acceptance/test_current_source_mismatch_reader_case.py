"""Private acceptance-harness checks, separate from frozen business CI."""

import importlib.util
from pathlib import Path

import pytest
from quant_investor.system.storage import SecureSystemStorage

spec = importlib.util.spec_from_file_location(
    "case8_helper", Path(__file__).with_name("current_source_mismatch_reader_case.py")
)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def put(root, relative, raw):
    root.mkdir(mode=0o700, exist_ok=True)
    leaf = root / relative
    leaf.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    for parent in leaf.parents:
        if parent == root:
            break
        parent.chmod(0o700)
    leaf.write_bytes(raw)
    leaf.chmod(0o600)


def test_exact_pair_only_and_native_metadata_and_limit_forwarding(tmp_path, monkeypatch):
    a, b, c = (tmp_path / name for name in ["original", "fault", "other"])
    path = "results/case8/market.json"
    second = "results/case8/other.json"
    for root, raw in [(a, b"original"), (b, b"corrupt!"), (c, b"outside!")]:
        put(root, path, raw)
        put(root, second, b"otherpath")
    saved = SecureSystemStorage.read_workspace_file_bytes
    calls = []

    def observe(reader, relative, *, maximum_bytes):
        calls.append((reader.workspace_root, str(relative), maximum_bytes))
        return saved(reader, relative, maximum_bytes=maximum_bytes)

    monkeypatch.setattr(SecureSystemStorage, "read_workspace_file_bytes", observe)
    original = SecureSystemStorage(str(a))
    other = SecureSystemStorage(str(c))
    expected = saved(SecureSystemStorage(str(b)), path, maximum_bytes=64)
    with helper.route_physical_fault(a, path, b) as counts:
        actual = original.read_workspace_file_bytes(path, maximum_bytes=64)
        assert actual == expected and actual.data == b"corrupt!"
        assert other.read_workspace_file_bytes(path, maximum_bytes=32).data == b"outside!"
        assert original.read_workspace_file_bytes(second, maximum_bytes=24).data == b"otherpath"
    assert calls == [(b.resolve(), path, 64), (c.resolve(), path, 32), (a.resolve(), second, 24)]
    assert counts == {
        "wrapper_calls": 3,
        "target_redirects": 1,
        "successful_physical_reads": 1,
        "non_target_redirects": 0,
        "recursive_calls": 0,
        "restored": True,
    }
    assert SecureSystemStorage.read_workspace_file_bytes is observe
    assert (a / path).read_bytes() == b"original"


def test_native_size_error_restores_reader(tmp_path):
    a, b = tmp_path / "original", tmp_path / "fault"
    path = "results/case8/pit.json"
    put(a, path, b"correct")
    put(b, path, b"corrupt")
    saved = SecureSystemStorage.read_workspace_file_bytes
    with pytest.raises(Exception) as baseline:
        saved(SecureSystemStorage(str(b)), path, maximum_bytes=2)
    with pytest.raises(type(baseline.value)) as observed:
        with helper.route_physical_fault(a, path, b) as counts:
            SecureSystemStorage(str(a)).read_workspace_file_bytes(path, maximum_bytes=2)
    assert str(observed.value) == str(baseline.value)
    assert counts["target_redirects"] == 1 and counts["successful_physical_reads"] == 0
    assert counts["restored"] and SecureSystemStorage.read_workspace_file_bytes is saved


def test_native_financial_0644_read_preserves_permissions(tmp_path):
    import hashlib

    root = tmp_path / "workspace"
    path = "results/strategy_records/CN/aggressive_tech_manufacturing/example/manifest.json"
    raw = b'{"fixture":true}'
    put(root, path, raw)
    leaf = root / path
    leaf.chmod(0o644)
    before = helper.file_identity(leaf)
    ref = {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
    assert (
        helper.read_selected_bytes(SecureSystemStorage(str(root)), root, ref, financial=True) == raw
    )
    assert helper.file_identity(leaf) == before
    with pytest.raises(Exception):
        helper.read_selected_bytes(SecureSystemStorage(str(root)), root, ref)


def test_financial_reader_cannot_widen_other_source_scope(tmp_path):
    import hashlib

    root = tmp_path / "workspace"
    path = "results/case8/market.json"
    raw = b"{}"
    put(root, path, raw)
    (root / path).chmod(0o644)
    ref = {"path": path, "sha256": hashlib.sha256(raw).hexdigest()}
    with pytest.raises(AssertionError, match="financial reader scope"):
        helper.read_selected_bytes(SecureSystemStorage(str(root)), root, ref, financial=True)
