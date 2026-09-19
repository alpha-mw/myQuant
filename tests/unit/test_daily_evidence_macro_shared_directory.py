"""Macro-first shared research directory safety; closure builder controlled."""

import stat
import pytest
from quant_investor.macro import readiness_closure as macro
from quant_investor.intelligence.storage import _store_parent


def seal(root, monkeypatch):
    monkeypatch.setattr(
        macro,
        "build_macro_readiness_closure",
        lambda **kw: {"target_date": "20260827", "fixture": "directory-scope-only"},
    )
    return macro.seal_macro_readiness_closure(
        workspace_root=root, terminal_path="terminal.json", terminal_sha256="a" * 64
    )


def test_macro_first_creates_pool_compatible_shared_ancestors(tmp_path, monkeypatch):
    first = seal(tmp_path, monkeypatch)
    for relative in [
        "results/intelligence",
        "results/intelligence/macro_readiness",
        "results/intelligence/macro_readiness/20260827",
    ]:
        assert stat.S_IMODE((tmp_path / relative).stat().st_mode) == 0o700
    before = (tmp_path / first["closure_path"]).read_bytes()
    _store_parent(tmp_path, ("intelligence", "research_pool", "aggressive_tech_manufacturing"))
    again = seal(tmp_path, monkeypatch)
    assert again["status"] == "NO_ACTION"
    assert first["closure_sha256"] == again["closure_sha256"]
    assert (tmp_path / first["closure_path"]).read_bytes() == before


@pytest.mark.parametrize("fault", ["mode", "symlink"])
@pytest.mark.parametrize(
    "relative",
    ["intelligence", "intelligence/macro_readiness", "intelligence/macro_readiness/20260827"],
)
def test_macro_never_repairs_or_follows_unsafe_existing_parent(
    tmp_path, monkeypatch, fault, relative
):
    results = tmp_path / "results"
    results.mkdir(mode=0o700)
    parent = results / relative
    parent.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    ancestor = parent.parent
    while ancestor != results:
        ancestor.chmod(0o700)
        ancestor = ancestor.parent
    if fault == "mode":
        parent.mkdir(mode=0o755)
    else:
        external = tmp_path / "external"
        external.mkdir(mode=0o700)
        parent.symlink_to(external, target_is_directory=True)
    before = {
        str(p): (
            p.lstat().st_mode,
            p.lstat().st_mtime_ns,
            str(p.readlink()) if p.is_symlink() else None,
        )
        for p in tmp_path.rglob("*")
    }
    with pytest.raises(
        macro.MacroReadinessClosureError, match="MACRO_READINESS_CLOSURE_DIRECTORY_UNSAFE"
    ):
        seal(tmp_path, monkeypatch)
    assert before == {
        str(p): (
            p.lstat().st_mode,
            p.lstat().st_mtime_ns,
            str(p.readlink()) if p.is_symlink() else None,
        )
        for p in tmp_path.rglob("*")
    }


def test_unexpected_parent_exception_propagates_unchanged(tmp_path, monkeypatch):
    from quant_investor.intelligence import storage

    failure = RuntimeError("sentinel parent failure")

    def unexpected(*args, **kwargs):
        raise failure

    monkeypatch.setattr(storage, "_store_parent", unexpected)
    with pytest.raises(RuntimeError) as caught:
        seal(tmp_path, monkeypatch)
    assert caught.value is failure
    assert list(tmp_path.iterdir()) == []
