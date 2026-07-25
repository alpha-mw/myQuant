from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import stat

import pytest

from quant_investor.v17.contracts import V17ContractError
from quant_investor.v17 import storage
from quant_investor.v17.storage import (
    atomic_copy_file_exact_once,
    atomic_write_json,
    atomic_write_json_exact_once,
    ensure_private_directory,
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_regular(path: Path, payload: bytes, *, mode: int = 0o600) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(mode)
    return _sha256(payload)


def _tmp_names(parent: Path) -> list[Path]:
    return sorted(parent.glob(".*.tmp"))


def test_streaming_copy_installs_file_larger_than_json_limit_without_temp_leaks(
    tmp_path: Path,
) -> None:
    payload = (b"v17-streaming-block:" + b"x" * 1021) * (16 * 1024 + 1)
    source = tmp_path / "source.bin"
    expected = _write_regular(source, payload)
    root = tmp_path / "private"
    target = root / "objects" / "aa" / "payload.bin"

    digest = atomic_copy_file_exact_once(
        source,
        target,
        root=root,
        expected_source_sha256=expected,
        expected_size_bytes=len(payload),
    )

    assert digest == expected
    assert target.read_bytes() == payload
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert _tmp_names(target.parent) == []


def test_wrong_source_digest_removes_temporary_file_and_leaves_target_missing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    _write_regular(source, b"sealed-source")
    root = tmp_path / "private"
    target = root / "objects" / "00" / "payload.bin"

    with pytest.raises(V17ContractError, match="byte SHA mismatch"):
        atomic_copy_file_exact_once(
            source,
            target,
            root=root,
            expected_source_sha256="0" * 64,
        )

    assert not target.exists()
    assert _tmp_names(target.parent) == []


def test_target_path_traversal_is_rejected_before_writing_outside_root(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.bin"
    expected = _write_regular(source, b"safe")
    root = tmp_path / "private"
    outside = tmp_path / "escape.bin"

    with pytest.raises(V17ContractError, match="escapes fixed root"):
        atomic_copy_file_exact_once(
            source,
            root / ".." / outside.name,
            root=root,
            expected_source_sha256=expected,
        )

    assert not outside.exists()


def test_filesystem_root_cannot_be_used_as_private_chmod_boundary(tmp_path: Path) -> None:
    with pytest.raises(V17ContractError, match="filesystem root"):
        ensure_private_directory(tmp_path / "nested", root=Path("/"))


def test_source_symlink_is_rejected(tmp_path: Path) -> None:
    real_source = tmp_path / "real.bin"
    expected = _write_regular(real_source, b"safe")
    link = tmp_path / "source-link.bin"
    link.symlink_to(real_source)

    with pytest.raises(V17ContractError, match="symlink"):
        atomic_copy_file_exact_once(
            link,
            tmp_path / "private" / "object.bin",
            root=tmp_path / "private",
            expected_source_sha256=expected,
        )


def test_source_hardlink_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    expected = _write_regular(source, b"safe")
    hardlink = tmp_path / "source-hardlink.bin"
    os.link(source, hardlink)

    with pytest.raises(V17ContractError, match="hard-linked stream source rejected"):
        atomic_copy_file_exact_once(
            source,
            tmp_path / "private" / "object.bin",
            root=tmp_path / "private",
            expected_source_sha256=expected,
        )


def test_source_path_replacement_during_copy_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    payload = b"a" * (1024 * 1024 + 5)
    expected = _write_regular(source, payload)
    original_inode = source.stat().st_ino
    original_read = os.read
    replaced = False

    def replacing_read(descriptor: int, count: int) -> bytes:
        nonlocal replaced
        entry = os.fstat(descriptor)
        if not replaced and entry.st_ino == original_inode and count == storage._STREAM_CHUNK_BYTES:
            source.rename(tmp_path / "source.old")
            _write_regular(source, b"b" * len(payload))
            replaced = True
        return original_read(descriptor, count)

    monkeypatch.setattr(storage.os, "read", replacing_read)

    with pytest.raises(
        V17ContractError, match="source changed during read|path changed during read"
    ):
        atomic_copy_file_exact_once(
            source,
            tmp_path / "private" / "object.bin",
            root=tmp_path / "private",
            expected_source_sha256=expected,
            expected_size_bytes=len(payload),
        )

    assert replaced is True


def test_existing_different_byte_target_is_left_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    expected = _write_regular(source, b"new-bytes")
    root = tmp_path / "private"
    target = root / "objects" / "aa" / "payload.bin"
    _write_regular(target, b"old-bytes")
    before = target.read_bytes()

    with pytest.raises(V17ContractError, match="target byte mismatch|identity invalid"):
        atomic_copy_file_exact_once(
            source,
            target,
            root=root,
            expected_source_sha256=expected,
        )

    assert target.read_bytes() == before
    assert _tmp_names(target.parent) == []


def test_same_byte_concurrent_install_is_idempotent(tmp_path: Path) -> None:
    payload = b"same immutable object"
    source_a = tmp_path / "source-a.bin"
    source_b = tmp_path / "source-b.bin"
    expected = _write_regular(source_a, payload)
    _write_regular(source_b, payload)
    target = tmp_path / "private" / "objects" / "aa" / "payload.bin"
    target.parent.mkdir(parents=True, mode=0o700)
    (tmp_path / "private").chmod(0o700)
    (tmp_path / "private" / "objects").chmod(0o700)
    target.parent.chmod(0o700)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda source: atomic_copy_file_exact_once(
                    source,
                    target,
                    root=tmp_path / "private",
                    expected_source_sha256=expected,
                    expected_size_bytes=len(payload),
                ),
                (source_a, source_b),
            )
        )

    assert results == [expected, expected]
    assert target.read_bytes() == payload
    assert target.stat().st_nlink == 1
    assert _tmp_names(target.parent) == []


def test_different_byte_concurrent_install_has_one_winner_and_no_overwrite(
    tmp_path: Path,
) -> None:
    sources = (tmp_path / "source-a.bin", tmp_path / "source-b.bin")
    payloads = (b"first immutable object", b"second immutable object")
    digests = tuple(_write_regular(source, payload) for source, payload in zip(sources, payloads))
    target = tmp_path / "private" / "objects" / "aa" / "payload.bin"

    def install(index: int) -> tuple[str, str]:
        try:
            result = atomic_copy_file_exact_once(
                sources[index],
                target,
                root=tmp_path / "private",
                expected_source_sha256=digests[index],
            )
            return ("installed", result)
        except V17ContractError as exc:
            return ("rejected", str(exc))

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(install, (0, 1)))

    assert [status for status, _ in outcomes].count("installed") == 1
    assert [status for status, _ in outcomes].count("rejected") == 1
    assert target.read_bytes() in payloads
    assert target.stat().st_nlink == 1
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert _tmp_names(target.parent) == []


def test_interrupted_link_cas_state_fails_closed_without_unlinking_evidence(
    tmp_path: Path,
) -> None:
    payload = b"durable but interrupted object"
    source = tmp_path / "source.bin"
    expected = _write_regular(source, payload)
    target = tmp_path / "private" / "objects" / "aa" / "payload.bin"
    _write_regular(target, payload)
    interrupted_temporary = target.parent / f".{target.name}.interrupted.tmp"
    os.link(target, interrupted_temporary)
    assert target.stat().st_nlink == 2

    with pytest.raises(V17ContractError, match="hard-link count"):
        atomic_copy_file_exact_once(
            source,
            target,
            root=tmp_path / "private",
            expected_source_sha256=expected,
        )

    assert target.read_bytes() == payload
    assert interrupted_temporary.read_bytes() == payload
    assert target.stat().st_nlink == 2


def test_atomic_replace_uses_parent_fd_when_parent_path_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "private"
    target = root / "nested" / "artifact.json"
    moved_parent = root / "nested-moved"
    outside = tmp_path / "outside"
    original_replace = storage.os.replace
    attacked = False

    def replacing_parent(source: str, destination: str, **kwargs: int) -> None:
        nonlocal attacked
        target.parent.rename(moved_parent)
        outside.mkdir()
        target.parent.symlink_to(outside, target_is_directory=True)
        attacked = True
        original_replace(source, destination, **kwargs)

    monkeypatch.setattr(storage.os, "replace", replacing_parent)

    with pytest.raises(V17ContractError, match="parent path identity drift"):
        atomic_write_json(target, {"safe": True}, root=root)

    assert attacked is True
    assert not (outside / target.name).exists()
    assert (moved_parent / target.name).read_bytes() == b'{"safe":true}\n'


def test_exact_once_link_uses_parent_fd_when_parent_path_is_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "private"
    target = root / "nested" / "artifact.json"
    moved_parent = root / "nested-moved"
    outside = tmp_path / "outside"
    original_link = storage.os.link
    attacked = False

    def replacing_parent(source: str, destination: str, **kwargs: int | bool) -> None:
        nonlocal attacked
        target.parent.rename(moved_parent)
        outside.mkdir()
        target.parent.symlink_to(outside, target_is_directory=True)
        attacked = True
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(storage.os, "link", replacing_parent)

    with pytest.raises(V17ContractError, match="parent path identity drift"):
        atomic_write_json_exact_once(target, {"safe": True}, root=root)

    assert attacked is True
    assert not (outside / target.name).exists()
    assert (moved_parent / target.name).read_bytes() == b'{"safe":true}\n'


@pytest.mark.parametrize("exact_once", [False, True])
def test_atomic_write_rejects_private_boundary_symlink_swap_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exact_once: bool,
) -> None:
    root = tmp_path / "private"
    target = root / "nested" / "artifact.json"
    target.parent.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    target.parent.chmod(0o700)
    moved_root = tmp_path / "private-moved"
    outside = tmp_path / "outside"
    (outside / "nested").mkdir(parents=True, mode=0o700)
    outside.chmod(0o700)
    original_open = storage.os.open
    attacked = False

    def replacing_boundary(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal attacked
        if not attacked and path == root.name and kwargs.get("dir_fd") is not None:
            root.rename(moved_root)
            root.symlink_to(outside, target_is_directory=True)
            attacked = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", replacing_boundary)
    writer = atomic_write_json_exact_once if exact_once else atomic_write_json

    with pytest.raises(V17ContractError, match="without symlinks|component unavailable"):
        writer(target, {"safe": True}, root=root)

    assert attacked is True
    assert not (outside / "nested" / target.name).exists()
    assert not (moved_root / "nested" / target.name).exists()


def test_stream_copy_rejects_private_boundary_symlink_swap_before_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.bin"
    expected = _write_regular(source, b"sealed-stream-source")
    root = tmp_path / "private"
    target = root / "nested" / "object.bin"
    target.parent.mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    target.parent.chmod(0o700)
    moved_root = tmp_path / "private-moved"
    outside = tmp_path / "outside"
    (outside / "nested").mkdir(parents=True, mode=0o700)
    outside.chmod(0o700)
    original_open = storage.os.open
    attacked = False

    def replacing_boundary(path: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal attacked
        if not attacked and path == root.name and kwargs.get("dir_fd") is not None:
            root.rename(moved_root)
            root.symlink_to(outside, target_is_directory=True)
            attacked = True
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(storage.os, "open", replacing_boundary)

    with pytest.raises(V17ContractError, match="without symlinks|component unavailable"):
        atomic_copy_file_exact_once(
            source,
            target,
            root=root,
            expected_source_sha256=expected,
        )

    assert attacked is True
    assert not (outside / "nested" / target.name).exists()
    assert not (moved_root / "nested" / target.name).exists()


def test_existing_same_byte_target_with_wrong_mode_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    payload = b"same-bytes"
    expected = _write_regular(source, payload)
    target = tmp_path / "private" / "objects" / "aa" / "payload.bin"
    _write_regular(target, payload, mode=0o644)

    with pytest.raises(V17ContractError, match="identity invalid|mode"):
        atomic_copy_file_exact_once(
            source,
            target,
            root=tmp_path / "private",
            expected_source_sha256=expected,
        )

    assert stat.S_IMODE(target.stat().st_mode) == 0o644
