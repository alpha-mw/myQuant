from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from quant_investor.cli.input import read_exact_request
from quant_investor.cli.output import CommandError


def _write(path: Path, value: object) -> tuple[bytes, str]:
    raw = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    path.write_bytes(raw)
    path.chmod(0o600)
    return raw, hashlib.sha256(raw).hexdigest()


def test_exact_request_round_trip(tmp_path: Path) -> None:
    raw, digest = _write(tmp_path / "request.json", {"operation": "research.forward"})

    observed_raw, document = read_exact_request(tmp_path, "request.json", digest)

    assert observed_raw == raw
    assert document == {"operation": "research.forward"}


@pytest.mark.parametrize(
    ("relative_path", "expected_code"),
    [
        ("../request.json", "REQUEST_PATH_INVALID"),
        ("/request.json", "REQUEST_PATH_INVALID"),
        ("request\\file.json", "REQUEST_PATH_INVALID"),
    ],
)
def test_request_path_is_canonical(tmp_path: Path, relative_path: str, expected_code: str) -> None:
    with pytest.raises(CommandError) as captured:
        read_exact_request(tmp_path, relative_path, "0" * 64)

    assert captured.value.blocker_code == expected_code


def test_request_rejects_symlink_and_wrong_hash(tmp_path: Path) -> None:
    _, digest = _write(tmp_path / "request.json", {"value": 1})
    os.symlink("request.json", tmp_path / "alias.json")

    with pytest.raises(CommandError) as symlink_error:
        read_exact_request(tmp_path, "alias.json", digest)
    with pytest.raises(CommandError) as hash_error:
        read_exact_request(tmp_path, "request.json", "0" * 64)

    assert symlink_error.value.blocker_code == "REQUEST_SYMLINK_REFUSED"
    assert hash_error.value.blocker_code == "REQUEST_SHA256_MISMATCH"


def test_request_rejects_noncanonical_and_unsafe_mode(tmp_path: Path) -> None:
    path = tmp_path / "request.json"
    raw = b'{"b":2, "a":1}'
    path.write_bytes(raw)
    path.chmod(0o600)
    digest = hashlib.sha256(raw).hexdigest()
    with pytest.raises(CommandError) as noncanonical:
        read_exact_request(tmp_path, "request.json", digest)

    path.chmod(0o622)
    with pytest.raises(CommandError) as unsafe:
        read_exact_request(tmp_path, "request.json", digest)

    assert noncanonical.value.blocker_code == "REQUEST_NOT_CANONICAL"
    assert unsafe.value.blocker_code == "REQUEST_FILE_UNSAFE"
