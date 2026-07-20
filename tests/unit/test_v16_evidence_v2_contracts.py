from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from quant_investor.v16.evidence_v2 import secure_io
from quant_investor.v16.evidence_v2.contracts import (
    BoundCanonicalArtifact,
    BoundRawArtifact,
    EVIDENCE_REF_SCHEMA,
    MAX_JSON_ITEMS,
    MAX_JSON_STRING_BYTES,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    encode_f64,
    parse_canonical_json_bytes,
    seal_semantic,
    semantic_sha256,
)
from quant_investor.v16.evidence_v2.secure_io import (
    PRIVATE_EVIDENCE_POLICY,
    RootPolicy,
    _read_bound_bytes,
    _read_bound_canonical_json,
    load_bound_canonical_artifact,
    load_bound_raw_artifact,
    platform_acl_absent,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reference(path: Path, payload: bytes, semantic_sha: str) -> EvidenceRef:
    return EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema="fixture.v1",
        absolute_path=str(path),
        byte_sha256=_sha(payload),
        semantic_sha256=semantic_sha,
        root_policy=PRIVATE_EVIDENCE_POLICY.policy_id,
    )


def _acl_absent(_fd: int, _label: str) -> bool:
    return True


def test_canonical_json_round_trip_and_binary64_normalization() -> None:
    payload = seal_semantic(
        {
            "schema_version": "fixture.v1",
            "number": encode_f64(-0.0),
            "label": "evidence",
        }
    )
    encoded = canonical_json_bytes(payload)

    assert encode_f64(-0.0) == "f64:0x0.0p+0"
    assert parse_canonical_json_bytes(encoded) == payload
    assert encoded.endswith(b"\n")


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1,"a":2}\n',
        b'{"a":1.5}\n',
        b'{"a":1}\n\n',
        b'{ "a":1}\n',
        b'{"a":9007199254740992}\n',
    ],
)
def test_canonical_json_rejects_ambiguous_or_noncanonical_bytes(payload: bytes) -> None:
    with pytest.raises(EvidenceV2Error):
        parse_canonical_json_bytes(payload)


def test_canonical_json_rejects_native_float_before_serialization() -> None:
    with pytest.raises(EvidenceV2Error, match="native JSON float"):
        canonical_json_bytes({"value": 0.5})


def test_canonical_json_enforces_aggregate_item_and_encoded_byte_limits() -> None:
    with pytest.raises(EvidenceV2Error, match="aggregate item limit"):
        canonical_json_bytes([None] * (MAX_JSON_ITEMS + 1))

    large_string = "x" * MAX_JSON_STRING_BYTES
    with pytest.raises(EvidenceV2Error, match="artifact byte limit"):
        canonical_json_bytes([large_string] * 16)


def test_evidence_ref_rejects_lexically_ambiguous_absolute_path() -> None:
    with pytest.raises(EvidenceV2Error, match="lexically canonical"):
        EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema="fixture.v1",
            absolute_path="/private/evidence/../replacement.json",
            byte_sha256="a" * 64,
            semantic_sha256="b" * 64,
            root_policy="v16.private-evidence-root.v2",
        )


def test_private_bound_read_checks_hash_mode_and_semantic_identity(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    path = root / "artifact.json"
    value = seal_semantic({"schema_version": "fixture.v1", "value": "bound"})
    payload = canonical_json_bytes(value)
    path.write_bytes(payload)
    path.chmod(0o600)
    reference = _reference(path, payload, semantic_sha256(value))

    bound = _read_bound_bytes(
        root=root,
        path=path,
        policy=PRIVATE_EVIDENCE_POLICY,
        expected_sha256=reference.byte_sha256,
        acl_checker=_acl_absent,
    )
    decoded = _read_bound_canonical_json(
        root=root,
        reference=reference,
        policy=PRIVATE_EVIDENCE_POLICY,
        acl_checker=_acl_absent,
    )

    assert bound.payload == payload
    assert decoded == value


def test_private_bound_read_rejects_writable_trusted_root_ancestor(
    tmp_path: Path,
) -> None:
    ancestor = tmp_path / "trusted"
    ancestor.mkdir(mode=0o700)
    root = ancestor / "private"
    root.mkdir(mode=0o700)
    path = root / "artifact.json"
    payload = b"bound\n"
    path.write_bytes(payload)
    path.chmod(0o600)

    bound = _read_bound_bytes(
        root=root,
        path=path,
        policy=PRIVATE_EVIDENCE_POLICY,
        expected_sha256=_sha(payload),
        acl_checker=_acl_absent,
    )

    assert bound.payload == payload

    ancestor.chmod(0o777)
    try:
        with pytest.raises(
            EvidenceV2Error,
            match="trusted root ancestor is group/world writable",
        ):
            _read_bound_bytes(
                root=root,
                path=path,
                policy=PRIVATE_EVIDENCE_POLICY,
                expected_sha256=_sha(payload),
                acl_checker=_acl_absent,
            )
    finally:
        ancestor.chmod(0o700)


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin ACL API required")
def test_private_bound_read_rejects_allow_acl_on_trusted_root_ancestor(
    tmp_path: Path,
) -> None:
    ancestor = tmp_path / "trusted"
    ancestor.mkdir(mode=0o700)
    root = ancestor / "private"
    root.mkdir(mode=0o700)
    path = root / "artifact.json"
    payload = b"bound\n"
    path.write_bytes(payload)
    path.chmod(0o600)
    acl_result = subprocess.run(
        ["chmod", "+a", "everyone allow write", str(ancestor)],
        check=False,
        capture_output=True,
        text=True,
    )
    if acl_result.returncode != 0:
        pytest.skip(f"extended ACL setup unavailable: {acl_result.stderr.strip()}")

    with pytest.raises(EvidenceV2Error, match="ancestor has an extended allow ACL"):
        load_bound_raw_artifact(
            root=root,
            reference=_reference(path, payload, "a" * 64),
            policy=PRIVATE_EVIDENCE_POLICY,
        )


def test_private_bound_read_rejects_hardlink_and_symlink(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    original = root / "artifact.json"
    payload = b"bound\n"
    original.write_bytes(payload)
    original.chmod(0o600)
    hardlink = root / "hardlink.json"
    os.link(original, hardlink)

    with pytest.raises(EvidenceV2Error, match="hard link"):
        _read_bound_bytes(
            root=root,
            path=original,
            policy=PRIVATE_EVIDENCE_POLICY,
            expected_sha256=_sha(payload),
            acl_checker=_acl_absent,
        )

    original.unlink()
    hardlink.unlink()
    outside = tmp_path / "outside.json"
    outside.write_bytes(payload)
    outside.chmod(0o600)
    symlink = root / "symlink.json"
    symlink.symlink_to(outside)
    with pytest.raises(EvidenceV2Error, match="open failed"):
        _read_bound_bytes(
            root=root,
            path=symlink,
            policy=PRIVATE_EVIDENCE_POLICY,
            expected_sha256=_sha(payload),
            acl_checker=_acl_absent,
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin ACL API required")
def test_private_bound_read_uses_platform_acl_absence_proof(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    path = root / "artifact.json"
    payload = b"bound\n"
    path.write_bytes(payload)
    path.chmod(0o600)

    bound = _read_bound_bytes(
        root=root,
        path=path,
        policy=PRIVATE_EVIDENCE_POLICY,
        expected_sha256=_sha(payload),
    )
    descriptor = os.open(path, os.O_RDONLY)
    try:
        assert platform_acl_absent(descriptor, str(path)) is True
    finally:
        os.close(descriptor)

    assert bound.payload == payload


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin ACL API required")
def test_private_bound_read_rejects_real_extended_acl(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    path = root / "artifact.json"
    payload = b"bound\n"
    path.write_bytes(payload)
    path.chmod(0o600)
    acl_result = subprocess.run(
        ["chmod", "+a", "everyone allow read", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if acl_result.returncode != 0:
        pytest.skip(f"extended ACL setup unavailable: {acl_result.stderr.strip()}")

    descriptor = os.open(path, os.O_RDONLY)
    try:
        assert platform_acl_absent(descriptor, str(path)) is False
    finally:
        os.close(descriptor)
    with pytest.raises(EvidenceV2Error, match="has an extended ACL"):
        _read_bound_bytes(
            root=root,
            path=path,
            policy=PRIVATE_EVIDENCE_POLICY,
            expected_sha256=_sha(payload),
        )


@pytest.mark.skipif(sys.platform != "darwin", reason="Darwin ACL API required")
def test_secure_factories_return_byte_bound_artifacts_and_reject_tamper(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)

    canonical_path = root / "artifact.json"
    value = seal_semantic({"schema_version": "fixture.v1", "value": "factory"})
    canonical_payload = canonical_json_bytes(value)
    canonical_path.write_bytes(canonical_payload)
    canonical_path.chmod(0o600)
    canonical_reference = _reference(
        canonical_path,
        canonical_payload,
        semantic_sha256(value),
    )

    canonical = load_bound_canonical_artifact(
        root=root,
        reference=canonical_reference,
        policy=PRIVATE_EVIDENCE_POLICY,
    )

    assert isinstance(canonical, BoundCanonicalArtifact)
    assert canonical.read() == value

    raw_path = root / "artifact.bin"
    raw_payload = b"opaque-evidence-v2\x00"
    raw_path.write_bytes(raw_payload)
    raw_path.chmod(0o600)
    raw_reference = _reference(raw_path, raw_payload, "b" * 64)
    raw = load_bound_raw_artifact(
        root=root,
        reference=raw_reference,
        policy=PRIVATE_EVIDENCE_POLICY,
    )

    assert isinstance(raw, BoundRawArtifact)
    assert raw.payload == raw_payload

    canonical_path.write_bytes(canonical_payload.replace(b"factory", b"tampered"))
    canonical_path.chmod(0o600)
    with pytest.raises(EvidenceV2Error, match="byte SHA mismatch"):
        load_bound_canonical_artifact(
            root=root,
            reference=canonical_reference,
            policy=PRIVATE_EVIDENCE_POLICY,
        )


def test_secure_io_public_surface_excludes_acl_injectable_readers() -> None:
    assert "read_bound_bytes" not in secure_io.__all__
    assert "read_bound_canonical_json" not in secure_io.__all__
    assert "_read_bound_bytes" not in secure_io.__all__
    assert "_read_bound_canonical_json" not in secure_io.__all__


def test_secure_factory_rejects_caller_weakened_root_policy(tmp_path: Path) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    path = root / "artifact.bin"
    payload = b"opaque-evidence-v2\n"
    path.write_bytes(payload)
    path.chmod(0o600)
    weakened = RootPolicy(
        policy_id=PRIVATE_EVIDENCE_POLICY.policy_id,
        directory_mode=0o700,
        file_mode=0o600,
        require_current_uid=True,
        reject_group_world_write=True,
        require_no_extended_acl=False,
    )

    with pytest.raises(EvidenceV2Error, match="canonical root policy"):
        load_bound_raw_artifact(
            root=root,
            reference=_reference(path, payload, "a" * 64),
            policy=weakened,
        )


def test_evidence_v2_is_not_imported_by_authorizing_runtime_modules() -> None:
    source_root = Path(__file__).resolve().parents[2] / "quant_investor"
    violations: list[str] = []
    for path in source_root.rglob("*.py"):
        if "evidence_v2" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            if any(module.startswith("quant_investor.v16.evidence_v2") for module in modules):
                violations.append(str(path.relative_to(source_root.parent)))
    assert violations == []
