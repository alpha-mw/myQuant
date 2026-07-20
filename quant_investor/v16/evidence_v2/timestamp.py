"""RFC3161 anchoring with first-response-wins terminal semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .contracts import (
    BoundCanonicalArtifact,
    EVIDENCE_REF_SCHEMA,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    require_sha256,
    seal_semantic,
    semantic_sha256,
    sha256_bytes,
    validate_semantic_seal,
)
from .runtime_identity import PINNED_OPENSSL_PATH, REQUIRED_ENVIRONMENT_CONTROLS

TIMESTAMP_ATTEMPT_SCHEMA = "v16.rfc3161-attempt-state.v2"
TIMESTAMP_RECEIPT_SCHEMA = "v16.rfc3161-validation-receipt.v2"
TIMESTAMP_RESPONSE_MAX_BYTES = 4 * 1024 * 1024
TIMESTAMP_STATES = (
    "awaiting_transport",
    "response_persisted",
    "validated",
    "failed_terminal",
)
_STATUS_GRANTED = "Status: Granted."
_OPENSSL_TIME_FORMATS = (
    "%b %d %H:%M:%S %Y GMT",
    "%b %d %H:%M:%S.%f %Y GMT",
)
_CERTIFICATE_BLOCK_PATTERN = re.compile(
    rb"-----BEGIN CERTIFICATE-----\s+.*?\s+-----END CERTIFICATE-----",
    flags=re.DOTALL,
)
_CRL_BLOCK_PATTERN = re.compile(
    rb"-----BEGIN X509 CRL-----\s+.*?\s+-----END X509 CRL-----",
    flags=re.DOTALL,
)


def _safe_id(value: Any, *, label: str) -> str:
    text = str(value or "")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not text or text != text.strip() or len(text) > 128:
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    if any(character not in allowed for character in text):
        raise EvidenceV2Error(f"{label} is not a safe identifier")
    return text


def _canonical_utc(value: Any, *, label: str) -> datetime:
    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EvidenceV2Error(f"{label} must be a UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise EvidenceV2Error(f"{label} must be UTC")
    canonical = parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if canonical != text:
        raise EvidenceV2Error(f"{label} must use canonical UTC form")
    return parsed


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise EvidenceV2Error("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _policy_oid(value: Any) -> str:
    text = str(value or "")
    if not re.fullmatch(r"[0-2](?:\.[0-9]+)+", text):
        raise EvidenceV2Error("RFC3161 policy OID is invalid")
    return text


@dataclass(frozen=True)
class BoundArtifact:
    reference: EvidenceRef
    payload: bytes

    def __post_init__(self) -> None:
        if sha256_bytes(self.payload) != self.reference.byte_sha256:
            raise EvidenceV2Error("bound timestamp artifact byte SHA mismatch")


@dataclass(frozen=True)
class RevocationBinding:
    certificate: BoundArtifact
    issuer_certificate: BoundArtifact
    crl: BoundArtifact


def _pem_blocks(payload: bytes, *, kind: str) -> tuple[bytes, ...]:
    pattern = _CERTIFICATE_BLOCK_PATTERN if kind == "certificate" else _CRL_BLOCK_PATTERN
    blocks = tuple(match.group(0).strip() for match in pattern.finditer(payload))
    if not blocks or len(blocks) != len(set(blocks)):
        raise EvidenceV2Error(f"{kind} PEM blocks are missing or duplicated")
    return blocks


@dataclass(frozen=True)
class TimestampVerificationBundle:
    anchored_artifact: BoundArtifact
    query: BoundArtifact
    response: BoundArtifact
    trust_anchor: BoundArtifact
    untrusted_chain: BoundArtifact
    revocations: tuple[RevocationBinding, ...]

    def __post_init__(self) -> None:
        if not self.revocations:
            raise EvidenceV2Error("RFC3161 verification requires CRL evidence")
        chain_blocks = _pem_blocks(self.untrusted_chain.payload, kind="certificate")
        root_blocks = _pem_blocks(self.trust_anchor.payload, kind="certificate")
        if len(root_blocks) != 1 or root_blocks[0] in chain_blocks:
            raise EvidenceV2Error("RFC3161 trust anchor/chain certificate set is invalid")
        covered_blocks: list[bytes] = []
        for binding in self.revocations:
            certificate_blocks = _pem_blocks(
                binding.certificate.payload,
                kind="certificate",
            )
            if len(certificate_blocks) != 1 or certificate_blocks[0] not in chain_blocks:
                raise EvidenceV2Error(
                    "every revocation certificate must be present in the untrusted chain"
                )
            covered_blocks.append(certificate_blocks[0])
            issuer_blocks = _pem_blocks(
                binding.issuer_certificate.payload,
                kind="certificate",
            )
            if len(issuer_blocks) != 1 or (
                issuer_blocks[0] not in chain_blocks and issuer_blocks[0] != root_blocks[0]
            ):
                raise EvidenceV2Error("every CRL issuer must be bound to the chain or trust anchor")
            if len(_pem_blocks(binding.crl.payload, kind="crl")) != 1:
                raise EvidenceV2Error("each revocation binding requires exactly one CRL")
        if set(covered_blocks) != set(chain_blocks):
            raise EvidenceV2Error("CRL bindings must cover every non-root certificate in the chain")
        certificate_shas = [
            binding.certificate.reference.byte_sha256 for binding in self.revocations
        ]
        if len(certificate_shas) != len(set(certificate_shas)):
            raise EvidenceV2Error("non-root RFC3161 certificates must be unique")


@dataclass(frozen=True)
class AnchorWindow:
    anchor_kind: str
    not_before: str | None
    not_after: str

    def __post_init__(self) -> None:
        if self.anchor_kind not in {"schedule_declaration", "prediction"}:
            raise EvidenceV2Error("anchor kind must be schedule_declaration or prediction")
        after = _canonical_utc(self.not_after, label="anchor.not_after")
        if self.anchor_kind == "prediction":
            if self.not_before is None:
                raise EvidenceV2Error("prediction anchor requires a lower time bound")
            before = _canonical_utc(self.not_before, label="anchor.not_before")
            if not before < after:
                raise EvidenceV2Error("prediction anchor window is empty")
        elif self.not_before is not None:
            raise EvidenceV2Error("schedule anchor must not carry a lower time bound")

    def validate_gen_time(self, gen_time: datetime) -> None:
        normalized = gen_time.astimezone(timezone.utc)
        upper = _canonical_utc(self.not_after, label="anchor.not_after")
        if not normalized < upper:
            raise EvidenceV2Error("RFC3161 genTime is not before the anchor upper bound")
        if self.not_before is not None:
            lower = _canonical_utc(self.not_before, label="anchor.not_before")
            if not lower < normalized:
                raise EvidenceV2Error("RFC3161 genTime is not after the anchor lower bound")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "anchor_kind": self.anchor_kind,
            "anchor_not_before": self.not_before,
            "anchor_not_after": self.not_after,
        }


def _normalize_revocation_refs(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise EvidenceV2Error("RFC3161 revocation refs must be a nonempty list")
    fields = {"certificate_ref", "issuer_certificate_ref", "crl_ref"}
    normalized: list[dict[str, Any]] = []
    certificate_shas: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != fields:
            raise EvidenceV2Error(f"RFC3161 revocation refs[{index}] fields mismatch")
        row = {
            field: EvidenceRef.from_dict(raw[field]).to_dict()
            for field in (
                "certificate_ref",
                "issuer_certificate_ref",
                "crl_ref",
            )
        }
        certificate_sha = row["certificate_ref"]["byte_sha256"]
        if certificate_sha in certificate_shas:
            raise EvidenceV2Error("RFC3161 revocation certificate refs must be unique")
        certificate_shas.add(certificate_sha)
        normalized.append(row)
    return normalized


def validate_timestamp_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "anchored_artifact_ref",
        "request_ref",
        "response_ref",
        "trust_anchor_ref",
        "untrusted_chain_ref",
        "revocation_refs",
        "policy_oid",
        "gen_time",
        "verified_at",
        "anchor_kind",
        "anchor_not_before",
        "anchor_not_after",
        "openssl_path",
        "openssl_binary_sha256",
        "response_projection_sha256",
        "verification_stdout_sha256",
        "data_verification_stdout_sha256",
        "warnings",
        "cryptographically_valid_at_gen_time",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != TIMESTAMP_RECEIPT_SCHEMA:
        raise EvidenceV2Error("RFC3161 validation receipt envelope mismatch")
    for field in (
        "anchored_artifact_ref",
        "request_ref",
        "response_ref",
        "trust_anchor_ref",
        "untrusted_chain_ref",
    ):
        EvidenceRef.from_dict(payload[field])
    payload["revocation_refs"] = _normalize_revocation_refs(payload["revocation_refs"])
    policy_oid = _policy_oid(payload["policy_oid"])
    gen_time = _canonical_utc(payload["gen_time"], label="receipt.gen_time")
    verified_at = _canonical_utc(payload["verified_at"], label="receipt.verified_at")
    if verified_at < gen_time:
        raise EvidenceV2Error("RFC3161 receipt verification predates genTime")
    window = AnchorWindow(
        anchor_kind=str(payload["anchor_kind"]),
        not_before=(
            None if payload["anchor_not_before"] is None else str(payload["anchor_not_before"])
        ),
        not_after=str(payload["anchor_not_after"]),
    )
    window.validate_gen_time(gen_time)
    if payload["openssl_path"] != PINNED_OPENSSL_PATH:
        raise EvidenceV2Error("RFC3161 validation receipt OpenSSL path drift")
    require_sha256(payload["openssl_binary_sha256"], label="receipt OpenSSL SHA")
    for field in (
        "response_projection_sha256",
        "verification_stdout_sha256",
        "data_verification_stdout_sha256",
    ):
        require_sha256(payload[field], label=f"receipt {field}")
    warnings = payload["warnings"]
    if (
        not isinstance(warnings, list)
        or warnings != sorted(set(str(item) for item in warnings))
        or any(not str(item) for item in warnings)
    ):
        raise EvidenceV2Error("RFC3161 validation receipt warnings are not canonical")
    if payload["cryptographically_valid_at_gen_time"] is not True:
        raise EvidenceV2Error("RFC3161 validation receipt is not cryptographically valid")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("RFC3161 validation receipt must be nonauthorizing")
    payload["policy_oid"] = policy_oid
    return payload


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: bytes
    stderr: bytes


CommandRunner = Callable[[Sequence[str], Path], CommandResult]


def _default_runner(command: Sequence[str], cwd: Path) -> CommandResult:
    environment = {
        **REQUIRED_ENVIRONMENT_CONTROLS,
        "HOME": "/nonexistent",
        "PATH": "/usr/bin:/bin",
    }
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvidenceV2Error("OpenSSL RFC3161 subprocess failed") from exc
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _write_private(path: Path, payload: bytes, *, mode: int = 0o600) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise EvidenceV2Error("private RFC3161 material write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _backend_signature(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _hash_backend(path: str) -> tuple[str, tuple[int, ...]]:
    if path != PINNED_OPENSSL_PATH:
        raise EvidenceV2Error("RFC3161 backend path is not pinned Homebrew OpenSSL")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceV2Error("pinned OpenSSL backend cannot be opened") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink < 1:
            raise EvidenceV2Error("pinned OpenSSL backend is not a regular file")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        signature_before = _backend_signature(before)
        signature_after = _backend_signature(after)
        if signature_before != signature_after:
            raise EvidenceV2Error("pinned OpenSSL backend changed during hashing")
        return digest.hexdigest(), signature_after
    finally:
        os.close(descriptor)


def _stage_verified_backend(
    *,
    source_path: str,
    destination: Path,
    expected_sha256: str,
    expected_signature: tuple[int, ...],
) -> str:
    """Copy the verified executable into the private root and execute that inode."""

    if source_path != PINNED_OPENSSL_PATH:
        raise EvidenceV2Error("RFC3161 backend path is not pinned Homebrew OpenSSL")
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source_path, source_flags)
    except OSError as exc:
        raise EvidenceV2Error("pinned OpenSSL backend cannot be staged") from exc
    destination_fd: int | None = None
    try:
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink < 1
            or _backend_signature(before) != expected_signature
        ):
            raise EvidenceV2Error("pinned OpenSSL backend changed before staging")
        destination_fd = os.open(destination, destination_flags, 0o700)
        digest = hashlib.sha256()
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            copied += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise EvidenceV2Error("OpenSSL backend staging made no progress")
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        staged = os.fstat(destination_fd)
        if (
            _backend_signature(after) != expected_signature
            or digest.hexdigest() != expected_sha256
            or not stat.S_ISREG(staged.st_mode)
            or stat.S_IMODE(staged.st_mode) != 0o700
            or staged.st_uid != os.getuid()
            or staged.st_nlink != 1
            or staged.st_size != copied
        ):
            raise EvidenceV2Error("staged OpenSSL backend identity mismatch")
    except OSError as exc:
        raise EvidenceV2Error("pinned OpenSSL backend staging failed") from exc
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)
    return str(destination)


def _run_checked(
    runner: CommandRunner,
    command: Sequence[str],
    cwd: Path,
    *,
    label: str,
) -> CommandResult:
    result = runner(command, cwd)
    if result.returncode != 0:
        stderr_sha = sha256_bytes(result.stderr)
        raise EvidenceV2Error(f"{label} failed:stderr_sha256={stderr_sha}")
    return result


def _parse_gen_time(text: str) -> datetime:
    values = [
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith("Time stamp:")
    ]
    if len(values) != 1:
        raise EvidenceV2Error("RFC3161 response must expose exactly one genTime")
    for format_string in _OPENSSL_TIME_FORMATS:
        try:
            return datetime.strptime(values[0], format_string).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise EvidenceV2Error("RFC3161 genTime text is not recognized")


def _parse_response_text(text: str, *, expected_policy_oid: str) -> datetime:
    statuses = [line for line in text.splitlines() if line.startswith("Status:")]
    if statuses != [_STATUS_GRANTED]:
        raise EvidenceV2Error("RFC3161 response status is not exactly Granted")
    policies = [
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.startswith("Policy OID:")
    ]
    if policies != [expected_policy_oid]:
        raise EvidenceV2Error("RFC3161 policy OID mismatch")
    return _parse_gen_time(text)


def _parse_certificate_serial(text: str) -> str:
    matches = re.findall(r"^serial=([0-9A-Fa-f]+)$", text, flags=re.MULTILINE)
    if len(matches) != 1:
        raise EvidenceV2Error("certificate serial projection is ambiguous")
    return matches[0].lstrip("0").upper() or "0"


def _parse_crl_revocations(text: str) -> dict[str, datetime]:
    lines = text.splitlines()
    revocations: dict[str, datetime] = {}
    for index, line in enumerate(lines):
        match = re.match(r"^\s*Serial Number:\s*([0-9A-Fa-f]+)\s*$", line)
        if match is None:
            continue
        if index + 1 >= len(lines):
            raise EvidenceV2Error("CRL serial lacks a revocation date")
        date_match = re.match(r"^\s*Revocation Date:\s*(.+?)\s*$", lines[index + 1])
        if date_match is None:
            raise EvidenceV2Error("CRL serial lacks an adjacent revocation date")
        parsed: datetime | None = None
        for format_string in _OPENSSL_TIME_FORMATS:
            try:
                parsed = datetime.strptime(date_match.group(1), format_string).replace(
                    tzinfo=timezone.utc
                )
                break
            except ValueError:
                continue
        if parsed is None:
            raise EvidenceV2Error("CRL revocation date text is not recognized")
        serial = match.group(1).lstrip("0").upper() or "0"
        if serial in revocations:
            raise EvidenceV2Error("CRL contains a duplicate serial")
        revocations[serial] = parsed
    return revocations


def _parse_crl_validity_window(text: str) -> tuple[datetime, datetime]:
    def extract(label: str) -> datetime:
        matches = [
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.strip().startswith(label + ":")
        ]
        if len(matches) != 1:
            raise EvidenceV2Error(f"CRL must expose exactly one {label}")
        for format_string in _OPENSSL_TIME_FORMATS:
            try:
                return datetime.strptime(matches[0], format_string).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        raise EvidenceV2Error(f"CRL {label} text is not recognized")

    last_update = extract("Last Update")
    next_update = extract("Next Update")
    if not last_update < next_update:
        raise EvidenceV2Error("CRL validity window is empty")
    return last_update, next_update


def verify_rfc3161_bundle(
    *,
    bundle: TimestampVerificationBundle,
    anchor_window: AnchorWindow,
    expected_policy_oid: str,
    openssl_binary_sha256: str,
    verification_time: str,
    openssl_path: str = PINNED_OPENSSL_PATH,
    runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Verify one already-persisted response without contacting a TSA."""

    policy_oid = _policy_oid(expected_policy_oid)
    verified_at = _canonical_utc(verification_time, label="verification_time")
    expected_backend_sha = require_sha256(
        openssl_binary_sha256,
        label="OpenSSL binary SHA",
    )
    backend_sha_before, backend_signature_before = _hash_backend(openssl_path)
    if backend_sha_before != expected_backend_sha:
        raise EvidenceV2Error("pinned OpenSSL backend byte SHA mismatch")
    command_runner = runner or _default_runner

    with tempfile.TemporaryDirectory(prefix="v16-rfc3161-") as temporary:
        root = Path(temporary)
        if stat.S_IMODE(root.stat().st_mode) != 0o700:
            raise EvidenceV2Error("RFC3161 temporary root is not mode 0700")
        command_path = _stage_verified_backend(
            source_path=openssl_path,
            destination=root / "openssl-verified",
            expected_sha256=expected_backend_sha,
            expected_signature=backend_signature_before,
        )
        material = {
            "anchored-artifact.bin": bundle.anchored_artifact.payload,
            "query.tsq": bundle.query.payload,
            "response.tsr": bundle.response.payload,
            "root.pem": bundle.trust_anchor.payload,
            "chain.pem": bundle.untrusted_chain.payload,
        }
        for index, binding in enumerate(bundle.revocations):
            material[f"certificate-{index}.pem"] = binding.certificate.payload
            material[f"issuer-{index}.pem"] = binding.issuer_certificate.payload
            material[f"crl-{index}.pem"] = binding.crl.payload
        for name, payload in material.items():
            _write_private(root / name, payload)

        response_text_result = _run_checked(
            command_runner,
            [command_path, "ts", "-reply", "-in", "response.tsr", "-text"],
            root,
            label="RFC3161 response projection",
        )
        try:
            response_text = response_text_result.stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise EvidenceV2Error("OpenSSL RFC3161 projection is not UTF-8") from exc
        gen_time = _parse_response_text(
            response_text,
            expected_policy_oid=policy_oid,
        )
        if verified_at < gen_time:
            raise EvidenceV2Error("RFC3161 verification time predates genTime")
        anchor_window.validate_gen_time(gen_time)

        verify_result = _run_checked(
            command_runner,
            [
                command_path,
                "ts",
                "-verify",
                "-queryfile",
                "query.tsq",
                "-in",
                "response.tsr",
                "-CAfile",
                "root.pem",
                "-untrusted",
                "chain.pem",
                "-purpose",
                "timestampsign",
                "-attime",
                str(int(gen_time.timestamp())),
                "-x509_strict",
                "-check_ss_sig",
            ],
            root,
            label="RFC3161 cryptographic verification at genTime",
        )
        data_verify_result = _run_checked(
            command_runner,
            [
                command_path,
                "ts",
                "-verify",
                "-data",
                "anchored-artifact.bin",
                "-in",
                "response.tsr",
                "-CAfile",
                "root.pem",
                "-untrusted",
                "chain.pem",
                "-purpose",
                "timestampsign",
                "-attime",
                str(int(gen_time.timestamp())),
                "-x509_strict",
                "-check_ss_sig",
            ],
            root,
            label="RFC3161 anchored-artifact verification at genTime",
        )

        warnings: list[str] = []
        for index, binding in enumerate(bundle.revocations):
            _run_checked(
                command_runner,
                [
                    command_path,
                    "verify",
                    "-CAfile",
                    f"issuer-{index}.pem",
                    "-partial_chain",
                    "-purpose",
                    "any",
                    "-no_check_time",
                    f"certificate-{index}.pem",
                ],
                root,
                label=f"RFC3161 certificate {index} issuer binding",
            )
            _run_checked(
                command_runner,
                [
                    command_path,
                    "crl",
                    "-in",
                    f"crl-{index}.pem",
                    "-noout",
                    "-verify",
                    "-CAfile",
                    f"issuer-{index}.pem",
                ],
                root,
                label=f"RFC3161 CRL {index} signature verification",
            )
            certificate_result = _run_checked(
                command_runner,
                [
                    command_path,
                    "x509",
                    "-in",
                    f"certificate-{index}.pem",
                    "-noout",
                    "-serial",
                ],
                root,
                label=f"RFC3161 certificate {index} serial projection",
            )
            crl_result = _run_checked(
                command_runner,
                [
                    command_path,
                    "crl",
                    "-in",
                    f"crl-{index}.pem",
                    "-noout",
                    "-text",
                ],
                root,
                label=f"RFC3161 CRL {index} projection",
            )
            serial = _parse_certificate_serial(certificate_result.stdout.decode("ascii"))
            crl_text = crl_result.stdout.decode("utf-8")
            last_update, next_update = _parse_crl_validity_window(crl_text)
            if not last_update <= verified_at <= next_update:
                raise EvidenceV2Error("RFC3161 CRL is not current at verification time")
            revocations = _parse_crl_revocations(crl_text)
            revoked_at = revocations.get(serial)
            if revoked_at is not None:
                if revoked_at <= gen_time:
                    raise EvidenceV2Error("RFC3161 certificate was revoked at or before genTime")
                warnings.append(
                    "certificate_revoked_after_gen_time:"
                    + binding.certificate.reference.byte_sha256
                )

    backend_sha_after, backend_signature_after = _hash_backend(openssl_path)
    if (
        backend_sha_after != expected_backend_sha
        or backend_signature_after != backend_signature_before
    ):
        raise EvidenceV2Error("pinned OpenSSL backend changed during verification")
    receipt = seal_semantic(
        {
            "schema_version": TIMESTAMP_RECEIPT_SCHEMA,
            "anchored_artifact_ref": bundle.anchored_artifact.reference.to_dict(),
            "request_ref": bundle.query.reference.to_dict(),
            "response_ref": bundle.response.reference.to_dict(),
            "trust_anchor_ref": bundle.trust_anchor.reference.to_dict(),
            "untrusted_chain_ref": bundle.untrusted_chain.reference.to_dict(),
            "revocation_refs": [
                {
                    "certificate_ref": binding.certificate.reference.to_dict(),
                    "issuer_certificate_ref": (binding.issuer_certificate.reference.to_dict()),
                    "crl_ref": binding.crl.reference.to_dict(),
                }
                for binding in bundle.revocations
            ],
            "policy_oid": policy_oid,
            "gen_time": _format_utc(gen_time),
            "verified_at": _format_utc(verified_at),
            "anchor_kind": anchor_window.anchor_kind,
            "anchor_not_before": anchor_window.not_before,
            "anchor_not_after": anchor_window.not_after,
            "openssl_path": openssl_path,
            "openssl_binary_sha256": expected_backend_sha,
            "response_projection_sha256": sha256_bytes(response_text_result.stdout),
            "verification_stdout_sha256": sha256_bytes(verify_result.stdout),
            "data_verification_stdout_sha256": sha256_bytes(data_verify_result.stdout),
            "warnings": sorted(warnings),
            "cryptographically_valid_at_gen_time": True,
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )
    return validate_timestamp_receipt(receipt)


@dataclass(frozen=True)
class PersistedTimestampResponse:
    absolute_path: str
    byte_sha256: str
    size: int


class TimestampPersistenceTerminalError(EvidenceV2Error):
    """Raised after a response file exists but persistence did not complete."""

    def __init__(self, message: str, persisted: PersistedTimestampResponse) -> None:
        super().__init__(message)
        self.persisted = persisted


def _canonical_response_directory(value: Any) -> str:
    text = str(value or "")
    if (
        not text.startswith("/")
        or "\x00" in text
        or os.path.normpath(text) != text
        or text.startswith("//")
        or text.endswith("/")
    ):
        raise EvidenceV2Error("RFC3161 response directory must be canonical and absolute")
    return text


def _expected_response_path(response_directory: str, anchor_id: str) -> str:
    return str(Path(response_directory) / f"{_safe_id(anchor_id, label='anchor_id')}.tsr")


def _validate_partial_response(value: Any, *, expected_path: str) -> dict[str, Any]:
    fields = {"absolute_path", "byte_sha256", "size"}
    if not isinstance(value, Mapping) or set(value) != fields:
        raise EvidenceV2Error("partial RFC3161 response fields mismatch")
    if value["absolute_path"] != expected_path:
        raise EvidenceV2Error("partial RFC3161 response path mismatch")
    require_sha256(value["byte_sha256"], label="partial response byte SHA")
    size = value["size"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or size > TIMESTAMP_RESPONSE_MAX_BYTES
    ):
        raise EvidenceV2Error("partial RFC3161 response size is invalid")
    return dict(value)


def _validate_attempt_envelope(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = validate_semantic_seal(value)
    fields = {
        "schema_version",
        "protocol_attempt_id",
        "anchor_id",
        "response_directory",
        "anchored_artifact_ref",
        "request_ref",
        "anchor_kind",
        "anchor_not_before",
        "anchor_not_after",
        "policy_oid",
        "openssl_path",
        "openssl_binary_sha256",
        "trust_anchor_ref",
        "untrusted_chain_ref",
        "revocation_refs",
        "state",
        "response_ref",
        "partial_response",
        "transport_failure_count",
        "validation_receipt_ref",
        "terminal_blockers",
        "activation_candidate",
        "new_risk_authorized",
        "production_apply_enabled",
        "semantic_sha256",
    }
    if set(payload) != fields or payload["schema_version"] != TIMESTAMP_ATTEMPT_SCHEMA:
        raise EvidenceV2Error("RFC3161 attempt state envelope mismatch")
    _safe_id(payload["protocol_attempt_id"], label="protocol_attempt_id")
    anchor_id = _safe_id(payload["anchor_id"], label="anchor_id")
    response_directory = _canonical_response_directory(payload["response_directory"])
    expected_response_path = _expected_response_path(response_directory, anchor_id)
    EvidenceRef.from_dict(payload["anchored_artifact_ref"])
    EvidenceRef.from_dict(payload["request_ref"])
    AnchorWindow(
        anchor_kind=str(payload["anchor_kind"]),
        not_before=(
            None if payload["anchor_not_before"] is None else str(payload["anchor_not_before"])
        ),
        not_after=str(payload["anchor_not_after"]),
    )
    _policy_oid(payload["policy_oid"])
    if payload["openssl_path"] != PINNED_OPENSSL_PATH:
        raise EvidenceV2Error("RFC3161 attempt OpenSSL path drift")
    require_sha256(payload["openssl_binary_sha256"], label="attempt OpenSSL SHA")
    EvidenceRef.from_dict(payload["trust_anchor_ref"])
    EvidenceRef.from_dict(payload["untrusted_chain_ref"])
    payload["revocation_refs"] = _normalize_revocation_refs(payload["revocation_refs"])
    if payload["state"] not in TIMESTAMP_STATES:
        raise EvidenceV2Error("RFC3161 attempt state is invalid")
    failures = payload["transport_failure_count"]
    if isinstance(failures, bool) or not isinstance(failures, int) or failures < 0:
        raise EvidenceV2Error("RFC3161 transport failure count is invalid")
    blockers = payload["terminal_blockers"]
    if not isinstance(blockers, list) or blockers != sorted(set(str(item) for item in blockers)):
        raise EvidenceV2Error("RFC3161 terminal blockers are not canonical")
    state = payload["state"]
    if state == "awaiting_transport":
        if (
            payload["response_ref"] is not None
            or payload["partial_response"] is not None
            or payload["validation_receipt_ref"] is not None
        ):
            raise EvidenceV2Error("awaiting RFC3161 attempt already carries a response")
        if blockers:
            raise EvidenceV2Error("awaiting RFC3161 attempt cannot carry terminal blockers")
    elif state == "response_persisted":
        response_ref = EvidenceRef.from_dict(payload["response_ref"])
        if response_ref.absolute_path != expected_response_path:
            raise EvidenceV2Error("persisted RFC3161 response path mismatch")
        if (
            payload["partial_response"] is not None
            or payload["validation_receipt_ref"] is not None
            or blockers
        ):
            raise EvidenceV2Error("persisted RFC3161 attempt has premature terminal data")
    elif state == "validated":
        response_ref = EvidenceRef.from_dict(payload["response_ref"])
        if response_ref.absolute_path != expected_response_path:
            raise EvidenceV2Error("validated RFC3161 response path mismatch")
        EvidenceRef.from_dict(payload["validation_receipt_ref"])
        if payload["partial_response"] is not None or blockers:
            raise EvidenceV2Error("validated RFC3161 attempt carries blockers")
    else:
        response_ref = payload["response_ref"]
        partial_response = payload["partial_response"]
        if response_ref is not None:
            normalized_ref = EvidenceRef.from_dict(response_ref)
            if normalized_ref.absolute_path != expected_response_path:
                raise EvidenceV2Error("failed RFC3161 response path mismatch")
            if partial_response is not None:
                raise EvidenceV2Error("failed RFC3161 attempt has two response identities")
        else:
            _validate_partial_response(
                partial_response,
                expected_path=expected_response_path,
            )
        if payload["validation_receipt_ref"] is not None or not blockers:
            raise EvidenceV2Error("failed RFC3161 attempt lacks canonical blockers")
    if any(
        payload[field] is not False
        for field in (
            "activation_candidate",
            "new_risk_authorized",
            "production_apply_enabled",
        )
    ):
        raise EvidenceV2Error("RFC3161 attempt state must be nonauthorizing")
    return payload


def build_timestamp_attempt(
    *,
    protocol_attempt_id: str,
    anchor_id: str,
    anchored_artifact_ref: EvidenceRef,
    request_ref: EvidenceRef,
    response_directory: str | Path,
    anchor_window: AnchorWindow,
    expected_policy_oid: str,
    openssl_binary_sha256: str,
    trust_anchor_ref: EvidenceRef,
    untrusted_chain_ref: EvidenceRef,
    revocations: Sequence[RevocationBinding],
) -> dict[str, Any]:
    if not isinstance(anchor_window, AnchorWindow):
        raise EvidenceV2Error("RFC3161 attempt requires an AnchorWindow")
    if not revocations:
        raise EvidenceV2Error("RFC3161 attempt requires CRL bindings")
    return seal_semantic(
        {
            "schema_version": TIMESTAMP_ATTEMPT_SCHEMA,
            "protocol_attempt_id": _safe_id(
                protocol_attempt_id,
                label="protocol_attempt_id",
            ),
            "anchor_id": _safe_id(anchor_id, label="anchor_id"),
            "response_directory": _canonical_response_directory(response_directory),
            "anchored_artifact_ref": anchored_artifact_ref.to_dict(),
            "request_ref": request_ref.to_dict(),
            **anchor_window.to_dict(),
            "policy_oid": _policy_oid(expected_policy_oid),
            "openssl_path": PINNED_OPENSSL_PATH,
            "openssl_binary_sha256": require_sha256(
                openssl_binary_sha256,
                label="OpenSSL binary SHA",
            ),
            "trust_anchor_ref": trust_anchor_ref.to_dict(),
            "untrusted_chain_ref": untrusted_chain_ref.to_dict(),
            "revocation_refs": [
                {
                    "certificate_ref": binding.certificate.reference.to_dict(),
                    "issuer_certificate_ref": (binding.issuer_certificate.reference.to_dict()),
                    "crl_ref": binding.crl.reference.to_dict(),
                }
                for binding in revocations
            ],
            "state": "awaiting_transport",
            "response_ref": None,
            "partial_response": None,
            "transport_failure_count": 0,
            "validation_receipt_ref": None,
            "terminal_blockers": [],
            "activation_candidate": False,
            "new_risk_authorized": False,
            "production_apply_enabled": False,
        }
    )


def record_transport_failure(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _validate_attempt_envelope(value)
    if payload["state"] != "awaiting_transport":
        raise EvidenceV2Error("transport retry is forbidden after response persistence")
    expected_path = _expected_response_path(
        payload["response_directory"],
        payload["anchor_id"],
    )
    if Path(expected_path).exists() or Path(expected_path).is_symlink():
        raise EvidenceV2Error("transport retry is forbidden after any response file exists")
    updated = dict(payload)
    updated.pop("semantic_sha256")
    updated["transport_failure_count"] += 1
    return seal_semantic(updated)


def record_persisted_response(
    value: Mapping[str, Any],
    *,
    response_ref: EvidenceRef,
) -> dict[str, Any]:
    payload = _validate_attempt_envelope(value)
    if payload["state"] != "awaiting_transport":
        raise EvidenceV2Error("the first persisted RFC3161 response already won")
    expected_path = _expected_response_path(
        payload["response_directory"],
        payload["anchor_id"],
    )
    if response_ref.absolute_path != expected_path:
        raise EvidenceV2Error("persisted RFC3161 response ref path mismatch")
    updated = dict(payload)
    updated.pop("semantic_sha256")
    updated["state"] = "response_persisted"
    updated["response_ref"] = response_ref.to_dict()
    return seal_semantic(updated)


def record_partial_response_failure(
    value: Mapping[str, Any],
    *,
    persisted: PersistedTimestampResponse,
) -> dict[str, Any]:
    payload = _validate_attempt_envelope(value)
    if payload["state"] != "awaiting_transport":
        raise EvidenceV2Error("partial response failure requires an awaiting attempt")
    expected_path = _expected_response_path(
        payload["response_directory"],
        payload["anchor_id"],
    )
    partial = {
        "absolute_path": persisted.absolute_path,
        "byte_sha256": persisted.byte_sha256,
        "size": persisted.size,
    }
    _validate_partial_response(partial, expected_path=expected_path)
    try:
        descriptor = os.open(
            expected_path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        )
    except OSError as exc:
        raise EvidenceV2Error("partial RFC3161 response cannot be read back") from exc
    try:
        metadata = os.fstat(descriptor)
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_uid != os.getuid()
            or metadata.st_nlink != 1
            or metadata.st_size != persisted.size
            or digest.hexdigest() != persisted.byte_sha256
        ):
            raise EvidenceV2Error("partial RFC3161 response readback mismatch")
    finally:
        os.close(descriptor)
    updated = dict(payload)
    updated.pop("semantic_sha256")
    updated["state"] = "failed_terminal"
    updated["partial_response"] = partial
    updated["terminal_blockers"] = ["rfc3161_partial_response_persisted"]
    return seal_semantic(updated)


def _validate_receipt_against_attempt(
    attempt: Mapping[str, Any],
    validation_receipt: BoundCanonicalArtifact,
) -> dict[str, Any]:
    receipt = validate_timestamp_receipt(validation_receipt.read())
    expected = {
        "anchored_artifact_ref": attempt["anchored_artifact_ref"],
        "request_ref": attempt["request_ref"],
        "response_ref": attempt["response_ref"],
        "trust_anchor_ref": attempt["trust_anchor_ref"],
        "untrusted_chain_ref": attempt["untrusted_chain_ref"],
        "revocation_refs": attempt["revocation_refs"],
        "policy_oid": attempt["policy_oid"],
        "anchor_kind": attempt["anchor_kind"],
        "anchor_not_before": attempt["anchor_not_before"],
        "anchor_not_after": attempt["anchor_not_after"],
        "openssl_path": attempt["openssl_path"],
        "openssl_binary_sha256": attempt["openssl_binary_sha256"],
    }
    mismatches = [field for field, value in expected.items() if receipt[field] != value]
    if mismatches:
        raise EvidenceV2Error(
            "RFC3161 validation receipt drifts from attempt: " + ",".join(mismatches)
        )
    return receipt


def _validate_bundle_against_attempt(
    attempt: Mapping[str, Any],
    bundle: TimestampVerificationBundle,
) -> None:
    if not isinstance(bundle, TimestampVerificationBundle):
        raise EvidenceV2Error("RFC3161 validation requires a bound verification bundle")
    expected = {
        "anchored_artifact_ref": bundle.anchored_artifact.reference.to_dict(),
        "request_ref": bundle.query.reference.to_dict(),
        "response_ref": bundle.response.reference.to_dict(),
        "trust_anchor_ref": bundle.trust_anchor.reference.to_dict(),
        "untrusted_chain_ref": bundle.untrusted_chain.reference.to_dict(),
        "revocation_refs": [
            {
                "certificate_ref": binding.certificate.reference.to_dict(),
                "issuer_certificate_ref": binding.issuer_certificate.reference.to_dict(),
                "crl_ref": binding.crl.reference.to_dict(),
            }
            for binding in bundle.revocations
        ],
    }
    mismatches = [field for field, value in expected.items() if attempt[field] != value]
    if mismatches:
        raise EvidenceV2Error(
            "RFC3161 verification bundle drifts from attempt: " + ",".join(mismatches)
        )


@dataclass(frozen=True)
class TimestampValidationResult:
    attempt: dict[str, Any]
    validation_receipt: BoundCanonicalArtifact


def verify_and_record_timestamp_validation(
    value: Mapping[str, Any],
    *,
    bundle: TimestampVerificationBundle,
    verification_time: str,
    validation_receipt_path: str,
) -> TimestampValidationResult:
    """Cryptographically verify the frozen bundle before recording success."""

    payload = _validate_attempt_envelope(value)
    if payload["state"] != "response_persisted":
        raise EvidenceV2Error("RFC3161 validation requires the canonical persisted response")
    _validate_bundle_against_attempt(payload, bundle)
    receipt_payload = verify_rfc3161_bundle(
        bundle=bundle,
        anchor_window=AnchorWindow(
            anchor_kind=str(payload["anchor_kind"]),
            not_before=(
                None if payload["anchor_not_before"] is None else str(payload["anchor_not_before"])
            ),
            not_after=str(payload["anchor_not_after"]),
        ),
        expected_policy_oid=str(payload["policy_oid"]),
        openssl_binary_sha256=str(payload["openssl_binary_sha256"]),
        verification_time=verification_time,
        openssl_path=str(payload["openssl_path"]),
    )
    receipt_bytes = canonical_json_bytes(receipt_payload)
    receipt_ref = EvidenceRef(
        schema_version=EVIDENCE_REF_SCHEMA,
        artifact_schema=TIMESTAMP_RECEIPT_SCHEMA,
        absolute_path=str(validation_receipt_path),
        byte_sha256=sha256_bytes(receipt_bytes),
        semantic_sha256=semantic_sha256(receipt_payload),
        root_policy="v16.private-evidence-root.v2",
    )
    receipt = BoundCanonicalArtifact(reference=receipt_ref, payload=receipt_bytes)
    _validate_receipt_against_attempt(payload, receipt)
    updated = dict(payload)
    updated.pop("semantic_sha256")
    updated["state"] = "validated"
    updated["validation_receipt_ref"] = receipt.reference.to_dict()
    return TimestampValidationResult(
        attempt=seal_semantic(updated),
        validation_receipt=receipt,
    )


def record_timestamp_validation(
    value: Mapping[str, Any],
    *,
    valid: bool,
    validation_receipt: BoundCanonicalArtifact | None = None,
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    payload = _validate_attempt_envelope(value)
    if payload["state"] != "response_persisted":
        raise EvidenceV2Error("RFC3161 validation requires the canonical persisted response")
    normalized_blockers = sorted(set(str(item) for item in blockers if str(item)))
    if valid:
        raise EvidenceV2Error(
            "caller-declared RFC3161 success is forbidden; use "
            "verify_and_record_timestamp_validation"
        )
    if validation_receipt is not None or not normalized_blockers:
        raise EvidenceV2Error("invalid RFC3161 response requires exact terminal blockers")
    updated = dict(payload)
    updated.pop("semantic_sha256")
    updated["state"] = "failed_terminal"
    updated["terminal_blockers"] = normalized_blockers
    return seal_semantic(updated)


def validate_timestamp_attempt(
    value: Mapping[str, Any],
    *,
    validation_receipt: BoundCanonicalArtifact | None = None,
) -> dict[str, Any]:
    payload = _validate_attempt_envelope(value)
    if payload["state"] == "validated":
        if validation_receipt is None:
            raise EvidenceV2Error("validated RFC3161 attempt requires its bound receipt")
        if payload["validation_receipt_ref"] != validation_receipt.reference.to_dict():
            raise EvidenceV2Error("validated RFC3161 attempt receipt ref mismatch")
        _validate_receipt_against_attempt(payload, validation_receipt)
    elif validation_receipt is not None:
        raise EvidenceV2Error("non-validated RFC3161 attempt cannot carry a receipt")
    return payload


@dataclass(frozen=True)
class TimestampAnchorBinding:
    attempt: BoundCanonicalArtifact
    validation_receipt: BoundCanonicalArtifact
    verification_bundle: TimestampVerificationBundle

    def read(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if any(
            artifact.reference.root_policy != "v16.private-evidence-root.v2"
            for artifact in (self.attempt, self.validation_receipt)
        ):
            raise EvidenceV2Error("RFC3161 state and receipt must use the private root")
        attempt = validate_timestamp_attempt(
            self.attempt.read(),
            validation_receipt=self.validation_receipt,
        )
        receipt = validate_timestamp_receipt(self.validation_receipt.read())
        _validate_bundle_against_attempt(attempt, self.verification_bundle)
        recomputed = verify_rfc3161_bundle(
            bundle=self.verification_bundle,
            anchor_window=AnchorWindow(
                anchor_kind=str(attempt["anchor_kind"]),
                not_before=(
                    None
                    if attempt["anchor_not_before"] is None
                    else str(attempt["anchor_not_before"])
                ),
                not_after=str(attempt["anchor_not_after"]),
            ),
            expected_policy_oid=str(attempt["policy_oid"]),
            openssl_binary_sha256=str(attempt["openssl_binary_sha256"]),
            verification_time=str(receipt["verified_at"]),
            openssl_path=str(attempt["openssl_path"]),
        )
        if recomputed != receipt:
            raise EvidenceV2Error(
                "RFC3161 receipt does not match independent cryptographic revalidation"
            )
        return attempt, receipt


def persist_first_timestamp_response(
    *,
    private_directory: str | Path,
    anchor_id: str,
    response: bytes,
) -> PersistedTimestampResponse:
    """Persist the canonical response with O_EXCL; partial writes remain terminal."""

    identifier = _safe_id(anchor_id, label="anchor_id")
    if not response or len(response) > TIMESTAMP_RESPONSE_MAX_BYTES:
        raise EvidenceV2Error("RFC3161 transport body is empty or exceeds its bound")
    directory = Path(private_directory)
    if not directory.is_absolute():
        raise EvidenceV2Error("RFC3161 private directory must be absolute")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        directory_fd = os.open(directory, flags)
    except OSError as exc:
        raise EvidenceV2Error("RFC3161 private directory open failed") from exc
    filename = f"{identifier}.tsr"
    descriptor: int | None = None
    try:
        metadata = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o700
            or metadata.st_uid != os.getuid()
        ):
            raise EvidenceV2Error("RFC3161 private directory ownership/mode mismatch")
        try:
            descriptor = os.open(
                filename,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
        except FileExistsError as exc:
            raise EvidenceV2Error("the first persisted RFC3161 response already exists") from exc
        except OSError as exc:
            raise EvidenceV2Error("RFC3161 response CAS creation failed") from exc
        digest = hashlib.sha256()
        total = 0
        try:
            view = memoryview(response)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("RFC3161 response persistence made no progress")
                digest.update(view[:written])
                total += written
                view = view[written:]
            os.fsync(descriptor)
            persisted = os.fstat(descriptor)
            if (
                not stat.S_ISREG(persisted.st_mode)
                or stat.S_IMODE(persisted.st_mode) != 0o600
                or persisted.st_uid != os.getuid()
                or persisted.st_nlink != 1
                or persisted.st_size != len(response)
            ):
                raise OSError("persisted RFC3161 response metadata mismatch")
            os.fsync(directory_fd)
        except Exception as exc:
            try:
                os.fsync(descriptor)
                os.fsync(directory_fd)
            except OSError:
                pass
            partial = PersistedTimestampResponse(
                absolute_path=str(directory / filename),
                byte_sha256=digest.hexdigest(),
                size=total,
            )
            raise TimestampPersistenceTerminalError(
                "RFC3161 response file exists but persistence is terminally incomplete",
                partial,
            ) from exc
        return PersistedTimestampResponse(
            absolute_path=str(directory / filename),
            byte_sha256=digest.hexdigest(),
            size=total,
        )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_fd)


__all__ = [
    "AnchorWindow",
    "BoundArtifact",
    "CommandResult",
    "PersistedTimestampResponse",
    "RevocationBinding",
    "TIMESTAMP_ATTEMPT_SCHEMA",
    "TIMESTAMP_RECEIPT_SCHEMA",
    "TimestampAnchorBinding",
    "TimestampValidationResult",
    "TimestampVerificationBundle",
    "TimestampPersistenceTerminalError",
    "build_timestamp_attempt",
    "persist_first_timestamp_response",
    "record_persisted_response",
    "record_partial_response_failure",
    "record_timestamp_validation",
    "record_transport_failure",
    "validate_timestamp_attempt",
    "validate_timestamp_receipt",
    "verify_and_record_timestamp_validation",
    "verify_rfc3161_bundle",
]
