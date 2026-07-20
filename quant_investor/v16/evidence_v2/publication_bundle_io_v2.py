"""Exclusive private publisher for a nonauthorizing publication bundle."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import stat
from collections.abc import Mapping
from typing import Any

from .candidate_report_source_v2 import (
    CandidateReportSourceEvidenceBundleV2,
    build_candidate_report_source_v2,
)
from .codex_authority_plan_v2 import PRIVATE_ROOT_POLICY, READINESS_V4_SCHEMA
from .contracts import (
    EVIDENCE_REF_SCHEMA,
    BoundCanonicalArtifact,
    EvidenceRef,
    EvidenceV2Error,
    canonical_json_bytes,
    semantic_sha256,
    sha256_bytes,
)
from .dashboard_source_v2 import (
    DashboardReportEvidenceBundleV2,
    DashboardSnapshotEvidenceBundleV2,
    build_dashboard_snapshot_v2,
    build_dashboard_source_status_v2,
)
from .publication_aggregate_v2 import (
    PublicationAggregateEvidenceBundleV2,
    build_publication_aggregate_v2,
    validate_publication_aggregate_v2,
)
from .publication_plan_v2 import (
    CANDIDATE_REPORT_SCHEMA,
    DASHBOARD_SNAPSHOT_SCHEMA,
    DASHBOARD_SOURCE_STATUS_SCHEMA,
    PUBLICATION_AGGREGATE_SCHEMA,
    PUBLICATION_PLAN_SCHEMA,
    PublicationPlanEvidenceBundleV2,
    validate_publication_source_plan_v2,
)
from .readiness_v4 import ReadinessEvidenceBundleV4
from .secure_io import platform_acl_absent

PRIVATE_DIR_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
PUBLICATION_WRITE_ORDER = (
    "publication_plan",
    "candidate_report",
    "dashboard_snapshot",
    "dashboard_source_status",
    "publication_aggregate",
)


@dataclass(frozen=True)
class PublishedPublicationBundleV2:
    publication_plan: BoundCanonicalArtifact
    candidate_report: BoundCanonicalArtifact
    dashboard_snapshot: BoundCanonicalArtifact
    dashboard_source_status: BoundCanonicalArtifact
    publication_aggregate: BoundCanonicalArtifact

    def references(self) -> dict[str, dict[str, str]]:
        return {
            "publication_plan": self.publication_plan.reference.to_dict(),
            "candidate_report": self.candidate_report.reference.to_dict(),
            "dashboard_snapshot": self.dashboard_snapshot.reference.to_dict(),
            "dashboard_source_status": self.dashboard_source_status.reference.to_dict(),
            "publication_aggregate": self.publication_aggregate.reference.to_dict(),
        }


@dataclass(frozen=True)
class _PinnedRoot:
    path: str
    descriptor: int
    device: int
    inode: int
    uid: int
    mode: int


def _require_acl_absent(descriptor: int, *, label: str) -> None:
    try:
        absent = platform_acl_absent(descriptor, label)
    except EvidenceV2Error:
        raise
    except Exception as exc:
        raise EvidenceV2Error(f"{label} ACL verification failed") from exc
    if absent is not True:
        raise EvidenceV2Error(f"{label} has an extended ACL")


def _bound(
    *,
    path: str,
    schema: str,
    payload: Mapping[str, Any],
) -> BoundCanonicalArtifact:
    raw = canonical_json_bytes(payload)
    return BoundCanonicalArtifact(
        reference=EvidenceRef(
            schema_version=EVIDENCE_REF_SCHEMA,
            artifact_schema=schema,
            absolute_path=path,
            byte_sha256=sha256_bytes(raw),
            semantic_sha256=semantic_sha256(payload),
            root_policy=PRIVATE_ROOT_POLICY,
        ),
        payload=raw,
    )


def _pin_private_root(path: str) -> _PinnedRoot:
    target = Path(path)
    try:
        resolved = str(target.resolve(strict=True))
    except OSError as exc:
        raise EvidenceV2Error("publication private root is missing") from exc
    if resolved != path:
        raise EvidenceV2Error("publication private root contains symlink indirection")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise EvidenceV2Error("publication private root cannot be pinned") from exc
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != PRIVATE_DIR_MODE
            or metadata.st_uid != os.getuid()
        ):
            raise EvidenceV2Error(
                "publication private root must be owner-owned mode 0700"
            )
        _require_acl_absent(descriptor, label="publication private root")
        return _PinnedRoot(
            path=path,
            descriptor=descriptor,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            uid=metadata.st_uid,
            mode=stat.S_IMODE(metadata.st_mode),
        )
    except Exception:
        os.close(descriptor)
        raise


def _assert_root_identity(root: _PinnedRoot) -> None:
    try:
        descriptor_metadata = os.fstat(root.descriptor)
        path_metadata = os.stat(root.path, follow_symlinks=False)
    except OSError as exc:
        raise EvidenceV2Error("publication private root identity became unavailable") from exc
    expected = (root.device, root.inode, root.uid, root.mode)
    for metadata in (descriptor_metadata, path_metadata):
        actual = (
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_uid,
            stat.S_IMODE(metadata.st_mode),
        )
        if not stat.S_ISDIR(metadata.st_mode) or actual != expected:
            raise EvidenceV2Error("publication private root identity changed")
    _require_acl_absent(root.descriptor, label="publication private root")


def _basename(path: str, *, root: _PinnedRoot) -> str:
    target = Path(path)
    if str(target.parent) != root.path or target.name in {"", ".", ".."}:
        raise EvidenceV2Error("publication output is not a direct root child")
    return target.name


def _preflight_absent(
    root: _PinnedRoot,
    artifacts: tuple[BoundCanonicalArtifact, ...],
) -> None:
    _assert_root_identity(root)
    names = [_basename(item.reference.absolute_path, root=root) for item in artifacts]
    if len(names) != len(set(names)):
        raise EvidenceV2Error("publication output names are duplicated")
    for name in names:
        try:
            os.stat(name, dir_fd=root.descriptor, follow_symlinks=False)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise EvidenceV2Error("publication output absence check failed") from exc
        raise EvidenceV2Error(f"publication output already exists: {name}")


def _stable_file_signature(metadata: os.stat_result) -> tuple[int, ...]:
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


def _readback(root: _PinnedRoot, artifact: BoundCanonicalArtifact) -> None:
    name = _basename(artifact.reference.absolute_path, root=root)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=root.descriptor)
    except OSError as exc:
        raise EvidenceV2Error("publication output readback open failed") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != PRIVATE_FILE_MODE
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or before.st_size != len(artifact.payload)
        ):
            raise EvidenceV2Error("publication output metadata readback mismatch")
        _require_acl_absent(descriptor, label=f"publication output {name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if _stable_file_signature(before) != _stable_file_signature(after):
            raise EvidenceV2Error("publication output changed during readback")
        if b"".join(chunks) != artifact.payload:
            raise EvidenceV2Error("publication output bytes readback mismatch")
    finally:
        os.close(descriptor)
    _assert_root_identity(root)


def _write_exclusive(root: _PinnedRoot, artifact: BoundCanonicalArtifact) -> None:
    _assert_root_identity(root)
    name = _basename(artifact.reference.absolute_path, root=root)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(
            name,
            flags,
            PRIVATE_FILE_MODE,
            dir_fd=root.descriptor,
        )
    except OSError as exc:
        raise EvidenceV2Error(f"publication exclusive creation failed: {name}") from exc
    try:
        os.fchmod(descriptor, PRIVATE_FILE_MODE)
        view = memoryview(artifact.payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise EvidenceV2Error("publication output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(root.descriptor)
    _readback(root, artifact)


def _build_bundle(
    *,
    plan_payload: Mapping[str, Any],
    readiness_v4: BoundCanonicalArtifact,
    readiness_evidence: ReadinessEvidenceBundleV4,
) -> PublishedPublicationBundleV2:
    plan = validate_publication_source_plan_v2(plan_payload)
    if (
        not isinstance(readiness_v4, BoundCanonicalArtifact)
        or readiness_v4.reference.artifact_schema != READINESS_V4_SCHEMA
        or plan["readiness_v4_ref"] != readiness_v4.reference.to_dict()
        or not isinstance(readiness_evidence, ReadinessEvidenceBundleV4)
    ):
        raise EvidenceV2Error("publication publisher readiness-v4 input is invalid")
    planned = plan["planned_artifacts"]
    plan_artifact = _bound(
        path=str(plan["plan_absolute_path"]),
        schema=PUBLICATION_PLAN_SCHEMA,
        payload=plan,
    )
    plan_evidence = PublicationPlanEvidenceBundleV2(
        plan=plan_artifact,
        readiness_v4=readiness_v4,
        readiness_evidence=readiness_evidence,
    )
    report_evidence = CandidateReportSourceEvidenceBundleV2(
        publication_plan=plan_evidence
    )
    report_payload = build_candidate_report_source_v2(evidence=report_evidence)
    report = _bound(
        path=str(planned["candidate_report"]["absolute_path"]),
        schema=CANDIDATE_REPORT_SCHEMA,
        payload=report_payload,
    )
    dashboard_report_evidence = DashboardReportEvidenceBundleV2(
        publication_plan=plan_evidence,
        candidate_report=report,
        report_evidence=report_evidence,
    )
    snapshot_payload = build_dashboard_snapshot_v2(evidence=dashboard_report_evidence)
    snapshot = _bound(
        path=str(planned["dashboard_snapshot"]["absolute_path"]),
        schema=DASHBOARD_SNAPSHOT_SCHEMA,
        payload=snapshot_payload,
    )
    snapshot_evidence = DashboardSnapshotEvidenceBundleV2(
        report=dashboard_report_evidence,
        snapshot=snapshot,
    )
    status_payload = build_dashboard_source_status_v2(evidence=snapshot_evidence)
    status = _bound(
        path=str(planned["dashboard_source_status"]["absolute_path"]),
        schema=DASHBOARD_SOURCE_STATUS_SCHEMA,
        payload=status_payload,
    )
    aggregate_evidence = PublicationAggregateEvidenceBundleV2(
        publication_plan=plan_evidence,
        candidate_report=report,
        dashboard_snapshot=snapshot,
        dashboard_source_status=status,
        dashboard_evidence=snapshot_evidence,
    )
    aggregate_payload = build_publication_aggregate_v2(evidence=aggregate_evidence)
    aggregate = _bound(
        path=str(planned["publication_aggregate"]["absolute_path"]),
        schema=PUBLICATION_AGGREGATE_SCHEMA,
        payload=aggregate_payload,
    )
    validate_publication_aggregate_v2(
        aggregate.read(),
        evidence=aggregate_evidence,
    )
    return PublishedPublicationBundleV2(
        publication_plan=plan_artifact,
        candidate_report=report,
        dashboard_snapshot=snapshot,
        dashboard_source_status=status,
        publication_aggregate=aggregate,
    )


def publish_publication_bundle_v2(
    *,
    plan_payload: Mapping[str, Any],
    readiness_v4: BoundCanonicalArtifact,
    readiness_evidence: ReadinessEvidenceBundleV4,
) -> PublishedPublicationBundleV2:
    """Publish one private diagnostic bundle; aggregate is always written last."""

    bundle = _build_bundle(
        plan_payload=plan_payload,
        readiness_v4=readiness_v4,
        readiness_evidence=readiness_evidence,
    )
    root = _pin_private_root(
        validate_publication_source_plan_v2(plan_payload)["private_root"]
    )
    artifacts = (
        bundle.publication_plan,
        bundle.candidate_report,
        bundle.dashboard_snapshot,
        bundle.dashboard_source_status,
        bundle.publication_aggregate,
    )
    try:
        if tuple(bundle.references()) != PUBLICATION_WRITE_ORDER:
            raise EvidenceV2Error("publication writer order drift")
        _preflight_absent(root, artifacts)
        for index, artifact in enumerate(artifacts):
            if index == len(artifacts) - 1:
                for prior in artifacts[:-1]:
                    _readback(root, prior)
            _write_exclusive(root, artifact)
        _assert_root_identity(root)
        os.fsync(root.descriptor)
        return bundle
    finally:
        os.close(root.descriptor)


__all__ = [
    "PRIVATE_DIR_MODE",
    "PRIVATE_FILE_MODE",
    "PUBLICATION_WRITE_ORDER",
    "PublishedPublicationBundleV2",
    "publish_publication_bundle_v2",
]
