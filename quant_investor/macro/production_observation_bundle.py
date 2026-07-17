"""Atomic production publication for the complete CN Macro observation scope.

The module composes two already fail-closed, offline compilers:

* 36 observations recompiled from a persisted official-web bundle; and
* three ``market.breadth`` observations compiled from three explicitly bound
  market-date snapshots and their exact coverage/scope contracts.

Publication is a single append-only observation-store CAS.  Exact source
entities are copied into the v2 evidence sidecar before the source paths are
used no further.  The readback is then rebuilt into the same Macro snapshot
that was validated before publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit

from quant_investor.macro.contracts import (
    MacroObservation,
    UTC,
    canonical_hash,
    parse_timestamp,
    published_cutoff,
)
from quant_investor.macro.local_market_observations import (
    LOCAL_MARKET_BREADTH_EVIDENCE_SCHEMA,
    LOCAL_MARKET_BREADTH_FORMULA_SHA256,
    compile_local_market_breadth_observation,
)
from quant_investor.macro.official_web_compiler import (
    NBS_NATIONAL_ECONOMY_PARSER,
    NBS_OFFICIAL_PMI_PARSER,
    NBS_QUARTERLY_GDP_PARSER,
    OfficialWebCompilationResult,
    PARSER_CONTRACT_SHA256,
    PBC_MONEY_STOCK_PARSER,
    recompile_official_web_bundle,
)
from quant_investor.macro.snapshot import build_macro_snapshot
from quant_investor.macro.store import (
    load_observations,
    pointer_sha256,
    publish_observations,
)


PRODUCTION_OBSERVATION_BUNDLE_SCHEMA = (
    "macro-production-observation-bundle.v1"
)
LOCAL_MARKET_OBSERVATION_PUBLICATION_SCHEMA = (
    "macro-local-market-observation-publication.v1"
)
LOCAL_BREADTH_BOOTSTRAP_PLAN_SCHEMA = (
    "cn-local-breadth-bootstrap-plan.v1"
)
_OFFICIAL_INDICATOR_COUNT = 12
_PRODUCTION_INDICATOR_COUNT = 13
_HISTORY_LENGTH = 3
_PRODUCTION_OBSERVATION_COUNT = 39
_EXPECTED_NATIONAL_COVERAGE = 0.8125
_MAX_JSON_BYTES = 4 * 1024 * 1024
_MAX_HTML_BYTES = 4 * 1024 * 1024
_MAX_PART_BYTES = 512 * 1024 * 1024
_MAX_SCOPE_BYTES = 16 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMPACT_DATE_RE = re.compile(r"^20\d{6}$")
_PAGE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,79}$")
_GENERATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_PRODUCTION_CHAIN_SCHEMAS = frozenset(
    {
        PRODUCTION_OBSERVATION_BUNDLE_SCHEMA,
        LOCAL_MARKET_OBSERVATION_PUBLICATION_SCHEMA,
    }
)
_GENERATION_V2_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "generation_id",
        "row_count",
        "parquet_sha256",
        "content_set_hash",
        "created_at",
        "parent_generation_id",
        "parent_pointer_sha256",
        "added_content_hashes",
        "min_available_at",
        "max_available_at",
        "observer_only",
        "production_eligible",
        "applied",
        "metadata",
        "evidence_file_count",
        "evidence_files",
        "evidence_set_sha256",
        "observation_evidence",
    }
)
_BOOTSTRAP_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "market",
        "as_of",
        "decision_cutoff_at",
        "official_bundle_manifest_sha256",
        "official_plan_sha256",
        "local_snapshot_manifest_sha256",
        "local_bootstrap_plan_sha256",
        "local_coverage_contract_sha256",
        "local_effective_available_at",
        "validated_snapshot_hash",
        "atomic_combined_publication",
    }
)
_UPDATE_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "market",
        "as_of",
        "decision_cutoff_at",
        "parent_as_of",
        "parent_decision_cutoff_at",
        "update_mode",
        "local_snapshot_manifest_sha256",
        "local_coverage_manifest_sha256",
        "local_scope_artifact_sha256",
        "local_target_trade_date",
        "local_coverage_contract_sha256",
        "local_effective_available_at",
        "validated_snapshot_hash",
    }
)
_EVIDENCE_FILE_KEYS = frozenset(
    {"path", "sha256", "size_bytes", "metadata", "metadata_sha256"}
)
_OFFICIAL_EVIDENCE_METADATA_KEYS = frozenset(
    {
        "extension",
        "evidence_kind",
        "page_id",
        "parser_id",
        "parser_contract_sha256",
        "source_system",
        "source_url",
        "source_record_id",
        "period",
        "release_at",
        "official_bundle_manifest_sha256",
        "support_only",
        "size_bytes",
    }
)
_LOCAL_INPUT_METADATA_KEYS = frozenset(
    {"extension", "evidence_kind", "size_bytes"}
)
_LOCAL_OBSERVATION_EVIDENCE_METADATA_KEYS = frozenset(
    {
        "extension",
        "evidence_kind",
        "schema_version",
        "target_trade_date",
        "evidence_semantic_sha256",
        "coverage_contract_sha256",
        "effective_available_at",
        "size_bytes",
    }
)
_OFFICIAL_SOURCE_BY_INDICATOR = {
    "cn.cpi_yoy": "nbs_official",
    "cn.exports_yoy": "nbs_official",
    "cn.fixed_asset_investment_yoy": "nbs_official",
    "cn.gdp_yoy": "nbs_official",
    "cn.imports_yoy": "nbs_official",
    "cn.industrial_value_added_yoy": "nbs_official",
    "cn.m1_yoy": "pboc_official",
    "cn.m2_yoy": "pboc_official",
    "cn.pmi_manufacturing": "nbs_official",
    "cn.ppi_yoy": "nbs_official",
    "cn.property_investment_yoy": "nbs_official",
    "cn.retail_sales_yoy": "nbs_official",
}
_OFFICIAL_HOST_BY_SOURCE = {
    "nbs_official": "www.stats.gov.cn",
    "pbc_official": "www.pbc.gov.cn",
}
_PARSER_SOURCE_SYSTEM = {
    NBS_NATIONAL_ECONOMY_PARSER: "nbs_official",
    NBS_OFFICIAL_PMI_PARSER: "nbs_official",
    NBS_QUARTERLY_GDP_PARSER: "nbs_official",
    PBC_MONEY_STOCK_PARSER: "pbc_official",
}
_EVIDENCE_SOURCE_BY_OBSERVATION_SOURCE = {
    "nbs_official": "nbs_official",
    "pboc_official": "pbc_official",
}


class ProductionObservationBundleError(RuntimeError):
    """Raised when a production observation publication fails closed."""


@dataclass(frozen=True)
class _EvidenceInputs:
    bodies: Mapping[str, bytes]
    metadata: Mapping[str, Mapping[str, Any]]
    observation_mapping: Mapping[str, Sequence[str]]
    official_file_count: int = 0
    local_file_count: int = 0


@dataclass(frozen=True)
class _LocalTargetBinding:
    target_trade_date: str
    snapshot_manifest_path: Path
    expected_snapshot_manifest_sha256: str
    coverage_manifest_path: Path
    expected_coverage_manifest_sha256: str
    scope_artifact_path: Path
    expected_scope_artifact_sha256: str


@dataclass(frozen=True)
class _ExistingProductionChain:
    observation_mapping: Mapping[str, Sequence[str]]
    logical_as_of: str
    decision_cutoff_at: datetime
    generation_id: str


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionObservationBundleError(
            "production_local_evidence_not_json_safe"
        ) from exc


def _required_sha256(
    value: Any,
    *,
    blocker: str,
    allow_empty: bool = False,
) -> str:
    text = str(value if value is not None else "").strip().lower()
    if allow_empty and not text:
        return ""
    if not _SHA256_RE.fullmatch(text):
        raise ProductionObservationBundleError(blocker)
    return text


def _stat_signature(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _absolute_path(value: str | Path, *, blocker: str) -> Path:
    raw = Path(value).expanduser()
    if not raw.parts or ".." in raw.parts:
        raise ProductionObservationBundleError(blocker)
    return Path(os.path.abspath(raw))


def _assert_no_symlink_components(path: Path, *, blocker: str) -> None:
    absolute = _absolute_path(path, blocker=blocker)
    cursor = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        cursor /= component
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise ProductionObservationBundleError(blocker) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise ProductionObservationBundleError(blocker)


def _directory_signature(
    path: Path,
    *,
    blocker: str,
    exact_mode: int | None = None,
) -> tuple[int, ...]:
    absolute = _absolute_path(path, blocker=blocker)
    _assert_no_symlink_components(absolute, blocker=blocker)
    try:
        metadata = os.lstat(absolute)
    except OSError as exc:
        raise ProductionObservationBundleError(blocker) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ProductionObservationBundleError(blocker)
    if exact_mode is not None and stat.S_IMODE(metadata.st_mode) != exact_mode:
        raise ProductionObservationBundleError(f"{blocker}_permissions_unsafe")
    return _stat_signature(metadata)


def _stable_file_bytes(
    path: Path,
    *,
    blocker: str,
    changed_blocker: str,
    max_bytes: int,
    exact_mode: int | None = None,
) -> tuple[bytes, tuple[int, ...]]:
    absolute = _absolute_path(path, blocker=blocker)
    _assert_no_symlink_components(absolute, blocker=blocker)
    descriptor: int | None = None
    try:
        before = os.lstat(absolute)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size <= 0
            or before.st_size > max_bytes
        ):
            raise ProductionObservationBundleError(blocker)
        if (
            exact_mode is not None
            and stat.S_IMODE(before.st_mode) != exact_mode
        ):
            raise ProductionObservationBundleError(
                f"{blocker}_permissions_unsafe"
            )
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(absolute, flags)
        signature = _stat_signature(before)
        if _stat_signature(os.fstat(descriptor)) != signature:
            raise ProductionObservationBundleError(changed_blocker)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ProductionObservationBundleError(blocker)
            chunks.append(chunk)
        if (
            _stat_signature(os.fstat(descriptor)) != signature
            or _stat_signature(os.lstat(absolute)) != signature
        ):
            raise ProductionObservationBundleError(changed_blocker)
        return b"".join(chunks), signature
    except ProductionObservationBundleError:
        raise
    except OSError as exc:
        raise ProductionObservationBundleError(blocker) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _json_object(payload: bytes, *, blocker: str) -> dict[str, Any]:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionObservationBundleError(blocker) from exc
    if not isinstance(decoded, dict):
        raise ProductionObservationBundleError(blocker)
    return decoded


def _safe_relative(value: Any, *, blocker: str) -> PurePosixPath:
    raw = str(value or "")
    if not raw or "\\" in raw:
        raise ProductionObservationBundleError(blocker)
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or any(component in {"", ".", ".."} for component in relative.parts)
        or str(relative) != raw
    ):
        raise ProductionObservationBundleError(blocker)
    return relative


def _safe_child(root: Path, relative: PurePosixPath, *, blocker: str) -> Path:
    root_absolute = _absolute_path(root, blocker=blocker)
    candidate = root_absolute.joinpath(*relative.parts)
    try:
        candidate.relative_to(root_absolute)
    except ValueError as exc:  # pragma: no cover - PurePosixPath guards this
        raise ProductionObservationBundleError(blocker) from exc
    _assert_no_symlink_components(candidate, blocker=blocker)
    return candidate


def _add_evidence(
    bodies: dict[str, bytes],
    metadata: dict[str, Mapping[str, Any]],
    *,
    body: bytes,
    item_metadata: Mapping[str, Any],
) -> str:
    digest = _sha256(body)
    previous_body = bodies.get(digest)
    previous_metadata = metadata.get(digest)
    normalized_metadata = dict(item_metadata)
    if previous_body is not None and (
        previous_body != body or previous_metadata != normalized_metadata
    ):
        raise ProductionObservationBundleError(
            "production_observation_evidence_digest_collision"
        )
    bodies[digest] = body
    metadata[digest] = normalized_metadata
    return digest


def _same_official_result(
    left: OfficialWebCompilationResult,
    right: OfficialWebCompilationResult,
) -> bool:
    return (
        left.observations == right.observations
        and left.receipts == right.receipts
        and dict(left.manifest) == dict(right.manifest)
    )


def _official_evidence_inputs(
    *,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    expected_plan_sha256: str,
) -> tuple[OfficialWebCompilationResult, _EvidenceInputs]:
    expected_manifest = _required_sha256(
        expected_manifest_sha256,
        blocker="production_official_manifest_sha256_invalid",
    )
    expected_plan = _required_sha256(
        expected_plan_sha256,
        blocker="production_official_plan_sha256_invalid",
    )
    path = _absolute_path(
        manifest_path,
        blocker="production_official_manifest_path_unsafe",
    )
    bundle_root = path.parent
    bundle_signature = _directory_signature(
        bundle_root,
        blocker="production_official_bundle_root_unsafe",
        exact_mode=0o700,
    )
    manifest_raw, manifest_signature = _stable_file_bytes(
        path,
        blocker="production_official_manifest_unsafe",
        changed_blocker="production_official_manifest_changed_during_read",
        max_bytes=_MAX_JSON_BYTES,
        exact_mode=0o600,
    )
    if _sha256(manifest_raw) != expected_manifest:
        raise ProductionObservationBundleError(
            "production_official_manifest_sha256_mismatch"
        )
    persisted_manifest = _json_object(
        manifest_raw,
        blocker="production_official_manifest_json_invalid",
    )
    raw_root = bundle_root / "raw"
    raw_root_signature = _directory_signature(
        raw_root,
        blocker="production_official_raw_root_unsafe",
        exact_mode=0o700,
    )

    compiled = recompile_official_web_bundle(
        path,
        expected_manifest_sha256=expected_manifest,
        expected_plan_sha256=expected_plan,
    )
    if len(compiled.observations) != 36 or len(compiled.receipts) != 36:
        raise ProductionObservationBundleError(
            "production_official_observation_count_invalid"
        )

    raw_artifacts = persisted_manifest.get("raw_artifacts")
    artifact_hashes = persisted_manifest.get("artifact_sha256")
    if not isinstance(raw_artifacts, Mapping) or len(raw_artifacts) != 12:
        raise ProductionObservationBundleError(
            "production_official_raw_artifacts_invalid"
        )
    if not isinstance(artifact_hashes, Mapping):
        raise ProductionObservationBundleError(
            "production_official_artifact_hashes_invalid"
        )
    plan_raw, _plan_signature = _stable_file_bytes(
        bundle_root / "plan.json",
        blocker="production_official_plan_artifact_unsafe",
        changed_blocker=(
            "production_official_plan_artifact_changed_during_read"
        ),
        max_bytes=_MAX_JSON_BYTES,
        exact_mode=0o600,
    )
    if (
        _sha256(plan_raw) != expected_plan
        or artifact_hashes.get("plan.json") != expected_plan
    ):
        raise ProductionObservationBundleError(
            "production_official_plan_artifact_sha256_mismatch"
        )
    plan = _json_object(
        plan_raw,
        blocker="production_official_plan_artifact_json_invalid",
    )
    raw_plan_pages = plan.get("pages")
    if not isinstance(raw_plan_pages, list) or len(raw_plan_pages) != 12:
        raise ProductionObservationBundleError(
            "production_official_plan_pages_invalid"
        )
    plan_pages: dict[str, Mapping[str, Any]] = {}
    for raw_page in raw_plan_pages:
        if not isinstance(raw_page, Mapping):
            raise ProductionObservationBundleError(
                "production_official_plan_page_invalid"
            )
        page_id = str(raw_page.get("page_id") or "")
        if not _PAGE_ID_RE.fullmatch(page_id) or page_id in plan_pages:
            raise ProductionObservationBundleError(
                "production_official_plan_page_id_invalid"
            )
        plan_pages[page_id] = raw_page
    if set(plan_pages) != set(raw_artifacts):
        raise ProductionObservationBundleError(
            "production_official_plan_raw_page_set_mismatch"
        )

    receipts_by_hash: dict[str, Mapping[str, Any]] = {}
    page_receipts: dict[str, list[Mapping[str, Any]]] = {}
    for receipt in compiled.receipts:
        content_hash = _required_sha256(
            receipt.get("content_hash"),
            blocker="production_official_receipt_content_hash_invalid",
        )
        pages = receipt.get("evidence_pages")
        if (
            not isinstance(pages, list)
            or len(pages) != 1
            or not isinstance(pages[0], Mapping)
        ):
            raise ProductionObservationBundleError(
                "production_official_receipt_evidence_invalid"
            )
        page = pages[0]
        page_id = str(page.get("page_id") or "")
        if not _PAGE_ID_RE.fullmatch(page_id):
            raise ProductionObservationBundleError(
                "production_official_receipt_page_id_invalid"
            )
        if content_hash in receipts_by_hash:
            raise ProductionObservationBundleError(
                "production_official_receipt_duplicate"
            )
        receipts_by_hash[content_hash] = receipt
        page_receipts.setdefault(page_id, []).append(receipt)
    observation_hashes = {item.content_hash for item in compiled.observations}
    if set(receipts_by_hash) != observation_hashes:
        raise ProductionObservationBundleError(
            "production_official_receipt_observation_set_mismatch"
        )
    if not set(page_receipts).issubset(raw_artifacts):
        raise ProductionObservationBundleError(
            "production_official_raw_reference_set_mismatch"
        )
    support_pages = set(raw_artifacts) - set(page_receipts)
    if len(support_pages) != 1:
        raise ProductionObservationBundleError(
            "production_official_support_page_set_invalid"
        )
    support_page_id = next(iter(support_pages))
    if str(plan_pages[support_page_id].get("parser_id") or "") != (
        "pbc-financial-statistics-html.v1"
    ):
        raise ProductionObservationBundleError(
            "production_official_support_page_role_invalid"
        )

    bodies: dict[str, bytes] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    digest_by_page: dict[str, str] = {}
    for page_id in sorted(raw_artifacts):
        if not _PAGE_ID_RE.fullmatch(str(page_id)):
            raise ProductionObservationBundleError(
                "production_official_raw_page_id_invalid"
            )
        artifact_name = str(raw_artifacts[page_id] or "")
        expected_name = f"raw/{page_id}.html"
        if artifact_name != expected_name:
            raise ProductionObservationBundleError(
                "production_official_raw_artifact_path_invalid"
            )
        relative = _safe_relative(
            artifact_name,
            blocker="production_official_raw_artifact_path_unsafe",
        )
        raw_path = _safe_child(
            bundle_root,
            relative,
            blocker="production_official_raw_artifact_unsafe",
        )
        body, _signature = _stable_file_bytes(
            raw_path,
            blocker="production_official_raw_artifact_unsafe",
            changed_blocker=(
                "production_official_raw_artifact_changed_during_read"
            ),
            max_bytes=_MAX_HTML_BYTES,
            exact_mode=0o600,
        )
        declared = _required_sha256(
            artifact_hashes.get(artifact_name),
            blocker="production_official_raw_artifact_sha256_invalid",
        )
        if _sha256(body) != declared:
            raise ProductionObservationBundleError(
                "production_official_raw_artifact_sha256_mismatch"
            )
        page_semantics = [
            receipt["evidence_pages"][0]
            for receipt in page_receipts.get(str(page_id), [])
        ]
        if page_semantics:
            semantic_hashes = {
                _required_sha256(
                    page.get("body_sha256"),
                    blocker=(
                        "production_official_receipt_body_sha256_invalid"
                    ),
                )
                for page in page_semantics
            }
            if semantic_hashes != {declared}:
                raise ProductionObservationBundleError(
                    "production_official_receipt_body_sha256_mismatch"
                )
            first = page_semantics[0]
        else:
            first = plan_pages[str(page_id)]
        stable_fields = (
            "parser_id",
            "parser_contract_sha256",
            "source_system",
            "source_url",
            "source_record_id",
            "period",
            "release_at",
            "body_sha256",
            "body_size_bytes",
        )
        if page_semantics and any(
            any(
                page.get(field) != first.get(field)
                for field in stable_fields
            )
            for page in page_semantics[1:]
        ):
            raise ProductionObservationBundleError(
                "production_official_receipt_page_semantics_mismatch"
            )
        digest = _add_evidence(
            bodies,
            metadata,
            body=body,
            item_metadata={
                "extension": ".html",
                "evidence_kind": "official_web_response_entity",
                "page_id": str(page_id),
                "parser_id": str(first.get("parser_id") or ""),
                "parser_contract_sha256": str(
                    first.get("parser_contract_sha256") or ""
                ),
                "source_system": str(first.get("source_system") or ""),
                "source_url": str(first.get("source_url") or ""),
                "source_record_id": str(first.get("source_record_id") or ""),
                "period": str(
                    first.get("period")
                    or first.get("expected_period")
                    or ""
                ),
                "release_at": str(first.get("release_at") or ""),
                "official_bundle_manifest_sha256": expected_manifest,
                "support_only": not bool(page_semantics),
                "size_bytes": len(body),
            },
        )
        digest_by_page[str(page_id)] = digest
    if len(bodies) != 12:
        raise ProductionObservationBundleError(
            "production_official_raw_digest_collision"
        )

    observation_mapping: dict[str, Sequence[str]] = {}
    for content_hash, receipt in receipts_by_hash.items():
        page_id = str(receipt["evidence_pages"][0]["page_id"])
        observation_mapping[content_hash] = [digest_by_page[page_id]]
    support_digest = digest_by_page[support_page_id]
    for observation in compiled.observations:
        if observation.source_system == "pboc_official":
            observation_mapping[observation.content_hash] = sorted(
                {
                    *observation_mapping[observation.content_hash],
                    support_digest,
                }
            )
    referenced = {
        digest
        for values in observation_mapping.values()
        for digest in values
    }
    if referenced != set(bodies):
        raise ProductionObservationBundleError(
            "production_official_evidence_unreferenced"
        )

    manifest_readback, readback_signature = _stable_file_bytes(
        path,
        blocker="production_official_manifest_unsafe",
        changed_blocker="production_official_manifest_changed_during_compile",
        max_bytes=_MAX_JSON_BYTES,
        exact_mode=0o600,
    )
    if (
        manifest_readback != manifest_raw
        or readback_signature != manifest_signature
    ):
        raise ProductionObservationBundleError(
            "production_official_manifest_changed_during_compile"
        )
    if (
        _directory_signature(
            bundle_root,
            blocker="production_official_bundle_root_unsafe",
            exact_mode=0o700,
        )
        != bundle_signature
        or _directory_signature(
            raw_root,
            blocker="production_official_raw_root_unsafe",
            exact_mode=0o700,
        )
        != raw_root_signature
    ):
        raise ProductionObservationBundleError(
            "production_official_bundle_changed_during_read"
        )
    replay = recompile_official_web_bundle(
        path,
        expected_manifest_sha256=expected_manifest,
        expected_plan_sha256=expected_plan,
    )
    if not _same_official_result(compiled, replay):
        raise ProductionObservationBundleError(
            "production_official_recompile_not_deterministic"
        )
    return compiled, _EvidenceInputs(
        bodies=bodies,
        metadata=metadata,
        observation_mapping=observation_mapping,
        official_file_count=len(bodies),
    )


def _snapshot_clock(
    snapshot_manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[Path, bytes, tuple[int, ...], datetime]:
    expected = _required_sha256(
        expected_manifest_sha256,
        blocker="production_local_manifest_sha256_invalid",
    )
    path = _absolute_path(
        snapshot_manifest_path,
        blocker="production_local_manifest_path_unsafe",
    )
    raw, signature = _stable_file_bytes(
        path,
        blocker="production_local_manifest_unsafe",
        changed_blocker="production_local_manifest_changed_during_read",
        max_bytes=_MAX_JSON_BYTES,
    )
    if _sha256(raw) != expected:
        raise ProductionObservationBundleError(
            "production_local_manifest_sha256_mismatch"
        )
    manifest = _json_object(
        raw,
        blocker="production_local_manifest_json_invalid",
    )
    snapshot_id = str(manifest.get("snapshot_id") or "")
    try:
        snapshot_at = datetime.strptime(snapshot_id, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=UTC
        )
    except ValueError as exc:
        raise ProductionObservationBundleError(
            "production_local_snapshot_id_invalid"
        ) from exc
    if path.name != f"{snapshot_id}.json":
        raise ProductionObservationBundleError(
            "production_local_snapshot_manifest_name_mismatch"
        )
    return path, raw, signature, snapshot_at


_LOCAL_PLAN_ROOT_KEYS = frozenset({"schema_version", "market", "targets"})
_LOCAL_PLAN_TARGET_KEYS = frozenset(
    {
        "target_trade_date",
        "snapshot_manifest_path",
        "expected_snapshot_manifest_sha256",
        "coverage_manifest_path",
        "expected_coverage_manifest_sha256",
        "scope_artifact_path",
        "expected_scope_artifact_sha256",
    }
)


def _load_local_bootstrap_plan(
    path_value: str | Path,
    *,
    expected_sha256: str,
) -> tuple[Path, bytes, tuple[int, ...], tuple[_LocalTargetBinding, ...]]:
    expected = _required_sha256(
        expected_sha256,
        blocker="production_local_bootstrap_plan_sha256_invalid",
    )
    path = _absolute_path(
        path_value,
        blocker="production_local_bootstrap_plan_path_unsafe",
    )
    raw, signature = _stable_file_bytes(
        path,
        blocker="production_local_bootstrap_plan_unsafe",
        changed_blocker="production_local_bootstrap_plan_changed_during_read",
        max_bytes=_MAX_JSON_BYTES,
        exact_mode=0o600,
    )
    if _sha256(raw) != expected:
        raise ProductionObservationBundleError(
            "production_local_bootstrap_plan_sha256_mismatch"
        )
    payload = _json_object(
        raw,
        blocker="production_local_bootstrap_plan_json_invalid",
    )
    if set(payload) != _LOCAL_PLAN_ROOT_KEYS:
        raise ProductionObservationBundleError(
            "production_local_bootstrap_plan_keys_invalid"
        )
    if (
        payload.get("schema_version") != LOCAL_BREADTH_BOOTSTRAP_PLAN_SCHEMA
        or str(payload.get("market") or "").upper() != "CN"
    ):
        raise ProductionObservationBundleError(
            "production_local_bootstrap_plan_contract_invalid"
        )
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or len(raw_targets) != 3:
        raise ProductionObservationBundleError(
            "production_local_bootstrap_plan_target_count_invalid"
        )
    bindings: list[_LocalTargetBinding] = []
    for raw_target in raw_targets:
        if not isinstance(raw_target, Mapping) or set(raw_target) != (
            _LOCAL_PLAN_TARGET_KEYS
        ):
            raise ProductionObservationBundleError(
                "production_local_bootstrap_plan_target_keys_invalid"
            )
        trade_date = str(raw_target.get("target_trade_date") or "")
        if not _COMPACT_DATE_RE.fullmatch(trade_date):
            raise ProductionObservationBundleError(
                "production_local_bootstrap_plan_trade_date_invalid"
            )
        bindings.append(
            _LocalTargetBinding(
                target_trade_date=trade_date,
                snapshot_manifest_path=_absolute_path(
                    str(raw_target["snapshot_manifest_path"]),
                    blocker="production_local_snapshot_manifest_path_unsafe",
                ),
                expected_snapshot_manifest_sha256=_required_sha256(
                    raw_target["expected_snapshot_manifest_sha256"],
                    blocker="production_local_snapshot_manifest_sha256_invalid",
                ),
                coverage_manifest_path=_absolute_path(
                    str(raw_target["coverage_manifest_path"]),
                    blocker="production_local_coverage_manifest_path_unsafe",
                ),
                expected_coverage_manifest_sha256=_required_sha256(
                    raw_target["expected_coverage_manifest_sha256"],
                    blocker="production_local_coverage_manifest_sha256_invalid",
                ),
                scope_artifact_path=_absolute_path(
                    str(raw_target["scope_artifact_path"]),
                    blocker="production_local_scope_artifact_path_unsafe",
                ),
                expected_scope_artifact_sha256=_required_sha256(
                    raw_target["expected_scope_artifact_sha256"],
                    blocker="production_local_scope_artifact_sha256_invalid",
                ),
            )
        )
    dates = [item.target_trade_date for item in bindings]
    if dates != sorted(dates) or len(set(dates)) != 3:
        raise ProductionObservationBundleError(
            "production_local_bootstrap_plan_dates_not_strictly_increasing"
        )
    return path, raw, signature, tuple(bindings)


def _generic_local_source_metadata(payload: bytes) -> dict[str, Any]:
    return {
        "extension": ".bin",
        "evidence_kind": "macro_local_bound_input",
        "size_bytes": len(payload),
    }


def _single_local_evidence_inputs(
    *,
    binding: _LocalTargetBinding,
    as_of: str,
) -> tuple[tuple[MacroObservation, ...], Mapping[str, Any], _EvidenceInputs]:
    cutoff = published_cutoff(as_of)
    observation, local_evidence = compile_local_market_breadth_observation(
        snapshot_manifest_path=binding.snapshot_manifest_path,
        expected_snapshot_manifest_sha256=(
            binding.expected_snapshot_manifest_sha256
        ),
        coverage_manifest_path=binding.coverage_manifest_path,
        expected_coverage_manifest_sha256=(
            binding.expected_coverage_manifest_sha256
        ),
        target_trade_date=binding.target_trade_date,
        scope_artifact_path=binding.scope_artifact_path,
        expected_scope_artifact_sha256=(
            binding.expected_scope_artifact_sha256
        ),
        as_of=as_of,
        clock=lambda: cutoff,
    )
    if (
        observation.indicator_id != "market.breadth"
        or observation.period_end.replace("-", "")
        != binding.target_trade_date
        or local_evidence.get("schema_version")
        != LOCAL_MARKET_BREADTH_EVIDENCE_SCHEMA
        or local_evidence.get("observation_content_hash")
        != observation.content_hash
    ):
        raise ProductionObservationBundleError(
            "production_local_observation_scope_invalid"
        )

    evidence_semantic_sha256 = _required_sha256(
        local_evidence.get("evidence_sha256"),
        blocker="production_local_evidence_semantic_sha256_invalid",
    )
    semantic_evidence = dict(local_evidence)
    semantic_evidence.pop("evidence_sha256", None)
    if canonical_hash(semantic_evidence) != evidence_semantic_sha256:
        raise ProductionObservationBundleError(
            "production_local_evidence_semantic_sha256_mismatch"
        )
    coverage_contract_sha256 = _required_sha256(
        local_evidence.get("coverage_contract_sha256"),
        blocker="production_local_coverage_contract_sha256_invalid",
    )
    coverage_summary = local_evidence.get("coverage_summary")
    if (
        not isinstance(coverage_summary, Mapping)
        or canonical_hash(dict(coverage_summary))
        != coverage_contract_sha256
    ):
        raise ProductionObservationBundleError(
            "production_local_coverage_contract_sha256_mismatch"
        )
    if local_evidence.get("formula_contract_sha256") != (
        LOCAL_MARKET_BREADTH_FORMULA_SHA256
    ):
        raise ProductionObservationBundleError(
            "production_local_formula_contract_sha256_mismatch"
        )
    effective_available_at = str(
        local_evidence.get("effective_available_at") or ""
    )
    try:
        parse_timestamp(
            effective_available_at,
            field_name="effective_available_at",
        )
    except ValueError as exc:
        raise ProductionObservationBundleError(
            "production_local_effective_available_at_invalid"
        ) from exc

    source_specs = (
        (
            binding.snapshot_manifest_path,
            binding.expected_snapshot_manifest_sha256,
            "snapshot_manifest_sha256",
            "production_local_snapshot_manifest",
            _MAX_JSON_BYTES,
        ),
        (
            binding.coverage_manifest_path,
            binding.expected_coverage_manifest_sha256,
            "coverage_manifest_sha256",
            "production_local_coverage_manifest",
            _MAX_JSON_BYTES,
        ),
        (
            binding.scope_artifact_path,
            binding.expected_scope_artifact_sha256,
            None,
            "production_local_scope_artifact",
            _MAX_SCOPE_BYTES,
        ),
    )
    bodies: dict[str, bytes] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    source_digests: list[str] = []
    readbacks: list[tuple[Path, bytes, tuple[int, ...], str, int]] = []
    for source_path, expected_hash, evidence_field, blocker, max_bytes in (
        source_specs
    ):
        raw, signature = _stable_file_bytes(
            source_path,
            blocker=f"{blocker}_unsafe",
            changed_blocker=f"{blocker}_changed_during_read",
            max_bytes=max_bytes,
        )
        if _sha256(raw) != expected_hash:
            raise ProductionObservationBundleError(
                f"{blocker}_sha256_mismatch"
            )
        if evidence_field is not None and local_evidence.get(
            evidence_field
        ) != expected_hash:
            raise ProductionObservationBundleError(
                f"{blocker}_binding_mismatch"
            )
        source_digests.append(
            _add_evidence(
                bodies,
                metadata,
                body=raw,
                item_metadata=_generic_local_source_metadata(raw),
            )
        )
        readbacks.append(
            (source_path, raw, signature, blocker, max_bytes)
        )
    scope_payload = local_evidence.get("scope_artifact")
    if (
        not isinstance(scope_payload, Mapping)
        or scope_payload.get("file_sha256")
        != binding.expected_scope_artifact_sha256
    ):
        raise ProductionObservationBundleError(
            "production_local_scope_artifact_binding_mismatch"
        )

    part = local_evidence.get("part_file")
    if not isinstance(part, Mapping):
        raise ProductionObservationBundleError(
            "production_local_part_evidence_invalid"
        )
    relative = _safe_relative(
        part.get("relative_path"),
        blocker="production_local_part_path_unsafe",
    )
    if (
        len(relative.parts) != 3
        or not re.fullmatch(r"year=20\d{2}", relative.parts[0])
        or not re.fullmatch(r"month=(?:0[1-9]|1[0-2])", relative.parts[1])
        or relative.parts[2] != "part.parquet"
    ):
        raise ProductionObservationBundleError(
            "production_local_part_path_invalid"
        )
    table_root = _absolute_path(
        str(local_evidence.get("table_root") or ""),
        blocker="production_local_table_root_unsafe",
    )
    part_path = _safe_child(
        table_root,
        relative,
        blocker="production_local_part_unsafe",
    )
    part_raw, part_signature = _stable_file_bytes(
        part_path,
        blocker="production_local_part_unsafe",
        changed_blocker="production_local_part_changed_during_read",
        max_bytes=_MAX_PART_BYTES,
    )
    part_hash = _required_sha256(
        part.get("sha256"),
        blocker="production_local_part_sha256_invalid",
    )
    if (
        _sha256(part_raw) != part_hash
        or part.get("size_bytes") != len(part_raw)
        or part.get("mtime_ns") != int(part_signature[4])
    ):
        raise ProductionObservationBundleError(
            "production_local_part_binding_mismatch"
        )
    source_digests.append(
        _add_evidence(
            bodies,
            metadata,
            body=part_raw,
            item_metadata=_generic_local_source_metadata(part_raw),
        )
    )
    readbacks.append(
        (
            part_path,
            part_raw,
            part_signature,
            "production_local_part",
            _MAX_PART_BYTES,
        )
    )

    local_evidence_bytes = _canonical_json_bytes(local_evidence)
    local_evidence_digest = _add_evidence(
        bodies,
        metadata,
        body=local_evidence_bytes,
        item_metadata={
            "extension": ".bin",
            "evidence_kind": "strict_parquet_local_observation_evidence",
            "schema_version": str(local_evidence.get("schema_version")),
            "target_trade_date": binding.target_trade_date,
            "evidence_semantic_sha256": evidence_semantic_sha256,
            "coverage_contract_sha256": coverage_contract_sha256,
            "effective_available_at": effective_available_at,
            "size_bytes": len(local_evidence_bytes),
        },
    )
    mapping = {
        observation.content_hash: sorted(
            {*source_digests, local_evidence_digest}
        )
    }
    replay, replay_evidence = compile_local_market_breadth_observation(
        snapshot_manifest_path=binding.snapshot_manifest_path,
        expected_snapshot_manifest_sha256=(
            binding.expected_snapshot_manifest_sha256
        ),
        coverage_manifest_path=binding.coverage_manifest_path,
        expected_coverage_manifest_sha256=(
            binding.expected_coverage_manifest_sha256
        ),
        target_trade_date=binding.target_trade_date,
        scope_artifact_path=binding.scope_artifact_path,
        expected_scope_artifact_sha256=(
            binding.expected_scope_artifact_sha256
        ),
        as_of=as_of,
        clock=lambda: cutoff,
    )
    if (
        replay != observation
        or replay_evidence != local_evidence
        or _canonical_json_bytes(replay_evidence) != local_evidence_bytes
    ):
        raise ProductionObservationBundleError(
            "production_local_recompile_not_deterministic"
        )
    for source_path, original, signature, blocker, max_bytes in readbacks:
        readback, readback_signature = _stable_file_bytes(
            source_path,
            blocker=f"{blocker}_unsafe",
            changed_blocker=f"{blocker}_changed_during_compile",
            max_bytes=max_bytes,
        )
        if readback != original or readback_signature != signature:
            raise ProductionObservationBundleError(
                f"{blocker}_changed_during_compile"
            )
    return (observation,), local_evidence, _EvidenceInputs(
        bodies=bodies,
        metadata=metadata,
        observation_mapping=mapping,
        local_file_count=len(bodies),
    )


def _bootstrap_local_evidence_inputs(
    *,
    plan_path: str | Path,
    expected_plan_sha256: str,
    as_of: str,
) -> tuple[tuple[MacroObservation, ...], Mapping[str, Any], _EvidenceInputs]:
    path, raw, signature, bindings = _load_local_bootstrap_plan(
        plan_path,
        expected_sha256=expected_plan_sha256,
    )
    bodies: dict[str, bytes] = {}
    metadata: dict[str, Mapping[str, Any]] = {}
    mapping: dict[str, Sequence[str]] = {}
    observations: list[MacroObservation] = []
    local_evidences: list[Mapping[str, Any]] = []
    for binding in bindings:
        compiled, local_evidence, evidence = _single_local_evidence_inputs(
            binding=binding,
            as_of=as_of,
        )
        observations.extend(compiled)
        local_evidences.append(local_evidence)
        for digest, body in evidence.bodies.items():
            if digest in bodies and (
                bodies[digest] != body
                or metadata[digest] != evidence.metadata[digest]
            ):
                raise ProductionObservationBundleError(
                    "production_observation_evidence_digest_collision"
                )
            bodies[digest] = body
            metadata[digest] = evidence.metadata[digest]
        mapping.update(evidence.observation_mapping)
    if len(observations) != 3 or len(mapping) != 3:
        raise ProductionObservationBundleError(
            "production_local_bootstrap_observation_count_invalid"
        )
    plan_digest = _add_evidence(
        bodies,
        metadata,
        body=raw,
        item_metadata=_generic_local_source_metadata(raw),
    )
    mapping = {
        content_hash: sorted({*digests, plan_digest})
        for content_hash, digests in mapping.items()
    }
    referenced = {
        digest for digests in mapping.values() for digest in digests
    }
    if referenced != set(bodies):
        raise ProductionObservationBundleError(
            "production_local_evidence_unreferenced"
        )
    readback, readback_signature = _stable_file_bytes(
        path,
        blocker="production_local_bootstrap_plan_unsafe",
        changed_blocker="production_local_bootstrap_plan_changed_during_compile",
        max_bytes=_MAX_JSON_BYTES,
        exact_mode=0o600,
    )
    if readback != raw or readback_signature != signature:
        raise ProductionObservationBundleError(
            "production_local_bootstrap_plan_changed_during_compile"
        )
    effective_values = [
        str(item.get("effective_available_at") or "")
        for item in local_evidences
    ]
    summary = {
        "bootstrap_plan_path": str(path),
        "bootstrap_plan_sha256": _sha256(raw),
        "target_trade_dates": [item.target_trade_date for item in bindings],
        "snapshot_manifest_sha256": canonical_hash(
            {
                "values": [
                    item.expected_snapshot_manifest_sha256
                    for item in bindings
                ]
            }
        ),
        "coverage_contract_sha256": canonical_hash(
            {
                "values": [
                    str(item.get("coverage_contract_sha256") or "")
                    for item in local_evidences
                ]
            }
        ),
        "effective_available_at": max(
            effective_values,
            key=lambda value: parse_timestamp(
                value,
                field_name="effective_available_at",
            ),
        ),
    }
    return tuple(observations), summary, _EvidenceInputs(
        bodies=bodies,
        metadata=metadata,
        observation_mapping=mapping,
        local_file_count=len(bodies),
    )


def _merge_evidence_inputs(
    official: _EvidenceInputs,
    local: _EvidenceInputs,
) -> _EvidenceInputs:
    bodies = dict(official.bodies)
    metadata = dict(official.metadata)
    for digest, body in local.bodies.items():
        if digest in bodies and (
            bodies[digest] != body
            or metadata[digest] != local.metadata[digest]
        ):
            raise ProductionObservationBundleError(
                "production_observation_evidence_digest_collision"
            )
        bodies[digest] = body
        metadata[digest] = local.metadata[digest]
    mapping = {
        key: list(value) for key, value in official.observation_mapping.items()
    }
    for content_hash, digests in local.observation_mapping.items():
        if content_hash in mapping:
            raise ProductionObservationBundleError(
                "production_observation_content_hash_collision"
            )
        mapping[content_hash] = list(digests)
    referenced = {
        digest for values in mapping.values() for digest in values
    }
    if referenced != set(bodies):
        raise ProductionObservationBundleError(
            "production_observation_evidence_unreferenced"
        )
    return _EvidenceInputs(
        bodies=bodies,
        metadata=metadata,
        observation_mapping=mapping,
        official_file_count=official.official_file_count,
        local_file_count=local.local_file_count,
    )


def _validated_production_snapshot(
    observations: Sequence[MacroObservation],
    *,
    as_of: str,
    decision_cutoff_at: Any | None = None,
) -> Mapping[str, Any]:
    if len(observations) != _PRODUCTION_OBSERVATION_COUNT:
        raise ProductionObservationBundleError(
            "production_observation_count_invalid"
        )
    counts = Counter(item.indicator_id for item in observations)
    if len(counts) != _PRODUCTION_INDICATOR_COUNT or set(counts.values()) != {
        _HISTORY_LENGTH
    }:
        raise ProductionObservationBundleError(
            "production_observation_history_scope_invalid"
        )
    if len({item.content_hash for item in observations}) != len(observations):
        raise ProductionObservationBundleError(
            "production_observation_content_hash_duplicate"
        )
    snapshot = build_macro_snapshot(
        observations,
        market="CN",
        as_of=as_of,
        decision_cutoff_at=decision_cutoff_at,
    )
    if snapshot.readiness_status != "pass":
        raise ProductionObservationBundleError(
            "production_observation_readiness_not_pass:"
            + ",".join(snapshot.blockers)
        )
    if snapshot.blockers:
        raise ProductionObservationBundleError(
            "production_observation_blockers_present"
        )
    try:
        national_coverage = float(snapshot.coverage.get("national"))
    except (TypeError, ValueError) as exc:
        raise ProductionObservationBundleError(
            "production_observation_national_coverage_invalid"
        ) from exc
    if national_coverage != _EXPECTED_NATIONAL_COVERAGE:
        raise ProductionObservationBundleError(
            "production_observation_national_coverage_not_0_8125"
        )
    return snapshot.to_dict()


def _normalized_rows(
    observations: Iterable[Mapping[str, Any] | MacroObservation],
) -> list[dict[str, Any]]:
    rows = [
        MacroObservation.from_mapping(
            item.to_dict() if isinstance(item, MacroObservation) else item
        ).to_dict()
        for item in observations
    ]
    return sorted(rows, key=lambda item: str(item["content_hash"]))


def _load_current(
    root: str | Path,
    *,
    expected_pointer_sha256: str,
) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    expected = _required_sha256(
        expected_pointer_sha256,
        blocker="production_observation_expected_pointer_sha256_invalid",
        allow_empty=True,
    )
    current = pointer_sha256(root)
    if current != expected:
        raise ProductionObservationBundleError(
            "production_observation_pointer_cas_mismatch"
        )
    if not current:
        return [], current, {}
    rows, pointer = load_observations(root)
    if pointer_sha256(root) != current:
        raise ProductionObservationBundleError(
            "production_observation_pointer_changed_during_read"
        )
    return rows, current, pointer


def _official_period_matches_row(period: str, period_end: str) -> bool:
    if re.fullmatch(r"20\d{4}", period):
        return period_end.replace("-", "")[:6] == period
    quarter_match = re.fullmatch(r"(20\d{2})Q([1-4])", period)
    if quarter_match is None:
        return False
    expected_month = int(quarter_match.group(2)) * 3
    return period_end.startswith(
        f"{quarter_match.group(1)}-{expected_month:02d}-"
    )


def _official_source_url_valid(source_system: str, source_url: str) -> bool:
    try:
        parsed = urlsplit(source_url)
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme == "https"
        and parsed.hostname == _OFFICIAL_HOST_BY_SOURCE.get(source_system)
        and parsed.username is None
        and parsed.password is None
        and port is None
        and not parsed.query
        and not parsed.fragment
        and bool(parsed.path)
    )


def _strict_chain_evidence(
    manifest: Mapping[str, Any],
    *,
    row_hashes: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    raw_files = manifest.get("evidence_files")
    raw_mapping = manifest.get("observation_evidence")
    if not isinstance(raw_files, list) or not isinstance(raw_mapping, Mapping):
        raise ProductionObservationBundleError(
            "production_observation_chain_evidence_shape_invalid"
        )
    evidence_count = manifest.get("evidence_file_count")
    if (
        isinstance(evidence_count, bool)
        or evidence_count != len(raw_files)
        or not raw_files
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_evidence_count_invalid"
        )

    files_by_digest: dict[str, dict[str, Any]] = {}
    normalized_files: list[dict[str, Any]] = []
    for raw_file in raw_files:
        if not isinstance(raw_file, Mapping) or set(raw_file) != (
            _EVIDENCE_FILE_KEYS
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_evidence_file_contract_invalid"
            )
        item = dict(raw_file)
        digest = _required_sha256(
            item.get("sha256"),
            blocker="production_observation_chain_evidence_sha256_invalid",
        )
        size_bytes = item.get("size_bytes")
        metadata = item.get("metadata")
        if (
            digest in files_by_digest
            or isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes <= 0
            or not isinstance(metadata, Mapping)
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_evidence_file_contract_invalid"
            )
        metadata = dict(metadata)
        if metadata.get("size_bytes") != size_bytes:
            raise ProductionObservationBundleError(
                "production_observation_chain_evidence_size_binding_invalid"
            )
        extension = str(metadata.get("extension") or "")
        if extension not in {".html", ".bin"} or item.get("path") != (
            f"evidence/raw/{digest}{extension}"
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_evidence_path_invalid"
            )
        if item.get("metadata_sha256") != _sha256(
            _canonical_json_bytes(metadata)
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_evidence_metadata_hash_invalid"
            )

        kind = str(metadata.get("evidence_kind") or "")
        if kind == "official_web_response_entity":
            parser_id = str(metadata.get("parser_id") or "")
            source_system = str(metadata.get("source_system") or "")
            period = str(metadata.get("period") or "")
            support_only = metadata.get("support_only")
            if (
                set(metadata) != _OFFICIAL_EVIDENCE_METADATA_KEYS
                or extension != ".html"
                or not _PAGE_ID_RE.fullmatch(
                    str(metadata.get("page_id") or "")
                )
                or parser_id not in _PARSER_SOURCE_SYSTEM
                or source_system != _PARSER_SOURCE_SYSTEM.get(parser_id)
                or metadata.get("parser_contract_sha256")
                != PARSER_CONTRACT_SHA256.get(parser_id)
                or not isinstance(support_only, bool)
                or not re.fullmatch(r"20\d{4}|20\d{2}Q[1-4]", period)
                or not _official_source_url_valid(
                    source_system,
                    str(metadata.get("source_url") or ""),
                )
            ):
                raise ProductionObservationBundleError(
                    "production_observation_chain_official_evidence_invalid"
                )
            _required_sha256(
                metadata.get("official_bundle_manifest_sha256"),
                blocker=(
                    "production_observation_chain_official_manifest_hash_invalid"
                ),
            )
            if support_only:
                if (
                    source_system != "pbc_official"
                    or parser_id != PBC_MONEY_STOCK_PARSER
                    or metadata.get("source_record_id") not in {"", None}
                    or metadata.get("release_at") not in {"", None}
                ):
                    raise ProductionObservationBundleError(
                        "production_observation_chain_support_role_invalid"
                    )
            else:
                if not str(metadata.get("source_record_id") or ""):
                    raise ProductionObservationBundleError(
                        "production_observation_chain_official_record_missing"
                    )
                try:
                    parse_timestamp(
                        metadata.get("release_at"), field_name="release_at"
                    )
                except ValueError as exc:
                    raise ProductionObservationBundleError(
                        "production_observation_chain_official_release_invalid"
                    ) from exc
        elif kind == "macro_local_bound_input":
            if (
                set(metadata) != _LOCAL_INPUT_METADATA_KEYS
                or extension != ".bin"
            ):
                raise ProductionObservationBundleError(
                    "production_observation_chain_local_input_role_invalid"
                )
        elif kind == "strict_parquet_local_observation_evidence":
            if (
                set(metadata)
                != _LOCAL_OBSERVATION_EVIDENCE_METADATA_KEYS
                or extension != ".bin"
                or metadata.get("schema_version")
                != LOCAL_MARKET_BREADTH_EVIDENCE_SCHEMA
                or not _COMPACT_DATE_RE.fullmatch(
                    str(metadata.get("target_trade_date") or "")
                )
            ):
                raise ProductionObservationBundleError(
                    "production_observation_chain_local_evidence_role_invalid"
                )
            for field_name in (
                "evidence_semantic_sha256",
                "coverage_contract_sha256",
            ):
                _required_sha256(
                    metadata.get(field_name),
                    blocker=(
                        "production_observation_chain_local_evidence_hash_invalid:"
                        + field_name
                    ),
                )
            try:
                parse_timestamp(
                    metadata.get("effective_available_at"),
                    field_name="effective_available_at",
                )
            except ValueError as exc:
                raise ProductionObservationBundleError(
                    "production_observation_chain_local_evidence_time_invalid"
                ) from exc
        else:
            raise ProductionObservationBundleError(
                "production_observation_chain_evidence_kind_invalid"
            )

        normalized_item = {**item, "metadata": metadata, "sha256": digest}
        files_by_digest[digest] = normalized_item
        normalized_files.append(normalized_item)

    if normalized_files != sorted(
        normalized_files, key=lambda item: item["sha256"]
    ) or manifest.get("evidence_set_sha256") != canonical_hash(
        {"evidence_files": normalized_files}
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_evidence_set_invalid"
        )
    if set(raw_mapping) != row_hashes:
        raise ProductionObservationBundleError(
            "production_observation_chain_evidence_mapping_incomplete"
        )

    mapping: dict[str, list[str]] = {}
    referenced: set[str] = set()
    for content_hash in sorted(row_hashes):
        raw_digests = raw_mapping.get(content_hash)
        if (
            not isinstance(raw_digests, list)
            or not raw_digests
            or raw_digests != sorted(set(raw_digests))
            or any(digest not in files_by_digest for digest in raw_digests)
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_evidence_mapping_invalid"
            )
        mapping[content_hash] = list(raw_digests)
        referenced.update(raw_digests)
    if referenced != set(files_by_digest):
        raise ProductionObservationBundleError(
            "production_observation_chain_evidence_reference_set_invalid"
        )
    return files_by_digest, mapping


def validate_production_observation_chain(
    observations: Sequence[Mapping[str, Any] | MacroObservation],
    *,
    generation_manifest: Mapping[str, Any],
    pointer_metadata: Mapping[str, Any] | None = None,
) -> dict[str, list[str]]:
    """Validate the complete production bootstrap/update evidence chain.

    ``macro_mart`` callers pass rows from :func:`load_observations`, that
    pointer's ``generation_manifest`` and its pointer-level ``metadata``.  A
    normalized observation-to-evidence mapping is returned only after the
    metadata, lineage, target, row-source policy and evidence roles all pass.
    """

    if (
        not isinstance(generation_manifest, Mapping)
        or generation_manifest.get("schema_version")
        != "macro-observation-generation.v2"
    ):
        raise ProductionObservationBundleError(
            "local_market_observation_existing_generation_v2_required"
        )
    manifest = dict(generation_manifest)
    if set(manifest) != _GENERATION_V2_KEYS:
        raise ProductionObservationBundleError(
            "production_observation_chain_generation_contract_invalid"
        )
    try:
        normalized_rows = _normalized_rows(observations)
    except (TypeError, ValueError) as exc:
        raise ProductionObservationBundleError(
            "production_observation_chain_rows_invalid"
        ) from exc
    if not normalized_rows:
        raise ProductionObservationBundleError(
            "production_observation_chain_rows_empty"
        )
    row_hashes = {str(row["content_hash"]) for row in normalized_rows}
    if len(row_hashes) != len(normalized_rows):
        raise ProductionObservationBundleError(
            "production_observation_chain_content_hash_duplicate"
        )

    row_count = manifest.get("row_count")
    generation_id = str(manifest.get("generation_id") or "")
    if (
        manifest.get("status") != "OK"
        or isinstance(row_count, bool)
        or row_count != len(normalized_rows)
        or not _GENERATION_ID_RE.fullmatch(generation_id)
        or manifest.get("observer_only") is not True
        or manifest.get("production_eligible") is not False
        or manifest.get("applied") is not False
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_generation_contract_invalid"
        )
    for field_name in ("parquet_sha256", "content_set_hash"):
        _required_sha256(
            manifest.get(field_name),
            blocker=(
                "production_observation_chain_generation_hash_invalid:"
                + field_name
            ),
        )
    if manifest.get("content_set_hash") != canonical_hash(
        {"hashes": sorted(row_hashes)}
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_content_set_hash_mismatch"
        )
    try:
        parse_timestamp(manifest.get("created_at"), field_name="created_at")
    except ValueError as exc:
        raise ProductionObservationBundleError(
            "production_observation_chain_created_at_invalid"
        ) from exc
    if (
        manifest.get("min_available_at")
        != min(str(row["available_at"]) for row in normalized_rows)
        or manifest.get("max_available_at")
        != max(str(row["available_at"]) for row in normalized_rows)
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_available_range_invalid"
        )
    added_hashes = manifest.get("added_content_hashes")
    if (
        not isinstance(added_hashes, list)
        or added_hashes != sorted(set(added_hashes))
        or any(digest not in row_hashes for digest in added_hashes)
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_added_hashes_invalid"
        )

    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_metadata_missing"
        )
    metadata = dict(metadata)
    if pointer_metadata is not None and (
        not isinstance(pointer_metadata, Mapping)
        or dict(pointer_metadata) != metadata
    ):
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_metadata_mismatch"
        )
    chain_schema = str(metadata.get("schema_version") or "")
    expected_metadata_keys = (
        _BOOTSTRAP_METADATA_KEYS
        if chain_schema == PRODUCTION_OBSERVATION_BUNDLE_SCHEMA
        else _UPDATE_METADATA_KEYS
        if chain_schema == LOCAL_MARKET_OBSERVATION_PUBLICATION_SCHEMA
        else frozenset()
    )
    if not expected_metadata_keys or set(metadata) != expected_metadata_keys:
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_schema_invalid"
        )
    logical_as_of = str(metadata.get("as_of") or "")
    if (
        str(metadata.get("market") or "") != "CN"
        or not _COMPACT_DATE_RE.fullmatch(logical_as_of)
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_target_invalid"
        )
    for field_name in (
        "local_snapshot_manifest_sha256",
        "local_coverage_contract_sha256",
        "validated_snapshot_hash",
    ):
        _required_sha256(
            metadata.get(field_name),
            blocker=(
                "local_market_observation_production_chain_hash_invalid:"
                + field_name
            ),
        )
    try:
        decision_cutoff = parse_timestamp(
            metadata.get("decision_cutoff_at"),
            field_name="decision_cutoff_at",
        )
        local_effective = parse_timestamp(
            metadata.get("local_effective_available_at"),
            field_name="local_effective_available_at",
        )
        available_times = [
            parse_timestamp(row["available_at"], field_name="available_at")
            for row in normalized_rows
        ]
    except ValueError as exc:
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_time_invalid"
        ) from exc
    if (
        max(available_times) > decision_cutoff
        or local_effective > decision_cutoff
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_after_decision_cutoff"
        )

    official_rows = [
        row
        for row in normalized_rows
        if row["indicator_id"] in _OFFICIAL_SOURCE_BY_INDICATOR
    ]
    local_rows = sorted(
        [
            row
            for row in normalized_rows
            if row["indicator_id"] == "market.breadth"
        ],
        key=lambda row: (
            str(row["period_end"]),
            str(row["available_at"]),
            str(row["content_hash"]),
        ),
    )
    official_counts = Counter(row["indicator_id"] for row in official_rows)
    if (
        set(official_counts) != set(_OFFICIAL_SOURCE_BY_INDICATOR)
        or set(official_counts.values()) != {_HISTORY_LENGTH}
        or len(official_rows) != 36
        or len(local_rows) < 3
        or len(normalized_rows) != len(official_rows) + len(local_rows)
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_row_scope_invalid"
        )
    for row in official_rows:
        if (
            row["source_system"]
            != _OFFICIAL_SOURCE_BY_INDICATOR[row["indicator_id"]]
            or row["dimension_type"] != "national"
            or row["quality_status"] != "pass"
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_official_source_policy_invalid"
            )
    local_dates = [
        str(row["period_end"]).replace("-", "") for row in local_rows
    ]
    if (
        local_dates != sorted(local_dates)
        or len(set(local_dates)) < 3
        or logical_as_of != local_dates[-1]
        or any(
            row["source_system"] != "local_strict_parquet"
            or row["dimension_type"] != "market_confirmation"
            or row["frequency"] != "daily"
            or row["unit"] != "%"
            or row["quality_status"] != "pass"
            for row in local_rows
        )
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_local_source_policy_invalid"
        )
    try:
        snapshot = build_macro_snapshot(
            normalized_rows,
            market="CN",
            as_of=logical_as_of,
            decision_cutoff_at=decision_cutoff,
        ).to_dict()
    except (TypeError, ValueError) as exc:
        raise ProductionObservationBundleError(
            "production_observation_chain_snapshot_invalid"
        ) from exc
    if (
        snapshot.get("readiness_status") != "pass"
        or snapshot.get("blockers") not in ([], ())
        or float(snapshot.get("coverage", {}).get("national", -1.0))
        != _EXPECTED_NATIONAL_COVERAGE
        or metadata.get("validated_snapshot_hash")
        != snapshot.get("snapshot_hash")
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_snapshot_binding_invalid"
        )

    files_by_digest, mapping = _strict_chain_evidence(
        manifest,
        row_hashes=row_hashes,
    )
    metadata_by_digest = {
        digest: item["metadata"] for digest, item in files_by_digest.items()
    }

    official_evidence = {
        digest
        for digest, item_metadata in metadata_by_digest.items()
        if item_metadata["evidence_kind"] == "official_web_response_entity"
    }
    support_evidence = {
        digest
        for digest in official_evidence
        if metadata_by_digest[digest]["support_only"] is True
    }
    strict_local_evidence = {
        digest
        for digest, item_metadata in metadata_by_digest.items()
        if item_metadata["evidence_kind"]
        == "strict_parquet_local_observation_evidence"
    }
    generic_local_evidence = {
        digest
        for digest, item_metadata in metadata_by_digest.items()
        if item_metadata["evidence_kind"] == "macro_local_bound_input"
    }
    if len(official_evidence) != 12 or len(support_evidence) != 1:
        raise ProductionObservationBundleError(
            "production_observation_chain_official_evidence_scope_invalid"
        )
    support_digest = next(iter(support_evidence))
    support_metadata = metadata_by_digest[support_digest]
    if (
        support_metadata["source_system"] != "pbc_official"
        or support_metadata["parser_id"] != PBC_MONEY_STOCK_PARSER
    ):
        raise ProductionObservationBundleError(
            "production_observation_chain_support_role_invalid"
        )
    official_bundle_hashes = {
        str(metadata_by_digest[digest]["official_bundle_manifest_sha256"])
        for digest in official_evidence
    }
    if len(official_bundle_hashes) != 1:
        raise ProductionObservationBundleError(
            "production_observation_chain_official_bundle_binding_invalid"
        )

    for row in official_rows:
        mapped = set(mapping[str(row["content_hash"])])
        if not mapped or not mapped.issubset(official_evidence):
            raise ProductionObservationBundleError(
                "production_observation_chain_official_mapping_role_invalid"
            )
        primary = [
            digest
            for digest in mapped
            if metadata_by_digest[digest]["support_only"] is False
        ]
        expected_support = (
            {support_digest}
            if row["source_system"] == "pboc_official"
            else set()
        )
        if len(primary) != 1 or (mapped & support_evidence) != expected_support:
            raise ProductionObservationBundleError(
                "production_observation_chain_official_mapping_role_invalid"
            )
        primary_metadata = metadata_by_digest[primary[0]]
        parser_id = str(primary_metadata["parser_id"])
        allowed_parsers = (
            {PBC_MONEY_STOCK_PARSER}
            if row["source_system"] == "pboc_official"
            else {NBS_OFFICIAL_PMI_PARSER}
            if row["indicator_id"] == "cn.pmi_manufacturing"
            else {
                NBS_NATIONAL_ECONOMY_PARSER,
                NBS_QUARTERLY_GDP_PARSER,
            }
            if row["indicator_id"] == "cn.gdp_yoy"
            else {NBS_NATIONAL_ECONOMY_PARSER}
        )
        try:
            primary_release = parse_timestamp(
                primary_metadata["release_at"], field_name="release_at"
            )
            row_release = parse_timestamp(
                row["release_at"], field_name="release_at"
            )
        except ValueError as exc:  # pragma: no cover - normalized above
            raise ProductionObservationBundleError(
                "production_observation_chain_official_mapping_invalid"
            ) from exc
        if (
            parser_id not in allowed_parsers
            or primary_metadata["source_system"]
            != _EVIDENCE_SOURCE_BY_OBSERVATION_SOURCE[row["source_system"]]
            or primary_metadata["source_record_id"] != row["source_record_id"]
            or primary_metadata["source_url"] != row["source_url"]
            or primary_release != row_release
            or not _official_period_matches_row(
                str(primary_metadata["period"]), str(row["period_end"])
            )
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_official_mapping_invalid"
            )

    if len(strict_local_evidence) != len(local_rows):
        raise ProductionObservationBundleError(
            "production_observation_chain_local_evidence_scope_invalid"
        )
    local_binding_by_hash: dict[str, dict[str, Any]] = {}
    for row in local_rows:
        content_hash = str(row["content_hash"])
        target = str(row["period_end"]).replace("-", "")
        mapped = set(mapping[content_hash])
        if not mapped.issubset(strict_local_evidence | generic_local_evidence):
            raise ProductionObservationBundleError(
                "production_observation_chain_local_mapping_role_invalid"
            )
        mapped_strict = mapped & strict_local_evidence
        mapped_generic = mapped & generic_local_evidence
        # The v4 market pointer may serve as both the immutable snapshot and
        # closing-coverage manifest.  Those two roles intentionally share one
        # byte hash, so the normalized evidence mapping has three generic
        # inputs (snapshot/coverage, scope, and part) instead of four.
        if len(mapped_strict) != 1 or len(mapped_generic) not in {3, 4, 5}:
            raise ProductionObservationBundleError(
                "production_observation_chain_local_mapping_role_invalid"
            )
        strict_metadata = metadata_by_digest[next(iter(mapped_strict))]
        try:
            strict_effective = parse_timestamp(
                strict_metadata["effective_available_at"],
                field_name="effective_available_at",
            )
            row_available = parse_timestamp(
                row["available_at"], field_name="available_at"
            )
        except ValueError as exc:  # pragma: no cover - normalized above
            raise ProductionObservationBundleError(
                "production_observation_chain_local_mapping_invalid"
            ) from exc
        if (
            strict_metadata["target_trade_date"] != target
            or strict_effective != row_available
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_local_mapping_invalid"
            )
        local_binding_by_hash[content_hash] = {
            "target": target,
            "strict_metadata": strict_metadata,
            "generic_digests": mapped_generic,
        }
    parent_generation_id = str(manifest.get("parent_generation_id") or "")
    parent_pointer_sha256 = str(manifest.get("parent_pointer_sha256") or "")
    if chain_schema == PRODUCTION_OBSERVATION_BUNDLE_SCHEMA:
        if (
            len(local_rows) != 3
            or parent_generation_id
            or parent_pointer_sha256
            or added_hashes != sorted(row_hashes)
            or metadata.get("atomic_combined_publication") is not True
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_bootstrap_lineage_invalid"
            )
        for field_name in (
            "official_bundle_manifest_sha256",
            "official_plan_sha256",
            "local_bootstrap_plan_sha256",
        ):
            _required_sha256(
                metadata.get(field_name),
                blocker=(
                    "local_market_observation_production_chain_hash_invalid:"
                    + field_name
                ),
            )
        if official_bundle_hashes != {
            str(metadata["official_bundle_manifest_sha256"])
        }:
            raise ProductionObservationBundleError(
                "production_observation_chain_official_bundle_binding_invalid"
            )
        plan_digest = str(metadata["local_bootstrap_plan_sha256"])
        if (
            plan_digest not in generic_local_evidence
            or any(
                plan_digest
                not in local_binding_by_hash[content_hash]["generic_digests"]
                for content_hash in local_binding_by_hash
            )
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_bootstrap_plan_role_invalid"
            )
        ordered_bootstrap = sorted(
            local_binding_by_hash,
            key=lambda content_hash: local_binding_by_hash[content_hash][
                "target"
            ],
        )
        expected_coverage_hash = canonical_hash(
            {
                "values": [
                    local_binding_by_hash[content_hash]["strict_metadata"][
                        "coverage_contract_sha256"
                    ]
                    for content_hash in ordered_bootstrap
                ]
            }
        )
        strict_effective_values = [
            str(
                local_binding_by_hash[content_hash]["strict_metadata"][
                    "effective_available_at"
                ]
            )
            for content_hash in ordered_bootstrap
        ]
        expected_effective = max(
            strict_effective_values,
            key=lambda value: parse_timestamp(
                value, field_name="effective_available_at"
            ),
        )
        if (
            metadata["local_coverage_contract_sha256"]
            != expected_coverage_hash
            or metadata["local_effective_available_at"] != expected_effective
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_bootstrap_local_binding_invalid"
            )
    else:
        update_mode = str(metadata.get("update_mode") or "")
        parent_as_of = str(metadata.get("parent_as_of") or "")
        try:
            parent_cutoff = parse_timestamp(
                metadata.get("parent_decision_cutoff_at"),
                field_name="parent_decision_cutoff_at",
            )
        except ValueError as exc:
            raise ProductionObservationBundleError(
                "production_observation_chain_update_parent_time_invalid"
            ) from exc
        if (
            len(local_rows) < 4
            or update_mode
            not in {"next_date_append", "same_date_correction"}
            or not _COMPACT_DATE_RE.fullmatch(parent_as_of)
            or not _GENERATION_ID_RE.fullmatch(parent_generation_id)
            or parent_generation_id == generation_id
            or parent_cutoff > decision_cutoff
            or (
                update_mode == "next_date_append"
                and not parent_as_of < logical_as_of
            )
            or (
                update_mode == "same_date_correction"
                and parent_as_of != logical_as_of
            )
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_update_lineage_invalid"
            )
        _required_sha256(
            parent_pointer_sha256,
            blocker="local_market_observation_update_parent_pointer_invalid",
        )
        target = str(metadata.get("local_target_trade_date") or "")
        if len(added_hashes) != 1 or added_hashes[0] not in (
            local_binding_by_hash
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_update_target_invalid"
            )
        added_hash = str(added_hashes[0])
        added_binding = local_binding_by_hash[added_hash]
        if target != logical_as_of or added_binding["target"] != target:
            raise ProductionObservationBundleError(
                "production_observation_chain_update_target_invalid"
            )
        declared_source_digests = {
            _required_sha256(
                metadata.get(field_name),
                blocker=(
                    "local_market_observation_production_chain_hash_invalid:"
                    + field_name
                ),
            )
            for field_name in (
                "local_snapshot_manifest_sha256",
                "local_coverage_manifest_sha256",
                "local_scope_artifact_sha256",
            )
        }
        # Snapshot and coverage may be the same immutable manifest, so their
        # two declared roles can collapse to one source hash.
        if (
            len(declared_source_digests) not in {2, 3}
            or not declared_source_digests.issubset(
                added_binding["generic_digests"]
            )
            or metadata["local_coverage_contract_sha256"]
            != added_binding["strict_metadata"]["coverage_contract_sha256"]
            or metadata["local_effective_available_at"]
            != added_binding["strict_metadata"]["effective_available_at"]
        ):
            raise ProductionObservationBundleError(
                "production_observation_chain_update_local_binding_invalid"
            )
    return mapping


def _validated_existing_production_chain(
    rows: Sequence[Mapping[str, Any]],
    pointer: Mapping[str, Any],
    *,
    canonical_root: str | Path | None = None,
) -> _ExistingProductionChain:
    manifest = pointer.get("generation_manifest")
    if not isinstance(manifest, Mapping) or manifest.get(
        "schema_version"
    ) != "macro-observation-generation.v2":
        raise ProductionObservationBundleError(
            "local_market_observation_existing_generation_v2_required"
        )
    metadata = manifest.get("metadata")
    pointer_metadata = pointer.get("metadata")
    if not isinstance(metadata, Mapping) or not isinstance(
        pointer_metadata, Mapping
    ):
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_metadata_missing"
        )
    if dict(metadata) != dict(pointer_metadata):
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_metadata_mismatch"
        )
    strict_mapping = validate_production_observation_chain(
        rows,
        generation_manifest=manifest,
        pointer_metadata=pointer_metadata,
    )
    chain_schema = str(metadata.get("schema_version") or "")
    if chain_schema not in _PRODUCTION_CHAIN_SCHEMAS:
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_schema_invalid"
        )
    if str(metadata.get("market") or "").upper() != "CN":
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_market_invalid"
        )
    logical_as_of = str(metadata.get("as_of") or "")
    if not _COMPACT_DATE_RE.fullmatch(logical_as_of):
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_as_of_invalid"
        )
    try:
        logical_cutoff = published_cutoff(logical_as_of)
        decision_cutoff = parse_timestamp(
            metadata.get("decision_cutoff_at"),
            field_name="decision_cutoff_at",
        )
        parse_timestamp(
            metadata.get("local_effective_available_at"),
            field_name="local_effective_available_at",
        )
    except ValueError as exc:
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_time_invalid"
        ) from exc
    if decision_cutoff < logical_cutoff:
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_cutoff_before_as_of"
        )
    local_periods = sorted(
        str(row.get("period_end") or "").replace("-", "")
        for row in rows
        if str(row.get("indicator_id") or "") == "market.breadth"
        and str(row.get("source_system") or "") == "local_strict_parquet"
    )
    if not local_periods or local_periods[-1] != logical_as_of:
        raise ProductionObservationBundleError(
            "local_market_observation_production_chain_as_of_row_mismatch"
        )
    for field_name in (
        "local_snapshot_manifest_sha256",
        "local_coverage_contract_sha256",
        "validated_snapshot_hash",
    ):
        _required_sha256(
            metadata.get(field_name),
            blocker=(
                "local_market_observation_production_chain_hash_invalid:"
                + field_name
            ),
        )
    if chain_schema == PRODUCTION_OBSERVATION_BUNDLE_SCHEMA:
        for field_name in (
            "official_bundle_manifest_sha256",
            "official_plan_sha256",
            "local_bootstrap_plan_sha256",
        ):
            _required_sha256(
                metadata.get(field_name),
                blocker=(
                    "local_market_observation_production_chain_hash_invalid:"
                    + field_name
                ),
            )
        if metadata.get("atomic_combined_publication") is not True:
            raise ProductionObservationBundleError(
                "local_market_observation_bootstrap_not_atomic"
            )
    else:
        for field_name in (
            "local_coverage_manifest_sha256",
            "local_scope_artifact_sha256",
        ):
            _required_sha256(
                metadata.get(field_name),
                blocker=(
                    "local_market_observation_production_chain_hash_invalid:"
                    + field_name
                ),
            )
        if not _COMPACT_DATE_RE.fullmatch(
            str(metadata.get("local_target_trade_date") or "")
        ):
            raise ProductionObservationBundleError(
                "local_market_observation_update_trade_date_invalid"
            )
        if str(metadata.get("local_target_trade_date")) != logical_as_of:
            raise ProductionObservationBundleError(
                "local_market_observation_update_trade_date_as_of_mismatch"
            )
        parent_as_of = str(metadata.get("parent_as_of") or "")
        if not _COMPACT_DATE_RE.fullmatch(parent_as_of):
            raise ProductionObservationBundleError(
                "local_market_observation_update_parent_as_of_invalid"
            )
        try:
            parent_decision_cutoff = parse_timestamp(
                metadata.get("parent_decision_cutoff_at"),
                field_name="parent_decision_cutoff_at",
            )
        except ValueError as exc:
            raise ProductionObservationBundleError(
                "local_market_observation_update_parent_cutoff_invalid"
            ) from exc
        if published_cutoff(parent_as_of) > parent_decision_cutoff:
            raise ProductionObservationBundleError(
                "local_market_observation_update_parent_cutoff_before_as_of"
            )
        if parent_as_of > logical_as_of:
            raise ProductionObservationBundleError(
                "local_market_observation_update_parent_as_of_after_child"
            )
        if parent_decision_cutoff > decision_cutoff:
            raise ProductionObservationBundleError(
                "local_market_observation_update_parent_cutoff_after_child"
            )
        parent_generation_id = str(
            manifest.get("parent_generation_id") or ""
        )
        if not parent_generation_id:
            raise ProductionObservationBundleError(
                "local_market_observation_update_parent_missing"
            )
        _required_sha256(
            manifest.get("parent_pointer_sha256"),
            blocker="local_market_observation_update_parent_pointer_invalid",
        )
        if canonical_root is not None:
            _parent_rows, parent_pointer = load_observations(
                canonical_root,
                generation_id=parent_generation_id,
            )
            parent_manifest = parent_pointer.get("generation_manifest")
            parent_metadata = (
                parent_manifest.get("metadata")
                if isinstance(parent_manifest, Mapping)
                else None
            )
            if not isinstance(parent_metadata, Mapping):
                raise ProductionObservationBundleError(
                    "local_market_observation_update_parent_metadata_missing"
                )
            actual_parent_as_of = str(parent_metadata.get("as_of") or "")
            try:
                actual_parent_cutoff = parse_timestamp(
                    parent_metadata.get("decision_cutoff_at"),
                    field_name="parent_decision_cutoff_at",
                )
            except ValueError as exc:
                raise ProductionObservationBundleError(
                    "local_market_observation_update_parent_metadata_invalid"
                ) from exc
            if (
                actual_parent_as_of != parent_as_of
                or actual_parent_cutoff != parent_decision_cutoff
            ):
                raise ProductionObservationBundleError(
                    "local_market_observation_update_parent_time_binding_"
                    "mismatch"
                )

    raw_mapping = manifest.get("observation_evidence")
    if not isinstance(raw_mapping, Mapping):
        raise ProductionObservationBundleError(
            "local_market_observation_existing_evidence_mapping_missing"
        )
    row_hashes = {str(row.get("content_hash") or "") for row in rows}
    if set(raw_mapping) != row_hashes:
        raise ProductionObservationBundleError(
            "local_market_observation_existing_evidence_mapping_incomplete"
        )
    normalized: dict[str, list[str]] = {}
    for content_hash in sorted(row_hashes):
        raw_digests = raw_mapping.get(content_hash)
        if not isinstance(raw_digests, list) or not raw_digests:
            raise ProductionObservationBundleError(
                "local_market_observation_existing_evidence_mapping_invalid"
            )
        digests = sorted(set(str(value or "") for value in raw_digests))
        if len(digests) != len(raw_digests) or any(
            not _SHA256_RE.fullmatch(digest) for digest in digests
        ):
            raise ProductionObservationBundleError(
                "local_market_observation_existing_evidence_mapping_invalid"
            )
        normalized[content_hash] = digests
    return _ExistingProductionChain(
        observation_mapping=strict_mapping,
        logical_as_of=logical_as_of,
        decision_cutoff_at=decision_cutoff,
        generation_id=str(pointer.get("generation_id") or ""),
    )


def _strict_publication_readback(
    *,
    final_rows: Sequence[Mapping[str, Any]],
    generation_manifest: Mapping[str, Any],
    expected_rows: Sequence[Mapping[str, Any]],
    expected_snapshot: Mapping[str, Any],
    as_of: str,
    decision_cutoff_at: Any | None,
    expected_incoming_evidence_hashes: set[str],
    expected_final_evidence_mapping: Mapping[str, Sequence[str]],
) -> None:
    normalized = _normalized_rows(final_rows)
    if normalized != list(expected_rows):
        raise ProductionObservationBundleError(
            "production_observation_readback_rows_mismatch"
        )
    rebuilt = build_macro_snapshot(
        normalized,
        market="CN",
        as_of=as_of,
        decision_cutoff_at=decision_cutoff_at,
    ).to_dict()
    if rebuilt != dict(expected_snapshot):
        raise ProductionObservationBundleError(
            "production_observation_readback_snapshot_mismatch"
        )
    if generation_manifest.get("schema_version") != (
        "macro-observation-generation.v2"
    ):
        raise ProductionObservationBundleError(
            "production_observation_readback_evidence_manifest_invalid"
        )
    files = generation_manifest.get("evidence_files")
    mapping = generation_manifest.get("observation_evidence")
    if not isinstance(files, list) or not isinstance(mapping, Mapping):
        raise ProductionObservationBundleError(
            "production_observation_readback_evidence_shape_invalid"
        )
    persisted_hashes = {
        str(item.get("sha256") or "")
        for item in files
        if isinstance(item, Mapping)
    }
    if not expected_incoming_evidence_hashes.issubset(persisted_hashes):
        raise ProductionObservationBundleError(
            "production_observation_readback_evidence_set_mismatch"
        )
    expected_mapping = {
        content_hash: sorted(set(digests))
        for content_hash, digests in expected_final_evidence_mapping.items()
    }
    persisted_mapping = {
        str(content_hash): sorted(set(digests))
        for content_hash, digests in mapping.items()
        if isinstance(digests, list)
    }
    final_row_hashes = {str(row["content_hash"]) for row in normalized}
    if (
        set(mapping) != final_row_hashes
        or set(expected_mapping) != final_row_hashes
        or persisted_mapping != expected_mapping
    ):
        raise ProductionObservationBundleError(
            "production_observation_readback_evidence_mapping_mismatch"
        )
    metadata = generation_manifest.get("metadata")
    validate_production_observation_chain(
        normalized,
        generation_manifest=generation_manifest,
        pointer_metadata=metadata if isinstance(metadata, Mapping) else None,
    )


def publish_macro_production_observation_bundle(
    *,
    official_bundle_manifest_path: str | Path,
    expected_official_bundle_manifest_sha256: str,
    expected_official_plan_sha256: str,
    local_bootstrap_plan_path: str | Path,
    expected_local_bootstrap_plan_sha256: str,
    as_of: str,
    canonical_observations_root: str | Path,
    run_id: str,
    expected_pointer_sha256: str,
) -> dict[str, Any]:
    """Publish the exact 36-official plus 3-local production scope."""

    as_of_value = str(as_of or "").strip()
    if not as_of_value:
        raise ProductionObservationBundleError(
            "production_observation_as_of_missing"
        )
    decision_cutoff = published_cutoff(as_of_value)
    if decision_cutoff > datetime.now(UTC):
        raise ProductionObservationBundleError(
            "production_observation_as_of_in_future"
        )
    official, official_evidence = _official_evidence_inputs(
        manifest_path=official_bundle_manifest_path,
        expected_manifest_sha256=expected_official_bundle_manifest_sha256,
        expected_plan_sha256=expected_official_plan_sha256,
    )
    local, local_manifest, local_evidence = _bootstrap_local_evidence_inputs(
        plan_path=local_bootstrap_plan_path,
        expected_plan_sha256=expected_local_bootstrap_plan_sha256,
        as_of=as_of_value,
    )
    observations = tuple(official.observations) + tuple(local)
    target_as_of = local[-1].period_end.replace("-", "")
    snapshot = _validated_production_snapshot(
        observations,
        as_of=target_as_of,
        decision_cutoff_at=decision_cutoff,
    )
    evidence = _merge_evidence_inputs(official_evidence, local_evidence)
    if len(evidence.observation_mapping) != _PRODUCTION_OBSERVATION_COUNT:
        raise ProductionObservationBundleError(
            "production_observation_evidence_mapping_count_invalid"
        )

    existing, current_pointer, _current_generation = _load_current(
        canonical_observations_root,
        expected_pointer_sha256=expected_pointer_sha256,
    )
    incoming_rows = _normalized_rows(observations)
    incoming_hashes = {row["content_hash"] for row in incoming_rows}
    existing_hashes = {row["content_hash"] for row in existing}
    if existing and existing_hashes != incoming_hashes:
        raise ProductionObservationBundleError(
            "production_observation_root_not_exact_target"
        )
    if existing and _normalized_rows(existing) != incoming_rows:
        raise ProductionObservationBundleError(
            "production_observation_existing_rows_mismatch"
        )

    def _validate_before_pointer_switch(
        final_rows: Sequence[Mapping[str, Any]],
        generation_manifest: Mapping[str, Any],
    ) -> None:
        _strict_publication_readback(
            final_rows=final_rows,
            generation_manifest=generation_manifest,
            expected_rows=incoming_rows,
            expected_snapshot=snapshot,
            as_of=target_as_of,
            decision_cutoff_at=decision_cutoff,
            expected_incoming_evidence_hashes=set(evidence.bodies),
            expected_final_evidence_mapping=evidence.observation_mapping,
        )

    publication = publish_observations(
        observations,
        root=canonical_observations_root,
        run_id=run_id,
        expected_pointer_sha256=current_pointer,
        metadata={
            "schema_version": PRODUCTION_OBSERVATION_BUNDLE_SCHEMA,
            "market": "CN",
            "as_of": target_as_of,
            "decision_cutoff_at": decision_cutoff.isoformat(),
            "official_bundle_manifest_sha256": (
                expected_official_bundle_manifest_sha256
            ),
            "official_plan_sha256": expected_official_plan_sha256,
            "local_snapshot_manifest_sha256": (
                local_manifest["snapshot_manifest_sha256"]
            ),
            "local_bootstrap_plan_sha256": (
                expected_local_bootstrap_plan_sha256
            ),
            "local_coverage_contract_sha256": str(
                local_manifest.get("coverage_contract_sha256") or ""
            ),
            "local_effective_available_at": str(
                local_manifest.get("effective_available_at") or ""
            ),
            "validated_snapshot_hash": snapshot["snapshot_hash"],
            "atomic_combined_publication": True,
        },
        evidence_bytes=evidence.bodies,
        evidence_metadata=evidence.metadata,
        observation_evidence=evidence.observation_mapping,
        precommit_validator=_validate_before_pointer_switch,
    )
    generation_manifest = publication.get("generation_manifest")
    raw_generation_evidence_count = (
        generation_manifest.get("evidence_file_count")
        if isinstance(generation_manifest, Mapping)
        else None
    )
    generation_evidence_count = (
        raw_generation_evidence_count
        if isinstance(raw_generation_evidence_count, int)
        and not isinstance(raw_generation_evidence_count, bool)
        else len(evidence.bodies)
    )
    next_pointer_sha = str(publication.get("pointer_sha256") or "")
    return {
        "schema_version": PRODUCTION_OBSERVATION_BUNDLE_SCHEMA,
        "status": "OK",
        "market": "CN",
        "as_of": target_as_of,
        "decision_cutoff_at": decision_cutoff.isoformat(),
        "run_id": run_id,
        "promoted": bool(publication.get("promoted")),
        "observation_count": len(incoming_rows),
        "indicator_count": _PRODUCTION_INDICATOR_COUNT,
        "history_length_per_indicator": _HISTORY_LENGTH,
        "official_observation_count": len(official.observations),
        "local_observation_count": len(local),
        "official_bundle_manifest_sha256": (
            expected_official_bundle_manifest_sha256
        ),
        "official_plan_sha256": expected_official_plan_sha256,
        "local_bootstrap_plan_path": str(
            local_manifest.get("bootstrap_plan_path") or ""
        ),
        "local_bootstrap_plan_sha256": (
            expected_local_bootstrap_plan_sha256
        ),
        "local_snapshot_manifest_sha256": str(
            local_manifest.get("snapshot_manifest_sha256") or ""
        ),
        "local_coverage_contract_sha256": str(
            local_manifest.get("coverage_contract_sha256") or ""
        ),
        "local_effective_available_at": str(
            local_manifest.get("effective_available_at") or ""
        ),
        "local_target_trade_dates": list(
            local_manifest.get("target_trade_dates") or []
        ),
        "incoming_evidence_file_count": len(evidence.bodies),
        "official_evidence_file_count": evidence.official_file_count,
        "local_evidence_file_count": evidence.local_file_count,
        "generation_evidence_file_count": generation_evidence_count,
        "generation_id": str(publication.get("generation_id") or ""),
        "manifest_sha256": str(publication.get("manifest_sha256") or ""),
        "content_set_hash": str(publication.get("content_set_hash") or ""),
        "previous_pointer_sha256": current_pointer,
        "pointer_sha256": next_pointer_sha,
        "snapshot_hash": str(snapshot["snapshot_hash"]),
        "snapshot_readiness_status": str(snapshot["readiness_status"]),
        "snapshot_national_coverage": float(snapshot["coverage"]["national"]),
        "snapshot_blockers": list(snapshot["blockers"]),
        "atomic_combined_publication": True,
        "strict_readback_validated": True,
    }


def publish_local_market_breadth_update(
    *,
    snapshot_manifest_path: str | Path,
    expected_snapshot_manifest_sha256: str,
    coverage_manifest_path: str | Path,
    expected_coverage_manifest_sha256: str,
    target_trade_date: str,
    scope_artifact_path: str | Path,
    expected_scope_artifact_sha256: str,
    as_of: str,
    canonical_observations_root: str | Path,
    run_id: str,
    expected_pointer_sha256: str,
) -> dict[str, Any]:
    """Append one coverage-certified local breadth observation by CAS."""

    as_of_value = str(as_of or "").strip()
    if not as_of_value:
        raise ProductionObservationBundleError(
            "production_observation_as_of_missing"
        )
    decision_cutoff = published_cutoff(as_of_value)
    if decision_cutoff > datetime.now(UTC):
        raise ProductionObservationBundleError(
            "production_observation_as_of_in_future"
        )
    binding = _LocalTargetBinding(
        target_trade_date=str(target_trade_date),
        snapshot_manifest_path=_absolute_path(
            snapshot_manifest_path,
            blocker="production_local_snapshot_manifest_path_unsafe",
        ),
        expected_snapshot_manifest_sha256=_required_sha256(
            expected_snapshot_manifest_sha256,
            blocker="production_local_snapshot_manifest_sha256_invalid",
        ),
        coverage_manifest_path=_absolute_path(
            coverage_manifest_path,
            blocker="production_local_coverage_manifest_path_unsafe",
        ),
        expected_coverage_manifest_sha256=_required_sha256(
            expected_coverage_manifest_sha256,
            blocker="production_local_coverage_manifest_sha256_invalid",
        ),
        scope_artifact_path=_absolute_path(
            scope_artifact_path,
            blocker="production_local_scope_artifact_path_unsafe",
        ),
        expected_scope_artifact_sha256=_required_sha256(
            expected_scope_artifact_sha256,
            blocker="production_local_scope_artifact_sha256_invalid",
        ),
    )
    if not _COMPACT_DATE_RE.fullmatch(binding.target_trade_date):
        raise ProductionObservationBundleError(
            "production_local_target_trade_date_invalid"
        )
    existing, current_pointer, current_generation = _load_current(
        canonical_observations_root,
        expected_pointer_sha256=expected_pointer_sha256,
    )
    if not existing:
        raise ProductionObservationBundleError(
            "local_market_observation_existing_generation_required"
        )
    parent_chain = _validated_existing_production_chain(
        existing,
        current_generation,
        canonical_root=canonical_observations_root,
    )
    if binding.target_trade_date < parent_chain.logical_as_of:
        raise ProductionObservationBundleError(
            "local_market_observation_target_trade_date_rollback"
        )
    if decision_cutoff < parent_chain.decision_cutoff_at:
        raise ProductionObservationBundleError(
            "local_market_observation_decision_cutoff_rollback"
        )
    same_date = binding.target_trade_date == parent_chain.logical_as_of
    if same_date and str(run_id) == parent_chain.generation_id:
        raise ProductionObservationBundleError(
            "local_market_observation_same_date_run_id_reuse"
        )

    local, local_manifest, evidence = _single_local_evidence_inputs(
        binding=binding,
        as_of=as_of_value,
    )
    expected_final_mapping = {
        content_hash: list(digests)
        for content_hash, digests in parent_chain.observation_mapping.items()
    }
    expected_by_hash = {
        row["content_hash"]: row for row in _normalized_rows(existing)
    }
    normalized_local = _normalized_rows(local)
    local_content_hash = str(normalized_local[0]["content_hash"])
    content_already_present = local_content_hash in expected_by_hash
    if same_date and not content_already_present:
        existing_same_date = [
            row
            for row in expected_by_hash.values()
            if row["indicator_id"] == "market.breadth"
            and row["source_system"] == "local_strict_parquet"
            and str(row["period_end"]).replace("-", "")
            == binding.target_trade_date
        ]
        if not existing_same_date:
            raise ProductionObservationBundleError(
                "local_market_observation_same_date_parent_row_missing"
            )
        latest_parent_available_at = max(
            parse_timestamp(
                row["available_at"],
                field_name="available_at",
            )
            for row in existing_same_date
        )
        incoming_available_at = parse_timestamp(
            normalized_local[0]["available_at"],
            field_name="available_at",
        )
        if incoming_available_at <= latest_parent_available_at:
            raise ProductionObservationBundleError(
                "local_market_observation_same_date_correction_"
                "available_at_not_increasing"
            )
    update_mode = (
        "same_date_idempotent_retry"
        if same_date and content_already_present
        else "same_date_correction"
        if same_date
        else "next_date_append"
    )
    for row in normalized_local:
        expected_by_hash[row["content_hash"]] = row
    for content_hash, digests in evidence.observation_mapping.items():
        normalized_digests = sorted(set(digests))
        previous = expected_final_mapping.get(content_hash)
        if previous is not None and previous != normalized_digests:
            raise ProductionObservationBundleError(
                "local_market_observation_existing_evidence_drift"
            )
        expected_final_mapping[content_hash] = normalized_digests
    expected_rows = [expected_by_hash[key] for key in sorted(expected_by_hash)]
    expected_snapshot = build_macro_snapshot(
        expected_rows,
        market="CN",
        as_of=binding.target_trade_date,
        decision_cutoff_at=decision_cutoff,
    ).to_dict()
    if (
        expected_snapshot["readiness_status"] != "pass"
        or expected_snapshot["blockers"]
        or float(expected_snapshot["coverage"]["national"])
        < _EXPECTED_NATIONAL_COVERAGE
    ):
        raise ProductionObservationBundleError(
            "local_market_observation_updated_snapshot_not_ready:"
            + ",".join(expected_snapshot["blockers"])
        )

    def _validate_before_pointer_switch(
        final_rows: Sequence[Mapping[str, Any]],
        generation_manifest: Mapping[str, Any],
    ) -> None:
        _strict_publication_readback(
            final_rows=final_rows,
            generation_manifest=generation_manifest,
            expected_rows=expected_rows,
            expected_snapshot=expected_snapshot,
            as_of=binding.target_trade_date,
            decision_cutoff_at=decision_cutoff,
            expected_incoming_evidence_hashes=set(evidence.bodies),
            expected_final_evidence_mapping=expected_final_mapping,
        )

    publication = publish_observations(
        local,
        root=canonical_observations_root,
        run_id=run_id,
        expected_pointer_sha256=current_pointer,
        metadata={
            "schema_version": LOCAL_MARKET_OBSERVATION_PUBLICATION_SCHEMA,
            "market": "CN",
            "as_of": binding.target_trade_date,
            "decision_cutoff_at": decision_cutoff.isoformat(),
            "parent_as_of": parent_chain.logical_as_of,
            "parent_decision_cutoff_at": (
                parent_chain.decision_cutoff_at.isoformat()
            ),
            "update_mode": update_mode,
            "local_snapshot_manifest_sha256": (
                binding.expected_snapshot_manifest_sha256
            ),
            "local_coverage_manifest_sha256": (
                binding.expected_coverage_manifest_sha256
            ),
            "local_scope_artifact_sha256": (
                binding.expected_scope_artifact_sha256
            ),
            "local_target_trade_date": binding.target_trade_date,
            "local_coverage_contract_sha256": str(
                local_manifest.get("coverage_contract_sha256") or ""
            ),
            "local_effective_available_at": str(
                local_manifest.get("effective_available_at") or ""
            ),
            "validated_snapshot_hash": expected_snapshot["snapshot_hash"],
        },
        evidence_bytes=evidence.bodies,
        evidence_metadata=evidence.metadata,
        observation_evidence=evidence.observation_mapping,
        precommit_validator=_validate_before_pointer_switch,
    )
    next_pointer_sha = str(publication.get("pointer_sha256") or "")
    return {
        "schema_version": LOCAL_MARKET_OBSERVATION_PUBLICATION_SCHEMA,
        "status": "OK",
        "market": "CN",
        "as_of": binding.target_trade_date,
        "decision_cutoff_at": decision_cutoff.isoformat(),
        "parent_as_of": parent_chain.logical_as_of,
        "parent_decision_cutoff_at": (
            parent_chain.decision_cutoff_at.isoformat()
        ),
        "update_mode": update_mode,
        "run_id": run_id,
        "promoted": bool(publication.get("promoted")),
        "local_observation_count": len(local),
        "local_snapshot_manifest_sha256": (
            binding.expected_snapshot_manifest_sha256
        ),
        "local_coverage_manifest_sha256": (
            binding.expected_coverage_manifest_sha256
        ),
        "local_scope_artifact_sha256": (
            binding.expected_scope_artifact_sha256
        ),
        "local_coverage_contract_sha256": str(
            local_manifest.get("coverage_contract_sha256") or ""
        ),
        "local_effective_available_at": str(
            local_manifest.get("effective_available_at") or ""
        ),
        "local_target_trade_dates": [binding.target_trade_date],
        "incoming_evidence_file_count": len(evidence.bodies),
        "generation_id": str(publication.get("generation_id") or ""),
        "manifest_sha256": str(publication.get("manifest_sha256") or ""),
        "content_set_hash": str(publication.get("content_set_hash") or ""),
        "previous_pointer_sha256": current_pointer,
        "pointer_sha256": next_pointer_sha,
        "snapshot_hash": str(expected_snapshot["snapshot_hash"]),
        "snapshot_readiness_status": str(
            expected_snapshot["readiness_status"]
        ),
        "snapshot_national_coverage": float(
            expected_snapshot["coverage"]["national"]
        ),
        "snapshot_blockers": list(expected_snapshot["blockers"]),
        "strict_readback_validated": True,
    }


__all__ = [
    "LOCAL_BREADTH_BOOTSTRAP_PLAN_SCHEMA",
    "LOCAL_MARKET_OBSERVATION_PUBLICATION_SCHEMA",
    "PRODUCTION_OBSERVATION_BUNDLE_SCHEMA",
    "ProductionObservationBundleError",
    "publish_local_market_breadth_update",
    "publish_macro_production_observation_bundle",
    "validate_production_observation_chain",
]
