from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import stat
import tempfile
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.etree import ElementTree
from zoneinfo import ZoneInfo


SCHEMA_VERSION = "pevc_thesis.v1"
APPROVAL_LEDGER_SCHEMA_VERSION = "pevc_approval_ledger.v2"
APPROVAL_LEDGER_FILENAME = "pevc_approval_ledger.jsonl"
APPROVAL_KEY_FILENAME = "pevc_approval.key"
APPROVAL_WAL_FILENAME = "pevc_approval_transaction.wal.json"
APPROVAL_DRAFT_ARCHIVE_DIRNAME = "approved_drafts"
APPROVAL_TRANSACTION_DIRNAME = "approval_transactions"
MIGRATION_EVIDENCE_ARCHIVE_DIRNAME = "migration_evidence"
APPROVAL_DRAFT_HASH_METADATA_KEY = "approval_draft_hash"
DEFAULT_PRIVATE_ROOT = Path("private/theme_knowledge")
DEFAULT_CANONICAL_PATH = DEFAULT_PRIVATE_ROOT / "pevc_theses.jsonl"
DEFAULT_DRAFT_DIR = DEFAULT_PRIVATE_ROOT / "drafts"
_DATE_FORMATS = ("%Y-%m-%d", "%Y%m%d")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_GENESIS_EVENT_HASH = "0" * 64
_SHANGHAI_TIMEZONE = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class PeVcThesis:
    thesis_id: str
    theme_id: str
    version: str
    status: str
    tam: str
    technology_maturity: float
    bottlenecks: tuple[str, ...]
    moat_strength: float
    customer_validation: float
    commercialization_stage: float
    milestones: tuple[str, ...]
    kill_criteria: tuple[str, ...]
    valuation_ceiling: float | None
    horizon_months: int
    confidence: float
    review_by: str
    available_at: str
    prior_score: float | None = None
    source_type: str = "local_structured"
    source_ref: str = ""
    source_hash: str = ""
    approved_at: str = ""
    content_hash: str = ""
    notes: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        require_approved: bool = False,
    ) -> "PeVcThesis":
        schema = str(payload.get("schema_version") or SCHEMA_VERSION).strip()
        if schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported pevc schema_version={schema}")
        thesis_id = str(payload.get("thesis_id") or "").strip()
        theme_id = str(payload.get("theme_id") or "").strip()
        version = str(payload.get("version") or "").strip()
        status = str(payload.get("status") or "draft").strip().lower()
        if not thesis_id or not theme_id or not version:
            raise ValueError("thesis_id, theme_id, and version are required")
        if "::" not in theme_id:
            raise ValueError("theme_id must be namespaced")
        if status not in {"draft", "approved", "superseded", "rejected"}:
            raise ValueError(f"invalid thesis status={status}")
        if require_approved and status != "approved":
            raise ValueError("only approved theses may enter the canonical knowledge base")
        tam = str(payload.get("tam") or "").strip()
        bottlenecks = tuple(_texts(payload.get("bottlenecks")))
        milestones = tuple(_texts(payload.get("milestones")))
        kill_criteria = tuple(_texts(payload.get("kill_criteria")))
        review_by = _date_text(payload.get("review_by"))
        available_at = _date_text(payload.get("available_at"))
        if not tam:
            raise ValueError("tam is required")
        if not bottlenecks:
            raise ValueError("bottlenecks requires at least one item")
        if not milestones:
            raise ValueError("milestones requires at least one item")
        if not kill_criteria:
            raise ValueError("kill_criteria requires at least one item")
        valuation_ceiling = _optional_nonnegative(payload.get("valuation_ceiling"))
        horizon_months = int(_finite(payload.get("horizon_months"), 0.0))
        if horizon_months <= 0:
            raise ValueError("horizon_months must be positive")
        thesis = cls(
            thesis_id=thesis_id,
            theme_id=theme_id,
            version=version,
            status=status,
            tam=tam,
            technology_maturity=_unit(payload.get("technology_maturity"), "technology_maturity"),
            bottlenecks=bottlenecks,
            moat_strength=_unit(payload.get("moat_strength"), "moat_strength"),
            customer_validation=_unit(payload.get("customer_validation"), "customer_validation"),
            commercialization_stage=_unit(
                payload.get("commercialization_stage"),
                "commercialization_stage",
            ),
            milestones=milestones,
            kill_criteria=kill_criteria,
            valuation_ceiling=valuation_ceiling,
            horizon_months=horizon_months,
            confidence=_unit(payload.get("confidence"), "confidence"),
            review_by=review_by,
            available_at=available_at,
            prior_score=_optional_unit(payload.get("prior_score"), "prior_score"),
            source_type=str(payload.get("source_type") or "local_structured").strip(),
            source_ref=str(payload.get("source_ref") or "").strip(),
            source_hash=str(payload.get("source_hash") or "").strip(),
            approved_at=_date_text(payload.get("approved_at"), required=False),
            content_hash=str(payload.get("content_hash") or "").strip(),
            notes=str(payload.get("notes") or "").strip(),
            metadata=dict(payload.get("metadata") or {}),
        )
        expected_hash = thesis.compute_content_hash()
        if thesis.content_hash and thesis.content_hash != expected_hash:
            raise ValueError("content_hash does not match canonical thesis content")
        if require_approved:
            if not thesis.approved_at:
                raise ValueError("approved_at is required for canonical theses")
            if not _is_sha256(thesis.content_hash):
                raise ValueError("canonical content_hash must be SHA-256")
            if not _is_sha256(thesis.approval_draft_hash):
                raise ValueError(
                    "canonical metadata.approval_draft_hash must be SHA-256"
                )
        return thesis

    @property
    def approval_draft_hash(self) -> str:
        return str(
            self.metadata.get(APPROVAL_DRAFT_HASH_METADATA_KEY) or ""
        ).strip()

    def compute_content_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_hash=False),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "thesis_id": self.thesis_id,
            "theme_id": self.theme_id,
            "version": self.version,
            "status": self.status,
            "tam": self.tam,
            "technology_maturity": self.technology_maturity,
            "bottlenecks": list(self.bottlenecks),
            "moat_strength": self.moat_strength,
            "customer_validation": self.customer_validation,
            "commercialization_stage": self.commercialization_stage,
            "milestones": list(self.milestones),
            "kill_criteria": list(self.kill_criteria),
            "valuation_ceiling": self.valuation_ceiling,
            "horizon_months": self.horizon_months,
            "confidence": self.confidence,
            "review_by": self.review_by,
            "available_at": self.available_at,
            "prior_score": self.prior_score,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "source_hash": self.source_hash,
            "approved_at": self.approved_at,
            "notes": self.notes,
            "metadata": dict(self.metadata),
        }
        if include_hash:
            payload["content_hash"] = self.content_hash or self.compute_content_hash()
        return payload


class PeVcKnowledgeStore:
    def __init__(self, canonical_path: str | Path = DEFAULT_CANONICAL_PATH) -> None:
        self.canonical_path = Path(canonical_path)

    @property
    def approval_ledger_path(self) -> Path:
        return self.canonical_path.parent / APPROVAL_LEDGER_FILENAME

    @property
    def approval_key_path(self) -> Path:
        return self.canonical_path.parent / APPROVAL_KEY_FILENAME

    @property
    def transaction_wal_path(self) -> Path:
        return self.canonical_path.parent / APPROVAL_WAL_FILENAME

    def load(self, *, as_of: str | None = None) -> list[PeVcThesis]:
        return self._load(as_of=as_of, ignore_active_wal=False)

    def _load(
        self,
        *,
        as_of: str | None = None,
        ignore_active_wal: bool,
    ) -> list[PeVcThesis]:
        point = _strict_date(as_of) if as_of is not None else None
        key: bytes | None = None
        if self.approval_key_path.exists():
            key = _load_approval_key(self.approval_key_path)
        if self.transaction_wal_path.exists() and not ignore_active_wal:
            raise ValueError("PEVC approval transaction recovery is required")
        if not self.canonical_path.exists():
            orphan_paths = (
                self.approval_ledger_path,
                self.transaction_wal_path,
                self.canonical_path.parent / APPROVAL_DRAFT_ARCHIVE_DIRNAME,
                self.canonical_path.parent / APPROVAL_TRANSACTION_DIRNAME,
                self.canonical_path.parent / MIGRATION_EVIDENCE_ARCHIVE_DIRNAME,
            )
            if any(path.exists() for path in orphan_paths):
                raise ValueError("PEVC approval artifacts exist without canonical store")
            return []
        if stat.S_IMODE(self.canonical_path.stat().st_mode) != 0o600:
            raise ValueError("PEVC canonical permissions must be 0600")
        if key is None:
            key = _load_approval_key(self.approval_key_path)
        events = _load_approval_ledger(self.approval_ledger_path, key=key)
        canonical_sha256 = _file_sha256(self.canonical_path)
        if not events or events[-1]["canonical_after_sha256"] != canonical_sha256:
            raise ValueError("PEVC canonical SHA does not match approval ledger head")
        for event in events:
            _validate_draft_archive(event, root=self.canonical_path.parent)
            _validate_migration_evidence_archive(
                event,
                root=self.canonical_path.parent,
            )
        approved_identities = {_approval_identity(event) for event in events}
        theses: list[PeVcThesis] = []
        seen_revisions: set[
            tuple[str, tuple[tuple[int, int | str], ...]]
        ] = set()
        for line_number, raw_line in enumerate(
            self.canonical_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
                thesis = PeVcThesis.from_mapping(payload, require_approved=True)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid canonical PEVC record at line {line_number}: {exc}"
                ) from exc
            revision = _revision_identity(thesis.thesis_id, thesis.version)
            if revision in seen_revisions:
                raise ValueError(
                    "duplicate canonical PEVC revision "
                    f"thesis_id={thesis.thesis_id} version={thesis.version}"
                )
            seen_revisions.add(revision)
            identity = _thesis_approval_identity(thesis)
            if identity not in approved_identities:
                raise ValueError(
                    "canonical PEVC record has no matching approval event "
                    f"thesis_id={thesis.thesis_id} version={thesis.version}"
                )
            if point is not None:
                available = _parse_date(thesis.available_at)
                approved = _parse_date(thesis.approved_at)
                review_by = _parse_date(thesis.review_by)
                if (
                    available is None
                    or approved is None
                    or available > point
                    or approved > point
                    or (review_by is not None and review_by < point)
                ):
                    continue
            theses.append(thesis)
        theses.sort(key=_thesis_revision_key)
        return theses

    def approve_draft(
        self,
        draft_path: str | Path,
        *,
        expected_draft_hash: str = "",
        approved_at: str | None = None,
        migration_mode: bool = False,
        migration_evidence_file: str | Path | None = None,
        expected_migration_evidence_hash: str = "",
    ) -> PeVcThesis:
        key = _load_approval_key(self.approval_key_path)
        self._recover_pending_transaction(key)
        source = Path(draft_path)
        raw_bytes = source.read_bytes()
        draft_hash = hashlib.sha256(raw_bytes).hexdigest()
        if not expected_draft_hash:
            raise ValueError("expected_draft_hash is required")
        if draft_hash != expected_draft_hash:
            raise ValueError("draft hash changed since review")
        draft = json.loads(raw_bytes.decode("utf-8"))
        if not isinstance(draft, Mapping):
            raise ValueError("draft must contain an object")
        if str(draft.get("draft_status") or "") != "pending_approval":
            raise ValueError("draft_status must be pending_approval")
        thesis_payload = dict(draft.get("thesis") or {})
        if self.canonical_path.exists():
            existing = self._load(as_of=None, ignore_active_wal=True)
        else:
            if self.approval_ledger_path.exists():
                raise ValueError("PEVC approval ledger exists without canonical store")
            existing = []
        thesis_id = str(thesis_payload.get("thesis_id") or "").strip()
        version = str(thesis_payload.get("version") or "").strip()
        same_revision = next(
            (
                item
                for item in existing
                if _revision_identity(item.thesis_id, item.version)
                == _revision_identity(thesis_id, version)
            ),
            None,
        )
        if same_revision is not None:
            if same_revision.approval_draft_hash == draft_hash:
                return same_revision
            raise ValueError(
                "approved PEVC revision is immutable; submit a new version"
            )

        recorded_at = _now_utc()
        current_date = _shanghai_business_date(recorded_at).isoformat()
        approved_date = _date_text(approved_at or current_date)
        if approved_date > current_date:
            raise ValueError("future approved_at is not allowed")
        evidence_hash: str | None = None
        evidence_path_summary: str | None = None
        evidence_bytes: bytes | None = None
        if approved_date != current_date:
            if not migration_mode:
                raise ValueError(
                    "backdated approved_at requires explicit migration_mode"
                )
            evidence_hash, evidence_path_summary, evidence_bytes = (
                _validate_migration_evidence(
                migration_evidence_file,
                expected_hash=expected_migration_evidence_hash,
            )
            )
        else:
            if migration_mode:
                raise ValueError("migration_mode is only valid for backdated approval")
            if migration_evidence_file or expected_migration_evidence_hash:
                raise ValueError(
                    "non-migration approval must not carry migration evidence"
                )
        source_available = _date_text(thesis_payload.get("available_at"))
        canonical_available = max(source_available, current_date)
        approval_metadata = dict(thesis_payload.get("metadata") or {})
        approval_metadata[APPROVAL_DRAFT_HASH_METADATA_KEY] = draft_hash
        thesis_payload.update(
            {
                "status": "approved",
                "approved_at": approved_date,
                "available_at": canonical_available,
                "content_hash": "",
                "metadata": approval_metadata,
            }
        )
        provisional = PeVcThesis.from_mapping(thesis_payload)
        approved_payload = provisional.to_dict()
        approved = PeVcThesis.from_mapping(approved_payload, require_approved=True)
        records = [*existing, approved]
        records.sort(key=_thesis_revision_key)
        self.canonical_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_after_text = "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in records
        )
        canonical_before = _private_file_state(self.canonical_path)
        ledger_before = _private_file_state(self.approval_ledger_path)
        canonical_after_sha256 = _sha256_bytes(
            canonical_after_text.encode("utf-8")
        )
        archive_summary = (
            f"{APPROVAL_DRAFT_ARCHIVE_DIRNAME}/{draft_hash}.json"
        )
        previous_event_hash = _GENESIS_EVENT_HASH
        if ledger_before["exists"]:
            previous_events = _load_approval_ledger(
                self.approval_ledger_path,
                key=key,
            )
            previous_event_hash = str(previous_events[-1]["event_hash"])
        event = _signed_approval_event(
            {
                "schema_version": APPROVAL_LEDGER_SCHEMA_VERSION,
                "approved_at": approved_date,
                "recorded_at": recorded_at.isoformat(),
                "thesis_id": approved.thesis_id,
                "theme_id": approved.theme_id,
                "version": approved.version,
                "draft_hash": draft_hash,
                "draft_archive_hash": draft_hash,
                "draft_archive_path_summary": archive_summary,
                "content_hash": approved.content_hash,
                "canonical_before_sha256": canonical_before["sha256"],
                "canonical_after_sha256": canonical_after_sha256,
                "prev_event_hash": previous_event_hash,
                "migration_mode": bool(migration_mode),
                "migration_evidence_hash": evidence_hash,
                "migration_evidence_path_summary": evidence_path_summary,
            },
            key=key,
        )
        event_line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
        ledger_after_text = str(ledger_before["text"]) + event_line
        ledger_after_sha256 = _sha256_bytes(ledger_after_text.encode("utf-8"))
        wal = _signed_transaction_wal(
            {
                "schema_version": "pevc_approval_transaction.v1",
                "transaction_id": event["event_hash"],
                "created_at": recorded_at.isoformat(),
                "canonical_path_summary": self.canonical_path.name,
                "ledger_path_summary": APPROVAL_LEDGER_FILENAME,
                "archive_path_summary": archive_summary,
                "before": {
                    "canonical_exists": canonical_before["exists"],
                    "canonical_sha256": canonical_before["sha256"],
                    "canonical_text": canonical_before["text"],
                    "ledger_exists": ledger_before["exists"],
                    "ledger_sha256": ledger_before["sha256"],
                    "ledger_text": ledger_before["text"],
                },
                "after": {
                    "canonical_exists": True,
                    "canonical_sha256": canonical_after_sha256,
                    "canonical_text": canonical_after_text,
                    "ledger_exists": True,
                    "ledger_sha256": ledger_after_sha256,
                    "ledger_text": ledger_after_text,
                },
                "inverse": {
                    "canonical_exists": canonical_before["exists"],
                    "canonical_sha256": canonical_before["sha256"],
                    "ledger_exists": ledger_before["exists"],
                    "ledger_sha256": ledger_before["sha256"],
                },
                "draft_archive_base64": base64.b64encode(raw_bytes).decode("ascii"),
                "draft_archive_sha256": draft_hash,
                "migration_evidence_archive_base64": (
                    base64.b64encode(evidence_bytes).decode("ascii")
                    if evidence_bytes is not None
                    else None
                ),
                "migration_evidence_archive_sha256": evidence_hash,
                "migration_evidence_archive_path_summary": evidence_path_summary,
                "event": event,
                "recovery": {
                    "status": "prepared",
                    "attempt_count": 0,
                    "last_recovered_at": None,
                },
            },
            key=key,
        )
        _write_private_json_exclusive(self.transaction_wal_path, wal)
        _transaction_checkpoint("wal_prepared")
        self._recover_pending_transaction(key)
        readback = self.load()
        matching = next(
            (
                item
                for item in readback
                if _thesis_approval_identity(item)
                == _thesis_approval_identity(approved)
            ),
            None,
        )
        if matching is None:
            raise RuntimeError("PEVC canonical approval readback mismatch")
        return matching

    def _recover_pending_transaction(self, key: bytes) -> None:
        path = self.transaction_wal_path
        if not path.exists():
            return
        wal = _load_transaction_wal(path, key=key)
        if wal.get("canonical_path_summary") != self.canonical_path.name:
            raise ValueError("PEVC transaction WAL canonical path mismatch")
        if wal.get("ledger_path_summary") != APPROVAL_LEDGER_FILENAME:
            raise ValueError("PEVC transaction WAL ledger path mismatch")
        if wal.get("archive_path_summary") != wal["event"].get(
            "draft_archive_path_summary"
        ):
            raise ValueError("PEVC transaction WAL archive path mismatch")
        before = dict(wal["before"])
        after = dict(wal["after"])
        archive_path = _resolve_private_summary(
            self.canonical_path.parent,
            str(wal["archive_path_summary"]),
        )
        archive_bytes = base64.b64decode(
            str(wal["draft_archive_base64"]),
            validate=True,
        )
        if _sha256_bytes(archive_bytes) != wal["draft_archive_sha256"]:
            raise ValueError("PEVC transaction draft archive payload hash mismatch")
        _ensure_private_archive(
            archive_path,
            archive_bytes,
            expected_sha256=str(wal["draft_archive_sha256"]),
        )
        evidence_summary = wal.get("migration_evidence_archive_path_summary")
        if evidence_summary is not None:
            evidence_path = _resolve_private_summary(
                self.canonical_path.parent,
                str(evidence_summary),
            )
            evidence_bytes = base64.b64decode(
                str(wal["migration_evidence_archive_base64"]),
                validate=True,
            )
            _ensure_private_archive(
                evidence_path,
                evidence_bytes,
                expected_sha256=str(
                    wal["migration_evidence_archive_sha256"]
                ),
            )
        _transaction_checkpoint("archive_written")
        _forward_transaction_file(
            self.canonical_path,
            before_exists=bool(before["canonical_exists"]),
            before_sha256=str(before["canonical_sha256"]),
            after_text=str(after["canonical_text"]),
            after_sha256=str(after["canonical_sha256"]),
        )
        _transaction_checkpoint("canonical_written")
        _forward_transaction_file(
            self.approval_ledger_path,
            before_exists=bool(before["ledger_exists"]),
            before_sha256=str(before["ledger_sha256"]),
            after_text=str(after["ledger_text"]),
            after_sha256=str(after["ledger_sha256"]),
        )
        _transaction_checkpoint("ledger_written")
        self._load(as_of=None, ignore_active_wal=True)
        recovery = dict(wal["recovery"])
        recovery["status"] = "committed"
        recovery["attempt_count"] = int(recovery.get("attempt_count") or 0) + 1
        recovery["last_recovered_at"] = datetime.now(timezone.utc).isoformat()
        wal["recovery"] = recovery
        wal = _signed_transaction_wal(wal, key=key)
        transaction_path = (
            self.canonical_path.parent
            / APPROVAL_TRANSACTION_DIRNAME
            / f"{wal['transaction_id']}.json"
        )
        _ensure_private_transaction_record(transaction_path, wal, key=key)
        _transaction_checkpoint("transaction_committed")
        path.unlink()
        _fsync_directory(path.parent)


def import_pevc_draft(
    source_path: str | Path,
    *,
    draft_dir: str | Path = DEFAULT_DRAFT_DIR,
    canonical_path: str | Path = DEFAULT_CANONICAL_PATH,
    source_type: str = "auto",
) -> dict[str, Any]:
    source = Path(source_path)
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    resolved_type = _resolve_source_type(source, source_type)
    payload = _extract_payload(source, raw, resolved_type)
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("status", "draft")
    payload["source_type"] = resolved_type
    payload["source_ref"] = str(source)
    payload["source_hash"] = source_hash
    payload.setdefault("available_at", datetime.now(timezone.utc).date().isoformat())
    payload["content_hash"] = ""
    validation_errors: list[str] = []
    try:
        thesis = PeVcThesis.from_mapping(payload)
        normalized_payload = thesis.to_dict()
        normalized_payload["status"] = "draft"
    except (TypeError, ValueError) as exc:
        validation_errors.append(str(exc))
        normalized_payload = dict(payload)
    diff = _canonical_diff(
        normalized_payload,
        canonical_path=canonical_path,
    )
    draft_id = hashlib.sha256(
        (
            f"{source_hash}:{normalized_payload.get('thesis_id', '')}:"
            f"{normalized_payload.get('version', '')}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    draft_payload = {
        "draft_schema_version": "pevc_thesis_draft.v1",
        "draft_id": draft_id,
        "draft_status": "pending_approval",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_hash": source_hash,
        "validation_status": "valid" if not validation_errors else "invalid",
        "validation_errors": validation_errors,
        "canonical_diff": diff,
        "thesis": normalized_payload,
    }
    target_dir = Path(draft_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{draft_id}.json"
    _atomic_write_text(
        target,
        json.dumps(draft_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        mode=0o600,
    )
    return {
        "status": "draft_created",
        "draft_path": str(target),
        "draft_hash": hashlib.sha256(target.read_bytes()).hexdigest(),
        "validation_status": draft_payload["validation_status"],
        "validation_errors": validation_errors,
        "canonical_diff": diff,
        "network_called": False,
    }


def _resolve_source_type(source: Path, requested: str) -> str:
    normalized = str(requested or "auto").strip().lower()
    if normalized != "auto":
        if normalized not in {"json", "markdown", "word", "notion_export"}:
            raise ValueError(f"unsupported source_type={requested}")
        return normalized
    suffix = source.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix == ".docx":
        return "word"
    if suffix in {".md", ".markdown", ".txt"}:
        return "markdown"
    raise ValueError(f"cannot infer source type from {source.name}")


def _extract_payload(source: Path, raw: bytes, source_type: str) -> dict[str, Any]:
    if source_type == "json":
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("structured PEVC input must contain an object")
        return dict(payload)
    if source_type == "word":
        text = _docx_text(source)
    else:
        text = raw.decode("utf-8")
    return _structured_text_payload(text)


def _docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(xml)
    namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespaces):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespaces)).strip()
        if text:
            paragraphs.append(text)
    return "\n".join(paragraphs)


def _structured_text_payload(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    list_fields = {"bottlenecks", "milestones", "kill_criteria"}
    aliases = {
        "thesis id": "thesis_id",
        "theme id": "theme_id",
        "technology maturity": "technology_maturity",
        "moat strength": "moat_strength",
        "customer validation": "customer_validation",
        "commercialization stage": "commercialization_stage",
        "valuation ceiling": "valuation_ceiling",
        "horizon months": "horizon_months",
        "review by": "review_by",
        "prior score": "prior_score",
        "技术成熟度": "technology_maturity",
        "客户验证": "customer_validation",
        "商业化阶段": "commercialization_stage",
        "壁垒": "moat_strength",
        "瓶颈": "bottlenecks",
        "里程碑": "milestones",
        "否决条件": "kill_criteria",
        "估值上限": "valuation_ceiling",
        "期限月": "horizon_months",
        "复核日期": "review_by",
    }
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("#").strip()
        if not line or line.startswith("---"):
            continue
        match = re.match(r"^[-*]?\s*([^:：]{2,40})\s*[:：]\s*(.+)$", line)
        if not match:
            continue
        raw_key, raw_value = match.groups()
        key = re.sub(r"[\s-]+", "_", raw_key.strip().lower())
        key = aliases.get(raw_key.strip().lower(), aliases.get(raw_key.strip(), key))
        value = raw_value.strip()
        if key in list_fields:
            payload[key] = [item.strip() for item in re.split(r"[;,；，|]", value) if item.strip()]
        elif key in {
            "technology_maturity",
            "moat_strength",
            "customer_validation",
            "commercialization_stage",
            "valuation_ceiling",
            "horizon_months",
            "confidence",
            "prior_score",
        }:
            try:
                payload[key] = float(value)
            except ValueError:
                payload[key] = value
        else:
            payload[key] = value
    return payload


def _canonical_diff(payload: Mapping[str, Any], *, canonical_path: str | Path) -> dict[str, Any]:
    thesis_id = str(payload.get("thesis_id") or "")
    version = str(payload.get("version") or "")
    try:
        existing = PeVcKnowledgeStore(canonical_path).load()
    except ValueError as exc:
        return {"status": "canonical_error", "error": str(exc), "changed_fields": []}
    matching = next(
        (
            item.to_dict()
            for item in existing
            if item.thesis_id == thesis_id and item.version == version
        ),
        None,
    )
    if matching is None:
        return {"status": "new_version", "changed_fields": sorted(payload)}
    ignored = {"content_hash", "approved_at", "status"}
    changed = sorted(
        key
        for key in set(payload) | set(matching)
        if key not in ignored and payload.get(key) != matching.get(key)
    )
    return {"status": "changed" if changed else "unchanged", "changed_fields": changed}


def _atomic_write_text(path: Path, text: str, *, mode: int) -> None:
    _atomic_write_bytes(path, text.encode("utf-8"), mode=mode)


def _atomic_write_bytes(path: Path, payload: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def initialize_pevc_approval_key(
    canonical_path: str | Path = DEFAULT_CANONICAL_PATH,
) -> dict[str, Any]:
    canonical = Path(canonical_path)
    root = canonical.parent
    key_path = root / APPROVAL_KEY_FILENAME
    protected_paths = (
        canonical,
        root / APPROVAL_LEDGER_FILENAME,
        root / APPROVAL_WAL_FILENAME,
        root / APPROVAL_DRAFT_ARCHIVE_DIRNAME,
        root / APPROVAL_TRANSACTION_DIRNAME,
        root / MIGRATION_EVIDENCE_ARCHIVE_DIRNAME,
        key_path,
    )
    if any(path.exists() for path in protected_paths):
        raise FileExistsError(
            "approval key initialization requires an empty PEVC canonical store"
        )
    key = os.urandom(32)
    _write_private_bytes_exclusive(key_path, key)
    loaded = _load_approval_key(key_path)
    return {
        "status": "initialized",
        "key_path": str(key_path),
        "key_fingerprint": hashlib.sha256(loaded).hexdigest(),
    }


def _load_approval_key(path: Path) -> bytes:
    if not path.exists():
        raise ValueError(
            "PEVC approval key is missing; run explicit init-key before approval"
        )
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("PEVC approval key permissions must be 0600")
    key = path.read_bytes()
    if len(key) != 32:
        raise ValueError("PEVC approval key must contain exactly 32 random bytes")
    return key


def _load_approval_ledger(path: Path, *, key: bytes) -> list[dict[str, Any]]:
    if not path.exists():
        raise ValueError("PEVC canonical requires an approval ledger")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("PEVC approval ledger permissions must be 0600")
    events: list[dict[str, Any]] = []
    event_hashes: set[str] = set()
    revisions: set[
        tuple[str, tuple[tuple[int, int | str], ...]]
    ] = set()
    previous_recorded_at: datetime | None = None
    previous_event_hash = _GENESIS_EVENT_HASH
    previous_canonical_sha256 = _EMPTY_SHA256
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            raw_event = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid PEVC approval event at line {line_number}: {exc}"
            ) from exc
        if not isinstance(raw_event, Mapping):
            raise ValueError(
                f"invalid PEVC approval event at line {line_number}: object required"
            )
        event = dict(raw_event)
        try:
            recorded_at = _validate_approval_event(event, key=key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid PEVC approval event at line {line_number}: {exc}"
            ) from exc
        event_hash = str(event["event_hash"])
        if event_hash in event_hashes:
            raise ValueError("duplicate PEVC approval event_hash")
        if event["prev_event_hash"] != previous_event_hash:
            raise ValueError("PEVC approval ledger prev_event_hash chain mismatch")
        if event["canonical_before_sha256"] != previous_canonical_sha256:
            raise ValueError("PEVC approval ledger canonical SHA chain mismatch")
        if previous_recorded_at is not None and recorded_at < previous_recorded_at:
            raise ValueError("PEVC approval ledger recorded_at is not append ordered")
        revision = _revision_identity(
            str(event["thesis_id"]),
            str(event["version"]),
        )
        if revision in revisions:
            raise ValueError("duplicate approved PEVC revision in approval ledger")
        event_hashes.add(event_hash)
        revisions.add(revision)
        previous_recorded_at = recorded_at
        previous_event_hash = event_hash
        previous_canonical_sha256 = str(event["canonical_after_sha256"])
        events.append(event)
    return events


def _validate_approval_event(event: Mapping[str, Any], *, key: bytes) -> datetime:
    if str(event.get("schema_version") or "") != APPROVAL_LEDGER_SCHEMA_VERSION:
        raise ValueError("unsupported approval ledger schema_version")
    event_hash = str(event.get("event_hash") or "").strip()
    if not _is_sha256(event_hash):
        raise ValueError("approval event_hash must be SHA-256")
    if event_hash != _approval_event_hash(event):
        raise ValueError("approval event_hash mismatch")
    signature = str(event.get("event_signature") or "").strip()
    if not _is_sha256(signature):
        raise ValueError("approval event_signature must be HMAC-SHA256")
    if not hmac.compare_digest(signature, _payload_hmac(event, key=key)):
        raise ValueError("approval event_signature mismatch")

    for field_name in ("thesis_id", "theme_id", "version"):
        if not str(event.get(field_name) or "").strip():
            raise ValueError(f"approval {field_name} is required")
    if "::" not in str(event.get("theme_id") or ""):
        raise ValueError("approval theme_id must be namespaced")
    for field_name in (
        "draft_hash",
        "draft_archive_hash",
        "content_hash",
        "canonical_before_sha256",
        "canonical_after_sha256",
        "prev_event_hash",
    ):
        if not _is_sha256(event.get(field_name)):
            raise ValueError(f"approval {field_name} must be SHA-256")
    if event.get("draft_archive_hash") != event.get("draft_hash"):
        raise ValueError("approval draft archive hash mismatch")
    archive_summary = str(event.get("draft_archive_path_summary") or "")
    expected_archive_summary = (
        f"{APPROVAL_DRAFT_ARCHIVE_DIRNAME}/{event['draft_hash']}.json"
    )
    if archive_summary != expected_archive_summary:
        raise ValueError("approval draft archive path summary mismatch")

    approved_at = _date_text(event.get("approved_at"))
    recorded_at = _recorded_datetime(event.get("recorded_at"))
    recorded_business_date = _shanghai_business_date(recorded_at).isoformat()
    migration_mode = event.get("migration_mode")
    if not isinstance(migration_mode, bool):
        raise ValueError("approval migration_mode must be boolean")
    migration_evidence_hash = event.get("migration_evidence_hash")
    if migration_mode:
        if not _is_sha256(migration_evidence_hash):
            raise ValueError(
                "migration approval requires migration_evidence_hash SHA-256"
            )
        if approved_at >= recorded_business_date:
            raise ValueError("migration approval must be backdated")
        expected_evidence_summary = (
            f"{MIGRATION_EVIDENCE_ARCHIVE_DIRNAME}/"
            f"{migration_evidence_hash}.evidence"
        )
        if event.get("migration_evidence_path_summary") != expected_evidence_summary:
            raise ValueError("migration approval evidence path summary mismatch")
    else:
        if migration_evidence_hash not in {None, ""}:
            raise ValueError(
                "non-migration approval must not carry migration evidence"
            )
        if approved_at != recorded_business_date:
            raise ValueError(
                "non-migration approval must use its recorded Shanghai date"
            )
        if event.get("migration_evidence_path_summary") not in {None, ""}:
            raise ValueError(
                "non-migration approval must not carry evidence path summary"
            )
    return recorded_at


def _approval_event_hash(event: Mapping[str, Any]) -> str:
    payload = dict(event)
    payload.pop("event_hash", None)
    payload.pop("event_signature", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _signed_approval_event(
    event: Mapping[str, Any],
    *,
    key: bytes,
) -> dict[str, Any]:
    payload = dict(event)
    payload.pop("event_hash", None)
    payload.pop("event_signature", None)
    payload["event_hash"] = _approval_event_hash(payload)
    payload["event_signature"] = _payload_hmac(payload, key=key)
    _validate_approval_event(payload, key=key)
    return payload


def _payload_hmac(payload: Mapping[str, Any], *, key: bytes) -> str:
    unsigned = dict(payload)
    unsigned.pop("event_signature", None)
    unsigned.pop("wal_signature", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, encoded, hashlib.sha256).hexdigest()


def _validate_draft_archive(event: Mapping[str, Any], *, root: Path) -> None:
    archive_path = _resolve_private_summary(
        root,
        str(event.get("draft_archive_path_summary") or ""),
    )
    if not archive_path.exists():
        raise ValueError("PEVC approved draft archive is missing")
    if stat.S_IMODE(archive_path.stat().st_mode) != 0o600:
        raise ValueError("PEVC approved draft archive permissions must be 0600")
    if _file_sha256(archive_path) != event.get("draft_archive_hash"):
        raise ValueError("PEVC approved draft archive hash mismatch")


def _validate_migration_evidence_archive(
    event: Mapping[str, Any],
    *,
    root: Path,
) -> None:
    if not event.get("migration_mode"):
        return
    evidence_path = _resolve_private_summary(
        root,
        str(event.get("migration_evidence_path_summary") or ""),
    )
    if not evidence_path.exists():
        raise ValueError("PEVC migration evidence archive is missing")
    if stat.S_IMODE(evidence_path.stat().st_mode) != 0o600:
        raise ValueError("PEVC migration evidence archive permissions must be 0600")
    if _file_sha256(evidence_path) != event.get("migration_evidence_hash"):
        raise ValueError("PEVC migration evidence archive hash mismatch")


def _validate_migration_evidence(
    evidence_file: str | Path | None,
    *,
    expected_hash: str,
) -> tuple[str, str, bytes]:
    if evidence_file is None:
        raise ValueError("migration_mode requires a 0600 evidence file")
    path = Path(evidence_file)
    if not path.is_file():
        raise ValueError("migration evidence file is missing")
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("migration evidence file permissions must be 0600")
    if not _is_sha256(expected_hash):
        raise ValueError("expected migration evidence SHA-256 is required")
    evidence_bytes = path.read_bytes()
    actual_hash = _sha256_bytes(evidence_bytes)
    if actual_hash != expected_hash:
        raise ValueError("migration evidence hash changed since review")
    archive_summary = (
        f"{MIGRATION_EVIDENCE_ARCHIVE_DIRNAME}/{actual_hash}.evidence"
    )
    return actual_hash, archive_summary, evidence_bytes


def _private_file_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "sha256": _EMPTY_SHA256, "text": ""}
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"private PEVC file permissions must be 0600: {path.name}")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"private PEVC file must be UTF-8: {path.name}") from exc
    return {"exists": True, "sha256": _sha256_bytes(raw), "text": text}


def _signed_transaction_wal(
    payload: Mapping[str, Any],
    *,
    key: bytes,
) -> dict[str, Any]:
    wal = dict(payload)
    wal.pop("wal_signature", None)
    wal["wal_signature"] = _payload_hmac(wal, key=key)
    return wal


def _load_transaction_wal(path: Path, *, key: bytes) -> dict[str, Any]:
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError("PEVC transaction WAL permissions must be 0600")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("invalid PEVC transaction WAL") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("invalid PEVC transaction WAL object")
    wal = dict(raw)
    if wal.get("schema_version") != "pevc_approval_transaction.v1":
        raise ValueError("unsupported PEVC transaction WAL schema")
    signature = str(wal.get("wal_signature") or "")
    if not _is_sha256(signature) or not hmac.compare_digest(
        signature,
        _payload_hmac(wal, key=key),
    ):
        raise ValueError("PEVC transaction WAL signature mismatch")
    event = wal.get("event")
    if not isinstance(event, Mapping):
        raise ValueError("PEVC transaction WAL event is invalid")
    _validate_approval_event(event, key=key)
    if wal.get("transaction_id") != event.get("event_hash"):
        raise ValueError("PEVC transaction WAL transaction_id mismatch")
    before = wal.get("before")
    after = wal.get("after")
    inverse = wal.get("inverse")
    recovery = wal.get("recovery")
    if not all(isinstance(value, Mapping) for value in (before, after, inverse, recovery)):
        raise ValueError("PEVC transaction WAL state is invalid")
    _validate_wal_file_state(before, prefix="canonical")
    _validate_wal_file_state(before, prefix="ledger")
    _validate_wal_file_state(after, prefix="canonical")
    _validate_wal_file_state(after, prefix="ledger")
    if event.get("canonical_before_sha256") != before.get("canonical_sha256"):
        raise ValueError("PEVC transaction WAL canonical before SHA mismatch")
    if event.get("canonical_after_sha256") != after.get("canonical_sha256"):
        raise ValueError("PEVC transaction WAL canonical after SHA mismatch")
    if inverse.get("canonical_sha256") != before.get("canonical_sha256"):
        raise ValueError("PEVC transaction WAL inverse canonical SHA mismatch")
    if inverse.get("ledger_sha256") != before.get("ledger_sha256"):
        raise ValueError("PEVC transaction WAL inverse ledger SHA mismatch")
    if wal.get("draft_archive_sha256") != event.get("draft_archive_hash"):
        raise ValueError("PEVC transaction WAL draft archive SHA mismatch")
    expected_event_line = json.dumps(
        event,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"
    if after.get("ledger_text") != before.get("ledger_text", "") + expected_event_line:
        raise ValueError("PEVC transaction WAL ledger transition mismatch")
    try:
        archive_bytes = base64.b64decode(
            str(wal.get("draft_archive_base64") or ""),
            validate=True,
        )
    except ValueError as exc:
        raise ValueError("PEVC transaction WAL draft archive encoding invalid") from exc
    if _sha256_bytes(archive_bytes) != wal.get("draft_archive_sha256"):
        raise ValueError("PEVC transaction WAL draft archive payload mismatch")
    evidence_payload = wal.get("migration_evidence_archive_base64")
    evidence_sha256 = wal.get("migration_evidence_archive_sha256")
    evidence_summary = wal.get("migration_evidence_archive_path_summary")
    if event.get("migration_mode"):
        if evidence_sha256 != event.get("migration_evidence_hash"):
            raise ValueError("PEVC transaction WAL migration evidence SHA mismatch")
        if evidence_summary != event.get("migration_evidence_path_summary"):
            raise ValueError("PEVC transaction WAL migration evidence path mismatch")
        try:
            evidence_bytes = base64.b64decode(
                str(evidence_payload or ""),
                validate=True,
            )
        except ValueError as exc:
            raise ValueError(
                "PEVC transaction WAL migration evidence encoding invalid"
            ) from exc
        if _sha256_bytes(evidence_bytes) != evidence_sha256:
            raise ValueError(
                "PEVC transaction WAL migration evidence payload mismatch"
            )
    elif any(
        value is not None
        for value in (evidence_payload, evidence_sha256, evidence_summary)
    ):
        raise ValueError("non-migration PEVC transaction carries evidence archive")
    return wal


def _validate_wal_file_state(state: Mapping[str, Any], *, prefix: str) -> None:
    exists = state.get(f"{prefix}_exists")
    sha256 = state.get(f"{prefix}_sha256")
    text = state.get(f"{prefix}_text")
    if not isinstance(exists, bool) or not isinstance(text, str) or not _is_sha256(sha256):
        raise ValueError(f"PEVC transaction WAL {prefix} state is invalid")
    if _sha256_bytes(text.encode("utf-8")) != sha256:
        raise ValueError(f"PEVC transaction WAL {prefix} text SHA mismatch")
    if not exists and (text or sha256 != _EMPTY_SHA256):
        raise ValueError(f"PEVC transaction WAL absent {prefix} state is invalid")


def _write_private_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _write_private_bytes_exclusive(path, text.encode("utf-8"))


def _write_private_bytes_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / (
        f".{path.name}.exclusive.{os.urandom(16).hex()}"
    )
    descriptor = os.open(
        temp_path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, path)
        _fsync_directory(path.parent)
    except Exception:
        raise
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def _ensure_private_archive(path: Path, payload: bytes, *, expected_sha256: str) -> None:
    if path.exists():
        if stat.S_IMODE(path.stat().st_mode) != 0o600:
            raise ValueError("PEVC approved draft archive permissions must be 0600")
        if _file_sha256(path) == expected_sha256:
            return
        _atomic_write_bytes(path, payload, mode=0o600)
    else:
        _write_private_bytes_exclusive(path, payload)
    if _file_sha256(path) != expected_sha256:
        raise RuntimeError("PEVC approved draft archive readback mismatch")


def _forward_transaction_file(
    path: Path,
    *,
    before_exists: bool,
    before_sha256: str,
    after_text: str,
    after_sha256: str,
) -> None:
    current = _private_file_state(path)
    if current["exists"] and current["sha256"] == after_sha256:
        return
    if current["exists"] != before_exists or current["sha256"] != before_sha256:
        raise ValueError(f"PEVC transaction state conflict: {path.name}")
    if _sha256_bytes(after_text.encode("utf-8")) != after_sha256:
        raise ValueError(f"PEVC transaction after SHA mismatch: {path.name}")
    _atomic_write_text(path, after_text, mode=0o600)
    if _file_sha256(path) != after_sha256:
        raise RuntimeError(f"PEVC transaction readback mismatch: {path.name}")


def _ensure_private_transaction_record(
    path: Path,
    wal: Mapping[str, Any],
    *,
    key: bytes,
) -> None:
    expected_text = json.dumps(
        wal,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if path.exists():
        existing = _load_transaction_wal(path, key=key)
        stable_fields = (
            "schema_version",
            "transaction_id",
            "canonical_path_summary",
            "ledger_path_summary",
            "archive_path_summary",
            "before",
            "after",
            "inverse",
            "draft_archive_base64",
            "draft_archive_sha256",
            "migration_evidence_archive_base64",
            "migration_evidence_archive_sha256",
            "migration_evidence_archive_path_summary",
            "event",
        )
        if any(existing.get(field) != wal.get(field) for field in stable_fields):
            raise ValueError("PEVC committed transaction record collision")
        return
    _write_private_bytes_exclusive(path, expected_text.encode("utf-8"))


def _resolve_private_summary(root: Path, summary: str) -> Path:
    relative = Path(summary)
    if relative.is_absolute() or not relative.parts or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise ValueError("invalid private PEVC path summary")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    if resolved_root not in resolved.parents:
        raise ValueError("private PEVC path summary escapes its root")
    return resolved


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _transaction_checkpoint(_stage: str) -> None:
    return None


def _approval_identity(event: Mapping[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        str(event.get("thesis_id") or "").strip(),
        str(event.get("theme_id") or "").strip(),
        str(event.get("version") or "").strip(),
        str(event.get("approved_at") or "").strip(),
        str(event.get("content_hash") or "").strip(),
        str(event.get("draft_hash") or "").strip(),
    )


def _thesis_approval_identity(
    thesis: PeVcThesis,
) -> tuple[str, str, str, str, str, str]:
    return (
        thesis.thesis_id,
        thesis.theme_id,
        thesis.version,
        thesis.approved_at,
        thesis.content_hash,
        thesis.approval_draft_hash,
    )


def _natural_version_key(value: Any) -> tuple[tuple[int, int | str], ...]:
    text = str(value or "").strip().lower()
    if text.startswith("v"):
        text = text[1:]
    parts: list[tuple[int, int | str]] = []
    for part in re.split(r"(\d+)", text):
        if not part:
            continue
        parts.append((1, int(part)) if part.isdigit() else (0, part))
    return tuple(parts)


def _revision_identity(
    thesis_id: Any,
    version: Any,
) -> tuple[str, tuple[tuple[int, int | str], ...]]:
    return str(thesis_id or "").strip(), _natural_version_key(version)


def _thesis_revision_key(
    thesis: PeVcThesis,
) -> tuple[str, tuple[tuple[int, int | str], ...], str, str]:
    return (
        thesis.thesis_id,
        _natural_version_key(thesis.version),
        thesis.approved_at,
        thesis.content_hash,
    )


def _recorded_datetime(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("approval recorded_at is required")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("invalid approval recorded_at") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("approval recorded_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _shanghai_business_date(value: datetime) -> date:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("recorded_at must be timezone-aware")
    return value.astimezone(_SHANGHAI_TIMEZONE).date()


def _is_sha256(value: Any) -> bool:
    return bool(_SHA256_PATTERN.fullmatch(str(value or "").strip()))


def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        values: Sequence[Any] = re.split(r"[;,；，|]", value)
    elif isinstance(value, Sequence):
        values = value
    else:
        values = ()
    result: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _unit(value: Any, field_name: str) -> float:
    numeric = _finite(value, math_nan())
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field_name} must be within 0..1")
    return numeric


def _optional_unit(value: Any, field_name: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return _unit(value, field_name)


def _optional_nonnegative(value: Any) -> float | None:
    if value is None or str(value).strip().lower() in {"", "unknown", "none", "null"}:
        return None
    numeric = _finite(value, -1.0)
    if numeric < 0:
        raise ValueError("valuation_ceiling must be non-negative or unknown")
    return numeric


def _finite(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if numeric == numeric and abs(numeric) != float("inf") else default


def math_nan() -> float:
    return float("nan")


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    for fmt in _DATE_FORMATS:
        candidate = text[:10] if fmt == "%Y-%m-%d" else text[:8]
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    return None


def _strict_date(value: Any) -> date:
    text = str(value or "").strip()
    if not re.fullmatch(r"(?:\d{4}-\d{2}-\d{2}|\d{8})", text):
        raise ValueError(f"invalid as_of={value}")
    parsed = _parse_date(text)
    if parsed is None:
        raise ValueError(f"invalid as_of={value}")
    return parsed


def _date_text(value: Any, *, required: bool = True) -> str:
    parsed = _parse_date(value)
    if parsed is None:
        if required:
            raise ValueError(f"invalid date={value}")
        return ""
    return parsed.isoformat()
