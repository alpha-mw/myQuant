from __future__ import annotations

import hashlib
import json
import stat
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import quant_investor.themes.pevc as pevc_module
from quant_investor.themes.pevc import (
    PeVcThesis,
    PeVcKnowledgeStore,
    import_pevc_draft,
    initialize_pevc_approval_key,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": "pevc_thesis.v1",
        "thesis_id": "ai-infra-001",
        "theme_id": "tech::ai-compute",
        "version": "1.0.0",
        "status": "draft",
        "tam": "Domestic AI infrastructure demand over a five-year horizon",
        "technology_maturity": 0.70,
        "bottlenecks": ["advanced packaging", "power density"],
        "moat_strength": 0.75,
        "customer_validation": 0.60,
        "commercialization_stage": 0.65,
        "milestones": ["customer qualification", "volume shipment"],
        "kill_criteria": ["qualification failure", "gross margin below floor"],
        "valuation_ceiling": 15000000000,
        "horizon_months": 36,
        "confidence": 0.72,
        "review_by": "2026-12-31",
        "available_at": "2026-07-10",
        "prior_score": 0.80,
    }


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _day_offset(days: int) -> str:
    return (datetime.now(timezone.utc).date() + timedelta(days=days)).isoformat()


def _import_and_approve(
    tmp_path: Path,
    canonical: Path,
    *,
    payload: dict[str, object] | None = None,
    source_name: str = "thesis.json",
) -> tuple[PeVcKnowledgeStore, PeVcThesis]:
    source = tmp_path / source_name
    source.write_text(json.dumps(payload or _payload()), encoding="utf-8")
    result = import_pevc_draft(
        source,
        draft_dir=tmp_path / "private" / "drafts",
        canonical_path=canonical,
    )
    key_path = canonical.parent / "pevc_approval.key"
    if not key_path.exists():
        initialize_pevc_approval_key(canonical)
    store = PeVcKnowledgeStore(canonical)
    approved = store.approve_draft(
        result["draft_path"],
        expected_draft_hash=result["draft_hash"],
    )
    return store, approved


def _event_hash(event: dict[str, object]) -> str:
    unsigned = dict(event)
    unsigned.pop("event_hash", None)
    unsigned.pop("event_signature", None)
    encoded = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def test_import_creates_private_draft_without_mutating_canonical_then_hash_guarded_approve(
    tmp_path: Path,
) -> None:
    source = tmp_path / "thesis.json"
    source.write_text(json.dumps(_payload(), ensure_ascii=False), encoding="utf-8")
    draft_dir = tmp_path / "private" / "drafts"
    canonical = tmp_path / "private" / "pevc_theses.jsonl"

    result = import_pevc_draft(
        source,
        draft_dir=draft_dir,
        canonical_path=canonical,
    )

    assert result["status"] == "draft_created"
    assert result["validation_status"] == "valid"
    assert result["network_called"] is False
    assert not canonical.exists()

    initialize_pevc_approval_key(canonical)
    store = PeVcKnowledgeStore(canonical)
    with pytest.raises(ValueError, match="draft hash changed"):
        store.approve_draft(result["draft_path"], expected_draft_hash="wrong")

    approved = store.approve_draft(
        result["draft_path"],
        expected_draft_hash=result["draft_hash"],
    )
    idempotent = store.approve_draft(
        result["draft_path"],
        expected_draft_hash=result["draft_hash"],
    )

    assert canonical.exists()
    assert idempotent.content_hash == approved.content_hash
    assert approved.status == "approved"
    assert approved.content_hash
    assert approved.approval_draft_hash == result["draft_hash"]
    assert [item.thesis_id for item in store.load(as_of=_today())] == [
        "ai-infra-001"
    ]
    assert stat.S_IMODE(canonical.stat().st_mode) == 0o600
    ledger = canonical.parent / "pevc_approval_ledger.jsonl"
    assert stat.S_IMODE(ledger.stat().st_mode) == 0o600
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1


def test_approval_key_requires_explicit_one_time_initialization(tmp_path: Path) -> None:
    source = tmp_path / "thesis.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    result = import_pevc_draft(
        source,
        draft_dir=tmp_path / "private" / "drafts",
        canonical_path=canonical,
    )
    store = PeVcKnowledgeStore(canonical)

    with pytest.raises(ValueError, match="explicit init-key"):
        store.approve_draft(
            result["draft_path"],
            expected_draft_hash=result["draft_hash"],
        )
    initialized = initialize_pevc_approval_key(canonical)

    key_path = Path(initialized["key_path"])
    assert stat.S_IMODE(key_path.stat().st_mode) == 0o600
    assert len(key_path.read_bytes()) == 32
    with pytest.raises(FileExistsError, match="empty PEVC canonical store"):
        initialize_pevc_approval_key(canonical)


def test_load_validates_as_of_and_existing_key_before_empty_store_return(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    store = PeVcKnowledgeStore(canonical)

    with pytest.raises(ValueError, match="invalid as_of"):
        store.load(as_of="invalid")

    initialized = initialize_pevc_approval_key(canonical)
    Path(initialized["key_path"]).chmod(0o644)
    with pytest.raises(ValueError, match="key permissions"):
        store.load()


@pytest.mark.parametrize(
    ("artifact_name", "is_directory"),
    [
        ("pevc_approval_ledger.jsonl", False),
        ("pevc_approval_transaction.wal.json", False),
        ("approved_drafts", True),
        ("approval_transactions", True),
        ("migration_evidence", True),
    ],
)
def test_init_key_refuses_orphan_approval_artifacts(
    tmp_path: Path,
    artifact_name: str,
    is_directory: bool,
) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    artifact = canonical.parent / artifact_name
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if is_directory:
        artifact.mkdir()
    else:
        artifact.write_text("orphan", encoding="utf-8")

    with pytest.raises(FileExistsError, match="empty PEVC canonical store"):
        initialize_pevc_approval_key(canonical)


def test_empty_store_rejects_orphan_archive_state(tmp_path: Path) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    initialize_pevc_approval_key(canonical)
    (canonical.parent / "approved_drafts").mkdir()

    with pytest.raises(ValueError, match="artifacts exist without canonical"):
        PeVcKnowledgeStore(canonical).load()


def test_invalid_draft_cannot_enter_canonical_store(tmp_path: Path) -> None:
    payload = _payload()
    payload["kill_criteria"] = []
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    result = import_pevc_draft(
        source,
        draft_dir=tmp_path / "private" / "drafts",
        canonical_path=canonical,
    )

    assert result["validation_status"] == "invalid"
    initialize_pevc_approval_key(canonical)
    with pytest.raises(ValueError, match="kill_criteria"):
        PeVcKnowledgeStore(canonical).approve_draft(
            result["draft_path"],
            expected_draft_hash=result["draft_hash"],
        )
    assert not canonical.exists()


def test_approval_time_is_a_pit_floor_for_canonical_availability(tmp_path: Path) -> None:
    payload = _payload()
    payload["available_at"] = "2020-01-01"
    source = tmp_path / "backdated.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    result = import_pevc_draft(
        source,
        draft_dir=tmp_path / "private" / "drafts",
        canonical_path=canonical,
    )

    initialize_pevc_approval_key(canonical)
    store = PeVcKnowledgeStore(canonical)
    approved = store.approve_draft(
        result["draft_path"],
        expected_draft_hash=result["draft_hash"],
    )

    assert approved.approved_at == _today()
    assert approved.available_at == _today()
    assert store.load(as_of=_day_offset(-1)) == []
    assert [item.thesis_id for item in store.load(as_of=_today())] == [
        "ai-infra-001"
    ]


def test_pit_floor_uses_shanghai_business_date_after_utc_rollover(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_at = datetime(2026, 7, 11, 16, 30, tzinfo=timezone.utc)
    monkeypatch.setattr(pevc_module, "_now_utc", lambda: recorded_at)
    payload = _payload()
    payload["available_at"] = "2020-01-01"
    source = tmp_path / "shanghai-floor.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    result = import_pevc_draft(
        source,
        draft_dir=tmp_path / "private" / "drafts",
        canonical_path=canonical,
    )
    initialize_pevc_approval_key(canonical)
    store = PeVcKnowledgeStore(canonical)

    approved = store.approve_draft(
        result["draft_path"],
        expected_draft_hash=result["draft_hash"],
    )

    assert approved.approved_at == "2026-07-12"
    assert approved.available_at == "2026-07-12"
    assert store.load(as_of="2026-07-11") == []
    assert len(store.load(as_of="2026-07-12")) == 1


def test_markdown_and_notion_exports_only_create_drafts(tmp_path: Path) -> None:
    text = "\n".join(
        [
            "thesis_id: robot-001",
            "theme_id: tech::humanoid-robot",
            "version: 1.0.0",
            "tam: Global humanoid robot components",
            "technology_maturity: 0.6",
            "bottlenecks: reducer; servo",
            "moat_strength: 0.7",
            "customer_validation: 0.5",
            "commercialization_stage: 0.5",
            "milestones: sample; mass production",
            "kill_criteria: sample failure; no customer order",
            "valuation_ceiling: 10000000000",
            "horizon_months: 36",
            "confidence: 0.65",
            "review_by: 2026-12-31",
            "available_at: 2026-07-10",
        ]
    )
    source = tmp_path / "notion-export.md"
    source.write_text(text, encoding="utf-8")
    canonical = tmp_path / "private" / "pevc_theses.jsonl"

    result = import_pevc_draft(
        source,
        source_type="notion_export",
        draft_dir=tmp_path / "private" / "drafts",
        canonical_path=canonical,
    )

    assert result["validation_status"] == "valid"
    assert not canonical.exists()
    draft = json.loads(Path(result["draft_path"]).read_text(encoding="utf-8"))
    assert draft["thesis"]["source_type"] == "notion_export"
    assert draft["draft_status"] == "pending_approval"


def test_word_import_reads_local_docx_without_network_or_canonical_write(tmp_path: Path) -> None:
    document_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:r><w:t>thesis_id: space-001</w:t></w:r></w:p>
        <w:p><w:r><w:t>theme_id: tech::commercial-space</w:t></w:r></w:p>
        <w:p><w:r><w:t>version: 1.0.0</w:t></w:r></w:p>
        <w:p><w:r><w:t>tam: Commercial launch and satellite manufacturing</w:t></w:r></w:p>
        <w:p><w:r><w:t>technology_maturity: 0.6</w:t></w:r></w:p>
        <w:p><w:r><w:t>bottlenecks: launch cadence; supply chain</w:t></w:r></w:p>
        <w:p><w:r><w:t>moat_strength: 0.7</w:t></w:r></w:p>
        <w:p><w:r><w:t>customer_validation: 0.6</w:t></w:r></w:p>
        <w:p><w:r><w:t>commercialization_stage: 0.5</w:t></w:r></w:p>
        <w:p><w:r><w:t>milestones: launch; customer contract</w:t></w:r></w:p>
        <w:p><w:r><w:t>kill_criteria: repeated launch failure</w:t></w:r></w:p>
        <w:p><w:r><w:t>valuation_ceiling: 12000000000</w:t></w:r></w:p>
        <w:p><w:r><w:t>horizon_months: 48</w:t></w:r></w:p>
        <w:p><w:r><w:t>confidence: 0.6</w:t></w:r></w:p>
        <w:p><w:r><w:t>review_by: 2026-12-31</w:t></w:r></w:p>
        <w:p><w:r><w:t>available_at: 2026-07-10</w:t></w:r></w:p>
      </w:body>
    </w:document>"""
    source = tmp_path / "thesis.docx"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("word/document.xml", document_xml)
    canonical = tmp_path / "private" / "pevc_theses.jsonl"

    result = import_pevc_draft(
        source,
        draft_dir=tmp_path / "private" / "drafts",
        canonical_path=canonical,
    )

    assert result["validation_status"] == "valid"
    assert result["network_called"] is False
    assert not canonical.exists()


def test_canonical_requires_private_approval_ledger(tmp_path: Path) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    store, _ = _import_and_approve(tmp_path, canonical)
    ledger = canonical.parent / "pevc_approval_ledger.jsonl"

    ledger.unlink()

    with pytest.raises(ValueError, match="requires an approval ledger"):
        store.load()


def test_canonical_and_ledger_permissions_fail_closed(tmp_path: Path) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    store, _ = _import_and_approve(tmp_path, canonical)
    ledger = canonical.parent / "pevc_approval_ledger.jsonl"

    canonical.chmod(0o644)
    with pytest.raises(ValueError, match="canonical permissions"):
        store.load()
    canonical.chmod(0o600)

    ledger.chmod(0o644)
    with pytest.raises(ValueError, match="ledger permissions"):
        store.load()
    ledger.chmod(0o600)

    key = canonical.parent / "pevc_approval.key"
    key.chmod(0o644)
    with pytest.raises(ValueError, match="key permissions"):
        store.load()


def test_self_consistent_forged_canonical_has_no_matching_approval(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    store, _ = _import_and_approve(tmp_path, canonical)
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    payload["tam"] = "Forged TAM after local approval"
    payload["content_hash"] = ""
    forged = PeVcThesis.from_mapping(payload).to_dict()
    canonical.write_text(
        json.dumps(forged, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical SHA"):
        store.load()


def test_self_hashed_but_incomplete_forged_ledger_is_rejected(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    store, _ = _import_and_approve(tmp_path, canonical)
    ledger = canonical.parent / "pevc_approval_ledger.jsonl"
    event = json.loads(ledger.read_text(encoding="utf-8"))
    event.pop("draft_hash")
    event["event_hash"] = _event_hash(event)
    ledger.write_text(
        json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="event_signature mismatch"):
        store.load()


def test_tampered_ledger_event_hash_is_rejected(tmp_path: Path) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    store, _ = _import_and_approve(tmp_path, canonical)
    ledger = canonical.parent / "pevc_approval_ledger.jsonl"
    event = json.loads(ledger.read_text(encoding="utf-8"))
    event["theme_id"] = "tech::forged-theme"
    ledger.write_text(
        json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="event_hash mismatch"):
        store.load()


def test_whole_self_hashed_canonical_and_ledger_forgery_fails_hmac(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    store, _ = _import_and_approve(tmp_path, canonical)
    payload = json.loads(canonical.read_text(encoding="utf-8"))
    payload["tam"] = "Forged but internally self-consistent"
    payload["content_hash"] = ""
    forged = PeVcThesis.from_mapping(payload).to_dict()
    canonical_text = json.dumps(
        forged,
        ensure_ascii=False,
        sort_keys=True,
    ) + "\n"
    canonical.write_text(canonical_text, encoding="utf-8")

    ledger = canonical.parent / "pevc_approval_ledger.jsonl"
    event = json.loads(ledger.read_text(encoding="utf-8"))
    event["content_hash"] = forged["content_hash"]
    event["canonical_after_sha256"] = hashlib.sha256(
        canonical_text.encode("utf-8")
    ).hexdigest()
    event["event_hash"] = _event_hash(event)
    ledger.write_text(
        json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="event_signature mismatch"):
        store.load()


def test_approved_revision_is_immutable_and_requires_new_version(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    store, _ = _import_and_approve(tmp_path, canonical)
    before_canonical = hashlib.sha256(canonical.read_bytes()).hexdigest()
    ledger = canonical.parent / "pevc_approval_ledger.jsonl"
    before_ledger = hashlib.sha256(ledger.read_bytes()).hexdigest()
    changed = _payload()
    changed["tam"] = "Changed thesis under an already-approved version"
    source = tmp_path / "changed.json"
    source.write_text(json.dumps(changed), encoding="utf-8")
    result = import_pevc_draft(
        source,
        draft_dir=tmp_path / "private" / "drafts",
        canonical_path=canonical,
    )

    with pytest.raises(ValueError, match="revision is immutable"):
        store.approve_draft(
            result["draft_path"],
            expected_draft_hash=result["draft_hash"],
        )

    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == before_canonical
    assert hashlib.sha256(ledger.read_bytes()).hexdigest() == before_ledger


def test_invalid_as_of_and_trailing_ledger_garbage_fail_closed(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    store, _ = _import_and_approve(tmp_path, canonical)

    for invalid_as_of in ("not-a-date", "", " ", "2026-07-12junk"):
        with pytest.raises(ValueError, match="invalid as_of"):
            store.load(as_of=invalid_as_of)

    ledger = canonical.parent / "pevc_approval_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("trailing-garbage\n")
    with pytest.raises(ValueError, match="invalid PEVC approval event"):
        store.load()


@pytest.mark.parametrize(
    "crash_stage",
    [
        "wal_prepared",
        "archive_written",
        "canonical_written",
        "ledger_written",
        "transaction_committed",
    ],
)
def test_approval_crash_windows_recover_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_stage: str,
) -> None:
    source = tmp_path / "thesis.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    result = import_pevc_draft(
        source,
        draft_dir=tmp_path / "private" / "drafts",
        canonical_path=canonical,
    )
    initialize_pevc_approval_key(canonical)
    store = PeVcKnowledgeStore(canonical)

    original_checkpoint = pevc_module._transaction_checkpoint
    crashed = False

    def _crash(stage: str) -> None:
        nonlocal crashed
        if stage == crash_stage and not crashed:
            crashed = True
            raise RuntimeError(f"simulated crash at {stage}")

    monkeypatch.setattr(pevc_module, "_transaction_checkpoint", _crash)
    with pytest.raises(RuntimeError, match="simulated crash"):
        store.approve_draft(
            result["draft_path"],
            expected_draft_hash=result["draft_hash"],
        )

    assert store.transaction_wal_path.exists()
    with pytest.raises(ValueError, match="recovery is required"):
        store.load()
    monkeypatch.setattr(
        pevc_module,
        "_transaction_checkpoint",
        original_checkpoint,
    )
    approved = store.approve_draft(
        result["draft_path"],
        expected_draft_hash=result["draft_hash"],
    )

    assert approved.thesis_id == "ai-infra-001"
    assert not store.transaction_wal_path.exists()
    assert [item.thesis_id for item in store.load()] == ["ai-infra-001"]


def test_backdated_approval_requires_migration_evidence(tmp_path: Path) -> None:
    source = tmp_path / "thesis.json"
    source.write_text(json.dumps(_payload()), encoding="utf-8")
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    result = import_pevc_draft(
        source,
        draft_dir=tmp_path / "private" / "drafts",
        canonical_path=canonical,
    )
    initialize_pevc_approval_key(canonical)
    store = PeVcKnowledgeStore(canonical)
    backdated = _day_offset(-1)

    with pytest.raises(ValueError, match="explicit migration_mode"):
        store.approve_draft(
            result["draft_path"],
            expected_draft_hash=result["draft_hash"],
            approved_at=backdated,
        )
    with pytest.raises(ValueError, match="evidence file"):
        store.approve_draft(
            result["draft_path"],
            expected_draft_hash=result["draft_hash"],
            approved_at=backdated,
            migration_mode=True,
        )

    evidence = tmp_path / "migration-evidence.json"
    evidence.write_text('{"source":"historic approval register"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="permissions must be 0600"):
        store.approve_draft(
            result["draft_path"],
            expected_draft_hash=result["draft_hash"],
            approved_at=backdated,
            migration_mode=True,
            migration_evidence_file=evidence,
            expected_migration_evidence_hash="0" * 64,
        )
    evidence.chmod(0o600)
    evidence_hash = hashlib.sha256(evidence.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="hash changed"):
        store.approve_draft(
            result["draft_path"],
            expected_draft_hash=result["draft_hash"],
            approved_at=backdated,
            migration_mode=True,
            migration_evidence_file=evidence,
            expected_migration_evidence_hash="0" * 64,
        )

    approved = store.approve_draft(
        result["draft_path"],
        expected_draft_hash=result["draft_hash"],
        approved_at=backdated,
        migration_mode=True,
        migration_evidence_file=evidence,
        expected_migration_evidence_hash=evidence_hash,
    )

    assert approved.approved_at == backdated
    assert approved.available_at == _today()
    archive = (
        canonical.parent
        / "migration_evidence"
        / f"{evidence_hash}.evidence"
    )
    assert stat.S_IMODE(archive.stat().st_mode) == 0o600
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == evidence_hash
    evidence.unlink()
    assert store.load(as_of=backdated) == []
    assert [item.thesis_id for item in store.load(as_of=_today())] == [
        "ai-infra-001"
    ]


def test_canonical_revisions_use_natural_version_order(tmp_path: Path) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    version_ten = _payload()
    version_ten["version"] = "v10"
    store, _ = _import_and_approve(
        tmp_path,
        canonical,
        payload=version_ten,
        source_name="v10.json",
    )
    version_nine = _payload()
    version_nine["version"] = "v9"
    _import_and_approve(
        tmp_path,
        canonical,
        payload=version_nine,
        source_name="v9.json",
    )

    assert [item.version for item in store.load()] == ["v9", "v10"]
    canonical_versions = [
        json.loads(line)["version"]
        for line in canonical.read_text(encoding="utf-8").splitlines()
    ]
    assert canonical_versions == ["v9", "v10"]


def test_natural_version_alias_is_same_immutable_revision(tmp_path: Path) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    first = _payload()
    first["version"] = "v1"
    store, _ = _import_and_approve(
        tmp_path,
        canonical,
        payload=first,
        source_name="v1.json",
    )
    alias = _payload()
    alias["version"] = "1"
    alias["tam"] = "Changed content under a natural-version alias"
    source = tmp_path / "alias-1.json"
    source.write_text(json.dumps(alias), encoding="utf-8")
    result = import_pevc_draft(
        source,
        draft_dir=tmp_path / "private" / "drafts",
        canonical_path=canonical,
    )

    with pytest.raises(ValueError, match="revision is immutable"):
        store.approve_draft(
            result["draft_path"],
            expected_draft_hash=result["draft_hash"],
        )


def test_load_rejects_hmac_valid_natural_version_alias_duplicate(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "private" / "pevc_theses.jsonl"
    first = _payload()
    first["version"] = "v1"
    store, incumbent = _import_and_approve(
        tmp_path,
        canonical,
        payload=first,
        source_name="v1.json",
    )
    alias_payload = incumbent.to_dict()
    alias_payload["version"] = "1"
    alias_payload["tam"] = "HMAC-valid alias duplicate"
    alias_payload["content_hash"] = ""
    alias_payload["metadata"] = {
        **dict(alias_payload["metadata"]),
        "approval_draft_hash": "b" * 64,
    }
    provisional = PeVcThesis.from_mapping(alias_payload)
    alias_thesis = PeVcThesis.from_mapping(
        provisional.to_dict(),
        require_approved=True,
    )
    records = [incumbent, alias_thesis]
    records.sort(key=pevc_module._thesis_revision_key)
    canonical_after_text = "".join(
        json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for item in records
    )
    archive = canonical.parent / "approved_drafts" / f"{'b' * 64}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"alias-archive")
    archive.chmod(0o600)
    alias_archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    alias_payload = alias_thesis.to_dict()
    alias_payload["metadata"] = {
        **dict(alias_payload["metadata"]),
        "approval_draft_hash": alias_archive_hash,
    }
    alias_payload["content_hash"] = ""
    alias_thesis = PeVcThesis.from_mapping(
        PeVcThesis.from_mapping(alias_payload).to_dict(),
        require_approved=True,
    )
    records = [incumbent, alias_thesis]
    records.sort(key=pevc_module._thesis_revision_key)
    canonical_after_text = "".join(
        json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for item in records
    )
    archive_target = (
        canonical.parent
        / "approved_drafts"
        / f"{alias_archive_hash}.json"
    )
    archive.replace(archive_target)
    event_one = json.loads(
        store.approval_ledger_path.read_text(encoding="utf-8")
    )
    recorded_at = pevc_module._now_utc()
    key = store.approval_key_path.read_bytes()
    event_two = pevc_module._signed_approval_event(
        {
            "schema_version": "pevc_approval_ledger.v2",
            "approved_at": pevc_module._shanghai_business_date(
                recorded_at
            ).isoformat(),
            "recorded_at": recorded_at.isoformat(),
            "thesis_id": alias_thesis.thesis_id,
            "theme_id": alias_thesis.theme_id,
            "version": alias_thesis.version,
            "draft_hash": alias_archive_hash,
            "draft_archive_hash": alias_archive_hash,
            "draft_archive_path_summary": (
                f"approved_drafts/{alias_archive_hash}.json"
            ),
            "content_hash": alias_thesis.content_hash,
            "canonical_before_sha256": hashlib.sha256(
                canonical.read_bytes()
            ).hexdigest(),
            "canonical_after_sha256": hashlib.sha256(
                canonical_after_text.encode("utf-8")
            ).hexdigest(),
            "prev_event_hash": event_one["event_hash"],
            "migration_mode": False,
            "migration_evidence_hash": None,
            "migration_evidence_path_summary": None,
        },
        key=key,
    )
    canonical.write_text(canonical_after_text, encoding="utf-8")
    store.approval_ledger_path.write_text(
        json.dumps(event_one, ensure_ascii=False, sort_keys=True)
        + "\n"
        + json.dumps(event_two, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate approved PEVC revision"):
        store.load()
