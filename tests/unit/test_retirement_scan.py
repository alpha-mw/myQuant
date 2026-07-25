from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

import quant_investor.market.retirement_scan as scan_module
from quant_investor.market.retirement import (
    ACTIVE_SCHEDULE_NAMES,
    REQUIRED_SCHEDULE_NAMES,
)
from quant_investor.market.retirement_scan import (
    ALLOWLIST_SCHEMA,
    REQUIRED_EXTERNAL_LOGICAL_PATHS,
    REQUIRED_RUNTIME_DOMAINS,
    RUNTIME_EVIDENCE_SCHEMA,
    SCHEDULE_LOGICAL_PATHS,
    SKILL_LOGICAL_PATH,
    RetirementScanError,
    scan_retirement,
    validate_runtime_evidence,
)

NOW = datetime(2026, 7, 22, 12, 0, 0, tzinfo=timezone.utc)
CUTOVER_ID = "cutover-1"
REPO_SHA = hashlib.sha256(b"repo-snapshot").hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _z(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _external_files(
    tmp_path: Path,
    *,
    skill_payload: bytes = b"installed skill points to v15 production\n",
) -> tuple[dict[str, Path], str, dict[str, str]]:
    external_root = tmp_path / "external-evidence"
    skill = external_root / "skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_bytes(skill_payload)
    files: dict[str, Path] = {SKILL_LOGICAL_PATH: skill}
    schedule_hashes: dict[str, str] = {}
    for name in REQUIRED_SCHEDULE_NAMES:
        schedule = external_root / "schedules" / f"{name}.toml"
        schedule.parent.mkdir(parents=True, exist_ok=True)
        payload = f"name={name}\nlane=v15-production\n".encode()
        schedule.write_bytes(payload)
        if name in ACTIVE_SCHEDULE_NAMES:
            files[SCHEDULE_LOGICAL_PATHS[name]] = schedule
        schedule_hashes[name] = _sha(payload)
    return files, _sha(skill_payload), schedule_hashes


def _bindings(
    repo: Path,
    skill_sha: str,
    schedule_shas: dict[str, str],
) -> dict[str, object]:
    return {
        "cutover_id": CUTOVER_ID,
        "repo_root": repo,
        "repo_sha256": REPO_SHA,
        "skill_sha256": skill_sha,
        "schedule_sha256": schedule_shas,
    }


def _runtime(
    path: Path,
    *,
    bindings: dict[str, object],
    generated_at: datetime = NOW - timedelta(minutes=1),
    expires_at: datetime = NOW + timedelta(minutes=5),
    bad_domain: str = "",
    match: bool = False,
) -> Path:
    raw_root = path.parent / f"{path.stem}-raw"
    domains: dict[str, dict[str, object]] = {}
    for name in REQUIRED_RUNTIME_DOMAINS:
        raw = raw_root / f"{name}.txt"
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw_payload = f"{name}: complete and clean\n".encode()
        raw.write_bytes(raw_payload)
        domains[name] = {
            "status": "INCOMPLETE" if name == bad_domain else "COMPLETE",
            "matches": ["legacy consumer"] if match and name == "processes" else [],
            "evidence_path": str(raw.absolute()),
            "evidence_sha256": _sha(raw_payload),
            "exit_code": 0,
        }
    value = {
        "schema_version": RUNTIME_EVIDENCE_SCHEMA,
        "cutover_id": bindings["cutover_id"],
        "repo_root": str(Path(bindings["repo_root"]).resolve()),
        "repo_sha256": bindings["repo_sha256"],
        "skill_sha256": bindings["skill_sha256"],
        "schedule_sha256": dict(sorted(bindings["schedule_sha256"].items())),
        "generated_at": _z(generated_at),
        "expires_at": _z(expires_at),
        "domains": domains,
        "authority": False,
    }
    value["semantic_sha256"] = _sha(_canonical(value))
    _write_json(path, value)
    return path


def _rewrite_runtime(path: Path, mutate: object) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    value.pop("semantic_sha256", None)
    value["semantic_sha256"] = _sha(_canonical(value))
    _write_json(path, value)


def _allowlist(path: Path, entries: list[dict[str, object]]) -> Path:
    _write_json(
        path,
        {
            "schema_version": ALLOWLIST_SCHEMA,
            "authority": False,
            "entries": entries,
        },
    )
    return path


def _entry(path: str, detector: str, count: int, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "detector": detector,
        "exact_count": count,
        "reason": "immutable historical reference",
        "expected_sha256": _sha(payload),
    }


def _fixture(
    tmp_path: Path,
    *,
    allowlist_entries: list[dict[str, object]] | None = None,
    skill_payload: bytes = b"installed skill points to v15 production\n",
) -> tuple[
    Path,
    Path,
    Path,
    dict[str, Path],
    dict[str, object],
]:
    repo = tmp_path / "repo"
    repo.mkdir()
    external, skill_sha, schedule_shas = _external_files(
        tmp_path,
        skill_payload=skill_payload,
    )
    bindings = _bindings(repo, skill_sha, schedule_shas)
    allowlist = _allowlist(repo / "allowlist.json", allowlist_entries or [])
    runtime = _runtime(tmp_path / "runtime.json", bindings=bindings)
    return repo, allowlist, runtime, external, bindings


def _scan(
    allowlist: Path,
    runtime: Path,
    external: dict[str, Path],
    bindings: dict[str, object],
) -> dict[str, object]:
    return scan_retirement(
        allowlist_path=allowlist,
        runtime_evidence_path=runtime,
        external_files=external,
        now_utc=NOW,
        **bindings,
    )


def test_clean_scan_binds_exact_external_and_runtime_evidence(tmp_path: Path) -> None:
    repo, allowlist, runtime, external, bindings = _fixture(tmp_path)
    (repo / "safe.py").write_text("VALUE = 'v15'\n", encoding="utf-8")

    report = _scan(allowlist, runtime, external, bindings)

    assert report["status"] == "CLEAN"
    assert report["unknown"] == []
    assert set(report["external_file_sha256"]) == REQUIRED_EXTERNAL_LOGICAL_PATHS
    assert report["repo_sha256"] == REPO_SHA
    assert len(report["semantic_sha256"]) == 64


def test_checked_in_allowlist_exactly_matches_repository_findings() -> None:
    repo = Path(__file__).resolve().parents[2]
    allowlist = (
        repo
        / "quant_investor"
        / "market"
        / "resources"
        / "retirement_scan_allowlist.json"
    )
    _, allowed = scan_module._load_allowlist(allowlist)
    inventory = scan_module._inventory(repo, allowlist)
    findings: dict[tuple[str, str], tuple[int, str]] = {}
    for logical_path, (path, _fingerprint) in inventory.items():
        payload, kind, _ = scan_module._stable_file_payload(path)
        digest = hashlib.sha256(payload).hexdigest()
        for detector, count in scan_module._detect(logical_path, payload, kind).items():
            findings[(logical_path, detector)] = (count, digest)

    assert set(findings) == set(allowed)
    for key, (count, digest) in findings.items():
        assert allowed[key]["exact_count"] == count
        assert allowed[key]["expected_sha256"] == digest


def test_unknown_text_or_python_import_blocks(tmp_path: Path) -> None:
    repo, allowlist, runtime, external, bindings = _fixture(tmp_path)
    (repo / "active.py").write_text(
        "from quant_investor.v16 import runtime\nVALUE = 'v16'\n",
        encoding="utf-8",
    )

    with pytest.raises(RetirementScanError) as exc_info:
        _scan(allowlist, runtime, external, bindings)

    message = str(exc_info.value)
    assert "text.v16-token" in message
    assert "python.import-v16" in message
    assert "python.string-v16" in message


def test_active_quant_investor_data_package_is_not_excluded(tmp_path: Path) -> None:
    repo, allowlist, runtime, external, bindings = _fixture(tmp_path)
    active = repo / "quant_investor" / "data" / "active.py"
    active.parent.mkdir(parents=True)
    active.write_text("VALUE = 'v16'\n", encoding="utf-8")
    root_data = repo / "data" / "large-market-cache.txt"
    root_data.parent.mkdir()
    root_data.write_text("v16 excluded data artifact", encoding="utf-8")

    with pytest.raises(RetirementScanError) as exc_info:
        _scan(allowlist, runtime, external, bindings)
    message = str(exc_info.value)
    assert "quant_investor/data/active.py" in message
    assert "large-market-cache.txt" not in message


def test_hash_and_exact_count_bound_historical_allowlist(tmp_path: Path) -> None:
    payload = b"retired v16 history\n"
    entries = [_entry("docs/history.md", "text.v16-token", 1, payload)]
    repo, allowlist, runtime, external, bindings = _fixture(
        tmp_path,
        allowlist_entries=entries,
    )
    history = repo / "docs" / "history.md"
    history.parent.mkdir(parents=True)
    history.write_bytes(payload)

    assert _scan(allowlist, runtime, external, bindings)["status"] == "CLEAN"

    history.write_bytes(b"retired v16 history changed\n")
    with pytest.raises(RetirementScanError, match="mismatches"):
        _scan(allowlist, runtime, external, bindings)


def test_stale_allowlist_and_self_reference_are_rejected(tmp_path: Path) -> None:
    entries = [_entry("missing.md", "text.v16-token", 1, b"v16")]
    repo, allowlist, runtime, external, bindings = _fixture(
        tmp_path,
        allowlist_entries=entries,
    )
    with pytest.raises(RetirementScanError, match="stale_allowlist_entries"):
        _scan(allowlist, runtime, external, bindings)

    _allowlist(
        allowlist,
        [_entry("allowlist.json", "text.v16-token", 1, allowlist.read_bytes())],
    )
    with pytest.raises(RetirementScanError, match="reference itself"):
        _scan(allowlist, runtime, external, bindings)


def test_allowlist_must_be_a_real_regular_file_inside_repository(tmp_path: Path) -> None:
    repo, allowlist, runtime, external, bindings = _fixture(tmp_path)
    outside = tmp_path / "outside-allowlist.json"
    outside.write_bytes(allowlist.read_bytes())
    allowlist.unlink()
    allowlist.symlink_to(outside)

    with pytest.raises(RetirementScanError, match="JSON evidence cannot be a symlink"):
        _scan(allowlist, runtime, external, bindings)

    allowlist.unlink()
    with pytest.raises(RetirementScanError, match="inside repo_root"):
        scan_retirement(
            allowlist_path=outside,
            runtime_evidence_path=runtime,
            external_files=external,
            now_utc=NOW,
            **bindings,
        )


@pytest.mark.parametrize(
    ("bad_domain", "match", "message"),
    [
        ("crontab", False, "incomplete: crontab"),
        ("", True, "still has v16 matches: processes"),
    ],
)
def test_runtime_evidence_must_cover_every_clean_domain(
    tmp_path: Path,
    bad_domain: str,
    match: bool,
    message: str,
) -> None:
    repo, _, _, _, bindings = _fixture(tmp_path)
    path = _runtime(
        tmp_path / "runtime-test.json",
        bindings=bindings,
        bad_domain=bad_domain,
        match=match,
    )
    with pytest.raises(RetirementScanError, match=message):
        validate_runtime_evidence(path, now_utc=NOW, **bindings)
    assert repo.is_dir()


def test_runtime_raw_evidence_must_exist_and_match_real_bytes(tmp_path: Path) -> None:
    _, _, runtime, _, bindings = _fixture(tmp_path)
    value = json.loads(runtime.read_text(encoding="utf-8"))
    raw = Path(value["domains"]["processes"]["evidence_path"])
    raw.write_text("changed after receipt self-report", encoding="utf-8")

    with pytest.raises(RetirementScanError, match="raw evidence SHA mismatch"):
        validate_runtime_evidence(runtime, now_utc=NOW, **bindings)

    raw.unlink()
    with pytest.raises(RetirementScanError, match="missing or unreadable"):
        validate_runtime_evidence(runtime, now_utc=NOW, **bindings)


@pytest.mark.parametrize(
    ("generated_at", "expires_at"),
    [
        (NOW - timedelta(minutes=20), NOW - timedelta(minutes=1)),
        (NOW + timedelta(minutes=1), NOW + timedelta(minutes=5)),
        (NOW - timedelta(minutes=1), NOW + timedelta(minutes=30)),
    ],
)
def test_runtime_evidence_must_be_current_and_short_lived(
    tmp_path: Path,
    generated_at: datetime,
    expires_at: datetime,
) -> None:
    _, _, _, _, bindings = _fixture(tmp_path)
    runtime = _runtime(
        tmp_path / "freshness.json",
        bindings=bindings,
        generated_at=generated_at,
        expires_at=expires_at,
    )
    with pytest.raises(RetirementScanError, match="freshness|stale|future"):
        validate_runtime_evidence(runtime, now_utc=NOW, **bindings)


def test_runtime_cutover_binding_mismatch_is_rejected(tmp_path: Path) -> None:
    _, _, runtime, _, bindings = _fixture(tmp_path)
    wrong = dict(bindings)
    wrong["repo_sha256"] = _sha(b"other-repo")
    with pytest.raises(RetirementScanError, match="binding mismatch"):
        validate_runtime_evidence(runtime, now_utc=NOW, **wrong)


def test_external_skill_file_is_scanned_under_exact_logical_path(tmp_path: Path) -> None:
    skill_payload = b"historical v16 note\n"
    entries = [_entry(SKILL_LOGICAL_PATH, "text.v16-token", 1, skill_payload)]
    _, allowlist, runtime, external, bindings = _fixture(
        tmp_path,
        allowlist_entries=entries,
        skill_payload=skill_payload,
    )

    report = _scan(allowlist, runtime, external, bindings)
    assert report["status"] == "CLEAN"


def test_external_skill_and_all_nine_schedule_identities_are_mandatory(
    tmp_path: Path,
) -> None:
    _, allowlist, runtime, external, bindings = _fixture(tmp_path)
    missing = dict(external)
    missing.pop(SCHEDULE_LOGICAL_PATHS["automation"])
    with pytest.raises(RetirementScanError, match="exact installed skill and nine schedules"):
        _scan(allowlist, runtime, missing, bindings)

    substituted = dict(external)
    substituted[SCHEDULE_LOGICAL_PATHS["automation"]] = external[SCHEDULE_LOGICAL_PATHS["a-2"]]
    with pytest.raises(RetirementScanError, match="SHA mismatch"):
        _scan(allowlist, runtime, substituted, bindings)


def test_version_literal_symlink_and_utf16_binary_are_not_silently_skipped(
    tmp_path: Path,
) -> None:
    repo, allowlist, runtime, external, bindings = _fixture(tmp_path)
    (repo / "metadata.txt").write_text("version = 16.0.0\n", encoding="utf-8")
    (repo / "utf16.bin").write_bytes("v16".encode("utf-16-le"))
    target = tmp_path / "elsewhere"
    target.mkdir()
    (repo / "v16_link").symlink_to(target, target_is_directory=True)

    with pytest.raises(RetirementScanError) as exc_info:
        _scan(allowlist, runtime, external, bindings)
    message = str(exc_info.value)
    assert "text.version-16.0.0" in message
    assert "binary.v16" in message
    assert "path.v16" in message


def test_repository_inventory_drift_during_scan_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, allowlist, runtime, external, bindings = _fixture(tmp_path)
    safe = repo / "safe.py"
    safe.write_text("VALUE = 'v15'\n", encoding="utf-8")
    original = scan_module._stable_file_payload
    injected = False

    def mutate_after_read(path: Path) -> object:
        nonlocal injected
        result = original(path)
        if path == safe and not injected:
            injected = True
            (repo / "late.py").write_text("VALUE = 'v16'\n", encoding="utf-8")
        return result

    monkeypatch.setattr(scan_module, "_stable_file_payload", mutate_after_read)
    with pytest.raises(RetirementScanError, match="inventory changed"):
        _scan(allowlist, runtime, external, bindings)


def test_runtime_domains_cannot_reuse_one_self_reported_raw_file(tmp_path: Path) -> None:
    _, _, runtime, _, bindings = _fixture(tmp_path)

    def reuse(value: dict[str, object]) -> None:
        domains = value["domains"]
        domains["scripts"]["evidence_path"] = domains["automations"]["evidence_path"]
        domains["scripts"]["evidence_sha256"] = domains["automations"]["evidence_sha256"]

    _rewrite_runtime(runtime, reuse)
    with pytest.raises(RetirementScanError, match="distinct raw evidence"):
        validate_runtime_evidence(runtime, now_utc=NOW, **bindings)
