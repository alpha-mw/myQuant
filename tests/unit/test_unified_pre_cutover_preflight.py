from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
VERIFIER_PATH = (
    ROOT / "operations/unified_cutover/verify_pre_cutover_dirty_inventory.py"
)
SPEC = importlib.util.spec_from_file_location("pre_cutover_preflight", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
SPEC.loader.exec_module(VERIFIER)

ABSORBED = VERIFIER.ABSORBED
BLOCKED_ABSORPTION = VERIFIER.BLOCKED_ABSORPTION
BLOCKED_HEAD_DRIFT = VERIFIER.BLOCKED_HEAD_DRIFT
BLOCKED_INCOMPLETE = VERIFIER.BLOCKED_INCOMPLETE
BLOCKED_INVENTORY_DRIFT = VERIFIER.BLOCKED_INVENTORY_DRIFT
BLOCKED_MANIFEST = VERIFIER.BLOCKED_MANIFEST
BLOCKED_UNCONFIRMED = VERIFIER.BLOCKED_UNCONFIRMED
EXPLICITLY_DISPOSITIONED = VERIFIER.EXPLICITLY_DISPOSITIONED
PreCutoverPreflightError = VERIFIER.PreCutoverPreflightError
build_unconfirmed_inventory = VERIFIER.build_unconfirmed_inventory
canonical_json_bytes = VERIFIER.canonical_json_bytes
validate_manifest_bytes = VERIFIER.validate_manifest_bytes
verify_pre_cutover_preflight = VERIFIER.verify_pre_cutover_preflight

CAPTURED_AT = "2026-08-14T11:19:24Z"
CONFIRMED_AT = "2026-08-14T11:19:25Z"

EXPECTED_REPOSITORY_ROWS = [
    {
        "byte_sha256": (
            "aa97f1c15783cdb4ba86d0d6e6df40e83dbae4a61c603b69b95065764a7026eb"
        ),
        "disposition_reason": None,
        "path": "portfolio_dashboard/public.html",
        "size": 9956,
        "status": " M",
        "user_disposition": "UNCONFIRMED",
    },
    {
        "byte_sha256": (
            "69201503f36216da38060efd29ffdbba66c9110d8968bf059ba56008b0fe54bd"
        ),
        "disposition_reason": None,
        "path": (
            "portfolio_dashboard/tests/"
            "cn_aggressive_dashboard_contract_v1.test.js"
        ),
        "size": 14803,
        "status": " M",
        "user_disposition": "UNCONFIRMED",
    },
    {
        "byte_sha256": (
            "51ae8764c6b75cd389bec752ffb10fa502b2a73b130423523b36a5744b82844e"
        ),
        "disposition_reason": None,
        "path": "scripts/build_cn_dashboard_public_site.py",
        "size": 8338,
        "status": " M",
        "user_disposition": "UNCONFIRMED",
    },
    {
        "byte_sha256": (
            "56b60641156d84ed423476eb568f5e5e3b687af2fe1e6b2fe3af39d94f913c0e"
        ),
        "disposition_reason": None,
        "path": "scripts/check_strategy_record_access.py",
        "size": 13390,
        "status": " M",
        "user_disposition": "UNCONFIRMED",
    },
    {
        "byte_sha256": (
            "671928e6ea3fd401c9f734cbb88b37648d82469fdc855a149521b6e62f523602"
        ),
        "disposition_reason": None,
        "path": "scripts/close_cn_dashboard_official_valuation.py",
        "size": 28864,
        "status": " M",
        "user_disposition": "UNCONFIRMED",
    },
    {
        "byte_sha256": (
            "638b698c6eeab833f23f32c5ade483e7bd736a9e1a1e92300eb3953de07109aa"
        ),
        "disposition_reason": None,
        "path": "tests/unit/test_close_cn_dashboard_official_valuation.py",
        "size": 11785,
        "status": "??",
        "user_disposition": "UNCONFIRMED",
    },
    {
        "byte_sha256": (
            "dc6b97ec103ab4772183da8d975379c146515edd71c4c69c0c5ff506c79f684f"
        ),
        "disposition_reason": None,
        "path": "tests/unit/test_cn_dashboard_export.py",
        "size": 64876,
        "status": " M",
        "user_disposition": "UNCONFIRMED",
    },
]


def _git(checkout: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _new_dirty_checkout(tmp_path: Path) -> tuple[Path, dict[str, Any], Path]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    _git(checkout, "init", "--quiet", "--initial-branch=main")
    _git(checkout, "config", "user.name", "Preflight Fixture")
    _git(checkout, "config", "user.email", "preflight-fixture@example.invalid")
    (checkout / "tracked.txt").write_bytes(b"baseline\n")
    _git(checkout, "add", "tracked.txt")
    _git(checkout, "commit", "--quiet", "-m", "baseline")

    (checkout / "tracked.txt").write_bytes(b"captured tracked bytes\n")
    (checkout / "untracked.txt").write_bytes(b"captured untracked bytes\n")
    document = build_unconfirmed_inventory(checkout, captured_at=CAPTURED_AT)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, document)
    return checkout, document, manifest_path


def _write_manifest(path: Path, document: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(document))


def _confirm(
    document: dict[str, Any],
    *,
    integration_commit: str,
    disposition: str,
) -> dict[str, Any]:
    confirmed = deepcopy(document)
    for entry in confirmed["entries"]:
        entry["user_disposition"] = disposition
        entry["disposition_reason"] = (
            "User confirmed byte-exact integration."
            if disposition == ABSORBED
            else "User confirmed that this captured row is intentionally excluded."
        )
    confirmed["integration_commit"] = integration_commit
    confirmed["user_confirmed_at"] = CONFIRMED_AT
    confirmed["user_confirmed_by"] = "fixture-user"
    return confirmed


def _assert_blocker(code: str, operation: Any) -> None:
    with pytest.raises(PreCutoverPreflightError) as captured:
        operation()
    assert captured.value.code == code


def test_repository_inventory_is_exact_canonical_unconfirmed_capture() -> None:
    path = ROOT / "operations/unified_cutover/pre-cutover-dirty-inventory.json"
    raw = path.read_bytes()
    document = validate_manifest_bytes(raw)

    assert raw == canonical_json_bytes(document)
    assert not raw.endswith(b"\n")
    assert hashlib.sha256(raw).hexdigest() == (
        "1f006225194771a798152e5573a2d7c3b11808982365c1f6a7655d1f59806dcd"
    )
    assert document["main_head"] == "8612ce51d8cb2b13076af60ac059a921dc104129"
    assert document["captured_at"] == CAPTURED_AT
    assert document["inventory_sha256"] == (
        "6bd29920affd707969d1b119e557632e13419795379eb3f311f06c503c3225aa"
    )
    assert document["entries"] == EXPECTED_REPOSITORY_ROWS
    assert document["integration_commit"] is None
    assert document["user_confirmed_at"] is None
    assert document["user_confirmed_by"] is None


def test_exact_unconfirmed_checkout_is_a_hard_stop(tmp_path: Path) -> None:
    checkout, _, manifest_path = _new_dirty_checkout(tmp_path)

    _assert_blocker(
        BLOCKED_UNCONFIRMED,
        lambda: verify_pre_cutover_preflight(
            manifest_path=manifest_path,
            checkout_root=checkout,
        ),
    )


@pytest.mark.parametrize("drift", ["bytes", "status", "path"])
def test_unconfirmed_checkout_rejects_inventory_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    checkout, _, manifest_path = _new_dirty_checkout(tmp_path)
    if drift == "bytes":
        (checkout / "tracked.txt").write_bytes(b"drifted bytes\n")
    elif drift == "status":
        _git(checkout, "add", "tracked.txt")
    else:
        (checkout / "new-path.txt").write_bytes(b"unexpected path\n")

    _assert_blocker(
        BLOCKED_INVENTORY_DRIFT,
        lambda: verify_pre_cutover_preflight(
            manifest_path=manifest_path,
            checkout_root=checkout,
        ),
    )


def test_unconfirmed_checkout_rejects_head_drift(tmp_path: Path) -> None:
    checkout, _, manifest_path = _new_dirty_checkout(tmp_path)
    _git(checkout, "commit", "--quiet", "--allow-empty", "-m", "head drift")

    _assert_blocker(
        BLOCKED_HEAD_DRIFT,
        lambda: verify_pre_cutover_preflight(
            manifest_path=manifest_path,
            checkout_root=checkout,
        ),
    )


def test_partially_dispositioned_capture_is_a_hard_stop(tmp_path: Path) -> None:
    checkout, document, manifest_path = _new_dirty_checkout(tmp_path)
    document["entries"][0]["user_disposition"] = ABSORBED
    document["entries"][0]["disposition_reason"] = "User disposition pending closure."
    _write_manifest(manifest_path, document)

    _assert_blocker(
        BLOCKED_INCOMPLETE,
        lambda: verify_pre_cutover_preflight(
            manifest_path=manifest_path,
            checkout_root=checkout,
        ),
    )


def test_clean_byte_exact_integration_commit_satisfies_only_this_gate(
    tmp_path: Path,
) -> None:
    checkout, document, manifest_path = _new_dirty_checkout(tmp_path)
    _git(checkout, "add", "--all")
    _git(checkout, "commit", "--quiet", "-m", "absorb captured inventory")
    integration_commit = _git(checkout, "rev-parse", "HEAD")
    confirmed = _confirm(
        document,
        integration_commit=integration_commit,
        disposition=ABSORBED,
    )
    _write_manifest(manifest_path, confirmed)

    report = verify_pre_cutover_preflight(
        manifest_path=manifest_path,
        checkout_root=checkout,
    )

    assert report["verified"] is True
    assert report["dirty_inventory_gate_satisfied"] is True
    assert report["absorbed_count"] == 2
    assert report["explicitly_dispositioned_count"] == 0
    assert report["final_build_authorized"] is False
    assert report["cas_authorized"] is False
    assert report["external_write_performed"] is False


def test_clean_explicit_disposition_commit_satisfies_only_this_gate(
    tmp_path: Path,
) -> None:
    checkout, document, manifest_path = _new_dirty_checkout(tmp_path)
    (checkout / "tracked.txt").write_bytes(b"baseline\n")
    (checkout / "untracked.txt").unlink()
    _git(checkout, "commit", "--quiet", "--allow-empty", "-m", "explicit disposition")
    integration_commit = _git(checkout, "rev-parse", "HEAD")
    confirmed = _confirm(
        document,
        integration_commit=integration_commit,
        disposition=EXPLICITLY_DISPOSITIONED,
    )
    _write_manifest(manifest_path, confirmed)

    report = verify_pre_cutover_preflight(
        manifest_path=manifest_path,
        checkout_root=checkout,
    )

    assert report["verified"] is True
    assert report["absorbed_count"] == 0
    assert report["explicitly_dispositioned_count"] == 2
    assert report["final_build_authorized"] is False
    assert report["cas_authorized"] is False


def test_absorbed_row_must_match_captured_bytes(tmp_path: Path) -> None:
    checkout, document, manifest_path = _new_dirty_checkout(tmp_path)
    (checkout / "tracked.txt").write_bytes(b"different committed bytes\n")
    _git(checkout, "add", "--all")
    _git(checkout, "commit", "--quiet", "-m", "wrong absorption")
    integration_commit = _git(checkout, "rev-parse", "HEAD")
    confirmed = _confirm(
        document,
        integration_commit=integration_commit,
        disposition=ABSORBED,
    )
    _write_manifest(manifest_path, confirmed)

    _assert_blocker(
        BLOCKED_ABSORPTION,
        lambda: verify_pre_cutover_preflight(
            manifest_path=manifest_path,
            checkout_root=checkout,
        ),
    )


def test_manifest_requires_exact_compact_canonical_bytes(tmp_path: Path) -> None:
    _, document, manifest_path = _new_dirty_checkout(tmp_path)
    manifest_path.write_text(json.dumps(document, ensure_ascii=False, indent=2))

    _assert_blocker(
        BLOCKED_MANIFEST,
        lambda: validate_manifest_bytes(manifest_path.read_bytes()),
    )
