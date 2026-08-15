from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tomllib

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERIFIER_PATH = ROOT / "operations/codex/verify_projection.py"
SPEC = importlib.util.spec_from_file_location("codex_projection_verifier", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFIER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFIER
SPEC.loader.exec_module(VERIFIER)

AUTOMATION_EXCLUDED_FIELDS = VERIFIER.AUTOMATION_EXCLUDED_FIELDS
AUTOMATION_INCLUDED_FIELDS = VERIFIER.AUTOMATION_INCLUDED_FIELDS
LEGACY_SEED_RELATIVE = VERIFIER.LEGACY_SEED_RELATIVE
ProjectionVerificationError = VERIFIER.ProjectionVerificationError
canonical_json_bytes = VERIFIER.canonical_json_bytes
reject_removed_entrypoint_tokens = VERIFIER.reject_removed_entrypoint_tokens
validate_automation_projection = VERIFIER.validate_automation_projection
verify_projection = VERIFIER.verify_projection


def _copy_projection(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "operations").mkdir(parents=True)
    shutil.copytree(ROOT / "operations/codex", root / "operations/codex")
    seed_target = root / LEGACY_SEED_RELATIVE
    seed_target.parent.mkdir(parents=True)
    shutil.copy2(ROOT / LEGACY_SEED_RELATIVE, seed_target)
    return root


def test_repository_codex_projection_verifies() -> None:
    result = verify_projection(ROOT)

    assert result["verified"] is True
    assert result["manifest_byte_sha256"] == hashlib.sha256(
        (ROOT / "operations/codex/projection-manifest.json").read_bytes()
    ).hexdigest()
    assert result["automation_status"] == "PAUSED"
    assert result["external_deployment_performed"] is False
    assert result["activation_performed"] is False


def test_skill_tree_rejects_extra_file_symlink_mode_and_hash_drift(tmp_path: Path) -> None:
    extra_root = _copy_projection(tmp_path / "extra")
    (extra_root / "operations/codex/skills/myquant/extra.md").write_text("extra")
    with pytest.raises(ProjectionVerificationError, match="missing or extra"):
        verify_projection(extra_root)

    symlink_root = _copy_projection(tmp_path / "symlink")
    source = symlink_root / "operations/codex/skills/myquant/SKILL.md"
    link = symlink_root / "operations/codex/skills/myquant/link.md"
    link.symlink_to(source)
    with pytest.raises(ProjectionVerificationError, match="symlink"):
        verify_projection(symlink_root)

    mode_root = _copy_projection(tmp_path / "mode")
    mode_path = mode_root / "operations/codex/skills/myquant/SKILL.md"
    mode_path.chmod(0o600)
    with pytest.raises(ProjectionVerificationError, match="mode drift"):
        verify_projection(mode_root)

    hash_root = _copy_projection(tmp_path / "hash")
    hash_path = hash_root / "operations/codex/skills/myquant/SKILL.md"
    hash_path.write_text(hash_path.read_text() + "\nchanged\n")
    with pytest.raises(ProjectionVerificationError, match="mapping mismatch"):
        verify_projection(hash_root)


def test_automation_projection_requires_exact_fields_and_paused_status() -> None:
    source = ROOT / "operations/codex/automations/myquant-2/automation.toml"
    document = tomllib.loads(source.read_text())

    projection = validate_automation_projection(document)
    assert tuple(projection) == AUTOMATION_INCLUDED_FIELDS
    assert set(document) - set(projection) == set(AUTOMATION_EXCLUDED_FIELDS)

    extra = {**document, "unexpected": True}
    with pytest.raises(ProjectionVerificationError, match="field set"):
        validate_automation_projection(extra)

    active = {**document, "status": "ACTIVE"}
    with pytest.raises(ProjectionVerificationError, match="remain PAUSED"):
        validate_automation_projection(active)


def test_manifest_requires_exact_canonical_bytes_and_contract_hash(tmp_path: Path) -> None:
    pretty_root = _copy_projection(tmp_path / "pretty")
    pretty_path = pretty_root / "operations/codex/projection-manifest.json"
    pretty = json.loads(pretty_path.read_bytes())
    pretty_path.write_text(json.dumps(pretty, ensure_ascii=False, indent=2))
    with pytest.raises(ProjectionVerificationError, match="compact canonical"):
        verify_projection(pretty_root)

    contract_root = _copy_projection(tmp_path / "contract")
    contract_path = contract_root / "operations/codex/projection-manifest.json"
    contract = json.loads(contract_path.read_bytes())
    contract["contract_sha256"] = "0" * 64
    contract_path.write_bytes(canonical_json_bytes(contract))
    with pytest.raises(ProjectionVerificationError, match="contract hash"):
        verify_projection(contract_root)


def test_removed_entrypoint_tokens_are_rejected_from_deployment_text(
    tmp_path: Path,
) -> None:
    seed = json.loads((ROOT / LEGACY_SEED_RELATIVE).read_bytes())
    token = seed["removed_entrypoint_tokens"][0]

    with pytest.raises(ProjectionVerificationError, match="removed entrypoint"):
        reject_removed_entrypoint_tokens({"projection": token}, [token])

    root = _copy_projection(tmp_path)
    readme = root / "operations/codex/README.md"
    readme.write_text(readme.read_text() + f"\n{token}\n")
    with pytest.raises(ProjectionVerificationError, match="removed entrypoint"):
        verify_projection(root)
