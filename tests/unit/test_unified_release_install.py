from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.system import (
    SystemPreconditionError,
    object_ref_for_artifact,
    prepare_operational_release,
    validate_release_install_evidence,
    verify_release_install_input,
)

BASE = "2026-08-16T00:00:00Z"


def _git(root: Path, *arguments: str) -> str:
    return (
        subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        .stdout.decode("ascii")
        .strip()
    )


def test_frozen_release_build_install_and_exact_origin_replay(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[2]
    repository = tmp_path / "repository"
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(source), str(repository)],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    commit = _git(repository, "rev-parse", "HEAD^{commit}")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    release_root = tmp_path / "release"
    release_root.mkdir(mode=0o700)

    prepared = prepare_operational_release(
        repository_root=repository,
        release_root=release_root,
        final_commit=commit,
        final_tree=tree,
        created_at=BASE,
    )
    release = prepared["release"]
    evidence = validate_release_install_evidence(prepared["release_install_evidence"])
    assert evidence["payload"]["release_ref"] == object_ref_for_artifact(release)
    exact_input = canonical_json_bytes(
        {"release_install_evidence": evidence, "deployed_release": release}
    )
    assert verify_release_install_input(exact_input, repository_root=repository)["state"] == "PASS"
    assert Path(evidence["payload"]["import_origin"]).is_relative_to(
        Path(evidence["payload"]["install_root"])
    )
    assert not Path(evidence["payload"]["import_origin"]).is_relative_to(repository)

    wheel = Path(evidence["payload"]["wheel"]["path"])
    wheel.write_bytes(wheel.read_bytes() + b"tamper")
    with pytest.raises(SystemPreconditionError, match="archive exact bytes"):
        verify_release_install_input(exact_input, repository_root=repository)
