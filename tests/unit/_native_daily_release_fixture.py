"""Real isolated release/install evidence for synthetic native DAG integration.

Snapshots current code into a disposable local Git repository; never commits the
user checkout, pushes, activates System/Factor, or changes the production install.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess

from quant_investor.contracts import canonical_json_bytes
from quant_investor.system.release_install import prepare_operational_release


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def snapshot_repository(source: Path, repository: Path) -> str:
    """Freeze the running source in a disposable repo, preserving caller checkout."""
    subprocess.run(
        ["git", "clone", "--quiet", "--shared", str(source), str(repository)], check=True
    )
    # Snapshot all task implementation changes in a separate repository. Only
    # source/test/build paths are eligible; no private results or credentials.
    eligible = ("quant_investor/", "scripts/", "tests/", "portfolio_dashboard/")
    changes = (
        subprocess.check_output(
            [
                "git",
                "-C",
                str(source),
                "ls-files",
                "-z",
                "--cached",
                "--others",
                "--exclude-standard",
            ]
        )
        .decode()
        .split("\0")
    )
    # Include clone paths so staged deletions are removed as well as staged edits copied.
    changes.extend(
        subprocess.check_output(["git", "-C", str(repository), "ls-files", "-z"])
        .decode()
        .split("\0")
    )
    for name in sorted(set(changes)):
        if not name or not (name.startswith(eligible) or name in {"pyproject.toml", "uv.lock"}):
            continue
        current, target = source / name, repository / name
        if current.is_symlink():
            raise ValueError("synthetic release rejects source symlinks")
        if current.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(current, target)
        elif target.exists():
            target.unlink()
    git(repository, "add", "--all")
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Synthetic DAG Fixture",
            "-c",
            "user.email=synthetic-dag@example.invalid",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "Snapshot implementation for isolated native DAG tests",
        ],
        check=True,
    )
    commit = git(repository, "rev-parse", "HEAD")
    git(repository, "checkout", "--detach", "--quiet", commit)
    return commit


def prepare_synthetic_release(source: Path, root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=False)
    repository = root / "repository"
    commit = snapshot_repository(source, repository)
    release_root = root / "release"
    release_root.mkdir(mode=0o700)
    prepared = prepare_operational_release(
        repository_root=repository,
        release_root=release_root,
        final_commit=commit,
        final_tree=git(repository, "rev-parse", "HEAD^{tree}"),
        created_at=None,
    )
    raw = canonical_json_bytes(
        {
            "deployed_release": prepared["release"],
            "release_install_evidence": prepared["release_install_evidence"],
        }
    )
    path = root / "release-input.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    python = prepared["release_install_evidence"]["payload"]["python_executable"]
    program = """import json,sys
from pathlib import Path
from quant_investor.system.release_install import verify_running_release_install_input
value = verify_running_release_install_input(
    Path(sys.argv[1]).read_bytes(), repository_root=Path(sys.argv[2]))
print(json.dumps(value))"""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [python, "-I", "-c", program, str(path), str(repository)],
        cwd=root,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    verified = json.loads(result.stdout)
    if verified["state"] != "PASS":
        raise AssertionError("native installed runtime verification failed")
    receipt = {
        "synthetic_test_environment": True,
        "repository": str(repository),
        "release_root": str(release_root),
        "release_input": str(path),
        "python": python,
        "commit": commit,
        "runtime_verification": verified,
        "full_dag_proof": False,
        "production_deployed": False,
    }
    (root / "fixture-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
