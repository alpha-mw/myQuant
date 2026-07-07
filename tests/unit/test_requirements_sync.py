from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_requirements_sync.py"


def test_requirements_sync_script_accepts_repo_files() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pyproject",
            str(ROOT / "pyproject.toml"),
            "--requirements",
            str(ROOT / "requirements.txt"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "in sync" in result.stdout


def test_requirements_sync_script_reports_missing_dependency(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[project]
dependencies = [
  "fastapi>=0.115.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=7.4.0",
]
""".strip(),
        encoding="utf-8",
    )
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("fastapi>=0.115.0\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pyproject",
            str(pyproject),
            "--requirements",
            str(requirements),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "pytest>=7.4.0" in result.stderr
