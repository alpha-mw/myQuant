from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_theme_governance_diagnostics_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "theme_snapshot.json"
    output_dir = tmp_path / "governance"
    snapshot_path.write_text(
        json.dumps(
            {
                "snapshot_schema_version": "theme_snapshot.v1",
                "market": "CN",
                "universe_key": "full_a",
                "as_of": "20260618",
                "theme_rotation": {
                    "schema_version": "theme_rotation.v1",
                    "enabled": True,
                    "status": "success",
                    "market": "CN",
                    "universe_key": "full_a",
                    "as_of": "20260618",
                    "theme_scores": {
                        "industry::ai": {
                            "theme_id": "industry::ai",
                            "theme_name": "AI",
                            "score": 72,
                            "smoothed_score": 61,
                            "heat_10d": 61,
                            "heat_delta_5d": 3,
                            "persistence_count": 6,
                            "trend_state": "warming",
                            "smoothing_status": "success",
                            "confidence": 0.66,
                            "breadth": 0.52,
                            "member_count": 18,
                            "phase": "confirmed_rotation",
                            "risk_flags": [],
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_theme_governance_diagnostics.py",
            "--market",
            "CN",
            "--universe-key",
            "full_a",
            "--snapshot-json",
            str(snapshot_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    json_files = sorted(output_dir.rglob("*.json"))
    markdown_files = sorted(output_dir.rglob("*.md"))
    assert len(json_files) == 1
    assert len(markdown_files) == 1

    payload = json.loads(json_files[0].read_text(encoding="utf-8"))
    markdown = markdown_files[0].read_text(encoding="utf-8")

    assert payload["schema_version"] == "theme_governance.v1"
    assert payload["decisions"][0]["gate_label"] == "admitted_shadow"
    assert "shadow/governance only" in markdown
    assert "final executable decision remains baseline" in markdown
