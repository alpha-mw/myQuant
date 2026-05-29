import json
import subprocess
import sys
from pathlib import Path

from quant_investor.factors.matrix import FactorMatrix
from quant_investor.factors.schema import (
    FACTOR_STATUS_PRODUCTION,
    FactorLibraryEntry,
    ProductionFactorLibrary,
)


def test_collect_factor_shadow_evidence_cli_smoke(tmp_path) -> None:
    library = ProductionFactorLibrary(
        library_id="prod-lib",
        generated_at="2026-04-01T00:00:00Z",
        entries=[
            FactorLibraryEntry(
                factor_id="factor-a",
                factor_version="v1",
                status=FACTOR_STATUS_PRODUCTION,
                admission_decision_id="decision-a",
                validation_report_id="validation-a",
                production_since="2026-04-01",
            )
        ],
    )
    matrix = FactorMatrix(
        matrix_id="matrix-a",
        factor_id="factor-a",
        factor_version="v1",
        expression="close / open",
        symbols=["AAA", "BBB"],
        dates=["2026-04-01"],
        values=[[2.0], [1.0]],
        coverage_ratio=1.0,
        missing_ratio=0.0,
    )
    library_path = tmp_path / "production_factors.json"
    matrix_path = tmp_path / "factor_matrices.jsonl"
    library_path.write_text(json.dumps(library.to_dict()), encoding="utf-8")
    matrix_path.write_text(json.dumps(matrix.to_dict()) + "\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "as_of_dates": ["2026-04-01"],
                "date_inputs": [
                    {
                        "as_of": "2026-04-01",
                        "candidates": [
                            {"symbol": "AAA", "official_score": 0.9, "official_rank": 1},
                            {"symbol": "BBB", "official_score": 0.8, "official_rank": 2},
                        ],
                        "production_library_path": str(library_path),
                        "factor_matrix_paths": [str(matrix_path)],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "evidence"
    script = Path("scripts/collect_factor_shadow_evidence.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--input-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--generated-at",
            "2026-04-03T00:00:00Z",
            "--min-observation-days",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "status:" in completed.stdout
    assert (output_dir / "evidence_report.md").exists()
    assert (output_dir / "evidence_dashboard.json").exists()
