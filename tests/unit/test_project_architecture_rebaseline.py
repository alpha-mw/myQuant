"""Architecture rebaseline report contract tests."""

from __future__ import annotations

import json

from scripts.project_architecture_rebaseline import (
    build_architecture_rebaseline_audit,
    main as architecture_rebaseline_main,
    write_architecture_rebaseline_audit,
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _seed_rebaseline_ready_workspace(root):
    for path in [
        "quant_investor/market/analyze.py",
        "quant_investor/market/full_report_helpers.py",
        "quant_investor/market/full_report_sections.py",
        "quant_investor/market/report_persistence.py",
        "quant_investor/market/runtime_profile.py",
        "quant_investor/monitoring/cn_aggressive_portfolio_tracker.py",
        "quant_investor/monitoring/cn_aggressive_review_layer.py",
        "quant_investor/monitoring/cn_aggressive_review_runtime.py",
        "quant_investor/monitoring/cn_aggressive_rebalance.py",
        "quant_investor/monitoring/cn_aggressive_report_renderer.py",
    ]:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("print('split')\n", encoding="utf-8")
    reader_path = root / "quant_investor" / "market" / "market_data_reader.py"
    reader_path.write_text(
        "\n".join(
            [
                "self._latest_payload = None",
                "self._snapshot_gate_cache = None",
                "self._serving_symbols_cache = None",
                "self._components_payload = None",
            ]
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "results" / "cn_analysis_full" / "CN_Runtime_Profile_20260103.json",
        {
            "schema_version": "market_runtime_profile.v1",
            "market": "CN",
            "universe": "full_a",
            "total_seconds": 11.2,
            "stages": [
                {"name": "dag_symbol_list", "seconds": 0.1, "metadata": {}},
                {
                    "name": "dag_batch_read",
                    "seconds": 2.0,
                    "metadata": {
                        "projected_column_count": 9,
                        "runtime_lookback_start_date": "20250416",
                        "batch_result_count": 5531,
                        "per_symbol_fallback_count": 0,
                    },
                },
                {"name": "dag_funnel", "seconds": 0.1, "metadata": {}},
                {"name": "dag_candidate_research", "seconds": 0.1, "metadata": {}},
                {"name": "dag_bayesian_selection", "seconds": 0.1, "metadata": {}},
                {"name": "dag_control_chain", "seconds": 0.1, "metadata": {}},
                {"name": "dag_reporting_artifacts", "seconds": 0.1, "metadata": {}},
                {
                    "name": "analysis_report_persistence",
                    "seconds": 0.1,
                    "metadata": {},
                },
            ],
        },
    )


def test_architecture_rebaseline_audit_clears_current_architecture(tmp_path):
    _seed_rebaseline_ready_workspace(tmp_path)

    audit = build_architecture_rebaseline_audit(tmp_path)

    assert audit["schema_version"] == "myquant_architecture_performance_audit.v1"
    assert audit["audit_kind"] == "current_rebaseline"
    assert audit["primary_findings"] == []
    assert audit["large_module_candidates"] == []
    assert audit["summary"] == {
        "primary_finding_count": 0,
        "large_module_candidate_count": 0,
        "stage_profile_available": True,
        "batch_read_profile_proven": True,
        "reader_cache_evidence": True,
        "market_report_split_evidence": True,
        "strategy_profile_split_evidence": True,
    }
    assert audit["mutation_status"] == {
        "source_edits": False,
        "data_deletions": False,
        "read_only_summary": True,
    }


def test_architecture_rebaseline_writes_reports_and_cli_output(tmp_path, capsys):
    _seed_rebaseline_ready_workspace(tmp_path)
    output_dir = tmp_path / "reports" / "project_cleanup" / "architecture_rebaseline"

    written = write_architecture_rebaseline_audit(tmp_path, output_dir=output_dir)

    payload = json.loads((output_dir / "architecture_performance_audit.json").read_text())
    markdown = (output_dir / "architecture_performance_audit.md").read_text()
    assert written["json"] == str(output_dir / "architecture_performance_audit.json")
    assert payload["summary"]["primary_finding_count"] == 0
    assert markdown.startswith("# myQuant Current Architecture Rebaseline")

    exit_code = architecture_rebaseline_main(
        [
            "--root",
            str(tmp_path),
            "--output-dir",
            str(output_dir / "cli"),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "architecture rebaseline status:" in stdout
    assert "primary_findings: 0" in stdout
