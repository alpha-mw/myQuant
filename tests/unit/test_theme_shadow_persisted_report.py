from __future__ import annotations

from pathlib import Path

from quant_investor.market import full_report


def _success_monitor() -> dict[str, object]:
    return {
        "status": "success",
        "final_decision_source": "baseline",
        "candidate_overlap_ratio": 1.0,
        "entered_candidates": ["000003.SZ"],
        "dropped_candidates": ["000002.SZ"],
        "selected_overlap_ratio": 1.0,
        "portfolio_weight_deltas": [],
        "theme_exposure_baseline": {},
        "theme_exposure_shadow": {},
        "risk_delta": {"theme_effect": False, "theme_risk_flags": []},
        "diagnostic_notes": ["portfolio_shadow_no_theme_exposure"],
        "artifact_path": "results/theme_shadow/CN/20260618_full_a_theme_shadow.json",
    }


def _all_results_with_theme_shadow() -> dict[str, list[dict[str, object]]]:
    return {
        "hs300": [
            {
                "stock_count": 1,
                "batch_id": 1,
                "execution_log": ["[INFO] batch finished"],
                "branches": {
                    "quant": {
                        "score": 0.1,
                        "confidence": 0.7,
                        "conclusion": "quant ok",
                    }
                },
                "strategy": {
                    "target_exposure": 0.0,
                    "style_bias": "防御",
                    "candidate_symbols": [],
                    "risk_summary": {},
                },
                "recommendations": [],
                "analysis_meta": {
                    "market": "CN",
                    "universe": "full_a",
                    "global_context": {
                        "metadata": {
                            "theme_shadow_monitor": _success_monitor(),
                        }
                    },
                    "data_snapshot": {"summary_text": "local snapshot ok"},
                },
            }
        ]
    }


def test_persisted_cn_report_writer_includes_theme_shadow(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(full_report, "load_stock_names", lambda market="CN", refresh=False: {})
    monkeypatch.setattr(full_report, "get_stock_name", lambda symbol, market="CN": symbol)

    output = full_report.generate_full_report(
        _all_results_with_theme_shadow(),
        market="CN",
        output_dir=str(tmp_path),
        total_capital=1_000_000,
        top_k=1,
    )

    trade_report = Path(output["trade_report"]).read_text(encoding="utf-8")
    full_report_text = Path(output["summary_report"]).read_text(encoding="utf-8")

    for text in (trade_report, full_report_text):
        assert "主题 Shadow Monitor" in text
        assert "final executable decision remains baseline" in text
        assert "candidate_overlap_ratio: 1.000000" in text
        assert "entered_candidates: 000003.SZ" in text
        assert "dropped_candidates: 000002.SZ" in text
        assert "selected_overlap_ratio: 1.000000" in text
        assert "portfolio_shadow_no_theme_exposure" in text
        assert "results/theme_shadow/CN/20260618_full_a_theme_shadow.json" in text
        assert text.count("主题 Shadow Monitor") == 1
