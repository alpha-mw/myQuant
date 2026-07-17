from __future__ import annotations

import pytest

from quant_investor.v16.protocol_matrix import (
    PROTOCOL_VERSIONS,
    protocol_envelope,
    require_exact_v16_protocol,
    results_v16_root,
)


def test_v16_protocol_matrix_is_exact_and_uses_dedicated_result_root(tmp_path) -> None:
    assert PROTOCOL_VERSIONS == {
        "architecture_version": "16.0.0",
        "branch_version": "v16.four-branch",
        "likelihood_version": "v16.four-evidence",
        "outcome_version": "v16.four-evidence",
        "calibration_version": "v16.four-evidence",
        "posterior_version": "v16.four-evidence",
        "ic_version": "v16.codex-authoritative",
        "handoff_version": "v1",
        "eligibility_version": "v1",
        "risk_advisor_version": "v1",
        "report_version": "v16",
        "readiness_version": "v16",
        "dashboard_version": "v16",
        "factor_governance_version": "v4",
    }
    assert results_v16_root(tmp_path) == tmp_path / "results" / "v16"
    assert not results_v16_root(tmp_path).exists()
    require_exact_v16_protocol(protocol_envelope())


def test_v16_protocol_matrix_rejects_v15_and_unknown_fields() -> None:
    legacy = protocol_envelope()
    legacy["architecture_version"] = "15.0.0-stable"
    with pytest.raises(ValueError, match="legacy or mismatched"):
        require_exact_v16_protocol(legacy)

    unknown = protocol_envelope()
    unknown["posterior_action_score"] = "retired"
    with pytest.raises(ValueError, match="unknown"):
        require_exact_v16_protocol(unknown)
