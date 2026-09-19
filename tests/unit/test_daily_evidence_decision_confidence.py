"""Source completeness mapping, independent of positive/negative investment states."""

from copy import deepcopy

import pytest

from quant_investor.intelligence import decision_report as module


def inputs(monkeypatch):
    company = "000001.SZ"

    def projection(**fields):
        return {"payload": {"company_rows": [{"company_code": company, **fields}]}}

    monkeypatch.setattr(module, "_freshness", lambda *a, **kw: ([], []))
    return company, {
        "projections": {
            "theme": projection(status="NO_MEMBERSHIP"),
            "industry": projection(status="AVAILABLE"),
            "exposure": projection(economic_exposure_state="LOW"),
            "macro": {"payload": {"classification": "CANONICAL_MACRO_READY"}},
        },
        "assessments": {company: {"payload": {"status": "COMPLETE", "blocker_codes": []}}},
        "freshness": {"fundamental": None, "macro": None},
        "portfolio": {"timing_status": "ON_TIME", "reason_codes": []},
        "physical": {domain: [{"fixture": "present binding"}] for domain in module.DOMAINS},
    }


def test_verified_negative_membership_and_low_exposure_are_complete(monkeypatch):
    company, kwargs = inputs(monkeypatch)
    states = module._domain_statuses(company, **kwargs)
    assert all(state == "COMPLETE" for state, _ in states.values())
    assert module.source_confidence(states) == "COMPLETE_SOURCE_BOUND"


@pytest.mark.parametrize(
    "condition", ["coverage", "fundamental_warning", "macro_warning", "late_portfolio"]
)
def test_noncritical_conditions_are_partial_not_missing(monkeypatch, condition):
    company, kwargs = inputs(monkeypatch)
    if condition == "coverage":
        kwargs["assessments"][company]["payload"]["status"] = "PARTIAL"
    elif condition == "late_portfolio":
        kwargs["portfolio"] = {
            "timing_status": "LATE_RECORDED",
            "reason_codes": ["PORTFOLIO_CUSTODY_AFTER_DECISION"],
        }
    else:
        domain = condition.removesuffix("_warning")
        monkeypatch.setattr(
            module,
            "_freshness",
            lambda *a, **kw: (["NATIVE_WARNING"], []) if kw["domain"] == domain else ([], []),
        )
    states = module._domain_statuses(company, **kwargs)
    assert module.source_confidence(states) == "PARTIAL_SOURCE_BOUND"
    assert all(state != "MISSING" for state, _ in states.values())


@pytest.mark.parametrize("domain", module.DOMAINS)
def test_critical_missing_binding_dominates_late_partial(monkeypatch, domain):
    company, kwargs = inputs(monkeypatch)
    kwargs["portfolio"] = {
        "timing_status": "LATE_RECORDED",
        "reason_codes": ["PORTFOLIO_CUSTODY_AFTER_DECISION"],
    }
    kwargs["physical"][domain] = []
    before = deepcopy(kwargs)
    states = module._domain_statuses(company, **kwargs)
    assert states[domain][0] == "MISSING"
    assert module.source_confidence(states) == "MISSING"
    assert kwargs == before
