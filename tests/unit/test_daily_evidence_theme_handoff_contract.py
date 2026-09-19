"""Theme handoff exact shape and bindings do not substitute for native source replay."""

import hashlib
import pytest
from quant_investor.contracts import canonical_json_bytes
from quant_investor.operations.daily_contract import ContractError, GRAPH_SHA256
from quant_investor.operations.daily_journal import DailyJournal, FALSE_AUTHORITY
from quant_investor.operations.theme_handoff import validate_theme_handoff, SCHEMA
from test_daily_evidence_theme_claim import identity


def fixture(root, fallback):
    journal = DailyJournal(str(root), "20260828")
    companies = ["000001.SZ", "000002.SZ"]
    bound = identity()
    bound["company_set_sha256"] = hashlib.sha256(canonical_json_bytes(companies)).hexdigest()
    binding = {"trade_date": journal.trade_date, "company_keyset": companies, "identity": bound}
    prefix = journal.root / "executions" / bound["request_ref"]["sha256"] / "theme-source"

    def ref(path):
        return {"path": str(path), "sha256": "a" * 64}

    claim = ref(journal.root / "theme-acquisition.v1.json")
    value = {
        "schema_version": SCHEMA,
        "trade_date": journal.trade_date,
        "graph_sha256": GRAPH_SHA256,
        **bound,
        "company_keyset": companies,
        "claim_ref": claim,
        "source_descriptor_ref": ref(prefix / "source.json"),
        "sealed_at": "2026-08-28T13:00:00Z",
        "authority": FALSE_AUTHORITY,
    }
    for label, rows in (("dc", companies), ("tdx", fallback)):
        value[label + "_plan_ref"] = ref(prefix / (label + "-plan.json")) if rows else None
        value[label + "_capture_ref"] = ref(prefix / label / "capture.json") if rows else None
        value[label + "_partition_refs"] = (
            [ref(prefix / label / "partitions" / f"{i:05d}.json") for i in range(1 + len(rows))]
            if rows
            else []
        )
    return value, dict(
        journal=journal, binding=binding, claim_ref=claim, fallback_company_keyset=fallback
    )


@pytest.mark.parametrize("fallback", [[], ["000002.SZ"]])
def test_exact_dc_and_native_fallback_shapes(tmp_path, fallback):
    value, kwargs = fixture(tmp_path, fallback)
    assert validate_theme_handoff(value, **kwargs) == value


@pytest.mark.parametrize(
    "fault", ["identity", "claim", "partition", "company", "tdx", "extra", "future"]
)
def test_handoff_rejects_changed_identity_or_scope(tmp_path, fault):
    value, kwargs = fixture(tmp_path, [])
    if fault == "identity":
        value["release_ref"] = {"path": "other.json", "sha256": "b" * 64}
    elif fault == "claim":
        value["claim_ref"] = {"path": "other.json", "sha256": "b" * 64}
    elif fault == "partition":
        value["dc_partition_refs"].reverse()
    elif fault == "company":
        value["company_keyset"] = ["000003.SZ"]
    elif fault == "tdx":
        value["tdx_plan_ref"] = value["dc_plan_ref"]
    elif fault == "extra":
        value["provider_override"] = True
    elif fault == "future":
        value["sealed_at"] = "2999-01-01T00:00:00Z"
    with pytest.raises(ContractError):
        validate_theme_handoff(value, **kwargs)
