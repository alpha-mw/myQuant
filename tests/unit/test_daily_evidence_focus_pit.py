"""Native PIT record decoding and required focus-company eligibility."""

import hashlib
import io

import pandas as pd
import pytest

from quant_investor.contracts import canonical_json_bytes
from quant_investor.factors.production_pit import decode_focus_pit, FOCUS_COMPANIES
from quant_investor.factors.governance.errors import FactorGovernanceError
from quant_investor.market.pit_universe import PITUniverseRecord, PITUniverseStore


def records():
    return [
        PITUniverseRecord(
            symbol=company,
            source_list_status="L",
            list_date="20000101",
            observed_at="2026-08-20T07:00:00Z",
        )
        for company in FOCUS_COMPANIES
    ]


def inputs(rows):
    frame = pd.DataFrame(
        [r.to_dict() for r in rows], columns=list(PITUniverseRecord.__dataclass_fields__)
    )
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    raw = buffer.getvalue()
    manifest = {
        "canonical_sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(rows),
        "records_sha256": PITUniverseStore._records_sha256(rows),
    }
    return raw, manifest


def test_required_focus_uses_native_listing_status_and_keeps_both_rows():
    raw, manifest = inputs(records())
    result = decode_focus_pit(
        membership_raw=raw, manifest_raw=canonical_json_bytes(manifest), trade_date="20260828"
    )
    assert [row["symbol"] for row in result] == list(FOCUS_COMPANIES)
    assert all(row["in_universe"] and row["research_eligible"] for row in result)
    raw, manifest = inputs(records()[:1])
    result = decode_focus_pit(
        membership_raw=raw, manifest_raw=canonical_json_bytes(manifest), trade_date="20260828"
    )
    assert len(result) == 2 and not result[1]["research_eligible"]
    assert result[1]["reason"] == "missing_pit_record"


@pytest.mark.parametrize("fault", ["sha", "row_count", "records_sha", "schema"])
def test_retained_pit_custody_and_record_binding_are_required(fault):
    raw, manifest = inputs(records())
    if fault == "sha":
        raw = b"not parquet"
    elif fault == "row_count":
        manifest["row_count"] = 1
    elif fault == "records_sha":
        manifest["records_sha256"] = "0" * 64
    else:
        buffer = io.BytesIO()
        pd.DataFrame({"symbol": list(FOCUS_COMPANIES)}).to_parquet(buffer, index=False)
        raw = buffer.getvalue()
        manifest["canonical_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(FactorGovernanceError):
        decode_focus_pit(
            membership_raw=raw, manifest_raw=canonical_json_bytes(manifest), trade_date="20260828"
        )


@pytest.mark.parametrize("kind", ["pre_listing", "delisted", "outside_scope"])
def test_native_unqualified_pit_status_is_not_promoted(kind):
    rows = records()
    if kind == "pre_listing":
        rows[0].list_date = rows[0].effective_from = "20260901"
    elif kind == "delisted":
        rows[0].source_list_status = "D"
        rows[0].delist_date = rows[0].effective_to = "20260801"
    else:
        rows[0].membership_quality = "outside_frozen_scope_pending"
    raw, manifest = inputs(rows)
    result = decode_focus_pit(
        membership_raw=raw, manifest_raw=canonical_json_bytes(manifest), trade_date="20260828"
    )
    assert not result[0]["research_eligible"]
    assert result[1]["research_eligible"]
