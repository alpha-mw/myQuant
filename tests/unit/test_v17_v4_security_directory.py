from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from quant_investor.v17_v4_runtime.security_directory import (
    STOCK_BASIC_FIELDS,
    SecurityDirectoryEntry,
    SourceAdmissionError,
    evaluate_membership,
    fetch_current_security_directory,
)
from quant_investor.v17_v4_runtime.tushare_https import TushareResponse

CUTOFF = "2026-07-27T08:00:00Z"


class _Client:
    def __init__(self, rows: dict[str, tuple[tuple[Any, ...], ...]]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, dict[str, Any], tuple[str, ...]]] = []

    def request(
        self,
        *,
        api_name: str,
        params: dict[str, Any],
        expected_fields: tuple[str, ...],
    ) -> TushareResponse:
        status = params["list_status"]
        self.calls.append((api_name, params, expected_fields))
        return TushareResponse(
            api_name=api_name,
            request_id=f"request-{status}",
            reported_count=0,
            has_more=False,
            fields=expected_fields,
            rows=self.rows.get(status, ()),
        )


def _row(
    code: str,
    status: str,
    *,
    list_date: str = "20000101",
    delist_date: str | None = None,
) -> tuple[Any, ...]:
    return (
        code,
        f"NAME-{code}",
        "深圳",
        "银行",
        "主板",
        list_date,
        delist_date,
        status,
    )


def _entries() -> tuple[SecurityDirectoryEntry, ...]:
    return fetch_current_security_directory(
        _Client(
            {
                "D": (_row("000002.SZ", "D", delist_date="20260720"),),
                "L": (_row("000001.SZ", "L"),),
                "P": (_row("000003.SZ", "P", list_date="20260701"),),
            }
        ),
        observed_at=CUTOFF,
    ).entries


def test_fetches_exact_d_l_p_snapshots_and_sorts_natural_keys() -> None:
    client = _Client(
        {
            "D": (_row("000002.SZ", "D", delist_date="20260720"),),
            "L": (_row("000001.SZ", "L"),),
            "P": (_row("000003.SZ", "P", list_date="20260701"),),
        }
    )
    snapshot = fetch_current_security_directory(client, observed_at=CUTOFF)
    entries = snapshot.entries

    assert [call[1]["list_status"] for call in client.calls] == ["L", "D", "P"]
    assert all(call[0] == "stock_basic" for call in client.calls)
    assert all(call[2] == STOCK_BASIC_FIELDS for call in client.calls)
    assert [entry.security_code for entry in entries] == [
        "000001.SZ",
        "000002.SZ",
        "000003.SZ",
    ]
    assert all(entry.available_at == CUTOFF for entry in entries)
    assert all(entry.revision_id.startswith("stock-basic-") for entry in entries)
    assert snapshot.exclusions == ()
    assert snapshot.request_ids == ("request-L", "request-D", "request-P")


@pytest.mark.parametrize(
    "rows",
    [
        {"L": (_row("BAD", "L"),)},
        {"L": (_row("000001.SZ", "D", delist_date="20260720"),)},
        {"D": (_row("000002.SZ", "D"),)},
        {"L": (_row("000001.SZ", "L"), _row("000001.SZ", "L"))},
        {"L": (_row("000001.SZ", "L", list_date="20261301"),)},
        {},
    ],
)
def test_directory_rows_fail_closed(rows: dict[str, tuple[tuple[Any, ...], ...]]) -> None:
    with pytest.raises(SourceAdmissionError, match="SOURCE_ADMISSION_BLOCKED"):
        fetch_current_security_directory(_Client(rows), observed_at=CUTOFF)


def test_non_a_code_is_hash_quarantined_not_silently_dropped() -> None:
    snapshot = fetch_current_security_directory(
        _Client(
            {
                "L": (_row("000001.SZ", "L"),),
                "D": (_row("T000018.SH", "D", delist_date="20260720"),),
            }
        ),
        observed_at=CUTOFF,
    )
    assert [entry.security_code for entry in snapshot.entries] == ["000001.SZ"]
    assert len(snapshot.exclusions) == 1
    assert snapshot.exclusions[0].reason == "UNSUPPORTED_SECURITY_CODE"
    assert len(snapshot.exclusions[0].source_row_sha256) == 64


def test_listing_pending_and_delisted_boundaries_are_exact() -> None:
    entries = _entries()
    listed = evaluate_membership(
        entries,
        security_code="000001.SZ",
        session="2026-07-27",
        decision_cutoff=CUTOFF,
    )
    pending = evaluate_membership(
        entries,
        security_code="000003.SZ",
        session="2026-07-27",
        decision_cutoff=CUTOFF,
    )
    before_delist = evaluate_membership(
        entries,
        security_code="000002.SZ",
        session="2026-07-19",
        decision_cutoff=CUTOFF,
    )
    on_delist = evaluate_membership(
        entries,
        security_code="000002.SZ",
        session="2026-07-20",
        decision_cutoff=CUTOFF,
    )

    assert (listed.reason, listed.tradable) == ("LISTED", True)
    assert (pending.reason, pending.tradable) == ("PENDING", False)
    assert (before_delist.reason, before_delist.in_universe) == ("LISTED", True)
    assert (on_delist.reason, on_delist.in_universe) == ("DELISTED", False)


def test_missing_future_and_conflicting_membership_fail_closed() -> None:
    entries = _entries()
    with pytest.raises(SourceAdmissionError):
        evaluate_membership(
            entries,
            security_code="999999.SZ",
            session="2026-07-27",
            decision_cutoff=CUTOFF,
        )
    with pytest.raises(SourceAdmissionError):
        evaluate_membership(
            entries,
            security_code="000001.SZ",
            session="2026-07-27",
            decision_cutoff="2026-07-27T07:59:59Z",
        )
    listed = next(entry for entry in entries if entry.security_code == "000001.SZ")
    conflict = replace(
        listed,
        source_list_status="P",
        revision_id="stock-basic-" + "f" * 64,
    )
    with pytest.raises(SourceAdmissionError):
        evaluate_membership(
            (*entries, conflict),
            security_code="000001.SZ",
            session="2026-07-27",
            decision_cutoff=CUTOFF,
        )


def test_directory_acquisition_does_not_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    before = tuple(tmp_path.rglob("*"))
    _entries()
    assert tuple(tmp_path.rglob("*")) == before
