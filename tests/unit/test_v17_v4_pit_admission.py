from __future__ import annotations

from copy import deepcopy
import hashlib
from typing import Any

import pytest

from quant_investor.v17_v4_contract import (
    canonical_resource_bytes,
    validate_artifact,
)
from quant_investor.v17_v4_contract.canonical import seal_semantic
from quant_investor.v17_v4_contract.schema_validation import SchemaValidationError
from quant_investor.v17_v4_runtime.pit_admission import (
    DatasetInput,
    NATURAL_KEYS,
    REQUIRED_ROLES,
    admit_pit_closure,
)
from quant_investor.v17_v4_runtime.pit_catalog import (
    CATALOG_VERSION,
    POINTER_VERSION,
    build_pit_catalog_pointer,
    build_pit_generation_catalog,
)
from quant_investor.v17_v4_runtime.security_directory import SourceAdmissionError

CUTOFF = "2026-07-27T08:00:00Z"
START = "2026-07-18"
END = "2026-07-19"


def _rows() -> dict[str, list[dict[str, Any]]]:
    return {
        "benchmark_total_return": [
            {
                "benchmark_id": "csi300",
                "session": session,
                "total_return_index": value,
                "available_at": CUTOFF,
            }
            for session, value in ((START, "1000.0"), (END, "1001.0"))
        ],
        "cn_open_day_calendar": [
            {
                "market_id": "cn",
                "session": session,
                "is_open": True,
                "available_at": CUTOFF,
            }
            for session in (START, END)
        ],
        "corporate_actions": [
            {
                "security_code": "000002.SZ",
                "ex_date": END,
                "action_type": "TERMINAL_DELISTING",
                "announced_at": "2026-07-17T08:00:00Z",
                "revision_id": "terminal-1",
                "cash_amount_per_share": "0",
                "split_ratio": "0",
                "currency": "CNY",
                "official_source_id": "szse.notice-1",
                "available_at": "2026-07-17T08:00:00Z",
            }
        ],
        "market_bars": [
            {
                "security_code": code,
                "trade_date": session,
                "open": "10.0",
                "high": "11.0",
                "low": "9.0",
                "close": "10.5",
                "volume": "1000",
                "amount": "10500",
                "adj_factor": "1.0",
                "available_at": CUTOFF,
            }
            for code in ("000001.SZ", "000002.SZ")
            for session in (START, END)
        ],
        "official_delisting_cash": [
            {
                "security_code": "000002.SZ",
                "terminal_session": END,
                "currency": "CNY",
                "official_source_id": "szse.notice-1",
                "cash_amount_per_share": "1.25",
                "settlement_date": "2026-07-20",
                "available_at": "2026-07-17T08:00:00Z",
            }
        ],
        "pit_fundamentals": [
            {
                "security_code": "000001.SZ",
                "report_period": "2026-03-31",
                "announce_date": "2026-04-30",
                "revision_id": "fundamental-1",
                "field_id": "roe",
                "value": "0.12",
                "unit": "ratio",
                "available_at": "2026-04-30T12:00:00Z",
            }
        ],
        "universe_membership": [
            {
                "security_code": "000001.SZ",
                "name": "平安银行",
                "area": "深圳",
                "industry": "银行",
                "board_market": "主板",
                "source_list_status": "L",
                "valid_from": "1991-04-03",
                "valid_to": "",
                "published_at": "2026-07-17T08:00:00Z",
                "revision_id": "stock-basic-1",
                "available_at": "2026-07-17T08:00:00Z",
                "source_id": "tushare.stock_basic",
            },
            {
                "security_code": "000002.SZ",
                "name": "退市样本",
                "area": "深圳",
                "industry": "制造",
                "board_market": "主板",
                "source_list_status": "D",
                "valid_from": "2000-01-01",
                "valid_to": "2026-07-20",
                "published_at": "2026-07-17T08:00:00Z",
                "revision_id": "stock-basic-2",
                "available_at": "2026-07-17T08:00:00Z",
                "source_id": "tushare.stock_basic",
            },
        ],
    }


def _sources(
    rows: dict[str, list[dict[str, Any]]] | None = None,
) -> list[DatasetInput]:
    resolved = _rows() if rows is None else rows
    return [
        DatasetInput(
            role=role,
            rows=resolved[role],
            expected_keys=[
                tuple(str(row[field]) for field in NATURAL_KEYS[role])
                for row in resolved[role]
            ],
        )
        for role in REQUIRED_ROLES
    ]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _ref(role: str, kind: str) -> dict[str, str]:
    return {
        "artifact_id": f"{kind}-{role}",
        "artifact_version": f"myquant.v17.v4.{kind}.{role}.v1",
        "byte_sha256": _sha(f"{kind}-bytes-{role}"),
        "cutoff": CUTOFF,
        "relative_path": (
            f"data/private/v17_v4_sources/{role}/{kind}-{role}.json"
        ),
        "semantic_sha256": _sha(f"{kind}-semantic-{role}"),
        "strategy_id": "quant-first",
    }


def _admitted() -> Any:
    return admit_pit_closure(
        _sources(),
        history_start=START,
        decision_session=END,
        decision_cutoff=CUTOFF,
    )


def _catalog() -> dict[str, Any]:
    return build_pit_generation_catalog(
        _admitted(),
        catalog_id="pit-catalog-1",
        generation_id="pit-generation-1",
        strategy_id="quant-first",
        dataset_refs={role: _ref(role, "dataset") for role in REQUIRED_ROLES},
        expected_key_inventory_refs={
            role: _ref(role, "expected-keys")
            for role in REQUIRED_ROLES
        },
    )


def test_admits_exact_seven_role_closure_and_delisting_cash() -> None:
    result = admit_pit_closure(
        _sources(),
        history_start=START,
        decision_session=END,
        decision_cutoff=CUTOFF,
    )
    assert tuple(dataset.role for dataset in result.datasets) == REQUIRED_ROLES
    assert result.for_role("market_bars").row_count == 4
    assert result.for_role("official_delisting_cash").row_count == 1
    assert len(result.closure_sha256) == 64


def test_admission_treats_json_object_field_order_as_nonsemantic() -> None:
    rows = _rows()
    rows["market_bars"][0] = dict(reversed(tuple(rows["market_bars"][0].items())))

    result = admit_pit_closure(
        _sources(rows),
        history_start=START,
        decision_session=END,
        decision_cutoff=CUTOFF,
    )

    assert result.for_role("market_bars").row_count == 4


def test_catalog_and_pointer_are_typed_sealed_and_non_authorizing() -> None:
    catalog = _catalog()
    assert catalog["version"] == CATALOG_VERSION
    assert validate_artifact(catalog).semantic_sha256 == catalog["semantic_sha256"]
    raw = canonical_resource_bytes(catalog)
    catalog_ref = {
        "artifact_id": catalog["catalog_id"],
        "artifact_version": catalog["version"],
        "byte_sha256": hashlib.sha256(raw).hexdigest(),
        "cutoff": catalog["cutoff"],
        "relative_path": (
            "data/private/v17_v4_sources/pit_catalog/generations/"
            "pit-generation-1.json"
        ),
        "semantic_sha256": catalog["semantic_sha256"],
        "strategy_id": catalog["strategy_id"],
    }
    pointer = build_pit_catalog_pointer(
        pointer_id="pit-pointer-1",
        strategy_id="quant-first",
        cutoff=CUTOFF,
        updated_at=CUTOFF,
        catalog_ref=catalog_ref,
    )
    assert pointer["version"] == POINTER_VERSION
    assert pointer["state"] == "PIT_CATALOG_ACTIVE"
    assert not any(pointer["authority"].values())
    validate_artifact(pointer)


def test_catalog_ref_maps_treat_json_object_order_as_nonsemantic() -> None:
    catalog = _catalog()
    catalog["dataset_refs"] = dict(
        reversed(tuple(catalog["dataset_refs"].items()))
    )
    catalog["expected_key_inventory_refs"] = dict(
        reversed(tuple(catalog["expected_key_inventory_refs"].items()))
    )

    validate_artifact(catalog)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["dataset_summaries"][0].update(
            {"row_count": 999}
        ),
        lambda payload: payload.update({"source_closure_sha256": "f" * 64}),
        lambda payload: payload["dataset_refs"][
            "market_bars"
        ].update(
            {
                "artifact_version": (
                    "myquant.v17.v3.dataset.market_bars.v1"
                )
            }
        ),
    ],
)
def test_catalog_resealed_tamper_fails_cross_document_validation(
    mutation: Any,
) -> None:
    catalog = _catalog()
    catalog.pop("semantic_sha256")
    mutation(catalog)
    tampered = seal_semantic(catalog)
    with pytest.raises((SchemaValidationError, ValueError)):
        validate_artifact(tampered)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows["market_bars"][0].update(
            {"available_at": "2026-07-27T08:00:01Z"}
        ),
        lambda rows: rows["market_bars"][0].update({"adj_factor": "0"}),
        lambda rows: rows["market_bars"].append(deepcopy(rows["market_bars"][0])),
        lambda rows: rows["cn_open_day_calendar"].pop(),
        lambda rows: rows["market_bars"][0].update({"daily_basic_close": "10.5"}),
        lambda rows: rows["corporate_actions"][0].update(
            {"announced_at": "2026-07-18T08:00:00Z"}
        ),
    ],
)
def test_invalid_source_rows_fail_closed_and_return_no_partial_result(
    mutation: Any,
) -> None:
    rows = _rows()
    mutation(rows)
    with pytest.raises(SourceAdmissionError, match="SOURCE_ADMISSION_BLOCKED"):
        admit_pit_closure(
            _sources(rows),
            history_start=START,
            decision_session=END,
            decision_cutoff=CUTOFF,
        )


def test_future_sessions_after_decision_session_are_rejected() -> None:
    rows = _rows()
    future_session = "2026-07-20"
    rows["cn_open_day_calendar"].append(
        {
            "market_id": "cn",
            "session": future_session,
            "is_open": True,
            "available_at": CUTOFF,
        }
    )
    rows["benchmark_total_return"].append(
        {
            "benchmark_id": "csi300",
            "session": future_session,
            "total_return_index": "1002.0",
            "available_at": CUTOFF,
        }
    )
    for code in ("000001.SZ", "000002.SZ"):
        rows["market_bars"].append(
            {
                "security_code": code,
                "trade_date": future_session,
                "open": "10.0",
                "high": "11.0",
                "low": "9.0",
                "close": "10.5",
                "volume": "1000",
                "amount": "10500",
                "adj_factor": "1.0",
                "available_at": CUTOFF,
            }
        )

    with pytest.raises(SourceAdmissionError, match="SOURCE_ADMISSION_BLOCKED"):
        admit_pit_closure(
            _sources(rows),
            history_start=START,
            decision_session=END,
            decision_cutoff=CUTOFF,
        )


def test_expected_key_inventory_is_independent_and_exact() -> None:
    sources = _sources()
    market = next(source for source in sources if source.role == "market_bars")
    tampered = DatasetInput(
        role=market.role,
        rows=market.rows,
        expected_keys=market.expected_keys[:-1],
    )
    sources[sources.index(market)] = tampered
    with pytest.raises(SourceAdmissionError):
        admit_pit_closure(
            sources,
            history_start=START,
            decision_session=END,
            decision_cutoff=CUTOFF,
        )


def test_negative_fundamental_is_valid_but_noncanonical_decimal_is_not() -> None:
    rows = _rows()
    rows["pit_fundamentals"][0]["value"] = "-0.12"
    admitted = admit_pit_closure(
        _sources(rows),
        history_start=START,
        decision_session=END,
        decision_cutoff=CUTOFF,
    )
    assert admitted.for_role("pit_fundamentals").row_count == 1

    rows["pit_fundamentals"][0]["value"] = "1e-1"
    with pytest.raises(SourceAdmissionError):
        admit_pit_closure(
            _sources(rows),
            history_start=START,
            decision_session=END,
            decision_cutoff=CUTOFF,
        )


def test_terminal_event_requires_exact_official_cash_not_last_close() -> None:
    rows = _rows()
    rows["official_delisting_cash"] = []
    with pytest.raises(SourceAdmissionError):
        admit_pit_closure(
            _sources(rows),
            history_start=START,
            decision_session=END,
            decision_cutoff=CUTOFF,
        )

    rows = _rows()
    rows["official_delisting_cash"][0]["cash_amount_per_share"] = rows[
        "market_bars"
    ][-1]["close"]
    admitted = admit_pit_closure(
        _sources(rows),
        history_start=START,
        decision_session=END,
        decision_cutoff=CUTOFF,
    )
    assert admitted.for_role("official_delisting_cash").row_count == 1


def test_delist_date_equality_excludes_a_market_bar() -> None:
    rows = _rows()
    rows["universe_membership"][1]["valid_to"] = END
    with pytest.raises(SourceAdmissionError):
        admit_pit_closure(
            _sources(rows),
            history_start=START,
            decision_session=END,
            decision_cutoff=CUTOFF,
        )


def test_role_order_and_unknown_role_are_closed() -> None:
    sources = _sources()
    with pytest.raises(SourceAdmissionError):
        admit_pit_closure(
            list(reversed(sources)),
            history_start=START,
            decision_session=END,
            decision_cutoff=CUTOFF,
        )
    with pytest.raises(SourceAdmissionError):
        admit_pit_closure(
            [
                *sources,
                DatasetInput(
                    role="daily_basic",
                    rows=(),
                    expected_keys=(),
                ),
            ],
            history_start=START,
            decision_session=END,
            decision_cutoff=CUTOFF,
        )
