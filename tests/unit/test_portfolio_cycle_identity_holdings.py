from __future__ import annotations

from decimal import Decimal
import hashlib
import os
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_investor.portfolio_cycle import (
    HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID,
    HOLDINGS_LEDGER_SCHEMA_ID,
    HOLDINGS_MANIFEST_SCHEMA_ID,
    HOLDINGS_POINTER_SCHEMA_ID,
    HOLDINGS_PRICE_SOURCE_SCHEMA_ID,
    IDENTITY_DECLARATION_SCHEMA_ID,
    PortfolioCycleError,
    canonical_json_bytes,
    resolve_holdings_baseline,
    resolve_strategy_identity,
    seal_document,
)

STRATEGY = "aggressive-tech-owner-v17"
HISTORICAL_LABEL = "aggressive_tech_manufacturing"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_exact(root: Path, relative_path: str, raw: bytes) -> tuple[str, str]:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)
    return relative_path, _sha(raw)


def _write_json(root: Path, relative_path: str, body: dict[str, Any]) -> tuple[str, str]:
    return _write_exact(root, relative_path, canonical_json_bytes(seal_document(body)))


def _identity_body(**overrides: Any) -> dict[str, Any]:
    body = {
        "schema_id": IDENTITY_DECLARATION_SCHEMA_ID,
        "protocol": "myquant.v17.v4",
        "historical_label": HISTORICAL_LABEL,
        "canonical_strategy_id": STRATEGY,
        "declared_by": "maxwell",
        "declared_at": "2026-08-06T01:00:00Z",
        "authority_kind": "owner_declaration",
        "provenance": ("explicit owner declaration for this exact holdings domain"),
    }
    body.update(overrides)
    return body


def _ledger_bytes(rows: list[dict[str, Any]]) -> bytes:
    fields = [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("shares", pa.int64(), nullable=False),
        pa.field("avg_cost", pa.decimal128(20, 4), nullable=False),
        pa.field("market_price", pa.decimal128(20, 4), nullable=False),
        pa.field("cost_basis", pa.decimal128(20, 4), nullable=False),
        pa.field("market_value", pa.decimal128(20, 4), nullable=False),
        pa.field("unrealized_pnl", pa.decimal128(20, 4), nullable=False),
        pa.field("realized_pnl", pa.decimal128(20, 4), nullable=False),
    ]
    schema = pa.schema(
        fields,
        metadata={b"schema_id": HOLDINGS_LEDGER_SCHEMA_ID.encode("ascii")},
    )
    arrays = [pa.array([row[field.name] for row in rows], type=field.type) for field in fields]
    table = pa.Table.from_arrays(arrays, schema=schema)
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression="NONE")
    return sink.getvalue().to_pybytes()


def _default_rows() -> list[dict[str, Any]]:
    return [
        {
            "symbol": "000001.SZ",
            "name": "平安银行",
            "shares": 100,
            "avg_cost": Decimal("10.0000"),
            "market_price": Decimal("11.0000"),
            "cost_basis": Decimal("1000.0000"),
            "market_value": Decimal("1100.0000"),
            "unrealized_pnl": Decimal("100.0000"),
            "realized_pnl": Decimal("25.0000"),
        },
        {
            "symbol": "600000.SH",
            "name": "浦发银行",
            "shares": 200,
            "avg_cost": Decimal("8.0000"),
            "market_price": Decimal("7.5000"),
            "cost_basis": Decimal("1600.0000"),
            "market_value": Decimal("1500.0000"),
            "unrealized_pnl": Decimal("-100.0000"),
            "realized_pnl": Decimal("-5.0000"),
        },
    ]


def _build_holdings(
    root: Path,
    *,
    rows: list[dict[str, Any]] | None = None,
    ledger_raw: bytes | None = None,
    pointer_strategy: str = STRATEGY,
    manifest_strategy: str = STRATEGY,
    manifest_overrides: dict[str, Any] | None = None,
    pointer_overrides: dict[str, Any] | None = None,
    policy_overrides: dict[str, Any] | None = None,
    price_overrides: dict[str, Any] | None = None,
) -> tuple[str, str, dict[str, str]]:
    policy = {
        "schema_id": HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID,
        "protocol": "myquant.v17.v4",
        "currency": "CNY",
        "money_scale": 4,
        "rounding_mode": "ROUND_HALF_EVEN",
        "capital_identity": ("NAV_EQUALS_CONTRIBUTED_CAPITAL_PLUS_REALIZED_PLUS_UNREALIZED"),
    }
    policy.update(policy_overrides or {})
    policy_path, policy_sha = _write_json(
        root,
        "governance/accounting-policy.json",
        policy,
    )
    price_source = {
        "schema_id": HOLDINGS_PRICE_SOURCE_SCHEMA_ID,
        "protocol": "myquant.v17.v4",
        "currency": "CNY",
        "source_id": "exact-close-evidence-v1",
        "as_of": "2026-08-05T07:00:00Z",
        "valuation_at": "2026-08-05T07:30:00Z",
    }
    price_source.update(price_overrides or {})
    price_path, price_sha = _write_json(
        root,
        "evidence/price-source.json",
        price_source,
    )
    ledger = ledger_raw if ledger_raw is not None else _ledger_bytes(rows or _default_rows())
    ledger_path, ledger_sha = _write_exact(root, "holdings/ledger.parquet", ledger)
    manifest = {
        "schema_id": HOLDINGS_MANIFEST_SCHEMA_ID,
        "protocol": "myquant.v17.v4",
        "canonical_strategy_id": manifest_strategy,
        "account_id": "maxwell-paper-cn",
        "currency": "CNY",
        "trade_date": "2026-08-05",
        "as_of": "2026-08-05T07:00:00Z",
        "valuation_at": "2026-08-05T07:30:00Z",
        "decision_cutoff": "2026-08-05T08:00:00Z",
        "accounting_policy_ref": {
            "schema_id": HOLDINGS_ACCOUNTING_POLICY_SCHEMA_ID,
            "relative_path": policy_path,
            "byte_sha256": policy_sha,
        },
        "price_source_ref": {
            "schema_id": HOLDINGS_PRICE_SOURCE_SCHEMA_ID,
            "relative_path": price_path,
            "byte_sha256": price_sha,
        },
        "ledger_ref": {
            "schema_id": HOLDINGS_LEDGER_SCHEMA_ID,
            "relative_path": ledger_path,
            "byte_sha256": ledger_sha,
        },
        "contributed_capital": "4980.0000",
        "cash": "2400.0000",
        "total_cost_basis": "2600.0000",
        "total_market_value": "2600.0000",
        "total_unrealized_pnl": "0.0000",
        "total_realized_pnl": "20.0000",
        "nav": "5000.0000",
    }
    manifest.update(manifest_overrides or {})
    manifest_path, manifest_sha = _write_json(root, "holdings/manifest.json", manifest)
    pointer = {
        "schema_id": HOLDINGS_POINTER_SCHEMA_ID,
        "protocol": "myquant.v17.v4",
        "canonical_strategy_id": pointer_strategy,
        "updated_at": "2026-08-05T08:01:00Z",
        "manifest_ref": {
            "schema_id": HOLDINGS_MANIFEST_SCHEMA_ID,
            "relative_path": manifest_path,
            "byte_sha256": manifest_sha,
        },
    }
    pointer.update(pointer_overrides or {})
    pointer_path, pointer_sha = _write_json(root, "holdings/current.json", pointer)
    return (
        pointer_path,
        pointer_sha,
        {
            "policy_path": policy_path,
            "price_path": price_path,
            "ledger_path": ledger_path,
            "manifest_path": manifest_path,
        },
    )


def _assert_code(exc_info: pytest.ExceptionInfo[PortfolioCycleError], code: str) -> None:
    assert exc_info.value.code == code
    assert str(exc_info.value).startswith(code + ":")


def test_resolves_exact_identity_and_holdings_happy_path(
    tmp_path: Path,
) -> None:
    identity_path, identity_sha = _write_json(
        tmp_path, "governance/identity.json", _identity_body()
    )
    identity = resolve_strategy_identity(
        tmp_path,
        declaration_path=identity_path,
        declaration_sha256=identity_sha,
        expected_historical_label=HISTORICAL_LABEL,
    )
    assert identity.verified is True
    assert identity.canonical_strategy_id == STRATEGY
    assert identity.historical_label == HISTORICAL_LABEL
    assert identity.declaration_ref.byte_sha256 == identity_sha

    pointer_path, pointer_sha, _ = _build_holdings(tmp_path)
    baseline = resolve_holdings_baseline(
        tmp_path,
        pointer_path=pointer_path,
        pointer_sha256=pointer_sha,
        expected_strategy_id=identity.canonical_strategy_id,
    )
    assert baseline.verified is True
    assert baseline.canonical_strategy_id == STRATEGY
    assert baseline.currency == "CNY"
    assert baseline.totals.nav == Decimal("5000.0000")
    assert [position.symbol for position in baseline.positions] == [
        "000001.SZ",
        "600000.SH",
    ]
    assert baseline.pointer_ref.byte_sha256 == pointer_sha
    assert baseline.accounting_policy_ref.byte_sha256
    assert baseline.price_source_ref.byte_sha256
    assert baseline.ledger_ref.byte_sha256


def test_identity_requires_exact_sha_and_explicit_historical_ref(
    tmp_path: Path,
) -> None:
    path, digest = _write_json(tmp_path, "identity.json", _identity_body())
    with pytest.raises(PortfolioCycleError) as mismatch:
        resolve_strategy_identity(tmp_path, declaration_path=path, declaration_sha256="0" * 64)
    _assert_code(mismatch, "PORTFOLIO_CYCLE_BYTE_SHA_MISMATCH")

    with pytest.raises(PortfolioCycleError) as label_mismatch:
        resolve_strategy_identity(
            tmp_path,
            declaration_path=path,
            declaration_sha256=digest,
            expected_historical_label="different_historical_label",
        )
    _assert_code(label_mismatch, "PORTFOLIO_CYCLE_IDENTITY_MISMATCH")


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        (
            {"canonical_strategy_id": "aggressive_tech_manufacturing"},
            "PORTFOLIO_CYCLE_IDENTITY_INVALID",
        ),
        (
            {"authority_kind": "directory_guess"},
            "PORTFOLIO_CYCLE_IDENTITY_INVALID",
        ),
        (
            {"authority_kind": "signed_attestation"},
            "PORTFOLIO_CYCLE_IDENTITY_INVALID",
        ),
        (
            {"declared_at": "2026-02-30T00:00:00Z"},
            "PORTFOLIO_CYCLE_IDENTITY_INVALID",
        ),
        ({"provenance": ""}, "PORTFOLIO_CYCLE_IDENTITY_INVALID"),
    ],
)
def test_identity_rejects_malformed_contract(
    tmp_path: Path, overrides: dict[str, Any], expected_code: str
) -> None:
    path, digest = _write_json(tmp_path, "identity.json", _identity_body(**overrides))
    with pytest.raises(PortfolioCycleError) as exc_info:
        resolve_strategy_identity(tmp_path, declaration_path=path, declaration_sha256=digest)
    _assert_code(exc_info, expected_code)


def test_exact_reader_rejects_traversal_symlink_hardlink_and_wrong_mode(
    tmp_path: Path,
) -> None:
    path, digest = _write_json(tmp_path, "identity.json", _identity_body())
    with pytest.raises(PortfolioCycleError) as traversal:
        resolve_strategy_identity(
            tmp_path,
            declaration_path="../identity.json",
            declaration_sha256=digest,
        )
    _assert_code(traversal, "PORTFOLIO_CYCLE_PATH_INVALID")

    source = tmp_path / path
    symlink = tmp_path / "identity-link.json"
    symlink.symlink_to(source)
    with pytest.raises(PortfolioCycleError) as symlink_error:
        resolve_strategy_identity(
            tmp_path,
            declaration_path=symlink.name,
            declaration_sha256=digest,
        )
    _assert_code(symlink_error, "PORTFOLIO_CYCLE_STORAGE_SECURITY")

    source.unlink()
    path, digest = _write_json(tmp_path, "identity.json", _identity_body())
    hardlink = tmp_path / "identity-hardlink.json"
    os.link(tmp_path / path, hardlink)
    with pytest.raises(PortfolioCycleError) as hardlink_error:
        resolve_strategy_identity(tmp_path, declaration_path=path, declaration_sha256=digest)
    _assert_code(hardlink_error, "PORTFOLIO_CYCLE_STORAGE_SECURITY")
    hardlink.unlink()

    (tmp_path / path).chmod(0o644)
    with pytest.raises(PortfolioCycleError) as mode_error:
        resolve_strategy_identity(tmp_path, declaration_path=path, declaration_sha256=digest)
    _assert_code(mode_error, "PORTFOLIO_CYCLE_STORAGE_SECURITY")


def test_exact_reader_rejects_casefold_collision(tmp_path: Path) -> None:
    path, digest = _write_json(tmp_path, "Identity.json", _identity_body())
    _write_json(tmp_path, "identity.json", _identity_body(declared_by="another-owner"))
    if os.path.samefile(tmp_path / "Identity.json", tmp_path / "identity.json"):
        pytest.skip("filesystem is case-insensitive and cannot represent the collision")
    with pytest.raises(PortfolioCycleError) as exc_info:
        resolve_strategy_identity(tmp_path, declaration_path=path, declaration_sha256=digest)
    _assert_code(exc_info, "PORTFOLIO_CYCLE_STORAGE_SECURITY")


def test_exact_reader_rejects_symlinked_workspace_root(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    path, digest = _write_json(real_root, "identity.json", _identity_body())
    alias_root = tmp_path / "alias"
    alias_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(PortfolioCycleError) as exc_info:
        resolve_strategy_identity(alias_root, declaration_path=path, declaration_sha256=digest)
    _assert_code(exc_info, "PORTFOLIO_CYCLE_STORAGE_SECURITY")


@pytest.mark.parametrize("rows_kind", ["duplicate", "out_of_order"])
def test_holdings_rejects_duplicate_or_out_of_order_symbols(tmp_path: Path, rows_kind: str) -> None:
    rows = _default_rows()
    if rows_kind == "duplicate":
        rows[1]["symbol"] = rows[0]["symbol"]
    else:
        rows.reverse()
    pointer_path, pointer_sha, _ = _build_holdings(tmp_path, rows=rows)
    with pytest.raises(PortfolioCycleError) as exc_info:
        resolve_holdings_baseline(
            tmp_path,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(exc_info, "PORTFOLIO_CYCLE_HOLDINGS_LEDGER_INVALID")


def test_holdings_rejects_malformed_parquet(tmp_path: Path) -> None:
    pointer_path, pointer_sha, _ = _build_holdings(tmp_path, ledger_raw=b"not parquet")
    with pytest.raises(PortfolioCycleError) as exc_info:
        resolve_holdings_baseline(
            tmp_path,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(exc_info, "PORTFOLIO_CYCLE_HOLDINGS_LEDGER_INVALID")


def test_holdings_rejects_row_decimal_identity_mismatch(
    tmp_path: Path,
) -> None:
    rows = _default_rows()
    rows[0]["cost_basis"] = Decimal("999.9999")
    rows[0]["unrealized_pnl"] = Decimal("100.0001")
    pointer_path, pointer_sha, _ = _build_holdings(tmp_path, rows=rows)
    with pytest.raises(PortfolioCycleError) as exc_info:
        resolve_holdings_baseline(
            tmp_path,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(exc_info, "PORTFOLIO_CYCLE_HOLDINGS_LEDGER_INVALID")


def test_holdings_rejects_zero_share_rows(tmp_path: Path) -> None:
    rows = _default_rows()
    rows[0].update(
        shares=0,
        cost_basis=Decimal("0.0000"),
        market_value=Decimal("0.0000"),
        unrealized_pnl=Decimal("0.0000"),
    )
    pointer_path, pointer_sha, _ = _build_holdings(tmp_path, rows=rows)
    with pytest.raises(PortfolioCycleError) as exc_info:
        resolve_holdings_baseline(
            tmp_path,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(exc_info, "PORTFOLIO_CYCLE_HOLDINGS_LEDGER_INVALID")


def test_holdings_rejects_manifest_totals_and_noncanonical_decimal(
    tmp_path: Path,
) -> None:
    pointer_path, pointer_sha, _ = _build_holdings(
        tmp_path,
        manifest_overrides={
            "total_market_value": "2601.0000",
            "nav": "5001.0000",
        },
    )
    with pytest.raises(PortfolioCycleError) as mismatch:
        resolve_holdings_baseline(
            tmp_path,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(mismatch, "PORTFOLIO_CYCLE_HOLDINGS_ACCOUNTING_MISMATCH")

    other = tmp_path / "other"
    pointer_path, pointer_sha, _ = _build_holdings(other, manifest_overrides={"cash": "2400.0"})
    with pytest.raises(PortfolioCycleError) as decimal_error:
        resolve_holdings_baseline(
            other,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(decimal_error, "PORTFOLIO_CYCLE_HOLDINGS_MANIFEST_INVALID")


def test_holdings_enforces_exact_capital_policy_and_price_binding(
    tmp_path: Path,
) -> None:
    capital_root = tmp_path / "capital"
    pointer_path, pointer_sha, _ = _build_holdings(
        capital_root,
        manifest_overrides={"contributed_capital": "5000.0000"},
    )
    with pytest.raises(PortfolioCycleError) as capital_error:
        resolve_holdings_baseline(
            capital_root,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(capital_error, "PORTFOLIO_CYCLE_HOLDINGS_ACCOUNTING_MISMATCH")

    policy_root = tmp_path / "policy"
    pointer_path, pointer_sha, _ = _build_holdings(
        policy_root, policy_overrides={"rounding_mode": "ROUND_HALF_UP"}
    )
    with pytest.raises(PortfolioCycleError) as policy_error:
        resolve_holdings_baseline(
            policy_root,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(policy_error, "PORTFOLIO_CYCLE_HOLDINGS_MANIFEST_INVALID")

    price_root = tmp_path / "price"
    pointer_path, pointer_sha, _ = _build_holdings(
        price_root,
        price_overrides={"valuation_at": "2026-08-05T07:29:59Z"},
    )
    with pytest.raises(PortfolioCycleError) as price_error:
        resolve_holdings_baseline(
            price_root,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(price_error, "PORTFOLIO_CYCLE_HOLDINGS_MANIFEST_INVALID")


@pytest.mark.parametrize(
    ("manifest_overrides", "pointer_overrides"),
    [
        ({"trade_date": "2026-08-06"}, {}),
        ({"as_of": "2026-08-05T08:30:00Z"}, {}),
        ({}, {"updated_at": "2026-08-05T07:59:59Z"}),
    ],
)
def test_holdings_rejects_invalid_chronology(
    tmp_path: Path,
    manifest_overrides: dict[str, Any],
    pointer_overrides: dict[str, Any],
) -> None:
    pointer_path, pointer_sha, _ = _build_holdings(
        tmp_path,
        manifest_overrides=manifest_overrides,
        pointer_overrides=pointer_overrides,
    )
    with pytest.raises(PortfolioCycleError) as exc_info:
        resolve_holdings_baseline(
            tmp_path,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(exc_info, "PORTFOLIO_CYCLE_HOLDINGS_CHRONOLOGY_INVALID")


@pytest.mark.parametrize(
    ("pointer_strategy", "manifest_strategy"),
    [("other-strategy", STRATEGY), (STRATEGY, "other-strategy")],
)
def test_holdings_rejects_strategy_mismatch(
    tmp_path: Path, pointer_strategy: str, manifest_strategy: str
) -> None:
    pointer_path, pointer_sha, _ = _build_holdings(
        tmp_path,
        pointer_strategy=pointer_strategy,
        manifest_strategy=manifest_strategy,
    )
    with pytest.raises(PortfolioCycleError) as exc_info:
        resolve_holdings_baseline(
            tmp_path,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(exc_info, "PORTFOLIO_CYCLE_HOLDINGS_STRATEGY_MISMATCH")


def test_holdings_requires_exact_supporting_policy_ref(tmp_path: Path) -> None:
    pointer_path, pointer_sha, refs = _build_holdings(tmp_path)
    (tmp_path / refs["policy_path"]).unlink()
    with pytest.raises(PortfolioCycleError) as exc_info:
        resolve_holdings_baseline(
            tmp_path,
            pointer_path=pointer_path,
            pointer_sha256=pointer_sha,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(exc_info, "PORTFOLIO_CYCLE_STORAGE_NOT_FOUND")


def test_holdings_rejects_pointer_sha_mismatch(tmp_path: Path) -> None:
    pointer_path, _, _ = _build_holdings(tmp_path)
    with pytest.raises(PortfolioCycleError) as exc_info:
        resolve_holdings_baseline(
            tmp_path,
            pointer_path=pointer_path,
            pointer_sha256="f" * 64,
            expected_strategy_id=STRATEGY,
        )
    _assert_code(exc_info, "PORTFOLIO_CYCLE_BYTE_SHA_MISMATCH")
