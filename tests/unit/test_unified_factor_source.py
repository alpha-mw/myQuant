from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import io
import os
from pathlib import Path
import threading
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_investor.factors.governance import FactorGovernanceError
from quant_investor.factors.governance.source import (
    DecodedSource,
    decode_source_role,
    role_schema,
)
import quant_investor.factors.governance.source as source_module
from quant_investor.system import SystemStore

STAMP = "2026-08-14T00:00:00Z"
ROOT_ID = "factor-source-test-root"


def _store(tmp_path: Path) -> tuple[SystemStore, Path]:
    system_root = tmp_path / "system"
    system_root.mkdir(mode=0o700)
    source_root = tmp_path / "source"
    source_root.mkdir(mode=0o700)
    return (
        SystemStore(
            system_root,
            source_root=source_root,
            source_root_id=ROOT_ID,
        ),
        source_root,
    )


def _put_table(
    store: SystemStore,
    source_root: Path,
    *,
    role: str,
    rows: list[dict],
    name: str | None = None,
    schema: pa.Schema | None = None,
) -> dict[str, str]:
    relative = name or f"{role}.parquet"
    path = source_root / relative
    table = pa.Table.from_pylist(rows, schema=schema or role_schema(role))
    pq.write_table(table, path)
    path.chmod(0o600)
    return store.put_source_file(
        relative,
        source_object_id=f"source-{relative.replace('.', '-')}",
        media_type="application/vnd.apache.parquet",
        source_format="PARQUET",
        created_at=STAMP,
    )


def _calendar_rows(count: int = 391) -> list[dict]:
    first = date(2026, 8, 17)
    rows = []
    for ordinal in range(count):
        session = first + timedelta(days=ordinal)
        rows.append(
            {
                "ordinal": ordinal,
                "open_session": session,
                "opens_at_utc": datetime.combine(session, datetime.min.time(), tzinfo=timezone.utc)
                + timedelta(hours=1),
                "closes_at_utc": datetime.combine(session, datetime.min.time(), tzinfo=timezone.utc)
                + timedelta(hours=7),
            }
        )
    return rows


def _discard_projection(table: pa.Table, binding: Mapping[str, object]) -> None:
    del table, binding
    return None


def test_strict_parquet_roles_decode_and_bind_exact_source_bytes(tmp_path: Path) -> None:
    store, source_root = _store(tmp_path)
    ref = _put_table(
        store,
        source_root,
        role="pit_universe",
        rows=[
            {
                "signal_session": date(2026, 8, 17),
                "symbol": "000001.SZ",
                "industry": "银行",
                "total_mv": 1_000_000.0,
                "tradable": True,
            },
            {
                "signal_session": date(2026, 8, 17),
                "symbol": "600000.SH",
                "industry": None,
                "total_mv": None,
                "tradable": None,
            },
        ],
    )

    decoded = decode_source_role(
        system_store=store,
        source_object_ref=ref,
        role="pit_universe",
        projector=lambda table, binding: {
            "row_count": table.num_rows,
            "column_names": table.column_names,
        },
    )
    assert isinstance(decoded, DecodedSource)
    assert decoded.projection == {
        "row_count": 2,
        "column_names": [
            "signal_session",
            "symbol",
            "industry",
            "total_mv",
            "tradable",
        ],
    }
    assert not hasattr(decoded, "table")
    assert set(decoded.binding) == {
        "role",
        "source_object_ref",
        "source_root_id",
        "source_object_created_at",
        "media_type",
        "source_format",
        "source_byte_sha256",
        "source_byte_count",
        "decoded_schema_sha256",
        "normalized_sha256",
        "row_count",
        "column_count",
        "decoded_cell_count",
        "minimum_session",
        "maximum_session",
    }
    assert decoded.binding["source_object_ref"] == ref
    assert decoded.binding["source_root_id"] == ROOT_ID
    assert decoded.binding["row_count"] == 2
    assert decoded.binding["decoded_cell_count"] == 10


def test_pit_object_is_full_denominator_and_has_no_eligible_column(tmp_path: Path) -> None:
    store, source_root = _store(tmp_path)
    rows = [
        {
            "signal_session": date(2026, 8, 17),
            "symbol": "000001.SZ",
            "industry": None,
            "total_mv": None,
            "tradable": None,
        }
    ]
    ref = _put_table(store, source_root, role="pit_universe", rows=rows)
    decoded = decode_source_role(
        system_store=store,
        source_object_ref=ref,
        role="pit_universe",
        projector=lambda table, binding: {"column_names": table.column_names},
    )
    assert decoded.binding["row_count"] == 1
    assert "eligible" not in decoded.projection["column_names"]

    extra_schema = pa.schema(
        list(role_schema("pit_universe")) + [pa.field("eligible", pa.bool_(), nullable=False)]
    )
    extra_ref = _put_table(
        store,
        source_root,
        role="pit_universe",
        rows=[{**rows[0], "eligible": True}],
        name="pit-extra.parquet",
        schema=extra_schema,
    )
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_SCHEMA_INVALID"):
        decode_source_role(
            system_store=store,
            source_object_ref=extra_ref,
            role="pit_universe",
            projector=_discard_projection,
        )


def test_sparse_weights_require_decimal128_38_12_and_reject_zero_duplicate(
    tmp_path: Path,
) -> None:
    store, source_root = _store(tmp_path)
    valid = [
        {
            "signal_session": date(2026, 8, 17),
            "configuration_id": "config-low",
            "symbol": "000001.SZ",
            "weight": Decimal("0.500000000000"),
        },
        {
            "signal_session": date(2026, 8, 17),
            "configuration_id": "config-low",
            "symbol": "600000.SH",
            "weight": Decimal("-0.500000000000"),
        },
    ]
    ref = _put_table(store, source_root, role="sparse_weights", rows=valid)
    decoded = decode_source_role(
        system_store=store,
        source_object_ref=ref,
        role="sparse_weights",
        projector=lambda table, binding: {"weight_type": str(table.schema.field("weight").type)},
    )
    assert decoded.projection["weight_type"] == "decimal128(38, 12)"

    zero_ref = _put_table(
        store,
        source_root,
        role="sparse_weights",
        rows=[{**valid[0], "weight": Decimal("0.000000000000")}],
        name="weights-zero.parquet",
    )
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_VALUE_INVALID"):
        decode_source_role(
            system_store=store,
            source_object_ref=zero_ref,
            role="sparse_weights",
            projector=_discard_projection,
        )

    duplicate_ref = _put_table(
        store,
        source_root,
        role="sparse_weights",
        rows=[valid[0], valid[0]],
        name="weights-duplicate.parquet",
    )
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_DUPLICATE"):
        decode_source_role(
            system_store=store,
            source_object_ref=duplicate_ref,
            role="sparse_weights",
            projector=_discard_projection,
        )

    float_schema = pa.schema(
        list(role_schema("sparse_weights"))[:-1]
        + [pa.field("weight", pa.float64(), nullable=False)]
    )
    float_ref = _put_table(
        store,
        source_root,
        role="sparse_weights",
        rows=[{**valid[0], "weight": 0.5}],
        name="weights-float.parquet",
        schema=float_schema,
    )
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_SCHEMA_INVALID"):
        decode_source_role(
            system_store=store,
            source_object_ref=float_ref,
            role="sparse_weights",
            projector=_discard_projection,
        )


def test_source_resolver_rejects_mode_hardlink_symlink_and_byte_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, source_root = _store(tmp_path)
    rows = [
        {
            "signal_session": date(2026, 8, 17),
            "symbol": "000001.SZ",
            "industry": "银行",
            "total_mv": 1.0,
            "tradable": True,
        }
    ]

    mode_ref = _put_table(store, source_root, role="pit_universe", rows=rows, name="mode.parquet")
    (source_root / "mode.parquet").chmod(0o644)
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_SECURITY_FAILED"):
        decode_source_role(
            system_store=store,
            source_object_ref=mode_ref,
            role="pit_universe",
            projector=_discard_projection,
        )

    hard_ref = _put_table(store, source_root, role="pit_universe", rows=rows, name="hard.parquet")
    os.link(source_root / "hard.parquet", source_root / "hard-alias.parquet")
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_SECURITY_FAILED"):
        decode_source_role(
            system_store=store,
            source_object_ref=hard_ref,
            role="pit_universe",
            projector=_discard_projection,
        )

    link_ref = _put_table(store, source_root, role="pit_universe", rows=rows, name="link.parquet")
    original = source_root / "link-original.parquet"
    (source_root / "link.parquet").rename(original)
    (source_root / "link.parquet").symlink_to(original.name)
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_SECURITY_FAILED"):
        decode_source_role(
            system_store=store,
            source_object_ref=link_ref,
            role="pit_universe",
            projector=_discard_projection,
        )

    changed_ref = _put_table(
        store, source_root, role="pit_universe", rows=rows, name="changed.parquet"
    )
    original_open = store.open_source_object

    @contextmanager
    def changed_open(*args, **kwargs):
        with original_open(*args, **kwargs) as (payload, stream):
            yield {**payload, "byte_sha256": "0" * 64}, stream

    monkeypatch.setattr(store, "open_source_object", changed_open)
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_BYTES_CHANGED"):
        decode_source_role(
            system_store=store,
            source_object_ref=changed_ref,
            role="pit_universe",
            projector=_discard_projection,
        )


def test_source_decoder_rejects_json_alias_and_close_volume_fallback(tmp_path: Path) -> None:
    store, source_root = _store(tmp_path)
    json_path = source_root / "market.json"
    json_path.write_bytes(b"{}")
    json_path.chmod(0o600)
    json_ref = store.put_source_file(
        "market.json",
        source_object_id="json-market",
        media_type="application/json",
        source_format="JSON",
        created_at=STAMP,
    )
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_FORMAT_INVALID"):
        decode_source_role(
            system_store=store,
            source_object_ref=json_ref,
            role="market_history",
            projector=_discard_projection,
        )

    alias_schema = pa.schema(
        [
            pa.field("trade_date", pa.date32(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("close", pa.float64(), nullable=True),
            pa.field("volume", pa.float64(), nullable=True),
        ]
    )
    alias_ref = _put_table(
        store,
        source_root,
        role="market_history",
        rows=[
            {
                "trade_date": date(2026, 8, 17),
                "symbol": "000001.SZ",
                "close": 10.0,
                "volume": 1_000.0,
            }
        ],
        name="market-alias.parquet",
        schema=alias_schema,
    )
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_SCHEMA_INVALID"):
        decode_source_role(
            system_store=store,
            source_object_ref=alias_ref,
            role="market_history",
            projector=_discard_projection,
        )


def test_calendar_requires_391_rows_and_exact_utc_windows(tmp_path: Path) -> None:
    store, source_root = _store(tmp_path)
    valid_ref = _put_table(
        store,
        source_root,
        role="exchange_calendar",
        rows=_calendar_rows(),
    )
    decoded = decode_source_role(
        system_store=store,
        source_object_ref=valid_ref,
        role="exchange_calendar",
        projector=_discard_projection,
    )
    assert decoded.binding["row_count"] == 391

    short_ref = _put_table(
        store,
        source_root,
        role="exchange_calendar",
        rows=_calendar_rows(390),
        name="calendar-short.parquet",
    )
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_CARDINALITY_INVALID"):
        decode_source_role(
            system_store=store,
            source_object_ref=short_ref,
            role="exchange_calendar",
            projector=_discard_projection,
        )

    reversed_rows = _calendar_rows()
    reversed_rows[2]["open_session"] = reversed_rows[1]["open_session"]
    reverse_ref = _put_table(
        store,
        source_root,
        role="exchange_calendar",
        rows=reversed_rows,
        name="calendar-reversed.parquet",
    )
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_ORDER_INVALID"):
        decode_source_role(
            system_store=store,
            source_object_ref=reverse_ref,
            role="exchange_calendar",
            projector=_discard_projection,
        )


def test_label_source_accepts_only_two_sessions_and_missing_without_coverage_gate(
    tmp_path: Path,
) -> None:
    store, source_root = _store(tmp_path)
    rows = [
        {
            "price_date": date(2026, 8, 18),
            "symbol": "000001.SZ",
            "adj_close": None,
        },
        {
            "price_date": date(2026, 9, 16),
            "symbol": "000001.SZ",
            "adj_close": 10.0,
        },
    ]
    ref = _put_table(store, source_root, role="matured_label_prices", rows=rows)
    decoded = decode_source_role(
        system_store=store,
        source_object_ref=ref,
        role="matured_label_prices",
        projector=lambda table, binding: {
            "adj_close_null_count": table.column("adj_close").null_count
        },
    )
    assert decoded.binding["row_count"] == 2
    assert decoded.projection["adj_close_null_count"] == 1

    one_session = _put_table(
        store,
        source_root,
        role="matured_label_prices",
        rows=[rows[0]],
        name="labels-one-session.parquet",
    )
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_CARDINALITY_INVALID"):
        decode_source_role(
            system_store=store,
            source_object_ref=one_session,
            role="matured_label_prices",
            projector=_discard_projection,
        )


def test_parquet_decoded_row_group_bound_fails_before_table_read(monkeypatch) -> None:
    class _Column:
        total_uncompressed_size = source_module.MAXIMUM_DECODED_ROW_GROUP_BYTES + 1

    class _RowGroup:
        num_columns = 1

        @staticmethod
        def column(index: int) -> _Column:
            assert index == 0
            return _Column()

    class _Metadata:
        serialized_size = 1
        num_row_groups = 1
        num_rows = 1

        @staticmethod
        def row_group(index: int) -> _RowGroup:
            assert index == 0
            return _RowGroup()

    class _Parquet:
        metadata = _Metadata()
        schema_arrow = role_schema("pit_universe")
        read_called = False

        @classmethod
        def read(cls, *, use_threads: bool):
            assert use_threads is False
            cls.read_called = True
            raise AssertionError("oversized row group must fail before decode")

    monkeypatch.setattr(source_module.pq, "ParquetFile", lambda _: _Parquet())
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_SIZE_EXCEEDED"):
        source_module._decode_parquet(io.BytesIO(b"PAR1"), "pit_universe")
    assert not _Parquet.read_called


def test_parquet_aggregate_decoded_bound_fails_before_table_read(monkeypatch) -> None:
    class _Column:
        total_uncompressed_size = 200 * 1024 * 1024

    class _RowGroup:
        num_columns = 1

        @staticmethod
        def column(index: int) -> _Column:
            assert index == 0
            return _Column()

    class _Metadata:
        serialized_size = 1
        num_row_groups = 2
        num_rows = 2

        @staticmethod
        def row_group(index: int) -> _RowGroup:
            assert index in {0, 1}
            return _RowGroup()

    class _Parquet:
        metadata = _Metadata()
        schema_arrow = role_schema("pit_universe")
        read_called = False

        @classmethod
        def read(cls, *, use_threads: bool):
            assert use_threads is False
            cls.read_called = True
            raise AssertionError("aggregate bound must fail before decode")

    monkeypatch.setattr(source_module.pq, "ParquetFile", lambda _: _Parquet())
    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_SIZE_EXCEEDED"):
        source_module._decode_parquet(io.BytesIO(b"PAR1"), "pit_universe")
    assert not _Parquet.read_called


def test_source_decoder_holds_system_lease_until_projection_finishes(monkeypatch) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "signal_session": date(2026, 8, 17),
                "symbol": "000001.SZ",
                "industry": "银行",
                "total_mv": 1_000_000.0,
                "tradable": True,
            }
        ],
        schema=role_schema("pit_universe"),
    )
    ref = {
        "kind": "system.source_object",
        "contract_sha256": "1" * 64,
        "artifact_id": "source-memory-bound",
        "semantic_sha256": "2" * 64,
        "byte_sha256": "3" * 64,
    }
    payload = {
        "source_root_id": ROOT_ID,
        "media_type": "application/vnd.apache.parquet",
        "source_format": "PARQUET",
        "byte_sha256": "4" * 64,
    }
    artifact = {"created_at": STAMP, "payload": payload}
    monkeypatch.setattr(
        source_module,
        "_resolve_source_ref",
        lambda system_store, source_ref: (artifact, payload, dict(ref)),
    )
    monkeypatch.setattr(
        source_module,
        "_normalize_table",
        lambda role, observed, schema_sha: (
            "5" * 64,
            "2026-08-17",
            "2026-08-17",
        ),
    )
    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def _decode(stream: object, role: str) -> pa.Table:
        assert role == "pit_universe"
        return table

    monkeypatch.setattr(source_module, "_decode_parquet", _decode)

    class _Store:
        @contextmanager
        def open_source_object(self, *args, **kwargs):
            nonlocal active, maximum_active
            del args, kwargs
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            try:
                yield payload, io.BytesIO(b"PAR1")
            finally:
                with lock:
                    active -= 1

    store = _Store()

    def consume(_: int) -> None:
        def project(observed: pa.Table, binding: Mapping[str, object]) -> dict:
            assert observed is table
            assert binding["row_count"] == 1
            with lock:
                assert active > 0
            time.sleep(0.05)
            return {"row_count": observed.num_rows}

        decoded = decode_source_role(
            system_store=store,
            source_object_ref=ref,
            role="pit_universe",
            projector=project,
        )
        assert decoded.projection == {"row_count": 1}
        assert not hasattr(decoded, "table")

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(consume, range(2)))
    assert maximum_active == 2


@pytest.mark.parametrize("forbidden", ["table", "array", "pandas", "numpy"])
def test_source_decoder_rejects_live_object_projection(
    tmp_path: Path,
    forbidden: str,
) -> None:
    store, source_root = _store(tmp_path)
    ref = _put_table(
        store,
        source_root,
        role="pit_universe",
        rows=[
            {
                "signal_session": date(2026, 8, 17),
                "symbol": "000001.SZ",
                "industry": "银行",
                "total_mv": 1_000_000.0,
                "tradable": True,
            }
        ],
    )

    def project(table: pa.Table, binding: Mapping[str, object]) -> object:
        del binding
        if forbidden == "table":
            return table
        if forbidden == "array":
            return table.column("symbol")
        if forbidden == "pandas":
            return table.to_pandas()
        return np.asarray([1.0])

    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_VALUE_INVALID"):
        decode_source_role(
            system_store=store,
            source_object_ref=ref,
            role="pit_universe",
            projector=project,
        )


def test_source_decoder_returns_a_bounded_canonical_copy(tmp_path: Path) -> None:
    store, source_root = _store(tmp_path)
    ref = _put_table(
        store,
        source_root,
        role="pit_universe",
        rows=[
            {
                "signal_session": date(2026, 8, 17),
                "symbol": "000001.SZ",
                "industry": "银行",
                "total_mv": 1_000_000.0,
                "tradable": True,
            }
        ],
    )
    projected = {"values": [1]}

    def project(table: pa.Table, binding: Mapping[str, object]) -> dict:
        del table
        nested_ref = binding["source_object_ref"]
        assert type(nested_ref) is dict
        nested_ref["artifact_id"] = "projector-local-mutation"
        return projected

    decoded = decode_source_role(
        system_store=store,
        source_object_ref=ref,
        role="pit_universe",
        projector=project,
    )
    projected["values"].append(2)
    assert decoded.projection == {"values": [1]}
    assert decoded.source_object_ref == ref
    assert decoded.binding["source_object_ref"] == ref

    with pytest.raises(FactorGovernanceError, match="FACTOR_SOURCE_SIZE_EXCEEDED"):
        decode_source_role(
            system_store=store,
            source_object_ref=ref,
            role="pit_universe",
            projector=lambda table, binding: {"value": "x" * (8 * 1024 * 1024)},
        )


def test_source_projection_has_no_hidden_5001_symbol_cap(tmp_path: Path) -> None:
    store, source_root = _store(tmp_path)
    rows = [
        {
            "signal_session": date(2026, 8, 17),
            "symbol": f"{index:06d}.SZ",
            "industry": "行业",
            "total_mv": 1_000_000.0,
            "tradable": True,
        }
        for index in range(1, 5_002)
    ]
    ref = _put_table(store, source_root, role="pit_universe", rows=rows)
    decoded = decode_source_role(
        system_store=store,
        source_object_ref=ref,
        role="pit_universe",
        projector=lambda table, binding: {
            "signal_values": {
                f"config-{configuration}": {
                    row["symbol"]: float(configuration)
                    for row in table.select(["symbol"]).to_pylist()
                }
                for configuration in range(10)
            }
        },
    )
    assert decoded.binding["row_count"] == 5_001
    assert len(decoded.projection["signal_values"]) == 10
    assert all(len(values) == 5_001 for values in decoded.projection["signal_values"].values())
