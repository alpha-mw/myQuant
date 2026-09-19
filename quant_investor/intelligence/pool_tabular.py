"""Native rank-to-Parquet representation and exact source-bound pool manifest."""

from collections.abc import Mapping, Sequence
from decimal import Decimal
import hashlib
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from quant_investor.contracts import canonical_json_bytes
from quant_investor.factors.production_observation import validate_factor_production_observation
from ._common import IntelligenceError, artifact_ref, build_artifact, business_identity, timestamp

LEGACY_MANIFEST_KIND = "daily_research_pool_manifest"
TABULAR_MANIFEST_KIND = "daily_research_tabular_pool_manifest"
POOL_MANIFEST_KINDS = frozenset({LEGACY_MANIFEST_KIND, TABULAR_MANIFEST_KIND})
MAX_TABLE_BYTES = 8 * 1024 * 1024
TABLE_SCHEMA = pa.schema(
    [
        pa.field("rank", pa.int32(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("low_percentile", pa.decimal128(13, 12), nullable=False),
        pa.field("w80_percentile", pa.decimal128(13, 12), nullable=False),
        pa.field("combined_percentile", pa.decimal128(13, 12), nullable=False),
    ],
    metadata={b"quant_investor.pool_format": b"top100"},
)


def rank_table(rank: Mapping[str, Any]) -> pa.Table:
    rows = [
        {
            "rank": index,
            "symbol": row["symbol"],
            "low_percentile": Decimal(row["factor_percentiles"]["LOW"]),
            "w80_percentile": Decimal(row["factor_percentiles"]["W80"]),
            "combined_percentile": Decimal(row["combined_percentile"]),
        }
        for index, row in enumerate(rank["payload"]["pool_rows"], 1)
    ]
    if len(rows) != 100:
        raise IntelligenceError("Top100 table requires exactly 100 native rank rows")
    return pa.Table.from_pylist(rows, schema=TABLE_SCHEMA)


def encode_top100(rank: Mapping[str, Any]) -> bytes:
    target = pa.BufferOutputStream()
    pq.write_table(
        rank_table(rank),
        target,
        version="2.6",
        data_page_version="2.0",
        compression="NONE",
        use_dictionary=False,
        write_statistics=False,
        row_group_size=100,
        store_schema=True,
        write_page_index=False,
        write_page_checksum=True,
    )
    return target.getvalue().to_pybytes()


def verify_top100(raw: bytes, *, expected_sha: str, rank: Mapping[str, Any]) -> None:
    # Byte custody precedes any Parquet metadata/decompression work.
    if len(raw) > MAX_TABLE_BYTES or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise IntelligenceError("Top100 table byte SHA or size differs")
    try:
        source = pq.ParquetFile(pa.BufferReader(raw), page_checksum_verification=True)
        metadata = source.metadata
        if (
            metadata.num_rows != 100
            or metadata.num_columns != 5
            or metadata.num_row_groups != 1
            or metadata.row_group(0).total_byte_size > MAX_TABLE_BYTES
            or not source.schema_arrow.equals(TABLE_SCHEMA, check_metadata=True)
        ):
            raise IntelligenceError("Top100 table metadata differs")
        table = source.read(use_threads=False)
        if any(column.null_count for column in table.columns) or not table.equals(
            rank_table(rank), check_metadata=True
        ):
            raise IntelligenceError("Top100 table rows differ from native rank")
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise IntelligenceError("Top100 table validation failed") from exc


def observation_bindings(
    rank: Mapping[str, Any], observations: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    observed = {}
    for value in observations:
        artifact = validate_factor_production_observation(value)
        alias = artifact["payload"]["factor_alias"]
        if alias in observed:
            raise IntelligenceError("Top100 observation alias duplicated")
        observed[alias] = artifact
    if set(observed) != {"LOW", "W80"}:
        raise IntelligenceError("Top100 requires exact LOW/W80 observations")
    payload = rank["payload"]
    if sorted(
        (artifact_ref(o) for o in observed.values()), key=lambda r: r["artifact_id"]
    ) != sorted(payload["observation_refs"], key=lambda r: r["artifact_id"]):
        raise IntelligenceError("Top100 observation refs differ from rank")
    generation = payload["factor_generation_ref"]
    common = (
        "market_pointer_sha256",
        "market_manifest_sha256",
        "pit_pointer_sha256",
        "pit_manifest_sha256",
        "pit_membership_sha256",
        "calendar_compilation_ref",
        "calendar_capture_custody_attestation_ref",
    )
    for alias, artifact in observed.items():
        row = artifact["payload"]
        factor = next(
            item for item in policy["payload"]["factor_rows"] if item["factor_alias"] == alias
        )
        expected = {
            "signal_date": payload["signal_date"],
            "factor_pointer_sha256": payload["factor_pointer_sha256"],
            "factor_generation_id": generation["artifact_id"],
            "factor_generation_sha256": generation["byte_sha256"],
            "factor_id": factor["factor_id"],
            "symbol_count": payload["common_symbol_count"],
            "signal_symbol_set_sha256": payload["common_symbol_set_sha256"],
            "state": "OPEN",
            "authority": "NON_AUTHORIZING",
        }
        if (
            any(row[key] != value for key, value in expected.items())
            or any(row[key] != observed["LOW"]["payload"][key] for key in common)
            or timestamp(artifact["created_at"], label="observation time") > rank["created_at"]
        ):
            raise IntelligenceError("Top100 observation source binding differs")
    return {
        "factor_generation_id": generation["artifact_id"],
        "factor_pointer_sha": payload["factor_pointer_sha256"],
        "low_observation_sha": artifact_ref(observed["LOW"])["byte_sha256"],
        "w80_observation_sha": artifact_ref(observed["W80"])["byte_sha256"],
        "market_pointer_sha": observed["LOW"]["payload"]["market_pointer_sha256"],
        "pit_pointer_sha": observed["LOW"]["payload"]["pit_pointer_sha256"],
    }


def tabular_documents(
    legacy: Mapping[str, dict], *, bindings: dict, generated_at: str, parquet: bytes
) -> dict[str, bytes]:
    rank = legacy["factor_research_rank.json"]
    stamp = timestamp(generated_at, label="pool generated_at")
    if stamp < timestamp(rank["created_at"], label="rank created_at"):
        raise IntelligenceError("Top100 publication predates native rank")
    original = legacy["manifest.json"]
    controls = {"authority", "production", "research_only", "run_state"}
    fields = {k: v for k, v in original["payload"].items() if k not in controls | {"manifest_id"}}
    day = fields["signal_date"]
    fields.update(
        bindings,
        expected_leaf_names=sorted([*legacy, "top100.parquet"]),
        trade_date=f"{day[:4]}-{day[4:6]}-{day[6:]}",
        generated_at=stamp,
        policy_sha=fields["policy_byte_sha256"],
        row_count=100,
        top100_sha=hashlib.sha256(parquet).hexdigest(),
    )
    manifest = build_artifact(
        kind=TABULAR_MANIFEST_KIND,
        identity_field="manifest_id",
        identity=business_identity(kind=TABULAR_MANIFEST_KIND, identity_inputs=fields),
        created_at=stamp,
        fields=fields,
    )
    receipt_fields = {
        k: v
        for k, v in legacy["publish_receipt.json"]["payload"].items()
        if k not in controls | {"receipt_id"}
    }
    receipt_fields.update(
        manifest_byte_sha256=hashlib.sha256(canonical_json_bytes(manifest)).hexdigest(),
        manifest_ref=artifact_ref(manifest),
    )
    receipt = build_artifact(
        kind="daily_research_pool_receipt",
        identity_field="receipt_id",
        identity=business_identity(
            kind="daily_research_pool_receipt",
            identity_inputs={"manifest_id": manifest["artifact_id"]},
        ),
        created_at=stamp,
        fields=receipt_fields,
    )
    documents = {**legacy, "manifest.json": manifest, "publish_receipt.json": receipt}
    return {
        **{name: canonical_json_bytes(doc) for name, doc in documents.items()},
        "top100.parquet": parquet,
    }
