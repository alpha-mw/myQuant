"""Read exact retained PIT evidence for a recorded Factor generation.

This is source extraction after the caller's core/Factor admission, not new
Factor, research, portfolio or trading authority. No current-head discovery.
"""

from collections.abc import Mapping
import hashlib
import io
from typing import Any

import pandas as pd

from quant_investor.contracts import canonical_json_bytes
from quant_investor.market.pit_universe import (
    PITUniverseRecord,
    PITUniverseStore,
    filter_symbols_by_pit_status,
)
from .production_authority import FactorProductionStore
from .governance.errors import FactorGovernanceError
from .governance.production_authority import (
    _canonical_json_mapping,
    _factor_source_topology,
    _deep_replay_market_pit_selection,
    _resolved_source_leaf,
)

FOCUS_COMPANIES = ("002384.SZ", "002463.SZ")


def decode_focus_pit(*, membership_raw: bytes, manifest_raw: bytes, trade_date: str) -> list[dict]:
    """Decode the native complete record schema, never a permissive symbol list."""
    manifest = _canonical_json_mapping(manifest_raw, label="focus PIT manifest")
    if hashlib.sha256(membership_raw).hexdigest() != manifest.get("canonical_sha256"):
        raise FactorGovernanceError("FOCUS_PIT_MEMBERSHIP_SHA_MISMATCH")
    try:
        frame = pd.read_parquet(io.BytesIO(membership_raw))
    except Exception as exc:
        raise FactorGovernanceError("FOCUS_PIT_MEMBERSHIP_PARQUET_INVALID") from exc
    if list(frame.columns) != list(PITUniverseRecord.__dataclass_fields__):
        raise FactorGovernanceError("FOCUS_PIT_MEMBERSHIP_SCHEMA_INVALID")
    records = [PITUniverseRecord.from_dict(row) for row in frame.to_dict(orient="records")]
    if len(records) != manifest.get("row_count") or PITUniverseStore._records_sha256(
        records
    ) != manifest.get("records_sha256"):
        raise FactorGovernanceError("FOCUS_PIT_RECORD_BINDING_MISMATCH")
    selected = filter_symbols_by_pit_status(
        FOCUS_COMPANIES, as_of=trade_date, records=records, required=True
    )
    return [selected.metadata["statuses"][company] for company in FOCUS_COMPANIES]


def read_bound_focus_pit(
    *,
    workspace: str,
    generation_ref: Mapping[str, str],
    trade_date: str,
    expected_manifest_sha256: str,
    expected_membership_sha256: str,
) -> dict[str, Any]:
    """Resolve native logical aliases through their immutable Factor mirrors."""
    store = FactorProductionStore(workspace)
    artifacts: dict[str, tuple[dict, dict]] = {}
    sources: dict[str, tuple[dict, int, bytes]] = {}
    physical: dict[str, str] = {}

    def resolve(reference):
        value = store._read_artifact_ref(reference, label="focus PIT source ancestry")
        key = reference["byte_sha256"]
        if key in artifacts and artifacts[key][1] != value:
            raise FactorGovernanceError("FOCUS_PIT_ARTIFACT_CHANGED")
        artifacts[key] = (dict(reference), value)
        physical[str(store._artifact_path(reference))] = key
        return value

    def source(reference, maximum_bytes):
        descriptor, raw = store._mirrored_source_resolver(reference, maximum_bytes)
        key = reference["byte_sha256"]
        if key in sources and sources[key][2] != raw:
            raise FactorGovernanceError("FOCUS_PIT_SOURCE_CHANGED")
        sources[key] = (dict(reference), maximum_bytes, raw)
        metadata_path, raw_path = store._source_mirror_paths(reference)
        physical[str(metadata_path)] = store.read(metadata_path).byte_sha256
        physical[str(raw_path)] = hashlib.sha256(raw).hexdigest()
        return descriptor, raw

    generation = store._read_factor_generation_ref(
        generation_ref, label="focus PIT recorded generation"
    )
    if generation["payload"]["as_of"] != trade_date:
        raise FactorGovernanceError("FOCUS_PIT_GENERATION_DATE_MISMATCH")
    resolve(generation_ref)
    payload = generation["payload"]
    _, branches = _factor_source_topology(
        payload["factor_source_bundle_ref"],
        artifact_resolver=resolve,
        source_resolver=source,
    )
    selection = resolve(payload["market_pit_selection_ref"])
    market_input = resolve(payload["market_input_ref"])
    market_leaves = [
        _resolved_source_leaf(
            market_input["payload"][field],
            artifact_resolver=resolve,
            source_resolver=source,
            label="focus PIT Market custody",
        )
        for field in ("market_pointer_source_ref", "market_snapshot_manifest_source_ref")
    ]
    selection, files = _deep_replay_market_pit_selection(
        selection,
        market_leaves=market_leaves,
        pit_leaves=branches["pit_universe"],
    )
    selected = selection["payload"]
    if (
        selected["as_of"] != trade_date
        or selected["pit_generation_manifest_sha256"] != expected_manifest_sha256
        or selected["pit_membership_sha256"] != expected_membership_sha256
        or market_input["payload"]["pit_membership_sha256"] != expected_membership_sha256
        or market_input["payload"]["market_pit_selection_ref"]
        != payload["market_pit_selection_ref"]
    ):
        raise FactorGovernanceError("FOCUS_PIT_CORE_BINDING_MISMATCH")
    rows = decode_focus_pit(
        membership_raw=files["pit_membership"]["raw"],
        manifest_raw=files["pit_generation_manifest"]["raw"],
        trade_date=trade_date,
    )

    def retained_ref(name):
        leaf = files[name]
        _, path = store._source_mirror_paths(leaf["ref"])
        return {"path": str(path), "sha256": hashlib.sha256(leaf["raw"]).hexdigest()}

    for reference, original in artifacts.values():
        if store._read_artifact_ref(reference, label="focus PIT final ancestry check") != original:
            raise FactorGovernanceError("FOCUS_PIT_ARTIFACT_CHANGED")
    for reference, maximum_bytes, original_raw in sources.values():
        if store._mirrored_source_resolver(reference, maximum_bytes)[1] != original_raw:
            raise FactorGovernanceError("FOCUS_PIT_SOURCE_CHANGED")
    if (
        store._read_factor_generation_ref(generation_ref, label="focus PIT final generation")
        != generation
    ):
        raise FactorGovernanceError("FOCUS_PIT_GENERATION_CHANGED")
    return {
        "trade_date": trade_date,
        "company_keyset": list(FOCUS_COMPANIES),
        "company_set_sha256": hashlib.sha256(
            canonical_json_bytes(list(FOCUS_COMPANIES))
        ).hexdigest(),
        "pit_selection_ref": {
            "path": str(store._artifact_path(payload["market_pit_selection_ref"])),
            "sha256": payload["market_pit_selection_ref"]["byte_sha256"],
        },
        "pit_generation_manifest_ref": retained_ref("pit_generation_manifest"),
        "pit_membership_ref": retained_ref("pit_membership"),
        "company_statuses": rows,
        "source_refs": [
            {"path": path, "sha256": digest} for path, digest in sorted(physical.items())
        ],
        "validation_scope": "RECORDED_FACTOR_PIT_SOURCE_EXTRACTION",
        "consumer_admission": False,
    }
