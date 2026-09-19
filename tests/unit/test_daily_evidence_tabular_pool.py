"""Real pool publisher, native observation validator and Parquet custody checks."""

from copy import deepcopy
import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_investor.contracts import seal_artifact
from quant_investor.intelligence import IntelligenceError
from quant_investor.intelligence import storage
from quant_investor.intelligence._common import artifact_ref
from quant_investor.intelligence.pool_tabular import (
    TABLE_SCHEMA,
    TABULAR_MANIFEST_KIND,
    observation_bindings,
    rank_table,
    tabular_documents,
    verify_top100,
)
from test_unified_daily_intelligence_storage import _pool_rank, _write_request


def setup(root, monkeypatch):
    policy_ref = storage.publish_theme_policy_v2(root)
    policy = storage.approved_theme_policy_v2()
    rank = _pool_rank(root, policy, signal_date="20260828")
    monkeypatch.setattr(storage, "_pool_generated_at", lambda: "2026-08-28T14:00:00Z")
    arguments = {
        "rank": rank,
        "expected_policy_sha256": policy_ref["daily_policy_sha256"],
        "policy_path": storage.THEME_POLICY_V2_RELATIVE_PATH,
    }
    return storage.DailyResearchPoolStore(root), arguments, policy


def inventory(root):
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in root.rglob("*")
        if p.is_file()
    }


def fail(*args, **kwargs):
    pytest.fail("unexpected publisher, callback or clock")


def test_native_table_manifest_and_zero_write_repeat(tmp_path, monkeypatch):
    store, args, policy = setup(tmp_path, monkeypatch)
    result = store.publish(**args, before_publish=lambda: None)
    root = tmp_path / result["pool_root"]
    manifest = json.loads((root / "manifest.json").read_bytes())
    payload = manifest["payload"]
    observations = store._observations(args["rank"], None)
    bindings = observation_bindings(args["rank"], observations, policy)
    assert {k: payload[k] for k in bindings} == bindings
    assert manifest["kind"] == TABULAR_MANIFEST_KIND
    assert payload["trade_date"] == "2026-08-28" and payload["row_count"] == 100
    assert payload["generated_at"] == manifest["created_at"] == "2026-08-28T14:00:00Z"
    assert (
        json.loads((root / "publish_receipt.json").read_bytes())["created_at"]
        == manifest["created_at"]
    )
    assert payload["policy_sha"] == args["expected_policy_sha256"]
    raw = (root / "top100.parquet").read_bytes()
    assert payload["top100_sha"] == hashlib.sha256(raw).hexdigest()
    parquet = pq.ParquetFile(pa.BufferReader(raw))
    table = parquet.read()
    assert table.schema.equals(TABLE_SCHEMA, check_metadata=True)
    assert table.equals(rank_table(args["rank"]), check_metadata=True)
    assert parquet.metadata.row_group(0).column(0).compression == "UNCOMPRESSED"
    assert parquet.metadata.row_group(0).column(0).statistics is None
    before = inventory(tmp_path)
    monkeypatch.setattr(storage, "_pool_generated_at", fail)
    monkeypatch.setattr(storage, "_write_pool_staging", fail)
    repeated = store.publish(**args, before_publish=fail)
    assert repeated["command_status"] == "NO_ACTION"
    assert repeated["publication_state"] == "ALREADY_SUCCEEDED"
    refs = store.verify(**args, required_format="TABULAR")
    assert len(refs) == 5 and refs["top100.parquet"]["sha256"] == payload["top100_sha"]
    assert before == inventory(tmp_path)


@pytest.mark.parametrize("fault", ["missing", "extra", "bytes", "rank", "format"])
def test_existing_conflicts_never_write(tmp_path, monkeypatch, fault):
    store, args, policy = setup(tmp_path, monkeypatch)
    if fault == "format":
        legacy = storage._pool_documents(
            rank=args["rank"],
            policy=policy,
            policy_path=args["policy_path"],
            policy_sha256=args["expected_policy_sha256"],
        )
        root = tmp_path / storage.POOL_ROOT_RELATIVE_PATH / storage.POOL_STRATEGY_ID / "2026-08-28"
        root.mkdir(parents=True, mode=0o700)
        for name, value in legacy.items():
            _write_request(root / name, value)
        assert len(store.verify(**args)) == 4
    else:
        result = store.publish(**args, before_publish=lambda: None)
        root = tmp_path / result["pool_root"]
        if fault == "missing":
            (root / "top100.parquet").unlink()
        elif fault == "extra":
            _write_request(root / "unknown.json", {})
        elif fault == "bytes":
            (root / "top100.parquet").write_bytes(b"bad binary")
        else:
            rank = deepcopy(args["rank"])
            row = rank["payload"]["pool_rows"][-1]
            row["combined_percentile"] = "0.005000000000"
            row["factor_percentiles"] = {"LOW": "0.005000000000", "W80": "0.005000000000"}
            args["rank"] = seal_artifact(
                rank["kind"], rank["payload"], created_at=rank["created_at"]
            )
    before = inventory(tmp_path)
    monkeypatch.setattr(storage, "_pool_generated_at", fail)
    with pytest.raises(storage.ResearchPoolConflict) as error:
        store.publish(**args, before_publish=fail)
    assert error.value.code == "RESEARCH_POOL_CONFLICT"
    assert error.value.public_fields == {"publication_state": "CONFLICT"}
    with pytest.raises(storage.ResearchPoolConflict):
        store.verify(**args, required_format="TABULAR")
    assert before == inventory(tmp_path)


@pytest.mark.parametrize(
    "field",
    [
        "trade_date",
        "generated_at",
        "factor_generation_id",
        "factor_pointer_sha",
        "low_observation_sha",
        "w80_observation_sha",
        "market_pointer_sha",
        "pit_pointer_sha",
        "policy_sha",
        "row_count",
        "top100_sha",
    ],
)
def test_resealed_manifest_alias_drift_is_rejected(tmp_path, monkeypatch, field):
    store, args, _ = setup(tmp_path, monkeypatch)
    result = store.publish(**args, before_publish=lambda: None)
    path = tmp_path / result["manifest_path"]
    manifest = json.loads(path.read_bytes())
    payload = manifest["payload"]
    payload[field] = (
        99
        if field == "row_count"
        else ("2026-08-28T14:01:00Z" if field == "generated_at" else "drift")
    )
    _write_request(
        path, seal_artifact(manifest["kind"], payload, created_at=manifest["created_at"])
    )
    before = inventory(tmp_path)
    with pytest.raises(storage.ResearchPoolConflict):
        store.verify(**args)
    assert before == inventory(tmp_path)


@pytest.mark.parametrize("fault", ["reverse", "null", "float", "metadata", "rows", "decimal"])
def test_table_semantics_reject_rehashed_corruption(tmp_path, monkeypatch, fault):
    _, args, _ = setup(tmp_path, monkeypatch)
    table = rank_table(args["rank"])
    if fault == "reverse":
        table = table.take(list(reversed(range(100))))
    elif fault == "null":
        rows = table.to_pylist()
        rows[0]["symbol"] = None
        schema = pa.schema(
            [pa.field(f.name, f.type) for f in TABLE_SCHEMA], metadata=TABLE_SCHEMA.metadata
        )
        table = pa.Table.from_pylist(rows, schema=schema)
    elif fault == "float":
        table = table.set_column(2, "low_percentile", table.column(2).cast(pa.float64()))
    elif fault == "metadata":
        table = table.replace_schema_metadata({})
    elif fault == "rows":
        table = table.slice(1)
    else:
        rows = table.to_pylist()
        rows[0]["low_percentile"] = rows[1]["low_percentile"]
        table = pa.Table.from_pylist(rows, schema=TABLE_SCHEMA)
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer)
    raw = buffer.getvalue().to_pybytes()
    with pytest.raises(IntelligenceError):
        verify_top100(raw, expected_sha=hashlib.sha256(raw).hexdigest(), rank=args["rank"])


def test_concurrent_winner_keeps_its_original_timestamp(tmp_path, monkeypatch):
    store, args, policy = setup(tmp_path, monkeypatch)
    rename = storage._atomic_no_replace
    winner_raw = {}

    def concurrent(staging, target):
        legacy = storage._pool_documents(
            rank=args["rank"],
            policy=policy,
            policy_path=args["policy_path"],
            policy_sha256=args["expected_policy_sha256"],
        )
        bindings = observation_bindings(
            args["rank"], store._observations(args["rank"], None), policy
        )
        winner_raw.update(
            tabular_documents(
                legacy,
                bindings=bindings,
                generated_at="2026-08-28T14:00:01Z",
                parquet=(staging / "top100.parquet").read_bytes(),
            )
        )
        other = staging.parent / ".winner"
        other.mkdir(mode=0o700)
        storage._write_pool_staging(other, winner_raw)
        rename(other, target)
        raise FileExistsError(target)

    monkeypatch.setattr(storage, "_atomic_no_replace", concurrent)
    result = store.publish(**args, before_publish=lambda: None)
    assert result["publication_state"] == "ALREADY_SUCCEEDED"
    root = tmp_path / result["pool_root"]
    assert {p.name: p.read_bytes() for p in root.iterdir()} == winner_raw
    assert not list(root.parent.glob(".2026-08-28.staging-*"))


def test_source_observation_drift_blocks_before_publication(tmp_path, monkeypatch):
    from quant_investor.factors.production_observation import build_factor_production_observation

    store, args, _ = setup(tmp_path, monkeypatch)
    path = tmp_path / "results/factors/observations/2026/08/28/W80.json"
    value = json.loads(path.read_bytes())
    original_id = value["artifact_id"]
    value["payload"]["market_pointer_sha256"] = "e" * 64
    value = build_factor_production_observation(
        inputs=value["payload"],
        factor_row=value["payload"],
        registered_at=value["payload"]["registered_at"],
    )
    _write_request(path, value)
    # Even a re-bound rank cannot hide mutually incompatible native observations.
    rank = args["rank"]
    refs = rank["payload"]["observation_refs"]
    refs[:] = [artifact_ref(value) if ref["artifact_id"] == original_id else ref for ref in refs]
    refs.sort(key=lambda r: (r["kind"], r["artifact_id"]))
    args["rank"] = seal_artifact(rank["kind"], rank["payload"], created_at=rank["created_at"])
    before = inventory(tmp_path)
    with pytest.raises(IntelligenceError, match="source binding differs"):
        store.publish(**args, before_publish=fail)
    assert before == inventory(tmp_path)


def test_public_pool_publish_uses_tabular_writer(tmp_path, monkeypatch, capsys):
    from quant_investor.cli.main import main
    from quant_investor.factors import production_authority
    import quant_investor.intelligence as intelligence

    store, args, _ = setup(tmp_path, monkeypatch)
    observations = store._observations(args["rank"], None)
    day = args["rank"]["payload"]["signal_date"]
    values = {
        "expected_factor_pointer_sha256": args["rank"]["payload"]["factor_pointer_sha256"],
        "expected_policy_sha256": args["expected_policy_sha256"],
        "policy_path": args["policy_path"],
    }
    for alias, observation in zip(("low", "w80"), observations):
        values[alias + "_observation_path"] = (
            f"results/factors/observations/2026/08/28/{alias.upper()}.json"
        )
        values[alias + "_observation_sha256"] = artifact_ref(observation)["byte_sha256"]
    sha = _write_request(tmp_path / "pool-request.json", values)
    # Only Factor snapshot/rank construction is a controlled seam; the public
    # request parser, source reads, native observation validators and publisher run.
    monkeypatch.setattr(
        production_authority,
        "read_factor_production_research_inputs",
        lambda *a, **k: {
            "signal_date": day,
            "factor_generation": {"created_at": args["rank"]["created_at"]},
        },
    )
    monkeypatch.setattr(intelligence, "build_factor_research_rank", lambda **k: args["rank"])
    calls = []
    monkeypatch.setattr(
        production_authority, "assert_factor_production_pointer", lambda *a, **k: calls.append(k)
    )
    command = [
        "research",
        "pool-publish",
        "--workspace-root",
        str(tmp_path),
        "--request",
        "pool-request.json",
        "--expected-request-sha256",
        sha,
    ]
    main(command)
    first = json.loads(capsys.readouterr().out)
    assert first["publication_state"] == "SUCCEEDED"
    main(command)
    second = json.loads(capsys.readouterr().out)
    assert second["publication_state"] == "ALREADY_SUCCEEDED"
    assert len(calls) == 1
    assert (tmp_path / first["pool_root"] / "top100.parquet").is_file()


def test_morning_verifies_complete_pool_without_writes(tmp_path, monkeypatch):
    from quant_investor.intelligence.morning import _morning_pool_reference

    store, args, _ = setup(tmp_path, monkeypatch)
    result = store.publish(**args, before_publish=lambda: None)
    values = {
        "pool_manifest_path": result["manifest_path"],
        "pool_manifest_sha256": result["manifest_sha256"],
    }
    before = inventory(tmp_path)
    monkeypatch.setattr(storage, "_pool_generated_at", fail)
    monkeypatch.setattr(storage.DailyResearchPoolStore, "publish", fail)
    blockers = []
    assert _morning_pool_reference(tmp_path, values, "20260828", blockers) == {
        "path": result["manifest_path"],
        "sha256": result["manifest_sha256"],
    }
    assert not blockers and before == inventory(tmp_path)
    (tmp_path / result["pool_root"] / "top100.parquet").unlink()
    with pytest.raises(IntelligenceError):
        _morning_pool_reference(tmp_path, values, "20260828", [])
