from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from quant_investor.contracts import canonical_json_bytes, seal_artifact
from quant_investor.factors.governance import (
    BLEND_W80,
    LOW_DOLLAR_VOLUME,
    FactorGovernanceError,
    FactorValidationStore,
    prospective_validation_namespace_id,
    validate_configuration_selection,
    validate_factor_status,
    validate_observation,
    validate_signal_capture,
)
from quant_investor.factors.governance.bootstrap import (
    _factor_set_sha256,
    _set_rows,
    bootstrap_factor_definitions,
    compute_bootstrap_signals,
)
from quant_investor.factors.governance.contextual import _signal_hashes
from quant_investor.factors.governance.custody import (
    custody_transaction_id,
    operation_request_sha256,
    replay_custody_chain,
)
from quant_investor.factors.governance.source import decode_source_role, role_schema
from quant_investor.system import (
    BOOTSTRAP_VALIDATION_PROFILE,
    EMPTY_POINTER_SHA256,
    PROSPECTIVE_VALIDATION_PROFILE,
    SystemCASMismatch,
    SystemStore,
    installed_code_manifest_sha256,
)
import quant_investor.system.store as system_store_module

STAMP = "2026-08-16T00:00:00Z"
DECISION_PATH = "operations/unified_cutover/bootstrap-decision.json"


def _write(root: Path, relative: str, raw: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    path.chmod(0o600)


def _write_parquet(
    root: Path,
    relative: str,
    rows: list[dict[str, Any]],
    schema: pa.Schema,
) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
    path.chmod(0o600)


def _source(
    store: SystemStore,
    relative: str,
    *,
    source_format: str,
    media_type: str,
) -> dict[str, str]:
    return store.put_source_file(
        relative,
        source_object_id="source-" + relative.replace("/", "-"),
        media_type=media_type,
        source_format=source_format,
        created_at=STAMP,
    )


def _bundle(
    store: SystemStore,
    bundle_id: str,
    role: str,
    source_ref: dict[str, str],
) -> dict[str, str]:
    artifact = seal_artifact(
        "system.source_bundle",
        {
            "source_bundle_id": bundle_id,
            "state": "IMMUTABLE",
            "sources": [{"role": role, "source_ref": source_ref}],
        },
        created_at=STAMP,
    )
    return store.put_object(artifact)


def _decision_bytes() -> bytes:
    repository = Path(__file__).resolve().parents[2]
    return (repository / DECISION_PATH).read_bytes()


def _calendar_rows() -> list[dict[str, Any]]:
    first = date(2026, 8, 17)
    rows = []
    for ordinal in range(391):
        session = first + timedelta(days=ordinal)
        opens = datetime.combine(session, datetime.min.time(), tzinfo=timezone.utc) + timedelta(
            hours=1
        )
        rows.append(
            {
                "ordinal": ordinal,
                "open_session": session,
                "opens_at_utc": opens,
                "closes_at_utc": opens + timedelta(hours=6),
            }
        )
    return rows


def _market_rows() -> list[dict[str, Any]]:
    symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"]
    first = date(2026, 4, 1)
    rows: list[dict[str, Any]] = []
    for ordinal in range(100):
        trade_date = first + timedelta(days=ordinal)
        for symbol_index, symbol in enumerate(symbols):
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "adj_close": 10.0 + symbol_index + ordinal * (0.01 + symbol_index * 0.001),
                    "amount": 1000.0 + symbol_index * 100.0 + ordinal,
                    "vol": 100.0 + symbol_index * 10.0 + np.sin(ordinal / 7.0),
                }
            )
    return rows


def _pit_rows() -> list[dict[str, Any]]:
    session = date(2026, 7, 9)
    return [
        {
            "signal_session": session,
            "symbol": symbol,
            "industry": "industry-a" if index < 2 else "industry-b",
            "total_mv": float(1_000_000 + index * 100_000),
            "tradable": True,
        }
        for index, symbol in enumerate(["000001.SZ", "000002.SZ", "600000.SH", "600001.SH"])
    ]


def _implementation_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in manifest["payload"]["implementation_rows"]:
        ref = row["implementation_component_ref"]
        rows.append(
            {
                "factor_id": row["factor_id"],
                "implementation_id": row["implementation_id"],
                "implementation_component_kind": ref["kind"],
                "implementation_component_contract_sha256": ref["contract_sha256"],
                "implementation_component_artifact_id": ref["artifact_id"],
                "implementation_component_semantic_sha256": ref["semantic_sha256"],
                "implementation_component_byte_sha256": ref["byte_sha256"],
                "module_name": row["module_name"],
                "qualified_name": row["qualified_name"],
                "code_sha256": row["code_sha256"],
                "family": row["family"],
                "primitive": row["primitive"],
                "direction": row["direction"],
                "formula": row["formula"],
                "normalized_expression": row["normalized_expression"],
                "parameters_json": row["parameters_json"],
                "input_fields": row["input_fields"],
            }
        )
    return rows


def _prospective_symbols() -> list[str]:
    return [
        *[f"{index:06d}.SZ" for index in range(1, 21)],
        *[f"{index:06d}.SH" for index in range(600000, 600020)],
    ]


def _prospective_market_rows(
    signal_session: date = date(2026, 8, 17),
) -> list[dict[str, Any]]:
    first = signal_session - timedelta(days=90)
    rows: list[dict[str, Any]] = []
    for ordinal in range(91):
        trade_date = first + timedelta(days=ordinal)
        for symbol_index, symbol in enumerate(_prospective_symbols()):
            rows.append(
                {
                    "trade_date": trade_date,
                    "symbol": symbol,
                    "adj_close": 10.0
                    + symbol_index * 0.2
                    + ordinal * (0.01 + symbol_index * 0.0001),
                    "amount": 1000.0 + symbol_index * 20.0 + ordinal,
                    "vol": 100.0
                    + symbol_index * 2.0
                    + np.sin(ordinal / (5.0 + symbol_index * 0.01)),
                }
            )
    return rows


def _prospective_pit_rows(
    signal_session: date = date(2026, 8, 17),
) -> list[dict[str, Any]]:
    return [
        {
            "signal_session": signal_session,
            "symbol": symbol,
            "industry": f"industry-{index % 4}",
            "total_mv": float(1_000_000 + index * 10_000),
            "tradable": True,
        }
        for index, symbol in enumerate(_prospective_symbols())
    ]


def _prospective_weight_rows(
    manifest: dict[str, Any],
    signal_session: date = date(2026, 8, 17),
) -> list[dict[str, Any]]:
    symbols = _prospective_symbols()
    return [
        {
            "signal_session": signal_session,
            "configuration_id": row["factor_id"],
            "symbol": symbols[index],
            "weight": Decimal("0.500000000000"),
        }
        for index, row in enumerate(manifest["payload"]["implementation_rows"])
    ]


def _prospective_label_rows(
    *,
    label_start_session: date,
    label_end_session: date,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for symbol_index, symbol in enumerate(_prospective_symbols()):
        start = 10.0 + symbol_index * 0.25
        rows.extend(
            [
                {
                    "price_date": label_start_session,
                    "symbol": symbol,
                    "adj_close": start,
                },
                {
                    "price_date": label_end_session,
                    "symbol": symbol,
                    "adj_close": start * (1.0 + 0.001 * (symbol_index + 1)),
                },
            ]
        )
    return sorted(rows, key=lambda row: (row["price_date"], row["symbol"]))


def _full_prospective_surface() -> dict[str, Any]:
    symbols = _prospective_symbols()
    symbol_count = len(symbols)
    generator = np.random.default_rng(20260814)

    def ranked_latent() -> np.ndarray:
        values = np.empty(symbol_count, dtype=float)
        values[generator.permutation(symbol_count)] = np.linspace(-1.0, 1.0, symbol_count)
        return values

    low_latent = ranked_latent()
    blend_latent = ranked_latent()
    noise_latent = ranked_latent()
    exposure_order = generator.permutation(symbol_count)
    prices = np.zeros((480, symbol_count), dtype=float)
    prices[0] = 20.0 + np.arange(symbol_count, dtype=float) * 0.1
    for position, ordinal in enumerate(range(-89, 390), start=1):
        low_weight = 0.80 + 0.45 * np.sin(ordinal / 19.0) + 0.20 * np.cos(ordinal / 24.0)
        blend_weight = 0.65 + 0.40 * np.sin(ordinal / 21.0 + 0.9) + 0.18 * np.cos(ordinal / 28.0)
        noise_weight = 0.45 * np.sin(ordinal / 11.0) + 0.22 * np.cos(ordinal / 7.0)
        micro = (
            np.sin((np.arange(symbol_count, dtype=float) + 1.0) * (ordinal + 101) * 0.173) * 0.22
        )
        log_return = 0.0007 * (
            low_weight * low_latent
            + blend_weight * blend_latent
            + noise_weight * noise_latent
            + micro
        )
        prices[position] = prices[position - 1] * np.exp(log_return)
    return {
        "symbols": symbols,
        "first_session": date(2026, 8, 17),
        "low_latent": low_latent,
        "blend_latent": blend_latent,
        "exposure_order": exposure_order,
        "prices": prices,
    }


def _surface_price(surface: dict[str, Any], ordinal: int) -> np.ndarray:
    return np.asarray(surface["prices"][ordinal + 90], dtype=float)


def _full_market_rows(surface: dict[str, Any], ordinal: int) -> list[dict[str, Any]]:
    first_session: date = surface["first_session"]
    symbols: list[str] = surface["symbols"]
    low_latent = np.asarray(surface["low_latent"], dtype=float)
    blend_latent = np.asarray(surface["blend_latent"], dtype=float)
    rows: list[dict[str, Any]] = []
    for trade_ordinal in range(ordinal - 90, ordinal + 1):
        price_row = _surface_price(surface, trade_ordinal)
        for symbol_index, symbol in enumerate(symbols):
            amplitude = 2.0 + (1.0 - (blend_latent[symbol_index] + 1.0) / 2.0) * 22.0
            rows.append(
                {
                    "trade_date": first_session + timedelta(days=trade_ordinal),
                    "symbol": symbol,
                    "adj_close": float(price_row[symbol_index]),
                    "amount": float(
                        1200.0
                        * np.exp(-0.28 * low_latent[symbol_index])
                        * (1.0 + 0.002 * np.sin(trade_ordinal / 13.0))
                    ),
                    "vol": float(
                        120.0
                        + amplitude * np.sin((trade_ordinal + 100) * 2.0 * np.pi / 9.0)
                        + 0.2 * np.cos(trade_ordinal / 5.0 + symbol_index)
                    ),
                }
            )
    return rows


def _full_pit_rows(surface: dict[str, Any], ordinal: int) -> list[dict[str, Any]]:
    first_session: date = surface["first_session"]
    symbols: list[str] = surface["symbols"]
    exposure_order = np.asarray(surface["exposure_order"], dtype=int)
    return [
        {
            "signal_session": first_session + timedelta(days=ordinal),
            "symbol": symbol,
            "industry": f"industry-{int(exposure_order[index]) % 4}",
            "total_mv": float(1_000_000 + int(exposure_order[index]) * 10_000),
            "tradable": True,
        }
        for index, symbol in enumerate(symbols)
    ]


def _full_label_rows(surface: dict[str, Any], ordinal: int) -> list[dict[str, Any]]:
    first_session: date = surface["first_session"]
    symbols: list[str] = surface["symbols"]
    rows: list[dict[str, Any]] = []
    for price_ordinal in (ordinal + 1, ordinal + 30):
        price_row = _surface_price(surface, price_ordinal)
        rows.extend(
            {
                "price_date": first_session + timedelta(days=price_ordinal),
                "symbol": symbol,
                "adj_close": float(price_row[index]),
            }
            for index, symbol in enumerate(symbols)
        )
    return sorted(rows, key=lambda row: (row["price_date"], row["symbol"]))


def _release_and_manifest(
    store: SystemStore,
    factor_store: FactorValidationStore,
    *,
    validation_profile: str = BOOTSTRAP_VALIDATION_PROFILE,
) -> tuple[dict[str, str], dict[str, Any], dict[str, str]]:
    release = seal_artifact(
        "system.release",
        {
            "release_id": "factor-store-bootstrap-release",
            "state": "OPERATIONAL",
            "code_sha256": hashlib.sha256(b"release-code").hexdigest(),
            "wheel_sha256": hashlib.sha256(b"release-wheel").hexdigest(),
            "code_manifest_sha256": installed_code_manifest_sha256(),
        },
        created_at=STAMP,
    )
    release_ref = store.put_object(release)
    contextual_ref = store.build_contextual_validator_component(
        validation_profile,
        release_manifest_ref=release_ref,
        created_at=STAMP,
    )
    decoder_ref = store.build_source_decoder_component(
        release_manifest_ref=release_ref,
        created_at=STAMP,
    )
    implementation_refs: dict[str, dict[str, str]] = {}
    from quant_investor.factors.governance.implementations import installed_semantic_row

    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80):
        row = installed_semantic_row(factor_id)
        implementation_refs[factor_id] = store.build_installed_component(
            component_id=row["implementation_id"],
            component_role="SOURCE_IMPLEMENTATION",
            package_name="quant_investor.factors.governance",
            module_names=[row["module_name"]],
            entrypoint_specs=[(row["module_name"], row["qualified_name"])],
            release_manifest_ref=release_ref,
            created_at=STAMP,
        )
    manifest = factor_store.build_validator_manifest(
        release_manifest_ref=release_ref,
        contextual_validator_component_ref=contextual_ref,
        source_decoder_component_ref=decoder_ref,
        implementation_component_refs=implementation_refs,
    )
    return release_ref, manifest, store.put_object(manifest)


def test_bootstrap_store_context_runner_and_status_are_exact_and_nonactivating(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    source_root = tmp_path / "source"
    workspace.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    system_store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id="factor-store-source-root",
    )

    def fixed_clock() -> datetime:
        return datetime(2026, 8, 16, tzinfo=timezone.utc)

    factor_store = FactorValidationStore._for_testing(
        system_store=system_store,
        clock=fixed_clock,
    )
    release_ref, manifest, manifest_ref = _release_and_manifest(system_store, factor_store)

    _write_parquet(
        source_root,
        "bootstrap/calendar.parquet",
        _calendar_rows(),
        role_schema("exchange_calendar"),
    )
    _write_parquet(
        source_root,
        "bootstrap/market.parquet",
        _market_rows(),
        role_schema("market_history"),
    )
    _write_parquet(
        source_root,
        "bootstrap/pit.parquet",
        _pit_rows(),
        role_schema("pit_universe"),
    )
    calendar_ref = _source(
        system_store,
        "bootstrap/calendar.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    market_ref = _source(
        system_store,
        "bootstrap/market.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    pit_ref = _source(
        system_store,
        "bootstrap/pit.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )

    normalized: dict[str, str] = {}
    for role, ref in (
        ("exchange_calendar", calendar_ref),
        ("market_history", market_ref),
        ("pit_universe", pit_ref),
    ):
        decoded = decode_source_role(
            system_store=system_store,
            source_object_ref=ref,
            role=role,
            projector=lambda table, binding: None,
        )
        normalized[role] = decoded.binding["normalized_sha256"]
    market_frame = pa.Table.from_pylist(
        _market_rows(), schema=role_schema("market_history")
    ).to_pandas()
    frames = {
        symbol: frame.drop(columns=["symbol"]).reset_index(drop=True)
        for symbol, frame in market_frame.groupby("symbol", sort=True)
    }
    signals = compute_bootstrap_signals(frames, source_format="PARQUET")
    signal_hashes = _signal_hashes(
        {
            factor_id: {
                symbol: None if np.isnan(value) else float(value).hex()
                for symbol, value in signals[factor_id].sort_index().items()
            }
            for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80)
        }
    )
    definitions = bootstrap_factor_definitions()
    factor_rows, control_rows = _set_rows(definitions)
    factor_set_sha = _factor_set_sha256(
        definitions=definitions,
        factor_rows=factor_rows,
        control_rows=control_rows,
    )

    decision_raw = _decision_bytes()
    implementation_raw = canonical_json_bytes(
        {
            "domain": "myquant-bootstrap-implementation-tree-manifest",
            "implementation_rows": manifest["payload"]["implementation_rows"],
        }
    )
    recomputation_raw = canonical_json_bytes(
        {
            "authority": "NON_AUTHORIZING",
            "domain": "myquant-bootstrap-recomputation",
            "factor_set_sha256": factor_set_sha,
            "factor_weights": [
                {"factor_id": row["factor_id"], "weight": row["weight"]} for row in factor_rows
            ],
            "implementation_rows": manifest["payload"]["implementation_rows"],
            "normalized_source_sha256s": normalized,
            "result": "EXACT_MATCH",
            "signal_sha256s": signal_hashes,
        }
    )
    source_rows = [
        {
            "role": role,
            "source_ref": ref,
            "source_byte_sha256": system_store.get_object(ref)["payload"]["byte_sha256"],
        }
        for role, ref in (
            ("exchange_calendar", calendar_ref),
            ("market", market_ref),
            ("pit_universe", pit_ref),
        )
    ]
    source_rows.sort(key=lambda row: row["role"])
    generation_body = {
        "authority": "NON_AUTHORIZING",
        "domain": "myquant-bootstrap-source-generation",
        "reader_contract": {
            "reader": "MarketDataReader",
            "market": "CN",
            "mode_policy": "strict",
            "source_format": "PARQUET",
            "fallback_allowed": False,
        },
        "source_rows": source_rows,
    }
    source_generation_raw = canonical_json_bytes(
        {
            **generation_body,
            "generation_sha256": hashlib.sha256(canonical_json_bytes(generation_body)).hexdigest(),
        }
    )
    for relative, raw in (
        (DECISION_PATH, decision_raw),
        ("bootstrap/implementation-tree.json", implementation_raw),
        ("bootstrap/recomputation.json", recomputation_raw),
        ("bootstrap/source-generation.json", source_generation_raw),
    ):
        _write(source_root, relative, raw)
    decision_ref = _source(
        system_store,
        DECISION_PATH,
        source_format="JSON",
        media_type="application/json",
    )
    implementation_ref = _source(
        system_store,
        "bootstrap/implementation-tree.json",
        source_format="JSON",
        media_type="application/json",
    )
    recomputation_ref = _source(
        system_store,
        "bootstrap/recomputation.json",
        source_format="JSON",
        media_type="application/json",
    )
    source_generation_ref = _source(
        system_store,
        "bootstrap/source-generation.json",
        source_format="JSON",
        media_type="application/json",
    )
    bundle_refs = {
        "decision_source_bundle_ref": _bundle(
            system_store, "decision", "bootstrap_decision", decision_ref
        ),
        "exchange_calendar_bundle_ref": _bundle(system_store, "calendar", "calendar", calendar_ref),
        "implementation_bundle_ref": _bundle(
            system_store,
            "implementation",
            "implementation_tree_manifest",
            implementation_ref,
        ),
        "market_bundle_ref": _bundle(system_store, "market", "market", market_ref),
        "pit_universe_bundle_ref": _bundle(system_store, "pit", "pit", pit_ref),
        "recomputation_bundle_ref": _bundle(
            system_store, "recomputation", "recomputation", recomputation_ref
        ),
        "source_generation_bundle_ref": _bundle(
            system_store,
            "source-generation",
            "source_generation",
            source_generation_ref,
        ),
    }
    closure = factor_store.initialize_bootstrap(release_ref=release_ref, **bundle_refs)
    request = system_store.build_validation_run_request(
        release_manifest_ref=release_ref,
        factor_validator_manifest_ref=manifest_ref,
        intrinsic_receipt_ref=closure.intrinsic_receipt_ref,
    )
    result = system_store.run_validation(request["validation_request_ref"])
    status = factor_store.build_status(
        active_factor_set_ref=closure.active_set_ref,
        active_validation_receipt_ref=closure.intrinsic_receipt_ref,
        active_contextual_result_ref=result["contextual_result_ref"],
        active_validation_attestation_ref=result["validation_attestation_ref"],
    )

    assert validate_factor_status(status)["payload"]["readiness"] == "READY"
    assert result["contextual_result"]["payload"]["source_object_refs"] == sorted(
        [
            decision_ref,
            calendar_ref,
            implementation_ref,
            market_ref,
            pit_ref,
            recomputation_ref,
            source_generation_ref,
        ],
        key=lambda ref: tuple(
            ref[field]
            for field in (
                "kind",
                "contract_sha256",
                "artifact_id",
                "semantic_sha256",
                "byte_sha256",
            )
        ),
    )
    assert not (workspace / "results/system/_active.json").exists()


def test_mine_reuses_system_transaction_intent_after_pre_cas_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    source_root = tmp_path / "source"
    workspace.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    system_store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id="factor-store-prospective-source-root",
    )

    def fixed_clock() -> datetime:
        return datetime(2026, 8, 16, tzinfo=timezone.utc)

    factor_store = FactorValidationStore._for_testing(
        system_store=system_store,
        clock=fixed_clock,
    )
    _, manifest, manifest_ref = _release_and_manifest(
        system_store,
        factor_store,
        validation_profile=PROSPECTIVE_VALIDATION_PROFILE,
    )
    _write_parquet(
        source_root,
        "prospective/calendar.parquet",
        _calendar_rows(),
        role_schema("exchange_calendar"),
    )
    _write_parquet(
        source_root,
        "prospective/implementation.parquet",
        _implementation_rows(manifest),
        role_schema("implementation_manifest"),
    )
    calendar_ref = _source(
        system_store,
        "prospective/calendar.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    implementation_ref = _source(
        system_store,
        "prospective/implementation.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    monkeypatch.setattr(system_store_module, "_utc_now", lambda: STAMP)
    original_cas = system_store.compare_and_swap_candidate_state

    def fail_before_cas(*args: object, **kwargs: object) -> dict[str, Any]:
        raise SystemCASMismatch(EMPTY_POINTER_SHA256, "f" * 64)

    monkeypatch.setattr(system_store, "compare_and_swap_candidate_state", fail_before_cas)
    with pytest.raises(FactorGovernanceError, match="CANDIDATE_STATE_CONFLICT"):
        factor_store.mine(
            exchange_calendar_ref=calendar_ref,
            implementation_manifest_ref=implementation_ref,
            factor_validator_manifest_ref=manifest_ref,
            expected_composite_state_ref=None,
        )
    namespace = prospective_validation_namespace_id(
        exchange_calendar_ref=calendar_ref,
        implementation_manifest_ref=implementation_ref,
        factor_validator_manifest_ref=manifest_ref,
    )
    assert system_store.read_candidate_state(namespace) is None
    intents = list(workspace.rglob("intent.json"))
    assert len(intents) == 1

    def forbidden_clock() -> str:
        raise AssertionError("candidate retry sampled a second System timestamp")

    monkeypatch.setattr(system_store_module, "_utc_now", forbidden_clock)
    monkeypatch.setattr(system_store, "compare_and_swap_candidate_state", original_cas)
    composite = factor_store.mine(
        exchange_calendar_ref=calendar_ref,
        implementation_manifest_ref=implementation_ref,
        factor_validator_manifest_ref=manifest_ref,
        expected_composite_state_ref=None,
    )
    payload = composite["payload"]
    request_sha = operation_request_sha256(
        operation="PREREGISTER",
        expected_composite_state_ref=None,
        input_refs={
            "exchange_calendar": calendar_ref,
            "factor_validator_manifest": manifest_ref,
            "implementation_manifest": implementation_ref,
        },
    )
    transaction_id = custody_transaction_id(
        custody_namespace_id=payload["custody_namespace_id"],
        transaction_sequence=1,
        previous_composite_state_ref=None,
        operation_request_sha256_value=request_sha,
    )
    intent = system_store.read_candidate_transaction(
        payload["custody_namespace_id"], transaction_id
    )
    assert intent is not None
    assert intent["trusted_at"] == STAMP
    assert composite["created_at"] == STAMP
    assert payload["last_stored_at"] == STAMP
    assert system_store.read_candidate_state(payload["custody_namespace_id"]) is not None
    assert not (workspace / "results/system/_active.json").exists()


def test_first_signal_capture_is_one_atomic_selection_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    source_root = tmp_path / "source"
    workspace.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    system_store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id="factor-store-signal-source-root",
    )
    current_stamp = [STAMP]

    def factor_clock() -> datetime:
        return datetime.strptime(current_stamp[0], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )

    monkeypatch.setattr(system_store_module, "_utc_now", lambda: current_stamp[0])
    factor_store = FactorValidationStore._for_testing(
        system_store=system_store,
        clock=factor_clock,
    )
    _, manifest, manifest_ref = _release_and_manifest(
        system_store,
        factor_store,
        validation_profile=PROSPECTIVE_VALIDATION_PROFILE,
    )
    for relative, rows, role in (
        ("prospective/calendar.parquet", _calendar_rows(), "exchange_calendar"),
        (
            "prospective/implementation.parquet",
            _implementation_rows(manifest),
            "implementation_manifest",
        ),
        ("prospective/pit-000.parquet", _prospective_pit_rows(), "pit_universe"),
        (
            "prospective/market-000.parquet",
            _prospective_market_rows(),
            "market_history",
        ),
        (
            "prospective/weights-000.parquet",
            _prospective_weight_rows(manifest),
            "sparse_weights",
        ),
    ):
        _write_parquet(source_root, relative, rows, role_schema(role))
    calendar_ref = _source(
        system_store,
        "prospective/calendar.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    implementation_ref = _source(
        system_store,
        "prospective/implementation.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    pit_ref = _source(
        system_store,
        "prospective/pit-000.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    market_ref = _source(
        system_store,
        "prospective/market-000.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    weights_ref = _source(
        system_store,
        "prospective/weights-000.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    mined = factor_store.mine(
        exchange_calendar_ref=calendar_ref,
        implementation_manifest_ref=implementation_ref,
        factor_validator_manifest_ref=manifest_ref,
        expected_composite_state_ref=None,
    )
    current_stamp[0] = "2026-08-17T07:00:00Z"
    captured = factor_store.observe_signal(
        preregistration_ref=mined["payload"]["preregistration_ref"],
        selection_ref=None,
        pit_universe_ref=pit_ref,
        market_history_ref=market_ref,
        sparse_weights_ref=weights_ref,
        expected_composite_state_ref=system_store.put_object(mined),
    )
    payload = captured["payload"]
    assert payload["cycle_state"] == "OBSERVING"
    assert payload["transaction_sequence"] == 2
    assert payload["custody_record_count"] == 3
    assert payload["signal_capture_count"] == 1
    assert payload["resolved_signal_slot_count"] == 1
    assert payload["selection_ref"] is not None
    assert payload["signal_capture_head_ref"] is not None
    selection = system_store.get_object(payload["selection_ref"])
    preregistration = system_store.get_object(payload["preregistration_ref"])
    capture = system_store.get_object(payload["signal_capture_head_ref"])
    assert (
        validate_configuration_selection(
            selection,
            preregistration=preregistration,
        )
        == selection
    )
    assert (
        validate_signal_capture(
            capture,
            preregistration=preregistration,
            selection=selection,
        )
        == capture
    )
    replay = replay_custody_chain(
        system_store=system_store,
        final_composite=captured,
    )
    assert replay.transaction_count == 2
    assert len(replay.custody_record_refs) == 3
    assert len(replay.source_attestation_refs) == 2
    assert len(replay.stage_slots) == 1
    assert replay.stage_slots[0]["state"] == "CAPTURED"
    assert not (workspace / "results/system/_active.json").exists()


def test_first_matured_label_replays_capture_sources_and_interleaved_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    source_root = tmp_path / "source"
    workspace.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    system_store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id="factor-store-label-source-root",
    )
    current_stamp = [STAMP]

    def factor_clock() -> datetime:
        return datetime.strptime(current_stamp[0], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )

    monkeypatch.setattr(system_store_module, "_utc_now", lambda: current_stamp[0])
    factor_store = FactorValidationStore._for_testing(
        system_store=system_store,
        clock=factor_clock,
    )
    _, manifest, manifest_ref = _release_and_manifest(
        system_store,
        factor_store,
        validation_profile=PROSPECTIVE_VALIDATION_PROFILE,
    )
    calendar_rows = _calendar_rows()
    _write_parquet(
        source_root,
        "prospective/calendar.parquet",
        calendar_rows,
        role_schema("exchange_calendar"),
    )
    _write_parquet(
        source_root,
        "prospective/implementation.parquet",
        _implementation_rows(manifest),
        role_schema("implementation_manifest"),
    )
    calendar_ref = _source(
        system_store,
        "prospective/calendar.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    implementation_ref = _source(
        system_store,
        "prospective/implementation.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    composite = factor_store.mine(
        exchange_calendar_ref=calendar_ref,
        implementation_manifest_ref=implementation_ref,
        factor_validator_manifest_ref=manifest_ref,
        expected_composite_state_ref=None,
    )
    composite_ref = system_store.put_object(composite)
    capture_refs: list[dict[str, str]] = []
    selection_ref: dict[str, str] | None = None
    for ordinal in range(31):
        signal_session = calendar_rows[ordinal]["open_session"]
        current_stamp[0] = calendar_rows[ordinal]["closes_at_utc"].strftime("%Y-%m-%dT%H:%M:%SZ")
        source_rows = (
            (
                f"prospective/pit-{ordinal:03d}.parquet",
                _prospective_pit_rows(signal_session),
                "pit_universe",
            ),
            (
                f"prospective/market-{ordinal:03d}.parquet",
                _prospective_market_rows(signal_session),
                "market_history",
            ),
            (
                f"prospective/weights-{ordinal:03d}.parquet",
                _prospective_weight_rows(manifest, signal_session),
                "sparse_weights",
            ),
        )
        refs: dict[str, dict[str, str]] = {}
        for relative, rows, role in source_rows:
            _write_parquet(source_root, relative, rows, role_schema(role))
            refs[role] = _source(
                system_store,
                relative,
                source_format="PARQUET",
                media_type="application/vnd.apache.parquet",
            )
        composite = factor_store.observe_signal(
            preregistration_ref=composite["payload"]["preregistration_ref"],
            selection_ref=selection_ref,
            pit_universe_ref=refs["pit_universe"],
            market_history_ref=refs["market_history"],
            sparse_weights_ref=refs["sparse_weights"],
            expected_composite_state_ref=composite_ref,
        )
        composite_ref = system_store.put_object(composite)
        selection_ref = composite["payload"]["selection_ref"]
        capture_refs.append(composite["payload"]["signal_capture_head_ref"])

    assert selection_ref is not None
    label_relative = "prospective/labels-000.parquet"
    _write_parquet(
        source_root,
        label_relative,
        _prospective_label_rows(
            label_start_session=calendar_rows[1]["open_session"],
            label_end_session=calendar_rows[30]["open_session"],
        ),
        role_schema("matured_label_prices"),
    )
    label_ref = _source(
        system_store,
        label_relative,
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    observed = factor_store.observe_label(
        preregistration_ref=composite["payload"]["preregistration_ref"],
        selection_ref=selection_ref,
        signal_capture_ref=capture_refs[0],
        matured_label_prices_ref=label_ref,
        expected_composite_state_ref=composite_ref,
    )
    payload = observed["payload"]
    assert payload["cycle_state"] == "OBSERVING"
    assert payload["transaction_sequence"] == 33
    assert payload["custody_record_count"] == 34
    assert payload["signal_capture_count"] == 31
    assert payload["observation_count"] == 1
    assert payload["resolved_label_slot_count"] == 1
    preregistration = system_store.get_object(payload["preregistration_ref"])
    selection = system_store.get_object(selection_ref)
    capture = system_store.get_object(capture_refs[0])
    observation = system_store.get_object(payload["observation_head_ref"])
    assert (
        validate_observation(
            observation,
            preregistration=preregistration,
            selection=selection,
            signal_capture=capture,
        )
        == observation
    )
    assert all(
        row["complete_case_count"] == len(_prospective_symbols())
        and row["held_missing_label_count"] == 0
        and row["gross_labeled_return"] is not None
        for row in observation["payload"]["configuration_rows"]
    )
    replay = replay_custody_chain(system_store=system_store, final_composite=observed)
    assert replay.transaction_count == 33
    assert len(replay.custody_record_refs) == 34
    assert len(replay.source_attestation_refs) == 33
    assert len(replay.stage_slots) == 32
    assert replay.stage_slots[1]["stage"] == "LABEL"
    assert replay.stage_slots[1]["ordinal"] == 0
    assert not (workspace / "results/system/_active.json").exists()


def test_full_prospective_store_context_replays_1442_raw_sources_without_activation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    source_root = tmp_path / "source"
    workspace.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    system_store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id="factor-full-prospective-source-root",
    )
    current_stamp = [STAMP]

    def factor_clock() -> datetime:
        return datetime.strptime(current_stamp[0], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )

    monkeypatch.setattr(system_store_module, "_utc_now", lambda: current_stamp[0])
    factor_store = FactorValidationStore._for_testing(
        system_store=system_store,
        clock=factor_clock,
    )
    release_ref, manifest, manifest_ref = _release_and_manifest(
        system_store,
        factor_store,
        validation_profile=PROSPECTIVE_VALIDATION_PROFILE,
    )
    calendar_rows = _calendar_rows()
    _write_parquet(
        source_root,
        "prospective-full/calendar.parquet",
        calendar_rows,
        role_schema("exchange_calendar"),
    )
    _write_parquet(
        source_root,
        "prospective-full/implementation.parquet",
        _implementation_rows(manifest),
        role_schema("implementation_manifest"),
    )
    calendar_ref = _source(
        system_store,
        "prospective-full/calendar.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    implementation_ref = _source(
        system_store,
        "prospective-full/implementation.parquet",
        source_format="PARQUET",
        media_type="application/vnd.apache.parquet",
    )
    composite = factor_store.mine(
        exchange_calendar_ref=calendar_ref,
        implementation_manifest_ref=implementation_ref,
        factor_validator_manifest_ref=manifest_ref,
        expected_composite_state_ref=None,
    )
    composite_ref = system_store.put_object(composite)
    selection_ref: dict[str, str] | None = None
    capture_refs: list[dict[str, str]] = []
    surface = _full_prospective_surface()

    def publish_source(
        relative: str,
        rows: list[dict[str, Any]],
        role: str,
    ) -> dict[str, str]:
        _write_parquet(source_root, relative, rows, role_schema(role))
        return _source(
            system_store,
            relative,
            source_format="PARQUET",
            media_type="application/vnd.apache.parquet",
        )

    for session_ordinal in range(390):
        current_stamp[0] = calendar_rows[session_ordinal]["closes_at_utc"].strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        if session_ordinal < 360:
            signal_session = calendar_rows[session_ordinal]["open_session"]
            pit_ref = publish_source(
                f"prospective-full/pit-{session_ordinal:03d}.parquet",
                _full_pit_rows(surface, session_ordinal),
                "pit_universe",
            )
            market_ref = publish_source(
                f"prospective-full/market-{session_ordinal:03d}.parquet",
                _full_market_rows(surface, session_ordinal),
                "market_history",
            )
            weights_ref = publish_source(
                f"prospective-full/weights-{session_ordinal:03d}.parquet",
                _prospective_weight_rows(manifest, signal_session),
                "sparse_weights",
            )
            composite = factor_store.observe_signal(
                preregistration_ref=composite["payload"]["preregistration_ref"],
                selection_ref=selection_ref,
                pit_universe_ref=pit_ref,
                market_history_ref=market_ref,
                sparse_weights_ref=weights_ref,
                expected_composite_state_ref=composite_ref,
            )
            composite_ref = system_store.put_object(composite)
            selection_ref = composite["payload"]["selection_ref"]
            capture_refs.append(composite["payload"]["signal_capture_head_ref"])
        if session_ordinal >= 30:
            label_ordinal = session_ordinal - 30
            label_ref = publish_source(
                f"prospective-full/labels-{label_ordinal:03d}.parquet",
                _full_label_rows(surface, label_ordinal),
                "matured_label_prices",
            )
            assert selection_ref is not None
            composite = factor_store.observe_label(
                preregistration_ref=composite["payload"]["preregistration_ref"],
                selection_ref=selection_ref,
                signal_capture_ref=capture_refs[label_ordinal],
                matured_label_prices_ref=label_ref,
                expected_composite_state_ref=composite_ref,
            )
            composite_ref = system_store.put_object(composite)

    assert selection_ref is not None
    assert composite["payload"]["cycle_state"] == "OBSERVATIONS_MATURED"
    assert composite["payload"]["transaction_sequence"] == 721
    assert composite["payload"]["custody_record_count"] == 722
    assert composite["payload"]["signal_capture_count"] == 360
    assert composite["payload"]["observation_count"] == 360
    for action, expected_state, expected_sequence in (
        ("FINALIZE_EXECUTION", "EXECUTION_FINALIZED", 722),
        ("EVALUATE_PREREGISTRATION", "EVALUATED_ELIGIBLE", 723),
        ("BUILD_ADMITTED_SET", "ADMITTED", 724),
        ("BUILD_INTRINSIC_RECEIPT", "INTRINSIC_VALIDATED", 725),
    ):
        composite = factor_store.evaluate(
            action=action,
            preregistration_ref=composite["payload"]["preregistration_ref"],
            selection_ref=selection_ref,
            expected_composite_state_ref=composite_ref,
        )
        composite_ref = system_store.put_object(composite)
        assert composite["payload"]["cycle_state"] == expected_state
        assert composite["payload"]["transaction_sequence"] == expected_sequence

    replay = replay_custody_chain(system_store=system_store, final_composite=composite)
    assert replay.transaction_count == 725
    assert len(replay.custody_record_refs) == 726
    assert len(replay.source_attestation_refs) == 721
    assert len(replay.stage_slots) == 720
    evaluation = system_store.get_object(composite["payload"]["evaluation_ref"])
    evaluation_rows = {
        row["configuration_id"]: row for row in evaluation["payload"]["candidate_rows"]
    }
    trial = evaluation["payload"]["trial_statistics"]
    assert {
        "effective_trial_count": trial["effective_trial_count"],
        "trial_icir_complete": trial["trial_icir_complete"],
        "trial_icir_count": trial["trial_icir_count"],
        "trial_sharpe_std": trial["trial_sharpe_std"],
        "pbo": trial["pbo"],
        "pbo_complete": trial["pbo_complete"],
        "pbo_block_count": trial["pbo_block_count"],
        "pbo_split_count": trial["pbo_split_count"],
        "cpcv_path_count": trial["cpcv_path_count"],
    } == {
        "effective_trial_count": 2,
        "trial_icir_complete": True,
        "trial_icir_count": 2,
        "trial_sharpe_std": "1.121410836628",
        "pbo": "0.103174603175",
        "pbo_complete": True,
        "pbo_block_count": 10,
        "pbo_split_count": 252,
        "cpcv_path_count": 45,
    }
    golden_fields = (
        "valid_daily_rankic_sessions",
        "closed_calendar_month_end_observations",
        "disjoint_30_open_session_cohort_means",
        "maturity_passed",
        "mean_rank_ic",
        "mean_purged_oos_rank_ic",
        "shrunk_ic",
        "t_statistic",
        "t_p_value",
        "deflated_sharpe_ratio",
        "bh_q_value",
        "cpcv_path_count",
        "positive_path_ratio",
        "turnover",
        "cluster_representative",
        "admission_eligible",
        "blockers",
    )
    assert {key: evaluation_rows[BLEND_W80][key] for key in golden_fields} == {
        "valid_daily_rankic_sessions": 360,
        "closed_calendar_month_end_observations": 12,
        "disjoint_30_open_session_cohort_means": 12,
        "maturity_passed": True,
        "mean_rank_ic": "0.567238899312",
        "mean_purged_oos_rank_ic": "0.567238899312",
        "shrunk_ic": "0.464104553983",
        "t_statistic": "17.651243496798",
        "t_p_value": "0.000000002032",
        "deflated_sharpe_ratio": "1.000000000000",
        "bh_q_value": "0.000000002032",
        "cpcv_path_count": 45,
        "positive_path_ratio": "1.000000000000",
        "turnover": "0.700000000000",
        "cluster_representative": True,
        "admission_eligible": True,
        "blockers": [],
    }
    assert {key: evaluation_rows[LOW_DOLLAR_VOLUME][key] for key in golden_fields} == {
        "valid_daily_rankic_sessions": 360,
        "closed_calendar_month_end_observations": 12,
        "disjoint_30_open_session_cohort_means": 12,
        "maturity_passed": True,
        "mean_rank_ic": "0.642960704607",
        "mean_purged_oos_rank_ic": "0.642960704607",
        "shrunk_ic": "0.526058758315",
        "t_statistic": "22.314333431990",
        "t_p_value": "0.000000000165",
        "deflated_sharpe_ratio": "1.000000000000",
        "bh_q_value": "0.000000000165",
        "cpcv_path_count": 45,
        "positive_path_ratio": "1.000000000000",
        "turnover": "0.700000000000",
        "cluster_representative": True,
        "admission_eligible": True,
        "blockers": [],
    }
    assert [row["configuration_ids"] for row in evaluation["payload"]["redundancy_clusters"]] == [
        [BLEND_W80],
        [LOW_DOLLAR_VOLUME],
    ]
    admitted = system_store.get_object(composite["payload"]["admitted_set_ref"])
    assert {row["factor_id"]: row["weight"] for row in admitted["payload"]["factor_rows"]} == {
        BLEND_W80: "0.468715158620",
        LOW_DOLLAR_VOLUME: "0.531284841380",
    }

    current_stamp[0] = (calendar_rows[389]["closes_at_utc"] + timedelta(seconds=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    request = system_store.build_validation_run_request(
        release_manifest_ref=release_ref,
        factor_validator_manifest_ref=manifest_ref,
        intrinsic_receipt_ref=composite["payload"]["intrinsic_receipt_ref"],
        candidate_state_ref=composite_ref,
    )
    result = system_store.run_validation(request["validation_request_ref"])
    context_payload = result["contextual_result"]["payload"]
    assert context_payload["lane"] == "PROSPECTIVE"
    assert context_payload["composite_state_ref"] == composite_ref
    assert len(context_payload["evidence_refs"]) == 1_445
    assert len(context_payload["source_attestation_refs"]) == 721
    assert len(context_payload["source_object_refs"]) == 1_442
    assert len(context_payload["custody_record_refs"]) == 726
    status = factor_store.build_status(
        active_factor_set_ref=composite["payload"]["admitted_set_ref"],
        active_validation_receipt_ref=composite["payload"]["intrinsic_receipt_ref"],
        active_contextual_result_ref=result["contextual_result_ref"],
        active_validation_attestation_ref=result["validation_attestation_ref"],
        observed_composite_state_ref=composite_ref,
    )
    assert validate_factor_status(status)["payload"]["readiness"] == "READY"
    assert not (workspace / "results/system/_active.json").exists()


def test_compact_execution_projection_replays_entry_rebalance_exit_costs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    source_root = tmp_path / "source"
    workspace.mkdir(mode=0o700)
    source_root.mkdir(mode=0o700)
    store = SystemStore(
        workspace,
        source_root=source_root,
        source_root_id="factor-execution-projection-source-root",
    )
    factor_store = FactorValidationStore(system_store=store)
    selection = {
        "payload": {
            "selected_configurations": [
                {
                    "selected_configuration_id": LOW_DOLLAR_VOLUME,
                    "selected_factor_id": LOW_DOLLAR_VOLUME,
                },
                {
                    "selected_configuration_id": BLEND_W80,
                    "selected_factor_id": BLEND_W80,
                },
            ]
        }
    }
    captures = [
        {
            "payload": {
                "signal_session": f"session-{ordinal:03d}",
                "configuration_rows": [
                    {
                        "configuration_id": factor_id,
                        "portfolio_weights_sha256": hashlib.sha256(
                            f"{factor_id}-{ordinal}".encode()
                        ).hexdigest(),
                        "nonzero_weight_count": 1,
                    }
                    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80)
                ],
            }
        }
        for ordinal in range(360)
    ]
    observations = [
        {
            "payload": {
                "configuration_rows": [
                    {
                        "configuration_id": factor_id,
                        "gross_labeled_return": "0.001000000000",
                        "held_missing_label_count": 0,
                    }
                    for factor_id in (LOW_DOLLAR_VOLUME, BLEND_W80)
                ]
            }
        }
        for _ in range(360)
    ]

    def replay_signal(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {
            "weights_by_configuration": {
                LOW_DOLLAR_VOLUME: {"000001.SZ": Decimal("0.500000000000")},
                BLEND_W80: {"000002.SZ": Decimal("0.500000000000")},
            }
        }

    def replay_observation(**kwargs: Any) -> dict[str, Any]:
        del kwargs
        return {}

    monkeypatch.setattr(factor_store, "_replay_signal_capture_projection", replay_signal)
    monkeypatch.setattr(factor_store, "_replay_observation_projection", replay_observation)
    rows, blockers = factor_store._execution_configuration_rows(
        preregistration={},
        selection=selection,
        captures=captures,
        observations=observations,
        manifest={},
    )

    assert blockers == []
    assert len(rows) == 2
    for row in rows:
        assert row["session_summary_count"] == 360
        assert row["initial_entry_turnover"] == "0.500000000000"
        assert row["rebalance_turnover"] == "0.000000000000"
        assert row["terminal_exit_turnover"] == "0.500000000000"
        assert row["total_turnover"] == "1.000000000000"
        assert row["annualized_turnover"] == "0.700000000000"
        assert row["total_estimated_cost"] == "0.000050000000"
        assert row["gross_labeled_return_count"] == 360
        assert row["gross_labeled_return_sum"] == "0.360000000000"
        assert row["net_labeled_return_sum"] == "0.359950000000"
