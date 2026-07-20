"""Deterministic data binding and five-factor scoring for operator advisory."""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.factors.price_volume import (
    compute_price_volume_factor,
    prepare_price_volume_frames,
)
from quant_investor.market.market_data_reader import MarketDataReader
from quant_investor.market.pit_universe import filter_symbols_by_pit_status
from quant_investor.v16.operator_advisory.contracts import (
    BRANCHES,
    BRANCH_SHARES,
    FACTOR_BUNDLE_SCHEMA,
    INPUT_MANIFEST_PATH,
    INPUT_MANIFEST_SCHEMA,
    PREPARED_EVIDENCE_SCHEMA,
    REPO_ROOT,
    AdvisoryError,
    centered_average_rank,
    file_sha256,
    read_json,
)

PRICE_FACTOR_NAMES = (
    "pv_blend_volstab19x2_mom90_amihud5_w80",
    "pv_short_reversal_25d",
    "pv_downside_volatility_15d",
)
FORMULA_FACTOR_NAME = "formula_cash_growth_lowlev_w65"
QUALITY_FACTOR_NAME = "fund_quality_cash_combo"
ORDERED_FACTOR_NAMES = PRICE_FACTOR_NAMES + (
    FORMULA_FACTOR_NAME,
    QUALITY_FACTOR_NAME,
)
LOOKBACK_CALENDAR_DAYS = 180
MIN_COMMON_DOMAIN = 300

_SEMANTIC_FIELDS = {
    "candidate_catalog": "semantic_sha256",
    "screening_evidence": "semantic_sha256",
    "primitive_ontology": "semantic_sha256",
    "pre_admission_report": "report_sha256",
}


def load_input_manifest() -> dict[str, Any]:
    manifest = read_json(INPUT_MANIFEST_PATH, max_bytes=512 * 1024)
    if manifest.get("schema_version") != INPUT_MANIFEST_SCHEMA:
        raise AdvisoryError("operator input manifest schema mismatch")
    _validate_manifest_policy(manifest)
    return manifest


def _repo_file(raw_path: Any, *, label: str) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise AdvisoryError(f"{label} path missing")
    declared = Path(text)
    if declared.is_absolute() or ".." in declared.parts:
        raise AdvisoryError(f"{label} path must be repository-relative")
    candidate = REPO_ROOT / declared
    current = REPO_ROOT
    for part in declared.parts:
        current = current / part
        if current.is_symlink():
            raise AdvisoryError(f"{label} symlink rejected")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(REPO_ROOT)
    except (OSError, ValueError) as exc:
        raise AdvisoryError(f"{label} unavailable") from exc
    if not resolved.is_file():
        raise AdvisoryError(f"{label} must be a regular file")
    return resolved


def _validate_manifest_policy(manifest: Mapping[str, Any]) -> None:
    factors = manifest.get("factors")
    if not isinstance(factors, list) or [item.get("name") for item in factors] != list(
        ORDERED_FACTOR_NAMES
    ):
        raise AdvisoryError("operator factor set/order mismatch")
    families = [str(item.get("family") or "") for item in factors]
    if len(set(families)) != 5:
        raise AdvisoryError("operator factors must span exactly five families")
    shares = [float(item.get("advisory_blend_share", -1.0)) for item in factors]
    if shares != [0.2] * 5 or not math.isclose(sum(shares), 1.0):
        raise AdvisoryError("operator factor blend must be five equal shares")
    if any(item.get("source_state") != "diagnostic_nonproduction" for item in factors):
        raise AdvisoryError("operator factor source state mismatch")

    branch_policy = manifest.get("branch_policy")
    if not isinstance(branch_policy, Mapping):
        raise AdvisoryError("operator branch policy missing")
    if tuple(branch_policy.get("ordered_branches") or ()) != BRANCHES:
        raise AdvisoryError("operator branch order mismatch")
    if branch_policy.get("branch_shares") != BRANCH_SHARES:
        raise AdvisoryError("operator branch shares mismatch")
    if branch_policy.get("retrieval_scoring_allowed") is not False:
        raise AdvisoryError("retrieval scoring must remain disabled")
    if branch_policy.get("risk_authority") != "advisory_only":
        raise AdvisoryError("risk authority must remain advisory-only")

    provider = manifest.get("provider_policy")
    if not isinstance(provider, Mapping) or provider != {
        "provider": "openai_responses_no_tools",
        "endpoint": "https://api.openai.com:443/v1/responses",
        "model": "gpt-5.4-2026-03-05",
        "store": False,
        "tools": [],
    }:
        raise AdvisoryError("operator provider policy mismatch")


def _artifact_payload(path: Path) -> dict[str, Any]:
    return read_json(path, max_bytes=16 * 1024 * 1024)


def _validate_governance_artifacts(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_sources = manifest.get("source_artifacts")
    if not isinstance(raw_sources, list) or len(raw_sources) != 7:
        raise AdvisoryError("operator source artifact inventory mismatch")
    bindings: list[dict[str, Any]] = []
    payloads: dict[str, dict[str, Any]] = {}
    for source in raw_sources:
        if not isinstance(source, Mapping):
            raise AdvisoryError("operator source artifact entry invalid")
        source_id = str(source.get("id") or "")
        path = _repo_file(source.get("path"), label=f"source artifact {source_id}")
        actual_sha = file_sha256(path)
        expected_sha = str(source.get("byte_sha256") or "")
        if actual_sha != expected_sha:
            raise AdvisoryError(f"source artifact byte hash mismatch: {source_id}")
        payload = _artifact_payload(path)
        payloads[source_id] = payload
        semantic_field = _SEMANTIC_FIELDS.get(source_id)
        expected_semantic = str(source.get("semantic_sha256") or "")
        if semantic_field:
            if payload.get(semantic_field) != expected_semantic:
                raise AdvisoryError(f"source artifact semantic hash mismatch: {source_id}")
        elif expected_semantic:
            raise AdvisoryError(f"unexpected semantic hash binding: {source_id}")
        bindings.append(
            {
                "id": source_id,
                "path": str(path.relative_to(REPO_ROOT)),
                "byte_sha256": actual_sha,
                "semantic_sha256": expected_semantic,
            }
        )

    expected_ids = {
        "candidate_queue",
        "candidate_catalog",
        "screening_evidence",
        "primitive_ontology",
        "market_data_inventory",
        "run_config",
        "pre_admission_report",
    }
    if set(payloads) != expected_ids:
        raise AdvisoryError("operator source artifact ids mismatch")

    queue = payloads["candidate_queue"]
    queued_names = [item.get("name") for item in queue.get("candidates", [])]
    if queued_names != list(ORDERED_FACTOR_NAMES):
        raise AdvisoryError("diversified candidate queue changed")
    if (
        queue.get("threshold_pass") is not True
        or queue.get("formal_admission_evidence") is not False
    ):
        raise AdvisoryError("diversified candidate queue state mismatch")

    catalog_rows = {
        str(item.get("name") or ""): item
        for item in payloads["candidate_catalog"].get("candidates", [])
        if isinstance(item, Mapping)
    }
    for factor in manifest["factors"]:
        name = str(factor["name"])
        row = catalog_rows.get(name)
        if row is None:
            raise AdvisoryError(f"operator factor missing from catalog: {name}")
        for key in (
            "name",
            "family",
            "slot",
            "implementation",
            "params",
            "expression",
            "direction",
            "definition_sha256",
        ):
            if row.get(key) != factor.get(key):
                raise AdvisoryError(f"operator factor descriptor mismatch: {name}:{key}")

    screening_rows = {
        str(item.get("name") or ""): item
        for item in payloads["screening_evidence"].get("rows", [])
        if isinstance(item, Mapping)
    }
    for name in ORDERED_FACTOR_NAMES:
        row = screening_rows.get(name)
        if not row or row.get("bh_pass") is not True:
            raise AdvisoryError(f"operator factor lost BH screening pass: {name}")

    source_notes = manifest.get("source_notes")
    if not isinstance(source_notes, Mapping) or source_notes.get("expected_absent") is not True:
        raise AdvisoryError("source notes absence policy mismatch")
    notes_path = REPO_ROOT / str(source_notes.get("path") or "")
    if notes_path.exists() or notes_path.is_symlink():
        raise AdvisoryError("unexpected source_notes artifact present")
    return bindings


def _validate_market_binding(
    manifest: Mapping[str, Any], reader: MarketDataReader
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = manifest.get("selection_snapshot")
    scoring = manifest.get("scoring_snapshot")
    if not isinstance(selection, Mapping) or not isinstance(scoring, Mapping):
        raise AdvisoryError("market snapshot bindings missing")
    pointer = _repo_file(scoring.get("market_pointer_path"), label="market pointer")
    pointer_sha = file_sha256(pointer)
    if pointer_sha != scoring.get("market_pointer_sha256") or pointer_sha != selection.get(
        "market_pointer_sha256"
    ):
        raise AdvisoryError("active market pointer hash mismatch")
    snapshot = reader.snapshot()
    if snapshot.get("healthy") is not True or snapshot.get("fallback_used") is not False:
        raise AdvisoryError("strict canonical market snapshot is not healthy")
    if snapshot.get("snapshot_id") != scoring.get("snapshot_id") or snapshot.get(
        "latest_complete_trade_date"
    ) != scoring.get("trade_date"):
        raise AdvisoryError("active market snapshot identity mismatch")

    pit = reader.coverage_bound_pit(refresh=True)
    if pit.get("status") != "passed":
        raise AdvisoryError(
            "PIT generation binding failed: "
            + ";".join(str(value) for value in pit.get("blockers", []))
        )
    expected_pit = {
        "generation_id": selection.get("pit_generation_id"),
        "generation_manifest_sha256": selection.get("pit_generation_manifest_sha256"),
        "canonical_sha256": selection.get("pit_membership_sha256"),
    }
    for key, expected in expected_pit.items():
        if pit.get(key) != expected:
            raise AdvisoryError(f"PIT generation identity mismatch: {key}")
    return snapshot, pit


def _validate_fundamental_binding(
    manifest: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = manifest.get("fundamental_source")
    if not isinstance(source, Mapping):
        raise AdvisoryError("fundamental source binding missing")
    pointer = _repo_file(source.get("pointer_path"), label="fundamental pointer")
    generation_manifest = _repo_file(
        source.get("generation_manifest_path"), label="fundamental manifest"
    )
    daily = _repo_file(source.get("daily_path"), label="fundamental daily")
    for label, path, expected in (
        ("pointer", pointer, source.get("pointer_sha256")),
        ("manifest", generation_manifest, source.get("generation_manifest_sha256")),
        ("daily", daily, source.get("daily_sha256")),
    ):
        if file_sha256(path) != expected:
            raise AdvisoryError(f"fundamental {label} hash mismatch")
    pointer_payload = _artifact_payload(pointer)
    generation_id = source.get("generation_id")
    if pointer_payload.get("generation_id") != generation_id:
        raise AdvisoryError("fundamental pointer generation mismatch")
    if pointer_payload.get("status") != "OK":
        raise AdvisoryError("fundamental pointer is not healthy")
    declared_manifest = (pointer.parent / str(pointer_payload.get("manifest_path") or "")).resolve()
    declared_daily = (
        pointer.parent / str((pointer_payload.get("tables") or {}).get("fundamental_daily") or "")
    ).resolve()
    if declared_manifest != generation_manifest or declared_daily != daily:
        raise AdvisoryError("fundamental pointer path binding mismatch")

    try:
        import pyarrow.dataset as ds

        target = datetime.strptime(str(source.get("as_of")), "%Y%m%d")
        table = ds.dataset(str(daily), format="parquet").to_table(
            columns=[
                "ts_code",
                "trade_date",
                "sector",
                "fin_roe",
                "fin_ocf_to_profit",
                "fin_debt_to_assets",
            ],
            filter=ds.field("trade_date") == target,
        )
        frame = table.to_pandas()
    except Exception as exc:
        raise AdvisoryError("fundamental exact-date read failed") from exc
    if frame.empty or frame["ts_code"].astype(str).duplicated().any():
        raise AdvisoryError("fundamental exact-date cross section invalid")
    frame = frame.set_index(frame["ts_code"].astype(str).str.upper(), drop=True)
    return frame, {
        "generation_id": generation_id,
        "pointer_sha256": source.get("pointer_sha256"),
        "generation_manifest_sha256": source.get("generation_manifest_sha256"),
        "daily_sha256": source.get("daily_sha256"),
        "as_of": source.get("as_of"),
    }


def _validate_macro_binding(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source = manifest.get("macro_source")
    if not isinstance(source, Mapping):
        raise AdvisoryError("macro source binding missing")
    generation_manifest = _repo_file(source.get("generation_manifest_path"), label="macro manifest")
    controls = _repo_file(source.get("controls_path"), label="macro controls")
    if file_sha256(generation_manifest) != source.get("generation_manifest_sha256"):
        raise AdvisoryError("macro manifest hash mismatch")
    if file_sha256(controls) != source.get("controls_sha256"):
        raise AdvisoryError("macro controls hash mismatch")
    manifest_payload = _artifact_payload(generation_manifest)
    controls_payload = _artifact_payload(controls)
    if manifest_payload.get("generation_id") != source.get("generation_id"):
        raise AdvisoryError("macro generation identity mismatch")
    if controls_payload.get("schema_version") != source.get("schema_version"):
        raise AdvisoryError("macro controls schema mismatch")
    if controls_payload.get("semantic_sha256") != source.get("semantic_sha256"):
        raise AdvisoryError("macro controls semantic hash mismatch")
    if controls_payload.get("snapshot_as_of") != source.get("as_of"):
        raise AdvisoryError("macro logical date mismatch")
    score = controls_payload.get(str(source.get("score_field")))
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
    ):
        raise AdvisoryError("macro score invalid")
    return controls_payload, {
        "generation_id": source.get("generation_id"),
        "generation_manifest_sha256": source.get("generation_manifest_sha256"),
        "controls_sha256": source.get("controls_sha256"),
        "semantic_sha256": source.get("semantic_sha256"),
        "as_of": source.get("as_of"),
    }


def _load_price_frames(
    reader: MarketDataReader,
    *,
    symbols: list[str],
    as_of: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=LOOKBACK_CALENDAR_DAYS)).strftime("%Y%m%d")
    results = reader.read_symbol_frames(
        symbols,
        universe_key="full_a",
        category="full_a",
        start_date=start,
        end_date=as_of,
        columns=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "vol",
            "amount",
            "adj_close",
        ],
    )
    frames: dict[str, pd.DataFrame] = {}
    latest_symbols: set[str] = set()
    issue_count = 0
    for symbol in symbols:
        result = results.get(symbol)
        if result is None:
            issue_count += 1
            continue
        if result.issues:
            issue_count += 1
        frame = result.frame
        if frame is None or frame.empty or "trade_date" not in frame.columns:
            continue
        working = frame.copy()
        working["trade_date"] = (
            working["trade_date"].astype(str).str.replace("-", "", regex=False).str[:8]
        )
        working = working.sort_values("trade_date", kind="stable").reset_index(drop=True)
        if str(working["trade_date"].iloc[-1]) == as_of:
            latest_symbols.add(symbol)
            frames[symbol] = working
    if len(frames) < MIN_COMMON_DOMAIN:
        raise AdvisoryError("insufficient exact-date strict price frames")
    return frames, {
        "requested_symbol_count": len(symbols),
        "exact_date_frame_count": len(frames),
        "read_issue_count": issue_count,
        "start_date": start,
        "end_date": as_of,
        "csv_fallback_used": False,
        "latest_symbol_count": len(latest_symbols),
    }


def _finite_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").replace(
        [float("inf"), float("-inf")], float("nan")
    )


def compute_fundamental_factor_signals(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Reproduce the pinned mining semantics on one cross section."""

    required = {"fin_roe", "fin_ocf_to_profit", "fin_debt_to_assets"}
    if not required.issubset(frame.columns):
        raise AdvisoryError("fundamental factor inputs missing")
    ocf_rank = frame["fin_ocf_to_profit"].rank(pct=True)
    low_debt_rank = (-frame["fin_debt_to_assets"]).rank(pct=True)
    roe_rank = frame["fin_roe"].rank(pct=True)
    return {
        FORMULA_FACTOR_NAME: _finite_series(ocf_rank.mul(0.65).add(low_debt_rank.mul(0.35))),
        QUALITY_FACTOR_NAME: _finite_series(roe_rank.add(ocf_rank)),
    }


def build_deterministic_inputs(
    *,
    max_candidates: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 1 <= int(max_candidates) <= 50:
        raise AdvisoryError("max_candidates must be within 1..50")
    manifest = load_input_manifest()
    governance_bindings = _validate_governance_artifacts(manifest)
    reader = MarketDataReader(market="CN", data_root=REPO_ROOT / "data", mode_policy="strict")
    snapshot, pit = _validate_market_binding(manifest, reader)
    records = pit.get("records")
    if not isinstance(records, Mapping) or not records:
        raise AdvisoryError("PIT records missing")
    serving_symbols = reader.list_symbols("full_a")
    pit_filter = filter_symbols_by_pit_status(
        serving_symbols,
        as_of=str(manifest["scoring_snapshot"]["trade_date"]),
        records=records,
        required=True,
    )
    symbols = pit_filter.symbols
    if len(symbols) < MIN_COMMON_DOMAIN:
        raise AdvisoryError("required PIT universe coverage failed")

    price_frames, price_diagnostics = _load_price_frames(
        reader,
        symbols=symbols,
        as_of=str(manifest["scoring_snapshot"]["trade_date"]),
    )
    fundamental, fundamental_binding = _validate_fundamental_binding(manifest)
    macro, macro_binding = _validate_macro_binding(manifest)

    prepared = prepare_price_volume_frames(price_frames)
    factor_cache: dict[str, Any] = {}
    raw_factors: dict[str, pd.Series] = {
        name: _finite_series(
            compute_price_volume_factor(
                name,
                price_frames,
                prepared_frames=prepared,
                factor_cache=factor_cache,
            )
        )
        for name in PRICE_FACTOR_NAMES
    }

    raw_factors.update(compute_fundamental_factor_signals(fundamental))

    common = set(price_frames) & set(fundamental.index)
    for name in ORDERED_FACTOR_NAMES:
        common &= set(raw_factors[name].dropna().index.astype(str))
    common_symbols = sorted(common)
    if len(common_symbols) < MIN_COMMON_DOMAIN:
        raise AdvisoryError(
            f"five-factor common domain below minimum: {len(common_symbols)}<{MIN_COMMON_DOMAIN}"
        )

    factor_ranks: dict[str, pd.Series] = {}
    for factor in manifest["factors"]:
        name = str(factor["name"])
        direction = float(factor["direction"])
        factor_ranks[name] = centered_average_rank(
            raw_factors[name].reindex(common_symbols).mul(direction)
        )
    quant_raw = sum(
        factor_ranks[str(factor["name"])].mul(float(factor["advisory_blend_share"]))
        for factor in manifest["factors"]
    )
    fundamental_raw = pd.concat(
        [
            centered_average_rank(fundamental["fin_roe"].reindex(common_symbols)),
            centered_average_rank(fundamental["fin_ocf_to_profit"].reindex(common_symbols)),
            centered_average_rank(-fundamental["fin_debt_to_assets"].reindex(common_symbols)),
        ],
        axis=1,
    ).mean(axis=1)
    macro_raw = float(macro[str(manifest["macro_source"]["score_field"])])

    rows: list[dict[str, Any]] = []
    for symbol in common_symbols:
        record = records[symbol]
        rows.append(
            {
                "symbol": symbol,
                "name": str(getattr(record, "name", "") or "UNKNOWN_NAME"),
                "industry": str(getattr(record, "industry", "") or "UNKNOWN_INDUSTRY"),
                "factor_raw": {
                    name: float(raw_factors[name].loc[symbol]) for name in ORDERED_FACTOR_NAMES
                },
                "factor_rank": {
                    name: float(factor_ranks[name].loc[symbol]) for name in ORDERED_FACTOR_NAMES
                },
                "quant_raw": float(quant_raw.loc[symbol]),
                "fundamental_raw": float(fundamental_raw.loc[symbol]),
                "macro_raw": macro_raw,
                "fundamental_metrics": {
                    "fin_roe": float(fundamental.loc[symbol, "fin_roe"]),
                    "fin_ocf_to_profit": float(fundamental.loc[symbol, "fin_ocf_to_profit"]),
                    "fin_debt_to_assets": float(fundamental.loc[symbol, "fin_debt_to_assets"]),
                },
            }
        )

    source_bindings = {
        "input_manifest_sha256": file_sha256(INPUT_MANIFEST_PATH),
        "market_pointer_sha256": manifest["scoring_snapshot"]["market_pointer_sha256"],
        "snapshot_id": snapshot["snapshot_id"],
        "trade_date": manifest["scoring_snapshot"]["trade_date"],
        "pit_generation_id": pit["generation_id"],
        "pit_membership_sha256": pit["canonical_sha256"],
        "governance_artifacts": governance_bindings,
        "fundamental": fundamental_binding,
        "macro": macro_binding,
    }
    factor_bundle = {
        "schema_version": FACTOR_BUNDLE_SCHEMA,
        "market": "CN",
        "source_run_id": manifest["source_run_id"],
        "source_bindings": source_bindings,
        "factors": manifest["factors"],
        "factor_family_count": 5,
        "factor_blend_share_sum": 1.0,
        "common_domain_count": len(rows),
        "universe_diagnostics": {
            **price_diagnostics,
            "serving_symbol_count": len(serving_symbols),
            "pit_eligible_symbol_count": len(symbols),
            "pit_excluded_symbol_count": len(pit_filter.quarantine_symbols),
            "pit_missing_record_excluded_count": int(pit_filter.metadata.get("missing_count", 0)),
            "fundamental_exact_date_row_count": len(fundamental),
        },
        "rows": rows,
    }

    top_rows = sorted(rows, key=lambda row: (-float(row["quant_raw"]), row["symbol"]))[
        : int(max_candidates)
    ]
    items: list[dict[str, Any]] = []
    for row in top_rows:
        symbol = row["symbol"]
        facts: list[dict[str, Any]] = []
        for name in ORDERED_FACTOR_NAMES:
            facts.append(
                {
                    "id": f"{symbol}:quant:{name}",
                    "branch": "quant",
                    "metric": name,
                    "value": row["factor_rank"][name],
                }
            )
        for metric, value in row["fundamental_metrics"].items():
            facts.append(
                {
                    "id": f"{symbol}:fundamental:{metric}",
                    "branch": "fundamental",
                    "metric": metric,
                    "value": value,
                }
            )
        facts.extend(
            [
                {
                    "id": f"{symbol}:macro:macro_score",
                    "branch": "macro",
                    "metric": "macro_score",
                    "value": macro_raw,
                },
                {
                    "id": f"{symbol}:macro:policy_signal",
                    "branch": "macro",
                    "metric": "policy_signal",
                    "value": str(macro.get("policy_signal") or "neutral"),
                },
                {
                    "id": f"{symbol}:macro:volatility_percentile",
                    "branch": "macro",
                    "metric": "volatility_percentile",
                    "value": float(macro.get("volatility_percentile", 0.0)),
                },
            ]
        )
        items.append(
            {
                "symbol": symbol,
                "name": row["name"],
                "industry": row["industry"],
                "quant_raw": row["quant_raw"],
                "fundamental_raw": row["fundamental_raw"],
                "macro_raw": row["macro_raw"],
                "fact_ids": [fact["id"] for fact in facts],
                "facts": facts,
            }
        )
    evidence = {
        "schema_version": PREPARED_EVIDENCE_SCHEMA,
        "market": "CN",
        "source_bindings": source_bindings,
        "branch_policy": manifest["branch_policy"],
        "sealed_symbol_count": len(items),
        "items": items,
        "authority": {
            "production_authority": False,
            "new_risk_authorized": False,
            "broker_enabled": False,
        },
    }

    # Reopen every binding after the expensive computation so source drift fails closed.
    _validate_governance_artifacts(manifest)
    _validate_market_binding(manifest, reader)
    _validate_fundamental_binding(manifest)
    _validate_macro_binding(manifest)
    return factor_bundle, evidence


__all__ = [
    "FORMULA_FACTOR_NAME",
    "MIN_COMMON_DOMAIN",
    "ORDERED_FACTOR_NAMES",
    "PRICE_FACTOR_NAMES",
    "QUALITY_FACTOR_NAME",
    "build_deterministic_inputs",
    "compute_fundamental_factor_signals",
    "load_input_manifest",
]
