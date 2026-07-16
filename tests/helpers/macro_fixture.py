from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from quant_investor.market import macro_mart
from quant_investor.macro.contracts import (
    MacroObservation,
    canonical_hash,
    parse_timestamp,
    published_cutoff,
)
from quant_investor.macro.local_market_observations import (
    LOCAL_MARKET_BREADTH_EVIDENCE_SCHEMA,
)
from quant_investor.macro.official_web_compiler import (
    NBS_NATIONAL_ECONOMY_PARSER,
    NBS_OFFICIAL_PMI_PARSER,
    NBS_QUARTERLY_GDP_PARSER,
    PARSER_CONTRACT_SHA256,
    PBC_MONEY_STOCK_PARSER,
)
from quant_investor.macro.production_observation_bundle import (
    PRODUCTION_OBSERVATION_BUNDLE_SCHEMA,
)
from quant_investor.macro.snapshot import build_macro_snapshot
from quant_investor.macro.store import pointer_sha256, publish_observations
from quant_investor.macro.v15_controls import (
    V15_MACRO_CONTROL_SCHEMA_VERSION,
    build_v15_macro_controls,
)


_POINTER_SHA = "1" * 64
_MARKET_INPUT_FILES = [
    {
        "path": "year=2024/month=05/part.parquet",
        "size_bytes": 1,
        "sha256": "2" * 64,
    }
]
_MARKET_FILES_SHA = macro_mart._canonical_json_sha256(
    {"files": _MARKET_INPUT_FILES}
)


def make_v15_controls(
    *,
    macro_score: float = 0.2,
    liquidity_score: float = 0.4,
    volatility_percentile: float = 45.0,
    policy_signal: str = "neutral",
) -> dict[str, Any]:
    """Build a compact, hash-bound DAG fixture for the v15 control contract."""

    controls: dict[str, Any] = {
        "schema_version": V15_MACRO_CONTROL_SCHEMA_VERSION,
        "production_control_projection": True,
        "macro_score": float(macro_score),
        "liquidity_score": float(liquidity_score),
        "volatility_percentile": float(volatility_percentile),
        "policy_signal": str(policy_signal),
    }
    controls["semantic_sha256"] = canonical_hash(controls)
    return controls


def write_ready_macro_observations(
    root: Path,
    *,
    as_of: str,
    run_id: str = "macro-observations-ready",
    decision_cutoff_at: str | None = None,
) -> str:
    """Publish a strict synthetic instance of the production v2 chain."""

    target = pd.Timestamp(str(as_of)).normalize()
    logical_as_of = target.date().isoformat()
    decision_cutoff = (
        published_cutoff(logical_as_of)
        if decision_cutoff_at is None
        else parse_timestamp(
            decision_cutoff_at,
            field_name="decision_cutoff_at",
        )
    )
    official_manifest_sha256 = "1" * 64
    evidence_bytes: dict[str, bytes] = {}
    evidence_metadata: dict[str, dict[str, Any]] = {}
    observation_evidence: dict[str, list[str]] = {}
    observations: list[MacroObservation] = []

    def add_evidence(
        label: str,
        metadata: Mapping[str, Any],
    ) -> str:
        body = json.dumps(
            {"fixture_evidence_role": label},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hashlib.sha256(body).hexdigest()
        item_metadata = {**dict(metadata), "size_bytes": len(body)}
        previous = evidence_bytes.get(digest)
        if previous is not None and (
            previous != body or evidence_metadata[digest] != item_metadata
        ):
            raise AssertionError("fixture_evidence_digest_collision")
        evidence_bytes[digest] = body
        evidence_metadata[digest] = item_metadata
        return digest

    def official_page(
        *,
        page_id: str,
        parser_id: str,
        source_system: str,
        period: str,
        release_at: str,
        source_record_id: str,
        source_url: str,
        support_only: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        metadata = {
            "extension": ".html",
            "evidence_kind": "official_web_response_entity",
            "page_id": page_id,
            "parser_id": parser_id,
            "parser_contract_sha256": PARSER_CONTRACT_SHA256[parser_id],
            "source_system": source_system,
            "source_url": source_url,
            "source_record_id": source_record_id,
            "period": period,
            "release_at": release_at,
            "official_bundle_manifest_sha256": official_manifest_sha256,
            "support_only": support_only,
        }
        return add_evidence(f"official:{page_id}", metadata), metadata

    def add_observation(
        *,
        indicator_id: str,
        period_end: pd.Timestamp,
        release_at: str,
        source_system: str,
        source_record_id: str,
        source_url: str,
        evidence_digests: list[str],
        frequency: str = "monthly",
        unit: str = "%",
        dimension_type: str = "national",
        value: float = 1.0,
    ) -> MacroObservation:
        observation = MacroObservation.from_mapping(
            {
                "indicator_id": indicator_id,
                "dimension_type": dimension_type,
                "industry_chain": "",
                "period_end": period_end.date().isoformat(),
                "release_at": release_at,
                "available_at": release_at,
                "vintage_id": "initial",
                "value": value,
                "unit": unit,
                "frequency": frequency,
                "source_system": source_system,
                "source_record_id": source_record_id,
                "source_url": source_url,
                "fetched_at": decision_cutoff.isoformat(),
                "quality_status": "pass",
            }
        )
        observations.append(observation)
        observation_evidence[observation.content_hash] = sorted(
            set(evidence_digests)
        )
        return observation

    latest_month = pd.offsets.MonthEnd().rollback(target)
    monthly_periods = [
        latest_month - pd.offsets.MonthEnd(offset) for offset in (2, 1, 0)
    ]
    economy_ids = (
        "cn.industrial_value_added_yoy",
        "cn.retail_sales_yoy",
        "cn.fixed_asset_investment_yoy",
        "cn.property_investment_yoy",
        "cn.exports_yoy",
        "cn.imports_yoy",
        "cn.cpi_yoy",
        "cn.ppi_yoy",
    )
    economy_pages: dict[str, tuple[str, dict[str, Any]]] = {}
    for period_end in monthly_periods:
        period = period_end.strftime("%Y%m")
        release = (period_end + pd.Timedelta(days=1, hours=1)).tz_localize(
            "UTC"
        ).isoformat()
        record_id = f"t{period_end + pd.Timedelta(days=1):%Y%m%d}_1"
        source_url = (
            "https://www.stats.gov.cn/sj/zxfb/"
            f"{period}/{record_id}.html"
        )
        page = official_page(
            page_id=f"economy-{period}",
            parser_id=NBS_NATIONAL_ECONOMY_PARSER,
            source_system="nbs_official",
            period=period,
            release_at=release,
            source_record_id=record_id,
            source_url=source_url,
        )
        economy_pages[period] = page
        for indicator_id in economy_ids:
            add_observation(
                indicator_id=indicator_id,
                period_end=period_end,
                release_at=release,
                source_system="nbs_official",
                source_record_id=record_id,
                source_url=source_url,
                evidence_digests=[page[0]],
            )

        pmi_record = f"t{period_end + pd.Timedelta(days=1):%Y%m%d}_9"
        pmi_url = (
            "https://www.stats.gov.cn/sj/zxfb/"
            f"{period}/{pmi_record}.html"
        )
        pmi_digest, _pmi_metadata = official_page(
            page_id=f"pmi-{period}",
            parser_id=NBS_OFFICIAL_PMI_PARSER,
            source_system="nbs_official",
            period=period,
            release_at=release,
            source_record_id=pmi_record,
            source_url=pmi_url,
        )
        add_observation(
            indicator_id="cn.pmi_manufacturing",
            period_end=period_end,
            release_at=release,
            source_system="nbs_official",
            source_record_id=pmi_record,
            source_url=pmi_url,
            evidence_digests=[pmi_digest],
            unit="index",
        )

    support_period = monthly_periods[0].strftime("%Y%m")
    support_digest, _support_metadata = official_page(
        page_id=f"pbc-support-{support_period}",
        parser_id=PBC_MONEY_STOCK_PARSER,
        source_system="pbc_official",
        period=support_period,
        release_at="",
        source_record_id="",
        source_url=(
            "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/"
            "fixture-support/index.html"
        ),
        support_only=True,
    )
    for period_end in monthly_periods:
        period = period_end.strftime("%Y%m")
        release = (period_end + pd.Timedelta(days=2, hours=1)).tz_localize(
            "UTC"
        ).isoformat()
        record_id = f"fixture-pbc-{period}"
        source_url = (
            "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/"
            f"{record_id}/index.html"
        )
        digest, _metadata = official_page(
            page_id=f"pbc-{period}",
            parser_id=PBC_MONEY_STOCK_PARSER,
            source_system="pbc_official",
            period=period,
            release_at=release,
            source_record_id=record_id,
            source_url=source_url,
        )
        for indicator_id in ("cn.m1_yoy", "cn.m2_yoy"):
            add_observation(
                indicator_id=indicator_id,
                period_end=period_end,
                release_at=release,
                source_system="pboc_official",
                source_record_id=record_id,
                source_url=source_url,
                evidence_digests=[digest, support_digest],
            )

    latest_quarter = pd.offsets.QuarterEnd().rollback(target)
    quarterly_periods = [
        latest_quarter - pd.offsets.QuarterEnd(offset)
        for offset in (2, 1, 0)
    ]
    for index, period_end in enumerate(quarterly_periods):
        if period_end == latest_quarter:
            digest, page_metadata = economy_pages[
                period_end.strftime("%Y%m")
            ]
        else:
            quarter = (period_end.month - 1) // 3 + 1
            period = f"{period_end.year}Q{quarter}"
            release = (
                period_end + pd.Timedelta(days=2, hours=1)
            ).tz_localize("UTC").isoformat()
            record_id = f"t{period_end + pd.Timedelta(days=2):%Y%m%d}_{index + 7}"
            source_url = (
                "https://www.stats.gov.cn/sj/zxfb/"
                f"{period_end:%Y%m}/{record_id}.html"
            )
            digest, page_metadata = official_page(
                page_id=f"gdp-{period.lower()}",
                parser_id=NBS_QUARTERLY_GDP_PARSER,
                source_system="nbs_official",
                period=period,
                release_at=release,
                source_record_id=record_id,
                source_url=source_url,
            )
        add_observation(
            indicator_id="cn.gdp_yoy",
            period_end=period_end,
            release_at=str(page_metadata["release_at"]),
            source_system="nbs_official",
            source_record_id=str(page_metadata["source_record_id"]),
            source_url=str(page_metadata["source_url"]),
            evidence_digests=[digest],
            frequency="quarterly",
        )

    plan_digest = add_evidence(
        "local:bootstrap-plan",
        {"extension": ".bin", "evidence_kind": "macro_local_bound_input"},
    )
    local_coverage_hashes: list[str] = []
    local_effective_values: list[str] = []
    for index, period_end in enumerate(
        target - pd.Timedelta(days=offset) for offset in (2, 1, 0)
    ):
        target_date = period_end.strftime("%Y%m%d")
        effective = (period_end + pd.Timedelta(hours=1)).tz_localize(
            "UTC"
        ).isoformat()
        coverage_hash = canonical_hash(
            {"fixture_coverage_target": target_date}
        )
        local_coverage_hashes.append(coverage_hash)
        local_effective_values.append(effective)
        strict_digest = add_evidence(
            f"local:strict-evidence:{target_date}",
            {
                "extension": ".bin",
                "evidence_kind": (
                    "strict_parquet_local_observation_evidence"
                ),
                "schema_version": LOCAL_MARKET_BREADTH_EVIDENCE_SCHEMA,
                "target_trade_date": target_date,
                "evidence_semantic_sha256": canonical_hash(
                    {"fixture_evidence_target": target_date}
                ),
                "coverage_contract_sha256": coverage_hash,
                "effective_available_at": effective,
            },
        )
        bound_inputs = [
            add_evidence(
                f"local:{role}:{target_date}",
                {
                    "extension": ".bin",
                    "evidence_kind": "macro_local_bound_input",
                },
            )
            for role in ("snapshot", "coverage", "scope", "part")
        ]
        add_observation(
            indicator_id="market.breadth",
            period_end=period_end,
            release_at=effective,
            source_system="local_strict_parquet",
            source_record_id=f"market.breadth:{target_date}:fixture",
            source_url=(
                "local://strict-parquet/cn/fixture/" + target_date
            ),
            evidence_digests=[strict_digest, plan_digest, *bound_inputs],
            frequency="daily",
            unit="%",
            dimension_type="market_confirmation",
            value=50.0,
        )

    snapshot = build_macro_snapshot(
        observations,
        market="CN",
        as_of=target.strftime("%Y%m%d"),
        decision_cutoff_at=decision_cutoff,
    ).to_dict()
    assert snapshot["readiness_status"] == "pass"
    assert snapshot["coverage"]["national"] == 0.8125
    result = publish_observations(
        observations,
        root=root,
        run_id=run_id,
        metadata={
            "schema_version": PRODUCTION_OBSERVATION_BUNDLE_SCHEMA,
            "market": "CN",
            "as_of": target.strftime("%Y%m%d"),
            "decision_cutoff_at": decision_cutoff.isoformat(),
            "official_bundle_manifest_sha256": official_manifest_sha256,
            "official_plan_sha256": "2" * 64,
            "local_bootstrap_plan_sha256": plan_digest,
            "local_snapshot_manifest_sha256": canonical_hash(
                {"fixture_targets": local_effective_values}
            ),
            "local_coverage_contract_sha256": canonical_hash(
                {"values": local_coverage_hashes}
            ),
            "local_effective_available_at": max(local_effective_values),
            "validated_snapshot_hash": snapshot["snapshot_hash"],
            "atomic_combined_publication": True,
        },
        evidence_bytes=evidence_bytes,
        evidence_metadata=evidence_metadata,
        observation_evidence=observation_evidence,
    )
    assert result["promoted"] is True
    return pointer_sha256(root)


def _formula_universe(*, trade_date: str) -> dict[str, Any]:
    symbols = [f"{index:06d}.SZ" for index in range(100)]
    empty_sha = macro_mart._formula_symbol_set_sha256([])
    symbol_sha = macro_mart._formula_symbol_set_sha256(symbols)
    return {
        "schema_version": macro_mart.MARKET_FORMULA_UNIVERSE_SCHEMA,
        "selection_rule": macro_mart.MARKET_FORMULA_SELECTION_RULE,
        "target_trade_date": str(trade_date).replace("-", ""),
        "input_symbol_count": 100,
        "target_terminal_symbol_count": 100,
        "stale_symbol_count": 0,
        "scored_symbol_count": 100,
        "input_row_count": 100,
        "target_terminal_row_count": 100,
        "stale_row_count": 0,
        "input_symbol_set_sha256": symbol_sha,
        "target_terminal_symbol_set_sha256": symbol_sha,
        "stale_symbol_set_sha256": empty_sha,
        "scored_symbol_set_sha256": symbol_sha,
        "macro_snapshot_sha256": "a" * 64,
        "macro_control_semantic_sha256": "b" * 64,
    }


def _bundle(*, trade_date: str) -> dict[str, Any]:
    fetched_at = "2024-05-10T08:00:00+00:00"
    endpoints: dict[str, Any] = {}
    selected: dict[str, Any] = {}
    for endpoint, spec in sorted(macro_mart._ENDPOINT_SPECS.items()):
        values = {
            field: 8.6 if field == "m2_yoy" else 1.0
            for field in spec["value_fields"]
        }
        record = {"month": "202404", **values}
        records = [record]
        endpoints[endpoint] = {
            "endpoint": endpoint,
            "query": {"start_m": "202401", "end_m": "202405"},
            "columns": sorted(record),
            "row_count": 1,
            "records": records,
            "records_sha256": hashlib.sha256(
                macro_mart._canonical_json_bytes({"records": records})
            ).hexdigest(),
        }
        selected[endpoint] = {
            "month": "202404",
            "values": values,
            "observed_available_at": fetched_at,
            "official_release_timestamp_known": False,
            "max_release_lag_days": int(spec["max_release_lag_days"]),
            "conservative_available_by": "2024-06-14",
            "transform_role": (
                "policy_signal" if endpoint == "cn_m" else "context_only"
            ),
        }
    return {
        "schema_version": macro_mart.LEGACY_PROVIDER_BUNDLE_SCHEMA,
        "provider_id": "tushare_pro",
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "trade_date": trade_date,
        "fetched_at": fetched_at,
        "decision_cutoff_at": fetched_at,
        "live_requested": True,
        "historical_replay_eligible": False,
        "official_release_timestamps_claimed": False,
        "query_window": {"start_month": "202401", "end_month": "202405"},
        "endpoints": endpoints,
        "selected_inputs": selected,
    }


def bind_macro_generation(
    root: Path,
    *,
    generation_id: str,
    row: Mapping[str, Any],
) -> tuple[Path, Path, Path, Path]:
    """Write a hash-consistent canonical fixture without invoking live code."""

    generation = root / "_generations" / generation_id
    generation.mkdir(parents=True, exist_ok=True)
    normalized_row = dict(row)
    normalized_row.setdefault(
        "macro_score_100",
        50.0 * (float(normalized_row["macro_score"]) + 1.0),
    )
    frame = pd.DataFrame([normalized_row])
    table = generation / "part.parquet"
    frame.to_parquet(table, index=False)
    table_sha = hashlib.sha256(table.read_bytes()).hexdigest()

    provider_bundle = _bundle(trade_date=str(row["trade_date"]))
    provider_path = generation / "provider_bundle.json"
    provider_path.write_bytes(
        macro_mart._canonical_json_bytes(provider_bundle) + b"\n"
    )
    provider_sha = hashlib.sha256(provider_path.read_bytes()).hexdigest()
    output_frame_sha = macro_mart._frame_sha256(frame)
    snapshot_payload: dict[str, Any] = {
        "schema_version": "macro-snapshot.v2",
        "market": "CN",
        "as_of": str(row["trade_date"]),
        "readiness_status": "pass",
        "national_states": {
            "growth": 0.48,
            "credit_liquidity": 0.4,
            "inflation": 0.0,
            "policy_fiscal": 0.0,
            "property": 0.0,
            "external": 0.0,
            "market_confirmation": 0.0,
        },
        "coverage": {"national": 0.8125},
    }
    snapshot_payload["snapshot_hash"] = canonical_hash(snapshot_payload)
    observation_generation = {
        "generation_id": "macro-observations-g1",
        "pointer_sha256": "3" * 64,
        "parquet_sha256": "4" * 64,
        "manifest_sha256": "5" * 64,
        "content_set_hash": "6" * 64,
        "row_count": 39,
    }
    row_volatility = float(row["volatility_percentile"])
    controls = build_v15_macro_controls(
        snapshot_payload,
        volatility_percentile=(
            row_volatility if 0.0 <= row_volatility <= 100.0 else 45.0
        ),
        observation_generation=observation_generation,
    )
    snapshot_path = generation / "macro_snapshot.json"
    snapshot_path.write_bytes(
        macro_mart._canonical_json_bytes(snapshot_payload) + b"\n"
    )
    snapshot_sha = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    controls_path = generation / "v15_controls.json"
    controls_path.write_bytes(
        macro_mart._canonical_json_bytes(controls) + b"\n"
    )
    controls_sha = hashlib.sha256(controls_path.read_bytes()).hexdigest()
    formula_universe = _formula_universe(
        trade_date=str(row["trade_date"])
    )
    formula_universe_sha = macro_mart._canonical_json_sha256(
        formula_universe
    )
    provenance: dict[str, Any] = {
        "schema_version": macro_mart.PRIMARY_PROVENANCE_SCHEMA,
        "status": "verified_live_tushare",
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "trade_date": str(row["trade_date"]),
        "fetched_at": "2024-05-10T08:00:00+00:00",
        "provider_bundle_sha256": provider_sha,
        "canonical_market_pointer_sha256": _POINTER_SHA,
        "market_input_files_sha256": _MARKET_FILES_SHA,
        "market_formula_universe_sha256": formula_universe_sha,
        "output_frame_sha256": output_frame_sha,
        "output_parquet_sha256": table_sha,
        "macro_snapshot_sha256": snapshot_sha,
        "v15_controls_sha256": controls_sha,
        "macro_observation_pointer_sha256": observation_generation[
            "pointer_sha256"
        ],
        "v15_controls_semantic_sha256": controls["semantic_sha256"],
        "transform_version": macro_mart.V15_TRANSFORM_VERSION,
        "historical_replay_eligible": False,
    }
    provenance["envelope_sha256"] = macro_mart._canonical_json_sha256(
        provenance
    )
    manifest_payload = {
        "schema_version": macro_mart.CANONICAL_MANIFEST_SCHEMA,
        "generation_id": generation_id,
        "table": "macro_daily",
        "table_path": "part.parquet",
        "parquet_sha256": table_sha,
        "provider_bundle_path": "provider_bundle.json",
        "provider_bundle_sha256": provider_sha,
        "row_count": 1,
        "source": "tushare_primary",
        "source_priority": "tushare_primary",
        "provider_status": "verified_provider_snapshot",
        "pit_status": "market_point_in_time",
        "as_of": str(row["trade_date"]),
        "decision_cutoff_at": "2024-05-10T08:00:00+00:00",
        "historical_replay_eligible": False,
        "transform_version": macro_mart.V15_TRANSFORM_VERSION,
        "market_input_files": _MARKET_INPUT_FILES,
        "market_input_files_sha256": _MARKET_FILES_SHA,
        "market_formula_universe": formula_universe,
        "market_formula_universe_sha256": formula_universe_sha,
        "macro_snapshot_path": "macro_snapshot.json",
        "macro_snapshot_sha256": snapshot_sha,
        "v15_controls_path": "v15_controls.json",
        "v15_controls_sha256": controls_sha,
        "v15_controls_schema_version": controls["schema_version"],
        "v15_controls_semantic_sha256": controls["semantic_sha256"],
        "macro_observation_generation": observation_generation,
        "primary_provenance": provenance,
        "production_eligible": True,
    }
    manifest = generation / "manifest.json"
    manifest.write_text(
        json.dumps(manifest_payload, sort_keys=True),
        encoding="utf-8",
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    catalog = root.parent / "_catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "schema_version": macro_mart.STRICT_CATALOG_SCHEMA,
                "required_tables": ["macro_daily"],
                "tables": {
                    "macro_daily": {
                        "path": table.relative_to(root.parent).as_posix(),
                        "generation_manifest": manifest.relative_to(
                            root.parent
                        ).as_posix(),
                        "generation_id": generation_id,
                        "parquet_sha256": table_sha,
                        "generation_manifest_sha256": manifest_sha,
                        "provider_bundle_sha256": provider_sha,
                        "macro_snapshot_sha256": snapshot_sha,
                        "v15_controls_sha256": controls_sha,
                        "v15_controls_semantic_sha256": controls[
                            "semantic_sha256"
                        ],
                        "macro_observation_generation": observation_generation,
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return catalog, table, manifest, provider_path
