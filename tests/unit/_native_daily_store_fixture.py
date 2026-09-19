"""Synthetic native Store integration inputs; no mocks and no real account data.

Market/Calendar files are fixture inputs validated by the native Store contract.
This is not a claim that the separate Market/Factor producer DAG was exercised.
Provider fields use production wire grammar; every value is generated test data,
not a provider response. SYNTHETIC-FIXTURE.json and the proof retain that boundary.
"""

from argparse import Namespace
from pathlib import Path
import hashlib
import json
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from scripts import cn_official_close_batch as batch
from scripts import manage_cn_strategy_records as manager
from close_cn_dashboard_official_valuation import build_record, content_sha256
from cn_dashboard_common import validate_record
from quant_investor.strategy_records import performance as perf
from quant_investor.strategy_records import store
from quant_investor.strategy_records import event_store
from quant_investor.market import cn_benchmark_store as benchmark
from test_close_cn_dashboard_official_valuation import _source_fixture, STOCKS

DAYS = ["2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(store.canonical_json_bytes(value))
    path.chmod(0o600)
    return sha(path)


def inventory(root, directory, day):
    return {
        "record_id": directory.name,
        "relative_path": directory.name,
        "state": "ONLINE",
        "storage_state": "ONLINE",
        "sealed_at": f"{day}T10:30:00Z",
        **manager.build_inventory(directory, enforce_new_record_budget=False),
        **batch._record_closure(root, directory),
        "history_eligible": True,
        "evidence_status": "HASH_VERIFIED",
    }


class NativeStoreFixture:
    def __init__(
        self, project: Path, *, stock_symbols=STOCKS, preserve_market=False, seed_transition=None
    ):
        self.stocks = tuple(stock_symbols)
        self.preserve_market = preserve_market
        self.project = project
        project.mkdir(parents=True, exist_ok=True)
        write(
            project / "SYNTHETIC-FIXTURE.json",
            {"synthetic": True, "real_account": False, "provider_calls": False},
        )
        _, source, _, market_pointer, evidence = _source_fixture(
            project,
            stock_symbols=self.stocks,
            market_subdir="fixtures/store-seed-market" if preserve_market else "data/parquet/cn",
        )
        self.root = project / "results/strategy_records/CN/aggressive_tech_manufacturing"
        self.root.mkdir(parents=True, exist_ok=True)
        new_source = self.root / source.name
        shutil.move(str(source), new_source)
        source = new_source
        # Establish a synthetic CNY1m baseline before any registration/sealing.
        manual = json.loads((source / "manual_execution_manifest.json").read_bytes())
        equity = manual["market_value_after"]
        cash = 1_000_000 - equity
        for target in (manual, manual["financial_state"]):
            target.update(
                cash_after=cash,
                total_value_after=1_000_000.0,
                portfolio_pnl_after=0.0,
                portfolio_return_after=0.0,
            )
        manual["financial_state_sha256"] = hashlib.sha256(
            store.canonical_json_bytes(manual["financial_state"])
        ).hexdigest()
        write(source / "manual_execution_manifest.json", manual)
        manifest = json.loads((source / "manifest.json").read_bytes())
        manifest["manual_execution"] = manual
        write(source / "manifest.json", manifest)
        pd.DataFrame(
            [
                {
                    k: manual[k]
                    for k in (
                        "cash_after",
                        "market_value_after",
                        "total_value_after",
                        "portfolio_pnl_after",
                    )
                }
            ]
        ).to_csv(source / "pnl_summary.csv", index=False)
        previous = self.root / "20260819_1321"
        shutil.copytree(source, previous)
        old_manifest = json.loads((previous / "manifest.json").read_bytes())
        old_manifest["timestamp"] = previous.name
        write(previous / "manifest.json", old_manifest)
        if seed_transition is not None:
            # Isolated fixture hook, before either native record is sealed or registered.
            seed_transition(previous, source)
        previous_record = inventory(self.root, previous, "2026-08-19")
        source_record = inventory(self.root, source, "2026-08-20")
        initial = store.bootstrap_catalog(
            self.root,
            records=[previous_record, source_record],
            active_record_id=source.name,
            previous_record_id=previous.name,
            generation_id="g-synthetic-source",
            published_at="2026-08-20T10:30:00Z",
            catalog_schema=store.CATALOG_SCHEMA_V2,
        )
        capture_path = "fixtures/benchmark-seed-capture.json"
        capture_sha = write(project / capture_path, {"synthetic": True, "kind": "benchmark_seed"})
        seed_rows = [
            {
                "date": d,
                "ts_code": code,
                "close": 1000.0 + i + j,
                "source_system": "local.strict.capture",
                "coverage": "exact_close",
                "value_date": d,
            }
            for j, d in enumerate(["2026-08-19", "2026-08-20", "2026-08-21"])
            for i, code in enumerate(benchmark.REQUIRED_CODES)
        ]
        seed_benchmark = benchmark.publish_generation(
            project / "data/parquet/cn/benchmarks",
            rows=seed_rows,
            generation_id="benchmark-20260821-synthetic-seed",
            captured_at="2026-08-21T10:00:00Z",
            expected_pointer_sha256=benchmark.EMPTY_POINTER_SHA256,
            acquisition_receipt_ref={"path": capture_path, "sha256": capture_sha},
        )
        benchmark_csv = project / "portfolio_dashboard/inputs/cn_index_benchmark.csv"
        benchmark_csv.write_bytes(benchmark.compatibility_csv_bytes(seed_benchmark["rows"]))
        market_doc = json.loads(market_pointer.read_bytes())
        market_doc.update(status="OK", blockers=[])
        write(market_pointer, market_doc)
        market_manifest = project / market_doc["manifest_path"]
        evidence = batch._market_evidence(
            project=project,
            market_pointer=market_doc,
            market_pointer_sha=sha(market_pointer),
            market_manifest_path=market_manifest,
            market_manifest=json.loads(market_manifest.read_bytes()),
            benchmark=seed_benchmark,
            compatibility_csv=benchmark_csv,
            trade_date="2026-08-21",
            symbols=list(self.stocks),
        )
        if preserve_market:
            evidence["market_pointer_path"] = market_pointer.relative_to(project).as_posix()
        self.seed = self.root / "20260821_1830"
        self.seed.mkdir()
        closure = batch._record_closure(self.root, source)
        build_record(
            staging_dir=self.seed,
            record_root=self.root,
            source_dir=source,
            registered_closure=closure,
            record_id=self.seed.name,
            trade_date="20260821",
            recorded_at_iso="2026-08-21T18:30:00+08:00",
            evidence=evidence,
            project_root=project,
            expected_market_pointer_sha256=sha(market_pointer),
            source_pointer_sha256=initial["pointer_sha256"],
            source_catalog_generation_id=initial["pointer"]["generation_id"],
            source_catalog_sha256=initial["pointer"]["catalog_sha256"],
            continuity_receipt_id="synthetic-seed-continuity",
            continuity_receipt_sha256=sha(market_pointer),
            continuity_receipt_created_at="2026-08-21T10:00:00Z",
            continuity_checkpoint_digest=content_sha256(closure),
            evidence_input_sha256=hashlib.sha256(store.canonical_json_bytes(evidence)).hexdigest(),
        )
        validate_record(self.seed, self.root, project)
        records = [previous_record, source_record, inventory(self.root, self.seed, "2026-08-21")]
        projection = {"historical_records": []}
        for directory, day in (
            (previous, "2026-08-19"),
            (source, "2026-08-20"),
            (self.seed, "2026-08-21"),
        ):
            doc = json.loads((directory / "manual_execution_manifest.json").read_bytes())
            projection["historical_records"].append(
                {
                    "record": directory.name,
                    "valuation_date": day,
                    "accounting": {
                        k: doc[k]
                        for k in (
                            "cash_after",
                            "market_value_after",
                            "total_value_after",
                            "portfolio_pnl_after",
                        )
                    },
                    "capital_base": 1_000_000,
                    "funding": None,
                    "funding_correction": None,
                    "evidence_status": "SYNTHETIC_REGISTERED",
                }
            )
        parent = store.publish_catalog(
            self.root,
            expected_pointer_sha256=initial["pointer_sha256"],
            records=records,
            dashboard_projection=projection,
            active_record_id=self.seed.name,
            previous_record_id=source.name,
            generation_id="g-synthetic-v2",
            published_at="2026-08-21T10:31:00Z",
            catalog_schema=store.CATALOG_SCHEMA_V2,
        )
        normalized, projection_sha, normalized_sha = perf.normalize_registered_projection(
            parent["catalog"]
        )
        rows = perf.build_seed_rows(normalized, catalog=parent["catalog"])
        identity = manager.command_declare_strategy_identity(
            Namespace(
                project_root=str(project),
                identity_path=manager.IDENTITY_RELATIVE_PATH,
                declared_at="2026-08-21T10:32:00Z",
                provenance="SYNTHETIC TEST FIXTURE ONLY",
            )
        )
        prefix = "_record_store/performance/p-synthetic-seed"
        directory = self.root / prefix
        directory.mkdir(parents=True)
        series_sha, series_bytes = perf.write_deterministic_parquet(
            rows, directory / "series.parquet"
        )
        owner = perf.build_owner_declaration(
            performance_generation_id="p-synthetic-seed",
            declared_at="2026-08-21T10:32:00Z",
            series_path=prefix + "/series.parquet",
            series_sha256=series_sha,
            series_bytes=series_bytes,
            source_pointer_sha256=parent["pointer_sha256"],
            source_catalog_sha256=parent["pointer"]["catalog_sha256"],
            normalized_projection_semantic_sha256=normalized_sha,
        )
        owner_sha = write(directory / "owner_declaration.v1.json", owner)
        pmanifest = perf.build_manifest(
            performance_generation_id="p-synthetic-seed",
            generated_at="2026-08-21T10:32:00Z",
            identity_path=manager.IDENTITY_RELATIVE_PATH,
            identity_sha256=identity["identity_sha256"],
            parent_performance_manifest_sha256=None,
            source_pointer_sha256=parent["pointer_sha256"],
            source_catalog_generation_id=parent["pointer"]["generation_id"],
            source_catalog_sha256=parent["pointer"]["catalog_sha256"],
            dashboard_projection_sha256=projection_sha,
            normalized_projection_semantic_sha256=normalized_sha,
            series_path=prefix + "/series.parquet",
            series_sha256=series_sha,
            series_bytes=series_bytes,
            owner_path=prefix + "/owner_declaration.v1.json",
            owner_sha256=owner_sha,
            owner_bytes=(directory / "owner_declaration.v1.json").stat().st_size,
            rows=rows,
        )
        manifest_sha = write(directory / "manifest.v1.json", pmanifest)
        pref = perf.build_performance_history_ref(
            manifest=pmanifest,
            manifest_sha256=manifest_sha,
            manifest_bytes=(directory / "manifest.v1.json").stat().st_size,
        )
        lineage = []
        for index, (record, day) in enumerate(
            zip(records, ["2026-08-19", "2026-08-20", "2026-08-21"])
        ):
            lineage.append(
                {
                    "record_id": record["record_id"],
                    "source_record_id": None if index == 0 else records[index - 1]["record_id"],
                    "supersedes_record_id": None,
                    "valuation_date": day,
                    "execution_class": "NO_TRADE",
                    "publication_class": "OFFICIAL_FINANCIAL_STATE",
                    "storage_state": "ONLINE",
                    "manifest_ref": {
                        "path": record["manifest_path"],
                        "sha256": record["manifest_sha256"],
                    },
                    "manual_manifest_ref": {
                        "path": record["manual_manifest_path"],
                        "sha256": record["manual_manifest_sha256"],
                    },
                    "effective_ledger_ref": {
                        "path": record["ledger_path"],
                        "sha256": record["ledger_sha256"],
                    },
                    "financial_state_sha256": record["financial_state_sha256"],
                    "ledger_parquet_sha256": record["ledger_sha256"],
                }
            )
        store.publish_catalog(
            self.root,
            expected_pointer_sha256=parent["pointer_sha256"],
            records=records,
            active_record_id=self.seed.name,
            previous_record_id=source.name,
            generation_id="g-synthetic-v3",
            published_at="2026-08-21T10:33:00Z",
            catalog_schema=store.CATALOG_SCHEMA_V3,
            inherit_history_registry=False,
            lineage_index=lineage,
            performance_history_ref=pref,
        )
        manager.command_verify(Namespace(record_root=str(self.root)))
        self.policy_path = "fixtures/official-close-policy.json"
        self.policy_sha = write(
            project / self.policy_path,
            {
                "schema_id": batch.POLICY_SCHEMA,
                "policy_id": "cn-daily-official-close-policy-v1",
                "strategy_label": "aggressive_tech_manufacturing",
                "record_root": str(self.root.relative_to(project)),
                "revoked_at": None,
                "max_backlog_open_days": 10,
                "allowed_writes": [
                    "DAILY_NO_ACTION_CONTINUITY_RECEIPT",
                    "OFFICIAL_VALUATION_RECORD",
                    "PERFORMANCE_APPEND",
                    "IMMUTABLE_CATALOG_GENERATION",
                    "STRATEGY_RECORD_POINTER_CAS",
                ],
                "forbidden": dict.fromkeys(
                    [
                        "broker_connection",
                        "order_creation",
                        "trade_execution",
                        "unregistered_share_mutation",
                        "unregistered_cash_mutation",
                    ],
                    True,
                ),
                "broker_order_trade_authority": False,
                "actual_holdings_mutation_authority": False,
                "synthetic": True,
            },
        )
        self.benchmark_pointer = seed_benchmark["pointer_sha256"]
        self.event_pointer = event_store.EMPTY_POINTER_SHA256
        from quant_investor.market.pit_universe import (
            PITUniverseStore,
            PITUniverseRecord,
            LIST_STATUS_LISTED,
        )

        if not preserve_market:
            pit = PITUniverseStore(
                root_dir=project / "data/parquet/cn/reference",
                raw_root=project / "data/pit_raw",
                compatibility_path=project / "data/pit_compat.json",
            )
            self.pit = pit.write_snapshot(
                raw_records=[
                    PITUniverseRecord(
                        symbol=symbol,
                        source_list_status=LIST_STATUS_LISTED,
                        list_date="20200101",
                        observed_at="2026-08-19T00:00:00Z",
                        source_run_id="synthetic-native-store",
                    )
                    for symbol in self.stocks
                ],
                observed_at="2026-08-19T00:00:00Z",
                source_run_id="synthetic-native-store",
            )
            write(
                project / "data/cn_universe/cn_index_components.json",
                {"full_a": list(self.stocks), "stats": {"total_unique": len(self.stocks)}},
            )
        self.days = []
        self.closures = []
        self.initial_holdings = pd.read_parquet(self.seed / "ledger_after_manual_switch.parquet")

    def advance(self, day, *, publish_events=True, adjustment_factors=None, update_risk_free=True):
        self.days.append(day)
        days = ["2026-08-20", "2026-08-21", *self.days]
        if self.preserve_market:
            market_path = self.project / "data/parquet/cn/_latest.json"
            current = json.loads(market_path.read_bytes())
            if current["latest_complete_trade_date"] < day.replace("-", ""):
                raise ValueError("shared Market does not cover requested date")
            self.market_sha = sha(market_path)
        else:
            snapshot = "synthetic-" + day.replace("-", "")
            serving = self.project / f"data/parquet/cn/_snapshots/{snapshot}/serving/bars"
            for index, symbol in enumerate(self.stocks, 1):
                path = serving / f"symbol={symbol}/bars.parquet"
                path.parent.mkdir(parents=True)
                pd.DataFrame(
                    [
                        {
                            "ts_code": symbol,
                            "trade_date": d.replace("-", ""),
                            "close": 19.0 + index + j,
                        }
                        for j, d in enumerate(days)
                    ]
                ).assign(
                    **(
                        {"adj_factor": [adjustment_factors.get(symbol, {}).get(d) for d in days]}
                        if adjustment_factors is not None
                        else {}
                    )
                ).to_parquet(
                    path, index=False
                )
            table = serving.parents[1] / "table/bars"
            table.mkdir(parents=True)
            pd.concat(
                [
                    pd.read_parquet(serving / f"symbol={symbol}/bars.parquet")
                    for symbol in self.stocks
                ]
            ).to_parquet(table / "part.parquet", index=False)
            compact = day.replace("-", "")
            coverage = {
                "coverage_schema_version": "cn-full-a-coverage.v4",
                "complete": True,
                "coverage_ratio": 1.0,
                "coverage_complete_count": len(self.stocks),
                "expected_scope_count": len(self.stocks),
                "observed_bar_count": len(self.stocks),
                "blocking_incomplete_count": 0,
                "latest_available_trade_date": compact,
                "latest_complete_trade_date": compact,
                "coverage_trade_date": compact,
                "expected_scope_sha256": hashlib.sha256(
                    store.canonical_json_bytes(list(self.stocks))
                ).hexdigest(),
                "suspended_symbols": [],
                "inactive_symbols": [],
                "verified_nontrading_bak_daily_zero_symbols": [],
                "verified_terminal_delisting_symbols": [],
                "allowed_stale_symbols": [],
                "non_blocking_absent_symbols": [],
                "true_missing_symbols": [],
                "classification_sets_disjoint": True,
                "pit_membership_path": str(self.pit["canonical_path"]),
                "pit_membership_sha256": self.pit["canonical_sha256"],
                "pit_generation_id": self.pit["generation_id"],
                "pit_generation_manifest_path": str(self.pit["generation_manifest_path"]),
                "pit_generation_manifest_sha256": self.pit["generation_manifest_sha256"],
            }
            manifest = self.project / f"data/parquet/cn/_snapshots/{snapshot}.json"
            payload = {
                "market": "CN",
                "snapshot_id": snapshot,
                "latest_complete_trade_date": compact,
                "latest_trade_date": compact,
                "manifest_path": str(manifest),
                "table_root": str(table),
                "derived_serving_root": str(serving),
                "coverage": coverage,
                "status": "OK",
                "blockers": [],
            }
            write(manifest, payload)
            self.market_sha = write(self.project / "data/parquet/cn/_latest.json", payload)
        from test_cn_dashboard_export import _write_risk_free

        if update_risk_free:
            _write_risk_free(self.project, ["2026-08-19", "2026-08-20", "2026-08-21", *self.days])
        capture = "fixtures/benchmark-capture-" + day + ".json"
        capture_sha = write(self.project / capture, {"synthetic": True, "day": day})
        bmrows = [
            {
                "date": d,
                "ts_code": symbol,
                "close": 1000.0 + i + j,
                "source_system": "tushare.index_daily",
                "coverage": "exact_close",
                "value_date": d,
            }
            for j, d in enumerate(["2026-08-19", "2026-08-20", "2026-08-21", *self.days])
            for i, symbol in enumerate(benchmark.REQUIRED_CODES)
        ]
        bm = benchmark.publish_generation(
            self.project / "data/parquet/cn/benchmarks",
            rows=bmrows,
            generation_id="benchmark-" + day.replace("-", "") + "-synthetic",
            captured_at=day + "T10:00:00Z",
            expected_pointer_sha256=self.benchmark_pointer,
            acquisition_receipt_ref={"path": capture, "sha256": capture_sha},
        )
        self.benchmark_pointer = bm["pointer_sha256"]
        (self.project / "portfolio_dashboard/inputs/cn_index_benchmark.csv").write_bytes(
            benchmark.compatibility_csv_bytes(bm["rows"])
        )
        if publish_events:
            decl = "fixtures/empty-events-" + day + ".json"
            declsha = write(
                self.project / decl,
                {
                    "synthetic": True,
                    "trade_date": day,
                    "dimensions": dict.fromkeys(event_store.EVENT_DIMENSIONS, []),
                },
            )
            self.closures.append(
                event_store.build_empty_closure(
                    trade_date=day,
                    sealed_at=day + "T10:00:00Z",
                    cutoff_at=day + "T07:30:00Z",
                    policy_ref={"path": self.policy_path, "sha256": self.policy_sha},
                    owner_declaration_ref={"path": decl, "sha256": declsha},
                    source_receipt_ref=None,
                )
            )
            ev = event_store.publish_generation(
                self.root / "_event_store",
                generation_id="event-" + day.replace("-", "") + "-synthetic",
                generated_at=day + "T10:00:00Z",
                expected_pointer_sha256=self.event_pointer,
                closures=self.closures,
                policy_ref={"path": self.policy_path, "sha256": self.policy_sha},
            )
            self.event_pointer = ev["pointer_sha256"]
        self.calendar_path = self.project / "fixtures" / ("calendar-" + day + ".json")
        self.calendar_sha = write(
            self.calendar_path,
            {
                "schema_version": "cn-close-session-receipt.v1",
                "status": "TARGET_AUTHORIZED",
                "ordered_open_dates": [d.replace("-", "") for d in days],
                "synthetic": True,
            },
        )
        return self.arguments()

    def arguments(self):
        return {
            "project_root": self.project,
            "record_root": self.root,
            "expected_store_pointer_sha": sha(self.root / "_record_store/current.v1.json"),
            "expected_market_pointer_sha": self.market_sha,
            "expected_benchmark_pointer_sha": self.benchmark_pointer,
            "expected_event_pointer_sha": self.event_pointer,
            "calendar_receipt_path": self.calendar_path,
            "calendar_receipt_sha": self.calendar_sha,
            "policy_path": self.policy_path,
            "policy_sha": self.policy_sha,
            "retrospective_path": None,
            "retrospective_sha": None,
        }
