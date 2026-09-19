"""Generated large-cohort inputs for real strict LOW/W80 recomputation.

Synthetic weekday calendar and generated prices are NOT provider or OOS evidence.
No Factor activation, release attestation, or native generation proof is fabricated.
"""

from datetime import datetime, time, timezone
import hashlib
import math
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from quant_investor.factors.governance.source import role_schema


def put_table(path, rows, role):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=role_schema(role)), path)
    path.chmod(0o600)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class NativeFactorInputs:
    def __init__(self, root: Path, count=3000, *, extra_future_sessions=0):
        self.root = root
        self.symbols = [f"{i:06d}.SZ" for i in range(1, count + 1)]
        self.sessions = [d.date() for d in pd.bdate_range(end="2026-08-28", periods=95)]
        if type(extra_future_sessions) is not int or extra_future_sessions < 0:
            raise ValueError("synthetic future-session count invalid")
        if extra_future_sessions:
            extra = pd.bdate_range(
                start=pd.Timestamp(self.sessions[-1]) + pd.offsets.BDay(1),
                periods=extra_future_sessions,
            )
            if len(extra) != extra_future_sessions:
                raise ValueError("synthetic future calendar size differs")
            self.sessions.extend(stamp.date() for stamp in extra)
        root.mkdir(parents=True, exist_ok=True)
        (root / "SYNTHETIC-FIXTURE.txt").write_text(
            "Synthetic calendar/prices/PIT. Not provider data or real-time OOS.\n"
        )

    def day(self, offset, *, extra_history=0):
        sessions = self.sessions[: 91 + offset]
        if extra_history:
            prior = pd.bdate_range(
                end=pd.Timestamp(self.sessions[0]) - pd.offsets.BDay(1), periods=extra_history
            )
            sessions = [stamp.date() for stamp in prior] + sessions
        target = sessions[-1]
        folder = self.root / target.strftime("%Y%m%d")
        calendar, pit, market = [
            folder / (name + ".parquet") for name in ("calendar", "pit", "market")
        ]
        calendar_sha = put_table(
            calendar,
            [
                {
                    "ordinal": i,
                    "open_session": day,
                    "opens_at_utc": datetime.combine(day, time(1, 30), tzinfo=timezone.utc),
                    "closes_at_utc": datetime.combine(day, time(7), tzinfo=timezone.utc),
                }
                for i, day in enumerate(sessions)
            ],
            "exchange_calendar",
        )
        pit_sha = put_table(
            pit,
            [
                {
                    "signal_session": target,
                    "symbol": symbol,
                    "industry": "synthetic_industry",
                    "total_mv": float(1_000_000 + i * 100),
                    "tradable": True,
                }
                for i, symbol in enumerate(self.symbols)
            ],
            "pit_universe",
        )
        rows = []
        for i, symbol in enumerate(self.symbols):
            for j, day in enumerate(sessions, start=-extra_history):
                rows.append(
                    {
                        "trade_date": day,
                        "symbol": symbol,
                        "adj_close": 10 + i * 0.01 + j * 0.02 + 0.01 * math.sin(j / (2 + i % 23)),
                        "amount": 1000 + i * 3 + j * (1 + i % 17),
                        "vol": 100 + i * 0.1 + (1 + i % 19) * (j % 19) * 0.2,
                    }
                )
        market_sha = put_table(market, rows, "market_history")
        return {
            "exchange_calendar_path": calendar,
            "pit_universe_path": pit,
            "market_history_path": market,
            "exchange_calendar_sha256": calendar_sha,
            "pit_universe_sha256": pit_sha,
            "market_history_sha256": market_sha,
            "as_of": target.strftime("%Y%m%d"),
        }


def strict_market_from_factor_inputs(
    workspace,
    arguments,
    *,
    snapshot_suffix="",
    pit_observed_at="2026-08-19T00:00:00Z",
    macro_ready_layout=False,
    simulated_available_at=None,
):
    """Publish synthetic native PIT and immutable v4 Market layout for real reader QA."""
    import json
    from quant_investor.market.pit_universe import PITUniverseStore, PITUniverseRecord
    from quant_investor.market.market_data_reader import MarketDataReader
    from quant_investor.contracts import canonical_json_bytes

    data = workspace / "data"
    simulated = None
    provenance = None
    if simulated_available_at is not None:
        if not macro_ready_layout:
            raise ValueError("simulated availability requires Macro fixture layout")
        simulated = datetime.strptime(simulated_available_at, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
        if simulated.strftime("%Y%m%d") != arguments["as_of"] or simulated.hour < 7:
            raise ValueError("simulated availability must follow target close")
        marker = workspace / "SYNTHETIC-DATA-CLOCK.json"
        expected = {
            "synthetic": True,
            "workspace": str(workspace.resolve()),
            "real_time_oos_eligible": False,
        }
        if data.exists() and (not marker.exists() or json.loads(marker.read_text()) != expected):
            raise ValueError("simulation cannot rewrite an unmarked data workspace")
        if (
            data / "parquet/cn/_snapshots" / (simulated.strftime("%Y%m%dT%H%M%SZ") + ".json")
        ).exists():
            raise ValueError("simulation snapshot already exists")
        marker.write_bytes(canonical_json_bytes(expected))
        marker.chmod(0o600)
        provenance = {
            **expected,
            "available_at_simulated": simulated_available_at,
            "wall_clock_created_at": datetime.now(timezone.utc).isoformat(),
        }
    market = pd.read_parquet(arguments["market_history_path"])
    pit_rows = pd.read_parquet(arguments["pit_universe_path"])
    symbols = sorted(pit_rows["symbol"].tolist())
    as_of = arguments["as_of"]
    published = PITUniverseStore(
        root_dir=data / "parquet/cn/reference",
        raw_root=data / "pit_raw",
        compatibility_path=data / "pit_compat.json",
    ).write_snapshot(
        raw_records=[
            PITUniverseRecord(
                symbol=symbol,
                industry="synthetic_industry",
                source_list_status="L",
                list_date="20200101",
                observed_at=pit_observed_at,
                source_run_id="synthetic-native-factor",
            )
            for symbol in symbols
        ],
        observed_at=pit_observed_at,
        source_run_id="synthetic-native-factor",
        source_bindings={
            "scope_expansion_pending": {
                "schema_version": "cn_pit_scope_expansion_pending.v1",
                "authority_scope": "FROZEN_FULL_A",
                "admission_status": "NOT_CONFIGURED",
                "count": 0,
                "sha256": hashlib.sha256(
                    (
                        json.dumps({"items": []}, ensure_ascii=False, indent=2, sort_keys=True)
                        + "\n"
                    ).encode()
                ).hexdigest(),
                "identities": [],
                "rows": [],
            }
        },
    )
    snapshot_id = (
        (simulated or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
        if macro_ready_layout
        else "synthetic-native-factor-" + as_of + snapshot_suffix
    )
    snapshot_root = data / "parquet/cn/_snapshots" / snapshot_id
    table = snapshot_root / "table/bars"
    serving = snapshot_root / "serving/bars"
    market["total_mv"] = market["symbol"].map(dict(zip(pit_rows["symbol"], pit_rows["total_mv"])))
    market["ts_code"] = market.pop("symbol")
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.strftime("%Y%m%d")
    market["close"] = market["adj_close"]
    market["open"] = market["close"]
    market["high"] = market["close"] * 1.01
    market["low"] = market["close"] * 0.99
    market["adj_factor"] = 1.0
    if macro_ready_layout:
        market["pct_chg"] = (
            market.groupby("ts_code", sort=False)["close"].pct_change(fill_method=None).fillna(0.0)
            * 100
        )
    table.mkdir(parents=True)
    if macro_ready_layout:
        for month, frame in market.groupby(market["trade_date"].str[:6], sort=True):
            part = table / ("year=" + month[:4]) / ("month=" + month[4:]) / "part.parquet"
            part.parent.mkdir(parents=True)
            frame.to_parquet(part, index=False)
            pd.testing.assert_frame_equal(
                pd.read_parquet(part),
                frame.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
    else:
        market.to_parquet(table / "part.parquet", index=False)
    for symbol, frame in market.groupby("ts_code", sort=True):
        path = serving / ("symbol=" + symbol) / "bars.parquet"
        path.parent.mkdir(parents=True)
        frame.to_parquet(path, index=False)
        if macro_ready_layout:
            pd.testing.assert_frame_equal(
                pd.read_parquet(path),
                frame.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
    coverage = {
        "coverage_schema_version": "cn-full-a-coverage.v4",
        "complete": True,
        "coverage_ratio": 1.0,
        "coverage_complete_count": len(symbols),
        "expected_scope_count": len(symbols),
        "observed_bar_count": len(symbols),
        "blocking_incomplete_count": 0,
        "latest_available_trade_date": as_of,
        "latest_complete_trade_date": as_of,
        "coverage_trade_date": as_of,
        "upsert_target_trade_date": as_of,
        "categories_checked": ["full_a"],
        "expected_scope_sha256": hashlib.sha256("\n".join(symbols).encode()).hexdigest(),
        "classification_sets_disjoint": True,
        **{
            key: []
            for key in (
                "suspended_symbols",
                "inactive_symbols",
                "verified_nontrading_bak_daily_zero_symbols",
                "verified_terminal_delisting_symbols",
                "allowed_stale_symbols",
                "non_blocking_absent_symbols",
                "true_missing_symbols",
            )
        },
        "pit_membership_path": str(published["canonical_path"]),
        "pit_membership_sha256": published["canonical_sha256"],
        "pit_generation_id": published["generation_id"],
        "pit_generation_manifest_path": str(published["generation_manifest_path"]),
        "pit_generation_manifest_sha256": published["generation_manifest_sha256"],
    }
    manifest = data / "parquet/cn/_snapshots" / (snapshot_id + ".json")
    document = {
        "market": "CN",
        **({"readback_validated": True} if macro_ready_layout else {}),
        "status": "OK",
        "blockers": [],
        "snapshot_id": snapshot_id,
        "latest_complete_trade_date": as_of,
        "latest_trade_date": as_of,
        "manifest_path": str(manifest),
        "table_root": str(table),
        "derived_serving_root": str(serving),
        "coverage": coverage,
        **(
            {
                "metadata": {
                    "coverage": coverage,
                    **({"synthetic_time": provenance} if provenance else {}),
                }
            }
            if macro_ready_layout
            else {}
        ),
    }
    for path, value in [
        (manifest, document),
        (data / "parquet/cn/_latest.json", document),
        (
            data / "cn_universe/cn_index_components.json",
            {"full_a": symbols, "stats": {"full_a": len(symbols)}},
        ),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(value))
        path.chmod(0o600)
    if simulated is not None:
        import os

        stamp_ns = int(simulated.timestamp()) * 1_000_000_000
        paths = [
            manifest,
            data / "parquet/cn/_latest.json",
            data / "cn_universe/cn_index_components.json",
        ]
        paths.extend(path for path in snapshot_root.rglob("*") if path.is_file())
        for path in paths:
            os.utime(path, ns=(stamp_ns, stamp_ns))
    return MarketDataReader(data_root=data), published
