"""Three-branch data readiness contracts and offline assessment helpers."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from quant_investor.market.fundamental_generation import (
    load_fundamental_pointer,
    resolve_fundamental_table_path,
)
from quant_investor.branch_config import CANONICAL_BRANCH_ORDER
from quant_investor.versioning import BRANCH_SCHEMA_VERSION


STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_BLOCK = "block"
SOURCE_TUSHARE = "tushare_primary"
SOURCE_PUBLIC_FALLBACK = "public_structured_fallback"
SOURCE_OFFLINE = "manual_offline_snapshot"
SOURCE_PRIORITY_ORDER = (SOURCE_TUSHARE, SOURCE_PUBLIC_FALLBACK, SOURCE_OFFLINE)

DEFAULT_PARQUET_CN_ROOT = Path("data/parquet/cn")
DEFAULT_FUNDAMENTAL_ROOT = DEFAULT_PARQUET_CN_ROOT / "fundamental_daily"
DEFAULT_MACRO_ROOT = DEFAULT_PARQUET_CN_ROOT / "macro_daily"
DEFAULT_READINESS_ROOT = Path("reports/v14/branch_readiness")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_FROZEN_READINESS_ROOT = (_REPOSITORY_ROOT / "reports/branch_readiness").resolve()

QUANT_REQUIRED_FIELDS = ("open", "high", "low", "close", "volume", "amount")
FUNDAMENTAL_REQUIRED_FIELDS = (
    "fin_roe",
    "fin_roa",
    "fin_debt_to_assets",
    "fin_net_profit_yoy",
    "fin_ocf_to_profit",
    "fin_fcf_to_profit",
    "fcf_to_price",
    "forecast_revision",
)
MACRO_REQUIRED_FIELDS = (
    "macro_score",
    "liquidity_score",
    "volatility_percentile",
    "policy_signal",
)


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(as_of: str = "") -> str:
    suffix = _date_text(as_of) or datetime.now().strftime("%Y%m%d")
    return f"branch_readiness_{suffix}_{datetime.now().strftime('%H%M%S')}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _date_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return ""
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _compact_date(value: Any) -> str:
    date = _date_text(value)
    return date.replace("-", "") if date else ""


def _normalize_symbol(symbol: Any) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        code, suffix = text.split(".", 1)
        return f"{code.zfill(6)}.{suffix}"
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) >= 6:
        code = digits[:6]
        suffix = "SH" if code.startswith(("6", "9")) else "SZ"
        return f"{code}.{suffix}"
    return text


def _source_priority(source: str = "", explicit: str = "") -> str:
    priority = str(explicit or "").strip()
    if priority in SOURCE_PRIORITY_ORDER:
        return priority
    source_text = str(source or "").lower()
    if "tushare" in source_text:
        return SOURCE_TUSHARE
    if any(token in source_text for token in ("akshare", "public", "rss", "official")):
        return SOURCE_PUBLIC_FALLBACK
    return SOURCE_OFFLINE


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip()
    return bool(text) and text.lower() not in {"nan", "nat", "none"}


def _latest_frame_date(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return ""
    for column in ("trade_date", "date"):
        if column not in frame.columns:
            continue
        values = [_compact_date(item) for item in frame[column].tail(256)]
        values = [item for item in values if item]
        if values:
            return max(values)
    return ""


@dataclass
class BranchDataReadiness:
    branch: str
    status: str = STATUS_BLOCK
    coverage_ratio: float = 0.0
    freshness_status: str = "unknown"
    pit_status: str = "unknown"
    source_priority: str = SOURCE_OFFLINE
    source: str = ""
    as_of: str = ""
    required_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    affected_symbols: list[str] = field(default_factory=list)
    fallback_used: bool = False
    provider_status: str = "not_requested"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(_json_safe(asdict(self)))


@dataclass
class BranchGovernanceReport:
    run_id: str
    market: str
    category: str
    as_of: str
    generated_at: str = field(default_factory=_now_utc)
    readiness: dict[str, BranchDataReadiness] = field(default_factory=dict)
    blocked_symbols: list[str] = field(default_factory=list)
    quantifiable_universe: list[str] = field(default_factory=list)
    investable_universe: list[str] = field(default_factory=list)
    branch_data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self, *, include_branch_data: bool = True) -> dict[str, Any]:
        payload = {
            "run_id": self.run_id,
            "market": self.market,
            "category": self.category,
            "as_of": self.as_of,
            "generated_at": self.generated_at,
            "readiness": {key: value.to_dict() for key, value in self.readiness.items()},
            "blocked_symbols": list(self.blocked_symbols),
            "quantifiable_universe": list(self.quantifiable_universe),
            "investable_universe": list(self.investable_universe),
            "metadata": dict(self.metadata),
        }
        if include_branch_data:
            payload["branch_data"] = dict(_json_safe(self.branch_data))
        return dict(_json_safe(payload))


def assess_quant_readiness(
    *,
    frames: Mapping[str, pd.DataFrame],
    read_results: Mapping[str, Any] | None = None,
    symbols: Sequence[str] | None = None,
    as_of: str = "",
) -> BranchDataReadiness:
    universe = [_normalize_symbol(symbol) for symbol in (symbols or frames.keys()) if _normalize_symbol(symbol)]
    target_date = _compact_date(as_of)
    pass_symbols: list[str] = []
    affected: list[str] = []
    missing_fields: set[str] = set()
    blockers: list[str] = []
    latest_dates: dict[str, str] = {}
    for symbol in universe:
        frame = frames.get(symbol)
        if frame is None and "." in symbol:
            frame = frames.get(symbol.split(".", 1)[0])
        if frame is None or frame.empty:
            affected.append(symbol)
            blockers.append("quant_empty_frame")
            continue
        columns = {str(column).strip().lower() for column in frame.columns}
        aliases = {"volume": {"volume", "vol"}, "amount": {"amount", "amt"}}
        symbol_missing: list[str] = []
        for field_name in QUANT_REQUIRED_FIELDS:
            allowed = aliases.get(field_name, {field_name})
            if not columns.intersection(allowed):
                symbol_missing.append(field_name)
                missing_fields.add(field_name)
        latest = _latest_frame_date(frame)
        latest_dates[symbol] = latest
        if target_date and latest and latest < target_date:
            symbol_missing.append("freshness")
            missing_fields.add("freshness")
        if symbol_missing:
            affected.append(symbol)
            blockers.append(f"quant_missing_or_stale:{symbol}")
            continue
        pass_symbols.append(symbol)
    coverage = len(pass_symbols) / max(len(universe), 1)
    unique_blockers = sorted(set(blockers))
    status = STATUS_PASS if coverage >= 0.995 and not unique_blockers else STATUS_BLOCK
    if not universe:
        unique_blockers.append("empty_quant_universe")
        status = STATUS_BLOCK
    freshness = "fresh" if not target_date or all(not latest or latest >= target_date for latest in latest_dates.values()) else "stale"
    if read_results:
        issue_count = sum(len(getattr(result, "issues", []) or []) for result in read_results.values())
        if issue_count:
            unique_blockers.append("market_data_read_diagnostics_present")
            status = STATUS_BLOCK
    return BranchDataReadiness(
        branch="quant",
        status=status,
        coverage_ratio=coverage,
        freshness_status=freshness,
        pit_status="daily_bar_snapshot",
        source_priority=SOURCE_TUSHARE,
        source="strict_parquet_canonical_bars",
        as_of=_date_text(as_of),
        required_fields=list(QUANT_REQUIRED_FIELDS),
        missing_fields=sorted(missing_fields),
        blockers=unique_blockers,
        affected_symbols=sorted(set(affected)),
        fallback_used=False,
        provider_status="strict_parquet_snapshot",
        metadata={"latest_dates": latest_dates, "symbol_count": len(universe), "pass_count": len(pass_symbols)},
    )


def _read_latest_manifest(root: Path) -> dict[str, Any]:
    path = root / "latest_manifest.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _resolve_parquet_table_root(root: str | Path, table_name: str) -> Path:
    path = Path(root).expanduser()
    if path.suffix.lower() == ".parquet":
        return path
    if table_name == "fundamental_daily":
        if load_fundamental_pointer(path) is not None:
            return resolve_fundamental_table_path(path, table_name)
    if (path / "part.parquet").exists():
        return path
    if table_name == "fundamental_daily":
        return resolve_fundamental_table_path(path, table_name)
    if path.name == table_name:
        return path
    return path / table_name


def _read_parquet_table(table_path: Path) -> pd.DataFrame:
    if not table_path.exists():
        return pd.DataFrame()
    if table_path.is_file():
        try:
            return pd.read_parquet(table_path)
        except Exception:
            return pd.DataFrame()
    part_path = table_path / "part.parquet"
    if part_path.exists():
        try:
            return pd.read_parquet(part_path)
        except Exception:
            return pd.DataFrame()
    try:
        return pd.read_parquet(table_path)
    except Exception:
        frames: list[pd.DataFrame] = []
        for path in sorted(table_path.rglob("*.parquet")):
            try:
                frames.append(pd.read_parquet(path))
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)


def _manifest_from_parquet(table_name: str, frame: pd.DataFrame, table_path: Path) -> dict[str, Any]:
    if frame.empty:
        return {}
    source = "parquet_canonical"
    if "source" in frame.columns:
        sources = frame["source"].dropna().astype(str).str.strip()
        source = str(sources.iloc[0]) if not sources.empty else source
    source_priority = ""
    if "source_priority" in frame.columns:
        priorities = frame["source_priority"].dropna().astype(str).str.strip()
        source_priority = str(priorities.iloc[0]) if not priorities.empty else ""
    return {
        "provider_status": source,
        "source": source,
        "source_priority": source_priority or _source_priority(source),
        "storage_backend": "parquet_canonical",
        "table": table_name,
        "table_path": str(table_path),
        "daily_rows": int(len(frame)),
    }


def _latest_records_by_symbol(
    frame: pd.DataFrame,
    *,
    symbols: Sequence[str],
    as_of: str,
) -> dict[str, dict[str, Any]]:
    if frame.empty or "ts_code" not in frame.columns:
        return {}
    working = frame.copy()
    working["ts_code"] = working["ts_code"].map(_normalize_symbol)
    if "trade_date" not in working.columns:
        working["trade_date"] = working.get("date", "")
    working["_trade_date"] = working["trade_date"].map(_compact_date)
    target = _compact_date(as_of)
    if target:
        working = working[working["_trade_date"].astype(str) <= target]
    records: dict[str, dict[str, Any]] = {}
    wanted = {_normalize_symbol(symbol) for symbol in symbols}
    for symbol, group in working[working["ts_code"].isin(wanted)].groupby("ts_code", sort=True):
        row = group.sort_values("_trade_date").iloc[-1].to_dict()
        row.pop("_trade_date", None)
        records[str(symbol)] = dict(_json_safe(row))
    return records


def load_fundamental_records(
    symbols: Sequence[str],
    *,
    as_of: str = "",
    root: str | Path = DEFAULT_FUNDAMENTAL_ROOT,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    table_path = _resolve_parquet_table_root(root, "fundamental_daily")
    frame = _read_parquet_table(table_path)
    pointer = load_fundamental_pointer(root)
    manifest = (
        dict(pointer.get("metadata", {}) or {})
        if pointer is not None
        else _read_latest_manifest(table_path)
    ) or _manifest_from_parquet("fundamental_daily", frame, table_path)
    if pointer is not None:
        manifest.update(
            {
                "generation_id": pointer.get("generation_id"),
                "pointer_path": pointer.get("pointer_path"),
                "storage_backend": "parquet_canonical_generation",
            }
        )
    return _latest_records_by_symbol(frame, symbols=symbols, as_of=as_of), manifest


def load_macro_record(
    *,
    as_of: str = "",
    root: str | Path = DEFAULT_MACRO_ROOT,
) -> tuple[dict[str, Any], dict[str, Any]]:
    table_path = _resolve_parquet_table_root(root, "macro_daily")
    frame = _read_parquet_table(table_path)
    manifest = _read_latest_manifest(table_path) or _manifest_from_parquet("macro_daily", frame, table_path)
    if frame.empty:
        return {}, manifest
    working = frame.copy()
    if "trade_date" not in working.columns:
        working["trade_date"] = working.get("date", "")
    working["_trade_date"] = working["trade_date"].map(_compact_date)
    target = _compact_date(as_of)
    if target:
        working = working[working["_trade_date"].astype(str) <= target]
    if working.empty:
        return {}, manifest
    row = working.sort_values("_trade_date").iloc[-1].to_dict()
    row.pop("_trade_date", None)
    return dict(_json_safe(row)), manifest


def _assess_symbol_records(
    *,
    branch: str,
    symbols: Sequence[str],
    records: Mapping[str, Mapping[str, Any]],
    required_fields: Sequence[str],
    manifest: Mapping[str, Any],
    as_of: str,
) -> BranchDataReadiness:
    universe = [_normalize_symbol(symbol) for symbol in symbols if _normalize_symbol(symbol)]
    affected: list[str] = []
    partial_symbols: list[str] = []
    missing_fields: set[str] = set()
    pass_count = 0
    record_count = 0
    for symbol in universe:
        record = dict(records.get(symbol, {}) or {})
        if not record:
            symbol_missing = list(required_fields) + ["record"]
            affected.append(symbol)
        else:
            record_count += 1
            symbol_missing = [field_name for field_name in required_fields if not _is_present(record.get(field_name))]
            if symbol_missing:
                partial_symbols.append(symbol)
            else:
                pass_count += 1
        if symbol_missing:
            missing_fields.update(symbol_missing)
    full_coverage = pass_count / max(len(universe), 1)
    record_coverage = record_count / max(len(universe), 1)
    source = str(manifest.get("provider_status") or manifest.get("source") or "parquet_canonical")
    priority = _source_priority(source, str(manifest.get("source_priority", "")))
    fallback_used = priority != SOURCE_TUSHARE
    blockers = []
    if not records:
        blockers.append(f"{branch}_parquet_table_missing_or_empty")
    if affected:
        blockers.append(f"{branch}_required_fields_missing")
    if fallback_used:
        blockers.append(f"{branch}_not_tushare_primary")
    if blockers:
        status = STATUS_BLOCK
    elif partial_symbols:
        status = STATUS_WARN
    else:
        status = STATUS_PASS if full_coverage >= 1.0 else STATUS_WARN
    return BranchDataReadiness(
        branch=branch,
        status=status,
        coverage_ratio=record_coverage if status == STATUS_WARN else full_coverage,
        freshness_status="fresh_or_pit_asof" if records else "unknown",
        pit_status="point_in_time" if records else "missing",
        source_priority=priority,
        source=source,
        as_of=_date_text(as_of),
        required_fields=list(required_fields),
        missing_fields=sorted(missing_fields),
        blockers=blockers,
        affected_symbols=sorted(set(affected)),
        fallback_used=fallback_used,
        provider_status=str(manifest.get("provider_status") or "local_snapshot"),
        metadata={
            "manifest": dict(manifest),
            "symbol_count": len(universe),
            "record_count": record_count,
            "pass_count": pass_count,
            "partial_symbols": sorted(set(partial_symbols)),
            "record_coverage_ratio": record_coverage,
            "full_field_coverage_ratio": full_coverage,
        },
    )


def assess_macro_readiness(
    *,
    macro_record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    as_of: str = "",
) -> BranchDataReadiness:
    missing = [field_name for field_name in MACRO_REQUIRED_FIELDS if not _is_present(macro_record.get(field_name))]
    source = str(macro_record.get("source") or manifest.get("provider_status") or manifest.get("source") or "parquet_canonical")
    priority = _source_priority(source, str(macro_record.get("source_priority") or manifest.get("source_priority") or ""))
    fallback_used = priority != SOURCE_TUSHARE
    blockers = []
    if not macro_record:
        blockers.append("macro_parquet_table_missing_or_empty")
    if missing:
        blockers.append("macro_required_fields_missing")
    if fallback_used:
        blockers.append("macro_not_tushare_primary")
    return BranchDataReadiness(
        branch="macro",
        status=STATUS_PASS if not blockers else STATUS_BLOCK,
        coverage_ratio=1.0 if not missing and macro_record else 0.0,
        freshness_status="fresh_or_pit_asof" if macro_record else "unknown",
        pit_status="market_point_in_time" if macro_record else "missing",
        source_priority=priority,
        source=source,
        as_of=_date_text(as_of),
        required_fields=list(MACRO_REQUIRED_FIELDS),
        missing_fields=missing,
        blockers=blockers,
        affected_symbols=[],
        fallback_used=fallback_used,
        provider_status=str(manifest.get("provider_status") or "local_snapshot"),
        metadata={"manifest": dict(manifest), "macro_record": dict(_json_safe(macro_record))},
    )


def assess_branch_data_readiness(
    *,
    frames: Mapping[str, pd.DataFrame],
    read_results: Mapping[str, Any] | None = None,
    candidate_symbols: Sequence[str] | None = None,
    market: str = "CN",
    category: str = "full_a",
    as_of: str = "",
    fundamental_root: str | Path = DEFAULT_FUNDAMENTAL_ROOT,
    macro_root: str | Path = DEFAULT_MACRO_ROOT,
    run_id: str | None = None,
) -> BranchGovernanceReport:
    symbols = [_normalize_symbol(symbol) for symbol in (candidate_symbols or frames.keys()) if _normalize_symbol(symbol)]
    run_id = run_id or make_run_id(as_of)
    quant = assess_quant_readiness(
        frames=frames,
        read_results=read_results,
        symbols=symbols,
        as_of=as_of,
    )
    fundamentals, fundamental_manifest = load_fundamental_records(symbols, as_of=as_of, root=fundamental_root)
    macro_record, macro_manifest = load_macro_record(as_of=as_of, root=macro_root)
    fundamental = _assess_symbol_records(
        branch="fundamental",
        symbols=symbols,
        records=fundamentals,
        required_fields=FUNDAMENTAL_REQUIRED_FIELDS,
        manifest=fundamental_manifest,
        as_of=as_of,
    )
    macro = assess_macro_readiness(macro_record=macro_record, manifest=macro_manifest, as_of=as_of)
    blocked = sorted(
        set(quant.affected_symbols)
        | set(fundamental.affected_symbols)
        | (set(symbols) if macro.status == STATUS_BLOCK else set())
    )
    quantifiable = [
        symbol
        for symbol in symbols
        if _normalize_symbol(symbol) and _normalize_symbol(symbol) not in set(quant.affected_symbols)
    ]
    investable = [symbol for symbol in symbols if symbol not in set(blocked)]
    branch_data = {
        "fundamentals": fundamentals,
        "macro_data": macro_record,
    }
    return BranchGovernanceReport(
        run_id=run_id,
        market=str(market).upper(),
        category=str(category or "full_a"),
        as_of=_date_text(as_of),
        readiness={
            "quant": quant,
            "fundamental": fundamental,
            "macro": macro,
        },
        blocked_symbols=blocked,
        quantifiable_universe=sorted(set(quantifiable)),
        investable_universe=investable,
        branch_data=branch_data,
        metadata={
            "policy": "strict_when_used",
            "source_priority_order": list(SOURCE_PRIORITY_ORDER),
            "branch_schema_version": BRANCH_SCHEMA_VERSION,
            "canonical_branch_order": list(CANONICAL_BRANCH_ORDER),
            "candidate_count": len(symbols),
            "blocked_count": len(blocked),
            "investable_count": len(investable),
        },
    )


def render_branch_readiness_md(report: BranchGovernanceReport) -> str:
    lines = [
        "# Branch Data Readiness",
        "",
        f"- Run: `{report.run_id}`",
        f"- Market: `{report.market}`",
        f"- Category: `{report.category}`",
        f"- As of: `{report.as_of}`",
        f"- Policy: `{report.metadata.get('policy', 'strict_when_used')}`",
        f"- Quantifiable universe: {len(report.quantifiable_universe)}",
        f"- Investable universe: {len(report.investable_universe)}",
        f"- Blocked symbols: {len(report.blocked_symbols)}",
        "",
        "| Branch | Status | Coverage | PIT | Source priority | Blockers |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for branch, readiness in report.readiness.items():
        lines.append(
            "| {branch} | {status} | {coverage:.2%} | {pit} | {source_priority} | {blockers} |".format(
                branch=branch,
                status=readiness.status,
                coverage=readiness.coverage_ratio,
                pit=readiness.pit_status,
                source_priority=readiness.source_priority,
                blockers=", ".join(readiness.blockers) or "-",
            )
        )
    return "\n".join(lines) + "\n"


def write_branch_readiness_report(
    report: BranchGovernanceReport,
    *,
    output_dir: str | Path = DEFAULT_READINESS_ROOT,
) -> dict[str, str]:
    out = Path(output_dir)
    resolved_out = out.resolve(strict=False)
    if (
        resolved_out == _FROZEN_READINESS_ROOT
        or _FROZEN_READINESS_ROOT in resolved_out.parents
    ):
        raise ValueError(
            "reports/branch_readiness is frozen v13 retirement evidence; "
            "write current artifacts under reports/v14/branch_readiness"
        )
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"{report.run_id}.json"
    md_path = out / f"{report.run_id}.md"
    csv_path = out / f"{report.run_id}.csv"
    json_path.write_text(json.dumps(report.to_dict(include_branch_data=True), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_branch_readiness_md(report), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "branch",
                "status",
                "coverage_ratio",
                "freshness_status",
                "pit_status",
                "source_priority",
                "source",
                "missing_fields",
                "blockers",
                "affected_symbols",
            ],
        )
        writer.writeheader()
        for readiness in report.readiness.values():
            writer.writerow(
                {
                    "branch": readiness.branch,
                    "status": readiness.status,
                    "coverage_ratio": readiness.coverage_ratio,
                    "freshness_status": readiness.freshness_status,
                    "pit_status": readiness.pit_status,
                    "source_priority": readiness.source_priority,
                    "source": readiness.source,
                    "missing_fields": ";".join(readiness.missing_fields),
                    "blockers": ";".join(readiness.blockers),
                    "affected_symbols": ";".join(readiness.affected_symbols[:128]),
                }
            )
    return {"json": str(json_path), "md": str(md_path), "csv": str(csv_path)}


__all__ = [
    "BranchDataReadiness",
    "BranchGovernanceReport",
    "FUNDAMENTAL_REQUIRED_FIELDS",
    "MACRO_REQUIRED_FIELDS",
    "QUANT_REQUIRED_FIELDS",
    "SOURCE_OFFLINE",
    "SOURCE_PRIORITY_ORDER",
    "SOURCE_PUBLIC_FALLBACK",
    "SOURCE_TUSHARE",
    "STATUS_BLOCK",
    "STATUS_PASS",
    "STATUS_WARN",
    "assess_branch_data_readiness",
    "assess_macro_readiness",
    "assess_quant_readiness",
    "load_fundamental_records",
    "load_macro_record",
    "write_branch_readiness_report",
]
