"""Real-data gap repair restricted to an explicit scope transition's additions."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any, Mapping

from .scope_transition import _bytes, _write_new, encoded, read_ref, reference

REQUIRED_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "vol",
    "amount",
    "adj_factor",
    "turnover_rate",
    "total_mv",
    "circ_mv",
)
NULLABLE_FIELDS = ("pe", "pb", "volume_ratio")
ACQUISITION_KEYS = {
    "schema_version",
    "request_sha256",
    "operation_id",
    "target_date",
    "before_market_pointer_ref",
    "before_pointer_artifact_ref",
    "before_manifest_ref",
    "incoming_ref",
    "calendar_ref",
    "added",
    "listing_dates",
    "expected_symbol_date_count",
    "repaired_row_count",
    "explained_missing",
    "suspension_evidence",
}


def validate_history_fields(frame) -> dict[str, Any]:
    """Explicit field contract: valuation ratios may be unavailable; core inputs may not."""
    import numpy as np
    import pandas as pd

    missing = set(REQUIRED_FIELDS + NULLABLE_FIELDS) - set(frame.columns)
    if missing:
        raise RuntimeError(f"SCOPE_HISTORY_REQUIRED_COLUMNS_MISSING:{sorted(missing)}")
    for col in REQUIRED_FIELDS:
        values = pd.to_numeric(frame[col], errors="coerce")
        positive = col in {"open", "high", "low", "close", "pre_close", "adj_factor", "total_mv"}
        if not np.isfinite(values).all() or ((values <= 0) if positive else (values < 0)).any():
            raise RuntimeError(f"SCOPE_HISTORY_REQUIRED_FIELD_INVALID:{col}")
    for col in NULLABLE_FIELDS:
        values = pd.to_numeric(frame[col].dropna(), errors="coerce")
        if not np.isfinite(values).all():
            raise RuntimeError(f"SCOPE_HISTORY_OPTIONAL_FIELD_INVALID:{col}")
    return {
        "required_finite_fields": list(REQUIRED_FIELDS),
        "permitted_null_fields": list(NULLABLE_FIELDS),
        "null_counts": {col: int(frame[col].isna().sum()) for col in NULLABLE_FIELDS},
    }


def _history_frame(payload, added):
    import pyarrow.dataset as ds

    frame = (
        ds.dataset(payload["table_root"], format="parquet")
        .to_table(
            columns=["ts_code", "trade_date", *REQUIRED_FIELDS, *NULLABLE_FIELDS],
            filter=ds.field("ts_code").isin(added),
        )
        .to_pandas()
    )
    if frame.duplicated(["ts_code", "trade_date"]).any():
        raise RuntimeError("SCOPE_HISTORY_CANONICAL_DUPLICATE_KEYS")
    validate_history_fields(frame)
    return frame


def _validate_acquisition(evidence, q, op, expected, listing_dates):
    import pandas as pd
    from .cn_nontrading_evidence import canonical_json_sha256

    if (
        set(evidence) != ACQUISITION_KEYS
        or evidence["schema_version"] != "cn-scope-history-acquisition.v1"
        or evidence["request_sha256"] != op.name
        or evidence["operation_id"] != q["operation_id"]
        or evidence["target_date"] != q["effective_date"]
        or evidence["added"] != q["added"]
        or evidence["listing_dates"] != {s: listing_dates[s] for s in q["added"]}
        or evidence["expected_symbol_date_count"] != len(expected)
    ):
        raise RuntimeError("SCOPE_HISTORY_ACQUISITION_BINDING_INVALID")
    fixed = {
        "incoming_ref": op / "history-incoming.parquet",
        "calendar_ref": op / "history-calendar.json",
        "before_pointer_artifact_ref": op / "history-before-pointer.json",
    }
    for key, path in fixed.items():
        if evidence[key]["path"] != str(path):
            raise RuntimeError("SCOPE_HISTORY_ARTIFACT_PATH_MISMATCH")
        read_ref(evidence[key])
    old = json.loads(read_ref(evidence["before_pointer_artifact_ref"]))
    if (
        evidence["before_market_pointer_ref"]["path"]
        != str(Path(q["workspace_root"]) / "data/parquet/cn/_latest.json")
        or evidence["before_pointer_artifact_ref"]["sha256"]
        != evidence["before_market_pointer_ref"]["sha256"]
        or old["manifest_path"] != evidence["before_manifest_ref"]["path"]
    ):
        raise RuntimeError("SCOPE_HISTORY_PREDECESSOR_BINDING_INVALID")
    read_ref(evidence["before_manifest_ref"])
    old_frame = _history_frame(old, q["added"])
    old_keys = set(zip(old_frame.ts_code, old_frame.trade_date))
    explained_rows = evidence["explained_missing"]
    if explained_rows != sorted([list(x) for x in {tuple(x) for x in explained_rows}]):
        raise RuntimeError("SCOPE_HISTORY_EXPLANATION_KEYS_INVALID")
    explained = {tuple(x) for x in explained_rows}
    if not explained <= expected - old_keys:
        raise RuntimeError("SCOPE_HISTORY_EXPLANATIONS_OUTSIDE_MISSING_KEYS")
    verified = set()
    for date, descriptor in evidence["suspension_evidence"].items():
        if set(descriptor) != {"symbols", "evidence_ref"}:
            raise RuntimeError("SCOPE_HISTORY_SUSPENSION_DESCRIPTOR_INVALID")
        source = json.loads(read_ref(descriptor["evidence_ref"]))
        body = dict(source)
        declared = body.pop("payload_sha256", None)
        events = source.get("exact_event_records", [])
        suspend = sorted({e["ts_code"] for e in events if e.get("suspend_type") == "S"})
        if (
            declared != canonical_json_sha256(body)
            or source.get("version") != 5
            or source.get("trade_date") != date
            or source.get("source") != "tushare.suspend_d"
            or source.get("query_succeeded") is not True
            or source.get("query_params") != {"trade_date": date}
            or source.get("exact_date_rows_validated") is not True
            or source.get("exact_event_row_count") != len(events)
            or source.get("exact_event_records_sha256") != canonical_json_sha256(events)
            or any(e.get("trade_date") != date for e in events)
            or source.get("symbols") != suspend
            or not set(descriptor["symbols"]) <= set(suspend)
        ):
            raise RuntimeError("SCOPE_HISTORY_SUSPENSION_EVIDENCE_INVALID")
        verified.update((s, date) for s in descriptor["symbols"])
    if explained != verified:
        raise RuntimeError("SCOPE_HISTORY_EXPLANATION_EVIDENCE_MISMATCH")
    incoming = pd.read_parquet(evidence["incoming_ref"]["path"])
    incoming_keys = set(zip(incoming.ts_code, incoming.trade_date)) if not incoming.empty else set()
    if not incoming.empty:
        validate_history_fields(incoming)
        if incoming.duplicated(["ts_code", "trade_date"]).any():
            raise RuntimeError("SCOPE_HISTORY_DUPLICATE_INCOMING_KEYS")
    if (
        incoming_keys != expected - old_keys - explained
        or len(incoming) != evidence["repaired_row_count"]
    ):
        raise RuntimeError("SCOPE_HISTORY_INCOMING_KEYSET_MISMATCH")
    return incoming, explained


def _history_result(evidence, pointer, frame, acquisition, *, recovered=False):
    return {
        **evidence,
        "after_market_pointer_ref": reference(pointer),
        "acquisition_ref": reference(acquisition),
        "unresolved_gap_count": 0,
        "symbol_coverage": frame.groupby("ts_code")
        .trade_date.agg(["min", "max", "count"])
        .to_dict("index"),
        "synthetic_bar_count": 0,
        "field_coverage": validate_history_fields(frame),
        "recovered": recovered,
    }


def repair_and_verify_added_history(
    q: Mapping[str, Any], *, operation_root: Path
) -> dict[str, Any]:
    import pandas as pd

    from .daily_components import _default_provider_factory, _default_suspension_loader
    from .download import CNParquetBatchMaintainer
    from .market_data_store import MarketDataStore

    root = Path(q["workspace_root"])
    pointer = root / "data/parquet/cn/_latest.json"
    completed = operation_root / "history-repair.json"
    if completed.exists():
        completed_payload = json.loads(_bytes(completed))
        current_ref = completed_payload.get("after_market_pointer_ref", {})
        if current_ref.get("path") != str(pointer):
            raise RuntimeError("SCOPE_HISTORY_COMPLETION_POINTER_INVALID")
        read_ref(current_ref)
    capture = json.loads(read_ref(q["pit_capture_ref"]))
    listed = json.loads(_bytes(capture["partitions"][0]["path"]))["items"]
    listing_dates = {row["ts_code"]: row["list_date"] for row in listed}
    added = list(q["added"])
    if not added:
        raise RuntimeError("SCOPE_TRANSITION_NO_ADDITIONS")
    start, end = min(listing_dates[s] for s in added), q["effective_date"]
    provider = _default_provider_factory()
    calendar_file = operation_root / "history-calendar.json"
    if calendar_file.exists():
        calendar = json.loads(_bytes(calendar_file))
    else:
        frame = provider.trade_cal(
            exchange="SSE",
            start_date=start,
            end_date=end,
            fields="exchange,cal_date,is_open,pretrade_date",
        )
        if frame is None or frame.empty:
            raise RuntimeError("SCOPE_HISTORY_CALENDAR_EMPTY")
        calendar = json.loads(frame.to_json(orient="records"))
        _write_new(calendar_file, encoded(calendar))
    days = []
    cursor = datetime.strptime(start, "%Y%m%d")
    while cursor.strftime("%Y%m%d") <= end:
        days.append(cursor.strftime("%Y%m%d"))
        cursor += timedelta(days=1)
    if sorted(row["cal_date"] for row in calendar) != days or any(
        row["exchange"] != "SSE" or row["is_open"] not in (0, 1) for row in calendar
    ):
        raise RuntimeError("SCOPE_HISTORY_CALENDAR_INCOMPLETE")
    open_dates = sorted(row["cal_date"] for row in calendar if row["is_open"] == 1)
    if not open_dates or open_dates[-1] != end:
        raise RuntimeError("SCOPE_HISTORY_TARGET_NOT_OPEN")

    def read_keys() -> tuple[dict[str, Any], pd.DataFrame, set[tuple[str, str]]]:
        payload = json.loads(_bytes(pointer))
        frame = _history_frame(payload, added)
        return payload, frame, set(zip(frame.ts_code, frame.trade_date))

    before_ref = reference(pointer)
    before, before_frame, keys = read_keys()
    expected = {(s, d) for s in added for d in open_dates if listing_dates[s] <= d}
    missing = expected - keys
    reasons: dict[str, dict[str, Any]] = {}
    repair_frames = []
    acquisition = operation_root / "history-acquisition.json"
    incoming_file = operation_root / "history-incoming.parquet"
    maintainer = CNParquetBatchMaintainer(
        data_dir=str(operation_root / "history-fetch"), data_root=root / "data"
    )
    maintainer.downloader.pro = provider
    if acquisition.exists():
        evidence = json.loads(_bytes(acquisition))
        incoming, explained = _validate_acquisition(
            evidence, q, operation_root, expected, listing_dates
        )
        if completed.exists():
            result = json.loads(_bytes(completed))
            if type(result.get("recovered")) is not bool:
                raise RuntimeError("SCOPE_HISTORY_COMPLETION_SCHEMA_INVALID")
            recomputed = _history_result(
                evidence, pointer, before_frame, acquisition, recovered=result["recovered"]
            )
            if result != recomputed or expected - keys - explained:
                raise RuntimeError("SCOPE_HISTORY_COMPLETION_REPLAY_MISMATCH")
            read_ref(result["acquisition_ref"])
            return result
        if evidence["before_market_pointer_ref"] != before_ref:
            # Recover a completed CAS whose completion receipt was interrupted.
            manifest = json.loads(_bytes(before["manifest_path"]))
            meta = manifest.get("metadata", {})
            if (
                meta.get("scope_transition_request_sha256") != operation_root.name
                or meta.get("scope_history_acquisition_ref") != reference(acquisition)
                or meta.get("expected_previous_latest_pointer_sha256")
                != evidence["before_market_pointer_ref"]["sha256"]
            ):
                raise RuntimeError("SCOPE_HISTORY_POINTER_DRIFT")
            if not (expected - keys - {tuple(x) for x in evidence["explained_missing"]}):
                result = _history_result(
                    evidence, pointer, before_frame, acquisition, recovered=True
                )
                _write_new(completed, encoded(result))
                return result
            raise RuntimeError("SCOPE_HISTORY_RECOVERY_INCOMPLETE")
    else:
        if completed.exists():
            raise RuntimeError("SCOPE_HISTORY_ACQUISITION_MISSING")
        _write_new(operation_root / "history-before-pointer.json", _bytes(pointer))
        explained = set()
        fields = {
            "daily": "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
            "adj_factor": "ts_code,trade_date,adj_factor",
            "daily_basic": "ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv",
        }
        for date in sorted({d for _, d in missing}):
            wanted = {s for s, d in missing if d == date}
            frames = {}
            for api, columns in fields.items():
                frame, error = maintainer._fetch_endpoint(api, date, columns)
                if error or frame is None or frame.empty:
                    raise RuntimeError(f"SCOPE_HISTORY_ENDPOINT_UNAVAILABLE:{date}:{api}")
                frames[api] = frame
            bars = maintainer._build_bars_frame(
                frames["daily"], frames["adj_factor"], frames["daily_basic"]
            )
            selected = bars.loc[bars.ts_code.isin(wanted)].copy()
            validate_history_fields(selected)
            observed = set(selected.ts_code)
            absent = wanted - observed
            if absent:
                suspended, evidence_path = _default_suspension_loader(
                    provider, date, operation_root / "history-suspensions"
                )
                if not absent <= suspended:
                    raise RuntimeError(
                        f"SCOPE_HISTORY_UNEXPLAINED_GAPS:{date}:{sorted(absent-suspended)}"
                    )
                explained.update((s, date) for s in absent)
                reasons[date] = {
                    "symbols": sorted(absent),
                    "evidence_ref": reference(evidence_path),
                }
            for api in ("daily_basic", "adj_factor"):
                if not observed <= set(frames[api].ts_code):
                    raise RuntimeError(f"SCOPE_HISTORY_AUXILIARY_COVERAGE_MISSING:{date}:{api}")
            repair_frames.append(selected)
        incoming = pd.concat(repair_frames, ignore_index=True) if repair_frames else pd.DataFrame()
        if not incoming.empty:
            if set(zip(incoming.ts_code, incoming.trade_date)) != missing - explained:
                raise RuntimeError("SCOPE_HISTORY_REPAIR_KEYSET_MISMATCH")
            if incoming.duplicated(["ts_code", "trade_date"]).any():
                raise RuntimeError("SCOPE_HISTORY_DUPLICATE_ROWS")
        incoming.to_parquet(incoming_file, index=False)
        incoming_file.chmod(0o600)
        evidence = {
            "schema_version": "cn-scope-history-acquisition.v1",
            "request_sha256": operation_root.name,
            "operation_id": q["operation_id"],
            "target_date": end,
            "before_market_pointer_ref": before_ref,
            "before_pointer_artifact_ref": reference(
                operation_root / "history-before-pointer.json"
            ),
            "before_manifest_ref": reference(before["manifest_path"]),
            "incoming_ref": reference(incoming_file),
            "calendar_ref": reference(calendar_file),
            "added": added,
            "listing_dates": {s: listing_dates[s] for s in added},
            "expected_symbol_date_count": len(expected),
            "repaired_row_count": len(incoming),
            "explained_missing": sorted([list(x) for x in explained]),
            "suspension_evidence": reasons,
        }
        _write_new(acquisition, encoded(evidence))
        _validate_acquisition(evidence, q, operation_root, expected, listing_dates)
    if not incoming.empty:
        dates = sorted(set(incoming.trade_date))
        if dates[-1] >= end:
            raise RuntimeError("SCOPE_HISTORY_CURRENT_CLOSE_MISSING")
        if reference(pointer) != before_ref:
            raise RuntimeError("SCOPE_HISTORY_POINTER_DRIFT")
        maintainer.store.upsert_bars(
            incoming,
            target_trade_date=dates[-1],
            target_trade_dates=dates,
            source="cn_scope_transition_historical_gap_repair",
            expected_latest_pointer_sha256=before_ref["sha256"],
            metadata={
                "status": "OK",
                "latest_available_trade_date": end,
                "latest_complete_trade_date": end,
                "blockers": [],
                "scope_transition_operation_id": q["operation_id"],
                "scope_transition_request_sha256": operation_root.name,
                "scope_history_acquisition_ref": reference(acquisition),
                "coverage": {
                    "coverage_trade_date": dates[-1],
                    "historical_repair_trade_dates": dates,
                    "historical_repair_row_count": len(incoming),
                },
            },
        )
    after, frame, after_keys = read_keys()
    gaps = expected - after_keys - explained
    if gaps or after["latest_complete_trade_date"] != end:
        raise RuntimeError("SCOPE_HISTORY_POST_REPAIR_GAPS")
    validation = MarketDataStore(market="CN", data_root=root / "data").validate_latest()
    if validation.get("status") != "passed":
        raise RuntimeError("SCOPE_HISTORY_STORAGE_INVALID")
    result = _history_result(evidence, pointer, frame, acquisition)
    _write_new(completed, encoded(result))
    return result
