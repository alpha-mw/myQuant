"""Registered component adapters for the post-close CN maintenance DAG."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable, Mapping

from .daily_maintenance import MaintenanceComponents, MaintenanceContext


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_bytes(path: str | Path) -> bytes:
    candidate = Path(path).expanduser()
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(candidate, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("daily_component_input_not_regular")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RuntimeError("daily_component_input_changed")
    return b"".join(chunks)


def _json_object(path: str | Path) -> tuple[dict[str, Any], bytes, str]:
    raw = _stable_bytes(path)
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("daily_component_json_invalid") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("daily_component_json_invalid")
    return dict(payload), raw, _sha(raw)


def _write_private_json(path: Path, payload: Mapping[str, Any]) -> tuple[str, str]:
    raw = (
        json.dumps(
            dict(payload),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        offset = 0
        while offset < len(raw):
            written = os.write(descriptor, raw[offset:])
            if written <= 0:
                raise RuntimeError("daily_component_write_short")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return str(path), _sha(raw)


def _default_provider_factory() -> Any:
    import tushare as ts

    from quant_investor.config import config
    from quant_investor.credential_utils import create_tushare_pro

    token = str(getattr(config, "TUSHARE_TOKEN", "") or "").strip()
    provider = create_tushare_pro(ts, token, getattr(config, "TUSHARE_URL", None))
    if provider is None:
        raise RuntimeError("TUSHARE_PROVIDER_UNAVAILABLE")
    return provider


def _default_suspension_loader(provider: Any, target: str, root: Path) -> tuple[set[str], Path]:
    from .download_cn import CNDataFetcher

    root.mkdir(parents=True, mode=0o700)
    fetcher = CNDataFetcher.__new__(CNDataFetcher)
    fetcher.data_dir = str(root)
    fetcher.pro = provider
    fetcher._latest_suspended_symbols_cache = {}
    fetcher._download_data_quality_issues = []
    fetcher._suspend_query_run_id = f"daily-{target}"
    symbols = fetcher._load_latest_suspended_symbols(
        target,
        force_refresh=True,
        query_run_id=f"daily-{target}",
    )
    return set(symbols), fetcher._suspend_cache_path(target)


@dataclass(frozen=True)
class DailyComponentAPIs:
    provider_factory: Callable[[], Any]
    pit_store_factory: Callable[[Path], Any]
    pit_acquire: Callable[..., Mapping[str, Any]]
    pit_validate: Callable[..., Mapping[str, Any]]
    pit_publish: Callable[..., Mapping[str, Any]]
    market_capture: Callable[..., Mapping[str, Any]]
    market_replay: Callable[..., Mapping[str, Any]]
    market_publish: Callable[..., Mapping[str, Any]]
    market_shadow: Callable[..., Mapping[str, Any]]
    history_audit: Callable[..., Any]
    macro_prepare: Callable[..., Mapping[str, Any]]
    macro_commit: Callable[..., Mapping[str, Any]]
    macro_recover: Callable[..., Mapping[str, Any]]
    macro_load_release: Callable[..., Any]
    macro_load_observations: Callable[..., Any]
    suspension_loader: Callable[[Any, str, Path], tuple[set[str], Path]]


class _DailyFrameCache:
    def __init__(self, provider: Any, frames: Mapping[str, Any]) -> None:
        self._provider = provider
        self._frames = dict(frames)

    def daily(self, **kwargs: Any) -> Any:
        trade_date = _compact_date(kwargs.get("trade_date"))
        if trade_date in self._frames:
            frame = self._frames[trade_date]
            return frame.copy() if hasattr(frame, "copy") else frame
        return self._provider.daily(**kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


def _sealed_nontrading_evidence(
    *,
    provider: Any,
    trade_date: str,
    scope_symbols: list[str],
    excluded_symbols: set[str],
    pit_binding: Mapping[str, Any],
    evidence_root: Path,
) -> tuple[Any, list[str], dict[str, Any] | None]:
    import pandas as pd

    from .cn_nontrading_evidence import (
        build_bak_daily_nontrading_evidence,
        evidence_cache_path,
        validate_bak_daily_nontrading_evidence,
        write_evidence_cache,
    )

    query = {"trade_date": trade_date}
    daily_frame = provider.daily(**query)
    if daily_frame is None or not isinstance(daily_frame, pd.DataFrame):
        daily_frame = pd.DataFrame()
    if "ts_code" not in daily_frame.columns and not daily_frame.empty:
        raise RuntimeError("DAILY_PRECAPTURE_SCHEMA_INVALID")
    observed = {
        str(value or "").strip().upper()
        for value in daily_frame.get("ts_code", pd.Series(dtype=str)).tolist()
        if str(value or "").strip()
    }
    primary_missing = sorted(set(scope_symbols) - observed - set(excluded_symbols))
    if not primary_missing:
        return _DailyFrameCache(provider, {trade_date: daily_frame}), [], None
    bak_frame = provider.bak_daily(**query)
    if bak_frame is None or not isinstance(bak_frame, pd.DataFrame):
        bak_frame = pd.DataFrame()
    if len(bak_frame) > 10_000:
        raise RuntimeError("BAK_DAILY_RESPONSE_BOUND_EXCEEDED")
    payload = build_bak_daily_nontrading_evidence(
        bak_frame,
        trade_date=trade_date,
        primary_missing_symbols=primary_missing,
        query_params=query,
        pit_membership_path=pit_binding["canonical_path"],
        pit_membership_sha256=pit_binding["canonical_sha256"],
    )
    blockers = validate_bak_daily_nontrading_evidence(
        payload,
        trade_date=trade_date,
        primary_missing_symbols=primary_missing,
        pit_membership_sha256=pit_binding["canonical_sha256"],
    )
    if blockers:
        raise RuntimeError("BAK_DAILY_EVIDENCE_INVALID")
    cache_path = evidence_cache_path(
        evidence_root,
        trade_date=trade_date,
        primary_missing_symbols=primary_missing,
        pit_membership_sha256=pit_binding["canonical_sha256"],
    )
    written = write_evidence_cache(cache_path, payload)
    reference = {
        "path": str(written),
        "sha256": _sha(_stable_bytes(written)),
        "payload_sha256": payload["payload_sha256"],
    }
    return (
        _DailyFrameCache(provider, {trade_date: daily_frame}),
        list(payload["verified_symbols"]),
        reference,
    )


def registered_component_apis() -> DailyComponentAPIs:
    from quant_investor.macro.maintenance import (
        commit_prepared_macro_transaction,
        prepare_cn_macro_maintenance_transaction,
        recover_macro_transaction,
    )
    from quant_investor.macro.release_calendar import load_release_calendar
    from quant_investor.macro.store import load_observations
    from quant_investor.market.cn_history_audit import run_cn_history_audit
    from quant_investor.market.market_daily_capture import (
        capture_market_daily,
        publish_market_daily_capture,
        replay_market_daily_capture,
        shadow_market_daily_capture,
    )
    from quant_investor.market.pit_universe import (
        PITUniverseStore,
        acquire_pit_universe_capture,
        publish_pit_universe_capture,
        validate_pit_universe_capture,
    )

    def pit_store_factory(workspace_root: Path) -> PITUniverseStore:
        return PITUniverseStore(
            root_dir=workspace_root / "data/parquet/cn/reference",
            raw_root=workspace_root / "data/cn_universe/raw",
            compatibility_path=(
                workspace_root / "data/cn_universe/stock_basic_membership_latest.json"
            ),
        )

    return DailyComponentAPIs(
        provider_factory=_default_provider_factory,
        pit_store_factory=pit_store_factory,
        pit_acquire=acquire_pit_universe_capture,
        pit_validate=validate_pit_universe_capture,
        pit_publish=publish_pit_universe_capture,
        market_capture=capture_market_daily,
        market_replay=replay_market_daily_capture,
        market_publish=publish_market_daily_capture,
        market_shadow=shadow_market_daily_capture,
        history_audit=run_cn_history_audit,
        macro_prepare=prepare_cn_macro_maintenance_transaction,
        macro_commit=commit_prepared_macro_transaction,
        macro_recover=recover_macro_transaction,
        macro_load_release=load_release_calendar,
        macro_load_observations=load_observations,
        suspension_loader=_default_suspension_loader,
    )


def _scope_reference(workspace: Path) -> tuple[Path, str, list[str]]:
    path = workspace / "data/cn_universe/cn_index_components.json"
    payload, _raw, digest = _json_object(path)
    values = payload.get("full_a")
    if not isinstance(values, list) or not values:
        raise RuntimeError("FULL_A_SCOPE_INVALID")
    symbols = sorted(str(value or "").strip().upper() for value in values)
    if not all(symbols) or len(symbols) != len(set(symbols)):
        raise RuntimeError("FULL_A_SCOPE_INVALID")
    return path.resolve(strict=True), digest, symbols


def _market_reference(data_root: Path) -> dict[str, Any]:
    pointer_path = data_root / "parquet/cn/_latest.json"
    pointer, _raw, pointer_sha = _json_object(pointer_path)
    manifest_text = str(pointer.get("manifest_path") or "")
    if not manifest_text:
        raise RuntimeError("MARKET_MANIFEST_REFERENCE_MISSING")
    declared = Path(manifest_text)
    if declared.is_absolute():
        manifest_path = declared
    elif declared.parts and declared.parts[0] == "data" and data_root.name == "data":
        manifest_path = data_root.parent / declared
    else:
        manifest_path = data_root / declared
    _manifest, _manifest_raw, manifest_sha = _json_object(manifest_path)
    return {
        "data_root": str(data_root),
        "pointer_path": str(pointer_path),
        "pointer_sha256": pointer_sha,
        "pointer": pointer,
        "snapshot_manifest_path": str(manifest_path.resolve(strict=True)),
        "snapshot_manifest_sha256": manifest_sha,
    }


def _stage_evidence(context: MaintenanceContext, stage: str) -> dict[str, Any]:
    for result in reversed(context.prior_stage_results):
        if result.get("stage") == stage:
            evidence = result.get("evidence")
            if isinstance(evidence, Mapping):
                return dict(evidence)
    raise RuntimeError(f"{stage}_EVIDENCE_MISSING")


def _compact_date(value: Any) -> str:
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return digits[:8] if len(digits) >= 8 else ""


def _open_session_window(
    *, close_receipt: Mapping[str, Any], parent_date: str, target_date: str
) -> list[str]:
    raw_dates = close_receipt.get("ordered_open_dates")
    if not isinstance(raw_dates, list):
        raise RuntimeError("CLOSE_OPEN_SESSION_WINDOW_MISSING")
    dates = [str(value or "") for value in raw_dates]
    if (
        not dates
        or any(len(value) != 8 or not value.isdigit() for value in dates)
        or dates != sorted(set(dates))
        or target_date not in dates
        or dates[-1] != target_date
    ):
        raise RuntimeError("CLOSE_OPEN_SESSION_WINDOW_INVALID")
    if parent_date == target_date:
        return []
    if parent_date not in dates:
        raise RuntimeError("MARKET_PARENT_SESSION_OUTSIDE_AUTHORITY_WINDOW")
    parent_index = dates.index(parent_date)
    target_index = dates.index(target_date)
    if target_index <= parent_index:
        raise RuntimeError("MARKET_SESSION_WINDOW_NOT_FORWARD")
    window = dates[parent_index + 1 : target_index + 1]
    if not 1 <= len(window) <= 5:
        raise RuntimeError("MARKET_SESSION_WINDOW_EXCEEDS_BOUND")
    if window != dates[parent_index + 1 : parent_index + 1 + len(window)]:
        raise RuntimeError("MARKET_SESSION_WINDOW_NONADJACENT")
    return window


def _pit_binding(result: Mapping[str, Any], *, mode: str) -> dict[str, str]:
    if mode == "execute":
        return {
            key: str(result.get(key) or "")
            for key in (
                "generation_id",
                "generation_manifest_path",
                "generation_manifest_sha256",
                "canonical_path",
                "canonical_sha256",
                "discovery_pointer_path",
                "discovery_pointer_sha256",
            )
        }
    candidate = dict(result.get("shadow_candidate") or {})
    raw_binding = candidate.get("generation_binding")
    source = dict(raw_binding) if isinstance(raw_binding, Mapping) else candidate
    binding = {
        key: str(source.get(key) or "")
        for key in (
            "generation_id",
            "generation_manifest_path",
            "generation_manifest_sha256",
            "canonical_path",
            "canonical_sha256",
            "discovery_pointer_path",
            "discovery_pointer_sha256",
        )
    }
    if not all(binding.values()):
        raise RuntimeError("PIT_SHADOW_GENERATION_BINDING_UNAVAILABLE")
    return binding


def _pit_reason_sets(
    *,
    context: MaintenanceContext,
    binding: Mapping[str, str],
    scope_symbols: list[str],
    provider: Any,
    apis: DailyComponentAPIs,
    trade_date: str | None = None,
) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    import pandas as pd

    from .pit_universe import (
        PITUniverseRecord,
        REASON_DELISTED,
        REASON_LISTED,
        REASON_PRE_LISTING,
        evaluate_listing_status,
        records_by_symbol,
    )

    effective_date = trade_date or context.target_date
    frame = pd.read_parquet(binding["canonical_path"])
    records = [PITUniverseRecord.from_dict(row) for row in frame.to_dict(orient="records")]
    by_symbol = records_by_symbol(records)
    reason_sets = {
        "suspended": [],
        "non_trading": [],
        "delisted": [],
        "prelisting": [],
        "inactive": [],
    }
    for symbol in scope_symbols:
        status = evaluate_listing_status(by_symbol.get(symbol), symbol=symbol, as_of=effective_date)
        if status.reason == REASON_LISTED:
            continue
        if status.reason == REASON_PRE_LISTING:
            reason_sets["prelisting"].append(symbol)
        elif status.reason == REASON_DELISTED:
            reason_sets["delisted"].append(symbol)
        else:
            reason_sets["inactive"].append(symbol)
    suspended, suspend_path = apis.suspension_loader(
        provider,
        effective_date,
        context.attempt_root / "suspension_evidence",
    )
    reason_sets["suspended"] = sorted(set(scope_symbols) & suspended)
    evidence_payload = {
        "schema_version": "cn-daily-pit-classification-evidence.v1",
        "target_trade_date": effective_date,
        "pit_binding": dict(binding),
        "reason_sets": reason_sets,
    }
    pit_path, pit_sha = _write_private_json(
        context.attempt_root / f"pit-classification-evidence-{effective_date}.json",
        evidence_payload,
    )
    evidence = {
        reason: {"path": pit_path, "sha256": pit_sha}
        for reason in ("delisted", "prelisting", "inactive")
    }
    if suspend_path.is_file():
        evidence["suspended"] = {
            "path": str(suspend_path),
            "sha256": _sha(_stable_bytes(suspend_path)),
        }
    return reason_sets, evidence


class _DefaultComponents:
    def __init__(self, workspace: Path, apis: DailyComponentAPIs) -> None:
        self.workspace = workspace
        self.apis = apis

    def pit(self, context: MaintenanceContext) -> Mapping[str, Any]:
        scope_path, scope_sha, scope_symbols = _scope_reference(self.workspace)
        store = self.apis.pit_store_factory(self.workspace)
        try:
            parent = store.load_generation_binding()
            parent_sha = str(parent.get("discovery_pointer_sha256") or "")
        except RuntimeError as exc:
            if str(exc) != "pit_latest_generation_binding_missing":
                raise
            parent = {}
            parent_sha = ""
        provider = self.apis.provider_factory()
        parent_manifest = dict(parent.get("manifest") or {})
        parent_scope = dict(
            dict(parent_manifest.get("source_bindings") or {}).get("full_a_scope") or {}
        )
        if (
            _compact_date(parent_manifest.get("observed_at")) >= context.target_date
            and parent_scope.get("sha256") == scope_sha
        ):
            binding = _pit_binding(parent, mode="execute")
            reason_sets, classification_evidence = _pit_reason_sets(
                context=context,
                binding=binding,
                scope_symbols=scope_symbols,
                provider=provider,
                apis=self.apis,
            )
            return {
                "status": "NO_ACTION",
                "write_performed": False,
                "blockers": [],
                "evidence": {
                    "scope_path": str(scope_path),
                    "scope_sha256": scope_sha,
                    "pit_binding": binding,
                    "reason_sets": reason_sets,
                    "classification_evidence": classification_evidence,
                    "provider_call_count": 1,
                },
            }
        capture = self.apis.pit_acquire(
            provider,
            capture_root=context.attempt_root / "pit_capture",
            source_run_id=f"daily-{context.target_date}-{context.attempt_slot}",
        )
        common = {
            "store": store,
            "canonical_scope_path": scope_path,
            "expected_scope_sha256": scope_sha,
            "expected_parent_pointer_sha256": parent_sha,
        }
        self.apis.pit_validate(
            capture["capture_receipt_path"],
            capture["capture_receipt_sha256"],
            **common,
        )
        if context.mode == "execute":
            published = self.apis.pit_publish(
                capture["capture_receipt_path"],
                capture["capture_receipt_sha256"],
                canonical=True,
                **common,
            )
        else:
            published = self.apis.pit_publish(
                capture["capture_receipt_path"],
                capture["capture_receipt_sha256"],
                canonical=False,
                shadow_root=(context.attempt_root / "market_candidate" / "data" / "parquet" / "cn"),
                **common,
            )
        try:
            binding = _pit_binding(published, mode=context.mode)
        except RuntimeError as exc:
            if str(exc) != "PIT_SHADOW_GENERATION_BINDING_UNAVAILABLE":
                raise
            return {
                "status": "BLOCKED",
                "write_performed": False,
                "blockers": ["PIT_SHADOW_GENERATION_BINDING_UNAVAILABLE"],
                "evidence": {
                    "capture_receipt_path": capture["capture_receipt_path"],
                    "capture_receipt_sha256": capture["capture_receipt_sha256"],
                },
            }
        reason_sets, classification_evidence = _pit_reason_sets(
            context=context,
            binding=binding,
            scope_symbols=scope_symbols,
            provider=provider,
            apis=self.apis,
        )
        return {
            "status": "READY",
            "write_performed": context.mode == "execute",
            "blockers": [],
            "evidence": {
                "scope_path": str(scope_path),
                "scope_sha256": scope_sha,
                "pit_binding": binding,
                "reason_sets": reason_sets,
                "classification_evidence": classification_evidence,
                "capture_receipt_path": capture["capture_receipt_path"],
                "capture_receipt_sha256": capture["capture_receipt_sha256"],
                "provider_call_count": capture.get("provider_call_count"),
            },
        }

    def market(self, context: MaintenanceContext) -> Mapping[str, Any]:
        pit = _stage_evidence(context, "PIT")
        production_data_root = self.workspace / "data"
        parent = _market_reference(production_data_root)
        parent_date = str(parent["pointer"].get("latest_complete_trade_date") or "")
        session_window = _open_session_window(
            close_receipt=context.close_session_receipt,
            parent_date=parent_date,
            target_date=context.target_date,
        )
        coverage = dict(parent["pointer"].get("coverage") or {})
        pit_binding = dict(pit["pit_binding"])
        pit_binding_matches = (
            coverage.get("pit_generation_id") == pit_binding.get("generation_id")
            and coverage.get("pit_generation_manifest_sha256")
            == pit_binding.get("generation_manifest_sha256")
            and coverage.get("pit_membership_sha256") == pit_binding.get("canonical_sha256")
        )
        if parent_date == context.target_date and pit_binding_matches:
            return {
                "status": "NO_ACTION",
                "write_performed": False,
                "blockers": [],
                "evidence": {**parent, "pit_binding": pit_binding},
            }
        same_target_pit_rebind = parent_date == context.target_date and not pit_binding_matches
        processing_dates = [context.target_date] if same_target_pit_rebind else session_window
        provider = self.apis.provider_factory()
        scope_symbols = _scope_reference(self.workspace)[2]
        reasons_by_date: dict[str, dict[str, list[str]]] = {}
        evidence_by_date: dict[str, dict[str, dict[str, Any]]] = {}
        for trade_date in processing_dates:
            if trade_date == context.target_date:
                reason_sets = {key: list(value) for key, value in dict(pit["reason_sets"]).items()}
                classification_evidence = {
                    key: dict(value) for key, value in dict(pit["classification_evidence"]).items()
                }
            else:
                reason_sets, classification_evidence = _pit_reason_sets(
                    context=context,
                    binding=pit_binding,
                    scope_symbols=scope_symbols,
                    provider=provider,
                    apis=self.apis,
                    trade_date=trade_date,
                )
            excluded = {
                symbol
                for reason, symbols in reason_sets.items()
                if reason != "non_trading"
                for symbol in symbols
            }
            provider, nontrading, nontrading_ref = _sealed_nontrading_evidence(
                provider=provider,
                trade_date=trade_date,
                scope_symbols=scope_symbols,
                excluded_symbols=excluded,
                pit_binding=pit_binding,
                evidence_root=context.attempt_root / "nontrading_evidence",
            )
            reason_sets["non_trading"] = nontrading
            if nontrading_ref is not None:
                classification_evidence["non_trading"] = nontrading_ref
            reasons_by_date[trade_date] = reason_sets
            evidence_by_date[trade_date] = classification_evidence
        capture = self.apis.market_capture(
            provider=provider,
            capture_root=context.attempt_root / "market_capture",
            target_authority_path=context.close_session_receipt_path,
            expected_target_authority_sha256=context.close_session_receipt_sha256,
            scope_path=pit["scope_path"],
            expected_scope_sha256=pit["scope_sha256"],
            pit_generation_binding=pit["pit_binding"],
            expected_market_pointer_sha256=parent["pointer_sha256"],
            reason_sets=reasons_by_date,
            classification_evidence=evidence_by_date,
            target_trade_dates=(
                [context.target_date] if same_target_pit_rebind else session_window
            ),
            parent_latest_complete_trade_date=parent_date,
            same_target_rebind=same_target_pit_rebind,
        )
        replay_kwargs = {
            "capture_manifest_path": capture["manifest_path"],
            "expected_capture_manifest_sha256": capture["manifest_sha256"],
            "scope_path": pit["scope_path"],
            "expected_scope_sha256": pit["scope_sha256"],
            "pit_generation_binding": pit["pit_binding"],
            "expected_market_pointer_sha256": parent["pointer_sha256"],
        }
        self.apis.market_replay(**replay_kwargs)
        if context.mode == "execute":
            publication = self.apis.market_publish(
                data_root=production_data_root,
                **replay_kwargs,
            )
            selected_data_root = production_data_root
        else:
            selected_data_root = context.attempt_root / "market_candidate" / "data"
            publication = self.apis.market_shadow(
                shadow_data_root=selected_data_root,
                production_data_root=production_data_root,
                **replay_kwargs,
            )
            selected_data_root = Path(str(publication["candidate_data_root"]))
        selected = _market_reference(selected_data_root)
        if context.mode == "shadow" and str(publication.get("candidate_pointer_path")) != (
            selected["pointer_path"]
        ):
            raise RuntimeError("MARKET_CANDIDATE_POINTER_MISMATCH")
        return {
            "status": "READY",
            "write_performed": context.mode == "execute",
            "blockers": [],
            "evidence": {
                **selected,
                "pit_binding": pit["pit_binding"],
                "capture_manifest_path": capture["manifest_path"],
                "capture_manifest_sha256": capture["manifest_sha256"],
                "classification": publication.get("classification", {}),
                "same_target_pit_rebind": same_target_pit_rebind,
            },
        }

    def history(self, context: MaintenanceContext) -> Mapping[str, Any]:
        market = _stage_evidence(context, "MARKET")
        kwargs: dict[str, Any] = {
            "data_root": self.workspace / "data",
            "output_root": context.attempt_root / "history_audit",
            "days": 100,
            "end_date": context.target_date,
            "allow_online": True,
        }
        provider = self.apis.provider_factory()

        def suspended_loader(trade_date: str, **_kwargs: Any) -> set[str]:
            symbols, _path = self.apis.suspension_loader(
                provider,
                trade_date,
                context.attempt_root / "history_audit",
            )
            return symbols

        kwargs.update(
            {
                "provider": provider,
                "suspended_loader": suspended_loader,
            }
        )
        candidate_root = Path(str(market["data_root"]))
        production_root = self.workspace / "data"
        if context.mode == "shadow" and candidate_root != production_root:
            kwargs.update(
                {
                    "candidate_data_root": market["data_root"],
                    "candidate_pointer_path": market["pointer_path"],
                }
            )
        report, path = self.apis.history_audit(**kwargs)
        ready = report.get("history_audit_status") == "passed"
        return {
            "status": "READY" if ready else "BLOCKED",
            "write_performed": False,
            "blockers": [] if ready else ["HISTORY_AUDIT_BLOCKED"],
            "evidence": {
                "audit_input_kind": report.get("audit_input_kind"),
                "audit_path": str(path),
                "audit_sha256": _sha(_stable_bytes(path)),
                "history_audit_status": report.get("history_audit_status"),
            },
        }

    def macro(self, context: MaintenanceContext) -> Mapping[str, Any]:
        market = _stage_evidence(context, "MARKET")
        pit = _stage_evidence(context, "PIT")
        release_root = self.workspace / "data/parquet/cn/macro_release_calendar"
        observations_root = self.workspace / "data/parquet/cn/macro_observations"
        release_pointer = release_root / "_latest.json"
        observations_pointer = observations_root / "_latest.json"
        _release_payload, _release_raw, release_sha = _json_object(release_pointer)
        observations_payload, _observations_raw, observations_sha = _json_object(
            observations_pointer
        )
        release_evidence = self.apis.macro_load_release(
            canonical_root=release_root,
            expected_pointer_sha256=release_sha,
        )
        _rows, loaded_observations = self.apis.macro_load_observations(observations_root)
        if _sha(_stable_bytes(observations_pointer)) != observations_sha:
            raise RuntimeError("MACRO_OBSERVATIONS_POINTER_CHANGED_DURING_READ")
        loaded_manifest = dict(loaded_observations.get("generation_manifest") or {})
        metadata = dict(
            loaded_manifest.get("metadata")
            or loaded_observations.get("metadata")
            or observations_payload.get("metadata")
            or {}
        )
        exact_market_binding = (
            metadata.get("local_snapshot_manifest_sha256") == market["snapshot_manifest_sha256"]
            and metadata.get("local_coverage_manifest_sha256") == market["snapshot_manifest_sha256"]
            and metadata.get("local_scope_artifact_sha256") == pit["scope_sha256"]
        )
        if (
            metadata.get("local_target_trade_date") == context.target_date
            and context.target_date in release_evidence.open_dates
            and _compact_date(release_evidence.captured_at) >= context.target_date
            and exact_market_binding
        ):
            return {
                "status": "NO_ACTION",
                "write_performed": False,
                "blockers": [],
                "evidence": {
                    "release_pointer_sha256": release_sha,
                    "release_manifest_sha256": (release_evidence.identity.manifest_sha256),
                    "release_semantic_sha256": (release_evidence.identity.semantic_sha256),
                    "observations_pointer_sha256": observations_sha,
                    "observations_manifest_sha256": loaded_observations.get("manifest_sha256"),
                    "market_manifest_sha256": market["snapshot_manifest_sha256"],
                },
            }
        transaction_id = f"macro-{context.target_date}-{context.attempt_slot}"
        journal_root = context.run_root / "journals" / "macro" / context.target_date
        journal_run_id = f"macro-{context.target_date}"
        journal_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(journal_root, 0o700)
        journal_run = journal_root / journal_run_id
        if context.mode == "execute" and journal_run.exists():
            recovery = self.apis.macro_recover(
                journal_root=journal_root,
                journal_run_id=journal_run_id,
                market_pointer_path=market["pointer_path"],
                expected_market_pointer_sha256=market["pointer_sha256"],
                pit_pointer_path=pit["pit_binding"]["discovery_pointer_path"],
                expected_pit_pointer_sha256=pit["pit_binding"]["discovery_pointer_sha256"],
                execute_forward=False,
            )
            if recovery.get("terminal") is True:
                result = recovery
            elif recovery.get("execute_forward_eligible") is True:
                result = self.apis.macro_recover(
                    journal_root=journal_root,
                    journal_run_id=journal_run_id,
                    market_pointer_path=market["pointer_path"],
                    expected_market_pointer_sha256=market["pointer_sha256"],
                    pit_pointer_path=pit["pit_binding"]["discovery_pointer_path"],
                    expected_pit_pointer_sha256=pit["pit_binding"]["discovery_pointer_sha256"],
                    execute_forward=True,
                )
            else:
                return {
                    "status": "BLOCKED",
                    "write_performed": False,
                    "blockers": ["MACRO_PROMOTION_UNCERTAIN"],
                    "evidence": {"recovery": dict(recovery)},
                }
        else:
            preparation_root = context.attempt_root / "macro_prepare"
            preparation_root.mkdir(mode=0o700)
            prepared = self.apis.macro_prepare(
                market="CN",
                target_date=context.target_date,
                snapshot_manifest_path=market["snapshot_manifest_path"],
                expected_snapshot_manifest_sha256=market["snapshot_manifest_sha256"],
                coverage_manifest_path=market["snapshot_manifest_path"],
                expected_coverage_manifest_sha256=market["snapshot_manifest_sha256"],
                scope_artifact_path=_stage_evidence(context, "PIT")["scope_path"],
                expected_scope_artifact_sha256=_stage_evidence(context, "PIT")["scope_sha256"],
                release_root=release_root,
                expected_release_pointer_sha256=release_sha,
                observations_root=observations_root,
                expected_observations_pointer_sha256=observations_sha,
                market_pointer_path=market["pointer_path"],
                expected_market_pointer_sha256=market["pointer_sha256"],
                pit_pointer_path=pit["pit_binding"]["discovery_pointer_path"],
                expected_pit_pointer_sha256=pit["pit_binding"]["discovery_pointer_sha256"],
                authority_mode=("candidate" if context.mode == "shadow" else "canonical"),
                release_run_id=f"release-{transaction_id}",
                observations_run_id=f"observations-{transaction_id}",
                private_run_root=preparation_root,
                transaction_run_id=transaction_id,
                allow_live=True,
            )
            if context.mode == "shadow":
                result = prepared
            else:
                result = self.apis.macro_commit(
                    prepared_path=prepared["prepared_path"],
                    expected_prepared_sha256=prepared["prepared_sha256"],
                    journal_root=journal_root,
                    journal_run_id=journal_run_id,
                    market_pointer_path=market["pointer_path"],
                    expected_market_pointer_sha256=market["pointer_sha256"],
                    pit_pointer_path=pit["pit_binding"]["discovery_pointer_path"],
                    expected_pit_pointer_sha256=pit["pit_binding"]["discovery_pointer_sha256"],
                )
        ready = str(result.get("status") or "") in {"PREPARED", "SUCCESS"}
        return {
            "status": "READY" if ready else "BLOCKED",
            "write_performed": context.mode == "execute" and ready,
            "blockers": [] if ready else ["MACRO_TRANSACTION_BLOCKED"],
            "evidence": {
                "transaction_status": result.get("status"),
                "prepared_path": result.get("prepared_path"),
                "prepared_sha256": result.get("prepared_sha256"),
                "journal_root": str(journal_root),
                "journal_run_id": journal_run_id,
            },
        }


def build_default_components(
    *, workspace_root: str | Path, apis: DailyComponentAPIs | None = None
) -> MaintenanceComponents:
    workspace = Path(workspace_root).expanduser().resolve(strict=True)
    owner = _DefaultComponents(workspace, apis or registered_component_apis())

    def guarded(stage: str, callback: Callable[[MaintenanceContext], Mapping[str, Any]]):
        def run(context: MaintenanceContext) -> Mapping[str, Any]:
            try:
                return callback(context)
            except Exception as exc:
                text = str(exc).casefold()
                retryable = any(
                    marker in text
                    for marker in (
                        "provider",
                        "tushare",
                        "timeout",
                        "connection",
                        "http_status",
                        "response_empty",
                        "query_failed",
                        "true_missing",
                        "keyset_missing",
                    )
                ) and not any(
                    marker in text
                    for marker in (
                        "cas_mismatch",
                        "sha256_mismatch",
                        "schema",
                        "promotion_uncertain",
                        "shrink",
                        "tamper",
                        "accounting",
                        "malformed",
                        "has_more",
                        "duplicate",
                        "scope_",
                        "identity",
                    )
                )
                return {
                    "status": "RETRY_PENDING" if retryable else "BLOCKED",
                    "write_performed": False,
                    "blockers": [
                        f"{stage}_{'PROVIDER_NOT_READY' if retryable else 'CONTRACT_BLOCKED'}"
                    ],
                    "evidence": {"error_type": type(exc).__name__},
                }

        return run

    return MaintenanceComponents(
        pit=guarded("PIT", owner.pit),
        market=guarded("MARKET", owner.market),
        history=guarded("HISTORY", owner.history),
        macro_release=guarded("MACRO_RELEASE", owner.macro),
    )


__all__ = [
    "DailyComponentAPIs",
    "build_default_components",
    "registered_component_apis",
]
