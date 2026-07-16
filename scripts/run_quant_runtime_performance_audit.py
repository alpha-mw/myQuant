#!/usr/bin/env python3
"""Run an explicit offline performance audit for the Quant runtime path.

This command is diagnostic only.  It builds deterministic synthetic frames,
never authorizes forward apply, and writes no audit or governance artifact.
Exactly one JSON document is emitted on stdout.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes
from dataclasses import dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from typing import Any, Callable, Mapping, Sequence
from unittest.mock import patch
import urllib.request

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import quant_investor.factors.governance_protocol_v3 as protocol_module  # noqa: E402
import quant_investor.factors.runtime as runtime_module  # noqa: E402
import quant_investor.market.dag.packets as packets_module  # noqa: E402
from quant_investor.factors.governance import (  # noqa: E402
    GATE_SPECS,
    FactorLifecycleState,
    FactorRecord,
    GateResult,
)
from quant_investor.factors.runtime import (  # noqa: E402
    MinedFactorRegistry,
    MinedFactorScorer,
    ProductionEvaluationContext,
    _mint_production_evaluation_context,
    production_runtime_input_sha256,
    production_symbol_set_sha256,
)


SCHEMA_VERSION = "myquant.quant-runtime-performance-audit.v1"
DEFAULT_REGISTRY = ROOT / "quant_investor" / "factor_registry" / "mined_factors.json"
EXPECTED_REGISTRY_SHA256 = (
    "b8369dfef7d27156999e93e3a1a12020e072db0296532fee10b0335d8bddca2f"
)
FACTOR_NAMES = (
    "pv_low_dollar_volume_5d",
    "pv_volume_stability_5d",
    "pv_momentum_20d",
    "pv_short_reversal_5d",
    "pv_amihud_illiquidity_5d",
    "pv_high_dollar_volume_10d",
    "pv_volatility_penalty_20d",
    "pv_downside_volatility_20d",
    "pv_price_efficiency_20d",
    "pv_dollar_volume_growth_5d_20d",
    "pv_volume_stability_smooth_10d_5d",
    "pv_blend_volstab19x2_mom90_amihud5_w70",
    "pv_low_dollar_volume_30d",
    "pv_momentum_60d",
)
BASE_BUDGET_SECONDS = {
    5: {"digest": 30.0, "score": 35.0, "plan": 45.0, "combined": 75.0},
    14: {"digest": 90.0, "score": 90.0, "plan": 125.0, "combined": 210.0},
}
REFERENCE_SYMBOLS = 5520
REFERENCE_VALIDATION_ROWS = 280
REFERENCE_RUNTIME_ROWS = 91
REFERENCE_WARMUP = 1
REFERENCE_SAMPLES = 3
REFERENCE_SEED = 20260714
REFERENCE_BUDGET_SCALE = 1.0
REFERENCE_VALIDATION_THROUGHPUT_MIN = 100_000.0
REFERENCE_FACTOR_THROUGHPUT_MIN = 80_000.0
REFERENCE_SCALING_MAX = 3.5
REFERENCE_NATIVE_INCREMENTAL_MAX_MIB = 128.0
MAX_DIAGNOSTIC_FLOAT = 1_000_000.0


@dataclass(frozen=True)
class RegistrySnapshot:
    sha256: str
    mode: int
    nlink: int
    uid: int
    device: int
    inode: int
    size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "mode": f"{self.mode:04o}",
            "nlink": self.nlink,
            "uid": self.uid,
            "device": self.device,
            "inode": self.inode,
            "size": self.size,
        }


def _registry_snapshot(path: Path) -> RegistrySnapshot:
    def identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            int(value.st_dev),
            int(value.st_ino),
            int(value.st_mode),
            int(value.st_uid),
            int(value.st_nlink),
            int(value.st_size),
            int(value.st_mtime_ns),
            int(value.st_ctime_ns),
        )

    try:
        path_before = os.lstat(path)
    except OSError as exc:
        raise RuntimeError("canonical registry lstat failed") from exc
    if not stat.S_ISREG(path_before.st_mode):
        raise RuntimeError("canonical registry must be a regular non-symlink file")
    if path_before.st_uid != os.getuid():
        raise RuntimeError("canonical registry owner mismatch")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("canonical registry nofollow open failed") from exc
    try:
        opened = os.fstat(fd)
        if identity(opened) != identity(path_before):
            raise RuntimeError("canonical registry identity changed before read")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(fd, min(65_536, remaining))
            if not chunk:
                raise RuntimeError("canonical registry shrank during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise RuntimeError("canonical registry grew during read")
        opened_after = os.fstat(fd)
    finally:
        os.close(fd)
    try:
        path_after = os.lstat(path)
    except OSError as exc:
        raise RuntimeError("canonical registry disappeared after read") from exc
    if (
        identity(opened) != identity(opened_after)
        or identity(opened) != identity(path_after)
    ):
        raise RuntimeError("canonical registry changed during read")
    raw = b"".join(chunks)
    return RegistrySnapshot(
        sha256=hashlib.sha256(raw).hexdigest(),
        mode=stat.S_IMODE(opened.st_mode),
        nlink=int(opened.st_nlink),
        uid=int(opened.st_uid),
        device=int(opened.st_dev),
        inode=int(opened.st_ino),
        size=int(opened.st_size),
    )


def _ru_maxrss_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


class _DarwinProcTaskInfo(ctypes.Structure):
    _fields_ = [
        ("pti_virtual_size", ctypes.c_uint64),
        ("pti_resident_size", ctypes.c_uint64),
        ("pti_total_user", ctypes.c_uint64),
        ("pti_total_system", ctypes.c_uint64),
        ("pti_threads_user", ctypes.c_uint64),
        ("pti_threads_system", ctypes.c_uint64),
        ("pti_policy", ctypes.c_int32),
        ("pti_faults", ctypes.c_int32),
        ("pti_pageins", ctypes.c_int32),
        ("pti_cow_faults", ctypes.c_int32),
        ("pti_messages_sent", ctypes.c_int32),
        ("pti_messages_received", ctypes.c_int32),
        ("pti_syscalls_mach", ctypes.c_int32),
        ("pti_syscalls_unix", ctypes.c_int32),
        ("pti_csw", ctypes.c_int32),
        ("pti_threadnum", ctypes.c_int32),
        ("pti_numrunning", ctypes.c_int32),
        ("pti_priority", ctypes.c_int32),
    ]


_DARWIN_LIBPROC: Any | None = None


def _current_rss_bytes() -> int:
    """Return current resident bytes without shelling out."""

    if sys.platform == "darwin":
        global _DARWIN_LIBPROC
        if _DARWIN_LIBPROC is None:
            libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
            libproc.proc_pidinfo.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_uint64,
                ctypes.c_void_p,
                ctypes.c_int,
            ]
            libproc.proc_pidinfo.restype = ctypes.c_int
            _DARWIN_LIBPROC = libproc
        libproc = _DARWIN_LIBPROC
        info = _DarwinProcTaskInfo()
        size = libproc.proc_pidinfo(
            os.getpid(),
            4,
            0,
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if size != ctypes.sizeof(info):
            raise RuntimeError("native current RSS readback failed")
        return int(info.pti_resident_size)
    if sys.platform.startswith("linux"):
        fields = Path("/proc/self/statm").read_text(encoding="ascii").split()
        if len(fields) < 2:
            raise RuntimeError("native current RSS readback failed")
        return int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
    raise RuntimeError("native current RSS is unsupported on this platform")


def _native_rss_probe(operation: Callable[[], None]) -> dict[str, int]:
    """Sample current RSS around one operation with its inputs resident."""

    baseline = _current_rss_bytes()
    peak = baseline
    peak_lock = threading.Lock()
    stop = threading.Event()
    errors: list[BaseException] = []

    def observe() -> None:
        nonlocal peak
        current = _current_rss_bytes()
        with peak_lock:
            peak = max(peak, current)

    def sample() -> None:
        try:
            while not stop.is_set():
                observe()
                stop.wait(0.005)
            observe()
        except BaseException as exc:
            errors.append(exc)
            stop.set()

    sampler = threading.Thread(
        target=sample,
        name="quant-runtime-rss-sampler",
        daemon=True,
    )
    sampler.start()
    try:
        operation()
        observe()
    finally:
        stop.set()
        sampler.join(timeout=1.0)
    if errors:
        raise RuntimeError("native RSS sampler failed") from errors[0]
    if sampler.is_alive():
        raise RuntimeError("native RSS sampler did not stop")
    return {
        "baseline_bytes": baseline,
        "peak_bytes": peak,
        "incremental_bytes": max(0, peak - baseline),
    }


def _factor_record(name: str) -> FactorRecord:
    gates = [
        GateResult(
            gate_id=spec.gate_id,
            gate_key=spec.key,
            title=spec.title,
            passed=True,
            metrics={"coverage_rate": 1.0} if spec.gate_id == 2 else {},
        )
        for spec in GATE_SPECS
    ]
    return FactorRecord(
        name=name,
        version="audit-v1",
        state=FactorLifecycleState.PRODUCTION_FACTOR,
        category="audit_only",
        implementation=f"price_volume:{name}",
        weight=0.05,
        direction=1.0,
        gate_results=gates,
        metadata={
            "factor_family": f"audit-{name}",
            "dominant_primitive_cluster": name,
        },
    )


def _runtime_status(
    records: Sequence[FactorRecord], runtime_rows: int
) -> dict[str, Any]:
    contracts = {
        record.name: {
            "required_columns": ["trade_date", "adj_close", "vol", "amount"],
            "lookback_rows": runtime_rows,
            "gate2_min_coverage_rate": 1.0,
            "min_cross_section": 20,
        }
        for record in records
    }
    return {
        "status": "ready",
        "factor_mode": "governed_mined_factors",
        "confidence_multiplier": 1.0,
        "production_eligible": True,
        "blockers": [],
        "production_factor_names": [record.name for record in records],
        "factor_runtime_contracts": contracts,
    }


def _scorer(
    records: Sequence[FactorRecord], runtime_status: Mapping[str, Any]
) -> MinedFactorScorer:
    scorer = MinedFactorScorer(MinedFactorRegistry.from_records(records))
    active = list(records)
    status = dict(runtime_status)
    scorer._runtime_contract = lambda: (active, status)  # type: ignore[method-assign]
    return scorer


def _synthetic_frames(
    *, symbols: int, rows: int, seed: int
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end="2026-01-06", periods=rows, freq="D")
    step = np.arange(rows, dtype=float)
    result: dict[str, pd.DataFrame] = {}
    for index in range(symbols):
        symbol = f"Q{index:06d}.SZ"
        phase = float(index % 19)
        trend = step * (0.0015 + (index % 11) * 0.00017)
        wave = np.sin((step + phase) / (4.0 + index % 7)) * (
            0.03 + (index % 5) * 0.006
        )
        close = 8.0 + index * 0.0007 + trend + wave
        volume = (
            900.0
            + index * 0.07
            + (step % (5 + index % 9)) * (7.0 + index % 13)
            + np.cos((step + phase) / 7.0) * 12.0
        )
        jitter = rng.normal(0.0, 0.0005, size=rows)
        close = np.maximum(close + jitter, 0.01)
        volume = np.maximum(volume, 1.0)
        amount = close * volume * (1.0 + (index % 17) * 0.003)
        result[symbol] = pd.DataFrame(
            {
                "ts_code": [symbol] * rows,
                "trade_date": dates,
                "adj_close": close,
                "vol": volume,
                "amount": amount,
            }
        )
    return result


def _evaluation_context(
    *, symbols: Sequence[str], directory: Path
) -> ProductionEvaluationContext:
    paths: dict[str, str] = {}
    hashes: dict[str, str] = {}
    for index, name in enumerate(
        (
            "snapshot_pointer",
            "snapshot_manifest",
            "open_day_calendar",
            "pit_manifest",
            "pit_canonical",
        )
    ):
        path = directory / f"{name}.audit"
        raw = json.dumps(
            {
                "audit_only": True,
                "name": name,
                "ordinal": index,
                "as_of": "20260106",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        path.write_bytes(raw)
        paths[name] = str(path.resolve())
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return _mint_production_evaluation_context(
        evaluation_as_of="20260106",
        market="CN",
        universe_key="synthetic_audit",
        universe_sha256=production_symbol_set_sha256(list(symbols)),
        snapshot_id="synthetic-runtime-audit",
        latest_complete_trade_date="20260106",
        pit_membership_status="verified",
        pit_membership_as_of="20260106",
        pit_membership_proof_sha256="a" * 64,
        pit_membership_not_applicable_reason="",
        open_day_proof_sha256=hashes["open_day_calendar"],
        read_result_provenance_sha256="c" * 64,
        verified_artifact_paths=paths,
        verified_artifact_sha256s=hashes,
    )


def _measure(
    operation: Callable[[], None], *, warmup: int, samples: int
) -> dict[str, Any]:
    for _ in range(warmup):
        operation()
    observed: list[float] = []
    for _ in range(samples):
        started = time.perf_counter()
        operation()
        observed.append(time.perf_counter() - started)
    median = float(statistics.median(observed))
    return {
        "samples_seconds": [round(value, 9) for value in observed],
        "median_seconds": round(median, 9),
    }


def _audit_factor_count(
    *,
    factor_count: int,
    frames: Mapping[str, pd.DataFrame],
    context: ProductionEvaluationContext,
    runtime_rows: int,
    warmup: int,
    samples: int,
) -> dict[str, Any]:
    records = [_factor_record(name) for name in FACTOR_NAMES[:factor_count]]
    status = _runtime_status(records, runtime_rows)
    contracts = dict(status["factor_runtime_contracts"])

    digest_result = _measure(
        lambda: production_runtime_input_sha256(frames, contracts),
        warmup=warmup,
        samples=samples,
    )

    score_scorer = _scorer(records, status)

    def score_once() -> None:
        result = score_scorer._score_production(
            frames,
            symbols=tuple(frames),
            active=records,
            skipped={},
            runtime_status=status,
            evaluation_context=context,
        )
        if (
            result.governance_status != "ready"
            or result.factor_count != factor_count
            or set(result.symbol_scores) != set(frames)
        ):
            raise RuntimeError("synthetic score path did not complete exactly")

    score_result = _measure(score_once, warmup=warmup, samples=samples)

    def plan_once() -> None:
        plan = _scorer(records, status).build_production_runtime_plan(frames)
        if (
            not plan.filter_applied
            or len(plan.active_factors) != factor_count
            or len(plan.eligible_symbols) != len(frames)
        ):
            raise RuntimeError("synthetic plan path did not complete exactly")

    plan_result = _measure(plan_once, warmup=warmup, samples=samples)
    medians = {
        "digest": float(digest_result["median_seconds"]),
        "score": float(score_result["median_seconds"]),
        "plan": float(plan_result["median_seconds"]),
    }
    medians["combined"] = medians["score"] + medians["plan"]
    work_units = factor_count * len(frames) * runtime_rows
    throughput = {
        name: round(work_units / seconds, 3) if seconds > 0.0 else 0.0
        for name, seconds in medians.items()
    }
    return {
        "digest": digest_result,
        "score": score_result,
        "plan": plan_result,
        "combined_median_seconds": round(medians["combined"], 9),
        "factor_symbol_row_observations": work_units,
        "factor_symbol_rows_per_second": throughput,
    }


def _orchestration_probe(
    *,
    frames: Mapping[str, pd.DataFrame],
    context: ProductionEvaluationContext,
    runtime_rows: int,
) -> dict[str, Any]:
    records = [_factor_record(name) for name in FACTOR_NAMES[:5]]
    status = _runtime_status(records, runtime_rows)
    calls = {
        "production_runtime_input_sha256": 0,
        "validate_production_frames": 0,
    }
    original_digest = runtime_module.production_runtime_input_sha256
    original_validate = runtime_module._validate_production_frames

    def counting_digest(runtime_frames: Any, runtime_contracts: Any) -> str:
        calls["production_runtime_input_sha256"] += 1
        return original_digest(runtime_frames, runtime_contracts)

    def counting_validate(
        runtime_frames: Any, *, symbols: Sequence[str], context: Any
    ) -> str | None:
        calls["validate_production_frames"] += 1
        return original_validate(runtime_frames, symbols=symbols, context=context)

    scorer = _scorer(records, status)
    with (
        patch.object(
            runtime_module,
            "production_runtime_input_sha256",
            counting_digest,
        ),
        patch.object(
            packets_module,
            "production_runtime_input_sha256",
            counting_digest,
        ),
        patch.object(
            runtime_module,
            "_validate_production_frames",
            counting_validate,
        ),
    ):
        plan = scorer.build_production_runtime_plan(frames)
        result, token = packets_module._build_quant_branch_result_with_validation(
            frames=frames,
            evaluation_context=context,
            scorer=scorer,
            production_runtime_plan=plan,
        )
    invariants_passed = (
        plan.filter_applied
        and calls["production_runtime_input_sha256"] == 2
        and calls["validate_production_frames"] == 1
        and token is None
        and result.metadata.get("production_eligible") is False
        and result.metadata.get("factor_mode") == "governance_blocked"
    )
    return {
        "observed_calls": calls,
        "validation_token_issued": token is not None,
        "non_authoritative_synthetic_context_expected_blocked": True,
        "observed_factor_mode": result.metadata.get("factor_mode"),
        "observed_production_eligible": result.metadata.get(
            "production_eligible"
        ),
        "invariants_passed": bool(invariants_passed),
    }


def _budgets(
    *,
    validation: Mapping[str, Any],
    factors: Mapping[str, Mapping[str, Any]],
    budget_scale: float,
    validation_throughput_min: float,
    factor_throughput_min: float,
    scaling_max: float,
    native_incremental_bytes: int,
    native_incremental_max_mib: float,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    validation_median = float(validation["median_seconds"])
    validation_rows = int(validation["symbol_rows"])
    validation_throughput = (
        validation_rows / validation_median if validation_median > 0.0 else 0.0
    )
    validation_seconds_max = 15.0 * budget_scale
    validation_passed = (
        validation_median <= validation_seconds_max
        and validation_throughput >= validation_throughput_min
    )
    if not validation_passed:
        blockers.append("frame_validation_budget_exceeded")

    factor_verdicts: dict[str, Any] = {}
    for count_text, result in factors.items():
        count = int(count_text)
        base = BASE_BUDGET_SECONDS[count]
        medians = {
            "digest": float(result["digest"]["median_seconds"]),
            "score": float(result["score"]["median_seconds"]),
            "plan": float(result["plan"]["median_seconds"]),
            "combined": float(result["combined_median_seconds"]),
        }
        throughputs = dict(result["factor_symbol_rows_per_second"])
        operation_passed: dict[str, bool] = {}
        for operation, median in medians.items():
            seconds_passed = median <= base[operation] * budget_scale
            throughput_passed = (
                True
                if operation == "combined"
                else float(throughputs[operation]) >= factor_throughput_min
            )
            operation_passed[operation] = seconds_passed and throughput_passed
            if not operation_passed[operation]:
                blockers.append(f"factor_{count}_{operation}_budget_exceeded")
        factor_verdicts[count_text] = {
            "seconds_max": {
                name: base[name] * budget_scale for name in sorted(base)
            },
            "operation_passed": operation_passed,
        }

    scaling: dict[str, float] = {}
    scaling_passed: dict[str, bool] = {}
    five = factors["5"]
    fourteen = factors["14"]
    for operation in ("digest", "score", "plan", "combined"):
        five_seconds = float(
            five["combined_median_seconds"]
            if operation == "combined"
            else five[operation]["median_seconds"]
        )
        fourteen_seconds = float(
            fourteen["combined_median_seconds"]
            if operation == "combined"
            else fourteen[operation]["median_seconds"]
        )
        ratio = fourteen_seconds / five_seconds if five_seconds > 0.0 else float("inf")
        scaling[operation] = round(ratio, 6)
        scaling_passed[operation] = ratio <= scaling_max
        if not scaling_passed[operation]:
            blockers.append(f"factor_{operation}_scaling_exceeded")

    native_limit_bytes = int(native_incremental_max_mib * 1024 * 1024)
    memory_passed = native_incremental_bytes <= native_limit_bytes
    if not memory_passed:
        blockers.append("native_incremental_memory_budget_exceeded")
    return (
        {
            "validation": {
                "median_seconds_max": validation_seconds_max,
                "symbol_rows_per_second_min": validation_throughput_min,
                "observed_symbol_rows_per_second": round(
                    validation_throughput, 3
                ),
                "passed": validation_passed,
            },
            "factors": factor_verdicts,
            "factor_symbol_rows_per_second_min": factor_throughput_min,
            "scaling": {
                "max": scaling_max,
                "observed": scaling,
                "passed": scaling_passed,
            },
            "native_incremental_bytes_max": native_limit_bytes,
            "native_incremental_passed": memory_passed,
        },
        list(dict.fromkeys(blockers)),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=REFERENCE_SYMBOLS)
    parser.add_argument(
        "--validation-rows", type=int, default=REFERENCE_VALIDATION_ROWS
    )
    parser.add_argument("--runtime-rows", type=int, default=REFERENCE_RUNTIME_ROWS)
    parser.add_argument("--warmup", type=int, default=REFERENCE_WARMUP)
    parser.add_argument("--samples", type=int, default=REFERENCE_SAMPLES)
    parser.add_argument("--seed", type=int, default=REFERENCE_SEED)
    parser.add_argument(
        "--budget-scale", type=float, default=REFERENCE_BUDGET_SCALE
    )
    parser.add_argument(
        "--validation-throughput-min",
        type=float,
        default=REFERENCE_VALIDATION_THROUGHPUT_MIN,
    )
    parser.add_argument(
        "--factor-throughput-min",
        type=float,
        default=REFERENCE_FACTOR_THROUGHPUT_MIN,
    )
    parser.add_argument(
        "--scaling-max", type=float, default=REFERENCE_SCALING_MAX
    )
    parser.add_argument(
        "--native-incremental-max-mib",
        type=float,
        default=REFERENCE_NATIVE_INCREMENTAL_MAX_MIB,
    )
    args = parser.parse_args(argv)
    float_values = (
        args.budget_scale,
        args.validation_throughput_min,
        args.factor_throughput_min,
        args.scaling_max,
        args.native_incremental_max_mib,
    )
    if (
        any(
            not math.isfinite(value) or value > MAX_DIAGNOSTIC_FLOAT
            for value in float_values
        )
        or args.symbols < 20
        or args.validation_rows < 2
        or args.runtime_rows < 91
        or args.warmup < 1
        or args.samples < 3
        or args.budget_scale < 0.0
        or args.validation_throughput_min < 0.0
        or args.factor_throughput_min < 0.0
        or args.scaling_max <= 0.0
        or args.native_incremental_max_mib < 0.0
    ):
        parser.error("audit dimensions and budgets are outside the safe contract")
    return args


def _is_reference_profile(args: argparse.Namespace) -> bool:
    return (
        args.symbols == REFERENCE_SYMBOLS
        and args.validation_rows == REFERENCE_VALIDATION_ROWS
        and args.runtime_rows == REFERENCE_RUNTIME_ROWS
        and args.warmup == REFERENCE_WARMUP
        and args.samples == REFERENCE_SAMPLES
        and args.seed == REFERENCE_SEED
        and args.budget_scale == REFERENCE_BUDGET_SCALE
        and args.validation_throughput_min
        == REFERENCE_VALIDATION_THROUGHPUT_MIN
        and args.factor_throughput_min == REFERENCE_FACTOR_THROUGHPUT_MIN
        and args.scaling_max == REFERENCE_SCALING_MAX
        and args.native_incremental_max_mib
        == REFERENCE_NATIVE_INCREMENTAL_MAX_MIB
    )


def _run(args: argparse.Namespace, poison_calls: dict[str, int]) -> dict[str, Any]:
    process_current_rss_start = _current_rss_bytes()
    process_ru_maxrss_start = _ru_maxrss_bytes()
    peak_python = 0
    with tempfile.TemporaryDirectory(prefix="myquant-runtime-audit-") as raw_temp:
        temp_root = Path(raw_temp)
        symbols = [f"Q{index:06d}.SZ" for index in range(args.symbols)]
        context = _evaluation_context(symbols=symbols, directory=temp_root)

        runtime_probe_frames = _synthetic_frames(
            symbols=args.symbols,
            rows=args.runtime_rows,
            seed=args.seed,
        )
        native_memory_records = [
            _factor_record(name) for name in FACTOR_NAMES[:14]
        ]
        native_memory_contracts = dict(
            _runtime_status(native_memory_records, args.runtime_rows)[
                "factor_runtime_contracts"
            ]
        )
        validation_frames = _synthetic_frames(
            symbols=args.symbols,
            rows=args.validation_rows,
            seed=args.seed,
        )

        def validate_once() -> None:
            blocker = runtime_module._validate_production_frames(
                validation_frames,
                symbols=tuple(validation_frames),
                context=context,
            )
            if blocker is not None:
                raise RuntimeError(f"synthetic validation blocked:{blocker}")

        def native_memory_once() -> None:
            production_runtime_input_sha256(
                runtime_probe_frames, native_memory_contracts
            )
            validate_once()

        native_memory_probe = _native_rss_probe(native_memory_once)
        runtime_probe_frames.clear()
        gc.collect()
        validation = _measure(
            validate_once,
            warmup=args.warmup,
            samples=args.samples,
        )
        validation["symbol_rows"] = args.symbols * args.validation_rows
        validation_frames.clear()
        gc.collect()

        runtime_frames = _synthetic_frames(
            symbols=args.symbols,
            rows=args.runtime_rows,
            seed=args.seed,
        )
        factor_results = {
            str(count): _audit_factor_count(
                factor_count=count,
                frames=runtime_frames,
                context=context,
                runtime_rows=args.runtime_rows,
                warmup=args.warmup,
                samples=args.samples,
            )
            for count in (5, 14)
        }
        orchestration = _orchestration_probe(
            frames=runtime_frames,
            context=context,
            runtime_rows=args.runtime_rows,
        )
        memory_records = [
            _factor_record(name) for name in FACTOR_NAMES[:5]
        ]
        memory_contracts = dict(
            _runtime_status(memory_records, args.runtime_rows)[
                "factor_runtime_contracts"
            ]
        )
        tracemalloc.start()
        try:
            production_runtime_input_sha256(runtime_frames, memory_contracts)
            _current_python, peak_python = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        runtime_frames.clear()
        gc.collect()

    native_incremental = native_memory_probe["incremental_bytes"]
    native_ru_maxrss_total = _ru_maxrss_bytes()
    native_current_rss_final = _current_rss_bytes()
    budgets, blockers = _budgets(
        validation=validation,
        factors=factor_results,
        budget_scale=args.budget_scale,
        validation_throughput_min=args.validation_throughput_min,
        factor_throughput_min=args.factor_throughput_min,
        scaling_max=args.scaling_max,
        native_incremental_bytes=native_incremental,
        native_incremental_max_mib=args.native_incremental_max_mib,
    )
    if not orchestration["invariants_passed"]:
        blockers.append("production_orchestration_call_invariant_failed")
    if any(poison_calls.values()):
        blockers.append("forbidden_external_or_apply_call_observed")
    return {
        "profile": {
            "factor_counts": [5, 14],
            "runtime_rows": args.runtime_rows,
            "samples": args.samples,
            "seed": args.seed,
            "symbols": args.symbols,
            "validation_rows": args.validation_rows,
            "timed_warmup_per_operation": args.warmup,
            "untimed_pre_timing_operations": [
                "fourteen_factor_runtime_input_digest",
                "production_frame_validation",
            ],
        },
        "timer": {
            "clock": "time.perf_counter",
            "median": True,
        },
        "platform": {
            "python": platform.python_version(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
        "benchmarks": {
            "validation": validation,
            "factors": factor_results,
        },
        "orchestration": orchestration,
        "memory": {
            "native_current_rss_process_start_bytes": process_current_rss_start,
            "native_current_rss_final_bytes": native_current_rss_final,
            "native_ru_maxrss_process_start_bytes": process_ru_maxrss_start,
            "native_ru_maxrss_total_bytes": native_ru_maxrss_total,
            "native_runtime_and_validation_probe": native_memory_probe,
            "native_incremental_bytes": native_incremental,
            "native_incremental_max_bytes": budgets[
                "native_incremental_bytes_max"
            ],
            "native_incremental_passed": budgets[
                "native_incremental_passed"
            ],
            "python_tracemalloc_peak_bytes": peak_python,
            "python_tracemalloc_scope": "five_factor_digest_single_pass",
            "native_measurement": (
                "sampled_current_rss_common_baseline_with_both_inputs_resident"
            ),
        },
        "budgets": budgets,
        "blockers": list(dict.fromkeys(blockers)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    reference_profile = _is_reference_profile(args)
    registry_path = DEFAULT_REGISTRY
    poison_calls = {
        "protocol.apply_governed_transition": 0,
        "socket.create_connection": 0,
        "urllib.request.urlopen": 0,
        "subprocess.Popen": 0,
    }
    before: RegistrySnapshot | None = None
    after: RegistrySnapshot | None = None
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit_only": True,
        "production_apply_authorized": False,
        "reference_profile": reference_profile,
        "reference_acceptance_eligible": False,
        "status": "fail",
        "blockers": [],
    }

    def poison(surface: str, message: str) -> Callable[..., Any]:
        def blocked(*_args: Any, **_kwargs: Any) -> Any:
            poison_calls[surface] += 1
            raise RuntimeError(message)

        return blocked

    try:
        before = _registry_snapshot(registry_path)
        if (
            before.sha256 != EXPECTED_REGISTRY_SHA256
            or before.mode != 0o644
            or before.nlink != 1
        ):
            raise RuntimeError("canonical registry baseline mismatch")
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    protocol_module,
                    "apply_governed_transition",
                    poison(
                        "protocol.apply_governed_transition",
                        "forward apply is forbidden during runtime audit",
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    socket,
                    "create_connection",
                    poison(
                        "socket.create_connection",
                        "external I/O is forbidden during runtime audit",
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    urllib.request,
                    "urlopen",
                    poison(
                        "urllib.request.urlopen",
                        "external I/O is forbidden during runtime audit",
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    subprocess,
                    "Popen",
                    poison(
                        "subprocess.Popen",
                        "subprocess execution is forbidden during runtime audit",
                    ),
                )
            )
            payload.update(_run(args, poison_calls))
    except Exception as exc:
        payload.setdefault("blockers", []).append(
            f"performance_audit_exception:{type(exc).__name__}"
        )
        print(
            f"quant runtime performance audit blocked: {type(exc).__name__}",
            file=sys.stderr,
        )
    finally:
        try:
            after = _registry_snapshot(registry_path)
        except Exception as exc:
            payload.setdefault("blockers", []).append(
                f"registry_readback_failed:{type(exc).__name__}"
            )
        unchanged = before is not None and after is not None and before == after
        if not unchanged:
            payload.setdefault("blockers", []).append("canonical_registry_drift")
        payload["registry_readback"] = {
            "before": before.to_dict() if before is not None else None,
            "after": after.to_dict() if after is not None else None,
            "unchanged": unchanged,
        }
        payload["safety"] = {
            "guarded_surface_calls": dict(poison_calls),
            "guard_scope_exhaustive": False,
            "guard_interpretation": (
                "negative evidence for the enumerated surfaces only"
            ),
            "guarded_external_or_apply_allowed": False,
        }
        payload["blockers"] = list(
            dict.fromkeys(str(item) for item in payload.get("blockers", []) if item)
        )
        payload["status"] = "pass" if not payload["blockers"] else "fail"
        payload["reference_acceptance_eligible"] = (
            reference_profile and payload["status"] == "pass"
        )
        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return 0 if payload["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
