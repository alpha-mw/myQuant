from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd

from quant_investor.themes.calibration import (
    evaluate_threshold,
    safe_hit_rate,
    safe_mean,
)
from quant_investor.themes.replay import (
    PIT_INDUSTRY_LABEL_NOTE,
    ThemeReplayRecord,
    build_theme_calibration_dataset,
)
from quant_investor.themes.storage import ThemeSnapshotStore


SCHEMA_VERSION = "theme_threshold_sweep.v1"
DEFAULT_OUTPUT_DIR = Path("results/theme_calibration")
_BASE_METADATA = {
    "deterministic": True,
    "no_llm": True,
    "no_network": True,
    "offline_only": True,
}
_GRID_PARAMETERS = {
    "momentum_normalization": {
        "expression": "clamp((theme_return_20d + offset) / scale)",
        "current": {"offset": 0.10, "scale": 0.30},
        "sweep": [
            {"offset": 0.08, "scale": 0.25},
            {"offset": 0.10, "scale": 0.30},
            {"offset": 0.12, "scale": 0.35},
        ],
        "evidence_proxy": "symbol_theme_score and theme_score threshold rows",
    },
    "overextension_return_5d_start": {
        "expression": "clamp((theme_return_5d - start) / 0.12)",
        "current": 0.08,
        "sweep": [0.06, 0.08, 0.10],
        "evidence_proxy": "overextended phase and risk-flag threshold rows",
    },
    "phase_score_gates": {
        "current": [35, 55, 70],
        "sweep": [
            [30, 50, 65],
            [35, 55, 70],
            [40, 60, 75],
        ],
        "evidence_proxy": "theme_score threshold rows",
    },
    "crowding_weights": {
        "current": {
            "turnover_share_stretch": 0.45,
            "limitup_norm": 0.35,
            "member_turnover_concentration": 0.20,
        },
        "sweep": [
            {
                "turnover_share_stretch": 0.40,
                "limitup_norm": 0.40,
                "member_turnover_concentration": 0.20,
            },
            {
                "turnover_share_stretch": 0.45,
                "limitup_norm": 0.35,
                "member_turnover_concentration": 0.20,
            },
            {
                "turnover_share_stretch": 0.50,
                "limitup_norm": 0.30,
                "member_turnover_concentration": 0.20,
            },
        ],
        "evidence_proxy": "theme_crowded and theme_narrow_leadership risk-flag rows",
    },
}
_PHASE_VALUES = (
    "accumulation",
    "early_acceleration",
    "confirmed_rotation",
    "overextended",
    "distribution",
)
_RISK_FLAG_VALUES = (
    "theme_overextended",
    "theme_fake_breakout_risk",
    "theme_low_breadth",
    "theme_distribution_risk",
    "theme_crowded",
    "theme_narrow_leadership",
)


@dataclass(frozen=True)
class ThemeThresholdSweepConfig:
    snapshot_dir: Path | str = Path("results/theme_snapshots")
    output_dir: Path | str = DEFAULT_OUTPUT_DIR
    market: str = "CN"
    universe_key: str | None = "full_a"
    history_limit: int = 10
    min_sample: int = 10
    as_of: str = ""
    price_dir: Path | str | None = None
    execution_cost_bps: float = 0.0


def run_threshold_sweep(
    config: ThemeThresholdSweepConfig,
    *,
    frames: Mapping[str, pd.DataFrame] | None = None,
    snapshots: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    snapshot_payloads = list(snapshots or [])
    if not snapshot_payloads:
        snapshot_payloads = ThemeSnapshotStore(config.snapshot_dir).load_recent(
            market=str(config.market or "CN"),
            universe_key=config.universe_key,
            limit=max(int(config.history_limit or 10), 0),
        )
    frame_map = dict(frames or {})
    if not frame_map and config.price_dir:
        frame_map = load_price_frames(config.price_dir)

    dataset = build_theme_calibration_dataset(
        snapshots=snapshot_payloads,
        frames=frame_map,
        horizons=(5, 10, 20),
        benchmark_horizons=(5, 10, 20),
    )
    records = list(dataset.records or [])
    execution_cost_bps = _non_negative_float(config.execution_cost_bps)
    threshold_rows = _build_threshold_rows(
        records,
        min_sample=int(config.min_sample or 10),
        execution_cost_bps=execution_cost_bps,
    )
    output_date = _output_date(config.as_of or _latest_as_of(snapshot_payloads))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "as_of": output_date,
        "market": str(config.market or "CN"),
        "universe_key": str(config.universe_key or ""),
        "metadata": {
            **_BASE_METADATA,
            "snapshot_count": len(snapshot_payloads),
            "record_count": len(records),
            "available_count": sum(1 for record in records if record.data_available),
            "min_sample": int(config.min_sample or 10),
            "history_limit": int(config.history_limit or 10),
            "pit_industry_labels": False,
            "industry_label_note": PIT_INDUSTRY_LABEL_NOTE,
            "execution_cost_bps": execution_cost_bps,
            "net_alpha_method": "gross_forward_alpha_minus_execution_cost_bps",
        },
        "grid_parameters": _GRID_PARAMETERS,
        "dataset": {
            "record_count": len(records),
            "available_count": sum(1 for record in records if record.data_available),
            "metadata": dict(dataset.metadata or {}),
        },
        "threshold_rows": threshold_rows,
    }

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"threshold_sweep_{output_date}.json"
    markdown_path = output_dir / f"threshold_sweep_{output_date}.md"
    _write_json(json_path, payload)
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "payload": payload,
    }


def load_price_frames(price_dir: Path | str) -> dict[str, pd.DataFrame]:
    root = Path(price_dir)
    frames: dict[str, pd.DataFrame] = {}
    if not root.exists():
        return frames
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        symbol = path.stem
        if path.suffix.lower() == ".csv":
            frames[symbol] = pd.read_csv(path)
        elif path.suffix.lower() == ".parquet":
            frames[symbol] = pd.read_parquet(path)
    return frames


def _build_threshold_rows(
    records: list[ThemeReplayRecord],
    *,
    min_sample: int,
    execution_cost_bps: float = 0.0,
) -> list[dict[str, Any]]:
    specs: list[tuple[str, float | str, Callable[[ThemeReplayRecord], bool]]] = []
    for value in (0.45, 0.55, 0.70, 0.85):
        specs.append(
            (
                f"symbol_theme_score >= {value:.2f}",
                value,
                lambda record, threshold=value: _safe_float(record.symbol_theme_score) >= threshold,
            )
        )
    for value in (35.0, 55.0, 70.0):
        specs.append(
            (
                f"theme_score >= {int(value)}",
                value,
                lambda record, threshold=value: _safe_float(record.theme_score) >= threshold,
            )
        )
    for phase in _PHASE_VALUES:
        specs.append(
            (
                f"phase == {phase}",
                phase,
                lambda record, phase_value=phase: str(record.phase or "") == phase_value,
            )
        )
    specs.append(
        (
            "no risk flags",
            "no_risk_flags",
            lambda record: not list(record.risk_flags or []),
        )
    )
    for flag in _RISK_FLAG_VALUES:
        specs.append(
            (
                f"has {flag}",
                flag,
                lambda record, risk_flag=flag: risk_flag in set(record.risk_flags or []),
            )
        )

    rows: list[dict[str, Any]] = []
    total = len(records)
    for threshold_name, threshold_value, predicate in specs:
        selected = [record for record in records if _predicate_passes(predicate, record)]
        diagnostic = evaluate_threshold(
            records,
            threshold_name=threshold_name,
            threshold_value=threshold_value,
            predicate=predicate,
            total_record_count=total,
            min_sample=min_sample,
        )
        row = diagnostic.to_dict()
        row["avg_forward_alpha_20d"] = safe_mean(
            record.forward_alpha_20d for record in selected if record.data_available
        )
        row["hit_rate_20d"] = safe_hit_rate(
            (record.forward_alpha_20d > 0.0 if record.forward_alpha_20d is not None else None)
            for record in selected
            if record.data_available
        )
        _attach_net_alpha_columns(
            row,
            execution_cost_return=execution_cost_bps / 10_000.0,
        )
        rows.append(_jsonable_mapping(row))
    return rows


def _render_markdown(payload: Mapping[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload, Mapping) else {}
    metadata = metadata if isinstance(metadata, Mapping) else {}
    lines = [
        "## Theme Threshold Sweep",
        "",
        f"Market: {payload.get('market', '')}",
        f"Universe: {payload.get('universe_key', '')}",
        f"As of: {payload.get('as_of', '')}",
        f"Snapshots: {metadata.get('snapshot_count', 0)}",
        f"Records: {metadata.get('record_count', 0)}",
        f"Available: {metadata.get('available_count', 0)}",
        f"Industry label note: {PIT_INDUSTRY_LABEL_NOTE}",
        "",
        "### Grid Parameters",
    ]
    grid = payload.get("grid_parameters")
    if isinstance(grid, Mapping):
        for key in sorted(grid):
            current = grid[key].get("current") if isinstance(grid[key], Mapping) else ""
            lines.append(f"- {key}: current={json.dumps(current, sort_keys=True)}")
    lines.extend(
        [
            "",
            "### Threshold Evidence",
            "| threshold | value | selected | available | alpha5 | net_alpha5 | alpha10 | net_alpha10 | alpha20 | net_alpha20 | hit5 | hit10 | action |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in list(payload.get("threshold_rows", []) or []):
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| "
            f"{row.get('threshold_name', '')} | "
            f"{row.get('threshold_value', '')} | "
            f"{row.get('selected_count', 0)} | "
            f"{row.get('available_count', 0)} | "
            f"{_format_optional(row.get('avg_forward_alpha_5d_gross'))} | "
            f"{_format_optional(row.get('avg_forward_alpha_5d_net'))} | "
            f"{_format_optional(row.get('avg_forward_alpha_10d_gross'))} | "
            f"{_format_optional(row.get('avg_forward_alpha_10d_net'))} | "
            f"{_format_optional(row.get('avg_forward_alpha_20d_gross'))} | "
            f"{_format_optional(row.get('avg_forward_alpha_20d_net'))} | "
            f"{_format_optional(row.get('hit_rate_5d'))} | "
            f"{_format_optional(row.get('hit_rate_10d'))} | "
            f"{row.get('recommended_action', '')} |"
        )
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _latest_as_of(snapshots: Sequence[Mapping[str, Any]]) -> str:
    for snapshot in reversed(list(snapshots or [])):
        if not isinstance(snapshot, Mapping):
            continue
        rotation = snapshot.get("theme_rotation")
        rotation = rotation if isinstance(rotation, Mapping) else {}
        value = snapshot.get("as_of") or rotation.get("as_of")
        if value:
            return str(value)
    return ""


def _output_date(value: str) -> str:
    text = str(value or "").strip()
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) >= 8:
        return digits[:8]
    return "unknown_date"


def _predicate_passes(
    predicate: Callable[[ThemeReplayRecord], bool],
    record: ThemeReplayRecord,
) -> bool:
    try:
        return bool(predicate(record))
    except Exception:
        return False


def _safe_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return numeric if math.isfinite(numeric) else -math.inf


def _non_negative_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(numeric) or numeric < 0.0:
        return 0.0
    return numeric


def _attach_net_alpha_columns(
    row: dict[str, Any],
    *,
    execution_cost_return: float,
) -> None:
    cost = max(0.0, float(execution_cost_return or 0.0))
    for horizon in (5, 10, 20):
        base_key = f"avg_forward_alpha_{horizon}d"
        gross = _optional_finite(row.get(base_key))
        row[f"{base_key}_gross"] = gross
        row[f"{base_key}_net"] = (gross - cost) if gross is not None else None


def _optional_finite(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _jsonable_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(value) for key, value in payload.items()}


def _jsonable(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _format_optional(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.4f}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic offline theme threshold sweep diagnostics."
    )
    parser.add_argument("--snapshot-dir", default="results/theme_snapshots")
    parser.add_argument("--price-dir", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--market", default="CN")
    parser.add_argument("--universe-key", default="full_a")
    parser.add_argument("--history-limit", type=int, default=10)
    parser.add_argument("--min-sample", type=int, default=10)
    parser.add_argument("--as-of", default="")
    parser.add_argument("--execution-cost-bps", type=float, default=0.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = ThemeThresholdSweepConfig(
        snapshot_dir=Path(args.snapshot_dir),
        output_dir=Path(args.output_dir),
        market=str(args.market or "CN"),
        universe_key=str(args.universe_key or "") or None,
        history_limit=int(args.history_limit or 10),
        min_sample=int(args.min_sample or 10),
        as_of=str(args.as_of or ""),
        price_dir=Path(args.price_dir) if args.price_dir else None,
        execution_cost_bps=float(args.execution_cost_bps or 0.0),
    )
    result = run_threshold_sweep(config)
    print(f"json_path: {result['json_path']}")
    print(f"markdown_path: {result['markdown_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
