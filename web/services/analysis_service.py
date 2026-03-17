"""Service layer for running and reading web analysis results."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from web.config import APP_DB_PATH, PROJECT_ROOT, PROJECT_VENV_PYTHON, RESULTS_DIR, STOCK_DB_PATH, WEB_ANALYSIS_DIR

logger = logging.getLogger(__name__)

RUNNER_PATH = PROJECT_ROOT / "web" / "tasks" / "run_analysis_job.py"
JOB_DIR = WEB_ANALYSIS_DIR / "jobs"
WEB_RESULT_FILE_RE = re.compile(r"^analysis_\d{8}_\d{6}\.json$")
WEB_REQUEST_FILE_RE = re.compile(r"^analysis_request_\d{8}_\d{6}_.+\.json$")
STALE_JOB_TIMEOUT = timedelta(minutes=10)
DEFAULT_ANALYSIS_TIMEOUT_SECONDS = 300
LARGE_ANALYSIS_TIMEOUT_CAP_SECONDS = 7200
DEFAULT_MARKET_BATCH_SIZE_CN = 80
DEFAULT_MARKET_BATCH_SIZE_US = 60
SESSION_INIT_SQL = """
CREATE TABLE IF NOT EXISTS analysis_sessions (
    analysis_id TEXT PRIMARY KEY,
    created_at TEXT,
    source TEXT,
    mode TEXT,
    request_json TEXT,
    result_json TEXT,
    updated_at TEXT
);
"""

BRANCH_ORDER = ["kline", "quant", "llm_debate", "intelligence", "macro"]
BRANCH_LABELS = {
    "kline": "K线分析",
    "quant": "传统量化分支",
    "llm_debate": "LLM 多空辩论",
    "intelligence": "多维智能融合",
    "macro": "宏观分支",
}

ANALYSIS_PRESETS: dict[str, dict[str, Any]] = {
    "quick_scan": {
        "id": "quick_scan",
        "label": "快速筛查",
        "description": "适合快速检查单股或小组合是否值得继续深挖。",
        "mode": "single",
        "branches": {
            "kline": {"enabled": True, "settings": {"prediction_horizon": "20d", "trend_window": "60d", "backend": "heuristic"}},
            "quant": {"enabled": True, "settings": {"factor_pack": "core", "rebalance": "monthly"}},
            "llm_debate": {"enabled": False, "settings": {"rounds": 2}},
            "intelligence": {"enabled": True, "settings": {"event_risk": True, "capital_flow": True}},
            "macro": {"enabled": True, "settings": {"overlay_strength": "medium"}},
        },
        "risk": {
            "capital": 1_000_000.0,
            "risk_level": "中等",
            "max_single_position": 0.2,
            "max_drawdown_limit": 0.15,
            "default_stop_loss": 0.08,
            "keep_cash_buffer": True,
        },
        "portfolio": {"candidate_limit": 8, "allocation_mode": "target_weight", "allow_cash_buffer": True},
        "llm_debate": {
            "enabled": False,
            "models": [],
            "rounds": 2,
            "assignment_mode": "random_balanced",
            "judge_mode": "auto",
            "judge_model": None,
        },
    },
    "single_deep_dive": {
        "id": "single_deep_dive",
        "label": "单股深研",
        "description": "适合对一只股票展开全维度研究，包含 LLM 多空辩论。",
        "mode": "single",
        "branches": {
            "kline": {"enabled": True, "settings": {"prediction_horizon": "60d", "trend_window": "120d", "backend": "heuristic"}},
            "quant": {"enabled": True, "settings": {"factor_pack": "expanded", "rebalance": "weekly"}},
            "llm_debate": {"enabled": True, "settings": {"rounds": 3}},
            "intelligence": {"enabled": True, "settings": {"event_risk": True, "capital_flow": True, "breadth": True}},
            "macro": {"enabled": True, "settings": {"overlay_strength": "medium"}},
        },
        "risk": {
            "capital": 1_000_000.0,
            "risk_level": "中等",
            "max_single_position": 0.25,
            "max_drawdown_limit": 0.18,
            "default_stop_loss": 0.08,
            "keep_cash_buffer": True,
        },
        "portfolio": {"candidate_limit": 5, "allocation_mode": "conviction_weight", "allow_cash_buffer": True},
        "llm_debate": {
            "enabled": True,
            "models": ["deepseek-chat", "gpt-4.1-mini", "claude-3-5-sonnet"],
            "rounds": 3,
            "assignment_mode": "random_balanced",
            "judge_mode": "auto",
            "judge_model": None,
        },
    },
    "portfolio_builder": {
        "id": "portfolio_builder",
        "label": "组合构建",
        "description": "适合多标的组合候选生成与仓位建议。",
        "mode": "holdings",
        "branches": {
            "kline": {"enabled": True, "settings": {"prediction_horizon": "20d", "trend_window": "60d", "backend": "heuristic"}},
            "quant": {"enabled": True, "settings": {"factor_pack": "portfolio", "rebalance": "monthly"}},
            "llm_debate": {"enabled": True, "settings": {"rounds": 2}},
            "intelligence": {"enabled": True, "settings": {"event_risk": True, "capital_flow": True, "breadth": True}},
            "macro": {"enabled": True, "settings": {"overlay_strength": "high"}},
        },
        "risk": {
            "capital": 1_000_000.0,
            "risk_level": "中等",
            "max_single_position": 0.2,
            "max_drawdown_limit": 0.12,
            "default_stop_loss": 0.08,
            "keep_cash_buffer": True,
        },
        "portfolio": {"candidate_limit": 10, "allocation_mode": "risk_budget", "allow_cash_buffer": True},
        "llm_debate": {
            "enabled": True,
            "models": ["deepseek-chat", "gpt-4.1-mini"],
            "rounds": 2,
            "assignment_mode": "random_balanced",
            "judge_mode": "auto",
            "judge_model": None,
        },
    },
    "risk_review": {
        "id": "risk_review",
        "label": "风控复核",
        "description": "更强调宏观和风险约束，适合复盘已有组合或候选池。",
        "mode": "holdings",
        "branches": {
            "kline": {"enabled": True, "settings": {"prediction_horizon": "20d", "trend_window": "60d", "backend": "heuristic"}},
            "quant": {"enabled": False, "settings": {"factor_pack": "core"}},
            "llm_debate": {"enabled": False, "settings": {"rounds": 2}},
            "intelligence": {"enabled": True, "settings": {"event_risk": True, "capital_flow": True, "breadth": True}},
            "macro": {"enabled": True, "settings": {"overlay_strength": "high"}},
        },
        "risk": {
            "capital": 1_000_000.0,
            "risk_level": "保守",
            "max_single_position": 0.15,
            "max_drawdown_limit": 0.1,
            "default_stop_loss": 0.06,
            "keep_cash_buffer": True,
        },
        "portfolio": {"candidate_limit": 6, "allocation_mode": "risk_budget", "allow_cash_buffer": True},
        "llm_debate": {
            "enabled": False,
            "models": [],
            "rounds": 2,
            "assignment_mode": "random_balanced",
            "judge_mode": "auto",
            "judge_model": None,
        },
    },
}

RISK_TEMPLATES = [
    {
        "id": "conservative",
        "label": "保守",
        "risk_level": "保守",
        "max_single_position": 0.15,
        "max_drawdown_limit": 0.1,
        "default_stop_loss": 0.06,
        "keep_cash_buffer": True,
    },
    {
        "id": "balanced",
        "label": "中等",
        "risk_level": "中等",
        "max_single_position": 0.2,
        "max_drawdown_limit": 0.15,
        "default_stop_loss": 0.08,
        "keep_cash_buffer": True,
    },
    {
        "id": "aggressive",
        "label": "积极",
        "risk_level": "积极",
        "max_single_position": 0.3,
        "max_drawdown_limit": 0.2,
        "default_stop_loss": 0.1,
        "keep_cash_buffer": False,
    },
]

LLM_MODELS = [
    {"id": "deepseek-chat", "label": "DeepSeek Chat", "provider": "DeepSeek", "env": "DEEPSEEK_API_KEY"},
    {"id": "gpt-4.1-mini", "label": "GPT-4.1 mini", "provider": "OpenAI", "env": "OPENAI_API_KEY"},
    {"id": "claude-3-5-sonnet", "label": "Claude 3.5 Sonnet", "provider": "Anthropic", "env": "ANTHROPIC_API_KEY"},
    {"id": "gemini-2.0-flash", "label": "Gemini 2.0 Flash", "provider": "Google", "env": "GOOGLE_API_KEY"},
    {"id": "qwen-plus", "label": "通义千问 Plus", "provider": "Alibaba", "env": "DASHSCOPE_API_KEY"},
]


def _ensure_results_dir() -> None:
    WEB_ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    JOB_DIR.mkdir(parents=True, exist_ok=True)


def _configure_sqlite_connection(conn: sqlite3.Connection, label: str) -> sqlite3.Connection:
    try:
        conn.execute("PRAGMA busy_timeout=60000")
    except sqlite3.DatabaseError:
        logger.warning("Failed to set busy_timeout for %s", label)

    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.DatabaseError:
        logger.warning("SQLite WAL unavailable for %s, falling back to default journal mode", label)

    return conn


def _connect_session_db() -> sqlite3.Connection:
    db_path = Path(APP_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=60)
    _configure_sqlite_connection(conn, str(db_path))
    conn.execute(SESSION_INIT_SQL)
    return conn


def _save_analysis_session(result: dict[str, Any]) -> None:
    conn = _connect_session_db()
    conn.execute(
        """
        INSERT OR REPLACE INTO analysis_sessions (
            analysis_id, created_at, source, mode, request_json, result_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(result.get("analysis_id", "")),
            str(result.get("created_at", datetime.now().isoformat(timespec="seconds"))),
            str(result.get("source", "web")),
            str(result.get("request", {}).get("mode", "single")),
            json.dumps(result.get("request", {}), ensure_ascii=False),
            json.dumps(result, ensure_ascii=False),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    conn.commit()
    conn.close()


def _analysis_python() -> str:
    if PROJECT_VENV_PYTHON.exists():
        return str(PROJECT_VENV_PYTHON)
    return sys.executable


def _result_file_for(analysis_id: str) -> Path:
    return WEB_ANALYSIS_DIR / f"analysis_{analysis_id}.json"


def _job_file_for(job_id: str) -> Path:
    return JOB_DIR / f"job_{job_id}.json"


def _analysis_id_from_job_id(job_id: str) -> str:
    parts = str(job_id).split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return str(job_id)


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _parse_legacy_datetime(value: str | None) -> str:
    if not value:
        return datetime.now().isoformat(timespec="seconds")
    try:
        return datetime.strptime(value, "%Y%m%d_%H%M%S").isoformat(timespec="seconds")
    except ValueError:
        return datetime.now().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_job_payload(job_id: str, payload: dict[str, Any]) -> None:
    _job_file_for(job_id).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _read_job_payload(job_id: str) -> dict[str, Any] | None:
    path = _job_file_for(job_id)
    raw = _load_json(path)
    return raw if isinstance(raw, dict) else None


def _session_rows() -> list[sqlite3.Row]:
    conn = _connect_session_db()
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """
            SELECT analysis_id, created_at, result_json
            FROM analysis_sessions
            WHERE source = 'web'
            ORDER BY created_at DESC
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _load_session_result(analysis_id: str) -> dict[str, Any] | None:
    conn = _connect_session_db()
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT result_json
            FROM analysis_sessions
            WHERE analysis_id = ?
            LIMIT 1
            """,
            (analysis_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()

    if not row:
        return None
    try:
        raw = json.loads(row["result_json"])
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return _normalize_web_result(raw)


def _default_branch_defaults() -> dict[str, dict[str, Any]]:
    preset = ANALYSIS_PRESETS["quick_scan"]["branches"]
    return {name: deepcopy(config) for name, config in preset.items()}


def _default_risk() -> dict[str, Any]:
    return deepcopy(ANALYSIS_PRESETS["quick_scan"]["risk"])


def _default_portfolio() -> dict[str, Any]:
    return deepcopy(ANALYSIS_PRESETS["quick_scan"]["portfolio"])


def _default_llm_debate() -> dict[str, Any]:
    return deepcopy(ANALYSIS_PRESETS["quick_scan"]["llm_debate"])


def _normalize_targets(payload: dict[str, Any]) -> list[str]:
    targets = payload.get("targets") or payload.get("stocks") or []
    if isinstance(targets, list):
        normalized = [str(item).strip().upper() for item in targets if str(item).strip()]
        return list(dict.fromkeys(normalized))
    return []


def _connect_stock_db() -> sqlite3.Connection:
    conn = sqlite3.connect(STOCK_DB_PATH, timeout=60)
    _configure_sqlite_connection(conn, STOCK_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _analysis_timeout_seconds(payload: dict[str, Any]) -> int:
    override = os.environ.get("ANALYSIS_JOB_TIMEOUT_SECONDS", "").strip()
    if override:
        try:
            return max(60, int(override))
        except ValueError:
            logger.warning("Ignoring invalid ANALYSIS_JOB_TIMEOUT_SECONDS=%s", override)

    targets = payload.get("targets") or payload.get("stocks") or []
    target_count = len(targets) if isinstance(targets, list) else 0
    mode = str(payload.get("mode") or "single")

    if mode == "market":
        return min(
            LARGE_ANALYSIS_TIMEOUT_CAP_SECONDS,
            max(1800, 600 + target_count * 4),
        )
    if target_count >= 100:
        return min(3600, 300 + target_count * 8)
    if target_count >= 20:
        return min(1800, 300 + target_count * 12)
    return DEFAULT_ANALYSIS_TIMEOUT_SECONDS


def _job_stale_timeout(payload: dict[str, Any]) -> timedelta:
    mode = str(payload.get("mode") or "")
    if mode != "market":
        return STALE_JOB_TIMEOUT

    targets = payload.get("targets") or payload.get("stocks") or []
    target_count = len(targets) if isinstance(targets, list) else int(payload.get("target_count") or 0)
    timeout_seconds = min(
        LARGE_ANALYSIS_TIMEOUT_CAP_SECONDS + 1800,
        max(5400, 1200 + target_count * 2),
    )
    return timedelta(seconds=timeout_seconds)


def _market_batch_size(payload: dict[str, Any]) -> int:
    override = os.environ.get("MARKET_ANALYSIS_BATCH_SIZE", "").strip()
    if override:
        try:
            return max(10, int(override))
        except ValueError:
            logger.warning("Ignoring invalid MARKET_ANALYSIS_BATCH_SIZE=%s", override)

    market = str(payload.get("market") or "CN").upper()
    return DEFAULT_MARKET_BATCH_SIZE_CN if market == "CN" else DEFAULT_MARKET_BATCH_SIZE_US


def _market_lot_size(symbol: str, market: str) -> int:
    return 100 if market.upper() == "CN" else 1


def _average(values: list[float], default: float = 0.0) -> float:
    cleaned = [float(value) for value in values if value is not None]
    return sum(cleaned) / len(cleaned) if cleaned else default


def _normalize_rank_weights(
    raw_scores: dict[str, float],
    total_target_exposure: float,
    max_single_weight: float,
) -> dict[str, float]:
    positive_scores = {symbol: score for symbol, score in raw_scores.items() if score > 0}
    if not positive_scores or total_target_exposure <= 0:
        return {}

    remaining = dict(positive_scores)
    weights = {symbol: 0.0 for symbol in positive_scores}
    remaining_exposure = total_target_exposure

    while remaining and remaining_exposure > 1e-8:
        total_score = sum(remaining.values())
        if total_score <= 0:
            break

        overflow_symbols: list[str] = []
        for symbol, score in list(remaining.items()):
            proposed = remaining_exposure * score / total_score
            if proposed > max_single_weight + 1e-8:
                weights[symbol] = max_single_weight
                remaining_exposure -= max_single_weight
                overflow_symbols.append(symbol)

        if overflow_symbols:
            for symbol in overflow_symbols:
                remaining.pop(symbol, None)
            continue

        for symbol, score in remaining.items():
            weights[symbol] = remaining_exposure * score / total_score
        break

    return {symbol: weight for symbol, weight in weights.items() if weight > 0}


def _rank_market_recommendation(item: dict[str, Any]) -> float:
    action_multiplier = 1.0 if str(item.get("action", "watch")) == "buy" else 0.45
    return (
        max(float(item.get("suggested_weight", 0.0)), 0.001)
        * (1 + max(float(item.get("consensus_score", 0.0)), 0.0))
        * (0.8 + max(float(item.get("confidence", 0.0)), 0.0))
        * (1 + max(float(item.get("branch_positive_count", 0.0)), 0.0) / 5.0)
        * action_multiplier
    )


def _build_market_trade_recommendations(
    recommendations: list[dict[str, Any]],
    market: str,
    capital: float,
    target_exposure: float,
    candidate_limit: int,
    max_single_position: float,
) -> list[dict[str, Any]]:
    ranked: dict[str, dict[str, Any]] = {}
    for item in recommendations:
        symbol = str(item.get("symbol", "")).strip().upper()
        if not symbol:
            continue
        scored = dict(item)
        scored["symbol"] = symbol
        scored["rank_score"] = _rank_market_recommendation(scored)
        existing = ranked.get(symbol)
        if existing is None or float(scored["rank_score"]) > float(existing.get("rank_score", 0.0)):
            ranked[symbol] = scored

    selected = sorted(
        ranked.values(),
        key=lambda item: (
            float(item.get("rank_score", 0.0)),
            float(item.get("confidence", 0.0)),
            float(item.get("consensus_score", 0.0)),
        ),
        reverse=True,
    )[: max(candidate_limit, 1)]

    weight_map = _normalize_rank_weights(
        {item["symbol"]: float(item.get("rank_score", 0.0)) for item in selected},
        total_target_exposure=target_exposure,
        max_single_weight=max_single_position,
    )

    normalized: list[dict[str, Any]] = []
    lot_size_default = _market_lot_size("", market)
    for item in selected:
        symbol = str(item["symbol"])
        raw_weight = weight_map.get(symbol, 0.0)
        entry_price = float(item.get("recommended_entry_price") or item.get("current_price") or 0.0)
        if entry_price <= 0 or raw_weight <= 0:
            continue
        lot_size = _market_lot_size(symbol, market) or lot_size_default
        suggested_shares = int((capital * raw_weight) // entry_price // lot_size) * lot_size
        suggested_amount = round(suggested_shares * entry_price, 2)
        suggested_weight = suggested_amount / capital if capital > 0 else 0.0
        if suggested_shares <= 0 or suggested_amount <= 0:
            continue

        next_item = dict(item)
        next_item["action"] = "buy"
        next_item["suggested_shares"] = suggested_shares
        next_item["suggested_amount"] = suggested_amount
        next_item["suggested_weight"] = round(suggested_weight, 4)
        normalized.append(next_item)

    return normalized


def _build_market_report(
    request: dict[str, Any],
    batch_count: int,
    scanned_count: int,
    recommendations: list[dict[str, Any]],
    target_exposure: float,
    style_bias: str,
    warnings: list[str],
) -> str:
    market_label = "A股" if str(request.get("market", "CN")).upper() == "CN" else "美股"
    lines = [
        f"# {market_label}全市场分批扫描报告",
        "",
        f"- 分析模式: 全市场分批扫描",
        f"- 扫描股票数: {scanned_count}",
        f"- 批次数量: {batch_count}",
        f"- 目标仓位: {target_exposure:.1%}",
        f"- 风格偏好: {style_bias}",
        "",
        "## 候选标的",
    ]
    if recommendations:
        for index, item in enumerate(recommendations, start=1):
            lines.append(
                f"{index}. {item['symbol']} | 仓位 {float(item.get('suggested_weight', 0.0)):.1%} | "
                f"建议买入 {float(item.get('recommended_entry_price', 0.0)):.2f} | "
                f"目标价 {float(item.get('target_price', 0.0)):.2f} | "
                f"止损价 {float(item.get('stop_loss_price', 0.0)):.2f}"
            )
    else:
        lines.append("暂无满足条件的候选标的。")

    if warnings:
        lines.extend(["", "## 执行提醒"])
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines)


def _run_market_analysis(
    payload: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    from web.tasks.run_analysis_job import run_job

    normalized_request = deepcopy(payload)
    targets = list(normalized_request.get("targets") or normalized_request.get("stocks") or [])
    if not targets:
        raise ValueError("当前市场下没有可分析的股票")

    market = str(normalized_request.get("market", "CN")).upper()
    batch_size = _market_batch_size(normalized_request)
    batches = [targets[index : index + batch_size] for index in range(0, len(targets), batch_size)]
    total_batches = len(batches)
    capital = float(normalized_request.get("risk", {}).get("capital", 1_000_000.0))
    candidate_limit = int(normalized_request.get("portfolio", {}).get("candidate_limit", 12) or 12)
    max_single_position = float(normalized_request.get("risk", {}).get("max_single_position", 0.2) or 0.2)

    warnings: list[str] = []
    if normalized_request.get("branches", {}).get("llm_debate", {}).get("enabled"):
        normalized_request["branches"]["llm_debate"]["enabled"] = False
        normalized_request["llm_debate"]["enabled"] = False
        normalized_request["llm_debate"]["models"] = []
        normalized_request["llm_debate"]["assignments"] = []
        warnings.append("全市场分析已自动关闭 LLM 多空辩论分支，以避免超长时延和过高调用成本。")

    branch_scores: dict[str, list[float]] = {name: [] for name in BRANCH_ORDER}
    branch_confidences: dict[str, list[float]] = {name: [] for name in BRANCH_ORDER}
    branch_top_symbols: dict[str, Counter[str]] = {name: Counter() for name in BRANCH_ORDER}
    branch_risks: dict[str, list[str]] = {name: [] for name in BRANCH_ORDER}
    risk_volatility: list[float] = []
    risk_drawdown: list[float] = []
    risk_sharpe: list[float] = []
    style_counter: Counter[str] = Counter()
    batch_target_exposures: list[float] = []
    aggregated_recommendations: list[dict[str, Any]] = []
    execution_log: list[str] = []
    started_at = time.perf_counter()

    for index, batch_targets in enumerate(batches, start=1):
        if progress_callback is not None:
            progress_callback(
                {
                    "status_message": f"全市场分批扫描中：第 {index}/{total_batches} 批，{len(batch_targets)} 只股票",
                    "progress": {"current_batch": index, "total_batches": total_batches},
                }
            )

        batch_request = deepcopy(normalized_request)
        batch_request["targets"] = batch_targets
        batch_request["stocks"] = batch_targets

        batch_started = time.perf_counter()
        batch_result = run_job(batch_request)
        batch_elapsed = time.perf_counter() - batch_started
        execution_log.append(
            f"[{datetime.now().strftime('%H:%M:%S')}] 第 {index}/{total_batches} 批完成，"
            f"扫描 {len(batch_targets)} 只，耗时 {batch_elapsed:.1f}s。"
        )

        batch_target_exposures.append(float(batch_result.get("target_exposure", 0.0)))
        style_counter[str(batch_result.get("style_bias", "均衡"))] += 1
        aggregated_recommendations.extend(
            item for item in batch_result.get("trade_recommendations", []) if isinstance(item, dict)
        )

        batch_risk = _ensure_dict(batch_result.get("risk"))
        risk_volatility.append(float(batch_risk.get("volatility", 0.0)))
        risk_drawdown.append(float(batch_risk.get("max_drawdown", 0.0)))
        risk_sharpe.append(float(batch_risk.get("sharpe_ratio", 0.0)))
        warnings.extend(str(item) for item in _ensure_list(batch_risk.get("warnings")))

        for branch in batch_result.get("branches", []):
            if not isinstance(branch, dict):
                continue
            branch_name = str(branch.get("branch_name", ""))
            if branch_name not in branch_scores:
                continue
            branch_scores[branch_name].append(float(branch.get("score", 0.0)))
            branch_confidences[branch_name].append(float(branch.get("confidence", 0.0)))
            branch_top_symbols[branch_name].update(str(item) for item in _ensure_list(branch.get("top_symbols")))
            branch_risks[branch_name].extend(str(item) for item in _ensure_list(branch.get("risks")))

    target_exposure = min(max(_average(batch_target_exposures, default=0.3), 0.1), 0.8)
    style_bias = style_counter.most_common(1)[0][0] if style_counter else "均衡"
    final_recommendations = _build_market_trade_recommendations(
        aggregated_recommendations,
        market=market,
        capital=capital,
        target_exposure=target_exposure,
        candidate_limit=candidate_limit,
        max_single_position=max_single_position,
    )
    candidate_symbols = [item["symbol"] for item in final_recommendations]

    deduped_warnings = list(dict.fromkeys(item for item in warnings if item))[:6]
    if not final_recommendations:
        deduped_warnings.append("本次全市场扫描未生成可执行候选，建议调整风险参数或等待市场信号改善。")

    branches = []
    for branch_name in BRANCH_ORDER:
        enabled = bool(normalized_request.get("branches", {}).get(branch_name, {}).get("enabled", True))
        top_symbols = [symbol for symbol, _ in branch_top_symbols[branch_name].most_common(5)]
        branches.append(
            {
                "branch_name": branch_name,
                "enabled": enabled,
                "score": round(_average(branch_scores[branch_name]), 4),
                "confidence": round(_average(branch_confidences[branch_name]), 4),
                "explanation": (
                    f"全市场分批扫描共 {total_batches} 批，平均分支得分 {_average(branch_scores[branch_name]):+.2f}。"
                    if enabled else "全市场扫描中未启用该分支。"
                ),
                "risks": list(dict.fromkeys(branch_risks[branch_name]))[:3],
                "top_symbols": top_symbols,
                "branch_mode": "market_batch",
                "settings": deepcopy(_ensure_dict(normalized_request.get("branches", {}).get(branch_name, {}).get("settings"))),
                "model_assignment": deepcopy(_ensure_list(normalized_request.get("llm_debate", {}).get("assignments", []))) if branch_name == "llm_debate" else [],
                "signals": {
                    "aggregation": "market_batch",
                    "batch_count": total_batches,
                    "scanned_symbols": len(targets),
                },
                "metadata": {
                    "batch_count": total_batches,
                    "scanned_symbols": len(targets),
                },
            }
        )

    market_label = "A股" if market == "CN" else "美股"
    execution_notes = [
        f"{market_label}全市场已按 {total_batches} 批完成扫描，共覆盖 {len(targets)} 只股票。",
        f"最终筛出 {len(final_recommendations)} 只候选，目标仓位 {target_exposure:.1%}。",
    ]
    execution_notes.extend(deduped_warnings[:3])

    result = {
        "analysis_id": str(normalized_request.get("analysis_id") or datetime.now().strftime("%Y%m%d_%H%M%S")),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "web",
        "request": normalized_request,
        "total_time": round(time.perf_counter() - started_at, 2),
        "research_mode": "market_batch",
        "final_decision": _decision_text(
            {
                "candidate_symbols": candidate_symbols,
                "target_exposure": target_exposure,
                "risk": {"risk_level": normalized_request.get("risk", {}).get("risk_level", "中等")},
            }
        ),
        "target_exposure": target_exposure,
        "style_bias": style_bias,
        "sector_preferences": [],
        "candidate_symbols": candidate_symbols,
        "execution_notes": execution_notes,
        "branches": branches,
        "risk": {
            "risk_level": str(normalized_request.get("risk", {}).get("risk_level", "中等")),
            "volatility": round(_average(risk_volatility), 4),
            "max_drawdown": round(_average(risk_drawdown), 4),
            "sharpe_ratio": round(_average(risk_sharpe), 4),
            "warnings": deduped_warnings,
            "max_single_position": max_single_position,
            "max_drawdown_limit": float(normalized_request.get("risk", {}).get("max_drawdown_limit", 0.15)),
            "default_stop_loss": float(normalized_request.get("risk", {}).get("default_stop_loss", 0.08)),
            "keep_cash_buffer": bool(normalized_request.get("risk", {}).get("keep_cash_buffer", True)),
            "stress_test": deduped_warnings[-1] if deduped_warnings else "",
        },
        "trade_recommendations": final_recommendations,
        "report_markdown": _build_market_report(
            normalized_request,
            batch_count=total_batches,
            scanned_count=len(targets),
            recommendations=final_recommendations,
            target_exposure=target_exposure,
            style_bias=style_bias,
            warnings=deduped_warnings,
        ),
        "execution_log": execution_log,
        "llm_assignments": deepcopy(_ensure_list(normalized_request.get("llm_debate", {}).get("assignments", []))),
        "config_applied": {
            "preset": normalized_request.get("preset"),
            "mode": "market",
            "enabled_branches": [
                name
                for name, branch in normalized_request.get("branches", {}).items()
                if branch.get("enabled")
            ],
        },
    }
    return result


def _targets_from_portfolio_state(key: str, market: str) -> list[str]:
    from web.services import portfolio_service

    state = portfolio_service.get_portfolio_state()
    items = state.get(key, [])
    normalized = [
        str(item.get("symbol", "")).strip().upper()
        for item in items
        if str(item.get("market", "")).upper() == market.upper() and str(item.get("symbol", "")).strip()
    ]
    return list(dict.fromkeys(normalized))


def _targets_for_market(market: str) -> list[str]:
    conn = _connect_stock_db()
    try:
        rows = conn.execute(
            """
            SELECT ts_code
            FROM stock_list
            WHERE market = ?
              AND EXISTS (
                  SELECT 1
                  FROM daily_data d
                  WHERE d.ts_code = stock_list.ts_code
              )
            ORDER BY ts_code ASC
            """,
            (market.upper(),),
        ).fetchall()
    finally:
        conn.close()
    return [str(row["ts_code"]).upper() for row in rows if str(row["ts_code"]).strip()]


def _ensure_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _ensure_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _merge_branch_settings(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(base)
    if not override:
        return merged
    if "enabled" in override:
        merged["enabled"] = bool(override["enabled"])
    settings = deepcopy(merged.get("settings", {}))
    settings.update(_ensure_dict(override.get("settings")))
    merged["settings"] = settings
    return merged


def _risk_template_for(level: str | None) -> dict[str, Any]:
    normalized = str(level or "中等")
    for template in RISK_TEMPLATES:
        if template["label"] == normalized or template["risk_level"] == normalized:
            return deepcopy(template)
    return deepcopy(RISK_TEMPLATES[1])


def _build_llm_assignments(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = [str(item).strip() for item in config.get("models", []) if str(item).strip()]
    if not models:
        return []

    shuffled = models[:]
    random.shuffle(shuffled)

    assignments: list[dict[str, Any]] = []
    if len(shuffled) == 1:
        return [{"model": shuffled[0], "role": "solo"}]

    judge_model: str | None = None
    remaining = shuffled
    if len(shuffled) >= 3 and config.get("judge_mode", "auto") == "auto":
        judge_model = shuffled[-1]
        remaining = shuffled[:-1]

    bull_models = remaining[::2]
    bear_models = remaining[1::2]

    for model in bull_models:
        assignments.append({"model": model, "role": "bull"})
    for model in bear_models:
        assignments.append({"model": model, "role": "bear"})

    if judge_model:
        assignments.append({"model": judge_model, "role": "judge"})
    elif len(shuffled) == 2:
        assignments.append({"model": "ensemble", "role": "judge"})

    return assignments


def _normalize_request_payload(raw: dict[str, Any]) -> dict[str, Any]:
    payload = deepcopy(raw)
    preset_id = str(payload.get("preset", "quick_scan"))
    preset = deepcopy(ANALYSIS_PRESETS.get(preset_id, ANALYSIS_PRESETS["quick_scan"]))

    raw_targets = _normalize_targets(payload)
    mode = str(payload.get("mode") or ("holdings" if len(raw_targets) > 1 else "single"))
    market = str(payload.get("market", "CN")).upper()
    if mode == "market":
        targets = _targets_for_market(market)
    elif mode == "holdings":
        targets = raw_targets or _targets_from_portfolio_state("holdings", market)
    elif mode == "watchlist":
        targets = raw_targets or _targets_from_portfolio_state("watchlist", market)
    else:
        targets = raw_targets

    branches = _default_branch_defaults()
    for name, config in preset.get("branches", {}).items():
        branches[name] = deepcopy(config)
    for name, config in (payload.get("branches") or {}).items():
        if name not in branches:
            branches[name] = {"enabled": True, "settings": {}}
        branches[name] = _merge_branch_settings(branches[name], config)

    legacy_to_branch = {
        "enable_macro": "macro",
        "enable_kronos": "kline",
        "enable_kline": "kline",
        "enable_intelligence": "intelligence",
        "enable_llm_debate": "llm_debate",
    }
    for legacy_key, branch_name in legacy_to_branch.items():
        if payload.get(legacy_key) is not None:
            branches[branch_name]["enabled"] = bool(payload[legacy_key])

    risk = _risk_template_for(
        (payload.get("risk") or {}).get("risk_level") or payload.get("risk_level") or preset["risk"].get("risk_level")
    )
    risk.update(deepcopy(preset.get("risk", {})))
    risk.update(deepcopy(payload.get("risk", {})))
    if payload.get("capital") is not None:
        risk["capital"] = float(payload["capital"])
    if payload.get("risk_level") is not None:
        risk["risk_level"] = str(payload["risk_level"])

    portfolio = _default_portfolio()
    portfolio.update(deepcopy(preset.get("portfolio", {})))
    portfolio.update(deepcopy(payload.get("portfolio", {})))

    llm_debate = _default_llm_debate()
    llm_debate.update(deepcopy(preset.get("llm_debate", {})))
    llm_debate.update(deepcopy(payload.get("llm_debate", {})))
    llm_debate["enabled"] = bool(branches.get("llm_debate", {}).get("enabled", llm_debate.get("enabled", True)))
    llm_debate["assignments"] = _build_llm_assignments(llm_debate)
    branches["llm_debate"] = _merge_branch_settings(
        branches.get("llm_debate", {"enabled": llm_debate["enabled"], "settings": {}}),
        {"enabled": llm_debate["enabled"], "settings": {"rounds": llm_debate.get("rounds", 2)}},
    )

    normalized = {
        "mode": mode,
        "targets": targets,
        "preset": preset_id,
        "market": market,
        "branches": {name: branches[name] for name in BRANCH_ORDER if name in branches},
        "risk": risk,
        "portfolio": portfolio,
        "llm_debate": llm_debate,
        "stocks": targets,
        "capital": float(risk.get("capital", 1_000_000.0)),
        "risk_level": str(risk.get("risk_level", "中等")),
        "enable_macro": bool(branches.get("macro", {}).get("enabled", True)),
        "enable_kline": bool(branches.get("kline", {}).get("enabled", True)),
        "enable_kronos": bool(branches.get("kline", {}).get("enabled", True)),
        "enable_intelligence": bool(branches.get("intelligence", {}).get("enabled", True)),
        "enable_llm_debate": bool(branches.get("llm_debate", {}).get("enabled", True)),
    }
    return normalized


def _title_for(payload: dict[str, Any]) -> str:
    stocks = payload.get("targets") or payload.get("stocks") or []
    market = payload.get("market", "CN")
    mode = {"single": "单只股票", "holdings": "我的持仓", "watchlist": "自选池", "market": "全市场"}.get(
        str(payload.get("mode", "single")),
        "研究任务",
    )
    preset = ANALYSIS_PRESETS.get(str(payload.get("preset", "quick_scan")), ANALYSIS_PRESETS["quick_scan"])
    return f"{market} · {mode} · {len(stocks)} 只标的 · {preset['label']}"


def _decision_text(result: dict[str, Any]) -> str:
    exposure = float(result.get("target_exposure", 0.0))
    risk_level = str(result.get("risk", {}).get("risk_level", "unknown"))
    if not result.get("candidate_symbols"):
        return "继续观察，暂不形成明确建仓候选。"
    if exposure >= 0.6:
        return f"可积极配置候选标的，当前风险状态为 {risk_level}。"
    if exposure >= 0.3:
        return f"以分批建仓为主，当前风险状态为 {risk_level}。"
    return f"以轻仓试探或继续观察为主，当前风险状态为 {risk_level}。"


def _build_symbol_decision(item: dict[str, Any]) -> dict[str, Any]:
    risk_flags = [str(flag) for flag in item.get("risk_flags", [])]
    rationale_parts = [
        f"共识得分 {float(item.get('consensus_score', 0.0)):+.2f}",
        f"支持分支 {int(item.get('branch_positive_count', 0))}/5",
    ]
    if item.get("trend_regime"):
        rationale_parts.append(f"趋势状态 {item['trend_regime']}")
    if risk_flags:
        rationale_parts.append(f"风险提示：{'；'.join(risk_flags[:2])}")
    return {
        "symbol": str(item.get("symbol", "")),
        "action": str(item.get("action", "watch")),
        "current_price": float(item.get("current_price", 0.0)),
        "recommended_entry_price": float(item.get("recommended_entry_price", 0.0)),
        "target_price": float(item.get("target_price", 0.0)),
        "stop_loss_price": float(item.get("stop_loss_price", 0.0)),
        "suggested_weight": float(item.get("suggested_weight", 0.0)),
        "suggested_amount": float(item.get("suggested_amount", 0.0)),
        "suggested_shares": int(item.get("suggested_shares", 0)),
        "confidence": float(item.get("confidence", 0.0)),
        "consensus_score": float(item.get("consensus_score", 0.0)),
        "branch_positive_count": int(item.get("branch_positive_count", 0)),
        "trend_regime": str(item.get("trend_regime", "")),
        "risk_flags": risk_flags,
        "rationale": "；".join(rationale_parts),
    }


def _normalize_web_result(raw: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw)
    request = _normalize_request_payload(payload.get("request", {}))
    raw_branches = payload.get("branches", [])
    if isinstance(raw_branches, dict):
        raw_branches = [
            {"branch_name": name, **branch} if isinstance(branch, dict) else {"branch_name": name}
            for name, branch in raw_branches.items()
        ]
    elif not isinstance(raw_branches, list):
        raw_branches = []
    normalized_branches = []
    for branch in raw_branches:
        if not isinstance(branch, dict):
            continue
        branch_name = str(branch.get("branch_name", "unknown"))
        branch_config = request.get("branches", {}).get(branch_name, {"enabled": True, "settings": {}})
        normalized_branches.append(
            {
                "branch_name": branch_name,
                "enabled": bool(branch_config.get("enabled", True)),
                "score": float(branch.get("score", 0.0)),
                "confidence": float(branch.get("confidence", 0.0)),
                "explanation": str(branch.get("explanation", "")),
                "risks": [str(item) for item in _ensure_list(branch.get("risks"))],
                "top_symbols": [str(item) for item in _ensure_list(branch.get("top_symbols"))],
                "branch_mode": str(branch.get("branch_mode", branch_config.get("settings", {}).get("branch_mode", ""))) or None,
                "settings": deepcopy(_ensure_dict(branch_config.get("settings"))),
                "model_assignment": _ensure_list(
                    request.get("llm_debate", {}).get("assignments", []) if branch_name == "llm_debate" else []
                ),
                "signals": deepcopy(_ensure_dict(branch.get("signals"))),
                "metadata": deepcopy(_ensure_dict(branch.get("metadata"))),
            }
        )

    trade_recommendations = [
        _build_symbol_decision(item)
        for item in payload.get("trade_recommendations", [])
        if isinstance(item, dict)
    ]
    capital = float(request.get("risk", {}).get("capital", 0.0))
    target_exposure = float(payload.get("target_exposure", 0.0))
    risk_review = {
        "risk_level": str(payload.get("risk", {}).get("risk_level", "unknown")),
        "volatility": float(payload.get("risk", {}).get("volatility", 0.0)),
        "max_drawdown": float(payload.get("risk", {}).get("max_drawdown", 0.0)),
        "sharpe_ratio": float(payload.get("risk", {}).get("sharpe_ratio", 0.0)),
        "warnings": [str(item) for item in payload.get("risk", {}).get("warnings", [])],
        "max_single_position": float(request.get("risk", {}).get("max_single_position", 0.2)),
        "max_drawdown_limit": float(request.get("risk", {}).get("max_drawdown_limit", 0.15)),
        "default_stop_loss": float(request.get("risk", {}).get("default_stop_loss", 0.08)),
        "keep_cash_buffer": bool(request.get("risk", {}).get("keep_cash_buffer", True)),
        "stress_test": (
            str(payload.get("risk", {}).get("warnings", [])[-1])
            if payload.get("risk", {}).get("warnings")
            else ""
        ),
    }
    normalized = {
        "analysis_id": str(payload.get("analysis_id", datetime.now().strftime("%Y%m%d_%H%M%S"))),
        "created_at": str(payload.get("created_at", datetime.now().isoformat(timespec="seconds"))),
        "source": str(payload.get("source", "web")),
        "request": request,
        "total_time": float(payload.get("total_time", 0.0)),
        "research_mode": str(payload.get("research_mode", "production")),
        "final_decision": str(payload.get("final_decision") or _decision_text(payload)),
        "target_exposure": target_exposure,
        "style_bias": str(payload.get("style_bias", "均衡")),
        "sector_preferences": [str(item) for item in payload.get("sector_preferences", [])],
        "candidate_symbols": [str(item) for item in payload.get("candidate_symbols", [])],
        "execution_notes": [str(item) for item in payload.get("execution_notes", [])],
        "branches": normalized_branches,
        "risk": risk_review,
        "execution_plan": {
            "capital": capital,
            "target_exposure": target_exposure,
            "investable_capital": capital * target_exposure,
            "reserved_cash": max(capital - capital * target_exposure, 0.0),
            "symbol_decisions": trade_recommendations,
        },
        "trade_recommendations": trade_recommendations,
        "report_markdown": str(payload.get("report_markdown") or payload.get("report", "")),
        "execution_log": [str(item) for item in payload.get("execution_log", [])],
        "llm_assignments": request.get("llm_debate", {}).get("assignments", []),
        "config_applied": {
            "preset": request.get("preset"),
            "mode": request.get("mode"),
            "enabled_branches": [
                name
                for name, branch in request.get("branches", {}).items()
                if branch.get("enabled")
            ],
        },
    }
    return normalized


def _normalize_legacy_result(raw: dict[str, Any], source: str) -> dict[str, Any]:
    market = str(raw.get("market") or ("US" if "us_backtest" in source else "CN")).upper()
    strategy = raw.get("strategy", {})
    timestamp = raw.get("timestamp")

    branches = []
    for branch_name, branch in (raw.get("branches") or {}).items():
        branches.append(
            {
                "branch_name": branch_name,
                "score": float(branch.get("score", 0.0)),
                "confidence": float(branch.get("confidence", 0.0)),
                "explanation": str(branch.get("signals_summary", "")),
                "risks": [],
                "top_symbols": [],
            }
        )

    normalized = {
        "analysis_id": timestamp or f"legacy_{Path(source).stem}",
        "created_at": _parse_legacy_datetime(timestamp),
        "source": "legacy",
        "request": {
            "targets": raw.get("stocks", []),
            "market": market,
            "preset": "portfolio_builder" if len(raw.get("stocks", [])) > 1 else "single_deep_dive",
            "mode": "holdings" if len(raw.get("stocks", [])) > 1 else "single",
        },
        "total_time": 0.0,
        "research_mode": "legacy",
        "target_exposure": float(strategy.get("target_exposure", 0.0)),
        "style_bias": str(strategy.get("style_bias", "均衡")),
        "sector_preferences": [str(item) for item in strategy.get("sector_preferences", [])],
        "candidate_symbols": [str(item) for item in strategy.get("candidate_symbols", [])],
        "execution_notes": [],
        "branches": branches,
        "risk": {
            "risk_level": str(strategy.get("risk_summary", {}).get("risk_level", "unknown")),
            "volatility": float(strategy.get("risk_summary", {}).get("volatility", 0.0)),
            "max_drawdown": float(strategy.get("risk_summary", {}).get("max_drawdown", 0.0)),
            "sharpe_ratio": 0.0,
            "warnings": [str(item) for item in strategy.get("risk_summary", {}).get("warnings", [])],
        },
        "trade_recommendations": [],
        "report_markdown": str(raw.get("report", "")),
        "execution_log": [],
    }
    return _normalize_web_result(normalized)


def _iter_result_paths() -> list[tuple[str, Path]]:
    _ensure_results_dir()
    paths: list[tuple[str, Path]] = []
    for path in sorted(WEB_ANALYSIS_DIR.glob("analysis_*.json"), reverse=True):
        if WEB_RESULT_FILE_RE.match(path.name):
            paths.append(("web", path))
    for path in sorted((RESULTS_DIR / "cn_analysis").glob("analysis_*.json"), reverse=True):
        paths.append(("legacy_cn", path))
    for path in sorted((RESULTS_DIR / "us_backtest").glob("analysis_*.json"), reverse=True):
        paths.append(("legacy_us", path))
    return paths


def _load_normalized_result(path: Path, source: str) -> dict[str, Any] | None:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        return None
    if source == "web":
        return _normalize_web_result(raw)
    return _normalize_legacy_result(raw, str(path))


def get_analysis_options() -> dict[str, Any]:
    return {
        "presets": [
            {
                "id": preset["id"],
                "label": preset["label"],
                "description": preset["description"],
                "mode": preset["mode"],
                "defaults": {
                    "branches": deepcopy(preset["branches"]),
                    "risk": deepcopy(preset["risk"]),
                    "portfolio": deepcopy(preset["portfolio"]),
                    "llm_debate": deepcopy(preset["llm_debate"]),
                },
            }
            for preset in ANALYSIS_PRESETS.values()
        ],
        "branch_defaults": {
            name: {"enabled": True, "settings": {}}
            for name in BRANCH_ORDER
        },
        "llm_models": [
            {
                "id": item["id"],
                "label": item["label"],
                "provider": item["provider"],
                "enabled": bool(os.environ.get(item["env"])),
                "note": None if os.environ.get(item["env"]) else "未配置对应 API Key",
            }
            for item in LLM_MODELS
        ],
        "risk_templates": deepcopy(RISK_TEMPLATES),
    }


def run_analysis(
    payload: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _ensure_results_dir()
    normalized_request = _normalize_request_payload(payload)
    timeout_seconds = _analysis_timeout_seconds(normalized_request)

    analysis_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = _result_file_for(analysis_id)
    job_payload = dict(normalized_request)
    job_payload["analysis_id"] = analysis_id

    if normalized_request.get("mode") == "market":
        raw_result = _run_market_analysis(job_payload, progress_callback=progress_callback)
        output_path.write_text(json.dumps(raw_result, ensure_ascii=False, indent=2), encoding="utf-8")
        normalized_result = _normalize_web_result(raw_result)
        _save_analysis_session(normalized_result)
        return normalized_result

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        prefix=f"analysis_request_{analysis_id}_",
        dir=WEB_ANALYSIS_DIR,
        encoding="utf-8",
        delete=False,
    ) as handle:
        request_path = Path(handle.name)
        json.dump(job_payload, handle, ensure_ascii=False, indent=2)

    try:
        try:
            completed = subprocess.run(
                [
                    _analysis_python(),
                    str(RUNNER_PATH),
                    "--payload-file",
                    str(request_path),
                    "--output-file",
                    str(output_path),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"分析超时：模式 {normalized_request.get('mode')}，标的 {len(normalized_request.get('targets') or [])} 只，"
                f"已等待 {timeout_seconds} 秒。"
            ) from exc
    finally:
        request_path.unlink(missing_ok=True)

    if completed.returncode != 0:
        error_message = completed.stderr.strip() or completed.stdout.strip() or "分析任务失败"
        output_path.unlink(missing_ok=True)
        raise RuntimeError(error_message)

    result = _load_json(output_path)
    if not isinstance(result, dict):
        raise RuntimeError("分析结果文件无效")

    normalized_result = _normalize_web_result(result)
    _save_analysis_session(normalized_result)
    return normalized_result


def _run_analysis_job(job_id: str, payload: dict[str, Any]) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    normalized_request = _normalize_request_payload(payload)
    target_count = len(normalized_request.get("targets") or [])
    try:
        _write_job_payload(
            job_id,
            {
                "ok": True,
                "job_id": job_id,
                "status": "running",
                "created_at": now,
                "updated_at": now,
                "result": None,
                "error": None,
                "mode": normalized_request.get("mode"),
                "target_count": target_count,
            },
        )

        def _update_progress(extra: dict[str, Any]) -> None:
            payload_update = {
                "ok": True,
                "job_id": job_id,
                "status": "running",
                "created_at": now,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "result": None,
                "error": None,
                "mode": normalized_request.get("mode"),
                "target_count": target_count,
            }
            payload_update.update(extra)
            _write_job_payload(job_id, payload_update)

        result = run_analysis(payload, progress_callback=_update_progress)
        completed_at = datetime.now().isoformat(timespec="seconds")
        _write_job_payload(
            job_id,
            {
                "ok": True,
                "job_id": job_id,
                "status": "completed",
                "created_at": now,
                "updated_at": completed_at,
                "result": result,
                "error": None,
                "mode": normalized_request.get("mode"),
                "target_count": target_count,
            },
        )
    except Exception as exc:
        logger.exception("Analysis job %s failed", job_id)
        failed_at = datetime.now().isoformat(timespec="seconds")
        _write_job_payload(
            job_id,
            {
                "ok": False,
                "job_id": job_id,
                "status": "failed",
                "created_at": now,
                "updated_at": failed_at,
                "result": None,
                "error": str(exc),
                "mode": normalized_request.get("mode"),
                "target_count": target_count,
            },
        )


def create_analysis_job(payload: dict[str, Any]) -> dict[str, Any]:
    _ensure_results_dir()
    prepared_payload = deepcopy(payload)
    normalized_request = _normalize_request_payload(prepared_payload)
    if normalized_request.get("mode") != "market" and normalized_request.get("targets"):
        from web.services import data_service

        ensured_targets = data_service.ensure_symbols_available(
            list(normalized_request.get("targets") or []),
            str(normalized_request.get("market") or "CN"),
        )
        prepared_payload["targets"] = ensured_targets
        prepared_payload["stocks"] = ensured_targets
        normalized_request = _normalize_request_payload(prepared_payload)

    if not normalized_request.get("targets"):
        raise ValueError("当前模式下没有可分析的股票，请先补充标的")
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    created_at = datetime.now().isoformat(timespec="seconds")
    initial_payload = {
        "ok": True,
        "job_id": job_id,
        "status": "queued",
        "created_at": created_at,
        "updated_at": created_at,
        "result": None,
        "error": None,
        "mode": normalized_request.get("mode"),
        "target_count": len(normalized_request.get("targets") or []),
    }
    _write_job_payload(job_id, initial_payload)
    worker = threading.Thread(
        target=_run_analysis_job,
        args=(job_id, prepared_payload),
        daemon=True,
        name=f"analysis-job-{job_id}",
    )
    worker.start()
    return initial_payload


def _reconcile_job_payload(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    status = str(payload.get("status", ""))
    if status not in {"queued", "running"}:
        result = payload.get("result")
        if isinstance(result, dict):
            payload = dict(payload)
            payload["result"] = _normalize_web_result(result)
        return payload

    analysis_id = _analysis_id_from_job_id(job_id)
    result_path = _result_file_for(analysis_id)
    if result_path.exists():
        result = _load_normalized_result(result_path, "web")
        if result is not None:
            _save_analysis_session(result)
            completed_payload = {
                "ok": True,
                "job_id": job_id,
                "status": "completed",
                "created_at": str(payload.get("created_at", datetime.now().isoformat(timespec="seconds"))),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "result": result,
                "error": None,
            }
            _write_job_payload(job_id, completed_payload)
            return completed_payload

    updated_at = _parse_iso_datetime(payload.get("updated_at")) or _parse_iso_datetime(payload.get("created_at"))
    if updated_at and datetime.now() - updated_at > _job_stale_timeout(payload):
        failed_payload = {
            "ok": False,
            "job_id": job_id,
            "status": "failed",
            "created_at": str(payload.get("created_at", datetime.now().isoformat(timespec="seconds"))),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "result": None,
            "error": str(payload.get("error") or "任务状态已过期，请重新发起分析"),
            "mode": payload.get("mode"),
            "target_count": payload.get("target_count"),
        }
        _write_job_payload(job_id, failed_payload)
        return failed_payload

    return payload


def get_analysis_job(job_id: str) -> dict[str, Any] | None:
    payload = _read_job_payload(job_id)
    if payload is None:
        return None
    return _reconcile_job_payload(job_id, payload)


def list_analysis_history(
    limit: int = 20,
    offset: int = 0,
    search: str | None = None,
    market: str | None = None,
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for row in _session_rows():
        try:
            raw = json.loads(row["result_json"])
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        result = _normalize_web_result(raw)
        request = result.get("request", {})
        item_market = str(request.get("market", "CN"))
        if market and item_market.upper() != market.upper():
            continue
        history_item = {
            "analysis_id": str(result.get("analysis_id", row["analysis_id"])),
            "created_at": str(result.get("created_at", row["created_at"] or datetime.now().isoformat(timespec="seconds"))),
            "source": str(result.get("source", "web")),
            "market": item_market,
            "mode": str(request.get("mode", "single")),
            "preset": str(request.get("preset", "quick_scan")),
            "stock_count": len(request.get("targets", request.get("stocks", []))),
            "stocks": request.get("targets", request.get("stocks", [])),
            "target_exposure": float(result.get("target_exposure", 0.0)),
            "style_bias": str(result.get("style_bias", "均衡")),
            "risk_level": str(result.get("risk", {}).get("risk_level", "unknown")),
            "candidate_symbols": result.get("candidate_symbols", []),
            "title": _title_for(request),
        }
        if search:
            search_lower = search.lower()
            searchable = f"{history_item['title']} {' '.join(history_item['stocks'])} {' '.join(history_item['candidate_symbols'])}".lower()
            if search_lower not in searchable:
                continue
        items.append(history_item)

    items.sort(key=lambda item: item["created_at"], reverse=True)
    total = len(items)
    return {"items": items[offset : offset + limit], "total": total}


def get_recent_jobs(limit: int = 8) -> list[dict[str, Any]]:
    _ensure_results_dir()
    jobs = []
    for path in sorted(JOB_DIR.glob("job_*.json"), reverse=True):
        raw = _load_json(path)
        if not isinstance(raw, dict):
            continue
        jobs.append(_reconcile_job_payload(path.stem.replace("job_", ""), raw))
    return jobs[:limit]


def get_analysis_result(analysis_id: str) -> dict[str, Any] | None:
    session_result = _load_session_result(analysis_id)
    if session_result is not None:
        return session_result

    for source, path in _iter_result_paths():
        if analysis_id in {path.stem, path.stem.replace("analysis_", "")}:
            result = _load_normalized_result(path, "web" if source == "web" else source)
            if result is not None and source == "web":
                _save_analysis_session(result)
            return result

        result = _load_normalized_result(path, "web" if source == "web" else source)
        if result and str(result.get("analysis_id")) == analysis_id:
            if source == "web":
                _save_analysis_session(result)
            return result
    return None


def delete_analysis_result(analysis_id: str) -> dict[str, Any]:
    _ensure_results_dir()
    deleted_count = 0

    conn = _connect_session_db()
    try:
        cursor = conn.execute("DELETE FROM analysis_sessions WHERE analysis_id = ?", (analysis_id,))
        deleted_count += cursor.rowcount
        conn.commit()
    finally:
        conn.close()

    result_path = _result_file_for(analysis_id)
    if result_path.exists():
        result_path.unlink()
        deleted_count += 1

    for path in JOB_DIR.glob(f"job_{analysis_id}_*.json"):
        path.unlink()
        deleted_count += 1

    for path in WEB_ANALYSIS_DIR.glob(f"analysis_request_{analysis_id}_*.json"):
        path.unlink()
        deleted_count += 1

    return {
        "ok": deleted_count > 0,
        "deleted_count": deleted_count,
        "message": "分析记录已删除" if deleted_count > 0 else "未找到对应分析记录",
    }


def clear_analysis_history() -> dict[str, Any]:
    _ensure_results_dir()
    deleted_count = 0

    conn = _connect_session_db()
    try:
        cursor = conn.execute("DELETE FROM analysis_sessions WHERE source = 'web'")
        deleted_count += cursor.rowcount
        conn.commit()
    finally:
        conn.close()

    for path in WEB_ANALYSIS_DIR.glob("analysis_*.json"):
        if WEB_RESULT_FILE_RE.match(path.name) or WEB_REQUEST_FILE_RE.match(path.name):
            path.unlink()
            deleted_count += 1

    for path in JOB_DIR.glob("job_*.json"):
        path.unlink()
        deleted_count += 1

    return {
        "ok": True,
        "deleted_count": deleted_count,
        "message": "已清空 Web 分析历史与任务缓存",
    }


def get_stock_analysis_mentions(ts_code: str, limit: int = 5) -> list[dict[str, Any]]:
    mentions: list[dict[str, Any]] = []
    normalized_code = ts_code.upper()

    for source, path in _iter_result_paths():
        result = _load_normalized_result(path, "web" if source == "web" else source)
        if result is None:
            continue

        request = result.get("request", {})
        candidate_symbols = set(result.get("candidate_symbols", []))
        stocks = set(request.get("targets", request.get("stocks", [])))
        trade_symbols = {
            item.get("symbol")
            for item in result.get("trade_recommendations", [])
            if isinstance(item, dict)
        }
        present = normalized_code in candidate_symbols or normalized_code in stocks or normalized_code in trade_symbols
        if not present:
            continue

        summary = "进入候选池" if normalized_code in candidate_symbols else "出现在研究样本中"
        mentions.append(
            {
                "analysis_id": str(result.get("analysis_id", path.stem)),
                "created_at": str(result.get("created_at", datetime.now().isoformat(timespec="seconds"))),
                "source": str(result.get("source", "web")),
                "title": _title_for(request),
                "candidate": normalized_code in candidate_symbols,
                "summary": summary,
            }
        )

    mentions.sort(key=lambda item: item["created_at"], reverse=True)
    return mentions[:limit]
