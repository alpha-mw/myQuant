"""
myQuant 每日分析配置文件 — 直接编辑此文件来调整分析参数。
"""

DAILY_CONFIG = {
    # ── 市场与资金 ──────────────────────────────────────────────────────────────
    "market": "CN",
    "universe": "full_a",         # CN 默认全 A universe；legacy basket 仅作兼容
    "risk_level": "中等",          # 可选: "保守" / "中等" / "积极"
    "total_capital": 1_000_000,   # 总资金（元）

    # ── LLM 模型配置 ────────────────────────────────────────────────────────────
    # Subagent（分支分析）—— 例: "moonshot-v1-128k" / "deepseek-reasoner"
    "agent_model": "deepseek-reasoner",
    "agent_fallback_model": "qwen-3.6",
    # Master Agent（决策综合）—— 例: "moonshot-v1-128k" / "deepseek-chat"
    "master_model": "moonshot-v1-128k",
    "master_fallback_model": "deepseek-chat",
    # Master Agent reasoning 强度—— 仅 deepseek-reasoner 支持
    "master_reasoning_effort": "",

    # ── 决策引擎（Bayesian Pipeline） ───────────────────────────────────────────
    # "bayesian" = 7 层 Bayesian 架构（默认）；"legacy" = 原 3 层流水线
    "pipeline_mode": "bayesian",
    # 漏斗压缩后保留最大候选数（全市场 -> 候选）
    "funnel_max_candidates": 400,
    # Bayesian shortlist 入选数（候选 -> Master Discussion 精选）
    "bayesian_shortlist_size": 20,
    # CN 数据新鲜度模式: "stable"（T-1 容忍）/ "strict"（要求当日）
    "freshness_mode": "stable",

    # ── 分析参数 ────────────────────────────────────────────────────────────────
    "kline_backend": "heuristic",  # 全市场扫描建议用 heuristic；精细分析可用 hybrid
    "top_k": 20,                   # 最终精选股票数量
    "agent_timeout": 20.0,         # 单个 subagent 超时（秒）
    "master_timeout": 45.0,        # master agent 超时（秒）
    "enable_agent_layer": True,    # 是否启用 LLM review layer

    # ── 数据下载 ────────────────────────────────────────────────────────────────
    "skip_download": False,        # False = 每次先检查并更新数据
    "years": 3,                    # 历史数据年数
    "workers": 4,                  # 并发下载线程数

    # ── 定时调度 ────────────────────────────────────────────────────────────────
    "schedule_time": "17:30",      # 每日触发时间（A 股 15:00 收盘后）

    # ── 输出配置 ────────────────────────────────────────────────────────────────
    "report_dir": "reports/daily",  # 报告保存目录
    "history_lookback": 5,          # 加载最近 N 次分析记录作为上下文

    # ── 后端 ────────────────────────────────────────────────────────────────────
    "backend_host": "127.0.0.1",
    "backend_port": 8000,
}
