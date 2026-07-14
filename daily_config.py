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
    # 本地 myQuant 运行不主动调用 LLM；需要 LLM 判断时由 Codex 读取产物接管。
    # 下列模型仅保留为显式启用本地 review layer 时的兼容配置。
    "review_model_priority": [
        "deepseek-chat",
        "moonshot-v1-128k",
        "qwen3.6-plus",
    ],
    # Daily runner 默认把 master 角色固定到 Kimi，缺失时回退到 DeepSeek reasoning。
    "master_model": "moonshot-v1-128k",
    "master_fallback_model": "deepseek-reasoner",
    # Master Agent reasoning 强度—— 仅 deepseek-reasoner 支持
    "master_reasoning_effort": "high",

    # ── 决策引擎（统一 DAG + Bayesian Pipeline） ───────────────────────────────
    # 漏斗压缩后保留最大候选数（全市场 -> 候选）
    "funnel_max_candidates": 500,
    # Bayesian shortlist 入选数（候选 -> Master Discussion 精选）
    "bayesian_shortlist_size": 50,
    # CN 数据新鲜度模式: "stable"（T-1 容忍）/ "strict"（要求当日）
    "freshness_mode": "stable",

    # ── 分析参数 ────────────────────────────────────────────────────────────────
    "kline_backend": "hybrid",  # 全市场扫描建议用 heuristic；精细分析可用 hybrid
    "top_k": 20,                   # 最终精选股票数量
    "agent_timeout": 180.0,        # 单个 subagent 超时（秒）
    "master_timeout": 900.0,       # master agent 超时（秒）
    # 进入 review layer；本地 LLM 禁用时生成 Codex handoff packets
    "enable_agent_layer": True,

    # ── 数据下载 ────────────────────────────────────────────────────────────────
    "skip_download": False,        # False = staged batch 检查并补目标交易日缺口
    "years": 3,                    # 历史数据年数
    "workers": 4,                  # 并发下载线程数
    "maintenance_batch_size": 200,  # 每批最多处理的 symbol 数
    "maintenance_max_batches_per_run": 200,  # 每次自动化最多跑的批次数
    "maintenance_min_symbol_success_rate": 0.95,
    "maintenance_target_date": "auto",
    "maintenance_daily_window": True,

    # ── 定时调度 ────────────────────────────────────────────────────────────────
    "schedule_time": "17:30",      # 每日触发时间（A 股 15:00 收盘后）

    # ── 输出配置 ────────────────────────────────────────────────────────────────
    "report_dir": "reports/v14/daily",  # v13 reports/daily 已冻结为退役证据
}
