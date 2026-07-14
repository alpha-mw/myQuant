"""
Quant-Investor V7.0 配置管理模块
安全地加载环境变量和配置
"""

import os

from quant_investor.credential_utils import get_secret
from quant_investor.env_loading import load_env_file

MAINLINE_ENV_DEFAULTS: dict[str, str] = {
    "TUSHARE_TOKEN": "",
    "TUSHARE_URL": "http://lianghua.nanyangqiankun.top",
    "TUSHARE_RATE_LIMIT_PER_MIN": "500",
    "MYQUANT_TUSHARE_AUTO_CLEAN": "1",
    "MYQUANT_TUSHARE_FACTOR_READINESS": "1",
    "MYQUANT_TUSHARE_CLEANING_REPORT_DIR": "data/cleaning_reports/tushare",
    "MYQUANT_TUSHARE_RAW_BACKUP_DIR": "data/raw_backups/tushare",
    "MYQUANT_TUSHARE_QUARANTINE_DIR": "data/quarantine/tushare",
    "MYQUANT_TUSHARE_FACTOR_READINESS_DIR": "data/factor_readiness/tushare",
    "MYQUANT_TUSHARE_STORAGE_AUDIT": "1",
    "MYQUANT_TUSHARE_PARQUET_SHADOW_WRITE": "0",
    "MYQUANT_TUSHARE_PARQUET_CANONICAL": "0",
    "MYQUANT_TUSHARE_PARQUET_DIR": "data/cn_market_parquet",
    "MYQUANT_TUSHARE_PARQUET_COMPRESSION": "snappy",
    "MYQUANT_TUSHARE_DELETE_REDUNDANT_CSV": "0",
    "MYQUANT_LLM_HANDOFF": "codex",
    "MYQUANT_DISABLE_LOCAL_LLM": "true",
    "QUANT_PRODUCTION_KILL_SWITCH": "true",
    "QUANT_PRODUCTION_ACTIVATION_RECEIPT": "",
    "QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256": "",
    "THEME_SCANNER_ENABLED": "1",
    "THEME_MIN_MEMBER_COUNT": "5",
    "THEME_TOP_N": "20",
    "THEME_METADATA_SYMBOL_LIMIT": "300",
    "THEME_SMOOTHING_WINDOW": "10",
    "THEME_SMOOTHING_MIN_OBSERVATIONS": "5",
    "THEME_POLICY_CATALYST_ENABLED": "0",
    "THEME_POLICY_CATALYST_WEIGHT": "0.16",
    "THEME_POLICY_LOOKBACK_DAYS": "30",
    "THEME_POLICY_EVENT_PATH": "data/theme_policy_events.jsonl",
    "THEME_SNAPSHOT_ENABLED": "1",
    "THEME_SNAPSHOT_DIR": "results/theme_snapshots",
    "THEME_SNAPSHOT_SAVE_DISABLED": "0",
    "THEME_HOLDING_GUARD_ENABLED": "0",
    "THEME_HOLDING_GUARD_TIGHTEN_RATIO": "0.5",
    "THEME_CROWDING_ENABLED": "0",
    "THEME_CROWDING_MIN_UNIVERSE": "30",
    "THEME_CONCEPT_MEMBERSHIP_ENABLED": "0",
    "THEME_CONCEPT_MEMBERSHIP_PATH": "data/theme_membership.jsonl",
    "THEME_CONCEPT_MEMBERSHIP_REQUIRED": "0",
    "THEME_CONCEPT_PRIMARY_MARGIN": "0.05",
    "THEME_MEMBERSHIP_V2_ENABLED": "1",
    "THEME_MEMBERSHIP_V2_PATH": "private/theme_knowledge/theme_membership.v2.jsonl",
    "THEME_MEMBERSHIP_V2_REQUIRED": "0",
    "THEME_MEMBERSHIP_V2_EXPECTED_SHA256": "",
    "THEME_PROTOCOL_V2_ENABLED": "1",
    "THEME_TAXONOMY_V2_PATH": "quant_investor/themes/data/theme_taxonomy.v2.json",
    "THEME_EVIDENCE_EVENT_V1_PATH": "private/theme_knowledge/theme_evidence_events.jsonl",
    "THEME_PEVC_CANONICAL_PATH": "private/theme_knowledge/pevc_theses.jsonl",
    "THEME_V2_FORMAL_ENABLED": "0",
    "THEME_V2_FORMAL_KILL_SWITCH": "1",
    "THEME_FORMAL_RECONCILIATION_PERSIST_ENABLED": "0",
    "THEME_FORMAL_RECONCILIATION_DIR": "private/theme_reconciliation",
    "THEME_V2_JOINT_MANIFEST_PATH": "",
    "THEME_V2_EXPECTED_JOINT_MANIFEST_SHA256": "",
    "THEME_STAT_CLUSTER_ENABLED": "0",
    "THEME_FUNNEL_BOOST_ENABLED": "1",
    "THEME_FUNNEL_BOOST_SCORE_SOURCE": "raw",
    "THEME_SYMBOL_BOOST_CAP": "0.10",
    "RISK_GUARD_SINGLE_NAME_WEIGHT_CAP": "0.15",
    "THEME_RISK_GUARD_ENABLED": "1",
    "THEME_RISK_OVEREXTENDED_GROSS_CAP": "0.60",
    "THEME_RISK_OVEREXTENDED_MAX_WEIGHT": "0.10",
    "THEME_RISK_DISTRIBUTION_GROSS_CAP": "0.45",
    "THEME_RISK_DISTRIBUTION_MAX_WEIGHT": "0.08",
    "THEME_RISK_FAKE_BREAKOUT_MAX_WEIGHT": "0.10",
    "THEME_PORTFOLIO_CAP_ENABLED": "1",
    "THEME_PORTFOLIO_MAX_THEME_EXPOSURE": "0.35",
    "THEME_PORTFOLIO_OVEREXTENDED_MAX_THEME_EXPOSURE": "0.25",
    "THEME_PORTFOLIO_DISTRIBUTION_MAX_THEME_EXPOSURE": "0.15",
    "THEME_SHADOW_MODE_ENABLED": "0",
    "THEME_SHADOW_EXECUTION_TARGET": "baseline",
    "THEME_SHADOW_FUNNEL_BOOST_ENABLED": "1",
    "THEME_SHADOW_RISK_GUARD_ENABLED": "1",
    "THEME_SHADOW_PORTFOLIO_CAP_ENABLED": "1",
    "THEME_SHADOW_ARTIFACT_ENABLED": "1",
    "THEME_SHADOW_ARTIFACT_DIR": "results/theme_shadow",
    "THEME_SHADOW_MAX_ROWS": "50",
    "THEME_GOVERNANCE_ENABLED": "0",
    "THEME_GOVERNANCE_REGISTRY_PATH": "",
    "THEME_GOVERNANCE_ARTIFACT_ENABLED": "0",
    "THEME_GOVERNANCE_OUTPUT_DIR": "results/theme_governance",
    "THEME_POOL_ENABLED": "1",
    "THEME_POOL_REQUIRED": "1",
    "THEME_POOL_USE_MARKOV_POLICY": "1",
    "THEME_POOL_SCORE_SOURCE": "smoothed",
    "THEME_POOL_FALLBACK_TO_RAW_SCORE": "1",
    "THEME_POOL_MIN_THEME_SCORE": "0.58",
    "THEME_POOL_MIN_SYMBOL_SCORE": "0.55",
    "THEME_POOL_TOP_THEMES": "8",
    "THEME_POOL_MAX_SYMBOLS_PER_THEME": "30",
    "THEME_POOL_RESIDUAL_RATIO": "0.25",
    "THEME_POOL_MIN_RESIDUAL_SYMBOLS": "20",
    "THEME_POOL_MIN_ADMITTED_THEMES": "0",
    "THEME_POOL_ALLOW_UNTHEMED_RESIDUAL": "0",
    "THEME_POOL_INCLUDE_RISK_WATCH": "1",
    "THEME_POOL_RISK_WATCH_MAX_RATIO": "0.20",
    "THEME_POOL_SYMBOL_GATE_MODE": "classify",
    "MARKOV_REGIME_ENABLED": "1",
    "MARKOV_REGIME_EXECUTION_TARGET": "production",
    "MARKOV_REGIME_HISTORY_PATH": "results/regime/markov_regime_history.jsonl",
    "MARKOV_REGIME_PERSIST_ENABLED": "1",
    "MARKOV_REGIME_MIN_MARKET_SAMPLE": "30",
    "MARKOV_REGIME_MAX_REFERENCE_SYMBOLS": "300",
    "MARKOV_REGIME_REFERENCE_UNIVERSE_CN": "full_a",
    "MARKOV_REGIME_REFERENCE_UNIVERSE_US": "full_us",
    "PIT_UNIVERSE_ENABLED": "0",
    "PIT_UNIVERSE_REQUIRED": "0",
    "PIT_UNIVERSE_SOURCE_ROOT": "data/parquet/cn/reference",
    "PIT_UNIVERSE_BACKFILL_ENABLED": "0",
    "KIMI_API_KEY": "",
    "DEEPSEEK_API_KEY": "",
    "DASHSCOPE_API_KEY": "",
    "FRED_API_KEY": "",
    "API_HOST": "127.0.0.1",
    "API_PORT": "8000",
    "WORKSPACE_AUTH_TOKEN": "",
    "CORS_ORIGINS": "http://localhost:5173,http://localhost:3000",
    "DB_PATH": "data/stock_database.db",
    "APP_DB_PATH": "data/app.db",
    "LOG_LEVEL": "INFO",
    "RESULTS_DIR": "results",
    "WEB_ANALYSIS_DIR": "results/web_analysis",
    "REDIS_HOST": "localhost",
    "REDIS_PORT": "6379",
    "REDIS_DB": "0",
    "INITIAL_CASH": "1000000",
    "COMMISSION_RATE": "0.0003",
    "STAMP_DUTY_RATE": "0.001",
    "SLIPPAGE": "0.001",
    "EXECUTION_COST_MODEL_ENABLED": "0",
}

MAINLINE_ENV_KEYS: tuple[str, ...] = tuple(MAINLINE_ENV_DEFAULTS)

load_env_file()

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return bool(default)
    return str(raw_value or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_str(
    name: str,
    default: str,
) -> str:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return str(default or "").strip() or default
    return str(raw_value or "").strip() or default


def _env_markov_execution_target(
    name: str,
    default: str = "production",
) -> str:
    raw_value = _env_str(name, default)
    normalized = str(raw_value or "").strip().lower()
    if normalized == "disabled":
        return "disabled"
    return "production"


def _env_int_list(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return tuple(int(item) for item in default)
    values: list[int] = []
    for item in raw.split(","):
        text = str(item).strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except (TypeError, ValueError):
            continue
    if not values:
        return tuple(int(item) for item in default)
    return tuple(values)


class Config:
    """配置类"""

    MAINLINE_ENV_DEFAULTS: dict[str, str] = MAINLINE_ENV_DEFAULTS
    MAINLINE_ENV_KEYS: tuple[str, ...] = MAINLINE_ENV_KEYS
    QUANT_PRODUCTION_KILL_SWITCH: str = _env_str(
        'QUANT_PRODUCTION_KILL_SWITCH',
        MAINLINE_ENV_DEFAULTS['QUANT_PRODUCTION_KILL_SWITCH'],
    )
    QUANT_PRODUCTION_ACTIVATION_RECEIPT: str = _env_str(
        'QUANT_PRODUCTION_ACTIVATION_RECEIPT',
        MAINLINE_ENV_DEFAULTS['QUANT_PRODUCTION_ACTIVATION_RECEIPT'],
    )
    QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256: str = _env_str(
        'QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256',
        MAINLINE_ENV_DEFAULTS['QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256'],
    )
    
    # Tushare配置
    TUSHARE_TOKEN: str = get_secret('TUSHARE_TOKEN')
    TUSHARE_URL: str = _env_str(
        'TUSHARE_URL',
        MAINLINE_ENV_DEFAULTS['TUSHARE_URL'],
    )
    TUSHARE_RATE_LIMIT_PER_MIN: int = _env_int(
        'TUSHARE_RATE_LIMIT_PER_MIN',
        int(MAINLINE_ENV_DEFAULTS['TUSHARE_RATE_LIMIT_PER_MIN']),
    )
    TUSHARE_AUTO_CLEAN: bool = _env_bool(
        'MYQUANT_TUSHARE_AUTO_CLEAN',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_AUTO_CLEAN'] == '1',
    )
    TUSHARE_FACTOR_READINESS: bool = _env_bool(
        'MYQUANT_TUSHARE_FACTOR_READINESS',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_FACTOR_READINESS'] == '1',
    )
    TUSHARE_CLEANING_REPORT_DIR: str = _env_str(
        'MYQUANT_TUSHARE_CLEANING_REPORT_DIR',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_CLEANING_REPORT_DIR'],
    )
    TUSHARE_RAW_BACKUP_DIR: str = _env_str(
        'MYQUANT_TUSHARE_RAW_BACKUP_DIR',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_RAW_BACKUP_DIR'],
    )
    TUSHARE_QUARANTINE_DIR: str = _env_str(
        'MYQUANT_TUSHARE_QUARANTINE_DIR',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_QUARANTINE_DIR'],
    )
    TUSHARE_FACTOR_READINESS_DIR: str = _env_str(
        'MYQUANT_TUSHARE_FACTOR_READINESS_DIR',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_FACTOR_READINESS_DIR'],
    )
    TUSHARE_STORAGE_AUDIT: bool = _env_bool(
        'MYQUANT_TUSHARE_STORAGE_AUDIT',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_STORAGE_AUDIT'] == '1',
    )
    TUSHARE_PARQUET_SHADOW_WRITE: bool = _env_bool(
        'MYQUANT_TUSHARE_PARQUET_SHADOW_WRITE',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_PARQUET_SHADOW_WRITE'] == '1',
    )
    TUSHARE_PARQUET_CANONICAL: bool = _env_bool(
        'MYQUANT_TUSHARE_PARQUET_CANONICAL',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_PARQUET_CANONICAL'] == '1',
    )
    TUSHARE_PARQUET_DIR: str = _env_str(
        'MYQUANT_TUSHARE_PARQUET_DIR',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_PARQUET_DIR'],
    )
    TUSHARE_PARQUET_COMPRESSION: str = _env_str(
        'MYQUANT_TUSHARE_PARQUET_COMPRESSION',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_PARQUET_COMPRESSION'],
    )
    TUSHARE_DELETE_REDUNDANT_CSV: bool = _env_bool(
        'MYQUANT_TUSHARE_DELETE_REDUNDANT_CSV',
        MAINLINE_ENV_DEFAULTS['MYQUANT_TUSHARE_DELETE_REDUNDANT_CSV'] == '1',
    )

    # LLM / 外部 API 凭据
    KIMI_API_KEY: str = get_secret('KIMI_API_KEY')
    DEEPSEEK_API_KEY: str = get_secret('DEEPSEEK_API_KEY')
    DASHSCOPE_API_KEY: str = get_secret('DASHSCOPE_API_KEY')
    FRED_API_KEY: str = get_secret('FRED_API_KEY')
    FINNHUB_API_KEY: str = get_secret('FINNHUB_API_KEY')
    
    # 数据库配置
    DB_PATH: str = os.environ.get('DB_PATH', 'data/stock_database.db')
    DATA_DIR: str = os.environ.get('DATA_DIR', 'data')
    CN_MARKET_DATA_DIR: str = os.environ.get('CN_MARKET_DATA_DIR', 'data/cn_market_full')
    CN_FRESHNESS_MODE: str = os.environ.get('CN_FRESHNESS_MODE', 'stable')
    CN_FRESHNESS_COVERAGE_THRESHOLD: float = _env_float('CN_FRESHNESS_COVERAGE_THRESHOLD', 0.95)
    CN_STRICT_EARLY_STOP_SAMPLE_SIZE: int = _env_int('CN_STRICT_EARLY_STOP_SAMPLE_SIZE', 10)
    CN_STRICT_EARLY_STOP_STALE_RATIO: float = _env_float('CN_STRICT_EARLY_STOP_STALE_RATIO', 0.80)
    EXECUTION_COST_MODEL_ENABLED: bool = _env_bool('EXECUTION_COST_MODEL_ENABLED', False)

    # Pipeline mode: "bayesian" (new 7-layer) or "legacy" (original 3-layer)
    PIPELINE_MODE: str = os.environ.get('PIPELINE_MODE', 'bayesian')
    DECISION_ENGINE: str = os.environ.get('DECISION_ENGINE', 'bayesian')
    BAYESIAN_SHORTLIST_SIZE: int = _env_int('BAYESIAN_SHORTLIST_SIZE', 50)
    FUNNEL_PROFILE: str = str(os.environ.get('FUNNEL_PROFILE', 'momentum_leader') or 'momentum_leader').strip().lower()
    FUNNEL_MAX_CANDIDATES: int = _env_int('FUNNEL_MAX_CANDIDATES', 500)
    FUNNEL_TREND_WINDOWS: tuple[int, ...] = _env_int_list('FUNNEL_TREND_WINDOWS', (20, 60, 120))
    FUNNEL_VOLUME_SPIKE_THRESHOLD: float = _env_float('FUNNEL_VOLUME_SPIKE_THRESHOLD', 1.35)
    FUNNEL_BREAKOUT_DISTANCE_PCT: float = _env_float('FUNNEL_BREAKOUT_DISTANCE_PCT', 0.06)
    FUNNEL_SECTOR_BUCKET_LIMIT: int = _env_int('FUNNEL_SECTOR_BUCKET_LIMIT', 2)
    THEME_SCANNER_ENABLED: bool = _env_bool(
        'THEME_SCANNER_ENABLED',
        MAINLINE_ENV_DEFAULTS['THEME_SCANNER_ENABLED'] == '1',
    )
    THEME_MIN_MEMBER_COUNT: int = _env_int('THEME_MIN_MEMBER_COUNT', 5)
    THEME_TOP_N: int = _env_int('THEME_TOP_N', 20)
    THEME_METADATA_SYMBOL_LIMIT: int = _env_int('THEME_METADATA_SYMBOL_LIMIT', 300)
    THEME_SMOOTHING_WINDOW: int = _env_int('THEME_SMOOTHING_WINDOW', 10)
    THEME_SMOOTHING_MIN_OBSERVATIONS: int = _env_int('THEME_SMOOTHING_MIN_OBSERVATIONS', 5)
    THEME_POLICY_CATALYST_ENABLED: bool = _env_bool('THEME_POLICY_CATALYST_ENABLED', False)
    THEME_POLICY_CATALYST_WEIGHT: float = _env_float('THEME_POLICY_CATALYST_WEIGHT', 0.16)
    THEME_POLICY_LOOKBACK_DAYS: int = _env_int('THEME_POLICY_LOOKBACK_DAYS', 30)
    THEME_POLICY_EVENT_PATH: str = _env_str('THEME_POLICY_EVENT_PATH', 'data/theme_policy_events.jsonl')
    THEME_SNAPSHOT_ENABLED: bool = _env_bool(
        'THEME_SNAPSHOT_ENABLED',
        MAINLINE_ENV_DEFAULTS['THEME_SNAPSHOT_ENABLED'] == '1',
    )
    THEME_SNAPSHOT_DIR: str = _env_str('THEME_SNAPSHOT_DIR', 'results/theme_snapshots')
    THEME_SNAPSHOT_SAVE_DISABLED: bool = _env_bool('THEME_SNAPSHOT_SAVE_DISABLED', False)
    THEME_HOLDING_GUARD_ENABLED: bool = _env_bool('THEME_HOLDING_GUARD_ENABLED', False)
    THEME_HOLDING_GUARD_TIGHTEN_RATIO: float = _env_float('THEME_HOLDING_GUARD_TIGHTEN_RATIO', 0.5)
    THEME_CROWDING_ENABLED: bool = _env_bool('THEME_CROWDING_ENABLED', False)
    THEME_CROWDING_MIN_UNIVERSE: int = _env_int('THEME_CROWDING_MIN_UNIVERSE', 30)
    THEME_CONCEPT_MEMBERSHIP_ENABLED: bool = _env_bool('THEME_CONCEPT_MEMBERSHIP_ENABLED', False)
    THEME_CONCEPT_MEMBERSHIP_PATH: str = _env_str('THEME_CONCEPT_MEMBERSHIP_PATH', 'data/theme_membership.jsonl')
    THEME_CONCEPT_MEMBERSHIP_REQUIRED: bool = _env_bool('THEME_CONCEPT_MEMBERSHIP_REQUIRED', False)
    THEME_CONCEPT_PRIMARY_MARGIN: float = _env_float('THEME_CONCEPT_PRIMARY_MARGIN', 0.05)
    THEME_MEMBERSHIP_V2_ENABLED: bool = _env_bool('THEME_MEMBERSHIP_V2_ENABLED', True)
    THEME_MEMBERSHIP_V2_PATH: str = _env_str(
        'THEME_MEMBERSHIP_V2_PATH',
        'private/theme_knowledge/theme_membership.v2.jsonl',
    )
    THEME_MEMBERSHIP_V2_REQUIRED: bool = _env_bool('THEME_MEMBERSHIP_V2_REQUIRED', False)
    THEME_MEMBERSHIP_V2_EXPECTED_SHA256: str = _env_str(
        'THEME_MEMBERSHIP_V2_EXPECTED_SHA256',
        '',
    )
    THEME_PROTOCOL_V2_ENABLED: bool = _env_bool('THEME_PROTOCOL_V2_ENABLED', True)
    THEME_TAXONOMY_V2_PATH: str = _env_str(
        'THEME_TAXONOMY_V2_PATH',
        'quant_investor/themes/data/theme_taxonomy.v2.json',
    )
    THEME_EVIDENCE_EVENT_V1_PATH: str = _env_str(
        'THEME_EVIDENCE_EVENT_V1_PATH',
        'private/theme_knowledge/theme_evidence_events.jsonl',
    )
    THEME_PEVC_CANONICAL_PATH: str = _env_str(
        'THEME_PEVC_CANONICAL_PATH',
        'private/theme_knowledge/pevc_theses.jsonl',
    )
    THEME_V2_FORMAL_ENABLED: bool = _env_bool('THEME_V2_FORMAL_ENABLED', False)
    THEME_V2_FORMAL_KILL_SWITCH: bool = _env_bool('THEME_V2_FORMAL_KILL_SWITCH', True)
    THEME_FORMAL_RECONCILIATION_PERSIST_ENABLED: bool = _env_bool(
        'THEME_FORMAL_RECONCILIATION_PERSIST_ENABLED', False
    )
    THEME_FORMAL_RECONCILIATION_DIR: str = _env_str(
        'THEME_FORMAL_RECONCILIATION_DIR', 'private/theme_reconciliation'
    )
    THEME_V2_JOINT_MANIFEST_PATH: str = _env_str(
        'THEME_V2_JOINT_MANIFEST_PATH', ''
    )
    THEME_V2_EXPECTED_JOINT_MANIFEST_SHA256: str = _env_str(
        'THEME_V2_EXPECTED_JOINT_MANIFEST_SHA256', ''
    )
    THEME_STAT_CLUSTER_ENABLED: bool = _env_bool('THEME_STAT_CLUSTER_ENABLED', False)
    THEME_FUNNEL_BOOST_ENABLED: bool = _env_bool(
        'THEME_FUNNEL_BOOST_ENABLED',
        MAINLINE_ENV_DEFAULTS['THEME_FUNNEL_BOOST_ENABLED'] == '1',
    )
    THEME_FUNNEL_BOOST_SCORE_SOURCE: str = _env_str('THEME_FUNNEL_BOOST_SCORE_SOURCE', 'raw')
    THEME_SYMBOL_BOOST_CAP: float = _env_float('THEME_SYMBOL_BOOST_CAP', 0.10)
    THEME_RISK_GUARD_ENABLED: bool = _env_bool(
        'THEME_RISK_GUARD_ENABLED',
        MAINLINE_ENV_DEFAULTS['THEME_RISK_GUARD_ENABLED'] == '1',
    )
    THEME_RISK_OVEREXTENDED_GROSS_CAP: float = _env_float('THEME_RISK_OVEREXTENDED_GROSS_CAP', 0.60)
    THEME_RISK_OVEREXTENDED_MAX_WEIGHT: float = _env_float('THEME_RISK_OVEREXTENDED_MAX_WEIGHT', 0.10)
    THEME_RISK_DISTRIBUTION_GROSS_CAP: float = _env_float('THEME_RISK_DISTRIBUTION_GROSS_CAP', 0.45)
    THEME_RISK_DISTRIBUTION_MAX_WEIGHT: float = _env_float('THEME_RISK_DISTRIBUTION_MAX_WEIGHT', 0.08)
    THEME_RISK_FAKE_BREAKOUT_MAX_WEIGHT: float = _env_float('THEME_RISK_FAKE_BREAKOUT_MAX_WEIGHT', 0.10)
    THEME_PORTFOLIO_CAP_ENABLED: bool = _env_bool(
        'THEME_PORTFOLIO_CAP_ENABLED',
        MAINLINE_ENV_DEFAULTS['THEME_PORTFOLIO_CAP_ENABLED'] == '1',
    )
    THEME_PORTFOLIO_MAX_THEME_EXPOSURE: float = _env_float('THEME_PORTFOLIO_MAX_THEME_EXPOSURE', 0.35)
    THEME_PORTFOLIO_OVEREXTENDED_MAX_THEME_EXPOSURE: float = _env_float('THEME_PORTFOLIO_OVEREXTENDED_MAX_THEME_EXPOSURE', 0.25)
    THEME_PORTFOLIO_DISTRIBUTION_MAX_THEME_EXPOSURE: float = _env_float('THEME_PORTFOLIO_DISTRIBUTION_MAX_THEME_EXPOSURE', 0.15)
    THEME_SHADOW_MODE_ENABLED: bool = _env_bool('THEME_SHADOW_MODE_ENABLED', False)
    THEME_SHADOW_EXECUTION_TARGET: str = _env_str('THEME_SHADOW_EXECUTION_TARGET', 'baseline')
    THEME_SHADOW_FUNNEL_BOOST_ENABLED: bool = _env_bool('THEME_SHADOW_FUNNEL_BOOST_ENABLED', True)
    THEME_SHADOW_RISK_GUARD_ENABLED: bool = _env_bool('THEME_SHADOW_RISK_GUARD_ENABLED', True)
    THEME_SHADOW_PORTFOLIO_CAP_ENABLED: bool = _env_bool('THEME_SHADOW_PORTFOLIO_CAP_ENABLED', True)
    THEME_SHADOW_ARTIFACT_ENABLED: bool = _env_bool('THEME_SHADOW_ARTIFACT_ENABLED', True)
    THEME_SHADOW_ARTIFACT_DIR: str = _env_str('THEME_SHADOW_ARTIFACT_DIR', 'results/theme_shadow')
    THEME_SHADOW_MAX_ROWS: int = _env_int('THEME_SHADOW_MAX_ROWS', 50)
    THEME_GOVERNANCE_ENABLED: bool = _env_bool('THEME_GOVERNANCE_ENABLED', False)
    THEME_GOVERNANCE_REGISTRY_PATH: str = _env_str('THEME_GOVERNANCE_REGISTRY_PATH', '')
    THEME_GOVERNANCE_ARTIFACT_ENABLED: bool = _env_bool('THEME_GOVERNANCE_ARTIFACT_ENABLED', False)
    THEME_GOVERNANCE_OUTPUT_DIR: str = _env_str('THEME_GOVERNANCE_OUTPUT_DIR', 'results/theme_governance')
    THEME_POOL_ENABLED: bool = _env_bool(
        'THEME_POOL_ENABLED',
        MAINLINE_ENV_DEFAULTS['THEME_POOL_ENABLED'] == '1',
    )
    THEME_POOL_REQUIRED: bool = _env_bool(
        'THEME_POOL_REQUIRED',
        MAINLINE_ENV_DEFAULTS['THEME_POOL_REQUIRED'] == '1',
    )
    THEME_POOL_USE_MARKOV_POLICY: bool = _env_bool(
        'THEME_POOL_USE_MARKOV_POLICY',
        MAINLINE_ENV_DEFAULTS['THEME_POOL_USE_MARKOV_POLICY'] == '1',
    )
    THEME_POOL_SCORE_SOURCE: str = _env_str(
        'THEME_POOL_SCORE_SOURCE',
        MAINLINE_ENV_DEFAULTS['THEME_POOL_SCORE_SOURCE'],
    )
    THEME_POOL_FALLBACK_TO_RAW_SCORE: bool = _env_bool(
        'THEME_POOL_FALLBACK_TO_RAW_SCORE',
        MAINLINE_ENV_DEFAULTS['THEME_POOL_FALLBACK_TO_RAW_SCORE'] == '1',
    )
    THEME_POOL_MIN_THEME_SCORE: float = _env_float(
        'THEME_POOL_MIN_THEME_SCORE',
        float(MAINLINE_ENV_DEFAULTS['THEME_POOL_MIN_THEME_SCORE']),
    )
    THEME_POOL_MIN_SYMBOL_SCORE: float = _env_float(
        'THEME_POOL_MIN_SYMBOL_SCORE',
        float(MAINLINE_ENV_DEFAULTS['THEME_POOL_MIN_SYMBOL_SCORE']),
    )
    THEME_POOL_TOP_THEMES: int = _env_int(
        'THEME_POOL_TOP_THEMES',
        int(MAINLINE_ENV_DEFAULTS['THEME_POOL_TOP_THEMES']),
    )
    THEME_POOL_MAX_SYMBOLS_PER_THEME: int = _env_int(
        'THEME_POOL_MAX_SYMBOLS_PER_THEME',
        int(MAINLINE_ENV_DEFAULTS['THEME_POOL_MAX_SYMBOLS_PER_THEME']),
    )
    THEME_POOL_RESIDUAL_RATIO: float = _env_float(
        'THEME_POOL_RESIDUAL_RATIO',
        float(MAINLINE_ENV_DEFAULTS['THEME_POOL_RESIDUAL_RATIO']),
    )
    THEME_POOL_MIN_RESIDUAL_SYMBOLS: int = _env_int(
        'THEME_POOL_MIN_RESIDUAL_SYMBOLS',
        int(MAINLINE_ENV_DEFAULTS['THEME_POOL_MIN_RESIDUAL_SYMBOLS']),
    )
    THEME_POOL_MIN_ADMITTED_THEMES: int = _env_int(
        'THEME_POOL_MIN_ADMITTED_THEMES',
        int(MAINLINE_ENV_DEFAULTS['THEME_POOL_MIN_ADMITTED_THEMES']),
    )
    THEME_POOL_ALLOW_UNTHEMED_RESIDUAL: bool = _env_bool(
        'THEME_POOL_ALLOW_UNTHEMED_RESIDUAL',
        MAINLINE_ENV_DEFAULTS['THEME_POOL_ALLOW_UNTHEMED_RESIDUAL'] == '1',
    )
    THEME_POOL_INCLUDE_RISK_WATCH: bool = _env_bool(
        'THEME_POOL_INCLUDE_RISK_WATCH',
        MAINLINE_ENV_DEFAULTS['THEME_POOL_INCLUDE_RISK_WATCH'] == '1',
    )
    THEME_POOL_RISK_WATCH_MAX_RATIO: float = _env_float(
        'THEME_POOL_RISK_WATCH_MAX_RATIO',
        float(MAINLINE_ENV_DEFAULTS['THEME_POOL_RISK_WATCH_MAX_RATIO']),
    )
    THEME_POOL_SYMBOL_GATE_MODE: str = _env_str(
        'THEME_POOL_SYMBOL_GATE_MODE',
        MAINLINE_ENV_DEFAULTS['THEME_POOL_SYMBOL_GATE_MODE'],
    )
    MARKOV_REGIME_ENABLED: bool = _env_bool('MARKOV_REGIME_ENABLED', True)
    MARKOV_REGIME_EXECUTION_TARGET: str = _env_markov_execution_target(
        'MARKOV_REGIME_EXECUTION_TARGET',
        'production',
    )
    MARKOV_REGIME_HISTORY_PATH: str = _env_str(
        'MARKOV_REGIME_HISTORY_PATH',
        'results/regime/markov_regime_history.jsonl',
    )
    MARKOV_REGIME_PERSIST_ENABLED: bool = _env_bool('MARKOV_REGIME_PERSIST_ENABLED', True)
    MARKOV_REGIME_MIN_MARKET_SAMPLE: int = _env_int('MARKOV_REGIME_MIN_MARKET_SAMPLE', 30)
    MARKOV_REGIME_MAX_REFERENCE_SYMBOLS: int = _env_int('MARKOV_REGIME_MAX_REFERENCE_SYMBOLS', 300)
    MARKOV_REGIME_REFERENCE_UNIVERSE_CN: str = _env_str('MARKOV_REGIME_REFERENCE_UNIVERSE_CN', 'full_a')
    MARKOV_REGIME_REFERENCE_UNIVERSE_US: str = _env_str('MARKOV_REGIME_REFERENCE_UNIVERSE_US', 'full_us')
    PIT_UNIVERSE_ENABLED: bool = _env_bool('PIT_UNIVERSE_ENABLED', False)
    PIT_UNIVERSE_REQUIRED: bool = _env_bool('PIT_UNIVERSE_REQUIRED', False)
    PIT_UNIVERSE_SOURCE_ROOT: str = _env_str('PIT_UNIVERSE_SOURCE_ROOT', 'data/parquet/cn/reference')
    PIT_UNIVERSE_BACKFILL_ENABLED: bool = _env_bool('PIT_UNIVERSE_BACKFILL_ENABLED', False)
    DEFAULT_AGENT_TIMEOUT_SECONDS: float = _env_float('AGENT_TIMEOUT_SECONDS', 180.0)
    DEFAULT_MASTER_TIMEOUT_SECONDS: float = _env_float('MASTER_TIMEOUT_SECONDS', 900.0)
    DEFAULT_AGENT_TOTAL_TIMEOUT_SECONDS: float = _env_float('TOTAL_TIMEOUT_SECONDS', 2400.0)

    # 日志配置
    LOG_LEVEL: str = os.environ.get('LOG_LEVEL', 'INFO')
    
    # Redis配置
    REDIS_HOST: str = os.environ.get('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.environ.get('REDIS_PORT', '6379'))
    REDIS_DB: int = int(os.environ.get('REDIS_DB', '0'))
    
    # 回测配置
    INITIAL_CASH: float = float(os.environ.get('INITIAL_CASH', '1000000'))
    COMMISSION_RATE: float = float(os.environ.get('COMMISSION_RATE', '0.0003'))
    STAMP_DUTY_RATE: float = float(os.environ.get('STAMP_DUTY_RATE', '0.001'))
    SLIPPAGE: float = float(os.environ.get('SLIPPAGE', '0.001'))

    @classmethod
    def validate(cls) -> list:
        """验证配置是否完整"""
        errors = []
        
        if not cls.TUSHARE_TOKEN:
            errors.append("TUSHARE_TOKEN 未设置")
        
        return errors


# 导出配置
config = Config()
