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
    "MACRO_V2_OBSERVER_ENABLED": "0",
    "MACRO_V2_OBSERVER_KILL_SWITCH": "1",
    "MACRO_V2_PRODUCTION_ENABLED": "0",
    "MACRO_V2_PRODUCTION_KILL_SWITCH": "1",
    "MACRO_V2_OBSERVATIONS_PATH": "data/parquet/cn/macro_observations",
    "MACRO_V2_OBSERVER_OUTPUT_DIR": "results/v14/macro_observer",
    "MYQUANT_LLM_HANDOFF": "codex",
    "MYQUANT_DISABLE_LOCAL_LLM": "true",
    "QUANT_PRODUCTION_KILL_SWITCH": "true",
    "QUANT_PRODUCTION_ACTIVATION_RECEIPT": "",
    "QUANT_PRODUCTION_ACTIVATION_RECEIPT_SHA256": "",
    "FUNDAMENTAL_RESEARCH_OVERLAY_MODE": "shadow",
    "FUNDAMENTAL_RESEARCH_ROOT": "results/fundamental_research",
    "FUNDAMENTAL_RESEARCH_ACTIVATION_PATH": "",
    "FUNDAMENTAL_RESEARCH_ACTIVATION_EXPECTED_SHA256": "",
    "RISK_GUARD_SINGLE_NAME_WEIGHT_CAP": "0.15",
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
    MACRO_V2_OBSERVER_ENABLED: bool = _env_bool(
        'MACRO_V2_OBSERVER_ENABLED',
        MAINLINE_ENV_DEFAULTS['MACRO_V2_OBSERVER_ENABLED'] == '1',
    )
    MACRO_V2_OBSERVER_KILL_SWITCH: bool = _env_bool(
        'MACRO_V2_OBSERVER_KILL_SWITCH',
        MAINLINE_ENV_DEFAULTS['MACRO_V2_OBSERVER_KILL_SWITCH'] == '1',
    )
    MACRO_V2_PRODUCTION_ENABLED: bool = _env_bool(
        'MACRO_V2_PRODUCTION_ENABLED',
        MAINLINE_ENV_DEFAULTS['MACRO_V2_PRODUCTION_ENABLED'] == '1',
    )
    MACRO_V2_PRODUCTION_KILL_SWITCH: bool = _env_bool(
        'MACRO_V2_PRODUCTION_KILL_SWITCH',
        MAINLINE_ENV_DEFAULTS['MACRO_V2_PRODUCTION_KILL_SWITCH'] == '1',
    )
    MACRO_V2_OBSERVATIONS_PATH: str = _env_str(
        'MACRO_V2_OBSERVATIONS_PATH',
        MAINLINE_ENV_DEFAULTS['MACRO_V2_OBSERVATIONS_PATH'],
    )
    MACRO_V2_OBSERVER_OUTPUT_DIR: str = _env_str(
        'MACRO_V2_OBSERVER_OUTPUT_DIR',
        MAINLINE_ENV_DEFAULTS['MACRO_V2_OBSERVER_OUTPUT_DIR'],
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
