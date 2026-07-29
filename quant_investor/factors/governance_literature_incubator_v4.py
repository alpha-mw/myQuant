"""Research-only, source-backed factor ideas for future Factor v4 cycles.

This module translates a small set of primary-source findings into causal,
filesystem-free signal definitions that can be preregistered in a later
governance cycle.  It is deliberately separate from the frozen v4.4
exact-five oracle.  No function in this module measures outcomes, evaluates
labels, writes governance state, or grants production authority.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
import pandas as pd

from quant_investor.factors import governance_screening_v4 as screening

PROTOCOL_VERSION = "v4"
INCUBATOR_VERSION = "v4-literature-incubator.v10"
SHELL_EXCLUSION_FRACTION = 0.30
LOW_BETA_WINDOW = 252
LOW_BETA_MIN_PERIODS = 126
HIGH_52_WEEK_CALENDAR_DAYS = 365
HIGH_52_WEEK_MIN_PERIODS = 200
HIGH_52_WEEK_SELECTION_DATE = "2026-07-28"
PRICE_DELAY_SELECTION_DATE = "2026-07-28"
PRICE_DELAY_WEEKDAY = 2
PRICE_DELAY_WINDOW_WEEKS = 52
PRICE_DELAY_MARKET_LAGS = 4
PRICE_DELAY_MIN_OBSERVATIONS = 40
LOW_MAX_WINDOW = 20
LOW_MAX_MIN_PERIODS = 15
LOW_MAX_SELECTION_DATE = "2026-07-28"
LOW_TOTAL_SKEW_MIN_PERIODS = 15
LOW_TOTAL_SKEW_SELECTION_DATE = "2026-07-28"
LOW_TOTAL_SKEW_WINDOW = 20
TAIL_ASYMMETRY_SELECTION_DATE = "2026-07-28"
TAIL_ASYMMETRY_WINDOW = 252
TAIL_ASYMMETRY_MIN_PERIODS = 126
TAIL_ASYMMETRY_SIGMA_THRESHOLD = 1.0
SAME_MONTH_LOOKBACK_SESSIONS = 1300
SAME_MONTH_MAX_ANNUAL_LAGS = 5
SAME_MONTH_MIN_ANNUAL_OBSERVATIONS = 3
SAME_MONTH_MIN_SESSIONS_PER_MONTH = 10
FIP_FORMATION_SESSIONS = 231
FIP_SKIP_SESSIONS = 21
FIP_MIN_PERIODS = 200
FIP_SELECTION_DATE = "2026-07-28"
LEFT_TAIL_VAR_MIN_PERIODS = 200
LEFT_TAIL_VAR_QUANTILE = 0.01
LEFT_TAIL_VAR_SELECTION_DATE = "2026-07-28"
LEFT_TAIL_VAR_WINDOW = 250
PROTECTED_EXACT_FIVE_CANDIDATE_NAMES = (
    "alpha_range_position_momentum_20d",
    "pv_low_overnight_gap_20d",
    "pv_low_vol_ratio_10_60",
    "pv_price_volume_consistency_20d",
    "pv_low_vol_of_vol_20d",
)

AUTHORITY_FLAGS = {
    "formal_preregistration": False,
    "measurement": False,
    "family_bh": False,
    "maturity": False,
    "walk_forward": False,
    "cost": False,
    "neutralization": False,
    "stability": False,
    "dedup": False,
    "canonical_replay": False,
    "candidate_qualified": False,
    "admission": False,
    "registry_write": False,
    "production": False,
    "activation": False,
}

SIDE_EFFECT_FLAGS = {
    "filesystem": False,
    "provider": False,
    "network": False,
    "registry": False,
    "wal": False,
    "portfolio": False,
    "broker": False,
    "order": False,
    "trade": False,
}

_CANDIDATE_ORDER = (
    "cn_earnings_yield_ex_shell_30pct",
    "cn_low_beta_252d",
    "cn_52_week_high_momentum_12m",
    "cn_high_price_delay_d1_52w",
    "cn_low_max_return_20d",
    "cn_low_total_skewness_20d",
    "cn_low_market_adjusted_tail_asymmetry_252d",
    "cn_quality_cash_low_leverage",
    "cn_same_month_seasonality_5y",
    "cn_fip_continuous_direction_12m",
    "cn_low_left_tail_var1_250d",
)

_CANDIDATE_DEFINITIONS = (
    {
        "name": "cn_earnings_yield_ex_shell_30pct",
        "implementation": "literature_incubator:earnings_yield_ex_shell_v4",
        "expression": ("cs_rank((1 / pe) where pe > 0 and " "cs_rank(total_mv) > 0.30)"),
        "direction": 1.0,
        "params": {"shell_exclusion_fraction": 0.30},
        "lookback": 1,
        "slot": "primitive:earnings_yield",
        "input_fields": ["pe", "total_mv"],
        "primitive_ids": ["earnings_yield"],
    },
    {
        "name": "cn_low_beta_252d",
        "implementation": "literature_incubator:low_beta_v4",
        "expression": (
            "-rolling_cov(pct_change(adj_close), "
            "equal_weight_pit_market_return, 252, min_periods=126) / "
            "rolling_var(equal_weight_pit_market_return, 252, "
            "min_periods=126)"
        ),
        "direction": 1.0,
        "params": {"min_periods": 126, "window": 252},
        "lookback": 252,
        "slot": "primitive:low_beta",
        "input_fields": ["adj_close"],
        "primitive_ids": ["low_beta"],
    },
    {
        "name": "cn_52_week_high_momentum_12m",
        "implementation": "literature_incubator:high_52_week_momentum_v4",
        "expression": (
            "cs_rank(adj_close / ts_max(adj_close, trailing_365_calendar_days, " "min_periods=200))"
        ),
        "direction": 1.0,
        "params": {
            "calendar_days": 365,
            "min_periods": 200,
            "source_formation_frequency": "monthly",
            "translation_frequency": "causal_daily_with_month_end_governance_measurement",
        },
        "lookback": 262,
        "slot": "primitive:price_to_52_week_high",
        "input_fields": ["adj_close"],
        "primitive_ids": ["price_to_52_week_high"],
    },
    {
        "name": "cn_high_price_delay_d1_52w",
        "implementation": "literature_incubator:high_price_delay_d1_v4",
        "expression": (
            "cs_rank(1 - R2(weekly_stock_return ~ current_value_weighted_"
            "market_return) / R2(weekly_stock_return ~ current_and_1_to_4_"
            "week_lagged_value_weighted_market_returns)); exact Wednesday "
            "anchors, rolling prior 52 response weeks, min_observations=40"
        ),
        "direction": 1.0,
        "params": {
            "anchor_weekday": "WEDNESDAY",
            "daily_expansion": "last_completed_wednesday_observation",
            "d1_bounds": [0.0, 1.0],
            "market_lags": [0, 1, 2, 3, 4],
            "market_weighting": "prior_wednesday_total_mv",
            "min_observations": 40,
            "restricted_market_lags": [0],
            "window_weeks": 52,
        },
        "lookback": 300,
        "slot": "primitive:price_delay_d1",
        "input_fields": ["adj_close", "total_mv"],
        "primitive_ids": ["price_delay_d1"],
    },
    {
        "name": "cn_low_max_return_20d",
        "implementation": "literature_incubator:low_max_return_v4",
        "expression": ("cs_rank(-ts_max(pct_change(adj_close), 20, min_periods=15))"),
        "direction": 1.0,
        "params": {"min_periods": 15, "window": 20},
        "lookback": 21,
        "slot": "primitive:max_daily_return",
        "input_fields": ["adj_close"],
        "primitive_ids": ["max_daily_return"],
    },
    {
        "name": "cn_low_total_skewness_20d",
        "implementation": "literature_incubator:low_total_skewness_v4",
        "expression": (
            "cs_rank(-unbiased_sample_skewness(" "pct_change(adj_close), 20, min_periods=15))"
        ),
        "direction": 1.0,
        "params": {
            "estimator": "unbiased_sample_skewness",
            "min_periods": 15,
            "window": 20,
        },
        "lookback": 21,
        "slot": "primitive:total_return_skewness",
        "input_fields": ["adj_close"],
        "primitive_ids": ["total_return_skewness"],
    },
    {
        "name": "cn_low_market_adjusted_tail_asymmetry_252d",
        "implementation": ("literature_incubator:low_market_adjusted_tail_asymmetry_v4"),
        "expression": (
            "cs_rank(-(P(z(market_adjusted_daily_return)>1) - "
            "P(z(market_adjusted_daily_return)<-1)) over 252, "
            "min_periods=126)"
        ),
        "direction": 1.0,
        "params": {
            "market_adjustment": "pit_equal_weight_daily_return",
            "min_periods": 126,
            "sigma_threshold": 1.0,
            "window": 252,
        },
        "lookback": 253,
        "slot": "primitive:market_adjusted_tail_asymmetry",
        "input_fields": ["adj_close"],
        "primitive_ids": ["market_adjusted_tail_asymmetry"],
    },
    {
        "name": "cn_quality_cash_low_leverage",
        "implementation": ("literature_incubator:quality_cash_low_leverage_v4"),
        "expression": (
            "mean(cs_rank(fin_roe), cs_rank(fin_ocf_to_profit), "
            "1 - cs_rank(fin_debt_to_assets)); complete cases only"
        ),
        "direction": 1.0,
        "params": {
            "aggregation": "arithmetic_mean",
            "complete_cases_only": True,
        },
        "lookback": 1,
        "slot": ("primitive:fin_debt_to_assets+fin_ocf_to_profit+fin_roe"),
        "input_fields": [
            "fin_debt_to_assets",
            "fin_ocf_to_profit",
            "fin_roe",
        ],
        "primitive_ids": [
            "fin_debt_to_assets",
            "fin_ocf_to_profit",
            "fin_roe",
        ],
    },
    {
        "name": "cn_same_month_seasonality_5y",
        "implementation": "literature_incubator:same_month_seasonality_v4",
        "expression": (
            "cs_rank(mean(market_adjusted_monthly_return at calendar "
            "month lags 12,24,36,48,60; min_annual_observations=3))"
        ),
        "direction": 1.0,
        "params": {
            "market_adjustment": "pit_equal_weight_monthly_return",
            "max_annual_lags": 5,
            "min_annual_observations": 3,
            "min_sessions_per_month": 10,
        },
        "lookback": 1300,
        "slot": "primitive:same_month_seasonality",
        "input_fields": ["adj_close"],
        "primitive_ids": ["same_month_seasonality"],
    },
    {
        "name": "cn_fip_continuous_direction_12m",
        "implementation": "literature_incubator:fip_continuous_direction_v4",
        "expression": (
            "cs_rank(sign(PRET) * (1 - ID) / 2); "
            "PRET=shift(adj_close,21)/shift(adj_close,252)-1; "
            "ID=sign(PRET)*(N_negative-N_positive)/N_valid over the "
            "same shifted 231-session formation window, min_periods=200"
        ),
        "direction": 1.0,
        "params": {
            "formation_sessions": 231,
            "min_periods": 200,
            "skip_sessions": 21,
            "source_formation_frequency": "monthly",
            "translation_frequency": ("causal_daily_with_closed_month_end_governance_measurement"),
            "single_score_translation": ("formation_return_direction_times_information_continuity"),
        },
        "lookback": 253,
        "slot": "primitive:fip_continuous_direction",
        "input_fields": ["adj_close"],
        "primitive_ids": ["fip_continuous_direction"],
    },
    {
        "name": "cn_low_left_tail_var1_250d",
        "implementation": "literature_incubator:low_left_tail_var1_v4",
        "expression": (
            "cs_rank(-VaR1); VaR1=-rolling_quantile("
            "pct_change(adj_close),0.01,250,min_periods=200)"
        ),
        "direction": 1.0,
        "params": {
            "min_periods": 200,
            "quantile": 0.01,
            "source_formation_frequency": "monthly",
            "translation_frequency": ("causal_daily_with_closed_month_end_governance_measurement"),
            "window": 250,
        },
        "lookback": 251,
        "slot": "primitive:left_tail_var1",
        "input_fields": ["adj_close"],
        "primitive_ids": ["left_tail_var1"],
    },
)

_CANDIDATE_SOURCE_IDS = {
    "cn_earnings_yield_ex_shell_30pct": "liu_stambaugh_yuan_2019",
    "cn_low_beta_252d": "frazzini_pedersen_2014",
    "cn_52_week_high_momentum_12m": "zhou_liu_guo_2021",
    "cn_high_price_delay_d1_52w": "qian_sun_yu_2017",
    "cn_low_max_return_20d": "gao_han_xiong_2021",
    "cn_low_total_skewness_20d": "wang_wang_wu_2023",
    "cn_low_market_adjusted_tail_asymmetry_252d": "chen_wu_zhu_2022",
    "cn_quality_cash_low_leverage": "asness_frazzini_pedersen_2019",
    "cn_same_month_seasonality_5y": "meng_du_shu_2024",
    "cn_fip_continuous_direction_12m": "da_gurun_warachka_2014",
    "cn_low_left_tail_var1_250d": "zhen_ruan_zhang_2020",
}

_CANDIDATE_PRIMITIVES = (
    {"primitive_id": "earnings_yield", "family": "earnings_yield"},
    {
        "primitive_id": "fin_debt_to_assets",
        "family": "fin_debt_to_assets",
    },
    {
        "primitive_id": "fin_ocf_to_profit",
        "family": "fin_ocf_to_profit",
    },
    {"primitive_id": "fin_roe", "family": "fin_roe"},
    {"primitive_id": "low_beta", "family": "low_beta"},
    {
        "primitive_id": "price_to_52_week_high",
        "family": "price_to_52_week_high",
    },
    {
        "primitive_id": "price_delay_d1",
        "family": "price_delay_d1",
    },
    {
        "primitive_id": "max_daily_return",
        "family": "max_daily_return",
    },
    {
        "primitive_id": "total_return_skewness",
        "family": "total_return_skewness",
    },
    {
        "primitive_id": "market_adjusted_tail_asymmetry",
        "family": "market_adjusted_tail_asymmetry",
    },
    {
        "primitive_id": "same_month_seasonality",
        "family": "same_month_seasonality",
    },
    {
        "primitive_id": "fip_continuous_direction",
        "family": "fip_continuous_direction",
    },
    {
        "primitive_id": "left_tail_var1",
        "family": "left_tail_var1",
    },
)

LITERATURE_IDEAS = (
    {
        "source_id": "liu_stambaugh_yuan_2019",
        "idea": "china_earnings_yield_ex_shell",
        "primary_url": "https://www.nber.org/papers/w24458",
        "source_scope": "China A shares",
        "translation": (
            "Use earnings-to-price rather than book-to-market and exclude the "
            "smallest 30 percent of the cross-section before testing value."
        ),
        "local_data_status": "IMPLEMENTABLE",
    },
    {
        "source_id": "frazzini_pedersen_2014",
        "idea": "low_beta",
        "primary_url": (
            "https://www.aqr.com/Insights/Research/Journal-Article/" "Betting-Against-Beta"
        ),
        "source_scope": "global asset classes and equities",
        "translation": (
            "Test a low-beta signal, but subject it to size/sector "
            "neutralization and correlation dedup against low-volatility."
        ),
        "local_data_status": "IMPLEMENTABLE",
    },
    {
        "source_id": "hou_moskowitz_2005",
        "idea": "price_delay_d1",
        "title": ("Market Frictions, Price Delay, and the Cross-Section of " "Expected Returns"),
        "authors": ["Kewei Hou", "Tobias J. Moskowitz"],
        "published_period": "2005",
        "journal": "The Review of Financial Studies",
        "primary_url": ("https://academic.oup.com/rfs/article-abstract/18/3/981/1597514"),
        "doi_url": "https://doi.org/10.1093/rfs/hhi023",
        "author_copy_url": "https://www.ruf.rice.edu/~jgsfss/moskowitz.pdf",
        "source_scope": "US equities; foundational method",
        "translation": (
            "Use the D1 measure: one minus restricted R-squared divided by "
            "unrestricted R-squared, where weekly stock returns are regressed "
            "on the contemporaneous value-weighted market return alone or "
            "together with four weekly market-return lags over the prior year."
        ),
        "evidence_role": "FOUNDATIONAL_PRICE_DELAY_METHOD",
        "local_data_status": "REFERENCE_FOUNDATIONAL_METHOD",
    },
    {
        "source_id": "qian_sun_yu_2017",
        "idea": "china_high_price_delay_d1",
        "title": (
            "High turnover with high price delay? Dissecting the puzzling "
            "phenomenon for China's A-shares"
        ),
        "authors": ["Meifen Qian", "Ping-Wen Sun", "Bin Yu"],
        "published_period": "2017-08",
        "journal": "Finance Research Letters",
        "primary_url": (
            "https://www.sciencedirect.com/science/article/abs/pii/" "S1544612316302781"
        ),
        "doi_url": "https://doi.org/10.1016/j.frl.2017.06.004",
        "source_scope": "China A shares",
        "translation": (
            "Apply Hou-Moskowitz D1 to China A shares. The paper reports a "
            "China price-delay premium and positive future-return relations "
            "for both turnover and non-turnover components of price delay; "
            "illiquidity, uncertainty, and attention are locked mechanism "
            "controls rather than substitutes for D1."
        ),
        "evidence_role": "DIRECT_CHINA_SUPPORT_WITH_MICROSTRUCTURE_CONTROLS",
        "local_data_status": "IMPLEMENTABLE_CAUSAL_ROLLING_TRANSLATION",
    },
    {
        "source_id": "george_hwang_2004",
        "idea": "price_to_52_week_high",
        "title": "The 52-Week High and Momentum Investing",
        "authors": ["Thomas J. George", "Chuan-Yang Hwang"],
        "published_period": "2004-10",
        "journal": "The Journal of Finance",
        "primary_url": (
            "https://onlinelibrary.wiley.com/doi/abs/" "10.1111/j.1540-6261.2004.00695.x"
        ),
        "doi_url": "https://doi.org/10.1111/j.1540-6261.2004.00695.x",
        "author_copy_url": "https://www.bauer.uh.edu/TGeorge/papers/gh4-paper.pdf",
        "source_scope": "US equities; foundational method",
        "translation": (
            "At each month end, rank current price divided by the highest "
            "price reached during the trailing 12 months. The original "
            "strategy uses top and bottom 30 percent portfolios and a "
            "six-month holding period."
        ),
        "evidence_role": "FOUNDATIONAL_52_WEEK_HIGH_METHOD",
        "local_data_status": "REFERENCE_FOUNDATIONAL_METHOD",
    },
    {
        "source_id": "zhou_liu_guo_2021",
        "idea": "china_52_week_high_momentum",
        "title": (
            "The 52-week High Momentum Strategy and Economic Policy "
            "Uncertainty: Evidence from China"
        ),
        "authors": ["Xuemei Zhou", "Qiang Liu", "Shuxin Guo"],
        "published_date": "2021-04-21",
        "journal": "Emerging Markets Finance and Trade",
        "primary_url": ("https://www.tandfonline.com/doi/full/" "10.1080/1540496X.2021.1904880"),
        "doi_url": "https://doi.org/10.1080/1540496X.2021.1904880",
        "source_scope": "China A shares",
        "translation": (
            "Apply the George-Hwang price-to-52-week-high ranking to China "
            "A shares. The paper reports significant unconditional momentum "
            "but virtually no momentum in high-EPU periods, so a PIT China "
            "EPU regime split is a mandatory falsification test."
        ),
        "evidence_role": "DIRECT_CHINA_SUPPORT_WITH_EPU_REGIME_RISK",
        "local_data_status": "IMPLEMENTABLE_SIGNAL_BLOCKED_PIT_EPU_REGIME",
    },
    {
        "source_id": "blitz_hanauer_van_vliet_2021",
        "idea": "china_low_risk_volatility_not_beta",
        "title": "The Volatility Effect in China",
        "authors": ["David Blitz", "Matthias X. Hanauer", "Pim van Vliet"],
        "published_date": "2021-04-20",
        "journal": "Journal of Asset Management",
        "primary_url": ("https://link.springer.com/article/10.1057/s41260-021-00218-0"),
        "doi_url": "https://doi.org/10.1057/s41260-021-00218-0",
        "source_scope": "China A shares",
        "translation": (
            "Treat raw beta as a control rather than a preregistration target: "
            "the China low-risk premium is reported as driven by volatility, "
            "not beta, and remains investable in large liquid stocks."
        ),
        "evidence_role": "ADVERSE_TO_RAW_LOW_BETA",
        "local_data_status": "IMPLEMENTABLE_CONTROL_ONLY",
    },
    {
        "source_id": "chen_huang_qiu_2021",
        "idea": "china_beta_anomaly_conditional",
        "title": (
            "Heterogeneous Beliefs and the Beta Anomaly in the Chinese " "A-share Stock Market"
        ),
        "authors": ["Shu Chen", "Zhuo Huang", "Zhimin Qiu"],
        "published_date": "2020-12-16",
        "journal": "Emerging Markets Finance and Trade",
        "primary_url": ("https://www.tandfonline.com/doi/full/" "10.1080/1540496X.2020.1822809"),
        "doi_url": "https://doi.org/10.1080/1540496X.2020.1822809",
        "source_scope": "China A shares",
        "translation": (
            "Use the reported China beta anomaly only as a conditional "
            "mechanism test: it disappears for low disagreement or low "
            "arbitrage-limit states."
        ),
        "evidence_role": "CONDITIONAL_SUPPORT_FOR_RAW_LOW_BETA",
        "local_data_status": "BLOCKED_MISSING_PIT_DISAGREEMENT_AND_ARBITRAGE_LIMITS",
    },
    {
        "source_id": "zhao_lin_2022",
        "idea": "china_beta_behavioral_volatility_mechanism",
        "title": (
            "Does behavioral-motivated volatility effect explain the beta "
            "anomaly? Evidence from China"
        ),
        "authors": ["Lu Zhao", "Lei Lin"],
        "published_period": "2022-05",
        "journal": "Finance Research Letters",
        "primary_url": (
            "https://www.sciencedirect.com/science/article/abs/pii/" "S154461232100307X"
        ),
        "doi_url": "https://doi.org/10.1016/j.frl.2021.102265",
        "source_scope": "China A shares",
        "translation": (
            "Require raw beta to add value beyond volatility, MAX lottery "
            "demand, and idiosyncratic risk; the paper reports that these "
            "behavioral effects explain the China beta anomaly."
        ),
        "evidence_role": "ADVERSE_TO_RAW_LOW_BETA_SUPPORTS_MAX_MECHANISM",
        "local_data_status": "IMPLEMENTABLE_PARTIAL_MAX_PROXY",
    },
    {
        "source_id": "gao_han_xiong_2021",
        "idea": "china_low_max_return",
        "title": "Loss from the chasing of MAX stocks: Evidence from China",
        "authors": ["Ya Gao", "Xing Han", "Xiong Xiong"],
        "published_period": "2021-11",
        "journal": "The North American Journal of Economics and Finance",
        "primary_url": (
            "https://www.sciencedirect.com/science/article/abs/pii/" "S1062940821000966"
        ),
        "doi_url": "https://doi.org/10.1016/j.najef.2021.101475",
        "source_scope": "China A shares",
        "translation": (
            "Rank the negative maximum daily return over the prior 20 "
            "sessions. The paper reports that chasing high-MAX China stocks "
            "lost money, with stronger effects in speculative states and a "
            "weaker effect after short selling was introduced."
        ),
        "evidence_role": "DIRECT_CHINA_SUPPORT_WITH_REGIME_RISK",
        "local_data_status": "IMPLEMENTABLE",
    },
    {
        "source_id": "wang_wang_wu_2023",
        "idea": "china_low_total_skewness",
        "title": (
            "The role of anchoring on investors' gambling preference: " "Evidence from China"
        ),
        "authors": ["Zhuo Wang", "Ziyue Wang", "Ke Wu"],
        "published_period": "2023-09",
        "journal": "Pacific-Basin Finance Journal",
        "primary_url": (
            "https://www.sciencedirect.com/science/article/abs/pii/" "S0927538X23001208"
        ),
        "doi_url": "https://doi.org/10.1016/j.pacfin.2023.102054",
        "source_scope": "China A shares",
        "translation": (
            "Rank the negative unbiased sample skewness of daily total "
            "returns over the prior 20 sessions, requiring at least 15 "
            "observations. Treat distance from the 52-week high, arbitrage "
            "risk, and investor sentiment as locked conditioning tests."
        ),
        "evidence_role": "DIRECT_CHINA_SUPPORT_WITH_52_WEEK_ANCHOR_CONDITION",
        "local_data_status": "IMPLEMENTABLE",
    },
    {
        "source_id": "jiang_wu_zhou_zhu_2020",
        "idea": "distribution_based_return_asymmetry",
        "title": "Stock Return Asymmetry: Beyond Skewness",
        "authors": ["Lei Jiang", "Ke Wu", "Guofu Zhou", "Yifeng Zhu"],
        "published_period": "2020",
        "journal": "Journal of Financial and Quantitative Analysis",
        "primary_url": (
            "https://www.cambridge.org/core/journals/"
            "journal-of-financial-and-quantitative-analysis/article/"
            "stock-return-asymmetry-beyond-skewness/"
            "6DB44C0DB241D0030AAA7F885CD078DB"
        ),
        "doi_url": "https://doi.org/10.1017/S0022109019000206",
        "author_copy_url": (
            "https://www.cb.cityu.edu.hk/ef/doc/2016%20Sofie/Papers/"
            "5_Wu_Stock%20Return%20Asymmetry%20Beyond%20Skewness.pdf"
        ),
        "source_scope": "US equities; foundational method",
        "translation": (
            "Estimate excess tail probability as the probability of a "
            "standardized return above one sigma minus the probability below "
            "minus one sigma. Prefer lower upside asymmetry."
        ),
        "evidence_role": "FOUNDATIONAL_DISTRIBUTION_METHOD",
        "local_data_status": "REFERENCE_FOUNDATIONAL_METHOD",
    },
    {
        "source_id": "chen_wu_zhu_2022",
        "idea": "china_distribution_based_return_asymmetry",
        "title": "Stock return asymmetry in China",
        "authors": ["Dongxu Chen", "Ke Wu", "Yifeng Zhu"],
        "published_period": "2022-06",
        "journal": "Pacific-Basin Finance Journal",
        "primary_url": (
            "https://www.sciencedirect.com/science/article/abs/pii/" "S0927538X2200052X"
        ),
        "doi_url": "https://doi.org/10.1016/j.pacfin.2022.101757",
        "source_scope": "China A shares",
        "translation": (
            "Test a PIT equal-weight-market-adjusted excess-tail-probability "
            "proxy, while recording that it is not the paper's full CH-3/CH-4 "
            "idiosyncratic implementation."
        ),
        "evidence_role": "DIRECT_CHINA_SUPPORT_PROXY_DEFINITION",
        "local_data_status": "IMPLEMENTABLE_PROXY_ONLY",
    },
    {
        "source_id": "asness_frazzini_pedersen_2019",
        "idea": "quality",
        "primary_url": (
            "https://www.aqr.com/Insights/Research/Working-Paper/" "Quality-Minus-Junk"
        ),
        "source_scope": "global equities",
        "translation": (
            "Use a conservative PIT quality proxy from profitability, cash "
            "conversion, and leverage; do not claim the full paper definition."
        ),
        "local_data_status": "IMPLEMENTABLE_PROXY_ONLY",
    },
    {
        "source_id": "novy_marx_2013",
        "idea": "gross_profitability",
        "primary_url": "https://www.nber.org/papers/w15940",
        "source_scope": "US equities",
        "translation": "Gross profits divided by assets.",
        "local_data_status": ("BLOCKED_MISSING_PIT_GROSS_PROFIT_AND_TOTAL_ASSETS_HISTORY"),
    },
    {
        "source_id": "hou_xue_zhang_2015",
        "idea": "investment_and_profitability",
        "primary_url": "https://academic.oup.com/rfs/article/28/3/650/1574802",
        "source_scope": "US equities",
        "translation": (
            "Test investment and profitability factors only after lagged PIT "
            "asset-growth inputs are available."
        ),
        "local_data_status": "BLOCKED_MISSING_PIT_ASSET_GROWTH_HISTORY",
    },
    {
        "source_id": "meng_du_shu_2024",
        "idea": "china_same_month_seasonality",
        "primary_url": (
            "https://research.birmingham.ac.uk/en/publications/"
            "return-seasonalities-in-the-chinese-stock-market/"
        ),
        "doi_url": "https://doi.org/10.1016/j.pacfin.2024.102391",
        "source_scope": "China stocks",
        "translation": (
            "Test whether stocks with stronger returns in a calendar month "
            "continue to outperform in that same calendar month in later "
            "years, while treating other-month reversal as adverse evidence."
        ),
        "local_data_status": "IMPLEMENTABLE",
    },
    {
        "source_id": "heston_sadka_2008",
        "idea": "same_month_seasonality",
        "primary_url": ("https://www.sciencedirect.com/science/article/pii/" "S0304405X0700195X"),
        "author_copy_url": (
            "https://w4.stern.nyu.edu/finance/docs/pdfs/Seminars/" "063f-sadka.pdf"
        ),
        "source_scope": "US equities with international follow-up",
        "translation": (
            "Average PIT equal-weight-market-adjusted returns from the same "
            "calendar month in the prior one to five years, requiring at "
            "least three complete annual observations."
        ),
        "local_data_status": "REFERENCE_FOUNDATIONAL_METHOD",
    },
    {
        "source_id": "da_gurun_warachka_2014",
        "idea": "frog_in_the_pan_continuous_information_momentum",
        "title": "Frog in the Pan: Continuous Information and Momentum",
        "authors": ["Zhi Da", "Umit G. Gurun", "Mitch Warachka"],
        "published_period": "2014",
        "journal": "The Review of Financial Studies",
        "primary_url": "https://academicweb.nd.edu/~zda/Frog.pdf",
        "doi_url": "https://doi.org/10.1093/rfs/hhu003",
        "source_scope": "US equities; foundational method",
        "translation": (
            "Use the source ID definition, sign(PRET) times the fraction of "
            "negative formation-period daily returns minus the fraction of "
            "positive formation-period daily returns. Translate the source "
            "double-sort into one non-authoritative score: the sign of PRET "
            "times (1-ID)/2, so continuous information strengthens the prior "
            "return direction while discrete information moves the score "
            "toward zero."
        ),
        "evidence_role": "FOUNDATIONAL_FIP_METHOD_SINGLE_SCORE_TRANSLATION",
        "local_data_status": "IMPLEMENTABLE_CAUSAL_DAILY_TRANSLATION",
    },
    {
        "source_id": "zhang_chen_feng_2026",
        "idea": "china_aggregate_information_discreteness",
        "title": "Information Discreteness and Stock Market Returns: Evidence from China",
        "authors": ["Yaojie Zhang", "Jie Chen", "Yuqing Feng"],
        "published_date": "2026-03-16",
        "journal": "Emerging Markets Finance and Trade",
        "primary_url": ("https://www.tandfonline.com/doi/full/" "10.1080/1540496X.2026.2641079"),
        "doi_url": "https://doi.org/10.1080/1540496X.2026.2641079",
        "source_scope": "China aggregate stock-market returns",
        "translation": (
            "Treat the reported negative predictive relation between ID and "
            "aggregate China excess stock-market returns as market-relevance "
            "evidence only. It is not direct support for a cross-sectional "
            "A-share signal and cannot replace local Factor v4 measurement."
        ),
        "evidence_role": "CHINA_AGGREGATE_RELEVANCE_NOT_CROSS_SECTIONAL_SUPPORT",
        "local_data_status": "REFERENCE_AGGREGATE_ONLY",
    },
    {
        "source_id": "atilgan_bali_demirtas_gunaydin_2020",
        "idea": "left_tail_momentum",
        "title": (
            "Left-tail momentum: Underreaction to bad news, costly arbitrage " "and equity returns"
        ),
        "authors": [
            "Yigit Atilgan",
            "Turan G. Bali",
            "K. Ozgur Demirtas",
            "A. Doruk Gunaydin",
        ],
        "published_period": "2020-03",
        "journal": "Journal of Financial Economics",
        "primary_url": ("https://www.sciencedirect.com/science/article/pii/" "S0304405X19301795"),
        "doi_url": "https://doi.org/10.1016/j.jfineco.2019.07.006",
        "source_scope": "US equities; foundational left-tail method",
        "translation": (
            "Use a nonparametric lower-tail return statistic measured from "
            "daily returns over the prior year, while requiring a separate "
            "China source before treating its direction as locally supported."
        ),
        "evidence_role": "FOUNDATIONAL_LEFT_TAIL_METHOD",
        "local_data_status": "REFERENCE_FOUNDATIONAL_METHOD",
    },
    {
        "source_id": "zhen_ruan_zhang_2020",
        "idea": "china_low_left_tail_var1",
        "title": "Left-tail risk in China",
        "authors": ["Fang Zhen", "Xinfeng Ruan", "Jin E. Zhang"],
        "published_period": "2020-10",
        "journal": "Pacific-Basin Finance Journal",
        "primary_url": ("https://www.sciencedirect.com/science/article/pii/" "S0927538X20301797"),
        "doi_url": "https://doi.org/10.1016/j.pacfin.2020.101391",
        "source_scope": "China A shares",
        "translation": (
            "At each closed month end, define VaR1 as negative one times the "
            "first percentile of daily returns over the prior 250 trading "
            "days, requiring at least 200 nonmissing returns. Prefer low "
            "VaR1 because the paper reports a negative relation between "
            "left-tail risk and next-month China stock returns."
        ),
        "evidence_role": "DIRECT_CHINA_SUPPORT_EXACT_VAR1_SIGNAL",
        "local_data_status": "IMPLEMENTABLE",
    },
    {
        "source_id": "lin_2019_china_residual_momentum",
        "idea": "china_residual_momentum",
        "title": ("Residual momentum and the cross-section of stock returns: " "Chinese evidence"),
        "authors": ["Qi Lin"],
        "published_period": "2019-06",
        "journal": "Finance Research Letters",
        "primary_url": ("https://www.sciencedirect.com/science/article/pii/" "S1544612318303325"),
        "doi_url": "https://doi.org/10.1016/j.frl.2018.07.009",
        "source_scope": "China A shares",
        "translation": (
            "Estimate residual momentum from monthly excess-return residuals "
            "of rolling China Fama-French three-factor regressions only after "
            "the exact PIT risk-free and China factor histories are bound."
        ),
        "evidence_role": "DIRECT_CHINA_SUPPORT_BLOCKED_EXACT_INPUTS",
        "local_data_status": ("BLOCKED_MISSING_PIT_MONTHLY_RISK_FREE_AND_CHINA_FF3_HISTORY"),
    },
    {
        "source_id": "lin_2022_china_idiosyncratic_momentum",
        "idea": "china_idiosyncratic_momentum_pls",
        "title": "Understanding idiosyncratic momentum in the Chinese stock market",
        "authors": ["Qi Lin"],
        "published_period": "2022-01",
        "journal": ("Journal of International Financial Markets, Institutions and Money"),
        "primary_url": ("https://www.sciencedirect.com/science/article/pii/" "S104244312100175X"),
        "doi_url": "https://doi.org/10.1016/j.intfin.2021.101469",
        "source_scope": "China A shares",
        "translation": (
            "Aggregate eight multifactor residual-momentum measures with the "
            "source partial-least-squares design only after all model factors "
            "and estimation histories are available."
        ),
        "evidence_role": "DIRECT_CHINA_SUPPORT_BLOCKED_MULTIFACTOR_INPUTS",
        "local_data_status": ("BLOCKED_MISSING_EIGHT_MULTIFACTOR_RESIDUAL_INPUTS_AND_PLS_DESIGN"),
    },
    {
        "source_id": "ma_liao_jiang_2024_factor_momentum",
        "idea": "china_factor_momentum",
        "title": "Factor momentum in the Chinese stock market",
        "authors": ["Tian Ma", "Cunfei Liao", "Fuwei Jiang"],
        "published_period": "2024-01",
        "journal": "Journal of Empirical Finance",
        "primary_url": ("https://www.sciencedirect.com/science/article/pii/" "S0927539823001251"),
        "doi_url": "https://doi.org/10.1016/j.jempfin.2023.101458",
        "source_scope": "China factor portfolios",
        "translation": (
            "Treat prior one-year returns of ten non-momentum factor "
            "portfolios as a future factor-timing idea, not as a new "
            "stock-level cross-sectional candidate."
        ),
        "evidence_role": "DIRECT_CHINA_FACTOR_TIMING_NOT_STOCK_SIGNAL",
        "local_data_status": ("BLOCKED_FACTOR_PORTFOLIO_LEVEL_INPUTS_AND_NOT_STOCK_SIGNAL"),
    },
    {
        "source_id": "zhang_ma_yang_fan_2024_change_salience",
        "idea": "china_change_in_salience",
        "title": (
            "The change in salience and the cross-section of stock returns: "
            "Empirical evidence from China A-shares"
        ),
        "authors": ["Ying Zhang", "Tian Ma", "Yang Yang", "Yue Fan"],
        "published_period": "2024",
        "journal": "Pacific-Basin Finance Journal",
        "primary_url": ("https://www.sciencedirect.com/science/article/pii/" "S0927538X24000702"),
        "doi_url": "https://doi.org/10.1016/j.pacfin.2024.102319",
        "source_scope": "China A shares",
        "translation": (
            "Preserve the direct China negative-return prediction as a "
            "research route, but do not implement until the exact change-in-"
            "salience aggregation and all source parameters are locked."
        ),
        "evidence_role": "DIRECT_CHINA_SUPPORT_BLOCKED_EXACT_CONSTRUCTION",
        "local_data_status": ("BLOCKED_EXACT_CS_AGGREGATION_AND_PARAMETERS_NOT_SOURCE_LOCKED"),
    },
)

_CANDIDATE_LITERATURE_ASSESSMENTS: tuple[dict[str, Any], ...] = (
    {
        "candidate_name": "cn_earnings_yield_ex_shell_30pct",
        "status": "DIRECT_CHINA_SUPPORT",
        "future_preregistration_eligible": True,
        "supporting_source_ids": ["liu_stambaugh_yuan_2019"],
        "adverse_source_ids": [],
        "locked_falsification_tests": [
            "must_add_value_after_size_and_sector_neutralization",
            "must_not_be_driven_by_smallest_30pct_shell_segment",
        ],
    },
    {
        "candidate_name": "cn_low_beta_252d",
        "status": "CONTROL_ONLY_CONFLICTING_CHINA_MECHANISM_EVIDENCE",
        "future_preregistration_eligible": False,
        "supporting_source_ids": [
            "frazzini_pedersen_2014",
            "chen_huang_qiu_2021",
        ],
        "adverse_source_ids": [
            "blitz_hanauer_van_vliet_2021",
            "zhao_lin_2022",
        ],
        "locked_falsification_tests": [
            "must_add_value_beyond_realized_volatility",
            "must_add_value_beyond_max_and_idiosyncratic_risk",
            "must_survive_size_and_sector_neutralization",
            "must_be_reported_by_disagreement_and_arbitrage_limit_state",
        ],
    },
    {
        "candidate_name": "cn_52_week_high_momentum_12m",
        "status": "DIRECT_CHINA_SUPPORT_WITH_EPU_REGIME_RISK",
        "future_preregistration_eligible": True,
        "supporting_source_ids": [
            "george_hwang_2004",
            "zhou_liu_guo_2021",
        ],
        "adverse_source_ids": ["zhou_liu_guo_2021"],
        "locked_falsification_tests": [
            "must_add_value_beyond_20_and_120_session_momentum",
            "must_add_value_beyond_20_session_range_position_momentum",
            "must_add_value_beyond_short_term_reversal",
            "must_survive_size_and_sector_neutralization",
            "must_report_low_moderate_and_high_China_EPU_states",
            "must_stop_if_high_EPU_state_is_not_available_as_strict_PIT_data",
            "must_not_claim_daily_rolling_translation_is_the_source_monthly_formation_schedule",
        ],
    },
    {
        "candidate_name": "cn_high_price_delay_d1_52w",
        "status": "DIRECT_CHINA_SUPPORT_CAUSAL_ROLLING_TRANSLATION",
        "future_preregistration_eligible": True,
        "supporting_source_ids": [
            "hou_moskowitz_2005",
            "qian_sun_yu_2017",
        ],
        "adverse_source_ids": [],
        "locked_falsification_tests": [
            "must_add_value_beyond_turnover",
            "must_add_value_beyond_amihud_illiquidity",
            "must_add_value_beyond_price_efficiency",
            "must_add_value_beyond_short_term_momentum_or_reversal",
            "must_survive_size_and_sector_neutralization",
            "must_report_illiquidity_uncertainty_and_attention_mechanisms_when_PIT_inputs_exist",
            "must_not_replace_exact_wednesday_weekly_D1_with_daily_proxy",
            "must_not_claim_the_rolling_monthly_translation_is_the_source_papers_annual_formation_schedule",
        ],
    },
    {
        "candidate_name": "cn_low_max_return_20d",
        "status": "DIRECT_CHINA_SUPPORT_WITH_REGIME_RISK",
        "future_preregistration_eligible": True,
        "supporting_source_ids": [
            "gao_han_xiong_2021",
            "zhao_lin_2022",
        ],
        "adverse_source_ids": [],
        "locked_falsification_tests": [
            "must_add_value_beyond_realized_and_downside_volatility",
            "must_add_value_beyond_short_term_momentum_or_reversal",
            "must_survive_size_and_sector_neutralization",
            "must_report_pre_and_post_short_sale_regimes_separately",
            "must_report_high_and_low_sentiment_regimes_separately",
        ],
    },
    {
        "candidate_name": "cn_low_total_skewness_20d",
        "status": "DIRECT_CHINA_SUPPORT_WITH_52_WEEK_ANCHOR_CONDITION",
        "future_preregistration_eligible": True,
        "supporting_source_ids": ["wang_wang_wu_2023"],
        "adverse_source_ids": ["chen_wu_zhu_2022"],
        "locked_falsification_tests": [
            "must_add_value_beyond_low_MAX",
            "must_add_value_beyond_distributional_tail_asymmetry",
            "must_add_value_beyond_realized_and_downside_volatility",
            "must_add_value_beyond_short_term_momentum_or_reversal",
            "must_survive_size_and_sector_neutralization",
            "must_report_far_and_near_52_week_high_states_separately",
            "must_report_arbitrage_risk_and_sentiment_states_when_PIT_inputs_exist",
            "must_not_be_substituted_for_distributional_tail_asymmetry",
        ],
    },
    {
        "candidate_name": "cn_low_market_adjusted_tail_asymmetry_252d",
        "status": "DIRECT_CHINA_SUPPORT_PROXY_DEFINITION",
        "future_preregistration_eligible": True,
        "supporting_source_ids": [
            "chen_wu_zhu_2022",
            "jiang_wu_zhou_zhu_2020",
        ],
        "adverse_source_ids": [
            "chen_wu_zhu_2022",
        ],
        "locked_falsification_tests": [
            "must_add_value_beyond_low_MAX",
            "must_add_value_beyond_realized_and_downside_volatility",
            "must_add_value_beyond_short_term_momentum_or_reversal",
            "must_survive_size_and_sector_neutralization",
            "must_not_be_described_as_conventional_skewness",
            "must_not_claim_exact_CH3_or_CH4_idiosyncratic_IE_replication",
        ],
    },
    {
        "candidate_name": "cn_quality_cash_low_leverage",
        "status": "GLOBAL_PROXY_ONLY",
        "future_preregistration_eligible": True,
        "supporting_source_ids": ["asness_frazzini_pedersen_2019"],
        "adverse_source_ids": [],
        "locked_falsification_tests": [
            "must_pass_formal_dedup_against_existing_quality_family",
            "must_not_claim_full_quality_minus_junk_definition",
        ],
    },
    {
        "candidate_name": "cn_same_month_seasonality_5y",
        "status": "DIRECT_CHINA_SUPPORT_WITH_OTHER_MONTH_REVERSAL",
        "future_preregistration_eligible": True,
        "supporting_source_ids": [
            "meng_du_shu_2024",
            "heston_sadka_2008",
        ],
        "adverse_source_ids": ["meng_du_shu_2024"],
        "locked_falsification_tests": [
            "same_calendar_month_is_the_only_primary_target",
            "other_month_reversal_must_be_reported_separately",
            "all_twelve_calendar_month_buckets_must_be_stable",
        ],
    },
    {
        "candidate_name": "cn_fip_continuous_direction_12m",
        "status": "FOUNDATIONAL_FIP_WITH_AGGREGATE_CHINA_RELEVANCE",
        "future_preregistration_eligible": True,
        "supporting_source_ids": [
            "da_gurun_warachka_2014",
            "zhang_chen_feng_2026",
        ],
        "adverse_source_ids": ["zhang_chen_feng_2026"],
        "locked_falsification_tests": [
            "must_add_value_beyond_20_and_120_session_momentum",
            "must_add_value_beyond_52_week_high_and_price_efficiency",
            "must_add_value_beyond_short_term_reversal",
            "must_survive_size_and_sector_neutralization",
            "must_report_past_winners_and_past_losers_separately",
            "must_report_zero_return_and_price_limit_sensitivity",
            "must_not_claim_the_single_score_is_the_source_double_sort",
            "must_not_claim_daily_session_translation_is_the_source_monthly_schedule",
            "must_not_claim_aggregate_China_evidence_is_cross_sectional_support",
        ],
    },
    {
        "candidate_name": "cn_low_left_tail_var1_250d",
        "status": "DIRECT_CHINA_SUPPORT_EXACT_VAR1_SIGNAL",
        "future_preregistration_eligible": True,
        "supporting_source_ids": [
            "atilgan_bali_demirtas_gunaydin_2020",
            "zhen_ruan_zhang_2020",
        ],
        "adverse_source_ids": [],
        "locked_falsification_tests": [
            "must_add_value_beyond_realized_and_downside_volatility",
            "must_add_value_beyond_low_MAX_total_skewness_and_tail_asymmetry",
            "must_add_value_beyond_20_and_120_session_momentum_or_reversal",
            "must_survive_size_and_sector_neutralization",
            "must_report_price_limit_suspension_and_relisting_sensitivity",
            "must_not_replace_primary_VaR1_with_VaR5_or_ES1_after_measurement",
            "must_not_claim_daily_translation_is_the_source_monthly_schedule",
        ],
    },
)


class FactorGovernanceLiteratureIncubatorV4Error(ValueError):
    """Raised when a research-only factor definition fails closed."""


def _error(message: str) -> FactorGovernanceLiteratureIncubatorV4Error:
    return FactorGovernanceLiteratureIncubatorV4Error(message)


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise _error(f"value is not canonical JSON: {exc}") from exc


def candidate_ontology_v4() -> dict[str, Any]:
    """Return the standalone v4 ontology for the research candidates."""

    return screening.build_primitive_ontology_v4(copy.deepcopy(list(_CANDIDATE_PRIMITIVES)))


def candidate_catalog_artifact_v4() -> dict[str, Any]:
    """Return an exact Factor v4-compatible research candidate catalog."""

    return screening.build_candidate_catalog_v4(
        ontology=candidate_ontology_v4(),
        candidates=copy.deepcopy(list(_CANDIDATE_DEFINITIONS)),
    )


def candidate_catalog_v4() -> list[dict[str, Any]]:
    """Return decorated v4 identities with no governance authority."""

    catalog = candidate_catalog_artifact_v4()
    by_name = {row["name"]: row for row in catalog["candidates"]}
    rows: list[dict[str, Any]] = []
    for order, name in enumerate(_CANDIDATE_ORDER, start=1):
        row = copy.deepcopy(by_name[name])
        row["order"] = order
        row["definition_identity_sha256"] = row["definition_sha256"]
        row["required_fields"] = copy.deepcopy(row["input_fields"])
        row["definition"] = row["expression"]
        row["source_id"] = _CANDIDATE_SOURCE_IDS[name]
        row["research_status"] = "IMPLEMENTABLE_RESEARCH_ONLY"
        row["authority"] = copy.deepcopy(AUTHORITY_FLAGS)
        row["side_effects"] = copy.deepcopy(SIDE_EFFECT_FLAGS)
        rows.append(row)
    return rows


def build_structural_audit_v4(
    *,
    comparison_ontology: Mapping[str, Any],
    comparison_catalog: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit exact definitions, primitives, slots, and families diagnostically."""

    normalized_ontology = screening.validate_primitive_ontology_v4(comparison_ontology)
    normalized_catalog = screening.validate_candidate_catalog_v4(
        comparison_catalog,
        ontology=normalized_ontology,
    )
    candidate_catalog = candidate_catalog_artifact_v4()
    comparison_rows = normalized_catalog["candidates"]
    results: list[dict[str, Any]] = []
    for candidate in candidate_catalog["candidates"]:
        candidate_primitives = set(candidate["primitive_ids"])
        overlaps: list[dict[str, Any]] = []
        for existing in comparison_rows:
            existing_primitives = set(existing["primitive_ids"])
            intersection = sorted(candidate_primitives.intersection(existing_primitives))
            if not intersection:
                continue
            union = candidate_primitives.union(existing_primitives)
            overlaps.append(
                {
                    "existing_factor_name": existing["name"],
                    "shared_primitive_ids": intersection,
                    "primitive_jaccard": (float(len(intersection) / len(union))),
                }
            )
        definition_collisions = sorted(
            row["name"]
            for row in comparison_rows
            if row["definition_sha256"] == candidate["definition_sha256"]
        )
        primitive_collisions = sorted(
            row["name"]
            for row in comparison_rows
            if row["primitive_ids"] == candidate["primitive_ids"]
        )
        results.append(
            {
                "candidate_name": candidate["name"],
                "candidate_definition_sha256": candidate["definition_sha256"],
                "candidate_primitive_ids": candidate["primitive_ids"],
                "definition_sha256_collision_names": definition_collisions,
                "exact_primitive_collision_names": primitive_collisions,
                "slot_collision_names": sorted(
                    row["name"] for row in comparison_rows if row["slot"] == candidate["slot"]
                ),
                "family_collision_names": sorted(
                    row["name"] for row in comparison_rows if row["family"] == candidate["family"]
                ),
                "primitive_overlap_rows": sorted(
                    overlaps,
                    key=lambda row: row["existing_factor_name"],
                ),
                "structural_collision_passed_diagnostic": bool(
                    not definition_collisions and not primitive_collisions
                ),
                "formal_dedup_evidence": False,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "factor-structural-dedup-diagnostic.v4",
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "candidate_ontology_sha256": candidate_ontology_v4()["semantic_sha256"],
        "candidate_catalog_sha256": candidate_catalog["semantic_sha256"],
        "comparison_ontology_sha256": normalized_ontology["semantic_sha256"],
        "comparison_catalog_sha256": normalized_catalog["semantic_sha256"],
        "comparison_factor_count": len(comparison_rows),
        "candidate_results": results,
        "formal_dedup_evidence": False,
        "authority": copy.deepcopy(AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
    }
    payload["semantic_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def build_protected_exact_five_audit_v4(
    *,
    protected_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit incubator identities against the immutable v4.4 exact-five set."""

    if isinstance(protected_candidates, (str, bytes)) or not isinstance(
        protected_candidates, Sequence
    ):
        raise _error("protected exact-five candidates must be a sequence")
    normalized: list[dict[str, str]] = []
    for row in protected_candidates:
        if not isinstance(row, Mapping):
            raise _error("protected exact-five candidate must be an object")
        raw_candidate = {
            "name": row.get("name"),
            "definition_identity_sha256": row.get("definition_identity_sha256"),
            "family": row.get("family"),
            "slot": row.get("slot"),
        }
        if any(type(value) is not str or not value for value in raw_candidate.values()):
            raise _error("protected exact-five identity fields must be non-empty strings")
        candidate: dict[str, str] = {key: cast(str, value) for key, value in raw_candidate.items()}
        identity = candidate["definition_identity_sha256"]
        if len(identity) != 64 or any(
            character not in "0123456789abcdef" for character in identity
        ):
            raise _error("protected exact-five definition identity must be lowercase SHA256")
        normalized.append(candidate)
    names = tuple(row["name"] for row in normalized)
    if names != PROTECTED_EXACT_FIVE_CANDIDATE_NAMES:
        raise _error("protected exact-five candidate names/order differ from the frozen set")
    for field in ("name", "definition_identity_sha256", "family", "slot"):
        if len({row[field] for row in normalized}) != len(normalized):
            raise _error(f"protected exact-five {field} values must be unique")
    incubator_rows = candidate_catalog_v4()
    namespace_collisions = sorted({row["name"] for row in incubator_rows}.intersection(names))
    if namespace_collisions:
        raise _error(
            "protected exact-five names collide with incubator candidates: "
            + ",".join(namespace_collisions)
        )
    results: list[dict[str, Any]] = []
    for candidate in incubator_rows:
        identity_collisions = sorted(
            row["name"]
            for row in normalized
            if row["definition_identity_sha256"] == candidate["definition_identity_sha256"]
        )
        slot_collisions = sorted(
            row["name"] for row in normalized if row["slot"] == candidate["slot"]
        )
        family_collisions = sorted(
            row["name"] for row in normalized if row["family"] == candidate["family"]
        )
        results.append(
            {
                "candidate_name": candidate["name"],
                "candidate_definition_identity_sha256": candidate["definition_identity_sha256"],
                "definition_identity_collision_names": identity_collisions,
                "slot_collision_names": slot_collisions,
                "family_collision_names": family_collisions,
                "protected_structural_collision_passed_diagnostic": bool(
                    not identity_collisions and not slot_collisions
                ),
                "formal_dedup_evidence": False,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "factor-protected-exact-five-audit.v4.4",
        "status": "RESEARCH_DIAGNOSTIC_ONLY",
        "protected_candidate_count": len(normalized),
        "protected_candidate_names": list(names),
        "protected_candidates": normalized,
        "candidate_results": results,
        "formal_dedup_evidence": False,
        "authority": copy.deepcopy(AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
    }
    payload["semantic_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def literature_idea_catalog_v4() -> list[dict[str, Any]]:
    """Return the complete source audit, including non-computable ideas."""

    return copy.deepcopy(list(LITERATURE_IDEAS))


def candidate_literature_assessments_v4() -> list[dict[str, Any]]:
    """Return source-supported and adverse-evidence routing constraints."""

    candidate_names = set(_CANDIDATE_ORDER)
    source_ids = {row["source_id"] for row in LITERATURE_IDEAS}
    assessment_names = {row["candidate_name"] for row in _CANDIDATE_LITERATURE_ASSESSMENTS}
    if assessment_names != candidate_names:
        raise _error("literature assessments must cover the exact candidate set")
    for row in _CANDIDATE_LITERATURE_ASSESSMENTS:
        referenced = set(row["supporting_source_ids"]) | set(row["adverse_source_ids"])
        if not referenced.issubset(source_ids):
            raise _error("literature assessment references an unknown source")
    return copy.deepcopy(list(_CANDIDATE_LITERATURE_ASSESSMENTS))


def low_max_future_preregistration_policy_v4() -> dict[str, Any]:
    """Return the frozen draft policy for a later, genuinely future cycle."""

    candidate = next(
        row for row in candidate_catalog_v4() if row["name"] == "cn_low_max_return_20d"
    )
    literature = next(
        row
        for row in candidate_literature_assessments_v4()
        if row["candidate_name"] == candidate["name"]
    )
    payload: dict[str, Any] = {
        "schema_version": "factor-governance-literature-preregistration-policy.v4",
        "protocol_version": PROTOCOL_VERSION,
        "incubator_version": INCUBATOR_VERSION,
        "status": "DRAFT_FUTURE_PREREGISTRATION_POLICY",
        "formal_preregistration_created": False,
        "candidate": {
            "name": candidate["name"],
            "definition_identity_sha256": candidate["definition_identity_sha256"],
            "family": candidate["family"],
            "slot": candidate["slot"],
            "initial_weight": 0,
        },
        "selection_provenance": {
            "selection_date": LOW_MAX_SELECTION_DATE,
            "external_published_outcomes_informed_idea": True,
            "local_labels_or_forward_returns_used": False,
            "historical_source_statistics_inherited_as_formal_evidence": False,
            "publication_time_authority": "NOT_YET_CREATED",
            "required_strict_cutoff_relation": ("cutoff_must_be_strictly_after_selection_date"),
        },
        "primary_hypothesis": {
            "signal": (
                "cross_sectional_rank_of_negative_maximum_daily_return_over_"
                "prior_20_open_sessions"
            ),
            "null": (
                "the_signal_has_no_positive_incremental_next_20_open_session_"
                "return_after_required_controls"
            ),
            "alternative": (
                "the_signal_has_positive_incremental_next_20_open_session_"
                "return_after_required_controls"
            ),
            "signal_direction": "higher_is_preferred",
            "primary_rebalance": ("last_strict_open_session_of_each_closed_natural_month"),
            "execution_lag_open_sessions": 1,
            "forward_horizon_open_sessions": 20,
            "primary_target": "next_20_open_session_total_return",
            "primary_group_test": (
                "long_high_signal_low_MAX_decile_minus_low_signal_high_MAX_decile"
            ),
            "long_only_leg_required_positive": True,
        },
        "future_sample_contract": {
            "universe": "strict_parquet_pit_full_a_after_canonical_eligibility",
            "embargo_strictly_later_open_sessions": 30,
            "measurement_starts_on_later_open_session": 31,
            "minimum_post_embargo_open_sessions": 240,
            "minimum_distinct_closed_month_ends": 12,
            "publication_day_excluded": True,
            "pre_publication_observations_are_diagnostic_only": True,
        },
        "multiple_testing_contract": {
            "method": "benjamini_hochberg_by_family",
            "family": "max_daily_return",
            "maximum_q_value": 0.10,
            "full_frozen_catalog_denominator_required": True,
        },
        "dedup_contract": {
            "metric": "median_monthly_cross_sectional_abs_spearman",
            "maximum_allowed": 0.70,
            "minimum_common_symbol_count": 20,
            "minimum_closed_month_count": 3,
            "required_existing_factor_names": [
                "pv_downside_volatility_60d",
                "pv_momentum_20d",
                "pv_volatility_penalty_60d",
            ],
            "required_incubator_candidate_names": [
                "cn_low_market_adjusted_tail_asymmetry_252d",
                "cn_low_total_skewness_20d",
            ],
            "required_protected_candidate_names": list(PROTECTED_EXACT_FIVE_CANDIDATE_NAMES),
            "formal_and_high_correlation_dedup_required": True,
        },
        "factor_v4_gate_contract": {
            "required_gate_ids": list(range(1, 9)),
            "minimum_coverage_rate": 0.60,
            "maximum_nan_rate": 0.40,
            "minimum_watch_icir": 0.30,
            "minimum_production_candidate_icir": 0.50,
            "minimum_positive_ic_ratio": 0.52,
            "minimum_production_positive_ic_ratio": 0.55,
            "minimum_neutralized_icir": 0.20,
            "minimum_oos_positive_ratio": 0.55,
            "walk_forward_minimum_purge_days": 30,
            "walk_forward_exact_embargo_days": 30,
            "canonical_abcd_replay_required": True,
        },
        "locked_adverse_and_robustness_tests": literature["locked_falsification_tests"],
        "parameter_robustness_contract": {
            "primary_window_open_sessions": 20,
            "secondary_windows_open_sessions": [15, 25],
            "secondary_windows_cannot_replace_primary_after_measurement": True,
            "a_different_winner_requires_a_new_governance_cycle": True,
        },
        "stop_rules": [
            "stop_on_any_data_safety_failure",
            "stop_on_structural_definition_or_primitive_collision",
            "stop_if_any_required_dedup_correlation_exceeds_0_70",
            "stop_if_family_bh_q_value_exceeds_0_10",
            "stop_if_size_and_sector_neutralized_icir_is_below_0_20",
            "stop_if_cost_adjusted_return_is_not_positive",
            "stop_if_purged_embargoed_walk_forward_gate_fails",
            "stop_if_canonical_abcd_incremental_replay_gate_fails",
        ],
        "authority": copy.deepcopy(AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
    }
    payload["semantic_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def low_total_skewness_future_preregistration_policy_v4() -> dict[str, Any]:
    """Return the frozen draft policy for conventional total-return skewness."""

    candidate = next(
        row for row in candidate_catalog_v4() if row["name"] == "cn_low_total_skewness_20d"
    )
    literature = next(
        row
        for row in candidate_literature_assessments_v4()
        if row["candidate_name"] == candidate["name"]
    )
    payload: dict[str, Any] = {
        "schema_version": "factor-governance-literature-preregistration-policy.v4",
        "protocol_version": PROTOCOL_VERSION,
        "incubator_version": INCUBATOR_VERSION,
        "status": "DRAFT_FUTURE_PREREGISTRATION_POLICY",
        "formal_preregistration_created": False,
        "candidate": {
            "name": candidate["name"],
            "definition_identity_sha256": candidate["definition_identity_sha256"],
            "family": candidate["family"],
            "slot": candidate["slot"],
            "initial_weight": 0,
        },
        "selection_provenance": {
            "selection_date": LOW_TOTAL_SKEW_SELECTION_DATE,
            "external_published_outcomes_informed_idea": True,
            "local_labels_or_forward_returns_used": False,
            "historical_source_statistics_inherited_as_formal_evidence": False,
            "publication_time_authority": "NOT_YET_CREATED",
            "required_strict_cutoff_relation": ("cutoff_must_be_strictly_after_selection_date"),
            "conventional_total_skewness_definition_acknowledged": True,
            "distributional_tail_asymmetry_replication_claimed": False,
        },
        "primary_hypothesis": {
            "signal": (
                "cross_sectional_rank_of_negative_unbiased_sample_skewness_of_"
                "daily_adjusted_total_returns_over_prior_20_open_sessions_with_"
                "at_least_15_observations"
            ),
            "null": (
                "the_signal_has_no_positive_incremental_next_20_open_session_"
                "return_after_required_controls"
            ),
            "alternative": (
                "the_signal_has_positive_incremental_next_20_open_session_"
                "return_after_required_controls"
            ),
            "signal_direction": "higher_is_preferred",
            "primary_rebalance": ("last_strict_open_session_of_each_closed_natural_month"),
            "execution_lag_open_sessions": 1,
            "forward_horizon_open_sessions": 20,
            "primary_target": "next_20_open_session_total_return",
            "primary_group_test": (
                "long_high_signal_low_total_skewness_decile_minus_"
                "low_signal_high_total_skewness_decile"
            ),
            "long_only_leg_required_positive": True,
        },
        "future_sample_contract": {
            "universe": "strict_parquet_pit_full_a_after_canonical_eligibility",
            "embargo_strictly_later_open_sessions": 30,
            "measurement_starts_on_later_open_session": 31,
            "minimum_post_embargo_open_sessions": 240,
            "minimum_distinct_closed_month_ends": 12,
            "publication_day_excluded": True,
            "pre_publication_observations_are_diagnostic_only": True,
        },
        "multiple_testing_contract": {
            "method": "benjamini_hochberg_by_family",
            "family": "total_return_skewness",
            "maximum_q_value": 0.10,
            "full_frozen_catalog_denominator_required": True,
        },
        "dedup_contract": {
            "metric": "median_monthly_cross_sectional_abs_spearman",
            "maximum_allowed": 0.70,
            "minimum_common_symbol_count": 20,
            "minimum_closed_month_count": 3,
            "required_existing_factor_names": [
                "pv_downside_volatility_60d",
                "pv_momentum_20d",
                "pv_volatility_penalty_60d",
            ],
            "required_incubator_candidate_names": [
                "cn_low_market_adjusted_tail_asymmetry_252d",
                "cn_low_max_return_20d",
            ],
            "required_protected_candidate_names": list(PROTECTED_EXACT_FIVE_CANDIDATE_NAMES),
            "formal_and_high_correlation_dedup_required": True,
        },
        "anchor_state_contract": {
            "metric": (
                "one_minus_adj_close_t_divided_by_max_adj_close_from_" "t_minus_251_through_t"
            ),
            "partition": ("cross_sectional_terciles_on_each_primary_rebalance_session"),
            "far_below_52_week_high_state": "largest_distance_tercile",
            "near_52_week_high_state": "smallest_distance_tercile",
            "required_interaction_direction": (
                "low_skewness_spread_far_below_must_exceed_near_52_week_high"
            ),
            "state_results_cannot_replace_primary_unconditional_test": True,
        },
        "conditional_mechanism_contract": {
            "arbitrage_risk_and_sentiment_reporting": "MANDATORY_IF_PIT_INPUTS_EXIST",
            "missing_inputs_cannot_be_replaced_with_inferred_or_non_PIT_proxies": True,
            "conditional_results_cannot_replace_primary_unconditional_test": True,
        },
        "factor_v4_gate_contract": {
            "required_gate_ids": list(range(1, 9)),
            "minimum_coverage_rate": 0.60,
            "maximum_nan_rate": 0.40,
            "minimum_watch_icir": 0.30,
            "minimum_production_candidate_icir": 0.50,
            "minimum_positive_ic_ratio": 0.52,
            "minimum_production_positive_ic_ratio": 0.55,
            "minimum_neutralized_icir": 0.20,
            "minimum_oos_positive_ratio": 0.55,
            "walk_forward_minimum_purge_days": 30,
            "walk_forward_exact_embargo_days": 30,
            "canonical_abcd_replay_required": True,
        },
        "locked_adverse_and_robustness_tests": literature["locked_falsification_tests"],
        "parameter_robustness_contract": {
            "primary_window_open_sessions": 20,
            "primary_minimum_observations": 15,
            "secondary_windows_open_sessions": [15, 25],
            "secondary_minimum_observations": 15,
            "secondary_windows_cannot_replace_primary_after_measurement": True,
            "a_different_winner_requires_a_new_governance_cycle": True,
        },
        "stop_rules": [
            "stop_on_any_data_safety_failure",
            "stop_on_structural_definition_or_primitive_collision",
            "stop_if_any_required_dedup_correlation_exceeds_0_70",
            "stop_if_family_bh_q_value_exceeds_0_10",
            "stop_if_incremental_value_beyond_low_MAX_is_not_positive",
            "stop_if_incremental_value_beyond_tail_asymmetry_is_not_positive",
            "stop_if_52_week_high_interaction_direction_is_reversed",
            "stop_if_size_and_sector_neutralized_icir_is_below_0_20",
            "stop_if_cost_adjusted_return_is_not_positive",
            "stop_if_purged_embargoed_walk_forward_gate_fails",
            "stop_if_canonical_abcd_incremental_replay_gate_fails",
        ],
        "authority": copy.deepcopy(AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
    }
    payload["semantic_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def tail_asymmetry_future_preregistration_policy_v4() -> dict[str, Any]:
    """Return the frozen draft policy for the market-adjusted tail proxy."""

    candidate = next(
        row
        for row in candidate_catalog_v4()
        if row["name"] == "cn_low_market_adjusted_tail_asymmetry_252d"
    )
    literature = next(
        row
        for row in candidate_literature_assessments_v4()
        if row["candidate_name"] == candidate["name"]
    )
    payload: dict[str, Any] = {
        "schema_version": "factor-governance-literature-preregistration-policy.v4",
        "protocol_version": PROTOCOL_VERSION,
        "incubator_version": INCUBATOR_VERSION,
        "status": "DRAFT_FUTURE_PREREGISTRATION_POLICY",
        "formal_preregistration_created": False,
        "candidate": {
            "name": candidate["name"],
            "definition_identity_sha256": candidate["definition_identity_sha256"],
            "family": candidate["family"],
            "slot": candidate["slot"],
            "initial_weight": 0,
        },
        "selection_provenance": {
            "selection_date": TAIL_ASYMMETRY_SELECTION_DATE,
            "external_published_outcomes_informed_idea": True,
            "local_labels_or_forward_returns_used": False,
            "historical_source_statistics_inherited_as_formal_evidence": False,
            "publication_time_authority": "NOT_YET_CREATED",
            "required_strict_cutoff_relation": ("cutoff_must_be_strictly_after_selection_date"),
            "proxy_definition_acknowledged": True,
            "exact_CH3_or_CH4_idiosyncratic_IE_replication_claimed": False,
        },
        "primary_hypothesis": {
            "signal": (
                "cross_sectional_rank_of_negative_excess_tail_probability_"
                "P_z_gt_1_minus_P_z_lt_minus_1_over_prior_252_open_sessions_"
                "using_PIT_equal_weight_market_adjusted_daily_returns"
            ),
            "null": (
                "the_signal_has_no_positive_incremental_next_20_open_session_"
                "return_after_required_controls"
            ),
            "alternative": (
                "the_signal_has_positive_incremental_next_20_open_session_"
                "return_after_required_controls"
            ),
            "signal_direction": "higher_is_preferred",
            "primary_rebalance": ("last_strict_open_session_of_each_closed_natural_month"),
            "execution_lag_open_sessions": 1,
            "forward_horizon_open_sessions": 20,
            "primary_target": "next_20_open_session_total_return",
            "primary_group_test": (
                "long_high_signal_low_upside_tail_asymmetry_decile_minus_"
                "low_signal_high_upside_tail_asymmetry_decile"
            ),
            "long_only_leg_required_positive": True,
        },
        "future_sample_contract": {
            "universe": "strict_parquet_pit_full_a_after_canonical_eligibility",
            "embargo_strictly_later_open_sessions": 30,
            "measurement_starts_on_later_open_session": 31,
            "minimum_post_embargo_open_sessions": 240,
            "minimum_distinct_closed_month_ends": 12,
            "publication_day_excluded": True,
            "pre_publication_observations_are_diagnostic_only": True,
        },
        "multiple_testing_contract": {
            "method": "benjamini_hochberg_by_family",
            "family": "market_adjusted_tail_asymmetry",
            "maximum_q_value": 0.10,
            "full_frozen_catalog_denominator_required": True,
        },
        "dedup_contract": {
            "metric": "median_monthly_cross_sectional_abs_spearman",
            "maximum_allowed": 0.70,
            "minimum_common_symbol_count": 20,
            "minimum_closed_month_count": 3,
            "required_existing_factor_names": [
                "pv_downside_volatility_60d",
                "pv_momentum_20d",
                "pv_volatility_penalty_60d",
            ],
            "required_incubator_candidate_names": [
                "cn_low_max_return_20d",
                "cn_low_total_skewness_20d",
            ],
            "required_protected_candidate_names": list(PROTECTED_EXACT_FIVE_CANDIDATE_NAMES),
            "formal_and_high_correlation_dedup_required": True,
        },
        "factor_v4_gate_contract": {
            "required_gate_ids": list(range(1, 9)),
            "minimum_coverage_rate": 0.60,
            "maximum_nan_rate": 0.40,
            "minimum_watch_icir": 0.30,
            "minimum_production_candidate_icir": 0.50,
            "minimum_positive_ic_ratio": 0.52,
            "minimum_production_positive_ic_ratio": 0.55,
            "minimum_neutralized_icir": 0.20,
            "minimum_oos_positive_ratio": 0.55,
            "walk_forward_minimum_purge_days": 30,
            "walk_forward_exact_embargo_days": 30,
            "canonical_abcd_replay_required": True,
        },
        "locked_adverse_and_robustness_tests": literature["locked_falsification_tests"],
        "parameter_robustness_contract": {
            "primary_window_open_sessions": 252,
            "secondary_windows_open_sessions": [126, 504],
            "primary_sigma_threshold": 1.0,
            "secondary_sigma_thresholds": [0.75, 1.25],
            "secondary_parameters_cannot_replace_primary_after_measurement": True,
            "a_different_winner_requires_a_new_governance_cycle": True,
        },
        "stop_rules": [
            "stop_on_any_data_safety_failure",
            "stop_on_structural_definition_or_primitive_collision",
            "stop_if_any_required_dedup_correlation_exceeds_0_70",
            "stop_if_family_bh_q_value_exceeds_0_10",
            "stop_if_incremental_value_beyond_low_MAX_is_not_positive",
            "stop_if_size_and_sector_neutralized_icir_is_below_0_20",
            "stop_if_cost_adjusted_return_is_not_positive",
            "stop_if_purged_embargoed_walk_forward_gate_fails",
            "stop_if_canonical_abcd_incremental_replay_gate_fails",
        ],
        "authority": copy.deepcopy(AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
    }
    payload["semantic_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def fip_future_preregistration_policy_v4() -> dict[str, Any]:
    """Return the frozen draft policy for the FIP single-score translation."""

    candidate = next(
        row for row in candidate_catalog_v4() if row["name"] == "cn_fip_continuous_direction_12m"
    )
    literature = next(
        row
        for row in candidate_literature_assessments_v4()
        if row["candidate_name"] == candidate["name"]
    )
    payload: dict[str, Any] = {
        "schema_version": "factor-governance-literature-preregistration-policy.v4",
        "protocol_version": PROTOCOL_VERSION,
        "incubator_version": INCUBATOR_VERSION,
        "status": "DRAFT_FUTURE_PREREGISTRATION_POLICY",
        "formal_preregistration_created": False,
        "candidate": {
            "name": candidate["name"],
            "definition_identity_sha256": candidate["definition_identity_sha256"],
            "family": candidate["family"],
            "slot": candidate["slot"],
            "initial_weight": 0,
        },
        "selection_provenance": {
            "selection_date": FIP_SELECTION_DATE,
            "external_published_outcomes_informed_idea": True,
            "local_labels_or_forward_returns_used": False,
            "historical_source_statistics_inherited_as_formal_evidence": False,
            "publication_time_authority": "NOT_YET_CREATED",
            "required_strict_cutoff_relation": ("cutoff_must_be_strictly_after_selection_date"),
            "foundational_source_double_sort_acknowledged": True,
            "single_score_translation_acknowledged": True,
            "China_aggregate_evidence_treated_as_cross_sectional_support": False,
        },
        "primary_hypothesis": {
            "signal": (
                "cross_sectional_rank_of_sign_PRET_times_one_minus_ID_over_two_"
                "using_231_formation_sessions_after_skipping_21_sessions"
            ),
            "null": (
                "the_signal_has_no_positive_incremental_next_20_open_session_"
                "return_after_required_controls"
            ),
            "alternative": (
                "the_signal_has_positive_incremental_next_20_open_session_"
                "return_after_required_controls"
            ),
            "signal_direction": "higher_is_preferred",
            "primary_rebalance": ("last_strict_open_session_of_each_closed_natural_month"),
            "execution_lag_open_sessions": 1,
            "forward_horizon_open_sessions": 20,
            "primary_target": "next_20_open_session_total_return",
            "primary_group_test": (
                "long_high_continuous_positive_direction_decile_minus_"
                "low_continuous_negative_direction_decile"
            ),
            "long_only_leg_required_positive": True,
        },
        "future_sample_contract": {
            "universe": "strict_parquet_pit_full_a_after_canonical_eligibility",
            "embargo_strictly_later_open_sessions": 30,
            "measurement_starts_on_later_open_session": 31,
            "minimum_post_embargo_open_sessions": 240,
            "minimum_distinct_closed_month_ends": 12,
            "publication_day_excluded": True,
            "pre_publication_observations_are_diagnostic_only": True,
        },
        "multiple_testing_contract": {
            "method": "benjamini_hochberg_by_family",
            "family": "fip_continuous_direction",
            "maximum_q_value": 0.10,
            "full_frozen_catalog_denominator_required": True,
        },
        "dedup_contract": {
            "metric": "median_monthly_cross_sectional_abs_spearman",
            "maximum_allowed": 0.70,
            "minimum_common_symbol_count": 20,
            "minimum_closed_month_count": 3,
            "required_existing_factor_names": [
                "pv_momentum_20d",
                "pv_momentum_120d",
                "pv_price_efficiency_60d",
                "pv_short_reversal_20d",
            ],
            "required_incubator_candidate_names": [
                "cn_52_week_high_momentum_12m",
            ],
            "required_protected_candidate_names": list(PROTECTED_EXACT_FIVE_CANDIDATE_NAMES),
            "formal_and_high_correlation_dedup_required": True,
        },
        "factor_v4_gate_contract": {
            "required_gate_ids": list(range(1, 9)),
            "minimum_coverage_rate": 0.60,
            "maximum_nan_rate": 0.40,
            "minimum_watch_icir": 0.30,
            "minimum_production_candidate_icir": 0.50,
            "minimum_positive_ic_ratio": 0.52,
            "minimum_production_positive_ic_ratio": 0.55,
            "minimum_neutralized_icir": 0.20,
            "minimum_oos_positive_ratio": 0.55,
            "walk_forward_minimum_purge_days": 30,
            "walk_forward_exact_embargo_days": 30,
            "canonical_abcd_replay_required": True,
        },
        "source_translation_contract": {
            "source_ID_definition": (
                "sign_PRET_times_fraction_negative_days_minus_fraction_positive_days"
            ),
            "source_monthly_double_sort_replication_claimed": False,
            "daily_translation_used_for_governance_measurement": True,
            "closed_month_end_measurement_required": True,
            "past_winner_and_past_loser_results_required_separately": True,
        },
        "microstructure_robustness_contract": {
            "zero_return_day_denominator": "all_finite_formation_return_days",
            "zero_return_sensitivity_required": True,
            "price_limit_sensitivity_required": True,
            "suspension_and_relisting_sensitivity_required": True,
        },
        "locked_adverse_and_robustness_tests": literature["locked_falsification_tests"],
        "parameter_robustness_contract": {
            "primary_formation_sessions": FIP_FORMATION_SESSIONS,
            "primary_skip_sessions": FIP_SKIP_SESSIONS,
            "secondary_formation_sessions": [210, 252],
            "secondary_skip_sessions": [20, 22],
            "secondary_parameters_cannot_replace_primary_after_measurement": True,
            "a_different_winner_requires_a_new_governance_cycle": True,
        },
        "stop_rules": [
            "stop_on_any_data_safety_failure",
            "stop_on_structural_definition_or_primitive_collision",
            "stop_if_any_required_dedup_correlation_exceeds_0_70",
            "stop_if_family_bh_q_value_exceeds_0_10",
            "stop_if_incremental_value_beyond_120_session_momentum_is_not_positive",
            "stop_if_past_winner_or_past_loser_direction_contradicts_the_hypothesis",
            "stop_if_size_and_sector_neutralized_icir_is_below_0_20",
            "stop_if_cost_adjusted_return_is_not_positive",
            "stop_if_purged_embargoed_walk_forward_gate_fails",
            "stop_if_canonical_abcd_incremental_replay_gate_fails",
        ],
        "authority": copy.deepcopy(AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
    }
    payload["semantic_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def left_tail_var1_future_preregistration_policy_v4() -> dict[str, Any]:
    """Return the frozen draft policy for the exact China VaR1 signal."""

    candidate = next(
        row for row in candidate_catalog_v4() if row["name"] == "cn_low_left_tail_var1_250d"
    )
    literature = next(
        row
        for row in candidate_literature_assessments_v4()
        if row["candidate_name"] == candidate["name"]
    )
    payload: dict[str, Any] = {
        "schema_version": "factor-governance-literature-preregistration-policy.v4",
        "protocol_version": PROTOCOL_VERSION,
        "incubator_version": INCUBATOR_VERSION,
        "status": "DRAFT_FUTURE_PREREGISTRATION_POLICY",
        "formal_preregistration_created": False,
        "candidate": {
            "name": candidate["name"],
            "definition_identity_sha256": candidate["definition_identity_sha256"],
            "family": candidate["family"],
            "slot": candidate["slot"],
            "initial_weight": 0,
        },
        "selection_provenance": {
            "selection_date": LEFT_TAIL_VAR_SELECTION_DATE,
            "external_published_outcomes_informed_idea": True,
            "local_labels_or_forward_returns_used": False,
            "historical_source_statistics_inherited_as_formal_evidence": False,
            "publication_time_authority": "NOT_YET_CREATED",
            "required_strict_cutoff_relation": ("cutoff_must_be_strictly_after_selection_date"),
            "direct_China_source_definition_acknowledged": True,
            "daily_signal_translation_acknowledged": True,
        },
        "primary_hypothesis": {
            "signal": (
                "cross_sectional_rank_of_negative_VaR1_where_VaR1_is_negative_"
                "first_percentile_of_daily_returns_over_prior_250_open_"
                "sessions_with_at_least_200_observations"
            ),
            "null": (
                "the_signal_has_no_positive_incremental_next_20_open_session_"
                "return_after_required_controls"
            ),
            "alternative": (
                "the_signal_has_positive_incremental_next_20_open_session_"
                "return_after_required_controls"
            ),
            "signal_direction": "higher_is_preferred",
            "primary_rebalance": ("last_strict_open_session_of_each_closed_natural_month"),
            "execution_lag_open_sessions": 1,
            "forward_horizon_open_sessions": 20,
            "primary_target": "next_20_open_session_total_return",
            "primary_group_test": (
                "long_high_signal_low_VaR1_decile_minus_low_signal_high_VaR1_decile"
            ),
            "long_only_leg_required_positive": True,
        },
        "future_sample_contract": {
            "universe": "strict_parquet_pit_full_a_after_canonical_eligibility",
            "embargo_strictly_later_open_sessions": 30,
            "measurement_starts_on_later_open_session": 31,
            "minimum_post_embargo_open_sessions": 240,
            "minimum_distinct_closed_month_ends": 12,
            "publication_day_excluded": True,
            "pre_publication_observations_are_diagnostic_only": True,
        },
        "multiple_testing_contract": {
            "method": "benjamini_hochberg_by_family",
            "family": "left_tail_var1",
            "maximum_q_value": 0.10,
            "full_frozen_catalog_denominator_required": True,
        },
        "dedup_contract": {
            "metric": "median_monthly_cross_sectional_abs_spearman",
            "maximum_allowed": 0.70,
            "minimum_common_symbol_count": 20,
            "minimum_closed_month_count": 3,
            "required_existing_factor_names": [
                "pv_downside_volatility_60d",
                "pv_momentum_20d",
                "pv_momentum_120d",
                "pv_short_reversal_20d",
                "pv_volatility_penalty_60d",
            ],
            "required_incubator_candidate_names": [
                "cn_low_market_adjusted_tail_asymmetry_252d",
                "cn_low_max_return_20d",
                "cn_low_total_skewness_20d",
            ],
            "required_protected_candidate_names": list(PROTECTED_EXACT_FIVE_CANDIDATE_NAMES),
            "formal_and_high_correlation_dedup_required": True,
        },
        "factor_v4_gate_contract": {
            "required_gate_ids": list(range(1, 9)),
            "minimum_coverage_rate": 0.60,
            "maximum_nan_rate": 0.40,
            "minimum_watch_icir": 0.30,
            "minimum_production_candidate_icir": 0.50,
            "minimum_positive_ic_ratio": 0.52,
            "minimum_production_positive_ic_ratio": 0.55,
            "minimum_neutralized_icir": 0.20,
            "minimum_oos_positive_ratio": 0.55,
            "walk_forward_minimum_purge_days": 30,
            "walk_forward_exact_embargo_days": 30,
            "canonical_abcd_replay_required": True,
        },
        "source_translation_contract": {
            "VaR1_definition": ("negative_one_times_first_percentile_of_daily_returns"),
            "primary_window_open_sessions": LEFT_TAIL_VAR_WINDOW,
            "primary_minimum_observations": LEFT_TAIL_VAR_MIN_PERIODS,
            "primary_quantile": LEFT_TAIL_VAR_QUANTILE,
            "source_monthly_measurement_acknowledged": True,
            "daily_translation_used_for_governance_measurement": True,
            "closed_month_end_measurement_required": True,
        },
        "microstructure_robustness_contract": {
            "price_limit_sensitivity_required": True,
            "suspension_and_relisting_sensitivity_required": True,
            "nontrading_returns_must_not_be_synthetically_imputed": True,
        },
        "locked_adverse_and_robustness_tests": literature["locked_falsification_tests"],
        "parameter_robustness_contract": {
            "primary_window_open_sessions": LEFT_TAIL_VAR_WINDOW,
            "primary_minimum_observations": LEFT_TAIL_VAR_MIN_PERIODS,
            "primary_quantile": LEFT_TAIL_VAR_QUANTILE,
            "secondary_risk_statistics": ["VaR5", "ES1"],
            "secondary_statistics_cannot_replace_primary_after_measurement": True,
            "a_different_winner_requires_a_new_governance_cycle": True,
        },
        "stop_rules": [
            "stop_on_any_data_safety_failure",
            "stop_on_structural_definition_or_primitive_collision",
            "stop_if_any_required_dedup_correlation_exceeds_0_70",
            "stop_if_family_bh_q_value_exceeds_0_10",
            "stop_if_incremental_value_beyond_downside_volatility_is_not_positive",
            "stop_if_incremental_value_beyond_low_MAX_or_tail_asymmetry_is_not_positive",
            "stop_if_size_and_sector_neutralized_icir_is_below_0_20",
            "stop_if_cost_adjusted_return_is_not_positive",
            "stop_if_purged_embargoed_walk_forward_gate_fails",
            "stop_if_canonical_abcd_incremental_replay_gate_fails",
        ],
        "authority": copy.deepcopy(AUTHORITY_FLAGS),
        "side_effects": copy.deepcopy(SIDE_EFFECT_FLAGS),
    }
    payload["semantic_sha256"] = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return payload


def _validate_axes(frame: pd.DataFrame, *, label: str) -> None:
    if type(frame) is not pd.DataFrame or frame.empty:
        raise _error(f"{label} must be a non-empty pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise _error(f"{label} date axis must be a DatetimeIndex")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise _error(f"{label} date axis must be strictly ordered and unique")
    if (
        not frame.columns.is_unique
        or not frame.columns.is_monotonic_increasing
        or any(type(value) is not str or not value for value in frame.columns)
    ):
        raise _error(f"{label} symbol axis must be sorted, unique strings")


def _normalize_inputs(
    *,
    required_fields: tuple[str, ...],
    inputs: Mapping[str, pd.DataFrame],
    pit_mask: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    if not isinstance(inputs, Mapping) or set(inputs) != set(required_fields):
        raise _error("inputs must contain the exact required fields: " + ",".join(required_fields))
    _validate_axes(pit_mask, label="PIT mask")
    if any(dtype != bool for dtype in pit_mask.dtypes):
        raise _error("PIT mask must be strictly boolean")
    normalized: dict[str, pd.DataFrame] = {}
    for field in required_fields:
        frame = inputs[field]
        _validate_axes(frame, label=field)
        if not frame.index.equals(pit_mask.index) or not frame.columns.equals(pit_mask.columns):
            raise _error(f"{field} axes differ from the PIT mask")
        try:
            numeric = frame.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise _error(f"{field} must contain numeric values") from exc
        normalized[field] = numeric.replace([np.inf, -np.inf], np.nan).where(pit_mask)
    return normalized, pit_mask.copy()


def _cross_sectional_rank(values: pd.DataFrame) -> pd.DataFrame:
    return values.rank(axis=1, method="average", pct=True, na_option="keep")


def earnings_yield_ex_shell_v4(
    *,
    pe: pd.DataFrame,
    total_mv: pd.DataFrame,
    pit_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Return positive E/P ranks after excluding the smallest 30% each day."""

    values, mask = _normalize_inputs(
        required_fields=("pe", "total_mv"),
        inputs={"pe": pe, "total_mv": total_mv},
        pit_mask=pit_mask,
    )
    valid_cap = values["total_mv"].where(values["total_mv"] > 0.0)
    cap_percentile = _cross_sectional_rank(valid_cap)
    eligible = mask & values["pe"].gt(0.0) & cap_percentile.gt(SHELL_EXCLUSION_FRACTION)
    earnings_yield = values["pe"].where(eligible).rdiv(1.0)
    return _cross_sectional_rank(earnings_yield).where(eligible)


def low_beta_v4(
    *,
    adj_close: pd.DataFrame,
    pit_mask: pd.DataFrame,
    window: int = LOW_BETA_WINDOW,
    min_periods: int = LOW_BETA_MIN_PERIODS,
) -> pd.DataFrame:
    """Return negative rolling beta to the contemporaneous PIT equal-weight market."""

    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise _error("low-beta window must be an integer of at least 2")
    if (
        isinstance(min_periods, bool)
        or not isinstance(min_periods, int)
        or min_periods < 2
        or min_periods > window
    ):
        raise _error("low-beta min_periods must be within [2, window]")
    values, mask = _normalize_inputs(
        required_fields=("adj_close",),
        inputs={"adj_close": adj_close},
        pit_mask=pit_mask,
    )
    close = values["adj_close"].where(values["adj_close"] > 0.0)
    returns = close.pct_change(fill_method=None).where(mask)
    market_return = returns.mean(axis=1, skipna=True)
    covariance = returns.rolling(window, min_periods=min_periods).cov(market_return)
    market_variance = market_return.rolling(window, min_periods=min_periods).var(ddof=1)
    beta = covariance.div(market_variance.replace(0.0, np.nan), axis=0)
    return (-beta).replace([np.inf, -np.inf], np.nan).where(mask)


def high_52_week_momentum_v4(
    *,
    adj_close: pd.DataFrame,
    pit_mask: pd.DataFrame,
    calendar_days: int = HIGH_52_WEEK_CALENDAR_DAYS,
    min_periods: int = HIGH_52_WEEK_MIN_PERIODS,
) -> pd.DataFrame:
    """Return causal price-to-trailing-52-week-high ranks."""

    if isinstance(calendar_days, bool) or not isinstance(calendar_days, int) or calendar_days < 252:
        raise _error("52-week-high calendar days must be an integer of at least 252")
    if (
        isinstance(min_periods, bool)
        or not isinstance(min_periods, int)
        or min_periods < 2
        or min_periods > calendar_days
    ):
        raise _error("52-week-high min_periods must be within [2, calendar_days]")
    values, mask = _normalize_inputs(
        required_fields=("adj_close",),
        inputs={"adj_close": adj_close},
        pit_mask=pit_mask,
    )
    if not mask.index.equals(mask.index.normalize()):
        raise _error("52-week-high input sessions must be normalized dates")
    close = values["adj_close"].where(values["adj_close"] > 0.0)
    trailing_high = close.rolling(
        f"{calendar_days}D",
        min_periods=min_periods,
    ).max()
    price_to_high = close.div(trailing_high.where(trailing_high > 0.0))
    return _cross_sectional_rank(price_to_high.replace([np.inf, -np.inf], np.nan)).where(mask)


def high_price_delay_d1_v4(
    *,
    adj_close: pd.DataFrame,
    total_mv: pd.DataFrame,
    pit_mask: pd.DataFrame,
    window_weeks: int = PRICE_DELAY_WINDOW_WEEKS,
    market_lags: int = PRICE_DELAY_MARKET_LAGS,
    min_observations: int = PRICE_DELAY_MIN_OBSERVATIONS,
) -> pd.DataFrame:
    """Return causal Wednesday-to-Wednesday Hou-Moskowitz D1 ranks."""

    if isinstance(window_weeks, bool) or not isinstance(window_weeks, int) or window_weeks < 8:
        raise _error("price-delay window must be an integer of at least 8 weeks")
    if (
        isinstance(market_lags, bool)
        or not isinstance(market_lags, int)
        or market_lags < 1
        or market_lags >= window_weeks
    ):
        raise _error("price-delay market lags must be within [1, window_weeks)")
    if (
        isinstance(min_observations, bool)
        or not isinstance(min_observations, int)
        or min_observations < market_lags + 2
        or min_observations > window_weeks
    ):
        raise _error(
            "price-delay minimum observations must be within " "[market_lags + 2, window_weeks]"
        )
    values, mask = _normalize_inputs(
        required_fields=("adj_close", "total_mv"),
        inputs={"adj_close": adj_close, "total_mv": total_mv},
        pit_mask=pit_mask,
    )
    if not mask.index.equals(mask.index.normalize()):
        raise _error("price-delay input sessions must be normalized dates")

    close = values["adj_close"].where(values["adj_close"] > 0.0)
    market_cap = values["total_mv"].where(values["total_mv"] > 0.0)
    weekly_dates = pd.date_range(
        start=mask.index[0].normalize(),
        end=mask.index[-1].normalize(),
        freq="W-WED",
        name=mask.index.name,
    )
    weekly_close = close.reindex(weekly_dates)
    weekly_cap = market_cap.reindex(weekly_dates)
    weekly_return = weekly_close.pct_change(fill_method=None)

    prior_cap = weekly_cap.shift(1)
    usable_weight = prior_cap.where(weekly_return.notna() & prior_cap.gt(0.0))
    market_denominator = usable_weight.sum(axis=1, min_count=1)
    market_return = (
        weekly_return.mul(usable_weight)
        .sum(axis=1, min_count=1)
        .div(market_denominator.replace(0.0, np.nan))
    )
    lagged_market = pd.concat(
        [market_return.shift(lag) for lag in range(market_lags + 1)],
        axis=1,
    )
    lagged_market.columns = [f"market_lag_{lag}" for lag in range(market_lags + 1)]

    weekly_output = pd.DataFrame(
        np.nan,
        index=weekly_dates,
        columns=mask.columns,
        dtype=float,
    )
    response = weekly_return.to_numpy(dtype=float)
    design = lagged_market.to_numpy(dtype=float)
    first_end = window_weeks + market_lags
    for end in range(first_end, len(weekly_dates)):
        start = end - window_weeks + 1
        design_window = design[start : end + 1]
        response_window = response[start : end + 1]
        finite_design = np.isfinite(design_window).all(axis=1)
        if int(finite_design.sum()) < min_observations:
            continue
        x_market = design_window[finite_design]
        y_all = response_window[finite_design]
        finite_y = np.isfinite(y_all)
        packed_patterns = np.packbits(finite_y, axis=0).T
        _, pattern_groups = np.unique(
            packed_patterns,
            axis=0,
            return_inverse=True,
        )
        d1 = np.full(response.shape[1], np.nan, dtype=float)
        for group_id in range(int(pattern_groups.max()) + 1):
            columns = np.flatnonzero(pattern_groups == group_id)
            valid_rows = finite_y[:, columns[0]]
            if int(valid_rows.sum()) < min_observations:
                continue
            market_sample = x_market[valid_rows]
            y_sample = y_all[valid_rows][:, columns]
            restricted = np.column_stack(
                [np.ones(len(market_sample), dtype=float), market_sample[:, 0]]
            )
            unrestricted = np.column_stack(
                [np.ones(len(market_sample), dtype=float), market_sample]
            )
            if (
                np.linalg.matrix_rank(restricted) != restricted.shape[1]
                or np.linalg.matrix_rank(unrestricted) != unrestricted.shape[1]
            ):
                continue
            restricted_coefficients = np.linalg.lstsq(
                restricted,
                y_sample,
                rcond=None,
            )[0]
            unrestricted_coefficients = np.linalg.lstsq(
                unrestricted,
                y_sample,
                rcond=None,
            )[0]
            centered = y_sample - y_sample.mean(axis=0)
            total_sum_squares = np.square(centered).sum(axis=0)
            restricted_sse = np.square(y_sample - restricted @ restricted_coefficients).sum(axis=0)
            unrestricted_sse = np.square(y_sample - unrestricted @ unrestricted_coefficients).sum(
                axis=0
            )
            usable_variance = total_sum_squares > np.finfo(float).eps
            restricted_r2 = np.full(len(columns), np.nan, dtype=float)
            unrestricted_r2 = np.full(len(columns), np.nan, dtype=float)
            restricted_r2[usable_variance] = (
                1.0 - restricted_sse[usable_variance] / total_sum_squares[usable_variance]
            )
            unrestricted_r2[usable_variance] = (
                1.0 - unrestricted_sse[usable_variance] / total_sum_squares[usable_variance]
            )
            restricted_r2 = np.clip(restricted_r2, 0.0, 1.0)
            unrestricted_r2 = np.clip(unrestricted_r2, 0.0, 1.0)
            valid_d1 = (
                np.isfinite(restricted_r2)
                & np.isfinite(unrestricted_r2)
                & (unrestricted_r2 > np.finfo(float).eps)
                & (restricted_r2 <= unrestricted_r2 + 1e-10)
            )
            group_d1 = np.full(len(columns), np.nan, dtype=float)
            group_d1[valid_d1] = 1.0 - (restricted_r2[valid_d1] / unrestricted_r2[valid_d1])
            d1[columns] = np.clip(group_d1, 0.0, 1.0)
        weekly_output.iloc[end] = d1

    weekly_rank = _cross_sectional_rank(weekly_output)
    actual_wednesdays = mask.index[mask.index.weekday == PRICE_DELAY_WEEKDAY]
    anchor_signal = weekly_rank.reindex(actual_wednesdays)
    daily_signal = anchor_signal.reindex(mask.index, method="ffill")
    return daily_signal.where(mask)


def low_max_return_v4(
    *,
    adj_close: pd.DataFrame,
    pit_mask: pd.DataFrame,
    window: int = LOW_MAX_WINDOW,
    min_periods: int = LOW_MAX_MIN_PERIODS,
) -> pd.DataFrame:
    """Return ranks that prefer a lower maximum daily return in the prior month."""

    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise _error("low-MAX window must be an integer of at least 2")
    if (
        isinstance(min_periods, bool)
        or not isinstance(min_periods, int)
        or min_periods < 2
        or min_periods > window
    ):
        raise _error("low-MAX min_periods must be within [2, window]")
    values, mask = _normalize_inputs(
        required_fields=("adj_close",),
        inputs={"adj_close": adj_close},
        pit_mask=pit_mask,
    )
    close = values["adj_close"].where(values["adj_close"] > 0.0)
    returns = close.pct_change(fill_method=None).where(mask)
    maximum_return = returns.rolling(window, min_periods=min_periods).max()
    return _cross_sectional_rank(-maximum_return).where(mask)


def low_left_tail_var1_v4(
    *,
    adj_close: pd.DataFrame,
    pit_mask: pd.DataFrame,
    window: int = LEFT_TAIL_VAR_WINDOW,
    min_periods: int = LEFT_TAIL_VAR_MIN_PERIODS,
    quantile: float = LEFT_TAIL_VAR_QUANTILE,
) -> pd.DataFrame:
    """Return ranks that prefer a lower source-defined one-percent VaR."""

    if isinstance(window, bool) or not isinstance(window, int) or window < 2:
        raise _error("left-tail VaR1 window must be an integer of at least 2")
    if (
        isinstance(min_periods, bool)
        or not isinstance(min_periods, int)
        or min_periods < 2
        or min_periods > window
    ):
        raise _error("left-tail VaR1 min_periods must be within [2, window]")
    if (
        isinstance(quantile, bool)
        or not isinstance(quantile, (int, float))
        or not np.isfinite(float(quantile))
        or not 0.0 < float(quantile) < 0.5
    ):
        raise _error("left-tail VaR1 quantile must be finite and within (0, 0.5)")
    values, mask = _normalize_inputs(
        required_fields=("adj_close",),
        inputs={"adj_close": adj_close},
        pit_mask=pit_mask,
    )
    close = values["adj_close"].where(values["adj_close"] > 0.0)
    returns = close.pct_change(fill_method=None).where(mask)
    left_tail_quantile = returns.rolling(
        window,
        min_periods=min_periods,
    ).quantile(float(quantile), interpolation="linear")
    return _cross_sectional_rank(left_tail_quantile).where(mask)


def low_total_skewness_v4(
    *,
    adj_close: pd.DataFrame,
    pit_mask: pd.DataFrame,
    window: int = LOW_TOTAL_SKEW_WINDOW,
    min_periods: int = LOW_TOTAL_SKEW_MIN_PERIODS,
) -> pd.DataFrame:
    """Return ranks that prefer lower prior-month total-return skewness."""

    if isinstance(window, bool) or not isinstance(window, int) or window < 3:
        raise _error("total-skewness window must be an integer of at least 3")
    if (
        isinstance(min_periods, bool)
        or not isinstance(min_periods, int)
        or min_periods < 3
        or min_periods > window
    ):
        raise _error("total-skewness min_periods must be within [3, window]")
    values, mask = _normalize_inputs(
        required_fields=("adj_close",),
        inputs={"adj_close": adj_close},
        pit_mask=pit_mask,
    )
    close = values["adj_close"].where(values["adj_close"] > 0.0)
    returns = close.pct_change(fill_method=None).where(mask)
    total_skewness = returns.rolling(window, min_periods=min_periods).skew()
    return _cross_sectional_rank(-total_skewness).where(mask)


def low_market_adjusted_tail_asymmetry_v4(
    *,
    adj_close: pd.DataFrame,
    pit_mask: pd.DataFrame,
    window: int = TAIL_ASYMMETRY_WINDOW,
    min_periods: int = TAIL_ASYMMETRY_MIN_PERIODS,
    sigma_threshold: float = TAIL_ASYMMETRY_SIGMA_THRESHOLD,
) -> pd.DataFrame:
    """Prefer lower upside tail probability in PIT market-adjusted returns."""

    if isinstance(window, bool) or not isinstance(window, int) or window < 3:
        raise _error("tail-asymmetry window must be an integer of at least 3")
    if (
        isinstance(min_periods, bool)
        or not isinstance(min_periods, int)
        or min_periods < 3
        or min_periods > window
    ):
        raise _error("tail-asymmetry min_periods must be within [3, window]")
    if (
        isinstance(sigma_threshold, bool)
        or not isinstance(sigma_threshold, (int, float))
        or not np.isfinite(float(sigma_threshold))
        or float(sigma_threshold) <= 0.0
    ):
        raise _error("tail-asymmetry sigma threshold must be finite and positive")
    values, mask = _normalize_inputs(
        required_fields=("adj_close",),
        inputs={"adj_close": adj_close},
        pit_mask=pit_mask,
    )
    close = values["adj_close"].where(values["adj_close"] > 0.0)
    returns = close.pct_change(fill_method=None).where(mask)
    market_return = returns.mean(axis=1, skipna=True)
    adjusted = returns.sub(market_return, axis=0).where(mask)
    raw = adjusted.to_numpy(dtype=float)
    output = np.full(raw.shape, np.nan, dtype=float)
    threshold = float(sigma_threshold)
    for end in range(min_periods, len(adjusted) + 1):
        start = max(0, end - window)
        sample = raw[start:end]
        finite = np.isfinite(sample)
        counts = finite.sum(axis=0)
        usable = counts >= min_periods
        if not np.any(usable):
            continue
        sums = np.where(finite, sample, 0.0).sum(axis=0)
        means = np.divide(
            sums,
            counts,
            out=np.full(sample.shape[1], np.nan, dtype=float),
            where=counts > 0,
        )
        centered = np.where(finite, sample - means, 0.0)
        variances = np.divide(
            np.square(centered).sum(axis=0),
            counts - 1,
            out=np.full(sample.shape[1], np.nan, dtype=float),
            where=counts > 1,
        )
        standard_deviations = np.sqrt(variances)
        valid_scale = usable & np.isfinite(standard_deviations) & (standard_deviations > 0.0)
        if not np.any(valid_scale):
            continue
        upper_thresholds = means + threshold * standard_deviations
        lower_thresholds = means - threshold * standard_deviations
        upper_counts = (finite & (sample > upper_thresholds)).sum(axis=0)
        lower_counts = (finite & (sample < lower_thresholds)).sum(axis=0)
        excess_tail_probability = np.divide(
            upper_counts - lower_counts,
            counts,
            out=np.full(sample.shape[1], np.nan, dtype=float),
            where=valid_scale,
        )
        output[end - 1] = -excess_tail_probability
    signal = pd.DataFrame(
        output,
        index=adjusted.index,
        columns=adjusted.columns,
        dtype=float,
    )
    return _cross_sectional_rank(signal).where(mask)


def quality_cash_low_leverage_v4(
    *,
    fin_roe: pd.DataFrame,
    fin_ocf_to_profit: pd.DataFrame,
    fin_debt_to_assets: pd.DataFrame,
    pit_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Return an equal-weight complete-case PIT quality proxy."""

    values, mask = _normalize_inputs(
        required_fields=(
            "fin_roe",
            "fin_ocf_to_profit",
            "fin_debt_to_assets",
        ),
        inputs={
            "fin_roe": fin_roe,
            "fin_ocf_to_profit": fin_ocf_to_profit,
            "fin_debt_to_assets": fin_debt_to_assets,
        },
        pit_mask=pit_mask,
    )
    roe_rank = _cross_sectional_rank(values["fin_roe"])
    cash_rank = _cross_sectional_rank(values["fin_ocf_to_profit"])
    leverage_rank = _cross_sectional_rank(values["fin_debt_to_assets"])
    complete = mask & roe_rank.notna() & cash_rank.notna() & leverage_rank.notna()
    signal = (roe_rank + cash_rank + (1.0 - leverage_rank)).div(3.0)
    return signal.where(complete)


def same_month_seasonality_v4(
    *,
    adj_close: pd.DataFrame,
    pit_mask: pd.DataFrame,
    max_annual_lags: int = SAME_MONTH_MAX_ANNUAL_LAGS,
    min_annual_observations: int = SAME_MONTH_MIN_ANNUAL_OBSERVATIONS,
    min_sessions_per_month: int = SAME_MONTH_MIN_SESSIONS_PER_MONTH,
) -> pd.DataFrame:
    """Return causal same-calendar-month seasonality ranks."""

    for value, label, minimum in (
        (max_annual_lags, "max annual lags", 1),
        (min_annual_observations, "minimum annual observations", 1),
        (min_sessions_per_month, "minimum sessions per month", 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise _error(f"same-month {label} must be an integer >= {minimum}")
    if min_annual_observations > max_annual_lags:
        raise _error("same-month minimum annual observations must not exceed max lags")
    values, mask = _normalize_inputs(
        required_fields=("adj_close",),
        inputs={"adj_close": adj_close},
        pit_mask=pit_mask,
    )
    close = values["adj_close"].where(values["adj_close"] > 0.0)
    returns = close.pct_change(fill_method=None)
    periods = returns.index.to_period("M")
    monthly_count = returns.groupby(periods).count()
    monthly_return = (
        returns.add(1.0)
        .groupby(periods)
        .prod(min_count=min_sessions_per_month)
        .sub(1.0)
        .where(monthly_count >= min_sessions_per_month)
    )
    market_return = monthly_return.mean(axis=1, skipna=True)
    adjusted = monthly_return.sub(market_return, axis=0)
    seasonal = pd.DataFrame(
        np.nan,
        index=adjusted.index,
        columns=adjusted.columns,
        dtype=float,
    )
    available = set(adjusted.index)
    for period in adjusted.index:
        prior_periods = [
            period - (12 * annual_lag)
            for annual_lag in range(1, max_annual_lags + 1)
            if period - (12 * annual_lag) in available
        ]
        if len(prior_periods) < min_annual_observations:
            continue
        history = adjusted.loc[prior_periods]
        observation_count = history.notna().sum(axis=0)
        seasonal.loc[period] = history.mean(axis=0, skipna=True).where(
            observation_count >= min_annual_observations
        )
    expanded = seasonal.reindex(periods)
    expanded.index = returns.index
    expanded.columns = returns.columns
    return _cross_sectional_rank(expanded).where(mask)


def fip_continuous_direction_v4(
    *,
    adj_close: pd.DataFrame,
    pit_mask: pd.DataFrame,
    formation_sessions: int = FIP_FORMATION_SESSIONS,
    skip_sessions: int = FIP_SKIP_SESSIONS,
    min_periods: int = FIP_MIN_PERIODS,
) -> pd.DataFrame:
    """Return a causal single-score translation of FIP information continuity."""

    for value, label, minimum in (
        (formation_sessions, "formation sessions", 2),
        (skip_sessions, "skip sessions", 1),
        (min_periods, "minimum periods", 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise _error(f"FIP {label} must be an integer >= {minimum}")
    if min_periods > formation_sessions:
        raise _error("FIP minimum periods must not exceed formation sessions")
    values, mask = _normalize_inputs(
        required_fields=("adj_close",),
        inputs={"adj_close": adj_close},
        pit_mask=pit_mask,
    )
    close = values["adj_close"].where(values["adj_close"] > 0.0)
    returns = close.pct_change(fill_method=None).where(mask)
    formation_returns = returns.shift(skip_sessions)
    valid_count = formation_returns.rolling(
        formation_sessions,
        min_periods=min_periods,
    ).count()
    positive_count = (
        formation_returns.gt(0.0)
        .rolling(
            formation_sessions,
            min_periods=min_periods,
        )
        .sum()
    )
    negative_count = (
        formation_returns.lt(0.0)
        .rolling(
            formation_sessions,
            min_periods=min_periods,
        )
        .sum()
    )
    pret = close.shift(skip_sessions).div(close.shift(skip_sessions + formation_sessions)).sub(1.0)
    pret_sign = np.sign(pret)
    information_discreteness = pret_sign.mul(
        negative_count.sub(positive_count).div(valid_count.where(valid_count > 0.0))
    )
    continuous_direction = pret_sign.mul(1.0 - information_discreteness).div(2.0)
    usable = (
        mask
        & valid_count.ge(min_periods)
        & pret.notna()
        & pret.ne(0.0)
        & information_discreteness.ge(-1.0)
        & information_discreteness.le(1.0)
    )
    return _cross_sectional_rank(continuous_direction.where(usable)).where(mask)


def evaluate_candidate_v4(
    *,
    name: str,
    inputs: Mapping[str, pd.DataFrame],
    pit_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate one allowlisted signal without accepting labels or outcomes."""

    if name == "cn_earnings_yield_ex_shell_30pct":
        values, _ = _normalize_inputs(
            required_fields=("pe", "total_mv"),
            inputs=inputs,
            pit_mask=pit_mask,
        )
        return earnings_yield_ex_shell_v4(
            pe=values["pe"], total_mv=values["total_mv"], pit_mask=pit_mask
        )
    if name == "cn_low_beta_252d":
        values, _ = _normalize_inputs(
            required_fields=("adj_close",),
            inputs=inputs,
            pit_mask=pit_mask,
        )
        return low_beta_v4(adj_close=values["adj_close"], pit_mask=pit_mask)
    if name == "cn_52_week_high_momentum_12m":
        values, _ = _normalize_inputs(
            required_fields=("adj_close",),
            inputs=inputs,
            pit_mask=pit_mask,
        )
        return high_52_week_momentum_v4(
            adj_close=values["adj_close"],
            pit_mask=pit_mask,
        )
    if name == "cn_high_price_delay_d1_52w":
        values, _ = _normalize_inputs(
            required_fields=("adj_close", "total_mv"),
            inputs=inputs,
            pit_mask=pit_mask,
        )
        return high_price_delay_d1_v4(
            adj_close=values["adj_close"],
            total_mv=values["total_mv"],
            pit_mask=pit_mask,
        )
    if name == "cn_low_max_return_20d":
        values, _ = _normalize_inputs(
            required_fields=("adj_close",),
            inputs=inputs,
            pit_mask=pit_mask,
        )
        return low_max_return_v4(
            adj_close=values["adj_close"],
            pit_mask=pit_mask,
        )
    if name == "cn_low_total_skewness_20d":
        values, _ = _normalize_inputs(
            required_fields=("adj_close",),
            inputs=inputs,
            pit_mask=pit_mask,
        )
        return low_total_skewness_v4(
            adj_close=values["adj_close"],
            pit_mask=pit_mask,
        )
    if name == "cn_low_market_adjusted_tail_asymmetry_252d":
        values, _ = _normalize_inputs(
            required_fields=("adj_close",),
            inputs=inputs,
            pit_mask=pit_mask,
        )
        return low_market_adjusted_tail_asymmetry_v4(
            adj_close=values["adj_close"],
            pit_mask=pit_mask,
        )
    if name == "cn_quality_cash_low_leverage":
        values, _ = _normalize_inputs(
            required_fields=(
                "fin_roe",
                "fin_ocf_to_profit",
                "fin_debt_to_assets",
            ),
            inputs=inputs,
            pit_mask=pit_mask,
        )
        return quality_cash_low_leverage_v4(
            fin_roe=values["fin_roe"],
            fin_ocf_to_profit=values["fin_ocf_to_profit"],
            fin_debt_to_assets=values["fin_debt_to_assets"],
            pit_mask=pit_mask,
        )
    if name == "cn_same_month_seasonality_5y":
        values, _ = _normalize_inputs(
            required_fields=("adj_close",),
            inputs=inputs,
            pit_mask=pit_mask,
        )
        return same_month_seasonality_v4(
            adj_close=values["adj_close"],
            pit_mask=pit_mask,
        )
    if name == "cn_fip_continuous_direction_12m":
        values, _ = _normalize_inputs(
            required_fields=("adj_close",),
            inputs=inputs,
            pit_mask=pit_mask,
        )
        return fip_continuous_direction_v4(
            adj_close=values["adj_close"],
            pit_mask=pit_mask,
        )
    if name == "cn_low_left_tail_var1_250d":
        values, _ = _normalize_inputs(
            required_fields=("adj_close",),
            inputs=inputs,
            pit_mask=pit_mask,
        )
        return low_left_tail_var1_v4(
            adj_close=values["adj_close"],
            pit_mask=pit_mask,
        )
    raise _error(f"candidate is not allowlisted: {name}")


__all__ = [
    "AUTHORITY_FLAGS",
    "HIGH_52_WEEK_CALENDAR_DAYS",
    "HIGH_52_WEEK_MIN_PERIODS",
    "HIGH_52_WEEK_SELECTION_DATE",
    "INCUBATOR_VERSION",
    "LITERATURE_IDEAS",
    "FIP_FORMATION_SESSIONS",
    "FIP_MIN_PERIODS",
    "FIP_SELECTION_DATE",
    "FIP_SKIP_SESSIONS",
    "LEFT_TAIL_VAR_MIN_PERIODS",
    "LEFT_TAIL_VAR_QUANTILE",
    "LEFT_TAIL_VAR_SELECTION_DATE",
    "LEFT_TAIL_VAR_WINDOW",
    "LOW_BETA_MIN_PERIODS",
    "LOW_BETA_WINDOW",
    "LOW_MAX_MIN_PERIODS",
    "LOW_MAX_SELECTION_DATE",
    "LOW_MAX_WINDOW",
    "LOW_TOTAL_SKEW_MIN_PERIODS",
    "LOW_TOTAL_SKEW_SELECTION_DATE",
    "LOW_TOTAL_SKEW_WINDOW",
    "PRICE_DELAY_MARKET_LAGS",
    "PRICE_DELAY_MIN_OBSERVATIONS",
    "PRICE_DELAY_SELECTION_DATE",
    "PRICE_DELAY_WEEKDAY",
    "PRICE_DELAY_WINDOW_WEEKS",
    "PROTECTED_EXACT_FIVE_CANDIDATE_NAMES",
    "PROTOCOL_VERSION",
    "SAME_MONTH_LOOKBACK_SESSIONS",
    "SAME_MONTH_MAX_ANNUAL_LAGS",
    "SAME_MONTH_MIN_ANNUAL_OBSERVATIONS",
    "SAME_MONTH_MIN_SESSIONS_PER_MONTH",
    "SHELL_EXCLUSION_FRACTION",
    "SIDE_EFFECT_FLAGS",
    "TAIL_ASYMMETRY_MIN_PERIODS",
    "TAIL_ASYMMETRY_SELECTION_DATE",
    "TAIL_ASYMMETRY_SIGMA_THRESHOLD",
    "TAIL_ASYMMETRY_WINDOW",
    "FactorGovernanceLiteratureIncubatorV4Error",
    "build_structural_audit_v4",
    "build_protected_exact_five_audit_v4",
    "candidate_catalog_artifact_v4",
    "candidate_catalog_v4",
    "candidate_literature_assessments_v4",
    "candidate_ontology_v4",
    "earnings_yield_ex_shell_v4",
    "evaluate_candidate_v4",
    "fip_continuous_direction_v4",
    "fip_future_preregistration_policy_v4",
    "high_52_week_momentum_v4",
    "high_price_delay_d1_v4",
    "literature_idea_catalog_v4",
    "left_tail_var1_future_preregistration_policy_v4",
    "low_beta_v4",
    "low_left_tail_var1_v4",
    "low_max_future_preregistration_policy_v4",
    "low_max_return_v4",
    "low_market_adjusted_tail_asymmetry_v4",
    "low_total_skewness_future_preregistration_policy_v4",
    "low_total_skewness_v4",
    "quality_cash_low_leverage_v4",
    "same_month_seasonality_v4",
    "tail_asymmetry_future_preregistration_policy_v4",
]
