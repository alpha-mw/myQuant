---
name: quant-investor
description: "一个端到端的AI量化投资技能，用于股票分析、因子挖掘、模型训练和回测。V6.0大一统版本将V2.7到V5.0的所有核心能力融合到一个统一、分层解耦的框架中，涵盖数据获取与清洗、因子计算与验证、ML模型训练与信号生成、多LLM多Agent辩论决策、组合优化与高级风险管理。当用户想要进行量化投资、分析股票、构建投资策略或进行回测时使用此技能。"
---

# 量化投资技能 (Quant-Investor) - V6.0

**版本**: 6.0 (大一统框架)
**作者**: Manus AI
**核心理念**: 数据驱动 + 分层解耦 + 端到端自动化

---

## 1. 技能简介

`quant-investor` V6.0 是一个**大一统的AI量化投资框架**，将V2.7到V5.0的所有核心能力无缝融合到一个统一、高效、可扩展的分层架构中。用户只需提供最简单的指令（市场、股票池），即可触发完整的端到端分析流水线。

### V6.0 核心升级

1. **统一分层架构**: 5层解耦设计（数据层→因子层→模型层→决策层→风控层），每层独立可测试、可替换。
2. **智能样本扩充**: 自定义股票池时自动拉取指数成分股（NASDAQ100+S&P500 / 沪深300+中证1000），确保因子验证和模型训练有充足的截面样本，决策层和风控层聚焦用户关注股票。
3. **多LLM多Agent辩论**: 整合Gemini、OpenAI、DeepSeek、Qwen四大LLM，5个专家Agent（财务、行业、护城河、估值、风险）进行多轮交叉质询。
4. **完整风险评估**: VaR/CVaR、压力测试、Alpha/Beta分析、Barra风险分解。
5. **统一MasterPipeline**: 单一入口串联所有层，自动生成Markdown投资报告。

---

## 2. V6.0 分层架构

```
MasterPipelineV6
├── 第1层: 数据层 (UnifiedDataLayer)
│   ├── 数据源: yfinance / Tushare / FRED
│   ├── 持久化: SQLite + Parquet (自动缓存/增量更新)
│   ├── 数据清洗: 去极值 / 补缺失 / 标准化
│   └── 样本扩充: 自定义池 → 自动补充指数成分股
│
├── 第2层: 因子层 (UnifiedFactorLayer)
│   ├── 因子库: 34+因子 (动量/反转/波动/RSI/MACD/成交量/价格位置)
│   ├── 因子验证: IC/IR/多空收益/分层回测
│   └── 候选选股: 多因子综合评分排名
│
├── 第3层: 模型层 (UnifiedModelLayer)
│   ├── ML模型: XGBoost / LightGBM / RandomForest
│   ├── 集成预测: 加权集成 + 特征重要性
│   └── 信号生成: 综合排名Top N
│
├── 第4层: 决策层 (UnifiedDecisionLayer)
│   ├── 多LLM适配器: Gemini / OpenAI / DeepSeek / Qwen
│   ├── 多Agent辩论: 5专家 × 多轮交叉质询
│   └── 投资评级: 强烈买入/买入/持有/卖出/强烈卖出
│
└── 第5层: 风控层 (UnifiedRiskLayer)
    ├── 组合优化: 最大夏普 / 风险平价 / 最小方差 / 等权
    ├── 风险评估: VaR / CVaR / 最大回撤 / 波动率
    ├── 基准对比: Alpha / Beta / 信息比率 / 胜率
    └── 压力测试: 5种极端场景
```

---

## 3. 使用方法

直接调用技能即可，系统会自动执行V6.0的完整流水线：

```
/quant-investor 分析美股市场

/quant-investor 分析 AAPL MSFT NVDA GOOGL AMZN

/quant-investor 分析我的A股持仓：600519 30%, 000858 20%
```

### 代码调用

```python
import sys
sys.path.insert(0, "/home/ubuntu/skills/quant-investor/scripts/v6.0")

from pipeline.master_pipeline import MasterPipelineV6

# 基本用法：分析美股市场（自动使用指数成分股）
pipeline = MasterPipelineV6(market="US")
report = pipeline.run()

# 自定义股票池（自动扩充样本 + 聚焦分析）
pipeline = MasterPipelineV6(
    market="US",
    stock_pool=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    lookback_years=3,
    optimization_method='max_sharpe',
    total_capital=1000000,
    max_debate_stocks=5,
    max_debate_rounds=1
)
report = pipeline.run()

# A股分析
pipeline = MasterPipelineV6(market="CN")
report = pipeline.run()
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `market` | str | "US" | 市场类型: "US"(美股) 或 "CN"(A股) |
| `stock_pool` | list | None | 自定义股票池，None则使用指数成分股 |
| `lookback_years` | int | 3 | 历史数据回溯年数 |
| `llm_preference` | list | None | LLM偏好顺序，如 ["gemini", "openai"] |
| `optimization_method` | str | "max_sharpe" | 组合优化方法 |
| `total_capital` | float | 1000000 | 总投资资金 |
| `top_n_candidates` | int | 20 | 因子层候选股票数 |
| `top_n_final` | int | 10 | 模型层最终排名数 |
| `max_debate_stocks` | int | 5 | 决策层深度分析股票数 |
| `max_debate_rounds` | int | 1 | 辩论轮数 |

---

## 4. V6.0 代码结构

```
scripts/v6.0/
├── data_layer/
│   └── unified_data_layer.py      # 统一数据层 (1014行)
├── factor_layer/
│   └── unified_factor_layer.py    # 统一因子层 (776行)
├── model_layer/
│   └── unified_model_layer.py     # 统一模型层 (502行)
├── decision_layer/
│   └── unified_decision_layer.py  # 统一决策层 (864行)
├── risk_layer/
│   └── unified_risk_layer.py      # 统一风控层 (687行)
└── pipeline/
    └── master_pipeline.py         # 统一流水线 (508行)
```

**总代码量**: ~4,400行 Python

---

## 5. 版本演进

| 版本 | 核心特性 |
|:---|:---|
| **V6.0** | **大一统框架** - 分层解耦，融合V2.7~V5.0全部能力，智能样本扩充 |
| V5.0 | 工业级量化框架 - 数据清洗、ML模型、组合优化、高级风险管理 |
| V4.1 | 基准对比升级版 - 以超越基准的长期稳定超额收益为核心 |
| V4.0 | 统一主流水线 - 整合所有能力，标准化流程 |
| V3.6 | 多LLM支持 - DeepSeek、千问、Kimi |
| V3.5 | 深度特征合成引擎 |
| V3.4 | 海纳百川因子库 - Alpha158 + tsfresh |
| V3.3 | 工业级因子分析 - Tear Sheet |
| V3.2 | 动态因子挖掘系统 |
| V3.1 | 动态智能框架 - 因子监控 + 组合优化 |
| V3.0 | 全景数据层 - 期货期权 + 行业数据 |
| V2.9 | 多Agent辩论系统 |
| V2.8 | 风险管理模块 |
| V2.7 | 持久化数据存储 |

---

**如有问题或建议，欢迎反馈！**
