---
name: quant-investor
description: "一个端到端的AI量化投资技能，用于股票分析、因子挖掘、模型训练和回测。V6.1版本在V6.0大一统框架基础上，全面升级因子层，整合Qlib、TA-Lib、WorldQuant因子库，并引入AI驱动的因子挖掘引擎（遗传编程、Transformer、LLM情绪分析），实现从500+因子库到自动化因子发现的全链路增强。当用户想要进行量化投资、分析股票、构建投资策略或进行回测时使用此技能。"
---

# 量化投资技能 (Quant-Investor) - V6.1

**版本**: 6.1 (AI因子挖掘增强)
**作者**: Manus AI
**核心理念**: 数据驱动 + 分层解耦 + AI因子工程

---

## 1. 技能简介

`quant-investor` V6.1 是在 V6.0 大一统框架基础上的**一次重大升级**，核心是**对因子层的革命性增强**。通过整合业界主流因子库和引入AI驱动的因子挖掘引擎，V6.1 将因子分析能力从“使用”提升到“创造”，实现了从海量因子库到自动化因子发现的全链路智能化。

### V6.1 核心升级

1.  **海纳百川因子库 (500+)**: 全面整合三大主流因子库，因子数量从 34+ 扩展到 500+。
    *   **Qlib**: Alpha158 / Alpha360
    *   **TA-Lib**: 50+ 技术指标
    *   **WorldQuant**: 101 Alphas 精选

2.  **AI 创新因子引擎**: 引入 AI 模型生成和提取新型因子。
    *   **Transformer 时序特征**: 使用 Transformer 提取深度时序模式作为新因子。
    *   **LLM 情绪因子**: 利用大模型分析财报、新闻文本，生成情绪得分因子。
    *   **另类数据因子**: 整合社交媒体情绪、搜索趋势等非传统数据源。

3.  **自动化因子挖掘**: 内置遗传编程（Genetic Programming）引擎，可自动发现新的有效因子表达式。

4.  **数据层增强**: 在数据获取阶段增加完整性检查和另类数据接口，为因子挖掘提供更坚实的基础。

---

## 2. V6.1 分层架构

```
MasterPipelineV6.1
├── 第1层: 数据层 (UnifiedDataLayer)
│   ├── 数据源: yfinance / Tushare / FRED
│   ├── 持久化: SQLite + Parquet
│   ├── 数据清洗: 去极值 / 补缺失 / 标准化
│   ├── 样本扩充: 自动补充指数成分股
│   └── (V6.1)数据增强: 完整性检查 + 另类数据接口
│
├── 第2层: 因子层 (UnifiedFactorLayer) - V6.1 全面升级
│   ├── 因子库 (500+): Qlib / TA-Lib / WorldQuant / 基础因子
│   ├── AI创新因子: Transformer特征 / LLM情绪 / 另类数据
│   ├── 因子挖掘引擎: 遗传编程自动发现新因子
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

使用方法与 V6.0 保持一致，技能会自动启用 V6.1 的增强功能。用户可以通过参数开启或关闭特定的因子挖掘功能。

```
/quant-investor 分析美股市场

/quant-investor 分析 AAPL MSFT NVDA GOOGL AMZN
```

### 代码调用

```python
import sys
sys.path.insert(0, "/home/ubuntu/skills/quant-investor/scripts/v6.1")

from pipeline.master_pipeline import MasterPipelineV6

# 启用所有新功能
pipeline = MasterPipelineV6(
    market="US",
    enable_factor_mining=True, # 开启遗传编程因子挖掘
    enable_transformer_factors=True, # 开启 Transformer 因子
    factor_libraries=["qlib", "talib", "worldquant"] # 加载所有因子库
)
report = pipeline.run()
```

### 新增参数说明

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `enable_factor_mining` | bool | False | 是否启用遗传编程进行因子挖掘 |
| `enable_transformer_factors` | bool | False | 是否启用 Transformer 提取时序特征因子 |
| `factor_libraries` | list | [] | 加载的外部因子库，可选 ["qlib", "talib", "worldquant"] |

---

## 4. V6.1 代码结构

```
scripts/v6.1/
├── data_layer/
│   └── unified_data_layer.py
├── factor_layer/
│   ├── unified_factor_layer.py          # 主入口
│   ├── factor_libraries/                # 新增：因子库模块
│   │   ├── qlib_factors.py             # Qlib Alpha158/360
│   │   ├── talib_factors.py            # TA-Lib 指标
│   │   ├── worldquant_factors.py       # WorldQuant Alphas
│   │   └── alternative_factors.py      # 另类数据与AI因子
│   └── factor_mining/                   # 新增：因子挖掘模块
│       └── genetic_miner.py            # 遗传编程引擎
├── model_layer/
│   └── unified_model_layer.py
├── decision_layer/
│   └── unified_decision_layer.py
├── risk_layer/
│   └── unified_risk_layer.py
└── pipeline/
    └── master_pipeline.py
```

---

## 5. 版本演进

| 版本 | 核心特性 |
|:---|:---|
| **V6.1** | **AI因子挖掘增强** - 整合Qlib/TA-Lib/WorldQuant，引入遗传编程、Transformer、LLM因子 |
| V6.0 | **大一统框架** - 分层解耦，融合V2.7~V5.0全部能力，智能样本扩充 |
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
