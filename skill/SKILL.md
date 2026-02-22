---
name: quant-investor
description: "端到端AI量化投资技能，用于股票分析、因子挖掘、模型训练和回测。V6.2版本新增A股宏观风控终端（Leverage/Growth/Valuation/Inflation四大模块），强制报告透明化（展示数据获取、分析过程和推理逻辑）。当用户想要进行量化投资、分析股票、构建投资策略或进行回测时使用此技能。"
---

# 量化投资技能 (Quant-Investor) - V6.2

**版本**: 6.2 (宏观风控终端 + 报告透明化)
**核心理念**: 数据驱动 + 分层解耦 + AI因子工程 + 过程透明

---

## 1. 技能简介

`quant-investor` V6.2 在 V6.1 基础上进行两项重大升级：

### V6.2 核心升级

1. **A股宏观风控终端**: 新增四大宏观风控模块，提供市场级别的系统性风险监控。
   - **资金杠杆与情绪 (Leverage)**: 两融余额、两融/流通市值比，含2015牛市顶参考
   - **经济景气度 (Growth)**: GDP同比增速，含季度趋势追踪
   - **整体估值锚 (Valuation)**: 巴菲特指标（市值/GDP），含2007/2015历史大顶对标
   - **通胀与货币 (Inflation & Money)**: CPI、PPI、M1-M2剪刀差、M2增速、社融增量

2. **报告透明化**: 所有报告必须详细展示分析过程，包括下载了什么数据、做了哪些分析、每一步的推理逻辑和结论依据。用户需要完整了解从数据到结论的全链路。

### V6.1 能力（保留）

- 海纳百川因子库 (500+): Qlib Alpha158/360 + TA-Lib 50+ + WorldQuant 101 Alphas
- AI创新因子引擎: Transformer时序特征 + LLM情绪因子 + 另类数据因子
- 自动化因子挖掘: 遗传编程引擎

---

## 2. V6.2 分层架构

```
MasterPipelineV6.2
├── 第0层: 宏观风控终端 (MacroRiskTerminal) ← V6.2新增
│   ├── 资金杠杆与情绪: 两融余额 / 两融占比 / 2015顶部对标
│   ├── 经济景气度: GDP同比 / 季度趋势
│   ├── 整体估值锚: 巴菲特指标 / 2007&2015大顶对标
│   ├── 通胀与货币: CPI / PPI / M1-M2剪刀差 / M2增速 / 社融
│   └── 综合风控信号: 🔴高风险 / 🟡中风险 / 🟢低风险 / 🔵极低风险
│
├── 第1层: 数据层 (UnifiedDataLayer)
│   ├── 数据源: yfinance / Tushare / FRED / AKShare
│   ├── 持久化: SQLite + Parquet
│   ├── 数据清洗: 去极值 / 补缺失 / 标准化
│   ├── 样本扩充: 自动补充指数成分股
│   └── 数据增强: 完整性检查 + 另类数据接口
│
├── 第2层: 因子层 (UnifiedFactorLayer) - V6.1升级
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

## 3. A股宏观风控终端

分析A股市场时，**必须首先运行宏观风控终端**，在个股分析之前建立市场级别的风险判断。

**详细指标体系和判断规则**: 见 `references/cn_macro_risk_terminal.md`

**数据获取和分析脚本**: 见 `scripts/v2.5/cn_macro_data/macro_risk_terminal.py`

### 使用方法

```python
import sys
sys.path.insert(0, "/home/ubuntu/skills/quant-investor/scripts/v2.5")
from cn_macro_data.macro_risk_terminal import MacroRiskTerminal

terminal = MacroRiskTerminal(tushare_token="YOUR_TOKEN")
report = terminal.generate_risk_report()
markdown = terminal.format_report_markdown(report)
```

### 宏观风控对投资决策的影响

| 综合信号 | 仓位建议 | 策略调整 |
|:---|:---|:---|
| 🔴 高风险 | ≤30% | 防御为主，优先现金和低波动资产 |
| 🟡 中风险 | 30%-60% | 控制仓位，精选高质量个股 |
| 🟢 低风险 | 60%-90% | 正常配置，积极布局成长股 |
| 🔵 极低风险 | 80%-100% | 加大配置，逆向布局超跌优质股 |

---

## 4. 报告透明化要求

**所有报告必须遵循以下透明化原则**，确保用户完整了解分析过程：

### 4.1 数据获取透明

报告必须明确说明：
- 下载了哪些数据（数据名称、来源、时间范围、数据量）
- 数据获取是否成功，失败时的降级策略
- 使用了哪些API和数据源

### 4.2 分析过程透明

报告必须展示：
- 每个指标的计算公式和中间结果
- 因子选择的依据和有效性验证结果
- 模型训练的配置、特征重要性和预测结果
- LLM辩论的核心论点和推理过程

### 4.3 结论推导透明

报告必须说明：
- 从数据到结论的完整推理链
- 每个建议背后的量化依据
- 风险判断的具体触发条件
- 与历史数据的对标结果

### 4.4 报告模板

使用 `templates/report.md.template` 作为报告结构参考。报告应包含以下核心章节：

1. **数据获取过程** — 详细的数据下载清单和日志
2. **A股宏观风控终端** — 四大模块的完整分析（A股分析时）
3. **量化因子分析** — 因子计算和验证过程
4. **模型预测与选股** — 模型配置和结果
5. **LLM多Agent辩论** — 辩论过程和结论
6. **风险评估与组合优化** — 风险指标和优化结果
7. **投资结论与建议** — 综合结论和具体建议
8. **附录** — 完整日志和原始数据文件路径

---

## 5. 使用方法

### A股分析（含宏观风控终端）

```
/quant-investor 分析A股市场

/quant-investor 分析 000001.SZ 600000.SH 300750.SZ
```

执行流程：
1. 运行宏观风控终端 → 获取市场级风险信号
2. 根据风控信号调整分析策略和仓位建议
3. 执行个股量化分析流水线
4. 生成透明化报告

### 美股分析

```
/quant-investor 分析美股市场

/quant-investor 分析 AAPL MSFT NVDA GOOGL AMZN
```

### 代码调用

```python
import sys
sys.path.insert(0, "/home/ubuntu/skills/quant-investor/scripts/v6.1")

from pipeline.master_pipeline import MasterPipelineV6

pipeline = MasterPipelineV6(
    market="CN",                              # A股
    enable_macro_risk_terminal=True,           # V6.2: 启用宏观风控终端
    enable_transparent_reporting=True,         # V6.2: 启用报告透明化
    enable_factor_mining=True,
    enable_transformer_factors=True,
    factor_libraries=["qlib", "talib", "worldquant"]
)
report = pipeline.run()
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `enable_macro_risk_terminal` | bool | True | V6.2: 是否启用A股宏观风控终端 |
| `enable_transparent_reporting` | bool | True | V6.2: 是否启用报告透明化 |
| `enable_factor_mining` | bool | False | 是否启用遗传编程进行因子挖掘 |
| `enable_transformer_factors` | bool | False | 是否启用Transformer提取时序特征因子 |
| `factor_libraries` | list | [] | 加载的外部因子库 ["qlib", "talib", "worldquant"] |

---

## 6. 代码结构

```
scripts/
├── v2.5/cn_macro_data/
│   ├── cn_macro_data_manager.py       # 中国宏观数据管理器
│   └── macro_risk_terminal.py         # V6.2: A股宏观风控终端
├── v6.0/                              # V6.0 大一统框架
│   ├── data_layer/
│   ├── factor_layer/
│   │   ├── factor_libraries/          # Qlib/TA-Lib/WorldQuant
│   │   └── factor_mining/             # 遗传编程引擎
│   ├── model_layer/
│   ├── decision_layer/
│   ├── risk_layer/
│   └── pipeline/
└── [v2.3-v5.0]/                       # 历史版本脚本

references/
├── cn_macro_risk_terminal.md          # V6.2: 宏观风控终端指标体系
├── data_sources.md                    # 数据源参考
├── feature_library.md                 # 特征库参考
├── bayesian_guide.md                  # 贝叶斯指南
├── causal_inference.md                # 因果推断
├── master_investors_wisdom.md         # 投资大师智慧
├── model_zoo.md                       # 模型库
├── qlib_integration.md                # Qlib集成
└── statistical_framework.md           # 统计框架

templates/
├── config.ini.template                # 配置模板
└── report.md.template                 # V6.2: 透明化报告模板
```

---

## 7. 版本演进

| 版本 | 核心特性 |
|:---|:---|
| **V6.2** | **宏观风控终端 + 报告透明化** - 四大宏观风控模块(Leverage/Growth/Valuation/Inflation)，强制报告展示完整分析过程 |
| V6.1 | AI因子挖掘增强 - 整合Qlib/TA-Lib/WorldQuant，引入遗传编程、Transformer、LLM因子 |
| V6.0 | 大一统框架 - 分层解耦，融合V2.7~V5.0全部能力，智能样本扩充 |
| V5.0 | 工业级量化框架 - 数据清洗、ML模型、组合优化、高级风险管理 |
| V4.1 | 基准对比升级版 - 超越基准的长期稳定超额收益 |
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
