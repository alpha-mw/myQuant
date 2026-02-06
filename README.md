# 🚀 Quant-Investor V6.0

<div align="center">

**大一统AI量化投资框架**

*数据驱动 · 分层解耦 · 端到端自动化*

[![Version](https://img.shields.io/badge/Version-6.0-blue.svg)](https://github.com/maxwelllee54/myQuant)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 项目简介

**Quant-Investor** 是一个整合了量化分析与AI深度思辨的智能投资研究平台。V6.0大一统版本将V2.7到V5.0的所有核心能力融合到一个统一、分层解耦的框架中，实现了从数据获取到投资报告的全自动化流水线。

### 核心投资逻辑

```
数据获取与清洗 → 因子计算与验证 → ML建模与信号生成 → 多LLM多Agent辩论 → 组合优化与风控
```

1. **数据驱动**：自动获取并清洗海量数据（价量、基本面、宏观），智能扩充截面样本。
2. **因子验证**：34+因子计算，IC/IR/多空收益严格验证，筛选有效因子。
3. **机器学习**：XGBoost/LightGBM/RandomForest集成预测，生成综合排名信号。
4. **AI辩论**：4大LLM（Gemini/OpenAI/DeepSeek/Qwen）驱动5个专家Agent多轮交叉质询。
5. **风险管理**：组合优化、VaR/CVaR、压力测试、Alpha/Beta基准对比。

---

## ✨ V6.0 分层架构

| 层级 | 模块 | 核心能力 |
|:---|:---|:---|
| **第1层** | 数据层 (UnifiedDataLayer) | yfinance/Tushare/FRED数据源；SQLite+Parquet持久化；去极值/补缺失/标准化；智能样本扩充 |
| **第2层** | 因子层 (UnifiedFactorLayer) | 34+因子（动量/反转/波动/RSI/MACD/成交量）；IC/IR验证；多因子综合评分 |
| **第3层** | 模型层 (UnifiedModelLayer) | XGBoost/LightGBM/RandomForest；加权集成预测；特征重要性分析 |
| **第4层** | 决策层 (UnifiedDecisionLayer) | 4大LLM适配器；5专家Agent多轮辩论；投资评级生成 |
| **第5层** | 风控层 (UnifiedRiskLayer) | 最大夏普/风险平价/最小方差优化；VaR/CVaR/压力测试；Alpha/Beta分析 |

### 智能样本扩充

V6.0的核心创新之一是**智能样本扩充**：当用户提供自定义股票池（如5只股票）时，数据层自动拉取完整指数成分股（约90只），确保因子验证和模型训练有充足的截面样本。决策层和风控层则聚焦用户关注的股票，实现"宽样本训练 + 窄焦点决策"。

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/maxwelllee54/myQuant.git
cd myQuant

# 安装依赖
pip install pandas numpy yfinance scipy
pip install xgboost lightgbm scikit-learn statsmodels
```

### API密钥配置

创建配置文件 `~/.quant_investor/credentials.env`：

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
DEEPSEEK_API_KEY=sk-...
DASHSCOPE_API_KEY=sk-...

# 数据源 API Keys
TUSHARE_TOKEN=...
```

### 使用示例

#### 方式1：Manus技能调用

```
/quant-investor 分析美股市场

/quant-investor 分析 AAPL MSFT NVDA GOOGL AMZN

/quant-investor 分析我的持仓：AAPL 25%, MSFT 25%, GOOGL 20%
```

#### 方式2：Python代码调用

```python
import sys
sys.path.insert(0, "scripts/v6.0")

from pipeline.master_pipeline import MasterPipelineV6

# 基本用法：分析美股市场
pipeline = MasterPipelineV6(market="US")
report = pipeline.run()

# 自定义股票池（自动扩充样本 + 聚焦分析）
pipeline = MasterPipelineV6(
    market="US",
    stock_pool=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    lookback_years=3,
    optimization_method='max_sharpe',
    max_debate_stocks=5
)
report = pipeline.run()
```

---

## 📁 代码结构

```
myQuant/
├── README.md
├── skill/
│   └── SKILL.md                    # Manus技能定义文件
├── scripts/
│   ├── v6.0/                       # V6.0 大一统版本 (~4,400行)
│   │   ├── data_layer/             # 统一数据层
│   │   ├── factor_layer/           # 统一因子层
│   │   ├── model_layer/            # 统一模型层
│   │   ├── decision_layer/         # 统一决策层
│   │   ├── risk_layer/             # 统一风控层
│   │   └── pipeline/               # 统一流水线
│   ├── v5.0/                       # V5.0 工业级量化框架
│   ├── v4.0~v4.1/                  # V4.x 统一流水线
│   ├── v3.0~v3.6/                  # V3.x 因子+LLM
│   └── v2.7~v2.9/                  # V2.x 基础框架
```

---

## 📈 版本演进

| 版本 | 发布日期 | 核心特性 |
|:---|:---|:---|
| **V6.0** | 2026-02-06 | **大一统框架** - 分层解耦，融合V2.7~V5.0全部能力，智能样本扩充 |
| V5.0 | 2026-02-06 | 工业级量化框架 - 全面升级量化能力 |
| V4.1 | 2026-02-05 | 基准对比升级版，以超越基准为核心 |
| V4.0 | 2026-02-04 | 统一主流水线，整合所有能力 |
| V3.6 | 2026-02-04 | 多LLM支持（DeepSeek/千问/Kimi） |
| V3.5 | 2026-02-04 | 深度特征合成引擎 |
| V3.4 | 2026-02-04 | 海纳百川因子库 |
| V3.3 | 2026-02-04 | 工业级因子分析 |
| V3.2 | 2026-02-04 | 动态因子挖掘系统 |
| V3.1 | 2026-02-04 | 动态智能框架 |
| V3.0 | 2026-02-04 | 全景数据层 |
| V2.9 | 2026-02-04 | 多Agent辩论系统 |
| V2.8 | 2026-02-04 | 风险管理模块 |
| V2.7 | 2026-02-04 | 持久化数据存储 |

---

## ⚠️ 免责声明

本项目生成的所有分析报告和投资建议**仅供参考**，不构成任何投资建议。投资有风险，入市需谨慎。使用者应自行承担投资决策的全部责任。

---

<div align="center">

**Built with ❤️ by Maxwell & Manus AI**

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

</div>
