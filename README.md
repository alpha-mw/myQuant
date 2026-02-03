# MyQuant - AI量化投资工具箱

一个端到端的AI量化投资框架，融合了统计学、贝叶斯推断、现代投资组合理论、投资大师智慧和大型语言模型增强分析。

**当前版本**: V2.8 (风险管理模块)
**核心理念**: 全球化一手数据驱动 + LLM深度分析 + 工业级框架设计 + 全方位风险管理

---

## 🎯 项目特色

- ✅ **全球化数据**：同时支持A股和美股市场，覆盖宏观、行业、个股全维度数据
- ✅ **一手数据驱动**：直接从FRED、Tushare、Finnhub、yfinance等专业数据源获取原始数据
- ✅ **全方位风险管理 (V2.8 NEW)**：内置专业风险管理模块，提供超过20种风险指标计算、因子风险分解、情景分析和压力测试。
- ✅ **持久化存储**：内置持久化数据存储系统，一次下载，永久使用，极大提升数据获取效率。
- ✅ **科学严谨**：基于统计显著性检验，避免过拟合和数据挖掘偏差
- ✅ **理论完备**：整合现代投资组合理论(MPT)、贝叶斯推断、因果推断
- ✅ **AI增强**：支持ChatGPT、Gemini等大模型深度分析，多Agent协作
- ✅ **工业级架构**：借鉴微软Qlib框架设计，表达式引擎、特征缓存、增强回测

---

## 🆕 V2.8 新特性：专业风险管理模块

V2.8版本引入了全新的**风险管理模块**，让您不仅关注收益，更能深刻理解策略的风险暴露和潜在弱点，从风险的角度全面评估投资策略。

### 核心特性

| 特性 | 描述 |
|:---|:---|
| **全面风险度量** | 计算超过20种业界标准的风险指标，包括波动率、回撤、Sharpe、Sortino、Calmar、VaR、CVaR等。 |
| **因子风险分解** | 基于Barra模型思想，将策略风险分解为**系统性风险**和**特异性风险**，并量化策略在各风格因子上的暴露。 |
| **风险归因分析** | 精确计算每个因子、每个资产对组合总风险的贡献度，洞察风险来源。 |
| **情景分析与压力测试** | 模拟不同市场情景和极端压力下策略的预期表现。 |
| **自动化风险报告** | 一键生成专业的、可视化的综合风险分析报告。 |

---

## 📁 项目结构

```
myquant/
└── scripts/                      # quant-investor技能核心脚本
    ├── v2.8/                   # V2.8 风险管理模块 (NEW)
    │   └── risk_management/
    │       ├── risk_metrics.py
    │       ├── factor_risk.py
    │       └── risk_manager.py
    ├── v2.7/                   # V2.7 持久化数据存储
    ├── v2.6/                   # V2.6 美股宏观数据层
    ├── v2.5/                   # V2.5 A股数据层
    └── ...
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install tushare akshare yfinance pandas numpy pyarrow requests scipy
```

### 2. 运行示例 (V2.8)

使用新的`RiskManager`对您的策略进行全面的风险体检。

```python
import sys
import numpy as np
import pandas as pd

sys.path.append("scripts/v2.8/risk_management")
from risk_manager import RiskManager

# --- 准备数据 ---
np.random.seed(42)
n_days = 504
dates = pd.date_range('2024-01-01', periods=n_days, freq='B')
strategy_returns = pd.Series(np.random.normal(0.0008, 0.015, n_days), index=dates)
benchmark_returns = pd.Series(np.random.normal(0.0004, 0.012, n_days), index=dates)

# --- 开始分析 ---
rm = RiskManager(risk_free_rate=0.02)
report = rm.generate_comprehensive_report(
    returns=strategy_returns,
    benchmark_returns=benchmark_returns,
    strategy_name="我的第一个量化策略"
)
print(report)
```

---

## 📖 版本演进

| 版本 | 核心特性 | 发布时间 |
|:---|:---|:---|
| **V2.8** | **风险管理模块** | 2026-02-02 |
| V2.7 | **持久化数据存储系统** | 2026-02-02 |
| V2.6 | 美股宏观数据层 | 2026-02-01 |
| V2.5 | A股一手数据驱动分析框架 | 2026-01-31 |
| V2.4 | LLM增强量化分析框架 | 2026-01-30 |

---

## ⚠️ 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。

---

**如果这个项目对你有帮助，请给个⭐️Star支持一下！** 🚀
