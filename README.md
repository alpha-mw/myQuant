# 🚀 Quant-Investor V5.0

<div align="center">

**一个工业级的AI量化投资平台**

*数据驱动 · 机器学习 · 风险管理*

[![Version](https://img.shields.io/badge/Version-5.0-blue.svg)](https://github.com/maxwelllee54/myQuant)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 项目简介

**Quant-Investor** 是一个整合了量化分析与AI深度思辨的智能投资研究平台。V5.0版本对量化能力进行了全面升级，涵盖数据清洗、特征工程、机器学习模型、完整因子体系、组合优化、回测和高级风险管理。

### 核心投资逻辑

```
数据清洗 → 因子工程 → 机器学习 → 多因子合成 → 组合优化 → 回测 & 风控
```

1. **数据驱动**：基于海量数据（价量、基本面、另类）进行分析。
2. **机器学习**：利用XGBoost、LSTM、Transformer等模型捕捉非线性关系，动态调整因子权重。
3. **风险管理**：采用Barra模型分解风险，GARCH预测波动率，并进行历史情景压力测试。

---

## ✨ V5.0 核心特性

### 统一化流程概览

| 阶段 | 核心能力 |
|:---|:---|
| **数据获取与清洗** | OHLCV/Tick/订单簿/基本面/另类数据；去极值/补缺失/标准化/复权 |
| **因子工程与筛选** | 500+因子（基本面/价量/宏观/另类）；IC/分层回测/换手率分析 |
| **机器学习建模** | XGBoost/LSTM/Transformer；时间序列交叉验证；自定义损失函数 |
| **多因子合成** | 非线性交互；动态权重；生成综合信号 |
| **组合优化与构建** | 均值-方差/风险平价/Black-Litterman |
| **回测与风险管理** | backtrader回测；Barra风险模型/GARCH/压力测试 |

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/maxwelllee54/myQuant.git
cd myQuant

# 安装依赖
pip install pandas numpy yfinance akshare tushare scipy
pip install xgboost lightgbm scikit-learn statsmodels
pip install backtrader
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

/quant-investor 分析我的持仓：AAPL 25%, MSFT 25%, GOOGL 20%
```

#### 方式2：Python代码调用

```python
import sys
sys.path.append("scripts/v5.0")

# 示例：使用高级风险管理器
from risk_management.advanced_risk_manager import AdvancedRiskManager
risk_manager = AdvancedRiskManager()
analysis = risk_manager.comprehensive_risk_analysis(returns, portfolio_value)
report = risk_manager.generate_risk_report(analysis)
print(report)
```

---

## 📈 版本演进

| 版本 | 发布日期 | 核心特性 |
|:---|:---|:---|
| **V5.0** | 2026-02-06 | **工业级量化框架** - 全面升级量化能力 |
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
