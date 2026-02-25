# 🚀 Quant-Investor V6.1

<div align="center">

**AI驱动的量化投资框架**

*数据驱动 · 分层解耦 · AI因子工程*

[![Version](https://img.shields.io/badge/Version-6.1-blue.svg)](https://github.com/alpha-mw/myQuant)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 项目简介

**Quant-Investor** 是一个整合了量化分析与AI深度思辨的智能投资研究平台。V6.1版本在V6.0大一统框架基础上，**全面升级因子层**，整合Qlib、TA-Lib、WorldQuant三大主流因子库，并引入AI驱动的因子挖掘引擎，实现从500+因子库到自动化因子发现的全链路智能化。

### 核心投资逻辑

```
数据获取与清洗 → AI因子工程(500+) → ML建模与信号生成 → 多LLM多Agent辩论 → 组合优化与风控
```

**V6.1核心升级**：因子层从"使用"到"创造"的革命性跃迁

1. **海纳百川因子库 (500+)**: 整合Qlib Alpha158/360、TA-Lib 50+指标、WorldQuant 101 Alphas
2. **AI创新因子引擎**: Transformer时序特征、LLM情绪因子、另类数据因子
3. **自动化因子挖掘**: 遗传编程（Genetic Programming）自动发现新因子表达式
4. **数据层增强**: 完整性检查、另类数据接口

---

## ✨ V6.1 分层架构

| 层级 | 模块 | 核心能力 |
|:---|:---|:---|
| **第1层** | 数据层 (UnifiedDataLayer) | yfinance/Tushare/FRED数据源；SQLite+Parquet持久化；去极值/补缺失/标准化；智能样本扩充；**(V6.1)完整性检查+另类数据接口** |
| **第2层** | 因子层 (UnifiedFactorLayer) | **(V6.1)500+因子库**（Qlib/TA-Lib/WorldQuant）；**AI创新因子**（Transformer/LLM情绪/另类数据）；**遗传编程因子挖掘**；IC/IR验证；多因子综合评分 |
| **第3层** | 模型层 (UnifiedModelLayer) | XGBoost/LightGBM/RandomForest；加权集成预测；特征重要性分析 |
| **第4层** | 决策层 (UnifiedDecisionLayer) | 4大LLM适配器；5专家Agent多轮辩论；投资评级生成 |
| **第5层** | 风控层 (UnifiedRiskLayer) | 最大夏普/风险平价/最小方差优化；VaR/CVaR/压力测试；Alpha/Beta分析 |

### V6.1 因子层革命性增强

**三大因子库整合**：

| 因子库 | 因子数量 | 核心特性 |
|--------|---------|----------|
| Qlib | 158-360 | Alpha158/Alpha360，工业级量化因子 |
| TA-Lib | 50+ | 技术分析指标（MACD/RSI/布林带等） |
| WorldQuant | 101 | WorldQuant 101 Alphas精选 |

**AI驱动因子创新**：

- **Transformer时序因子**：使用Transformer提取深度时序模式作为新因子
- **LLM情绪因子**：分析财报、新闻文本，生成情绪得分因子
- **遗传编程挖掘**：自动发现新的有效因子表达式

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/alpha-mw/myQuant.git
cd myQuant

# 安装依赖
pip install pandas numpy yfinance scipy
pip install xgboost lightgbm scikit-learn statsmodels
pip install qlib ta-lib  # V6.1新增
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
sys.path.insert(0, "scripts/v6.1")

from pipeline.master_pipeline import MasterPipelineV6

# V6.1: 启用所有新功能
pipeline = MasterPipelineV6(
    market="US",
    stock_pool=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    enable_factor_mining=True,  # 开启遗传编程因子挖掘
    enable_transformer_factors=True,  # 开启Transformer因子
    factor_libraries=["qlib", "talib", "worldquant"]  # 加载所有因子库
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
│   ├── v6.1/                       # V6.1 AI因子挖掘增强版
│   │   ├── data_layer/             # 统一数据层
│   │   ├── factor_layer/           # 统一因子层 (V6.1全面升级)
│   │   │   ├── unified_factor_layer.py
│   │   │   ├── factor_libraries/   # 新增：因子库模块
│   │   │   │   ├── qlib_factors.py
│   │   │   │   ├── talib_factors.py
│   │   │   │   ├── worldquant_factors.py
│   │   │   │   └── alternative_factors.py
│   │   │   └── factor_mining/      # 新增：因子挖掘模块
│   │   │       └── genetic_miner.py
│   │   ├── model_layer/            # 统一模型层
│   │   ├── decision_layer/         # 统一决策层
│   │   ├── risk_layer/             # 统一风控层
│   │   └── pipeline/               # 统一流水线
│   ├── v6.0/                       # V6.0 大一统版本
│   ├── v5.0/                       # V5.0 工业级量化框架
│   └── v2.7~v4.1/                  # 历史版本
```

---

## 📈 版本演进

| 版本 | 发布日期 | 核心特性 |
|:---|:---|:---|
| **V6.1** | 2026-02-25 | **AI因子挖掘增强** - 整合Qlib/TA-Lib/WorldQuant，引入遗传编程、Transformer、LLM因子 |
| V6.0 | 2026-02-06 | **大一统框架** - 分层解耦，融合V2.7~V5.0全部能力，智能样本扩充 |
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

**Built with ❤️ by Maxwell**

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

</div>
