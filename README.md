# 🚀 Quant-Investor V7.0 (Unified + MacroRiskTerminal V6.3)

<div align="center">

**AI驱动的量化投资框架 + 多市场宏观风控终端**

*数据驱动 · 分层解耦 · AI因子工程 · 宏观风控 · 完全透明化*

[![Version](https://img.shields.io/badge/Version-7.0-blue.svg)](https://github.com/alpha-mw/myQuant)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 项目简介

**Quant-Investor** 是一个整合了量化分析与AI深度思辨的智能投资研究平台。**V7.0 统一版本**在V6.x基础上，实现了历史版本的大一统整合，并引入了完全透明化的多市场宏观风控终端。

### V7.0 核心升级

```
量化投资分析 + 宏观风控终端 = 完整投资决策体系
     ↓                    ↓
  数据/因子/模型      货币政策/经济增长
  决策/风控          估值/通胀/情绪
```

**三大核心特性：**

1. **🔥 统一版本 (Unified v7.0)** - 整合 V2.7~V6.0 所有历史版本功能
2. **🌍 宏观风控终端 (MacroRiskTerminal V6.3)** - 第0层风控，多市场适配
3. **📊 完全透明化** - 所有分析步骤可追溯、可验证

---

## ✨ V7.0 架构

### 第一层：量化投资分析 (Unified v7.0)

| 层级 | 模块 | 核心能力 |
|:---|:---|:---|
| **数据层** | UnifiedDataLayer | yfinance/Tushare多源支持；自动数据获取 |
| **因子层** | UnifiedFactorLayer | 7大基础因子（动量/波动率/均值回归/成交量） |
| **模型层** | UnifiedModelLayer | XGBoost/LightGBM/RandomForest集成 |
| **决策层** | UnifiedDecisionLayer | 股票排名与投资建议生成 |
| **风控层** | UnifiedRiskLayer | 组合优化（最大夏普）与风险评估 |

### 第二层：宏观风控终端 (MacroRiskTerminal V6.3)

**第0层风控 - 市场趋势判断**

| 市场 | 模块数 | 核心指标 |
|:---|:---:|:---|
| **A股 (CN)** | 4大模块 | 两融余额、GDP、巴菲特指标、CPI/PPI、M1-M2剪刀差、M2增速、社融 |
| **美股 (US)** | 5大模块 | 联邦基金利率、美联储资产负债表、GDP、失业率、巴菲特指标、Shiller PE、CPI/PPI、核心PCE、国债收益率曲线、VIX、消费者信心 |

**综合风控信号：**
- 🔴 高风险 → ≤30%仓位，防御为主
- 🟡 中风险 → 30%-60%仓位，精选个股
- 🟢 低风险 → 60%-90%仓位，积极布局
- 🔵 极低风险 → 80%-100%仓位，逆向布局

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/alpha-mw/myQuant.git
cd myQuant

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install pandas numpy yfinance scipy
pip install xgboost lightgbm scikit-learn statsmodels
pip install tushare  # A股数据源
```

### Tushare 配置 (A股分析必需)

```python
# 在代码中设置（或设置环境变量）
import tushare as ts

token = "your_tushare_token"
ts.set_token(token)
pro = ts.pro_api(token)
pro._DataApi__token = token
pro._DataApi__http_url = 'http://lianghua.nanyangqiankun.top'
```

### 使用示例

#### 方式1: 一行代码分析 (推荐)

```python
from scripts.unified.unified_tushare import analyze_with_tushare

# A股分析
results = analyze_with_tushare(
    market="CN",
    stocks=["000001.SZ", "600000.SH", "000858.SZ"],
    lookback_years=1.0,
    verbose=True
)

print(results['final_recommendation'])
# 输出: 🟡 中风险 | 控制仓位，精选个股
```

#### 方式2: 美股分析

```python
from scripts.unified.unified_transparent import analyze_transparent

results = analyze_transparent(
    market="US",
    stocks=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    lookback_years=0.5,
    verbose=True
)

print(results['final_recommendation'])
```

#### 方式3: 仅运行宏观风控

```python
from scripts.unified.macro_terminal_tushare import create_terminal

# 创建宏观风控终端
terminal = create_terminal("CN")  # 或 "US"

# 生成报告
report = terminal.generate_risk_report()
markdown = terminal.format_report_markdown(report)

print(markdown)
```

---

## 📁 代码结构

```
myQuant/
├── README.md                           # 本文件
├── scripts/
│   ├── unified/                        ⭐ V7.0 统一版本
│   │   ├── unified_tushare.py          # Tushare优先集成入口
│   │   ├── unified_transparent.py      # 完全透明化版本
│   │   ├── macro_terminal_tushare.py   # Tushare优先宏观终端
│   │   ├── macro_terminal_transparent.py # 透明化宏观终端
│   │   ├── MACRO_RISK_GUIDE.md         # 宏观风控指标体系文档
│   │   ├── FINAL_SUMMARY.md            # V7.0 功能总结
│   │   ├── data_layer/                 # 数据层
│   │   ├── factor_layer/               # 因子层
│   │   ├── model_layer/                # 模型层
│   │   ├── decision_layer/             # 决策层
│   │   ├── risk_layer/                 # 风控层
│   │   └── pipeline/                   # 流水线
│   │
│   ├── v6.1/                           # V6.1 AI因子挖掘
│   ├── v6.0/                           # V6.0 大一统框架
│   ├── v5.0/                           # V5.0 工业级量化
│   └── v2.7~v4.1/                      # 历史版本
│
└── skill/
    └── SKILL.md                        # Manus技能定义
```

---

## 📊 分析示例

### 输入
```python
market = "CN"
stocks = ["000001.SZ", "600000.SH", "000858.SZ"]
lookback_years = 0.5
```

### 输出

```
🎯 综合结论
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
综合信号: 🟡 中风险
仓位建议: 30%-60% 仓位
策略调整: 控制仓位，精选个股

📊 量化分析
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 分析标的: 3只股票 (平安银行、浦发银行、五粮液)
• 有效因子: 7个 (momentum_5d/20d/60d, volatility_20d, ma_bias_20d等)
• 模型排名: 3只 (基于动量+均值回归评分)
• 组合配置: 等权重配置

🌍 宏观风控 (Tushare实时数据)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模块1 资金杠杆与情绪      🟡 偏冷
  └─ 两融余额             0.01万亿 (历史低位)
  └─ 两融/流通市值比      0.01% (极度冷清)

模块2 经济景气度          🟡 中速增长
  └─ GDP同比增速          5.0%

模块3 整体估值锚          🟢 合理区间
  └─ 巴菲特指标           92.2% (80-100%合理)

模块4 通胀与货币          🔴 注意
  └─ PPI同比              -1.9% (下行)
  └─ M1-M2剪刀差          -4.7% (严重定期化)
  └─ M2增速               8.5% (适度)

综合信号计算:
  步骤1: 收集模块信号 → 黄:3, 绿:1, 红:1
  步骤2: 应用规则 → 1个红色 = 中风险
  结论: 🟡 中风险
```

---

## 🔧 高级用法

### 自定义分析

```python
from scripts.unified.unified_tushare import UnifiedTushare

analyzer = UnifiedTushare(
    market="CN",
    stock_pool=["000001.SZ", "600000.SH"],
    lookback_years=1.0,
    enable_macro=True,
    verbose=True
)

results = analyzer.run()

# 生成完整报告
report_md = analyzer.generate_full_report(results)
with open('my_report.md', 'w') as f:
    f.write(report_md)
```

### 扩展新市场

```python
from scripts.unified.macro_terminal_tushare import MacroRiskTerminalBase, ModuleResult

class HKMacroRiskTerminal(MacroRiskTerminalBase):
    MARKET = "HK"
    MARKET_NAME = "港股"
    
    def get_modules(self) -> List[ModuleResult]:
        modules = []
        modules.append(self._analyze_exchange_rate())  # 联系汇率
        modules.append(self._analyze_stock_connect())  # 港股通
        modules.append(self._analyze_hangseng())       # 恒生指数
        return modules
```

---

## 📈 版本演进

| 版本 | 发布日期 | 核心特性 |
|:---|:---|:---|
| **V7.0** | 2026-02-25 | **统一版本 + 宏观风控** - 整合V2.7~V6.0，多市场宏观风控终端，Tushare优先，完全透明化 |
| V6.1 | 2026-02-25 | AI因子挖掘增强 - 整合Qlib/TA-Lib/WorldQuant，遗传编程、Transformer、LLM因子 |
| V6.0 | 2026-02-06 | 大一统框架 - 分层解耦，融合V2.7~V5.0全部能力 |
| V5.0 | 2026-02-06 | 工业级量化框架 - 全面升级量化能力 |
| V4.1 | 2026-02-05 | 基准对比升级版 |
| V4.0 | 2026-02-04 | 统一主流水线 |
| V3.6 | 2026-02-04 | 多LLM支持 |
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

## 🎯 核心特性

| 特性 | 描述 | 状态 |
|:---|:---|:---:|
| **统一版本** | 整合V2.7~V6.0所有功能 | ✅ |
| **多市场适配** | CN(4模块)/US(5模块)/可扩展 | ✅ |
| **Tushare优先** | A股数据源首选Tushare | ✅ |
| **宏观风控** | 第0层风控，市场趋势判断 | ✅ |
| **完全透明化** | 所有步骤可追溯、可验证 | ✅ |
| **自动市场检测** | 根据代码自动识别市场 | ✅ |
| **综合信号** | 四档信号+仓位建议+策略 | ✅ |
| **可扩展架构** | 继承基类即可添加市场 | ✅ |

---

## ⚠️ 免责声明

本项目生成的所有分析报告和投资建议**仅供参考**，不构成任何投资建议。投资有风险，入市需谨慎。使用者应自行承担投资决策的全部责任。

---

<div align="center">

**Built with ❤️ by Maxwell**

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

</div>
