# 🚀 Quant-Investor V7.0 - 六层智能量化投资框架

<div align="center">

**AI驱动的六层量化投资框架 · LLM多模型多空辩论 · 完全透明化**

*数据层 → 因子层 → 模型层 → 宏观层 → 风控层 → 决策层*

[![Version](https://img.shields.io/badge/Version-7.0.0-blue.svg)](https://github.com/alpha-mw/myQuant)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 📖 项目简介

**Quant-Investor V7.0** 是一个融合传统量化分析与AI深度思辨的六层智能投资研究平台。通过分层解耦架构，将数据获取、因子挖掘、模型预测、宏观风控、组合风控、AI决策六大模块有机结合，实现从原始数据到最终投资建议的完整闭环。

### 核心创新：六层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Quant-Investor V7.0                          │
│                        六层量化投资架构                            │
├─────────────────────────────────────────────────────────────────┤
│  第6层 │ 决策层 (Decision Layer)                                  │
│        │ LLM多模型多空辩论 → 具体投资建议                         │
│        │ 5个专业模型：财务/行业/宏观/技术/风险                     │
├─────────────────────────────────────────────────────────────────┤
│  第5层 │ 风控层 (Risk Layer)                                      │
│        │ 组合风控：波动率控制/仓位管理/止损止盈/压力测试           │
│        │ 风险分解：Barra风格因子风险分解                          │
├─────────────────────────────────────────────────────────────────┤
│  第4层 │ 宏观层 (Macro Layer) - 第0层风控                         │
│        │ 市场趋势判断：货币政策/经济增长/估值/通胀/情绪            │
│        │ 多市场适配：CN(4模块) / US(5模块) / 可扩展               │
├─────────────────────────────────────────────────────────────────┤
│  第3层 │ 模型层 (Model Layer)                                     │
│        │ ML模型：Random Forest / XGBoost / SVM / LSTM             │
│        │ 时序交叉验证 / 模型集成 / 特征重要性排序                  │
├─────────────────────────────────────────────────────────────────┤
│  第2层 │ 因子层 (Factor Layer)                                    │
│        │ 因子计算 → IC分析 / 分层回测 / 换手率分析 → 因子筛选      │
│        │ 多维度因子：动量/波动率/均值回归/成交量/基本面            │
├─────────────────────────────────────────────────────────────────┤
│  第1层 │ 数据层 (Data Layer)                                      │
│        │ 多源数据：OHLCV / 基本面 / 宏观数据 / 资金流              │
│        │ 数据清理：去极值 / 缺失值处理 / 标准化                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ 核心特性

### 🎯 第6层：LLM多模型多空辩论系统

**5个专业分析模型，每个模型分别输出看多/看空观点：**

| 模型 | 分析维度 | 看多示例 | 看空示例 |
|:---|:---|:---|:---|
| **📊 财务分析模型** | ROE/估值/现金流 | ROE 18%、PE 15倍、FCF充裕 | 应收账款增加、资本支出上升 |
| **🏭 行业研究模型** | 生命周期/竞争格局/护城河 | CAGR 15%、市占率30%、技术壁垒 | 竞争加剧、新进入者威胁 |
| **🌍 宏观分析模型** | 经济周期/货币政策/通胀 | 货币宽松、经济复苏、政策支持 | 通胀压力、加息风险、地缘冲突 |
| **📈 技术分析模型** | 趋势/支撑阻力/量价 | 突破前高、放量上涨、MACD金叉 | RSI超买、接近阻力位 |
| **⚠️ 风险评估模型** | 波动率/回撤/流动性 | 波动率可控、流动性充足 | 最大回撤25%、集中度风险 |

**决策输出：**
```json
{
  "decision": "买入",
  "confidence": 0.72,
  "position_size": 0.15,
  "target_price": 150.00,
  "stop_loss": 120.00,
  "time_horizon": "中期",
  "logic_chain": [
    "财务指标健康，ROE 18%，估值合理PE 15倍",
    "行业处于成长期，公司市占率30%龙头地位",
    "宏观环境中性偏正面，政策支持",
    "技术面上升趋势确立，资金流入",
    "风险可控，设置止损位保护"
  ],
  "model_consensus": {
    "财务分析模型": "bullish (75%)",
    "行业研究模型": "bullish (70%)",
    "宏观分析模型": "neutral (65%)",
    "技术分析模型": "bullish (60%)",
    "风险评估模型": "caution (65%)"
  }
}
```

### 🛡️ 第5层：组合风控系统

| 功能 | 说明 |
|:---|:---|
| **风险指标计算** | 年化波动率、最大回撤、VaR/CVaR、夏普比率、Beta/Alpha |
| **仓位管理** | 基于宏观信号和波动率的动态仓位调整 (🔴30% → 🟢80%) |
| **止损止盈** | 固定百分比 + ATR-based 跟踪止损 |
| **风险分解** | Barra风格因子风险分解 (系统性/特异性风险) |
| **压力测试** | 2008金融危机/2015股灾/2020疫情 极端行情模拟 |
| **风险预算** | 等风险贡献(ERC)资产配置策略 |

### 🌍 第4层：宏观风控终端

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

### 🤖 第3层：增强模型层

| 模型 | 适用场景 | 特点 |
|:---|:---|:---|
| **Random Forest** | 通用预测 | 抗过拟合、特征重要性可解释 |
| **XGBoost** | 大规模数据 | 高效、准确、支持正则化 |
| **SVM** | 小样本 | 高维数据处理、核技巧 |
| **LSTM** | 时序预测 | 捕捉长期依赖、适合价格预测 |

**特性：**
- 时序交叉验证 (TimeSeriesSplit)
- 模型集成 (加权平均)
- 特征重要性自动排序

### 📊 第2层：因子分析系统

| 分析维度 | 指标 |
|:---|:---|
| **IC分析** | IC均值、IC_IR、正IC比率、t统计量 |
| **分层回测** | 单调性检验、多空收益、换手率 |
| **相关性分析** | 因子间相关性、多重共线性检测 |
| **综合评分** | 收益能力、稳定性、换手率加权评分 |

**因子类型：**
- 动量因子 (5d/20d/60d)
- 波动率因子 (20d)
- 均值回归因子 (20d偏离度)
- 成交量因子
- 基本面因子 (ROE/PE/PB)

### 📥 第1层：增强数据层

| 数据类型 | 来源 | 说明 |
|:---|:---|:---|
| **OHLCV** | Tushare/yfinance | 日线数据，自动复权 |
| **基本面** | Tushare | ROE、PE、PB、营收、利润 |
| **资金流** | Tushare | 主力资金、散户资金、大单动向 |
| **宏观数据** | Tushare/FRED | GDP、CPI、利率、货币供应量 |

**数据清理：**
- 去极值 (Winsorization)
- 缺失值处理 (前向填充/插值)
- 标准化 (Z-score)

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
pip install pandas numpy yfinance scipy scikit-learn
pip install xgboost lightgbm tensorflow  # 模型层
pip install tushare  # A股数据源
```

### API配置

```bash
# 设置OpenAI API key (第6层LLM分析必需)
export OPENAI_API_KEY="sk-your-key-here"

# 可选：设置多个key避免rate limit
export OPENAI_API_KEYS="sk-key-1,sk-key-2,sk-key-3"

# 设置Tushare token (A股数据必需)
export TUSHARE_TOKEN="your-tushare-token"
```

### 使用示例

#### 方式1: 一行代码完整分析 (推荐)

```python
from scripts.unified.quant_investor_v7 import analyze

# 六层完整分析
result = analyze(
    market="CN",
    stocks=["000001.SZ", "600000.SH", "000858.SZ"],
    lookback_years=1.0,
    verbose=True
)

# 查看最终投资建议
print(result.decision_result.final_report)
```

**输出示例：**
```markdown
# 🎯 量化投资决策报告
**生成时间**: 2026-02-26 11:30:00

## 📊 市场展望
结构性机会，精选个股 | 宏观环境支持，可适当积极

## 💼 组合配置建议
| 标的 | 配置比例 |
|:---|:---:|
| 000001.SZ | 35.0% |
| 600000.SH | 30.0% |
| CASH | 35.0% |

## 📈 个股投资决策

### 🟢 000001.SZ (平安银行)
- **决策**: 买入
- **置信度**: 75%
- **建议仓位**: 15%
- **目标价**: ¥15.50
- **止损位**: ¥12.00
- **投资周期**: 中期

**决策逻辑**:
1. 财务指标健康，ROE 15%，估值合理PE 8倍
2. 银行业受益于货币政策宽松
3. 技术面突破前期整理平台
4. 股息率5%，防御属性强

**模型共识**:
- 财务分析模型: bullish (80%)
- 行业研究模型: bullish (75%)
- 宏观分析模型: bullish (70%)
- 技术分析模型: bullish (65%)
- 风险评估模型: neutral (60%)
```

#### 方式2: 仅运行宏观风控

```python
from scripts.unified.macro_terminal_tushare import create_terminal

# 创建宏观风控终端
terminal = create_terminal("CN")  # 或 "US"

# 生成报告
report = terminal.generate_risk_report()
print(terminal.format_report_markdown(report))
```

#### 方式3: 自定义六层流水线

```python
from scripts.unified.quant_investor_v7 import QuantInvestorV7

# 创建分析器
analyzer = QuantInvestorV7(
    market="CN",
    stock_pool=["000001.SZ", "600000.SH"],
    lookback_years=1.0,
    enable_macro=True,
    verbose=True
)

# 运行六层分析
result = analyzer.run()

# 访问各层结果
print(f"宏观信号: {result.macro_signal}")
print(f"风控等级: {result.risk_layer_result.risk_level}")
print(f"模型预测: {result.model_predictions.mean():.2%}")

# 查看LLM多模型辩论结果
for decision in result.decision_result.investment_decisions:
    print(f"{decision.symbol}: {decision.decision} (置信度{decision.confidence:.0%})")
```

---

## 📁 代码结构

```
myQuant/
├── README.md                              # 本文件
├── scripts/
│   └── unified/                           ⭐ V7.0 六层架构
│       ├── quant_investor_v7.py           # 主入口：六层流水线
│       ├── multi_model_debate.py          # 第6层：多模型多空辩论
│       ├── decision_layer.py              # 第6层：决策层封装
│       ├── risk_management_layer.py       # 第5层：风控层
│       ├── macro_terminal_tushare.py      # 第4层：宏观风控终端
│       ├── enhanced_model_layer.py        # 第3层：模型层
│       ├── factor_analyzer.py             # 第2层：因子层
│       ├── enhanced_data_layer.py         # 第1层：数据层
│       └── llm_rate_limiter.py            # LLM速率限制器
│
└── skill/
    └── SKILL.md                           # Manus技能定义
```

---

## 🔧 高级配置

### 速率限制配置

当使用OpenAI API时，为避免rate limit：

```python
from llm_rate_limiter import configure_rate_limiter

# 配置速率限制器
configure_rate_limiter(
    requests_per_minute=15,    # 每分钟15次
    min_interval=4.0,          # 最少4秒间隔
    max_retries=3,             # 最大重试3次
    api_keys=[                 # 多个API key轮询
        "sk-key-1",
        "sk-key-2",
        "sk-key-3"
    ]
)
```

### 扩展新市场

```python
from scripts.unified.macro_terminal_tushare import MacroRiskTerminalBase

class HKMacroRiskTerminal(MacroRiskTerminalBase):
    MARKET = "HK"
    MARKET_NAME = "港股"
    
    def get_modules(self):
        modules = []
        modules.append(self._analyze_exchange_rate())
        modules.append(self._analyze_stock_connect())
        return modules
```

---

## 📊 分析示例

### 完整六层分析输出

```
================================================================================
Quant-Investor V7.0 六层架构开始执行
版本: 7.0.0-six-layer
市场: CN
股票池: ['000001.SZ', '600000.SH', '000858.SZ']
================================================================================

[10:30:00] [Layer1] 【第1层】数据层 - 数据获取与清理
[10:30:05] [Layer1] 数据层完成: 获取 3 只股票数据

[10:30:05] [Layer2] 【第2层】因子层 - 因子计算与检验
[10:30:08] [Layer2] 因子层完成: 7 个有效因子

[10:30:08] [Layer3] 【第3层】模型层 - ML模型训练
[10:30:15] [Layer3] 模型层完成: 4 个模型训练完成

[10:30:15] [Layer4] 【第4层】宏观层 - 市场趋势判断
[10:30:18] [Layer4] 宏观信号: 🟡 中风险

[10:30:18] [Layer5] 【第5层】风控层 - 组合风控
[10:30:20] [Layer5] 风控完成: warning

[10:30:20] [Layer6] 【第6层】决策层 - LLM多模型多空辩论
[10:30:20] [DebateSystem] 开始多模型多空辩论: 000001.SZ
[10:30:24] [DebateSystem]   财务分析模型: bullish (置信度75%)
[10:30:28] [DebateSystem]   行业研究模型: bullish (置信度70%)
[10:30:32] [DebateSystem]   宏观分析模型: neutral (置信度65%)
[10:30:36] [DebateSystem]   技术分析模型: bullish (置信度60%)
[10:30:40] [DebateSystem]   风险评估模型: caution (置信度65%)
[10:30:40] [DecisionLayer] 决策: 买入 (置信度75%)

================================================================================
六层流程执行完成
================================================================================
```

---

## 🎯 技术亮点

| 特性 | 说明 | 状态 |
|:---|:---|:---:|
| **六层架构** | 分层解耦，每层可独立运行 | ✅ |
| **多模型辩论** | 5个专业模型多空观点碰撞 | ✅ |
| **深度研究** | 产品/竞争/行业/政策全方位分析 | ✅ |
| **完全透明** | 所有步骤可追溯、可验证 | ✅ |
| **速率限制** | 自动重试、指数退避、多key轮询 | ✅ |
| **模拟回退** | API不可用时自动使用模拟数据 | ✅ |
| **多市场** | CN/US/HK可扩展 | ✅ |
| **时序CV** | 避免数据泄露的交叉验证 | ✅ |
| **风险分解** | Barra风格因子风险归因 | ✅ |
| **压力测试** | 极端行情模拟 | ✅ |

---

## 📈 版本演进

| 版本 | 发布日期 | 核心特性 |
|:---|:---|:---|
| **V7.0** | 2026-02-26 | **六层架构 + 多模型辩论** - 数据/因子/模型/宏观/风控/决策六层闭环，LLM多空辩论 |
| V6.3 | 2026-02-25 | 宏观风控终端 - 多市场适配，第0层风控 |
| V6.1 | 2026-02-25 | AI因子挖掘 - 遗传编程、Transformer、LLM因子 |
| V6.0 | 2026-02-06 | 大一统框架 - 分层解耦架构 |
| V5.0 | 2026-02-06 | 工业级量化框架 |
| V2.7~V4.1 | 2026-02-04 | 基础量化能力构建 |

---

## ⚠️ 免责声明

本项目生成的所有分析报告和投资建议**仅供参考**，不构成任何投资建议。投资有风险，入市需谨慎。使用者应自行承担投资决策的全部责任。

---

<div align="center">

**Built with ❤️ by Maxwell**

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

</div>
