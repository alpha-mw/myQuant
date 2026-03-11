# 🚀 Quant-Investor V8.0 - 专业级量化投资平台

<div align="center">

**七层架构 · Multi-LLM集成裁判 · Alpha挖掘 · Walk-Forward回测**

[![Version](https://img.shields.io/badge/Version-8.0.0-blue.svg)](https://github.com/alpha-mw/myQuant)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-59%20passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-60%25-yellow.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**七层架构 · 事件驱动 · 微服务 · 全球多市场 · 实时风控**

</div>

---

## 📖 项目简介

**Quant-Investor V8.0** 是在 V7 六层架构基础上进一步演进的七层智能投资研究平台。新增第7层 **Multi-LLM 集成裁判**（Claude + GPT-4o + DeepSeek + Gemini 四大模型并行），同时引入系统性 Alpha 挖掘和 Walk-Forward 组合回测，并完成了全面的代码质量提升。

### 核心创新：七层架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     Quant-Investor V8.0                          │
│                        量化投资架构                               │
├─────────────────────────────────────────────────────────────────┤
│  第1层 │ 数据层 (Data Layer)                                      │
│        │ 多源数据：OHLCV / 基本面 / 宏观数据 / 资金流              │
│        │ 数据清理：去极值 / 缺失值处理 / 标准化                    │
├─────────────────────────────────────────────────────────────────┤
│  第2层 │ 因子层 (Factor Layer)                                    │
│        │ 因子计算 → IC分析 / 分层回测 / 换手率分析 → 因子筛选      │
│        │ 多维度因子：动量/波动率/均值回归/成交量/基本面            │
├─────────────────────────────────────────────────────────────────┤
│  第3层 │ 模型层 (Model Layer)                                     │
│        │ ML模型：Random Forest / XGBoost / SVM / LSTM             │
│        │ 时序交叉验证 / 模型集成 / 特征重要性排序                  │
├─────────────────────────────────────────────────────────────────┤
│  第4层 │ 宏观层 (Macro Layer) - 第0层风控                         │
│        │ 市场趋势判断：货币政策/经济增长/估值/通胀/情绪            │
│        │ 多市场适配：CN(4模块) / US(5模块) / 可扩展               │
├─────────────────────────────────────────────────────────────────┤
│  第5层 │ 风控层 (Risk Layer)                                      │
│        │ 组合风控：波动率控制/仓位管理/止损止盈/压力测试           │
│        │ 风险分解：Barra风格因子风险分解                          │
├─────────────────────────────────────────────────────────────────┤
│  第6层 │ 决策层 (Decision Layer)                                  │
│        │ LLM多模型多空辩论 → 具体投资建议                         │
│        │ 5个专业模型：财务/行业/宏观/技术/风险                     │
├─────────────────────────────────────────────────────────────────┤
│  第7层 │ 集成裁判层 (Ensemble Layer) ★ V8 新增                    │
│        │ Claude + GPT-4o + DeepSeek + Gemini 四大LLM并行          │
│        │ 置信度加权投票 · 分歧度风险信号 · 结构化执行报告          │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ 核心特性

### ★ V8.0 新增：第7层 Multi-LLM 集成裁判

**四大独立 LLM 并行裁判，消除单模型偏差：**

| 模型 | 角色定位 | 权重 |
|:---|:---|:---:|
| **Claude (Anthropic)** | 整体战略分析师 | 0.35 |
| **GPT-4o (OpenAI)** | 量化基本面分析师 | 0.30 |
| **DeepSeek** | 中国市场专家 | 0.20 |
| **Gemini (Google)** | 技术面+替代数据 | 0.15 |

**核心机制：**
- 同一量化数据发送给四个独立 LLM，角色各异
- 基于置信度的加权投票聚合
- **分歧度本身作为风险信号**（高分歧 = 不确定性高，自动降仓）
- 传统量化结论 + LLM 集成 → 最终裁决 + 分步执行计划

### ★ V8.0 新增：系统性 Alpha 挖掘

三层 Alpha 发掘框架（`alpha_mining.py`）：

| 层级 | 方法 | 说明 |
|:---|:---|:---|
| **Layer A** | 因子库 | 50+ 预定义系统性因子（动量/价值/质量/低波/成长） |
| **Layer B** | 遗传搜索 | 遗传算法组合因子基元，进化出新因子 |
| **Layer C** | LLM 头脑风暴 | 大模型提出文字逻辑 → 量化成因子 |

验证流程：IC/IR → 信息衰减 → 因子正交化 → 容量/换手分析

### ★ V8.0 新增：Walk-Forward 组合回测

- 滚动窗口训练，无未来数据泄露
- 自动报告：年化收益、夏普比率、最大回撤、卡玛比率
- 与基准（沪深300/标普500）对比

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

---

## 🔧 代码质量改进 (2026-03)

本轮代码质量提升主要包含以下内容：

### 1. 集中化日志管理 (`logging_config.py`)
- 使用 **loguru** 替代分散的 `print` / `logging.getLogger`
- 统一日志格式：时间戳 | 级别 | 模块:函数:行号
- 自动按日轮转，保留30天日志
- 独立 error.log 文件，便于问题排查

### 2. 自定义异常体系 (`exceptions.py`)
- 替代裸 `except Exception` 捕获
- 分层异常：`DataError` / `FactorError` / `ModelError` / `RiskError` / `LLMError`
- 携带上下文信息（来源、字段、期望类型等），定位问题更快

### 3. 关键性能修复
- **O(n²) IC计算修复**：`factor_analyzer.py` 中滚动IC计算从逐行循环改为向量化操作，大数据集性能提升 10-100x
- 类型注解补全，提升 IDE 推断精度与代码可读性

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
pip install -r requirements.txt
```

### API配置

```bash
# 设置多个LLM API key（第7层集成裁判）
export OPENAI_API_KEY="sk-your-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export DEEPSEEK_API_KEY="sk-your-deepseek-key"
export GOOGLE_API_KEY="your-gemini-key"

# 设置Tushare token (A股数据必需)
export TUSHARE_TOKEN="your-tushare-token"
```

### 使用示例

#### 方式1: 命令行一键分析 (推荐)

```bash
# V8 七层完整分析
python scripts/unified/quant_investor_v8.py \
    --stocks 000001.SZ 600519.SH 000858.SZ \
    --market CN \
    --capital 1000000 \
    --risk-level 中等
```

#### 方式2: Python API

```python
from scripts.unified.quant_investor_v8 import QuantInvestorV8

analyzer = QuantInvestorV8(
    stock_pool=["000001.SZ", "600519.SH"],
    market="CN",
    total_capital=1_000_000,
    risk_level="中等",
    enable_macro=True,
    enable_backtest=True,
    verbose=True
)

result = analyzer.run()
print(result.final_report)   # 完整 Markdown 报告，含执行步骤
```

#### 方式3: 仅运行宏观风控

```python
from scripts.unified.macro_terminal_tushare import create_terminal

terminal = create_terminal("CN")  # 或 "US"
report = terminal.generate_risk_report()
print(terminal.format_report_markdown(report))
```

#### 方式4: 独立 Alpha 挖掘

```python
from scripts.unified.alpha_mining import AlphaMiningEngine

engine = AlphaMiningEngine(price_data=df)
result = engine.run_full_mining()

for factor in result.top_factors[:5]:
    print(f"{factor.name}: IC={factor.ic_mean:.3f}, IR={factor.ir:.2f}")
```

---

## 📁 代码结构

```
myQuant/
├── README.md                              # 本文件
├── requirements.txt
├── pyproject.toml
├── tests/                                 # 单元测试（59 passing）
│   └── unit/
│       ├── test_backtest.py
│       ├── test_data_layer.py
│       ├── test_factor_layer.py
│       ├── test_factor_neutralizer.py
│       ├── test_market_impact.py
│       ├── test_model_layer.py
│       ├── test_risk_management.py
│       └── test_var_cvar.py
├── scripts/
│   ├── unified/                           ⭐ V8.0 七层架构（当前主线）
│   │   ├── quant_investor_v8.py           # 主入口：七层流水线
│   │   ├── multi_llm_ensemble.py          # 第7层：四大LLM集成裁判
│   │   ├── investment_report.py           # 结构化执行报告生成
│   │   ├── alpha_mining.py                # 系统性Alpha挖掘（三层）
│   │   ├── portfolio_backtest.py          # Walk-Forward 组合回测
│   │   ├── multi_model_debate.py          # 第6层：多模型多空辩论
│   │   ├── decision_layer.py              # 第6层：决策层封装
│   │   ├── advanced_risk_metrics.py       # 第5层：高级风险指标
│   │   ├── macro_terminal_tushare.py      # 第4层：宏观风控终端
│   │   ├── enhanced_model_layer.py        # 第3层：模型层
│   │   ├── factor_analyzer.py             # 第2层：因子层（O(n²)已修复）
│   │   ├── factor_neutralizer.py          # 因子中性化
│   │   ├── enhanced_data_layer.py         # 第1层：数据层
│   │   ├── logging_config.py              # ★ 集中化日志管理
│   │   ├── exceptions.py                  # ★ 自定义异常体系
│   │   ├── logger.py                      # Logger工厂
│   │   ├── config.py                      # 全局配置
│   │   ├── cache_manager.py               # 缓存管理
│   │   ├── di_container.py                # 依赖注入容器
│   │   ├── event_bus.py                   # 事件总线
│   │   ├── backtest_engine.py             # 回测引擎
│   │   ├── market_impact.py               # 市场冲击模型
│   │   ├── compliance_framework.py        # 合规框架
│   │   └── archive/                       # 历史版本归档（不建议使用）
│   │       ├── README.md                  # 归档说明
│   │       ├── quant_investor_v7.py       # V7.0 六层原版
│   │       ├── quant_investor_v71.py      # V7.1 过渡版
│   │       ├── quant_investor_v72.py      # V7.2 健壮版
│   │       ├── macro_terminal_v63.py      # V6.3 宏观终端
│   │       └── macro_terminal_transparent.py  # 实验性版本
│   └── v4.0/ ~ v6.0/                      # 早期版本目录（历史参考）
└── skill/                                 # Manus技能定义
    ├── SKILL.md
    └── references/                        # 参考文档
```

---

## 📊 分析示例

### V8 完整七层分析输出

```
================================================================================
Quant-Investor V8.0 七层架构开始执行
版本: 8.0.0-seven-layer
市场: CN
股票池: ['000001.SZ', '600519.SH', '000858.SZ']
================================================================================

[10:30:00] [Layer1] 数据层完成: 获取 3 只股票数据
[10:30:08] [Layer2] 因子层完成: 7 个有效因子 (IC均值 0.051)
[10:30:15] [Layer3] 模型层完成: 4 个模型，集成预测 +2.3%
[10:30:18] [Layer4] 宏观信号: 🟡 中风险
[10:30:20] [Layer5] 风控完成: warning (VaR 3.2%)
[10:30:40] [Layer6] 多空辩论完成: 000001.SZ 买入(75%)
[10:31:20] [Layer7] LLM集成裁判:
  Claude:   买入 (85%, 权重0.35)
  GPT-4o:   买入 (78%, 权重0.30)
  DeepSeek: 持有 (62%, 权重0.20)
  Gemini:   买入 (71%, 权重0.15)
  → 最终裁决: 买入 | 置信度 77% | 分歧度 0.18 (低)

================================================================================
七层流程执行完成 (总耗时: 91s)
================================================================================
```

---

## 🎯 技术亮点

| 特性 | 说明 | 状态 |
|:---|:---|:---:|
| **七层架构** | 分层解耦，每层可独立运行 | ✅ |
| **Multi-LLM集成** | 4个独立LLM并行，加权投票聚合 | ✅ |
| **分歧度风险信号** | LLM分歧高时自动降仓 | ✅ |
| **系统性Alpha挖掘** | 遗传算法 + LLM生成因子 | ✅ |
| **Walk-Forward回测** | 无未来数据泄露的滚动回测 | ✅ |
| **集中化日志** | loguru结构化日志，自动轮转 | ✅ |
| **分层异常体系** | 精确定位错误来源 | ✅ |
| **O(n²) IC修复** | 向量化操作，性能提升10-100x | ✅ |
| **多市场** | CN/US/HK可扩展 | ✅ |
| **时序CV** | 避免数据泄露的交叉验证 | ✅ |
| **风险分解** | Barra风格因子风险归因 | ✅ |
| **压力测试** | 极端行情模拟 | ✅ |

---

## 📈 版本演进

| 版本 | 发布日期 | 核心特性 |
|:---|:---|:---|
| **V8.0** | 2026-03-11 | **七层架构** - Multi-LLM集成裁判、Alpha挖掘、Walk-Forward回测、代码质量全面提升 |
| V7.2 | 2026-02-28 | 健壮六层框架 - 完整6层，稳定性增强 |
| V7.1 | 2026-02-27 | 修复Path导入问题 |
| V7.0 | 2026-02-26 | 六层架构 + 多模型辩论 |
| V6.3 | 2026-02-25 | 宏观风控终端 - 多市场适配 |
| V6.0 | 2026-02-06 | 大一统框架 - 分层解耦架构 |
| V5.0 | 2026-02-06 | 工业级量化框架 |
| V2.7~V4.1 | 2026-02-04 | 基础量化能力构建 |

> 历史版本代码见 [`scripts/unified/archive/`](scripts/unified/archive/)

---

## ⚠️ 免责声明

本项目生成的所有分析报告和投资建议**仅供参考**，不构成任何投资建议。投资有风险，入市需谨慎。使用者应自行承担投资决策的全部责任。

---

<div align="center">

**Built with ❤️ by Maxwell**

⭐ 如果这个项目对你有帮助，请给个Star支持一下！

</div>
