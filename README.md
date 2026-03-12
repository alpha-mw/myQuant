# Quant-Investor V9.0 - 专业级量化投资平台

<div align="center">

**九层架构 · Kronos基础模型 · 财务/新闻/情绪分析 · Multi-LLM集成裁判 · Alpha挖掘 · Walk-Forward回测**

[![Version](https://img.shields.io/badge/Version-9.0.0-blue.svg)](https://github.com/alpha-mw/myQuant)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-59%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 项目简介

**Quant-Investor V9.0** 是在 V8 七层架构基础上，集成 [Kronos 金融基础模型](https://github.com/shiyu-coder/Kronos)（AAAI 2026），并新增财务分析、新闻分析、市场情绪分析三大智能模块，构建完整的**九层全面智能投资研究平台**。

### 九层架构

```
┌──────────────────────────────────────────────────────────────────────┐
│  第9层 │ 多维智能融合层 (Intelligence Layer)            ★ V9 新增    │
│        │ 财务(Piotroski/Beneish/Altman/DCF/DuPont)                  │
│        │ + 新闻情感(AKShare/LLM事件检测)                            │
│        │ + 市场情绪(恐慌贪婪/资金流/广度)                           │
│        │ → 四维信号加权融合 + LLM综合摘要                           │
├──────────────────────────────────────────────────────────────────────┤
│  第8层 │ Kronos 金融基础模型层                          ★ V9 新增    │
│        │ 45个全球交易所·120亿K线预训练·零样本RankIC +93%            │
│        │ OHLCV离散分词 + 自回归Transformer预测                      │
│        │ 支持 kronos-mini(4M) / small(25M) / base(102M)             │
├──────────────────────────────────────────────────────────────────────┤
│  第7层 │ 集成裁判层 (Ensemble Layer)                    ★ V8 引入    │
│        │ Claude + GPT-4o + DeepSeek + Gemini 四大LLM并行            │
│        │ 置信度加权投票 · 分歧度风险信号 · 结构化执行报告            │
├──────────────────────────────────────────────────────────────────────┤
│  第6层 │ 决策层 (Decision Layer)                                     │
│        │ LLM多模型多空辩论 → 具体投资建议                           │
│        │ 5个专业角色：财务/行业/宏观/技术/风险                       │
├──────────────────────────────────────────────────────────────────────┤
│  第5层 │ 风控层 (Risk Layer)                                         │
│        │ VaR/CVaR · 压力测试 · Barra因子风险分解                    │
│        │ 动态仓位管理 · 止损止盈 · 风险预算                         │
├──────────────────────────────────────────────────────────────────────┤
│  第4层 │ 宏观层 (Macro Layer)                                        │
│        │ A股：货币/信用/增长/估值/情绪 (4大模块)                    │
│        │ 美股：利率/美联储/就业/估值/通胀 (5大模块)                 │
├──────────────────────────────────────────────────────────────────────┤
│  第3层 │ 模型层 (Model Layer)                                        │
│        │ Random Forest / XGBoost / SVM / LSTM 集成                  │
│        │ 时序交叉验证 · 特征重要性排序                               │
├──────────────────────────────────────────────────────────────────────┤
│  第2层 │ 因子层 (Factor Layer)                                       │
│        │ 50+ 因子计算 → IC/IR分析 → 因子筛选                        │
│        │ 动量/波动率/均值回归/成交量/基本面                          │
├──────────────────────────────────────────────────────────────────────┤
│  第1层 │ 数据层 (Data Layer)                                         │
│        │ Tushare Pro (主) + AKShare (辅) + yfinance (美股)          │
│        │ 数据清理：去极值/缺失值处理/标准化                          │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 核心特性

### V9 新增：Kronos 金融基础模型（第8层）

| 特性 | 描述 |
|:---|:---|
| **预训练规模** | 45个全球交易所，120亿条K线记录（AAAI 2026） |
| **零样本预测** | RankIC 比最佳 TSFM 提升 **93%** |
| **波动率预测** | MAE 降低 **9%** |
| **可用模型** | Kronos-mini(4M) / Kronos-small(25M) / Kronos-base(102M) |
| **降级模式** | Kronos 库未安装时自动使用 EWMA统计替代预测 |

### V9 新增：三大智能分析模块（第9层）

**财务分析引擎**：
- Piotroski F-Score（财务健康度 0-9分）
- Beneish M-Score（盈利操纵检测）
- Altman Z-Score（破产风险评估）
- 两阶段DCF估值（内在价值 + 安全边际）
- DuPont ROE三因素拆解

**新闻与替代数据分析**：
- AKShare / RSS / 新浪财经多源新闻获取
- LLM驱动情感分析（Claude-Haiku / GPT-4o-mini）
- 9类金融事件自动检测（盈利超预期/监管/并购/高管变动等）
- 新闻冲击评分量化

**市场情绪分析**：
- 恐慌贪婪指数（7维度加权，仿 CNN Fear & Greed）
- 技术情绪：RSI / MACD / 布林带 / 量比 / KDJ
- 资金流向：主力资金 / 北向资金 / 融资融券
- 市场广度：涨跌家数比 / 创新高新低
- 行业轮动阶段判断（早/中/晚周期/防御）
- 极端情绪逆向信号提醒

### V8 引入：第7层 Multi-LLM 集成裁判

| 模型 | 角色 | 权重 |
|:---|:---|:---:|
| Claude (Anthropic) | 整体战略分析师 | 0.35 |
| GPT-4o (OpenAI) | 量化基本面分析师 | 0.30 |
| DeepSeek | 中国市场专家 | 0.20 |
| Gemini (Google) | 技术面+替代数据 | 0.15 |

**分歧度作为风险信号**：各LLM意见高度分歧时自动降仓。

### 系统性 Alpha 挖掘（alpha_mining.py）

| 层级 | 方法 | 说明 |
|:---|:---|:---|
| Layer A | 因子库 | 50+ 预定义系统性因子 |
| Layer B | 遗传算法 | 进化搜索新因子组合 |
| Layer C | LLM头脑风暴 | 文字逻辑→量化因子 |

### Walk-Forward 组合回测（portfolio_backtest.py）

- 滚动窗口，无未来数据泄露
- 自动计算：年化收益/夏普/最大回撤/卡玛比率
- 与沪深300/标普500基准对比

---

## 快速开始

### 环境配置

```bash
git clone https://github.com/alpha-mw/myQuant.git
cd myQuant
pip install -r requirements.txt

# 可选：安装 Kronos 以使用原生基础模型预测
git clone https://github.com/shiyu-coder/Kronos
cd Kronos && pip install -e . && cd ..
```

### API 密钥配置

```bash
# LLM APIs（第7/9层使用）
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
export GOOGLE_API_KEY="..."

# 数据源（A股分析必需）
export TUSHARE_TOKEN="your-token"
```

### 使用示例

#### V9 完整九层分析（推荐）

```bash
# 命令行一键分析
python scripts/unified/quant_investor_v9.py \
    --stocks 000001.SZ 600519.SH 000858.SZ \
    --market CN \
    --capital 1000000 \
    --kronos-model kronos-small \
    --output report_v9.md

# 快速模式（禁用宏观层，加快速度）
python scripts/unified/quant_investor_v9.py \
    --stocks 600519.SH \
    --no-macro \
    --output quick_report.md
```

```python
from scripts.unified.quant_investor_v9 import QuantInvestorV9

analyzer = QuantInvestorV9(
    stock_pool=["000001.SZ", "600519.SH"],
    market="CN",
    total_capital=1_000_000,
    kronos_model="kronos-small",  # kronos-mini / kronos-small / kronos-base
    pred_len=20,                  # 预测未来20个交易日
    enable_financial_analysis=True,
    enable_news_analysis=True,
    enable_sentiment_analysis=True,
)
result = analyzer.run()
print(result.final_report)
```

#### V8 七层分析

```python
from scripts.unified.quant_investor_v8 import QuantInvestorV8

analyzer = QuantInvestorV8(
    stock_pool=["000001.SZ", "600519.SH"],
    market="CN",
    total_capital=1_000_000,
    enable_macro=True,
    enable_backtest=False,
)
result = analyzer.run()
print(result.final_report)
```

#### 单模块使用

```python
# Kronos 预测
from scripts.unified.kronos_predictor import KronosIntegrator
signal = KronosIntegrator("kronos-small").analyze_portfolio(
    {"600519.SH": df_price}, pred_len=20
)

# 财务分析
from scripts.unified.financial_analysis import FinancialAnalyzer
report = FinancialAnalyzer().full_analysis("600519.SH", "贵州茅台", df_financial)

# 新闻分析
from scripts.unified.news_analysis import NewsAnalyzer
result = NewsAnalyzer().analyze("600519.SH", "贵州茅台", days=7)

# 市场情绪
from scripts.unified.sentiment_analysis import MarketSentimentAnalyzer
report = MarketSentimentAnalyzer().analyze("600519.SH", "贵州茅台", price_df=df)

# 宏观风控终端（独立运行）
from scripts.unified.macro_terminal_tushare import create_terminal
terminal = create_terminal("CN")
print(terminal.format_report_markdown(terminal.generate_risk_report()))

# Alpha 挖掘
from scripts.unified.alpha_mining import AlphaMiner
miner = AlphaMiner(df, enable_genetic=True)
result = miner.run_full_mining()
```

---

## 代码结构

```
myQuant/
├── README.md
├── requirements.txt
├── pyproject.toml
├── tests/                                 # 单元测试（59 passing）
│   └── unit/
├── skill/                                 # 技能定义文档
│   ├── SKILL_V3.0.md                      # 最新版本（V9 架构）
│   ├── SKILL_V2.5.md                      # V8 版本
│   └── references/                        # 参考文档
└── scripts/
    └── unified/                           ⭐ 主代码目录
        │
        │  ── 入口文件 ──
        ├── quant_investor_v9.py           # 主入口：九层流水线
        ├── quant_investor_v8.py           # V8：七层流水线
        ├── quant_investor_v7.py           # V7：六层基础流水线
        │
        │  ── Layer 9：多维智能融合 ──
        ├── intelligence_layer.py          # Layer 8+9 编排器
        ├── financial_analysis.py          # 财务分析（Piotroski/Beneish/Altman/DCF/DuPont）
        ├── news_analysis.py               # 新闻&替代数据分析
        ├── sentiment_analysis.py          # 市场情绪（恐慌贪婪/技术/资金/广度）
        │
        │  ── Layer 8：Kronos 基础模型 ──
        ├── kronos_predictor.py            # Kronos K线预测（含统计降级模式）
        │
        │  ── Layer 7：Multi-LLM ──
        ├── multi_llm_ensemble.py          # 四大LLM集成裁判
        ├── investment_report.py           # 结构化执行报告
        │
        │  ── Layer 6：决策层 ──
        ├── decision_layer.py              # 决策层封装
        ├── multi_model_debate.py          # 多模型多空辩论
        │
        │  ── Layer 5：风控层 ──
        ├── risk_management_layer.py       # 组合风控
        ├── var_calculator.py              # VaR/CVaR 计算
        ├── stress_tester.py               # 压力测试
        ├── market_impact.py               # 市场冲击模型
        ├── advanced_risk_metrics.py       # 高级风险指标
        ├── risk_dashboard.py              # 风险可视化
        │
        │  ── Layer 4：宏观层 ──
        ├── macro_terminal_tushare.py      # 宏观风控终端（Tushare+AKShare）
        │
        │  ── Layer 3：模型层 ──
        ├── enhanced_model_layer.py        # ML模型集成
        ├── hyperparameter_tuner.py        # 超参数优化
        ├── shap_explainer.py              # SHAP 可解释性
        │
        │  ── Layer 2：因子层 ──
        ├── factor_analyzer.py             # 因子计算与IC分析
        ├── factor_neutralizer.py          # 因子中性化
        ├── alpha_mining.py                # 系统性Alpha挖掘（三层）
        ├── alpha158.py                    # Qlib Alpha158因子库
        │
        │  ── Layer 1：数据层 ──
        ├── enhanced_data_layer.py         # 多源数据采集与清理
        ├── batch_data_fetcher.py          # 批量数据获取
        ├── stock_database.py              # SQLite数据持久化
        ├── stock_universe.py              # 股票池管理
        ├── data_manager.py                # 数据生命周期管理
        ├── download_all.py                # 批量数据下载
        ├── download_continuous.sh         # 持续数据同步
        │
        │  ── 扩展模块 ──
        ├── portfolio_backtest.py          # Walk-Forward组合回测
        ├── llm_rate_limiter.py            # LLM API限速器
        ├── cache_manager.py               # 缓存管理
        │
        │  ── 基础设施 ──
        ├── config.py                      # 全局配置
        ├── logger.py                      # 日志工厂
        ├── logging_config.py              # 集中化日志管理
        ├── exceptions.py                  # 自定义异常体系
        │
        └── archive/                       # 历史版本归档
            ├── quant_investor_v71.py      # V7.1 过渡版
            ├── quant_investor_v72.py      # V7.2 健壮版
            ├── macro_terminal_v63.py      # V6.3 宏观终端
            ├── macro_terminal_transparent.py
            ├── legacy_scripts/            # 旧版一次性分析脚本
            ├── superseded_modules/        # 已被更新版本取代的模块
            ├── unused_infrastructure/     # 预留但未集成的基础设施
            └── old_layer_packages/        # 旧版模块化包（已被.py文件取代）
```

---

## 模块依赖关系

```
quant_investor_v9.py
  ├── quant_investor_v8.py
  │   ├── quant_investor_v7.py  ← 六层基础流水线
  │   │   ├── enhanced_data_layer.py    (Layer 1)
  │   │   ├── factor_analyzer.py        (Layer 2)
  │   │   ├── enhanced_model_layer.py   (Layer 3)
  │   │   ├── macro_terminal_tushare.py (Layer 4)
  │   │   ├── risk_management_layer.py  (Layer 5)
  │   │   └── decision_layer.py         (Layer 6)
  │   │       └── multi_model_debate.py
  │   ├── multi_llm_ensemble.py  (Layer 7)
  │   ├── investment_report.py
  │   ├── alpha_mining.py        (可选)
  │   └── portfolio_backtest.py  (可选)
  └── intelligence_layer.py
      ├── kronos_predictor.py    (Layer 8)
      ├── financial_analysis.py  (Layer 9)
      ├── news_analysis.py       (Layer 9)
      └── sentiment_analysis.py  (Layer 9)
```

所有模块共享基础设施：`config.py` · `logger.py` · `exceptions.py` · `cache_manager.py`

---

## 宏观风控信号

| 信号 | 含义 | 建议仓位 |
|:---|:---|:---:|
| 🔴 高风险 | 市场过热/流动性紧张 | ≤30% |
| 🟡 中风险 | 市场中性 | 30-60% |
| 🟢 低风险 | 基本面良好 | 60-90% |
| 🔵 极低风险 | 估值低/政策宽松 | 80-100% |

## 综合信号体系

| 综合得分 | 信号 | 建议行动 |
|:---|:---|:---|
| > 0.35 | 强烈买入 | 积极加仓 |
| 0.15–0.35 | 买入 | 适度加仓 |
| ±0.15 | 持有 | 维持仓位 |
| -0.35–-0.15 | 卖出 | 适度减仓 |
| < -0.35 | 强烈卖出 | 减仓/清仓 |

---

## 版本历史

| 版本 | 发布时间 | 核心特性 |
|:---|:---|:---|
| **V9.0** | 2026-03-12 | Kronos基础模型(Layer 8) + 财务/新闻/情绪分析(Layer 9)，代码清理 |
| V8.0 | 2026-03-11 | Multi-LLM集成裁判(Layer 7) + Alpha挖掘 + Walk-Forward回测 |
| V7.0 | 2026-02-26 | 六层架构 + Multi-LLM多空辩论 |
| V6.3 | 2026-02-25 | 宏观风控终端 |

---

## 配置说明（.env）

```ini
TUSHARE_TOKEN=your_token          # A股数据API（必需）
DB_PATH=data/stock_database.db    # SQLite数据库路径
LOG_LEVEL=INFO                    # 日志级别
INITIAL_CASH=1000000              # 回测初始资金
COMMISSION_RATE=0.0003            # 佣金率
STAMP_DUTY_RATE=0.001             # 印花税（A股卖出）
SLIPPAGE=0.001                    # 滑点率
```

---

**免责声明**：本项目仅供学习和研究使用，不构成任何投资建议。量化模型的历史表现不代表未来收益，投资有风险，入市需谨慎。
