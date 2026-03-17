# Quant-Investor V10.0 - 专业级量化投资平台

<div align="center">

**五支柱并行架构 · Kronos基础模型 · LLM多空辩论 · 多维情报融合 · 集成裁判层 · 因子工程**

[![Version](https://img.shields.io/badge/Version-10.0.0-blue.svg)](https://github.com/alpha-mw/myQuant)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-59%20passing-brightgreen.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

## 项目简介

**Quant-Investor V10.0** 彻底重构了 V9 的顺序链式架构，引入**五支柱并行智能框架**：数据层之后，五大分支通过 `ThreadPoolExecutor` 真正并行执行，最终经过风控层约束，由**集成裁判层**（EnsembleJudgeEngine）汇聚所有信号，输出可执行的投资建议。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                     数据层（Data Layer）                              │
│  Tushare Pro / AKShare / yfinance · OHLCV + 财务 + 新闻 + 宏观       │
└───────────────────────────┬─────────────────────────────────────────┘
                            │ 共享数据，五分支并行启动
        ┌───────────────────┼────────────────────────┐
        │                   │                        │
┌───────▼──────┐  ┌─────────▼────────┐  ┌──────────▼──────────┐
│  Branch 1    │  │  Branch 2        │  │  Branch 3           │
│  Kronos      │  │  传统量化         │  │  LLM多空辩论         │
│  图形预测     │  │  因子层 + 模型层  │  │  5模型角色辩论       │
│              │  │  Alpha158+       │  │  财务/行业/宏观/     │
│ kronos-mini  │  │  FactorEngineer  │  │  技术/风险           │
│ kronos-small │  │  IC-IR加权合成   │  │                     │
│ kronos-base  │  │  行业中性化       │  │                     │
└───────┬──────┘  └─────────┬────────┘  └──────────┬──────────┘
        │                   │                        │
┌───────▼──────────────┐  ┌─▼─────────────────────────────────┐
│  Branch 4            │  │  Branch 5                          │
│  多维情报融合         │  │  宏观数据                           │
│  财务分析            │  │  货币/信用/增长/估值/情绪            │
│  新闻情感            │  │  A股：4模块  美股：5模块             │
│  市场情绪            │  │  宏观一票否决机制                    │
│  恐慌贪婪指数         │  │                                    │
└───────┬──────────────┘  └─▲──────────────────────────────────┘
        └──────────┬─────────┘
                   │ 五分支信号汇聚
┌──────────────────▼──────────────────────────────────────────────────┐
│                     风控层（Risk Layer）                              │
│  VaR/CVaR · 仓位上限 · Barra因子风险分解 · 止损止盈                   │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────────────┐
│                  集成裁判层（Ensemble Judge Layer）         ★ V10 新增 │
│                                                                      │
│  RegimeDetector   → 识别市场状态（趋势/震荡/高波动/危机）              │
│  动态权重调整      → 不同市场状态下各分支权重不同                       │
│  宏观一票否决      → macro_score ≤ -0.6 时强制降仓                    │
│  置信度加权集成    → 方向一致性 × 权重 → 综合得分 [-1, +1]             │
│  组合配置优化      → 最大单股仓位约束 + 总仓位上限                      │
└──────────────────┬──────────────────────────────────────────────────┘
                   │
            最终投资建议 + Markdown 报告
```

---

## 集成裁判层：市场状态感知与动态权重

| 市场状态 | 触发条件 | Kronos | 量化 | 辩论 | 情报融合 | 宏观 |
|:--------|:--------|:------:|:----:|:----:|:--------:|:----:|
| 趋势上行 | 低波动 + 5日均涨 | 30% | 25% | 15% | 15% | 15% |
| 趋势下行 | 低波动 + 5日均跌 | 25% | 20% | 15% | 15% | 25% |
| 高波动 | 20日vol > 2.5% | 20% | 15% | 15% | 20% | 30% |
| 震荡盘整 | 低波动 + 无趋势 | 20% | 20% | 20% | 25% | 15% |
| **极端风险** | vol > 4% 或宏观极端 | 15% | 10% | 10% | 15% | **50%** |

---

## 核心特性

### Branch 1: Kronos 金融基础模型（K线图形分析）

| 特性 | 描述 |
|:---|:---|
| **预训练规模** | 45个全球交易所，120亿条K线记录（AAAI 2026） |
| **零样本预测** | RankIC 比最佳 TSFM 提升 **93%** |
| **可用模型** | kronos-mini(4M) / kronos-small(25M) / kronos-base(102M) |
| **降级模式** | 库未安装时自动使用 EWMA+动量统计替代预测 |

### Branch 2: 传统量化（改进因子层 + 模型层）

**Alpha158+ 因子库（200+ 因子）**：

| 因子类别 | 代表因子 | 改进点 |
|:--------|:--------|:------|
| 价格动量 | RETURN_5D/20D/60D, LOG_RETURN | 多周期覆盖 |
| 成交量 | VOLUME_MA, AMOUNT_RATIO, VWAP偏离 | 量价关系 |
| 波动率 | VOLATILITY_20D, PARKINSON | 历史波动 |
| 技术指标 | RSI/MACD/布林带/均线偏离 | 多参数 |
| **短期反转** | REVERSAL_5D/10D, MOM_12M_SKIP1M | ★ V10 新增 |
| **质量代理** | PRICE_STABILITY, CLOSE_POSITION, VOL_DIVERGENCE | ★ V10 新增 |
| **波动率状态** | VOL_RATIO_5_20, VOL_OF_VOL, OVERNIGHT_GAP | ★ V10 新增 |

**FactorEngineer 因子工程流水线**（V10 新增）：
```
原始因子 → 3σ去极值 → rolling Z-Score标准化
→ Spearman IC-IR加权筛选（IC阈值0.02）
→ 截面排名归一化 → 行业中性化
→ [-1, +1] 合成评分
```

### Branch 3: LLM 多空辩论

5个专业分析角色对每只股票展开多空辩论：
- **财务分析模型**：报表、盈利、估值、现金流
- **行业研究模型**：生命周期、竞争格局、护城河
- **宏观分析模型**：经济周期、货币政策、地缘政治
- **技术分析模型**：趋势、量价、技术指标
- **风险评估模型**：波动率、回撤、流动性、尾部风险

### Branch 4: 多维情报融合

| 模块 | 分析内容 |
|:---|:---|
| **财务分析** | Piotroski F-Score · Beneish M-Score · Altman Z-Score · DCF估值 · DuPont拆解 |
| **新闻分析** | AKShare/RSS/新浪多源 · LLM事件检测(9类) · 情感打分 |
| **市场情绪** | 恐慌贪婪指数(7维度) · 技术情绪 · 资金流 · 市场广度 · 行业轮动 |

### Branch 5: 宏观数据

| 市场 | 监测维度 |
|:---|:---|
| **A股** | 货币（M2/利率/DR007） · 信用（社融/PMI） · 增长（GDP/工业） · 估值（PE/PB） |
| **美股** | 利率（美债/联储） · 就业（非农/失业率） · 估值（标普PE） · 通胀（CPI/PCE） |

---

## 快速开始

### 环境配置

```bash
git clone https://github.com/alpha-mw/myQuant.git
cd myQuant
pip install -r requirements.txt

# 可选：安装 Kronos 原生基础模型
git clone https://github.com/shiyu-coder/Kronos
cd Kronos && pip install -e . && cd ..
```

### API 密钥配置

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # LLM辩论 + 情报融合
export OPENAI_API_KEY="sk-..."          # 可选LLM
export DEEPSEEK_API_KEY="sk-..."        # 可选LLM
export TUSHARE_TOKEN="your-token"       # A股数据
```

### 使用示例

#### V10 五支柱并行分析（推荐）

```bash
# 命令行
python scripts/unified/quant_investor_v10.py \
    --stocks 000001.SZ 600519.SH 000858.SZ \
    --market CN \
    --capital 1000000 \
    --risk-level 中等 \
    --output report_v10.md

# 禁用部分分支加速
python scripts/unified/quant_investor_v10.py \
    --stocks 600519.SH \
    --no-debate \
    --no-macro \
    --workers 3 \
    --output quick.md
```

```python
from scripts.unified.quant_investor_v10 import QuantInvestorV10

analyzer = QuantInvestorV10(
    stock_pool=["000001.SZ", "600519.SH"],
    market="CN",
    total_capital=1_000_000,
    risk_level="中等",           # 保守 / 中等 / 积极
    kronos_model="kronos-small",
    enable_kronos=True,
    enable_quant=True,
    enable_debate=True,
    enable_intelligence=True,
    enable_macro=True,
    max_workers=5,               # 并行线程数
)
result = analyzer.run()
print(result.final_report)

# 查看集成裁判结果
for sym, j in result.ensemble.stock_judgments.items():
    print(f"{sym}: {j.decision} | 得分={j.ensemble_score:.2f} | 仓位={j.risk_adjusted_weight:.1%}")
```

#### 单模块使用

```python
# 集成裁判
from scripts.unified.ensemble_judge import EnsembleJudgeEngine
judge = EnsembleJudgeEngine(max_single_position=0.25)
result = judge.judge(
    stock_pool=["600519.SH"],
    kronos_result=..., quant_result=..., debate_result=...,
    intelligence_result=..., macro_result=..., risk_result=...
)

# 因子工程（截面打分）
from scripts.unified.alpha158 import FactorEngineer
engineer = FactorEngineer(ic_threshold=0.02)
scores = engineer.cross_sectional_score({"600519.SH": df_price})

# Kronos 预测
from scripts.unified.kronos_predictor import KronosIntegrator
signal = KronosIntegrator("kronos-small").analyze_portfolio(
    {"600519.SH": df_price}, pred_len=20
)

# 财务分析
from scripts.unified.financial_analysis import FinancialAnalyzer
report = FinancialAnalyzer().full_analysis("600519.SH", "贵州茅台", df_financial)

# 宏观终端
from scripts.unified.macro_terminal_tushare import create_terminal
terminal = create_terminal("CN")
print(terminal.format_report_markdown(terminal.generate_risk_report()))
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
│   └── SKILL_V3.0.md
└── scripts/
    └── unified/                           ⭐ 主代码目录（42个模块）
        │
        │  ── 主入口 ──
        ├── quant_investor_v10.py          # ★ V10 主入口：五支柱并行框架
        ├── quant_investor_v9.py           # V9：九层串行（保留）
        ├── quant_investor_v8.py           # V8：七层流水线
        ├── quant_investor_v7.py           # V7：六层基础流水线
        │
        │  ── 集成裁判层（V10 新增）──
        ├── ensemble_judge.py              # ★ 集成裁判：市场状态感知+动态权重
        │
        │  ── Branch 4：多维情报融合 ──
        ├── intelligence_layer.py          # 编排器
        ├── financial_analysis.py          # Piotroski/Beneish/Altman/DCF/DuPont
        ├── news_analysis.py               # 新闻&替代数据分析
        ├── sentiment_analysis.py          # 市场情绪（恐慌贪婪/资金/广度）
        │
        │  ── Branch 1：Kronos ──
        ├── kronos_predictor.py            # Kronos K线预测（含统计降级）
        │
        │  ── Branch 3：LLM多空辩论 ──
        ├── decision_layer.py              # 决策层封装
        ├── multi_model_debate.py          # 5模型多空辩论
        ├── multi_llm_ensemble.py          # 多LLM集成
        ├── investment_report.py           # 结构化执行报告
        │
        │  ── 风控层 ──
        ├── risk_management_layer.py       # 组合风控
        ├── var_calculator.py              # VaR/CVaR
        ├── stress_tester.py               # 压力测试
        ├── market_impact.py               # 市场冲击模型
        ├── advanced_risk_metrics.py       # 高级风险指标
        ├── risk_dashboard.py              # 风险可视化
        │
        │  ── Branch 5：宏观层 ──
        ├── macro_terminal_tushare.py      # 宏观风控终端
        │
        │  ── Branch 2：传统量化（因子+模型）──
        ├── enhanced_model_layer.py        # ML模型集成（RF/XGB/LSTM）
        ├── hyperparameter_tuner.py        # 超参数优化
        ├── shap_explainer.py              # SHAP可解释性
        ├── factor_analyzer.py             # IC/IR因子分析
        ├── factor_neutralizer.py          # 行业/市值中性化
        ├── alpha_mining.py                # Alpha挖掘（系统/遗传/LLM三层）
        ├── alpha158.py                    # ★ Alpha158+因子库（200+因子）+ FactorEngineer
        │
        │  ── 数据层 ──
        ├── enhanced_data_layer.py         # 多源数据采集与清理
        ├── batch_data_fetcher.py          # 批量数据获取
        ├── stock_database.py              # SQLite数据持久化
        ├── stock_universe.py              # 股票池管理
        ├── data_manager.py                # 数据生命周期管理
        ├── download_all.py                # 批量数据下载
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
            ├── legacy_scripts/            # 旧版一次性分析脚本（13个）
            ├── superseded_modules/        # 已被更新版本取代（2个）
            ├── unused_infrastructure/     # 未集成的基础设施（9个）
            └── old_layer_packages/        # 旧版模块化包（6个目录）
```

---

## 模块依赖关系

```
quant_investor_v10.py
  ├── [数据层] enhanced_data_layer.py
  │
  ├── [Branch 1] kronos_predictor.py
  ├── [Branch 2] quant_investor_v7.py  ← 降级: alpha158.FactorEngineer
  │               ├── enhanced_data_layer.py
  │               ├── factor_analyzer.py + alpha158.py
  │               ├── enhanced_model_layer.py
  │               ├── macro_terminal_tushare.py
  │               ├── risk_management_layer.py
  │               └── decision_layer.py → multi_model_debate.py
  ├── [Branch 3] decision_layer.py → multi_model_debate.py
  ├── [Branch 4] intelligence_layer.py
  │               ├── financial_analysis.py
  │               ├── news_analysis.py
  │               └── sentiment_analysis.py
  ├── [Branch 5] macro_terminal_tushare.py
  │
  ├── [风控层]   risk_management_layer.py
  └── [裁判层]   ensemble_judge.py
                  ├── RegimeDetector（市场状态检测）
                  ├── SignalNormalizer（五分支归一化）
                  └── EnsembleJudgeEngine（集成裁判）

所有模块共享：config.py · logger.py · exceptions.py · cache_manager.py
```

---

## 综合信号体系

| 集成裁判得分 | 信号 | 建议行动 |
|:-----------|:-----|:--------|
| > 0.7 | 强烈买入 | 积极加仓（至最大仓位上限） |
| 0.3–0.7 | 买入 | 适度加仓 |
| ±0.3 | 中性 | 维持或观望 |
| -0.3–-0.7 | 卖出 | 适度减仓 |
| < -0.7 | 强烈卖出 | 减仓/清仓 |

## 宏观风控信号（一票否决）

| 信号 | 含义 | 集成行为 |
|:---|:---|:---:|
| 🔴 极端风险 | 宏观得分 ≤ -0.6 | 一票否决：全部股票强制降至空仓 |
| 🟠 高风险 | 宏观权重升至 25-50% | 压制所有多头信号 |
| 🟡 中性 | 正常权重分配 | 五分支均衡决策 |
| 🟢 宽松 | 宏观权重降至 15% | 量化/Kronos主导 |

---

## 版本历史

| 版本 | 发布时间 | 核心特性 |
|:---|:---|:---|
| **V10.0** | 2026-03-12 | 五支柱并行架构、集成裁判层、FactorEngineer、市场状态感知 |
| V9.0 | 2026-03-12 | Kronos基础模型(L8) + 财务/新闻/情绪分析(L9) |
| V8.0 | 2026-03-11 | Multi-LLM集成裁判(L7) + Alpha挖掘 + Walk-Forward回测 |
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
