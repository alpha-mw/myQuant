# Quant-Investor V8.0

专业级量化研究平台，当前主线采用：

`数据层 -> 五路并行研究分支 -> 风控层 -> 集成裁判层`

系统目标不是给出单一“买/卖”标签，而是形成**组合级投资策略**：总仓位、风格偏好、候选标的、仓位上限和执行约束。
当前主线默认输出 provenance、分支 `mode`、可靠度和降级状态；当全部候选只来自 synthetic/degraded 数据时，系统会自动降级为 `research_only`，仅保留研究参考结论。

---

## 当前架构

```text
Data Layer
  -> Kronos Branch
  -> Quant Branch
  -> LLM Debate Branch
  -> Intelligence Branch
  -> Macro Branch
  -> Risk Layer
  -> Ensemble Layer
```

### 1. 数据层
- 统一生成 `UnifiedDataBundle`
- 按股票保存 OHLCV、基本面、事件、情绪、宏观上下文
- 对单只股票的数据失败支持降级，不拖垮整篮子

### 2. 五个研究分支

| 分支 | 作用 | 主要输出 |
|:---|:---|:---|
| `kronos` | 基于 OHLCV 的时序趋势与未来收益判断 | `predicted_return`, `trend_regime`, `symbol_scores` |
| `quant` | Alpha 挖掘优先，失败时回退经典因子 | `alpha_factors`, `factor_exposures`, `expected_return` |
| `llm_debate` | 基本面、行业、事件驱动的多空辩论 | `bull_case`, `bear_case`, `key_risks` |
| `intelligence` | 财务质量、事件风险、情绪、资金流、广度 | `financial_health_score`, `event_risk_score`, `alerts` |
| `macro` | 宏观 regime、流动性、政策和风险状态 | `macro_score`, `macro_regime`, `risk_level` |

所有分支都输出统一的 `BranchResult`，并在 `metadata` 中显式标记：
- `branch_name`
- `score`
- `confidence`
- `signals`
- `risks`
- `explanation`
- `symbol_scores`
- `data_source_status`
- `is_synthetic`
- `degraded_reason`
- `branch_mode`
- `reliability`

### 3. 风控层
- 消费五个分支的统一结果，不依赖旧串行字段
- 做仓位管理、波动率控制、止损止盈、压力测试、风险预算

### 4. 集成裁判层
- 先形成研究共识
- 再叠加风控硬约束
- 最终输出组合级策略：
  - `target_exposure`
  - `style_bias`
  - `sector_preferences`
  - `candidate_symbols`
  - `position_limits`
  - `stop_loss_policy`
  - `execution_notes`

---

## 一揽子股票的数据传导

当输入一揽子股票时，数据传导分三步：

### 1. 先按股票分开收集
- `stock_pool` 中每只股票分别抓取和处理
- 结果进入 `UnifiedDataBundle.symbol_data`
- 结构是 `symbol -> DataFrame`

### 2. 分支内各自处理
- `kronos`、`llm_debate`、`intelligence` 主要按股票逐只处理
- `quant` 会先把全篮子股票合并成一张按 `date, symbol` 排序的截面表，再做 Alpha/因子分析
- `macro` 是市场级信号，最后会映射到整个篮子

### 3. 最后回到组合级聚合
- 五个分支都产出 `symbol_scores`
- 风控层将这些分数聚成组合共识
- 集成裁判层根据共识与风险约束选出候选标的并决定总仓位

---

## 核心文件

### 当前主线
- `scripts/unified/quant_investor_v8.py`
- `scripts/unified/parallel_research_pipeline.py`
- `scripts/unified/branch_contracts.py`

### 当前主线依赖的关键模块
- `scripts/unified/enhanced_data_layer.py`
- `scripts/unified/alpha_mining.py`
- `scripts/unified/macro_terminal_tushare.py`
- `scripts/unified/risk_management_layer.py`

### 兼容保留模块（Legacy，不作为当前主线入口）
- `scripts/unified/enhanced_model_layer.py`
- `scripts/unified/factor_analyzer.py`
- `scripts/unified/factor_neutralizer.py`
- `scripts/unified/portfolio_backtest.py`

---

## 快速开始

### 环境准备

```bash
cd myQuant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 可选环境变量

```bash
export TUSHARE_TOKEN="your-tushare-token"
export OPENAI_API_KEY="sk-your-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export DEEPSEEK_API_KEY="sk-your-deepseek-key"
export GOOGLE_API_KEY="your-gemini-key"
```

安全约定：
- 所有 token / API key 只从环境变量或本地 `.env` 读取，不写入仓库文件
- 主线 Tushare 客户端采用内存模式初始化，不执行落盘持久化
- 日志、报告和异常信息不得输出 credential 明文

### 命令行

```bash
python scripts/unified/quant_investor_v8.py \
  --stocks 000001.SZ 600519.SH 000858.SZ \
  --market CN \
  --capital 1000000 \
  --risk 中等
```

可选开关：
- `--no-kronos`
- `--no-intelligence`
- `--no-llm-debate`
- `--no-macro`

### Python API

官方公共入口：

```python
from scripts.unified import QuantInvestorV8
```

```python
analyzer = QuantInvestorV8(
    stock_pool=["000001.SZ", "600519.SH"],
    market="CN",
    total_capital=1_000_000,
    risk_level="中等",
    enable_kronos=True,
    enable_intelligence=True,
    enable_llm_debate=True,
    enable_macro=True,
    verbose=True,
)

result = analyzer.run()
print(result.final_strategy)
print(result.final_report)
```

最终 `final_strategy` 会固定包含：
- `research_mode`
- `provenance_summary`
- `branch_consensus`
- `position_limits`

### 独立运行 Alpha 挖掘

```python
from scripts.unified.alpha_mining import AlphaMiner

miner = AlphaMiner(df, enable_genetic=True, enable_llm=False)
result = miner.mine()

for factor in result.selected_factors[:5]:
    print(f"{factor.name}: IC={factor.ic_mean:.3f}, IR={factor.ir:.2f}")
```

---

## 当前目录建议

```text
myQuant/
├── README.md
├── pyproject.toml
├── requirements.txt
├── tests/
├── scripts/
│   ├── unified/
│   │   ├── quant_investor_v8.py
│   │   ├── parallel_research_pipeline.py
│   │   ├── branch_contracts.py
│   │   ├── enhanced_data_layer.py
│   │   ├── alpha_mining.py
│   │   ├── macro_terminal_tushare.py
│   │   ├── risk_management_layer.py
│   │   └── archive/
│   └── v4.0~v6.0/
└── skill/
```

---

## 归档说明

旧串行主线、旧 demo 和旧静态报告样例已经移到：

- `scripts/unified/archive/serial_research_legacy_20260312/`

归档说明见：

- `scripts/unified/archive/README_20260312_parallel_research.md`

---

## 版本说明

| 版本 | 日期 | 说明 |
|:---|:---|:---|
| `V8.0` | 2026-03-13 | 五路并行研究架构、统一分支契约、可信度治理与组合级集成裁判 |
| `V7.x` | 2026-02 | 六层/七层串行主线 |
| `V6.x` 及更早 | 2026-02 | 历史能力演进阶段 |

---

## 免责声明

本项目生成的所有分析结果仅供研究和参考，不构成任何投资建议。投资有风险，决策需谨慎。
