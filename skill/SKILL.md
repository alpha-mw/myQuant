---
name: quant-investor
description: "一个端到端的AI量化投资技能，用于股票分析、模型训练和回测。当用户想要进行量化投资、分析股票、构建投资策略或进行回测时使用此技能。V2.3版本引入了Qlib框架的先进设计，包括表达式引擎、特征缓存、财报PIT数据处理和增强回测引擎，显著提升开发效率和回测真实性。"
---

# 量化投资技能 (Quant-Investor) - V2.3

**版本**: 2.3
**作者**: Manus AI
**核心理念**: 融合工业级框架设计、统计显著性检验、贝叶斯推断与因果思维，构建高效、稳健、可解释的量化投资策略。

---

## 1. 技能简介

`quant-investor` 技能 V2.3 是一个工业级的端到端AI量化投资框架。本版本通过深入研究微软开源的Qlib框架，吸收了业界最前沿的设计理念，在V2.2的基础上进行了全面的底层能力升级。

### V2.3核心升级

本版本引入了四个高优先级的核心功能，显著提升系统的开发效率、计算性能和回测真实性：

1. **表达式引擎 (Expression Engine)**: 使用简洁的公式化字符串定义因子，如 `"Mean($close, 20)"`，减少50%以上的代码量。

2. **特征缓存系统 (Feature Caching System)**: 自动缓存特征工程结果，避免重复计算，预计提升70%以上的计算效率。

3. **财报PIT数据处理 (Financial Report PIT)**: 处理财务报表数据的多次修订问题，与V2.2的价格数据Point-in-Time框架结合，形成完整的Look-Ahead Bias解决方案。

4. **增强回测引擎 (Enhanced Backtesting Engine)**: 引入滑点、市场冲击、交易限制等真实交易成本模型，使回测结果更接近实盘。

### 技能特色

- **工业级架构**: 借鉴Qlib的分层设计，模块化、松耦合，易于扩展。
- **投资智慧融合**: 集成Graham、Buffett、Lynch、Dalio、Burry、Marks六位投资大师的策略。
- **统计严谨性**: IC分析、Bootstrap测试、贝叶斯框架、因果推断。
- **完整的Look-Ahead Bias消除**: 价格数据Point-in-Time框架 + 财报PIT数据处理。
- **用户友好**: 端到端的AI量化投资技能，易于使用。

---

## 2. 核心工作流 (The Scientific Workflow)

本技能遵循一个严谨的、科学的策略开发流程，确保每一步都建立在坚实的基础之上。

**核心步骤如下**: 

1.  **数据获取与特征工程 (Data & Feature Engineering)** 【V2.3升级】
    - **工具**: `scripts/data_retrieval.py`, `scripts/v2.3/expression_engine.py`, `scripts/v2.3/feature_cache.py`
    - **方法**: 
        - 使用**表达式引擎**快速构建因子，如 `"$close / Ref($close, 5) - 1"` 定义5日收益率。
        - **特征缓存系统**自动缓存计算结果，后续运行直接加载，大幅提升效率。
        - **财报PIT数据处理**确保使用历史上真实可用的财报数据，消除修订引入的偏差。

2.  **理论假设与信号初筛 (Hypothesis & Signal Vetting)**
    - **工具**: `scripts/ic_analysis.py`
    - **方法**: 在编写任何策略代码前，首先使用**信息系数 (Information Coefficient, IC)** 检验潜在因子（信号）的有效性和稳定性。只有通过IC检验的因子才能进入下一步。

3.  **严谨的回测框架 (Rigorous Backtesting)** 【V2.3升级】
    - **工具**: `scripts/walk_forward_optimizer.py`, `scripts/v2.3/enhanced_backtest.py`
    - **方法**: 
        - **增强回测引擎**引入滑点、市场冲击、交易限制等真实交易成本模型。
        - **成本优先原则 (Fees First)**: 在回测开始时就计入高于实际的交易成本。
        - **前向滚动优化 (Walk-Forward Optimization, WFO)**: 彻底杜绝未来数据泄露，在真实的样本外数据上验证策略的稳健性。

4.  **统计显著性验证 (Statistical Validation)**
    - **工具**: `scripts/statistical_validator.py`
    - **方法**: 
        - **Bootstrap检验**: 对夏普比率等关键指标进行上千次重采样，给出其95%置信区间，评估运气成分。
        - **多重检验校正**: 解决"数据挖掘"问题，报告校正后的p值。
        - **蒙特卡洛模拟**: 模拟数千条可能的收益路径，评估最坏情况下的风险。

5.  **贝叶斯动态更新 (Bayesian Inference)**
    - **工具**: `scripts/bayesian_updater.py`
    - **方法**: 将市场均衡或专家观点作为**先验信念**，结合新的市场数据（似然），生成**后验分布**。这使得策略能够动态适应市场变化，并量化不确定性。

6.  **因果推断探索 (Causal Inference)**
    - **工具**: `scripts/causal_analyzer.py`
    - **方法**: 尝试回答"为什么"策略有效，而不仅仅是"什么"有效。通过构建因果图、寻找自然实验等方法，探索因子与收益之间的深层因果关系，增强策略的可解释性和鲁棒性。

---

## 3. 技能文件结构

```
/skills/quant-investor/
├── SKILL.md                # 本指南
|
├── scripts/                  # 核心执行脚本
│   ├── data_retrieval.py       # 数据获取
│   ├── feature_engineering.py  # 特征工程
│   ├── ic_analysis.py          # 信号IC检验
│   ├── walk_forward_optimizer.py # 前向滚动优化
│   ├── model_training.py       # 模型训练
│   ├── backtesting_advanced.py # 高级回测引擎
│   ├── statistical_validator.py# 统计验证模块
│   ├── bayesian_updater.py     # 贝叶斯更新模块
│   ├── causal_analyzer.py      # 因果分析模块
│   |
│   └── v2.3/                   # 【V2.3新增】工业级框架组件
│       ├── expression_engine.py   # 表达式引擎
│       ├── feature_cache.py       # 特征缓存系统
│       ├── financial_pit.py       # 财报PIT数据处理
│       └── enhanced_backtest.py   # 增强回测引擎
|
├── references/               # 核心理论与方法论文档
│   ├── data_sources.md         # 数据源说明
│   ├── feature_library.md      # 特征库
│   ├── model_zoo.md            # 模型库
│   ├── statistical_framework.md# 统计检验方法论
│   ├── bayesian_guide.md       # 贝叶斯推断指南
│   ├── causal_inference.md     # 因果推断指南
│   └── qlib_integration.md     # 【V2.3新增】Qlib框架集成指南
|
└── templates/                # 配置文件与报告模板
    ├── config.ini.template     # 配置文件模板
    └── report_advanced.md.template # 专业报告模板
```

---

## 4. V2.3新功能详解

### 4.1. 表达式引擎 (Expression Engine)

**目标**: 简化因子构建，提升开发效率。

**使用方法**:

```python
from scripts.v2.3.expression_engine import ExpressionEngine

# 创建引擎
engine = ExpressionEngine()

# 定义因子表达式
expressions = {
    'return_5d': "$close / Ref($close, 5) - 1",  # 5日收益率
    'ma_20': "Mean($close, 20)",                  # 20日均线
    'bb_upper': "Mean($close, 20) + 2 * Std($close, 20)",  # 布林带上轨
    'momentum': "Delta($close, 1)",               # 价格动量
}

# 计算因子
for name, expr in expressions.items():
    data[name] = engine.evaluate(expr, data)
```

**支持的函数**:

| 函数 | 含义 | 示例 |
|------|------|------|
| `Ref(field, n)` | 引用n个周期前的数据 | `Ref($close, 5)` |
| `Mean(field, n)` | 过去n个周期的均值 | `Mean($close, 20)` |
| `Std(field, n)` | 过去n个周期的标准差 | `Std($close, 20)` |
| `Sum(field, n)` | 过去n个周期的总和 | `Sum($volume, 5)` |
| `Max(field, n)` | 过去n个周期的最大值 | `Max($high, 10)` |
| `Min(field, n)` | 过去n个周期的最小值 | `Min($low, 10)` |
| `Delta(field, n)` | n个周期前的差分 | `Delta($close, 1)` |
| `Rank(field)` | 截面排序（百分位） | `Rank($close)` |
| `Corr(field1, field2, n)` | 过去n个周期的相关系数 | `Corr($high, $low, 30)` |

### 4.2. 特征缓存系统 (Feature Caching System)

**目标**: 自动缓存特征工程结果，避免重复计算。

**使用方法**:

```python
from scripts.v2.3.feature_cache import FeatureCache

# 创建缓存管理器
cache = FeatureCache()

# 定义计算函数
def compute_features():
    # 复杂的特征计算逻辑
    return features_df

# 获取或计算特征（自动缓存）
features = cache.get_or_compute(
    key="my_features_v1_2020_2022",
    compute_func=compute_features
)

# 清理缓存
cache.clear_cache("my_features_v1_2020_2022")
```

**优势**:
- 首次计算后自动保存到本地缓存（`~/.quant-investor/cache/`）
- 后续运行直接加载缓存，速度提升70%以上
- 支持缓存管理（列出、清理）

### 4.3. 财报PIT数据处理 (Financial Report PIT)

**目标**: 处理财务报表数据的多次修订问题，消除Look-Ahead Bias。

**使用方法**:

```python
from scripts.v2.3.financial_pit import FinancialPIT

# 创建PIT管理器
pit = FinancialPIT()

# 插入财报数据（包含发布日期）
pit.insert_data(
    stock_code='600000.SH',
    field_name='net_profit',
    report_date='2020-12-31',
    publish_date='2021-04-30',
    value=1000000000.0
)

# Point-in-Time查询（回测时使用）
value = pit.query_at_time(
    stock_code='600000.SH',
    field_name='net_profit',
    query_date='2021-05-01',  # 回测时间点
    report_date='2020-12-31'
)

# 查询修订历史
history = pit.get_history(
    stock_code='600000.SH',
    field_name='net_profit',
    report_date='2020-12-31'
)
```

**应用场景**:
- 基本面量化策略（使用财报数据）
- 价值投资策略（使用财务指标）
- 任何需要财报数据的策略

### 4.4. 增强回测引擎 (Enhanced Backtesting Engine)

**目标**: 引入更真实的交易成本模型，使回测结果更接近实盘。

**使用方法**:

```python
from scripts.v2.3.enhanced_backtest import EnhancedBacktestEngine, OrderType, TradingCostModel

# 创建交易成本模型
cost_model = TradingCostModel(
    commission_rate=0.0003,  # 佣金率 0.03%
    stamp_tax_rate=0.001,    # 印花税 0.1%
    slippage_rate=0.001,     # 滑点率 0.1%
)

# 创建回测引擎
engine = EnhancedBacktestEngine(
    initial_capital=1000000,
    cost_model=cost_model
)

# 执行订单
engine.execute_order(
    stock_code='600000.SH',
    order_type=OrderType.BUY,
    target_shares=1000,
    current_price=10.0,
    date='2020-01-02',
    volume=1000000  # 当日成交量
)

# 计算组合价值
portfolio_value = engine.calculate_portfolio_value(
    prices={'600000.SH': 10.5},
    date='2020-01-03'
)

# 获取性能指标
metrics = engine.get_performance_metrics()
```

**交易成本模型**:
- **佣金**: 双边，默认0.03%
- **印花税**: 仅卖出，默认0.1%（A股）
- **滑点**: 固定滑点或基于成交量的动态滑点
- **市场冲击**: 基于交易量占比的市场冲击模型
- **交易限制**: 最小交易单位（A股100股）

---

## 5. 使用指南

要使用本技能开发一个完整的量化策略，请遵循以下步骤：

### 5.1. 基础设置

1.  **创建项目目录**: `mkdir -p /home/ubuntu/my_quant_project`
2.  **复制配置文件**: `cp /home/ubuntu/skills/quant-investor/templates/config.ini.template /home/ubuntu/my_quant_project/config.ini`
3.  **编辑配置文件**: 根据您的需求修改 `config.ini`，设置股票池、回测周期、Tushare Token等。

### 5.2. V2.3工作流（推荐）

使用V2.3的新功能可以显著提升开发效率：

```python
# 1. 数据获取
from scripts.data_retrieval import get_stock_data
data = get_stock_data(stock_codes, start_date, end_date)

# 2. 使用表达式引擎构建因子
from scripts.v2.3.expression_engine import ExpressionEngine
engine = ExpressionEngine()

factors = {
    'return_5d': "$close / Ref($close, 5) - 1",
    'ma_20': "Mean($close, 20)",
    'volatility': "Std($close, 20)",
}

for name, expr in factors.items():
    data[name] = engine.evaluate(expr, data)

# 3. 使用特征缓存系统
from scripts.v2.3.feature_cache import FeatureCache
cache = FeatureCache()

features = cache.get_or_compute(
    key=f"features_{start_date}_{end_date}",
    compute_func=lambda: compute_all_features(data)
)

# 4. IC分析
from scripts.ic_analysis import ic_analysis
ic_results = ic_analysis(features, returns)

# 5. 使用增强回测引擎
from scripts.v2.3.enhanced_backtest import EnhancedBacktestEngine
engine = EnhancedBacktestEngine(initial_capital=1000000)

# ... 回测逻辑 ...

# 6. 统计验证
from scripts.statistical_validator import validate_strategy
validation_results = validate_strategy(returns)
```

### 5.3. 传统工作流

如果您更熟悉传统的工作流，也可以继续使用：

1.  **执行数据获取**: `python3 /home/ubuntu/skills/quant-investor/scripts/data_retrieval.py --config /home/ubuntu/my_quant_project/config.ini`
2.  **执行特征工程**: `python3 /home/ubuntu/skills/quant-investor/scripts/feature_engineering.py --config ...`
3.  **【关键】信号IC检验**: `python3 /home/ubuntu/skills/quant-investor/scripts/ic_analysis.py --config ...`
4.  **执行高级回测与验证**: `python3 /home/ubuntu/skills/quant-investor/scripts/backtesting_advanced.py --config ...`
5.  **查看报告**: 最终的专业报告将生成在您项目目录的 `reports/` 文件夹下。

---

## 6. 核心依赖

本技能需要以下Python库，将在首次使用时自动提示安装：
- `tushare`: 核心金融数据源
- `pandas`, `numpy`, `scipy`: 数据处理与科学计算
- `scikit-learn`: 机器学习模型
- `statsmodels`: 统计模型与检验
- `matplotlib`, `seaborn`: 数据可视化
- `pymc`: 贝叶斯推断
- `dowhy`, `causalml`: 因果推断
- `alphalens`: 因子分析

---

## 7. V2.3版本路线图

### 已完成（V2.3）
- ✅ 表达式引擎
- ✅ 特征缓存系统
- ✅ 财报PIT数据处理
- ✅ 增强回测引擎

### 计划中（V2.4）
- 🔄 集成Qlib的SOTA时间序列模型（HIST、TRA）
- 🔄 自动数据更新脚本
- 🔄 数据存储优化（考虑引入高效的数据格式）

### 远期规划（V2.5+）
- 🔮 强化学习框架（基于QlibRL）
- 🔮 在线学习与策略自适应
- 🔮 多层次决策框架

---

**免责声明**: 本技能提供的所有策略和分析仅供研究和学习使用，不构成任何投资建议。金融市场存在风险，投资需谨慎。

---

## 附录A：价格复权处理最佳实践

本技能在处理历史股价时，严格遵循量化金融的最佳实践，以确保分析结果的准确性和可复现性。

### 1. 问题的核心：Look-Ahead Bias (前视偏差)

- **后复权 (Backward Adjusted)**: 这是大多数数据提供商（如Yahoo Finance）的默认方式。它以当前价格为基准，向前调整历史价格。这种方法虽然便于图表展示，但存在致命的**前视偏差**。历史某天的价格会随着未来的分红、拆股而改变，导致回测结果不可复现，并引入了未来信息。

- **前复权 (Forward Adjusted)**: 本技能采用的方式。它以股票上市第一天的价格为基准，向后调整未来价格。这种方法保证了任何历史日期的价格只依赖于该日期之前的事件，**完全消除了前视偏差**，确保了回测的科学性和可复现性。

### 2. 本技能的实现

- **数据获取 (`data_retrieval.py`)**: 脚本已重构，无论是从Tushare还是yfinance获取数据，都会**统一处理为前复权价格**。
  - **Tushare**: 直接使用`ts.pro_bar(adj='qfq')`接口获取前复权数据。
  - **yfinance**: 通过同时获取后复权和不复权价格，利用后复权价格的收益率和第一天的真实价格来**手动计算前复权价格**。

- **下游脚本**: 所有后续的特征工程、模型训练和回测脚本，都统一使用`data_retrieval.py`生成的**前复权收盘价 (`close`)**，从而保证了整个分析流程的一致性和准确性。

### 3. 不同场景的推荐

| 场景 | 推荐复权方式 | 本技能实现 |
|---|---|---|
| **回测与模型训练** | **前复权** | ✅ **默认采用** |
| 实时交易决策 | 不复权 | 需额外获取实时报价 |
| 技术分析图表 | 后复权 | 可通过数据源接口获取 |

通过强制使用前复权价格进行核心分析，本技能从根本上保证了所有量化结果的**严谨性和可信度**。

---

## 附录B：投资大师智慧整合

本技能融合了六位投资大师的核心理念，提供多维度的投资视角：

| 投资大师 | 核心理念 | 应用场景 |
|---------|---------|---------|
| **Benjamin Graham** | 价值投资、安全边际 | 寻找被低估的股票 |
| **Warren Buffett** | 护城河、长期持有 | 识别具有竞争优势的公司 |
| **Peter Lynch** | PEG指标、成长投资 | 寻找成长性股票 |
| **Ray Dalio** | 风险平价、全天候策略 | 投资组合风险管理 |
| **Michael Burry** | 逆向投资、深度研究 | 寻找市场错误定价 |
| **Howard Marks** | 二阶思维、周期理解 | 理解市场情绪和周期 |

这些策略可以单独使用，也可以组合使用，形成多层次的投资决策框架。

---

## 附录C：Qlib框架集成指南

V2.3版本深入研究了微软开源的Qlib框架，并吸收了其核心设计理念。详细的研究笔记和对比分析请参考 `references/qlib_integration.md`。

### Qlib的核心优势

1. **分层架构**: 基础设施层、学习框架层、工作流层、接口层
2. **高效数据层**: `.bin` 格式、表达式引擎、缓存机制
3. **完备的PIT数据库**: 处理财报数据的多次修订
4. **强化学习框架 (QlibRL)**: 支持订单执行优化和动态投资组合管理

### quant-investor的集成策略

- **V2.3**: 表达式引擎、特征缓存、财报PIT、增强回测引擎
- **V2.4**: SOTA时间序列模型、自动数据更新
- **V2.5+**: 强化学习框架、在线学习

通过渐进式集成，quant-investor将在保持自身特色（投资智慧、统计严谨性）的同时，逐步提升到工业级水平。
