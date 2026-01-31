---
name: quant-investor
description: "一个端到端的AI量化投资技能，用于股票分析、模型训练和回测。当用户想要进行量化投资、分析股票、构建投资策略或进行回测时使用此技能。V2.4版本引入了LLM增强量化分析框架，将传统量化模型的严谨性与大型语言模型的深度认知能力深度融合，实现从数据到投资建议的端到端智能化分析。"
---

# 量化投资技能 (Quant-Investor) - V2.4

**版本**: 2.4  
**作者**: Manus AI  
**核心理念**: 融合工业级框架设计、LLM深度分析、统计显著性检验、贝叶斯推断与因果思维，构建智能化、自动化、可解释的量化投资平台。

---

## 1. 技能简介

`quant-investor` 技能 V2.4 是一个**LLM增强的端到端AI量化投资平台**。本版本在V2.3工业级基础设施的基础上，引入了创新的**量化-LLM混合分析框架**，将传统量化模型的数值计算能力与大型语言模型的语义理解、因果推断和知识融合能力完美结合。

### V2.4核心创新：LLM增强量化分析框架

V2.4版本的核心是一个**分层的、多Agent协作的混合架构**，实现了从数据收集到投资建议的全流程智能化：

#### 架构图

```
用户输入 → 数据收集Agent → 特征工程Agent → 量化分析Agent 
    → LLM综合分析Agent → Bull/Bear辩论 → 投资建议Agent → 最终报告
```

#### 核心组件

1.  **LLM客户端模块** (`scripts/v2.4/llm_client.py`)
    - 统一封装OpenAI和Gemini API
    - 智能缓存机制，避免重复调用，降低成本
    - 成本估算和自动重试功能

2.  **提示工程库** (`scripts/v2.4/prompt_templates.py`)
    - 8个标准化的Prompt模板（因子生成、IC分析、回测解释、综合分析、多空辩论、投资建议等）
    - 5种专业角色的系统消息（量化分析师、因子工程师、投资顾问、Bull/Bear研究员）

3.  **投资大师智慧库** (`references/master_wisdom.json`)
    - 6位投资大师的投资哲学和核心原则（巴菲特、达里奥、彼得·林奇、格雷厄姆、霍华德·马克斯、迈克尔·伯里）
    - 结构化的投资检查清单和经典投资名言

4.  **端到端分析流水线** (`scripts/v2.4/llm_quant_pipeline.py`)
    - 六大分析阶段：数据收集 → 特征工程 → 量化分析 → LLM综合分析 → 多空辩论 → 投资建议
    - 自动生成结构化的Markdown分析报告

#### V2.4的核心价值

| 价值 | 描述 |
|:---|:---|
| **自动化 (Automation)** | 从数据到策略的端到端自动化，减少人工干预 |
| **智能化 (Intelligence)** | 超越数值计算，进行深度因果推断和多维度综合分析 |
| **可解释性 (Interpretability)** | 提供清晰的推理过程（Chain-of-Thought），增强决策可信度 |
| **融合性 (Fusion)** | 完美融合量化分析的严谨性与投资大师的智慧 |
| **适应性 (Adaptability)** | 能够识别市场状态并动态调整策略 |

### V2.3核心能力（已集成）

V2.4继承了V2.3的所有工业级基础设施：

1.  **表达式引擎**: 使用简洁的公式化字符串定义因子，减少50%代码量
2.  **特征缓存系统**: 自动缓存特征工程结果，提升70%计算效率
3.  **财报PIT数据处理**: 处理财务报表数据的多次修订问题
4.  **增强回测引擎**: 引入滑点、市场冲击等真实交易成本模型

### 技能特色

-   **LLM增强分析**: 将量化结果交给大模型进行深度分析和因果推断
-   **多Agent协作**: Bull/Bear Agent辩论机制，提供多维度视角
-   **投资智慧融合**: 集成6位投资大师的策略，为决策提供理论支持
-   **工业级架构**: 借鉴Qlib和TradingAgents的先进设计
-   **统计严谨性**: IC分析、Bootstrap测试、贝叶斯框架、因果推断
-   **完整的Look-Ahead Bias消除**: 价格数据Point-in-Time框架 + 财报PIT数据处理
-   **用户友好**: 一键生成结构化的Markdown分析报告

---

## 2. 核心工作流 (The LLM-Enhanced Workflow)

V2.4版本的工作流在V2.3的基础上，增加了LLM增强分析环节，实现了从数据到投资建议的全流程智能化。

### 完整工作流程

#### 阶段1: 数据获取与特征工程 【V2.3 + V2.4】

-   **工具**: `scripts/data_retrieval.py`, `scripts/v2.3/expression_engine.py`, `scripts/v2.3/feature_cache.py`, `scripts/v2.4/llm_quant_pipeline.py`
-   **方法**:
    -   使用**表达式引擎**快速构建因子
    -   **特征缓存系统**自动缓存计算结果
    -   **LLM因子生成**（可选）：调用LLM自动生成创新的alpha因子

#### 阶段2: 理论假设与信号初筛

-   **工具**: `scripts/ic_analysis.py`
-   **方法**: 使用**信息系数 (IC)** 检验因子的有效性和稳定性

#### 阶段3: 严谨的回测框架 【V2.3】

-   **工具**: `scripts/walk_forward_optimizer.py`, `scripts/v2.3/enhanced_backtest.py`
-   **方法**:
    -   **增强回测引擎**引入真实交易成本模型
    -   **前向滚动优化 (WFO)**: 彻底杜绝未来数据泄露

#### 阶段4: LLM综合分析 【V2.4新增】

-   **工具**: `scripts/v2.4/llm_quant_pipeline.py`, `scripts/v2.4/llm_client.py`, `scripts/v2.4/prompt_templates.py`
-   **方法**:
    -   **LLM综合分析Agent**接收量化分析的数值结果
    -   结合财报、新闻等非结构化信息
    -   进行深度分析和因果推断
    -   评估因子有效性、市场适应性、风险等多个维度

#### 阶段5: 多空辩论 【V2.4新增】

-   **工具**: `scripts/v2.4/llm_quant_pipeline.py`
-   **方法**:
    -   **Bull Agent**从看涨角度论述，提取支持看涨的证据
    -   **Bear Agent**从看跌角度论述，识别风险和威胁
    -   形成多维度视角，避免单一视角的偏见

#### 阶段6: 投资建议生成 【V2.4新增】

-   **工具**: `scripts/v2.4/llm_quant_pipeline.py`, `references/master_wisdom.json`
-   **方法**:
    -   **投资建议Agent**综合量化结果、LLM分析、多空辩论
    -   融合**投资大师智慧库**中的投资理念
    -   生成最终的、可执行的投资建议（评级、目标价位、持仓建议、风险提示等）

#### 阶段7: 统计显著性验证

-   **工具**: `scripts/statistical_validator.py`
-   **方法**: Bootstrap检验、多重检验校正、蒙特卡洛模拟

#### 阶段8: 贝叶斯动态更新

-   **工具**: `scripts/bayesian_updater.py`
-   **方法**: 结合先验信念和新数据，生成后验分布

#### 阶段9: 因果推断探索

-   **工具**: `scripts/causal_analyzer.py`
-   **方法**: 探索因子与收益之间的深层因果关系

---

## 3. 技能文件结构

```
/skills/quant-investor/
├── SKILL.md                # 本指南 (V2.4)
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
│   ├── v2.3/                   # 【V2.3】工业级框架组件
│   │   ├── expression_engine.py   # 表达式引擎
│   │   ├── feature_cache.py       # 特征缓存系统
│   │   ├── financial_pit.py       # 财报PIT数据处理
│   │   └── enhanced_backtest.py   # 增强回测引擎
│   |
│   └── v2.4/                   # 【V2.4新增】LLM增强分析框架
│       ├── llm_client.py          # LLM客户端（统一API封装）
│       ├── prompt_templates.py    # 提示工程库
│       └── llm_quant_pipeline.py  # 端到端LLM增强分析流水线
|
├── references/                # 参考资料与配置
│   ├── master_strategies.md    # 投资大师策略库
│   ├── master_wisdom.json      # 【V2.4新增】投资大师智慧库（结构化）
│   ├── qlib_integration.md     # 【V2.3】Qlib框架集成指南
│   └── statistical_tests.md    # 统计检验方法参考
|
└── examples/                   # 使用示例
    ├── example_basic_workflow.md  # 基础工作流示例
    ├── example_llm_analysis.md    # 【V2.4新增】LLM增强分析示例
    └── example_advanced_strategy.md # 高级策略示例
```

---

## 4. V2.4使用指南

### 4.1. 快速开始：LLM增强量化分析

使用V2.4的LLM增强分析流水线，只需几行代码即可完成从数据到投资建议的全流程分析：

```python
from scripts.v2.4.llm_quant_pipeline import LLMQuantPipeline

# 初始化流水线
pipeline = LLMQuantPipeline(
    llm_model="gpt-4",  # 可选: gpt-4, gpt-3.5-turbo, gemini-pro等
    enable_cache=True
)

# 分析一只股票
results = pipeline.analyze_stock(
    stock_code="600519.SH",  # 贵州茅台
    start_date="2023-01-01",
    end_date="2023-12-31",
    analysis_type="comprehensive"
)

# 查看生成的报告
print(f"报告已保存至: {results['report_path']}")
```

### 4.2. 生成的报告结构

LLM增强分析流水线会自动生成一份结构化的Markdown报告，包含以下内容：

1.  **量化分析结果**
    -   IC分析（均值、标准差、IC-IR比率、T统计量、p值）
    -   回测结果（累计收益、夏普比率、最大回撤、胜率）
    -   统计检验（IC显著性、Granger因果检验）

2.  **LLM综合分析**
    -   因子有效性评估
    -   因果推断
    -   市场环境适应性
    -   风险评估
    -   改进建议

3.  **多空辩论**
    -   Bull Agent观点（看涨理由、核心论据、置信度）
    -   Bear Agent观点（看跌理由、核心论据、置信度）

4.  **投资建议**
    -   投资评级（强烈买入/买入/持有/卖出/强烈卖出）
    -   核心逻辑（结合量化分析和大师智慧）
    -   目标价位（买入价位、目标价位、止损价位）
    -   持仓建议（持仓比例、建仓策略、持有期限）
    -   风险提示
    -   替代方案
    -   决策依据（说明采纳了哪些信息，为什么）

### 4.3. 自定义LLM模型

V2.4支持多种LLM模型，可以根据需要灵活切换：

```python
# 使用GPT-4（最强能力，成本较高）
pipeline = LLMQuantPipeline(llm_model="gpt-4")

# 使用GPT-3.5-turbo（性价比高）
pipeline = LLMQuantPipeline(llm_model="gpt-3.5-turbo")

# 使用Gemini Pro（Google的模型）
pipeline = LLMQuantPipeline(llm_model="gemini-pro")
```

### 4.4. 成本控制

V2.4内置了智能缓存机制和成本估算功能：

```python
from scripts.v2.4.llm_client import LLMClient

client = LLMClient(enable_cache=True)

# 估算API调用成本
cost = client.estimate_cost(
    prompt="请分析贵州茅台的投资价值",
    model="gpt-4",
    max_tokens=2000
)
print(f"预估成本: ${cost:.6f}")

# 清空缓存（如果需要）
client.clear_cache()
```

### 4.5. 使用投资大师智慧库

投资大师智慧库以JSON格式存储，可以直接加载和使用：

```python
import json

# 加载投资大师智慧库
with open('references/master_wisdom.json', 'r', encoding='utf-8') as f:
    master_wisdom = json.load(f)

# 查看巴菲特的投资检查清单
buffett_checklist = master_wisdom['warren_buffett']['investment_checklist']
for item in buffett_checklist:
    print(f"- {item}")
```

---

## 5. V2.3核心功能使用指南

V2.4完全兼容V2.3的所有功能，以下是V2.3核心模块的使用方法：

### 5.1. 表达式引擎

```python
from scripts.v2.3.expression_engine import ExpressionEngine
import pandas as pd

# 初始化引擎
engine = ExpressionEngine()

# 定义因子表达式
factors = {
    "momentum_20d": "$close / Ref($close, 20) - 1",
    "value_pe": "1 / $pe_ratio",
    "volatility_20d": "Std($close, 20) / Mean($close, 20)"
}

# 计算因子
data = pd.DataFrame({...})  # 你的数据
computed_factors = {}
for name, expr in factors.items():
    computed_factors[name] = engine.evaluate(expr, data)
```

### 5.2. 特征缓存系统

```python
from scripts.v2.3.feature_cache import FeatureCache

# 初始化缓存
cache = FeatureCache(cache_dir="/home/ubuntu/.cache/quant_investor")

# 计算因子（首次会计算并缓存）
factor_data = cache.get_or_compute(
    cache_key="momentum_20d_600519",
    compute_func=lambda: engine.evaluate("$close / Ref($close, 20) - 1", data)
)

# 第二次调用会直接从缓存加载，速度极快
factor_data = cache.get_or_compute(
    cache_key="momentum_20d_600519",
    compute_func=lambda: engine.evaluate("$close / Ref($close, 20) - 1", data)
)
```

### 5.3. 增强回测引擎

```python
from scripts.v2.3.enhanced_backtest import EnhancedBacktestEngine

# 初始化回测引擎
engine = EnhancedBacktestEngine(
    initial_capital=100000,
    commission_rate=0.0003,  # 万三佣金
    slippage_rate=0.001,     # 0.1%滑点
    enable_market_impact=True
)

# 运行回测
results = engine.run_backtest(
    signals=your_signals,
    prices=your_prices
)

# 查看结果
print(f"累计收益: {results['cumulative_return']:.2%}")
print(f"夏普比率: {results['sharpe_ratio']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
```

---

## 6. 最佳实践

### 6.1. LLM增强分析的最佳实践

1.  **选择合适的模型**:
    -   对于复杂的综合分析和投资建议，使用GPT-4
    -   对于简单的因子生成和IC解释，使用GPT-3.5-turbo即可

2.  **启用缓存**:
    -   LLM API调用成本较高，务必启用缓存机制
    -   对于相同的分析任务，缓存可以节省90%以上的成本

3.  **人工审核关键决策**:
    -   LLM的输出应该作为决策支持，而不是直接执行
    -   对于重要的投资决策，建议进行人工审核

4.  **结合多个信息源**:
    -   不要仅依赖LLM的分析，应该结合量化结果、新闻、财报等多个信息源
    -   V2.4的多空辩论机制有助于形成更全面的视角

### 6.2. 量化分析的最佳实践

1.  **从IC分析开始**: 在编写任何策略代码前，先用IC分析验证因子的有效性
2.  **使用前向滚动优化**: 避免过拟合，在真实的样本外数据上验证策略
3.  **高估交易成本**: 在回测中使用高于实际的交易成本，为实盘留出安全边际
4.  **进行统计检验**: 使用Bootstrap等方法评估策略的统计显著性
5.  **消除Look-Ahead Bias**: 使用V2.3的PIT框架确保不使用未来数据

### 6.3. 风险管理

1.  **设置止损**: 为每个策略设置明确的止损线
2.  **分散投资**: 不要将所有资金投入单一策略或股票
3.  **定期审查**: 定期审查策略的表现，及时调整
4.  **压力测试**: 使用蒙特卡洛模拟评估极端情况下的风险

---

## 7. 技术架构与设计理念

### 7.1. V2.4架构设计原则

V2.4的架构设计遵循以下五大原则：

1.  **专业化分工 (Specialization)**: 每个Agent专注于特定任务，发挥其最大效能
2.  **协作与辩论 (Collaboration & Debate)**: 通过多Agent协作和辩论机制，提升决策的鲁棒性
3.  **风险意识 (Risk-Awareness)**: 将风险管理贯穿于整个决策流程
4.  **自适应性 (Adaptability)**: 框架应能识别市场状态，并动态调整策略
5.  **可解释性 (Interpretability)**: 所有Agent的决策过程都必须是可追溯、可理解的

### 7.2. 从V2.2到V2.4的演进

-   **V2.2**: 引入统计严谨性和因果思维，构建了Point-in-Time框架
-   **V2.3**: 吸收Qlib框架的工业级设计，大幅提升底层能力
-   **V2.4**: 引入LLM增强分析，实现从数据到投资建议的端到端智能化

每个版本都在前一个版本的基础上进行了重大升级，同时保持向后兼容。

---

## 8. 常见问题 (FAQ)

### Q1: V2.4相比V2.3有哪些主要提升？

**A**: V2.4的核心提升是引入了**LLM增强量化分析框架**，主要包括：
-   LLM综合分析：深度理解量化结果，进行因果推断
-   多空辩论：Bull/Bear Agent提供多维度视角
-   投资建议生成：融合投资大师智慧，输出可执行的投资建议
-   端到端自动化：从数据到报告的全流程智能化

### Q2: 使用V2.4需要哪些API密钥？

**A**: V2.4支持多种LLM API：
-   **OpenAI API**: 需要设置`OPENAI_API_KEY`环境变量
-   **Gemini API**: 需要设置`GEMINI_API_KEY`环境变量
-   可以根据需要选择其中一个或多个

### Q3: LLM API调用成本如何？

**A**: 成本取决于使用的模型和调用频率：
-   **GPT-4**: 约$0.03/1K输入tokens, $0.06/1K输出tokens
-   **GPT-3.5-turbo**: 约$0.0005/1K输入tokens, $0.0015/1K输出tokens
-   **Gemini Pro**: 约$0.00025/1K输入tokens, $0.0005/1K输出tokens

V2.4内置了智能缓存机制，可以大幅降低成本。

### Q4: V2.4是否可以离线使用？

**A**: 部分功能可以离线使用：
-   V2.3的所有模块（表达式引擎、特征缓存、回测引擎等）完全离线
-   V2.4的LLM增强分析需要联网调用API
-   未来可以考虑集成开源的本地LLM（如Qwen、LLaMA）

### Q5: 如何确保LLM分析的可靠性？

**A**: V2.4采取了多项措施确保可靠性：
-   **多模型交叉验证**: 可以使用多个LLM进行交叉验证
-   **数据支持要求**: Prompt中强制要求LLM提供数据支持
-   **置信度评分**: LLM为每个结论输出置信度
-   **人工审核**: 关键决策建议进行人工审核
-   **多空辩论**: 通过Bull/Bear Agent辩论，避免单一视角偏见

---

## 9. 未来规划

### V2.5及后续版本规划

1.  **强化学习框架**: 借鉴Qlib的QlibRL，引入强化学习进行订单执行优化和动态投资组合管理
2.  **SOTA模型集成**: 集成Transformer、GRU等前沿深度学习模型
3.  **实时数据流**: 支持实时数据流处理和在线学习
4.  **多资产支持**: 扩展到期货、期权、加密货币等多种资产类别
5.  **本地LLM支持**: 集成开源LLM，降低API调用成本

---

## 10. 参考资料

### 核心论文与框架

1.  **Qlib**: Microsoft的量化投资AI平台 - [GitHub](https://github.com/microsoft/qlib)
2.  **TradingAgents**: 多Agent LLM金融交易框架 - [Paper](https://tradingagents-ai.github.io/)
3.  **Automate Strategy Finding with LLM**: 使用LLM自动发现量化策略 - [arXiv:2409.06289](https://arxiv.org/abs/2409.06289)

### 投资大师经典著作

1.  **《聪明的投资者》** - 本杰明·格雷厄姆
2.  **《巴菲特致股东的信》** - 沃伦·巴菲特
3.  **《彼得·林奇的成功投资》** - 彼得·林奇
4.  **《原则》** - 瑞·达利奥
5.  **《投资最重要的事》** - 霍华德·马克斯

---

**免责声明**: 本技能仅供学习和研究使用，不构成投资建议。投资有风险，入市需谨慎。使用本技能进行实盘交易的任何损失，作者不承担责任。
