---
name: quant-investor
description: "一个端到端的AI量化投资技能，用于股票分析、模型训练和回测。当用户想要进行量化投资、分析股票、构建投资策略或进行回测时使用此技能。V2.5版本引入了完整的一手数据驱动分析框架，直接从Tushare/AKShare等专业数据源获取宏观、行业、个股的原始数据，结合V2.4的LLM增强分析能力，实现真正的data-driven智能投资。"
---

# 量化投资技能 (Quant-Investor) - V2.5

**版本**: 2.5  
**作者**: Manus AI  
**核心理念**: 一手数据驱动 + LLM深度分析 + 工业级框架设计 + 统计严谨性，构建真正data-driven的智能化量化投资平台。

---

## 1. 技能简介

`quant-investor` 技能 V2.5 是一个**一手数据驱动的LLM增强量化投资平台**。本版本在V2.4的LLM增强分析框架基础上，引入了完整的**专业数据源集成层**，实现了从底层原始数据到顶层投资决策的全链路自动化。

### V2.5核心创新：一手数据驱动分析框架

V2.5版本的核心突破是**彻底重构了数据获取层**，不再依赖二手的新闻稿或他人总结，而是直接从**Tushare Pro**和**AKShare**等专业数据源获取一手原始数据。

#### 数据覆盖范围

| 数据类型 | 覆盖内容 | 数据源 |
|:---|:---|:---|
| **宏观经济数据** | GDP、CPI、PPI、PMI、货币供应量（M0/M1/M2）、社会融资规模 | Tushare Pro (主) + AKShare (辅) |
| **行业数据** | 行业景气度指数、行业资金流向、行业估值水平（PE/PB） | Tushare Pro + AKShare |
| **个股行情数据** | 日线、分钟线价格和成交量、前复权/后复权/不复权 | Tushare Pro (主) + AKShare (辅) |
| **个股基本面数据** | 财务报表（资产负债表、利润表、现金流量表）、财务指标（ROE、毛利率、负债率等）、业绩预告 | Tushare Pro (主) + AKShare (辅) |
| **资金流向数据** | 主力资金、散户资金、融资融券数据 | Tushare Pro + AKShare (主) |
| **概念/行业板块** | 概念板块列表、行业板块列表、成分股 | AKShare (主) |
| **市场情绪数据** | 龙虎榜、人气榜、热度排名 | AKShare |

#### V2.5架构图

```
┌─────────────────────────────────────────────────────────────┐
│                  应用层：V2.4 LLM增强分析框架                  │
│  (数据收集Agent → 特征工程Agent → 量化分析Agent →             │
│   LLM综合分析Agent → Bull/Bear辩论 → 投资建议Agent)           │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                    数据服务层：统一数据API                     │
│         (DataAPI: 为上层提供统一、简洁的数据访问接口)          │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│              数据处理层：数据清洗、标准化、指标计算              │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│          数据获取与存储层：DataSourceManager                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │TushareClient │  │ AKShareClient│  │ 缓存系统      │      │
│  │  (主数据源)   │  │  (辅数据源)   │  │(Redis/Disk) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         ↓                  ↓                                 │
│  ┌──────────────┐  ┌──────────────┐                        │
│  │ Tushare API  │  │  AKShare API │                        │
│  └──────────────┘  └──────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

#### V2.5核心组件

1.  **TushareClient** (`scripts/v2.5/data_acquisition/tushare_client.py`)
    - 封装Tushare Pro的核心API
    - 股票、财务、宏观、资金流向等数据的统一调用接口
    - 自动频率限制和权限检测

2.  **AKShareClient** (`scripts/v2.5/data_acquisition/akshare_client.py`)
    - 封装AKShare的核心API
    - 作为Tushare的补充数据源
    - 提供概念板块、市场情绪等特色数据

3.  **DataSourceManager** (`scripts/v2.5/data_acquisition/data_source_manager.py`)
    - **智能调度**：根据数据类型和可用性，自动选择最优数据源
    - **自动降级**：当主数据源（Tushare）不可用或权限不足时，自动切换到备用数据源（AKShare）
    - **智能缓存**：使用Parquet格式缓存数据，大幅降低API调用成本和时间
    - **权限管理**：当检测到Tushare积分不足时，立即提示用户解决

4.  **DataAPI** (`scripts/v2.5/data_service/data_api.py`)
    - 为上层应用提供统一、简洁、语义化的数据访问接口
    - 自动处理股票代码标准化、日期计算等细节
    - 屏蔽底层数据源的差异

5.  **IntegratedAnalysisPipeline** (`scripts/v2.5/integrated_analysis_pipeline.py`)
    - 将V2.5数据层与V2.4的LLM分析框架深度集成
    - 实现从一手数据到投资建议的端到端自动化
    - 自动生成结构化的量化分析报告

#### V2.5的核心价值

| 价值 | 描述 |
|:---|:---|
| **Data-Driven (数据驱动)** | 真正基于一手原始数据的量化分析，而非二手新闻或总结 |
| **Cost-Effective (成本效益)** | 智能缓存和双数据源策略，大幅降低数据成本 |
| **Reliable (稳定可靠)** | 多数据源容错机制，保证分析的连续性 |
| **Professional (专业深度)** | 数据覆盖的广度和深度远超以往，支持更复杂的量化策略 |
| **Integrated (深度集成)** | 与V2.4的LLM分析框架无缝集成，形成完整的智能投资平台 |

### V2.4核心能力（已集成）

V2.5继承了V2.4的所有LLM增强分析能力：

1.  **LLM客户端模块**: 统一封装OpenAI和Gemini API，智能缓存，成本估算
2.  **提示工程库**: 8个标准化Prompt模板，5种专业角色
3.  **投资大师智慧库**: 6位投资大师的投资哲学和核心原则
4.  **端到端分析流水线**: 六大分析阶段，自动生成Markdown报告

### V2.3核心能力（已集成）

V2.5继承了V2.3的所有工业级基础设施：

1.  **表达式引擎**: 使用简洁的公式化字符串定义因子
2.  **特征缓存系统**: 自动缓存特征工程结果
3.  **财报PIT数据处理**: 处理财务报表数据的多次修订问题
4.  **增强回测引擎**: 引入滑点、市场冲击等真实交易成本模型

### 技能特色

-   **一手数据驱动**: 直接从Tushare/AKShare获取原始数据，不依赖二手信息
-   **智能数据调度**: 自动选择最优数据源，自动降级和容错
-   **LLM增强分析**: 将量化结果交给大模型进行深度分析和因果推断
-   **多Agent协作**: Bull/Bear Agent辩论机制，提供多维度视角
-   **投资智慧融合**: 集成6位投资大师的策略
-   **工业级架构**: 借鉴Qlib和TradingAgents的先进设计
-   **统计严谨性**: IC分析、Bootstrap测试、贝叶斯框架、因果推断
-   **完整的Look-Ahead Bias消除**: 价格数据Point-in-Time框架 + 财报PIT数据处理

---

## 2. 快速开始

### 2.1 环境配置

**必需**：
- Tushare Pro Token（用户需提供）
- Python 3.11+
- 依赖库：`tushare`, `akshare`, `pandas`, `numpy`, `pyarrow`

**可选**：
- OpenAI API Key（用于LLM增强分析）
- Gemini API Key（用于LLM增强分析）

### 2.2 使用示例

#### 示例1：分析单只股票（V2.5完整流程）

```python
import sys
sys.path.append('/home/ubuntu/skills/quant-investor/scripts/v2.5')
from integrated_analysis_pipeline import IntegratedAnalysisPipeline

# 初始化流水线（需要提供Tushare Token）
pipeline = IntegratedAnalysisPipeline(tushare_token='YOUR_TUSHARE_TOKEN')

# 分析单只股票
result = pipeline.analyze_stock('600519', '贵州茅台')

# 查看报告
print(f"报告路径: {result['report_path']}")
```

**输出**：
- 自动生成包含价格分析、技术指标、基本面分析的完整Markdown报告
- 所有数据来自Tushare Pro一手数据源
- 自动计算MA、RSI等技术指标
- 提供数据质量说明和风险提示

#### 示例2：获取宏观经济数据

```python
from data_api import DataAPI

api = DataAPI(tushare_token='YOUR_TUSHARE_TOKEN')

# 获取GDP数据
gdp_data = api.get_gdp()
print(gdp_data.tail())

# 获取CPI数据
cpi_data = api.get_cpi()
print(cpi_data.tail())
```

#### 示例3：获取股票价格和财务数据

```python
from data_api import DataAPI

api = DataAPI(tushare_token='YOUR_TUSHARE_TOKEN')

# 获取最近30天的价格数据（前复权）
price_data = api.get_stock_price('600519.SH', days=30, adjust='qfq')
print(price_data.head())

# 获取最新的财务指标
financial_data = api.get_stock_financial('600519.SH')
print(financial_data.head())

# 获取资金流向
fund_flow = api.get_stock_fund_flow('600519.SH', days=30)
print(fund_flow.head())
```

---

## 3. 核心工作流 (The Data-Driven LLM-Enhanced Workflow)

V2.5版本的工作流在V2.4的基础上，增加了一手数据获取环节，实现了从底层原始数据到顶层投资建议的全链路自动化。

### 完整工作流程

#### 阶段1: 一手数据获取 【V2.5 NEW】

-   **工具**: `scripts/v2.5/data_acquisition/data_source_manager.py`, `scripts/v2.5/data_service/data_api.py`
-   **方法**:
    -   使用**DataSourceManager**智能调度Tushare和AKShare
    -   自动缓存数据，避免重复API调用
    -   获取宏观、行业、个股的一手原始数据

#### 阶段2: 特征工程与量化分析 【V2.3 + V2.4】

-   **工具**: `scripts/v2.3/expression_engine.py`, `scripts/v2.3/feature_cache.py`, `scripts/ic_analysis.py`
-   **方法**:
    -   使用**表达式引擎**快速构建因子
    -   **特征缓存系统**自动缓存计算结果
    -   使用**IC分析**检验因子的有效性

#### 阶段3: LLM增强分析 【V2.4】

-   **工具**: `scripts/v2.4/llm_quant_pipeline.py`, `scripts/v2.4/llm_client.py`
-   **方法**:
    -   **LLM综合分析Agent**：对量化结果进行深度分析和因果推断
    -   **Bull/Bear辩论**：多维度视角评估投资机会
    -   **投资建议Agent**：融合投资大师智慧，给出最终建议

#### 阶段4: 回测与验证 【V2.3】

-   **工具**: `scripts/v2.3/enhanced_backtest.py`, `scripts/walk_forward_optimizer.py`
-   **方法**:
    -   使用**增强回测引擎**，引入滑点、市场冲击等真实交易成本
    -   **Walk-Forward优化**，避免过拟合

---

## 4. 核心模块详解

### 4.1 数据获取层 (V2.5)

#### TushareClient

**功能**：封装Tushare Pro API，提供统一的数据获取接口。

**核心方法**：
- `get_stock_basic()`: 获取股票列表
- `get_daily()`: 获取日线行情
- `get_fina_indicator()`: 获取财务指标
- `get_cn_gdp()`, `get_cn_cpi()`: 获取宏观数据
- `get_moneyflow()`: 获取资金流向

**特性**：
- 自动频率限制（200ms间隔）
- 权限检测（积分不足时抛出`PermissionError`）
- API调用统计

#### AKShareClient

**功能**：封装AKShare API，作为Tushare的补充数据源。

**核心方法**：
- `get_stock_zh_a_hist()`: 获取历史行情
- `get_macro_china_gdp()`, `get_macro_china_cpi()`: 获取宏观数据
- `get_stock_individual_fund_flow()`: 获取资金流向
- `get_stock_board_concept_name_em()`: 获取概念板块

**特性**：
- 免费使用，无需Token
- 数据覆盖广泛，包括概念板块、市场情绪等特色数据

#### DataSourceManager

**功能**：智能调度Tushare和AKShare，实现自动降级和容错。

**核心特性**：
1. **智能调度**：根据数据类型和可用性，自动选择最优数据源
2. **自动降级**：当主数据源不可用时，自动切换到备用数据源
3. **智能缓存**：使用Parquet格式缓存数据，TTL可配置
4. **权限管理**：检测到Tushare权限不足时，立即提示用户

**核心方法**：
- `get_stock_list()`: 获取股票列表
- `get_daily_price()`: 获取日线行情
- `get_financial_indicator()`: 获取财务指标
- `get_macro_gdp()`, `get_macro_cpi()`: 获取宏观数据
- `get_fund_flow()`: 获取资金流向

### 4.2 数据服务层 (V2.5)

#### DataAPI

**功能**：为上层应用提供统一、简洁、语义化的数据访问接口。

**核心方法**：
- `get_stock_list(market='all')`: 获取股票列表
- `get_stock_price(stock_code, days=250, adjust='qfq')`: 获取价格数据
- `get_stock_financial(stock_code)`: 获取财务指标
- `get_stock_fund_flow(stock_code, days=30)`: 获取资金流向
- `get_gdp()`, `get_cpi()`: 获取宏观数据
- `normalize_stock_code(code)`: 标准化股票代码

**特性**：
- 自动处理日期计算（如"最近30天"）
- 自动标准化股票代码（600519 → 600519.SH）
- 屏蔽底层数据源的差异

### 4.3 集成分析流水线 (V2.5)

#### IntegratedAnalysisPipeline

**功能**：将V2.5数据层与V2.4的LLM分析框架深度集成，实现端到端自动化。

**核心方法**：
- `analyze_stock(stock_code, stock_name)`: 对单只股票进行完整的量化分析
- `analyze_market()`: 分析整体市场环境（宏观数据）

**分析流程**：
1. 获取价格数据（前复权）
2. 获取财务数据
3. 计算技术指标（MA、RSI等）
4. 生成分析报告

**输出**：
- 结构化的Markdown分析报告
- 包含价格分析、技术指标、基本面分析、综合评估
- 提供数据质量说明和风险提示

### 4.4 LLM增强分析层 (V2.4)

#### LLMClient

**功能**：统一封装OpenAI和Gemini API。

**特性**：
- 智能缓存，避免重复调用
- 成本估算
- 自动重试和错误处理

#### PromptTemplates

**功能**：提供标准化的Prompt模板。

**模板类型**：
- 因子生成
- IC分析解释
- 回测结果解释
- LLM综合分析
- 多空辩论
- 投资建议

#### LLMQuantPipeline

**功能**：端到端的LLM增强量化分析流水线。

**六大分析阶段**：
1. 数据收集
2. 特征工程（LLM生成因子）
3. 量化分析（IC分析、回测、统计检验）
4. LLM综合分析
5. 多空辩论（Bull/Bear Agent）
6. 投资建议（融合投资大师智慧）

---

## 5. 数据质量保证

### 5.1 Look-Ahead Bias消除

V2.5继承了V2.2和V2.3的Look-Ahead Bias消除机制：

1.  **价格数据**：使用前复权（qfq）处理，并在报告中明确说明
2.  **财报数据**：使用PIT（Point-in-Time）数据库处理财报的多次修订问题

### 5.2 数据来源说明

所有分析报告都会明确标注数据来源和数据质量说明：

```markdown
**数据质量说明**：
- 价格数据：采用前复权（qfq）处理，已考虑分红送股对价格的影响
- 财务数据：来自上市公司定期报告，具有滞后性
- 数据来源：Tushare Pro（主）+ AKShare（辅）
```

### 5.3 缓存机制

V2.5引入了智能缓存机制，大幅降低API调用成本：

- **缓存格式**：Parquet（高效、压缩）
- **缓存目录**：`/home/ubuntu/.quant_cache`
- **TTL配置**：
  - 股票列表：7天
  - 日线数据：1天
  - 财务数据：30天
  - 宏观数据：30天

---

## 6. 使用注意事项

### 6.1 Tushare权限

V2.5需要Tushare Pro的Token才能正常工作。当检测到Tushare积分不足时，系统会：

1. 抛出`PermissionError`
2. 打印明确的错误信息
3. 提示用户解决权限问题
4. 自动尝试使用AKShare作为备用数据源

**用户需要**：
- 注册Tushare Pro账号
- 获取Token
- 根据需要的数据接口，确保有足够的积分

### 6.2 数据源选择

V2.5的数据源优先级策略：

| 数据类型 | 优先级 |
|:---|:---|
| 股票基础数据 | Tushare (主) → AKShare (辅) |
| 日线行情 | Tushare (主) → AKShare (辅) |
| 财务数据 | Tushare (主) → AKShare (辅) |
| 宏观数据 | Tushare (主) → AKShare (辅) |
| 资金流向 | AKShare (主) → Tushare (辅) |
| 概念板块 | AKShare (唯一) |
| 市场情绪 | AKShare (唯一) |

### 6.3 LLM API配置

如果需要使用V2.4的LLM增强分析功能，需要配置：

- OpenAI API Key（环境变量：`OPENAI_API_KEY`）
- 或 Gemini API Key（环境变量：`GEMINI_API_KEY`）

---

## 7. 版本演进

| 版本 | 核心特性 | 发布时间 |
|:---|:---|:---|
| **V2.5** | 一手数据驱动分析框架，集成Tushare/AKShare | 2026-01-31 |
| **V2.4** | LLM增强量化分析框架，多Agent协作 | 2026-01-30 |
| **V2.3** | 工业级基础设施，表达式引擎、特征缓存、增强回测 | 2026-01-29 |
| **V2.2** | 统计严谨性和因果思维，IC分析、Bootstrap测试 | 2026-01 |
| **V2.1** | 投资大师策略融合 | 2025-12 |
| **V2.0** | 基础量化分析框架 | 2025-11 |

---

## 8. 参考资料

### 8.1 数据源文档

- [Tushare Pro官方文档](https://tushare.pro/document/2)
- [AKShare官方文档](https://akshare.akfamily.xyz/)

### 8.2 技术参考

- [Qlib框架](https://qlib.readthedocs.io/)
- [TradingAgents](https://tradingagents-ai.github.io/)

### 8.3 内部文档

- `references/qlib_integration.md`: Qlib框架集成指南
- `references/master_wisdom.json`: 投资大师智慧库

---

## 9. 联系与支持

如有问题或建议，请通过以下方式联系：

- GitHub: [maxwelllee54/myQuant](https://github.com/maxwelllee54/myQuant)
- Manus帮助中心: [https://help.manus.im](https://help.manus.im)

---

**免责声明**：本技能仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。
