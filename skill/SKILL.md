---
name: quant-investor
description: "一个端到端的AI量化投资技能，用于股票分析、模型训练和回测。当用户想要进行量化投资、分析股票、构建投资策略或进行回测时使用此技能。V2.6版本引入了完整的美股宏观数据层，支持FRED、Finnhub、yfinance等专业数据源，获取美联储利率、非农就业、CPI等核心宏观经济指标，结合V2.5的A股数据层和V2.4的LLM增强分析能力，实现真正的全球化data-driven智能投资。"
---

# 量化投资技能 (Quant-Investor) - V2.6

**版本**: 2.6 (V2.5.2 A股高级数据接口已集成)  
**作者**: Manus AI  
**核心理念**: 全球化一手数据驱动 + LLM深度分析 + 工业级框架设计 + 统计严谨性，构建真正data-driven的智能化量化投资平台。

---

## 1. 技能简介

`quant-investor` 技能 V2.6 是一个**全球化一手数据驱动的LLM增强量化投资平台**。本版本在V2.5的A股数据层基础上，引入了完整的**美股宏观数据层**，实现了对美国市场的全面数据覆盖。

### V2.6核心创新：美股宏观数据层

V2.6版本的核心突破是**构建了专业的美股数据获取体系**，直接从**FRED（美联储经济数据库）**、**Finnhub**、**yfinance**等权威数据源获取一手原始数据。

#### 美股数据覆盖范围

| 数据类型 | 覆盖内容 | 数据源 |
|:---|:---|:---|
| **宏观经济数据** | GDP、CPI、PCE（美联储首选通胀指标）、PPI、失业率、非农就业 | FRED (官方权威) |
| **利率数据** | 联邦基金利率、国债收益率（3M/2Y/10Y）、收益率曲线 | FRED |
| **货币数据** | M2货币供应量、美元指数 | FRED + yfinance |
| **市场指数** | 标普500、纳斯达克、道琼斯、VIX恐慌指数 | yfinance (主) + FRED (辅) |
| **美股行情** | 日线、分钟线、历史数据 | yfinance (主) + Finnhub (辅) |
| **实时报价** | 美股实时价格、涨跌幅 | Finnhub |
| **公司信息** | 公司简介、市值、行业、财务指标 | Finnhub + yfinance |
| **新闻数据** | 公司新闻、市场新闻 | Finnhub |
| **经济日历** | 即将发布的经济数据、财报日历 | Finnhub |

#### V2.6架构图

```
┌─────────────────────────────────────────────────────────────┐
│                  应用层：V2.4 LLM增强分析框架                  │
│  (数据收集Agent → 特征工程Agent → 量化分析Agent →             │
│   LLM综合分析Agent → Bull/Bear辩论 → 投资建议Agent)           │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                    数据服务层：统一数据API                     │
│    ┌──────────────────┐  ┌──────────────────┐              │
│    │   A股数据API      │  │   美股数据API     │              │
│    │   (V2.5)         │  │   (V2.6 NEW)     │              │
│    └──────────────────┘  └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                    数据获取与存储层                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              A股数据层 (V2.5)                        │   │
│  │  TushareClient + AKShareClient + DataSourceManager  │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              美股数据层 (V2.6 NEW)                   │   │
│  │  FREDClient + YFinanceClient + FinnhubClient        │   │
│  │              + USMacroDataManager                   │   │
│  └─────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              缓存系统 (Redis/Disk)                   │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

#### V2.6核心组件

1.  **FREDClient** (`scripts/v2.6/us_macro_data/fred_client.py`)
    - 封装FRED（美联储经济数据库）API
    - 提供GDP、CPI、PCE、失业率、非农就业、联邦基金利率、国债收益率等核心宏观数据
    - 支持收益率曲线分析

2.  **YFinanceClient** (`scripts/v2.6/us_macro_data/yfinance_client.py`)
    - 封装yfinance库
    - 提供美股历史行情、公司信息、财务报表
    - 提供市场指数（标普500、纳斯达克、VIX等）

3.  **FinnhubClient** (`scripts/v2.6/us_macro_data/finnhub_client.py`)
    - 封装Finnhub API
    - 提供实时报价、公司新闻、市场新闻
    - 提供经济日历、财报日历、财务指标

4.  **USMacroDataManager** (`scripts/v2.6/us_macro_data/us_macro_data_manager.py`)
    - **智能调度**：根据数据类型自动选择最优数据源
    - **自动降级**：当主数据源不可用时，自动切换到备用数据源
    - **智能缓存**：使用磁盘缓存，大幅降低API调用成本
    - **统一接口**：为上层应用提供统一的数据访问接口

#### 数据源优先级策略

| 数据类型 | 优先级 |
|:---|:---|
| 宏观经济数据 | FRED (官方权威) |
| 美股行情数据 | yfinance > Finnhub > Tushare |
| 实时报价/新闻 | Finnhub > yfinance |
| 市场指数数据 | yfinance > FRED |
| 公司信息 | yfinance > Finnhub |

#### V2.6的核心价值

| 价值 | 描述 |
|:---|:---|
| **全球化覆盖** | 同时支持A股和美股市场，实现全球化投资分析 |
| **官方权威数据** | 宏观经济数据直接来自FRED（美联储经济数据库），权威可靠 |
| **多源容错** | 四个数据源（FRED、yfinance、Finnhub、Tushare）智能调度，确保数据可用性 |
| **实时性** | Finnhub提供实时报价和新闻，支持盘中分析 |
| **深度分析** | 收益率曲线、经济日历等专业分析工具 |

---

## 2. 快速开始

### 2.1 环境配置

**A股分析必需**：
- Tushare Pro Token

**美股分析必需**：
- FRED API Key（免费注册：https://fred.stlouisfed.org/docs/api/api_key.html）
- Finnhub API Key（免费注册：https://finnhub.io/）

**可选**：
- OpenAI API Key（用于LLM增强分析）
- Gemini API Key（用于LLM增强分析）

**依赖库**：
```bash
pip install tushare akshare yfinance pandas numpy pyarrow requests
```

### 2.2 使用示例

#### 示例1：获取美国宏观经济数据快照

```python
import sys
sys.path.append('/home/ubuntu/skills/quant-investor/scripts/v2.6')
from us_macro_data import USMacroDataManager

# 初始化管理器（需要提供FRED和Finnhub API密钥）
manager = USMacroDataManager(
    fred_api_key='YOUR_FRED_API_KEY',
    finnhub_api_key='YOUR_FINNHUB_API_KEY'
)

# 获取宏观经济数据快照
snapshot = manager.get_macro_snapshot()
print(f"时间戳: {snapshot['timestamp']}")
for key, data in snapshot['data'].items():
    print(f"{key}: {data['value']:.2f} {data['unit']} ({data['date']})")
```

**输出示例**：
```
时间戳: 2026-02-01 10:09:41
gdp: 24026.83 Billion USD (Real) (2025-07-01)
cpi_core: 331.86 Index (2025-12-01)
unemployment: 4.40 % (2025-12-01)
fed_rate: 3.64 % (2026-01-29)
treasury_10y: 4.24 % (2026-01-29)
vix: 17.44 Index (2026-01-30)
```

#### 示例2：分析美股个股

```python
from us_macro_data import USMacroDataManager

manager = USMacroDataManager()

# 获取AAPL历史行情
aapl = manager.get_stock_history('AAPL', period='1mo')
print(f"获取到 {len(aapl)} 条数据")
print(f"最新价格: ${aapl.iloc[-1]['close']:.2f}")

# 获取实时报价
quote = manager.get_stock_quote('AAPL')
print(f"实时价格: ${quote['price']:.2f} ({quote['change_pct']:+.2f}%)")

# 获取公司新闻
news = manager.get_company_news('AAPL', days=3)
print(f"获取到 {len(news)} 条新闻")
```

#### 示例3：获取收益率曲线

```python
from us_macro_data import USMacroDataManager

manager = USMacroDataManager()

# 获取收益率曲线
yield_curve = manager.get_yield_curve()
print(f"3个月国债: {yield_curve['3m']:.2f}%")
print(f"2年期国债: {yield_curve['2y']:.2f}%")
print(f"10年期国债: {yield_curve['10y']:.2f}%")
print(f"10Y-2Y利差: {yield_curve['spread_10y_2y']:.2f}%")
print(f"曲线形态: {yield_curve['curve_shape']}")
```

#### 示例4：获取经济日历

```python
from us_macro_data import USMacroDataManager

manager = USMacroDataManager()

# 获取经济日历
calendar = manager.get_economic_calendar()
for event in calendar[:5]:
    print(f"{event['date']} - {event['event']} ({event['country']})")
```

---

## 3. 核心工作流

### 3.1 美股分析工作流

#### 阶段1: 宏观环境分析

-   **工具**: `USMacroDataManager.get_macro_snapshot()`
-   **内容**:
    -   GDP增长率
    -   通胀指标（CPI、PCE）
    -   就业市场（失业率、非农就业）
    -   货币政策（联邦基金利率、收益率曲线）
    -   市场情绪（VIX恐慌指数）

#### 阶段2: 个股数据获取

-   **工具**: `USMacroDataManager.get_stock_history()`, `get_stock_info()`, `get_basic_financials()`
-   **内容**:
    -   历史价格数据
    -   公司基本信息
    -   财务指标（PE、PB、ROE等）

#### 阶段3: 实时信息获取

-   **工具**: `USMacroDataManager.get_stock_quote()`, `get_company_news()`, `get_market_news()`
-   **内容**:
    -   实时报价
    -   公司新闻
    -   市场新闻

#### 阶段4: LLM增强分析

-   **工具**: V2.4的LLM分析框架
-   **内容**:
    -   宏观环境解读
    -   个股基本面分析
    -   技术面分析
    -   多空辩论
    -   投资建议

---

## 4. 核心模块详解

### 4.0 A股宏观数据层 (V2.5.1 NEW)

**V2.5.1更新**：对标美股V2.6的数据结构，为A股增加了完整的宏观经济数据支持。

#### CNMacroDataManager

**功能**：中国宏观数据管理器，提供完整的A股宏观经济数据支持。

**数据覆盖范围**：

| 数据类型 | 覆盖内容 | 数据源 |
|:---|:---|:---|
| **宏观经济数据** | GDP、CPI、PPI、PMI | Tushare (主) + AKShare (辅) |
| **货币政策数据** | LPR、Shibor、国债收益率 | Tushare (主) + AKShare (辅) |
| **货币供应数据** | M0、M1、M2、社会融资规模 | Tushare (主) + AKShare (辅) |
| **市场指数数据** | 上证、深证、沪深300、中证500、创业板、科创50 | Tushare (主) + AKShare (辅) |

**核心方法**：
- `get_gdp()`: 获取GDP数据
- `get_cpi()`: 获取CPI数据
- `get_ppi()`: 获取PPI数据
- `get_pmi()`: 获取PMI数据（制造业采购经理指数）
- `get_lpr()`: 获取LPR贷款基础利率
- `get_shibor()`: 获取Shibor上海银行间同业拆放利率
- `get_bond_yield()`: 获取中国国债收益率
- `get_yield_curve()`: 获取中国国债收益率曲线
- `get_money_supply()`: 获取货币供应量数据（M0、M1、M2）
- `get_social_financing()`: 获取社会融资规模数据
- `get_index_daily()`: 获取指数日线行情
- `get_market_indices()`: 获取主要市场指数的最新数据
- `get_macro_snapshot()`: 获取中国宏观经济数据快照

**使用示例**：

```python
import sys
sys.path.append('/home/ubuntu/skills/quant-investor/scripts/v2.5/cn_macro_data')
from cn_macro_data_manager import CNMacroDataManager

# 初始化管理器
manager = CNMacroDataManager(tushare_token='YOUR_TUSHARE_TOKEN')

# 获取宏观数据快照
snapshot = manager.get_macro_snapshot()
print(f"GDP: {snapshot['economy']['gdp']}")
print(f"CPI: {snapshot['economy']['cpi']}")
print(f"PMI: {snapshot['economy']['pmi']}")
print(f"LPR: {snapshot['monetary_policy']['lpr']}")
print(f"Shibor: {snapshot['monetary_policy']['shibor']}")
```

**A股与美股数据结构对比**：

| 数据类型 | A股 (V2.5.1) | 美股 (V2.6) |
|:---|:---|:---|
| 宏观经济 | GDP、CPI、PPI、PMI | GDP、CPI、PCE、PPI |
| 货币政策 | LPR、Shibor、国债收益率 | 联邦基金利率、国债收益率 |
| 货币供应 | M0、M1、M2、社会融资规模 | M2 |
| 市场指数 | 上证、深证、沪深300、中证500 | 标普500、纳斯达克、道琼斯 |
| 市场情绪 | - | VIX恐慌指数 |

### 4.0.1 A股高级数据接口 (一万分权限)

**V2.5.2更新**：充分利用Tushare一万分权限，解锁并集成所有高级数据接口，全面强化情绪、筹码、衍生品等数据获取能力。

#### TushareClientExtended

**功能**：Tushare Pro扩展客户端，提供一万分权限解锁的高级数据接口。

**数据覆盖范围**：

| 数据类别 | 接口 | 描述 | 价值 |
|:---|:---|:---|:---|
| **龙虎榜** | `get_top_list()`, `get_top_inst()` | 每日龙虎榜单，机构席位追踪 | 识别游资和机构的短线动向 |
| **融资融券** | `get_margin()`, `get_margin_detail()` | 融资余额、融券余额变化 | 判断市场杠杆水平和多空情绪 |
| **股东数据** | `get_stk_holdernumber()`, `get_stk_holdertrade()` | 股东人数变化，重要股东增减持 | 分析筹码集中度和"聪明钱"动向 |
| **大宗交易** | `get_block_trade()` | 折价、溢价交易情况 | 洞察产业资本和机构的真实意图 |
| **沪深股通** | `get_hk_hold()` | 北向资金持仓明细 | 跟踪外资的配置偏好 |
| **限售解禁** | `get_share_float()` | 限售股解禁数量、比例 | 预判减持压力 |
| **期权数据** | `get_opt_basic()`, `get_opt_daily()` | 期权合约、日线行情 | 捕捉市场波动率预期 |
| **期货数据** | `get_fut_basic()`, `get_fut_daily()`, `get_fut_holding()` | 期货合约、行情、持仓排名 | 宏观对冲，跨品种套利 |
| **基金数据** | `get_fund_basic()`, `get_fund_nav()`, `get_fund_portfolio()` | 公募基金列表、净值、持仓明细 | 跟踪明星基金经理的调仓路径 |
| **指数数据** | `get_index_basic()`, `get_index_daily()`, `get_index_weight()`, `get_index_dailybasic()` | 指数信息、行情、成分股权重、每日指标 | 指数增强、ETF套利策略 |
| **行业分类** | `get_index_classify()`, `get_index_member()` | 申万行业分类及成分股 | 构建行业轮动策略 |
| **可转债** | `get_cb_basic()`, `get_cb_daily()` | 可转债基础信息、日线数据 | 可转债套利策略 |
| **外汇数据** | `get_fx_obasic()`, `get_fx_daily()` | 外汇基础信息、日线行情 | 汇率风险分析 |

**使用示例**：

```python
import sys
sys.path.append('/home/ubuntu/skills/quant-investor/scripts/v2.5/data_acquisition')
from tushare_client_extended import TushareClientExtended

# 初始化客户端（使用自定义API服务）
client = TushareClientExtended(
    token='YOUR_TUSHARE_TOKEN',
    http_url='http://lianghua.nanyangqiankun.top'  # 可选：自定义API服务地址
)

# 获取龙虎榜数据
top_list = client.get_top_list(trade_date='20260128')
print(f"龙虎榜数据: {len(top_list)}条")

# 获取融资融券数据
margin = client.get_margin(trade_date='20260128')
print(f"融资融券汇总: {len(margin)}条")

# 获取股东人数变化
holder_num = client.get_stk_holdernumber(ts_code='600519.SH')
print(f"贵州茅台股东人数: {len(holder_num)}条")

# 获取北向资金持股
hk_hold = client.get_hk_hold(ts_code='600519.SH')
print(f"贵州茅台北向持股: {len(hk_hold)}条")

# 获取沪深300成分股权重
index_weight = client.get_index_weight(index_code='000300.SH')
print(f"沪深300成分股: {len(index_weight)}条")

# 获取申万一级行业分类
sw_class = client.get_index_classify(level='L1')
print(f"申万一级行业: {len(sw_class)}条")
```

**核心价值**：

| 价值 | 描述 |
|:---|:---|
| **情绪洞察** | 龙虎榜、融资融券、大宗交易等数据揭示市场情绪和资金动向 |
| **筹码分析** | 股东人数、北向资金、限售解禁等数据分析筹码结构 |
| **衍生品覆盖** | 期权、期货、可转债等衍生品数据支持复杂策略 |
| **行业轮动** | 申万行业分类、指数成分权重支持行业轮动策略 |
| **基金跟踪** | 公募基金持仓数据跟踪明星基金经理调仓 |

---

### 4.1 美股数据层 (V2.6)

#### FREDClient

**功能**：封装FRED（美联储经济数据库）API，提供权威的宏观经济数据。

**核心方法**：
- `get_gdp(real=True)`: 获取GDP数据
- `get_cpi(core=False)`: 获取CPI数据
- `get_pce(core=True)`: 获取PCE数据（美联储首选通胀指标）
- `get_unemployment()`: 获取失业率
- `get_nfp()`: 获取非农就业人数
- `get_fed_rate()`: 获取联邦基金利率
- `get_treasury_yield(maturity='10y')`: 获取国债收益率
- `get_yield_curve()`: 获取收益率曲线
- `get_m2()`: 获取M2货币供应量
- `get_vix()`: 获取VIX恐慌指数

**特性**：
- 官方权威数据源
- 智能缓存（可配置TTL）
- API调用统计

#### YFinanceClient

**功能**：封装yfinance库，提供美股行情和公司信息。

**核心方法**：
- `get_stock_history(ticker, period, interval)`: 获取历史行情
- `get_stock_info(ticker)`: 获取公司信息
- `get_financials(ticker)`: 获取财务报表
- `get_index_history(index_name, period)`: 获取指数历史

**特性**：
- 免费使用，无需API密钥
- 数据覆盖全面
- 支持多种时间周期

#### FinnhubClient

**功能**：封装Finnhub API，提供实时数据和新闻。

**核心方法**：
- `get_quote(ticker)`: 获取实时报价
- `get_company_profile(ticker)`: 获取公司简介
- `get_basic_financials(ticker)`: 获取财务指标
- `get_company_news(ticker)`: 获取公司新闻
- `get_market_news(category)`: 获取市场新闻
- `get_economic_calendar()`: 获取经济日历
- `get_earnings_calendar()`: 获取财报日历

**特性**：
- 实时数据
- 丰富的新闻和日历功能
- 智能缓存

#### USMacroDataManager

**功能**：智能调度多个数据源，提供统一的数据访问接口。

**核心特性**：
1. **智能调度**：根据数据类型自动选择最优数据源
2. **自动降级**：当主数据源不可用时，自动切换到备用数据源
3. **智能缓存**：使用磁盘缓存，大幅降低API调用成本
4. **统一接口**：屏蔽底层数据源的差异

**核心方法**：
- `get_macro_snapshot()`: 获取宏观经济数据快照
- `get_gdp()`, `get_cpi()`, `get_pce()`: 获取宏观数据
- `get_fed_rate()`, `get_treasury_yield()`, `get_yield_curve()`: 获取利率数据
- `get_stock_history()`, `get_stock_info()`: 获取个股数据
- `get_stock_quote()`: 获取实时报价
- `get_company_news()`, `get_market_news()`: 获取新闻
- `get_economic_calendar()`, `get_earnings_calendar()`: 获取日历

---

## 5. API密钥获取指南

### 5.1 FRED API Key

1. 访问 https://fred.stlouisfed.org/docs/api/api_key.html
2. 点击"Request API Key"
3. 填写注册信息（免费）
4. 获取API Key

### 5.2 Finnhub API Key

1. 访问 https://finnhub.io/
2. 点击"Get free API key"
3. 注册账号（免费）
4. 获取API Key

### 5.3 Tushare Pro Token

1. 访问 https://tushare.pro/
2. 注册账号
3. 获取Token
4. 根据需要的数据接口，确保有足够的积分

**自定义API服务配置**（可选）：

如果您使用第三方Tushare数据服务，可以通过以下方式配置：

```python
from tushare_client import TushareClient

# 方式1：直接传入参数
client = TushareClient(
    token="YOUR_TOKEN",
    http_url="http://your-custom-api-server.com"
)

# 方式2：使用环境变量
# export TUSHARE_TOKEN="YOUR_TOKEN"
# export TUSHARE_HTTP_URL="http://your-custom-api-server.com"
client = TushareClient()
```

这样可以使用第三方提供的高权限Tushare数据服务，获取更多数据接口的访问权限。

---

## 6. 版本演进

| 版本 | 核心特性 | 发布时间 |
|:---|:---|:---|
| **V2.6** | 美股宏观数据层，FRED/Finnhub/yfinance集成 | 2026-02-01 |
| **V2.5.1** | A股宏观数据层，对标美股V2.6数据结构，CNMacroDataManager | 2026-02-01 |
| **V2.5** | A股一手数据驱动分析框架，Tushare/AKShare集成 | 2026-01-31 |
| **V2.4** | LLM增强量化分析框架，多Agent协作 | 2026-01-30 |
| **V2.3** | 工业级基础设施，表达式引擎、特征缓存、增强回测 | 2026-01-29 |
| **V2.2** | 统计严谨性和因果思维，IC分析、Bootstrap测试 | 2026-01 |
| **V2.1** | 投资大师策略融合 | 2025-12 |
| **V2.0** | 基础量化分析框架 | 2025-11 |

---

## 7. 参考资料

### 7.1 数据源文档

- [FRED API文档](https://fred.stlouisfed.org/docs/api/fred/)
- [Finnhub API文档](https://finnhub.io/docs/api)
- [yfinance文档](https://pypi.org/project/yfinance/)
- [Tushare Pro官方文档](https://tushare.pro/document/2)
- [AKShare官方文档](https://akshare.akfamily.xyz/)

### 7.2 技术参考

- [Qlib框架](https://qlib.readthedocs.io/)
- [TradingAgents](https://tradingagents-ai.github.io/)

### 7.3 内部文档

- `references/qlib_integration.md`: Qlib框架集成指南
- `references/master_wisdom.json`: 投资大师智慧库

---

## 8. 联系与支持

如有问题或建议，请通过以下方式联系：

- GitHub: [maxwelllee54/myQuant](https://github.com/maxwelllee54/myQuant)
- Manus帮助中心: [https://help.manus.im](https://help.manus.im)

---

**免责声明**：本技能仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。
