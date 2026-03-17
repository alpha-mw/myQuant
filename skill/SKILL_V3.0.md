---
name: quant-investor
description: "一个端到端的AI量化投资技能，用于股票分析、模型训练和回测。V3.0版本引入Kronos金融基础模型集成（Layer 8）、财务分析引擎、新闻情感分析、市场情绪分析，构建九层全面智能投资框架。"
---

# 量化投资技能 (Quant-Investor) - V3.0

**版本**: 3.0
**作者**: Manus AI
**核心理念**: Kronos基础模型预测 + 四维智能分析（财务/新闻/情绪/量化） + 九层架构，构建业界最全面的AI量化投资平台。

---

## 1. 技能简介

`quant-investor` 技能 V3.0 是 V2.5 基础上的重大架构升级，引入了以下核心能力：

### V3.0 核心创新

#### 1.1 Kronos 金融基础模型集成（Layer 8）

[Kronos](https://github.com/shiyu-coder/Kronos) 是首个专为金融K线序列预训练的开源基础模型：

| 特性 | 描述 |
|:---|:---|
| **预训练规模** | 45个全球交易所，120亿条K线记录 |
| **预测性能** | 零样本RankIC比最佳TSFM提升93% |
| **波动率预测** | MAE降低9% |
| **架构** | 两阶段：OHLCV离散分词 + 自回归Transformer |
| **可用模型** | Kronos-mini(4.1M) / Kronos-small(24.7M) / Kronos-base(102.3M) |

**Kronos集成功能**：
- 单标的K线序列预测（未来N个交易日）
- 组合批量预测（predict_batch并行化）
- 波动率预测（年化）
- RankIC组合内排名信号
- 当原生库不可用时自动降级为统计替代预测（EWMA波动率 + 动量/均值回归）

#### 1.2 财务分析引擎（FinancialAnalysisEngine）

| 模型 | 功能 | 参考标准 |
|:---|:---|:---|
| **Piotroski F-Score** | 财务健康度0-9分（盈利/偿债/效率） | Joseph Piotroski (2000) |
| **Beneish M-Score** | 盈利操纵检测（8维度） | Messod D. Beneish (1999) |
| **Altman Z-Score** | 破产风险评估（安全/灰色/危险区） | Edward Altman (1968) |
| **DCF估值** | 两阶段自由现金流折现内在价值 | 内在价值投资 |
| **DuPont分析** | ROE三因素拆解（利润率×周转率×杠杆） | DuPont Corp. |
| **综合评级** | A+/A/B/C/D 五级财务健康评级 | 综合模型 |

#### 1.3 新闻与替代数据分析（NewsAnalysisEngine）

- **多源新闻获取**: AKShare个股新闻 + RSS订阅 + 新浪财经API
- **LLM情感提取**: Claude/GPT-4o驱动的8维情感标注
- **事件检测**: 9类事件自动识别（盈利超预期/监管/并购/高管变动等）
- **主题建模**: 市场热点主题自动聚类
- **新闻冲击评分**: 量化预期对股价的影响幅度
- **分析师动态追踪**: 升降级次数统计

#### 1.4 市场情绪分析（MarketSentimentEngine）

| 指标 | 描述 |
|:---|:---|
| **恐慌贪婪指数** | 7维度综合市场情绪（0=极度恐慌，100=极度贪婪） |
| **技术情绪** | RSI/MACD/布林带/量比/KDJ综合技术情绪 |
| **资金流向** | 主力资金/北向资金/融资融券净流入 |
| **市场广度** | 涨跌家数比/创新高新低/均线覆盖率 |
| **行业轮动** | 热门行业追踪/轮动阶段判断（早/中/晚周期/防御） |
| **逆向信号** | 极端情绪反转提醒（>80极度贪婪警告，<20极度恐慌机遇） |

#### 1.5 九层架构总览

```
Layer 1: 数据层        → 多源数据采集（OHLCV/基本面/宏观/资金流）
Layer 2: 因子层        → 50+因子计算与IC分析
Layer 3: 模型层        → ML模型集成（RF/XGBoost/SVM/LSTM）
Layer 4: 宏观层        → 市场趋势判断（CN 4模块/US 5模块）
Layer 5: 风险层        → 组合风险管理（VaR/CVaR/压力测试）
Layer 6: 决策层        → 多模型辩论（5个专家角色）
Layer 7: 集成层        → Multi-LLM集成裁判（Claude+GPT-4o+DeepSeek+Gemini）
Layer 8: Kronos层      → 金融基础模型K线预测（零样本RankIC+93%）  ← NEW
Layer 9: 智能融合层    → 财务+新闻+情绪+LLM综合分析                ← NEW
```

---

## 2. 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│              Layer 9: 多维智能融合层（NEW in V3.0）                  │
│  财务分析（Piotroski/Beneish/Altman/DCF/DuPont）                     │
│  新闻情感（AKShare/RSS/LLM情感/事件检测）                            │
│  市场情绪（恐慌贪婪/技术情绪/资金流/市场广度）                        │
│  LLM综合分析（Claude/GPT-4o 综合各层信号）                           │
└─────────────────────────────────────────────────────────────────────┘
                              ↑ 接入
┌─────────────────────────────────────────────────────────────────────┐
│              Layer 8: Kronos 金融基础模型层（NEW in V3.0）            │
│  KronosPredictor（kronos-mini/small/base）                          │
│  批量预测 → RankIC信号 → 组合多空排名                                 │
│  降级模式：统计预测（EWMA波动率 + 动量/均值回归）                      │
└─────────────────────────────────────────────────────────────────────┘
                              ↑ 接入
┌─────────────────────────────────────────────────────────────────────┐
│              Layer 7: Multi-LLM 集成裁判（V8 继承）                   │
│  Claude + GPT-4o + DeepSeek + Gemini 四模型投票                      │
│  置信度加权融合 + 分歧度风险信号                                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↑
┌───────────────────────────────────────────────┐
│         Layers 1-6: 量化基础层（V8继承）        │
│  数据 → 因子 → 模型 → 宏观 → 风险 → 决策       │
└───────────────────────────────────────────────┘
```

---

## 3. 快速开始

### 3.1 环境配置

**必需**：
- Python 3.10+
- `pip install pandas numpy scipy scikit-learn xgboost loguru`

**推荐**（获取真实数据）：
- Tushare Pro Token（A股数据）
- AKShare（免费，`pip install akshare`）

**可选**（增强功能）：
- Kronos原生库：`git clone https://github.com/shiyu-coder/Kronos && pip install -e .`
- Anthropic API Key（`ANTHROPIC_API_KEY`）：LLM情感分析和综合报告
- OpenAI API Key（`OPENAI_API_KEY`）：备选LLM

### 3.2 运行V9全面分析

```bash
# 分析A股标的（完整九层）
python scripts/unified/quant_investor_v9.py \
    --stocks 000001.SZ 600519.SH 000858.SZ \
    --market CN \
    --capital 1000000 \
    --kronos-model kronos-small \
    --pred-len 20 \
    --output report_v9.md

# 快速分析（禁用宏观和新闻，主要测试Kronos+财务+情绪）
python scripts/unified/quant_investor_v9.py \
    --stocks 600519.SH \
    --no-macro \
    --no-news \
    --output quick_report.md
```

### 3.3 Python API

```python
from scripts.unified.quant_investor_v9 import QuantInvestorV9

# 完整九层分析
analyzer = QuantInvestorV9(
    stock_pool=["600519.SH", "000001.SZ"],
    market="CN",
    total_capital=1_000_000,
    kronos_model="kronos-small",   # 推荐：性能/速度平衡
    pred_len=20,                   # 预测未来20个交易日
    enable_financial_analysis=True,
    enable_news_analysis=True,
    enable_sentiment_analysis=True,
)
result = analyzer.run()
print(result.final_report)
```

### 3.4 单模块使用

```python
# 仅使用 Kronos 预测
from scripts.unified.kronos_predictor import KronosIntegrator
integrator = KronosIntegrator(model_name="kronos-small")
signal = integrator.analyze_portfolio(
    stock_data_dict={"600519.SH": df_maotai},
    pred_len=20,
)
print(signal.summary)

# 仅使用财务分析
from scripts.unified.financial_analysis import FinancialAnalyzer
analyzer = FinancialAnalyzer()
report = analyzer.full_analysis(
    symbol="600519.SH",
    stock_name="贵州茅台",
    financial_df=df_financial,
)
print(report.summary)

# 仅使用市场情绪分析
from scripts.unified.sentiment_analysis import MarketSentimentAnalyzer
analyzer = MarketSentimentAnalyzer()
report = analyzer.analyze(
    symbol="600519.SH",
    stock_name="贵州茅台",
    price_df=df_price,
)
print(report.summary)

# 仅使用新闻分析
from scripts.unified.news_analysis import NewsAnalyzer
analyzer = NewsAnalyzer()
result = analyzer.analyze(
    symbol="600519.SH",
    stock_name="贵州茅台",
    days=7,
)
print(result.summary)
```

---

## 4. 核心模块详解

### 4.1 Kronos集成层 (`kronos_predictor.py`)

**KronosIntegrator** 类核心方法：

| 方法 | 功能 |
|:---|:---|
| `predict_single(symbol, df, pred_len)` | 单标的预测，返回KronosForecast |
| `analyze_portfolio(stock_data_dict, pred_len)` | 组合批量预测，返回KronosPortfolioSignal |
| `prepare_ohlcv(df)` | 将任意格式数据转换为Kronos标准输入 |

**KronosForecast 字段**：
- `pred_close_pct`: 预测收盘价涨跌幅(%)
- `volatility_forecast`: 年化波动率预测(%)
- `direction_signal`: "看多" | "看空" | "中性"
- `confidence`: 置信度 0~1
- `rank_ic_score`: 组合内RankIC排名

**降级策略**：
1. 尝试加载 Kronos 原生库（`from model import Kronos, KronosTokenizer, KronosPredictor`）
2. 原生库不可用时，使用 EWMA波动率 + 动量/均值回归统计预测
3. 两种模式的输出接口完全一致

### 4.2 财务分析引擎 (`financial_analysis.py`)

**FinancialAnalyzer** 类核心方法：

| 方法 | 模型 | 评分标准 |
|:---|:---|:---|
| `calc_piotroski(symbol, df)` | Piotroski F-Score | 7-9强/4-6中/0-3弱 |
| `calc_beneish(symbol, df)` | Beneish M-Score | >-1.78高风险/中/-2.22低风险 |
| `calc_altman_z(symbol, df)` | Altman Z-Score | >2.99安全/>1.81灰色/<1.81危险 |
| `calc_dcf(symbol, fcf, price)` | DCF估值 | 上行空间>50%=严重低估 |
| `calc_dupont(symbol, df)` | DuPont ROE拆解 | 质量评分0~1 |
| `full_analysis(...)` | 五模型综合 | A+/A/B/C/D五级 |

### 4.3 新闻分析引擎 (`news_analysis.py`)

**NewsAnalyzer** 类三层架构：

```
NewsDataFetcher → SentimentEngine → NewsAnalyzer
   (多源获取)      (情感分析)        (结果聚合)
```

**数据源优先级**：
1. AKShare个股新闻（东方财富/同花顺）
2. RSS订阅（雪球/Yahoo Finance）
3. 新浪财经API
4. 无法获取时生成提示占位符

**情感分析双轨制**：
- 规则分析（30+正面词/30+负面词词典 + 9类事件模式）
- LLM深度分析（claude-haiku/gpt-4o-mini，重要新闻优先）

### 4.4 市场情绪分析 (`sentiment_analysis.py`)

**MarketSentimentAnalyzer** 核心组件：

| 组件 | 功能 |
|:---|:---|
| `TechnicalSentimentCalculator` | RSI/MACD/布林带/量比/KDJ技术情绪 |
| `CapitalFlowAnalyzer` | AKShare资金流向数据 |
| `MarketBreadthAnalyzer` | 涨跌家数/创新高新低/均线覆盖率 |
| `FearGreedCalculator` | 7维度加权恐慌贪婪指数 |
| `SectorRotationAnalyzer` | 行业轮动阶段判断 |

**恐慌贪婪指数权重**：
- 价格动量 25% + RSI强弱 15% + 量价关系 15%
- 资金流向 20% + 市场广度 10% + 波动率恐慌 10% + 布林带位置 5%

### 4.5 智能融合层 (`intelligence_layer.py`)

**IntelligenceLayerEngine** 四维信号融合：

```
Kronos预测得分  × 25%
财务分析得分    × 25%   →  加权综合得分  →  买入/持有/卖出
新闻情感得分    × 20%      (-1 ~ +1)
市场情绪得分    × 20%
V7量化基础得分  × 10%
```

**置信度计算**：基于四维信号方向一致性（同向越多，置信度越高）

**LLM综合分析**：将四维量化信号输入Claude/GPT-4o，生成200字专业投资摘要

---

## 5. 信号体系

### 5.1 综合信号分级

| 综合得分 | 信号 | 建议仓位（中等风险） |
|:---|:---|:---|
| > 0.35 | 🟢 强烈买入 | 80-100% |
| 0.15 ~ 0.35 | 🟩 买入 | 60-80% |
| -0.15 ~ 0.15 | 🟡 持有 | 40-60% |
| -0.35 ~ -0.15 | 🟧 卖出 | 20-40% |
| < -0.35 | 🔴 强烈卖出 | 0-20% |

### 5.2 Kronos信号与其他信号协同

| 场景 | Kronos | 财务 | 新闻 | 情绪 | 结论 |
|:---|:---|:---|:---|:---|:---|
| 四维共振看多 | 看多 | A+ | 看多 | 贪婪区 | 强烈买入（高置信度）|
| Kronos看多但财务差 | 看多 | D | 中性 | 中性 | 持有（低置信度）|
| 情绪极度恐慌 | 中性 | B | 利空 | 极度恐慌 | 逆向布局机会提示 |
| 高涨跌幅预测+盈利操纵 | 强烈看多 | M高风险 | 中性 | 贪婪 | 谨慎（调整为持有）|

---

## 6. 版本演进

| 版本 | 核心特性 | 发布时间 |
|:---|:---|:---|
| **V3.0** | Kronos金融基础模型(Layer 8) + 财务/新闻/情绪分析(Layer 9) | 2026-03-12 |
| V2.5 | 一手数据驱动（Tushare/AKShare） | 2026-01-31 |
| V2.4 | LLM增强量化分析，多Agent协作 | 2026-01-30 |
| V2.3 | 工业级基础设施（表达式引擎/增强回测） | 2026-01-29 |
| V2.2 | 统计严谨性（IC/Bootstrap/贝叶斯） | 2026-01 |

---

## 7. Kronos 安装指南

### 完整安装（使用原生Kronos预测能力）

```bash
# 1. 克隆Kronos仓库
git clone https://github.com/shiyu-coder/Kronos
cd Kronos

# 2. 安装依赖
pip install -r requirements.txt

# 3. 安装为Python包
pip install -e .

# 4. 验证安装
python -c "from model import Kronos, KronosTokenizer, KronosPredictor; print('Kronos OK')"
```

### 降级模式（无需Kronos，使用统计预测）

不安装Kronos时，系统自动使用统计替代模式：
- EWMA波动率建模（λ=0.94）
- 历史动量 + 均值回归混合预测
- Monte Carlo路径模拟（sample_count次）
- 输出接口与Kronos原生模式完全兼容

---

## 8. 性能基准

| 分析层 | 典型耗时（5只股票）|
|:---|:---|
| Layers 1-6 (量化基础) | 10-30s |
| Layer 7 (Multi-LLM裁判) | 20-60s（取决于API延迟）|
| Layer 8 (Kronos预测) | 2-10s（GPU）/ 10-60s（CPU）|
| Layer 9 (财务+新闻+情绪) | 15-45s |
| **总计** | **50-150s** |

*耗时受API访问速度影响，可通过 `--no-macro` 和禁用部分模块加速*

---

## 9. 参考资料

### Kronos
- [Kronos GitHub](https://github.com/shiyu-coder/Kronos)
- [Kronos论文 (AAAI 2026)](https://arxiv.org/abs/2508.02739)
- [Kronos HuggingFace Models](https://huggingface.co/NeoQuasar)
- [Kronos Live Demo](https://shiyu-coder.github.io/Kronos-demo/)

### 财务分析模型
- Piotroski, J.D. (2000). "Value Investing: The Use of Historical Financial Statement Information"
- Beneish, M.D. (1999). "The Detection of Earnings Manipulation"
- Altman, E.I. (1968). "Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy"

### 市场情绪
- CNN Fear & Greed Index方法论
- AAII Investor Sentiment Survey

### 数据源
- [Tushare Pro](https://tushare.pro/document/2)
- [AKShare](https://akshare.akfamily.xyz/)

---

**免责声明**：本技能仅供学习和研究使用，不构成任何投资建议。量化模型的历史表现不代表未来收益，投资有风险，入市需谨慎。
