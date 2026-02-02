# MyQuant - AI量化投资工具箱

一个端到端的AI量化投资框架，融合了统计学、贝叶斯推断、现代投资组合理论、投资大师智慧和大型语言模型增强分析。

**当前版本**: V2.7 (持久化数据存储)
**核心理念**: 全球化一手数据驱动 + LLM深度分析 + 工业级框架设计 + 持久化数据存储

---

## 🎯 项目特色

- ✅ **全球化数据**：同时支持A股和美股市场，覆盖宏观、行业、个股全维度数据
- ✅ **一手数据驱动**：直接从FRED、Tushare、Finnhub、yfinance等专业数据源获取原始数据
- ✅ **持久化存储 (V2.7 NEW)**：内置持久化数据存储系统，一次下载，永久使用，避免重复下载，极大提升数据获取效率。
- ✅ **科学严谨**：基于统计显著性检验，避免过拟合和数据挖掘偏差
- ✅ **理论完备**：整合现代投资组合理论(MPT)、贝叶斯推断、因果推断
- ✅ **大师智慧**：融入巴菲特、格雷厄姆、林奇等6位投资大师的可量化策略
- ✅ **AI增强**：支持ChatGPT、Gemini等大模型深度分析，多Agent协作
- ✅ **工业级架构**：借鉴微软Qlib框架设计，表达式引擎、特征缓存、增强回测
- ✅ **可复现**：引入Point-in-Time回测框架，彻底消除Look-Ahead Bias

---

## 🆕 V2.7 新特性：持久化数据存储系统

V2.7版本引入了全新的**持久化数据存储系统**，彻底解决了重复下载数据的痛点，为用户提供更流畅、高效的量化研究体验。

### 核心特性

| 特性 | 描述 |
|:---|:---|
| **持久化存储** | 所有下载的数据都将永久保存在本地`~/.quant_investor/data/`目录中。 |
| **增量更新** | 系统能够智能判断本地数据的覆盖范围，只下载缺失或过期的数据。 |
| **统一管理** | 通过`PersistentDataManager`提供对所有数据的统一访问接口。 |
| **高性能** | 采用**Parquet**列式存储格式，结合**SQLite**进行元数据索引，大幅优化读写性能。 |

---

## 📁 项目结构

```
myquant/
└── skill/                          # quant-investor技能完整代码
    ├── SKILL.md                    # 技能说明文档
    ├── scripts/
    │   ├── v2.7/                   # V2.7 持久化数据存储 (NEW)
    │   │   ├── persistent_data_manager.py
    │   │   ├── persistent_us_data_manager.py
    │   │   └── persistent_cn_data_manager.py
    │   ├── v2.6/                   # V2.6 美股宏观数据层
    │   ├── v2.5/                   # V2.5 A股数据层
    │   ├── v2.4/                   # V2.4 LLM增强分析
    │   └── v2.3/                   # V2.3 工业级基础设施
    └── references/                 # 理论文档和参考资料
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install tushare akshare yfinance pandas numpy pyarrow requests scikit-learn scipy matplotlib seaborn
```

### 2. 配置API密钥

(与之前版本相同)

### 3. 运行示例 (V2.7)

从V2.7开始，您无需再关心底层的数据下载和缓存，只需调用新的`Persistent`数据管理器即可。

#### 美股分析（自动持久化）

```python
import sys
sys.path.append("skill/scripts/v2.7")
from persistent_us_data_manager import PersistentUSDataManager

# 初始化管理器（API Key会自动从环境变量读取）
manager = PersistentUSDataManager()

# 第一次获取AAPL历史行情（会从网络下载并保存）
print("首次获取AAPL数据...")
aapl_first = manager.get_stock_history("AAPL", period="3mo")
print(f"获取到 {len(aapl_first)} 条数据")

# 第二次获取AAPL历史行情（将直接从本地持久化存储中读取）
print("\n再次获取AAPL数据...")
aapl_second = manager.get_stock_history("AAPL", period="3mo")
print(f"获取到 {len(aapl_second)} 条数据")
```

---

## 📖 版本演进

| 版本 | 核心特性 | 发布时间 |
|:---|:---|:---|
| **V2.7** | **持久化数据存储系统** | 2026-02-02 |
| V2.5.2 | A股高级数据接口，Tushare一万分权限 | 2026-02-02 |
| V2.6 | 美股宏观数据层，FRED/Finnhub/yfinance集成 | 2026-02-01 |
| V2.5 | A股一手数据驱动分析框架，Tushare/AKShare集成 | 2026-01-31 |
| V2.4 | LLM增强量化分析框架，多Agent协作 | 2026-01-30 |
| V2.3 | 工业级基础设施，表达式引擎、特征缓存、增强回测 | 2026-01-29 |

---

## ⚠️ 免责声明

本项目仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。

---

**如果这个项目对你有帮助，请给个⭐️Star支持一下！** 🚀
