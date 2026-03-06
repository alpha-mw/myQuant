# Quant-Investor

一套完整的量化投资研究框架，从数据到决策，涵盖六个核心层面。

## 它能做什么

这套系统适合系统性地研究股票市场：

- **量化研究员** —— 快速验证因子有效性，自动完成IC测试、分层回测
- **基金经理** —— 实时监控组合风险，获取仓位调整建议  
- **个人投资者** —— 基于AI分析获取投资决策参考

## 架构 overview

系统按数据流动顺序分为六层：

### 1️⃣ 数据层

处理原始数据的获取和清洗。

接入 Tushare（A股）、yfinance（美股）、FRED（宏观），覆盖中、港、美、欧、日五个市场。数据清洗包括去极值、缺失值填充、标准化，确保下游计算质量。

### 2️⃣ 因子层

计算并验证各类因子的有效性。

内置 Alpha158 因子库，130+因子涵盖动量、波动率、成交量等维度。每个因子经过 IC 分析、分层回测、换手率分析，输出综合评分帮助筛选。

### 3️⃣ 模型层

用机器学习预测股票收益。

支持 Random Forest、XGBoost、SVM、LSTM。时序交叉验证避免数据泄露，Optuna 自动调参，SHAP 解释特征重要性，MLflow 管理模型版本。

### 4️⃣ 宏观层  

判断市场环境，作为风控第一道防线。

A股监控两融余额、GDP、巴菲特指标、M1-M2剪刀差；美股监控联邦利率、失业率、Shiller PE、VIX。输出 🔴🟡🟢🔵 四档信号指引仓位。

### 5️⃣ 风控层

组合层面的风险控制。

计算 VaR/CVaR、夏普比率、最大回撤。压力测试覆盖 2008、2015、2020 等极端行情。根据宏观信号动态调仓，红灯 30% 仓位，绿灯 80%。

### 6️⃣ 决策层

AI 多模型辩论，输出投资建议。

五个专业模型各抒己见：财务看 ROE 估值，行业看竞争格局，宏观看经济周期，技术看价格趋势，风险评估潜在损失。综合形成决策：买不买、买多少、目标价、止损位。

## 核心能力

**工程化实践**

59 个单元测试覆盖核心逻辑，pytest + GitHub Actions 持续集成，代码遵循 PEP 8，用 black 格式化、flake8 检查、mypy 类型注解。

**完整回测**

Backtrader 事件驱动回测，包含滑点、手续费、印花税、市场冲击成本。支持因子中性化（市值/行业），消除风格暴露。

**可扩展架构**

依赖注入解耦模块，Redis 缓存热点数据，异步处理提升 LLM 吞吐。支持微服务拆分，可水平扩展到多节点。

**合规与审计**

内置风险管理政策，支持 Form PF（美国）、AIFMD（欧盟）、CSRC（中国）三种合规报告格式。完整审计日志追踪所有操作。

## 快速开始

安装：

```bash
git clone https://github.com/alpha-mw/myQuant.git
cd myQuant
pip install -r requirements.txt
```

配置：

```bash
cp .env.example .env
# 填入 Tushare token 和 OpenAI API key
```

运行全市场分析：

```python
from scripts.unified.v72_full_market_analysis import run_full_analysis

result = run_full_analysis()
print(result['top_15_stocks'])
```

启动监控面板：

```bash
streamlit run scripts/unified/risk_dashboard.py
```

## 项目结构

```
myQuant/
├── scripts/unified/          # 核心代码
│   ├── v72_full_market_analysis.py   # 全市场分析入口
│   ├── quant_investor_v7.py          # 六层流水线
│   ├── enhanced_data_layer.py        # 数据层
│   ├── factor_analyzer.py            # 因子层  
│   ├── enhanced_model_layer.py       # 模型层
│   ├── macro_terminal_tushare.py     # 宏观层
│   ├── risk_management_layer.py      # 风控层
│   ├── decision_layer.py             # 决策层
│   ├── alpha158.py                   # 因子库
│   ├── backtest_engine.py            # 回测引擎
│   ├── event_bus.py                  # 事件总线
│   ├── microservices.py              # 微服务框架
│   └── ...
├── tests/unit/               # 59 个单元测试
├── .github/workflows/        # CI/CD 配置
└── README.md                 # 本文件
```

## 演进历程

从 V2.7 开始迭代至今：

- **V7.0** —— 六层架构成熟，工程化改造完成，支持全市场 1900 只股票分析
- **V6.x** —— 引入宏观风控和多模型辩论系统
- **V5.x** —— 建立工业级量化框架基础
- **V2-4** —— 基础数据获取和简单策略回测

早期版本缺乏测试、配置硬编码，现在测试覆盖率 60%+，配置集中管理，错误处理完善，整体可用性和可维护性显著提升。

## 注意事项

系统生成的分析报告仅供参考，不构成投资建议。模型基于历史数据训练，无法预测未来。使用者需理解量化投资风险，独立做出判断。

全市场分析需要 30-60 分钟计算时间，建议先从小股票池测试。

## License

MIT
