# 沪深300双因子量化平台部署手册

## 1. 环境准备
- Python 3.9+（Windows/Linux）
- 建议使用虚拟环境：`python -m venv .venv`

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 数据配置
1. 申请 Tushare Pro Token 并配置：
   - 创建 `config.yaml`，填入 `tushare_token`。
2. 若 Tushare 不可用，可准备本地数据：
   - `data/hs300_members.csv`
   - `data/<ts_code>_daily.csv` 和 `data/<ts_code>_basic.csv`
3. 备用数据源：iFinD（需补充账号配置与 SDK）。

示例 `config.yaml`：
```yaml
data:
  tushare_token: "YOUR_TOKEN"
  start_date: "20210101"
  end_date: "20260101"
```

## 3. 运行全流程
```bash
python scripts/run_pipeline.py --config config.yaml
```

输出：
- `outputs/backtest_report.pdf`
- `outputs/沪深300双因子策略_最优版本_YYYYMMDD.zip`

## 4. 实盘接入说明
- 模拟交易：可接入同花顺模拟交易或券商模拟接口。
- 实盘接口预留：请在 `quant_platform/live_trading.py` 补充券商 API 适配器。

## 5. 风控规则
- 单票最大仓位 10%
- 日内止损 2% 自动平仓

## 6. 常见问题
- 若接口权限不足，请准备本地 CSV 或使用 iFinD 备用数据源。
- 回测报表中的图表可用于策略评估与实盘监控参考。
