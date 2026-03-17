"""
风险监控仪表板 V8（Risk Dashboard V8）— 改进七
================================================

从回测结果生成交互式 HTML 报告，包含：
  1. 组合净值曲线 vs 基准（支持对数坐标）
  2. 最大回撤水下图
  3. 滚动 Sharpe/IR（30/60/90 日窗口）
  4. 月度收益热图
  5. 五路分支信号历史时序
  6. 月度超额收益分布

依赖：plotly（requirements.txt 中已有）
用法：
    from risk_dashboard_v8 import RiskDashboard, DashboardData
    data = RiskDashboard.from_backtest_result(backtest_result)
    RiskDashboard().generate(data, "results/dashboard.html")
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class DashboardData:
    """仪表板所需数据"""
    portfolio_nav: pd.Series
    benchmark_nav: pd.Series
    monthly_returns: pd.Series
    excess_return: float
    sharpe: float
    max_drawdown: float
    branch_scores: dict[str, list] = None  # type: ignore
    title: str = "Quant-Investor V8.0 风险监控仪表板"

    def __post_init__(self) -> None:
        if self.branch_scores is None:
            self.branch_scores = {}


class RiskDashboard:
    """交互式风险监控仪表板（Plotly HTML 输出）"""

    ROLLING_WINDOWS = [30, 60, 90]
    BRANCH_COLORS = {
        "kronos": "#2196F3",
        "quant": "#4CAF50",
        "llm_debate": "#FF9800",
        "intelligence": "#9C27B0",
        "macro": "#F44336",
    }

    def generate(
        self,
        data: DashboardData,
        output_path: str = "results/dashboard.html",
        log_scale: bool = False,
    ) -> str:
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly 未安装，运行：pip install plotly")

        port = data.portfolio_nav.dropna()
        bm = data.benchmark_nav.reindex(port.index).fillna(method="ffill").fillna(1.0)
        port_ret = port.pct_change().fillna(0)
        bm_ret = bm.pct_change().fillna(0)
        rf_daily = 0.025 / 252

        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                "组合净值 vs 基准",
                "最大回撤水下图 (%)",
                "滚动夏普比率",
                "月度收益热图",
                "五路分支信号历史",
                "月度超额收益分布",
            ],
            row_heights=[0.35, 0.35, 0.30],
            vertical_spacing=0.10,
            horizontal_spacing=0.08,
        )

        # 净值曲线
        fig.add_trace(go.Scatter(x=port.index, y=port.values,
            name="策略", line=dict(color="#1565C0", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=bm.index, y=bm.values,
            name="基准", line=dict(color="#B71C1C", width=1.5, dash="dash")), row=1, col=1)
        if log_scale:
            fig.update_yaxes(type="log", row=1, col=1)

        # 回撤水下图
        cum = (1 + port_ret).cumprod()
        dd = (cum / cum.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values,
            fill="tozeroy", name="回撤(%)",
            line=dict(color="#C62828"),
            fillcolor="rgba(198,40,40,0.3)"), row=1, col=2)

        # 滚动夏普
        colors3 = ["#1976D2", "#388E3C", "#F57C00"]
        for w, c in zip(self.ROLLING_WINDOWS, colors3):
            rs = (port_ret.rolling(w).mean() - rf_daily) / (port_ret.rolling(w).std() + 1e-8) * np.sqrt(252)
            fig.add_trace(go.Scatter(x=rs.index, y=rs.values,
                name=f"Sharpe-{w}d", line=dict(color=c, width=1.2)), row=2, col=1)
        fig.add_hline(y=0, row=2, col=1, line_color="gray", line_dash="dot")
        fig.add_hline(y=1, row=2, col=1, line_color="green", line_dash="dot")

        # 月度热图
        try:
            monthly = data.monthly_returns.dropna()
            if not monthly.empty:
                monthly.index = pd.DatetimeIndex(monthly.index)
                df_h = monthly.to_frame("r")
                df_h["yr"] = df_h.index.year
                df_h["mo"] = df_h.index.month
                pv = df_h.pivot(index="yr", columns="mo", values="r")
                mlabs = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
                fig.add_trace(go.Heatmap(
                    z=pv.values * 100,
                    x=[mlabs[m-1] for m in pv.columns],
                    y=pv.index.astype(str),
                    colorscale="RdYlGn", zmid=0,
                    text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in r] for r in pv.values * 100],
                    texttemplate="%{text}", showscale=True, name="月度收益",
                ), row=2, col=2)
        except Exception:
            pass

        # 分支信号历史
        for branch, scores in data.branch_scores.items():
            if scores:
                fig.add_trace(go.Scatter(
                    x=list(range(len(scores))), y=scores,
                    name=branch,
                    line=dict(color=self.BRANCH_COLORS.get(branch, "#607D8B"), width=1.5),
                ), row=3, col=1)
        fig.add_hline(y=0, row=3, col=1, line_color="gray", line_dash="dot")

        # 月度超额分布
        try:
            bm_monthly = bm_ret.resample("ME").apply(lambda x: (1+x).prod()-1)
            port_monthly = port_ret.resample("ME").apply(lambda x: (1+x).prod()-1)
            exc_m = (port_monthly - bm_monthly.reindex(port_monthly.index).fillna(0)) * 100
            fig.add_trace(go.Histogram(x=exc_m.values, name="月超额(%)",
                marker_color="#1565C0", nbinsx=20, opacity=0.7), row=3, col=2)
            fig.add_vline(x=0, row=3, col=2, line_color="red", line_dash="dash")
        except Exception:
            pass

        summary = (
            f"年化超额: {data.excess_return:.1%} | "
            f"夏普: {data.sharpe:.2f} | "
            f"最大回撤: {data.max_drawdown:.1%} | "
            f"最终净值: {float(port.iloc[-1]):.3f}"
        )
        fig.update_layout(
            title=dict(text=data.title + f"<br><sub>{summary}</sub>",
                       font=dict(size=14), x=0.5),
            height=1100,
            showlegend=True,
            legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE")
        fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE")

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.write_html(output_path, include_plotlyjs="cdn")
        print(f"仪表板已生成: {output_path}")
        return output_path

    @classmethod
    def from_backtest_result(cls, result) -> DashboardData:
        m = result.combined_metrics
        return DashboardData(
            portfolio_nav=result.full_portfolio_nav,
            benchmark_nav=result.full_benchmark_nav,
            monthly_returns=m.monthly_returns if m else pd.Series(),
            excess_return=m.excess_return if m else 0.0,
            sharpe=m.sharpe_ratio if m else 0.0,
            max_drawdown=m.max_drawdown if m else 0.0,
        )
