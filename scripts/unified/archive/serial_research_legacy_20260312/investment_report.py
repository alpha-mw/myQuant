"""
Investment Report Generator — 结构化投资报告
============================================
核心目标：让普通用户也能看懂量化系统的结论，并知道该怎么做。

报告结构：
  Part 1: 执行摘要（30秒阅读版）
  Part 2: 市场环境（宏观背景）
  Part 3: 股票推荐（每只股票：逻辑 + 具体执行步骤 + 监控触发条件）
  Part 4: 组合配置方案（比例 + 分批买入计划）
  Part 5: AI裁决面板（各LLM观点 + 分歧分析）
  Part 6: 回测表现（历史验证）
  Part 7: 风险警示（止损/监控点）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from logger import get_logger

_logger = get_logger("InvestmentReport")

# 各评级的中英对照 + 行动指南
VOTE_GUIDE: dict[str, dict] = {
    "强烈买入": {
        "emoji": "🔥",
        "color": "green",
        "action": "建立核心仓位",
        "position_range": "10%–15%",
        "timing": "当日或次日分批建仓",
    },
    "买入": {
        "emoji": "🟢",
        "color": "lightgreen",
        "action": "逢低加仓或新建仓位",
        "position_range": "5%–10%",
        "timing": "1–3个交易日内建仓",
    },
    "持有": {
        "emoji": "🟡",
        "color": "yellow",
        "action": "维持现有仓位，不追高",
        "position_range": "维持原仓位",
        "timing": "无需操作",
    },
    "卖出": {
        "emoji": "🔴",
        "color": "orange",
        "action": "减仓50%以上",
        "position_range": "控制在2%以内",
        "timing": "1–2个交易日内执行",
    },
    "强烈卖出": {
        "emoji": "⛔",
        "color": "red",
        "action": "清仓离场",
        "position_range": "0%",
        "timing": "当日盘中分批卖出",
    },
}

# 分歧指数的含义说明
DISAGREEMENT_GUIDE = {
    (0.0, 0.2): ("✅ 高度一致", "各AI模型判断高度一致，信号可靠性强"),
    (0.2, 0.4): ("🔶 基本一致", "模型间存在小幅分歧，主方向明确"),
    (0.4, 0.6): ("⚠️ 分歧明显", "模型分歧较大，建议减半仓位，等待更多确认信号"),
    (0.6, 1.0): ("🚨 高度分歧", "模型判断严重对立，建议观望，本信号不宜执行"),
}


@dataclass
class StockRecommendation:
    """单只股票的完整推荐信息"""
    symbol: str
    name: str = ""
    current_price: float = 0.0
    final_vote: str = "持有"
    ensemble_confidence: float = 0.5
    disagreement_index: float = 0.3
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    quant_score: float = 0.0
    ensemble_score: float = 0.0
    combined_score: float = 0.0

    # 因子信号
    factor_signals: dict[str, float] = field(default_factory=dict)
    # 量化模型预测
    model_predictions: dict[str, float] = field(default_factory=dict)
    # LLM 各模型裁决
    llm_verdicts: list[dict] = field(default_factory=list)
    # 多方论点
    bull_points: list[str] = field(default_factory=list)
    # 空方论点
    bear_points: list[str] = field(default_factory=list)
    # 风险点
    risk_points: list[str] = field(default_factory=list)


@dataclass
class ReportInput:
    """报告生成所需的所有输入数据"""
    stocks: list[StockRecommendation] = field(default_factory=list)
    macro_signal: str = "🟡"
    macro_summary: str = ""
    market_outlook: str = ""
    backtest_report: str = ""
    total_capital: float = 1_000_000.0
    risk_level: str = "中等"     # 保守/中等/积极
    analysis_date: str = ""
    # 选填：组合级别数据
    portfolio_metrics: dict[str, Any] = field(default_factory=dict)


class InvestmentReportGenerator:
    """
    将量化系统各层的输出整合为用户可执行的投资报告。
    """

    def __init__(self, report_input: ReportInput) -> None:
        self.inp = report_input
        self.date = report_input.analysis_date or time.strftime("%Y-%m-%d")

    def generate(self) -> str:
        """生成完整 Markdown 报告"""
        sections = [
            self._header(),
            self._executive_summary(),
            self._market_environment(),
            self._stock_recommendations(),
            self._portfolio_allocation(),
            self._ai_panel(),
            self._backtest_section(),
            self._risk_warnings(),
            self._footer(),
        ]
        return "\n\n---\n\n".join(s for s in sections if s.strip())

    # ----------------------------------------------------------------
    # Part 0: 标题
    # ----------------------------------------------------------------

    def _header(self) -> str:
        macro_desc = {
            "🟢": "积极进攻",
            "🟡": "中性精选",
            "🔴": "防御保守",
        }.get(self.inp.macro_signal, "中性")

        return f"""# 量化投资分析报告
**分析日期：** {self.date}　｜　**市场环境：** {self.inp.macro_signal} {macro_desc}　｜　**风险偏好：** {self.inp.risk_level}

> 本报告由 六层量化引擎 + 四大AI模型（Claude / GPT-4o / DeepSeek / Gemini）联合生成。
> 所有投资建议均基于历史数据与模型推断，**不构成投资建议，请结合自身情况谨慎决策。**"""

    # ----------------------------------------------------------------
    # Part 1: 执行摘要
    # ----------------------------------------------------------------

    def _executive_summary(self) -> str:
        buy_stocks   = [s for s in self.inp.stocks if s.final_vote in ("强烈买入", "买入")]
        hold_stocks  = [s for s in self.inp.stocks if s.final_vote == "持有"]
        sell_stocks  = [s for s in self.inp.stocks if s.final_vote in ("卖出", "强烈卖出")]

        lines = ["## 📋 执行摘要（30秒阅读版）", ""]

        if buy_stocks:
            names = "、".join(s.symbol for s in buy_stocks[:5])
            lines.append(f"🟢 **建议买入（{len(buy_stocks)}只）：** {names}")
        if hold_stocks:
            names = "、".join(s.symbol for s in hold_stocks[:5])
            lines.append(f"🟡 **建议持有（{len(hold_stocks)}只）：** {names}")
        if sell_stocks:
            names = "、".join(s.symbol for s in sell_stocks[:5])
            lines.append(f"🔴 **建议卖出（{len(sell_stocks)}只）：** {names}")

        lines += [
            "",
            f"**宏观信号：** {self.inp.macro_signal} {self.inp.macro_summary[:80]}",
            "",
            "**今日核心操作：**",
        ]

        for s in buy_stocks[:3]:
            guide = VOTE_GUIDE.get(s.final_vote, {})
            pct = guide.get("position_range", "5%–10%")
            timing = guide.get("timing", "")
            tp = f"，目标价 ¥{s.target_price:.2f}" if s.target_price else ""
            sl = f"，止损 ¥{s.stop_loss:.2f}" if s.stop_loss else ""
            lines.append(
                f"  - **{s.symbol}** {s.final_vote} — 建议仓位 {pct}，{timing}{tp}{sl}"
            )

        for s in sell_stocks[:3]:
            guide = VOTE_GUIDE.get(s.final_vote, {})
            timing = guide.get("timing", "")
            lines.append(
                f"  - **{s.symbol}** {s.final_vote} — {guide.get('action', '减仓')}，{timing}"
            )

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Part 2: 市场环境
    # ----------------------------------------------------------------

    def _market_environment(self) -> str:
        signal = self.inp.macro_signal
        env_map = {
            "🟢": ("宏观环境积极", "经济数据向好、流动性宽松、风险偏好上升，适合适度进攻。"),
            "🟡": ("宏观环境中性", "市场方向不明朗，建议精选个股、控制总仓位在 50%–70%。"),
            "🔴": ("宏观环境偏空", "经济下行压力大或流动性收紧，优先防御，总仓位控制在 30%以内。"),
        }
        title, desc = env_map.get(signal, ("宏观环境未知", ""))

        lines = [
            "## 🌍 市场环境分析",
            "",
            f"### {signal} {title}",
            desc,
            "",
        ]

        if self.inp.market_outlook:
            lines += ["**详细分析：**", self.inp.market_outlook, ""]

        # 操作建议
        action_map = {
            "🟢": [
                "总仓位可提升至 80%–90%",
                "优先配置高弹性成长股",
                "可适当参与主题炒作",
            ],
            "🟡": [
                "总仓位维持在 50%–70%",
                "精选高质量个股，回避概念炒作",
                "保留足够现金等待机会",
            ],
            "🔴": [
                "总仓位压缩至 30%以内",
                "优先配置防御性行业（消费/医药/公用事业）",
                "严格执行止损纪律",
            ],
        }
        actions = action_map.get(signal, [])
        if actions:
            lines.append("**当前市场操作指引：**")
            for a in actions:
                lines.append(f"  - {a}")

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Part 3: 股票推荐（核心部分）
    # ----------------------------------------------------------------

    def _stock_recommendations(self) -> str:
        lines = ["## 📈 个股推荐详情"]

        # 按评级排序
        order = {"强烈买入": 0, "买入": 1, "持有": 2, "卖出": 3, "强烈卖出": 4}
        sorted_stocks = sorted(self.inp.stocks, key=lambda s: order.get(s.final_vote, 2))

        for s in sorted_stocks:
            lines.append("")
            lines.append(self._single_stock_section(s))

        return "\n".join(lines)

    def _single_stock_section(self, s: StockRecommendation) -> str:
        guide = VOTE_GUIDE.get(s.final_vote, {})
        emoji = guide.get("emoji", "🟡")
        action = guide.get("action", "观察")

        # 分歧指数解释
        di_label, di_desc = "⚠️ 未知", ""
        for (lo, hi), (label, desc) in DISAGREEMENT_GUIDE.items():
            if lo <= s.disagreement_index < hi:
                di_label, di_desc = label, desc
                break

        # 目标价/止损
        tp_str = f"¥{s.target_price:.2f}" if s.target_price else "待定"
        sl_str = f"¥{s.stop_loss:.2f}"    if s.stop_loss    else "待定"
        upside = (
            f"{(s.target_price / s.current_price - 1):.1%}" if s.target_price and s.current_price
            else "—"
        )

        lines = [
            f"### {emoji} {s.symbol} {s.name}  —  {s.final_vote}",
            "",
            "#### 关键数据一览",
            f"| 项目 | 数值 |",
            f"|------|------|",
            f"| 当前价格 | ¥{s.current_price:.2f} |" if s.current_price else "",
            f"| 目标价格 | {tp_str}（预期上涨空间 {upside}）|",
            f"| 止损价格 | {sl_str} |",
            f"| 综合得分 | {s.combined_score:+.3f}（量化 {s.quant_score:+.3f} + AI {s.ensemble_score:+.3f}）|",
            f"| AI置信度 | {s.ensemble_confidence:.1%} |",
            f"| 模型分歧 | {di_label}（{s.disagreement_index:.2f}）|",
            "",
            "#### 核心逻辑",
        ]
        lines = [l for l in lines if l]  # 去空行

        if s.bull_points:
            lines.append("**多方理由：**")
            for p in s.bull_points[:4]:
                lines.append(f"  ✅ {p}")

        if s.bear_points:
            lines.append("**风险因素：**")
            for p in s.bear_points[:3]:
                lines.append(f"  ⚠️ {p}")

        # 因子信号可视化
        if s.factor_signals:
            lines += ["", "#### 量化因子信号"]
            for fname, fval in list(s.factor_signals.items())[:6]:
                bar = self._signal_bar(fval)
                lines.append(f"  - {fname}: {bar} ({fval:+.4f})")

        # 具体执行步骤（最重要的部分）
        lines += ["", "#### 📋 具体执行步骤"]
        lines.extend(self._execution_steps(s))

        # 监控触发条件
        lines += ["", "#### 🔔 监控触发条件（需要重新评估时）"]
        lines.extend(self._monitoring_triggers(s))

        return "\n".join(lines)

    def _execution_steps(self, s: StockRecommendation) -> list[str]:
        """生成具体的操作步骤（量化 + AI 综合）"""
        vote = s.final_vote
        cp   = s.current_price
        sl   = s.stop_loss
        tp   = s.target_price

        if vote in ("强烈买入", "买入"):
            pct = 0.12 if vote == "强烈买入" else 0.07
            capital_each = self.inp.total_capital * pct
            lot = int(capital_each / cp / 100) * 100 if cp > 0 else 0

            steps = [
                f"**Step 1 — 建仓（{time.strftime('%Y-%m-%d')} 或次日）**",
                f"  - 首次买入：计划资金的 **40%**（约 {capital_each*0.4:,.0f} 元，约 {int(lot*0.4)} 股）",
                f"  - 买入价格：以当日**收盘价委托**或**明日开盘后5分钟内**分价买入",
                f"  - 避免追涨：若开盘即大幅高开（>3%），等待回调再入",
                "",
                f"**Step 2 — 分批加仓**",
                f"  - 若股价回落到买入价 **-3%** 时，加仓计划资金的 **30%**",
                f"  - 若持仓盈利 **+5%** 时，加仓计划资金的 **30%**",
                f"  - 总仓位上限：{self.inp.total_capital * pct:,.0f} 元（{pct:.0%} 组合比例）",
            ]
            if sl:
                steps += [
                    "",
                    f"**Step 3 — 止损设置**",
                    f"  - 硬止损：¥{sl:.2f}（跌破即无条件止损）",
                    f"  - 时间止损：买入后 **20个交易日** 内股价未按预期运动，重新评估",
                ]
            if tp:
                steps += [
                    "",
                    f"**Step 4 — 止盈策略**",
                    f"  - 目标价：¥{tp:.2f}，到达后卖出 **50%** 仓位",
                    f"  - 剩余仓位跟踪止盈：每涨 **5%** 上移止损位置",
                    f"  - 强制止盈：盈利超过 **30%** 后全部清仓",
                ]
        elif vote in ("卖出", "强烈卖出"):
            steps = [
                f"**Step 1 — 减仓计划（尽快执行）**",
                f"  - 若 {vote == '强烈卖出' and '当日盘中分批卖出 100%' or '2个交易日内减仓 50%以上'}",
                f"  - 避免单笔大单（可能触发涨跌停），分 3–5 笔卖出",
                "",
                f"**Step 2 — 优先卖出条件**",
                f"  - 开盘后若有反弹，优先利用反弹卖出",
                f"  - 不要等待'解套'再卖，执行纪律优先",
                "",
                f"**Step 3 — 剩余仓位处理**",
                f"  - 剩余仓位设置严格止损：{'¥'+str(round(sl,2)) if sl else '前期低点'}",
                f"  - 若止损触发，无条件清仓",
            ]
        else:  # 持有
            steps = [
                f"**当前建议：持有，无需操作**",
                f"  - 维持现有仓位不变",
                f"  - 检查现有止损位是否合理（建议：最新成本 × 0.92）",
                f"  - 下次重新评估时间：{self._next_review_date()}",
            ]

        return steps

    def _monitoring_triggers(self, s: StockRecommendation) -> list[str]:
        """生成监控触发条件（何时需要重新评估）"""
        triggers = [
            "📊 **量化信号触发（系统自动检测）：**",
            f"  - 因子得分下降超过 **0.3**（当前 {s.quant_score:+.3f}）",
            f"  - AI模型分歧指数上升超过 **0.6**（当前 {s.disagreement_index:.2f}）",
            f"  - 30日波动率超过 **25%**",
            "",
            "📰 **基本面事件触发（需人工关注）：**",
            "  - 公司发布盈利预警或重大负面公告",
            "  - 行业政策重大变化（监管新规、补贴取消等）",
            "  - 主要高管离职或股东大比例减持",
            "",
            "📉 **技术面触发：**",
            f"  - 价格跌破 **20日均线**（确认后信号）",
            f"  - 成交量异常放大（超过60日均量 **3倍以上**）",
        ]
        if s.stop_loss:
            triggers.append(f"  - 价格触及止损位 ¥{s.stop_loss:.2f}（**立即执行止损**）")
        return triggers

    @staticmethod
    def _signal_bar(value: float, width: int = 10) -> str:
        """将信号值转换为可视化进度条"""
        normalized = max(-1.0, min(1.0, value))
        filled = int((normalized + 1) / 2 * width)
        bar = "█" * filled + "░" * (width - filled)
        direction = "▲" if value > 0 else "▼" if value < 0 else "■"
        return f"{direction} [{bar}]"

    @staticmethod
    def _next_review_date() -> str:
        from datetime import datetime, timedelta
        next_date = datetime.now() + timedelta(days=7)
        return next_date.strftime("%Y-%m-%d")

    # ----------------------------------------------------------------
    # Part 4: 组合配置方案
    # ----------------------------------------------------------------

    def _portfolio_allocation(self) -> str:
        buy_stocks = [s for s in self.inp.stocks if s.final_vote in ("强烈买入", "买入")]
        if not buy_stocks:
            return "## 💼 组合配置\n\n当前无买入推荐，建议持币观望。"

        total_capital = self.inp.total_capital
        lines = [
            "## 💼 组合配置方案",
            "",
            f"**总资金：** ¥{total_capital:,.0f}　｜　**参考仓位：** {self.inp.risk_level}",
            "",
            "| 股票代码 | 评级 | 建议仓位% | 建议金额(元) | 目标价 | 止损价 |",
            "|----------|------|-----------|-------------|--------|--------|",
        ]

        weight_map = {"强烈买入": 0.12, "买入": 0.07}
        total_pct = 0.0

        for s in buy_stocks:
            pct = weight_map.get(s.final_vote, 0.05)
            # 高分歧时减半
            if s.disagreement_index > 0.5:
                pct *= 0.5
            pct = round(pct, 2)
            total_pct += pct
            amount = total_capital * pct
            tp_str = f"¥{s.target_price:.2f}" if s.target_price else "—"
            sl_str = f"¥{s.stop_loss:.2f}"    if s.stop_loss    else "—"
            lines.append(
                f"| {s.symbol} | {s.final_vote} | {pct:.0%} | {amount:,.0f} | {tp_str} | {sl_str} |"
            )

        cash_pct = max(0, 1.0 - total_pct)
        lines += [
            f"| **现金** | — | **{cash_pct:.0%}** | **{total_capital*cash_pct:,.0f}** | — | — |",
            "",
            "**分批建仓建议：**",
            "  - 第一周：建立各仓位的 **40%**",
            "  - 第二周：根据走势追加 **30%**（上涨确认后）",
            "  - 第三周：剩余 **30%** 根据市场情况决定是否补齐",
        ]
        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Part 5: AI裁决面板
    # ----------------------------------------------------------------

    def _ai_panel(self) -> str:
        lines = ["## 🤖 AI模型裁决面板", ""]

        for s in self.inp.stocks:
            if not s.llm_verdicts:
                continue
            lines.append(f"### {s.symbol} — AI多模型裁决")
            lines.append("")
            lines.append("| 模型 | 裁决 | 置信度 | 延迟 | 核心观点 |")
            lines.append("|------|------|--------|------|----------|")
            for v in s.llm_verdicts:
                model = v.get("model_name", "—")
                vote  = v.get("vote", "—")
                conf  = v.get("confidence", 0)
                lat   = v.get("latency_ms", 0)
                reason = v.get("reasoning", "")[:50] + "..."
                lines.append(f"| {model} | {vote} | {conf:.0%} | {lat:.0f}ms | {reason} |")

            di = s.disagreement_index
            di_label, di_desc = "—", ""
            for (lo, hi), (label, desc) in DISAGREEMENT_GUIDE.items():
                if lo <= di < hi:
                    di_label, di_desc = label, desc
                    break
            lines += [
                "",
                f"> **分歧分析：** {di_label} — {di_desc}",
                "",
            ]

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Part 6: 回测表现
    # ----------------------------------------------------------------

    def _backtest_section(self) -> str:
        if not self.inp.backtest_report:
            return ""
        return f"## 📊 历史回测表现\n\n{self.inp.backtest_report}"

    # ----------------------------------------------------------------
    # Part 7: 风险警示
    # ----------------------------------------------------------------

    def _risk_warnings(self) -> str:
        high_disagree = [
            s for s in self.inp.stocks
            if s.disagreement_index > 0.5 and s.final_vote in ("强烈买入", "买入")
        ]
        sell_stocks = [s for s in self.inp.stocks if s.final_vote in ("卖出", "强烈卖出")]

        lines = [
            "## ⚠️ 风险警示",
            "",
            "### 系统性风险",
            "  - 本系统基于历史数据，无法预测黑天鹅事件（政策突变、系统性危机等）",
            f"  - 当前宏观信号：{self.inp.macro_signal}，请据此调整总体仓位",
            "  - 模型在极端行情（单日涨跌 >5%）下预测效果会显著下降",
            "",
        ]

        if high_disagree:
            lines.append("### ⚠️ 高分歧警告（以下股票AI模型分歧较大）")
            for s in high_disagree:
                lines.append(
                    f"  - **{s.symbol}**：分歧指数 {s.disagreement_index:.2f}，"
                    f"建议**减半仓位**并等待分歧收敛"
                )
            lines.append("")

        if sell_stocks:
            lines.append("### 🔴 急需处理的卖出信号")
            for s in sell_stocks:
                guide = VOTE_GUIDE.get(s.final_vote, {})
                lines.append(
                    f"  - **{s.symbol}** {s.final_vote}：{guide.get('action','减仓')}，"
                    f"{guide.get('timing','尽快执行')}"
                )
            lines.append("")

        lines += [
            "### 止损纪律提醒",
            "  - **止损是保护本金的最后防线，不要因侥幸心理而跳过**",
            "  - 单只股票最大亏损：建议不超过总资金的 **3%**",
            "  - 单周最大亏损上限：超过总资金 **8%** 时，停止所有新建仓操作",
            "  - 月度最大亏损上限：超过 **15%** 时，启动全面减仓模式",
        ]

        return "\n".join(lines)

    # ----------------------------------------------------------------
    # Part 8: 页脚
    # ----------------------------------------------------------------

    def _footer(self) -> str:
        return (
            f"## 📌 说明\n\n"
            f"- **生成时间：** {self.date}\n"
            f"- **量化层：** 六层因子-模型-宏观-风险架构\n"
            f"- **AI层：** Claude-Sonnet / GPT-4o / DeepSeek-V3 / Gemini-2.0-Flash\n"
            f"- **权重：** 量化 40% + AI集成 60%\n"
            f"- **数据来源：** Tushare（A股）/ yfinance（美股）/ FRED（宏观）\n\n"
            f"*请记住：任何模型都无法保证未来收益，请谨慎管理仓位和风险。*"
        )


# ---------------------------------------------------------------------------
# 便捷函数：从 EnsembleConsensus 对象构建 StockRecommendation
# ---------------------------------------------------------------------------

def from_ensemble_result(
    consensus: Any,         # EnsembleConsensus from multi_llm_ensemble
    quant_context: dict,    # build_quant_context 的返回值
) -> StockRecommendation:
    """从 MultiLLMEnsemble 的结果构建 StockRecommendation"""
    verdicts_raw = []
    for v in getattr(consensus, "verdicts", []):
        verdicts_raw.append({
            "model_name": v.model_name,
            "vote": v.vote.value,
            "confidence": v.confidence,
            "latency_ms": v.latency_ms,
            "reasoning": v.reasoning_chain[:100],
        })

    # 聚合多方/空方观点
    bull: list[str] = []
    bear: list[str] = []
    for v in getattr(consensus, "verdicts", []):
        bull.extend(getattr(v, "key_bull_points", [])[:2])
        bear.extend(getattr(v, "key_bear_points", [])[:2])

    # 从量化上下文取因子信号
    factors = quant_context.get("factor_signals", {})
    model_preds = quant_context.get("ml_model_predictions", {})
    risk = quant_context.get("risk_profile", {})

    # 目标价：取各模型目标价的中位数
    import statistics
    tps = [v.target_price for v in getattr(consensus, "verdicts", []) if v.target_price]
    tp = statistics.median(tps) if tps else None

    sls = [v.stop_loss for v in getattr(consensus, "verdicts", []) if v.stop_loss]
    sl = statistics.median(sls) if sls else None

    return StockRecommendation(
        symbol=consensus.symbol,
        current_price=quant_context.get("current_price") or 0.0,
        final_vote=consensus.final_vote.value,
        ensemble_confidence=consensus.ensemble_confidence,
        disagreement_index=consensus.disagreement_index,
        target_price=round(tp, 2) if tp else None,
        stop_loss=round(sl, 2) if sl else None,
        quant_score=consensus.quant_score,
        ensemble_score=consensus.ensemble_score,
        combined_score=consensus.final_combined_score,
        factor_signals={k: float(v) for k, v in factors.items() if isinstance(v, (int, float))},
        model_predictions={k: float(v) for k, v in model_preds.items() if isinstance(v, (int, float))},
        llm_verdicts=verdicts_raw,
        bull_points=list(dict.fromkeys(bull))[:5],
        bear_points=list(dict.fromkeys(bear))[:4],
        risk_points=[
            f"VaR(95%): {risk.get('var_95', 'N/A')}",
            f"最大回撤: {risk.get('max_drawdown', 'N/A')}",
        ],
    )


# ---------------------------------------------------------------------------
# CLI 测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stocks = [
        StockRecommendation(
            symbol="000001.SZ", name="平安银行",
            current_price=12.80, final_vote="强烈买入",
            ensemble_confidence=0.78, disagreement_index=0.18,
            target_price=15.50, stop_loss=11.80,
            quant_score=0.65, ensemble_score=0.72, combined_score=0.69,
            factor_signals={"momentum_1m": 0.032, "pe_inverse": 0.080,
                             "roe": 0.142, "realized_vol_20d": -0.218},
            bull_points=["估值处于历史低位，PE仅12.5倍",
                         "ROE连续三年改善，盈利能力提升",
                         "量化因子综合得分前10%"],
            bear_points=["利率下行压缩净息差",
                         "房地产敞口仍有不确定性"],
            llm_verdicts=[
                {"model_name": "Claude-Sonnet", "vote": "强烈买入",
                 "confidence": 0.80, "latency_ms": 1200,
                 "reasoning": "估值历史低位+盈利质量改善，风险收益比优秀"},
                {"model_name": "GPT-4o", "vote": "买入",
                 "confidence": 0.72, "latency_ms": 900,
                 "reasoning": "基本面改善，但需关注房地产风险敞口"},
                {"model_name": "DeepSeek-V3", "vote": "强烈买入",
                 "confidence": 0.75, "latency_ms": 800,
                 "reasoning": "技术形态突破，量价配合良好"},
                {"model_name": "Gemini-2.0-Flash", "vote": "买入",
                 "confidence": 0.68, "latency_ms": 1100,
                 "reasoning": "宏观货币政策偏松，银行股受益"},
            ],
        ),
        StockRecommendation(
            symbol="600519.SH", name="贵州茅台",
            current_price=1650.0, final_vote="持有",
            ensemble_confidence=0.60, disagreement_index=0.35,
            target_price=1800.0, stop_loss=1500.0,
            quant_score=0.10, ensemble_score=0.15, combined_score=0.13,
            factor_signals={"momentum_1m": -0.015, "pe_inverse": 0.032, "roe": 0.285},
            bull_points=["品牌护城河无与伦比", "ROE 28.5%，行业顶尖"],
            bear_points=["估值偏贵，PE>30倍", "动量信号转弱"],
        ),
        StockRecommendation(
            symbol="000858.SZ", name="五粮液",
            current_price=145.0, final_vote="卖出",
            ensemble_confidence=0.65, disagreement_index=0.22,
            stop_loss=135.0,
            quant_score=-0.45, ensemble_score=-0.50, combined_score=-0.48,
            bull_points=["品牌价值稳固"],
            bear_points=["技术形态破位", "量化信号全面偏空", "行业需求疲软"],
        ),
    ]

    report_input = ReportInput(
        stocks=stocks,
        macro_signal="🟡",
        macro_summary="CPI温和，M2增速放缓，市场等待政策催化剂",
        market_outlook="A股处于震荡整理阶段，指数缺乏明确方向，建议精选个股。",
        total_capital=1_000_000.0,
        risk_level="中等",
        analysis_date="2026-03-11",
    )

    generator = InvestmentReportGenerator(report_input)
    report = generator.generate()
    print(report)
    # 保存为文件
    with open("/tmp/investment_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n报告已保存到 /tmp/investment_report.md")
