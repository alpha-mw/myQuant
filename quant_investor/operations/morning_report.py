"""Pure Morning report authority and native quote-timing declarations."""

from .daily_contract import ContractError


def validate_morning_report(raw: bytes, prepared: dict) -> None:
    if "threshold_review" in prepared:
        if raw != render_morning_report(prepared):
            raise ContractError("MORNING_REPORT_NATIVE_CONTENT_DIFFERS")
        return
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise ContractError("MORNING_REPORT_ENCODING_INVALID") from exc
    required = {
        "research_only": "true",
        "broker": "false",
        "live_order": "false",
        "actual_holdings_mutation": "false",
        **{key: str(value) for key, value in prepared["quote_timing"].items()},
    }
    for key, value in required.items():
        declarations = [line.strip() for line in lines if line.strip().startswith(key + "=")]
        if declarations != [f"{key}={value}"]:
            raise ContractError("MORNING_REPORT_DECLARATION_INVALID:" + key)


def render_morning_report(prepared: dict) -> bytes:
    """Deterministic v3 content; no wall clock or file writer."""
    from collections import Counter
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.intelligence.morning_threshold_review import validate_threshold_review
    from quant_investor.intelligence.investment_decision import DECISION_STATES

    review = validate_threshold_review(prepared["threshold_review"])
    for field in (
        "run_date",
        "previous_trade_date",
        "previous_completion_ref",
        "quote_capture_ref",
        "quote_raw_ref",
        "threshold_policy_refs",
    ):
        if prepared[field] != review[field]:
            raise ContractError("MORNING_REPORT_REVIEW_BINDING_DIFFERS")
    decisions = prepared["decision"].get("decisions")
    if type(decisions) is not list or any(
        type(r) is not dict or r.get("state") not in DECISION_STATES for r in decisions
    ):
        raise ContractError("MORNING_REPORT_DECISION_ROWS_INVALID")
    if len({row["company_code"] for row in decisions}) != len(decisions):
        raise ContractError("MORNING_REPORT_DECISION_SYMBOL_DUPLICATED")
    counts = Counter(row["state"] for row in decisions)
    declarations = {
        "research_only": "true",
        "broker": "false",
        "live_order": "false",
        "actual_holdings_mutation": "false",
        **prepared["quote_timing"],
    }
    lines = [
        "# Morning 晨间研究复核" + ("（合成测试）" if review["synthetic"] else ""),
        "",
        *[str(k) + "=" + str(v) for k, v in declarations.items()],
        "",
        "复核日期：" + review["run_date"] + "；上一交易日：" + review["previous_trade_date"],
        "复核状态：" + review["summary_state"] + "；证据模式：" + review["evidence_mode"],
        "报价采集时点：" + review["quote_requested_at"],
        "报告用途："
        + ("历史研究回放" if prepared.get("admission") == "RESEARCH_ONLY" else "晨间研究复核"),
        "合成证据："
        + str(review["synthetic"]).lower()
        + "；日终核验时点："
        + review["eod_validated_at"],
        "风险阈值依据上一日终严格收盘数据。盘中触及只提示，不确认收盘违约，不更新移动峰值，不产生交易指令。",
        "",
        "上一日终研究结论："
        + "；".join(state + " " + str(counts[state]) for state in sorted(DECISION_STATES)),
        "",
        (
            "| 持仓 | 移动窗口末日收盘 | 移动复核线 | 移动减仓复核线 | owner 初始止损（政策配置） | 当日报价 | "
            "盘中观察（移动／初始止损） | 上日初始止损／owner 复核 |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def cell(value):
        return (
            "未确认"
            if value is None
            else str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        )

    for row in review["rows"]:
        risk, quote = row["eod_risk"], row["quote_observation"]
        values = [
            row["symbol"] + " " + row["name"],
            risk["strict_close"],
            risk["moving_take_profit_review_price"],
            risk["moving_take_profit_reduce_price"],
            row["policy_binding"]["initial_stop"]["configured_price"],
            quote["price"],
            quote["trailing_review_comparison"] + " / " + quote["initial_stop_comparison"],
            risk["owner_stop_trigger"] + " / " + risk["owner_review_state"],
        ]
        lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    lines.extend(["", "## 未确认项与政策边界", ""])
    for row in review["rows"]:
        binding = row["policy_binding"]
        lines.append(
            "- "
            + row["symbol"]
            + "：移动阈值 "
            + binding["trailing"]["state"]
            + "；初始止损 "
            + binding["initial_stop"]["state"]
            + "；"
            + ("、".join(row["eod_risk"]["blockers"]) or "无证据阻断项")
        )
    lines.extend(
        [
            "",
            "仅采集报价的额外标的：" + (", ".join(review["quote_only_symbols"]) or "无"),
            "",
            "## 证据引用",
            "",
            *[
                "- " + cell(ref["path"]) + " · SHA256 " + ref["sha256"]
                for ref in review["source_refs"]
            ],
            "",
            "复核内容 SHA256：" + review["content_sha256"],
            "",
            "```json",
            canonical_json_bytes(review).decode("utf-8"),
            "```",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")
