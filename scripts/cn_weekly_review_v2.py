"""Independent coverage and report projections for the weekly v2 consumer."""

from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

SCHEMA_V2 = "cn_weekly_portfolio_evidence.v2"
DAILY_SCHEMA_V2 = "cn_weekly_daily_review_input.v2"
SHANGHAI = ZoneInfo("Asia/Shanghai")


def daily_domain(value, source_ref, *, window, expected_trade_dates):
    from scripts.export_cn_weekly_review_evidence import WeeklyEvidenceError, _domain, _parse_utc

    selected = {}
    start = _parse_utc(window["start_at"], label="window start")
    end = _parse_utc(window["end_at"], label="window end")
    rows = [] if value is None else value.get("items")
    if not isinstance(rows, list) or len(rows) > 100:
        raise WeeklyEvidenceError("v2 daily review items invalid")
    identities = {}
    for row in rows:
        if not isinstance(row, dict):
            raise WeeklyEvidenceError("v2 daily review row invalid")
        if row.get("automation_id") != "automation":
            raise WeeklyEvidenceError(
                "v2 automation identity differs from registered weekly consumer"
            )
        for key in ("thread_id", "run_id"):
            if not isinstance(row.get(key), str) or not row[key] or len(row[key]) > 200:
                raise WeeklyEvidenceError("v2 daily run identity invalid")
        identity = (row["automation_id"], row["run_id"])
        raw = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
        if identity in identities and identities[identity] != raw:
            raise WeeklyEvidenceError("v2 conflicting run identity")
        identities[identity] = raw
        begun = _parse_utc(row.get("started_at"), label="daily started_at")
        state = row.get("run_status")
        research = row.get("research_status")
        if state not in {"COMPLETED", "FAILED", "IN_PROGRESS"} or research not in {
            "COMPLETE",
            "PARTIAL",
            "BLOCKED",
            "NOT_RUN",
        }:
            raise WeeklyEvidenceError("v2 daily status invalid")
        finished = None
        if row.get("completed_at") is not None:
            finished = _parse_utc(row["completed_at"], label="daily completed_at")
            if finished < begun:
                raise WeeklyEvidenceError("v2 daily chronology invalid")
        if (state == "IN_PROGRESS") != (finished is None) or (
            research == "COMPLETE" and state != "COMPLETED"
        ):
            raise WeeklyEvidenceError("v2 daily completion inconsistent")
        day = row.get("trade_date")
        if day not in expected_trade_dates or not start <= begun < end:
            continue
        if begun.astimezone(SHANGHAI).date().isoformat() < day:
            raise WeeklyEvidenceError("v2 run starts before target date")
        within = finished is not None and finished < end
        timing = (
            "SAME_SESSION"
            if within and finished.astimezone(SHANGHAI).date().isoformat() == day
            else "LATE_RECOVERY" if within else "UNFINISHED_IN_WINDOW"
        )
        selected[identity] = {
            **row,
            "timing": timing,
            "completed_in_window": within,
            "timing_semantics": "FACTUAL_SAME_LOCAL_TRADE_DATE",
            "scheduled_punctuality": "NOT_CONFIGURED",
            "research_complete_in_window": within
            and state == "COMPLETED"
            and research == "COMPLETE",
        }
    items = [selected[key] for key in sorted(selected)]
    task_dates = sorted({r["trade_date"] for r in items})
    complete_dates = sorted({r["trade_date"] for r in items if r["research_complete_in_window"]})
    missing = sorted(set(expected_trade_dates) - set(task_dates))
    evidence = {
        "source_ref": source_ref,
        "expected_trade_dates": list(expected_trade_dates),
        "covered_trade_dates": task_dates,
        "task_review_dates": task_dates,
        "research_complete_dates": complete_dates,
        "research_incomplete_dates": sorted(set(expected_trade_dates) - set(complete_dates)),
        "late_recovery_dates": sorted(
            {r["trade_date"] for r in items if r["timing"] == "LATE_RECOVERY"}
        ),
        "failed_dates": sorted({r["trade_date"] for r in items if r["run_status"] == "FAILED"}),
        "missing_task_dates": missing,
        "formal_closure_dates": [],
        "formal_closure_source": "EXACT_MAINLINE_ONLY_NOT_NARRATIVE",
    }
    return (
        _domain(
            "FRESH" if set(complete_dates) == set(expected_trade_dates) else "PARTIAL",
            blockers=(["DAILY_REVIEW_EXPECTED_TRADING_DAY_MISSING"] if missing else [])
            + (["DAILY_RESEARCH_INCOMPLETE"] if evidence["research_incomplete_dates"] else []),
            evidence=evidence,
        ),
        items,
    )


def period_projection(bundle: dict[str, Any], expected: list[str]) -> dict[str, Any]:
    performance = bundle.get("performance_benchmark")
    if not performance:
        return {"state": "BLOCKED", "blockers": ["PERFORMANCE_UNAVAILABLE"]}
    points = performance["portfolio"]["performance_points"]
    start = bundle["report_window"]["start_date"]
    end = bundle["report_window"]["end_date"]
    by_date = {row["date"]: row for row in points if row["date"] <= end}
    baselines = [day for day in by_date if day < start]
    available = sorted(set(expected) & set(by_date))
    missing = sorted(set(expected) - set(available))
    if not baselines or not available:
        return {
            "state": "BLOCKED",
            "blockers": ["PERIOD_BASELINE_OR_END_MISSING"],
            "missing_dates": missing,
        }
    baseline, last = by_date[max(baselines)], by_date[available[-1]]
    portfolio_return = last["portfolio_unit_nav"] / baseline["portfolio_unit_nav"] - 1
    benchmark_return = None
    if baseline.get("csi300_nav") and last.get("csi300_nav"):
        benchmark_return = last["csi300_nav"] / baseline["csi300_nav"] - 1
    full = (
        not missing
        and benchmark_return is not None
        and all(
            by_date[day].get("benchmark_coverage") == "exact_close"
            and by_date[day].get("benchmark_value_date") == day
            for day in [baseline["date"], *available]
        )
    )
    return {
        "state": "FULL_WEEK" if full else "PARTIAL_WEEK",
        "expected_dates": expected,
        "complete_dates": available,
        "missing_dates": missing,
        "baseline_date": baseline["date"],
        "baseline_record": baseline["record"],
        "end_date": last["date"],
        "end_record": last["record"],
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "excess_percentage_points": (
            None if benchmark_return is None else 100 * (portfolio_return - benchmark_return)
        ),
        "nav_change_cny": last["total_value"] - baseline["total_value"],
        "realized_pnl": None,
        "fee_basis": "UNKNOWN",
    }


def assess_production_days(rows, expected):
    """Count consecutive component closures, without inferring scheduler dispatch."""
    by_day = {row["trade_date"]: row for row in rows}
    complete, late = [], []
    run = maximum = 0
    for day in expected:
        row = by_day.get(day, {})
        ready = all(
            row.get(key) is True
            for key in ("maintenance_verified", "observations_verified", "top100_verified")
        )
        if ready and row.get("artifact_completion_at"):
            completed = datetime.fromisoformat(row["artifact_completion_at"].replace("Z", "+00:00"))
            if completed.astimezone(SHANGHAI).date().isoformat() == day:
                complete.append(day)
                run += 1
                maximum = max(maximum, run)
                continue
            late.append(day)
        run = 0
    return {
        "complete_same_day_dates": complete,
        "late_recovery_dates": late,
        "missing_or_unverified_dates": sorted(set(expected) - set(complete) - set(late)),
        "max_consecutive_complete_days": maximum,
        "five_day_checkpoint": "MET" if maximum >= 5 else "NOT_MET",
        "ten_day_acceptance": "MET" if maximum >= 10 else "NOT_MET",
        "acceptance_scope": "COMPONENT_CLOSURE_ONLY_NOT_UNATTENDED_DISPATCH",
    }


def factor_coverage(project: Path, expected: list[str], producer_refs=None) -> dict[str, Any]:
    from quant_investor.factors.production_observation import validate_factor_production_observation
    from quant_investor.factors.production_authority import (
        validate_factor_active_pointer,
        validate_factor_production_rollover_bundle,
    )
    from quant_investor.factors.production_rollover import validate_daily_maintenance_receipt
    from quant_investor.strategy_records.store import regular_file_sha256
    from quant_investor.contracts import validate_artifact, canonical_json_bytes
    from quant_investor.intelligence._common import artifact_ref

    refs = []

    def exact(path):
        if path.resolve(strict=True) != path.absolute() or not path.is_relative_to(project):
            raise ValueError("FACTOR_SOURCE_PATH_UNSAFE")
        sha, _ = regular_file_sha256(path, label="Factor weekly evidence")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("FACTOR_EVIDENCE_DRIFT")
        refs.append({"path": str(path.relative_to(project)), "sha256": sha})
        return json.loads(raw), sha

    result: dict[str, Any] = {
        "status": "PARTIAL",
        "dates": [],
        "unattended_successful_dates": [],
        "continuous_unattended_operation": "NOT_PROVEN",
        "five_day_checkpoint": "NOT_PROVEN",
        "ten_day_acceptance": "NOT_PROVEN",
        "outcome_consumer_state": "DEPENDENCY_BLOCKED",
        "outcome_blocker": "VERSIONED_PRODUCER_HANDOFF_REQUIRED",
        "source_refs": refs,
    }
    try:
        pointer, pointer_sha = exact(project / "results/factors/_active.json")
        original_head = pointer_sha
        generations, seen, receipt_refs = {}, set(), list(producer_refs or [])
        generation_refs = {}
        while pointer_sha not in seen:
            seen.add(pointer_sha)
            validated = validate_factor_active_pointer(canonical_json_bytes(pointer))
            path = (
                project
                / "results/factors/generations"
                / validated["factor_generation_id"]
                / "generation.json"
            )
            generation, generation_sha = exact(path)
            if generation_sha != validated["factor_generation_sha256"]:
                raise ValueError("FACTOR_GENERATION_SHA_MISMATCH")
            generations[pointer_sha] = validate_artifact(
                generation, expected_kind="factor.production_generation"
            )["payload"]
            generation_refs[pointer_sha] = artifact_ref(generation)
            rollover = project / "results/factors/rollover_bundles" / f"{pointer_sha}.json"
            if rollover.exists():
                rb, _ = exact(rollover)
                rp = validate_factor_production_rollover_bundle(rb)["payload"]
                if rp["target_date"] in {day.replace("-", "") for day in expected}:
                    receipt_refs.append(
                        {
                            "path": rp["maintenance_receipt_path"],
                            "sha256": rp["maintenance_receipt_sha256"],
                        }
                    )
            previous = validated.get("previous_pointer_sha256")
            if previous == "EMPTY":
                break
            pointer, pointer_sha = exact(
                project / "results/factors/pointer_history" / f"{previous}.json"
            )
            if pointer_sha != previous:
                raise ValueError("FACTOR_LINEAGE_SHA_MISMATCH")
        else:
            raise ValueError("FACTOR_LINEAGE_CYCLE")
        maintenance = {}
        for ref in receipt_refs:
            path = Path(ref["path"])
            if not path.is_absolute():
                path = project / path
            allowed = project / "data/private/cn_daily_maintenance/attempts"
            if not path.is_relative_to(allowed):
                raise ValueError("PRODUCER_RECEIPT_OUTSIDE_REGISTERED_ROOT")
            payload, sha = exact(path)
            if sha != ref["sha256"]:
                raise ValueError("PRODUCER_RECEIPT_SHA_MISMATCH")
            target = str(payload.get("target_date", ""))
            day = f"{target[:4]}-{target[4:6]}-{target[6:]}"
            if day not in expected:
                continue
            row = {
                "path": str(path.relative_to(project)),
                "sha256": sha,
                "declared_status": payload.get("status"),
                "verified": False,
            }
            try:
                validate_daily_maintenance_receipt(
                    workspace_root=project, receipt_path=path, expected_receipt_sha256=sha
                )
                row["verified"] = True
            except (ValueError, RuntimeError, OSError) as exc:
                row["validation_blocker"] = str(exc)
            if day not in maintenance or row["verified"]:
                maintenance[day] = row
        for day in expected:
            observations: list[dict[str, Any]] = []
            observation_refs = []
            for alias in ("LOW", "W80"):
                path = (
                    project
                    / "results/factors/observations"
                    / day[:4]
                    / day[5:7]
                    / day[8:]
                    / f"{alias}.json"
                )
                if not path.exists():
                    observations.append({"alias": alias, "state": "MISSING"})
                    continue
                value, _ = exact(path)
                payload = validate_factor_production_observation(value)["payload"]
                observation_refs.append(artifact_ref(value))
                origin = generations.get(payload["factor_pointer_sha256"])
                signal_key = "low_signal_sha256" if alias == "LOW" else "w80_signal_sha256"
                if (
                    payload["signal_date"] != day.replace("-", "")
                    or payload["factor_alias"] != alias
                    or origin is None
                    or origin["as_of"] != payload["signal_date"]
                    or origin[signal_key] != payload["signal_sha256"]
                    or origin["factor_production_generation_id"] != payload["factor_generation_id"]
                ):
                    raise ValueError("FACTOR_OBSERVATION_ORIGIN_MISMATCH")
                observations.append(
                    {
                        "alias": alias,
                        "state": "OPEN",
                        "registered_at": payload["registered_at"],
                        "late_registration": datetime.fromisoformat(
                            payload["registered_at"].replace("Z", "+00:00")
                        )
                        .astimezone(SHANGHAI)
                        .date()
                        .isoformat()
                        > day,
                    }
                )
            pool = (
                project
                / "results/intelligence/research_pool/aggressive_tech_manufacturing"
                / day
                / "manifest.json"
            )
            pool_state, pool_at = "MISSING", None
            if pool.exists():
                from quant_investor.intelligence.storage import DailyResearchPoolStore

                manifest, _ = exact(pool)
                mp = manifest["payload"]
                policy, policy_sha = exact(project / mp["policy_path"])
                if mp["signal_date"] < policy["payload"]["effective_signal_date"]:
                    raise ValueError("TOP100_POLICY_NOT_EFFECTIVE")
                rank, _ = exact(pool.parent / "factor_research_rank.json")
                verified_pool = DailyResearchPoolStore(project).verify(
                    rank=rank,
                    policy_path=mp["policy_path"],
                    expected_policy_sha256=policy_sha,
                )
                if (
                    verified_pool["manifest.json"]["sha256"]
                    != hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
                ):
                    raise ValueError("TOP100_REPLAY_MISMATCH")
                if (
                    mp["signal_date"] != day.replace("-", "")
                    or mp["factor_pointer_sha256"] not in generations
                ):
                    raise ValueError("TOP100_ORIGIN_MISMATCH")
                if mp["factor_generation_ref"] != generation_refs[mp["factor_pointer_sha256"]]:
                    raise ValueError("TOP100_GENERATION_REFERENCE_MISMATCH")
                if {json.dumps(ref, sort_keys=True) for ref in mp["observation_refs"]} != {
                    json.dumps(ref, sort_keys=True) for ref in observation_refs
                }:
                    raise ValueError("TOP100_OBSERVATION_REFERENCE_MISMATCH")
                pool_state, pool_at = "VERIFIED", manifest["created_at"]
            stamps = [x["registered_at"] for x in observations if x["state"] == "OPEN"]
            result["dates"].append(
                {
                    "trade_date": day,
                    "observations": observations,
                    "top100_state": pool_state,
                    "maintenance": maintenance.get(day),
                    "maintenance_verified": maintenance.get(day, {}).get("verified", False),
                    "observations_verified": len(stamps) == 2,
                    "top100_verified": pool_state == "VERIFIED",
                    "artifact_completion_at": (
                        max([*stamps, pool_at]) if len(stamps) == 2 and pool_at else None
                    ),
                }
            )
        result.update(assess_production_days(result["dates"], expected))
        result["blockers"] = [
            "DAILY_COMPONENT_CLOSURE_UNPROVEN:" + day
            for day in result["missing_or_unverified_dates"]
        ]
        result["blockers"] += [
            "LATE_FACTOR_RECOVERY:" + day for day in result["late_recovery_dates"]
        ]
        result["status"] = "FRESH" if not result["blockers"] else "PARTIAL"
        _, after = exact(project / "results/factors/_active.json")
        if after != original_head:
            raise ValueError("FACTOR_POINTER_CHANGED_DURING_READ")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        result.update(status="BLOCKED", blocker=str(exc), blockers=[str(exc)])
    return result


def enrich(bundle: dict[str, Any], project: Path, expected: list[str]) -> None:
    from scripts.export_cn_research_risk import build_risk_monitor
    from scripts.export_cn_weekly_review_evidence import _domain
    from scripts.export_cn_weekly_review_evidence import _registered_cn_trade_dates

    period = period_projection(bundle, expected)
    bundle["period"] = period
    bundle["risk_monitor"] = (
        build_risk_monitor(project, as_of=expected[-1])
        if expected
        else {"status": "BLOCKED", "blockers": ["REGISTERED_CALENDAR_UNAVAILABLE"]}
    )
    acceptance_dates = expected
    if expected:
        lookback = (datetime.fromisoformat(expected[-1]) - timedelta(days=45)).date().isoformat()
        acceptance_dates = _registered_cn_trade_dates(
            project, start_date=lookback, end_date=expected[-1]
        )[0][-10:]
    bundle["factor_daily_coverage"] = factor_coverage(
        project, acceptance_dates, bundle.get("producer_receipt_refs")
    )
    bundle["factor_daily_coverage"]["report_trade_dates"] = expected
    bundle["factor_daily_coverage"]["acceptance_trade_dates"] = acceptance_dates
    risk = bundle["risk_monitor"]
    risk_blockers = sorted(
        set(
            [
                *risk.get("blockers", []),
                *[
                    row["symbol"] + ":" + reason
                    for row in risk.get("rows", [])
                    for reason in row["blockers"]
                ],
            ]
        )
    )
    bundle["domains"]["RISK_MONITOR"] = _domain(
        "FRESH" if risk["status"] == "READY" else risk["status"],
        blockers=risk_blockers,
        evidence={"content_sha256": risk.get("content_sha256"), "as_of": risk.get("as_of")},
    )
    factor = bundle["factor_daily_coverage"]
    bundle["domains"]["FACTOR_DAILY_COVERAGE"] = _domain(
        factor["status"],
        blockers=factor.get("blockers", ["CONTINUOUS_DAILY_PRODUCTION_NOT_PROVEN"]),
        evidence={"continuous_unattended_operation": factor["continuous_unattended_operation"]},
    )
    bundle["domains"]["FACTOR_EFFECTIVENESS"] = _domain(
        "DEPENDENCY_BLOCKED",
        blockers=[factor["outcome_blocker"]],
        evidence={"cost_adjusted": "UNAVAILABLE", "portfolio_attribution": "UNAVAILABLE"},
    )
    bundle["attribution"] = {
        "factor": "DEPENDENCY_BLOCKED_VERSIONED_PRODUCER_HANDOFF_REQUIRED",
        "selection": "UNAVAILABLE_NO_SAME_DATE_SELECTION_CONSUMPTION_BINDING",
        "portfolio": "DESCRIPTIVE_NAV_CHANGE_ONLY",
        "risk_decisions": "UNAVAILABLE_NO_EXECUTION_PAIRING",
        "cost_adjusted": "UNAVAILABLE_NO_COST_BINDING",
        "formal_authority": False,
    }
    if period["state"] != "FULL_WEEK":
        domain = bundle["domains"]["PERFORMANCE_BENCHMARK"]
        if domain["status"] == "FRESH":
            domain["status"] = "PARTIAL"
            domain["warnings"].append("FULL_WEEK_PERFORMANCE_UNAVAILABLE")
    if expected and bundle.get("holdings") and bundle["holdings"]["as_of"] < expected[-1]:
        domain = bundle["domains"]["STORE_HOLDINGS"]
        if domain["status"] == "FRESH":
            domain["status"] = "PARTIAL"
            domain["warnings"].append("CURRENT_HOLDINGS_CONTINUITY_UNCONFIRMED")


def render_summary(bundle: dict[str, Any]) -> str:
    """Compact investment-facing projection; full evidence remains the bundle."""
    period = bundle["period"]
    lines = [
        f"本周状态：{bundle['status']}；收益覆盖：{period['state']}。",
        "",
        f"风险证据：{bundle['risk_monitor']['status']}；"
        f"Factor日更：{bundle['factor_daily_coverage']['status']}；"
        "无人值守证据：NOT_PROVEN；正式建议：BLOCKED。",
        "",
    ]
    if "portfolio_return" in period:
        lines += [
            f"区间 {period['baseline_date']} → {period['end_date']}；"
            f"组合收益 {period['portfolio_return']:.2%}；"
            f"资产变化 {period['nav_change_cny']:+,.2f} 元。",
            "",
        ]
    lines += [
        "| 持仓 | 移动复核价 | 进一步复核价 | Owner stop | 状态 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in bundle["risk_monitor"].get("rows", []):
        values = [
            row.get(k) or "—"
            for k in (
                "moving_take_profit_review_price",
                "moving_take_profit_reduce_price",
                "owner_stop_price",
            )
        ]
        lines.append(
            f"| {row['symbol']} {row['name']} | {' | '.join(values)} | "
            f"{row['calculation_state']} / {row['threshold_state']} |"
        )
    lines += [
        "",
        "利润回吐100%指峰值浮盈全部回吐，不是股价下跌100%。",
        "",
        "所有阈值仅为研究复核；退休或未确认的旧值仅保留在证据附件。",
        "正式建议：FORMAL_ADVISORY_BLOCKED；Decision Log：NOT_APPLICABLE。",
    ]
    return "\n".join(lines) + "\n"
