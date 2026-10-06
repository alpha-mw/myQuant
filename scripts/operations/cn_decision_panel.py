#!/usr/bin/env python3
"""Run the daily decision panel over one sealed digest and record its memo.

Four lanes answer the same evidence bundle in JSON, then the chair writes a memo:

| lane | profile | 产出 |
| --- | --- | --- |
| 量化筛选 | myquant | 候选解释（分位/主题/闸门漏斗） |
| 价值研究 | myquant_value | 论点、估值、催化、风险、失效条件 |
| 怀疑 | myquant_skeptic | 反驳、严重度、什么能推翻它 |
| 风控 | myquant_risk | PASS / BLOCKED / INSUFFICIENT_EVIDENCE |

The chair (`quant-trading`, the TradeDesk profile with the PM role) receives the
digest and every lane's own words and writes the memo, preserving disagreements.

Two things this runner deliberately does:

- it drives the **deterministic layer**: every lane reads the same digest by
  path+SHA, and nothing here decides anything by itself;
- the risk lane gets a **redacted** digest — its role forbids absolute money
  figures, so cash/NAV/PnL are removed and only weights and percentages survive.

A lane that fails or times out is recorded as `LLM_UNAVAILABLE:<lane>`; the panel
never invents its answer, and the deterministic paper rules run regardless.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
DIGEST_ROOT = WORKSPACE / "data/private/decision_digests"
MEMO_ROOT = WORKSPACE / "data/private/decision_memos"
LANES = (
    ("quant", "myquant", "解释候选：分位、主题归属、闸门漏斗；不要重排因子、不要新增候选"),
    (
        "value",
        "myquant_value",
        "对 shortlist 前 5 个候选给出论点、估值、催化、风险、失效条件；"
        "行业与基本面证据在 digest.research，引用时必须标注报表期与日频 cutoff 滞后",
    ),
    ("skeptic", "myquant_skeptic", "反驳上述候选与论点，给出严重度与什么能推翻你"),
    ("risk", "myquant_risk", "对本轮候选与持仓出具 PASS / BLOCKED / INSUFFICIENT_EVIDENCE"),
)
CHAIR_PROFILE = "quant-trading"
LANE_TIMEOUT_SECONDS = 900
CHAIR_TIMEOUT_SECONDS = 1200
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def build_digest(session: str) -> tuple[Path, str, dict]:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "cn_decision_digest", WORKSPACE / "scripts/operations/cn_decision_digest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    digest = module.build(session)
    path = DIGEST_ROOT / session / "cn-decision-digest.v1.json"
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    written = json.dumps(digest, ensure_ascii=False, indent=1, sort_keys=True).encode()
    path.write_bytes(written)
    path.chmod(0o600)
    # The ref must hash the bytes the lanes actually read, not the canonical form.
    return path, _sha(written), digest


def risk_summary(digest: dict, digest_ref: dict[str, str]) -> dict:
    """Minimal-disclosure risk input: derived metrics only.

    The risk role forbids account identifiers, real holdings and financial
    detail in its context, so it receives no symbols, no share counts, no cost
    and no money — only per-rule results, derived ratios, evidence refs and
    status, which is what its check list actually needs.
    """

    flags: dict[str, int] = {}
    for row in digest.get("risk", []):
        if row.get("owner_stop_trigger") == "BREACH":
            flags["owner_stop_breach"] = flags.get("owner_stop_breach", 0) + 1
        if row.get("trailing_trigger") == "REDUCTION_REVIEW":
            flags["trailing_reduction_review"] = flags.get("trailing_reduction_review", 0) + 1
        if row.get("trailing_trigger") == "REVIEW":
            flags["trailing_review"] = flags.get("trailing_review", 0) + 1
        for blocker in row.get("blockers", []):
            key = "corporate_action_review" if "CORPORATE_ACTION" in blocker else "other_blocker"
            flags[key] = flags.get(key, 0) + 1
        if row.get("hard_stop") is None and row.get("giveback_ratio") is None:
            flags["unmanaged_position"] = flags.get("unmanaged_position", 0) + 1
    orders = digest.get("orders", {}).get("plans", [])
    actions: dict[str, int] = {}
    for plan in orders:
        for order in plan.get("orders", []):
            actions[order["action"]] = actions.get(order["action"], 0) + 1
    drawdown = digest.get("drawdown") or {}
    clock = _clock_evidence()
    policies = {
        "paper_execution_policy_ref": _policy_ref(
            WORKSPACE
            / "results/policies/paper/aggressive_tech_manufacturing"
            / "owner-paper-risk-execution-policy-20261005-v4.json"
        ),
        "owner_stop_policy_ref": _policy_ref(
            WORKSPACE
            / "results/policies/risk/aggressive_tech_manufacturing/initial-risk-stop.v1"
            / "owner-stop-policy-20260828-v1.json"
        ),
        "trailing_anchor_policy_ref": _policy_ref(
            WORKSPACE
            / "results/policies/risk/aggressive_tech_manufacturing/trailing-anchor.v1"
            / "owner-trailing-anchor-policy-20260901-v1.json"
        ),
    }
    # The evidence envelope the risk role's ROLE.md/RISK_WORKFLOW.md require.
    # Everything factual is filled in; the governance pieces no policy provides
    # are named as gaps instead of being invented.
    gaps = [
        "SOURCE_ALLOWLIST_NOT_PROVIDED",
        "VALIDITY_RULE_NOT_PROVIDED",
        "DRAWDOWN_THRESHOLD_NOT_PROVIDED",
    ]
    envelope = {
        "mode": "monitor+pretrade",
        "issuer_ref": {"issuer": "cn-paper-panel-runner.v1", **digest_ref},
        "source_ref": dict(digest_ref),
        "snapshot_ref": dict(clock["snapshot_pointer_ref"]),
        "clock_evidence_ref": dict(clock["snapshot_pointer_ref"]),
        "observed_at": digest.get("generated_at"),
        "received_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timezone": "Asia/Shanghai",
        "policy_refs": [dict(ref) for ref in policies.values()],
        "valid_from": sorted(
            ref["effective_from"] for ref in policies.values() if ref.get("effective_from")
        ),
        "valid_until": None,
        "source_allowlist_ref": None,
        "freshness_rule_ref": None,
        "gaps": gaps,
    }
    return {
        "schema_version": "cn-risk-input.v1",
        "trade_date": digest["trade_date"],
        "sampled_at": digest.get("generated_at"),
        "timezone": "Asia/Shanghai",
        "digest_ref": dict(digest_ref),
        "scope": "derived metrics only; no symbols, shares, cost or money",
        "disclosure": "observed 为派生比率，仅供规则判定；不得写入对外简报或告警",
        "envelope": envelope,
        "concentration": digest.get("concentration", {}),
        "drawdown": {
            "status": drawdown.get("status"),
            "reason": drawdown.get("reason"),
            "observations": drawdown.get("observations"),
            "max_drawdown_fraction": drawdown.get("max_drawdown_fraction"),
            "current_drawdown_fraction": drawdown.get("current_drawdown_fraction"),
            "partial_sessions": drawdown.get("partial_sessions"),
            "stopped_at": drawdown.get("stopped_at"),
            "basis": drawdown.get("basis"),
            "rule": {
                "threshold": None,
                "status": "NO_APPROVED_THRESHOLD",
                "note": "未获批准的账户回撤阈值；缺少获批阈值时该项不得 PASS",
            },
            "evidence_ref": dict(digest_ref),
        },
        "risk_flags": flags,
        "proposed_actions": actions,
        "evidence": {
            "missing_lanes": digest["evidence"]["missing"],
            "stale_lanes": sorted(digest["evidence"].get("stale") or {}),
            "present_lane_count": len(digest["evidence"]["present"]),
            "digest_integrity": "MATCHES_SUPPLIED_SHA",
        },
        "policies": policies,
        "validity": {
            "ttl_rule": None,
            "clock_skew_rule": None,
            "status": "NO_APPROVED_TTL",
            "note": "政策只提供 effective_from；未提供时效上限与允许时钟偏差，缺失即 fail closed",
        },
        "seal_veto": digest.get("seal_veto", {"state": "UNKNOWN"}),
        "clock": clock,
        "boundary": "本输入为派生指标，供风控出具 PASS / BLOCKED / INSUFFICIENT_EVIDENCE",
    }


def _policy_ref(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    ref = {"path": str(path), "sha256": _sha(raw)}
    value = json.loads(raw)
    if isinstance(value, dict) and value.get("effective_from"):
        ref["effective_from"] = str(value["effective_from"])
    return ref


def _clock_evidence() -> dict:
    """The session's own time evidence: the strict snapshot pointer it reads."""

    path = WORKSPACE / "data/parquet/cn/_latest.json"
    raw = path.read_bytes()
    value = json.loads(raw)
    return {
        "snapshot_pointer_ref": {"path": "data/parquet/cn/_latest.json", "sha256": _sha(raw)},
        "snapshot_updated_at": value.get("updated_at"),
        "latest_complete_trade_date": value.get("latest_complete_trade_date"),
        "timezone": "Asia/Shanghai",
    }


def _call(profile: str, prompt: str, timeout: int) -> tuple[bool, dict | None, str]:
    try:
        completed = subprocess.run(
            ["hermes", "-p", profile, "-z", prompt],
            cwd=str(WORKSPACE),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, None, "TIMEOUT"
    output = (completed.stdout or "").strip()
    if completed.returncode != 0:
        return False, None, f"EXIT_{completed.returncode}:{output[-200:]}"
    match = _JSON_BLOCK.search(output)
    if match is None:
        return False, None, f"NO_JSON:{output[-200:]}"
    try:
        return True, json.loads(match.group(0)), ""
    except ValueError as exc:
        return False, None, f"BAD_JSON:{exc}"


def _lane_prompt(role: str, instruction: str, ref_path: Path, sha: str, *, redacted: bool) -> str:
    target = ref_path
    if redacted:
        target = ref_path.with_name("cn-risk-input.v1.json")
    return (
        f"你是 {role} lane。只读这一份输入（**绝对路径** path={target}，"
        f"sha256={sha}），不要访问其他项目数据。按你的角色文件输出**一个 JSON 对象**，不要输出其他文字。"
        f"任务：{instruction}。evidence.missing 的 lane 必须报 INSUFFICIENT_EVIDENCE；"
        "evidence.stale 的 lane 必须标注滞后并按你的角色判断可用范围。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    session = args.trade_date

    ref_path, sha, digest = build_digest(session)
    redacted_path = ref_path.with_name("cn-risk-input.v1.json")
    redacted_payload = risk_summary(digest, {"path": str(ref_path), "sha256": sha})
    redacted_written = json.dumps(
        redacted_payload, ensure_ascii=False, indent=1, sort_keys=True
    ).encode()
    redacted_path.write_bytes(redacted_written)
    redacted_path.chmod(0o600)
    redacted_sha = _sha(redacted_written)

    report: dict = {
        "schema_version": "cn-decision-panel.v1",
        "trade_date": session,
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "digest_ref": {"path": str(ref_path.relative_to(WORKSPACE)), "sha256": sha},
        "lanes": {},
        "unavailable": [],
    }
    for role, profile, instruction in LANES:
        redacted = role == "risk"
        prompt = _lane_prompt(
            role,
            instruction,
            ref_path,
            redacted_sha if redacted else sha,
            redacted=redacted,
        )
        ok, value, error = _call(profile, prompt, LANE_TIMEOUT_SECONDS)
        if ok and value is not None:
            report["lanes"][role] = value
        else:
            report["unavailable"].append(f"LLM_UNAVAILABLE:{role}")
            report["lanes"][role] = {"error": error}

    chair_prompt = (
        "你是投资决策主席（PM）。只读 digest（path="
        f"{ref_path.relative_to(WORKSPACE).as_posix()}，sha256={sha}）与下面四个 lane 的原始输出，"
        "写当日 decision memo 的 JSON：候选结论、各 lane 要点、**逐条保留异议原文**、"
        "最终动作建议（仓位/触发/失效条件）、证据缺口。风控 BLOCKED 即 BLOCKED，不得抹平分歧。"
        "只输出一个 JSON 对象。\n\nlane 输出：\n" + json.dumps(report["lanes"], ensure_ascii=False)
    )
    ok, memo, error = _call(CHAIR_PROFILE, chair_prompt, CHAIR_TIMEOUT_SECONDS)
    if not ok or memo is None:
        report["memo"] = {"error": error}
        report["unavailable"].append("LLM_UNAVAILABLE:chair")
    else:
        report["memo"] = memo

    raw = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    report["content_sha256"] = _sha(raw)
    if args.write:
        out = MEMO_ROOT / session / "panel-memo.v1.json"
        out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        out.write_bytes(json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True).encode())
        out.chmod(0o600)
        print(
            json.dumps(
                {
                    "status": "WRITTEN",
                    "path": str(out.relative_to(WORKSPACE)),
                    "content_sha256": report["content_sha256"],
                    "unavailable": report["unavailable"],
                },
                ensure_ascii=False,
            )
        )
    else:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
