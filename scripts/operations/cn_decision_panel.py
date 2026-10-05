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
    ("value", "myquant_value", "对 shortlist 前 5 个候选给出论点、估值、催化、风险、失效条件"),
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
    path.write_bytes(json.dumps(digest, ensure_ascii=False, indent=1, sort_keys=True).encode())
    path.chmod(0o600)
    return path, digest["content_sha256"], digest


def redacted_digest(digest: dict) -> dict:
    """The risk lane may not receive absolute money; weights and flags survive."""

    account = dict(digest["account"])
    for field in ("cash", "realized_pnl", "cumulative_fees"):
        account.pop(field, None)
    account["positions"] = [
        {
            "symbol": row["symbol"],
            "shares_bucket": "LOT" if row["shares"] >= 100 else "ODD",
            "settled": row["settled_shares"] == row["shares"],
            "avg_cost": row["avg_cost"],
        }
        for row in account.get("positions", [])
    ]
    return {**digest, "account": account, "redaction": "RISK_LANE_NO_ABSOLUTE_MONEY"}


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
        target = ref_path.with_name("cn-decision-digest.risk-redacted.v1.json")
    return (
        f"你是 {role} lane。只读这一份 digest（path={target.relative_to(WORKSPACE).as_posix()}，"
        f"sha256={sha}），不要访问其他项目数据。按你的角色文件输出**一个 JSON 对象**，不要输出其他文字。"
        f"任务：{instruction}。digest 里 evidence.missing 的 lane 必须报 INSUFFICIENT_EVIDENCE。"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    session = args.trade_date

    ref_path, sha, digest = build_digest(session)
    redacted_path = ref_path.with_name("cn-decision-digest.risk-redacted.v1.json")
    redacted_payload = redacted_digest(digest)
    redacted_path.write_bytes(
        json.dumps(redacted_payload, ensure_ascii=False, indent=1, sort_keys=True).encode()
    )
    redacted_path.chmod(0o600)

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
            sha,
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
