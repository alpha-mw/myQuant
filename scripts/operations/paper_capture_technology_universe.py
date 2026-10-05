#!/usr/bin/env python3
"""Capture the strategy's technology universe for one session.

The sealed research pool ranks the **whole market** on two price/volume factors; on
2026-09-30 none of its 100 names belonged to any of the strategy's technology
themes, so pool membership cannot define the candidate universe. This captures the
universe directly instead: one `dc_member` request per DC technology theme from the
research policy (航天航空/新材料/人工智能/PCB/国产芯片/半导体概念/数据中心/商业航天/
机器人概念/算力概念/人形机器人).

Everything lands owner-only under `data/private/paper_evidence/<date>/`:

- `technology-universe/<theme>/raw.json`  exact response bytes per theme
- `technology-universe/capture.json`      request/time/response evidence
- `paper-technology-universe.v1.json`     canonical evidence (symbol → theme ids)

The TDX themes in the policy cannot be served by the DC endpoint and are recorded
as uncovered rather than approximated.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import stat

WORKSPACE = Path("/Users/maxwell/mySpace/myQuant")
EVIDENCE_ROOT = WORKSPACE / "data/private/paper_evidence"
POLICY = WORKSPACE / "results/policies/research/aggressive_tech_manufacturing/v2.json"
DAY = re.compile(r"^[0-9]{8}$")
SYMBOL = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
FIELDS = ("trade_date", "ts_code", "con_code", "name")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_token() -> None:
    if os.environ.get("TUSHARE_TOKEN"):
        return
    env = WORKSPACE / ".env"
    for line in env.read_text().splitlines():
        if line.startswith("TUSHARE_TOKEN="):
            os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip()
            return
    raise SystemExit("TUSHARE_TOKEN missing")


def _write_owner_only(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    path.chmod(0o600)
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise SystemExit(f"{path} is not owner-only")


def dc_technology_themes() -> list[str]:
    payload = json.loads(POLICY.read_text())["payload"]
    if payload["technology_policy_state"] != "ACTIVE":
        raise SystemExit("technology policy is not ACTIVE")
    return sorted(
        theme for theme in payload["technology_theme_ids"] if theme.startswith("TUSHARE_DC:")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if DAY.fullmatch(args.trade_date) is None:
        raise SystemExit("--trade-date must be YYYYMMDD")

    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.market.tushare_transport import (
        OfficialTushareHttpsClient,
        replay_tushare_response_bytes,
    )

    _load_token()
    themes = dc_technology_themes()
    client = OfficialTushareHttpsClient(strict_decimal_decode=True)
    members: dict[str, list[str]] = {}
    raw_by_theme: dict[str, bytes] = {}
    requests: list[dict] = []
    for theme in themes:
        code = theme.split(":", 1)[1]
        response = client.request(
            api_name="dc_member",
            params={"trade_date": args.trade_date, "ts_code": code},
            expected_fields=FIELDS,
        )
        replayed = replay_tushare_response_bytes(
            response.raw_body, api_name="dc_member", expected_fields=FIELDS
        )
        index = {name: position for position, name in enumerate(replayed.fields)}
        found: list[str] = []
        for row in replayed.rows:
            symbol = row[index["con_code"]]
            day = row[index["trade_date"]]
            if type(symbol) is not str or SYMBOL.fullmatch(symbol) is None:
                continue
            if day != args.trade_date:
                raise SystemExit("provider row trade_date differs")
            found.append(symbol)
        if not found:
            raise SystemExit(f"theme {theme} returned no constituents")
        members[code] = sorted(set(found))
        raw_by_theme[code] = response.raw_body
        requests.append(
            {
                "theme": theme,
                "request_id": replayed.request_id,
                "rows": len(found),
                "raw_sha256": _sha(response.raw_body),
            }
        )

    symbols: dict[str, list[str]] = {}
    for code, found in members.items():
        for symbol in found:
            symbols.setdefault(symbol, []).append(code)
    uncovered = sorted(
        theme
        for theme in json.loads(POLICY.read_text())["payload"]["technology_theme_ids"]
        if theme.startswith("TUSHARE_TDX:")
    )
    evidence = canonical_json_bytes(
        {
            "schema_version": "paper-technology-universe.v1",
            "trade_date": args.trade_date,
            "source": "tushare.dc_member",
            "policy_ref": {
                "path": str(POLICY.relative_to(WORKSPACE)),
                "sha256": _sha(POLICY.read_bytes()),
            },
            "themes": {code: members[code] for code in members},
            "symbols": {symbol: sorted(symbols[symbol]) for symbol in sorted(symbols)},
            "uncovered_themes": uncovered,
            "authority": {"broker": False, "real_order": False, "research_only": True},
        }
    )
    capture = canonical_json_bytes(
        {
            "schema_version": "paper-technology-universe-capture.v1",
            "trade_date": args.trade_date,
            "api_name": "dc_member",
            "expected_fields": list(FIELDS),
            "requests": requests,
            "symbol_count": len(symbols),
            "evidence_sha256": _sha(evidence),
            "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "broker": False,
            "real_order": False,
        }
    )

    day_root = EVIDENCE_ROOT / args.trade_date
    result = {
        "trade_date": args.trade_date,
        "themes": len(themes),
        "symbols": len(symbols),
        "uncovered_themes": uncovered,
        "evidence_path": str(
            (day_root / "paper-technology-universe.v1.json").relative_to(WORKSPACE)
        ),
        "evidence_sha256": _sha(evidence),
        "write": bool(args.write),
    }
    if args.write:
        for code, raw in raw_by_theme.items():
            _write_owner_only(day_root / "technology-universe" / f"{code}.json", raw)
        _write_owner_only(day_root / "technology-universe" / "capture.json", capture)
        _write_owner_only(day_root / "paper-technology-universe.v1.json", evidence)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
