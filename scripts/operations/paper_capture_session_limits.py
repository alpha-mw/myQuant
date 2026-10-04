#!/usr/bin/env python3
"""Capture and seal one session's exchange price limits for the Paper writer.

`paper-input-eligibility.v1` needs a `price_limit_ref`, and the Board/ST rule
that would derive it disagrees with the exchange on ~0.33% of real symbol-days
(ST renames and first sessions of new listings), so the limits are captured from
the provider instead.

One request per session to the official endpoint (`stk_limit`), through the same
transport the market pipeline uses. Everything lands under
`data/private/paper_evidence/<trade_date>/`, owner-only:

- `stk_limit/raw.json`            exact response bytes
- `stk_limit/capture.json`        request/time/response evidence
- `paper-price-limit-evidence.v1.json`  canonical evidence consumed downstream
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
SYMBOL = re.compile(r"^[0-9]{6}\.(?:SH|SZ|BJ)$")
DAY = re.compile(r"^[0-9]{8}$")
FIELDS = ("trade_date", "ts_code", "up_limit", "down_limit")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_token() -> None:
    """The transport reads TUSHARE_TOKEN from the environment only."""

    if os.environ.get("TUSHARE_TOKEN"):
        return
    env = WORKSPACE / ".env"
    if not env.exists():
        raise SystemExit("TUSHARE_TOKEN is not set and no workspace .env exists")
    for line in env.read_text().splitlines():
        if line.startswith("TUSHARE_TOKEN="):
            os.environ["TUSHARE_TOKEN"] = line.split("=", 1)[1].strip()
            return
    raise SystemExit("TUSHARE_TOKEN missing from the workspace .env")


def _write_owner_only(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_bytes(raw)
    path.chmod(0o600)
    stat_result = path.stat()
    if stat.S_IMODE(stat_result.st_mode) != 0o600:
        raise SystemExit(f"{path} is not owner-only")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trade-date", required=True, help="YYYYMMDD session")
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
    requested_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    response = OfficialTushareHttpsClient(strict_decimal_decode=True).request(
        api_name="stk_limit",
        params={"trade_date": args.trade_date},
        expected_fields=FIELDS,
    )
    replayed = replay_tushare_response_bytes(
        response.raw_body, api_name="stk_limit", expected_fields=FIELDS
    )
    index = {name: position for position, name in enumerate(replayed.fields)}
    limits: dict[str, dict[str, str]] = {}
    for row in replayed.rows:
        symbol = row[index["ts_code"]]
        day = row[index["trade_date"]]
        if type(symbol) is not str or SYMBOL.fullmatch(symbol) is None or day != args.trade_date:
            raise SystemExit("provider row identity is invalid")
        limits[symbol] = {
            "limit_up": f"{float(row[index['up_limit']]):.2f}",
            "limit_down": f"{float(row[index['down_limit']]):.2f}",
        }
    if not limits:
        raise SystemExit("provider returned no limits for the session")

    evidence = canonical_json_bytes(
        {
            "schema_version": "paper-price-limit-evidence.v1",
            "trade_date": args.trade_date,
            "source": "tushare.stk_limit",
            "provider_request_id": replayed.request_id,
            "symbols": dict(sorted(limits.items())),
        }
    )
    capture = canonical_json_bytes(
        {
            "schema_version": "paper-price-limit-capture.v1",
            "trade_date": args.trade_date,
            "api_name": "stk_limit",
            "params": {"trade_date": args.trade_date},
            "expected_fields": list(FIELDS),
            "request_id": replayed.request_id,
            "provider_reported_count": replayed.provider_reported_count,
            "item_count": replayed.item_count,
            "raw_sha256": _sha(response.raw_body),
            "evidence_sha256": _sha(evidence),
            "requested_at": requested_at,
            "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "broker": False,
            "real_order": False,
        }
    )

    day_root = EVIDENCE_ROOT / args.trade_date
    result = {
        "trade_date": args.trade_date,
        "symbol_count": len(limits),
        "request_id": replayed.request_id,
        "raw_sha256": _sha(response.raw_body),
        "evidence_path": str(
            (day_root / "paper-price-limit-evidence.v1.json").relative_to(WORKSPACE)
        ),
        "evidence_sha256": _sha(evidence),
        "write": bool(args.write),
    }
    if args.write:
        _write_owner_only(day_root / "stk_limit" / "raw.json", response.raw_body)
        _write_owner_only(day_root / "stk_limit" / "capture.json", capture)
        _write_owner_only(day_root / "paper-price-limit-evidence.v1.json", evidence)
    print(json.dumps(result, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
