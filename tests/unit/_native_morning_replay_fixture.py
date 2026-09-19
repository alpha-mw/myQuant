"""Installed EOD->Morning research replay with explicitly synthetic Sina transport."""

import quant_investor
from pathlib import Path
from datetime import datetime, timezone
from argparse import Namespace
import hashlib
import json
import sys


def run(root: Path, dependencies: Path, *, trade_date="20260827", completion_ref=None):
    receipt = json.loads((root / "fixture-receipt.json").read_text())
    if (
        Path(quant_investor.__file__).resolve()
        != Path(receipt["runtime_verification"]["import_origin"]).resolve()
    ):
        raise ValueError("installed runtime required")
    repository = root / "repository"
    sys.path.extend(
        [
            str(repository),
            str(repository / "scripts"),
            str(repository / "tests/unit"),
            str(dependencies),
        ]
    )
    import pandas as pd
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.operations.completion_readback import inspect_recorded_completion
    from quant_investor.operations.daily_journal import FALSE_AUTHORITY
    from quant_investor.market.next_session_proof import read_next_session_proof
    from scripts.daily_morning_consumer import prepare_morning_consumer
    from capture_sina_cn_quotes import run as capture_quotes
    from _verify_five_native_days import inventory, readonly_replay_guard

    workspace = root / "factor-workspace"
    if completion_ref is None:
        completion_ref = json.loads((root / "full-completion-ref.json").read_text())
    completed = inspect_recorded_completion(
        workspace=str(workspace), trade_date=trade_date, completion_ref=completion_ref
    )["recorded_completion"]
    calendar = json.loads(
        (workspace / completed["node_terminal_refs"]["calendar"]["path"]).read_text()
    )
    future = read_next_session_proof(
        workspace=str(workspace),
        eod_trade_date=trade_date,
        publication_ref=calendar["output_refs"]["next_session_calendar_proof"],
    )
    day = future["proof"]["next_open_session"]
    store = json.loads((workspace / completed["node_terminal_refs"]["store"]["path"]).read_text())
    holdings = pd.read_parquet(workspace / store["output_refs"]["ledger"]["path"])
    symbols = sorted(holdings.loc[holdings["shares"] > 0, "symbol"].astype(str).tolist())

    def put(relative, value):
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = canonical_json_bytes(value)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("fixture input conflict")
        path.write_bytes(raw)
        path.chmod(0o600)
        return {"path": relative, "sha256": hashlib.sha256(raw).hexdigest()}

    policy = put(
        "fixtures/morning-owner-policy.json",
        {
            "schema_version": "morning-quote-policy.v1",
            "strategy_id": "aggressive_tech_manufacturing",
            "market": "CN",
            "effective_from": day,
            "effective_through": day,
            "revoked_at": None,
            "additional_symbols": [],
            "authority": FALSE_AUTHORITY,
        },
    )
    quote_request = put(
        "fixtures/morning-quote-request.json", {"run_date": day, "symbols": symbols}
    )
    iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    lines = []
    for symbol in symbols:
        fields = [
            "Synthetic",
            "12",
            "11.9",
            "12",
            "12.1",
            "11.8",
            "12",
            "12",
            "100",
            "1200",
            *(["0"] * 20),
            iso,
            "09:45:00",
            "00",
        ]
        lines.append(f'var hq_str_{symbol[-2:].lower()}{symbol[:6]}="{",".join(fields)}";')
    raw = ("\n".join(lines) + "\n").encode("gb18030")
    moments = iter(
        [
            datetime.fromisoformat(iso + "T01:45:00+00:00"),
            datetime.fromisoformat(iso + "T01:45:01+00:00"),
        ]
    )
    result = capture_quotes(
        Namespace(
            allow_live=True,
            workspace_root=str(workspace),
            request_path=str(workspace / quote_request["path"]),
            request_sha256=quote_request["sha256"],
            output_root=str(workspace / f"data/private/cn_public_quotes/{day}/sina-0945"),
        ),
        fetcher=lambda url: raw,
        now=lambda: next(moments),
    )
    capture = json.loads((workspace / result["capture_path"]).read_text())
    request = {
        "schema_version": "morning-strategy-request.v2",
        "action": "REPLAY",
        "run_date": day,
        "previous_completion_ref": completion_ref,
        "quote_capture_ref": {"path": result["capture_path"], "sha256": result["capture_sha256"]},
        "quote_raw_ref": {key: capture["raw_ref"][key] for key in ("path", "sha256")},
        "owner_policy_ref": policy,
        "output_ref": None,
    }
    request_ref = put("fixtures/morning-replay-request.json", request)
    before = inventory(workspace)

    print("START installed Morning native replay", flush=True)
    with readonly_replay_guard():
        replay = prepare_morning_consumer(workspace=str(workspace), request=request)
    assert inventory(workspace) == before
    assert replay["synthetic"] is True and replay["admission"] == "RESEARCH_ONLY"
    (root / "native-morning-replay-result.json").write_text(json.dumps(replay, indent=2) + "\n")
    (root / "native-morning-replay-proof.json").write_text(
        json.dumps(
            {
                "synthetic": True,
                "eod_trade_date": trade_date,
                "live_provider_calls": 0,
                "quote_clock_simulated": True,
                "consumer_clock_simulated": False,
                "request_ref": request_ref,
                "command_status": replay["command_status"],
                "unchanged_inventory": True,
                "entries": len(before),
                "data_writers_forbidden": True,
                "existing_native_factor_read_lock_allowed": True,
                "live_morning_success": False,
            },
            indent=2,
        )
        + "\n"
    )
    print("PASS installed EOD->Morning research replay; no writes", flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]))
