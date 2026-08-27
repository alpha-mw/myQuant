from __future__ import annotations

from argparse import Namespace
import hashlib
import json
from pathlib import Path

import quant_investor

from quant_investor.contracts import canonical_json_bytes
from quant_investor.intelligence._common import artifact_ref, build_artifact, business_identity
from quant_investor.intelligence.storage import approved_theme_policy_v2
from quant_investor.market.tushare._core import canonical_bytes
from scripts.operations.run_cn_theme_replay import run
from tests.unit.tushare_response_fixtures import make_tushare_response

ROOT = Path(__file__).resolve().parents[2]
REPLAY_SCRIPT = ROOT / "scripts/operations/run_cn_theme_replay.py"


class Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    def request(self, *, api_name: str, params: dict, expected_fields: tuple):
        company = params.get("con_code")
        self.calls.append((api_name, company))
        if api_name == "dc_index":
            rows = [("BK1001.DC", "20260827", "机器人", "概念板块", "1")]
        elif api_name == "dc_member" and company == "000001.SZ":
            rows = [("20260827", "BK1001.DC", company, "平安银行")]
        elif api_name == "dc_member" and company == "000002.SZ":
            rows = [("20260827", "BK1001.DC", "999999.SZ", "错误主体")]
        elif api_name == "tdx_index":
            rows = [("880001.TDX", "20260827", "机器人", "概念板块", "1")]
        elif api_name == "tdx_member":
            rows = [("880001.TDX", "20260827", company, "万科A")]
        else:
            rows = []
        return make_tushare_response(
            api_name=api_name,
            request_id=f"request-{len(self.calls)}",
            reported_count=len(rows),
            has_more=False,
            fields=tuple(expected_fields),
            rows=tuple(rows),
        )


def _write(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.parent.chmod(0o700)
    raw = canonical_bytes(value)
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest()


def _inputs(tmp_path: Path) -> Namespace:
    policy = approved_theme_policy_v2()
    policy_path = tmp_path / "results/policies/research/aggressive_tech_manufacturing/v2.json"
    policy_sha = _write(policy_path, policy)
    symbols = ["000001.SZ", "000002.SZ"]
    selection = build_artifact(
        kind="daily_research_selected_symbols",
        identity_field="selection_id",
        identity=business_identity(
            kind="daily_research_selected_symbols",
            identity_inputs={"test": "cn-theme-replay"},
        ),
        created_at="2026-08-27T11:59:00Z",
        fields={
            "ordered_symbols": symbols,
            "rank_ref": artifact_ref(policy),
            "signal_date": "20260827",
            "strategy_id": "aggressive_tech_manufacturing",
            "symbol_count": len(symbols),
            "symbol_set_sha256": hashlib.sha256(canonical_json_bytes(symbols)).hexdigest(),
        },
    )
    selected_path = tmp_path / "results/intelligence/pool/selected_symbols.json"
    selected_sha = _write(selected_path, selection)
    return Namespace(
        allow_live=True,
        expected_import_root=str(Path(quant_investor.__file__).resolve().parent),
        expected_policy_sha256=policy_sha,
        expected_selected_symbols_sha256=selected_sha,
        policy=policy_path.relative_to(tmp_path).as_posix(),
        selected_symbols=selected_path.relative_to(tmp_path).as_posix(),
        workspace_root=str(tmp_path),
    )


def test_daily_theme_replay_uses_dc_then_only_registered_tdx_fallback(tmp_path: Path) -> None:
    client = Client()
    result = run(
        _inputs(tmp_path),
        client=client,
        now=lambda: "2026-08-27T12:00:00Z",
    )

    assert result["command_status"] == "COMPLETE"
    assert result["fallback_company_count"] == 1
    assert result["token_material_recorded"] is False
    assert result["token_hash_recorded"] is False
    assert client.calls == [
        ("dc_index", None),
        ("dc_member", "000001.SZ"),
        ("dc_member", "000002.SZ"),
        ("tdx_index", None),
        ("tdx_member", "000002.SZ"),
    ]
    receipt = json.loads((tmp_path / result["receipt_path"]).read_bytes())
    assert receipt["fallback_company_keyset"] == ["000002.SZ"]
    assert receipt["credential_source"] == "PROJECT_ENV"
    assert receipt["broker"] is False
    assert receipt["order"] is False
    assert receipt["trade"] is False


def test_daily_theme_replay_replays_without_network_calls(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    first_client = Client()
    first = run(args, client=first_client, now=lambda: "2026-08-27T12:00:00Z")
    replay_client = Client()
    second = run(args, client=replay_client, now=lambda: "2026-08-27T12:00:00Z")

    assert second == first
    assert replay_client.calls == []


def test_daily_theme_replay_accepts_existing_read_only_plan_parent(tmp_path: Path) -> None:
    args = _inputs(tmp_path)
    plans = tmp_path / "data/private/intelligence_sources/theme/plans"
    plans.mkdir(parents=True, mode=0o755)
    plans.chmod(0o755)

    result = run(args, client=Client(), now=lambda: "2026-08-27T12:00:00Z")

    assert result["command_status"] == "COMPLETE"
    assert plans.stat().st_mode & 0o022 == 0


def test_daily_theme_replay_uses_only_project_env_without_secret_logging() -> None:
    source = REPLAY_SCRIPT.read_text(encoding="utf-8")
    assert "read_project_env_token" in source
    assert 'workspace / ".env"' in source
    assert "/usr/bin/security" not in source
    assert "Keychain" not in source
    assert "source .env" not in source
    assert 'token_material_recorded": False' in source
    assert 'token_hash_recorded": False' in source
    assert "print(token" not in source
