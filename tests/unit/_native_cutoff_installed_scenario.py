"""Installed native Factor/Core plus cutoff profile; only data/transport/clocks are synthetic."""

from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch


def focus_theme_source(workspace, day):
    from quant_investor.contracts import canonical_json_bytes
    from quant_investor.factors.production_pit import FOCUS_COMPANIES
    from quant_investor.intelligence.theme_governance import TECHNOLOGY_THEME_IDS
    from quant_investor.market.tushare import (
        build_theme_provider_execution_plan,
        build_theme_partition_capture,
        build_theme_provider_capture,
    )

    prefix = Path("synthetic-cutoff-focus") / day
    iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    stamp = iso + "T13:10:00Z"
    theme = next(
        value.split(":", 1)[1] for value in TECHNOLOGY_THEME_IDS if value.startswith("TUSHARE_DC:")
    )
    plan = build_theme_provider_execution_plan(
        provider="TUSHARE_DC",
        trade_date=day,
        company_keyset=list(FOCUS_COMPANIES),
        document_observed_at=stamp,
        created_at=stamp,
    )
    parts = [
        build_theme_partition_capture(
            plan=plan,
            partition_ordinal=0,
            provider_request_id="synthetic-focus-registry",
            reported_count=1,
            rows=[
                {
                    "idx_type": "概念板块",
                    "level": "1",
                    "name": "synthetic focus",
                    "trade_date": day,
                    "ts_code": theme,
                }
            ],
            blocker_codes=[],
            captured_at=stamp,
        )
    ]
    for index, company in enumerate(FOCUS_COMPANIES, 1):
        parts.append(
            build_theme_partition_capture(
                plan=plan,
                partition_ordinal=index,
                provider_request_id="synthetic-focus-" + company,
                reported_count=1,
                rows=[
                    {"con_code": company, "name": "synthetic", "trade_date": day, "ts_code": theme}
                ],
                blocker_codes=[],
                captured_at=stamp,
            )
        )
    capture = build_theme_provider_capture(plan=plan, partition_documents=parts, completed_at=stamp)

    def emit(name, value):
        raw = canonical_json_bytes(value)
        path = workspace / prefix / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_bytes(raw)
        path.chmod(0o600)
        return {"path": str(path.relative_to(workspace)), "sha256": hashlib.sha256(raw).hexdigest()}

    return {
        "dc_plan": emit("plan.json", plan),
        "dc_capture": emit("capture.json", capture),
        "dc_partitions": [emit(f"part-{index}.json", part) for index, part in enumerate(parts)],
        "tdx_plan": None,
        "tdx_capture": None,
        "tdx_partitions": [],
    }


def run(root, dependency_path):
    import quant_investor

    receipt = json.loads((root / "fixture-receipt.json").read_bytes())
    if (
        Path(quant_investor.__file__).resolve()
        != Path(receipt["runtime_verification"]["import_origin"]).resolve()
    ):
        raise ValueError("exact installed native package required")
    repository = root / "repository"
    sys.path.extend(
        [
            str(repository),
            str(repository / "scripts"),
            str(repository / "tests/unit"),
            str(dependency_path),
        ]
    )
    import _native_full_dag_scenario as scenario

    state = {"real": False}
    real_datetime = datetime

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            if state["real"]:
                return real_datetime.now(tz)
            return real_datetime(2026, 8, 27, 14, tzinfo=timezone.utc)

    names = [
        "quant_investor.cli.unified",
        "quant_investor.factors.production_authority",
        "quant_investor.factors.production_observation",
        "quant_investor.operations.daily_journal",
        "quant_investor.operations.maintenance_handoff",
        "quant_investor.operations.portfolio_binding",
        "quant_investor.operations.research_timing",
        "quant_investor.operations.research_cutoff",
        "scripts.daily_materialization",
        "scripts.cn_official_close_batch",
        "cn_official_close_batch",
    ]
    with ExitStack() as stack:
        for name in names:
            owner = importlib.import_module(name)
            stack.enter_context(patch.object(owner, "datetime", Clock))
        result = scenario.run(
            root,
            dependency_path,
            cutoff_profile=True,
            switch_to_actual_clock=lambda: state.update(real=True),
        )
    result.update(
        production_deployed=False,
        real_provider_calls=False,
        factor_and_core_mocked=False,
        release_verifier_mocked=False,
        data_and_core_cutoff_clock_simulated=True,
    )
    (root / "native-cutoff-installed-proof.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    print(json.dumps(run(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
