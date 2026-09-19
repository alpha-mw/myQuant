"""Native Macro transaction on a marked shared synthetic Market/PIT workspace."""

from datetime import datetime, timezone, timedelta
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from quant_investor.macro import readiness_closure as readiness
from quant_investor.macro import maintenance_transaction as transaction
from quant_investor.macro.release_calendar import publish_release_calendar
from quant_investor.macro.production_observation_bundle import publish_local_market_breadth_roll


def build_shared_macro(
    workspace: Path,
    target: str,
    *,
    transaction_identity: str | None = None,
    prepare_only: bool = False,
    calendar_through: str | None = None,
) -> dict:
    from tests.unit.test_macro_release_calendar import _write_fixture
    from tests.unit.test_macro_production_observation_bundle import _inputs, _publish

    marker = json.loads((workspace / "SYNTHETIC-DATA-CLOCK.json").read_text())
    if marker.get("synthetic") is not True or marker.get("real_time_oos_eligible") is not False:
        raise ValueError("explicit synthetic clock workspace required")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    release_root = workspace / readiness.RELEASE_POINTER.parent
    obs_root = workspace / readiness.OBSERVATIONS_POINTER.parent
    present = [(base / "_latest.json").exists() for base in (release_root, obs_root)]
    if present[0] != present[1]:
        raise ValueError("partial Macro parents require native recovery")
    prior_release = (
        sha(release_root / "_latest.json") if present[0] else transaction.EMPTY_POINTER_SHA256
    )
    prior_observations = (
        sha(obs_root / "_latest.json") if present[1] else transaction.EMPTY_POINTER_SHA256
    )
    compact = target.replace("-", "")
    raw_root = workspace / "synthetic-macro" / compact
    calendar = _write_fixture(raw_root / "calendar")
    open_dates = [
        stamp.strftime("%Y%m%d")
        for stamp in pd.bdate_range("2026-04-20", calendar_through or target)
    ]
    calendar.open_days["open_dates"] = open_dates
    calendar.open_days_path.write_text(json.dumps(calendar.open_days, sort_keys=True))
    calendar.open_days_path.chmod(0o600)
    if present[0]:
        shutil.copytree(release_root, calendar.canonical_root)
    publish_release_calendar(
        **calendar.kwargs(run_id="calendar-" + compact, expected_pointer_sha256=prior_release)
    )
    source = raw_root / "observations-source"
    if present[1]:
        source.mkdir(parents=True, mode=0o700)
        shutil.copytree(obs_root, source / "observations")
        initial = {"pointer_sha256": prior_observations}
    else:
        inputs = _inputs(source)
        from tests.unit.test_macro_official_web_compiler import (
            _fixture_pages,
            _requested_scope,
            _page,
            _pmi_html,
            _pbc_html,
            _seal_inputs,
        )
        from quant_investor.macro.official_web_compiler import (
            PBC_MONEY_STOCK_PARSER,
            PBC_MONEY_STOCK_PARSER_V2,
            NBS_OFFICIAL_PMI_PARSER,
            PARSER_CONTRACT_SHA256,
            compile_official_web_bundle_file,
        )

        pages = _fixture_pages()
        for index, (page, relative, body) in enumerate(pages):
            if page["parser_id"] == PBC_MONEY_STOCK_PARSER:
                page = {
                    **page,
                    "parser_id": PBC_MONEY_STOCK_PARSER_V2,
                    "parser_contract_sha256": PARSER_CONTRACT_SHA256[PBC_MONEY_STOCK_PARSER_V2],
                }
                pages[index] = (page, relative, body)
        pages.extend(
            [
                (
                    _page(
                        "synthetic-pmi-july",
                        NBS_OFFICIAL_PMI_PARSER,
                        "nbs_official",
                        "https://www.stats.gov.cn/sj/zxfb/202608/t20260801_123.html",
                        "202607",
                    ),
                    "pmi-july.html",
                    _pmi_html("202607", "2026/08/01 09:30", "50.2"),
                ),
                (
                    _page(
                        "synthetic-money-july",
                        PBC_MONEY_STOCK_PARSER_V2,
                        "pbc_official",
                        "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/20260815/index.html",
                        "202607",
                    ),
                    "money-july.html",
                    _pbc_html("202607", "2026-08-15 15:00:00", "4.2", "8.2", "23.0"),
                ),
            ]
        )
        requested = _requested_scope() + [
            {"indicator_id": key, "period_end": "2026-07-31"}
            for key in ("cn.pmi_manufacturing", "cn.m1_yoy", "cn.m2_yoy")
        ]

        for indicator in ("cn.pmi_manufacturing", "cn.m1_yoy", "cn.m2_yoy"):
            oldest = min(row["period_end"] for row in requested if row["indicator_id"] == indicator)
            requested = [
                row
                for row in requested
                if not (row["indicator_id"] == indicator and row["period_end"] == oldest)
            ]

        pages.append(
            (
                _page(
                    "synthetic-money-june",
                    PBC_MONEY_STOCK_PARSER_V2,
                    "pbc_official",
                    "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/20260715/index.html",
                    "202606",
                ),
                "money-june.html",
                _pbc_html(
                    "202606", "2026-07-15 15:00:09", "4.0", "8.0", "22.83", half_year_title=True
                ),
            )
        )
        requested = [
            row for row in requested if row["indicator_id"] not in {"cn.m1_yoy", "cn.m2_yoy"}
        ]
        requested.extend(
            {"indicator_id": indicator, "period_end": period}
            for indicator in ("cn.m1_yoy", "cn.m2_yoy")
            for period in ("2026-05-31", "2026-06-30", "2026-07-31")
        )

        pages = [
            row
            for row in pages
            if not (
                row[0]["parser_id"] == NBS_OFFICIAL_PMI_PARSER
                and row[0]["expected_period"] == "202604"
            )
            and not (
                row[0]["parser_id"] == PBC_MONEY_STOCK_PARSER_V2
                and row[0]["expected_period"] in {"202602", "202603"}
            )
        ]

        def capture_times(capture):
            for page in capture["pages"]:
                page["fetch_started_at"] = "2026-08-20T12:00:00+08:00"
                page["fetch_completed_at"] = "2026-08-20T12:00:01+08:00"

        plan, capture, raw, _, _ = _seal_inputs(
            raw_root / "fresh-official",
            {
                "schema_version": "macro-official-web-plan.v1",
                "market": "CN",
                "requested_scope": requested,
                "pages": [row[0] for row in pages],
            },
            pages,
            capture_mutator=capture_times,
        )
        official = compile_official_web_bundle_file(
            plan,
            capture_manifest_path=capture,
            raw_root=raw,
            output_root=raw_root / "fresh-official-bundle",
            run_id="fresh-official-" + compact,
        )
        inputs.update(
            official_bundle_manifest_path=official["artifacts"]["manifest"],
            expected_official_bundle_manifest_sha256=official["normalization_manifest_sha256"],
            expected_official_plan_sha256=official["plan_file_sha256"],
        )
        scope = workspace / "data/cn_universe/cn_index_components.json"

        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        earlier = [day for day in open_dates if day < compact][-3:]
        targets = []
        for day in earlier:
            manifest = workspace / "data/parquet/cn/_snapshots" / (day + "T073000Z.json")
            targets.append(
                {
                    "target_trade_date": day,
                    "snapshot_manifest_path": str(manifest),
                    "expected_snapshot_manifest_sha256": sha(manifest),
                    "coverage_manifest_path": str(manifest),
                    "expected_coverage_manifest_sha256": sha(manifest),
                    "scope_artifact_path": str(scope),
                    "expected_scope_artifact_sha256": sha(scope),
                }
            )
        local_plan = raw_root / "shared-bootstrap-plan.json"
        local_plan.write_text(
            json.dumps(
                {
                    "schema_version": "cn-local-breadth-bootstrap-plan.v1",
                    "market": "CN",
                    "targets": targets,
                },
                sort_keys=True,
            )
        )
        local_plan.chmod(0o600)
        inputs.update(
            local_bootstrap_plan_path=local_plan,
            expected_local_bootstrap_plan_sha256=sha(local_plan),
        )
        initial = _publish(
            source, inputs, as_of=target + "T13:00:00Z", run_id="bootstrap-" + compact
        )
    market_path = workspace / readiness.MARKET_POINTER
    market = json.loads(market_path.read_text())
    manifest = Path(market["manifest_path"])
    scope = workspace / "data/cn_universe/cn_index_components.json"

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    publish_local_market_breadth_roll(
        snapshot_manifest_path=manifest,
        expected_snapshot_manifest_sha256=sha(manifest),
        coverage_manifest_path=manifest,
        expected_coverage_manifest_sha256=sha(manifest),
        target_trade_date=compact,
        scope_artifact_path=scope,
        expected_scope_artifact_sha256=sha(scope),
        target_as_of=compact,
        decision_cutoff_at=target + "T13:00:00Z",
        pinned_open_dates=open_dates,
        market_open_days_path=calendar.open_days_path,
        expected_market_open_days_sha256=sha(calendar.open_days_path),
        canonical_observations_root=source / "observations",
        run_id="shared-local-" + compact,
        expected_pointer_sha256=initial["pointer_sha256"],
    )
    release_root = workspace / readiness.RELEASE_POINTER.parent
    obs_root = workspace / readiness.OBSERVATIONS_POINTER.parent
    for base in (release_root, obs_root):
        base.mkdir(parents=True, mode=0o700, exist_ok=True)
        (base / "_generations").mkdir(mode=0o700, exist_ok=True)
    identity = transaction_identity or "shared-macro-" + compact
    transaction_parent = workspace / readiness.TRANSACTION_ROOT
    transaction_parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    base = transaction_parent / identity
    prepared = (
        base / "prepared" if transaction_identity else base / "_prepared" / identity / "prepared"
    )
    prepared.mkdir(parents=True, mode=0o700)
    pit = workspace / readiness.PIT_POINTER
    args = {
        "market_pointer_path": market_path,
        "expected_market_pointer_sha256": sha(market_path),
        "pit_pointer_path": pit,
        "expected_pit_pointer_sha256": sha(pit),
    }
    veto_path = workspace / readiness.VETO_PATH
    veto_ref = {"path": str(veto_path), "sha256": sha(veto_path)} if veto_path.exists() else None
    sealed = transaction.seal_prepared_macro_transaction(
        prepared_root=prepared,
        release_candidate_root=calendar.canonical_root,
        observations_candidate_root=source / "observations",
        release_canonical_root=release_root,
        observations_canonical_root=obs_root,
        expected_release_pointer_sha256=prior_release,
        expected_observations_pointer_sha256=prior_observations,
        authority_mode="canonical",
        input_bindings={"macro_veto": veto_ref} if veto_ref is not None else {},
        target_date=compact,
        **args,
    )

    if prepare_only:
        return {
            "synthetic": True,
            "prepared": sealed,
            "authority_args": args,
            "real_time_oos_eligible": False,
            "committed": False,
        }

    (base / "journals").mkdir(mode=0o700)

    class LogicalClock(datetime):
        tick = 0

        @classmethod
        def now(cls, tz=None):
            cls.tick += 1
            value = datetime.fromisoformat(target + "T13:10:00+00:00") + timedelta(seconds=cls.tick)
            return value.astimezone(tz) if tz else value.replace(tzinfo=None)

    with patch.object(transaction, "datetime", LogicalClock):
        transaction.commit_prepared_macro_transaction(
            prepared_path=sealed["prepared_path"],
            expected_prepared_sha256=sealed["prepared_sha256"],
            journal_root=base / "journals",
            journal_run_id=identity,
            **args,
        )
    if veto_ref is not None:
        from quant_investor.market import daily_maintenance

        with patch.object(daily_maintenance, "datetime", LogicalClock):
            daily_maintenance.clear_cn_daily_write_veto(
                run_root=veto_path.parent,
                expected_veto_sha256=veto_ref["sha256"],
                reason="Synthetic native Macro transaction and postcheck completed",
                lane="macro",
            )
    terminal = base / "journals" / identity / "0007-terminal.json"
    closure = readiness.build_macro_readiness_closure(
        workspace_root=workspace,
        terminal_path=str(terminal.relative_to(workspace)),
        terminal_sha256=sha(terminal),
    )
    verified = readiness.verify_current_macro_readiness_closure(
        workspace_root=workspace,
        closure=closure,
        expected_target_date=compact,
        decision_as_of=target + "T13:30:00Z",
    )
    sealed_readiness = readiness.seal_macro_readiness_closure(
        workspace_root=workspace,
        terminal_path=str(terminal.relative_to(workspace)),
        terminal_sha256=sha(terminal),
    )
    return {
        "closure_ref": {
            "path": sealed_readiness["closure_path"],
            "sha256": sealed_readiness["closure_sha256"],
        },
        "synthetic": True,
        "logical_clock_simulated": True,
        "real_time_oos_eligible": False,
        "wall_clock_captured_at": datetime.now(timezone.utc).isoformat(),
        "closure": closure,
        "verification": verified,
    }
