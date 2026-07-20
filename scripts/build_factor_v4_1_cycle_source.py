#!/usr/bin/env python3
"""Build one offline, private v4.1 cutoff source bundle.

All market inputs are explicit absolute paths with caller-supplied raw hashes.
The runner is research-only and has no registry, replay, proposal, provider,
portfolio, broker, order, trade, or network interface.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import quant_investor.factors.governance_cycle_state_v4_1 as cycle_state  # noqa: E402
import quant_investor.factors.governance_source_readback_v4_1 as readback  # noqa: E402
import quant_investor.factors.governance_source_v4_1 as source  # noqa: E402


class FactorV4_1CycleSourceRunnerError(ValueError):
    """Raised when the bounded runner cannot produce fail-closed evidence."""


def _source_binding_sha256() -> str:
    return readback.source_code_binding_sha256_v4_1(
        [
            Path(source.__file__).resolve(strict=True),
            Path(cycle_state.__file__).resolve(strict=True),
            Path(readback.__file__).resolve(strict=True),
            Path(__file__).resolve(strict=True),
        ]
    )


def _attempted_inputs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "snapshot_id": args.snapshot_id,
        "analysis_start": args.analysis_start,
        "cutoff_date": args.cutoff_date,
        "expected_full_a_count": args.expected_full_a_count,
        "expected_serving_inventory_count": (
            args.expected_serving_inventory_count
        ),
        "latest_pointer": {
            "absolute_path": args.latest_pointer_path,
            "expected_sha256": args.expected_latest_pointer_sha256,
        },
        "snapshot_manifest": {
            "absolute_path": args.snapshot_manifest_path,
            "expected_sha256": args.expected_snapshot_manifest_sha256,
        },
        "components": {
            "absolute_path": args.components_path,
            "expected_sha256": args.expected_components_sha256,
            "expected_full_a_semantic_sha256": (
                args.expected_full_a_semantic_sha256
            ),
        },
        "pit_generation_manifest": {
            "absolute_path": args.pit_generation_manifest_path,
            "expected_sha256": args.expected_pit_generation_manifest_sha256,
        },
        "pit_membership": {
            "absolute_path": args.pit_membership_path,
            "expected_sha256": args.expected_pit_membership_sha256,
        },
        "table_root": {"absolute_path": args.table_root},
    }


def _bind(args: argparse.Namespace) -> readback.BoundCutoffInputsV4_1:
    return readback.bind_explicit_cutoff_inputs_v4_1(
        latest_pointer_path=args.latest_pointer_path,
        expected_latest_pointer_sha256=args.expected_latest_pointer_sha256,
        snapshot_manifest_path=args.snapshot_manifest_path,
        expected_snapshot_manifest_sha256=(
            args.expected_snapshot_manifest_sha256
        ),
        components_path=args.components_path,
        expected_components_sha256=args.expected_components_sha256,
        expected_full_a_semantic_sha256=(
            args.expected_full_a_semantic_sha256
        ),
        pit_generation_manifest_path=args.pit_generation_manifest_path,
        expected_pit_generation_manifest_sha256=(
            args.expected_pit_generation_manifest_sha256
        ),
        pit_membership_path=args.pit_membership_path,
        expected_pit_membership_sha256=args.expected_pit_membership_sha256,
        table_root=args.table_root,
        snapshot_id=args.snapshot_id,
        analysis_start=args.analysis_start,
        cutoff_date=args.cutoff_date,
        expected_full_a_count=args.expected_full_a_count,
        expected_serving_inventory_count=(
            args.expected_serving_inventory_count
        ),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Bind inputs, run the pure source contract, and publish private evidence."""

    attempted = _attempted_inputs(args)
    try:
        bound = _bind(args)
    except (ValueError, RuntimeError, OSError) as exc:
        result = readback.publish_input_binding_failure_v4_1(
            private_root=args.private_root,
            run_id=args.run_id,
            cycle_id=args.cycle_id,
            attempted_inputs=attempted,
            blocker_code="INPUT_BINDING_REJECTED",
            blocker_detail=str(exc),
            expected_state_sha256=args.expected_state_sha256,
        )
        return {
            **result,
            "snapshot_id": args.snapshot_id,
            "analysis_window": {
                "start": args.analysis_start,
                "cutoff": args.cutoff_date,
            },
            "qualification": False,
        }

    input_binding_sha = readback.binding_semantic_sha256_v4_1(bound.binding)
    try:
        source_binding_sha = _source_binding_sha256()
    except (ValueError, RuntimeError, OSError) as exc:
        result = readback.publish_blocked_cutoff_readback_v4_1(
            private_root=args.private_root,
            run_id=args.run_id,
            cycle_id=args.cycle_id,
            input_binding=bound.binding,
            blocker_code="SOURCE_BINDING_REJECTED",
            blocker_detail=str(exc),
            expected_state_sha256=args.expected_state_sha256,
        )
        return {
            **result,
            "snapshot_id": args.snapshot_id,
            "analysis_window": {
                "start": args.analysis_start,
                "cutoff": args.cutoff_date,
                "open_session_count": len(bound.calendar_sessions),
            },
            "input_binding_semantic_sha256": input_binding_sha,
            "qualification": False,
        }
    try:
        normalized_pit = source.validate_pit_records_v4_1(
            list(bound.pit_records)
        )
        design = source.build_design_source_node_v4_1(
            cycle_id=args.cycle_id,
            pit_records=normalized_pit,
            component_symbols=list(bound.component_symbols),
            calendar_sessions=list(bound.calendar_sessions),
            market_binding_sha256=input_binding_sha,
            source_binding_sha256=source_binding_sha,
            expected_component_count=args.expected_full_a_count,
        )
        source_node = readback.build_cutoff_source_node_v4_1(
            cycle_id=args.cycle_id,
            input_binding=bound.binding,
            design_source=design,
            source_binding_sha256=source_binding_sha,
        )
        cycle_root_sha = readback.cycle_root_semantic_sha256_v4_1(
            cycle_id=args.cycle_id,
            input_binding=bound.binding,
            design_source=design,
        )
        precommitted_state = cycle_state.build_genesis_cycle_state_v4_1(
            cycle_id=args.cycle_id,
            cycle_root_sha256=cycle_root_sha,
            source_chain_node_sha256=source_node["semantic_sha256"],
        )
    except (ValueError, RuntimeError) as exc:
        result = readback.publish_blocked_cutoff_readback_v4_1(
            private_root=args.private_root,
            run_id=args.run_id,
            cycle_id=args.cycle_id,
            input_binding=bound.binding,
            blocker_code="SOURCE_CONTRACT_REJECTED",
            blocker_detail=str(exc),
            expected_state_sha256=args.expected_state_sha256,
        )
        return {
            **result,
            "snapshot_id": args.snapshot_id,
            "analysis_window": {
                "start": args.analysis_start,
                "cutoff": args.cutoff_date,
                "open_session_count": len(bound.calendar_sessions),
            },
            "input_binding_semantic_sha256": input_binding_sha,
            "source_binding_sha256": source_binding_sha,
            "qualification": False,
        }

    result = readback.publish_precommitted_cutoff_source_v4_1(
        private_root=args.private_root,
        run_id=args.run_id,
        cycle_id=args.cycle_id,
        input_binding=bound.binding,
        design_source=design,
        source_chain_node=source_node,
        precommitted_cycle_state=precommitted_state,
        pit_records=bound.pit_records,
        expected_component_count=args.expected_full_a_count,
        expected_source_binding_sha256=source_binding_sha,
        expected_state_sha256=args.expected_state_sha256,
    )
    return {
        **result,
        "snapshot_id": args.snapshot_id,
        "analysis_window": {
            "start": args.analysis_start,
            "cutoff": args.cutoff_date,
            "open_session_count": len(bound.calendar_sessions),
        },
        "coverage": {
            "universe": "full_a",
            "component_count": len(bound.component_symbols),
            "component_semantic_sha256": bound.binding["components"][
                "newline_set_sha256"
            ],
            "pit_record_count": len(bound.pit_records),
            "bound_table_symbol_count": len(
                bound.bound_table_symbol_row_counts
            ),
        },
        "input_binding_semantic_sha256": input_binding_sha,
        "source_binding_sha256": source_binding_sha,
        "cycle_root_semantic_sha256": cycle_root_sha,
        "design_source_semantic_sha256": design["semantic_sha256"],
        "source_chain_node_semantic_sha256": source_node["semantic_sha256"],
        "out_of_bound_calendar_nonparticipating": source_node[
            "out_of_bound_calendar_nonparticipating"
        ],
        "blockers": [
            "holdout_not_appended",
            "statistics_not_run",
            "verified_v4_replay_not_run",
            "qualification_not_evaluated",
        ],
        "qualification": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a private offline v4.1 cutoff source bundle"
    )
    parser.add_argument("--latest-pointer-path", required=True)
    parser.add_argument("--expected-latest-pointer-sha256", required=True)
    parser.add_argument("--snapshot-manifest-path", required=True)
    parser.add_argument("--expected-snapshot-manifest-sha256", required=True)
    parser.add_argument("--components-path", required=True)
    parser.add_argument("--expected-components-sha256", required=True)
    parser.add_argument("--expected-full-a-semantic-sha256", required=True)
    parser.add_argument("--pit-generation-manifest-path", required=True)
    parser.add_argument(
        "--expected-pit-generation-manifest-sha256", required=True
    )
    parser.add_argument("--pit-membership-path", required=True)
    parser.add_argument("--expected-pit-membership-sha256", required=True)
    parser.add_argument("--table-root", required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--analysis-start", required=True)
    parser.add_argument("--cutoff-date", required=True)
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cycle-id", required=True)
    parser.add_argument("--expected-state-sha256", required=True)
    parser.add_argument(
        "--expected-full-a-count",
        type=int,
        default=readback.EXPECTED_FULL_A_COUNT,
    )
    parser.add_argument(
        "--expected-serving-inventory-count",
        type=int,
        default=readback.EXPECTED_SERVING_INVENTORY_COUNT,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "readiness": "BLOCKED_FAIL_CLOSED",
                    "error": str(exc),
                    "qualification": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result.get("readiness") == "EXPLORATORY_PRECOMMITTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
