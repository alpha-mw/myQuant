#!/usr/bin/env python3
"""Capture the sealed SW2021 membership keyset, one exact partition at a time."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Callable

from quant_investor.intelligence_v2._core import canonical_bytes
from quant_investor.intelligence_v2.sources.tushare import (
    TushareContractError,
    build_industry_membership_capture,
    capture_industry_membership_partition,
    validate_industry_membership_execution_plan,
    validate_industry_membership_capture,
    validate_industry_membership_partition_capture,
    validate_industry_taxonomy_capture,
    validate_industry_taxonomy_execution_plan,
)
from quant_investor.v17_v4_runtime.tushare_https import OfficialTushareHttpsClient
from scripts.probe_tushare_10000_capabilities import (
    ProbeSafetyError,
    _unique_object,
    _validate_new_output_root,
    _write_exact,
)


def _load_exact(path: Path, expected_sha256: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ProbeSafetyError("MEMBERSHIP_INPUT_PATH_INVALID")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ProbeSafetyError("MEMBERSHIP_INPUT_SHA_MISMATCH")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda _: (_ for _ in ()).throw(
                ProbeSafetyError("MEMBERSHIP_INPUT_NONFINITE")
            ),
        )
    except ProbeSafetyError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProbeSafetyError("MEMBERSHIP_INPUT_INVALID") from exc
    if type(value) is not dict or canonical_bytes(value) != raw:
        raise ProbeSafetyError("MEMBERSHIP_INPUT_NOT_CANONICAL")
    return value


def _load_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    taxonomy_plan = validate_industry_taxonomy_execution_plan(
        _load_exact(Path(args.taxonomy_plan_path), args.taxonomy_plan_sha256)
    )
    taxonomy_capture = validate_industry_taxonomy_capture(
        _load_exact(Path(args.taxonomy_capture_path), args.taxonomy_capture_sha256),
        plan=taxonomy_plan,
    )
    membership_plan = validate_industry_membership_execution_plan(
        _load_exact(Path(args.membership_plan_path), args.membership_plan_sha256),
        taxonomy_plan=taxonomy_plan,
        taxonomy_capture=taxonomy_capture,
    )
    return taxonomy_plan, taxonomy_capture, membership_plan


def _validate_resume_root(path: Path) -> None:
    observed = path.lstat()
    if (
        not path.is_absolute()
        or not stat.S_ISDIR(observed.st_mode)
        or stat.S_ISLNK(observed.st_mode)
        or stat.S_IMODE(observed.st_mode) != 0o700
    ):
        raise ProbeSafetyError("MEMBERSHIP_RESUME_ROOT_INVALID")
    partitions = path / "partitions"
    part_stat = partitions.lstat()
    if (
        not stat.S_ISDIR(part_stat.st_mode)
        or stat.S_ISLNK(part_stat.st_mode)
        or stat.S_IMODE(part_stat.st_mode) != 0o700
    ):
        raise ProbeSafetyError("MEMBERSHIP_RESUME_PARTITIONS_INVALID")


def _prepare_root(
    output_root: Path,
    *,
    resume: bool,
    inputs: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> Path:
    taxonomy_plan, taxonomy_capture, membership_plan = inputs
    if resume:
        _validate_resume_root(output_root)
        expected = {
            "membership_plan.json": membership_plan,
            "taxonomy_capture.json": taxonomy_capture,
            "taxonomy_plan.json": taxonomy_plan,
        }
        for name, document in expected.items():
            path = output_root / name
            if path.is_symlink() or path.read_bytes() != canonical_bytes(document):
                raise ProbeSafetyError("MEMBERSHIP_RESUME_INPUT_MISMATCH")
        return output_root / "partitions"
    _validate_new_output_root(output_root, create=True)
    partitions = output_root / "partitions"
    partitions.mkdir(mode=0o700)
    os.chmod(partitions, 0o700)
    _write_exact(output_root / "taxonomy_plan.json", taxonomy_plan)
    _write_exact(output_root / "taxonomy_capture.json", taxonomy_capture)
    _write_exact(output_root / "membership_plan.json", membership_plan)
    return partitions


def _load_partition(
    path: Path,
    *,
    validator: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ProbeSafetyError("MEMBERSHIP_PARTITION_PATH_INVALID")
    observed = path.lstat()
    if observed.st_nlink != 1 or stat.S_IMODE(observed.st_mode) != 0o600:
        raise ProbeSafetyError("MEMBERSHIP_PARTITION_MODE_INVALID")
    raw = path.read_bytes()
    try:
        document = json.loads(raw, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProbeSafetyError("MEMBERSHIP_PARTITION_INVALID") from exc
    validated = validator(document)
    if canonical_bytes(validated) != raw:
        raise ProbeSafetyError("MEMBERSHIP_PARTITION_NOT_CANONICAL")
    return validated


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run(
    args: argparse.Namespace,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    taxonomy_plan, taxonomy_capture, membership_plan = _load_inputs(args)
    output_root = Path(args.output_root)
    if not args.allow_live:
        _validate_new_output_root(output_root, create=False)
        return {
            "completed_partitions": 0,
            "membership_plan_id": membership_plan["membership_plan_id"],
            "network_attempts": 0,
            "planned_partitions": membership_plan["endpoint_plan"][
                "planned_terminal_request_count"
            ],
            "status": "DRY_RUN_VALIDATED",
        }
    partitions_root = _prepare_root(
        output_root,
        resume=bool(args.resume),
        inputs=(taxonomy_plan, taxonomy_capture, membership_plan),
    )
    transport = client or OfficialTushareHttpsClient(strict_decimal_decode=True)
    keyset = membership_plan["endpoint_plan"]["ordered_expected_partition_keyset"]
    network_attempts = 0
    partition_documents: list[dict[str, Any]] = []

    def validate_partition(document: dict[str, Any]) -> dict[str, Any]:
        return validate_industry_membership_partition_capture(
            document,
            membership_plan=membership_plan,
            taxonomy_plan=taxonomy_plan,
            taxonomy_capture=taxonomy_capture,
        )

    for ordinal, partition_key in enumerate(keyset):
        path = partitions_root / f"{ordinal:04d}.json"
        if path.exists():
            partition = _load_partition(path, validator=validate_partition)
            if (
                partition["partition_ordinal"] != ordinal
                or partition["partition_key"] != partition_key
            ):
                raise ProbeSafetyError("MEMBERSHIP_RESUME_KEYSET_MISMATCH")
            partition_documents.append(partition)
            continue
        partition = capture_industry_membership_partition(
            membership_plan=membership_plan,
            taxonomy_plan=taxonomy_plan,
            taxonomy_capture=taxonomy_capture,
            partition_ordinal=ordinal,
            captured_at=_now(),
            client=transport,
        )
        network_attempts += 1
        _write_exact(path, partition)
        partition_documents.append(partition)
        if (ordinal + 1) % 25 == 0:
            print(f"membership progress {ordinal + 1}/{len(keyset)}", file=sys.stderr, flush=True)
    capture_path = output_root / "capture.json"
    if capture_path.exists():
        capture = _load_partition(
            capture_path,
            validator=lambda document: validate_industry_membership_capture(
                document,
                membership_plan=membership_plan,
                taxonomy_plan=taxonomy_plan,
                taxonomy_capture=taxonomy_capture,
                partition_documents=partition_documents,
            ),
        )
    else:
        capture = build_industry_membership_capture(
            membership_plan=membership_plan,
            taxonomy_plan=taxonomy_plan,
            taxonomy_capture=taxonomy_capture,
            partition_documents=partition_documents,
            completed_at=_now(),
        )
        _write_exact(capture_path, capture)
    summary = {
        "capture_id": capture["capture_id"],
        "completed_partitions": len(keyset),
        "membership_plan_id": membership_plan["membership_plan_id"],
        "network_attempts": network_attempts,
        "planned_partitions": len(keyset),
        "status": "COMPLETE",
    }
    if not (output_root / "summary.json").exists():
        _write_exact(output_root / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--taxonomy-plan-path", required=True)
    parser.add_argument("--taxonomy-plan-sha256", required=True)
    parser.add_argument("--taxonomy-capture-path", required=True)
    parser.add_argument("--taxonomy-capture-sha256", required=True)
    parser.add_argument("--membership-plan-path", required=True)
    parser.add_argument("--membership-plan-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def main() -> int:
    try:
        summary = run(parse_args())
    except (ProbeSafetyError, TushareContractError) as exc:
        print(json.dumps({"blocker": str(exc), "status": "MEMBERSHIP_CAPTURE_BLOCKED"}))
        return 2
    except Exception:
        print(json.dumps({"status": "MEMBERSHIP_CAPTURE_BLOCKED"}))
        return 2
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
