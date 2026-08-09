"""The preregistration loader must not let the trial count grow.

The deflated Sharpe charges a result for the size of the search behind it, so
the loader's job is as much about what it refuses as what it returns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.mine_quant_branch_factors import (
    MiningCandidate,
    fundamental_candidates,
    preregistered_candidates,
)
from scripts.preregister_factor_candidates import build_record

SEALED = Path("results/factor_governance/candidate_preregistration.json")


def _write(tmp_path: Path, record: dict) -> Path:
    path = tmp_path / "prereg.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False))
    return path


def test_loads_only_primaries_one_per_family(tmp_path: Path) -> None:
    path = _write(tmp_path, build_record(as_of="2026-08-07"))
    loaded = preregistered_candidates(path)

    assert len(loaded) == 5
    assert all(isinstance(item, MiningCandidate) for item in loaded)
    families = [item.family for item in loaded]
    assert len(set(families)) == len(families)
    assert set(families) == {
        "value",
        "size",
        "quality",
        "leverage",
        "earnings_quality",
    }


def test_alternates_are_not_extra_trials(tmp_path: Path) -> None:
    record = build_record(as_of="2026-08-07")
    alternates = [row for row in record["candidates"] if row["role"] != "primary"]
    assert alternates, "fixture should carry alternates for this to mean anything"

    loaded = preregistered_candidates(_write(tmp_path, record))

    assert len(loaded) == record["trial_accounting"]["declared_trial_count"]
    names = {item.name for item in loaded}
    assert not names & {row["candidate_id"] for row in alternates}


def test_generates_no_smoothing_variants(tmp_path: Path) -> None:
    """fundamental_candidates expands each field into base/20d/60d; this must not."""

    generated = fundamental_candidates()
    assert any(item.name.endswith("_20d") for item in generated)

    loaded = preregistered_candidates(_write(tmp_path, build_record(as_of="2026-08-07")))

    assert not any(item.name.endswith(("_20d", "_60d")) for item in loaded)
    assert len({item.expression for item in loaded}) == len(loaded)


def test_rejects_a_record_whose_trial_count_was_edited(tmp_path: Path) -> None:
    record = build_record(as_of="2026-08-07")
    record["candidates"] = list(record["candidates"]) + [
        {
            "candidate_id": "snuck_in",
            "family": "value",
            "role": "primary",
            "expression": "cs_rank(1.0 / pe)",
            "hypothesis": "added after sealing",
            "rationale": "",
            "inputs": ["pe"],
        }
    ]

    with pytest.raises(ValueError, match="declared_trial_count"):
        preregistered_candidates(_write(tmp_path, record))


def test_rejects_a_record_with_no_primaries(tmp_path: Path) -> None:
    record = build_record(as_of="2026-08-07")
    for row in record["candidates"]:
        row["role"] = "alternate_for:nothing"

    with pytest.raises(ValueError, match="no primary candidates"):
        preregistered_candidates(_write(tmp_path, record))


@pytest.mark.skipif(not SEALED.exists(), reason="sealed record not published")
def test_sealed_record_still_hashes_to_its_recorded_digest() -> None:
    """The record on disk must not have drifted since it was sealed."""

    payload = json.loads(SEALED.read_text())
    stored = payload.pop("record_sha256")
    recomputed = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()

    assert recomputed == stored
    assert len(preregistered_candidates(SEALED)) == payload["trial_accounting"][
        "declared_trial_count"
    ]
