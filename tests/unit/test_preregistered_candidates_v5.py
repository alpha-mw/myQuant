from __future__ import annotations

import json

from scripts.preregister_factor_candidates_v5 import candidate_catalog


def test_catalog_has_one_alternate_at_most_and_no_generated_variants():
    rows = candidate_catalog()
    primaries = {row["candidate_id"] for row in rows if row["role"] == "PRIMARY"}
    alternates = [row for row in rows if row["role"].startswith("ALTERNATE_FOR:")]
    targets = [row["role"].split(":", 1)[1] for row in alternates]
    assert set(targets) <= primaries
    assert len(targets) == len(set(targets))
    assert not any(row["candidate_id"].endswith(("_20d", "_60d")) for row in rows)


def test_catalog_is_json_safe_and_deterministic():
    first = json.dumps(candidate_catalog(), sort_keys=True, separators=(",", ":"))
    second = json.dumps(candidate_catalog(), sort_keys=True, separators=(",", ":"))
    assert first == second
