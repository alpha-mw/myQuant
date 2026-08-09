"""Test-collection policy for the owner-machine-bound evidence archive.

The Factor Governance evidence archive verifies its seals against the exact
machine that published them. See docs/architecture/evidence_archive_boundary.md.

Three of its bindings cannot hold anywhere else:

* ``AST_RUNTIME_FINGERPRINT`` pins the absolute path of the publishing
  interpreter, that binary's SHA-256, and its Clang build string.
* ``governance_private_bundle_io`` publishes through ``renameatx_np`` with
  ``RENAME_EXCL``, a Darwin-only syscall, and fails closed elsewhere.
* Source identity is read with ``git cat-file blob`` against blobs that exist
  only in the owner's local object store.

These are properties of the sealed archive, not defects to repair - repairing
them would mean editing byte-pinned files. So the modules below are collected
on the owner's machine, where the seals are meaningful, and skipped under CI,
where they can only ever fail. Everything else - the entire runtime layer - is
unaffected and must stay green in CI.
"""

from __future__ import annotations

import os

import pytest

OWNER_MACHINE_BOUND_MODULES = frozenset(
    {
        "test_build_factor_v4_3_candidate_preregistration",
        "test_build_factor_v4_3_prior_diagnostic_nomination",
        "test_build_factor_v4_4_candidate_preregistration",
        "test_build_factor_v4_4_future_strict_signal_computability",
        "test_build_factor_v4_4_signal_computability",
        "test_factor_governance_candidate_preregistration_bundle_v4_3",
        "test_factor_governance_candidate_preregistration_bundle_v4_4",
        "test_factor_governance_candidate_preregistration_v4_3",
        "test_factor_governance_candidate_preregistration_v4_4",
        "test_factor_governance_discovery_v4_1",
        "test_factor_governance_future_strict_exact_five_eval_v4_4",
        "test_factor_governance_future_strict_signal_computability_v4_4",
        "test_factor_governance_no_label_diagnostic_v4_1",
        "test_factor_governance_prior_diagnostic_nomination_bundle_v4_3",
        "test_factor_governance_prior_diagnostic_nomination_v4_3",
        "test_factor_governance_same_snapshot_screening_v4_1",
    }
)

_SKIP_REASON = (
    "evidence archive is bound to the publishing machine "
    "(interpreter identity, Darwin renameatx_np, local git blobs); "
    "see docs/architecture/evidence_archive_boundary.md"
)


def _archive_tests_are_runnable() -> bool:
    """Owner machine only. ``RUN_EVIDENCE_ARCHIVE_TESTS`` forces either answer."""

    override = os.environ.get("RUN_EVIDENCE_ARCHIVE_TESTS")
    if override is not None:
        return override == "1"
    return not os.environ.get("CI")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if _archive_tests_are_runnable():
        return
    skip = pytest.mark.skip(reason=_SKIP_REASON)
    for item in items:
        if item.path.stem in OWNER_MACHINE_BOUND_MODULES:
            item.add_marker(skip)
