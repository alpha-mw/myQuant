"""One immutable day claim prevents new request subtrees resetting acquisition scope."""

from copy import deepcopy
import pytest
from quant_investor.operations.theme_acquisition import reserve_theme_acquisition, IDENTITY_REFS
from quant_investor.operations.daily_journal import DailyJournal
from quant_investor.operations.daily_contract import ContractError


def identity():
    return {
        **{key: {"path": key + ".json", "sha256": "a" * 64} for key in IDENTITY_REFS},
        "company_set_sha256": "b" * 64,
    }


def test_claim_requires_lock_and_reuses_original_bytes(tmp_path):
    journal = DailyJournal(str(tmp_path), "20260908")
    with pytest.raises(ContractError, match="LOCK_REQUIRED"):
        reserve_theme_acquisition(journal=journal, identity=identity())
    with journal.locked():
        first = reserve_theme_acquisition(journal=journal, identity=identity())
    path = tmp_path / first["path"]
    before = path.read_bytes(), path.stat().st_mtime_ns
    with journal.locked():
        assert reserve_theme_acquisition(journal=journal, identity=identity()) == first
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("field", sorted(IDENTITY_REFS | {"company_set_sha256"}))
def test_competing_identity_cannot_replace_day_claim(tmp_path, field):
    journal = DailyJournal(str(tmp_path), "20260908")
    original = identity()
    with journal.locked():
        first = reserve_theme_acquisition(journal=journal, identity=original)
    changed = deepcopy(original)
    if field == "company_set_sha256":
        changed[field] = "c" * 64
    else:
        changed[field]["sha256"] = "c" * 64
    before = (tmp_path / first["path"]).read_bytes()
    with journal.locked():
        with pytest.raises(ContractError, match="ALREADY_CLAIMED"):
            reserve_theme_acquisition(journal=journal, identity=changed)
    assert (tmp_path / first["path"]).read_bytes() == before
