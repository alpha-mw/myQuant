"""Public read-only daily status routing and invalid-date boundary."""

import json
import pytest
from quant_investor.cli.main import main


def command(root):
    return [
        "production",
        "daily-status",
        "--workspace-root",
        str(root),
        "--market",
        "CN",
        "--strategy",
        "aggressive_tech_manufacturing",
        "--trade-date",
        "20260825",
    ]


def test_public_status_has_no_filesystem_writes(tmp_path, capsys):
    main(command(tmp_path))
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "NOT_STARTED"
    assert output["completion_ref"] is None
    assert output["validation_scope"] == "JOURNAL_AND_OUTPUT_BYTES"
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("day", ["20260230", "2026-08-25", "2026825"])
def test_public_status_rejects_noncanonical_day(tmp_path, capsys, day):
    args = command(tmp_path)
    args[-1] = day
    with pytest.raises(SystemExit):
        main(args)
    assert list(tmp_path.iterdir()) == []
