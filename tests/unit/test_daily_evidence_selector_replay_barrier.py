from scripts.cn_dashboard_v2_selector import publish_selector
from quant_investor.operations.dashboard_replay_sources import (
    RetainedDashboardSources,
    retained_dashboard_sources,
)
import pytest


def test_selector_cannot_publish_inside_historical_read_context(tmp_path):
    with retained_dashboard_sources(RetainedDashboardSources(tmp_path, {})):
        with pytest.raises(ValueError, match="DASHBOARD_REPLAY_CANNOT_PUBLISH"):
            publish_selector(
                {},
                json_path=tmp_path / "selector.json",
                js_path=tmp_path / "selector.js",
                project_root=tmp_path,
                js_first=False,
            )
    assert not list(tmp_path.iterdir())
