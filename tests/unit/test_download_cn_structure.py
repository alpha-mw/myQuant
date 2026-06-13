from __future__ import annotations

from quant_investor.market.download_cn import CNFullMarketDownloader
from quant_investor.market.download_cn_freshness import CNDownloadFreshnessMixin


def test_cn_downloader_uses_freshness_mixin_boundary() -> None:
    assert issubclass(CNFullMarketDownloader, CNDownloadFreshnessMixin)
    assert (
        CNFullMarketDownloader._build_completeness_report_for_target
        is CNDownloadFreshnessMixin._build_completeness_report_for_target
    )
    assert CNFullMarketDownloader.build_completeness_report is CNDownloadFreshnessMixin.build_completeness_report
    assert CNFullMarketDownloader._load_freshness_index is CNDownloadFreshnessMixin._load_freshness_index
