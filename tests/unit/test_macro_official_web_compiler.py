from __future__ import annotations

import calendar
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pytest

import quant_investor.macro.official_web_compiler as compiler
from quant_investor.macro.contracts import MacroObservation
from quant_investor.macro.official_web_compiler import (
    NBS_NATIONAL_ECONOMY_PARSER,
    NBS_OFFICIAL_PMI_PARSER,
    NBS_QUARTERLY_GDP_PARSER,
    OFFICIAL_WEB_CAPTURE_SCHEMA,
    OFFICIAL_WEB_PLAN_SCHEMA,
    PARSER_CONTRACT_SHA256,
    PBC_MONEY_STOCK_PARSER,
    OfficialWebCompilerError,
    compile_official_web_bundle_file,
    recompile_official_web_bundle,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _month_end(period: str) -> str:
    year, month = int(period[:4]), int(period[4:])
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def _quarter_end(period: str) -> str:
    year, quarter = int(period[:4]), int(period[-1])
    month = quarter * 3
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def _html(title: str, pubdate: str, paragraphs: list[str]) -> bytes:
    body = "".join(f"<p>{item}</p>" for item in paragraphs)
    return (
        "<!doctype html><html><head>"
        f'<meta name="ArticleTitle" content="{title}">'
        f'<meta name="PubDate" content="{pubdate}">'
        f"<title>{title} - 国家统计局</title>"
        f"</head><body>{body}</body></html>"
    ).encode("utf-8")


def _economy_html(period: str, pubdate: str, offset: int) -> bytes:
    month = int(period[4:])
    cumulative = "上半年" if month == 6 else f"1—{month}月份"
    industrial = DecimalText(4 + offset / 10)
    retail = DecimalText(1 + offset / 10)
    fai = DecimalText(2 + offset / 10)
    property_value = DecimalText(10 + offset / 10)
    exports = DecimalText(8 + offset / 10)
    imports = DecimalText(6 + offset / 10)
    cpi = DecimalText(1 + offset / 10)
    ppi = DecimalText(2 + offset / 10)
    paragraphs = [
        f"{month}月份，全国规模以上工业增加值同比增长{industrial}%.",
        f"{month}月份，社会消费品零售总额100亿元，同比增长{retail}%.",
        (
            f"{cumulative}，全国固定资产投资（不含农户）100亿元，"
            f"同比下降{fai}%。分领域看，房地产开发投资下降{property_value}%。"
        ),
        (
            f"{month}月份，货物进出口总额100亿元，同比增长1%。"
            f"其中，出口60亿元，增长{exports}%；进口40亿元，增长{imports}%。"
        ),
        f"{month}月份，全国居民消费价格（CPI）同比上涨{cpi}%。",
        f"{month}月份，全国工业生产者出厂价格同比上涨{ppi}%。",
    ]
    title = f"{month}月份国民经济运行总体平稳"
    if month == 6:
        title = "上半年经济运行在合理区间"
        paragraphs.insert(
            0,
            (
                "初步核算，上半年国内生产总值100亿元，按不变价格计算，"
                "同比增长4.7%。分季度看，一季度国内生产总值同比增长5.0%，"
                "二季度增长4.3%。从环比看，二季度国内生产总值增长0.9%。"
            ),
        )
    return _html(
        title,
        pubdate,
        paragraphs,
    )


class DecimalText:
    def __init__(self, value: float) -> None:
        self.value = value

    def __str__(self) -> str:
        return f"{self.value:.1f}"


def _pmi_html(period: str, pubdate: str, value: str) -> bytes:
    year, month = int(period[:4]), int(period[4:])
    title = f"{year}年{month}月中国采购经理指数运行情况"
    return _html(
        title,
        pubdate,
        [f"{month}月份，制造业采购经理指数（PMI）为{value}%。"],
    )


def _gdp_q4_html() -> bytes:
    return _html(
        "2025年经济发展向新向优全年目标实现",
        "2026/01/19 10:00",
        [
            "全年国内生产总值同比增长5.0%。分季度看，"
            "一季度国内生产总值同比增长5.4%，二季度增长5.2%，"
            "三季度增长4.8%，四季度增长4.5%。从环比看，四季度增长1.2%。"
        ],
    )


def _gdp_q1_html() -> bytes:
    return _html(
        "一季度国民经济实现良好开局",
        "2026/04/16 10:00",
        [
            "一季度国内生产总值334193亿元，按不变价格计算，"
            "同比增长5.0%，比上年四季度加快0.5个百分点。"
        ],
    )


def _pbc_html(period: str, release: str, m1: str, m2: str, cumulative: str) -> bytes:
    year, month = int(period[:4]), int(period[4:])
    title_period = "一季度" if month == 3 else f"{month}月"
    title = f"{year}年{title_period}金融统计数据报告"
    pubdate = release.split(" ", 1)[0]
    return _html(
        title,
        pubdate,
        [
            release,
            f"初步统计，社会融资规模增量累计为{cumulative}万亿元。",
            (
                f"{month}月末，广义货币（M2）余额100万亿元，同比增长{m2}%。"
                f"狭义货币（M1）余额50万亿元，同比增长{m1}%。"
            ),
        ],
    )


def _requested_scope() -> list[dict[str, str]]:
    monthly_nbs = [
        "cn.industrial_value_added_yoy",
        "cn.retail_sales_yoy",
        "cn.fixed_asset_investment_yoy",
        "cn.property_investment_yoy",
        "cn.exports_yoy",
        "cn.imports_yoy",
        "cn.cpi_yoy",
        "cn.ppi_yoy",
        "cn.pmi_manufacturing",
    ]
    rows = [
        {"indicator_id": indicator_id, "period_end": _month_end(period)}
        for indicator_id in monthly_nbs
        for period in ("202604", "202605", "202606")
    ]
    rows.extend(
        {
            "indicator_id": "cn.gdp_yoy",
            "period_end": _quarter_end(period),
        }
        for period in ("2025Q4", "2026Q1", "2026Q2")
    )
    rows.extend(
        {"indicator_id": indicator_id, "period_end": _month_end(period)}
        for indicator_id in ("cn.m1_yoy", "cn.m2_yoy")
        for period in ("202603", "202604", "202605")
    )
    return rows


def _page(
    page_id: str,
    parser_id: str,
    source_system: str,
    source_url: str,
    period: str,
) -> dict[str, str]:
    return {
        "page_id": page_id,
        "parser_id": parser_id,
        "parser_contract_sha256": PARSER_CONTRACT_SHA256[parser_id],
        "source_system": source_system,
        "source_url": source_url,
        "expected_period": period,
    }


def _fixture_pages() -> list[tuple[dict[str, str], str, bytes]]:
    rows: list[tuple[dict[str, str], str, bytes]] = []
    for period, release, offset in (
        ("202604", "2026/05/18 10:00", 1),
        ("202605", "2026/06/16 10:00", 2),
        ("202606", "2026/07/15 10:00", 3),
    ):
        record_date = release[:10].replace("/", "")
        page_id = f"economy-{period}"
        url = (
            "https://www.stats.gov.cn/sj/zxfbhjd/" f"{record_date[:6]}/t{record_date}_{offset}.html"
        )
        rows.append(
            (
                _page(
                    page_id,
                    NBS_NATIONAL_ECONOMY_PARSER,
                    "nbs_official",
                    url,
                    period,
                ),
                f"nbs/{page_id}.html",
                _economy_html(period, release, offset),
            )
        )
    for period, release, value in (
        ("202604", "2026/04/30 09:30", "50.3"),
        ("202605", "2026/05/31 09:30", "50.0"),
        ("202606", "2026/06/30 09:30", "50.3"),
    ):
        record_date = release[:10].replace("/", "")
        page_id = f"pmi-{period}"
        url = "https://www.stats.gov.cn/sj/zxfb/" f"{record_date[:6]}/t{record_date}_9.html"
        rows.append(
            (
                _page(
                    page_id,
                    NBS_OFFICIAL_PMI_PARSER,
                    "nbs_official",
                    url,
                    period,
                ),
                f"nbs/{page_id}.html",
                _pmi_html(period, release, value),
            )
        )
    for page_id, period, release, body in (
        ("gdp-2025q4", "2025Q4", "20260119", _gdp_q4_html()),
        ("gdp-2026q1", "2026Q1", "20260416", _gdp_q1_html()),
    ):
        url = "https://www.stats.gov.cn/sj/zxfbhjd/" f"{release[:6]}/t{release}_8.html"
        rows.append(
            (
                _page(
                    page_id,
                    NBS_QUARTERLY_GDP_PARSER,
                    "nbs_official",
                    url,
                    period,
                ),
                f"nbs/{page_id}.html",
                body,
            )
        )
    pbc_values = (
        ("202602", "2026-03-13 17:00:01", "5.9", "9.0", "9.60"),
        ("202603", "2026-04-13 17:00:00", "5.1", "8.5", "14.83"),
        ("202604", "2026-05-14 17:00:02", "5.0", "8.6", "15.45"),
        ("202605", "2026-06-12 17:00:01", "5.5", "8.6", "17.48"),
    )
    for period, release, m1, m2, cumulative in pbc_values:
        record_id = release[:10].replace("-", "") + "14270000000"
        page_id = f"pbc-{period}"
        url = "https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/" f"{record_id}/index.html"
        rows.append(
            (
                _page(
                    page_id,
                    PBC_MONEY_STOCK_PARSER,
                    "pbc_official",
                    url,
                    period,
                ),
                f"pbc/{page_id}.html",
                _pbc_html(period, release, m1, m2, cumulative),
            )
        )
    return rows


def _seal_inputs(
    root: Path,
    plan: dict[str, Any],
    page_files: list[tuple[dict[str, str], str, bytes]],
    *,
    capture_mutator: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    raw_root = root / "raw"
    raw_root.mkdir(parents=True)
    for _page_row, relative, body in page_files:
        path = raw_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    plan_path = root / "plan.json"
    plan_bytes = _json_bytes(plan)
    plan_path.write_bytes(plan_bytes)
    captures = []
    for page_row, relative, body in page_files:
        captures.append(
            {
                "page_id": page_row["page_id"],
                "source_system": page_row["source_system"],
                "source_url": page_row["source_url"],
                "effective_url": page_row["source_url"],
                "raw_path": relative,
                "raw_sha256": _sha(body),
                "size_bytes": len(body),
                "fetch_started_at": "2026-07-16T12:00:00+08:00",
                "fetch_completed_at": "2026-07-16T12:00:01+08:00",
                "content_type": "text/html",
                "charset": "utf-8",
                "redirect_chain": [page_row["source_url"]],
            }
        )
    capture = {
        "schema_version": OFFICIAL_WEB_CAPTURE_SCHEMA,
        "market": "CN",
        "plan_sha256": _sha(plan_bytes),
        "pages": captures,
    }
    if capture_mutator is not None:
        capture_mutator(capture)
    capture_path = root / "capture.json"
    capture_path.write_bytes(_json_bytes(capture))
    return plan_path, capture_path, raw_root, plan, capture


def _bundle(
    tmp_path: Path,
    *,
    plan_mutator: Callable[[dict[str, Any]], None] | None = None,
    capture_mutator: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Path, Path, Path, dict[str, Any], dict[str, Any]]:
    page_files = _fixture_pages()
    plan = {
        "schema_version": OFFICIAL_WEB_PLAN_SCHEMA,
        "market": "CN",
        "requested_scope": _requested_scope(),
        "pages": [item[0] for item in page_files],
    }
    if plan_mutator is not None:
        plan_mutator(plan)
    return _seal_inputs(
        tmp_path,
        plan,
        page_files,
        capture_mutator=capture_mutator,
    )


def test_compile_and_recompile_exact_36_official_observations(tmp_path: Path) -> None:
    plan_path, capture_path, raw_root, _plan, _capture = _bundle(tmp_path)
    report = compile_official_web_bundle_file(
        plan_path,
        capture_manifest_path=capture_path,
        raw_root=raw_root,
        output_root=tmp_path / "output",
        run_id="official-20260716",
    )

    assert report["status"] == "OK"
    assert report["observation_count"] == 36
    assert report["receipt_count"] == 36
    assert report["national_registry_coverage"] == 0.75
    assert report["social_financing_flow_emitted"] is False
    assert (
        report["unsupported_indicator_reasons"]["cn.social_financing_flow"]
        == "rounded_cumulative_difference_not_exact"
    )

    artifacts = report["artifacts"]
    observations = [
        MacroObservation.from_mapping(json.loads(line))
        for line in Path(artifacts["observations"]).read_text().splitlines()
    ]
    assert len(observations) == 36
    assert Counter(item.indicator_id for item in observations) == Counter(
        {indicator_id: 3 for indicator_id in report["supported_indicator_ids"]}
    )
    assert all(item.indicator_id != "cn.social_financing_flow" for item in observations)
    assert {
        item.source_system
        for item in observations
        if item.indicator_id in {"cn.m1_yoy", "cn.m2_yoy"}
    } == {"pboc_official"}
    assert [item.value for item in observations if item.indicator_id == "cn.gdp_yoy"] == [
        4.5,
        5.0,
        4.3,
    ]

    bundle_path = Path(artifacts["bundle"])
    assert bundle_path.stat().st_mode & 0o777 == 0o700
    for path in bundle_path.rglob("*"):
        expected_mode = 0o700 if path.is_dir() else 0o600
        assert path.stat().st_mode & 0o777 == expected_mode
    for page_row, _relative, original in _fixture_pages():
        assert (bundle_path / "raw" / f"{page_row['page_id']}.html").read_bytes() == original

    replay = recompile_official_web_bundle(
        artifacts["manifest"],
        expected_manifest_sha256=report["normalization_manifest_sha256"],
        expected_plan_sha256=report["plan_file_sha256"],
    )
    assert replay.observations == tuple(observations)
    assert replay.manifest == {
        key: value
        for key, value in json.loads(Path(artifacts["manifest"]).read_text()).items()
        if key not in {"artifact_sha256", "raw_artifacts"}
    }


def test_compile_rejects_raw_hash_mismatch_before_output(tmp_path: Path) -> None:
    def mutate(capture: dict[str, Any]) -> None:
        capture["pages"][0]["raw_sha256"] = "0" * 64

    plan_path, capture_path, raw_root, _plan, _capture = _bundle(tmp_path, capture_mutator=mutate)
    output = tmp_path / "output"
    with pytest.raises(OfficialWebCompilerError, match="raw_hash_mismatch"):
        compile_official_web_bundle_file(
            plan_path,
            capture_manifest_path=capture_path,
            raw_root=raw_root,
            output_root=output,
            run_id="rejected",
        )
    assert not (output / "CN" / "rejected").exists()


def test_compile_rejects_wrong_parser_contract(tmp_path: Path) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        plan["pages"][0]["parser_contract_sha256"] = "0" * 64

    plan_path, capture_path, raw_root, _plan, _capture = _bundle(tmp_path, plan_mutator=mutate)
    with pytest.raises(OfficialWebCompilerError, match="parser_contract_mismatch"):
        compile_official_web_bundle_file(
            plan_path,
            capture_manifest_path=capture_path,
            raw_root=raw_root,
            output_root=tmp_path / "output",
            run_id="rejected",
        )


def test_compile_rejects_wrong_issuer(tmp_path: Path) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        page = next(item for item in plan["pages"] if item["page_id"] == "pbc-202605")
        page["source_url"] = "https://www.stats.gov.cn/not-pbc/t20260612_1.html"

    plan_path, capture_path, raw_root, _plan, _capture = _bundle(tmp_path, plan_mutator=mutate)
    with pytest.raises(OfficialWebCompilerError, match="source_url_issuer_mismatch"):
        compile_official_web_bundle_file(
            plan_path,
            capture_manifest_path=capture_path,
            raw_root=raw_root,
            output_root=tmp_path / "output",
            run_id="rejected",
        )


def test_compile_rejects_date_only_pbc_without_visible_exact_time(tmp_path: Path) -> None:
    pages = _fixture_pages()
    for index, (page, relative, body) in enumerate(pages):
        if page["page_id"] == "pbc-202605":
            pages[index] = (
                page,
                relative,
                body.replace(b"2026-06-12 17:00:01", b"2026-06-12"),
            )
    plan = {
        "schema_version": OFFICIAL_WEB_PLAN_SCHEMA,
        "market": "CN",
        "requested_scope": _requested_scope(),
        "pages": [item[0] for item in pages],
    }
    plan_path, capture_path, raw_root, _plan, _capture = _seal_inputs(tmp_path, plan, pages)
    with pytest.raises(OfficialWebCompilerError, match="pubdate_exact_time_missing"):
        compile_official_web_bundle_file(
            plan_path,
            capture_manifest_path=capture_path,
            raw_root=raw_root,
            output_root=tmp_path / "output",
            run_id="rejected",
        )


def test_compile_rejects_raw_path_traversal(tmp_path: Path) -> None:
    def mutate(capture: dict[str, Any]) -> None:
        capture["pages"][0]["raw_path"] = "../escape.html"

    plan_path, capture_path, raw_root, _plan, _capture = _bundle(tmp_path, capture_mutator=mutate)
    with pytest.raises(OfficialWebCompilerError, match="raw_path_unsafe"):
        compile_official_web_bundle_file(
            plan_path,
            capture_manifest_path=capture_path,
            raw_root=raw_root,
            output_root=tmp_path / "output",
            run_id="rejected",
        )


def test_compile_rejects_symlinked_raw_file(tmp_path: Path) -> None:
    plan_path, capture_path, raw_root, _plan, capture = _bundle(tmp_path)
    row = capture["pages"][0]
    path = raw_root / row["raw_path"]
    target = tmp_path / "outside.html"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(OfficialWebCompilerError, match="symlink_rejected"):
        compile_official_web_bundle_file(
            plan_path,
            capture_manifest_path=capture_path,
            raw_root=raw_root,
            output_root=tmp_path / "output",
            run_id="rejected",
        )


def test_stable_reader_detects_in_place_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "large.html"
    path.write_bytes(b"a" * (128 * 1024))
    original_read = os.read
    mutated = False

    def mutating_read(descriptor: int, count: int) -> bytes:
        nonlocal mutated
        payload = original_read(descriptor, count)
        if payload and not mutated:
            mutated = True
            with path.open("ab") as handle:
                handle.write(b"changed")
        return payload

    monkeypatch.setattr(compiler.os, "read", mutating_read)
    with pytest.raises(OfficialWebCompilerError, match="toctou_detected"):
        compiler._stable_file_bytes(
            path,
            error_prefix="official_web_test",
            max_bytes=256 * 1024,
        )


def test_recompile_rejects_plan_cas_mismatch_and_artifact_tamper(tmp_path: Path) -> None:
    plan_path, capture_path, raw_root, _plan, _capture = _bundle(tmp_path)
    report = compile_official_web_bundle_file(
        plan_path,
        capture_manifest_path=capture_path,
        raw_root=raw_root,
        output_root=tmp_path / "output",
        run_id="official-20260716",
    )
    artifacts = report["artifacts"]
    with pytest.raises(OfficialWebCompilerError, match="plan_hash_mismatch"):
        recompile_official_web_bundle(
            artifacts["manifest"],
            expected_manifest_sha256=report["normalization_manifest_sha256"],
            expected_plan_sha256="0" * 64,
        )

    observations_path = Path(artifacts["observations"])
    observations_path.write_bytes(observations_path.read_bytes() + b"\n")
    with pytest.raises(OfficialWebCompilerError, match="artifact_hash_mismatch"):
        recompile_official_web_bundle(
            artifacts["manifest"],
            expected_manifest_sha256=report["normalization_manifest_sha256"],
            expected_plan_sha256=report["plan_file_sha256"],
        )


def test_compile_rejects_incomplete_36_row_scope(tmp_path: Path) -> None:
    def mutate(plan: dict[str, Any]) -> None:
        plan["requested_scope"].pop()

    plan_path, capture_path, raw_root, _plan, _capture = _bundle(tmp_path, plan_mutator=mutate)
    with pytest.raises(OfficialWebCompilerError, match="requested_scope_count_invalid"):
        compile_official_web_bundle_file(
            plan_path,
            capture_manifest_path=capture_path,
            raw_root=raw_root,
            output_root=tmp_path / "output",
            run_id="rejected",
        )
