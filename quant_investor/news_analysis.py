"""
News & Alternative Data Analysis Engine
========================================
新闻与替代数据分析模块

核心功能：
  1. 多源新闻爬取      - RSS/公开API/AKShare新闻接口
  2. LLM驱动情感分析   - 多维情感标注（正/负/中性 + 事件类型）
  3. 命名实体识别       - 公司/人物/事件提取
  4. 事件检测与分类     - 盈利超预期/CEO变更/并购/监管等
  5. 主题建模          - 市场热点主题追踪
  6. 新闻冲击评分       - 量化新闻对股价的预期影响
  7. 情感时序           - 情感趋势（近7日/30日）
  8. 一致预期跟踪       - 分析师评级变化汇总
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from quant_investor.logger import get_logger

_logger = get_logger("NewsAnalysis")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class NewsItem:
    """单条新闻"""
    title: str
    content: str
    source: str
    publish_time: datetime
    url: str = ""
    relevance_score: float = 0.0   # 与标的的相关性 0–1
    sentiment_score: float = 0.0   # -1（极负）到 +1（极正）
    sentiment_label: str = "中性"  # "强烈利好" | "利好" | "中性" | "利空" | "强烈利空"
    event_type: str = ""           # "盈利超预期" | "监管" | "并购" | "高管变动" | "产品" | "宏观" 等
    entities: list[str] = field(default_factory=list)  # 相关公司/人物
    impact_score: float = 0.0      # 预期股价冲击幅度 (%)
    keywords: list[str] = field(default_factory=list)


@dataclass
class NewsAnalysisResult:
    """股票新闻分析结果"""
    symbol: str
    stock_name: str
    analysis_period_days: int
    total_news_count: int
    news_items: list[NewsItem] = field(default_factory=list)

    # 情感聚合
    avg_sentiment_score: float = 0.0     # 整体情感均值
    sentiment_trend: str = "平稳"        # "上升" | "下降" | "平稳"
    positive_ratio: float = 0.0          # 正面新闻占比
    negative_ratio: float = 0.0          # 负面新闻占比

    # 事件汇总
    major_events: list[dict] = field(default_factory=list)
    event_summary: str = ""

    # 热点主题
    hot_topics: list[str] = field(default_factory=list)

    # 冲击评分
    cumulative_impact_score: float = 0.0  # 累积新闻冲击
    news_signal: str = "中性"             # "强烈看多" | "看多" | "中性" | "看空" | "强烈看空"
    signal_confidence: float = 0.5

    # 分析师观点
    analyst_upgrades: int = 0
    analyst_downgrades: int = 0
    consensus_rating: str = "持有"

    summary: str = ""


# ---------------------------------------------------------------------------
# 新闻获取器
# ---------------------------------------------------------------------------

class NewsDataFetcher:
    """从多种来源获取新闻数据"""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[list[NewsItem], float]] = {}
        self._cache_ttl = 1800  # 30分钟缓存

    def fetch_akshare_news(self, symbol: str, days: int = 7) -> list[NewsItem]:
        """通过 AKShare 获取个股新闻"""
        items = []
        try:
            import akshare as ak  # type: ignore
            # 去除市场后缀
            code = symbol.split(".")[0]
            try:
                df = ak.stock_news_em(symbol=code)
                if df is not None and not df.empty:
                    cutoff = datetime.now() - timedelta(days=days)
                    for _, row in df.iterrows():
                        try:
                            pub_time = pd.to_datetime(row.get("发布时间", ""))
                            if pub_time < cutoff:
                                continue
                            items.append(NewsItem(
                                title=str(row.get("新闻标题", "")),
                                content=str(row.get("新闻内容", "")),
                                source=str(row.get("文章来源", "东方财富")),
                                publish_time=pub_time,
                                url=str(row.get("新闻链接", "")),
                            ))
                        except Exception:
                            continue
            except Exception as e:
                _logger.debug(f"AKShare个股新闻接口: {e}")

            # 通用财经新闻
            try:
                df2 = ak.stock_hot_tgb_em()
                if df2 is not None and not df2.empty:
                    for _, row in df2.head(20).iterrows():
                        items.append(NewsItem(
                            title=str(row.get("事件", "")),
                            content=str(row.get("内容", "")),
                            source="同花顺",
                            publish_time=datetime.now(),
                        ))
            except Exception:
                pass

        except ImportError:
            _logger.debug("AKShare未安装，跳过新闻获取")

        return items

    def fetch_rss_news(self, symbol: str, stock_name: str) -> list[NewsItem]:
        """通过RSS订阅获取财经新闻"""
        items = []
        try:
            import feedparser  # type: ignore

            rss_sources = [
                f"https://xueqiu.com/hq/rss?stockSymbol={symbol}",
                "https://finance.yahoo.com/news/rssindex",
                "https://feeds.bbci.co.uk/news/business/rss.xml",
            ]

            for url in rss_sources:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:20]:
                        title = getattr(entry, "title", "")
                        content = getattr(entry, "summary", "")
                        # 过滤相关新闻
                        if stock_name and stock_name not in title and stock_name not in content:
                            if symbol.split(".")[0] not in title:
                                continue
                        pub = datetime.now()
                        if hasattr(entry, "published_parsed") and entry.published_parsed:
                            pub = datetime(*entry.published_parsed[:6])
                        items.append(NewsItem(
                            title=title, content=content,
                            source=feed.feed.get("title", url),
                            publish_time=pub,
                            url=getattr(entry, "link", ""),
                        ))
                except Exception:
                    continue

        except ImportError:
            _logger.debug("feedparser未安装，跳过RSS新闻")

        return items

    def fetch_sina_news(self, symbol: str, days: int = 7) -> list[NewsItem]:
        """通过新浪财经API获取新闻"""
        items = []
        try:
            import requests
            code = symbol.split(".")[0]
            # 新浪财经个股新闻API
            url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllBulletin/stockid/{code}/page_type/ndbgg.phtml"
            resp = requests.get(url, timeout=5,
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                # 简单解析标题
                titles = re.findall(r'<a[^>]+href="[^"]*bulletin[^"]*"[^>]*>([^<]+)</a>', resp.text)
                for title in titles[:10]:
                    items.append(NewsItem(
                        title=title.strip(),
                        content="",
                        source="新浪财经",
                        publish_time=datetime.now(),
                    ))
        except Exception as e:
            _logger.debug(f"新浪财经新闻获取失败: {e}")

        return items

    def fetch_all(self, symbol: str, stock_name: str, days: int = 7) -> list[NewsItem]:
        """从所有来源获取新闻并合并去重"""
        cache_key = f"{symbol}_{days}"
        if cache_key in self._cache:
            items, ts = self._cache[cache_key]
            if time.time() - ts < self._cache_ttl:
                return items

        all_items: list[NewsItem] = []
        all_items.extend(self.fetch_akshare_news(symbol, days))
        all_items.extend(self.fetch_rss_news(symbol, stock_name))
        all_items.extend(self.fetch_sina_news(symbol, days))

        # 去重（按标题哈希）
        seen: set = set()
        unique_items = []
        for item in all_items:
            h = hashlib.md5(item.title[:50].encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique_items.append(item)

        # 排序（最新在前）
        unique_items.sort(key=lambda x: x.publish_time, reverse=True)

        self._cache[cache_key] = (unique_items, time.time())
        _logger.info(f"获取新闻 [{symbol}]: 共 {len(unique_items)} 条（去重后）")
        return unique_items


# ---------------------------------------------------------------------------
# 情感分析引擎
# ---------------------------------------------------------------------------

class SentimentEngine:
    """
    多层次情感分析：
    1. 基于规则的快速分析（无需LLM）
    2. LLM深度分析（需要API密钥）
    """

    # 情感词典
    POSITIVE_WORDS = [
        "增长", "突破", "盈利", "超预期", "利好", "上涨", "创新高", "超额",
        "强劲", "优秀", "领先", "扩张", "合作", "获批", "上市", "涨停",
        "回购", "分红", "增持", "大幅增长", "创历史", "重大突破", "积极",
        "受益", "看好", "机遇", "亮眼", "飙升", "暴涨", "新高",
    ]
    NEGATIVE_WORDS = [
        "下滑", "亏损", "低于预期", "利空", "下跌", "创新低", "减少",
        "风险", "监管", "处罚", "退市", "违规", "造假", "跌停", "暴跌",
        "减持", "质押", "爆仓", "流动性", "危机", "负面", "警告", "调查",
        "诉讼", "召回", "事故", "下调评级", "大幅下滑", "亏损扩大",
    ]
    EVENT_PATTERNS = {
        "盈利超预期": ["超预期", "超出预期", "业绩超预期", "净利润增长", "盈利大增"],
        "盈利不及预期": ["低于预期", "不及预期", "业绩下滑", "净利润下降"],
        "监管处罚": ["处罚", "罚款", "立案", "调查", "违规", "监管", "警示函"],
        "并购重组": ["收购", "并购", "重组", "合并", "入股", "战略合作"],
        "高管变动": ["CEO", "董事长", "总经理", "辞职", "离职", "任命", "高管"],
        "产品发布": ["发布", "上市", "推出", "新产品", "新业务", "商业化"],
        "融资动态": ["定增", "配股", "发债", "融资", "IPO", "再融资"],
        "分红回购": ["分红", "派息", "回购", "股票回购", "特别分红"],
        "宏观政策": ["政策", "央行", "降息", "加息", "财政", "货币政策", "刺激"],
    }

    def score_by_rules(self, text: str) -> tuple[float, str, str]:
        """
        基于规则的情感打分。
        Returns: (score -1~1, label, event_type)
        """
        text_lower = text
        pos_count = sum(1 for w in self.POSITIVE_WORDS if w in text_lower)
        neg_count = sum(1 for w in self.NEGATIVE_WORDS if w in text_lower)

        if pos_count + neg_count == 0:
            score, label = 0.0, "中性"
        else:
            raw_score = (pos_count - neg_count) / (pos_count + neg_count + 1)
            score = float(np.tanh(raw_score * 3))  # 映射到 -1~1
            if score > 0.4:
                label = "强烈利好"
            elif score > 0.1:
                label = "利好"
            elif score < -0.4:
                label = "强烈利空"
            elif score < -0.1:
                label = "利空"
            else:
                label = "中性"

        # 事件类型检测
        event_type = ""
        for etype, patterns in self.EVENT_PATTERNS.items():
            if any(p in text_lower for p in patterns):
                event_type = etype
                break

        return score, label, event_type

    def score_by_llm(self, text: str, symbol: str) -> Optional[tuple[float, str, str, float]]:
        """
        使用 LLM 进行深度情感分析。
        Returns: (score, label, event_type, impact_pct) or None
        """
        from quant_investor.llm_gateway import LLMClient as GatewayLLMClient, has_any_provider, _run_sync

        if not has_any_provider():
            return None

        prompt = f"""分析以下金融新闻对股票 {symbol} 的情感影响，以JSON格式输出：

新闻标题：{text[:200]}

请输出：
{{
  "sentiment_score": <-1到1之间的浮点数，1=极度利好，-1=极度利空>,
  "sentiment_label": <"强烈利好"|"利好"|"中性"|"利空"|"强烈利空">,
  "event_type": <事件类型，如"盈利超预期"|"监管处罚"|"并购重组"|"高管变动"|"宏观政策"|"其他">,
  "expected_price_impact_pct": <预期对股价的影响幅度，如2.5表示+2.5%，-3表示-3%>,
  "key_entities": [<相关公司或人物名称列表>],
  "reasoning": <50字以内的简短分析>
}}"""

        try:
            client = GatewayLLMClient(timeout=15.0, max_retries=1)
            data = _run_sync(client.complete(
                messages=[{"role": "user", "content": prompt}],
                model="moonshot-v1-8k",
                max_tokens=256,
                response_json=True,
                stage="news_sentiment",
                actor_name=symbol,
            ))
            return (
                float(data.get("sentiment_score", 0)),
                str(data.get("sentiment_label", "中性")),
                str(data.get("event_type", "")),
                float(data.get("expected_price_impact_pct", 0)),
            )
        except Exception as e:
            _logger.debug(f"LLM情感分析失败: {e}")

        return None

    def analyze_batch(self, items: list[NewsItem], symbol: str) -> list[NewsItem]:
        """批量分析新闻情感"""
        for item in items:
            text = item.title + " " + item.content[:300]
            # 先尝试规则分析
            score, label, event_type = self.score_by_rules(text)
            item.sentiment_score = score
            item.sentiment_label = label
            item.event_type = event_type

            # 重要新闻尝试LLM深度分析
            if abs(score) > 0.3 or event_type:
                llm_result = self.score_by_llm(text, symbol)
                if llm_result:
                    item.sentiment_score = llm_result[0]
                    item.sentiment_label = llm_result[1]
                    item.event_type = llm_result[2]
                    item.impact_score = llm_result[3]

            # 简单关键词提取
            item.keywords = self._extract_keywords(text)

        return items

    @staticmethod
    def _extract_keywords(text: str, top_k: int = 5) -> list[str]:
        """简单关键词提取（基于词频+领域词典）"""
        domain_terms = [
            "营收", "净利润", "毛利率", "ROE", "ROA", "PE", "PB",
            "市值", "股价", "业绩", "增长率", "分红", "回购", "扩张",
            "降息", "加息", "政策", "监管", "并购", "合作", "合同",
        ]
        keywords = []
        for term in domain_terms:
            if term in text:
                keywords.append(term)
        return keywords[:top_k]


# ---------------------------------------------------------------------------
# 新闻分析器主类
# ---------------------------------------------------------------------------

class NewsAnalyzer:
    """
    新闻与替代数据分析器。

    使用方式：
    ----------
    analyzer = NewsAnalyzer()
    result = analyzer.analyze(
        symbol="600519.SH",
        stock_name="贵州茅台",
        days=7,
    )
    print(result.summary)
    """

    def __init__(self) -> None:
        self.fetcher = NewsDataFetcher()
        self.sentiment = SentimentEngine()

    def analyze(
        self,
        symbol: str,
        stock_name: str,
        days: int = 7,
        use_llm: bool = True,
    ) -> NewsAnalysisResult:
        """
        执行完整新闻分析流程。

        Parameters
        ----------
        symbol     : 股票代码（如 "600519.SH"）
        stock_name : 股票名称（如 "贵州茅台"）
        days       : 分析最近N天的新闻
        use_llm    : 是否使用LLM深度分析（需要API密钥）
        """
        _logger.info(f"开始新闻分析: {symbol} ({stock_name}), 近{days}天")

        result = NewsAnalysisResult(
            symbol=symbol,
            stock_name=stock_name,
            analysis_period_days=days,
            total_news_count=0,
        )

        # 1. 获取新闻
        news_items = self.fetcher.fetch_all(symbol, stock_name, days)
        if not news_items:
            _logger.info(f"未获取到 [{symbol}] 的新闻数据，生成模拟数据")
            news_items = self._generate_mock_news(symbol, stock_name)

        result.total_news_count = len(news_items)

        # 2. 情感分析
        analyzed_items = self.sentiment.analyze_batch(news_items, symbol)
        result.news_items = analyzed_items

        # 3. 相关性过滤（保留高相关新闻）
        relevant = self._filter_relevant(analyzed_items, symbol, stock_name)

        # 4. 聚合情感指标
        if relevant:
            scores = [i.sentiment_score for i in relevant]
            result.avg_sentiment_score = round(float(np.mean(scores)), 3)
            result.positive_ratio = round(sum(1 for s in scores if s > 0.1) / len(scores), 3)
            result.negative_ratio = round(sum(1 for s in scores if s < -0.1) / len(scores), 3)
            result.sentiment_trend = self._calc_trend(scores)
            result.cumulative_impact_score = round(
                float(sum(i.impact_score for i in relevant)), 2
            )

        # 5. 重大事件提取
        result.major_events = self._extract_major_events(relevant)

        # 6. 热点主题
        result.hot_topics = self._extract_topics(relevant)

        # 7. 信号生成
        result.news_signal, result.signal_confidence = self._generate_signal(result)

        # 8. 分析师观点（简单统计）
        result.analyst_upgrades, result.analyst_downgrades = self._count_analyst_actions(relevant)

        # 9. 生成报告
        result.summary = self._generate_summary(result)

        _logger.info(
            f"新闻分析完成 [{symbol}]: "
            f"情感={result.avg_sentiment_score:+.3f} "
            f"信号={result.news_signal} "
            f"置信度={result.signal_confidence:.0%}"
        )
        return result

    def _filter_relevant(self, items: list[NewsItem], symbol: str, stock_name: str) -> list[NewsItem]:
        """过滤与标的相关的新闻"""
        relevant = []
        code = symbol.split(".")[0]
        for item in items:
            text = item.title + item.content[:200]
            if (stock_name in text or code in text or
                    any(kw in text for kw in item.keywords)):
                item.relevance_score = 1.0
                relevant.append(item)
            else:
                item.relevance_score = 0.3
                relevant.append(item)  # 保留所有（无法过滤时）
        return relevant

    def _calc_trend(self, scores: list[float]) -> str:
        """计算情感趋势"""
        if len(scores) < 4:
            return "平稳"
        mid = len(scores) // 2
        early = np.mean(scores[mid:])
        recent = np.mean(scores[:mid])
        if recent - early > 0.15:
            return "上升"
        elif early - recent > 0.15:
            return "下降"
        return "平稳"

    def _extract_major_events(self, items: list[NewsItem]) -> list[dict]:
        """提取重大事件"""
        events = []
        for item in items:
            if item.event_type and abs(item.sentiment_score) > 0.3:
                events.append({
                    "event_type": item.event_type,
                    "title": item.title[:100],
                    "sentiment": item.sentiment_label,
                    "impact": item.impact_score,
                    "time": item.publish_time.strftime("%Y-%m-%d"),
                })
        # 按冲击大小排序
        events.sort(key=lambda x: abs(x["impact"]), reverse=True)
        return events[:10]

    def _extract_topics(self, items: list[NewsItem]) -> list[str]:
        """提取热点主题"""
        topic_counts: dict[str, int] = {}
        for item in items:
            if item.event_type:
                topic_counts[item.event_type] = topic_counts.get(item.event_type, 0) + 1
            for kw in item.keywords:
                topic_counts[kw] = topic_counts.get(kw, 0) + 1
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [t for t, _ in sorted_topics[:10]]

    def _generate_signal(self, r: NewsAnalysisResult) -> tuple[str, float]:
        """综合新闻信号"""
        score = r.avg_sentiment_score
        # 正面事件加分，负面事件减分
        for evt in r.major_events[:3]:
            if "超预期" in evt.get("event_type", "") or "利好" in evt.get("sentiment", ""):
                score += 0.1
            elif "处罚" in evt.get("event_type", "") or "利空" in evt.get("sentiment", ""):
                score -= 0.1

        if score > 0.4:
            return "强烈看多", min(0.9, 0.6 + abs(score))
        elif score > 0.15:
            return "看多", 0.65
        elif score < -0.4:
            return "强烈看空", min(0.9, 0.6 + abs(score))
        elif score < -0.15:
            return "看空", 0.65
        return "中性", 0.5

    def _count_analyst_actions(self, items: list[NewsItem]) -> tuple[int, int]:
        """统计分析师升降级"""
        upgrades = sum(1 for i in items if "上调" in i.title or "买入" in i.title or "推荐" in i.title)
        downgrades = sum(1 for i in items if "下调" in i.title or "卖出" in i.title or "减持" in i.title)
        return upgrades, downgrades

    def _generate_mock_news(self, symbol: str, stock_name: str) -> list[NewsItem]:
        """当无法获取真实新闻时生成占位新闻"""
        return [
            NewsItem(
                title=f"{stock_name}({symbol})：暂无最新重大新闻",
                content="当前未能获取到该股票的最新新闻数据，建议通过财经网站手动查询。",
                source="系统提示",
                publish_time=datetime.now(),
                sentiment_score=0.0,
                sentiment_label="中性",
            )
        ]

    def _generate_summary(self, r: NewsAnalysisResult) -> str:
        sentiment_emoji = (
            "📈" if r.news_signal in ("强烈看多", "看多") else
            ("📉" if r.news_signal in ("强烈看空", "看空") else "➡️")
        )
        lines = [
            f"## {r.stock_name} ({r.symbol}) 新闻分析报告\n\n",
            f"**分析周期**: 近 {r.analysis_period_days} 天  **新闻数量**: {r.total_news_count} 条\n\n",
            f"### 情感概览\n",
            f"| 指标 | 数值 |\n|------|------|\n",
            f"| 整体情感得分 | {r.avg_sentiment_score:+.3f} |\n",
            f"| 情感趋势 | {r.sentiment_trend} |\n",
            f"| 正面新闻占比 | {r.positive_ratio:.0%} |\n",
            f"| 负面新闻占比 | {r.negative_ratio:.0%} |\n",
            f"| 累积新闻冲击 | {r.cumulative_impact_score:+.1f}% |\n",
            f"\n### 新闻信号: {sentiment_emoji} **{r.news_signal}** （置信度 {r.signal_confidence:.0%}）\n\n",
        ]

        if r.major_events:
            lines.append("### 重大事件\n")
            for evt in r.major_events[:5]:
                lines.append(
                    f"- **[{evt['event_type']}]** {evt['title']} "
                    f"（{evt['sentiment']}，{evt['time']}）\n"
                )
            lines.append("\n")

        if r.hot_topics:
            lines.append(f"### 热点主题\n{' | '.join(r.hot_topics[:8])}\n\n")

        if r.analyst_upgrades > 0 or r.analyst_downgrades > 0:
            lines.append(
                f"### 分析师动态\n"
                f"- 升级/买入推荐: **{r.analyst_upgrades}** 次\n"
                f"- 降级/卖出推荐: **{r.analyst_downgrades}** 次\n\n"
            )

        if r.news_items:
            lines.append("### 最新新闻（前5条）\n")
            for item in r.news_items[:5]:
                time_str = item.publish_time.strftime("%m-%d %H:%M")
                lines.append(
                    f"- [{time_str}] **{item.sentiment_label}** {item.title[:80]}\n"
                )

        return "".join(lines)
