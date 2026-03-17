import { useEffect, useRef } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, ArrowUpRight } from 'lucide-react';
import {
  CandlestickSeries,
  HistogramSeries,
  createChart,
  type IChartApi,
} from 'lightweight-charts';
import { fetchOHLCV, fetchStockDossier } from '../api/data';
import type {
  OHLCVRecord,
  StockAnalysisMention,
  StockFactorSignal,
  StockMetric,
} from '../types/api';

const sections = [
  { id: 'overview', label: '概览' },
  { id: 'technical', label: '技术面' },
  { id: 'fundamentals', label: '基本面' },
  { id: 'industry', label: '行业与竞对' },
  { id: 'business', label: '产品与业务' },
  { id: 'history', label: '历史分析与决策' },
];

function formatPercent(value: number | null | undefined) {
  if (value == null) return '-';
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
}

function formatNumber(value: number | null | undefined, digits = 2) {
  if (value == null) return '-';
  return value.toLocaleString('zh-CN', {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function formatMaybeRatio(value: number | null | undefined) {
  if (value == null) return '-';
  if (Math.abs(value) <= 1) return formatPercent(value);
  return formatNumber(value, 2);
}

function formatDateTime(value: string) {
  return value.replace('T', ' ').slice(0, 16);
}

function toneClass(tone: string) {
  if (tone === 'positive') return 'text-emerald-700';
  if (tone === 'negative') return 'text-rose-700';
  return 'text-[var(--ink)]';
}

export default function StockDetail() {
  const { tsCode } = useParams<{ tsCode: string }>();

  const dossierQuery = useQuery({
    queryKey: ['stock-dossier', tsCode],
    queryFn: () => fetchStockDossier(tsCode!),
    enabled: !!tsCode,
  });

  const chartQuery = useQuery({
    queryKey: ['ohlcv', tsCode],
    queryFn: () => fetchOHLCV(tsCode!),
    enabled: !!tsCode,
  });

  if (dossierQuery.isLoading || !dossierQuery.data) {
    return <div className="empty-card">加载中...</div>;
  }

  const dossier = dossierQuery.data;
  const completeness = Object.entries(dossier.completeness);
  const fundamentalCards = [
    { label: '营业收入', value: formatNumber(dossier.fundamentals.revenue, 0) },
    { label: '净利润', value: formatNumber(dossier.fundamentals.net_income, 0) },
    { label: '毛利率', value: formatMaybeRatio(dossier.fundamentals.gross_margin) },
    { label: 'ROE', value: formatMaybeRatio(dossier.fundamentals.roe) },
    { label: '经营现金流', value: formatNumber(dossier.fundamentals.operating_cashflow, 0) },
    { label: '总资产', value: formatNumber(dossier.fundamentals.total_assets, 0) },
    { label: '总负债', value: formatNumber(dossier.fundamentals.total_liabilities, 0) },
    { label: '市值', value: formatNumber(dossier.fundamentals.market_cap, 0) },
  ];

  const seriesByPeriod = dossier.fundamental_series.reduce<Record<string, Record<string, string | number | null>>>(
    (accumulator, item) => {
      accumulator[item.period] ??= { period: item.period };
      accumulator[item.period][item.metric_name] = item.value;
      return accumulator;
    },
    {},
  );
  const chartData = Object.values(seriesByPeriod)
    .sort((left, right) => String(left.period).localeCompare(String(right.period)))
    .map((item) => ({
      period: String(item.period),
      revenue: typeof item.revenue === 'number' ? item.revenue / 1_000_000 : null,
      net_income: typeof item.net_income === 'number' ? item.net_income / 1_000_000 : null,
      operating_cashflow: typeof item.operating_cashflow === 'number' ? item.operating_cashflow / 1_000_000 : null,
    }));

  return (
    <div className="space-y-6">
      <section className="paper-card">
        <div className="flex items-start gap-4">
          <Link to="/stocks" className="icon-button mt-1">
            <ArrowLeft size={16} />
          </Link>
          <div className="flex-1 space-y-4">
            <div className="flex flex-col gap-4 xl:flex-row xl:items-start xl:justify-between">
              <div>
                <p className="panel-kicker">股票档案页</p>
                <h1 className="hero-title text-[2.2rem]">
                  {dossier.display_name}
                  <span className="ml-3 font-mono text-lg text-[var(--muted)]">{dossier.stock.ts_code}</span>
                </h1>
                <p className="hero-copy mt-3 max-w-4xl">{dossier.profile_summary}</p>
              </div>
              <div className="grid min-w-[280px] gap-3 md:grid-cols-2">
                <MetricSummary title="最新价" value={formatNumber(dossier.quote.latest_close, 2)} note={formatPercent(dossier.quote.change_pct)} />
                <MetricSummary title="20日收益" value={formatPercent(dossier.quote.return_20d)} note="趋势强弱" />
                <MetricSummary title="20日波动率" value={formatPercent(dossier.quote.volatility_20d)} note="风险水平" />
                <MetricSummary title="最近命中" value={String(dossier.analysis_history.length)} note="分析记录" />
              </div>
            </div>

            <div className="flex flex-wrap gap-2">
              {dossier.tags.map((tag) => (
                <span key={tag} className="candidate-chip">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        </div>
      </section>

      <nav className="sticky top-4 z-10 rounded-[24px] border border-[var(--line)] bg-[rgba(252,249,243,0.9)] p-2 backdrop-blur">
        <div className="flex flex-wrap gap-2">
          {sections.map((section) => (
            <a key={section.id} href={`#${section.id}`} className="anchor-chip">
              {section.label}
            </a>
          ))}
        </div>
      </nav>

      <section id="overview" className="paper-card">
        <div className="section-header">
          <div>
            <p className="panel-kicker">概览</p>
            <h3>摘要与完整度</h3>
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[1.1fr_0.9fr]">
          <div className="space-y-4">
            <div className="metric-grid">
              {dossier.technical.key_metrics.map((metric) => (
                <MetricTile key={metric.label} metric={metric} />
              ))}
            </div>
            <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
              <div className="text-sm font-semibold text-[var(--ink)]">公司摘要</div>
              <p className="mt-3 text-sm leading-7 text-[var(--muted)]">{dossier.business_profile.summary}</p>
            </div>
          </div>

          <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
            <div className="text-sm font-semibold text-[var(--ink)]">数据完整度</div>
            <div className="mt-4 space-y-3">
              {completeness.map(([key, value]) => (
                <div key={key} className="flex items-center justify-between gap-4 rounded-[18px] bg-[rgba(12,33,60,0.04)] px-4 py-3">
                  <div>
                    <div className="font-semibold text-[var(--ink)]">{completenessLabel[key] ?? key}</div>
                    <div className="mt-1 text-xs text-[var(--muted)]">{value.source ?? '未标注来源'}</div>
                  </div>
                  <div className={value.ready ? 'text-sm font-semibold text-emerald-700' : 'text-sm font-semibold text-[var(--danger)]'}>
                    {value.ready ? '已就绪' : value.note ?? '待补齐'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section id="technical" className="paper-card">
        <div className="section-header">
          <div>
            <p className="panel-kicker">技术面</p>
            <h3>K线、量能与趋势</h3>
          </div>
        </div>

        <div className="space-y-6">
          <PriceChart records={chartQuery.data?.records ?? []} />
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {dossier.technical.factors.length ? (
              dossier.technical.factors.map((factor) => <FactorCard key={factor.key} factor={factor} />)
            ) : (
              <div className="empty-card col-span-full">数据不足。</div>
            )}
          </div>
          {dossier.technical.notes.length > 0 && (
            <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
              <div className="text-sm font-semibold text-[var(--ink)]">技术观察</div>
              <div className="mt-3 space-y-2">
                {dossier.technical.notes.map((note) => (
                  <div key={note} className="rounded-[18px] bg-[rgba(12,33,60,0.04)] px-3 py-3 text-sm text-[var(--muted)]">
                    {note}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      <section id="fundamentals" className="paper-card">
        <div className="section-header">
          <div>
            <p className="panel-kicker">基本面</p>
            <h3>财务快照与历史</h3>
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[0.95fr_1.05fr]">
          <div className="metric-grid">
            {fundamentalCards.map((item) => (
              <div key={item.label} className="metric-block">
                <div className="metric-label">{item.label}</div>
                <div className="metric-value text-[1.2rem]">{item.value}</div>
                <div className="metric-note">
                  {dossier.fundamentals.report_period ? `报告期 ${dossier.fundamentals.report_period}` : '暂无最新报告期'}
                </div>
              </div>
            ))}
          </div>

          <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
            <div className="mb-3 flex items-center justify-between">
              <div className="text-sm font-semibold text-[var(--ink)]">财务序列走势</div>
              <div className="text-xs text-[var(--muted)]">单位：百万</div>
            </div>
            {chartData.length ? (
              <FundamentalSeriesBoard data={chartData} />
            ) : (
              <div className="empty-card">暂无财务数据。</div>
            )}
          </div>
        </div>
      </section>

      <section id="industry" className="paper-card">
        <div className="section-header">
          <div>
            <p className="panel-kicker">行业与竞对</p>
            <h3>可比公司与行业定位</h3>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
            <div className="text-sm font-semibold text-[var(--ink)]">行业摘要</div>
            <p className="mt-3 text-sm leading-7 text-[var(--muted)]">{dossier.industry_context.summary}</p>
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              <MetricSummary title="市场" value={dossier.industry_context.market ?? '-'} note="交易市场" />
              <MetricSummary title="行业样本" value={String(dossier.industry_context.industry_stock_count)} note="本地同业数量" />
              <MetricSummary title="可比公司" value={String(dossier.industry_context.peer_count)} note="已识别 peer" />
            </div>
          </div>

          <div className="overflow-x-auto rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)]">
            <table className="data-table">
              <thead>
                <tr>
                  <th>代码</th>
                  <th>名称</th>
                  <th>行业</th>
                  <th>识别依据</th>
                  <th>最新价</th>
                </tr>
              </thead>
              <tbody>
                {dossier.competitors.length ? (
                  dossier.competitors.map((item) => (
                    <tr key={item.ts_code}>
                      <td className="font-mono text-[var(--ink-soft)]">{item.ts_code}</td>
                      <td>
                        <Link to={`/stocks/${item.ts_code}`} className="inline-flex items-center gap-1 font-semibold text-[var(--ink)]">
                          {item.name || item.ts_code}
                          <ArrowUpRight size={14} />
                        </Link>
                      </td>
                      <td>{item.industry || '-'}</td>
                      <td>{item.reason ?? '可比样本'}</td>
                      <td>{item.latest_close != null ? formatNumber(item.latest_close, 2) : '-'}</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="table-empty">
                      暂无可比公司。
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section id="business" className="paper-card">
        <div className="section-header">
          <div>
            <p className="panel-kicker">产品与业务</p>
            <h3>主营业务与公司画像</h3>
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[1fr_0.9fr]">
          <div className="space-y-4">
            <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
              <div className="text-sm font-semibold text-[var(--ink)]">业务概述</div>
              <p className="mt-3 text-sm leading-7 text-[var(--muted)]">{dossier.business_profile.summary}</p>
            </div>

            <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
              <div className="text-sm font-semibold text-[var(--ink)]">产品关键词</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {dossier.business_profile.products.length ? (
                  dossier.business_profile.products.map((item) => (
                    <span key={item} className="candidate-chip">
                      {item}
                    </span>
                  ))
                ) : (
                  <div className="empty-inline">暂无产品关键词。</div>
                )}
              </div>
            </div>
          </div>

          <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
            <div className="text-sm font-semibold text-[var(--ink)]">业务线与公司信息</div>
            <div className="mt-3 space-y-3">
              {dossier.business_profile.business_lines.length ? (
                dossier.business_profile.business_lines.map((item) => (
                  <div key={item} className="rounded-[18px] bg-[rgba(12,33,60,0.04)] px-3 py-3 text-sm text-[var(--muted)]">
                    {item}
                  </div>
                ))
              ) : (
                <div className="empty-inline">暂无业务线数据。</div>
              )}
            </div>

            <div className="mt-5 grid gap-3 text-sm text-[var(--muted)]">
              <BusinessMeta label="官网" value={dossier.business_profile.website} />
              <BusinessMeta label="城市" value={dossier.business_profile.city} />
              <BusinessMeta label="地区" value={dossier.business_profile.region} />
              <BusinessMeta label="国家" value={dossier.business_profile.country} />
              <BusinessMeta label="员工数" value={dossier.business_profile.employees != null ? String(dossier.business_profile.employees) : null} />
              <BusinessMeta label="来源" value={dossier.business_profile.source} />
            </div>
          </div>
        </div>
      </section>

      <section id="history" className="paper-card">
        <div className="section-header">
          <div>
            <p className="panel-kicker">历史分析与决策</p>
            <h3>历次研究结论</h3>
          </div>
        </div>

        <div className="space-y-3">
          {dossier.analysis_history.length ? (
            dossier.analysis_history.map((item) => (
              <HistoryCard key={`${item.analysis_id}-${item.created_at}`} item={item} />
            ))
          ) : (
            <div className="empty-card">暂无历史分析。</div>
          )}
        </div>
      </section>
    </div>
  );
}

function MetricSummary({ title, value, note }: { title: string; value: string; note: string }) {
  return (
    <div className="metric-block">
      <div className="metric-label">{title}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </div>
  );
}

function MetricTile({ metric }: { metric: StockMetric }) {
  return (
    <div className="metric-block">
      <div className="metric-label">{metric.label}</div>
      <div className={`metric-value ${toneClass(metric.tone)}`}>{metric.value}</div>
      <div className="metric-note">&nbsp;</div>
    </div>
  );
}

function FactorCard({ factor }: { factor: StockFactorSignal }) {
  return (
    <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
      <div className="flex items-center justify-between gap-3">
        <div className="text-sm font-semibold text-[var(--ink)]">{factor.label}</div>
        <div className={`text-sm font-semibold ${toneClass(factor.signal)}`}>{factor.display_value}</div>
      </div>
      <p className="mt-3 text-sm leading-6 text-[var(--muted)]">{factor.description}</p>
    </div>
  );
}

function BusinessMeta({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="rounded-[18px] bg-[rgba(12,33,60,0.04)] px-3 py-3">
      <div className="text-xs text-[var(--muted)]">{label}</div>
      <div className="mt-1 font-medium text-[var(--ink)]">{value || '-'}</div>
    </div>
  );
}

function HistoryCard({ item }: { item: StockAnalysisMention }) {
  return (
    <div className="list-card">
      <div className="list-card-main">
        <div className="list-card-title">{item.title}</div>
        <div className="list-card-subtitle">{item.candidate ? '进入候选池' : '作为研究样本出现'}</div>
      </div>
      <div className="list-card-meta">
        <div>{formatDateTime(item.created_at)}</div>
        <div>{item.summary}</div>
      </div>
    </div>
  );
}

function FundamentalSeriesBoard({
  data,
}: {
  data: Array<{
    period: string;
    revenue: number | null;
    net_income: number | null;
    operating_cashflow: number | null;
  }>;
}) {
  const values = data.flatMap((item) => [item.revenue, item.net_income, item.operating_cashflow].filter((value): value is number => value != null));
  const maxValue = values.length ? Math.max(...values.map((item) => Math.abs(item))) : 1;

  return (
    <div className="space-y-3">
      {data.map((item) => (
        <div key={item.period} className="rounded-[18px] bg-[rgba(12,33,60,0.04)] px-4 py-3">
          <div className="text-sm font-semibold text-[var(--ink)]">{item.period}</div>
          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <SeriesCell label="营收" value={item.revenue} maxValue={maxValue} color="rgba(181,109,42,0.78)" />
            <SeriesCell label="净利润" value={item.net_income} maxValue={maxValue} color="rgba(33,79,104,0.78)" />
            <SeriesCell label="经营现金流" value={item.operating_cashflow} maxValue={maxValue} color="rgba(46,139,87,0.78)" />
          </div>
        </div>
      ))}
    </div>
  );
}

function SeriesCell({
  label,
  value,
  maxValue,
  color,
}: {
  label: string;
  value: number | null;
  maxValue: number;
  color: string;
}) {
  const width = value == null || maxValue === 0 ? 0 : Math.min(100, (Math.abs(value) / maxValue) * 100);
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-xs text-[var(--muted)]">
        <span>{label}</span>
        <span>{value != null ? formatNumber(value, 1) : '-'}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-white/80">
        <div className="h-full rounded-full" style={{ width: `${width}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

function PriceChart({ records }: { records: OHLCVRecord[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || records.length === 0) return undefined;

    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 420,
      layout: {
        background: { color: '#fcf9f3' },
        textColor: '#183247',
      },
      grid: {
        vertLines: { color: 'rgba(24,50,71,0.08)' },
        horzLines: { color: 'rgba(24,50,71,0.08)' },
      },
      timeScale: { timeVisible: false },
    });

    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#be5c2c',
      downColor: '#2e8b57',
      borderUpColor: '#be5c2c',
      borderDownColor: '#2e8b57',
      wickUpColor: '#be5c2c',
      wickDownColor: '#2e8b57',
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });

    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.78, bottom: 0 },
    });

    const formatDate = (value: string) =>
      value.length === 8 ? `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6, 8)}` : value;

    candleSeries.setData(
      records.map((item) => ({
        time: formatDate(item.trade_date),
        open: item.open,
        high: item.high,
        low: item.low,
        close: item.close,
      })) as never,
    );

    volumeSeries.setData(
      records.map((item) => ({
        time: formatDate(item.trade_date),
        value: item.volume,
        color: item.close >= item.open ? 'rgba(190,92,44,0.28)' : 'rgba(46,139,87,0.28)',
      })) as never,
    );

    chart.timeScale().fitContent();

    function handleResize() {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    }

    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [records]);

  if (!records.length) {
    return <div className="empty-card">暂时还没有本地行情数据。</div>;
  }

  return <div ref={containerRef} className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-2" />;
}

const completenessLabel: Record<string, string> = {
  technical: '技术面',
  fundamentals: '基本面',
  industry: '行业信息',
  competitors: '竞对关系',
  business: '产品/业务',
  profile: '公司档案',
};
