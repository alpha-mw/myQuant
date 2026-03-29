import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Link, useParams, useSearchParams } from 'react-router-dom'
import {
  ArrowLeft,
  Database,
  Search,
  TrendingUp,
  BarChart3,
  Building2,
  Users,
  FileText,
} from 'lucide-react'
import { fetchMarketOverview, fetchStocks, fetchStockDossier } from '../api/data'
import { formatDateTime } from '../lib/format'
import type {
  MarketOverviewResponse,
  StockDossierResponse,
  StockMetric,
  StockFactorSignal,
  CompetitorInfo,
  FundamentalSnapshot,
  QuoteOverview,
  BusinessProfile,
  IndustryContext,
  StockAnalysisMention,
} from '../types/api'

const PAGE_SIZE = 50

export default function DataExplorerPage() {
  const { tsCode } = useParams<{ tsCode: string }>()
  return tsCode ? <StockDossierView tsCode={tsCode} /> : <MarketBrowser />
}

// ── Market Browser ────────────────────────────────────────────────────────────

function MarketBrowser() {
  const [searchParams, setSearchParams] = useSearchParams()
  const market = searchParams.get('market') || undefined
  const index = searchParams.get('index') || undefined
  const search = searchParams.get('search') || undefined
  const page = Math.max(1, Number(searchParams.get('page')) || 1)

  const { data: overview } = useQuery({
    queryKey: ['market-overview'],
    queryFn: fetchMarketOverview,
    staleTime: 60_000,
  })

  const { data: stockData, isLoading: stocksLoading } = useQuery({
    queryKey: ['stocks', { market, index, search, page }],
    queryFn: () =>
      fetchStocks({
        market,
        index,
        search,
        offset: (page - 1) * PAGE_SIZE,
        limit: PAGE_SIZE,
      }),
    staleTime: 30_000,
  })

  const stocks = stockData?.items ?? []
  const total = stockData?.total ?? 0
  const totalPages = Math.ceil(total / PAGE_SIZE)

  function updateFilter(key: string, value: string | undefined) {
    const next = new URLSearchParams(searchParams)
    if (value) {
      next.set(key, value)
    } else {
      next.delete(key)
    }
    next.delete('page')
    setSearchParams(next)
  }

  function setPage(p: number) {
    const next = new URLSearchParams(searchParams)
    if (p > 1) next.set('page', String(p))
    else next.delete('page')
    setSearchParams(next)
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Header */}
      <header className="shrink-0 border-b border-white/10 px-4 py-3 lg:px-6">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.24em] text-teal-300/70">
          <Database size={14} aria-hidden="true" />
          Data Explorer
        </div>
        <h1 className="mt-1 text-lg font-semibold text-white">Market & Stock Data</h1>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 lg:px-6">
        {/* Overview cards */}
        {overview && <OverviewCards overview={overview} />}

        {/* Filters */}
        <div className="mt-4 flex flex-wrap items-center gap-2">
          {/* Market */}
          {[undefined, 'CN', 'US'].map((m) => (
            <button
              key={m ?? 'all'}
              type="button"
              onClick={() => updateFilter('market', m)}
              className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
                market === m
                  ? 'border-teal-400/30 bg-teal-400/10 text-teal-100'
                  : 'border-white/10 bg-white/[0.02] text-slate-400 hover:border-white/16 hover:text-slate-200'
              }`}
            >
              {m ?? 'All'}
            </button>
          ))}

          <span className="mx-1 text-slate-700">|</span>

          {/* Index */}
          {[undefined, 'hs300', 'zz500', 'zz1000'].map((idx) => (
            <button
              key={idx ?? 'all-idx'}
              type="button"
              onClick={() => updateFilter('index', idx)}
              className={`rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
                index === idx
                  ? 'border-amber-400/30 bg-amber-400/10 text-amber-100'
                  : 'border-white/10 bg-white/[0.02] text-slate-400 hover:border-white/16 hover:text-slate-200'
              }`}
            >
              {idx?.toUpperCase() ?? 'All Indices'}
            </button>
          ))}

          {/* Search */}
          <div className="relative ml-auto">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search code or name..."
              defaultValue={search}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  updateFilter('search', (e.target as HTMLInputElement).value || undefined)
                }
              }}
              className="rounded-full border border-white/10 bg-white/[0.03] py-1.5 pl-8 pr-3 text-xs text-slate-200 placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
            />
          </div>
        </div>

        {/* Stock table */}
        <div className="mt-4 overflow-hidden rounded-[1.75rem] border border-white/8 bg-slate-950/50 shadow-2xl shadow-black/20">
          {stocksLoading ? (
            <div className="p-6 text-sm text-slate-400">Loading stocks...</div>
          ) : stocks.length === 0 ? (
            <div className="p-6 text-sm text-slate-500">No stocks found.</div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[48rem] text-sm">
                <thead>
                  <tr className="border-b border-white/8 text-slate-400">
                    <th className="px-4 py-3 text-left font-medium">Code</th>
                    <th className="px-4 py-3 text-left font-medium">Name</th>
                    <th className="px-4 py-3 text-left font-medium">Industry</th>
                    <th className="px-4 py-3 text-left font-medium">Market</th>
                    <th className="px-4 py-3 text-right font-medium">Price</th>
                    <th className="px-4 py-3 text-right font-medium">Change</th>
                    <th className="px-4 py-3 text-center font-medium">Index</th>
                  </tr>
                </thead>
                <tbody>
                  {stocks.map((s) => (
                    <tr
                      key={s.ts_code}
                      className="border-b border-white/6 align-top last:border-b-0 hover:bg-white/[0.03]"
                    >
                      <td className="px-4 py-3">
                        <Link
                          to={`/data/${s.ts_code}`}
                          className="font-medium text-sky-300 hover:text-sky-200"
                        >
                          {s.ts_code}
                        </Link>
                      </td>
                      <td className="px-4 py-3 text-slate-200">{s.name ?? '—'}</td>
                      <td className="px-4 py-3 text-slate-400">{s.industry ?? '—'}</td>
                      <td className="px-4 py-3 text-slate-400">{s.market ?? '—'}</td>
                      <td className="px-4 py-3 text-right font-mono text-slate-200">
                        {s.latest_close != null ? s.latest_close.toFixed(2) : '—'}
                      </td>
                      <td className="px-4 py-3 text-right">
                        {s.change_pct != null ? (
                          <span
                            className={
                              s.change_pct >= 0 ? 'text-emerald-400' : 'text-red-400'
                            }
                          >
                            {s.change_pct >= 0 ? '+' : ''}
                            {s.change_pct.toFixed(2)}%
                          </span>
                        ) : (
                          '—'
                        )}
                      </td>
                      <td className="px-4 py-3 text-center">
                        <div className="flex justify-center gap-1">
                          {s.is_hs300 && (
                            <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] text-amber-300">
                              300
                            </span>
                          )}
                          {s.is_zz500 && (
                            <span className="rounded bg-sky-500/10 px-1.5 py-0.5 text-[10px] text-sky-300">
                              500
                            </span>
                          )}
                          {s.is_zz1000 && (
                            <span className="rounded bg-purple-500/10 px-1.5 py-0.5 text-[10px] text-purple-300">
                              1000
                            </span>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="mt-4 flex items-center justify-center gap-2">
            <button
              type="button"
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              className="rounded-full border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-white/16 hover:text-white disabled:opacity-30"
            >
              Prev
            </button>
            <span className="text-xs text-slate-500">
              {page} / {totalPages} ({total} stocks)
            </span>
            <button
              type="button"
              onClick={() => setPage(Math.min(totalPages, page + 1))}
              disabled={page === totalPages}
              className="rounded-full border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-white/16 hover:text-white disabled:opacity-30"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Overview Cards ────────────────────────────────────────────────────────────

function OverviewCards({ overview }: { overview: MarketOverviewResponse }) {
  const { summary, market_pulse } = overview
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      <StatCard label="Total Stocks" value={String(summary.total_stocks)}>
        <span className="text-[10px] text-slate-500">
          CN {summary.cn_count} · US {summary.us_count}
        </span>
      </StatCard>
      <StatCard label="With Data" value={String(summary.stocks_with_data)}>
        <span className="text-[10px] text-slate-500">{summary.date_range}</span>
      </StatCard>
      <StatCard label="Market Breadth" value={market_pulse.breadth_label}>
        <span className="text-[10px] text-slate-500">
          {(market_pulse.positive_ratio_20d * 100).toFixed(0)}% rising (20d)
        </span>
      </StatCard>
      <StatCard label="Risk State" value={market_pulse.risk_state}>
        <span className="text-[10px] text-slate-500">
          Vol {(market_pulse.avg_volatility_20d * 100).toFixed(1)}% · Ret{' '}
          <span className={market_pulse.avg_return_20d >= 0 ? 'text-emerald-400' : 'text-red-400'}>
            {(market_pulse.avg_return_20d * 100).toFixed(1)}%
          </span>
        </span>
      </StatCard>
    </div>
  )
}

function StatCard({
  label,
  value,
  children,
}: {
  label: string
  value: string
  children?: React.ReactNode
}) {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.03] p-4">
      <p className="text-[10px] uppercase tracking-[0.18em] text-slate-500">{label}</p>
      <p className="mt-1.5 text-xl font-semibold text-white">{value}</p>
      {children && <div className="mt-1">{children}</div>}
    </div>
  )
}

// ── Stock Dossier View ────────────────────────────────────────────────────────

function StockDossierView({ tsCode }: { tsCode: string }) {
  const { data: dossier, isLoading, isError } = useQuery({
    queryKey: ['stock-dossier', tsCode],
    queryFn: () => fetchStockDossier(tsCode),
    staleTime: 60_000,
  })

  type DossierTab = 'overview' | 'fundamentals' | 'technical' | 'competitors' | 'business' | 'history'
  const [tab, setTab] = useState<DossierTab>('overview')

  return (
    <div className="flex h-full min-h-0 flex-col">
      {/* Header */}
      <header className="shrink-0 border-b border-white/10 px-4 py-3 lg:px-6">
        <Link
          to="/data"
          className="inline-flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200"
        >
          <ArrowLeft size={14} />
          Back to stocks
        </Link>
        {dossier && (
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <h1 className="text-lg font-semibold text-white">
              {dossier.display_name}
              <span className="ml-2 text-sm font-normal text-slate-400">{dossier.stock.ts_code}</span>
            </h1>
            {dossier.tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-white/8 bg-white/[0.04] px-2 py-0.5 text-[10px] text-slate-400"
              >
                {tag}
              </span>
            ))}
          </div>
        )}
      </header>

      {isLoading && (
        <div className="p-6 text-sm text-slate-400">Loading stock data...</div>
      )}
      {isError && (
        <div className="p-6 text-sm text-red-300">Failed to load stock data.</div>
      )}
      {dossier && (
        <>
          {/* Tabs */}
          <div className="shrink-0 border-b border-white/8 px-4 lg:px-6">
            <div className="flex gap-1 overflow-x-auto py-2">
              {(
                [
                  { id: 'overview', label: 'Overview', icon: BarChart3 },
                  { id: 'fundamentals', label: 'Fundamentals', icon: FileText },
                  { id: 'technical', label: 'Technical', icon: TrendingUp },
                  { id: 'competitors', label: 'Competitors', icon: Users },
                  { id: 'business', label: 'Business', icon: Building2 },
                  { id: 'history', label: 'Analysis', icon: Database },
                ] as const
              ).map((t) => {
                const Icon = t.icon
                return (
                  <button
                    key={t.id}
                    type="button"
                    onClick={() => setTab(t.id)}
                    className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors ${
                      tab === t.id
                        ? 'border-teal-400/30 bg-teal-400/10 text-teal-100'
                        : 'border-white/8 text-slate-400 hover:border-white/14 hover:text-slate-200'
                    }`}
                  >
                    <Icon size={13} />
                    {t.label}
                  </button>
                )
              })}
            </div>
          </div>

          {/* Tab content */}
          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 lg:px-6">
            {tab === 'overview' && <OverviewTab dossier={dossier} />}
            {tab === 'fundamentals' && <FundamentalsTab fundamentals={dossier.fundamentals} />}
            {tab === 'technical' && <TechnicalTab technical={dossier.technical} />}
            {tab === 'competitors' && <CompetitorsTab competitors={dossier.competitors} industry={dossier.industry_context} />}
            {tab === 'business' && <BusinessTab profile={dossier.business_profile} />}
            {tab === 'history' && <AnalysisHistoryTab history={dossier.analysis_history} />}
          </div>
        </>
      )}
    </div>
  )
}

// ── Dossier Tabs ──────────────────────────────────────────────────────────────

function OverviewTab({ dossier }: { dossier: StockDossierResponse }) {
  const { quote } = dossier
  return (
    <div className="space-y-4">
      {/* Profile */}
      {dossier.profile_summary && (
        <Card title="Profile">
          <p className="text-sm leading-6 text-slate-300">{dossier.profile_summary}</p>
        </Card>
      )}

      {/* Quote */}
      <Card title="Quote">
        <QuoteGrid quote={quote} />
      </Card>

      {/* Key metrics */}
      {dossier.technical.key_metrics.length > 0 && (
        <Card title="Key Metrics">
          <MetricsGrid metrics={dossier.technical.key_metrics} />
        </Card>
      )}

      {/* Completeness */}
      <Card title="Data Completeness">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
          {Object.entries(dossier.completeness).map(([key, avail]) => (
            <div key={key} className="flex items-center gap-2 text-xs">
              <span
                className={`h-2 w-2 rounded-full ${
                  avail.ready ? 'bg-emerald-400' : 'bg-slate-600'
                }`}
              />
              <span className="text-slate-300 capitalize">{key.replace('_', ' ')}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}

function QuoteGrid({ quote }: { quote: QuoteOverview }) {
  const items = [
    { label: 'Latest Close', value: quote.latest_close?.toFixed(2) },
    {
      label: 'Change',
      value: quote.change_pct != null ? `${quote.change_pct >= 0 ? '+' : ''}${quote.change_pct.toFixed(2)}%` : null,
      color: quote.change_pct != null ? (quote.change_pct >= 0 ? 'text-emerald-400' : 'text-red-400') : undefined,
    },
    { label: '20d Return', value: quote.return_20d != null ? `${(quote.return_20d * 100).toFixed(1)}%` : null },
    { label: '60d Return', value: quote.return_60d != null ? `${(quote.return_60d * 100).toFixed(1)}%` : null },
    { label: '20d Volatility', value: quote.volatility_20d != null ? `${(quote.volatility_20d * 100).toFixed(1)}%` : null },
    { label: '52W High', value: quote.high_52w?.toFixed(2) },
    { label: '52W Low', value: quote.low_52w?.toFixed(2) },
    { label: 'Support', value: quote.support_level?.toFixed(2) },
    { label: 'Resistance', value: quote.resistance_level?.toFixed(2) },
  ]
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
      {items.map((item) => (
        <div key={item.label} className="rounded-xl bg-white/[0.03] px-3 py-2">
          <p className="text-[10px] text-slate-500">{item.label}</p>
          <p className={`mt-0.5 text-sm font-medium ${item.color ?? 'text-slate-200'}`}>
            {item.value ?? '—'}
          </p>
        </div>
      ))}
    </div>
  )
}

function MetricsGrid({ metrics }: { metrics: StockMetric[] }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
      {metrics.map((m) => (
        <div key={m.label} className="rounded-xl bg-white/[0.03] px-3 py-2">
          <p className="text-[10px] text-slate-500">{m.label}</p>
          <p
            className={`mt-0.5 text-sm font-medium ${
              m.tone === 'positive'
                ? 'text-emerald-400'
                : m.tone === 'negative'
                  ? 'text-red-400'
                  : 'text-slate-200'
            }`}
          >
            {m.value}
          </p>
        </div>
      ))}
    </div>
  )
}

function FundamentalsTab({ fundamentals }: { fundamentals: FundamentalSnapshot }) {
  const rows: [string, string | null][] = [
    ['Report Period', fundamentals.report_period],
    ['Revenue', fundamentals.revenue != null ? `${(fundamentals.revenue / 1e8).toFixed(2)} 亿` : null],
    ['Net Income', fundamentals.net_income != null ? `${(fundamentals.net_income / 1e8).toFixed(2)} 亿` : null],
    ['Gross Margin', fundamentals.gross_margin != null ? `${(fundamentals.gross_margin * 100).toFixed(1)}%` : null],
    ['Operating Margin', fundamentals.operating_margin != null ? `${(fundamentals.operating_margin * 100).toFixed(1)}%` : null],
    ['ROE', fundamentals.roe != null ? `${(fundamentals.roe * 100).toFixed(1)}%` : null],
    ['ROA', fundamentals.roa != null ? `${(fundamentals.roa * 100).toFixed(1)}%` : null],
    ['Debt/Asset', fundamentals.debt_to_asset != null ? `${(fundamentals.debt_to_asset * 100).toFixed(1)}%` : null],
    ['PE (TTM)', fundamentals.pe_ttm?.toFixed(1) ?? null],
    ['PB', fundamentals.pb?.toFixed(2) ?? null],
    ['PS', fundamentals.ps?.toFixed(2) ?? null],
    ['Market Cap', fundamentals.market_cap != null ? `${(fundamentals.market_cap / 1e8).toFixed(1)} 亿` : null],
  ]

  return (
    <Card title="Financial Snapshot">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
        {rows.map(([label, value]) => (
          <div key={label} className="rounded-xl bg-white/[0.03] px-3 py-2">
            <p className="text-[10px] text-slate-500">{label}</p>
            <p className="mt-0.5 text-sm font-medium text-slate-200">{value ?? '—'}</p>
          </div>
        ))}
      </div>
    </Card>
  )
}

function TechnicalTab({ technical }: { technical: { key_metrics: StockMetric[]; factors: StockFactorSignal[]; notes: string[] } }) {
  return (
    <div className="space-y-4">
      {technical.key_metrics.length > 0 && (
        <Card title="Key Metrics">
          <MetricsGrid metrics={technical.key_metrics} />
        </Card>
      )}
      {technical.factors.length > 0 && (
        <Card title="Factor Signals">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/8 text-slate-400">
                  <th className="px-3 py-2 text-left font-medium">Factor</th>
                  <th className="px-3 py-2 text-right font-medium">Value</th>
                  <th className="px-3 py-2 text-left font-medium">Signal</th>
                  <th className="px-3 py-2 text-left font-medium">Description</th>
                </tr>
              </thead>
              <tbody>
                {technical.factors.map((f) => (
                  <tr key={f.key} className="border-b border-white/6">
                    <td className="px-3 py-2 text-slate-200">{f.label}</td>
                    <td className="px-3 py-2 text-right font-mono text-slate-300">
                      {f.display_value}
                    </td>
                    <td className="px-3 py-2">
                      <span
                        className={`rounded px-1.5 py-0.5 text-xs ${
                          f.signal === 'bullish'
                            ? 'bg-emerald-500/10 text-emerald-300'
                            : f.signal === 'bearish'
                              ? 'bg-red-500/10 text-red-300'
                              : 'bg-slate-500/10 text-slate-400'
                        }`}
                      >
                        {f.signal}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-xs text-slate-500">{f.description}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
      {technical.notes.length > 0 && (
        <Card title="Notes">
          <ul className="space-y-1 text-sm text-slate-400">
            {technical.notes.map((note, i) => (
              <li key={`${note.slice(0, 20)}-${i}`}>· {note}</li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  )
}

function CompetitorsTab({
  competitors,
  industry,
}: {
  competitors: CompetitorInfo[]
  industry: IndustryContext
}) {
  return (
    <div className="space-y-4">
      <Card title="Industry Context">
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 text-sm">
          <div>
            <p className="text-[10px] text-slate-500">Market</p>
            <p className="text-slate-200">{industry.market ?? '—'}</p>
          </div>
          <div>
            <p className="text-[10px] text-slate-500">Sector</p>
            <p className="text-slate-200">{industry.sector ?? '—'}</p>
          </div>
          <div>
            <p className="text-[10px] text-slate-500">Industry</p>
            <p className="text-slate-200">{industry.industry ?? '—'}</p>
          </div>
          <div>
            <p className="text-[10px] text-slate-500">Peers</p>
            <p className="text-slate-200">{industry.peer_count}</p>
          </div>
          <div className="col-span-2">
            <p className="text-[10px] text-slate-500">Summary</p>
            <p className="text-slate-300">{industry.summary || '—'}</p>
          </div>
        </div>
      </Card>

      {competitors.length > 0 && (
        <Card title={`Competitors (${competitors.length})`}>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/8 text-slate-400">
                  <th className="px-3 py-2 text-left font-medium">Code</th>
                  <th className="px-3 py-2 text-left font-medium">Name</th>
                  <th className="px-3 py-2 text-left font-medium">Industry</th>
                  <th className="px-3 py-2 text-right font-medium">Price</th>
                </tr>
              </thead>
              <tbody>
                {competitors.map((c) => (
                  <tr key={c.ts_code} className="border-b border-white/6">
                    <td className="px-3 py-2">
                      <Link
                        to={`/data/${c.ts_code}`}
                        className="text-sky-300 hover:text-sky-200"
                      >
                        {c.ts_code}
                      </Link>
                    </td>
                    <td className="px-3 py-2 text-slate-200">{c.name ?? '—'}</td>
                    <td className="px-3 py-2 text-slate-400">{c.industry ?? '—'}</td>
                    <td className="px-3 py-2 text-right font-mono text-slate-300">
                      {c.latest_close?.toFixed(2) ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}

function BusinessTab({ profile }: { profile: BusinessProfile }) {
  return (
    <div className="space-y-4">
      {profile.summary && (
        <Card title="Business Summary">
          <p className="text-sm leading-6 text-slate-300">{profile.summary}</p>
        </Card>
      )}
      {profile.products.length > 0 && (
        <Card title="Products">
          <div className="flex flex-wrap gap-1.5">
            {profile.products.map((p, i) => (
              <span
                key={`${p}-${i}`}
                className="rounded-full border border-white/8 bg-white/[0.03] px-2.5 py-1 text-xs text-slate-300"
              >
                {p}
              </span>
            ))}
          </div>
        </Card>
      )}
      {profile.business_lines.length > 0 && (
        <Card title="Business Lines">
          <div className="flex flex-wrap gap-1.5">
            {profile.business_lines.map((bl, i) => (
              <span
                key={`${bl}-${i}`}
                className="rounded-full border border-white/8 bg-white/[0.03] px-2.5 py-1 text-xs text-slate-300"
              >
                {bl}
              </span>
            ))}
          </div>
        </Card>
      )}
      <Card title="Company Info">
        <div className="grid grid-cols-2 gap-2 text-sm sm:grid-cols-3">
          {profile.city && (
            <div>
              <p className="text-[10px] text-slate-500">City</p>
              <p className="text-slate-200">{profile.city}</p>
            </div>
          )}
          {profile.region && (
            <div>
              <p className="text-[10px] text-slate-500">Region</p>
              <p className="text-slate-200">{profile.region}</p>
            </div>
          )}
          {profile.country && (
            <div>
              <p className="text-[10px] text-slate-500">Country</p>
              <p className="text-slate-200">{profile.country}</p>
            </div>
          )}
          {profile.employees != null && (
            <div>
              <p className="text-[10px] text-slate-500">Employees</p>
              <p className="text-slate-200">{profile.employees.toLocaleString()}</p>
            </div>
          )}
          {profile.website && (
            <div className="col-span-2">
              <p className="text-[10px] text-slate-500">Website</p>
              <p className="truncate text-sky-300">{profile.website}</p>
            </div>
          )}
        </div>
      </Card>
    </div>
  )
}

function AnalysisHistoryTab({ history }: { history: StockAnalysisMention[] }) {
  if (history.length === 0) {
    return (
      <Card title="Analysis History">
        <p className="text-sm text-slate-500">No prior analysis mentions for this stock.</p>
      </Card>
    )
  }
  return (
    <Card title={`Analysis History (${history.length})`}>
      <div className="space-y-2">
        {history.map((h) => (
          <div
            key={h.analysis_id}
            className="rounded-xl border border-white/6 bg-white/[0.02] p-3"
          >
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-medium text-slate-200">{h.title}</span>
              {h.candidate && (
                <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-300">
                  Candidate
                </span>
              )}
            </div>
            <p className="mt-1 text-xs text-slate-400">{h.summary}</p>
            <p className="mt-1 text-[10px] text-slate-600">
              {h.source} · {formatDateTime(h.created_at)}
            </p>
          </div>
        ))}
      </div>
    </Card>
  )
}

// ── Shared Card ───────────────────────────────────────────────────────────────

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/8 bg-white/[0.02] p-4">
      <h3 className="text-xs uppercase tracking-[0.18em] text-slate-500">{title}</h3>
      <div className="mt-3">{children}</div>
    </div>
  )
}
