import { useDeferredValue, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { ChevronLeft, ChevronRight, Search } from 'lucide-react';
import { fetchMarketOverview, fetchStocks } from '../api/data';
import type { StockInfo } from '../types/api';

const PAGE_SIZE = 40;

const marketOptions = [
  { value: '', label: '全部市场' },
  { value: 'CN', label: 'A股' },
  { value: 'US', label: '美股' },
];

const indexOptions = [
  { value: '', label: '全部指数' },
  { value: 'hs300', label: 'HS300' },
  { value: 'zz500', label: '中证500' },
  { value: 'zz1000', label: '中证1000' },
];

const completenessOptions = [
  { value: '', label: '全部完整度' },
  { value: 'complete', label: '完整度高' },
  { value: 'needs_attention', label: '待补齐' },
  { value: 'missing_fundamentals', label: '缺基本面' },
  { value: 'missing_profile', label: '缺公司档案' },
  { value: 'missing_business', label: '缺业务/产品' },
  { value: 'missing_competitors', label: '缺竞对关系' },
];

type SortKey = 'ts_code' | 'record_count' | 'latest_close' | 'change_pct';

function formatDateRange(start?: string | null, end?: string | null) {
  if (!start || !end) return '-';
  return `${start.slice(0, 4)}-${start.slice(4, 6)}-${start.slice(6, 8)} ~ ${end.slice(0, 4)}-${end.slice(4, 6)}-${end.slice(6, 8)}`;
}

function formatPercent(value: number | null | undefined) {
  if (value == null) return '-';
  return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
}

function completenessScore(stock: StockInfo) {
  return Object.values(stock.completeness).filter((item) => item.ready).length;
}

export default function DataExplorer() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [industry, setIndustry] = useState('');
  const [market, setMarket] = useState('');
  const [indexFilter, setIndexFilter] = useState('');
  const [completeness, setCompleteness] = useState('');
  const [onlyAnalyzed, setOnlyAnalyzed] = useState(false);
  const [hasProfile, setHasProfile] = useState<'' | 'true' | 'false'>('');
  const [hasFundamentals, setHasFundamentals] = useState<'' | 'true' | 'false'>('');
  const [page, setPage] = useState(0);
  const [sortKey, setSortKey] = useState<SortKey>('record_count');
  const [sortAsc, setSortAsc] = useState(false);

  const deferredSearch = useDeferredValue(search);
  const deferredIndustry = useDeferredValue(industry);

  const { data: overview } = useQuery({
    queryKey: ['market-overview'],
    queryFn: fetchMarketOverview,
  });

  const { data, isLoading } = useQuery({
    queryKey: [
      'stocks',
      market,
      indexFilter,
      deferredSearch,
      deferredIndustry,
      completeness,
      onlyAnalyzed,
      hasProfile,
      hasFundamentals,
      page,
    ],
    queryFn: () =>
      fetchStocks({
        market: market || undefined,
        index: indexFilter || undefined,
        search: deferredSearch || undefined,
        industry: deferredIndustry || undefined,
        completeness: completeness || undefined,
        recently_analyzed: onlyAnalyzed ? true : undefined,
        has_profile: hasProfile === '' ? undefined : hasProfile === 'true',
        has_fundamentals: hasFundamentals === '' ? undefined : hasFundamentals === 'true',
        offset: page * PAGE_SIZE,
        limit: PAGE_SIZE,
      }),
  });

  const totalPages = data ? Math.ceil(data.total / PAGE_SIZE) : 0;
  const sortedItems = [...(data?.items ?? [])].sort((left, right) => {
    const direction = sortAsc ? 1 : -1;
    if (sortKey === 'ts_code') {
      return left.ts_code.localeCompare(right.ts_code) * direction;
    }
    const leftValue = left[sortKey] ?? -Infinity;
    const rightValue = right[sortKey] ?? -Infinity;
    if (Number(leftValue) === Number(rightValue)) {
      return left.ts_code.localeCompare(right.ts_code);
    }
    return (Number(leftValue) > Number(rightValue) ? 1 : -1) * direction;
  });

  function toggleSort(nextKey: SortKey) {
    if (sortKey === nextKey) {
      setSortAsc((value) => !value);
      return;
    }
    setSortKey(nextKey);
    setSortAsc(nextKey === 'ts_code');
  }

  return (
    <div className="grid gap-6 xl:grid-cols-[280px_minmax(0,1fr)]">
      <aside className="paper-card h-fit xl:sticky xl:top-6">
        <div className="section-header">
          <div>
            <p className="panel-kicker">筛选侧栏</p>
            <h3>筛选条件</h3>
          </div>
        </div>

        <div className="space-y-4">
          <div className="rounded-[24px] border border-[var(--line)] bg-[rgba(255,255,255,0.74)] p-4">
            <div className="text-sm font-semibold text-[var(--ink)]">数据库概况</div>
            <div className="mt-3 grid grid-cols-2 gap-3 text-sm">
              <FilterMetric label="总覆盖" value={String(overview?.summary.total_stocks ?? '-')} />
              <FilterMetric label="记录数" value={String(overview?.summary.total_records ?? '-')} />
              <FilterMetric label="A股" value={String(overview?.summary.cn_count ?? '-')} />
              <FilterMetric label="美股" value={String(overview?.summary.us_count ?? '-')} />
            </div>
          </div>

          <div className="space-y-3">
            <label className="filter-label" htmlFor="stock-search">
              搜索代码或名称
            </label>
            <div className="relative">
              <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--muted)]" size={16} />
              <input
                id="stock-search"
                value={search}
                onChange={(event) => {
                  setSearch(event.target.value);
                  setPage(0);
                }}
                placeholder="000001.SZ / Apple"
                className="app-input pl-10"
              />
            </div>
          </div>

          <div className="space-y-3">
            <label className="filter-label" htmlFor="industry-search">
              行业关键词
            </label>
            <input
              id="industry-search"
              value={industry}
              onChange={(event) => {
                setIndustry(event.target.value);
                setPage(0);
              }}
              placeholder="银行 / 半导体 / Software"
              className="app-input"
            />
          </div>

          <SelectGroup
            label="市场"
            value={market}
            onChange={(value) => {
              setMarket(value);
              setPage(0);
            }}
            options={marketOptions}
          />
          <SelectGroup
            label="指数"
            value={indexFilter}
            onChange={(value) => {
              setIndexFilter(value);
              setPage(0);
            }}
            options={indexOptions}
          />
          <SelectGroup
            label="完整度"
            value={completeness}
            onChange={(value) => {
              setCompleteness(value);
              setPage(0);
            }}
            options={completenessOptions}
          />
          <SelectGroup
            label="公司档案"
            value={hasProfile}
            onChange={(value) => {
              setHasProfile(value as '' | 'true' | 'false');
              setPage(0);
            }}
            options={[
              { value: '', label: '全部' },
              { value: 'true', label: '已补齐' },
              { value: 'false', label: '待补齐' },
            ]}
          />
          <SelectGroup
            label="基本面"
            value={hasFundamentals}
            onChange={(value) => {
              setHasFundamentals(value as '' | 'true' | 'false');
              setPage(0);
            }}
            options={[
              { value: '', label: '全部' },
              { value: 'true', label: '已补齐' },
              { value: 'false', label: '待补齐' },
            ]}
          />

          <label className="flex items-center justify-between rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.74)] px-4 py-3 text-sm text-[var(--ink)]">
            <span>只看最近分析过的股票</span>
            <input
              type="checkbox"
              checked={onlyAnalyzed}
              onChange={(event) => {
                setOnlyAnalyzed(event.target.checked);
                setPage(0);
              }}
            />
          </label>
        </div>
      </aside>

      <section className="paper-card overflow-hidden">
        <div className="section-header">
          <div>
            <p className="panel-kicker">股票列表</p>
            <h3>覆盖与完整度</h3>
          </div>
          <div className="text-sm text-[var(--muted)]">当前 {data?.total ?? 0} 只</div>
        </div>

        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>
                  <button type="button" className="table-sort" onClick={() => toggleSort('ts_code')}>
                    代码 / 名称
                  </button>
                </th>
                <th>行业 / 市场</th>
                <th>
                  <button type="button" className="table-sort" onClick={() => toggleSort('latest_close')}>
                    最新价
                  </button>
                </th>
                <th>
                  <button type="button" className="table-sort" onClick={() => toggleSort('change_pct')}>
                    当日涨跌
                  </button>
                </th>
                <th>
                  <button type="button" className="table-sort" onClick={() => toggleSort('record_count')}>
                    数据覆盖
                  </button>
                </th>
                <th>完整度</th>
              </tr>
            </thead>
            <tbody>
              {isLoading ? (
                <tr>
                  <td colSpan={6} className="table-empty">
                    加载中...
                  </td>
                </tr>
              ) : sortedItems.length ? (
                sortedItems.map((stock) => (
                  <tr
                    key={stock.ts_code}
                    className="cursor-pointer hover:bg-[rgba(12,33,60,0.04)]"
                    onClick={() => navigate(`/stocks/${stock.ts_code}`)}
                  >
                    <td>
                      <div className="font-mono text-sm text-[var(--ink-soft)]">{stock.ts_code}</div>
                      <div className="mt-1 font-semibold text-[var(--ink)]">{stock.name || stock.ts_code}</div>
                    </td>
                    <td>
                      <div>{stock.industry || '未分类'}</div>
                      <div className="mt-1 flex flex-wrap gap-2">
                        <span className="data-chip">{stock.market === 'CN' ? 'A股' : '美股'}</span>
                        {stock.is_hs300 && <span className="data-chip">HS300</span>}
                        {stock.is_zz500 && <span className="data-chip">中证500</span>}
                        {stock.is_zz1000 && <span className="data-chip">中证1000</span>}
                      </div>
                    </td>
                    <td>{stock.latest_close != null ? stock.latest_close.toFixed(2) : '-'}</td>
                    <td className={stock.change_pct != null && stock.change_pct >= 0 ? 'text-emerald-700' : 'text-rose-700'}>
                      {formatPercent(stock.change_pct)}
                    </td>
                    <td>
                      <div>{stock.record_count.toLocaleString()} 条</div>
                      <div className="mt-1 text-xs text-[var(--muted)]">{formatDateRange(stock.date_start, stock.date_end)}</div>
                    </td>
                    <td>
                      <div className="text-sm font-semibold text-[var(--ink)]">{completenessScore(stock)}/6</div>
                      <div className="mt-1 flex flex-wrap gap-1">
                        {Object.entries(stock.completeness).map(([key, value]) => (
                          <span key={key} className={`mini-chip ${value.ready ? 'is-ready' : 'is-missing'}`}>
                            {labelMap[key] ?? key}
                          </span>
                        ))}
                      </div>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={6} className="table-empty">
                    无匹配结果。
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        {totalPages > 1 && (
          <div className="table-footer">
            <span>
              当前显示 {page * PAGE_SIZE + 1}-{Math.min((page + 1) * PAGE_SIZE, data?.total ?? 0)} / {data?.total}
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                className="icon-button"
                disabled={page === 0}
                onClick={() => setPage((value) => Math.max(0, value - 1))}
              >
                <ChevronLeft size={16} />
              </button>
              <button
                type="button"
                className="icon-button"
                disabled={page >= totalPages - 1}
                onClick={() => setPage((value) => Math.min(totalPages - 1, value + 1))}
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function SelectGroup({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string }>;
}) {
  return (
    <div className="space-y-2">
      <label className="filter-label">{label}</label>
      <select value={value} onChange={(event) => onChange(event.target.value)} className="app-input">
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

function FilterMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[18px] bg-[rgba(12,33,60,0.04)] px-3 py-3">
      <div className="text-xs text-[var(--muted)]">{label}</div>
      <div className="mt-1 text-base font-semibold text-[var(--ink)]">{value}</div>
    </div>
  );
}

const labelMap: Record<string, string> = {
  technical: '技术',
  fundamentals: '基本面',
  industry: '行业',
  competitors: '竞对',
  business: '业务',
  profile: '档案',
};
