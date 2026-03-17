import { useDeferredValue, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { fetchStocks } from '../api/data';
import { deleteHolding, deleteWatchlist, fetchPortfolioState, upsertHolding, upsertWatchlist } from '../api/portfolio';
import type { StockInfo, UserHolding } from '../types/api';

const DEFAULT_ACCOUNT = '默认账户';

function formatDateTime(value: string) {
  return value.replace('T', ' ').slice(0, 16);
}

export default function Watchlists() {
  const queryClient = useQueryClient();
  const [visibleAccount, setVisibleAccount] = useState('ALL');
  const [editingHoldingId, setEditingHoldingId] = useState<number | null>(null);
  const [holdingFeedback, setHoldingFeedback] = useState<{ tone: 'success' | 'error'; text: string } | null>(null);
  const [watchlistFeedback, setWatchlistFeedback] = useState<{ tone: 'success' | 'error'; text: string } | null>(null);
  const [isSubmittingHolding, setIsSubmittingHolding] = useState(false);
  const [isSubmittingWatchlist, setIsSubmittingWatchlist] = useState(false);
  const [holdingForm, setHoldingForm] = useState({
    holding_id: null as number | null,
    account_name: DEFAULT_ACCOUNT,
    symbol: '',
    name: '',
    market: 'CN',
    quantity: '100',
    cost_basis: '',
    notes: '',
  });
  const [watchlistForm, setWatchlistForm] = useState({
    symbol: '',
    name: '',
    market: 'CN',
    priority: 'normal',
    notes: '',
  });
  const deferredHoldingSearch = useDeferredValue(holdingForm.symbol.trim());
  const deferredWatchlistSearch = useDeferredValue(watchlistForm.symbol.trim());

  const portfolioQuery = useQuery({
    queryKey: ['portfolio-state'],
    queryFn: fetchPortfolioState,
  });
  const holdingSearchQuery = useQuery({
    queryKey: ['holding-search', holdingForm.market, deferredHoldingSearch],
    queryFn: () => fetchStocks({ market: holdingForm.market, search: deferredHoldingSearch, limit: 8 }),
    enabled: deferredHoldingSearch.length > 0,
  });
  const watchlistSearchQuery = useQuery({
    queryKey: ['watchlist-search', watchlistForm.market, deferredWatchlistSearch],
    queryFn: () => fetchStocks({ market: watchlistForm.market, search: deferredWatchlistSearch, limit: 8 }),
    enabled: deferredWatchlistSearch.length > 0,
  });

  const saveHoldingMutation = useMutation({
    mutationFn: upsertHolding,
    onSuccess: (response) => {
      queryClient.setQueryData(['portfolio-state'], response.state);
      resetHoldingForm();
      setHoldingFeedback({ tone: 'success', text: response.message });
    },
    onError: (error: Error) => setHoldingFeedback({ tone: 'error', text: error.message }),
  });
  const deleteHoldingMutation = useMutation({
    mutationFn: deleteHolding,
    onSuccess: (response) => {
      queryClient.setQueryData(['portfolio-state'], response.state);
      setHoldingFeedback({ tone: 'success', text: response.message });
      if (editingHoldingId != null) {
        resetHoldingForm();
      }
    },
    onError: (error: Error) => setHoldingFeedback({ tone: 'error', text: error.message }),
  });
  const saveWatchlistMutation = useMutation({
    mutationFn: upsertWatchlist,
    onSuccess: (response) => {
      queryClient.setQueryData(['portfolio-state'], response.state);
      setWatchlistForm({ symbol: '', name: '', market: 'CN', priority: 'normal', notes: '' });
      setWatchlistFeedback({ tone: 'success', text: response.message });
    },
    onError: (error: Error) => setWatchlistFeedback({ tone: 'error', text: error.message }),
  });
  const deleteWatchlistMutation = useMutation({
    mutationFn: deleteWatchlist,
    onSuccess: (response) => {
      queryClient.setQueryData(['portfolio-state'], response.state);
      setWatchlistFeedback({ tone: 'success', text: response.message });
    },
    onError: (error: Error) => setWatchlistFeedback({ tone: 'error', text: error.message }),
  });

  const state = portfolioQuery.data;
  const watchlist = state?.watchlist ?? [];
  const accounts = state?.summary.accounts.length ? state.summary.accounts : [DEFAULT_ACCOUNT];
  const selectedHoldingAccountOption = accounts.includes(holdingForm.account_name)
    ? holdingForm.account_name
    : '__new__';
  const matchedHolding = resolveMatchedStock(holdingForm.symbol, holdingSearchQuery.data?.items ?? []);
  const matchedWatchlist = resolveMatchedStock(watchlistForm.symbol, watchlistSearchQuery.data?.items ?? []);

  const groupedHoldings = useMemo(() => {
    const sourceHoldings = state?.holdings ?? [];
    const next = new Map<string, UserHolding[]>();
    sourceHoldings.forEach((item) => {
      if (visibleAccount !== 'ALL' && item.account_name !== visibleAccount) {
        return;
      }
      const current = next.get(item.account_name) ?? [];
      current.push(item);
      next.set(item.account_name, current);
    });
    return Array.from(next.entries());
  }, [state?.holdings, visibleAccount]);

  function resetHoldingForm() {
    setEditingHoldingId(null);
    setHoldingForm({
      holding_id: null,
      account_name: visibleAccount === 'ALL' ? (accounts[0] ?? DEFAULT_ACCOUNT) : visibleAccount,
      symbol: '',
      name: '',
      market: 'CN',
      quantity: '100',
      cost_basis: '',
      notes: '',
    });
  }

  function selectHoldingStock(stock: StockInfo) {
    setHoldingForm((current) => ({
      ...current,
      symbol: stock.ts_code,
      name: stock.name ?? '',
      market: stock.market ?? current.market,
    }));
  }

  function selectWatchlistStock(stock: StockInfo) {
    setWatchlistForm((current) => ({
      ...current,
      symbol: stock.ts_code,
      name: stock.name ?? '',
      market: stock.market ?? current.market,
    }));
  }

  function startEditingHolding(holding: UserHolding) {
    setEditingHoldingId(holding.holding_id);
    setHoldingFeedback(null);
    setHoldingForm({
      holding_id: holding.holding_id,
      account_name: holding.account_name,
      symbol: holding.symbol,
      name: holding.name ?? '',
      market: holding.market,
      quantity: String(holding.quantity),
      cost_basis: holding.cost_basis == null ? '' : String(holding.cost_basis),
      notes: holding.notes,
    });
  }

  return (
    <div className="space-y-6">
      <section className="hero-panel">
        <div>
          <p className="panel-kicker">持仓与自选池</p>
          <h2 className="hero-title text-[2rem]">多账户持仓、自选池与研究联动</h2>
          <p className="hero-copy">
            维护多个账户的真实仓位与待观察股票，分析中心会按“我的持仓 / 自选池 / 全市场”自动带入标的。
          </p>
        </div>
        <div className="hero-actions">
          <Link to="/research" className="primary-button">
            去分析中心
          </Link>
        </div>
      </section>

      <section className="paper-card">
        <div className="section-header">
          <div>
            <p className="panel-kicker">摘要</p>
            <h3>当前研究池概览</h3>
          </div>
        </div>

        <div className="metric-grid">
          <MetricBlock label="账户数量" value={String(state?.summary.account_count ?? 0)} note="持仓分账户管理" />
          <MetricBlock label="持仓数量" value={String(state?.summary.holdings_count ?? 0)} note="分析中心可自动带入" />
          <MetricBlock label="自选池数量" value={String(state?.summary.watchlist_count ?? 0)} note="待观察股票池" />
          <MetricBlock label="账户列表" value={accounts.slice(0, 3).join('、') || '暂无'} note="可按账户筛选" />
        </div>
      </section>

      <div className="grid gap-6 xl:grid-cols-2">
        <section className="paper-card">
          <div className="section-header">
            <div>
              <p className="panel-kicker">我的持仓</p>
              <h3>{editingHoldingId ? '更新账户仓位' : '新增账户仓位'}</h3>
            </div>
          </div>

          <div className="mb-4 flex flex-wrap gap-2">
            <button
              type="button"
              className={`mode-chip ${visibleAccount === 'ALL' ? 'is-active' : ''}`}
              onClick={() => setVisibleAccount('ALL')}
            >
              全部账户
            </button>
            {accounts.map((account) => (
              <button
                key={account}
                type="button"
                className={`mode-chip ${visibleAccount === account ? 'is-active' : ''}`}
                onClick={() => setVisibleAccount(account)}
              >
                {account}
              </button>
            ))}
          </div>

          <form
            className="space-y-3"
            onSubmit={async (event) => {
              event.preventDefault();
              setHoldingFeedback(null);
              setIsSubmittingHolding(true);
              try {
                const matchedHoldingForSubmit = await findMatchedStock(
                  holdingForm.symbol,
                  holdingForm.market,
                  holdingSearchQuery.data?.items ?? [],
                );
                if (selectedHoldingAccountOption === '__new__' && !holdingForm.account_name.trim()) {
                  setHoldingFeedback({ tone: 'error', text: '请输入新账户名称。' });
                  return;
                }
                const fallbackSymbol = normalizeSymbolInput(holdingForm.symbol);
                if (!matchedHoldingForSubmit && !fallbackSymbol) {
                  setHoldingFeedback({ tone: 'error', text: '未找到匹配股票，请输入代码或从下拉结果里选择。' });
                  return;
                }
                await saveHoldingMutation.mutateAsync({
                  holding_id: holdingForm.holding_id,
                  account_name: holdingForm.account_name.trim() || DEFAULT_ACCOUNT,
                  symbol: matchedHoldingForSubmit?.ts_code ?? fallbackSymbol!,
                  name: matchedHoldingForSubmit?.name ?? (holdingForm.name || null),
                  market: matchedHoldingForSubmit?.market ?? holdingForm.market,
                  quantity: Number(holdingForm.quantity || 0),
                  cost_basis: holdingForm.cost_basis ? Number(holdingForm.cost_basis) : null,
                  notes: holdingForm.notes,
                });
              } finally {
                setIsSubmittingHolding(false);
              }
            }}
          >
            <div className="grid gap-3 md:grid-cols-2">
              <SelectField
                label="账户"
                value={selectedHoldingAccountOption}
                onChange={(value) => {
                  if (value === '__new__') {
                    setHoldingForm((current) => ({
                      ...current,
                      account_name: current.account_name.trim() && !accounts.includes(current.account_name)
                        ? current.account_name
                        : '',
                    }));
                    return;
                  }
                  setHoldingForm((current) => ({ ...current, account_name: value }));
                }}
                options={[
                  ...accounts.map((account) => ({ value: account, label: account })),
                  { value: '__new__', label: '新增账户…' },
                ]}
              />
              {selectedHoldingAccountOption === '__new__' ? (
                <div className="space-y-2">
                  <label className="filter-label">新账户名称</label>
                  <input
                    value={holdingForm.account_name}
                    onChange={(event) => setHoldingForm((current) => ({ ...current, account_name: event.target.value }))}
                    placeholder="例如 招商证券 / 雪球 / 长线账户"
                    className="app-input"
                  />
                </div>
              ) : (
                <div className="space-y-2">
                  <label className="filter-label">新账户名称</label>
                  <div className="rounded-[18px] border border-[var(--line)] bg-[rgba(12,33,60,0.04)] px-3 py-3 text-sm text-[var(--muted)]">
                    已选择现有账户，若要新建请切换到“新增账户…”。
                  </div>
                </div>
              )}
              <SelectField
                label="市场"
                value={holdingForm.market}
                onChange={(value) => setHoldingForm((current) => ({ ...current, market: value, name: '' }))}
                options={[
                  { value: 'CN', label: 'A股' },
                  { value: 'US', label: '美股' },
                ]}
              />
              <div className="space-y-2 md:col-span-2">
                <label className="filter-label">股票代码或公司名称</label>
                <input
                  value={holdingForm.symbol}
                  onChange={(event) => setHoldingForm((current) => ({ ...current, symbol: event.target.value, name: '' }))}
                  placeholder="例如 平安银行 / Apple / 000001.SZ / AAPL"
                  className="app-input"
                />
                {holdingForm.symbol.trim() && (
                  <StockSearchResultList
                    isLoading={holdingSearchQuery.isLoading}
                    items={holdingSearchQuery.data?.items ?? []}
                    onSelect={selectHoldingStock}
                  />
                )}
              </div>
              <InputField
                label="持仓数量"
                value={holdingForm.quantity}
                onChange={(value) => setHoldingForm((current) => ({ ...current, quantity: value }))}
              />
              <InputField
                label="成本价"
                value={holdingForm.cost_basis}
                onChange={(value) => setHoldingForm((current) => ({ ...current, cost_basis: value }))}
              />
            </div>

            {matchedHolding && (
              <div className="rounded-[18px] bg-[rgba(12,33,60,0.04)] px-3 py-2 text-sm text-[var(--muted)]">
                将保存为 {matchedHolding.ts_code}{matchedHolding.name ? ` · ${matchedHolding.name}` : ''}
              </div>
            )}
            {holdingFeedback && (
              <div className={`rounded-[18px] px-3 py-2 text-sm ${holdingFeedback.tone === 'success' ? 'bg-[rgba(14,116,74,0.08)] text-emerald-700' : 'bg-[rgba(190,92,44,0.08)] text-[var(--danger)]'}`}>
                {holdingFeedback.text}
              </div>
            )}

            <InputField
              label="备注"
              value={holdingForm.notes}
              onChange={(value) => setHoldingForm((current) => ({ ...current, notes: value }))}
              placeholder="例如 长线底仓 / 分析后准备减仓"
            />

            <div className="flex flex-col gap-3 md:flex-row">
              <button
                type="submit"
                className="primary-button w-full justify-center"
                disabled={
                  (saveHoldingMutation.isPending || isSubmittingHolding)
                  || !holdingForm.symbol.trim()
                  || Number(holdingForm.quantity || 0) <= 0
                  || (selectedHoldingAccountOption === '__new__' && !holdingForm.account_name.trim())
                }
              >
                {saveHoldingMutation.isPending || isSubmittingHolding ? '保存中...' : editingHoldingId ? '更新仓位' : '保存持仓'}
              </button>
              {editingHoldingId && (
                <button
                  type="button"
                  className="secondary-button w-full justify-center"
                  onClick={resetHoldingForm}
                >
                  取消编辑
                </button>
              )}
            </div>
          </form>

          <div className="mt-5 space-y-4">
            {portfolioQuery.isLoading ? (
              <div className="empty-card">加载中...</div>
            ) : groupedHoldings.length ? (
              groupedHoldings.map(([accountName, items]) => (
                <div key={accountName} className="space-y-3">
                  <div className="rounded-[18px] bg-[rgba(12,33,60,0.04)] px-4 py-3 text-sm font-semibold text-[var(--ink)]">
                    {accountName} · {items.length} 只
                  </div>
                  {items.map((item) => (
                    <div key={item.holding_id} className="list-card">
                      <div className="list-card-main">
                        <div className="list-card-title">{item.symbol} {item.name ? `· ${item.name}` : ''}</div>
                        <div className="list-card-subtitle">
                          {item.market} · 数量 {item.quantity.toLocaleString('zh-CN')} · 成本 {item.cost_basis ?? '未填'} · {item.notes || '无备注'}
                        </div>
                      </div>
                      <div className="list-card-meta">
                        <div>{formatDateTime(item.updated_at)}</div>
                        <button type="button" className="secondary-button" onClick={() => startEditingHolding(item)}>
                          编辑
                        </button>
                        <button
                          type="button"
                          className="secondary-button"
                          onClick={() => deleteHoldingMutation.mutate(item.holding_id)}
                          disabled={deleteHoldingMutation.isPending}
                        >
                          删除
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ))
            ) : (
              <div className="empty-card">还没有持仓，分析中心的“我的持仓”会因此为空。</div>
            )}
          </div>
        </section>

        <section className="paper-card">
          <div className="section-header">
            <div>
              <p className="panel-kicker">自选池</p>
              <h3>维护待观察股票</h3>
            </div>
          </div>

          <form
            className="space-y-3"
            onSubmit={async (event) => {
              event.preventDefault();
              setWatchlistFeedback(null);
              setIsSubmittingWatchlist(true);
              try {
                const matchedWatchlistForSubmit = await findMatchedStock(
                  watchlistForm.symbol,
                  watchlistForm.market,
                  watchlistSearchQuery.data?.items ?? [],
                );
                const fallbackSymbol = normalizeSymbolInput(watchlistForm.symbol);
                if (!matchedWatchlistForSubmit && !fallbackSymbol) {
                  setWatchlistFeedback({ tone: 'error', text: '未找到匹配股票，请输入代码或从下拉结果里选择。' });
                  return;
                }
                await saveWatchlistMutation.mutateAsync({
                  symbol: matchedWatchlistForSubmit?.ts_code ?? fallbackSymbol!,
                  name: matchedWatchlistForSubmit?.name ?? (watchlistForm.name || null),
                  market: matchedWatchlistForSubmit?.market ?? watchlistForm.market,
                  priority: watchlistForm.priority,
                  notes: watchlistForm.notes,
                });
              } finally {
                setIsSubmittingWatchlist(false);
              }
            }}
          >
            <div className="grid gap-3 md:grid-cols-2">
              <SelectField
                label="市场"
                value={watchlistForm.market}
                onChange={(value) => setWatchlistForm((current) => ({ ...current, market: value, name: '' }))}
                options={[
                  { value: 'CN', label: 'A股' },
                  { value: 'US', label: '美股' },
                ]}
              />
              <SelectField
                label="优先级"
                value={watchlistForm.priority}
                onChange={(value) => setWatchlistForm((current) => ({ ...current, priority: value }))}
                options={[
                  { value: 'high', label: '高优先级' },
                  { value: 'normal', label: '常规跟踪' },
                  { value: 'low', label: '低频观察' },
                ]}
              />
              <div className="space-y-2 md:col-span-2">
                <label className="filter-label">股票代码或公司名称</label>
                <input
                  value={watchlistForm.symbol}
                  onChange={(event) => setWatchlistForm((current) => ({ ...current, symbol: event.target.value, name: '' }))}
                  placeholder="例如 贵州茅台 / NVIDIA / 600519.SH / NVDA"
                  className="app-input"
                />
                {watchlistForm.symbol.trim() && (
                  <StockSearchResultList
                    isLoading={watchlistSearchQuery.isLoading}
                    items={watchlistSearchQuery.data?.items ?? []}
                    onSelect={selectWatchlistStock}
                  />
                )}
              </div>
            </div>

            {matchedWatchlist && (
              <div className="rounded-[18px] bg-[rgba(12,33,60,0.04)] px-3 py-2 text-sm text-[var(--muted)]">
                将保存为 {matchedWatchlist.ts_code}{matchedWatchlist.name ? ` · ${matchedWatchlist.name}` : ''}
              </div>
            )}
            {watchlistFeedback && (
              <div className={`rounded-[18px] px-3 py-2 text-sm ${watchlistFeedback.tone === 'success' ? 'bg-[rgba(14,116,74,0.08)] text-emerald-700' : 'bg-[rgba(190,92,44,0.08)] text-[var(--danger)]'}`}>
                {watchlistFeedback.text}
              </div>
            )}

            <InputField
              label="备注"
              value={watchlistForm.notes}
              onChange={(value) => setWatchlistForm((current) => ({ ...current, notes: value }))}
              placeholder="例如 等财报 / 等回调 / 等仓位腾挪"
            />

            <button
              type="submit"
              className="primary-button w-full justify-center"
              disabled={(saveWatchlistMutation.isPending || isSubmittingWatchlist) || !watchlistForm.symbol.trim()}
            >
              {saveWatchlistMutation.isPending || isSubmittingWatchlist ? '保存中...' : '加入自选池'}
            </button>
          </form>

          <div className="mt-5 space-y-3">
            {portfolioQuery.isLoading ? (
              <div className="empty-card">加载中...</div>
            ) : watchlist.length ? (
              watchlist.map((item) => (
                <div key={item.symbol} className="list-card">
                  <div className="list-card-main">
                    <div className="list-card-title">{item.symbol} {item.name ? `· ${item.name}` : ''}</div>
                    <div className="list-card-subtitle">
                      {item.market} · {priorityLabelMap[item.priority] ?? item.priority} · {item.notes || '暂无备注'}
                    </div>
                  </div>
                  <div className="list-card-meta">
                    <div>{formatDateTime(item.updated_at)}</div>
                    <button
                      type="button"
                      className="secondary-button"
                      onClick={() => deleteWatchlistMutation.mutate(item.symbol)}
                      disabled={deleteWatchlistMutation.isPending}
                    >
                      删除
                    </button>
                  </div>
                </div>
              ))
            ) : (
              <div className="empty-card">自选池还是空的，可以把准备观察的股票先加进来。</div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

function MetricBlock({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className="metric-block">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      <div className="metric-note">{note}</div>
    </div>
  );
}

function InputField({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) {
  return (
    <div className="space-y-2">
      <label className="filter-label">{label}</label>
      <input value={value} onChange={(event) => onChange(event.target.value)} placeholder={placeholder} className="app-input" />
    </div>
  );
}

function SelectField({
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

function StockSearchResultList({
  isLoading,
  items,
  onSelect,
}: {
  isLoading: boolean;
  items: StockInfo[];
  onSelect: (stock: StockInfo) => void;
}) {
  return (
    <div className="rounded-[20px] border border-[var(--line)] bg-white/85 p-2">
      {isLoading ? (
        <div className="empty-inline">搜索中...</div>
      ) : items.length ? (
        <div className="space-y-2">
          {items.map((item) => (
            <button
              key={item.ts_code}
              type="button"
              className="flex w-full items-center justify-between rounded-[16px] px-3 py-2 text-left text-sm transition-colors hover:bg-[rgba(12,33,60,0.05)]"
              onClick={() => onSelect(item)}
            >
              <span className="text-[var(--ink)]">{item.ts_code}{item.name ? ` · ${item.name}` : ''}</span>
              <span className="text-[var(--muted)]">{item.market ?? '-'}</span>
            </button>
          ))}
        </div>
      ) : (
        <div className="empty-inline">没有匹配结果，支持输入股票代码或公司名称。</div>
      )}
    </div>
  );
}

function resolveMatchedStock(input: string, items: StockInfo[]) {
  const normalized = input.trim().toUpperCase();
  if (!normalized) {
    return null;
  }
  return items.find((item) => item.ts_code.toUpperCase() === normalized) ?? items[0] ?? null;
}

async function findMatchedStock(input: string, market: string, items: StockInfo[]) {
  const localMatched = resolveMatchedStock(input, items);
  if (localMatched) {
    return localMatched;
  }
  const keyword = input.trim();
  if (!keyword) {
    return null;
  }
  try {
    const response = await fetchStocks({ market, search: keyword, limit: 1 });
    return response.items[0] ?? null;
  } catch {
    return null;
  }
}

function normalizeSymbolInput(input: string) {
  const normalized = input.trim().toUpperCase();
  return /^[A-Z0-9.]+$/.test(normalized) ? normalized : '';
}

const priorityLabelMap: Record<string, string> = {
  high: '高优先级',
  normal: '常规跟踪',
  low: '低频观察',
};
