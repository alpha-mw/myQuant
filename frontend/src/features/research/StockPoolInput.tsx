import { useState, useRef, useEffect, type KeyboardEvent, type MouseEvent } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useResearchStore } from '../../stores/researchStore'
import { getUniversePresets, resolveUniverse } from '../../api/universe'
import { fetchStocks } from '../../api/data'
import { LoadingSpinner } from '../../components/LoadingSpinner'

type InputMode = 'universe' | 'custom' | 'browse'

const PANEL_LABEL_CLASS_NAME = 'text-[11px] font-medium uppercase tracking-[0.18em] text-slate-500'
const MODE_BUTTON_CLASS_NAME =
  'rounded-full border px-3 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300'

export function StockPoolInput() {
  const market = useResearchStore((state) => state.market)
  const stockPool = useResearchStore((state) => state.stock_pool)
  const universeKeys = useResearchStore((state) => state.universe_keys) ?? []
  const universeOperation = useResearchStore((state) => state.universe_operation) ?? 'replace'
  const setField = useResearchStore((state) => state.setField)

  const [mode, setMode] = useState<InputMode>('universe')
  const [customInput, setCustomInput] = useState('')
  const [loadingKeys, setLoadingKeys] = useState<Set<string>>(new Set())
  const [browseSearch, setBrowseSearch] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setField('universe_keys', [])
    setField('universe_operation', 'replace')
  }, [market]) // eslint-disable-line react-hooks/exhaustive-deps

  const {
    data: presetsData,
    isLoading: presetsLoading,
    isError: presetsIsError,
    error: presetsError,
  } = useQuery({
    queryKey: ['universe', market, 'presets'],
    queryFn: () => getUniversePresets(market),
    staleTime: Infinity,
  })

  const resolveMutation = useMutation({
    mutationFn: ({ keys, operation }: { keys: string[]; operation: 'replace' | 'merge' }) =>
      resolveUniverse(market, {
        keys,
        operation,
        existing_pool: operation === 'merge' ? stockPool : [],
      }),
    onMutate: ({ keys }) => setLoadingKeys(new Set(keys)),
    onSuccess: (data, { keys, operation }) => {
      setField('stock_pool', data.symbols)
      setField('universe_keys', keys)
      setField('stock_input_mode', keys.length > 1 ? 'multi' : 'universe')
      setField('universe_operation', operation)
      setLoadingKeys(new Set())
    },
    onError: () => setLoadingKeys(new Set()),
  })

  const { data: browseData, isLoading: browseLoading } = useQuery({
    queryKey: ['browse-stocks', market, browseSearch],
    queryFn: () =>
      fetchStocks({
        market,
        search: browseSearch || undefined,
        limit: 30,
      }),
    enabled: mode === 'browse',
    staleTime: 30_000,
  })

  const handlePresetClick = (key: string, event: MouseEvent<HTMLButtonElement>) => {
    const isMulti = event.ctrlKey || event.metaKey

    if (isMulti) {
      const nextKeys = universeKeys.includes(key)
        ? universeKeys.filter((item) => item !== key)
        : [...universeKeys, key]

      if (nextKeys.length === 0) {
        setField('stock_pool', [])
        setField('universe_keys', [])
        setField('stock_input_mode', 'custom')
        return
      }

      setField('universe_keys', nextKeys)
      setField('stock_input_mode', nextKeys.length > 1 ? 'multi' : 'universe')
      if (universeOperation === 'replace') {
        setField('stock_pool', [])
      }
      resolveMutation.mutate({ keys: nextKeys, operation: universeOperation })
      return
    }

    if (universeKeys.length === 1 && universeKeys[0] === key) {
      setField('stock_pool', [])
      setField('universe_keys', [])
      setField('stock_input_mode', 'custom')
      return
    }

    setField('universe_keys', [key])
    setField('stock_input_mode', 'universe')
    if (universeOperation === 'replace') {
      setField('stock_pool', [])
    }
    resolveMutation.mutate({ keys: [key], operation: universeOperation })
  }

  const addSymbol = (raw: string) => {
    const symbol = raw.trim().toUpperCase()

    if (symbol && !stockPool.includes(symbol)) {
      setField('stock_pool', [...stockPool, symbol])
      setField('stock_input_mode', 'custom')
      setField('universe_keys', [])
    }

    setCustomInput('')
  }

  const handleKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' || event.key === ',' || event.key === ' ') {
      event.preventDefault()
      addSymbol(customInput)
    }

    if (event.key === 'Backspace' && !customInput && stockPool.length > 0) {
      setField('stock_pool', stockPool.slice(0, -1))
      setField('universe_keys', [])
    }
  }

  const removeSymbol = (symbol: string) => {
    setField('stock_pool', stockPool.filter((item) => item !== symbol))
    setField('universe_keys', [])
  }

  const toggleBrowseSymbol = (symbol: string) => {
    if (stockPool.includes(symbol)) {
      removeSymbol(symbol)
    } else {
      setField('stock_pool', [...stockPool, symbol])
      setField('stock_input_mode', 'custom')
      setField('universe_keys', [])
    }
  }

  const clear = () => {
    setField('stock_pool', [])
    setField('universe_keys', [])
    setField('stock_input_mode', 'custom')
  }

  const presets = presetsData?.presets ?? []
  const browseStocks = browseData?.items ?? []

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <label className={PANEL_LABEL_CLASS_NAME}>Stock pool</label>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex flex-wrap gap-2">
            {(['universe', 'custom', 'browse'] as InputMode[]).map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setMode(item)}
                className={`${MODE_BUTTON_CLASS_NAME} ${
                  mode === item
                    ? 'border-cyan-300/22 bg-cyan-300/10 text-cyan-50'
                    : 'border-white/10 bg-white/[0.03] text-slate-400 hover:border-white/16 hover:bg-white/[0.05] hover:text-slate-200'
                }`}
              >
                {item === 'universe' ? 'Universe' : item === 'custom' ? 'Custom' : 'Browse'}
              </button>
            ))}
          </div>

          {mode === 'universe' && (
            <div className="flex gap-2">
              {(['replace', 'merge'] as const).map((operation) => (
                <button
                  key={operation}
                  type="button"
                  onClick={() => setField('universe_operation', operation)}
                  className={`${MODE_BUTTON_CLASS_NAME} ${
                    universeOperation === operation
                      ? 'border-emerald-300/20 bg-emerald-300/10 text-emerald-50'
                      : 'border-white/10 bg-white/[0.03] text-slate-400 hover:border-white/16 hover:bg-white/[0.05] hover:text-slate-200'
                  }`}
                >
                  {operation === 'replace' ? 'Replace' : 'Merge'}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {mode === 'universe' && (
        <div className="space-y-3">
          <p className="text-xs leading-6 text-slate-400">
            Use Ctrl or Command for multi-select. The current mode chooses whether each preset replaces or merges into the working pool.
          </p>

          <div className="grid gap-2 sm:grid-cols-2">
            {presets.map((preset) => {
              const isLoading = loadingKeys.has(preset.key)
              const isSelected = universeKeys.includes(preset.key)

              return (
                <button
                  key={preset.key}
                  type="button"
                  onClick={(event) => handlePresetClick(preset.key, event)}
                  disabled={resolveMutation.isPending}
                  title={preset.description}
                  className={`flex items-center justify-between gap-3 rounded-2xl border px-3 py-3 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300 disabled:cursor-wait ${
                    isSelected
                      ? 'border-emerald-300/20 bg-emerald-300/10 text-white'
                      : 'border-white/10 bg-white/[0.03] text-slate-300 hover:border-white/16 hover:bg-white/[0.05] disabled:opacity-70'
                  }`}
                >
                  <span className="min-w-0">
                    <span className="block truncate font-medium">{preset.label}</span>
                    <span className="mt-1 block text-xs leading-5 text-slate-500">{preset.description}</span>
                  </span>

                  <span className="shrink-0 text-xs text-slate-400">
                    {isLoading ? (
                      <LoadingSpinner size="sm" />
                    ) : isSelected ? (
                      'Selected'
                    ) : (
                      `~${fmtCount(preset.estimated_count)}`
                    )}
                  </span>
                </button>
              )
            })}
          </div>

          {resolveMutation.isError && (
            <p className="text-sm text-red-300">
              {resolveMutation.error instanceof Error
                ? resolveMutation.error.message
                : 'Failed to resolve the selected universe.'}
            </p>
          )}

          {presetsLoading && presets.length === 0 && <p className="text-sm text-slate-500">Loading universes…</p>}

          {presetsIsError && (
            <p className="text-sm text-red-300">
              {presetsError instanceof Error ? presetsError.message : 'Failed to load stock universes.'}
            </p>
          )}
        </div>
      )}

      {mode === 'custom' && (
        <div
          className="flex min-h-[3.25rem] flex-wrap gap-2 rounded-[1.5rem] border border-white/10 bg-slate-950/72 p-2.5"
          onClick={() => inputRef.current?.focus()}
        >
          {stockPool.map((symbol) => (
            <span
              key={symbol}
              className="inline-flex items-center gap-2 rounded-full border border-cyan-300/16 bg-cyan-300/10 px-3 py-1.5 text-xs font-medium text-cyan-50"
            >
              {symbol}
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  removeSymbol(symbol)
                }}
                className="rounded-full text-cyan-100/70 transition-colors hover:text-red-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
                aria-label={`Remove ${symbol}`}
              >
                ×
              </button>
            </span>
          ))}

          <input
            aria-label="Custom stock symbols"
            autoComplete="off"
            name="custom_symbols"
            ref={inputRef}
            spellCheck={false}
            value={customInput}
            onChange={(event) => setCustomInput(event.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={() => customInput && addSymbol(customInput)}
            placeholder={stockPool.length === 0 ? '000001.SZ, AAPL, MSFT…' : ''}
            className="min-w-[10rem] flex-1 bg-transparent px-2 py-1 text-sm text-slate-100 outline-none placeholder:text-slate-500"
          />
        </div>
      )}

      {mode === 'browse' && (
        <div className="space-y-3">
          <input
            aria-label="Search stocks"
            autoComplete="off"
            name="stock_search"
            spellCheck={false}
            type="text"
            value={browseSearch}
            onChange={(event) => setBrowseSearch(event.target.value)}
            placeholder="Search by ticker or company name…"
            className="w-full rounded-2xl border border-white/10 bg-slate-950/72 px-3 py-2.5 text-sm text-slate-100 outline-none transition-colors placeholder:text-slate-500 focus:border-cyan-300/35 focus:bg-slate-950/90"
          />

          <div className="max-h-56 overflow-y-auto rounded-[1.5rem] border border-white/10 bg-slate-950/72">
            {browseLoading ? (
              <div className="flex items-center justify-center py-5">
                <LoadingSpinner size="sm" />
              </div>
            ) : browseStocks.length === 0 ? (
              <p className="px-4 py-5 text-center text-sm text-slate-500">No symbols match the current search.</p>
            ) : (
              browseStocks.map((stock) => {
                const selected = stockPool.includes(stock.ts_code)

                return (
                  <button
                    key={stock.ts_code}
                    type="button"
                    onClick={() => toggleBrowseSymbol(stock.ts_code)}
                    className={`flex w-full items-center justify-between gap-3 border-b border-white/8 px-4 py-3 text-left text-sm transition-colors last:border-b-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300 ${
                      selected
                        ? 'bg-emerald-300/10 text-white'
                        : 'text-slate-300 hover:bg-white/[0.04]'
                    }`}
                  >
                    <span className="min-w-0">
                      <span className="block font-mono font-medium">{stock.ts_code}</span>
                      <span className="mt-1 block truncate text-xs text-slate-500">{stock.name ?? ''}</span>
                    </span>
                    <span className="shrink-0 text-xs text-slate-400">
                      {selected ? 'Added' : stock.industry ?? 'Select'}
                    </span>
                  </button>
                )
              })
            )}
          </div>

          {browseData && (
            <p className="text-xs text-slate-500">
              Showing {browseStocks.length} of {browseData.total} results.
            </p>
          )}
        </div>
      )}

      {stockPool.length > 0 && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/[0.03] px-3 py-3">
          <div className="text-sm text-slate-300">
            {universeKeys.length > 0 ? (
              <>
                <span className="font-medium text-white">{universeKeys.join(' + ')}</span>
                <span className="text-slate-500"> · {universeOperation}</span>
                <span className="text-slate-500"> · </span>
                <span>{stockPool.length} selected</span>
              </>
            ) : (
              <span>{stockPool.length} manual selections</span>
            )}
          </div>

          <button
            type="button"
            onClick={clear}
            className="rounded-full border border-white/12 bg-white/[0.04] px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-red-300/20 hover:bg-red-300/10 hover:text-red-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
          >
            Clear selection
          </button>
        </div>
      )}
    </div>
  )
}

function fmtCount(value: number): string {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(value % 1000 === 0 ? 0 : 1)}k`
  }

  return String(value)
}
