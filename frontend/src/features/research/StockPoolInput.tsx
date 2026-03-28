import { useState, useRef, useEffect, type KeyboardEvent, type MouseEvent } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useResearchStore } from '../../stores/researchStore'
import { getUniversePresets, resolveUniverse } from '../../api/universe'
import { LoadingSpinner } from '../../components/LoadingSpinner'

type InputMode = 'universe' | 'custom'

export function StockPoolInput() {
  const market = useResearchStore((s) => s.market)
  const stockPool = useResearchStore((s) => s.stock_pool)
  const universeKeys = useResearchStore((s) => s.universe_keys) ?? []
  const universeOperation = useResearchStore((s) => s.universe_operation) ?? 'replace'
  const setField = useResearchStore((s) => s.setField)

  const [mode, setMode] = useState<InputMode>('universe')
  const [customInput, setCustomInput] = useState('')
  const [loadingKeys, setLoadingKeys] = useState<Set<string>>(new Set())
  const inputRef = useRef<HTMLInputElement>(null)

  // Reset selected keys when market switches
  useEffect(() => {
    setField('universe_keys', [])
    setField('universe_operation', 'replace')
  }, [market]) // eslint-disable-line react-hooks/exhaustive-deps

  // Load preset metadata (labels + estimated counts) — instant, no Tushare call
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

  // Resolve one or more preset keys → actual symbols
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

  // ── Preset click handler ──────────────────────────────────────────────────
  const handlePresetClick = (key: string, e: MouseEvent) => {
    const isMulti = e.ctrlKey || e.metaKey
    if (isMulti) {
      const nextKeys = universeKeys.includes(key)
        ? universeKeys.filter((k) => k !== key)
        : [...universeKeys, key]
      if (nextKeys.length === 0) {
        setField('stock_pool', [])
        setField('universe_keys', [])
        setField('stock_input_mode', 'custom')
        return
      }
      resolveMutation.mutate({ keys: nextKeys, operation: universeOperation })
    } else {
      if (universeKeys.length === 1 && universeKeys[0] === key) {
        setField('stock_pool', [])
        setField('universe_keys', [])
        setField('stock_input_mode', 'custom')
        return
      }
      resolveMutation.mutate({ keys: [key], operation: universeOperation })
    }
  }

  // ── Custom tag input helpers ──────────────────────────────────────────────
  const addSymbol = (raw: string) => {
    const symbol = raw.trim().toUpperCase()
    if (symbol && !stockPool.includes(symbol)) {
      setField('stock_pool', [...stockPool, symbol])
      setField('stock_input_mode', 'custom')
      setField('universe_keys', [])
    }
    setCustomInput('')
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' || e.key === ',' || e.key === ' ') {
      e.preventDefault()
      addSymbol(customInput)
    }
    if (e.key === 'Backspace' && !customInput && stockPool.length > 0) {
      setField('stock_pool', stockPool.slice(0, -1))
      setField('universe_keys', [])
    }
  }

  const removeSymbol = (symbol: string) => {
    setField('stock_pool', stockPool.filter((s) => s !== symbol))
    setField('universe_keys', [])
  }

  const clear = () => {
    setField('stock_pool', [])
    setField('universe_keys', [])
    setField('stock_input_mode', 'custom')
  }

  const presets = presetsData?.presets ?? []

  return (
    <div className="space-y-2">
      {/* Header row */}
      <div className="flex items-center justify-between">
        <label className="text-xs text-gray-400">Stock Pool</label>
        <div className="flex items-center gap-2">
          {/* Mode tabs */}
          <div className="flex rounded overflow-hidden border border-gray-700">
            {(['universe', 'custom'] as InputMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`px-2 py-0.5 text-xs transition-colors ${
                  mode === m
                    ? 'bg-gray-700 text-gray-200'
                    : 'bg-gray-900 text-gray-500 hover:text-gray-300'
                }`}
              >
                {m === 'universe' ? '索引' : '自选'}
              </button>
            ))}
          </div>
          {mode === 'universe' && (
            <div className="flex rounded overflow-hidden border border-gray-700">
              {(['replace', 'merge'] as const).map((operation) => (
                <button
                  key={operation}
                  onClick={() => setField('universe_operation', operation)}
                  className={`px-2 py-0.5 text-xs transition-colors ${
                    universeOperation === operation
                      ? 'bg-emerald-700 text-white'
                      : 'bg-gray-900 text-gray-500 hover:text-gray-300'
                  }`}
                >
                  {operation === 'replace' ? '替换' : '合并'}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* Universe preset buttons */}
      {mode === 'universe' && (
        <div className="space-y-1.5">
          <p className="text-[10px] text-gray-600 px-0.5">
            Ctrl/⌘ 多选，顶部切换替换/合并
          </p>
          <div className="grid grid-cols-2 gap-1">
            {presets.map((p) => {
              const isLoading = loadingKeys.has(p.key)
              const isSelected = universeKeys.includes(p.key)
              return (
                <button
                  key={p.key}
                  onClick={(e) => handlePresetClick(p.key, e)}
                  disabled={resolveMutation.isPending}
                  title={p.description + (universeKeys.length > 0 ? ' — Ctrl/⌘ 多选' : '')}
                  className={`flex items-center justify-between px-2.5 py-2 rounded border text-xs transition-colors disabled:cursor-wait ${
                    isSelected
                      ? 'border-emerald-600 bg-emerald-900/30 text-emerald-300'
                      : 'border-gray-700 bg-gray-900 text-gray-300 hover:border-gray-500 hover:bg-gray-800 disabled:opacity-60'
                  }`}
                >
                  <span className="font-medium">{p.label}</span>
                  {isLoading ? (
                    <LoadingSpinner size="sm" />
                  ) : isSelected ? (
                    <span className="text-emerald-500 text-[10px] font-medium">✓</span>
                  ) : (
                    <span className="text-gray-600 text-[10px]">~{fmtCount(p.estimated_count)}</span>
                  )}
                </button>
              )
            })}
          </div>

          {resolveMutation.isError && (
            <p className="text-xs text-red-400 px-0.5">
              {resolveMutation.error instanceof Error
                ? resolveMutation.error.message
                : '加载失败 — 请在 Settings 中配置 Tushare / FRED API Key'}
            </p>
          )}

          {presetsLoading && presets.length === 0 && (
            <p className="text-xs text-gray-600 px-0.5">加载索引列表中...</p>
          )}

          {presetsIsError && (
            <p className="text-xs text-red-400 px-0.5">
              {presetsError instanceof Error ? presetsError.message : '加载股票池索引失败'}
            </p>
          )}
        </div>
      )}

      {/* Custom tag input */}
      {mode === 'custom' && (
        <div
          className="flex flex-wrap gap-1 p-1.5 bg-gray-900 border border-gray-700 rounded text-xs min-h-[36px] cursor-text"
          onClick={() => inputRef.current?.focus()}
        >
          {stockPool.map((s) => (
            <span
              key={s}
              className="inline-flex items-center gap-0.5 px-1.5 py-0.5 bg-gray-800 rounded text-gray-300"
            >
              {s}
              <button
                onClick={(e) => { e.stopPropagation(); removeSymbol(s) }}
                className="text-gray-500 hover:text-red-400 ml-0.5"
              >
                &times;
              </button>
            </span>
          ))}
          <input
            ref={inputRef}
            value={customInput}
            onChange={(e) => setCustomInput(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={() => customInput && addSymbol(customInput)}
            placeholder={stockPool.length === 0 ? '000001.SZ, AAPL ... Enter 或逗号分隔' : ''}
            className="flex-1 min-w-[120px] bg-transparent outline-none text-gray-200 placeholder-gray-600"
          />
        </div>
      )}

      {/* Footer: selection summary + clear */}
      {stockPool.length > 0 && (
        <div className="flex items-center justify-between px-0.5">
          <span className="text-[11px] text-gray-400">
            {universeKeys.length > 0 ? (
              <>
                <span className="text-emerald-500">{universeKeys.join(' + ')}</span>
                <span className="text-gray-600"> · </span>
                <span className="text-gray-500">{universeOperation === 'merge' ? '合并' : '替换'}</span>
                <span className="text-gray-600"> · </span>
                {stockPool.length} 只
              </>
            ) : (
              <>{stockPool.length} 只已选</>
            )}
          </span>
          <button onClick={clear} className="text-[11px] text-gray-600 hover:text-red-400">
            清空
          </button>
        </div>
      )}
    </div>
  )
}

function fmtCount(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(n % 1000 === 0 ? 0 : 1)}k`
  return String(n)
}
