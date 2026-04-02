import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Play, Sparkles } from 'lucide-react'
import { resolveUniverse } from '../../api/universe'
import { useResearchStore } from '../../stores/researchStore'
import { LoadingSpinner } from '../../components/LoadingSpinner'
import type { ResearchRunRequest } from '../../types/research'

interface Props {
  submit: (req: ResearchRunRequest) => void
  isSubmitting: boolean
  isRunning: boolean
}

export function RunButton({ submit, isSubmitting, isRunning }: Props) {
  const toRequest = useResearchStore((state) => state.toRequest)
  const stockPool = useResearchStore((state) => state.stock_pool)
  const market = useResearchStore((state) => state.market)
  const universeKeys = useResearchStore((state) => state.universe_keys) ?? []
  const universeOperation = useResearchStore((state) => state.universe_operation) ?? 'replace'
  const setField = useResearchStore((state) => state.setField)
  const [resolutionError, setResolutionError] = useState<string | null>(null)

  const hasSymbols = stockPool.length > 0
  const hasUniverseSelection = universeKeys.length > 0

  const resolveMutation = useMutation({
    mutationFn: () =>
      resolveUniverse(market, {
        keys: universeKeys,
        operation: universeOperation,
        existing_pool: universeOperation === 'merge' ? stockPool : [],
      }),
    onSuccess: (data) => {
      if (data.symbols.length === 0) {
        setResolutionError('The selected universe resolved to 0 symbols. Refresh the universe data or choose a different pool.')
        return
      }

      setResolutionError(null)
      setField('stock_pool', data.symbols)
      setField('universe_keys', data.resolved_keys)
      setField('stock_input_mode', data.resolved_keys.length > 1 ? 'multi' : 'universe')
      submit({
        ...toRequest(),
        stock_pool: data.symbols,
        universe_keys: data.resolved_keys,
      })
    },
  })

  const isResolvingUniverse = resolveMutation.isPending
  const disabled = (!hasSymbols && !hasUniverseSelection) || isRunning || isSubmitting || isResolvingUniverse

  const helperText = resolutionError
    ? resolutionError
    : resolveMutation.isError
      ? resolveMutation.error instanceof Error
        ? resolveMutation.error.message
        : 'Failed to resolve the selected universe.'
      : !hasSymbols && !hasUniverseSelection
        ? 'Add symbols or choose a universe to activate the run.'
        : isResolvingUniverse
          ? 'Resolving the selected universe before launch…'
          : isRunning || isSubmitting
            ? 'Streaming progress updates from the active research session.'
            : hasSymbols
              ? `Ready to evaluate ${stockPool.length} selected symbols.`
              : `Ready to resolve and run ${universeKeys.join(' + ')}.`

  const handleRun = () => {
    if (isRunning || isSubmitting || isResolvingUniverse) {
      return
    }

    setResolutionError(null)

    if (hasSymbols) {
      submit(toRequest())
      return
    }

    if (hasUniverseSelection) {
      resolveMutation.mutate()
    }
  }

  return (
    <button
      type="button"
      onClick={handleRun}
      disabled={disabled}
      className={`group flex w-full flex-col items-start gap-1.5 rounded-[1.4rem] border px-4 py-3 text-left transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300 ${
        disabled
          ? 'cursor-not-allowed border-white/8 bg-white/[0.03] text-slate-500'
          : 'border-cyan-300/24 bg-[linear-gradient(135deg,rgba(34,211,238,0.18),rgba(8,18,34,0.92))] text-white hover:border-cyan-200/38 hover:shadow-[0_18px_60px_rgba(8,145,178,0.18)]'
      }`}
    >
      <span className="flex items-center gap-2 text-sm font-semibold">
        {isSubmitting || isRunning || isResolvingUniverse ? (
          <>
            <LoadingSpinner size="sm" />
            {isResolvingUniverse ? 'Resolving universe' : 'Run in progress'}
          </>
        ) : (
          <>
            <span className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-white/14 bg-white/[0.08] text-cyan-50">
              <Play size={14} aria-hidden="true" />
            </span>
            Run mainline research
          </>
        )}
      </span>

      <span
        className={`text-sm leading-6 ${
          resolutionError || resolveMutation.isError
            ? 'text-red-100'
            : disabled && !isResolvingUniverse
              ? 'text-slate-500'
              : 'text-slate-100/80'
        }`}
      >
        {helperText}
      </span>

      {!disabled && !isRunning && !isSubmitting && !isResolvingUniverse && !resolutionError && !resolveMutation.isError && (
        <span className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-cyan-100/70">
          <Sparkles size={13} aria-hidden="true" />
          Structured output only
        </span>
      )}
    </button>
  )
}
