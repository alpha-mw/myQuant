import { Play, Sparkles } from 'lucide-react'
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

  const disabled = stockPool.length === 0 || isRunning || isSubmitting
  const helperText =
    stockPool.length === 0
      ? 'Add at least one symbol to activate the run.'
      : isRunning || isSubmitting
        ? 'Streaming progress updates from the active research session.'
        : `Ready to evaluate ${stockPool.length} selected symbols.`

  const handleRun = () => {
    if (disabled) {
      return
    }

    submit(toRequest())
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
        {isSubmitting || isRunning ? (
          <>
            <LoadingSpinner size="sm" />
            Run in progress
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

      <span className={`text-sm leading-6 ${disabled ? 'text-slate-500' : 'text-slate-100/80'}`}>{helperText}</span>

      {!disabled && !isRunning && !isSubmitting && (
        <span className="inline-flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-cyan-100/70">
          <Sparkles size={13} aria-hidden="true" />
          Structured output only
        </span>
      )}
    </button>
  )
}
