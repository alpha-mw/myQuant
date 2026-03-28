import { useResearchStore } from '../../stores/researchStore'
import { LoadingSpinner } from '../../components/LoadingSpinner'
import type { ResearchRunRequest } from '../../types/research'

interface Props {
  submit: (req: ResearchRunRequest) => void
  isSubmitting: boolean
  isRunning: boolean
}

export function RunButton({ submit, isSubmitting, isRunning }: Props) {
  const toRequest = useResearchStore((s) => s.toRequest)
  const stockPool = useResearchStore((s) => s.stock_pool)

  const disabled = stockPool.length === 0 || isRunning || isSubmitting

  const handleRun = () => {
    if (disabled) return
    submit(toRequest())
  }

  return (
    <button
      onClick={handleRun}
      disabled={disabled}
      className={`w-full flex items-center justify-center gap-2 px-4 py-2.5 rounded text-sm font-medium transition-colors ${
        disabled
          ? 'bg-gray-800 text-gray-600 cursor-not-allowed'
          : 'bg-emerald-700 text-white hover:bg-emerald-600'
      }`}
    >
      {isSubmitting || isRunning ? (
        <>
          <LoadingSpinner size="sm" />
          Running...
        </>
      ) : (
        <>{'\u25B6'} Run Research</>
      )}
    </button>
  )
}
