interface Props {
  progress: number
  status: string | null
}

export function ProgressBar({ progress, status }: Props) {
  const pct = Math.round(progress * 100)

  const barColor =
    status === 'failed'
      ? 'bg-red-600'
      : status === 'completed'
        ? 'bg-emerald-500'
        : 'bg-blue-500'

  return (
    <div className="flex items-center gap-3">
      <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full ${barColor} transition-all duration-500 rounded-full`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-xs text-gray-500 w-8 text-right">{pct}%</span>
    </div>
  )
}
