const STATUS_STYLES: Record<string, string> = {
  queued: 'bg-yellow-900/50 text-yellow-400 border-yellow-700',
  running: 'bg-blue-900/50 text-blue-400 border-blue-700',
  completed: 'bg-emerald-900/50 text-emerald-400 border-emerald-700',
  failed: 'bg-red-900/50 text-red-400 border-red-700',
}

interface Props {
  status: string
}

export function Badge({ status }: Props) {
  const style = STATUS_STYLES[status] ?? 'bg-gray-800 text-gray-400 border-gray-700'
  return (
    <span className={`inline-flex items-center px-2 py-0.5 text-xs font-medium rounded border ${style}`}>
      {status}
    </span>
  )
}
