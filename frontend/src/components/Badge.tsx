const STATUS_STYLES: Record<string, string> = {
  queued: 'border-amber-300/24 bg-amber-300/12 text-amber-100',
  running: 'border-cyan-300/24 bg-cyan-300/12 text-cyan-100',
  completed: 'border-emerald-300/24 bg-emerald-300/12 text-emerald-100',
  failed: 'border-red-300/24 bg-red-300/12 text-red-100',
}

interface Props {
  status: string
}

export function Badge({ status }: Props) {
  const style = STATUS_STYLES[status] ?? 'border-white/10 bg-white/[0.04] text-slate-300'

  return (
    <span className={`inline-flex items-center rounded-full border px-3 py-1.5 text-xs font-medium capitalize ${style}`}>
      {status}
    </span>
  )
}
