import { useEffect, useRef } from 'react'

interface Props {
  logs: string[]
  progress: number
  phaseLabel?: string
  phaseKey?: string
}

export function LiveLogPanel({ logs, progress, phaseLabel, phaseKey }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs.length])

  const pct = Math.round(progress * 100)

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-white/8 px-4 py-4 lg:px-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-[11px] uppercase tracking-[0.22em] text-cyan-200/65">Execution feed</p>
            <h3 className="mt-2 text-lg font-semibold text-white">{phaseLabel || 'Preparing research run'}</h3>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              Live logs stream here while the workspace advances through the current phase.
            </p>
          </div>

          <div className="min-w-[10rem] rounded-[1.4rem] border border-cyan-300/18 bg-cyan-300/10 px-4 py-3">
            <div className="flex items-center justify-between text-[11px] uppercase tracking-[0.18em] text-cyan-100/70">
              <span>Progress</span>
              <span className="font-semibold tabular-nums text-cyan-50">{pct}%</span>
            </div>
            <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-slate-900/90">
              <div
                className="h-full rounded-full bg-[linear-gradient(90deg,rgba(34,211,238,1),rgba(16,185,129,0.92))] transition-all duration-500"
                style={{ width: `${pct}%` }}
              />
            </div>
            {phaseKey && <p className="mt-2 text-xs text-cyan-100/65">{phaseKey}</p>}
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 lg:px-6">
        <div className="min-h-full rounded-[1.8rem] border border-white/10 bg-[linear-gradient(180deg,rgba(6,12,22,0.96),rgba(3,7,18,0.92))] shadow-[inset_0_1px_0_rgba(255,255,255,0.04)]">
          <div className="flex items-center justify-between border-b border-white/8 px-4 py-3">
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-slate-500">Terminal</p>
              <p className="mt-1 text-sm text-slate-300">{logs.length} log lines captured</p>
            </div>
            <span className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1.5 text-xs text-slate-400">
              Auto-scroll enabled
            </span>
          </div>

          <div className="space-y-2 px-4 py-4 font-mono text-[13px] leading-6 text-slate-300">
            {logs.length === 0 ? (
              <p className="text-sm text-slate-500">Waiting for the first log line…</p>
            ) : (
              logs.map((line, index) => (
                <div
                  key={`${index}-${line}`}
                  className="grid grid-cols-[3rem_minmax(0,1fr)] gap-3 rounded-xl px-3 py-2 transition-colors hover:bg-white/[0.03]"
                >
                  <span className="select-none text-right text-slate-600">{String(index + 1).padStart(3, '0')}</span>
                  <span className="break-words text-slate-200">{line}</span>
                </div>
              ))
            )}
            <div ref={bottomRef} />
          </div>
        </div>
      </div>
    </div>
  )
}
