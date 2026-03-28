import { useEffect, useRef } from 'react'

interface Props {
  logs: string[]
  progress: number
}

export function LiveLogPanel({ logs, progress }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs.length])

  const pct = Math.round(progress * 100)

  return (
    <div className="flex flex-col h-full">
      {/* Progress bar */}
      <div className="h-1 bg-gray-800 shrink-0">
        <div
          className="h-full bg-emerald-600 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-gray-800 shrink-0">
        <span className="text-xs text-gray-400">Live Logs</span>
        <span className="text-xs text-emerald-500">{pct}%</span>
      </div>

      {/* Log lines */}
      <div className="flex-1 overflow-y-auto p-3 font-mono text-xs leading-relaxed">
        {logs.map((line, i) => (
          <div key={i} className="text-gray-400 hover:text-gray-200">
            <span className="text-gray-600 select-none mr-2">{String(i + 1).padStart(3, ' ')}</span>
            {line}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
