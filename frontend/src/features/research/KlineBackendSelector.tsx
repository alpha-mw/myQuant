import { useResearchStore } from '../../stores/researchStore'

const BACKENDS = ['heuristic', 'kronos', 'chronos', 'hybrid'] as const

export function KlineBackendSelector() {
  const backend = useResearchStore((s) => s.kline_backend)
  const setField = useResearchStore((s) => s.setField)

  return (
    <div>
      <label className="block text-xs text-gray-400 mb-1">K-Line Backend</label>
      <select
        value={backend}
        onChange={(e) => setField('kline_backend', e.target.value)}
        className="w-full px-2 py-1.5 text-xs bg-gray-900 border border-gray-700 rounded text-gray-200 outline-none focus:border-emerald-700"
      >
        {BACKENDS.map((b) => (
          <option key={b} value={b}>{b}</option>
        ))}
      </select>
    </div>
  )
}
