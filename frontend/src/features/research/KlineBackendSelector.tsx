import { useResearchStore } from '../../stores/researchStore'

const BACKENDS = ['heuristic', 'kronos', 'chronos', 'hybrid'] as const

export function KlineBackendSelector() {
  const backend = useResearchStore((state) => state.kline_backend)
  const setField = useResearchStore((state) => state.setField)

  return (
    <div>
      <label className="mb-2 block text-[11px] font-medium uppercase tracking-[0.18em] text-slate-500" htmlFor="research-kline-backend">
        K-Line backend
      </label>
      <select
        id="research-kline-backend"
        name="kline_backend"
        value={backend}
        onChange={(event) => setField('kline_backend', event.target.value)}
        className="w-full rounded-2xl border border-white/10 bg-slate-950/72 px-3 py-2.5 text-sm text-slate-100 outline-none transition-colors focus:border-cyan-300/35 focus:bg-slate-950/90"
      >
        {BACKENDS.map((item) => (
          <option key={item} value={item}>
            {item}
          </option>
        ))}
      </select>
    </div>
  )
}
