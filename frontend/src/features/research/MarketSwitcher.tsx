import { useResearchStore } from '../../stores/researchStore'

export function MarketSwitcher() {
  const market = useResearchStore((state) => state.market)
  const setField = useResearchStore((state) => state.setField)

  return (
    <div>
      <label className="mb-2 block text-[11px] font-medium uppercase tracking-[0.18em] text-slate-500">
        Market
      </label>
      <div className="grid grid-cols-2 gap-2">
        {(['CN', 'US'] as const).map((item) => (
          <button
            key={item}
            type="button"
            onClick={() => setField('market', item)}
            className={`rounded-2xl border px-3 py-3 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300 ${
              market === item
                ? 'border-cyan-300/22 bg-cyan-300/10 text-white'
                : 'border-white/10 bg-white/[0.03] text-slate-400 hover:border-white/16 hover:bg-white/[0.05] hover:text-slate-200'
            }`}
          >
            <span className="block font-medium">{item === 'CN' ? 'A股 CN' : 'US'}</span>
            <span className={`mt-1 block text-xs leading-5 ${market === item ? 'text-slate-100/80' : 'text-slate-500'}`}>
              {item === 'CN' ? 'China A-share universe' : 'US-listed universe'}
            </span>
          </button>
        ))}
      </div>
    </div>
  )
}
