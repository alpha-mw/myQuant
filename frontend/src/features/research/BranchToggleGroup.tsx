import { useResearchStore } from '../../stores/researchStore'
import type { ResearchRunRequest } from '../../types/research'

const BRANCHES: {
  key: keyof Pick<
    ResearchRunRequest,
    'enable_macro' | 'enable_quant' | 'enable_kline' | 'enable_fundamental' | 'enable_intelligence'
  >
  label: string
}[] = [
  { key: 'enable_kline', label: 'K-Line' },
  { key: 'enable_quant', label: 'Quant' },
  { key: 'enable_fundamental', label: 'Fundamental' },
  { key: 'enable_intelligence', label: 'Intelligence' },
  { key: 'enable_macro', label: 'Macro' },
]

const BUTTON_CLASS_NAME =
  'rounded-2xl border px-3 py-3 text-left text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300'

export function BranchToggleGroup() {
  const store = useResearchStore()

  return (
    <div>
      <label className="mb-2 block text-[11px] font-medium uppercase tracking-[0.18em] text-slate-500">
        Research branches
      </label>
      <div className="grid gap-2 sm:grid-cols-2">
        {BRANCHES.map(({ key, label }) => {
          const enabled = store[key] as boolean

          return (
            <button
              key={key}
              type="button"
              onClick={() => store.setField(key, !enabled)}
              className={`${BUTTON_CLASS_NAME} ${
                enabled
                  ? 'border-emerald-300/20 bg-emerald-300/10 text-white hover:border-emerald-300/30'
                  : 'border-white/10 bg-white/[0.03] text-slate-400 hover:border-white/16 hover:bg-white/[0.05] hover:text-slate-200'
              }`}
            >
              <span className="flex items-center gap-2">
                <span
                  className={`inline-flex h-2.5 w-2.5 rounded-full ${
                    enabled ? 'bg-emerald-300 shadow-[0_0_12px_rgba(110,231,183,0.4)]' : 'bg-slate-600'
                  }`}
                  aria-hidden="true"
                />
                <span className="font-medium">{label}</span>
              </span>
              <span className={`mt-2 block text-xs leading-5 ${enabled ? 'text-slate-100/80' : 'text-slate-500'}`}>
                {enabled ? 'Included in the evidence set.' : 'Excluded from this run.'}
              </span>
            </button>
          )
        })}

        <button
          type="button"
          onClick={() => store.setField('enable_agent_layer', !store.enable_agent_layer)}
          className={`${BUTTON_CLASS_NAME} ${
            store.enable_agent_layer
              ? 'border-cyan-300/22 bg-cyan-300/10 text-white hover:border-cyan-300/30'
              : 'border-white/10 bg-white/[0.03] text-slate-400 hover:border-white/16 hover:bg-white/[0.05] hover:text-slate-200'
          }`}
        >
          <span className="flex items-center gap-2">
            <span
              className={`inline-flex h-2.5 w-2.5 rounded-full ${
                store.enable_agent_layer ? 'bg-cyan-300 shadow-[0_0_12px_rgba(103,232,249,0.36)]' : 'bg-slate-600'
              }`}
              aria-hidden="true"
            />
            <span className="font-medium">Agent Review</span>
          </span>
          <span className={`mt-2 block text-xs leading-5 ${store.enable_agent_layer ? 'text-slate-100/80' : 'text-slate-500'}`}>
            {store.enable_agent_layer
              ? 'Keep the review layer online before the control chain.'
              : 'Run the deterministic chain without agent review.'}
          </span>
        </button>
      </div>
    </div>
  )
}
