import { useQuery } from '@tanstack/react-query'
import { getLLMModels } from '../../api/settings'
import { useResearchStore } from '../../stores/researchStore'

const FIELD_CLASS_NAME =
  'w-full rounded-2xl border border-white/10 bg-slate-950/72 px-3 py-2.5 text-sm text-slate-100 outline-none transition-colors focus:border-cyan-300/35 focus:bg-slate-950/90'
const LABEL_CLASS_NAME = 'mb-2 block text-[11px] font-medium uppercase tracking-[0.18em] text-slate-500'

export function LLMModelSelector() {
  const agentModel = useResearchStore((state) => state.agent_model)
  const masterModel = useResearchStore((state) => state.master_model)
  const setField = useResearchStore((state) => state.setField)

  const { data } = useQuery({
    queryKey: ['settings', 'models'],
    queryFn: getLLMModels,
    staleTime: 60_000,
  })

  const available = data?.models?.filter((model) => model.available) ?? []

  return (
    <div className="grid gap-3">
      <div>
        <label className={LABEL_CLASS_NAME} htmlFor="branch-agent-model">
          Branch agent model
        </label>
        <select
          id="branch-agent-model"
          name="agent_model"
          value={agentModel}
          onChange={(event) => setField('agent_model', event.target.value)}
          className={FIELD_CLASS_NAME}
        >
          <option value="">default (claude-sonnet-4-6)</option>
          {available.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label} - ${model.prompt_price}/1M
            </option>
          ))}
        </select>
      </div>

      <div>
        <label className={LABEL_CLASS_NAME} htmlFor="master-ic-model">
          Master IC model
        </label>
        <select
          id="master-ic-model"
          name="master_model"
          value={masterModel}
          onChange={(event) => setField('master_model', event.target.value)}
          className={FIELD_CLASS_NAME}
        >
          <option value="">default (gpt-5.4-mini)</option>
          {available.map((model) => (
            <option key={model.id} value={model.id}>
              {model.label} - ${model.prompt_price}/1M
            </option>
          ))}
        </select>
      </div>
    </div>
  )
}
