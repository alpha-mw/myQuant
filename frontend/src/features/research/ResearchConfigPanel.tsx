import { Settings2, TimerReset, Waypoints } from 'lucide-react'
import { useResearchStore } from '../../stores/researchStore'
import { StockPoolInput } from './StockPoolInput'
import { MarketSwitcher } from './MarketSwitcher'
import { RiskLevelSelector } from './RiskLevelSelector'
import { BranchToggleGroup } from './BranchToggleGroup'
import { KlineBackendSelector } from './KlineBackendSelector'
import { LLMModelSelector } from './LLMModelSelector'
import { PresetSelector } from './PresetSelector'
import { RunButton } from './RunButton'
import type { ResearchRunRequest } from '../../types/research'

interface Props {
  submit: (req: ResearchRunRequest) => void
  isSubmitting: boolean
  isRunning: boolean
}

const SECTION_CLASS_NAME = 'border-t border-white/8 px-4 py-4 first:border-t-0 lg:px-5'
const FIELD_CLASS_NAME =
  'w-full rounded-2xl border border-white/10 bg-slate-950/72 px-3 py-2.5 text-sm text-slate-100 outline-none transition-colors placeholder:text-slate-500 focus:border-cyan-300/35 focus:bg-slate-950/90'
const LABEL_CLASS_NAME = 'mb-2 block text-[11px] font-medium uppercase tracking-[0.18em] text-slate-500'

export function ResearchConfigPanel({ submit, isSubmitting, isRunning }: Props) {
  const { capital, lookback_years, agent_timeout, master_timeout, setField } = useResearchStore()

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="border-b border-white/8 px-4 py-4 lg:px-5 lg:py-5">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.24em] text-cyan-200/65">
          <Settings2 size={14} aria-hidden="true" />
          Config dock
        </div>
        <h2 className="mt-2 text-lg font-semibold text-white">Research inputs</h2>
        <p className="mt-2 text-sm leading-6 text-slate-400">
          Shape the stock pool, risk posture, model stack, and runtime limits before launching the mainline pipeline.
        </p>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <section className={SECTION_CLASS_NAME}>
          <SectionHeading
            title="Preset memory"
            description="Reuse saved workspace configurations or capture the current setup as a new preset."
            icon={Waypoints}
          />
          <PresetSelector />
        </section>

        <section className={SECTION_CLASS_NAME}>
          <SectionHeading
            title="Selection"
            description="Load a universe, browse the symbol list, or paste a custom stock basket."
            icon={Waypoints}
          />
          <StockPoolInput />
        </section>

        <section className={SECTION_CLASS_NAME}>
          <SectionHeading
            title="Risk posture"
            description="Confirm the market scope, risk level, and capital before the run begins."
            icon={Settings2}
          />
          <div className="grid gap-3">
            <div className="grid gap-3 sm:grid-cols-2">
              <MarketSwitcher />
              <RiskLevelSelector />
            </div>

            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <label className={LABEL_CLASS_NAME} htmlFor="research-capital">
                  Capital
                </label>
                <input
                  autoComplete="off"
                  id="research-capital"
                  name="capital"
                  type="number"
                  value={capital}
                  onChange={(event) => setField('capital', Number(event.target.value))}
                  className={FIELD_CLASS_NAME}
                />
              </div>

              <div>
                <label className={LABEL_CLASS_NAME} htmlFor="research-lookback">
                  Lookback years
                </label>
                <input
                  autoComplete="off"
                  id="research-lookback"
                  name="lookback_years"
                  type="number"
                  step="0.5"
                  value={lookback_years}
                  onChange={(event) => setField('lookback_years', Number(event.target.value))}
                  className={FIELD_CLASS_NAME}
                />
              </div>
            </div>
          </div>
        </section>

        <section className={SECTION_CLASS_NAME}>
          <SectionHeading
            title="Branch coverage"
            description="Choose which research agents contribute evidence and whether the review layer stays online."
            icon={Waypoints}
          />
          <BranchToggleGroup />
        </section>

        <section className={SECTION_CLASS_NAME}>
          <SectionHeading
            title="Execution models"
            description="Define the K-Line engine plus optional model overrides for the branch and IC layers."
            icon={Settings2}
          />
          <div className="grid gap-3">
            <KlineBackendSelector />
            <LLMModelSelector />
          </div>
        </section>

        <section className={SECTION_CLASS_NAME}>
          <SectionHeading
            title="Runtime limits"
            description="Keep hard time caps explicit so long reasoning runs can finish, while true failures still terminate."
            icon={TimerReset}
          />
          <div className="grid gap-3 sm:grid-cols-2">
            <div>
              <label className={LABEL_CLASS_NAME} htmlFor="research-agent-timeout">
                Agent timeout (s)
              </label>
              <input
                autoComplete="off"
                id="research-agent-timeout"
                name="agent_timeout"
                type="number"
                value={agent_timeout}
                onChange={(event) => setField('agent_timeout', Number(event.target.value))}
                className={FIELD_CLASS_NAME}
              />
            </div>

            <div>
              <label className={LABEL_CLASS_NAME} htmlFor="research-master-timeout">
                Master timeout (s)
              </label>
              <input
                autoComplete="off"
                id="research-master-timeout"
                name="master_timeout"
                type="number"
                value={master_timeout}
                onChange={(event) => setField('master_timeout', Number(event.target.value))}
                className={FIELD_CLASS_NAME}
              />
            </div>
          </div>
        </section>
      </div>

      <div className="border-t border-white/8 bg-slate-950/72 px-4 py-4 lg:px-5">
        <RunButton submit={submit} isSubmitting={isSubmitting} isRunning={isRunning} />
      </div>
    </div>
  )
}

function SectionHeading({
  title,
  description,
  icon: Icon,
}: {
  title: string
  description: string
  icon: typeof Settings2
}) {
  return (
    <div className="mb-4">
      <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
        <Icon size={14} aria-hidden="true" />
        {title}
      </div>
      <p className="mt-2 text-sm leading-6 text-slate-400">{description}</p>
    </div>
  )
}
