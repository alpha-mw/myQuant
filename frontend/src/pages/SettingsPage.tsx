import { useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  CheckCircle2,
  Database,
  KeyRound,
  RotateCcw,
  Save,
  Settings2,
  SlidersHorizontal,
  XCircle,
} from 'lucide-react'
import { useSearchParams } from 'react-router-dom'
import { fetchSettings, updateSettings } from '../api/settings'
import { formatDateTime, formatInteger } from '../lib/format'
import type { SettingsResponse, SettingsUpdateRequest } from '../types/settings'

const TABS = [
  {
    id: 'credentials',
    label: 'API 密钥',
    description: 'Update provider tokens without leaving the workspace.',
    icon: KeyRound,
  },
  {
    id: 'backtest',
    label: '回测默认值',
    description: 'Adjust default capital, fees, slippage, and log verbosity.',
    icon: SlidersHorizontal,
  },
  {
    id: 'database',
    label: '数据库',
    description: 'Inspect the stock DB path and workspace persistence health.',
    icon: Database,
  },
] as const

type TabId = (typeof TABS)[number]['id']

const KEY_FIELDS = [
  { field: 'tushare_token', envKey: 'TUSHARE_TOKEN', label: 'Tushare Token' },
  { field: 'openai_api_key', envKey: 'OPENAI_API_KEY', label: 'OpenAI API Key' },
  { field: 'anthropic_api_key', envKey: 'ANTHROPIC_API_KEY', label: 'Anthropic API Key' },
  { field: 'deepseek_api_key', envKey: 'DEEPSEEK_API_KEY', label: 'DeepSeek API Key' },
  { field: 'google_api_key', envKey: 'GOOGLE_API_KEY', label: 'Google API Key' },
  { field: 'fred_api_key', envKey: 'FRED_API_KEY', label: 'FRED API Key' },
  { field: 'finnhub_api_key', envKey: 'FINNHUB_API_KEY', label: 'Finnhub API Key' },
  { field: 'dashscope_api_key', envKey: 'DASHSCOPE_API_KEY', label: 'Dashscope API Key' },
  { field: 'kimi_api_key', envKey: 'KIMI_API_KEY', label: 'Moonshot Kimi API Key' },
] as const

const LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR'] as const

function parseActiveTab(value: string | null): TabId {
  return TABS.some((tab) => tab.id === value) ? (value as TabId) : 'credentials'
}

function formatBytes(value: number | null) {
  if (value == null) {
    return '—'
  }

  if (value < 1024) {
    return `${value} B`
  }

  const units = ['KB', 'MB', 'GB', 'TB']
  let size = value
  let unitIndex = -1
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024
    unitIndex += 1
  }

  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[unitIndex]}`
}

export default function SettingsPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const activeTab = parseActiveTab(searchParams.get('tab'))
  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: fetchSettings,
  })

  function setActiveTab(tabId: TabId) {
    const nextParams = new URLSearchParams(searchParams)
    if (tabId === 'credentials') {
      nextParams.delete('tab')
    } else {
      nextParams.set('tab', tabId)
    }
    setSearchParams(nextParams)
  }

  if (isLoading) {
    return <div className="px-4 py-6 text-sm text-slate-400 lg:px-6">正在加载系统设置…</div>
  }
  if (!settings) {
    return <div className="px-4 py-6 text-sm text-red-300 lg:px-6">设置加载失败。</div>
  }

  const apiKeysStateKey = settings.credentials
    .map((credential) => `${credential.env_key}:${credential.is_set ? 1 : 0}:${credential.masked_value ?? ''}`)
    .join('|')
  const backtestStateKey = [
    settings.backtest.initial_cash,
    settings.backtest.commission_rate,
    settings.backtest.stamp_duty_rate,
    settings.backtest.slippage,
    settings.log_level,
  ].join('|')

  return (
    <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4 lg:px-6">
      <header className="rounded-[2rem] border border-white/10 bg-white/[0.03] p-6">
        <div className="flex items-center gap-2 text-[11px] uppercase tracking-[0.24em] text-teal-300/70">
          <Settings2 size={14} aria-hidden="true" />
          Settings
        </div>
        <h1 className="mt-3 text-2xl font-semibold text-white">Configure the Workspace</h1>
        <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
          Manage provider credentials, default research parameters, and the persistence layer used by the research workspace.
        </p>
      </header>

      <div className="mt-4 grid gap-3 lg:grid-cols-3">
        {TABS.map((tab) => {
          const Icon = tab.icon
          return (
            <button
              key={tab.id}
              type="button"
              onClick={() => setActiveTab(tab.id)}
              className={`rounded-[1.6rem] border p-4 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300 ${
                activeTab === tab.id
                  ? 'border-teal-400/30 bg-teal-400/10 text-white'
                  : 'border-white/10 bg-white/[0.03] text-slate-300 hover:border-white/16 hover:bg-white/[0.05]'
              }`}
            >
              <div className="flex items-center gap-2 text-sm font-medium">
                <Icon size={16} aria-hidden="true" />
                {tab.label}
              </div>
              <p className="mt-2 text-sm leading-6 text-slate-400">{tab.description}</p>
            </button>
          )
        })}
      </div>

      <section className="mt-4 rounded-[2rem] border border-white/10 bg-slate-950/55 p-6">
        {activeTab === 'credentials' && <APIKeysTab key={apiKeysStateKey} settings={settings} />}
        {activeTab === 'backtest' && <BacktestTab key={backtestStateKey} settings={settings} />}
        {activeTab === 'database' && <DatabaseTab settings={settings} />}
      </section>
    </div>
  )
}

function APIKeysTab({ settings }: { settings: SettingsResponse }) {
  const queryClient = useQueryClient()
  const [draft, setDraft] = useState<Record<string, string>>({})
  const [clearKeys, setClearKeys] = useState<string[]>([])
  const [saved, setSaved] = useState(false)

  const mutation = useMutation({
    mutationFn: (updates: SettingsUpdateRequest) => updateSettings(updates),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['settings'] })
      setDraft({})
      setClearKeys([])
      setSaved(true)
      window.setTimeout(() => setSaved(false), 2000)
    },
  })

  const credentialMap = useMemo(
    () => new Map(settings.credentials.map((credential) => [credential.env_key, credential])),
    [settings.credentials],
  )

  const pendingPayload = buildApiKeyPayload(draft, clearKeys)
  const hasChanges = Object.keys(pendingPayload).length > 0

  return (
    <div>
      <div className="max-w-3xl">
        <h2 className="text-lg font-semibold text-white">Provider Credentials</h2>
        <p className="mt-2 text-sm leading-6 text-slate-400">
          Tokens are stored in the local workspace environment file and masked on readback. Empty values are ignored until you explicitly clear them.
        </p>
      </div>

      <div className="mt-6 space-y-4">
        {KEY_FIELDS.map((field) => {
          const credential = credentialMap.get(field.envKey)
          const isClearing = clearKeys.includes(field.field)
          const currentValue = draft[field.field] ?? ''
          const inputId = `credential-${field.field}`

          return (
            <div key={field.field} className="rounded-[1.5rem] border border-white/10 bg-white/[0.02] p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <label htmlFor={inputId} className="text-sm font-medium text-white">
                    {field.label}
                  </label>
                  <div className="mt-1 flex items-center gap-2 text-xs text-slate-400">
                    {credential?.is_set ? (
                      <>
                        <CheckCircle2 size={14} className="text-emerald-300" aria-hidden="true" />
                        已配置
                      </>
                    ) : (
                      <>
                        <XCircle size={14} className="text-slate-500" aria-hidden="true" />
                        未配置
                      </>
                    )}
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setDraft((prev) => ({ ...prev, [field.field]: '' }))
                    setClearKeys((prev) =>
                      prev.includes(field.field) ? prev : [...prev, field.field],
                    )
                  }}
                  className="rounded-full border border-white/10 px-3 py-1.5 text-xs font-medium text-slate-300 transition-colors hover:border-red-400/30 hover:bg-red-500/10 hover:text-red-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
                >
                  清空
                </button>
              </div>

              <input
                id={inputId}
                name={field.field}
                type="password"
                value={currentValue}
                autoComplete="off"
                spellCheck={false}
                placeholder={
                  isClearing
                    ? '保存后清空…'
                    : credential?.is_set
                      ? credential.masked_value || '********'
                      : '粘贴新的密钥…'
                }
                onChange={(event) => {
                  const nextValue = event.target.value
                  setDraft((prev) => ({ ...prev, [field.field]: nextValue }))
                  if (nextValue.trim()) {
                    setClearKeys((prev) => prev.filter((item) => item !== field.field))
                  }
                }}
                className="mt-3 w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
              />
            </div>
          )
        })}
      </div>

      <SaveActions
        hasChanges={hasChanges}
        isPending={mutation.isPending}
        saved={saved}
        onSave={() => {
          if (hasChanges) {
            mutation.mutate(pendingPayload)
          }
        }}
        onCancel={() => {
          setDraft({})
          setClearKeys([])
        }}
      />
    </div>
  )
}

function BacktestTab({ settings }: { settings: SettingsResponse }) {
  const queryClient = useQueryClient()
  const [draft, setDraft] = useState({
    initial_cash: String(settings.backtest.initial_cash),
    commission_rate: String(settings.backtest.commission_rate),
    stamp_duty_rate: String(settings.backtest.stamp_duty_rate),
    slippage: String(settings.backtest.slippage),
    log_level: settings.log_level,
  })
  const [saved, setSaved] = useState(false)

  const mutation = useMutation({
    mutationFn: (updates: SettingsUpdateRequest) => updateSettings(updates),
    onSuccess: () => {
      void queryClient.invalidateQueries({ queryKey: ['settings'] })
      setSaved(true)
      window.setTimeout(() => setSaved(false), 2000)
    },
  })

  const pendingPayload = buildBacktestPayload(draft, settings)
  const hasChanges = Object.keys(pendingPayload).length > 0

  return (
    <div className="max-w-2xl">
      <h2 className="text-lg font-semibold text-white">Backtest Defaults</h2>
      <p className="mt-2 text-sm leading-6 text-slate-400">
        These values pre-fill new workspace runs and remain editable before each submission.
      </p>

      <div className="mt-6 grid gap-4 md:grid-cols-2">
        <NumericField
          name="initial_cash"
          label="初始资金"
          value={draft.initial_cash}
          suffix="CNY"
          onChange={(value) => setDraft((prev) => ({ ...prev, initial_cash: value }))}
        />
        <NumericField
          name="commission_rate"
          label="佣金费率"
          value={draft.commission_rate}
          onChange={(value) => setDraft((prev) => ({ ...prev, commission_rate: value }))}
        />
        <NumericField
          name="stamp_duty_rate"
          label="印花税率"
          value={draft.stamp_duty_rate}
          onChange={(value) => setDraft((prev) => ({ ...prev, stamp_duty_rate: value }))}
        />
        <NumericField
          name="slippage"
          label="滑点"
          value={draft.slippage}
          onChange={(value) => setDraft((prev) => ({ ...prev, slippage: value }))}
        />
      </div>

      <div className="mt-4 max-w-sm rounded-[1.5rem] border border-white/10 bg-white/[0.02] p-4">
        <label htmlFor="log-level" className="block text-sm font-medium text-white">
          日志级别
        </label>
        <select
          id="log-level"
          name="log_level"
          value={draft.log_level}
          onChange={(event) => setDraft((prev) => ({ ...prev, log_level: event.target.value }))}
          className="mt-3 w-full rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
        >
          {LOG_LEVELS.map((level) => (
            <option key={level} value={level}>
              {level}
            </option>
          ))}
        </select>
      </div>

      <SaveActions
        hasChanges={hasChanges}
        isPending={mutation.isPending}
        saved={saved}
        onSave={() => {
          if (hasChanges) {
            mutation.mutate(pendingPayload)
          }
        }}
        onCancel={() => {
          setDraft({
            initial_cash: String(settings.backtest.initial_cash),
            commission_rate: String(settings.backtest.commission_rate),
            stamp_duty_rate: String(settings.backtest.stamp_duty_rate),
            slippage: String(settings.backtest.slippage),
            log_level: settings.log_level,
          })
        }}
      />
    </div>
  )
}

function DatabaseTab({ settings }: { settings: SettingsResponse }) {
  return (
    <div>
      <div className="max-w-3xl">
        <h2 className="text-lg font-semibold text-white">Database Health</h2>
        <p className="mt-2 text-sm leading-6 text-slate-400">
          The workspace persists run history, presets, and pending trade ideas in a dedicated SQLite database while market data remains in the stock DB path below.
        </p>
      </div>

      <div className="mt-6 grid gap-4 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <div className="rounded-[1.8rem] border border-white/10 bg-white/[0.02] p-5">
          <div className="flex items-center gap-2 text-sm font-medium text-white">
            <Database size={16} aria-hidden="true" />
            Workspace Persistence
          </div>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            <StatTile label="Runs" value={formatInteger(settings.workspace_db.run_count)} />
            <StatTile label="Completed" value={formatInteger(settings.workspace_db.completed_runs)} />
            <StatTile label="Failed" value={formatInteger(settings.workspace_db.failed_runs)} />
            <StatTile label="Presets" value={formatInteger(settings.workspace_db.preset_count)} />
            <StatTile label="Pending Trades" value={formatInteger(settings.workspace_db.pending_trades)} />
            <StatTile label="DB Size" value={formatBytes(settings.workspace_db.size_bytes)} />
          </div>
          <dl className="mt-5 space-y-3 text-sm text-slate-300">
            <MetadataRow label="Workspace DB Path" value={settings.workspace_db.path} mono />
            <MetadataRow label="Exists" value={settings.workspace_db.exists ? 'Yes' : 'No'} />
            <MetadataRow
              label="Last Run"
              value={settings.workspace_db.last_run_at ? formatDateTime(settings.workspace_db.last_run_at) : '—'}
            />
            <MetadataRow
              label="Last Modified"
              value={settings.workspace_db.modified_at ? formatDateTime(settings.workspace_db.modified_at) : '—'}
            />
          </dl>
        </div>

        <div className="rounded-[1.8rem] border border-white/10 bg-white/[0.02] p-5">
          <div className="flex items-center gap-2 text-sm font-medium text-white">
            <Database size={16} aria-hidden="true" />
            Stock Data Store
          </div>
          <dl className="mt-5 space-y-3 text-sm text-slate-300">
            <MetadataRow label="Configured Path" value={settings.db_path} mono />
            <MetadataRow label="Resolved Path" value={settings.stock_db.path} mono />
            <MetadataRow label="Exists" value={settings.stock_db.exists ? 'Yes' : 'No'} />
            <MetadataRow label="DB Size" value={formatBytes(settings.stock_db.size_bytes)} />
            <MetadataRow
              label="Last Modified"
              value={settings.stock_db.modified_at ? formatDateTime(settings.stock_db.modified_at) : '—'}
            />
            <MetadataRow label="Log Level" value={settings.log_level} />
          </dl>
        </div>
      </div>
    </div>
  )
}

function NumericField(props: {
  name: string
  label: string
  value: string
  suffix?: string
  onChange: (value: string) => void
}) {
  const inputId = `settings-${props.name}`

  return (
    <div className="rounded-[1.5rem] border border-white/10 bg-white/[0.02] p-4">
      <label htmlFor={inputId} className="block text-sm font-medium text-white">
        {props.label}
      </label>
      <div className="mt-3 flex items-center gap-3">
        <input
          id={inputId}
          name={props.name}
          type="number"
          inputMode="decimal"
          value={props.value}
          onChange={(event) => props.onChange(event.target.value)}
          className="min-w-0 flex-1 rounded-2xl border border-white/10 bg-slate-950/60 px-4 py-3 text-sm text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
        />
        {props.suffix && <span className="text-sm text-slate-400">{props.suffix}</span>}
      </div>
    </div>
  )
}

function StatTile(props: { label: string; value: string }) {
  return (
    <div className="rounded-[1.2rem] border border-white/8 bg-slate-950/60 p-4">
      <p className="text-xs uppercase tracking-[0.18em] text-slate-500">{props.label}</p>
      <p className="mt-2 text-xl font-semibold text-white">{props.value}</p>
    </div>
  )
}

function MetadataRow(props: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-[0.18em] text-slate-500">{props.label}</dt>
      <dd className={`mt-1 break-words ${props.mono ? 'font-mono text-xs text-slate-300' : 'text-slate-300'}`}>
        {props.value}
      </dd>
    </div>
  )
}

function SaveActions(props: {
  hasChanges: boolean
  isPending: boolean
  saved: boolean
  onSave: () => void
  onCancel: () => void
}) {
  return (
    <div className="mt-6 flex flex-wrap items-center gap-3">
      <button
        type="button"
        onClick={props.onSave}
        disabled={!props.hasChanges || props.isPending}
        className="inline-flex items-center gap-2 rounded-full bg-teal-400 px-4 py-2 text-sm font-medium text-slate-950 transition-colors hover:bg-teal-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <Save size={16} aria-hidden="true" />
        {props.isPending ? '保存中…' : '保存修改'}
      </button>
      <button
        type="button"
        onClick={props.onCancel}
        disabled={props.isPending}
        className="inline-flex items-center gap-2 rounded-full border border-white/10 px-4 py-2 text-sm font-medium text-slate-200 transition-colors hover:border-white/16 hover:bg-white/[0.04] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <RotateCcw size={16} aria-hidden="true" />
        重置草稿
      </button>
      <span aria-live="polite" className="text-sm text-emerald-300">
        {props.saved ? '已保存。' : ''}
      </span>
    </div>
  )
}

function buildApiKeyPayload(
  draft: Record<string, string>,
  clearKeys: string[],
): SettingsUpdateRequest {
  const payload: SettingsUpdateRequest = {}
  for (const field of KEY_FIELDS) {
    const nextValue = draft[field.field]
    if (clearKeys.includes(field.field)) {
      payload[field.field] = ''
      continue
    }
    if (typeof nextValue === 'string' && nextValue.trim()) {
      payload[field.field] = nextValue.trim()
    }
  }
  return payload
}

function buildBacktestPayload(
  draft: {
    initial_cash: string
    commission_rate: string
    stamp_duty_rate: string
    slippage: string
    log_level: string
  },
  settings: SettingsResponse,
): SettingsUpdateRequest {
  const payload: SettingsUpdateRequest = {}
  const initialCash = Number(draft.initial_cash)
  const commissionRate = Number(draft.commission_rate)
  const stampDutyRate = Number(draft.stamp_duty_rate)
  const slippage = Number(draft.slippage)

  if (!Number.isNaN(initialCash) && initialCash !== settings.backtest.initial_cash) {
    payload.initial_cash = initialCash
  }
  if (!Number.isNaN(commissionRate) && commissionRate !== settings.backtest.commission_rate) {
    payload.commission_rate = commissionRate
  }
  if (!Number.isNaN(stampDutyRate) && stampDutyRate !== settings.backtest.stamp_duty_rate) {
    payload.stamp_duty_rate = stampDutyRate
  }
  if (!Number.isNaN(slippage) && slippage !== settings.backtest.slippage) {
    payload.slippage = slippage
  }
  if (draft.log_level !== settings.log_level) {
    payload.log_level = draft.log_level
  }
  return payload
}
