import { useEffect, useMemo, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { CheckCircle, RotateCcw, Save, XCircle } from 'lucide-react'
import { fetchSettings, updateSettings } from '../api/settings'
import type { SettingsResponse, SettingsUpdateRequest } from '../types/settings'

const TABS = ['API 密钥', '回测默认值', '数据库'] as const
type Tab = typeof TABS[number]

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

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<Tab>('API 密钥')
  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: fetchSettings,
  })

  if (isLoading) return <p className="text-gray-400">正在加载系统设置...</p>
  if (!settings) return <p className="text-gray-400">设置加载失败</p>

  return (
    <div>
      <h1 className="text-2xl font-bold text-gray-900 mb-6">系统设置</h1>

      <div className="flex gap-1 border-b border-gray-200 mb-6">
        {TABS.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 -mb-px transition-colors ${
              activeTab === tab
                ? 'text-blue-600 border-blue-600'
                : 'text-gray-500 border-transparent hover:text-gray-700'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {activeTab === 'API 密钥' && <APIKeysTab settings={settings} />}
      {activeTab === '回测默认值' && <BacktestTab settings={settings} />}
      {activeTab === '数据库' && <DatabaseTab settings={settings} />}
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

  useEffect(() => {
    setDraft({})
    setClearKeys([])
  }, [settings])

  const credentialMap = useMemo(
    () => new Map(settings.credentials.map((credential) => [credential.env_key, credential])),
    [settings.credentials],
  )

  const pendingPayload = buildApiKeyPayload(draft, clearKeys)
  const hasChanges = Object.keys(pendingPayload).length > 0

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 max-w-3xl">
      <div className="space-y-4">
        {KEY_FIELDS.map((field) => {
          const credential = credentialMap.get(field.envKey)
          const isClearing = clearKeys.includes(field.field)
          const currentValue = draft[field.field] ?? ''
          return (
            <div key={field.field} className="space-y-1.5">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-gray-700">{field.label}</label>
                  {credential?.is_set ? (
                    <CheckCircle size={14} className="text-green-500" />
                  ) : (
                    <XCircle size={14} className="text-gray-300" />
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => {
                    setDraft((prev) => ({ ...prev, [field.field]: '' }))
                    setClearKeys((prev) =>
                      prev.includes(field.field) ? prev : [...prev, field.field],
                    )
                  }}
                  className="text-xs text-gray-500 hover:text-red-500"
                >
                  清空
                </button>
              </div>
              <input
                type="password"
                value={currentValue}
                placeholder={
                  isClearing
                    ? '保存后清空'
                    : credential?.is_set
                      ? credential.masked_value || '********'
                      : '未设置'
                }
                onChange={(event) => {
                  const nextValue = event.target.value
                  setDraft((prev) => ({ ...prev, [field.field]: nextValue }))
                  if (nextValue.trim()) {
                    setClearKeys((prev) => prev.filter((item) => item !== field.field))
                  }
                }}
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
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
          if (hasChanges) mutation.mutate(pendingPayload)
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

  useEffect(() => {
    setDraft({
      initial_cash: String(settings.backtest.initial_cash),
      commission_rate: String(settings.backtest.commission_rate),
      stamp_duty_rate: String(settings.backtest.stamp_duty_rate),
      slippage: String(settings.backtest.slippage),
      log_level: settings.log_level,
    })
  }, [settings])

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
    <div className="bg-white rounded-lg border border-gray-200 p-6 max-w-xl">
      <div className="space-y-4">
        <NumericField
          label="初始资金"
          value={draft.initial_cash}
          suffix="CNY"
          onChange={(value) => setDraft((prev) => ({ ...prev, initial_cash: value }))}
        />
        <NumericField
          label="佣金费率"
          value={draft.commission_rate}
          onChange={(value) => setDraft((prev) => ({ ...prev, commission_rate: value }))}
        />
        <NumericField
          label="印花税率"
          value={draft.stamp_duty_rate}
          onChange={(value) => setDraft((prev) => ({ ...prev, stamp_duty_rate: value }))}
        />
        <NumericField
          label="滑点"
          value={draft.slippage}
          onChange={(value) => setDraft((prev) => ({ ...prev, slippage: value }))}
        />
        <div>
          <label className="text-sm font-medium text-gray-700 block mb-1">日志级别</label>
          <select
            value={draft.log_level}
            onChange={(event) => setDraft((prev) => ({ ...prev, log_level: event.target.value }))}
            className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {LOG_LEVELS.map((level) => (
              <option key={level} value={level}>
                {level}
              </option>
            ))}
          </select>
        </div>
      </div>
      <SaveActions
        hasChanges={hasChanges}
        isPending={mutation.isPending}
        saved={saved}
        onSave={() => {
          if (hasChanges) mutation.mutate(pendingPayload)
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
    <div className="bg-white rounded-lg border border-gray-200 p-6 max-w-lg space-y-4">
      <div>
        <p className="text-sm text-gray-500">数据库路径</p>
        <p className="font-mono text-sm text-gray-900">{settings.db_path}</p>
      </div>
      <div>
        <p className="text-sm text-gray-500">当前日志级别</p>
        <p className="font-medium text-gray-900">{settings.log_level}</p>
      </div>
      <p className="text-xs text-gray-500">
        API Key、回测默认值和日志级别请在上方两个标签页中修改并保存。
      </p>
    </div>
  )
}

function NumericField(props: {
  label: string
  value: string
  suffix?: string
  onChange: (value: string) => void
}) {
  return (
    <div>
      <label className="text-sm font-medium text-gray-700 block mb-1">{props.label}</label>
      <div className="flex items-center gap-2">
        <input
          type="text"
          value={props.value}
          onChange={(event) => props.onChange(event.target.value)}
          className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        {props.suffix && <span className="text-sm text-gray-500">{props.suffix}</span>}
      </div>
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
    <div className="flex items-center gap-3 mt-6">
      <button
        onClick={props.onSave}
        disabled={!props.hasChanges || props.isPending}
        className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
      >
        <Save size={16} />
        {props.isPending ? '保存中...' : '保存修改'}
      </button>
      <button
        onClick={props.onCancel}
        disabled={props.isPending}
        className="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded-lg text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-50"
      >
        <RotateCcw size={16} />
        取消
      </button>
      {props.saved && <span className="text-sm text-green-600">已保存</span>}
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
