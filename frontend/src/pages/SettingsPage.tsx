import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Save, CheckCircle, XCircle } from 'lucide-react';
import { fetchSettings, updateSettings } from '../api/settings';
import type { SettingsResponse } from '../types/api';

const TABS = ['API 密钥', '回测默认值', '数据库'] as const;
type Tab = typeof TABS[number];

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<Tab>('API 密钥');

  const { data: settings, isLoading } = useQuery({
    queryKey: ['settings'],
    queryFn: fetchSettings,
  });

  if (isLoading) return <p className="text-gray-400">正在加载系统设置...</p>;
  if (!settings) return <p className="text-gray-400">设置加载失败</p>;

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
  );
}

function APIKeysTab({ settings }: { settings: SettingsResponse }) {
  const queryClient = useQueryClient();
  const [values, setValues] = useState<Record<string, string>>({});
  const [saved, setSaved] = useState(false);

  const mutation = useMutation({
    mutationFn: (updates: Record<string, string>) => updateSettings(updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
      setValues({});
    },
  });

  const keyFields = [
    { envKey: 'tushare_token', label: 'Tushare Token' },
    { envKey: 'openai_api_key', label: 'OpenAI API Key' },
    { envKey: 'anthropic_api_key', label: 'Anthropic API Key' },
    { envKey: 'deepseek_api_key', label: 'DeepSeek API Key' },
    { envKey: 'google_api_key', label: 'Google API Key' },
    { envKey: 'fred_api_key', label: 'FRED API Key' },
    { envKey: 'finnhub_api_key', label: 'Finnhub API Key' },
    { envKey: 'dashscope_api_key', label: 'Dashscope API Key' },
  ];

  const handleSave = () => {
    const nonEmpty = Object.fromEntries(
      Object.entries(values).filter(([, v]) => v.trim())
    );
    if (Object.keys(nonEmpty).length > 0) {
      mutation.mutate(nonEmpty);
    }
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 max-w-2xl">
      <div className="space-y-4">
        {keyFields.map((field) => {
          const cred = settings.credentials.find((c) => c.env_key === field.envKey.toUpperCase());
          const isSet = cred?.is_set;

          return (
            <div key={field.envKey}>
              <div className="flex items-center gap-2 mb-1">
                <label className="text-sm font-medium text-gray-700">{field.label}</label>
                {isSet ? (
                  <CheckCircle size={14} className="text-green-500" />
                ) : (
                  <XCircle size={14} className="text-gray-300" />
                )}
              </div>
              <input
                type="password"
                placeholder={isSet ? cred?.masked_value || '********' : '未设置'}
                value={values[field.envKey] || ''}
                onChange={(e) =>
                  setValues((prev) => ({ ...prev, [field.envKey]: e.target.value }))
                }
                className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-3 mt-6">
        <button
          onClick={handleSave}
          disabled={mutation.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
        >
          <Save size={16} />
          {mutation.isPending ? '保存中...' : '保存修改'}
        </button>
        {saved && <span className="text-sm text-green-600">已保存</span>}
      </div>
    </div>
  );
}

function BacktestTab({ settings }: { settings: SettingsResponse }) {
  const queryClient = useQueryClient();
  const [values, setValues] = useState({
    initial_cash: String(settings.backtest.initial_cash),
    commission_rate: String(settings.backtest.commission_rate),
    stamp_duty_rate: String(settings.backtest.stamp_duty_rate),
    slippage: String(settings.backtest.slippage),
  });
  const [saved, setSaved] = useState(false);

  const mutation = useMutation({
    mutationFn: (updates: Record<string, unknown>) => updateSettings(updates),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    },
  });

  const handleSave = () => {
    mutation.mutate({
      initial_cash: parseFloat(values.initial_cash),
      commission_rate: parseFloat(values.commission_rate),
      stamp_duty_rate: parseFloat(values.stamp_duty_rate),
      slippage: parseFloat(values.slippage),
    });
  };

  const fields = [
    { key: 'initial_cash', label: '初始资金', suffix: 'CNY' },
    { key: 'commission_rate', label: '佣金费率', suffix: '' },
    { key: 'stamp_duty_rate', label: '印花税率', suffix: '' },
    { key: 'slippage', label: '滑点', suffix: '' },
  ] as const;

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 max-w-md">
      <div className="space-y-4">
        {fields.map((field) => (
          <div key={field.key}>
            <label className="text-sm font-medium text-gray-700 block mb-1">{field.label}</label>
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={values[field.key]}
                onChange={(e) => setValues((prev) => ({ ...prev, [field.key]: e.target.value }))}
                className="flex-1 px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              {field.suffix && <span className="text-sm text-gray-500">{field.suffix}</span>}
            </div>
          </div>
        ))}
      </div>
      <div className="flex items-center gap-3 mt-6">
        <button
          onClick={handleSave}
          disabled={mutation.isPending}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50"
        >
          <Save size={16} />
          {mutation.isPending ? '保存中...' : '保存修改'}
        </button>
        {saved && <span className="text-sm text-green-600">已保存</span>}
      </div>
    </div>
  );
}

function DatabaseTab({ settings }: { settings: SettingsResponse }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6 max-w-md">
      <div className="space-y-3">
        <div>
          <p className="text-sm text-gray-500">数据库路径</p>
          <p className="font-mono text-sm text-gray-900">{settings.db_path}</p>
        </div>
        <div>
          <p className="text-sm text-gray-500">日志级别</p>
          <p className="font-medium">{settings.log_level}</p>
        </div>
      </div>
    </div>
  );
}
