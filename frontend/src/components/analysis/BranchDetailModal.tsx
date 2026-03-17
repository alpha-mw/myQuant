import { useEffect, useRef, type MouseEvent } from 'react';
import { X } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
} from 'recharts';
import type { BranchDetailResult } from '../../types/api';

const branchTitleMap: Record<string, string> = {
  kline: 'K线分析',
  quant: '传统量化分支',
  llm_debate: 'LLM 多空辩论',
  intelligence: '多维智能融合',
  macro: '宏观分支',
};

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

interface Props {
  branch: BranchDetailResult;
  onClose: () => void;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asNumberRecord(value: unknown): Record<string, number> {
  return Object.fromEntries(
    Object.entries(asRecord(value)).filter(([, item]) => typeof item === 'number'),
  ) as Record<string, number>;
}

function asStringRecord(value: unknown): Record<string, string> {
  return Object.fromEntries(
    Object.entries(asRecord(value)).filter(([, item]) => typeof item === 'string'),
  ) as Record<string, string>;
}

function asNestedNumberRecord(value: unknown): Record<string, Record<string, number>> {
  return Object.fromEntries(
    Object.entries(asRecord(value)).map(([key, item]) => [key, asNumberRecord(item)]),
  );
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === 'string') : [];
}

function asNestedStringArrayRecord(value: unknown): Record<string, string[]> {
  return Object.fromEntries(
    Object.entries(asRecord(value)).map(([key, item]) => [key, asStringArray(item)]),
  );
}

function normalizeBranch(branch: BranchDetailResult): BranchDetailResult {
  return {
    ...branch,
    explanation: typeof branch.explanation === 'string' ? branch.explanation : '',
    risks: asStringArray(branch.risks),
    top_symbols: asStringArray(branch.top_symbols),
    settings: asRecord(branch.settings),
    model_assignment: Array.isArray(branch.model_assignment) ? branch.model_assignment : [],
    signals: asRecord(branch.signals),
    metadata: asRecord(branch.metadata),
  };
}

export default function BranchDetailModal({ branch, onClose }: Props) {
  const overlayRef = useRef<HTMLDivElement>(null);
  const safeBranch = normalizeBranch(branch);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [onClose]);

  function handleOverlayClick(e: MouseEvent<HTMLDivElement>) {
    if (e.target === overlayRef.current) onClose();
  }

  const hasSignals = Object.keys(safeBranch.signals).length > 0;

  return (
    <div
      ref={overlayRef}
      onClick={handleOverlayClick}
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/40 p-6 backdrop-blur-sm"
    >
      <div className="w-full max-w-4xl rounded-[28px] border border-[var(--line)] bg-[var(--surface)] p-6 shadow-xl">
        {/* Header */}
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="panel-kicker">分支详情</p>
            <h2 className="text-xl font-semibold text-[var(--ink)]">
              {branchTitleMap[safeBranch.branch_name] ?? safeBranch.branch_name}
            </h2>
            <div className="mt-1 text-sm text-[var(--muted)]">
              {safeBranch.enabled ? '已启用' : '未启用'} · 置信度 {formatPercent(safeBranch.confidence)} · 得分{' '}
              <span className={safeBranch.score >= 0 ? 'text-emerald-700' : 'text-rose-700'}>
                {safeBranch.score >= 0 ? '+' : ''}{safeBranch.score.toFixed(2)}
              </span>
            </div>
          </div>
          <button type="button" onClick={onClose} className="icon-button">
            <X size={20} />
          </button>
        </div>

        <div className="mt-6 space-y-6">
          {/* Explanation */}
          <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
            <div className="text-sm font-semibold text-[var(--ink)]">分析结论</div>
            <p className="mt-2 text-sm leading-7 text-[var(--muted)]">{safeBranch.explanation || '暂无分支解释。'}</p>
          </div>

          {/* Settings used */}
          {Object.keys(safeBranch.settings).length > 0 && (
            <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
              <div className="text-sm font-semibold text-[var(--ink)]">使用的配置</div>
              <div className="mt-3 grid gap-2 md:grid-cols-3">
                {Object.entries(safeBranch.settings).map(([key, value]) => (
                  <div key={key} className="rounded-[14px] bg-[rgba(12,33,60,0.04)] px-3 py-2 text-sm">
                    <span className="text-[var(--muted)]">{key}:</span>{' '}
                    <span className="font-semibold text-[var(--ink)]">{String(value)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Top symbols */}
          {safeBranch.top_symbols.length > 0 && (
            <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
              <div className="text-sm font-semibold text-[var(--ink)]">支持标的</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {safeBranch.top_symbols.map((symbol) => (
                  <span key={symbol} className="candidate-chip">{symbol}</span>
                ))}
              </div>
            </div>
          )}

          {/* Branch-specific visualizations */}
          {hasSignals ? (
            <BranchVisualization branch={safeBranch} />
          ) : (
            <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
              <div className="text-sm text-[var(--muted)]">
                该分析没有详细信号数据。新启动的分析将包含可视化数据。
              </div>
            </div>
          )}

          {/* Model assignments (LLM) */}
          {safeBranch.model_assignment.length > 0 && (
            <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
              <div className="text-sm font-semibold text-[var(--ink)]">模型分配</div>
              <div className="mt-3 flex flex-wrap gap-2">
                {safeBranch.model_assignment.map((item, i) => (
                  <span key={i} className="candidate-chip">
                    {String(item.model)} · {String(item.role)}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Risks */}
          {safeBranch.risks.length > 0 && (
            <div className="rounded-[20px] border border-[rgba(190,92,44,0.2)] bg-[rgba(190,92,44,0.06)] p-4">
              <div className="text-sm font-semibold text-[var(--danger)]">风险提示</div>
              <ul className="mt-2 space-y-1">
                {safeBranch.risks.map((risk, i) => (
                  <li key={i} className="text-sm text-[var(--danger)]">· {risk}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function BranchVisualization({ branch }: { branch: BranchDetailResult }) {
  switch (branch.branch_name) {
    case 'kline':
      return <KronosDetail signals={branch.signals} />;
    case 'quant':
      return <QuantDetail signals={branch.signals} />;
    case 'llm_debate':
      return <LlmDebateDetail signals={branch.signals} />;
    case 'intelligence':
      return <IntelligenceDetail signals={branch.signals} />;
    case 'macro':
      return <MacroDetail signals={branch.signals} />;
    default:
      return <GenericSignals signals={branch.signals} />;
  }
}

function KronosDetail({ signals }: { signals: Record<string, unknown> }) {
  const predictedReturn = asNumberRecord(signals.predicted_return);
  const trendRegime = asStringRecord(signals.trend_regime);
  const modelMode = signals.model_mode as string | undefined;

  const chartData = Object.keys(predictedReturn).length
    ? Object.entries(predictedReturn).map(([symbol, ret]) => ({
        symbol,
        predicted_return: Number((Number(ret) * 100).toFixed(2)),
        regime: trendRegime?.[symbol] ?? '未知',
      }))
    : [];

  return (
    <div className="space-y-4">
      <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
        <div className="text-sm font-semibold text-[var(--ink)]">Kronos 预测收益率</div>
        {modelMode && <div className="mt-1 text-xs text-[var(--muted)]">模型: {modelMode}</div>}
        {chartData.length > 0 ? (
          <div className="mt-4" style={{ height: 300 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} layout="vertical" margin={{ left: 80 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(12,33,60,0.08)" />
                <XAxis type="number" unit="%" tick={{ fontSize: 12 }} />
                <YAxis type="category" dataKey="symbol" tick={{ fontSize: 12 }} width={75} />
                <Tooltip
                  formatter={(value) => [`${value}%`, '预测收益率']}
                  contentStyle={{ borderRadius: 12, fontSize: 12 }}
                />
                <Bar dataKey="predicted_return" fill="#2e8b57" radius={[0, 6, 6, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="mt-4 text-sm text-[var(--muted)]">暂无预测数据。</div>
        )}
      </div>

      {chartData.length > 0 && (
        <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
          <div className="text-sm font-semibold text-[var(--ink)]">趋势状态</div>
          <div className="mt-3 grid gap-2 md:grid-cols-2">
            {chartData.map((item) => (
              <div key={item.symbol} className="flex items-center justify-between rounded-[14px] bg-[rgba(12,33,60,0.04)] px-3 py-2 text-sm">
                <span className="font-mono text-[var(--ink)]">{item.symbol}</span>
                <div className="flex items-center gap-3">
                  <span className={item.predicted_return >= 0 ? 'text-emerald-700' : 'text-rose-700'}>
                    {item.predicted_return >= 0 ? '+' : ''}{item.predicted_return}%
                  </span>
                  <span className="rounded-full bg-[rgba(12,33,60,0.08)] px-2 py-0.5 text-xs">{item.regime}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function QuantDetail({ signals }: { signals: Record<string, unknown> }) {
  const alphaFactors = asStringArray(signals.alpha_factors);
  const factorExposures = asNestedNumberRecord(signals.factor_exposures);
  const expectedReturn = asNumberRecord(signals.expected_return);
  const reasoning = signals.feature_reasoning as string | undefined;

  const chartData = Object.keys(factorExposures).length > 0
    ? Object.entries(factorExposures).flatMap(([symbol, factors]) =>
        Object.entries(factors).map(([factor, value]) => ({
          symbol,
          factor,
          exposure: Number(Number(value).toFixed(3)),
        })),
      )
    : [];

  // Group by symbol for bar chart
  const symbols = [...new Set(chartData.map((d) => d.symbol))];
  const factors = [...new Set(chartData.map((d) => d.factor))];
  const groupedData = factors.map((factor) => {
    const row: Record<string, string | number> = { factor };
    for (const sym of symbols) {
      const found = chartData.find((d) => d.symbol === sym && d.factor === factor);
      row[sym] = found?.exposure ?? 0;
    }
    return row;
  });

  const colors = ['#2e8b57', '#b56d2a', '#4a7fb5', '#c04040', '#8b5cf6'];

  return (
    <div className="space-y-4">
      {alphaFactors.length > 0 && (
        <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
          <div className="text-sm font-semibold text-[var(--ink)]">选中的 Alpha 因子</div>
          <div className="mt-3 flex flex-wrap gap-2">
            {alphaFactors.map((f) => (
              <span key={f} className="candidate-chip">{f}</span>
            ))}
          </div>
        </div>
      )}

      {groupedData.length > 0 && (
        <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
          <div className="text-sm font-semibold text-[var(--ink)]">因子暴露</div>
          <div className="mt-4" style={{ height: Math.max(250, factors.length * 40) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={groupedData} layout="vertical" margin={{ left: 100 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(12,33,60,0.08)" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis type="category" dataKey="factor" tick={{ fontSize: 11 }} width={95} />
                <Tooltip contentStyle={{ borderRadius: 12, fontSize: 12 }} />
                {symbols.map((sym, i) => (
                  <Bar key={sym} dataKey={sym} fill={colors[i % colors.length]} radius={[0, 4, 4, 0]} />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {Object.keys(expectedReturn).length > 0 && (
        <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
          <div className="text-sm font-semibold text-[var(--ink)]">预期收益</div>
          <div className="mt-3 grid gap-2 md:grid-cols-2">
            {Object.entries(expectedReturn).map(([symbol, ret]) => (
              <div key={symbol} className="flex items-center justify-between rounded-[14px] bg-[rgba(12,33,60,0.04)] px-3 py-2 text-sm">
                <span className="font-mono text-[var(--ink)]">{symbol}</span>
                <span className={Number(ret) >= 0 ? 'text-emerald-700' : 'text-rose-700'}>
                  {Number(ret) >= 0 ? '+' : ''}{(Number(ret) * 100).toFixed(2)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {reasoning && (
        <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
          <div className="text-sm font-semibold text-[var(--ink)]">因子选择逻辑</div>
          <p className="mt-2 text-sm leading-7 text-[var(--muted)]">{reasoning}</p>
        </div>
      )}
    </div>
  );
}

function LlmDebateDetail({ signals }: { signals: Record<string, unknown> }) {
  const bullCase = asNestedStringArrayRecord(signals.bull_case);
  const bearCase = asNestedStringArrayRecord(signals.bear_case);
  const keyRisks = asNestedStringArrayRecord(signals.key_risks);

  const symbols = [...new Set([
    ...Object.keys(bullCase),
    ...Object.keys(bearCase),
  ])];

  if (symbols.length === 0) {
    return (
      <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
        <div className="text-sm text-[var(--muted)]">暂无辩论详情数据。</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {symbols.map((symbol) => (
        <div key={symbol} className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
          <div className="text-sm font-semibold text-[var(--ink)]">{symbol}</div>
          <div className="mt-3 grid gap-4 md:grid-cols-2">
            <div className="rounded-[14px] bg-[rgba(46,139,87,0.06)] p-3">
              <div className="text-xs font-semibold text-emerald-700">多方观点</div>
              <ul className="mt-2 space-y-1">
                {(bullCase[symbol] ?? []).map((point, i) => (
                  <li key={i} className="text-sm text-emerald-800">+ {point}</li>
                ))}
                {!(bullCase[symbol]?.length) && (
                  <li className="text-sm text-[var(--muted)]">暂无</li>
                )}
              </ul>
            </div>
            <div className="rounded-[14px] bg-[rgba(190,92,44,0.06)] p-3">
              <div className="text-xs font-semibold text-rose-700">空方观点</div>
              <ul className="mt-2 space-y-1">
                {(bearCase[symbol] ?? []).map((point, i) => (
                  <li key={i} className="text-sm text-rose-800">- {point}</li>
                ))}
                {!(bearCase[symbol]?.length) && (
                  <li className="text-sm text-[var(--muted)]">暂无</li>
                )}
              </ul>
            </div>
          </div>
          {keyRisks[symbol]?.length ? (
            <div className="mt-3 rounded-[14px] bg-[rgba(190,92,44,0.08)] px-3 py-2 text-sm text-[var(--danger)]">
              关键风险：{keyRisks[symbol].join('；')}
            </div>
          ) : null}
        </div>
      ))}
    </div>
  );
}

function IntelligenceDetail({ signals }: { signals: Record<string, unknown> }) {
  const financialHealth = asNumberRecord(signals.financial_health_score);
  const eventRisk = asNumberRecord(signals.event_risk_score);
  const sentiment = asNumberRecord(signals.sentiment_score);
  const breadth = asNumberRecord(signals.breadth_score);
  const alerts = asStringArray(signals.alerts);

  const symbols = [...new Set([
    ...Object.keys(financialHealth),
    ...Object.keys(eventRisk),
    ...Object.keys(sentiment),
    ...Object.keys(breadth),
  ])];

  const radarData = symbols.length > 0
    ? symbols.map((symbol) => ({
        symbol,
        '财务健康': Number(((financialHealth?.[symbol] ?? 0) * 100).toFixed(0)),
        '事件风险': Number(((eventRisk?.[symbol] ?? 0) * 100).toFixed(0)),
        '情绪': Number(((sentiment?.[symbol] ?? 0) * 100).toFixed(0)),
        '市场广度': Number(((breadth?.[symbol] ?? 0) * 100).toFixed(0)),
      }))
    : [];

  // Build data for radar chart (one chart per symbol)
  const radarAxes = ['财务健康', '事件风险', '情绪', '市场广度'];

  return (
    <div className="space-y-4">
      {radarData.length > 0 && (
        <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
          <div className="text-sm font-semibold text-[var(--ink)]">多维评分</div>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            {radarData.map((item) => {
              const data = radarAxes.map((axis) => ({
                axis,
                value: item[axis as keyof typeof item] as number,
              }));
              return (
                <div key={item.symbol} className="text-center">
                  <div className="text-sm font-semibold text-[var(--ink)]">{item.symbol}</div>
                  <div style={{ height: 220 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={data} cx="50%" cy="50%" outerRadius="70%">
                        <PolarGrid stroke="rgba(12,33,60,0.12)" />
                        <PolarAngleAxis dataKey="axis" tick={{ fontSize: 11 }} />
                        <PolarRadiusAxis tick={{ fontSize: 10 }} domain={[-100, 100]} />
                        <Radar dataKey="value" stroke="#2e8b57" fill="#2e8b57" fillOpacity={0.2} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {alerts.length > 0 && (
        <div className="rounded-[20px] border border-[rgba(190,92,44,0.2)] bg-[rgba(190,92,44,0.06)] p-4">
          <div className="text-sm font-semibold text-[var(--danger)]">风险警报</div>
          <ul className="mt-2 space-y-1">
            {alerts.map((alert, i) => (
              <li key={i} className="text-sm text-[var(--danger)]">· {alert}</li>
            ))}
          </ul>
        </div>
      )}

      {radarData.length === 0 && alerts.length === 0 && (
        <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
          <div className="text-sm text-[var(--muted)]">暂无多维评分数据。</div>
        </div>
      )}
    </div>
  );
}

function MacroDetail({ signals }: { signals: Record<string, unknown> }) {
  const macroRegime = signals.macro_regime as string | undefined;
  const liquiditySignal = signals.liquidity_signal as string | undefined;
  const policySignal = signals.policy_signal as string | undefined;
  const riskLevel = signals.risk_level as string | undefined;
  const macroScore = signals.macro_score as number | undefined;

  const indicators = [
    { label: '宏观状态', value: macroRegime ?? '未知', color: macroRegime?.includes('低') ? '#2e8b57' : macroRegime?.includes('高') ? '#be5c2c' : '#b56d2a' },
    { label: '流动性信号', value: liquiditySignal ?? '未知', color: '#4a7fb5' },
    { label: '政策信号', value: policySignal ?? '未知', color: '#8b5cf6' },
    { label: '风险等级', value: riskLevel ?? '未知', color: riskLevel?.includes('低') ? '#2e8b57' : riskLevel?.includes('高') ? '#be5c2c' : '#b56d2a' },
  ];

  return (
    <div className="space-y-4">
      <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
        <div className="text-sm font-semibold text-[var(--ink)]">宏观指标</div>
        {macroScore != null && (
          <div className="mt-2 text-xs text-[var(--muted)]">
            宏观得分: <span className={macroScore >= 0 ? 'text-emerald-700' : 'text-rose-700'}>
              {macroScore >= 0 ? '+' : ''}{macroScore.toFixed(2)}
            </span>
          </div>
        )}
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {indicators.map((ind) => (
            <div key={ind.label} className="flex items-center justify-between rounded-[14px] bg-[rgba(12,33,60,0.04)] px-4 py-3">
              <span className="text-sm text-[var(--muted)]">{ind.label}</span>
              <span className="text-sm font-semibold" style={{ color: ind.color }}>{ind.value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function GenericSignals({ signals }: { signals: Record<string, unknown> }) {
  return (
    <div className="rounded-[20px] border border-[var(--line)] bg-[rgba(255,255,255,0.72)] p-4">
      <div className="text-sm font-semibold text-[var(--ink)]">信号数据</div>
      <pre className="mt-3 max-h-64 overflow-auto rounded-[14px] bg-[rgba(12,33,60,0.04)] p-3 text-xs text-[var(--muted)]">
        {JSON.stringify(signals, null, 2)}
      </pre>
    </div>
  );
}
