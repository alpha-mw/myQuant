import { useState, type FormEvent, type ReactNode } from 'react'
import { useMutation } from '@tanstack/react-query'
import { Database, LockKeyhole, Search, ShieldCheck } from 'lucide-react'
import { getActiveResearchRun } from '../api/research'
import type { V17ArtifactRef, V17MainlinePublicRun } from '../types/research'

export function ResearchPage() {
  const [strategyId, setStrategyId] = useState('cn-mainline')
  const [expectedPointerSha256, setExpectedPointerSha256] = useState('')
  const query = useMutation({
    mutationFn: () => getActiveResearchRun(strategyId, expectedPointerSha256),
  })

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (strategyId.trim()) query.mutate()
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      <header className="overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/90 p-7 text-white shadow-2xl shadow-slate-950/20 sm:p-10">
        <div className="flex flex-wrap items-center gap-3 text-xs uppercase tracking-[0.22em] text-teal-200">
          <span>V17 Mainline</span>
          <span className="rounded-full border border-emerald-300/20 bg-emerald-300/10 px-3 py-1 text-emerald-100">
            Read only
          </span>
        </div>
        <h2 className="mt-5 max-w-3xl font-serif text-4xl leading-tight sm:text-5xl">
          Active research portfolio
        </h2>
        <p className="mt-4 max-w-3xl text-sm leading-7 text-slate-300 sm:text-base">
          Read one governed CN A-share V17 active pointer. This view never starts a job,
          scans history, selects another protocol, or writes a fallback artifact.
        </p>

        <form onSubmit={submit} className="mt-8 grid gap-4 lg:grid-cols-[1fr_1.5fr_auto] lg:items-end">
          <Field label="Strategy ID">
            <input
              value={strategyId}
              onChange={(event) => setStrategyId(event.target.value)}
              required
              placeholder="cn-mainline"
              className="w-full rounded-2xl border border-white/12 bg-white/8 px-4 py-3 text-sm text-white outline-none placeholder:text-slate-500 focus:border-teal-300/60"
            />
          </Field>
          <Field label="Expected pointer SHA-256 (optional)">
            <input
              value={expectedPointerSha256}
              onChange={(event) => setExpectedPointerSha256(event.target.value)}
              placeholder="64 lowercase hex characters"
              pattern="[0-9a-f]{64}"
              className="w-full rounded-2xl border border-white/12 bg-white/8 px-4 py-3 font-mono text-xs text-white outline-none placeholder:text-slate-500 focus:border-teal-300/60"
            />
          </Field>
          <button
            type="submit"
            disabled={!strategyId.trim() || query.isPending}
            className="inline-flex min-h-12 items-center justify-center gap-2 rounded-2xl bg-teal-300 px-5 text-sm font-semibold text-slate-950 transition hover:bg-teal-200 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Search size={17} aria-hidden="true" />
            {query.isPending ? 'Reading…' : 'Read active run'}
          </button>
        </form>
      </header>

      {query.isError && (
        <section className="mt-6 rounded-3xl border border-amber-500/25 bg-amber-50 p-6 text-amber-950">
          <p className="text-xs font-semibold uppercase tracking-[0.18em]">Fail closed</p>
          <h3 className="mt-2 text-xl font-semibold">Active V17 run unavailable</h3>
          <p className="mt-2 break-words text-sm leading-6 text-amber-900/80">
            {query.error instanceof Error ? query.error.message : 'V17_MAINLINE_UNAVAILABLE'}
          </p>
        </section>
      )}

      {query.data ? <RunResult run={query.data} /> : !query.isError && <EmptyReadState />}
    </div>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block">
      <span className="mb-2 block text-xs font-medium uppercase tracking-[0.16em] text-slate-400">{label}</span>
      {children}
    </label>
  )
}

function EmptyReadState() {
  return (
    <section className="mt-6 grid gap-4 md:grid-cols-3">
      <BoundaryCard icon={Database} title="One active pointer" text="No directory scan or stored-run history is consulted." />
      <BoundaryCard icon={LockKeyhole} title="Exact-byte closure" text="Pointer, run, portfolio, formal output, and source closure are hash-bound." />
      <BoundaryCard icon={ShieldCheck} title="No side effects" text="Provider, LLM control, selector, broker, order, execution, and trade calls stay false." />
    </section>
  )
}

function BoundaryCard({ icon: Icon, title, text }: { icon: typeof Database; title: string; text: string }) {
  return (
    <article className="rounded-3xl border border-slate-900/10 bg-white/75 p-6 shadow-sm">
      <Icon size={20} className="text-teal-700" aria-hidden="true" />
      <h3 className="mt-4 font-semibold text-slate-900">{title}</h3>
      <p className="mt-2 text-sm leading-6 text-slate-600">{text}</p>
    </article>
  )
}

function RunResult({ run }: { run: V17MainlinePublicRun }) {
  const refs: Array<[string, V17ArtifactRef]> = [
    ['Active pointer', run.active_pointer_ref],
    ['Mainline run', run.mainline_run_ref],
    ['Formal output', run.formal_output_ref],
    ['Portfolio output', run.portfolio_output_ref],
    ['Source closure', run.source_closure_ref],
  ]
  return (
    <div className="mt-6 space-y-6">
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Metric label="State" value={run.state} />
        <Metric label="Strategy" value={run.canonical_strategy_id} />
        <Metric label="Gross weight" value={run.gross_weight} />
        <Metric label="Cash weight" value={run.cash_weight} />
      </section>

      <section className="overflow-hidden rounded-3xl border border-slate-900/10 bg-white/80 shadow-sm">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-900/8 px-6 py-5">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-teal-700">{run.protocol}</p>
            <h3 className="mt-1 text-xl font-semibold text-slate-900">Portfolio targets</h3>
          </div>
          <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-800">
            {run.targets.length} symbols
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[680px] text-left text-sm">
            <thead className="bg-slate-50 text-xs uppercase tracking-[0.14em] text-slate-500">
              <tr><th className="px-6 py-3">Symbol</th><th className="px-6 py-3">Lane</th><th className="px-6 py-3">Current</th><th className="px-6 py-3">Final</th></tr>
            </thead>
            <tbody>
              {run.targets.map((target) => (
                <tr key={target.symbol} className="border-t border-slate-900/7 text-slate-700">
                  <td className="px-6 py-4 font-mono font-semibold text-slate-950">{target.symbol}</td>
                  <td className="px-6 py-4">{target.lane}</td>
                  <td className="px-6 py-4 font-mono">{target.current_target}</td>
                  <td className="px-6 py-4 font-mono">{target.final_target}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="rounded-3xl border border-slate-900/10 bg-white/80 p-6 shadow-sm">
        <h3 className="text-lg font-semibold text-slate-900">Exact artifact closure</h3>
        <div className="mt-4 grid gap-3">
          {refs.map(([label, ref]) => (
            <div key={label} className="rounded-2xl border border-slate-900/8 bg-slate-50 px-4 py-3">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <span className="text-sm font-semibold text-slate-800">{label}</span>
                <span className="text-xs text-slate-500">{ref.schema_id}</span>
              </div>
              <p className="mt-2 break-all font-mono text-xs text-slate-600">{ref.relative_path}</p>
              <p className="mt-1 break-all font-mono text-[11px] text-slate-400">{ref.byte_sha256}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <article className="rounded-3xl border border-slate-900/10 bg-white/80 p-5 shadow-sm">
      <p className="text-xs uppercase tracking-[0.16em] text-slate-500">{label}</p>
      <p className="mt-2 break-words font-mono text-lg font-semibold text-slate-900">{value}</p>
    </article>
  )
}
