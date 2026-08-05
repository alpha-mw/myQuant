import { ChartNoAxesCombined, Settings2, ShieldCheck, Workflow } from 'lucide-react'
import { Link, NavLink } from 'react-router-dom'

const NAV_ITEMS = [
  {
    to: '/research',
    label: 'Research Workspace',
    description: 'Read the governed active V17 research portfolio.',
    icon: ChartNoAxesCombined,
  },
  {
    to: '/settings',
    label: 'Settings',
    description: 'Manage credentials, defaults, and database health.',
    icon: Settings2,
  },
]

export function Sidebar() {
  return (
    <aside className="border-b border-white/10 bg-slate-950/90 px-4 py-4 backdrop-blur lg:flex lg:h-screen lg:flex-col lg:border-b-0 lg:border-r lg:border-white/8 lg:px-5 lg:py-6">
      <div className="flex items-start justify-between gap-4">
        <Link
          to="/research"
          className="min-w-0 rounded-2xl focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
        >
          <p className="text-[11px] uppercase tracking-[0.28em] text-teal-300/70">Quant Investor</p>
          <h1 className="mt-2 font-serif text-2xl text-white">myQuant Workspace</h1>
          <p className="mt-2 max-w-56 text-sm leading-6 text-slate-400">
            Exact-pointer V17 research portfolio readout with fail-closed provenance.
          </p>
        </Link>
        <span className="rounded-full border border-amber-300/20 bg-amber-300/10 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-amber-100">
          V17
        </span>
      </div>

      <nav
        aria-label="Primary"
        className="mt-5 flex gap-2 overflow-x-auto pb-1 lg:flex-1 lg:flex-col lg:overflow-visible lg:pb-0"
      >
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end
              className={({ isActive }) =>
                `group flex min-w-[15rem] items-start gap-3 rounded-3xl border px-4 py-4 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300 lg:min-w-0 ${
                  isActive
                    ? 'border-teal-300/25 bg-teal-300/10 text-white'
                    : 'border-white/6 bg-white/[0.03] text-slate-200 hover:border-white/14 hover:bg-white/[0.05]'
                }`
              }
            >
              <span className="mt-0.5 rounded-2xl border border-white/8 bg-slate-900/80 p-2 text-teal-200">
                <Icon size={18} aria-hidden="true" />
              </span>
              <span className="min-w-0">
                <span className="block text-sm font-semibold">{item.label}</span>
                <span className="mt-1 block text-xs leading-5 text-slate-400 group-hover:text-slate-300">
                  {item.description}
                </span>
              </span>
            </NavLink>
          )
        })}
      </nav>

      <div className="mt-5 hidden rounded-3xl border border-white/8 bg-white/[0.03] p-4 lg:block">
        <div className="flex items-center gap-2 text-xs uppercase tracking-[0.18em] text-slate-400">
          <Workflow size={14} aria-hidden="true" />
          Governance
        </div>
        <p className="mt-3 text-sm leading-6 text-slate-300">
          One canonical active pointer closes the V17 run, formal output,
          portfolio output, and source evidence by exact bytes.
        </p>
        <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-200">
          <ShieldCheck size={14} aria-hidden="true" />
          Read only · no fallback
        </div>
      </div>
    </aside>
  )
}
