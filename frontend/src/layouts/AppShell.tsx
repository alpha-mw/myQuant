import { Outlet } from 'react-router-dom'
import { Sidebar } from './Sidebar'

export function AppShell() {
  return (
    <div className="min-h-screen bg-[#08111a] text-slate-100 lg:grid lg:h-screen lg:grid-cols-[18rem_minmax(0,1fr)] lg:overflow-hidden">
      <a
        href="#workspace-content"
        className="sr-only absolute left-4 top-4 z-50 rounded-full bg-amber-200 px-4 py-2 text-sm font-medium text-slate-950 focus:not-sr-only"
      >
        跳转到主内容
      </a>
      <Sidebar />
      <main
        id="workspace-content"
        className="min-w-0 lg:overflow-hidden bg-[radial-gradient(circle_at_top_left,_rgba(15,118,110,0.18),_transparent_26%),radial-gradient(circle_at_80%_0%,_rgba(245,158,11,0.12),_transparent_20%),linear-gradient(180deg,_rgba(9,14,24,0.98),_rgba(3,7,18,1))]"
      >
        <div className="flex h-full flex-col">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
