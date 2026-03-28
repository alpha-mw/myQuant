import { NavLink } from 'react-router-dom'

const NAV_ITEMS = [
  { to: '/', label: 'Research', icon: '\u25B6' },      // ▶
  { to: '/history', label: 'History', icon: '\u29D6' }, // ⧖
  { to: '/settings', label: 'Settings', icon: '\u2699' }, // ⚙
]

export function Sidebar() {
  return (
    <nav className="w-12 bg-gray-900 border-r border-gray-800 flex flex-col items-center py-3 gap-1 shrink-0">
      <div className="text-emerald-400 font-bold text-sm mb-4">Q</div>
      {NAV_ITEMS.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.to === '/'}
          className={({ isActive }) =>
            `w-9 h-9 flex items-center justify-center rounded text-sm transition-colors ${
              isActive
                ? 'bg-gray-700 text-emerald-400'
                : 'text-gray-500 hover:text-gray-300 hover:bg-gray-800'
            }`
          }
          title={item.label}
        >
          {item.icon}
        </NavLink>
      ))}
    </nav>
  )
}
