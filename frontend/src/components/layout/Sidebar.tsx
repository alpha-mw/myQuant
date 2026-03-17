import { NavLink } from 'react-router-dom';
import {
  Activity,
  ArrowRight,
  Bookmark,
  Database,
  Files,
  FlaskConical,
  History,
  Settings,
  TrendingUp,
} from 'lucide-react';

const navItems = [
  {
    to: '/',
    icon: Activity,
    label: '总览',
    subtitle: '快速入口与状态概要',
  },
  {
    to: '/market',
    icon: TrendingUp,
    label: '市场状态',
    subtitle: '数据覆盖与市场脉搏',
  },
  {
    to: '/stocks',
    icon: Database,
    label: '数据浏览',
    subtitle: '股票覆盖与完整度',
  },
  {
    to: '/research',
    icon: FlaskConical,
    label: '分析中心',
    subtitle: '五维分析与决策',
  },
  {
    to: '/history',
    icon: History,
    label: '过往分析',
    subtitle: '分析记录与回溯',
  },
  {
    to: '/watchlists',
    icon: Bookmark,
    label: '持仓与观察',
    subtitle: '用户持仓、自选与研究池',
  },
  {
    to: '/settings',
    icon: Settings,
    label: '系统设置',
    subtitle: '凭证与回测参数',
  },
];

export default function Sidebar() {
  return (
    <aside className="app-sidebar">
      <div className="app-brand">
        <div>
          <p className="app-brand-kicker">myQuant V8</p>
          <h1>投研平台</h1>
        </div>
        <div className="app-brand-badge">Desktop First</div>
      </div>

      <div className="workflow-card">
        <p className="workflow-label">核心工作流</p>
        <div className="workflow-row">
          <span>数据掌握</span>
          <ArrowRight size={14} />
          <span>并行研究</span>
          <ArrowRight size={14} />
          <span>投资决策</span>
        </div>
      </div>

      <nav className="app-nav">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) => `app-nav-item ${isActive ? 'is-active' : ''}`}
          >
            <item.icon size={18} />
            <div>
              <div className="app-nav-label">{item.label}</div>
              <div className="app-nav-subtitle">{item.subtitle}</div>
            </div>
          </NavLink>
        ))}
      </nav>

      <div className="app-sidebar-footer">
        <Files size={16} />
        <span>所有数据与结果存储在本地。</span>
      </div>
    </aside>
  );
}
