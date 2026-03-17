import { Outlet, useLocation } from 'react-router-dom';
import { SearchSlash } from 'lucide-react';
import Sidebar from './Sidebar';

const pageTitles: Record<string, { title: string; subtitle: string }> = {
  '/': {
    title: '总览',
    subtitle: '快速入口与状态概要。',
  },
  '/market': {
    title: '市场状态',
    subtitle: '数据覆盖、市场脉搏与行业分布。',
  },
  '/stocks': {
    title: '数据浏览',
    subtitle: '股票筛选与档案查看。',
  },
  '/research': {
    title: '分析中心',
    subtitle: '五维分析与投资决策。',
  },
  '/history': {
    title: '过往分析',
    subtitle: '研究记录与回溯。',
  },
  '/settings': {
    title: '系统设置',
    subtitle: '凭证、回测参数与运行环境。',
  },
};

export default function AppShell() {
  const location = useLocation();
  const header = location.pathname.startsWith('/stocks/')
    ? {
        title: '个股档案',
        subtitle: '概览、技术、基本面、竞对与历史。',
      }
    : pageTitles[location.pathname] ?? pageTitles['/'];

  return (
    <div className="app-frame">
      <Sidebar />
      <div className="app-main">
        <header className="page-header">
          <div>
            <p className="page-header-kicker">myQuant V8</p>
            <h2>{header.title}</h2>
            <p>{header.subtitle}</p>
          </div>
          <div className="page-header-note">
            <SearchSlash size={16} />
            <span>缺失字段会明确标注。</span>
          </div>
        </header>
        <main className="page-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
