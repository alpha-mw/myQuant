import { Suspense, lazy, type ReactNode } from 'react'
import { BrowserRouter, Link, Navigate, Route, Routes } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AppShell } from './layouts/AppShell'

const ResearchPage = lazy(() => import('./pages/ResearchPage').then((mod) => ({ default: mod.ResearchPage })))
const HistoryPage = lazy(() => import('./pages/HistoryPage').then((mod) => ({ default: mod.HistoryPage })))
const RunDetailPage = lazy(() => import('./pages/RunDetailPage').then((mod) => ({ default: mod.RunDetailPage })))
const SettingsPage = lazy(() => import('./pages/SettingsPage'))
const DataExplorerPage = lazy(() => import('./pages/DataExplorerPage'))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/research" replace />} />
          <Route element={<AppShell />}>
            <Route path="/research" element={<RouteView><ResearchPage /></RouteView>} />
            <Route path="/history" element={<RouteView><HistoryPage /></RouteView>} />
            <Route path="/history/:jobId" element={<RouteView><RunDetailPage /></RouteView>} />
            <Route path="/data" element={<RouteView><DataExplorerPage /></RouteView>} />
            <Route path="/data/:tsCode" element={<RouteView><DataExplorerPage /></RouteView>} />
            <Route path="/settings" element={<RouteView><SettingsPage /></RouteView>} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}

function RouteView({ children }: { children: ReactNode }) {
  return (
    <Suspense fallback={<PageLoading />}>
      {children}
    </Suspense>
  )
}

function PageLoading() {
  return <div className="px-4 py-6 text-sm text-slate-400">页面加载中…</div>
}

function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center py-32 text-center">
      <h1 className="text-6xl font-bold text-slate-500">404</h1>
      <p className="mt-4 text-lg text-slate-400">页面不存在</p>
      <Link
        to="/research"
        className="mt-6 rounded-full border border-white/10 px-4 py-2 text-sm text-slate-200 transition-colors hover:border-white/16 hover:bg-white/[0.04] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-amber-300"
      >
        返回研究工作台
      </Link>
    </div>
  )
}
