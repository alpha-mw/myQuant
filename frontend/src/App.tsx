import { Suspense, lazy, type ReactNode } from 'react';
import { BrowserRouter, Link, Navigate, Route, Routes, useParams } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AppShell from './components/layout/AppShell';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const MarketStatus = lazy(() => import('./pages/MarketStatus'));
const DataExplorer = lazy(() => import('./pages/DataExplorer'));
const StockDetail = lazy(() => import('./pages/StockDetail'));
const AnalysisHub = lazy(() => import('./pages/AnalysisHub'));
const AnalysisHistory = lazy(() => import('./pages/AnalysisHistory'));
const Watchlists = lazy(() => import('./pages/Watchlists'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/" element={<RouteView><Dashboard /></RouteView>} />
            <Route path="/market" element={<RouteView><MarketStatus /></RouteView>} />
            <Route path="/stocks" element={<RouteView><DataExplorer /></RouteView>} />
            <Route path="/stocks/:tsCode" element={<RouteView><StockDetail /></RouteView>} />
            <Route path="/research" element={<RouteView><AnalysisHub /></RouteView>} />
            <Route path="/history" element={<RouteView><AnalysisHistory /></RouteView>} />
            <Route path="/watchlists" element={<RouteView><Watchlists /></RouteView>} />
            <Route path="/settings" element={<RouteView><SettingsPage /></RouteView>} />
            <Route path="/data" element={<Navigate to="/stocks" replace />} />
            <Route path="/data/:tsCode" element={<LegacyStockRedirect />} />
            <Route path="/analysis" element={<Navigate to="/research" replace />} />
            <Route path="/regime" element={<Navigate to="/" replace />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

function LegacyStockRedirect() {
  const { tsCode } = useParams<{ tsCode: string }>();
  return <Navigate to={tsCode ? `/stocks/${tsCode}` : '/stocks'} replace />;
}

function RouteView({ children }: { children: ReactNode }) {
  return (
    <Suspense fallback={<PageLoading />}>
      {children}
    </Suspense>
  );
}

function PageLoading() {
  return <div className="empty-card">页面加载中...</div>;
}

function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center py-32 text-center">
      <h1 className="text-6xl font-bold text-[var(--muted)]">404</h1>
      <p className="mt-4 text-lg text-[var(--muted)]">页面不存在</p>
      <Link to="/" className="mt-6 text-sm text-[var(--accent)] hover:underline">返回首页</Link>
    </div>
  );
}
