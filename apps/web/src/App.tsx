import React, { useState, useEffect } from 'react';
import { AuthProvider, useAuth } from './context/AuthContext';
import { Navbar } from './components/layout/Navbar';
import { Sidebar } from './components/layout/Sidebar';
import { DashboardPage } from './pages/DashboardPage';
import { DatasetsPage } from './pages/DatasetsPage';
import { AutoMLPage } from './pages/AutoMLPage';
import { ForecastingPage } from './pages/ForecastingPage';
import { AnomaliesPage } from './pages/AnomaliesPage';
import { AgentsPage } from './pages/AgentsPage';
import { RAGPage } from './pages/RAGPage';
import { KnowledgeGraphPage } from './pages/KnowledgeGraphPage';
import { StreamingPage } from './pages/StreamingPage';
import { MLOpsPage } from './pages/MLOpsPage';
import { ObservabilityPage } from './pages/ObservabilityPage';
import { ReportsPage } from './pages/ReportsPage';
import { LoginPage } from './pages/LoginPage';
import { RegisterPage } from './pages/RegisterPage';
import { api } from './services/api';
import { SystemHealth } from './types';

const MainLayout: React.FC = () => {
  const { user, isLoading } = useAuth();
  const [authView, setAuthView] = useState<'login' | 'register'>('login');
  const [activeTab, setActiveTab] = useState<string>('dashboard');
  const [health, setHealth] = useState<SystemHealth | null>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await api.getHealth();
        setHealth(data);
      } catch (err) {
        console.error('Health fetch failed:', err);
      }
    };
    fetchHealth();
    const interval = setInterval(fetchHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-nexus-900 flex items-center justify-center">
        <div className="space-y-3 text-center">
          <div className="w-10 h-10 border-2 border-nexus-accent border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-xs font-mono text-slate-400">Initializing NEXUS Engine...</p>
        </div>
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-nexus-900 flex flex-col justify-center py-12">
        {authView === 'login' ? (
          <LoginPage onNavigateToRegister={() => setAuthView('register')} />
        ) : (
          <RegisterPage onNavigateToLogin={() => setAuthView('login')} />
        )}
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-nexus-900 flex flex-col">
      <Navbar 
        systemStatus={health?.status || 'operational'} 
        uptime={health?.services.api.uptime_seconds || 0} 
      />
      <div className="flex-1 flex overflow-hidden">
        <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
        <main className="flex-1 overflow-y-auto p-6 lg:p-8 bg-grid-pattern">
          {activeTab === 'dashboard' && <DashboardPage />}
          {activeTab === 'datasets' && <DatasetsPage />}
          {activeTab === 'agents' && <AgentsPage />}
          {activeTab === 'automl' && <AutoMLPage />}
          {activeTab === 'forecasting' && <ForecastingPage />}
          {activeTab === 'anomalies' && <AnomaliesPage />}
          {activeTab === 'rag' && <RAGPage />}
          {activeTab === 'graph' && <KnowledgeGraphPage />}
          {activeTab === 'streaming' && <StreamingPage />}
          {activeTab === 'mlops' && <MLOpsPage />}
          {activeTab === 'observability' && <ObservabilityPage />}
          {activeTab === 'reports' && <ReportsPage />}
          {activeTab !== 'dashboard' && activeTab !== 'datasets' && activeTab !== 'agents' && activeTab !== 'automl' && activeTab !== 'forecasting' && activeTab !== 'anomalies' && activeTab !== 'rag' && activeTab !== 'graph' && activeTab !== 'streaming' && activeTab !== 'mlops' && activeTab !== 'observability' && activeTab !== 'reports' && (
            <div className="glass-panel p-12 rounded-2xl border-slate-800 text-center space-y-3 max-w-2xl mx-auto my-12">
              <div className="inline-block px-3 py-1 rounded-full text-xs font-mono bg-nexus-accent/10 text-nexus-accent border border-nexus-accent/30">
                MODULE SCHEDULED FOR NEXT MILESTONE
              </div>
              <h2 className="text-xl font-bold text-white capitalize">{activeTab} Engine</h2>
              <p className="text-xs text-slate-400">
                This capability is scheduled in the incremental development roadmap. The foundation layer and API endpoints for this module are active and testable via the OpenAPI documentation.
              </p>
              <button
                onClick={() => setActiveTab('dashboard')}
                className="px-4 py-2 rounded-xl bg-slate-800 text-xs font-semibold text-slate-200 hover:bg-slate-700 transition-colors"
              >
                Return to Command Center
              </button>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

export const App: React.FC = () => {
  return (
    <AuthProvider>
      <MainLayout />
    </AuthProvider>
  );
};

export default App;
