import React from 'react';
import { 
  LayoutDashboard, 
  Database, 
  Cpu, 
  Bot, 
  TrendingUp, 
  AlertTriangle, 
  FileText, 
  Network, 
  Activity, 
  Layers,
  GitMerge,
  Radio,
  FlaskConical,
  Eye
} from 'lucide-react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab }) => {
  const navItems = [
    { id: 'dashboard', label: 'Command Center', icon: LayoutDashboard, badge: 'Live' },
    { id: 'datasets', label: 'Dataset Explorer', icon: Database, badge: 'Phase 2' },
    { id: 'agents', label: 'Multi-Agent Swarm', icon: Bot, badge: 'Phase 5 Live' },
    { id: 'automl', label: 'AutoML & Models', icon: Cpu, badge: 'Phase 3' },
    { id: 'forecasting', label: 'Time-Series Forecast', icon: TrendingUp, badge: 'Phase 4 Live' },
    { id: 'anomalies', label: 'Anomaly Center', icon: AlertTriangle, badge: 'Phase 4 Live' },
    { id: 'rag', label: 'Knowledge & RAG', icon: Network, badge: 'Phase 6 Live' },
    { id: 'graph', label: 'Knowledge Graph', icon: GitMerge, badge: 'Phase 7 Live' },
    { id: 'streaming', label: 'Live Monitor', icon: Radio, badge: 'Phase 8 Live' },
    { id: 'mlops', label: 'MLOps & Registry', icon: FlaskConical, badge: 'Phase 9 Live' },
    { id: 'observability', label: 'Observability', icon: Eye, badge: 'Phase 10 Live' },
    { id: 'reports', label: 'Executive Reports', icon: FileText, badge: 'Dossiers' },
  ];

  return (
    <aside className="w-64 border-r border-slate-800 bg-nexus-900/60 backdrop-blur-md flex flex-col justify-between p-4 shrink-0 hidden lg:flex">
      <div className="space-y-6">
        <div>
          <div className="text-[11px] font-mono uppercase tracking-wider text-slate-400 font-semibold px-3 mb-2">
            Intelligence Modules
          </div>
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setActiveTab(item.id)}
                  className={`w-full flex items-center justify-between px-3 py-2.5 rounded-xl text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-gradient-to-r from-nexus-accent/20 to-nexus-purple/10 text-white border border-nexus-accent/30 shadow-glow'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <div className="flex items-center space-x-3">
                    <Icon className={`w-4 h-4 ${isActive ? 'text-nexus-accent' : 'text-slate-400'}`} />
                    <span>{item.label}</span>
                  </div>
                  {item.badge && (
                    <span className={`px-1.5 py-0.5 text-[9px] font-mono rounded ${
                      isActive
                        ? 'bg-nexus-accent/20 text-nexus-accent border border-nexus-accent/40'
                        : 'bg-slate-800 text-slate-400 border border-slate-700'
                    }`}>
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>

        {/* Phase Indicator */}
        <div className="p-3.5 rounded-xl glass-panel border-slate-800/80">
          <div className="flex items-center space-x-2 text-nexus-accent mb-1.5">
            <Layers className="w-3.5 h-3.5" />
            <span className="text-[11px] font-mono font-semibold uppercase tracking-wider">Roadmap Status</span>
          </div>
          <p className="text-xs text-slate-300 font-medium">Phase 10: Production Hardened</p>
          <div className="w-full bg-slate-800 h-1.5 rounded-full mt-2 overflow-hidden">
            <div className="bg-gradient-to-r from-nexus-accent via-nexus-purple to-nexus-cyan h-full w-full" />
          </div>
          <p className="text-[10px] text-slate-400 mt-1.5">Streaming · MLOps · Observability</p>
        </div>
      </div>

      {/* Footer Info */}
      <div className="text-[11px] text-slate-400 font-mono px-3 py-2 border-t border-slate-800/60">
        <div>NEXUS Engine v0.1.0</div>
        <div className="text-slate-400 text-[10px]">Deterministic ML + Multi-Agent Swarm</div>
      </div>
    </aside>
  );
};
