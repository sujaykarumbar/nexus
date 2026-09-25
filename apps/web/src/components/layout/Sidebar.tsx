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
    { id: 'dashboard', label: 'Command Center', icon: LayoutDashboard },
    { id: 'datasets', label: 'Dataset Explorer', icon: Database },
    { id: 'agents', label: 'Multi-Agent Swarm', icon: Bot },
    { id: 'automl', label: 'AutoML & Models', icon: Cpu },
    { id: 'forecasting', label: 'Time-Series Forecast', icon: TrendingUp },
    { id: 'anomalies', label: 'Anomaly Center', icon: AlertTriangle },
    { id: 'rag', label: 'Knowledge & RAG', icon: Network },
    { id: 'graph', label: 'Knowledge Graph', icon: GitMerge },
    { id: 'streaming', label: 'Live Monitor', icon: Radio },
    { id: 'mlops', label: 'MLOps & Registry', icon: FlaskConical },
    { id: 'observability', label: 'Observability', icon: Eye },
    { id: 'reports', label: 'Executive Reports', icon: FileText },
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
                  className={`w-full flex items-center space-x-3 px-3 py-2.5 rounded-xl text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-gradient-to-r from-nexus-accent/20 to-nexus-purple/10 text-white border border-nexus-accent/30 shadow-glow'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/60'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-nexus-accent' : 'text-slate-400'}`} />
                  <span>{item.label}</span>
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
