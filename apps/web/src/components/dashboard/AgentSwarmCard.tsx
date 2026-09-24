import React from 'react';
import { AgentInfo } from '../../types';
import { Bot, CheckCircle2, ShieldAlert, Sparkles, Terminal, Wrench } from 'lucide-react';

interface AgentSwarmCardProps {
  agents: AgentInfo[];
}

export const AgentSwarmCard: React.FC<AgentSwarmCardProps> = ({ agents }) => {
  return (
    <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-nexus-purple/10 border border-nexus-purple/20 text-nexus-purple">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">Multi-Agent Swarm Registry</h3>
            <p className="text-xs text-slate-400">Autonomous roles & deterministic tool execution bindings</p>
          </div>
        </div>
        <span className="px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-nexus-purple/10 text-nexus-purple border border-nexus-purple/30">
          {agents.length} Agents Configured
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5 pt-1">
        {agents.map((agent) => (
          <div
            key={agent.id}
            className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 hover:border-slate-700 transition-all space-y-2.5"
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center space-x-2">
                <div 
                  className="w-2.5 h-2.5 rounded-full" 
                  style={{ backgroundColor: agent.color }} 
                />
                <h4 className="text-xs font-semibold text-slate-200">{agent.name}</h4>
              </div>
              <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded uppercase ${
                agent.status === 'active' 
                  ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' 
                  : 'bg-slate-800 text-slate-400'
              }`}>
                {agent.status}
              </span>
            </div>

            <p className="text-[11px] text-slate-400 line-clamp-2 leading-relaxed">
              {agent.role}
            </p>

            <div className="pt-2 border-t border-slate-800/80">
              <div className="text-[10px] font-mono text-slate-400 mb-1 flex items-center space-x-1">
                <Wrench className="w-3 h-3 text-slate-400" />
                <span>Deterministic Tools ({agent.tools.length}):</span>
              </div>
              <div className="flex flex-wrap gap-1">
                {agent.tools.slice(0, 3).map((tool, idx) => (
                  <span
                    key={idx}
                    className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-slate-800/80 text-slate-300 border border-slate-700/60"
                  >
                    {tool}()
                  </span>
                ))}
                {agent.tools.length > 3 && (
                  <span className="px-1.5 py-0.5 rounded text-[9px] font-mono bg-slate-800 text-slate-400">
                    +{agent.tools.length - 3}
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
