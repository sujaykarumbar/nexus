import React from 'react';
import { SystemHealth } from '../../types';
import { Activity, Cpu, Database, HardDrive, Server, ShieldCheck, Zap } from 'lucide-react';

interface SystemTelemetryCardProps {
  health: SystemHealth | null;
}

export const SystemTelemetryCard: React.FC<SystemTelemetryCardProps> = ({ health }) => {
  if (!health) {
    return (
      <div className="glass-panel p-6 rounded-2xl border-slate-800 animate-pulse text-xs text-slate-400">
        Loading system telemetry...
      </div>
    );
  }

  return (
    <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-nexus-accent/10 border border-nexus-accent/20 text-nexus-accent">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">System Health & Telemetry</h3>
            <p className="text-xs text-slate-400">Real-time resource utilization & microservice probes</p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-400" />
          <span className="text-xs font-mono text-emerald-400 font-semibold uppercase">
            {health.status}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5 pt-1">
        {/* CPU Usage */}
        <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-nexus-accent" />
              <span>CPU Load</span>
            </span>
            <span className="font-mono text-white font-semibold">{health.system_metrics.cpu_percent}%</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-2">
            <div 
              className="bg-nexus-accent h-full transition-all duration-500" 
              style={{ width: `${Math.min(health.system_metrics.cpu_percent, 100)}%` }} 
            />
          </div>
        </div>

        {/* Memory Usage */}
        <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="flex items-center space-x-1.5">
              <HardDrive className="w-3.5 h-3.5 text-nexus-purple" />
              <span>RAM Used</span>
            </span>
            <span className="font-mono text-white font-semibold">{health.system_metrics.memory_percent}%</span>
          </div>
          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden mt-2">
            <div 
              className="bg-nexus-purple h-full transition-all duration-500" 
              style={{ width: `${Math.min(health.system_metrics.memory_percent, 100)}%` }} 
            />
          </div>
        </div>

        {/* Database Engine & Latency */}
        <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="flex items-center space-x-1.5">
              <Database className="w-3.5 h-3.5 text-emerald-400" />
              <span>DB Latency</span>
            </span>
            <span className="font-mono text-emerald-400 font-semibold">{health.services.database.latency_ms} ms</span>
          </div>
          <p className="text-[10px] text-slate-400 font-mono mt-1">Engine: {health.services.database.engine.toUpperCase()}</p>
        </div>

        {/* Async Job Worker */}
        <div className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
          <div className="flex items-center justify-between text-slate-400 text-xs">
            <span className="flex items-center space-x-1.5">
              <Zap className="w-3.5 h-3.5 text-amber-400" />
              <span>Job Broker</span>
            </span>
            <span className="font-mono text-amber-400 font-semibold uppercase">{health.services.job_worker.status}</span>
          </div>
          <p className="text-[10px] text-slate-400 font-mono mt-1">Mode: Async Worker</p>
        </div>
      </div>
    </div>
  );
};
