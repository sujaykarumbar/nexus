import React from 'react';
import { useAuth } from '../../context/AuthContext';
import { Activity, ShieldCheck, Database, Cpu, LogOut, Sparkles, Terminal } from 'lucide-react';

interface NavbarProps {
  systemStatus?: string;
  uptime?: number;
}

export const Navbar: React.FC<NavbarProps> = ({ systemStatus = 'operational', uptime = 0 }) => {
  const { user, logout } = useAuth();

  const formatUptime = (sec: number) => {
    const mins = Math.floor(sec / 60);
    const hours = Math.floor(mins / 60);
    if (hours > 0) return `${hours}h ${mins % 60}m`;
    return `${mins}m ${sec % 60}s`;
  };

  return (
    <header className="h-16 border-b border-slate-800 bg-nexus-900/90 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-50">
      {/* Brand & Platform Badge */}
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-2.5">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-nexus-accent via-nexus-purple to-indigo-500 p-[1px] shadow-glow">
            <div className="w-full h-full bg-nexus-900 rounded-xl flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-nexus-accent animate-pulse-subtle" />
            </div>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-extrabold text-lg tracking-wider text-white">NEXUS</span>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-semibold bg-nexus-accent/10 text-nexus-accent border border-nexus-accent/30">
                v0.1.0
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium hidden sm:block">
              Autonomous Multi-Agent Data Intelligence
            </p>
          </div>
        </div>
      </div>

      {/* Center Platform Telemetry */}
      <div className="hidden md:flex items-center space-x-6">
        <div className="flex items-center space-x-2 px-3 py-1.5 rounded-lg bg-slate-900/80 border border-slate-800 text-xs font-mono">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
          <span className="text-slate-400">STATUS:</span>
          <span className="text-emerald-400 font-semibold uppercase">{systemStatus}</span>
        </div>

        <div className="flex items-center space-x-2 text-xs text-slate-400 font-mono">
          <Cpu className="w-3.5 h-3.5 text-nexus-purple" />
          <span>UPTIME: {formatUptime(uptime)}</span>
        </div>
      </div>

      {/* Right User & Actions */}
      <div className="flex items-center space-x-4">
        {user ? (
          <div className="flex items-center space-x-3">
            <div className="text-right hidden sm:block">
              <div className="text-xs font-semibold text-slate-200">{user.full_name || user.email}</div>
              <div className="text-[10px] text-nexus-accent uppercase font-mono tracking-wider">{user.role}</div>
            </div>
            <div className="w-8 h-8 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center text-xs font-bold text-nexus-accent">
              {(user.full_name || user.email).charAt(0).toUpperCase()}
            </div>
            <button
              onClick={logout}
              title="Sign Out"
              className="p-1.5 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-slate-800 transition-colors"
            >
              <LogOut className="w-4 h-4" />
            </button>
          </div>
        ) : (
          <div className="text-xs text-slate-400">Not authenticated</div>
        )}
      </div>
    </header>
  );
};
