import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Eye, Activity, Server, Database, Cpu, MemoryStick, RefreshCw,
  CheckCircle, XCircle, AlertTriangle, Clock, Zap, ChevronRight,
  TrendingUp, BarChart2, Globe, Shield, Loader2, Wifi
} from 'lucide-react';
import { observabilityApi, api } from '../services/api';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

// ─── Types ────────────────────────────────────────────────────────────────────
interface ServiceHealth {
  status: string;
  latency_ms?: number;
  engine?: string;
  uptime_seconds?: number;
  platform?: string;
  python_version?: string;
  mode?: string;
}

interface SystemMetrics {
  cpu_percent: number;
  memory_used_mb: number;
  memory_total_mb: number;
  memory_percent: number;
}

interface HealthData {
  status: string;
  version: string;
  environment: string;
  timestamp: string;
  services: Record<string, ServiceHealth>;
  system_metrics: SystemMetrics;
}

interface RequestTrace {
  trace_id?: string;
  method?: string;
  path?: string;
  status_code?: number;
  duration_ms?: number;
  timestamp?: string;
  client_ip?: string;
}

// ─── Gauge Ring ───────────────────────────────────────────────────────────────
function GaugeRing({ value, max = 100, color, label, unit = '%' }: { value: number; max?: number; color: string; label: string; unit?: string }) {
  const pct = Math.min(100, (value / max) * 100);
  const r = 36;
  const circ = 2 * Math.PI * r;
  const dash = (pct / 100) * circ;
  return (
    <div className="flex flex-col items-center space-y-2">
      <div className="relative w-20 h-20">
        <svg width="80" height="80" viewBox="0 0 80 80">
          <circle cx="40" cy="40" r={r} fill="none" stroke="#1E293B" strokeWidth="8" />
          <circle
            cx="40" cy="40" r={r} fill="none"
            stroke={color} strokeWidth="8"
            strokeDasharray={`${dash} ${circ - dash}`}
            strokeLinecap="round"
            transform="rotate(-90 40 40)"
            style={{ transition: 'stroke-dasharray 0.6s ease' }}
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-xs font-bold text-white">{value.toFixed(0)}{unit}</span>
        </div>
      </div>
      <span className="text-[10px] text-slate-400 font-mono text-center">{label}</span>
    </div>
  );
}

// ─── Status Badge ─────────────────────────────────────────────────────────────
function StatusBadge({ status }: { status: string }) {
  const isGood = ['healthy', 'operational', 'online', 'alive', 'ready'].includes(status?.toLowerCase() || '');
  return (
    <span className={`inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold border ${
      isGood
        ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30'
        : 'bg-red-500/10 text-red-400 border-red-500/30'
    }`}>
      {isGood ? <CheckCircle className="w-2.5 h-2.5" /> : <XCircle className="w-2.5 h-2.5" />}
      <span>{status?.toUpperCase() || 'UNKNOWN'}</span>
    </span>
  );
}

// ─── Live Health Panel ────────────────────────────────────────────────────────
const LiveHealthPanel: React.FC = () => {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [history, setHistory] = useState<Array<{ t: string; cpu: number; mem: number }>>([]);
  const [loading, setLoading] = useState(true);
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetch = useCallback(async () => {
    try {
      const data = await api.getHealth();
      setHealth(data as any);
      setLastRefresh(new Date());
      const sm = (data as any).system_metrics;
      if (sm) {
        setHistory(prev => {
          const next = [...prev, { t: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }), cpu: sm.cpu_percent, mem: sm.memory_percent }];
          return next.slice(-30);
        });
      }
    } catch { }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    fetch();
    intervalRef.current = setInterval(fetch, 10000);
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [fetch]);

  if (loading) return (
    <div className="flex items-center justify-center h-64">
      <Loader2 className="w-8 h-8 text-nexus-accent animate-spin" />
    </div>
  );

  const sm = health?.system_metrics;
  const services = health?.services || {};

  return (
    <div className="space-y-5">
      {/* Top bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={`w-2.5 h-2.5 rounded-full ${health?.status === 'operational' ? 'bg-emerald-400 animate-pulse' : 'bg-red-400'}`} />
          <span className="text-xs font-semibold text-white">{health?.status?.toUpperCase() || 'UNKNOWN'}</span>
          <span className="text-[10px] text-slate-500">v{health?.version} · {health?.environment}</span>
        </div>
        <div className="flex items-center space-x-2 text-[10px] text-slate-500 font-mono">
          <Clock className="w-3 h-3" />
          <span>Updated {lastRefresh.toLocaleTimeString()}</span>
          <button onClick={fetch} className="ml-1 w-6 h-6 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center hover:bg-slate-700 transition-colors">
            <RefreshCw className="w-3 h-3 text-slate-400" />
          </button>
        </div>
      </div>

      {/* Resource gauges */}
      {sm && (
        <div className="p-5 rounded-2xl bg-slate-800/40 border border-slate-700/60">
          <div className="text-[10px] text-slate-400 font-mono mb-4">SYSTEM RESOURCES</div>
          <div className="flex justify-around">
            <GaugeRing value={sm.cpu_percent} color="#3B82F6" label="CPU" />
            <GaugeRing value={sm.memory_percent} color="#8B5CF6" label="Memory" />
            <GaugeRing value={sm.memory_used_mb} max={sm.memory_total_mb} color="#10B981" label="RAM Used" unit=" MB" />
            <GaugeRing value={sm.memory_total_mb} max={sm.memory_total_mb} color="#06B6D4" label="RAM Total" unit=" MB" />
          </div>
        </div>
      )}

      {/* Live chart */}
      {history.length > 2 && (
        <div className="p-4 rounded-2xl bg-slate-800/40 border border-slate-700/60">
          <div className="text-[10px] text-slate-400 font-mono mb-3">RESOURCE HISTORY (30 samples)</div>
          <ResponsiveContainer width="100%" height={120}>
            <AreaChart data={history} margin={{ top: 0, right: 0, bottom: 0, left: -30 }}>
              <defs>
                <linearGradient id="cpuGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="memGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis dataKey="t" tick={{ fill: '#475569', fontSize: 8 }} interval="preserveStartEnd" />
              <YAxis domain={[0, 100]} tick={{ fill: '#475569', fontSize: 8 }} />
              <Tooltip contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #334155', borderRadius: 8, fontSize: 10 }} labelStyle={{ color: '#94A3B8' }} />
              <Area type="monotone" dataKey="cpu" stroke="#3B82F6" fill="url(#cpuGrad)" strokeWidth={1.5} dot={false} name="CPU %" />
              <Area type="monotone" dataKey="mem" stroke="#8B5CF6" fill="url(#memGrad)" strokeWidth={1.5} dot={false} name="Mem %" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Services table */}
      <div className="rounded-2xl border border-slate-700/60 overflow-hidden">
        <div className="bg-slate-800/80 px-4 py-2.5 flex items-center space-x-2 border-b border-slate-700/60">
          <Server className="w-3.5 h-3.5 text-nexus-accent" />
          <span className="text-[10px] font-mono text-slate-300 font-semibold">SERVICE STATUS</span>
        </div>
        <div className="divide-y divide-slate-700/40">
          {Object.entries(services).map(([svc, info]) => (
            <div key={svc} className="px-4 py-3 flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div className="w-7 h-7 rounded-lg bg-slate-800 border border-slate-700/60 flex items-center justify-center">
                  {svc === 'database' ? <Database className="w-3.5 h-3.5 text-blue-400" /> :
                   svc === 'api' ? <Globe className="w-3.5 h-3.5 text-emerald-400" /> :
                   <Cpu className="w-3.5 h-3.5 text-violet-400" />}
                </div>
                <div>
                  <div className="text-xs font-semibold text-white capitalize">{svc.replace('_', ' ')}</div>
                  {info.engine && <div className="text-[10px] text-slate-500 font-mono">{info.engine}</div>}
                  {info.mode && <div className="text-[10px] text-slate-500 font-mono">{info.mode}</div>}
                  {info.python_version && <div className="text-[10px] text-slate-500 font-mono">Python {info.python_version}</div>}
                </div>
              </div>
              <div className="flex items-center space-x-3">
                {info.latency_ms !== undefined && (
                  <span className="text-[10px] text-slate-400 font-mono">{info.latency_ms.toFixed(1)} ms</span>
                )}
                {info.uptime_seconds !== undefined && (
                  <span className="text-[10px] text-slate-400 font-mono">{Math.floor(info.uptime_seconds / 60)}m uptime</span>
                )}
                <StatusBadge status={info.status} />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

// ─── Request Traces Panel ─────────────────────────────────────────────────────
const TracesPanel: React.FC = () => {
  const [traces, setTraces] = useState<RequestTrace[]>([]);
  const [loading, setLoading] = useState(false);
  const [limit, setLimit] = useState(50);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await observabilityApi.getRequestTrace(limit);
      setTraces(Array.isArray(data) ? data : data?.traces || []);
    } catch { setTraces([]); }
    finally { setLoading(false); }
  }, [limit]);

  useEffect(() => { load(); }, [load]);

  const methodColor = (m?: string) => {
    switch (m?.toUpperCase()) {
      case 'GET': return 'text-emerald-400';
      case 'POST': return 'text-blue-400';
      case 'PUT': return 'text-amber-400';
      case 'DELETE': return 'text-red-400';
      default: return 'text-slate-400';
    }
  };

  const statusColor = (code?: number) => {
    if (!code) return 'text-slate-400';
    if (code < 300) return 'text-emerald-400';
    if (code < 400) return 'text-amber-400';
    return 'text-red-400';
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="text-[10px] text-slate-400 font-mono">{traces.length} recent requests</div>
        <div className="flex items-center space-x-2">
          <select value={limit} onChange={e => setLimit(Number(e.target.value))} className="bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded-lg px-2 py-1 focus:outline-none">
            {[25, 50, 100].map(n => <option key={n} value={n}>Last {n}</option>)}
          </select>
          <button onClick={load} disabled={loading} className="w-7 h-7 flex items-center justify-center rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 transition-colors">
            <RefreshCw className={`w-3.5 h-3.5 text-slate-400 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-48">
          <Loader2 className="w-7 h-7 text-nexus-accent animate-spin" />
        </div>
      ) : traces.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-48 space-y-3 text-center">
          <div className="w-12 h-12 rounded-2xl bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center">
            <Activity className="w-6 h-6 text-cyan-400/60" />
          </div>
          <p className="text-xs text-slate-400">No request traces available yet. Make API calls to populate this view.</p>
        </div>
      ) : (
        <div className="rounded-xl border border-slate-700/60 overflow-hidden">
          <div className="bg-slate-800/80 grid grid-cols-5 px-4 py-2 border-b border-slate-700/60 text-[9px] text-slate-400 font-mono font-semibold uppercase">
            <span>Method</span><span>Path</span><span>Status</span><span>Latency</span><span>Timestamp</span>
          </div>
          <div className="max-h-96 overflow-y-auto divide-y divide-slate-700/30">
            {traces.map((t, i) => (
              <div key={i} className="grid grid-cols-5 px-4 py-2 hover:bg-slate-800/30 transition-colors items-center">
                <span className={`text-[10px] font-mono font-bold ${methodColor(t.method)}`}>{t.method || 'GET'}</span>
                <span className="text-[10px] text-slate-300 font-mono truncate" title={t.path}>{t.path || '/api'}</span>
                <span className={`text-[10px] font-mono font-semibold ${statusColor(t.status_code)}`}>{t.status_code || '200'}</span>
                <span className="text-[10px] text-slate-400 font-mono">{t.duration_ms ? `${t.duration_ms.toFixed(0)}ms` : '-'}</span>
                <span className="text-[10px] text-slate-500 font-mono">{t.timestamp ? new Date(t.timestamp).toLocaleTimeString() : '-'}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ─── Metrics Panel ────────────────────────────────────────────────────────────
const MetricsPanel: React.FC = () => {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try { const data = await observabilityApi.getMetrics(); setMetrics(data); }
    catch { setMetrics(null); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  // Build chart data from metrics if available
  const barData = metrics ? Object.entries(metrics).slice(0, 12).map(([k, v]: any) => ({
    name: k.replace(/_/g, ' ').slice(0, 12),
    value: typeof v === 'number' ? v : 0,
  })) : [];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="text-[10px] text-slate-400 font-mono">Application performance metrics</div>
        <button onClick={load} disabled={loading} className="w-7 h-7 flex items-center justify-center rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 transition-colors">
          <RefreshCw className={`w-3.5 h-3.5 text-slate-400 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-48"><Loader2 className="w-7 h-7 text-nexus-accent animate-spin" /></div>
      ) : !metrics ? (
        <div className="flex flex-col items-center justify-center h-48 space-y-3 text-center">
          <div className="w-12 h-12 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center">
            <BarChart2 className="w-6 h-6 text-violet-400/60" />
          </div>
          <p className="text-xs text-slate-400">No metrics endpoint available or the server has not recorded metrics yet.</p>
          <p className="text-[10px] text-slate-500">The system health and resource metrics are available on the Live Health tab.</p>
        </div>
      ) : (
        <div className="space-y-4">
          {barData.length > 0 && (
            <div className="p-4 rounded-2xl bg-slate-800/40 border border-slate-700/60">
              <div className="text-[10px] text-slate-400 font-mono mb-3">METRICS OVERVIEW</div>
              <ResponsiveContainer width="100%" height={160}>
                <BarChart data={barData} margin={{ top: 0, right: 0, bottom: 20, left: -20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
                  <XAxis dataKey="name" tick={{ fill: '#475569', fontSize: 8 }} angle={-30} textAnchor="end" />
                  <YAxis tick={{ fill: '#475569', fontSize: 8 }} />
                  <Tooltip contentStyle={{ backgroundColor: '#0F172A', border: '1px solid #334155', borderRadius: 8, fontSize: 10 }} />
                  <Bar dataKey="value" fill="#6366F1" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
          <div className="grid grid-cols-3 gap-3">
            {Object.entries(metrics).slice(0, 9).map(([k, v]: any) => (
              <div key={k} className="rounded-xl bg-slate-800/60 border border-slate-700/60 p-3">
                <div className="text-[10px] text-slate-400 font-mono truncate">{k.replace(/_/g, ' ')}</div>
                <div className="text-sm font-bold text-white mt-1">{typeof v === 'number' ? v.toFixed(2) : String(v).slice(0, 12)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

// ─── Main ObservabilityPage ───────────────────────────────────────────────────
export const ObservabilityPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'health' | 'traces' | 'metrics'>('health');
  const tabs = [
    { id: 'health' as const,   label: 'Live Health',   icon: Wifi,     description: 'Real-time system resource monitoring and service status' },
    { id: 'traces' as const,   label: 'Request Traces', icon: Activity, description: 'HTTP request log with method, status, and latency' },
    { id: 'metrics' as const,  label: 'App Metrics',    icon: BarChart2, description: 'Application-level performance and throughput metrics' },
  ];

  return (
    <div className="flex flex-col h-full space-y-5">
      {/* Header */}
      <div className="flex items-center justify-between flex-shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500/20 to-teal-500/20 border border-cyan-500/30 flex items-center justify-center">
            <Eye className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Observability</h1>
            <p className="text-xs text-slate-400">System health, request tracing and performance monitoring</p>
          </div>
        </div>
        <div className="hidden lg:flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-emerald-500/10 border border-emerald-500/20">
          <div className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-[10px] text-emerald-400 font-mono font-semibold">LIVE MONITORING</span>
        </div>
      </div>

      {/* Tab strip */}
      <div className="flex rounded-2xl bg-slate-800/60 border border-slate-700/60 p-1 flex-shrink-0">
        {tabs.map(t => {
          const Icon = t.icon;
          const isActive = activeTab === t.id;
          return (
            <button
              key={t.id}
              id={`obs-tab-${t.id}`}
              onClick={() => setActiveTab(t.id)}
              className={`flex-1 flex items-center justify-center space-x-2 py-2.5 rounded-xl text-xs font-semibold transition-all ${
                isActive
                  ? 'bg-gradient-to-r from-cyan-600/80 to-teal-600/80 text-white shadow-sm border border-cyan-500/30'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/40'
              }`}
            >
              <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-cyan-300' : ''}`} />
              <span>{t.label}</span>
            </button>
          );
        })}
      </div>

      {/* Tab hint */}
      <div className="flex items-center space-x-2 px-3 py-2 rounded-xl bg-slate-800/30 border border-slate-700/40 flex-shrink-0">
        <ChevronRight className="w-3 h-3 text-cyan-400" />
        <p className="text-[10px] text-slate-400">{tabs.find(t => t.id === activeTab)?.description}</p>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'health' && <LiveHealthPanel />}
        {activeTab === 'traces' && <TracesPanel />}
        {activeTab === 'metrics' && <MetricsPanel />}
      </div>
    </div>
  );
};

export default ObservabilityPage;
