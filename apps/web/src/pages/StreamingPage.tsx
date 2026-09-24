import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Radio, Play, Square, Zap, AlertTriangle, Activity,
  RefreshCw, ChevronDown, Database, Cpu, Info,
  TrendingUp, Loader2, CheckCircle, XCircle, Wifi, WifiOff
} from 'lucide-react';
import {
  ComposedChart, Area, Scatter, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine
} from 'recharts';
import { streamingApi } from '../services/api';
import { api } from '../services/api';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────
interface AlertEvent {
  anomaly_id: string;
  session_id: string;
  timestamp: string;
  column: string;
  value: number;
  z_score: number;
  iqr_score?: number;
  severity: 'low' | 'medium' | 'high' | 'critical';
  anomaly_type: string;
  window_mean?: number;
  window_std?: number;
}

interface ChartPoint {
  idx: number;
  value: number;
  isAnomaly: boolean;
  severity?: string;
  timestamp: string;
}

const SEVERITY_CONFIG = {
  low:      { color: '#10B981', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', text: 'text-emerald-300' },
  medium:   { color: '#F59E0B', bg: 'bg-amber-500/10',   border: 'border-amber-500/30',   text: 'text-amber-300'  },
  high:     { color: '#EF4444', bg: 'bg-red-500/10',     border: 'border-red-500/30',     text: 'text-red-300'    },
  critical: { color: '#A855F7', bg: 'bg-purple-500/10',  border: 'border-purple-500/30',  text: 'text-purple-300' },
};

// ─────────────────────────────────────────────────────────────────────────────
// Live Chart with anomaly scatter overlay
// ─────────────────────────────────────────────────────────────────────────────
const LiveStreamChart: React.FC<{
  points: ChartPoint[];
  column: string;
}> = ({ points, column }) => {
  const maxPoints = 120;
  const visible = points.slice(-maxPoints);
  const anomalyPoints = visible.filter(p => p.isAnomaly).map(p => ({
    idx: p.idx,
    anomalyVal: p.value,
    severity: p.severity,
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={visible} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <defs>
          <linearGradient id="streamGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor="#06B6D4" stopOpacity={0.3} />
            <stop offset="95%" stopColor="#06B6D4" stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" vertical={false} />
        <XAxis
          dataKey="idx"
          tick={{ fontSize: 9, fill: '#64748B' }}
          tickLine={false}
          axisLine={false}
          tickFormatter={(v) => `${v}`}
        />
        <YAxis
          tick={{ fontSize: 9, fill: '#64748B' }}
          tickLine={false}
          axisLine={false}
          width={50}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#0F172A',
            border: '1px solid #1E293B',
            borderRadius: '8px',
            fontSize: '11px',
            color: '#CBD5E1',
          }}
          formatter={(v: any) => [Number(v).toFixed(3), column]}
        />
        <Area
          type="monotone"
          dataKey="value"
          stroke="#06B6D4"
          strokeWidth={1.5}
          fill="url(#streamGrad)"
          dot={false}
          isAnimationActive={false}
        />
        <Scatter
          data={anomalyPoints}
          dataKey="anomalyVal"
          fill="#EF4444"
          shape={(props: any) => {
            const { cx, cy } = props;
            return <circle cx={cx} cy={cy} r={5} fill="#EF4444" opacity={0.9} />;
          }}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Alert feed item
// ─────────────────────────────────────────────────────────────────────────────
const AlertFeedItem: React.FC<{ alert: AlertEvent; isNew?: boolean }> = ({ alert, isNew }) => {
  const cfg = SEVERITY_CONFIG[alert.severity] || SEVERITY_CONFIG.low;
  return (
    <div className={`flex items-start space-x-3 px-3 py-2 rounded-xl border ${cfg.bg} ${cfg.border} transition-all ${isNew ? 'animate-pulse-once' : ''}`}>
      <div className="flex-shrink-0 mt-0.5">
        <AlertTriangle className={`w-3.5 h-3.5 ${cfg.text}`} />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <span className={`text-[10px] font-mono font-bold uppercase ${cfg.text}`}>{alert.severity}</span>
          <span className="text-[9px] text-slate-500 font-mono">{alert.timestamp.slice(11, 19)}</span>
        </div>
        <p className="text-xs text-slate-200 mt-0.5">
          <span className="font-semibold text-white">{alert.column}</span>
          {' = '}<span className="font-mono">{alert.value.toFixed(3)}</span>
        </p>
        <p className="text-[10px] text-slate-400 mt-0.5">
          Z={alert.z_score.toFixed(2)} · {alert.anomaly_type.replace('_', ' ')}
          {alert.window_mean != null && ` · μ=${alert.window_mean.toFixed(2)}`}
        </p>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Main Page
// ─────────────────────────────────────────────────────────────────────────────
export const StreamingPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Array<{ id: string; name: string }>>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState('');
  const [rowsPerSec, setRowsPerSec] = useState(8);
  const [windowSize, setWindowSize] = useState(50);
  const [zThreshold, setZThreshold] = useState(2.8);

  const [session, setSession] = useState<any | null>(null);
  const [wsStatus, setWsStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected');
  const [alerts, setAlerts] = useState<AlertEvent[]>([]);
  const [chartData, setChartData] = useState<Record<string, ChartPoint[]>>({});
  const [stats, setStats] = useState({ records: 0, anomalies: 0, rate: 0, spikes: 0 });
  const [selectedColumn, setSelectedColumn] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const alertsRef = useRef<AlertEvent[]>([]);
  const pointIdxRef = useRef(0);
  const statsIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Load datasets on mount ─────────────────────────────────────────────────
  useEffect(() => {
    api.listDatasets?.().then((d: any) => {
      const list = Array.isArray(d) ? d : d?.items || [];
      setDatasets(list);
      if (list.length > 0) setSelectedDatasetId(list[0].id);
    }).catch(() => {});
  }, []);

  // ── Poll session stats while running ─────────────────────────────────────
  useEffect(() => {
    if (!session?.session_id) return;
    statsIntervalRef.current = setInterval(async () => {
      try {
        const s = await streamingApi.getSession(session.session_id);
        setStats({
          records: s.records_processed,
          anomalies: s.anomalies_detected,
          rate: Math.round(s.anomaly_rate * 100),
          spikes: s.spikes_injected,
        });
        if (s.status === 'stopped') {
          clearInterval(statsIntervalRef.current!);
        }
      } catch { /* ignore */ }
    }, 2000);
    return () => { if (statsIntervalRef.current) clearInterval(statsIntervalRef.current); };
  }, [session?.session_id]);

  // ── WebSocket connection ───────────────────────────────────────────────────
  const connectWs = useCallback((sessionId: string) => {
    setWsStatus('connecting');
    const url = streamingApi.getWebSocketUrl(sessionId);
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setWsStatus('connected');
    ws.onclose = () => setWsStatus('disconnected');
    ws.onerror = () => setWsStatus('disconnected');

    ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === 'alert' && msg.data) {
          const alert: AlertEvent = msg.data;
          alertsRef.current = [alert, ...alertsRef.current].slice(0, 200);
          setAlerts([...alertsRef.current]);

          // Update chart data for this column
          const col = alert.column;
          setChartData(prev => {
            const prev_pts = prev[col] || [];
            const idx = ++pointIdxRef.current;
            return {
              ...prev,
              [col]: [...prev_pts, {
                idx,
                value: alert.value,
                isAnomaly: true,
                severity: alert.severity,
                timestamp: alert.timestamp,
              }].slice(-150),
            };
          });
          if (!selectedColumn) setSelectedColumn(col);
        } else if (msg.type === 'heartbeat') {
          setStats(prev => ({
            ...prev,
            records: msg.records_processed || prev.records,
            anomalies: msg.anomalies_detected || prev.anomalies,
          }));
        }
      } catch { /* ignore malformed */ }
    };
  }, [selectedColumn]);

  const handleStart = async () => {
    if (!selectedDatasetId) return;
    setLoading(true);
    setError(null);
    setAlerts([]);
    alertsRef.current = [];
    setChartData({});
    pointIdxRef.current = 0;
    setStats({ records: 0, anomalies: 0, rate: 0, spikes: 0 });

    try {
      const s = await streamingApi.startSession({
        dataset_id: selectedDatasetId,
        rows_per_second: rowsPerSec,
        window_size: windowSize,
        z_threshold: zThreshold,
        spike_probability: 0.04,
        spike_factor: 4.5,
      });
      setSession(s);
      setSelectedColumn('');
      // Give the backend a moment to initialise, then connect WS
      setTimeout(() => connectWs(s.session_id), 800);
    } catch (e: any) {
      setError(e?.response?.data?.detail || 'Failed to start streaming session.');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    if (!session?.session_id) return;
    wsRef.current?.close();
    setWsStatus('disconnected');
    try {
      await streamingApi.stopSession(session.session_id);
    } catch { /* ignore */ }
    setSession((prev: any) => prev ? { ...prev, status: 'stopped' } : prev);
  };

  const isRunning = session?.status === 'running' || session?.status === 'queued';
  const columns = session?.columns_monitored || [];
  const severityCounts = alerts.reduce<Record<string, number>>((acc, a) => {
    acc[a.severity] = (acc[a.severity] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <div className="flex items-center justify-between flex-shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500/20 to-blue-500/20 border border-cyan-500/30 flex items-center justify-center">
            <Radio className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">Live Data Monitor</h1>
            <p className="text-xs text-slate-400">Real-time streaming anomaly detection · WebSocket alert push</p>
          </div>
        </div>

        {/* WS Status badge */}
        <div className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-full text-[10px] font-mono border ${
          wsStatus === 'connected'
            ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
            : wsStatus === 'connecting'
            ? 'bg-amber-500/10 border-amber-500/30 text-amber-300'
            : 'bg-slate-800 border-slate-700 text-slate-400'
        }`}>
          {wsStatus === 'connected'
            ? <><Wifi className="w-3 h-3" /><span>WS LIVE</span></>
            : wsStatus === 'connecting'
            ? <><Loader2 className="w-3 h-3 animate-spin" /><span>CONNECTING</span></>
            : <><WifiOff className="w-3 h-3" /><span>OFFLINE</span></>
          }
        </div>
      </div>

      {/* ── Config + Controls ──────────────────────────────────────────────── */}
      <div className="bg-slate-800/50 border border-slate-700/60 rounded-2xl p-4 flex-shrink-0">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 items-end">
          {/* Dataset selector */}
          <div className="md:col-span-2">
            <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1.5">Dataset</label>
            <div className="relative">
              <Database className="w-3.5 h-3.5 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <select
                id="stream-dataset-select"
                value={selectedDatasetId}
                onChange={e => setSelectedDatasetId(e.target.value)}
                disabled={isRunning}
                className="w-full bg-slate-900/80 border border-slate-700 text-slate-200 text-xs rounded-xl pl-9 pr-4 py-2.5 focus:outline-none focus:border-cyan-500 disabled:opacity-50"
              >
                <option value="">— Select Dataset —</option>
                {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
              </select>
            </div>
          </div>

          {/* Rows/sec */}
          <div>
            <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1.5">
              Speed ({rowsPerSec} rows/s)
            </label>
            <input
              type="range" min={1} max={50} step={1}
              value={rowsPerSec}
              onChange={e => setRowsPerSec(Number(e.target.value))}
              disabled={isRunning}
              className="w-full accent-cyan-500 disabled:opacity-50"
            />
          </div>

          {/* Window size */}
          <div>
            <label className="text-[10px] font-mono text-slate-400 uppercase tracking-wider block mb-1.5">
              Window ({windowSize})
            </label>
            <input
              type="range" min={10} max={200} step={10}
              value={windowSize}
              onChange={e => setWindowSize(Number(e.target.value))}
              disabled={isRunning}
              className="w-full accent-cyan-500 disabled:opacity-50"
            />
          </div>

          {/* Start/Stop */}
          <div>
            {!isRunning ? (
              <button
                id="stream-start-btn"
                onClick={handleStart}
                disabled={!selectedDatasetId || loading}
                className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white text-xs font-semibold transition-all disabled:opacity-40 shadow-lg"
              >
                {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                <span>{loading ? 'Starting…' : 'Start Stream'}</span>
              </button>
            ) : (
              <button
                id="stream-stop-btn"
                onClick={handleStop}
                className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-xl bg-red-600/80 hover:bg-red-500 text-white text-xs font-semibold transition-all shadow-lg"
              >
                <Square className="w-3.5 h-3.5" />
                <span>Stop Stream</span>
              </button>
            )}
          </div>
        </div>

        {error && (
          <div className="mt-3 flex items-center space-x-2 text-xs text-red-400 bg-red-500/10 border border-red-500/30 rounded-xl px-3 py-2">
            <XCircle className="w-3.5 h-3.5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* ── Stats Row ──────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-4 gap-3 flex-shrink-0">
        {[
          { label: 'Records Processed', val: stats.records.toLocaleString(), icon: Activity, color: 'text-cyan-400' },
          { label: 'Anomalies Detected', val: alerts.length.toLocaleString(), icon: AlertTriangle, color: 'text-red-400' },
          { label: 'Anomaly Rate', val: `${stats.rate}%`, icon: TrendingUp, color: 'text-amber-400' },
          { label: 'Spikes Injected', val: stats.spikes.toLocaleString(), icon: Zap, color: 'text-purple-400' },
        ].map(item => (
          <div key={item.label} className="bg-slate-800/50 border border-slate-700/60 rounded-2xl p-4">
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">{item.label}</span>
              <item.icon className={`w-3.5 h-3.5 ${item.color}`} />
            </div>
            <div className={`text-2xl font-bold ${item.color}`}>{item.val}</div>
            {isRunning && <div className="flex items-center space-x-1 mt-1">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
              <span className="text-[9px] text-slate-500 font-mono">LIVE</span>
            </div>}
          </div>
        ))}
      </div>

      {/* ── Main Content ───────────────────────────────────────────────────── */}
      <div className="flex-1 flex space-x-4 min-h-0">

        {/* Left: Chart + Column selector */}
        <div className="flex-1 flex flex-col space-y-3 min-h-0">
          {/* Column tabs */}
          {columns.length > 0 && (
            <div className="flex space-x-1.5 flex-wrap gap-y-1 flex-shrink-0">
              {columns.map((col: string) => {
                const colAlerts = alerts.filter(a => a.column === col).length;
                return (
                  <button
                    key={col}
                    onClick={() => setSelectedColumn(col)}
                    className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition-all border ${
                      selectedColumn === col
                        ? 'bg-cyan-500/15 border-cyan-500/40 text-cyan-300'
                        : 'bg-slate-800/60 border-slate-700 text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    <span>{col}</span>
                    {colAlerts > 0 && (
                      <span className="px-1 py-0.5 rounded-full bg-red-500/20 text-red-300 text-[9px] font-mono">
                        {colAlerts}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          )}

          {/* Chart */}
          <div className="flex-1 bg-slate-800/40 border border-slate-700/60 rounded-2xl p-4 min-h-0">
            {selectedColumn && chartData[selectedColumn]?.length > 0 ? (
              <>
                <div className="flex items-center justify-between mb-3 flex-shrink-0">
                  <div className="flex items-center space-x-2">
                    <Activity className="w-3.5 h-3.5 text-cyan-400" />
                    <span className="text-xs font-semibold text-slate-200">{selectedColumn}</span>
                    <span className="text-[9px] font-mono text-slate-500">live stream</span>
                  </div>
                  <div className="flex items-center space-x-3 text-[9px] font-mono text-slate-500">
                    <span className="flex items-center space-x-1"><span className="w-3 h-0.5 bg-cyan-400 inline-block" /><span>Value</span></span>
                    <span className="flex items-center space-x-1"><span className="w-2 h-2 rounded-full bg-red-500 inline-block" /><span>Anomaly</span></span>
                  </div>
                </div>
                <LiveStreamChart points={chartData[selectedColumn]} column={selectedColumn} />
              </>
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-center space-y-3 opacity-50">
                <Radio className="w-10 h-10 text-cyan-400/40" />
                <div>
                  <p className="text-sm font-semibold text-slate-300">No stream data yet</p>
                  <p className="text-xs text-slate-500 mt-1">Start a session to see real-time values and anomaly markers.</p>
                </div>
              </div>
            )}
          </div>

          {/* Severity distribution */}
          {alerts.length > 0 && (
            <div className="grid grid-cols-4 gap-2 flex-shrink-0">
              {(['critical', 'high', 'medium', 'low'] as const).map(sev => {
                const cfg = SEVERITY_CONFIG[sev];
                const cnt = severityCounts[sev] || 0;
                const pct = alerts.length > 0 ? Math.round((cnt / alerts.length) * 100) : 0;
                return (
                  <div key={sev} className={`${cfg.bg} border ${cfg.border} rounded-xl p-3`}>
                    <div className={`text-[10px] font-mono font-bold uppercase ${cfg.text}`}>{sev}</div>
                    <div className={`text-xl font-bold ${cfg.text} mt-0.5`}>{cnt}</div>
                    <div className="text-[9px] text-slate-500 font-mono">{pct}% of total</div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Right: Alert Feed */}
        <div className="w-80 flex flex-col space-y-2 flex-shrink-0">
          <div className="flex items-center justify-between flex-shrink-0">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-3.5 h-3.5 text-red-400" />
              <span className="text-xs font-semibold text-slate-200">Alert Feed</span>
              {alerts.length > 0 && (
                <span className="px-1.5 py-0.5 rounded-full bg-red-500/20 text-red-300 text-[9px] font-mono">
                  {alerts.length}
                </span>
              )}
            </div>
            {alerts.length > 0 && (
              <button
                onClick={() => { setAlerts([]); alertsRef.current = []; }}
                className="text-[10px] text-slate-500 hover:text-slate-300 transition-colors"
              >
                Clear
              </button>
            )}
          </div>

          <div className="flex-1 overflow-y-auto space-y-1.5 pr-1">
            {alerts.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-center opacity-40 space-y-2">
                <CheckCircle className="w-8 h-8 text-slate-500" />
                <p className="text-xs text-slate-500">No anomalies detected yet</p>
              </div>
            ) : (
              alerts.slice(0, 100).map((a, i) => (
                <AlertFeedItem key={a.anomaly_id} alert={a} isNew={i === 0} />
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
