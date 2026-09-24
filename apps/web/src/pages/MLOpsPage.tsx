import React, { useState, useEffect, useCallback } from 'react';
import {
  FlaskConical, GitMerge, Activity, BarChart2, Play, Trash2,
  ChevronRight, CheckCircle, XCircle, AlertTriangle, RefreshCw,
  Plus, Clock, Cpu, Database, TrendingUp, TrendingDown, Loader2,
  ArrowRight, Shield, Star, Archive, Layers, ChevronDown, ChevronUp
} from 'lucide-react';
import { mlopsApi, api } from '../services/api';

// Type definitions
interface ModelVersion {
  version_id: string;
  model_id: string;
  model_type: string;
  algorithm: string;
  version_number: number;
  lifecycle_stage: string;
  metrics: Record<string, any>;
  dataset_id: string;
  dataset_hash: string;
  artifact_path?: string | null;
  parent_version_id?: string | null;
  tags: string[];
  notes?: string | null;
  created_at: string;
}

interface Pipeline {
  pipeline_id: string;
  name: string;
  project_id: string;
  dataset_id: string;
  model_type: string;
  target_column: string;
  cron_expression: string;
  description?: string;
  status: string;
  last_run_at?: string | null;
  next_run_at?: string | null;
  run_count: number;
  created_at: string;
}

interface DriftFeature {
  feature: string;
  psi: number;
  ks_statistic: number;
  p_value: number;
  drift_detected: boolean;
  severity: string;
}

interface DriftReport {
  reference_dataset_id: string;
  current_dataset_id: string;
  overall_drift_score: number;
  drift_detected: boolean;
  features_drifted: number;
  total_features: number;
  drift_percentage: number;
  feature_drift: DriftFeature[];
  analysis_timestamp: string;
}

const STAGE_CONFIG: Record<string, { color: string; bg: string; icon: React.ElementType; label: string }> = {
  experimental: { color: '#94A3B8', bg: '#1E293B', icon: FlaskConical,  label: 'Experimental' },
  staging:      { color: '#F59E0B', bg: '#451A03', icon: Shield,         label: 'Staging' },
  production:   { color: '#10B981', bg: '#064E3B', icon: Star,           label: 'Production' },
  archived:     { color: '#6366F1', bg: '#1E1B4B', icon: Archive,        label: 'Archived' },
};

function StageChip({ stage }: { stage: string }) {
  const cfg = STAGE_CONFIG[stage] || STAGE_CONFIG['experimental'];
  const Icon = cfg.icon;
  return (
    <span
      className="inline-flex items-center space-x-1 px-2 py-0.5 rounded-full text-[10px] font-mono font-semibold border"
      style={{ color: cfg.color, backgroundColor: `${cfg.bg}cc`, borderColor: `${cfg.color}55` }}
    >
      <Icon className="w-2.5 h-2.5" />
      <span>{cfg.label.toUpperCase()}</span>
    </span>
  );
}

function MetricBadge({ name, value }: { name: string; value: any }) {
  const v = typeof value === 'number' ? value.toFixed(4) : String(value);
  return (
    <div className="flex items-center justify-between px-2 py-1 bg-slate-900/60 rounded-lg">
      <span className="text-[10px] text-slate-400 font-mono uppercase">{name}</span>
      <span className="text-[10px] font-bold text-nexus-accent ml-2">{v}</span>
    </div>
  );
}

function DriftBar({ psi, drifted }: { psi: number; drifted: boolean }) {
  const pct = Math.min(100, psi * 400);
  return (
    <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{
          width: `${pct}%`,
          backgroundColor: drifted ? (psi > 0.25 ? '#EF4444' : '#F59E0B') : '#10B981',
        }}
      />
    </div>
  );
}

// ─── Registry Panel ───────────────────────────────────────────────────────────
const RegistryPanel: React.FC = () => {
  const [versions, setVersions] = useState<ModelVersion[]>([]);
  const [loading, setLoading] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [promoteTarget, setPromoteTarget] = useState<{ id: string; stage: string } | null>(null);
  const [promoting, setPromoting] = useState(false);
  const [compareIds, setCompareIds] = useState<[string, string]>(['', '']);
  const [compareResult, setCompareResult] = useState<any | null>(null);
  const [comparing, setComparing] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try { const data = await mlopsApi.listVersions(); setVersions(data); }
    catch { setVersions([]); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const handleCompare = async () => {
    if (!compareIds[0] || !compareIds[1]) return;
    setComparing(true);
    try { const r = await mlopsApi.compareVersions(compareIds[0], compareIds[1]); setCompareResult(r); }
    catch { setCompareResult(null); }
    finally { setComparing(false); }
  };

  if (loading) return (
    <div className="flex flex-col items-center justify-center h-64 space-y-3">
      <Loader2 className="w-8 h-8 text-nexus-accent animate-spin" />
      <p className="text-xs text-slate-400">Loading model registry...</p>
    </div>
  );

  if (versions.length === 0) return (
    <div className="flex flex-col items-center justify-center h-64 space-y-3 text-center">
      <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center">
        <GitMerge className="w-7 h-7 text-indigo-400/60" />
      </div>
      <div>
        <p className="text-sm font-semibold text-slate-300">No registered model versions</p>
        <p className="text-xs text-slate-500 mt-1">Train AutoML or Forecast models, then register versions via the API.</p>
      </div>
    </div>
  );

  return (
    <div className="space-y-3">
      {versions.length >= 2 && (
        <div className="flex items-center space-x-3 p-3 rounded-xl bg-slate-800/60 border border-slate-700/60">
          <span className="text-[10px] text-slate-400 font-mono">COMPARE:</span>
          <select className="bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded-lg px-2 py-1 focus:outline-none" onChange={e => setCompareIds(p => [e.target.value, p[1]])} defaultValue="">
            <option value="" disabled>Version A</option>
            {versions.map(v => <option key={v.version_id} value={v.version_id}>{v.algorithm} v{v.version_number}</option>)}
          </select>
          <ArrowRight className="w-3 h-3 text-slate-500" />
          <select className="bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded-lg px-2 py-1 focus:outline-none" onChange={e => setCompareIds(p => [p[0], e.target.value])} defaultValue="">
            <option value="" disabled>Version B</option>
            {versions.map(v => <option key={v.version_id} value={v.version_id}>{v.algorithm} v{v.version_number}</option>)}
          </select>
          <button onClick={handleCompare} disabled={!compareIds[0] || !compareIds[1] || comparing} className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-indigo-600/80 hover:bg-indigo-500 text-white text-xs font-semibold transition-all disabled:opacity-40">
            {comparing ? <Loader2 className="w-3 h-3 animate-spin" /> : <BarChart2 className="w-3 h-3" />}
            <span>Compare</span>
          </button>
        </div>
      )}

      {compareResult && (
        <div className="p-4 rounded-xl bg-indigo-500/5 border border-indigo-500/30">
          <div className="text-[10px] text-indigo-400 font-mono mb-3">COMPARISON RESULTS</div>
          <div className="grid grid-cols-2 gap-4">
            {(['version_a', 'version_b'] as const).map(k => (
              <div key={k} className="space-y-1.5">
                <div className="text-[10px] text-slate-400 font-mono">{k.replace('_', ' ').toUpperCase()}</div>
                {compareResult[k]?.metrics && Object.entries(compareResult[k].metrics).slice(0, 5).map(([m, v]) => <MetricBadge key={m} name={m} value={v} />)}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-2">
        {versions.map(v => {
          const isExpanded = expandedId === v.version_id;
          return (
            <div key={v.version_id} className="rounded-xl border border-slate-700/60 bg-slate-800/40 overflow-hidden hover:border-slate-600/60 transition-all">
              <button className="w-full flex items-center justify-between px-4 py-3 text-left" onClick={() => setExpandedId(isExpanded ? null : v.version_id)}>
                <div className="flex items-center space-x-3 min-w-0">
                  <div className="w-8 h-8 rounded-lg bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center flex-shrink-0">
                    <Cpu className="w-4 h-4 text-indigo-400" />
                  </div>
                  <div className="min-w-0">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-semibold text-white truncate">{v.algorithm}</span>
                      <span className="text-[10px] text-slate-500 font-mono">v{v.version_number}</span>
                      <StageChip stage={v.lifecycle_stage} />
                    </div>
                    <div className="text-[10px] text-slate-500 mt-0.5">{v.model_type} · {new Date(v.created_at).toLocaleDateString()}</div>
                  </div>
                </div>
                <div className="flex items-center space-x-2 flex-shrink-0">
                  {Object.entries(v.metrics).slice(0, 1).map(([m, val]) => (
                    <span key={m} className="text-[10px] font-mono text-nexus-accent hidden sm:block">{m}: {typeof val === 'number' ? val.toFixed(3) : val}</span>
                  ))}
                  {isExpanded ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
                </div>
              </button>
              {isExpanded && (
                <div className="px-4 pb-4 border-t border-slate-700/40 space-y-4">
                  <div className="grid grid-cols-2 gap-4 pt-3">
                    <div className="space-y-1.5">
                      <div className="text-[10px] text-slate-400 font-mono mb-1.5">METRICS</div>
                      {Object.entries(v.metrics).slice(0, 6).map(([m, val]) => <MetricBadge key={m} name={m} value={val} />)}
                    </div>
                    <div className="space-y-2">
                      <div className="text-[10px] text-slate-400 font-mono mb-1.5">PROMOTE TO</div>
                      <div className="flex flex-wrap gap-1.5">
                        {['staging', 'production', 'archived'].filter(s => s !== v.lifecycle_stage).map(s => {
                          const cfg = STAGE_CONFIG[s];
                          return (
                            <button key={s} onClick={() => setPromoteTarget({ id: v.version_id, stage: s })} className="px-2.5 py-1 rounded-lg text-[10px] font-semibold border transition-all hover:opacity-80" style={{ color: cfg.color, borderColor: `${cfg.color}55`, backgroundColor: `${cfg.bg}aa` }}>
                              {cfg.label}
                            </button>
                          );
                        })}
                      </div>
                      <div className="text-[10px] text-slate-500 font-mono pt-1">Hash: {v.dataset_hash.slice(0, 10)}...</div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {promoteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl p-6 w-80 space-y-4 shadow-2xl">
            <h3 className="text-sm font-bold text-white">Confirm Promotion</h3>
            <p className="text-xs text-slate-400">Promote to <span className="font-semibold text-white">{promoteTarget.stage}</span>?</p>
            <div className="flex space-x-3">
              <button onClick={() => setPromoteTarget(null)} className="flex-1 py-2 rounded-xl bg-slate-800 text-xs text-slate-300 hover:bg-slate-700 transition-colors">Cancel</button>
              <button onClick={async () => { setPromoting(true); try { await load(); } finally { setPromoting(false); setPromoteTarget(null); } }} disabled={promoting} className="flex-1 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold transition-colors disabled:opacity-50 flex items-center justify-center space-x-1.5">
                {promoting ? <Loader2 className="w-3 h-3 animate-spin" /> : <CheckCircle className="w-3 h-3" />}
                <span>Promote</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ─── Drift Panel ──────────────────────────────────────────────────────────────
const DriftPanel: React.FC = () => {
  const [datasets, setDatasets] = useState<Array<{ id: string; name: string }>>([]);
  const [refId, setRefId] = useState('');
  const [curId, setCurId] = useState('');
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState<DriftReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getDatasets?.().then((d: any) => { const list = Array.isArray(d) ? d : d?.items || []; setDatasets(list); }).catch(() => {});
  }, []);

  const runDrift = async () => {
    if (!refId || !curId) return;
    setLoading(true); setError(null);
    try { const r = await mlopsApi.computeDrift(refId, curId); setReport(r); }
    catch (e: any) { setError(e?.response?.data?.detail || 'Drift analysis failed'); }
    finally { setLoading(false); }
  };

  return (
    <div className="space-y-4">
      <div className="p-4 rounded-xl bg-slate-800/60 border border-slate-700/60 space-y-3">
        <div className="text-[10px] text-slate-400 font-mono">DATASET SELECTION</div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-[10px] text-slate-500 font-mono block mb-1">REFERENCE (baseline)</label>
            <select id="drift-ref-select" value={refId} onChange={e => setRefId(e.target.value)} className="w-full bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:border-amber-500">
              <option value="">Select...</option>
              {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[10px] text-slate-500 font-mono block mb-1">CURRENT (new batch)</label>
            <select id="drift-cur-select" value={curId} onChange={e => setCurId(e.target.value)} className="w-full bg-slate-900 border border-slate-700 text-xs text-slate-200 rounded-lg px-3 py-2 focus:outline-none focus:border-amber-500">
              <option value="">Select...</option>
              {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </div>
        </div>
        <button id="run-drift-btn" onClick={runDrift} disabled={!refId || !curId || loading} className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-gradient-to-r from-amber-600 to-orange-600 hover:from-amber-500 hover:to-orange-500 text-white text-xs font-semibold transition-all disabled:opacity-40">
          {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Activity className="w-3.5 h-3.5" />}
          <span>{loading ? 'Analysing...' : 'Run Drift Analysis'}</span>
        </button>
      </div>

      {error && <div className="flex items-center space-x-2 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/30 text-xs text-red-300"><XCircle className="w-4 h-4 flex-shrink-0" /><span>{error}</span></div>}

      {report && (
        <div className="space-y-4">
          <div className="grid grid-cols-4 gap-3">
            {[
              { label: 'Drift Score', value: report.overall_drift_score.toFixed(3), hi: report.drift_detected },
              { label: 'Drifted Features', value: `${report.features_drifted} / ${report.total_features}`, hi: report.drift_detected },
              { label: 'Drift %', value: `${report.drift_percentage.toFixed(1)}%`, hi: report.drift_detected },
              { label: 'Status', value: report.drift_detected ? 'DRIFTED' : 'STABLE', hi: report.drift_detected },
            ].map(item => (
              <div key={item.label} className={`rounded-xl p-3 border ${item.hi ? 'bg-amber-500/10 border-amber-500/30' : 'bg-emerald-500/10 border-emerald-500/30'}`}>
                <div className="text-[10px] text-slate-400 font-mono">{item.label}</div>
                <div className={`text-sm font-bold mt-0.5 ${item.hi ? 'text-amber-300' : 'text-emerald-300'}`}>{item.value}</div>
              </div>
            ))}
          </div>
          {report.feature_drift && report.feature_drift.length > 0 && (
            <div className="rounded-xl border border-slate-700/60 overflow-hidden">
              <div className="bg-slate-800/80 px-4 py-2.5 flex items-center space-x-2 border-b border-slate-700/60">
                <BarChart2 className="w-3.5 h-3.5 text-amber-400" />
                <span className="text-[10px] font-mono text-slate-300 font-semibold">FEATURE-LEVEL DRIFT (PSI + KS Test)</span>
              </div>
              <div className="divide-y divide-slate-700/40">
                {report.feature_drift.slice(0, 12).map(f => (
                  <div key={f.feature} className="px-4 py-2.5 grid grid-cols-5 items-center gap-3">
                    <span className="text-xs text-slate-300 font-mono col-span-1 truncate">{f.feature}</span>
                    <div className="col-span-2"><DriftBar psi={f.psi} drifted={f.drift_detected} /></div>
                    <span className="text-[10px] font-mono text-slate-400">PSI: {f.psi.toFixed(3)}</span>
                    <span className={`text-[10px] font-semibold text-right ${f.drift_detected ? f.psi > 0.25 ? 'text-red-400' : 'text-amber-400' : 'text-emerald-400'}`}>
                      {f.drift_detected ? (f.psi > 0.25 ? 'HIGH' : 'MEDIUM') : 'OK'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
      {!report && !loading && (
        <div className="flex flex-col items-center justify-center h-48 space-y-3 text-center">
          <div className="w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center"><Activity className="w-6 h-6 text-amber-400/60" /></div>
          <p className="text-xs text-slate-400">Select reference and current datasets to run PSI + KS drift analysis</p>
        </div>
      )}
    </div>
  );
};

// ─── Pipelines Panel ──────────────────────────────────────────────────────────
const PipelinesPanel: React.FC = () => {
  const [pipelines, setPipelines] = useState<Pipeline[]>([]);
  const [loading, setLoading] = useState(false);
  const [showCreate, setShowCreate] = useState(false);
  const [creating, setCreating] = useState(false);
  const [triggering, setTriggering] = useState<string | null>(null);
  const [projects, setProjects] = useState<Array<{ id: string; name: string }>>([]);
  const [datasets, setDatasets] = useState<Array<{ id: string; name: string }>>([]);
  const [form, setForm] = useState({ name: '', project_id: '', dataset_id: '', model_type: 'automl', target_column: '', cron_expression: '0 2 * * *', description: '' });

  const load = useCallback(async () => {
    setLoading(true);
    try { const data = await mlopsApi.listPipelines(); setPipelines(data); }
    catch { setPipelines([]); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);
  useEffect(() => {
    api.getProjects?.().then((p: any) => { const list = Array.isArray(p) ? p : p?.items || []; setProjects(list); }).catch(() => {});
    api.getDatasets?.().then((d: any) => { const list = Array.isArray(d) ? d : d?.items || []; setDatasets(list); }).catch(() => {});
  }, []);

  const handleCreate = async () => {
    if (!form.name || !form.project_id || !form.dataset_id || !form.target_column) return;
    setCreating(true);
    try { await mlopsApi.createPipeline(form); await load(); setShowCreate(false); setForm({ name: '', project_id: '', dataset_id: '', model_type: 'automl', target_column: '', cron_expression: '0 2 * * *', description: '' }); }
    finally { setCreating(false); }
  };

  const CRON_PRESETS = [
    { label: 'Daily 2 AM', value: '0 2 * * *' },
    { label: 'Weekly Mon', value: '0 2 * * 1' },
    { label: 'Hourly', value: '0 * * * *' },
    { label: 'Monthly', value: '0 2 1 * *' },
  ];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="text-[10px] text-slate-400 font-mono">{pipelines.length} pipeline{pipelines.length !== 1 ? 's' : ''} registered</div>
        <div className="flex items-center space-x-2">
          <button onClick={load} className="w-7 h-7 flex items-center justify-center rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 transition-colors">
            <RefreshCw className={`w-3.5 h-3.5 text-slate-400 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button id="create-pipeline-btn" onClick={() => setShowCreate(true)} className="flex items-center space-x-1.5 px-3 py-1.5 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white text-xs font-semibold transition-all">
            <Plus className="w-3.5 h-3.5" /><span>New Pipeline</span>
          </button>
        </div>
      </div>

      {pipelines.length === 0 && !loading && (
        <div className="flex flex-col items-center justify-center h-48 space-y-3 text-center">
          <div className="w-12 h-12 rounded-2xl bg-violet-500/10 border border-violet-500/20 flex items-center justify-center"><Clock className="w-6 h-6 text-violet-400/60" /></div>
          <p className="text-xs text-slate-400">No scheduled pipelines. Create one to automate model retraining.</p>
        </div>
      )}

      <div className="space-y-3">
        {pipelines.map(p => (
          <div key={p.pipeline_id} className="rounded-xl border border-slate-700/60 bg-slate-800/40 p-4">
            <div className="flex items-start justify-between">
              <div className="space-y-1 min-w-0">
                <div className="flex items-center space-x-2">
                  <span className="text-sm font-semibold text-white">{p.name}</span>
                  <span className={`px-1.5 py-0.5 rounded-full text-[9px] font-mono font-semibold ${p.status === 'active' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30' : 'bg-slate-700 text-slate-400 border border-slate-600'}`}>{p.status.toUpperCase()}</span>
                </div>
                {p.description && <p className="text-[10px] text-slate-400">{p.description}</p>}
                <div className="flex flex-wrap gap-3 text-[10px] text-slate-500 font-mono">
                  <span className="flex items-center space-x-1"><Clock className="w-3 h-3" /><span>{p.cron_expression}</span></span>
                  <span className="flex items-center space-x-1"><Cpu className="w-3 h-3" /><span>{p.model_type}</span></span>
                  <span className="flex items-center space-x-1"><Database className="w-3 h-3" /><span>runs: {p.run_count}</span></span>
                </div>
              </div>
              <div className="flex items-center space-x-2 flex-shrink-0 ml-4">
                <button onClick={async () => { setTriggering(p.pipeline_id); try { await load(); } finally { setTriggering(null); } }} disabled={triggering === p.pipeline_id} title="Trigger now" className="w-7 h-7 flex items-center justify-center rounded-lg bg-emerald-500/10 border border-emerald-500/30 hover:bg-emerald-500/20 transition-colors">
                  {triggering === p.pipeline_id ? <Loader2 className="w-3.5 h-3.5 text-emerald-400 animate-spin" /> : <Play className="w-3.5 h-3.5 text-emerald-400" />}
                </button>
                <button onClick={async () => { try { await mlopsApi.deletePipeline(p.pipeline_id); await load(); } catch {} }} title="Delete pipeline" className="w-7 h-7 flex items-center justify-center rounded-lg bg-red-500/10 border border-red-500/30 hover:bg-red-500/20 transition-colors">
                  <Trash2 className="w-3.5 h-3.5 text-red-400" />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {showCreate && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div className="bg-slate-900 border border-slate-700 rounded-2xl p-6 w-[480px] shadow-2xl space-y-4 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-bold text-white">Create Re-Training Pipeline</h3>
              <button onClick={() => setShowCreate(false)} className="text-slate-400 hover:text-white text-lg leading-none">x</button>
            </div>
            <div className="space-y-3">
              {[{ label: 'Pipeline Name', key: 'name', placeholder: 'e.g. Churn Model Weekly' }, { label: 'Target Column', key: 'target_column', placeholder: 'e.g. churn' }, { label: 'Description', key: 'description', placeholder: 'Optional...' }].map(f => (
                <div key={f.key}>
                  <label className="text-[10px] text-slate-400 font-mono block mb-1">{f.label}</label>
                  <input value={(form as any)[f.key]} onChange={e => setForm(p => ({ ...p, [f.key]: e.target.value }))} placeholder={f.placeholder} className="w-full bg-slate-800 border border-slate-700 focus:border-violet-500 text-xs text-slate-200 rounded-lg px-3 py-2 placeholder-slate-600 focus:outline-none" />
                </div>
              ))}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-[10px] text-slate-400 font-mono block mb-1">PROJECT</label>
                  <select value={form.project_id} onChange={e => setForm(p => ({ ...p, project_id: e.target.value }))} className="w-full bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded-lg px-3 py-2 focus:outline-none">
                    <option value="">Select...</option>
                    {projects.map(pr => <option key={pr.id} value={pr.id}>{pr.name}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-[10px] text-slate-400 font-mono block mb-1">DATASET</label>
                  <select value={form.dataset_id} onChange={e => setForm(p => ({ ...p, dataset_id: e.target.value }))} className="w-full bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded-lg px-3 py-2 focus:outline-none">
                    <option value="">Select...</option>
                    {datasets.map(d => <option key={d.id} value={d.id}>{d.name}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-[10px] text-slate-400 font-mono block mb-1">MODEL TYPE</label>
                  <select value={form.model_type} onChange={e => setForm(p => ({ ...p, model_type: e.target.value }))} className="w-full bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded-lg px-3 py-2 focus:outline-none">
                    <option value="automl">AutoML</option>
                    <option value="forecast">Forecast</option>
                  </select>
                </div>
                <div>
                  <label className="text-[10px] text-slate-400 font-mono block mb-1">SCHEDULE</label>
                  <select value={form.cron_expression} onChange={e => setForm(p => ({ ...p, cron_expression: e.target.value }))} className="w-full bg-slate-800 border border-slate-700 text-xs text-slate-200 rounded-lg px-3 py-2 focus:outline-none">
                    {CRON_PRESETS.map(cp => <option key={cp.value} value={cp.value}>{cp.label}</option>)}
                  </select>
                </div>
              </div>
            </div>
            <div className="flex space-x-3 pt-2">
              <button onClick={() => setShowCreate(false)} className="flex-1 py-2.5 rounded-xl bg-slate-800 text-xs text-slate-300 hover:bg-slate-700 transition-colors">Cancel</button>
              <button onClick={handleCreate} disabled={creating || !form.name || !form.project_id || !form.dataset_id || !form.target_column} className="flex-1 py-2.5 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white text-xs font-semibold transition-all disabled:opacity-40 flex items-center justify-center space-x-1.5">
                {creating ? <Loader2 className="w-3 h-3 animate-spin" /> : <Plus className="w-3 h-3" />}
                <span>Create Pipeline</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ─── Main MLOpsPage ───────────────────────────────────────────────────────────
export const MLOpsPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'registry' | 'drift' | 'pipelines'>('registry');
  const tabs = [
    { id: 'registry' as const, label: 'Model Registry', icon: GitMerge, description: 'Version lineage, metrics, lifecycle promotion' },
    { id: 'drift' as const,    label: 'Drift Detection', icon: Activity,  description: 'PSI + KS-test feature drift analysis' },
    { id: 'pipelines' as const, label: 'Pipelines',      icon: Clock,     description: 'Scheduled re-training automation' },
  ];
  return (
    <div className="flex flex-col h-full space-y-5">
      <div className="flex items-center justify-between flex-shrink-0">
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500/20 to-violet-500/20 border border-indigo-500/30 flex items-center justify-center">
            <FlaskConical className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-white">MLOps &amp; Model Registry</h1>
            <p className="text-xs text-slate-400">Version control, drift monitoring &amp; automated retraining pipelines</p>
          </div>
        </div>
      </div>
      <div className="flex rounded-2xl bg-slate-800/60 border border-slate-700/60 p-1 flex-shrink-0">
        {tabs.map(t => {
          const Icon = t.icon;
          const isActive = activeTab === t.id;
          return (
            <button key={t.id} id={`mlops-tab-${t.id}`} onClick={() => setActiveTab(t.id)} className={`flex-1 flex items-center justify-center space-x-2 py-2.5 rounded-xl text-xs font-semibold transition-all ${isActive ? 'bg-gradient-to-r from-indigo-600/80 to-violet-600/80 text-white shadow-sm border border-indigo-500/30' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/40'}`}>
              <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-indigo-300' : ''}`} />
              <span>{t.label}</span>
            </button>
          );
        })}
      </div>
      <div className="flex items-center space-x-2 px-3 py-2 rounded-xl bg-slate-800/30 border border-slate-700/40 flex-shrink-0">
        <ChevronRight className="w-3 h-3 text-indigo-400" />
        <p className="text-[10px] text-slate-400">{tabs.find(t => t.id === activeTab)?.description}</p>
      </div>
      <div className="flex-1 overflow-y-auto">
        {activeTab === 'registry' && <RegistryPanel />}
        {activeTab === 'drift' && <DriftPanel />}
        {activeTab === 'pipelines' && <PipelinesPanel />}
      </div>
    </div>
  );
};

export default MLOpsPage;
