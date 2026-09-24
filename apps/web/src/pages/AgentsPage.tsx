import React, { useState, useEffect } from 'react';
import { 
  Bot, 
  Play, 
  ShieldCheck, 
  ShieldAlert, 
  FileText, 
  Activity, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  Clock, 
  Sparkles, 
  Database, 
  Cpu, 
  TrendingUp, 
  Search, 
  ChevronRight, 
  ExternalLink,
  RefreshCw,
  Award,
  Layers,
  Flame
} from 'lucide-react';
import { api, agentsApi } from '../services/api';
import { Project, Dataset, SwarmRunResponse, SwarmStepTrace } from '../types';

export const AgentsPage: React.FC = () => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>('');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');

  // Agent Swarm Configuration
  const [goal, setGoal] = useState<string>('Autonomous multi-agent analytical discovery, AutoML benchmarking, and decision intelligence.');
  const [targetColumn, setTargetColumn] = useState<string>('');
  const [timeColumn, setTimeColumn] = useState<string>('');
  const [runAutoML, setRunAutoML] = useState<boolean>(true);
  const [runForecasting, setRunForecasting] = useState<boolean>(true);
  const [runAnomalies, setRunAnomalies] = useState<boolean>(true);
  const [runRAG, setRunRAG] = useState<boolean>(true);
  const [docQuery, setDocQuery] = useState<string>('');

  // Execution state
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [swarmResult, setSwarmResult] = useState<SwarmRunResponse | null>(null);
  const [selectedStep, setSelectedStep] = useState<SwarmStepTrace | null>(null);
  const [activeTab, setActiveTab] = useState<'dag' | 'trace' | 'critic' | 'recommendations' | 'dossier'>('dag');
  const [error, setError] = useState<string | null>(null);

  // Load projects & datasets
  useEffect(() => {
    const initData = async () => {
      try {
        const projs = await api.getProjects();
        setProjects(projs);
        if (projs.length > 0) {
          setSelectedProjectId(projs[0].id);
        }
      } catch (err) {
        console.error('Failed to load projects:', err);
      }
    };
    initData();
  }, []);

  useEffect(() => {
    if (!selectedProjectId) return;
    const fetchDatasets = async () => {
      try {
        const ds = await api.getDatasets(selectedProjectId);
        setDatasets(ds);
        if (ds.length > 0) {
          setSelectedDatasetId(ds[0].id);
          if (ds[0].target_column) setTargetColumn(ds[0].target_column);
        }
      } catch (err) {
        console.error('Failed to load datasets:', err);
      }
    };
    fetchDatasets();
  }, [selectedProjectId]);

  const handleRunSwarm = async () => {
    if (!selectedDatasetId || !selectedProjectId) {
      setError('Please select a dataset to launch the swarm.');
      return;
    }
    setError(null);
    setIsRunning(true);
    try {
      const res = await agentsApi.runSwarm({
        project_id: selectedProjectId,
        dataset_id: selectedDatasetId,
        goal,
        target_column: targetColumn || undefined,
        time_column: timeColumn || undefined,
        run_automl: runAutoML,
        run_forecasting: runForecasting,
        run_anomalies: runAnomalies,
        run_rag: runRAG,
        document_query: docQuery || undefined
      });
      setSwarmResult(res);
      if (res.trace.length > 0) {
        setSelectedStep(res.trace[0]);
      }
      setActiveTab('dag');
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Multi-agent swarm execution encountered an error.');
    } finally {
      setIsRunning(false);
    }
  };

  const agentsList = [
    { id: 'data_agent', name: 'Data Agent', role: 'Data Hygiene & Quality Scoring', icon: Database, color: 'border-blue-500/50 bg-blue-500/10 text-blue-400' },
    { id: 'eda_agent', name: 'Data Scientist Agent', role: 'Statistical EDA & Hypotheses', icon: Sparkles, color: 'border-emerald-500/50 bg-emerald-500/10 text-emerald-400' },
    { id: 'ml_agent', name: 'ML Engineer Agent', role: 'AutoML & Cross-Validation', icon: Cpu, color: 'border-purple-500/50 bg-purple-500/10 text-purple-400' },
    { id: 'forecast_agent', name: 'Forecasting Agent', role: 'Time-Series & Uncertainty', icon: TrendingUp, color: 'border-amber-500/50 bg-amber-500/10 text-amber-400' },
    { id: 'anomaly_agent', name: 'Anomaly Agent', role: 'Outlier Ranking & Change Points', icon: AlertTriangle, color: 'border-red-500/50 bg-red-500/10 text-red-400' },
    { id: 'rag_agent', name: 'Research & RAG Agent', role: 'Domain Evidence & Citations', icon: Search, color: 'border-indigo-500/50 bg-indigo-500/10 text-indigo-400' },
    { id: 'critic_agent', name: 'Critic Gatekeeper', role: 'Mathematical Anti-Hallucination', icon: ShieldCheck, color: 'border-pink-500/50 bg-pink-500/10 text-pink-400' },
    { id: 'recommendation_agent', name: 'Recommendation Agent', role: 'Risk-Bounded Strategic Actions', icon: Award, color: 'border-cyan-500/50 bg-cyan-500/10 text-cyan-400' },
  ];

  const getAgentStepStatus = (agentId: string) => {
    if (!swarmResult) return isRunning ? 'running' : 'idle';
    const found = swarmResult.trace.find(s => s.agent_id === agentId);
    if (!found) return 'idle';
    return found.status;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center space-x-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-nexus-accent/10 text-nexus-accent border border-nexus-accent/30">
              Phase 5 Active
            </span>
            <span className="text-xs text-slate-400 font-mono">LangGraph Stateful DAG Engine</span>
          </div>
          <h1 className="text-2xl font-bold text-white tracking-tight mt-1 flex items-center space-x-3">
            <Bot className="w-7 h-7 text-nexus-accent" />
            <span>Autonomous Multi-Agent Swarm</span>
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">
            Orchestrates dedicated data scientists, ML engineers, and a strict mathematical Critic Gatekeeper to eliminate hallucinations.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-3">
          <button
            onClick={handleRunSwarm}
            disabled={isRunning || !selectedDatasetId}
            className={`px-5 py-2.5 rounded-xl text-xs font-semibold flex items-center space-x-2 shadow-glow transition-all ${
              isRunning || !selectedDatasetId
                ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                : 'bg-gradient-to-r from-nexus-accent to-nexus-purple text-white hover:opacity-95'
            }`}
          >
            {isRunning ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Executing Swarm DAG...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Launch Swarm Run</span>
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-xs flex items-center space-x-3">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Dataset & Configuration Card */}
      <div className="glass-panel p-5 rounded-2xl border-slate-800 space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div>
            <label className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block mb-1">Project</label>
            <select
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-nexus-accent"
            >
              {projects.map((p) => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block mb-1">Target Dataset</label>
            <select
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-nexus-accent"
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>{d.name} ({d.row_count} rows)</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block mb-1">Supervised Target (Optional)</label>
            <input
              type="text"
              placeholder="e.g. churned, revenue"
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-nexus-accent"
            />
          </div>

          <div>
            <label className="text-[11px] font-mono text-slate-400 uppercase tracking-wider block mb-1">Time Column (Optional)</label>
            <input
              type="text"
              placeholder="e.g. date, timestamp"
              value={timeColumn}
              onChange={(e) => setTimeColumn(e.target.value)}
              className="w-full bg-slate-900 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-nexus-accent"
            />
          </div>
        </div>

        {/* Module Toggles */}
        <div className="flex flex-wrap items-center gap-4 pt-2 border-t border-slate-800/80 text-xs">
          <span className="text-slate-400 font-mono text-[11px]">Enabled Sub-Swarm Modules:</span>
          <label className="flex items-center space-x-2 text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={runAutoML}
              onChange={(e) => setRunAutoML(e.target.checked)}
              className="rounded bg-slate-900 border-slate-800 text-nexus-accent focus:ring-0"
            />
            <span>AutoML Engineer</span>
          </label>
          <label className="flex items-center space-x-2 text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={runForecasting}
              onChange={(e) => setRunForecasting(e.target.checked)}
              className="rounded bg-slate-900 border-slate-800 text-nexus-accent focus:ring-0"
            />
            <span>Time-Series Forecaster</span>
          </label>
          <label className="flex items-center space-x-2 text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={runAnomalies}
              onChange={(e) => setRunAnomalies(e.target.checked)}
              className="rounded bg-slate-900 border-slate-800 text-nexus-accent focus:ring-0"
            />
            <span>Anomaly Detector</span>
          </label>
          <label className="flex items-center space-x-2 text-slate-300 cursor-pointer">
            <input
              type="checkbox"
              checked={runRAG}
              onChange={(e) => setRunRAG(e.target.checked)}
              className="rounded bg-slate-900 border-slate-800 text-nexus-accent focus:ring-0"
            />
            <span>Research & RAG</span>
          </label>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center space-x-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab('dag')}
          className={`px-4 py-2 rounded-xl text-xs font-medium transition-all ${
            activeTab === 'dag'
              ? 'bg-nexus-accent/20 text-white border border-nexus-accent/30'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          Swarm DAG Architecture
        </button>
        <button
          onClick={() => setActiveTab('trace')}
          className={`px-4 py-2 rounded-xl text-xs font-medium transition-all ${
            activeTab === 'trace'
              ? 'bg-nexus-accent/20 text-white border border-nexus-accent/30'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          Execution Trace ({swarmResult ? swarmResult.trace.length : 0} Steps)
        </button>
        <button
          onClick={() => setActiveTab('critic')}
          className={`px-4 py-2 rounded-xl text-xs font-medium transition-all flex items-center space-x-2 ${
            activeTab === 'critic'
              ? 'bg-nexus-accent/20 text-white border border-nexus-accent/30'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          <ShieldCheck className="w-3.5 h-3.5 text-pink-400" />
          <span>Critic Gatekeeper</span>
          {swarmResult?.critic_verdict && (
            <span className={`px-1.5 py-0.2 text-[9px] font-mono rounded ${
              swarmResult.critic_verdict.gatekeeper_verdict === 'PASS' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'
            }`}>
              {swarmResult.critic_verdict.gatekeeper_verdict}
            </span>
          )}
        </button>
        <button
          onClick={() => setActiveTab('recommendations')}
          className={`px-4 py-2 rounded-xl text-xs font-medium transition-all ${
            activeTab === 'recommendations'
              ? 'bg-nexus-accent/20 text-white border border-nexus-accent/30'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          Decision Intelligence ({swarmResult ? swarmResult.recommendations.length : 0})
        </button>
        <button
          onClick={() => setActiveTab('dossier')}
          className={`px-4 py-2 rounded-xl text-xs font-medium transition-all ${
            activeTab === 'dossier'
              ? 'bg-nexus-accent/20 text-white border border-nexus-accent/30'
              : 'text-slate-400 hover:text-white'
          }`}
        >
          Executive Dossier
        </button>
      </div>

      {/* TAB 1: Swarm DAG Architecture */}
      {activeTab === 'dag' && (
        <div className="space-y-6">
          <div className="glass-panel p-6 rounded-2xl border-slate-800">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-sm font-semibold text-white">Multi-Agent Swarm DAG State</h2>
              {swarmResult && (
                <div className="flex items-center space-x-2 text-xs font-mono">
                  <span className="text-slate-400">Run ID:</span>
                  <span className="text-nexus-accent">{swarmResult.job_id}</span>
                  <span className="text-slate-400 ml-2">Status:</span>
                  <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 uppercase text-[10px]">
                    {swarmResult.status}
                  </span>
                </div>
              )}
            </div>

            {/* Grid of Agent Nodes in DAG Flow */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {agentsList.map((agent) => {
                const Icon = agent.icon;
                const status = getAgentStepStatus(agent.id);
                return (
                  <div
                    key={agent.id}
                    className={`p-4 rounded-xl border bg-slate-900/70 transition-all ${agent.color}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-center space-x-2.5">
                        <div className="p-2 rounded-lg bg-slate-800/80">
                          <Icon className="w-5 h-5" />
                        </div>
                        <div>
                          <h3 className="text-xs font-semibold text-white">{agent.name}</h3>
                          <p className="text-[11px] text-slate-400">{agent.role}</p>
                        </div>
                      </div>
                    </div>

                    <div className="mt-4 pt-3 border-t border-slate-800/60 flex items-center justify-between text-[10px] font-mono">
                      <span className="text-slate-500">Status:</span>
                      <span className={`px-2 py-0.5 rounded uppercase font-semibold ${
                        status === 'completed' || status === 'verified'
                          ? 'bg-emerald-500/20 text-emerald-300'
                          : status === 'running'
                          ? 'bg-nexus-accent/20 text-nexus-accent animate-pulse'
                          : status === 'rejected'
                          ? 'bg-red-500/20 text-red-300'
                          : 'bg-slate-800 text-slate-400'
                      }`}>
                        {status}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Critic Verification Highlight Box */}
          {swarmResult?.critic_verdict && (
            <div className="glass-panel p-5 rounded-2xl border-pink-500/30 bg-gradient-to-r from-pink-500/5 to-purple-500/5">
              <div className="flex items-start justify-between">
                <div className="flex items-center space-x-3">
                  <div className="p-2.5 rounded-xl bg-pink-500/20 border border-pink-500/40 text-pink-400">
                    <ShieldCheck className="w-6 h-6" />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                      <span>Critic Gatekeeper Verification:</span>
                      <span className="text-emerald-400 font-mono">
                        {swarmResult.critic_verdict.gatekeeper_verdict}
                      </span>
                    </h3>
                    <p className="text-xs text-slate-300 mt-0.5">
                      Audited {swarmResult.critic_verdict.audit_count} analytical claims against deterministic computational arrays. 
                      Verified: {swarmResult.critic_verdict.supported_count} | Confidence: {(swarmResult.critic_verdict.overall_confidence * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
                <button
                  onClick={() => setActiveTab('critic')}
                  className="px-3 py-1.5 rounded-lg bg-pink-500/20 text-pink-300 hover:bg-pink-500/30 text-xs font-semibold transition-colors"
                >
                  Inspect Mathematical Proofs
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* TAB 2: Execution Trace Timeline */}
      {activeTab === 'trace' && swarmResult && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Steps List */}
          <div className="lg:col-span-1 space-y-2">
            <h3 className="text-xs font-mono uppercase text-slate-400 font-semibold px-1">Pipeline Steps</h3>
            {swarmResult.trace.map((step, idx) => (
              <button
                key={step.step_id}
                onClick={() => setSelectedStep(step)}
                className={`w-full p-3.5 rounded-xl text-left border transition-all ${
                  selectedStep?.step_id === step.step_id
                    ? 'bg-slate-800/90 border-nexus-accent text-white shadow-glow'
                    : 'bg-slate-900/60 border-slate-800/80 text-slate-300 hover:bg-slate-800/40'
                }`}
              >
                <div className="flex items-center justify-between text-[11px] font-mono mb-1">
                  <span className="text-nexus-accent">Step {idx + 1}</span>
                  <span className="text-slate-400">{step.duration_ms.toFixed(1)}ms</span>
                </div>
                <div className="text-xs font-semibold">{step.agent_name}</div>
                <p className="text-[11px] text-slate-400 truncate mt-1">{step.summary}</p>
              </button>
            ))}
          </div>

          {/* Step Detail Card */}
          <div className="lg:col-span-2">
            {selectedStep ? (
              <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
                <div className="flex items-start justify-between border-b border-slate-800 pb-4">
                  <div>
                    <span className="text-[10px] font-mono uppercase text-nexus-accent">Agent Step Trace</span>
                    <h3 className="text-base font-bold text-white mt-0.5">{selectedStep.agent_name}</h3>
                    <p className="text-xs text-slate-400 mt-1">{selectedStep.summary}</p>
                  </div>
                  <div className="text-right">
                    <span className="px-2.5 py-1 rounded-full text-[10px] font-mono bg-emerald-500/20 text-emerald-300 uppercase">
                      {selectedStep.status}
                    </span>
                    <p className="text-[11px] font-mono text-slate-400 mt-1">{selectedStep.duration_ms.toFixed(1)} ms</p>
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-semibold text-slate-300 mb-1">Agent Reasoning:</h4>
                  <p className="text-xs text-slate-400 bg-slate-900/70 p-3 rounded-xl border border-slate-800">
                    {selectedStep.reasoning || 'Executed standard deterministic routines.'}
                  </p>
                </div>

                <div>
                  <h4 className="text-xs font-semibold text-slate-300 mb-1">Tools Executed:</h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedStep.tools_used.map((t) => (
                      <span key={t} className="px-2.5 py-1 rounded-lg text-xs font-mono bg-slate-800 text-slate-300 border border-slate-700">
                        {t}()
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-xs font-semibold text-slate-300 mb-1">Generated Data Artifacts:</h4>
                  <pre className="text-[11px] font-mono text-slate-300 bg-slate-950 p-3.5 rounded-xl border border-slate-800 overflow-x-auto max-h-64">
                    {JSON.stringify(selectedStep.data_artifacts, null, 2)}
                  </pre>
                </div>
              </div>
            ) : (
              <div className="glass-panel p-12 rounded-2xl border-slate-800 text-center text-slate-400 text-xs">
                Select a step on the left to view full execution reasoning and tool artifacts.
              </div>
            )}
          </div>
        </div>
      )}

      {/* TAB 3: Critic Gatekeeper Proofs */}
      {activeTab === 'critic' && swarmResult?.critic_verdict && (
        <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-6">
          <div className="flex items-center justify-between border-b border-slate-800 pb-4">
            <div>
              <h2 className="text-base font-bold text-white flex items-center space-x-2">
                <ShieldCheck className="w-5 h-5 text-pink-400" />
                <span>Deterministic Evidence Audits</span>
              </h2>
              <p className="text-xs text-slate-400 mt-1">
                Every numerical assertion made by agents is strictly tested against computed ground-truth arrays.
              </p>
            </div>
            <div className="text-right">
              <span className="text-xs font-mono text-slate-400">Gatekeeper Verdict:</span>
              <div className="text-lg font-bold text-emerald-400 font-mono">
                {swarmResult.critic_verdict.gatekeeper_verdict}
              </div>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-900/80 text-slate-400 font-mono uppercase text-[10px]">
                <tr>
                  <th className="py-2.5 px-3">Status</th>
                  <th className="py-2.5 px-3">Claim & Metric</th>
                  <th className="py-2.5 px-3">Reported</th>
                  <th className="py-2.5 px-3">Ground Truth</th>
                  <th className="py-2.5 px-3">Discrepancy (Δ)</th>
                  <th className="py-2.5 px-3">Confidence</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 font-mono">
                {swarmResult.critic_verdict.audits.map((a, i) => (
                  <tr key={i} className="hover:bg-slate-800/40">
                    <td className="py-3 px-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                        a.status === 'SUPPORTED'
                          ? 'bg-emerald-500/20 text-emerald-300'
                          : a.status === 'PARTIALLY_SUPPORTED'
                          ? 'bg-amber-500/20 text-amber-300'
                          : 'bg-red-500/20 text-red-300'
                      }`}>
                        {a.status}
                      </span>
                    </td>
                    <td className="py-3 px-3">
                      <div className="text-white font-sans text-xs">{a.claim}</div>
                      <div className="text-slate-500 text-[10px]">{a.metric_name}</div>
                    </td>
                    <td className="py-3 px-3 text-slate-200">{String(a.reported_value)}</td>
                    <td className="py-3 px-3 text-emerald-400">{String(a.ground_truth_value)}</td>
                    <td className="py-3 px-3 text-slate-300">{typeof a.delta === 'number' ? `${(a.delta * 100).toFixed(2)}%` : 'N/A'}</td>
                    <td className="py-3 px-3 text-cyan-400">{(a.confidence * 100).toFixed(0)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* TAB 4: Recommendations */}
      {activeTab === 'recommendations' && swarmResult && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {swarmResult.recommendations.map((rec) => (
            <div key={rec.id} className="glass-panel p-5 rounded-2xl border-slate-800 space-y-3">
              <div className="flex items-start justify-between">
                <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-nexus-accent/10 text-nexus-accent border border-nexus-accent/30">
                  {rec.category}
                </span>
                <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                  rec.risk_level === 'LOW' ? 'bg-emerald-500/20 text-emerald-300' : 'bg-amber-500/20 text-amber-300'
                }`}>
                  Risk: {rec.risk_level}
                </span>
              </div>
              <h3 className="text-sm font-bold text-white">{rec.title}</h3>
              <p className="text-xs text-slate-300 leading-relaxed">{rec.action}</p>
              <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800/80 text-[11px] text-slate-400">
                <span className="text-slate-300 font-semibold">Expected Impact:</span> {rec.expected_impact}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* TAB 5: Executive Dossier */}
      {activeTab === 'dossier' && swarmResult?.report_markdown && (
        <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h2 className="text-sm font-semibold text-white flex items-center space-x-2">
              <FileText className="w-4 h-4 text-nexus-accent" />
              <span>Executive Analytical Dossier</span>
            </h2>
            <button
              onClick={() => navigator.clipboard.writeText(swarmResult.report_markdown || '')}
              className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold"
            >
              Copy Markdown
            </button>
          </div>
          <pre className="text-xs font-mono text-slate-300 bg-slate-950 p-5 rounded-xl border border-slate-800 overflow-x-auto whitespace-pre-wrap leading-relaxed">
            {swarmResult.report_markdown}
          </pre>
        </div>
      )}
    </div>
  );
};
