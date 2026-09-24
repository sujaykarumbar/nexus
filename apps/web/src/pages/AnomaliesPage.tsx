import React, { useState, useEffect } from 'react';
import { 
  AlertOctagon, 
  ShieldAlert, 
  Play, 
  Activity, 
  Sliders, 
  CheckCircle2, 
  AlertTriangle, 
  RefreshCw, 
  Zap, 
  Search, 
  ArrowDownRight, 
  ArrowUpRight,
  Filter,
  Layers,
  Thermometer,
  Gauge
} from 'lucide-react';
import { api, anomalyApi } from '../services/api';
import { 
  Project, 
  Dataset, 
  AnomalyReportResponse, 
  AnomalyEventItem, 
  ChangePointItem 
} from '../types';

export const AnomaliesPage: React.FC = () => {
  // Global Project & Dataset
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>('');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');
  const [columns, setColumns] = useState<string[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string>('');
  const [includeDeepLearning, setIncludeDeepLearning] = useState<boolean>(true);

  // Scan & Job State
  const [isScanning, setIsScanning] = useState(false);
  const [scanJobId, setScanJobId] = useState<string | null>(null);
  const [scanProgress, setScanProgress] = useState<number>(0);
  const [scanStage, setScanStage] = useState<string>('');

  // Report State
  const [report, setReport] = useState<AnomalyReportResponse | null>(null);
  const [severityFilter, setSeverityFilter] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      const projs = await api.getProjects();
      setProjects(projs);
      if (projs.length > 0) {
        setSelectedProjectId(projs[0].id);
      }
    } catch (err) {
      console.error('Failed to load projects', err);
    }
  };

  useEffect(() => {
    if (selectedProjectId) {
      loadDatasets(selectedProjectId);
    }
  }, [selectedProjectId]);

  const loadDatasets = async (projId: string) => {
    try {
      const data = await api.getDatasets(projId);
      setDatasets(data);
      if (data.length > 0) {
        setSelectedDatasetId(data[0].id);
      } else {
        setSelectedDatasetId('');
        setReport(null);
      }
    } catch (err) {
      console.error('Failed to load datasets', err);
    }
  };

  useEffect(() => {
    if (selectedDatasetId) {
      loadColumnsAndExistingReport(selectedDatasetId);
    }
  }, [selectedDatasetId]);

  const loadColumnsAndExistingReport = async (dsId: string) => {
    try {
      const preview = await api.getDatasetPreview(dsId, 1, 5);
      const numCols = preview.columns.filter((col: string) => {
        const dtype = (preview.dtypes[col] || '').toLowerCase();
        return dtype.includes('int') || dtype.includes('float') || dtype.includes('double') || dtype.includes('numeric');
      });
      setColumns(numCols);
      if (numCols.length > 0) {
        setSelectedMetric(numCols[0]);
      }

      // Check if report already exists for this dataset
      try {
        const existingReport = await anomalyApi.getLatestAnomalyReport(dsId);
        setReport(existingReport);
      } catch {
        setReport(null);
      }
    } catch (err) {
      console.error('Failed to load dataset columns', err);
    }
  };

  const handleStartScan = async () => {
    if (!selectedProjectId || !selectedDatasetId) return;

    setIsScanning(true);
    setScanProgress(5);
    setScanStage('QUEUED');

    try {
      const res = await anomalyApi.startAnomalyDetection({
        project_id: selectedProjectId,
        dataset_id: selectedDatasetId,
        target_column: selectedMetric || undefined,
        include_deep_learning: includeDeepLearning,
      });

      setScanJobId(res.job_id);
      pollScanJob(res.job_id);
    } catch (err) {
      console.error('Failed to start anomaly scan', err);
      setIsScanning(false);
    }
  };

  const pollScanJob = (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const job = await anomalyApi.getAnomalyJobStatus(jobId);
        setScanProgress(job.progress || 10);
        setScanStage(job.stage || job.status);

        if (job.status === 'completed') {
          clearInterval(interval);
          setIsScanning(false);
          // Fetch report
          const rep = await anomalyApi.getLatestAnomalyReport(selectedDatasetId);
          setReport(rep);
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setIsScanning(false);
          alert(`Anomaly scan failed: ${job.error_message}`);
        }
      } catch (err) {
        console.error('Error polling anomaly job', err);
      }
    }, 2000);
  };

  // Filtered Anomalies
  const filteredAnomalies = (report?.anomalies || []).filter(item => {
    const matchesSeverity = severityFilter === 'ALL' || item.severity === severityFilter;
    const matchesSearch = searchQuery === '' || 
      item.timestamp.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.explanation.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.metric_name.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSeverity && matchesSearch;
  });

  const getSeverityBadge = (sev: string) => {
    switch (sev) {
      case 'CRITICAL':
        return 'bg-rose-500/10 border-rose-500/30 text-rose-400 font-bold';
      case 'HIGH':
        return 'bg-orange-500/10 border-orange-500/30 text-orange-400 font-bold';
      case 'MEDIUM':
        return 'bg-amber-500/10 border-amber-500/30 text-amber-300 font-medium';
      default:
        return 'bg-sky-500/10 border-sky-500/30 text-sky-400 font-medium';
    }
  };

  // Render SVG Timeline Chart
  const renderTimelineChart = () => {
    if (!report || !report.timeline || report.timeline.length === 0) return null;

    const timeline = report.timeline;
    const allVals = timeline.map(t => t.actual);
    const minVal = Math.min(...allVals) * 0.95;
    const maxVal = Math.max(...allVals) * 1.05;
    const valRange = maxVal - minVal || 1;

    const svgWidth = 720;
    const svgHeight = 240;
    const paddingLeft = 55;
    const paddingRight = 30;
    const paddingTop = 20;
    const paddingBottom = 35;
    const plotWidth = svgWidth - paddingLeft - paddingRight;
    const plotHeight = svgHeight - paddingTop - paddingBottom;

    const getX = (idx: number) => paddingLeft + (idx / Math.max(1, timeline.length - 1)) * plotWidth;
    const getY = (val: number) => paddingTop + plotHeight - ((val - minVal) / valRange) * plotHeight;

    const actualCoords = timeline.map((pt, i) => `${getX(i)},${getY(pt.actual)}`);
    const actualPath = `M ${actualCoords.join(' L ')}`;

    const expectedCoords = timeline.map((pt, i) => `${getX(i)},${getY(pt.expected)}`);
    const expectedPath = `M ${expectedCoords.join(' L ')}`;

    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Activity className="w-4 h-4 text-rose-400" />
              Observed Trajectory vs Expected Normal Baseline
            </h4>
            <p className="text-xs text-slate-400">
              Telemetry: <span className="text-rose-300 font-mono font-semibold">{report.analyzed_metric}</span>
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-0.5 bg-slate-400 border-dashed border-t"></div>
              <span className="text-slate-400">Expected Normal</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-0.5 bg-sky-400"></div>
              <span className="text-slate-300">Observed Series</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
              <span className="text-slate-300 font-medium">Anomaly Breach</span>
            </div>
          </div>
        </div>

        <div className="w-full overflow-x-auto">
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto min-w-[600px]">
            {/* Horizontal Gridlines */}
            {[0, 0.25, 0.5, 0.75, 1.0].map((frac, i) => {
              const y = paddingTop + frac * plotHeight;
              const val = maxVal - frac * valRange;
              return (
                <g key={i}>
                  <line x1={paddingLeft} y1={y} x2={svgWidth - paddingRight} y2={y} stroke="#334155" strokeDasharray="3 3" />
                  <text x={paddingLeft - 10} y={y + 4} fill="#94a3b8" fontSize="10" textAnchor="end">
                    {val.toFixed(0)}
                  </text>
                </g>
              );
            })}

            {/* Expected baseline path */}
            <path d={expectedPath} fill="none" stroke="#64748b" strokeWidth="1.5" strokeDasharray="3 3" />

            {/* Observed path */}
            <path d={actualPath} fill="none" stroke="#38bdf8" strokeWidth="2" />

            {/* Anomaly Scatter Points */}
            {timeline.map((pt, idx) => {
              if (!pt.is_anomaly) return null;
              const pointColor = pt.severity === 'CRITICAL' ? '#f43f5e' : pt.severity === 'HIGH' ? '#f97316' : '#eab308';
              return (
                <g key={idx}>
                  <circle
                    cx={getX(idx)}
                    cy={getY(pt.actual)}
                    r="5"
                    fill={pointColor}
                    stroke="#ffffff"
                    strokeWidth="1.5"
                    className="hover:r-7 transition-all cursor-pointer"
                  >
                    <title>{`Timestamp: ${pt.timestamp}\nValue: ${pt.actual}\nExpected: ${pt.expected}\nSeverity: ${pt.severity || 'ANOMALY'}`}</title>
                  </circle>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Page Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-tr from-rose-600 to-amber-600 rounded-xl text-white shadow-lg shadow-rose-500/20">
              <AlertOctagon className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">Anomaly Detection Studio</h1>
              <p className="text-sm text-slate-400">
                Multi-Detector Consensus (Isolation Forest, Autoencoder, Change-Point & Sensor Bounds)
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs rounded-full font-medium flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Real Engine Active
          </span>
          <span className="px-3 py-1 bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs rounded-full font-medium">
            Deterministic Severity Rules
          </span>
        </div>
      </div>

      {/* Configuration & Trigger Card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 shadow-xl backdrop-blur-sm">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Project */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              1. Project
            </label>
            <select
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
            >
              {projects.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>

          {/* Dataset */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              2. Target Dataset
            </label>
            <select
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
            >
              {datasets.map(d => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>

          {/* Metric Column */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              3. Telemetry / Metric
            </label>
            <select
              value={selectedMetric}
              onChange={(e) => setSelectedMetric(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-rose-500"
            >
              {columns.map(c => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Deep Learning Toggle */}
          <div className="flex flex-col justify-between">
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              4. Deep Learning Autoencoder
            </label>
            <label className="flex items-center gap-2 cursor-pointer bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-slate-300">
              <input
                type="checkbox"
                checked={includeDeepLearning}
                onChange={(e) => setIncludeDeepLearning(e.target.checked)}
                className="rounded border-slate-800 text-rose-500 focus:ring-rose-500 bg-slate-900"
              />
              <span>Enable PyTorch Reconstruction</span>
            </label>
          </div>
        </div>

        {/* Action Button & Progress */}
        <div className="mt-6 pt-6 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="w-full sm:w-2/3">
            {isScanning && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-slate-300">
                  <span className="font-medium flex items-center gap-1.5">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin text-rose-400" />
                    {scanStage || 'Scanning for Anomalies...'}
                  </span>
                  <span className="font-mono text-rose-400">{scanProgress}%</span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                  <div 
                    className="bg-gradient-to-r from-rose-500 to-amber-500 h-full transition-all duration-300"
                    style={{ width: `${scanProgress}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>

          <button
            onClick={handleStartScan}
            disabled={isScanning || !selectedDatasetId}
            className="w-full sm:w-auto px-6 py-2.5 bg-gradient-to-r from-rose-600 to-amber-600 hover:from-rose-500 hover:to-amber-500 disabled:opacity-50 text-white font-medium text-xs rounded-lg shadow-lg shadow-rose-600/20 flex items-center justify-center gap-2 transition-all"
          >
            {isScanning ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-white" />}
            <span>Run Anomaly Detection Scan</span>
          </button>
        </div>
      </div>

      {/* Report Dashboard */}
      {report && (
        <div className="space-y-6">
          {/* Summary Metric Cards */}
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">Total Scanned</span>
              <span className="text-xl font-bold text-white font-mono mt-1 block">
                {report.dataset_summary.total_observations}
              </span>
              <span className="text-[10px] text-slate-500 mt-0.5 block">Observations</span>
            </div>

            <div className="bg-slate-900/80 border border-slate-800 rounded-xl p-4 shadow-lg">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">Total Anomalies</span>
              <span className="text-xl font-bold text-amber-400 font-mono mt-1 block">
                {report.dataset_summary.total_anomalies}
              </span>
              <span className="text-[10px] text-slate-400 mt-0.5 block">
                ({report.dataset_summary.anomaly_percentage}% rate)
              </span>
            </div>

            <div className="bg-rose-950/20 border border-rose-900/30 rounded-xl p-4 shadow-lg">
              <span className="text-[11px] font-semibold text-rose-400 uppercase tracking-wider block">Critical</span>
              <span className="text-xl font-bold text-rose-400 font-mono mt-1 block">
                {report.dataset_summary.critical_count}
              </span>
              <span className="text-[10px] text-rose-400/80 mt-0.5 block">Immediate Breach</span>
            </div>

            <div className="bg-orange-950/20 border border-orange-900/30 rounded-xl p-4 shadow-lg">
              <span className="text-[11px] font-semibold text-orange-400 uppercase tracking-wider block">High Severity</span>
              <span className="text-xl font-bold text-orange-400 font-mono mt-1 block">
                {report.dataset_summary.high_count}
              </span>
              <span className="text-[10px] text-orange-400/80 mt-0.5 block">Strong Outliers</span>
            </div>

            <div className="bg-amber-950/20 border border-amber-900/30 rounded-xl p-4 shadow-lg">
              <span className="text-[11px] font-semibold text-amber-300 uppercase tracking-wider block">Medium Severity</span>
              <span className="text-xl font-bold text-amber-300 font-mono mt-1 block">
                {report.dataset_summary.medium_count}
              </span>
              <span className="text-[10px] text-amber-300/80 mt-0.5 block">Deviations</span>
            </div>

            <div className="bg-purple-950/20 border border-purple-900/30 rounded-xl p-4 shadow-lg">
              <span className="text-[11px] font-semibold text-purple-300 uppercase tracking-wider block">Change Points</span>
              <span className="text-xl font-bold text-purple-300 font-mono mt-1 block">
                {report.dataset_summary.change_points_count}
              </span>
              <span className="text-[10px] text-purple-300/80 mt-0.5 block">Regime Shifts</span>
            </div>
          </div>

          {/* Timeline Chart */}
          {renderTimelineChart()}

          {/* Change-Points & Regime Shifts Section */}
          {report.change_points && report.change_points.length > 0 && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 shadow-xl">
              <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4 text-purple-400" />
                Detected Change-Points & Structural Regime Shifts
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {report.change_points.map((cp, idx) => (
                  <div key={idx} className="p-4 bg-slate-950/80 border border-purple-900/30 rounded-xl text-xs space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="font-mono text-purple-300 font-semibold">{cp.timestamp}</span>
                      <span className="px-2 py-0.5 bg-purple-500/20 text-purple-300 rounded text-[10px] font-bold">
                        {cp.regime_type}
                      </span>
                    </div>
                    <div className="text-slate-300 flex items-center justify-between pt-1 border-t border-slate-800">
                      <span>Baseline Shift:</span>
                      <span className="font-mono font-bold text-white">
                        {cp.previous_mean} → {cp.new_mean}
                      </span>
                    </div>
                    <div className="text-slate-400 flex items-center justify-between">
                      <span>Magnitude:</span>
                      <span className="font-mono text-emerald-400">
                        {cp.magnitude > 0 ? `+${cp.magnitude}` : cp.magnitude} ({cp.percentage_change}%)
                      </span>
                    </div>
                    <div className="text-slate-500 text-[10px]">
                      Confidence: {(cp.confidence * 100).toFixed(0)}% evidence
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Filterable Anomaly Events Table */}
          <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
            <div className="px-6 py-4 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4">
              <div>
                <h4 className="text-xs font-semibold text-white uppercase tracking-wider">
                  Diagnostic Anomaly Log
                </h4>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Showing {filteredAnomalies.length} detected anomaly records with deterministic severity and evidence
                </p>
              </div>

              {/* Filters */}
              <div className="flex items-center gap-3">
                <div className="relative">
                  <Search className="w-3.5 h-3.5 absolute left-3 top-2.5 text-slate-500" />
                  <input
                    type="text"
                    placeholder="Search timestamp, explanation..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="bg-slate-950 border border-slate-800 rounded-lg pl-8 pr-3 py-1.5 text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-rose-500"
                  />
                </div>

                <div className="flex items-center gap-1 bg-slate-950 border border-slate-800 p-1 rounded-lg text-xs">
                  {['ALL', 'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map(sev => (
                    <button
                      key={sev}
                      onClick={() => setSeverityFilter(sev)}
                      className={`px-2.5 py-1 rounded text-[11px] font-medium transition-all ${
                        severityFilter === sev
                          ? 'bg-rose-600 text-white shadow-sm'
                          : 'text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {sev}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-slate-950 text-[11px] uppercase font-semibold text-slate-400">
                  <tr>
                    <th className="px-5 py-3">Timestamp</th>
                    <th className="px-5 py-3 text-right">Actual</th>
                    <th className="px-5 py-3 text-right">Expected</th>
                    <th className="px-5 py-3 text-right">Deviation</th>
                    <th className="px-5 py-3 text-center">Severity</th>
                    <th className="px-5 py-3 text-right">Score</th>
                    <th className="px-5 py-3">Methods</th>
                    <th className="px-5 py-3">Evidence & Recommendation</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/60 font-mono">
                  {filteredAnomalies.length > 0 ? (
                    filteredAnomalies.map((anom) => (
                      <tr key={anom.id} className="hover:bg-slate-800/30 font-sans">
                        <td className="px-5 py-3 font-mono text-slate-300 whitespace-nowrap">{anom.timestamp}</td>
                        <td className="px-5 py-3 text-right font-mono font-bold text-white">{anom.actual}</td>
                        <td className="px-5 py-3 text-right font-mono text-slate-400">{anom.expected}</td>
                        <td className={`px-5 py-3 text-right font-mono font-bold ${anom.deviation > 0 ? 'text-rose-400' : 'text-amber-400'}`}>
                          {anom.deviation > 0 ? `+${anom.deviation}` : anom.deviation}
                        </td>
                        <td className="px-5 py-3 text-center">
                          <span className={`px-2 py-0.5 border rounded text-[10px] ${getSeverityBadge(anom.severity)}`}>
                            {anom.severity}
                          </span>
                        </td>
                        <td className="px-5 py-3 text-right font-mono font-semibold text-rose-300">
                          {anom.anomaly_score.toFixed(2)}
                        </td>
                        <td className="px-5 py-3">
                          <div className="flex flex-wrap gap-1">
                            {anom.detection_methods.map((dm, i) => (
                              <span key={i} className="px-1.5 py-0.5 bg-slate-800 text-slate-300 rounded text-[9px]">
                                {dm}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td className="px-5 py-3 max-w-xs">
                          <div className="text-slate-300 font-medium text-xs">{anom.explanation}</div>
                          <div className="text-[11px] text-slate-500 mt-0.5">{anom.recommendation}</div>
                        </td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={8} className="px-5 py-8 text-center text-slate-500 font-sans">
                        No anomalies matching filter criteria.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
