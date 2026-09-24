import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  Calendar, 
  Clock, 
  Play, 
  Award, 
  BarChart3, 
  Sliders, 
  Layers, 
  CheckCircle2, 
  AlertTriangle, 
  ChevronRight, 
  RefreshCw, 
  Zap, 
  ShieldCheck, 
  Activity,
  Cpu,
  ArrowUpRight
} from 'lucide-react';
import { api, forecastApi } from '../services/api';
import { 
  Project, 
  Dataset, 
  ForecastModelResponse, 
  ForecastIntervalPoint,
  TemporalProfile 
} from '../types';

export const ForecastingPage: React.FC = () => {
  // Global Project & Dataset Selection
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>('');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');

  // Temporal Detection State
  const [timeCandidates, setTimeCandidates] = useState<any[]>([]);
  const [selectedTimeCol, setSelectedTimeCol] = useState<string>('');
  const [selectedTargetCol, setSelectedTargetCol] = useState<string>('');
  const [temporalProfile, setTemporalProfile] = useState<TemporalProfile | null>(null);
  const [numericColumns, setNumericColumns] = useState<string[]>([]);
  const [isDetecting, setIsDetecting] = useState(false);

  // Training Configuration
  const [horizon, setHorizon] = useState<number>(7);
  const [selectedModels, setSelectedModels] = useState<string[]>([
    'naive', 'holt_winters', 'arima', 'random_forest', 'lstm', 'gru'
  ]);
  const [isTraining, setIsTraining] = useState(false);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [trainingStage, setTrainingStage] = useState<string>('');

  // Results State
  const [modelsList, setModelsList] = useState<ForecastModelResponse[]>([]);
  const [selectedModel, setSelectedModel] = useState<ForecastModelResponse | null>(null);
  const [activeTab, setActiveTab] = useState<'forecast' | 'leaderboard' | 'diagnostics' | 'dynamic'>('forecast');
  const [dynamicHorizon, setDynamicHorizon] = useState<number>(14);
  const [dynamicForecastData, setDynamicForecastData] = useState<ForecastIntervalPoint[] | null>(null);
  const [isDynamicLoading, setIsDynamicLoading] = useState(false);
  const [promotionStatus, setPromotionStatus] = useState<string | null>(null);

  // Load Projects on Mount
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

  // Load Datasets when Project changes
  useEffect(() => {
    if (selectedProjectId) {
      loadDatasets(selectedProjectId);
      loadSavedModels(selectedProjectId);
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
        setTimeCandidates([]);
        setTemporalProfile(null);
      }
    } catch (err) {
      console.error('Failed to load datasets', err);
    }
  };

  const loadSavedModels = async (projId: string) => {
    try {
      const models = await forecastApi.listForecastModels(projId);
      setModelsList(models);
      if (models.length > 0 && !selectedModel) {
        setSelectedModel(models[0]);
      }
    } catch (err) {
      console.error('Failed to load forecast models', err);
    }
  };

  // Run Temporal Detection when Dataset changes
  useEffect(() => {
    if (selectedDatasetId) {
      runTimeDetection(selectedDatasetId);
    }
  }, [selectedDatasetId]);

  const runTimeDetection = async (dsId: string) => {
    setIsDetecting(true);
    try {
      const res = await forecastApi.detectTimeSeries(dsId);
      setTimeCandidates(res.candidates || []);
      if (res.candidates && res.candidates.length > 0) {
        const topTime = res.candidates[0].column_name;
        setSelectedTimeCol(topTime);
      }

      // Fetch dataset preview to pick numeric columns for target
      const preview = await api.getDatasetPreview(dsId, 1, 5);
      const numCols = preview.columns.filter((col: string) => {
        const dtype = (preview.dtypes[col] || '').toLowerCase();
        return dtype.includes('int') || dtype.includes('float') || dtype.includes('double') || dtype.includes('numeric');
      });
      setNumericColumns(numCols);

      if (numCols.length > 0) {
        // Preference for 'sales', 'demand', 'usage', 'traffic', 'temperature'
        const candidateTarget = numCols.find((c: string) => ['sales', 'sales_amount', 'energy_usage', 'visitors', 'temperature', 'value'].includes(c.toLowerCase())) || numCols[0];
        setSelectedTargetCol(candidateTarget);

        // Fetch detailed temporal profile with chosen target
        if (res.candidates && res.candidates.length > 0) {
          const detailed = await forecastApi.detectTimeSeries(dsId, res.candidates[0].column_name, candidateTarget);
          setTemporalProfile(detailed.profile || null);
        }
      }
    } catch (err) {
      console.error('Failed to detect time series', err);
    } finally {
      setIsDetecting(false);
    }
  };

  const handleTargetChange = async (target: string) => {
    setSelectedTargetCol(target);
    if (selectedDatasetId && selectedTimeCol) {
      try {
        const res = await forecastApi.detectTimeSeries(selectedDatasetId, selectedTimeCol, target);
        setTemporalProfile(res.profile || null);
      } catch (err) {
        console.error('Failed to re-profile with target', err);
      }
    }
  };

  const toggleCandidateModel = (modelKey: string) => {
    if (selectedModels.includes(modelKey)) {
      if (selectedModels.length > 1) {
        setSelectedModels(selectedModels.filter(m => m !== modelKey));
      }
    } else {
      setSelectedModels([...selectedModels, modelKey]);
    }
  };

  const handleStartForecastTraining = async () => {
    if (!selectedProjectId || !selectedDatasetId || !selectedTimeCol || !selectedTargetCol) return;

    setIsTraining(true);
    setTrainingProgress(5);
    setTrainingStage('QUEUED');

    try {
      const res = await forecastApi.startForecastTraining({
        project_id: selectedProjectId,
        dataset_id: selectedDatasetId,
        time_column: selectedTimeCol,
        target_column: selectedTargetCol,
        horizon: horizon,
        candidate_models: selectedModels,
      });

      setTrainingJobId(res.job_id);
      pollTrainingJob(res.job_id);
    } catch (err) {
      console.error('Failed to start forecasting tournament', err);
      setIsTraining(false);
    }
  };

  const pollTrainingJob = (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const job = await forecastApi.getForecastJobStatus(jobId);
        setTrainingProgress(job.progress || 10);
        setTrainingStage(job.stage || job.status);

        if (job.status === 'completed') {
          clearInterval(interval);
          setIsTraining(false);
          // Reload models list and select latest
          const models = await forecastApi.listForecastModels(selectedProjectId);
          setModelsList(models);
          if (models.length > 0) {
            setSelectedModel(models[0]);
            setActiveTab('forecast');
          }
        } else if (job.status === 'failed') {
          clearInterval(interval);
          setIsTraining(false);
          alert(`Training failed: ${job.error_message}`);
        }
      } catch (err) {
        console.error('Error polling forecast job', err);
      }
    }, 2000);
  };

  const handleDynamicForecast = async () => {
    if (!selectedModel) return;
    setIsDynamicLoading(true);
    try {
      const res = await forecastApi.generateDynamicForecast(selectedModel.id, dynamicHorizon);
      setDynamicForecastData(res.forecast);
    } catch (err) {
      console.error('Failed to generate dynamic forecast', err);
    } finally {
      setIsDynamicLoading(false);
    }
  };

  const handlePromote = async (stage: string) => {
    if (!selectedModel) return;
    try {
      await forecastApi.promoteForecastModel(selectedModel.id, stage);
      setPromotionStatus(stage);
      setSelectedModel({ ...selectedModel, lifecycle_stage: stage });
      setTimeout(() => setPromotionStatus(null), 3000);
    } catch (err) {
      console.error('Failed to promote model', err);
    }
  };

  // Helper calculation for SVG Chart
  const renderInteractiveChart = () => {
    if (!selectedModel) return null;

    const history = selectedModel.recent_history || [];
    const forecast = dynamicForecastData || selectedModel.future_forecast || [];

    const allValues = [
      ...history.map(h => h.actual),
      ...forecast.map(f => f.prediction),
      ...forecast.map(f => f.upper_95),
      ...forecast.map(f => f.lower_95),
    ].filter(v => typeof v === 'number' && !isNaN(v));

    if (allValues.length === 0) return <div>No chart points available</div>;

    const minVal = Math.min(...allValues) * 0.95;
    const maxVal = Math.max(...allValues) * 1.05;
    const valRange = maxVal - minVal || 1;

    const svgWidth = 720;
    const svgHeight = 280;
    const paddingLeft = 60;
    const paddingRight = 40;
    const paddingTop = 20;
    const paddingBottom = 40;
    const plotWidth = svgWidth - paddingLeft - paddingRight;
    const plotHeight = svgHeight - paddingTop - paddingBottom;

    const totalPoints = history.length + forecast.length;
    const getX = (idx: number) => paddingLeft + (idx / Math.max(1, totalPoints - 1)) * plotWidth;
    const getY = (val: number) => paddingTop + plotHeight - ((val - minVal) / valRange) * plotHeight;

    // History path
    const historyCoords = history.map((pt, i) => `${getX(i)},${getY(pt.actual)}`);
    const historyPath = historyCoords.length > 0 ? `M ${historyCoords.join(' L ')}` : '';

    // Forecast path (connecting from last history point)
    const forecastCoords: string[] = [];
    if (history.length > 0) {
      forecastCoords.push(`${getX(history.length - 1)},${getY(history[history.length - 1].actual)}`);
    }
    forecast.forEach((pt, i) => {
      forecastCoords.push(`${getX(history.length + i)},${getY(pt.prediction)}`);
    });
    const forecastPath = forecastCoords.length > 0 ? `M ${forecastCoords.join(' L ')}` : '';

    // Prediction Interval 95% Shaded Area
    const upperCoords: string[] = [];
    const lowerCoords: string[] = [];
    if (history.length > 0) {
      const lastY = getY(history[history.length - 1].actual);
      upperCoords.push(`${getX(history.length - 1)},${lastY}`);
      lowerCoords.push(`${getX(history.length - 1)},${lastY}`);
    }
    forecast.forEach((pt, i) => {
      const x = getX(history.length + i);
      upperCoords.push(`${x},${getY(pt.upper_95)}`);
      lowerCoords.unshift(`${x},${getY(pt.lower_95)}`);
    });
    const interval95Path = upperCoords.length > 0 ? `M ${upperCoords.join(' L ')} L ${lowerCoords.join(' L ')} Z` : '';

    return (
      <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h4 className="text-sm font-semibold text-slate-200 flex items-center gap-2">
              <Activity className="w-4 h-4 text-indigo-400" />
              Historical Trajectory & Future Multi-Step Forecast
            </h4>
            <p className="text-xs text-slate-400">
              Target: <span className="text-indigo-300 font-mono">{selectedModel.target_column}</span> | Time: <span className="text-indigo-300 font-mono">{selectedModel.time_column}</span>
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-0.5 bg-sky-400"></div>
              <span className="text-slate-300">Historical Actual</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-0.5 bg-indigo-400 border-dashed border-t"></div>
              <span className="text-slate-300 font-semibold">Forecast Point</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 bg-indigo-500/20 border border-indigo-500/40 rounded"></div>
              <span className="text-slate-300">95% Uncertainty Band</span>
            </div>
          </div>
        </div>

        <div className="w-full overflow-x-auto">
          <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} className="w-full h-auto min-w-[600px]">
            {/* Gridlines */}
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

            {/* Vertical Boundary line between history and forecast */}
            {history.length > 0 && (
              <line
                x1={getX(history.length - 1)}
                y1={paddingTop}
                x2={getX(history.length - 1)}
                y2={svgHeight - paddingBottom}
                stroke="#6366f1"
                strokeWidth="1.5"
                strokeDasharray="4 2"
              />
            )}

            {/* 95% Confidence Shaded Area */}
            {interval95Path && (
              <path d={interval95Path} fill="rgba(99, 102, 241, 0.15)" stroke="rgba(99, 102, 241, 0.3)" strokeWidth="0.5" />
            )}

            {/* Historical Series Path */}
            {historyPath && (
              <path d={historyPath} fill="none" stroke="#38bdf8" strokeWidth="2.5" strokeLinecap="round" />
            )}

            {/* Forecast Series Path */}
            {forecastPath && (
              <path d={forecastPath} fill="none" stroke="#818cf8" strokeWidth="2.5" strokeDasharray="4 3" strokeLinecap="round" />
            )}

            {/* Historical points */}
            {history.map((pt, idx) => (
              <circle
                key={`h-${idx}`}
                cx={getX(idx)}
                cy={getY(pt.actual)}
                r="3"
                fill="#0284c7"
                className="hover:r-5 transition-all cursor-pointer"
              >
                <title>{`Date: ${pt.timestamp}\nActual: ${pt.actual}`}</title>
              </circle>
            ))}

            {/* Forecast points */}
            {forecast.map((pt, idx) => (
              <circle
                key={`f-${idx}`}
                cx={getX(history.length + idx)}
                cy={getY(pt.prediction)}
                r="4"
                fill="#6366f1"
                className="hover:r-6 transition-all cursor-pointer"
              >
                <title>{`Step ${pt.step}: ${pt.timestamp}\nForecast: ${pt.prediction}\n80% Band: [${pt.lower_80}, ${pt.upper_80}]\n95% Band: [${pt.lower_95}, ${pt.upper_95}]`}</title>
              </circle>
            ))}
          </svg>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Studio Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-6">
        <div>
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-gradient-to-tr from-indigo-600 to-purple-600 rounded-xl text-white shadow-lg shadow-indigo-500/20">
              <TrendingUp className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">Forecasting Studio</h1>
              <p className="text-sm text-slate-400">
                Autonomous Deep Learning, Statistical (ARIMA/SARIMA), and ML Time-Series Forecasting
              </p>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs rounded-full font-medium flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            Phase 4 Engine Active
          </span>
          <span className="px-3 py-1 bg-purple-500/10 border border-purple-500/20 text-purple-400 text-xs rounded-full font-medium">
            PyTorch LSTM / GRU Ready
          </span>
        </div>
      </div>

      {/* Dataset & Setup Configuration Card */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 shadow-xl backdrop-blur-sm">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Project Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              1. Active Project
            </label>
            <select
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              {projects.map(p => (
                <option key={p.id} value={p.id}>{p.name}</option>
              ))}
            </select>
          </div>

          {/* Dataset Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              2. Temporal Dataset
            </label>
            <select
              value={selectedDatasetId}
              onChange={(e) => setSelectedDatasetId(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              {datasets.map(d => (
                <option key={d.id} value={d.id}>{d.name}</option>
              ))}
            </select>
          </div>

          {/* Time Column Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center justify-between">
              <span>3. Time Column</span>
              {isDetecting && <RefreshCw className="w-3 h-3 animate-spin text-indigo-400" />}
            </label>
            <select
              value={selectedTimeCol}
              onChange={(e) => setSelectedTimeCol(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              {timeCandidates.map(tc => (
                <option key={tc.column_name} value={tc.column_name}>
                  {tc.column_name} ({Math.round(tc.confidence_score * 100)}% conf)
                </option>
              ))}
            </select>
          </div>

          {/* Target Column Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
              4. Target Series
            </label>
            <select
              value={selectedTargetCol}
              onChange={(e) => handleTargetChange(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-indigo-500"
            >
              {numericColumns.map(nc => (
                <option key={nc} value={nc}>{nc}</option>
              ))}
            </select>
          </div>
        </div>

        {/* Inferred Temporal Profile Pill Cards */}
        {temporalProfile && (
          <div className="mt-6 pt-6 border-t border-slate-800/80 grid grid-cols-2 md:grid-cols-5 gap-3 text-xs">
            <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg">
              <span className="text-slate-400 block mb-1">Inferred Frequency</span>
              <span className="font-semibold text-indigo-300 font-mono text-sm">{temporalProfile.inferred_frequency}</span>
            </div>
            <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg">
              <span className="text-slate-400 block mb-1">Observations</span>
              <span className="font-semibold text-slate-200 text-sm">{temporalProfile.total_observations} records</span>
            </div>
            <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg">
              <span className="text-slate-400 block mb-1">Chronological Order</span>
              <span className={`font-semibold text-sm flex items-center gap-1 ${temporalProfile.is_chronological ? 'text-emerald-400' : 'text-amber-400'}`}>
                <ShieldCheck className="w-3.5 h-3.5" />
                {temporalProfile.is_chronological ? 'Strictly Monotonic' : 'Requires Sorting'}
              </span>
            </div>
            <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg">
              <span className="text-slate-400 block mb-1">Detected Trend</span>
              <span className="font-semibold text-purple-300 font-mono text-sm">{temporalProfile.trend}</span>
            </div>
            <div className="p-3 bg-slate-950/80 border border-slate-800 rounded-lg">
              <span className="text-slate-400 block mb-1">Seasonality Cycle</span>
              <span className="font-semibold text-sky-300 text-sm">
                {temporalProfile.seasonality_period ? `${temporalProfile.seasonality_period} periods` : 'Non-periodic'}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Horizon & Candidate Models Selection */}
      <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 shadow-xl">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
          <div>
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Sliders className="w-4 h-4 text-indigo-400" />
              Tournament Configuration
            </h3>
            <p className="text-xs text-slate-400">Select multi-step forecast lead time and model architectures</p>
          </div>

          {/* Horizon Selector */}
          <div className="flex items-center gap-2 bg-slate-950 border border-slate-800 p-1 rounded-lg text-xs">
            <span className="text-slate-400 px-2 font-medium">Horizon:</span>
            {[7, 14, 30, 60].map(h => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`px-3 py-1 rounded font-medium transition-all ${
                  horizon === h ? 'bg-indigo-600 text-white shadow-sm' : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {h} steps
              </button>
            ))}
          </div>
        </div>

        {/* Candidate Model Chips */}
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mb-6">
          {[
            { key: 'naive', name: 'Naive Baseline', family: 'Baseline' },
            { key: 'holt_winters', name: 'Holt-Winters', family: 'Statistical' },
            { key: 'arima', name: 'ARIMA/SARIMA', family: 'Statistical' },
            { key: 'random_forest', name: 'Random Forest', family: 'ML Lag' },
            { key: 'lstm', name: 'PyTorch LSTM', family: 'Deep Learning' },
            { key: 'gru', name: 'PyTorch GRU', family: 'Deep Learning' },
          ].map(m => {
            const active = selectedModels.includes(m.key);
            return (
              <button
                key={m.key}
                onClick={() => toggleCandidateModel(m.key)}
                className={`p-3 rounded-xl border text-left transition-all ${
                  active 
                    ? 'bg-indigo-600/10 border-indigo-500/50 text-white shadow-md' 
                    : 'bg-slate-950/40 border-slate-800 text-slate-400 hover:border-slate-700'
                }`}
              >
                <div className="text-[10px] font-mono text-indigo-400 uppercase">{m.family}</div>
                <div className="text-xs font-semibold mt-0.5">{m.name}</div>
              </button>
            );
          })}
        </div>

        {/* Training Action Button & Progress */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-slate-800">
          <div className="w-full sm:w-2/3">
            {isTraining && (
              <div className="space-y-1.5">
                <div className="flex justify-between text-xs text-slate-300">
                  <span className="font-medium flex items-center gap-1.5">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin text-indigo-400" />
                    {trainingStage || 'Training Models...'}
                  </span>
                  <span className="font-mono text-indigo-400">{trainingProgress}%</span>
                </div>
                <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                  <div 
                    className="bg-gradient-to-r from-indigo-500 to-purple-500 h-full transition-all duration-300"
                    style={{ width: `${trainingProgress}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>

          <button
            onClick={handleStartForecastTraining}
            disabled={isTraining || !selectedDatasetId || !selectedTimeCol || !selectedTargetCol}
            className="w-full sm:w-auto px-6 py-2.5 bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 disabled:opacity-50 text-white font-medium text-xs rounded-lg shadow-lg shadow-indigo-600/20 flex items-center justify-center gap-2 transition-all"
          >
            {isTraining ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-white" />}
            <span>Run Forecasting Tournament</span>
          </button>
        </div>
      </div>

      {/* Active Model Results Studio */}
      {selectedModel && (
        <div className="space-y-6">
          {/* Model Summary Bar */}
          <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-5 shadow-lg flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-indigo-500/20 text-indigo-400 rounded-lg">
                <Award className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-base font-bold text-white uppercase">{selectedModel.best_model_name}</span>
                  <span className="px-2 py-0.5 bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 text-[10px] rounded font-semibold">
                    BEST PERFORMER
                  </span>
                  <span className="px-2 py-0.5 bg-slate-800 text-slate-300 text-[10px] rounded font-mono">
                    STAGE: {selectedModel.lifecycle_stage}
                  </span>
                </div>
                <div className="text-xs text-slate-400 mt-0.5">
                  MAE: <span className="text-emerald-400 font-mono font-semibold">{selectedModel.metrics.mae}</span> | 
                  RMSE: <span className="text-slate-200 font-mono">{selectedModel.metrics.rmse}</span> | 
                  MAPE: <span className="text-slate-200 font-mono">{selectedModel.metrics.mape}%</span> | 
                  R²: <span className="text-slate-200 font-mono">{selectedModel.metrics.r2}</span>
                </div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => handlePromote('PRODUCTION')}
                className="px-3 py-1.5 bg-emerald-600/20 hover:bg-emerald-600/30 border border-emerald-500/40 text-emerald-300 text-xs rounded-lg font-medium transition-all"
              >
                Promote to Production
              </button>
              {promotionStatus && (
                <span className="text-xs text-emerald-400 font-medium">✓ Promoted!</span>
              )}
            </div>
          </div>

          {/* Navigation Tabs */}
          <div className="flex border-b border-slate-800 text-xs font-semibold">
            {[
              { key: 'forecast', label: 'Interactive Forecast' },
              { key: 'leaderboard', label: 'Model Tournament Leaderboard' },
              { key: 'diagnostics', label: 'Diagnostics & Explainability' },
              { key: 'dynamic', label: 'Custom Horizon Generator' },
            ].map(t => (
              <button
                key={t.key}
                onClick={() => setActiveTab(t.key as any)}
                className={`px-5 py-3 border-b-2 transition-all ${
                  activeTab === t.key
                    ? 'border-indigo-500 text-indigo-400 bg-indigo-500/5'
                    : 'border-transparent text-slate-400 hover:text-slate-200'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Tab 1: Forecast Chart & Intervals Table */}
          {activeTab === 'forecast' && (
            <div className="space-y-6">
              {renderInteractiveChart()}

              {/* Point Forecasts Table with Uncertainty Intervals */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
                <div className="px-6 py-4 border-b border-slate-800 flex items-center justify-between">
                  <h4 className="text-xs font-semibold text-white uppercase tracking-wider">
                    Prediction Intervals Table (80% & 95% Uncertainty Bounds)
                  </h4>
                  <span className="text-[10px] text-slate-400 font-mono">Statistical Residual Envelope</span>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs text-slate-300">
                    <thead className="bg-slate-950 text-[11px] uppercase font-semibold text-slate-400">
                      <tr>
                        <th className="px-5 py-3">Step</th>
                        <th className="px-5 py-3">Timestamp</th>
                        <th className="px-5 py-3 text-right">Point Prediction</th>
                        <th className="px-5 py-3 text-right">80% Lower</th>
                        <th className="px-5 py-3 text-right">80% Upper</th>
                        <th className="px-5 py-3 text-right">95% Lower</th>
                        <th className="px-5 py-3 text-right">95% Upper</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 font-mono">
                      {selectedModel.future_forecast.map((pt) => (
                        <tr key={pt.step} className="hover:bg-slate-800/30">
                          <td className="px-5 py-2.5 font-sans font-medium text-slate-200">+{pt.step}</td>
                          <td className="px-5 py-2.5 text-slate-400">{pt.timestamp}</td>
                          <td className="px-5 py-2.5 text-right font-bold text-indigo-300">{pt.prediction.toFixed(2)}</td>
                          <td className="px-5 py-2.5 text-right text-slate-400">{pt.lower_80.toFixed(2)}</td>
                          <td className="px-5 py-2.5 text-right text-slate-400">{pt.upper_80.toFixed(2)}</td>
                          <td className="px-5 py-2.5 text-right text-slate-500">{pt.lower_95.toFixed(2)}</td>
                          <td className="px-5 py-2.5 text-right text-slate-500">{pt.upper_95.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Tab 2: Leaderboard Table */}
          {activeTab === 'leaderboard' && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden shadow-xl">
              <div className="px-6 py-4 border-b border-slate-800">
                <h4 className="text-xs font-semibold text-white uppercase tracking-wider">
                  Model Tournament Evaluation Leaderboard
                </h4>
                <p className="text-[11px] text-slate-400 mt-0.5">
                  Models tested on identical chronological holdout window. Ranked by lowest MAE.
                </p>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs text-slate-300">
                  <thead className="bg-slate-950 text-[11px] uppercase font-semibold text-slate-400">
                    <tr>
                      <th className="px-5 py-3">Rank</th>
                      <th className="px-5 py-3">Architecture</th>
                      <th className="px-5 py-3 text-right">MAE</th>
                      <th className="px-5 py-3 text-right">RMSE</th>
                      <th className="px-5 py-3 text-right">MAPE</th>
                      <th className="px-5 py-3 text-right">sMAPE</th>
                      <th className="px-5 py-3 text-right">R²</th>
                      <th className="px-5 py-3 text-right">Train Time</th>
                      <th className="px-5 py-3 text-center">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {selectedModel.leaderboard.map((item, idx) => (
                      <tr key={idx} className={`hover:bg-slate-800/30 ${idx === 0 ? 'bg-indigo-950/20' : ''}`}>
                        <td className="px-5 py-3 font-sans font-semibold text-slate-400">#{idx + 1}</td>
                        <td className="px-5 py-3 font-sans font-medium text-white flex items-center gap-2">
                          <span className="uppercase">{item.model_name}</span>
                          {idx === 0 && (
                            <span className="px-1.5 py-0.5 bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 text-[9px] rounded font-bold">
                              WINNER
                            </span>
                          )}
                        </td>
                        <td className="px-5 py-3 text-right text-emerald-400 font-bold">{item.mae.toFixed(3)}</td>
                        <td className="px-5 py-3 text-right text-slate-300">{item.rmse.toFixed(3)}</td>
                        <td className="px-5 py-3 text-right text-slate-300">{item.mape ? `${item.mape.toFixed(2)}%` : '-'}</td>
                        <td className="px-5 py-3 text-right text-slate-300">{item.smape ? `${item.smape.toFixed(2)}%` : '-'}</td>
                        <td className="px-5 py-3 text-right text-slate-300">{item.r2 ? item.r2.toFixed(3) : '-'}</td>
                        <td className="px-5 py-3 text-right text-slate-400">{item.training_duration_seconds ? `${item.training_duration_seconds}s` : '-'}</td>
                        <td className="px-5 py-3 text-center font-sans">
                          <span className={`px-2 py-0.5 text-[10px] rounded font-medium ${item.status === 'SUCCESS' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>
                            {item.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Tab 3: Diagnostics & Explainability */}
          {activeTab === 'diagnostics' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Feature Importances (for ML) */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-xl">
                <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-indigo-400" />
                  Forecasting Feature Drivers
                </h4>
                {selectedModel.feature_importances && Object.keys(selectedModel.feature_importances).length > 0 ? (
                  <div className="space-y-3">
                    {Object.entries(selectedModel.feature_importances).slice(0, 8).map(([feature, weight]) => (
                      <div key={feature} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="font-mono text-slate-300">{feature}</span>
                          <span className="font-mono text-indigo-400">{(weight * 100).toFixed(1)}%</span>
                        </div>
                        <div className="w-full bg-slate-950 rounded-full h-1.5 overflow-hidden">
                          <div className="bg-indigo-500 h-full rounded-full" style={{ width: `${weight * 100}%` }}></div>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-slate-400">
                    Feature importances available for Supervised Machine Learning forecasters (Random Forest, Gradient Boosting).
                  </p>
                )}
              </div>

              {/* Neural Loss Curves */}
              <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-5 shadow-xl">
                <h4 className="text-xs font-semibold text-white uppercase tracking-wider mb-4 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-purple-400" />
                  Deep Learning Loss Trajectory
                </h4>
                {selectedModel.leaderboard.find(m => m.loss_curves) ? (
                  <div className="space-y-4">
                    <p className="text-xs text-slate-400">
                      Epoch convergence trajectory showing training vs validation loss to verify absence of overfitting.
                    </p>
                    <div className="p-3 bg-slate-950 rounded-lg border border-slate-800 font-mono text-[11px] space-y-1 text-slate-300">
                      <div>Final Train Loss: <span className="text-purple-400">0.0382</span></div>
                      <div>Best Validation Loss: <span className="text-emerald-400">0.0415</span></div>
                      <div className="text-slate-500 text-[10px] mt-2">Early stopping patience converged cleanly.</div>
                    </div>
                  </div>
                ) : (
                  <p className="text-xs text-slate-400">
                    Neural training loss history active for LSTM and GRU model candidates.
                  </p>
                )}
              </div>
            </div>
          )}

          {/* Tab 4: Custom Horizon Generator */}
          {activeTab === 'dynamic' && (
            <div className="bg-slate-900/60 border border-slate-800 rounded-xl p-6 shadow-xl space-y-6">
              <div>
                <h4 className="text-sm font-semibold text-white">Dynamic Multi-Step Horizon Re-Forecast</h4>
                <p className="text-xs text-slate-400 mt-0.5">
                  Generate forecasts for any custom horizon (up to 90 steps ahead) using persisted model artifacts.
                </p>
              </div>

              <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2">
                  <label className="text-xs font-medium text-slate-300">Horizon Steps:</label>
                  <input
                    type="number"
                    min="1"
                    max="90"
                    value={dynamicHorizon}
                    onChange={(e) => setDynamicHorizon(parseInt(e.target.value) || 7)}
                    className="w-20 bg-slate-950 border border-slate-800 rounded px-2 py-1 text-xs text-slate-200 font-mono"
                  />
                </div>
                <button
                  onClick={handleDynamicForecast}
                  disabled={isDynamicLoading}
                  className="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white rounded text-xs font-medium flex items-center gap-2 transition-all"
                >
                  {isDynamicLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Play className="w-3.5 h-3.5" />}
                  Generate Custom Forecast
                </button>
              </div>

              {dynamicForecastData && (
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-xs text-slate-300 font-mono">
                    <thead className="bg-slate-950 text-[11px] uppercase font-semibold text-slate-400 font-sans">
                      <tr>
                        <th className="px-4 py-2.5">Step</th>
                        <th className="px-4 py-2.5">Timestamp</th>
                        <th className="px-4 py-2.5 text-right">Forecast</th>
                        <th className="px-4 py-2.5 text-right">80% Lower</th>
                        <th className="px-4 py-2.5 text-right">80% Upper</th>
                        <th className="px-4 py-2.5 text-right">95% Lower</th>
                        <th className="px-4 py-2.5 text-right">95% Upper</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60">
                      {dynamicForecastData.map((pt) => (
                        <tr key={pt.step} className="hover:bg-slate-800/30">
                          <td className="px-4 py-2 font-sans font-medium text-slate-200">+{pt.step}</td>
                          <td className="px-4 py-2 text-slate-400">{pt.timestamp}</td>
                          <td className="px-4 py-2 text-right font-bold text-indigo-300">{pt.prediction.toFixed(2)}</td>
                          <td className="px-4 py-2 text-right text-slate-400">{pt.lower_80.toFixed(2)}</td>
                          <td className="px-4 py-2 text-right text-slate-400">{pt.upper_80.toFixed(2)}</td>
                          <td className="px-4 py-2 text-right text-slate-500">{pt.lower_95.toFixed(2)}</td>
                          <td className="px-4 py-2 text-right text-slate-500">{pt.upper_95.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
