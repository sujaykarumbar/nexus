import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Sparkles, 
  ShieldAlert, 
  CheckCircle2, 
  Play, 
  Award, 
  BarChart3, 
  Sliders, 
  Layers, 
  ArrowUpRight, 
  AlertTriangle, 
  FileText, 
  ChevronRight, 
  RefreshCw, 
  Zap, 
  Send,
  Eye
} from 'lucide-react';
import { api } from '../services/api';
import { 
  Project, 
  Dataset, 
  TargetSuggestion, 
  LeakageAuditResponse, 
  MLModelSummary, 
  MLModelDetail, 
  PredictResponse 
} from '../types';

export const AutoMLPage: React.FC = () => {
  // Global Selection State
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<string>('');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');

  // Target & Problem State
  const [targetSuggestions, setTargetSuggestions] = useState<TargetSuggestion[]>([]);
  const [selectedTarget, setSelectedTarget] = useState<string>('');
  const [leakageAudit, setLeakageAudit] = useState<LeakageAuditResponse | null>(null);
  const [excludedColumns, setExcludedColumns] = useState<string[]>([]);
  const [isLoadingTarget, setIsLoadingTarget] = useState(false);
  const [isAuditingLeakage, setIsAuditingLeakage] = useState(false);

  // Training Configuration State
  const [cvSplits, setCvSplits] = useState<number>(5);
  const [optimizeHyperparameters, setOptimizeHyperparameters] = useState<boolean>(true);
  const [optunaTrials, setOptunaTrials] = useState<number>(10);
  const [selectedAlgorithms, setSelectedAlgorithms] = useState<string[]>([
    'logistic_regression',
    'random_forest',
    'hist_gradient_boosting'
  ]);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [trainingJobId, setTrainingJobId] = useState<string | null>(null);
  const [trainingProgress, setTrainingProgress] = useState<number>(0);
  const [trainingStage, setTrainingStage] = useState<string>('');

  // Models & Registry State
  const [models, setModels] = useState<MLModelSummary[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [modelDetail, setModelDetail] = useState<MLModelDetail | null>(null);
  const [isLoadingDetail, setIsLoadingDetail] = useState(false);
  const [activeDetailTab, setActiveDetailTab] = useState<'metrics' | 'features' | 'matrix' | 'card'>('metrics');

  // Prediction Sandbox State
  const [predictInputs, setPredictInputs] = useState<Record<string, any>>({});
  const [predictionResult, setPredictionResult] = useState<PredictResponse | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);

  // 1. Initial Load: Projects
  useEffect(() => {
    const fetchProjects = async () => {
      try {
        const projs = await api.listProjects();
        setProjects(projs);
        if (projs.length > 0) {
          setSelectedProjectId(projs[0].id);
        }
      } catch (err) {
        console.error('Failed to load projects:', err);
      }
    };
    fetchProjects();
  }, []);

  // 2. Load Datasets when Project changes
  useEffect(() => {
    if (!selectedProjectId) return;
    const fetchDatasets = async () => {
      try {
        const data = await api.listDatasets(selectedProjectId);
        setDatasets(data);
        if (data.length > 0) {
          setSelectedDatasetId(data[0].id);
        } else {
          setSelectedDatasetId('');
        }
      } catch (err) {
        console.error('Failed to load datasets:', err);
      }
    };
    fetchDatasets();
  }, [selectedProjectId]);

  // 3. Load Target Suggestions & Models when Dataset changes
  useEffect(() => {
    if (!selectedDatasetId) return;

    const loadTargetsAndModels = async () => {
      setIsLoadingTarget(true);
      try {
        const [targetsRes, modelsRes] = await Promise.all([
          api.getTargetSuggestions(selectedDatasetId),
          api.listMLModels(selectedProjectId, selectedDatasetId)
        ]);
        setTargetSuggestions(targetsRes.suggestions || []);
        if (targetsRes.suggestions && targetsRes.suggestions.length > 0) {
          const topCol = targetsRes.suggestions[0].column;
          setSelectedTarget(topCol);
        }
        setModels(modelsRes);
        if (modelsRes.length > 0 && !selectedModelId) {
          setSelectedModelId(modelsRes[0].id);
        }
      } catch (err) {
        console.error('Failed loading targets/models:', err);
      } finally {
        setIsLoadingTarget(false);
      }
    };
    loadTargetsAndModels();
  }, [selectedDatasetId]);

  // 4. Audit Leakage whenever selectedTarget changes
  useEffect(() => {
    if (!selectedDatasetId || !selectedTarget) return;

    const runLeakageAudit = async () => {
      setIsAuditingLeakage(true);
      try {
        const audit = await api.auditDataLeakage(selectedDatasetId, selectedTarget);
        setLeakageAudit(audit);
        // Automatically pre-check recommended drop columns
        if (audit.recommended_drop_columns && audit.recommended_drop_columns.length > 0) {
          setExcludedColumns(audit.recommended_drop_columns);
        } else {
          setExcludedColumns([]);
        }
      } catch (err) {
        console.error('Leakage audit failed:', err);
      } finally {
        setIsAuditingLeakage(false);
      }
    };
    runLeakageAudit();
  }, [selectedDatasetId, selectedTarget]);

  // 5. Load Model Detail when selectedModelId changes
  useEffect(() => {
    if (!selectedModelId) return;
    const fetchDetail = async () => {
      setIsLoadingDetail(true);
      try {
        const detail = await api.getMLModel(selectedModelId);
        setModelDetail(detail);
        // Pre-populate prediction inputs from feature names
        if (detail.feature_names) {
          const initialInputs: Record<string, any> = {};
          detail.feature_names.slice(0, 10).forEach((f) => {
            initialInputs[f] = 0;
          });
          setPredictInputs(initialInputs);
        }
        setPredictionResult(null);
      } catch (err) {
        console.error('Failed to load model details:', err);
      } finally {
        setIsLoadingDetail(false);
      }
    };
    fetchDetail();
  }, [selectedModelId]);

  // 6. Polling Job status if training is active
  useEffect(() => {
    if (!trainingJobId || !isTraining) return;

    const interval = setInterval(async () => {
      try {
        const job = await api.getJob(trainingJobId);
        setTrainingProgress(job.progress_percentage || 0);
        setTrainingStage(job.current_stage || 'Processing...');

        if (job.status === 'completed') {
          setIsTraining(false);
          setTrainingJobId(null);
          // Refresh models
          const updatedModels = await api.listMLModels(selectedProjectId, selectedDatasetId);
          setModels(updatedModels);
          if (updatedModels.length > 0) {
            setSelectedModelId(updatedModels[0].id);
          }
        } else if (job.status === 'failed') {
          setIsTraining(false);
          setTrainingJobId(null);
          alert(`AutoML training failed: ${job.error_message || 'Unknown error'}`);
        }
      } catch (err) {
        console.error('Job polling error:', err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [trainingJobId, isTraining, selectedProjectId, selectedDatasetId]);

  // Handler: Start AutoML Training
  const handleStartTraining = async () => {
    if (!selectedDatasetId || !selectedTarget) return;

    setIsTraining(true);
    setTrainingProgress(5);
    setTrainingStage('Initializing AutoML Engine...');

    try {
      const res = await api.trainAutoML({
        dataset_id: selectedDatasetId,
        target_column: selectedTarget,
        candidate_algorithms: selectedAlgorithms,
        excluded_columns: excludedColumns,
        cv_splits: cvSplits,
        optimize_hyperparameters: optimizeHyperparameters,
        optuna_trials: optunaTrials
      });
      setTrainingJobId(res.job_id);
    } catch (err: any) {
      setIsTraining(false);
      alert(`Failed to launch training: ${err.response?.data?.error || err.message}`);
    }
  };

  // Handler: Single Prediction
  const handleRunPrediction = async () => {
    if (!selectedModelId) return;
    setIsPredicting(true);
    try {
      const res = await api.predictSingle(selectedModelId, predictInputs);
      setPredictionResult(res);
    } catch (err: any) {
      alert(`Prediction failed: ${err.response?.data?.error || err.message}`);
    } finally {
      setIsPredicting(false);
    }
  };

  // Handler: Promote Model Lifecycle Stage
  const handlePromote = async (newStage: string) => {
    if (!selectedModelId) return;
    try {
      const updated = await api.promoteModel(selectedModelId, newStage);
      setModels((prev) => prev.map((m) => (m.id === selectedModelId ? { ...m, lifecycle_stage: updated.lifecycle_stage as any } : m)));
      if (modelDetail) {
        setModelDetail({ ...modelDetail, lifecycle_stage: updated.lifecycle_stage as any });
      }
    } catch (err: any) {
      alert(`Failed to promote model: ${err.response?.data?.error || err.message}`);
    }
  };

  const selectedDatasetObj = datasets.find((d) => d.id === selectedDatasetId);

  return (
    <div className="space-y-8 animate-fade-in pb-16">
      {/* Header Banner */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <div className="flex items-center space-x-2 text-nexus-accent mb-1 font-mono text-xs uppercase tracking-wider">
            <Cpu className="w-4 h-4" />
            <span>Phase 3: Autonomous Machine Learning & MLOps Studio</span>
          </div>
          <h1 className="text-2xl lg:text-3xl font-bold text-white tracking-tight flex items-center gap-3">
            AutoML Engine & Model Registry
            <span className="text-xs px-2.5 py-1 rounded-full bg-nexus-purple/20 text-nexus-purple border border-nexus-purple/30 font-mono font-medium">
              Optuna Bayesian Optimization
            </span>
          </h1>
          <p className="text-xs text-slate-400 mt-1 max-w-3xl">
            Autonomous multi-model training, leakage defense, baseline benchmarking, and automated Model Cards.
          </p>
        </div>

        {/* Dataset Selection Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          <select
            value={selectedProjectId}
            onChange={(e) => setSelectedProjectId(e.target.value)}
            className="px-3 py-2 rounded-xl bg-slate-800/80 border border-slate-700 text-xs text-slate-200 font-medium focus:outline-none focus:border-nexus-accent"
          >
            {projects.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>

          <select
            value={selectedDatasetId}
            onChange={(e) => setSelectedDatasetId(e.target.value)}
            className="px-3 py-2 rounded-xl bg-slate-800/80 border border-slate-700 text-xs text-slate-200 font-medium focus:outline-none focus:border-nexus-accent"
          >
            {datasets.map((d) => (
              <option key={d.id} value={d.id}>{d.name} ({d.row_count.toLocaleString()} rows)</option>
            ))}
          </select>
        </div>
      </div>

      {/* Main Grid: 2 Columns */}
      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
        {/* Left Column: Target Selection, Leakage Auditor, Training Setup (5 Cols) */}
        <div className="xl:col-span-5 space-y-6">
          {/* Step 1: Target Variable & Task Formulation */}
          <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-5">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-white font-semibold text-sm">
                <Sparkles className="w-4 h-4 text-nexus-accent" />
                <span>1. Intelligent Target Detection</span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400 border border-slate-700">
                Auto-Inference
              </span>
            </div>

            {isLoadingTarget ? (
              <div className="py-6 text-center text-xs text-slate-400 font-mono flex items-center justify-center gap-2">
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-nexus-accent" />
                Analyzing column distributions...
              </div>
            ) : targetSuggestions.length > 0 ? (
              <div className="space-y-3">
                <label className="text-xs text-slate-300 font-medium">Select Prediction Target:</label>
                <div className="grid grid-cols-1 gap-2">
                  {targetSuggestions.map((sug) => {
                    const isSelected = selectedTarget === sug.column;
                    return (
                      <div
                        key={sug.column}
                        onClick={() => setSelectedTarget(sug.column)}
                        className={`p-3 rounded-xl cursor-pointer border transition-all ${
                          isSelected
                            ? 'bg-nexus-accent/10 border-nexus-accent text-white shadow-glow'
                            : 'bg-slate-800/40 border-slate-800 hover:border-slate-700 text-slate-300'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-2">
                            <span className="font-mono text-xs font-semibold">{sug.column}</span>
                            <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-nexus-cyan font-mono border border-slate-700 uppercase">
                              {sug.suggested_type.replace('_', ' ')}
                            </span>
                          </div>
                          <div className="flex items-center space-x-1 text-xs font-mono text-nexus-accent font-bold">
                            <span>{(sug.confidence * 100).toFixed(0)}%</span>
                            <span className="text-[10px] text-slate-400">conf</span>
                          </div>
                        </div>
                        <p className="text-[11px] text-slate-400 mt-1">
                          {sug.reasons.join(' • ')}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="text-xs text-slate-400 py-4 text-center">
                No target columns detected. Ensure the dataset contains valid data.
              </div>
            )}
          </div>

          {/* Step 2: Data Leakage & Risk Auditor */}
          <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-white font-semibold text-sm">
                <ShieldAlert className="w-4 h-4 text-amber-400" />
                <span>2. Zero-Leakage Defense Audit</span>
              </div>
              {leakageAudit && (
                <span className={`text-[10px] font-mono px-2 py-0.5 rounded font-semibold ${
                  leakageAudit.has_critical_risk 
                    ? 'bg-red-500/20 text-red-400 border border-red-500/30' 
                    : leakageAudit.has_leakage_risk 
                    ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' 
                    : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                }`}>
                  {leakageAudit.has_critical_risk ? 'CRITICAL LEAKAGE DETECTED' : leakageAudit.has_leakage_risk ? 'WARNINGS FOUND' : 'CLEAN DATASET'}
                </span>
              )}
            </div>

            {isAuditingLeakage ? (
              <div className="py-4 text-center text-xs text-slate-400 font-mono flex items-center justify-center gap-2">
                <RefreshCw className="w-3.5 h-3.5 animate-spin text-amber-400" />
                Auditing target cross-correlation and IDs...
              </div>
            ) : leakageAudit && leakageAudit.warnings.length > 0 ? (
              <div className="space-y-3">
                <p className="text-xs text-slate-400">
                  Target contamination checks identified suspicious columns that may artificially inflate metrics:
                </p>
                <div className="space-y-2 max-h-56 overflow-y-auto pr-1">
                  {leakageAudit.warnings.map((w, idx) => (
                    <div 
                      key={idx} 
                      className={`p-3 rounded-xl border text-xs space-y-1 ${
                        w.severity === 'CRITICAL' 
                          ? 'bg-red-950/20 border-red-900/50 text-red-200' 
                          : 'bg-amber-950/20 border-amber-900/50 text-amber-200'
                      }`}
                    >
                      <div className="flex items-center justify-between font-mono font-semibold">
                        <span className="text-white">{w.column}</span>
                        <span className="text-[10px] uppercase px-1.5 py-0.2 rounded bg-slate-900/60">
                          {w.risk_type.replace(/_/g, ' ')}
                        </span>
                      </div>
                      <p className="text-[11px] text-slate-300">{w.description}</p>
                      <p className="text-[10px] text-slate-400 italic">Action: {w.recommendation}</p>
                    </div>
                  ))}
                </div>

                {leakageAudit.recommended_drop_columns.length > 0 && (
                  <div className="pt-2 border-t border-slate-800">
                    <span className="text-[11px] text-slate-400 font-mono">Excluded Feature Columns:</span>
                    <div className="flex flex-wrap gap-1.5 mt-1.5">
                      {leakageAudit.recommended_drop_columns.map((col) => (
                        <span key={col} className="px-2 py-0.5 rounded text-[11px] font-mono bg-red-500/10 text-red-400 border border-red-500/20">
                          ✕ {col}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="flex items-center gap-2 text-xs text-emerald-400 bg-emerald-950/20 border border-emerald-900/40 p-3 rounded-xl">
                <CheckCircle2 className="w-4 h-4 shrink-0" />
                <span>Zero target contamination detected. All feature vectors are safe for training.</span>
              </div>
            )}
          </div>

          {/* Step 3: AutoML Hyperparameter & Optimization Configuration */}
          <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-5">
            <div className="flex items-center space-x-2 text-white font-semibold text-sm">
              <Sliders className="w-4 h-4 text-nexus-purple" />
              <span>3. Swarm Training Hyperparameters</span>
            </div>

            {/* Candidate Algorithms Selection */}
            <div className="space-y-2">
              <label className="text-xs text-slate-300 font-medium">Candidate Algorithm Swarm:</label>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { id: 'logistic_regression', label: 'Logistic Regression' },
                  { id: 'random_forest', label: 'Random Forest' },
                  { id: 'hist_gradient_boosting', label: 'Hist Gradient Boosting' },
                  { id: 'decision_tree', label: 'Decision Tree' },
                  { id: 'gradient_boosting', label: 'Gradient Boosting' },
                  { id: 'svm', label: 'Support Vector Classifier' }
                ].map((algo) => {
                  const isChecked = selectedAlgorithms.includes(algo.id);
                  return (
                    <button
                      key={algo.id}
                      type="button"
                      onClick={() => {
                        if (isChecked) {
                          if (selectedAlgorithms.length > 1) {
                            setSelectedAlgorithms(selectedAlgorithms.filter((a) => a !== algo.id));
                          }
                        } else {
                          setSelectedAlgorithms([...selectedAlgorithms, algo.id]);
                        }
                      }}
                      className={`px-3 py-2 rounded-xl text-left text-xs font-mono transition-all border ${
                        isChecked
                          ? 'bg-nexus-purple/20 border-nexus-purple text-white'
                          : 'bg-slate-800/40 border-slate-800 text-slate-400 hover:text-slate-200'
                      }`}
                    >
                      {isChecked ? '✓ ' : '+ '}{algo.label}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* CV Splits & Optuna Configuration */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs text-slate-300 font-medium">Cross-Validation Folds:</label>
                <select
                  value={cvSplits}
                  onChange={(e) => setCvSplits(Number(e.target.value))}
                  className="w-full mt-1.5 px-3 py-2 rounded-xl bg-slate-800/80 border border-slate-700 text-xs text-slate-200 font-mono"
                >
                  <option value={3}>3-Fold Stratified</option>
                  <option value={5}>5-Fold Stratified (Recommended)</option>
                  <option value={10}>10-Fold Stratified</option>
                </select>
              </div>

              <div>
                <label className="text-xs text-slate-300 font-medium">Optuna Bayesian Trials:</label>
                <select
                  value={optunaTrials}
                  onChange={(e) => setOptunaTrials(Number(e.target.value))}
                  disabled={!optimizeHyperparameters}
                  className="w-full mt-1.5 px-3 py-2 rounded-xl bg-slate-800/80 border border-slate-700 text-xs text-slate-200 font-mono disabled:opacity-50"
                >
                  <option value={5}>5 Trials (Fast)</option>
                  <option value={10}>10 Trials (Balanced)</option>
                  <option value={20}>20 Trials (Exhaustive)</option>
                </select>
              </div>
            </div>

            {/* Optuna Toggle */}
            <div className="flex items-center justify-between p-3 rounded-xl bg-slate-800/40 border border-slate-800">
              <div>
                <div className="text-xs text-slate-200 font-medium">Optuna TPE Hyperparameter Tuning</div>
                <div className="text-[10px] text-slate-400">Explores tree depth, learning rate, and regularizations.</div>
              </div>
              <input
                type="checkbox"
                checked={optimizeHyperparameters}
                onChange={(e) => setOptimizeHyperparameters(e.target.checked)}
                className="w-4 h-4 rounded text-nexus-accent focus:ring-0 focus:outline-none"
              />
            </div>

            {/* Launch Button */}
            <button
              onClick={handleStartTraining}
              disabled={isTraining || !selectedDatasetId || !selectedTarget}
              className="w-full py-3 rounded-xl bg-gradient-to-r from-nexus-accent via-nexus-purple to-nexus-cyan text-white font-semibold text-xs tracking-wide shadow-glow hover:opacity-95 transition-all flex items-center justify-center space-x-2 disabled:opacity-50"
            >
              {isTraining ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  <span>Training in Progress ({trainingProgress}%)...</span>
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 fill-white" />
                  <span>Launch Autonomous AutoML Swarm</span>
                </>
              )}
            </button>

            {/* Live Progress Bar */}
            {isTraining && (
              <div className="space-y-1.5 pt-2">
                <div className="flex items-center justify-between text-[11px] font-mono text-slate-400">
                  <span>{trainingStage}</span>
                  <span className="text-nexus-accent font-bold">{trainingProgress}%</span>
                </div>
                <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                  <div 
                    className="bg-gradient-to-r from-nexus-accent via-nexus-purple to-nexus-cyan h-full transition-all duration-300" 
                    style={{ width: `${trainingProgress}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Model Leaderboard & Diagnostics Inspector (7 Cols) */}
        <div className="xl:col-span-7 space-y-6">
          {/* Models Leaderboard Table */}
          <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2 text-white font-semibold text-sm">
                <Award className="w-4 h-4 text-nexus-cyan" />
                <span>Model Benchmark Leaderboard</span>
              </div>
              <span className="text-[11px] font-mono text-slate-400">
                {models.length} Models Registered
              </span>
            </div>

            {models.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="text-[11px] font-mono text-slate-400 uppercase border-b border-slate-800 bg-slate-900/40">
                    <tr>
                      <th className="py-2.5 px-3">Algorithm</th>
                      <th className="py-2.5 px-3">Primary Metric</th>
                      <th className="py-2.5 px-3">Baseline Δ%</th>
                      <th className="py-2.5 px-3">Stage</th>
                      <th className="py-2.5 px-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/60 font-mono">
                    {models.map((m, idx) => {
                      const isSelected = selectedModelId === m.id;
                      return (
                        <tr 
                          key={m.id}
                          onClick={() => setSelectedModelId(m.id)}
                          className={`cursor-pointer transition-colors ${
                            isSelected 
                              ? 'bg-nexus-accent/10 text-white' 
                              : 'hover:bg-slate-800/40 text-slate-300'
                          }`}
                        >
                          <td className="py-3 px-3">
                            <div className="flex items-center space-x-2">
                              {idx === 0 && (
                                <span className="text-amber-400 text-xs" title="Top Ranked Model">★</span>
                              )}
                              <span className="font-semibold">{m.name.split('—')[1] || m.algorithm}</span>
                            </div>
                          </td>
                          <td className="py-3 px-3">
                            <span className="text-nexus-cyan font-bold">
                              {m.primary_metric_name.toUpperCase()}: {m.primary_metric_value.toFixed(4)}
                            </span>
                          </td>
                          <td className="py-3 px-3">
                            <span className="text-emerald-400 font-semibold">
                              +{m.delta_improvement_pct ? m.delta_improvement_pct.toFixed(1) : 0}%
                            </span>
                          </td>
                          <td className="py-3 px-3">
                            <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                              m.lifecycle_stage === 'PRODUCTION' 
                                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40' 
                                : m.lifecycle_stage === 'STAGING'
                                ? 'bg-nexus-accent/20 text-nexus-accent border border-nexus-accent/40'
                                : 'bg-slate-800 text-slate-400 border border-slate-700'
                            }`}>
                              {m.lifecycle_stage}
                            </span>
                          </td>
                          <td className="py-3 px-3 text-right">
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setSelectedModelId(m.id);
                              }}
                              className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 text-[11px]"
                            >
                              Inspect
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="py-12 text-center text-xs text-slate-400">
                No machine learning models trained yet for this dataset. Configure the target on the left and launch an AutoML job.
              </div>
            )}
          </div>

          {/* Model Inspector & Diagnostic Deep-Dive */}
          {selectedModelId && modelDetail && (
            <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-5">
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-4 border-b border-slate-800">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm font-bold text-white">{modelDetail.name}</span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                      {modelDetail.version}
                    </span>
                  </div>
                  <div className="text-xs text-slate-400 font-mono mt-0.5">
                    Target: {modelDetail.target_column} • Task: {modelDetail.problem_type}
                  </div>
                </div>

                {/* Stage Promotion Actions */}
                <div className="flex items-center gap-2">
                  {modelDetail.lifecycle_stage !== 'PRODUCTION' && (
                    <button
                      onClick={() => handlePromote('PRODUCTION')}
                      className="px-3 py-1.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-colors flex items-center space-x-1"
                    >
                      <ArrowUpRight className="w-3.5 h-3.5" />
                      <span>Promote to Production</span>
                    </button>
                  )}
                  {modelDetail.lifecycle_stage !== 'STAGING' && (
                    <button
                      onClick={() => handlePromote('STAGING')}
                      className="px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs transition-colors"
                    >
                      Stage Model
                    </button>
                  )}
                </div>
              </div>

              {/* Inspector Navigation Tabs */}
              <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
                {[
                  { id: 'metrics', label: 'Evaluation Metrics' },
                  { id: 'features', label: 'Feature Importance' },
                  { id: 'matrix', label: 'Confusion Matrix / Curves' },
                  { id: 'card', label: 'Model Card' }
                ].map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveDetailTab(tab.id as any)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                      activeDetailTab === tab.id
                        ? 'bg-nexus-accent text-nexus-900 font-semibold shadow-glow'
                        : 'text-slate-400 hover:text-slate-200'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              {/* Tab 1: Comprehensive Metrics Grid */}
              {activeDetailTab === 'metrics' && (
                <div className="space-y-4">
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    {Object.entries(modelDetail.all_metrics).map(([metricKey, metricVal]) => (
                      <div key={metricKey} className="p-3 rounded-xl bg-slate-900/60 border border-slate-800">
                        <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">{metricKey}</div>
                        <div className="text-lg font-bold text-white font-mono mt-1">
                          {typeof metricVal === 'number' ? metricVal.toFixed(4) : String(metricVal)}
                        </div>
                      </div>
                    ))}
                  </div>

                  {modelDetail.cv_scores && (
                    <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 space-y-2">
                      <div className="text-xs font-semibold text-slate-200">5-Fold Cross Validation Reliability:</div>
                      <div className="grid grid-cols-3 gap-2 text-xs font-mono text-slate-300">
                        <div>Mean Score: <span className="text-nexus-cyan font-bold">{modelDetail.cv_scores.mean_score?.toFixed(4)}</span></div>
                        <div>Std Dev: <span className="text-slate-400">±{modelDetail.cv_scores.std_dev?.toFixed(4)}</span></div>
                        <div>Folds Evaluated: <span className="text-white">{modelDetail.cv_scores.folds_evaluated}</span></div>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Tab 2: Feature Importance */}
              {activeDetailTab === 'features' && (
                <div className="space-y-3">
                  <p className="text-xs text-slate-400">
                    Global feature importance rankings derived from tree splits and permutation importance:
                  </p>
                  <div className="space-y-2">
                    {modelDetail.feature_importance && modelDetail.feature_importance.length > 0 ? (
                      modelDetail.feature_importance.map((f) => (
                        <div key={f.feature} className="space-y-1">
                          <div className="flex items-center justify-between text-xs font-mono">
                            <span className="text-slate-300 font-medium">#{f.rank} {f.feature}</span>
                            <span className="text-nexus-accent font-bold">{f.percentage}%</span>
                          </div>
                          <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                            <div 
                              className="bg-gradient-to-r from-nexus-accent to-nexus-purple h-full"
                              style={{ width: `${Math.min(100, f.percentage * 2.5)}%` }}
                            />
                          </div>
                        </div>
                      ))
                    ) : (
                      <div className="text-xs text-slate-400 py-4 text-center">
                        Feature importances not available for this algorithm.
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Tab 3: Confusion Matrix & ROC Curves */}
              {activeDetailTab === 'matrix' && (
                <div className="space-y-4">
                  {modelDetail.confusion_matrix ? (
                    <div className="space-y-3">
                      <div className="text-xs font-semibold text-slate-200">Confusion Matrix:</div>
                      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 overflow-x-auto">
                        <table className="text-center font-mono text-xs mx-auto">
                          <thead>
                            <tr>
                              <th className="p-2 text-slate-400">Actual \ Predicted</th>
                              {modelDetail.confusion_matrix.labels.map((lbl: any) => (
                                <th key={lbl} className="p-2 text-nexus-cyan font-bold">{String(lbl)}</th>
                              ))}
                            </tr>
                          </thead>
                          <tbody>
                            {modelDetail.confusion_matrix.matrix.map((row: number[], rowIdx: number) => (
                              <tr key={rowIdx}>
                                <td className="p-2 text-nexus-cyan font-bold">{String(modelDetail.confusion_matrix?.labels[rowIdx])}</td>
                                {row.map((val: number, colIdx: number) => (
                                  <td 
                                    key={colIdx} 
                                    className={`p-3 border border-slate-800 font-bold ${
                                      rowIdx === colIdx ? 'bg-nexus-accent/20 text-white' : 'text-slate-400'
                                    }`}
                                  >
                                    {val}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : (
                    <div className="text-xs text-slate-400 text-center py-6">
                      Confusion Matrix not applicable for regression models.
                    </div>
                  )}
                </div>
              )}

              {/* Tab 4: Standardized Model Card */}
              {activeDetailTab === 'card' && (
                <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-800 max-h-96 overflow-y-auto">
                  {modelDetail.model_card?.markdown_card ? (
                    <pre className="text-xs font-mono text-slate-300 whitespace-pre-wrap">
                      {modelDetail.model_card.markdown_card}
                    </pre>
                  ) : (
                    <div className="text-xs text-slate-400 text-center py-4">
                      Model Card documentation in progress.
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Interactive Prediction Sandbox */}
          {selectedModelId && modelDetail && (
            <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2 text-white font-semibold text-sm">
                  <Zap className="w-4 h-4 text-amber-400" />
                  <span>Real-Time Prediction & Explainability Sandbox</span>
                </div>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-400">
                  Live Inference Endpoint
                </span>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                {Object.keys(predictInputs).map((fName) => (
                  <div key={fName}>
                    <label className="text-[11px] font-mono text-slate-400 block truncate" title={fName}>
                      {fName}:
                    </label>
                    <input
                      type="text"
                      value={predictInputs[fName]}
                      onChange={(e) => setPredictInputs({ ...predictInputs, [fName]: e.target.value })}
                      className="w-full mt-1 px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700 text-xs font-mono text-white focus:outline-none focus:border-nexus-accent"
                    />
                  </div>
                ))}
              </div>

              <button
                onClick={handleRunPrediction}
                disabled={isPredicting}
                className="px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-white font-semibold text-xs transition-colors flex items-center space-x-2 disabled:opacity-50"
              >
                {isPredicting ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Predicting...</span>
                  </>
                ) : (
                  <>
                    <Send className="w-3.5 h-3.5 text-nexus-accent" />
                    <span>Generate Instant Prediction</span>
                  </>
                )}
              </button>

              {/* Prediction Output & Explainability Attribution */}
              {predictionResult && (
                <div className="p-4 rounded-xl bg-slate-900/80 border border-slate-700 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-[11px] font-mono text-slate-400 uppercase">Predicted Outcome:</div>
                      <div className="text-xl font-bold text-white font-mono mt-0.5">
                        {String(predictionResult.prediction)}
                      </div>
                    </div>
                    {predictionResult.probability !== undefined && predictionResult.probability !== null && (
                      <div className="text-right">
                        <div className="text-[11px] font-mono text-slate-400 uppercase">Confidence:</div>
                        <div className="text-lg font-bold text-nexus-accent font-mono">
                          {(predictionResult.probability * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                    <div className="text-right">
                      <div className="text-[11px] font-mono text-slate-400 uppercase">Latency:</div>
                      <div className="text-xs font-mono text-slate-300">
                        {predictionResult.latency_ms.toFixed(1)} ms
                      </div>
                    </div>
                  </div>

                  {/* Local Feature Attribution */}
                  {predictionResult.contributing_features && predictionResult.contributing_features.length > 0 && (
                    <div className="pt-2 border-t border-slate-800 space-y-2">
                      <div className="text-xs font-semibold text-slate-200">Local Feature Attribution (Why this prediction?):</div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                        {predictionResult.contributing_features.map((cf) => (
                          <div 
                            key={cf.feature} 
                            className="p-2 rounded-lg bg-slate-800/40 border border-slate-800 flex items-center justify-between text-xs font-mono"
                          >
                            <span className="text-slate-300">{cf.feature}: {cf.feature_value}</span>
                            <span className={`font-bold ${cf.direction === 'positive' ? 'text-emerald-400' : 'text-red-400'}`}>
                              {cf.direction === 'positive' ? '+' : ''}{cf.impact.toFixed(3)}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
