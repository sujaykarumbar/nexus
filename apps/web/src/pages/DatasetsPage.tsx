import React, { useState, useEffect } from 'react';
import { 
  Dataset, 
  Project, 
  DatasetProfile, 
  DataQualityReportData, 
  EDAResponseData, 
  InsightItem, 
  PaginatedPreviewData 
} from '../types';
import { api } from '../services/api';
import { 
  Database, 
  Upload, 
  Sparkles, 
  CheckCircle2, 
  AlertCircle, 
  Layers, 
  Activity, 
  BarChart3, 
  Search, 
  RefreshCw, 
  ArrowRight, 
  ChevronLeft, 
  ChevronRight, 
  ShieldCheck, 
  FileText, 
  Flame, 
  Sliders, 
  Table, 
  TrendingUp, 
  Trash2,
  PieChart
} from 'lucide-react';
import { SampleDatasetModal } from '../components/dashboard/SampleDatasetModal';
import { UploadDatasetModal } from '../components/datasets/UploadDatasetModal';

export const DatasetsPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedDatasetId, setSelectedDatasetId] = useState<string | null>(null);
  const [activeSubTab, setActiveSubTab] = useState<'quality' | 'schema' | 'eda' | 'insights' | 'preview'>('quality');
  
  // Data for active dataset
  const [profile, setProfile] = useState<DatasetProfile | null>(null);
  const [quality, setQuality] = useState<DataQualityReportData | null>(null);
  const [eda, setEda] = useState<EDAResponseData | null>(null);
  const [insights, setInsights] = useState<InsightItem[]>([]);
  const [preview, setPreview] = useState<PaginatedPreviewData | null>(null);
  const [previewPage, setPreviewPage] = useState<number>(1);
  const [previewPageSize, setPreviewPageSize] = useState<number>(15);

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isDetailLoading, setIsDetailLoading] = useState<boolean>(false);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [isSampleModalOpen, setIsSampleModalOpen] = useState<boolean>(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState<boolean>(false);
  const [columnSearch, setColumnSearch] = useState<string>('');
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const selectedDataset = datasets.find(d => d.id === selectedDatasetId) || datasets[0] || null;

  const loadInitialData = async () => {
    try {
      setIsLoading(true);
      const [datasetList, projectList] = await Promise.all([
        api.listDatasets(),
        api.listProjects()
      ]);
      setDatasets(datasetList);
      setProjects(projectList);

      if (datasetList.length > 0 && !selectedDatasetId) {
        setSelectedDatasetId(datasetList[0].id);
      }
    } catch (err) {
      console.error('Failed to load datasets:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadDatasetDetails = async (datasetId: string) => {
    try {
      setIsDetailLoading(true);
      const [profData, qualData, edaData, insData, prevData] = await Promise.all([
        api.getDatasetProfile(datasetId).catch(() => null),
        api.getDatasetQuality(datasetId).catch(() => null),
        api.getDatasetEDA(datasetId).catch(() => null),
        api.getDatasetInsights(datasetId).catch(() => []),
        api.previewDatasetPaginated(datasetId, previewPage, previewPageSize).catch(() => null)
      ]);
      setProfile(profData);
      setQuality(qualData);
      setEda(edaData);
      setInsights(insData || []);
      setPreview(prevData);
    } catch (err) {
      console.error('Failed to load dataset details:', err);
    } finally {
      setIsDetailLoading(false);
    }
  };

  useEffect(() => {
    if (selectedDataset?.id) {
      loadDatasetDetails(selectedDataset.id);
    }
  }, [selectedDataset?.id, previewPage, previewPageSize]);

  const handleTriggerAnalysis = async () => {
    if (!selectedDataset) return;
    try {
      setIsAnalyzing(true);
      await api.triggerAnalysis(selectedDataset.id);
      setStatusMessage(`Analysis pipeline triggered for '${selectedDataset.name}'.`);
      setTimeout(() => setStatusMessage(null), 4000);
      setTimeout(() => {
        loadDatasetDetails(selectedDataset.id);
        setIsAnalyzing(false);
      }, 1500);
    } catch (err: any) {
      console.error('Failed to trigger analysis:', err);
      setIsAnalyzing(false);
    }
  };

  const handleDeleteDataset = async (datasetId: string) => {
    if (!confirm('Are you sure you want to delete this dataset?')) return;
    try {
      await api.deleteDataset(datasetId);
      const updated = datasets.filter(d => d.id !== datasetId);
      setDatasets(updated);
      if (selectedDatasetId === datasetId) {
        setSelectedDatasetId(updated[0]?.id || null);
      }
      setStatusMessage('Dataset deleted.');
      setTimeout(() => setStatusMessage(null), 3000);
    } catch (err) {
      console.error('Failed to delete dataset:', err);
    }
  };

  const getGradeColor = (grade: string) => {
    switch (grade) {
      case 'A': return 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10';
      case 'B': return 'text-cyan-400 border-cyan-500/40 bg-cyan-500/10';
      case 'C': return 'text-amber-400 border-amber-500/40 bg-amber-500/10';
      default: return 'text-rose-400 border-rose-500/40 bg-rose-500/10';
    }
  };

  const getTypeBadgeColor = (type: string) => {
    switch (type) {
      case 'NUMERICAL': return 'bg-blue-500/10 text-blue-400 border-blue-500/30';
      case 'CATEGORICAL': return 'bg-purple-500/10 text-purple-400 border-purple-500/30';
      case 'DATETIME': return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
      case 'BOOLEAN': return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
      case 'TEXT': return 'bg-rose-500/10 text-rose-400 border-rose-500/30';
      case 'IDENTIFIER': return 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30';
      default: return 'bg-slate-500/10 text-slate-400 border-slate-500/30';
    }
  };

  return (
    <div className="space-y-6 pb-12">
      {/* Top Banner & Action Buttons */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 p-6 rounded-2xl glass-panel bg-gradient-to-r from-nexus-850 via-slate-900 to-indigo-950/40 border-slate-800">
        <div className="space-y-1">
          <div className="flex items-center space-x-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-nexus-accent/20 text-nexus-accent border border-nexus-accent/40">
              DATA INTELLIGENCE STUDIO
            </span>
            <span className="text-xs text-slate-400 font-mono">Statistical Profiling & Verification</span>
          </div>
          <h1 className="text-2xl font-extrabold text-white tracking-tight">
            Dataset Explorer & Schema Profiler
          </h1>
          <p className="text-xs text-slate-400 max-w-2xl">
            Deterministic data quality audits, correlation heatmaps, distribution statistics, and mathematically verified insights.
          </p>
        </div>

        <div className="flex items-center flex-wrap gap-2.5 shrink-0">
          <button
            onClick={() => setIsSampleModalOpen(true)}
            className="px-3.5 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 hover:border-slate-600 transition-all flex items-center space-x-2"
          >
            <Sparkles className="w-3.5 h-3.5 text-nexus-accent" />
            <span>Sample Datasets</span>
          </button>
          <button
            onClick={() => setIsUploadModalOpen(true)}
            className="px-4 py-2 rounded-xl bg-gradient-to-r from-nexus-accent to-nexus-purple text-white text-xs font-semibold hover:opacity-95 transition-opacity shadow-glow flex items-center space-x-2"
          >
            <Upload className="w-3.5 h-3.5" />
            <span>Upload File</span>
          </button>
        </div>
      </div>

      {statusMessage && (
        <div className="p-3 bg-nexus-accent/10 border border-nexus-accent/30 rounded-xl text-xs text-nexus-accent flex items-center space-x-2 animate-fadeIn">
          <CheckCircle2 className="w-4 h-4" />
          <span>{statusMessage}</span>
        </div>
      )}

      {/* Dataset Selector Carousel / Bar */}
      {datasets.length === 0 && !isLoading ? (
        <div className="glass-panel p-12 text-center rounded-2xl border-slate-800 space-y-4">
          <Database className="w-12 h-12 text-slate-600 mx-auto" />
          <div className="space-y-1">
            <h3 className="text-base font-semibold text-white">No Datasets Ingested Yet</h3>
            <p className="text-xs text-slate-400 max-w-md mx-auto">
              Load one of the bundled benchmark datasets or upload your own CSV, Excel, JSON, or Parquet file.
            </p>
          </div>
          <div className="flex justify-center space-x-3">
            <button
              onClick={() => setIsSampleModalOpen(true)}
              className="px-4 py-2 bg-nexus-accent text-white text-xs font-semibold rounded-xl"
            >
              Load Sample Dataset
            </button>
            <button
              onClick={() => setIsUploadModalOpen(true)}
              className="px-4 py-2 bg-slate-800 text-slate-200 text-xs font-semibold rounded-xl"
            >
              Upload Custom File
            </button>
          </div>
        </div>
      ) : (
        <div className="space-y-4">
          {/* Selector Horizontal Strip */}
          <div className="flex items-center space-x-3 overflow-x-auto pb-2 scrollbar-thin">
            {datasets.map((d) => {
              const isSelected = d.id === (selectedDataset?.id || '');
              return (
                <button
                  key={d.id}
                  onClick={() => setSelectedDatasetId(d.id)}
                  className={`flex items-center space-x-3 px-4 py-2.5 rounded-xl border text-left shrink-0 transition-all ${
                    isSelected
                      ? 'bg-nexus-850 border-nexus-accent/50 text-white shadow-glow'
                      : 'glass-panel border-slate-800 text-slate-400 hover:text-slate-200 hover:border-slate-700'
                  }`}
                >
                  <Database className={`w-4 h-4 ${isSelected ? 'text-nexus-accent' : 'text-slate-500'}`} />
                  <div>
                    <div className="text-xs font-semibold truncate max-w-[160px]">{d.name}</div>
                    <div className="text-[10px] font-mono text-slate-400 flex items-center space-x-2">
                      <span>{d.row_count} rows</span>
                      <span>•</span>
                      <span>{d.column_count} cols</span>
                      {d.detected_problem_type && (
                        <>
                          <span>•</span>
                          <span className="capitalize text-nexus-accent">{d.detected_problem_type}</span>
                        </>
                      )}
                    </div>
                  </div>
                  {d.data_quality_score !== null && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                      {d.data_quality_score.toFixed(0)}%
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Active Dataset Detail Header */}
          {selectedDataset && (
            <div className="glass-panel p-5 rounded-2xl border-slate-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <h2 className="text-lg font-bold text-white">{selectedDataset.name}</h2>
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold uppercase bg-slate-800 text-slate-300 border border-slate-700">
                    {selectedDataset.file_type}
                  </span>
                  {selectedDataset.detected_problem_type && (
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold uppercase bg-nexus-accent/20 text-nexus-accent border border-nexus-accent/30">
                      {selectedDataset.detected_problem_type}
                    </span>
                  )}
                  {selectedDataset.target_column && (
                    <span className="text-xs font-mono text-slate-400">
                      Target: <span className="text-slate-200 font-semibold">{selectedDataset.target_column}</span>
                    </span>
                  )}
                </div>
                <p className="text-xs text-slate-400">
                  {selectedDataset.description || 'Uploaded dataset with automated heuristic schema inference and quality scoring.'}
                </p>
              </div>

              <div className="flex items-center space-x-2 shrink-0">
                <button
                  onClick={handleTriggerAnalysis}
                  disabled={isAnalyzing}
                  className="px-3.5 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-xs font-semibold text-slate-200 border border-slate-700 flex items-center space-x-1.5 transition-colors disabled:opacity-50"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isAnalyzing ? 'animate-spin text-nexus-accent' : ''}`} />
                  <span>{isAnalyzing ? 'Profiling...' : 'Re-Run Profile'}</span>
                </button>
                <button
                  onClick={() => handleDeleteDataset(selectedDataset.id)}
                  className="p-2 rounded-xl bg-slate-800/80 hover:bg-red-500/20 text-slate-400 hover:text-red-400 border border-slate-700 hover:border-red-500/30 transition-colors"
                  title="Delete dataset"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* Navigation Tabs */}
          <div className="flex border-b border-slate-800 space-x-2">
            {[
              { id: 'quality', label: 'Data Quality Audit', icon: ShieldCheck, badge: quality ? `${quality.score.toFixed(0)}/100` : null },
              { id: 'schema', label: 'Schema & Profiler', icon: Sliders, badge: profile ? `${profile.column_count} Cols` : null },
              { id: 'eda', label: 'Correlations & EDA', icon: Flame, badge: eda ? `${eda.correlations.significant_correlations.length} Pairs` : null },
              { id: 'insights', label: 'Verified Insights', icon: Sparkles, badge: insights.length > 0 ? `${insights.length}` : null },
              { id: 'preview', label: 'Raw Data Table', icon: Table, badge: preview ? `${preview.total_rows}` : null },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeSubTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveSubTab(tab.id as any)}
                  className={`flex items-center space-x-2 px-4 py-3 text-xs font-semibold border-b-2 transition-all ${
                    isActive
                      ? 'border-nexus-accent text-white'
                      : 'border-transparent text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-nexus-accent' : 'text-slate-500'}`} />
                  <span>{tab.label}</span>
                  {tab.badge && (
                    <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${
                      isActive ? 'bg-nexus-accent/20 text-nexus-accent' : 'bg-slate-800 text-slate-400'
                    }`}>
                      {tab.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </div>

          {/* Tab Content Container */}
          {isDetailLoading && (
            <div className="p-12 text-center">
              <div className="w-8 h-8 border-2 border-nexus-accent border-t-transparent rounded-full animate-spin mx-auto mb-2" />
              <p className="text-xs font-mono text-slate-400">Loading statistical intelligence reports...</p>
            </div>
          )}

          {!isDetailLoading && (
            <div>
              {/* TAB 1: DATA QUALITY */}
              {activeSubTab === 'quality' && quality && (
                <div className="space-y-6">
                  {/* Quality Score Hero Card */}
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div className="glass-panel p-5 rounded-2xl border-slate-800 flex flex-col justify-between">
                      <span className="text-xs font-mono uppercase text-slate-400">Overall Quality</span>
                      <div className="my-2 flex items-baseline space-x-3">
                        <span className="text-4xl font-extrabold text-white">{quality.score.toFixed(1)}</span>
                        <span className="text-sm font-mono text-slate-400">/ 100</span>
                      </div>
                      <div className={`px-2.5 py-1 rounded-xl text-xs font-bold font-mono border inline-flex items-center justify-between ${getGradeColor(quality.grade)}`}>
                        <span>GRADE {quality.grade}</span>
                        <span>{quality.score >= 90 ? 'Production Ready' : 'Cleaning Needed'}</span>
                      </div>
                    </div>

                    <div className="glass-panel p-5 rounded-2xl border-slate-800 space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-400">Completeness</span>
                        <span className="font-mono text-white">{quality.score_breakdown.components.completeness.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                        <div className="bg-emerald-500 h-full rounded-full" style={{ width: `${quality.score_breakdown.components.completeness}%` }} />
                      </div>
                      <p className="text-[11px] text-slate-400">Non-null records integrity</p>
                    </div>

                    <div className="glass-panel p-5 rounded-2xl border-slate-800 space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-400">Validity</span>
                        <span className="font-mono text-white">{quality.score_breakdown.components.validity.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                        <div className="bg-cyan-500 h-full rounded-full" style={{ width: `${quality.score_breakdown.components.validity}%` }} />
                      </div>
                      <p className="text-[11px] text-slate-400">Range & format adherence</p>
                    </div>

                    <div className="glass-panel p-5 rounded-2xl border-slate-800 space-y-2">
                      <div className="flex justify-between text-xs">
                        <span className="text-slate-400">Uniqueness</span>
                        <span className="font-mono text-white">{quality.score_breakdown.components.uniqueness.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                        <div className="bg-indigo-500 h-full rounded-full" style={{ width: `${quality.score_breakdown.components.uniqueness}%` }} />
                      </div>
                      <p className="text-[11px] text-slate-400">Zero row/key duplication</p>
                    </div>
                  </div>

                  {/* Issues and Recommendations Grid */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Issues Audit */}
                    <div className="glass-panel p-5 rounded-2xl border-slate-800 space-y-4">
                      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                        <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                          <AlertCircle className="w-4 h-4 text-amber-400" />
                          <span>Detected Quality Issues</span>
                        </h3>
                        <span className="text-[11px] font-mono text-slate-400">
                          {quality.issues.missing_values.length + quality.issues.outliers.length + (quality.issues.duplicates.count > 0 ? 1 : 0)} items
                        </span>
                      </div>

                      <div className="space-y-3">
                        {quality.issues.missing_values.length > 0 ? (
                          quality.issues.missing_values.map((m, idx) => (
                            <div key={idx} className="p-3 rounded-xl bg-slate-850 border border-slate-800 flex items-center justify-between">
                              <div>
                                <p className="text-xs font-semibold text-slate-200">{m.column}</p>
                                <p className="text-[11px] text-slate-400">{m.count} null entries ({m.percentage.toFixed(1)}%)</p>
                              </div>
                              <span className={`px-2 py-0.5 text-[10px] font-mono rounded ${
                                m.severity === 'CRITICAL' ? 'bg-red-500/20 text-red-400' : 'bg-amber-500/20 text-amber-400'
                              }`}>
                                {m.severity}
                              </span>
                            </div>
                          ))
                        ) : (
                          <div className="p-3 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-xs text-emerald-400 flex items-center space-x-2">
                            <CheckCircle2 className="w-4 h-4 shrink-0" />
                            <span>Zero missing values detected across dataset.</span>
                          </div>
                        )}

                        {quality.issues.outliers.map((o, idx) => (
                          <div key={idx} className="p-3 rounded-xl bg-slate-850 border border-slate-800 flex items-center justify-between">
                            <div>
                              <p className="text-xs font-semibold text-slate-200">{o.column}</p>
                              <p className="text-[11px] text-slate-400">{o.count} outliers outside [{o.lower_bound}, {o.upper_bound}]</p>
                            </div>
                            <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-purple-500/20 text-purple-400">
                              {o.percentage}% OUTLIERS
                            </span>
                          </div>
                        ))}

                        {quality.issues.duplicates.count > 0 && (
                          <div className="p-3 rounded-xl bg-slate-850 border border-slate-800 flex items-center justify-between">
                            <div>
                              <p className="text-xs font-semibold text-slate-200">Duplicate Records</p>
                              <p className="text-[11px] text-slate-400">{quality.issues.duplicates.count} duplicate rows ({quality.issues.duplicates.percentage}%)</p>
                            </div>
                            <span className="px-2 py-0.5 text-[10px] font-mono rounded bg-amber-500/20 text-amber-400">
                              DUPLICATE
                            </span>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Cleaning Recommendations */}
                    <div className="glass-panel p-5 rounded-2xl border-slate-800 space-y-4">
                      <div className="flex items-center justify-between border-b border-slate-800 pb-3">
                        <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                          <span>Cleaning Recommendations</span>
                        </h3>
                        <span className="text-[11px] font-mono text-slate-400">{quality.recommendations.length} actions</span>
                      </div>

                      <div className="space-y-3">
                        {quality.recommendations.map((rec) => (
                          <div key={rec.id} className="p-3.5 rounded-xl bg-slate-850 border border-slate-800 space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-bold text-white">{rec.title}</span>
                              <span className={`px-1.5 py-0.5 rounded text-[9px] font-mono font-bold ${
                                rec.priority === 'HIGH' ? 'bg-red-500/20 text-red-400' : 'bg-slate-700 text-slate-300'
                              }`}>
                                {rec.priority}
                              </span>
                            </div>
                            <p className="text-xs text-slate-400">{rec.description}</p>
                            {rec.affected_columns.length > 0 && (
                              <div className="flex flex-wrap gap-1 pt-1">
                                {rec.affected_columns.map(c => (
                                  <span key={c} className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-slate-800 text-slate-300">
                                    {c}
                                  </span>
                                ))}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 2: SCHEMA & PROFILER */}
              {activeSubTab === 'schema' && profile && (
                <div className="space-y-4">
                  {/* Summary Pills & Filter Bar */}
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 glass-panel p-4 rounded-2xl border-slate-800">
                    <div className="flex flex-wrap items-center gap-2">
                      {Object.entries(profile.schema.column_types_summary).map(([type, count]) => (
                        count > 0 && (
                          <span key={type} className={`px-2.5 py-1 rounded-xl text-xs font-mono font-semibold border ${getTypeBadgeColor(type)}`}>
                            {count} {type}
                          </span>
                        )
                      ))}
                    </div>

                    <div className="relative w-full md:w-64">
                      <Search className="w-3.5 h-3.5 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
                      <input
                        type="text"
                        value={columnSearch}
                        onChange={(e) => setColumnSearch(e.target.value)}
                        placeholder="Search column..."
                        className="w-full pl-9 pr-3 py-1.5 bg-slate-800/80 border border-slate-700 rounded-xl text-xs text-slate-200 placeholder-slate-500 focus:outline-none focus:border-nexus-accent"
                      />
                    </div>
                  </div>

                  {/* Column Profile Table */}
                  <div className="glass-panel rounded-2xl border-slate-800 overflow-hidden">
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-slate-850/80 text-slate-400 font-mono text-[11px] uppercase border-b border-slate-800">
                          <tr>
                            <th className="py-3 px-4">Column</th>
                            <th className="py-3 px-4">Type</th>
                            <th className="py-3 px-4">Missingness</th>
                            <th className="py-3 px-4">Unique Values</th>
                            <th className="py-3 px-4">Statistical Summary</th>
                            <th className="py-3 px-4">Distribution</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60 font-sans">
                          {Object.values(profile.columns)
                            .filter(col => col.name.toLowerCase().includes(columnSearch.toLowerCase()))
                            .map((col) => (
                              <tr key={col.name} className="hover:bg-slate-800/40 transition-colors">
                                <td className="py-3.5 px-4 font-semibold text-white font-mono">{col.name}</td>
                                <td className="py-3.5 px-4">
                                  <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold border ${getTypeBadgeColor(col.inferred_type)}`}>
                                    {col.inferred_type}
                                  </span>
                                </td>
                                <td className="py-3.5 px-4">
                                  <div className="space-y-1">
                                    <span className="font-mono text-slate-300">{col.null_percentage.toFixed(1)}%</span>
                                    <div className="w-20 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                                      <div 
                                        className={`h-full rounded-full ${col.null_percentage > 10 ? 'bg-red-500' : 'bg-emerald-500'}`}
                                        style={{ width: `${Math.min(100, col.null_percentage)}%` }} 
                                      />
                                    </div>
                                  </div>
                                </td>
                                <td className="py-3.5 px-4 font-mono text-slate-300">
                                  {col.unique_count} ({col.unique_percentage.toFixed(1)}%)
                                </td>
                                <td className="py-3.5 px-4">
                                  {col.inferred_type === 'NUMERICAL' && col.mean !== undefined ? (
                                    <div className="text-[11px] font-mono text-slate-300 space-y-0.5">
                                      <div>μ = {col.mean} • σ = {col.std}</div>
                                      <div className="text-slate-500">[{col.min} to {col.max}]</div>
                                    </div>
                                  ) : col.top_categories ? (
                                    <div className="text-[11px] text-slate-300 truncate max-w-[200px]">
                                      {col.top_categories.slice(0, 2).map(c => `${c.category} (${c.count})`).join(', ')}
                                    </div>
                                  ) : col.date_range_days !== undefined ? (
                                    <div className="text-[11px] font-mono text-slate-300">
                                      {col.date_range_days} days range
                                    </div>
                                  ) : (
                                    <span className="text-slate-500 text-[11px]">—</span>
                                  )}
                                </td>
                                <td className="py-3.5 px-4">
                                  {col.histogram && col.histogram.length > 0 ? (
                                    <div className="flex items-end space-x-1 h-6 w-24">
                                      {col.histogram.map((bin, i) => {
                                        const maxCount = Math.max(...col.histogram!.map(b => b.count), 1);
                                        const hPct = Math.max(10, (bin.count / maxCount) * 100);
                                        return (
                                          <div
                                            key={i}
                                            className="flex-1 bg-nexus-accent/60 hover:bg-nexus-accent rounded-t transition-colors"
                                            style={{ height: `${hPct}%` }}
                                            title={`Bin: ${bin.bin_start} - ${bin.bin_end}: ${bin.count}`}
                                          />
                                        );
                                      })}
                                    </div>
                                  ) : (
                                    <span className="text-slate-500 text-[11px]">—</span>
                                  )}
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 3: CORRELATIONS & EDA */}
              {activeSubTab === 'eda' && eda && (
                <div className="space-y-6">
                  {/* Significant Correlation Pairs */}
                  <div className="glass-panel p-5 rounded-2xl border-slate-800 space-y-4">
                    <h3 className="text-sm font-bold text-white flex items-center space-x-2">
                      <Flame className="w-4 h-4 text-nexus-accent" />
                      <span>Top Significant Feature Correlations</span>
                    </h3>
                    {eda.correlations.significant_correlations.length > 0 ? (
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        {eda.correlations.significant_correlations.map((pair, i) => (
                          <div key={i} className="p-3.5 rounded-xl bg-slate-850 border border-slate-800 space-y-2">
                            <div className="flex items-center justify-between">
                              <span className="text-xs font-semibold text-white font-mono">{pair.feature_a} ↔ {pair.feature_b}</span>
                              <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                                pair.pearson_r > 0 ? 'bg-indigo-500/20 text-indigo-400' : 'bg-cyan-500/20 text-cyan-400'
                              }`}>
                                r = {pair.pearson_r.toFixed(3)}
                              </span>
                            </div>
                            <div className="flex items-center space-x-2 text-[11px] text-slate-400">
                              <span className="capitalize">{pair.strength}</span>
                              <span>•</span>
                              <span className="capitalize">{pair.direction} association</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-xs text-slate-400">No highly correlated pairs (|r| &gt; 0.4) detected among numerical variables.</p>
                    )}
                  </div>

                  {/* Correlation Matrix Heatmap Table */}
                  {eda.correlations.numerical_columns.length > 0 && (
                    <div className="glass-panel p-5 rounded-2xl border-slate-800 space-y-3">
                      <div className="flex items-center justify-between">
                        <h3 className="text-sm font-bold text-white">Full Pearson Correlation Heatmap</h3>
                        <span className="text-[11px] font-mono text-slate-400">Matrix ({eda.correlations.numerical_columns.length} × {eda.correlations.numerical_columns.length})</span>
                      </div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-center text-[11px] font-mono border-collapse">
                          <thead>
                            <tr>
                              <th className="py-2 px-3 text-left text-slate-400 font-medium">Feature</th>
                              {eda.correlations.numerical_columns.map(col => (
                                <th key={col} className="py-2 px-3 text-slate-400 truncate max-w-[90px] font-medium" title={col}>
                                  {col}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-800/40">
                            {eda.correlations.numerical_columns.map(rowCol => (
                              <tr key={rowCol}>
                                <td className="py-2 px-3 text-left font-semibold text-slate-200 truncate max-w-[120px]" title={rowCol}>
                                  {rowCol}
                                </td>
                                {eda.correlations.numerical_columns.map(colCol => {
                                  const val = eda.correlations.matrix[rowCol]?.[colCol] ?? 0;
                                  const absVal = Math.abs(val);
                                  const isSelf = rowCol === colCol;
                                  let bgColor = 'bg-slate-800/40';
                                  if (!isSelf) {
                                    if (val > 0.7) bgColor = 'bg-indigo-600/70 text-white font-bold';
                                    else if (val > 0.4) bgColor = 'bg-indigo-500/40 text-indigo-200';
                                    else if (val > 0.1) bgColor = 'bg-indigo-900/30 text-indigo-300';
                                    else if (val < -0.7) bgColor = 'bg-cyan-600/70 text-white font-bold';
                                    else if (val < -0.4) bgColor = 'bg-cyan-500/40 text-cyan-200';
                                    else if (val < -0.1) bgColor = 'bg-cyan-900/30 text-cyan-300';
                                  }
                                  return (
                                    <td key={colCol} className={`py-2 px-3 rounded ${bgColor} transition-colors`}>
                                      {val.toFixed(2)}
                                    </td>
                                  );
                                })}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* TAB 4: MATHEMATICALLY VERIFIED INSIGHTS */}
              {activeSubTab === 'insights' && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 rounded-2xl glass-panel border-slate-800">
                    <div className="flex items-center space-x-2">
                      <ShieldCheck className="w-5 h-5 text-emerald-400" />
                      <div>
                        <h3 className="text-sm font-bold text-white">Deterministic Verification Layer</h3>
                        <p className="text-[11px] text-slate-400">All claims are checked against raw ground-truth metrics to prevent hallucinations.</p>
                      </div>
                    </div>
                    <span className="px-2.5 py-1 rounded-xl text-xs font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                      100% MATHEMATICALLY VERIFIED
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {insights.map((ins) => (
                      <div key={ins.id} className="glass-panel p-5 rounded-2xl border-slate-800 space-y-3">
                        <div className="flex items-center justify-between">
                          <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-slate-800 text-slate-300 border border-slate-700">
                            {ins.category}
                          </span>
                          <span className="flex items-center space-x-1 text-[11px] font-mono text-emerald-400">
                            <ShieldCheck className="w-3.5 h-3.5" />
                            <span>{(ins.confidence * 100).toFixed(0)}% Confident</span>
                          </span>
                        </div>
                        <h4 className="text-sm font-bold text-white">{ins.title}</h4>
                        <p className="text-xs text-slate-300 leading-relaxed">{ins.description}</p>
                        
                        {/* Metric Evidence Box */}
                        {ins.metric_evidence && Object.keys(ins.metric_evidence).length > 0 && (
                          <div className="p-2.5 rounded-xl bg-slate-850/80 border border-slate-800 text-[11px] font-mono text-slate-400 space-y-1">
                            <div className="text-[10px] uppercase tracking-wider text-slate-400 font-semibold">Audited Evidence</div>
                            <div className="flex flex-wrap gap-2 text-slate-300">
                              {Object.entries(ins.metric_evidence).map(([k, v]) => (
                                <span key={k} className="bg-slate-800 px-1.5 py-0.5 rounded border border-slate-700">
                                  {k}: <strong className="text-white">{String(v)}</strong>
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {ins.recommendation && (
                          <div className="text-xs text-nexus-accent flex items-center space-x-1.5 pt-1">
                            <ArrowRight className="w-3.5 h-3.5 shrink-0" />
                            <span>{ins.recommendation}</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* TAB 5: RAW DATA PREVIEW */}
              {activeSubTab === 'preview' && preview && (
                <div className="space-y-4">
                  {/* Table Container */}
                  <div className="glass-panel rounded-2xl border-slate-800 overflow-hidden">
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-xs">
                        <thead className="bg-slate-850 text-slate-400 font-mono text-[11px] border-b border-slate-800">
                          <tr>
                            <th className="py-3 px-4 text-slate-400">#</th>
                            {preview.columns.map((col) => (
                              <th key={col} className="py-3 px-4 whitespace-nowrap">
                                <div className="space-y-0.5">
                                  <div className="text-slate-200 font-semibold">{col}</div>
                                  <span className={`inline-block px-1.5 py-0.2 rounded text-[9px] font-mono uppercase border ${getTypeBadgeColor(preview.inferred_types[col] || 'UNKNOWN')}`}>
                                    {preview.inferred_types[col] || preview.dtypes[col]}
                                  </span>
                                </div>
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/60 font-mono text-[11px]">
                          {preview.rows.map((row, rIdx) => (
                            <tr key={rIdx} className="hover:bg-slate-800/30 transition-colors">
                              <td className="py-2.5 px-4 text-slate-400">{(preview.page - 1) * preview.page_size + rIdx + 1}</td>
                              {preview.columns.map((col) => {
                                const val = row[col];
                                const isNull = val === null || val === undefined;
                                return (
                                  <td key={col} className="py-2.5 px-4 whitespace-nowrap">
                                    {isNull ? (
                                      <span className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 text-[10px] italic">null</span>
                                    ) : (
                                      <span className="text-slate-300">{String(val)}</span>
                                    )}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Pagination Controls */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 px-2">
                    <div className="text-xs text-slate-400 font-mono">
                      Showing records {(preview.page - 1) * preview.page_size + 1} to {Math.min(preview.page * preview.page_size, preview.total_rows)} of {preview.total_rows}
                    </div>

                    <div className="flex items-center space-x-3">
                      <div className="flex items-center space-x-1.5">
                        <span className="text-xs text-slate-400 font-mono">Page size:</span>
                        <select
                          value={previewPageSize}
                          onChange={(e) => {
                            setPreviewPageSize(Number(e.target.value));
                            setPreviewPage(1);
                          }}
                          className="px-2 py-1 bg-slate-800 border border-slate-700 rounded-lg text-xs text-slate-200 focus:outline-none"
                        >
                          <option value={10}>10</option>
                          <option value={15}>15</option>
                          <option value={25}>25</option>
                          <option value={50}>50</option>
                        </select>
                      </div>

                      <div className="flex items-center space-x-1">
                        <button
                          disabled={preview.page <= 1}
                          onClick={() => setPreviewPage(p => Math.max(1, p - 1))}
                          className="p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 disabled:opacity-40"
                        >
                          <ChevronLeft className="w-4 h-4" />
                        </button>
                        <span className="px-3 py-1 font-mono text-xs text-slate-300">
                          {preview.page} / {preview.total_pages}
                        </span>
                        <button
                          disabled={preview.page >= preview.total_pages}
                          onClick={() => setPreviewPage(p => Math.min(preview.total_pages, p + 1))}
                          className="p-1.5 rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 disabled:opacity-40"
                        >
                          <ChevronRight className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Modals */}
      <SampleDatasetModal
        isOpen={isSampleModalOpen}
        onClose={() => setIsSampleModalOpen(false)}
        projects={projects}
        onDatasetLoaded={loadInitialData}
        onProjectCreated={(newProj) => setProjects(prev => [newProj, ...prev])}
      />

      <UploadDatasetModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        projects={projects}
        onDatasetUploaded={loadInitialData}
        onProjectCreated={(newProj) => setProjects(prev => [newProj, ...prev])}
      />
    </div>
  );
};
