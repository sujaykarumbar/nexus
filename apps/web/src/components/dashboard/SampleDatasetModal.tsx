import React, { useState } from 'react';
import { Project } from '../../types';
import { api } from '../../services/api';
import { Database, Plus, Sparkles, X, Check, ArrowRight } from 'lucide-react';

interface SampleDatasetModalProps {
  isOpen: boolean;
  onClose: () => void;
  projects: Project[];
  onDatasetLoaded: () => void;
  onProjectCreated: (newProject: Project) => void;
}

export const SampleDatasetModal: React.FC<SampleDatasetModalProps> = ({
  isOpen,
  onClose,
  projects,
  onDatasetLoaded,
  onProjectCreated,
}) => {
  const [selectedProjectId, setSelectedProjectId] = useState<string>(projects[0]?.id || '');
  const [newProjectName, setNewProjectName] = useState<string>('');
  const [isCreatingProject, setIsCreatingProject] = useState<boolean>(projects.length === 0);
  const [selectedSample, setSelectedSample] = useState<'churn' | 'sales' | 'sensor'>('churn');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const sampleDatasets = [
    {
      id: 'churn',
      title: 'Customer Churn Intelligence',
      badge: 'Classification',
      description: 'Telecom customer subscription records, support tickets, monthly charges, and churn flags.',
      rows: '20 Rows • 10 Features',
      color: '#3B82F6'
    },
    {
      id: 'sales',
      title: 'Store Sales & Demand Forecasting',
      badge: 'Time-Series',
      description: 'Historical daily retail revenue, customer foot traffic, promotion flags, and weather metrics.',
      rows: '20 Dates • 6 Features',
      color: '#10B981'
    },
    {
      id: 'sensor',
      title: 'Industrial IoT Vibration & Thermal',
      badge: 'Anomaly Detection',
      description: 'Multi-sensor high-frequency vibration (Hz), temperature (°C), pressure (kPa), and voltage.',
      rows: '15 Records • 7 Features',
      color: '#EF4444'
    }
  ];

  const handleLoad = async () => {
    setIsLoading(true);
    setError(null);
    try {
      let targetProjectId = selectedProjectId;

      if (isCreatingProject || !targetProjectId) {
        if (!newProjectName.trim()) {
          setError('Please provide a name for the new project.');
          setIsLoading(false);
          return;
        }
        const created = await api.createProject({
          name: newProjectName.trim(),
          description: 'Autonomous analytics project initialized with benchmark dataset.'
        });
        targetProjectId = created.id;
        onProjectCreated(created);
      }

      await api.loadSampleDataset(targetProjectId, selectedSample);
      onDatasetLoaded();
      onClose();
    } catch (err: any) {
      console.error('Failed to load sample dataset:', err);
      setError(err?.response?.data?.error?.message || err?.message || 'Failed to load dataset.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-nexus-900/80 backdrop-blur-md animate-in fade-in duration-200">
      <div className="w-full max-w-xl glass-panel p-6 rounded-2xl border-slate-700 shadow-2xl space-y-5">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-2.5">
            <div className="p-2 rounded-xl bg-nexus-accent/10 text-nexus-accent border border-nexus-accent/20">
              <Database className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-white">Ingest Benchmark Dataset</h3>
              <p className="text-xs text-slate-400">Select a real production dataset to test the platform</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-mono">
            {error}
          </div>
        )}

        {/* Dataset Selector */}
        <div className="space-y-2">
          <label className="text-xs font-semibold text-slate-300">Choose Benchmark Dataset</label>
          <div className="grid grid-cols-1 gap-2.5">
            {sampleDatasets.map((sample) => (
              <div
                key={sample.id}
                onClick={() => setSelectedSample(sample.id as any)}
                className={`p-3.5 rounded-xl border cursor-pointer transition-all ${
                  selectedSample === sample.id
                    ? 'bg-slate-800/80 border-nexus-accent shadow-glow'
                    : 'bg-slate-900/60 border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: sample.color }} />
                    <span className="text-xs font-bold text-white">{sample.title}</span>
                  </div>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                    {sample.badge}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 mt-1.5">{sample.description}</p>
                <div className="text-[10px] font-mono text-slate-400 mt-2">{sample.rows}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Project Target */}
        <div className="space-y-2 pt-2 border-t border-slate-800">
          <div className="flex items-center justify-between">
            <label className="text-xs font-semibold text-slate-300">Target Workspace Project</label>
            {projects.length > 0 && (
              <button
                type="button"
                onClick={() => setIsCreatingProject(!isCreatingProject)}
                className="text-[11px] text-nexus-accent hover:underline flex items-center space-x-1"
              >
                {isCreatingProject ? 'Select Existing Project' : '+ Create New Project'}
              </button>
            )}
          </div>

          {isCreatingProject || projects.length === 0 ? (
            <input
              type="text"
              placeholder="e.g. Telecom Churn Analysis 2026"
              value={newProjectName}
              onChange={(e) => setNewProjectName(e.target.value)}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white focus:outline-none focus:border-nexus-accent"
            />
          ) : (
            <select
              value={selectedProjectId}
              onChange={(e) => setSelectedProjectId(e.target.value)}
              className="w-full px-3.5 py-2.5 rounded-xl bg-slate-950 border border-slate-700 text-xs text-white focus:outline-none focus:border-nexus-accent"
            >
              {projects.map((proj) => (
                <option key={proj.id} value={proj.id}>
                  {proj.name} ({new Date(proj.created_at).toLocaleDateString()})
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-end space-x-3 pt-4 border-t border-slate-800">
          <button
            onClick={onClose}
            disabled={isLoading}
            className="px-4 py-2 rounded-xl text-xs text-slate-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleLoad}
            disabled={isLoading}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-nexus-accent to-nexus-purple text-nexus-900 font-bold text-xs shadow-glow hover:opacity-95 transition-opacity flex items-center space-x-2 disabled:opacity-50"
          >
            {isLoading ? (
              <span>Ingesting & Profiling...</span>
            ) : (
              <>
                <span>Ingest Dataset</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
