import React, { useState, useRef } from 'react';
import { Project } from '../../types';
import { api } from '../../services/api';
import { Upload, X, FileSpreadsheet, Plus, CheckCircle2, AlertCircle } from 'lucide-react';

interface UploadDatasetModalProps {
  isOpen: boolean;
  onClose: () => void;
  projects: Project[];
  onDatasetUploaded: () => void;
  onProjectCreated: (newProject: Project) => void;
}

export const UploadDatasetModal: React.FC<UploadDatasetModalProps> = ({
  isOpen,
  onClose,
  projects,
  onDatasetUploaded,
  onProjectCreated,
}) => {
  const [selectedProjectId, setSelectedProjectId] = useState<string>(projects[0]?.id || '');
  const [newProjectName, setNewProjectName] = useState<string>('');
  const [isCreatingProject, setIsCreatingProject] = useState<boolean>(projects.length === 0);
  const [file, setFile] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState<string>('');
  const [description, setDescription] = useState<string>('');
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  if (!isOpen) return null;

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selected = e.target.files[0];
      setFile(selected);
      if (!datasetName) {
        setDatasetName(selected.name.replace(/\.[^/.]+$/, ""));
      }
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const dropped = e.dataTransfer.files[0];
      setFile(dropped);
      if (!datasetName) {
        setDatasetName(dropped.name.replace(/\.[^/.]+$/, ""));
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file to upload.');
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      let targetProjectId = selectedProjectId;
      if (isCreatingProject) {
        if (!newProjectName.trim()) {
          setError('Please specify a project name.');
          setIsUploading(false);
          return;
        }
        const createdProject = await api.createProject({
          name: newProjectName.trim(),
          description: 'Created during dataset upload'
        });
        onProjectCreated(createdProject);
        targetProjectId = createdProject.id;
      }

      await api.uploadDataset(targetProjectId, file, datasetName.trim() || undefined, description.trim() || undefined);
      onDatasetUploaded();
      onClose();
    } catch (err: any) {
      setError(err?.response?.data?.message || err?.message || 'Failed to upload dataset.');
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fadeIn">
      <div className="bg-nexus-900 border border-slate-800 rounded-2xl w-full max-w-lg overflow-hidden shadow-2xl space-y-6 p-6">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div className="flex items-center space-x-2">
            <div className="p-2 rounded-xl bg-nexus-accent/20 text-nexus-accent border border-nexus-accent/30">
              <Upload className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Upload Dataset</h2>
              <p className="text-xs text-slate-400">CSV, Excel (.xlsx, .xls), JSON, or Parquet</p>
            </div>
          </div>
          <button onClick={onClose} className="p-1 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800 transition-colors">
            <X className="w-5 h-5" />
          </button>
        </div>

        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-xl text-xs text-red-400 flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Project Target */}
        <div className="space-y-2">
          <label className="text-xs font-mono text-slate-400 uppercase tracking-wider">Target Project</label>
          {projects.length > 0 && !isCreatingProject ? (
            <div className="flex space-x-2">
              <select
                value={selectedProjectId}
                onChange={(e) => setSelectedProjectId(e.target.value)}
                className="flex-1 px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-nexus-accent"
              >
                {projects.map((p) => (
                  <option key={p.id} value={p.id}>{p.name}</option>
                ))}
              </select>
              <button
                type="button"
                onClick={() => setIsCreatingProject(true)}
                className="px-3 py-2 bg-slate-800 hover:bg-slate-700 text-xs font-mono text-slate-300 rounded-xl border border-slate-700 flex items-center space-x-1"
              >
                <Plus className="w-3.5 h-3.5" />
                <span>New</span>
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              <input
                type="text"
                value={newProjectName}
                onChange={(e) => setNewProjectName(e.target.value)}
                placeholder="Enter new workspace/project name..."
                className="w-full px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-nexus-accent"
              />
              {projects.length > 0 && (
                <button
                  type="button"
                  onClick={() => setIsCreatingProject(false)}
                  className="text-[11px] text-slate-400 hover:text-nexus-accent underline"
                >
                  Or select existing project
                </button>
              )}
            </div>
          )}
        </div>

        {/* Dropzone */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-slate-700 hover:border-nexus-accent/60 rounded-xl p-6 text-center cursor-pointer transition-colors bg-slate-900/40"
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".csv,.xlsx,.xls,.json,.parquet"
            className="hidden"
          />
          {file ? (
            <div className="flex items-center justify-center space-x-3 text-slate-200">
              <FileSpreadsheet className="w-8 h-8 text-nexus-accent" />
              <div className="text-left">
                <p className="text-xs font-semibold">{file.name}</p>
                <p className="text-[11px] font-mono text-slate-400">{(file.size / 1024).toFixed(1)} KB</p>
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <Upload className="w-8 h-8 text-slate-500 mx-auto" />
              <p className="text-xs text-slate-300">Click or drag file here to upload</p>
              <p className="text-[11px] font-mono text-slate-400">Supported: .csv, .xlsx, .xls, .json, .parquet</p>
            </div>
          )}
        </div>

        {/* Optional Metadata */}
        <div className="grid grid-cols-1 gap-3">
          <div>
            <label className="text-xs font-mono text-slate-400 uppercase tracking-wider">Dataset Title (Optional)</label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="e.g. Q3 Customer Behavioral Signals"
              className="w-full mt-1 px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-nexus-accent"
            />
          </div>
          <div>
            <label className="text-xs font-mono text-slate-400 uppercase tracking-wider">Description (Optional)</label>
            <input
              type="text"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Brief context about this dataset..."
              className="w-full mt-1 px-3 py-2 bg-slate-800/80 border border-slate-700 rounded-xl text-xs text-slate-200 focus:outline-none focus:border-nexus-accent"
            />
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex items-center justify-end space-x-3 pt-3 border-t border-slate-800">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-xs font-medium text-slate-400 hover:text-white rounded-xl hover:bg-slate-800 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={!file || isUploading}
            onClick={handleUpload}
            className="px-5 py-2 bg-gradient-to-r from-nexus-accent to-nexus-purple text-white text-xs font-semibold rounded-xl hover:opacity-95 transition-opacity disabled:opacity-50 flex items-center space-x-2"
          >
            {isUploading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Processing Pipeline...</span>
              </>
            ) : (
              <>
                <CheckCircle2 className="w-4 h-4" />
                <span>Upload & Profile</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
};
