import React from 'react';
import { AnalysisJob } from '../../types';
import { Activity, CheckCircle, Clock, PlayCircle, RefreshCw, XCircle } from 'lucide-react';

interface JobQueueCardProps {
  jobs: AnalysisJob[];
  onRefresh: () => void;
}

export const JobQueueCard: React.FC<JobQueueCardProps> = ({ jobs, onRefresh }) => {
  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return (
          <span className="flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            <CheckCircle className="w-3 h-3" />
            <span>COMPLETED</span>
          </span>
        );
      case 'running':
        return (
          <span className="flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-mono bg-nexus-accent/10 text-nexus-accent border border-nexus-accent/30 animate-pulse">
            <PlayCircle className="w-3 h-3" />
            <span>RUNNING</span>
          </span>
        );
      case 'failed':
        return (
          <span className="flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-mono bg-rose-500/10 text-rose-400 border border-rose-500/30">
            <XCircle className="w-3 h-3" />
            <span>FAILED</span>
          </span>
        );
      default:
        return (
          <span className="flex items-center space-x-1 px-2 py-0.5 rounded text-[10px] font-mono bg-amber-500/10 text-amber-400 border border-amber-500/30">
            <Clock className="w-3 h-3" />
            <span>QUEUED</span>
          </span>
        );
    }
  };

  return (
    <div className="glass-panel p-6 rounded-2xl border-slate-800 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">Asynchronous Execution Queue</h3>
            <p className="text-xs text-slate-400">Background tasks, stage progression & telemetry</p>
          </div>
        </div>
        <button
          onClick={onRefresh}
          className="p-1.5 rounded-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-300 transition-colors"
          title="Refresh Queue"
        >
          <RefreshCw className="w-3.5 h-3.5" />
        </button>
      </div>

      {jobs.length === 0 ? (
        <div className="p-8 text-center rounded-xl bg-slate-900/40 border border-slate-800/80">
          <p className="text-xs text-slate-400">No active or historical background jobs found in workspace.</p>
          <p className="text-[11px] text-slate-400 mt-1">Trigger an analysis from the Dataset Explorer to start the pipeline.</p>
        </div>
      ) : (
        <div className="space-y-2.5">
          {jobs.slice(0, 5).map((job) => (
            <div
              key={job.id}
              className="p-3.5 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2 hover:border-slate-700 transition-all"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <span className="text-xs font-semibold text-slate-200 capitalize">
                    {job.job_type.replace('_', ' ')} Pipeline
                  </span>
                  <span className="text-[10px] font-mono text-slate-400">
                    ID: {job.id.slice(0, 8)}...
                  </span>
                </div>
                {getStatusBadge(job.status)}
              </div>

              {/* Progress bar */}
              <div className="space-y-1">
                <div className="flex justify-between text-[10px] font-mono text-slate-400">
                  <span>{job.current_stage || 'Processing...'}</span>
                  <span className="text-slate-200 font-semibold">{Math.round(job.progress_percentage)}%</span>
                </div>
                <div className="w-full bg-slate-800 h-1.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      job.status === 'completed'
                        ? 'bg-emerald-400'
                        : job.status === 'failed'
                        ? 'bg-rose-500'
                        : 'bg-nexus-accent'
                    }`}
                    style={{ width: `${job.progress_percentage}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
