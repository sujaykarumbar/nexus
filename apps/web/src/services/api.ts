import axios from 'axios';
import { 
  User, 
  AuthTokens, 
  Project, 
  Dataset, 
  AnalysisJob, 
  AgentInfo, 
  SystemHealth,
  DatasetProfile,
  DataQualityReportData,
  EDAResponseData,
  InsightItem,
  VisualizationSpec,
  PaginatedPreviewData,
  TargetSuggestionsResponse,
  LeakageAuditResponse,
  MLTrainRequest,
  MLModelSummary,
  MLModelDetail,
  PredictResponse,
  SwarmRunRequest,
  SwarmRunResponse,
  CriticVerificationResult,
  RAGDocument,
  RAGQueryResponse,
  HybridSearchResponse,
  KnowledgeGraphData,
  GraphBuildResult,
  GraphRAGResult
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Auto-attach JWT token if available
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('nexus_token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export const api = {
  // Authentication
  register: async (data: { email: string; password: string; full_name: string }) => {
    const res = await apiClient.post<User>('/auth/register', data);
    return res.data;
  },
  
  login: async (data: { email: string; password: string }) => {
    const res = await apiClient.post<AuthTokens>('/auth/login', data);
    return res.data;
  },

  getMe: async () => {
    const res = await apiClient.get<User>('/auth/me');
    return res.data;
  },

  // Health
  getHealth: async () => {
    const res = await apiClient.get<SystemHealth>('/health');
    return res.data;
  },

  // Projects
  listProjects: async () => {
    const res = await apiClient.get<Project[]>('/projects');
    return res.data;
  },

  getProjects: async () => {
    const res = await apiClient.get<Project[]>('/projects');
    return res.data;
  },

  createProject: async (data: { name: string; description?: string }) => {
    const res = await apiClient.post<Project>('/projects', data);
    return res.data;
  },

  // Datasets
  listDatasets: async (projectId?: string) => {
    const params = projectId ? { project_id: projectId } : {};
    const res = await apiClient.get<Dataset[]>('/datasets', { params });
    return res.data;
  },

  getDatasets: async (projectId?: string) => {
    const params = projectId ? { project_id: projectId } : {};
    const res = await apiClient.get<Dataset[]>('/datasets', { params });
    return res.data;
  },

  getDatasetPreview: async (datasetId: string, page = 1, pageSize = 20) => {
    const res = await apiClient.get<PaginatedPreviewData>(`/datasets/${datasetId}/preview`, { 
      params: { page, page_size: pageSize } 
    });
    return res.data;
  },

  getDataset: async (datasetId: string) => {
    const res = await apiClient.get<Dataset>(`/datasets/${datasetId}`);
    return res.data;
  },

  loadSampleDataset: async (projectId: string, sampleType: 'churn' | 'sales' | 'sensor') => {
    const formData = new FormData();
    formData.append('project_id', projectId);
    formData.append('sample_type', sampleType);
    const res = await apiClient.post<Dataset>('/datasets/sample', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return res.data;
  },

  uploadDataset: async (projectId: string, file: File, name?: string, description?: string) => {
    const formData = new FormData();
    formData.append('project_id', projectId);
    formData.append('file', file);
    if (name) formData.append('name', name);
    if (description) formData.append('description', description);
    const res = await apiClient.post<Dataset>('/datasets/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return res.data;
  },

  getDatasetProfile: async (datasetId: string) => {
    const res = await apiClient.get<DatasetProfile>(`/datasets/${datasetId}/profile`);
    return res.data;
  },

  getDatasetQuality: async (datasetId: string) => {
    const res = await apiClient.get<DataQualityReportData>(`/datasets/${datasetId}/quality`);
    return res.data;
  },

  getDatasetEDA: async (datasetId: string) => {
    const res = await apiClient.get<EDAResponseData>(`/datasets/${datasetId}/eda`);
    return res.data;
  },

  getDatasetInsights: async (datasetId: string) => {
    const res = await apiClient.get<InsightItem[]>(`/datasets/${datasetId}/insights`);
    return res.data;
  },

  getDatasetVisualizations: async (datasetId: string) => {
    const res = await apiClient.get<VisualizationSpec[]>(`/datasets/${datasetId}/visualizations`);
    return res.data;
  },

  previewDatasetPaginated: async (datasetId: string, page = 1, pageSize = 20) => {
    const res = await apiClient.get<PaginatedPreviewData>(`/datasets/${datasetId}/preview`, { 
      params: { page, page_size: pageSize } 
    });
    return res.data;
  },

  triggerAnalysis: async (datasetId: string) => {
    const res = await apiClient.post<{ job_id: string; status: string; message: string }>(`/datasets/${datasetId}/analyze`);
    return res.data;
  },

  deleteDataset: async (datasetId: string) => {
    await apiClient.delete(`/datasets/${datasetId}`);
  },

  previewDataset: async (datasetId: string, limit = 20) => {
    const res = await apiClient.get(`/datasets/${datasetId}/preview`, { params: { limit } });
    return res.data;
  },

  // Jobs
  listJobs: async (projectId?: string) => {
    const params = projectId ? { project_id: projectId } : {};
    const res = await apiClient.get<AnalysisJob[]>('/jobs', { params });
    return res.data;
  },

  createJob: async (data: { project_id: string; dataset_id?: string; job_type: string }) => {
    const res = await apiClient.post<AnalysisJob>('/jobs', data);
    return res.data;
  },

  getJob: async (jobId: string) => {
    const res = await apiClient.get<AnalysisJob>(`/jobs/${jobId}`);
    return res.data;
  },

  // Agents
  getAgents: async () => {
    const res = await apiClient.get<AgentInfo[]>('/agents');
    return res.data;
  },

  // Phase 3: Autonomous Machine Learning & Model Registry
  getTargetSuggestions: async (datasetId: string) => {
    const res = await apiClient.get<TargetSuggestionsResponse>(`/ml/datasets/${datasetId}/targets`);
    return res.data;
  },

  auditDataLeakage: async (datasetId: string, targetColumn: string) => {
    const res = await apiClient.get<LeakageAuditResponse>(`/ml/datasets/${datasetId}/leakage`, {
      params: { target_column: targetColumn }
    });
    return res.data;
  },

  trainAutoML: async (data: MLTrainRequest) => {
    const res = await apiClient.post<{ job_id: string; status: string; message: string }>('/ml/train', data);
    return res.data;
  },

  listMLModels: async (projectId?: string, datasetId?: string) => {
    const params: Record<string, string> = {};
    if (projectId) params.project_id = projectId;
    if (datasetId) params.dataset_id = datasetId;
    const res = await apiClient.get<MLModelSummary[]>('/ml/models', { params });
    return res.data;
  },

  getMLModel: async (modelId: string) => {
    const res = await apiClient.get<MLModelDetail>(`/ml/models/${modelId}`);
    return res.data;
  },

  predictSingle: async (modelId: string, features: Record<string, any>) => {
    const res = await apiClient.post<PredictResponse>(`/ml/models/${modelId}/predict`, { features });
    return res.data;
  },

  promoteModel: async (modelId: string, lifecycleStage: string) => {
    const res = await apiClient.post<MLModelSummary>(`/ml/models/${modelId}/promote`, { lifecycle_stage: lifecycleStage });
    return res.data;
  },

  // Knowledge Graph & GraphRAG (also available on agentsApi for backward compat)
  buildKnowledgeGraph: async (req: {
    project_id: string;
    dataset_id?: string;
    target_column?: string;
    correlation_threshold?: number;
    include_models?: boolean;
    include_anomalies?: boolean;
    include_forecasts?: boolean;
  }) => {
    const res = await apiClient.post<GraphBuildResult>('/graph/build', req);
    return res.data;
  },

  getKnowledgeGraph: async (projectId: string) => {
    const res = await apiClient.get<KnowledgeGraphData>(`/graph/projects/${projectId}`);
    return res.data;
  },

  queryGraphRAG: async (query: string, projectId?: string, maxHops: number = 2) => {
    const res = await apiClient.post<GraphRAGResult>('/graph/query', {
      query,
      project_id: projectId,
      max_hops: maxHops
    });
    return res.data;
  },
};

export const forecastApi = {
  detectTimeSeries: async (datasetId: string, timeColumn?: string, targetColumn?: string) => {
    const res = await apiClient.post<any>('/forecast/detect', {
      dataset_id: datasetId,
      time_column: timeColumn,
      target_column: targetColumn,
    });
    return res.data;
  },

  startForecastTraining: async (payload: any) => {
    const res = await apiClient.post<{ message: string; job_id: string; status: string }>('/forecast/train', payload);
    return res.data;
  },

  getForecastJobStatus: async (jobId: string) => {
    const res = await apiClient.get<{
      job_id: string;
      status: string;
      progress: number;
      stage: string;
      error_message?: string;
      result_summary?: any;
    }>(`/forecast/jobs/${jobId}`);
    return res.data;
  },

  listForecastModels: async (projectId?: string) => {
    const params: Record<string, string> = {};
    if (projectId) params.project_id = projectId;
    const res = await apiClient.get<any[]>('/forecast/models', { params });
    return res.data;
  },

  getForecastModel: async (modelId: string) => {
    const res = await apiClient.get<any>(`/forecast/models/${modelId}`);
    return res.data;
  },

  generateDynamicForecast: async (modelId: string, horizon: number) => {
    const res = await apiClient.post<any>(`/forecast/models/${modelId}/forecast`, { horizon });
    return res.data;
  },

  promoteForecastModel: async (modelId: string, lifecycleStage: string) => {
    const res = await apiClient.post<{ model_id: string; lifecycle_stage: string }>(`/forecast/models/${modelId}/promote`, {
      lifecycle_stage: lifecycleStage,
    });
    return res.data;
  },
};

export const anomalyApi = {
  startAnomalyDetection: async (payload: {
    project_id: string;
    dataset_id: string;
    time_column?: string;
    target_column?: string;
    feature_columns?: string[];
    include_deep_learning?: boolean;
  }) => {
    const res = await apiClient.post<{ message: string; job_id: string; status: string }>('/anomalies/detect', payload);
    return res.data;
  },

  getAnomalyJobStatus: async (jobId: string) => {
    const res = await apiClient.get<{
      job_id: string;
      status: string;
      progress: number;
      stage: string;
      error_message?: string;
      result_summary?: any;
    }>(`/anomalies/jobs/${jobId}`);
    return res.data;
  },

  getLatestAnomalyReport: async (datasetId: string) => {
    const res = await apiClient.get<any>(`/anomalies/${datasetId}`);
    return res.data;
  },

  getAnomalySummary: async (datasetId: string) => {
    const res = await apiClient.get<any>(`/anomalies/${datasetId}/summary`);
    return res.data;
  },

  getAnomalyTimeline: async (datasetId: string) => {
    const res = await apiClient.get<{ timeline: any[]; analyzed_metric: string }>(`/anomalies/${datasetId}/timeline`);
    return res.data;
  },

  getAnomalyChangePoints: async (datasetId: string) => {
    const res = await apiClient.get<any[]>(`/anomalies/${datasetId}/change-points`);
    return res.data;
  },
};

export const agentsApi = {
  getSwarmStatus: async () => {
    const res = await apiClient.get<AgentInfo[]>('/agents');
    return res.data;
  },

  runSwarm: async (payload: SwarmRunRequest) => {
    const res = await apiClient.post<SwarmRunResponse>('/agents/swarm/run', payload);
    return res.data;
  },

  getSwarmRunTrace: async (jobId: string) => {
    const res = await apiClient.get<SwarmRunResponse>(`/agents/swarm/runs/${jobId}`);
    return res.data;
  },

  verifyCriticClaims: async (payload: { claims: any[]; computed_metrics: Record<string, any> }) => {
    const res = await apiClient.post<CriticVerificationResult>('/agents/critic/verify', payload);
    return res.data;
  },

  // Knowledge Graph & GraphRAG
  buildKnowledgeGraph: async (req: {
    project_id: string;
    dataset_id?: string;
    target_column?: string;
    correlation_threshold?: number;
    include_models?: boolean;
    include_anomalies?: boolean;
    include_forecasts?: boolean;
  }) => {
    const res = await apiClient.post<GraphBuildResult>('/graph/build', req);
    return res.data;
  },

  getKnowledgeGraph: async (projectId: string) => {
    const res = await apiClient.get<KnowledgeGraphData>(`/graph/projects/${projectId}`);
    return res.data;
  },

  queryGraphRAG: async (query: string, projectId?: string, maxHops: number = 2) => {
    const res = await apiClient.post<GraphRAGResult>('/graph/query', {
      query,
      project_id: projectId,
      max_hops: maxHops
    });
    return res.data;
  },

  getNodeNeighborhood: async (nodeId: string, maxHops: number = 2) => {
    const res = await apiClient.get(`/graph/node/${nodeId}/neighborhood`, {
      params: { max_hops: maxHops }
    });
    return res.data;
  }
};

export const ragApi = {
  uploadDocument: async (file: File, projectId?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    if (projectId) formData.append('project_id', projectId);
    const res = await apiClient.post<RAGDocument>('/rag/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return res.data;
  },

  listDocuments: async (projectId?: string) => {
    const params: Record<string, string> = {};
    if (projectId) params.project_id = projectId;
    const res = await apiClient.get<RAGDocument[]>('/rag/documents', { params });
    return res.data;
  },

  deleteDocument: async (documentId: string) => {
    const res = await apiClient.delete<{ status: string; document_id: string }>(`/rag/documents/${documentId}`);
    return res.data;
  },

  hybridSearch: async (query: string, projectId?: string, topK: number = 5) => {
    const res = await apiClient.post<HybridSearchResponse>('/rag/search', {
      query,
      project_id: projectId,
      top_k: topK
    });
    return res.data;
  },

  queryRAG: async (query: string, projectId?: string, topK: number = 4) => {
    const res = await apiClient.post<RAGQueryResponse>('/rag/query', {
      query,
      project_id: projectId,
      top_k: topK
    });
    return res.data;
  }
};

export const streamingApi = {
  startSession: async (payload: {
    dataset_id: string;
    target_columns?: string[];
    rows_per_second?: number;
    window_size?: number;
    z_threshold?: number;
    spike_probability?: number;
    spike_factor?: number;
  }) => {
    const res = await apiClient.post<any>('/stream/start', payload);
    return res.data;
  },

  listSessions: async () => {
    const res = await apiClient.get<any[]>('/stream/sessions');
    return res.data;
  },

  getSession: async (sessionId: string) => {
    const res = await apiClient.get<any>(`/stream/sessions/${sessionId}`);
    return res.data;
  },

  stopSession: async (sessionId: string) => {
    const res = await apiClient.post<any>(`/stream/sessions/${sessionId}/stop`);
    return res.data;
  },

  getAlerts: async (sessionId: string, limit = 100, severity?: string) => {
    const params: Record<string, any> = { limit };
    if (severity) params.severity = severity;
    const res = await apiClient.get<any>(`/stream/sessions/${sessionId}/alerts`, { params });
    return res.data;
  },

  getWebSocketUrl: (sessionId: string): string => {
    const base = (import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1')
      .replace(/^http/, 'ws');
    return `${base}/stream/sessions/${sessionId}/live`;
  },
};

export const mlopsApi = {
  listVersions: async (modelId?: string) => {
    const params: Record<string, any> = {};
    if (modelId) params.model_id = modelId;
    const res = await apiClient.get<any[]>('/mlops/registry', { params });
    return res.data;
  },

  getVersion: async (versionId: string) => {
    const res = await apiClient.get<any>(`/mlops/registry/${versionId}`);
    return res.data;
  },

  registerVersion: async (modelId: string, payload: any) => {
    const res = await apiClient.post<any>(`/mlops/registry/${modelId}/versions`, payload);
    return res.data;
  },

  compareVersions: async (versionId: string, compareToId: string) => {
    const res = await apiClient.post<any>(`/mlops/registry/${versionId}/compare`, { compare_to: compareToId });
    return res.data;
  },

  createPipeline: async (payload: any) => {
    const res = await apiClient.post<any>('/mlops/pipelines', payload);
    return res.data;
  },

  listPipelines: async () => {
    const res = await apiClient.get<any[]>('/mlops/pipelines');
    return res.data;
  },

  deletePipeline: async (pipelineId: string) => {
    await apiClient.delete(`/mlops/pipelines/${pipelineId}`);
  },

  computeDrift: async (referenceDatasetId: string, currentDatasetId: string) => {
    const res = await apiClient.post<any>('/mlops/drift', {
      reference_dataset_id: referenceDatasetId,
      current_dataset_id: currentDatasetId,
    });
    return res.data;
  },
};

export const observabilityApi = {
  getDeepHealth: async () => {
    const res = await apiClient.get<any>('/health/deep');
    return res.data;
  },

  getMetrics: async () => {
    const res = await apiClient.get<any>('/observability/metrics');
    return res.data;
  },

  getRequestTrace: async (limit = 50) => {
    const res = await apiClient.get<any>('/observability/traces', { params: { limit } });
    return res.data;
  },
};
