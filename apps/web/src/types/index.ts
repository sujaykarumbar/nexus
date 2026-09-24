export interface User {
  id: string;
  email: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
}

export interface AuthTokens {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface Project {
  id: string;
  name: string;
  description: string | null;
  owner_id: string;
  status: string;
  created_at: string;
  updated_at: string;
  is_active: boolean;
}

export interface Dataset {
  id: string;
  project_id: string;
  name: string;
  description: string | null;
  file_path: string;
  file_type: string;
  file_size_bytes: number;
  row_count: number;
  column_count: number;
  data_quality_score: number | null;
  detected_problem_type: string | null;
  target_column: string | null;
  schema_metadata?: {
    columns: string[];
    dtypes: Record<string, string>;
    null_counts: Record<string, number>;
    unique_counts: Record<string, number>;
  } | null;
  created_at: string;
}

export interface AnalysisJob {
  id: string;
  project_id: string;
  dataset_id?: string | null;
  creator_id: string;
  job_type: string;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled';
  progress_percentage: number;
  current_stage: string | null;
  logs: Array<{
    timestamp: string;
    level: string;
    message: string;
  }>;
  results?: Record<string, any> | null;
  error_message?: string | null;
  created_at: string;
}

export interface AgentInfo {
  id: string;
  name: string;
  role: string;
  status: 'idle' | 'active' | 'evaluating' | 'verifying';
  tools: string[];
  color: string;
}

export interface SystemHealth {
  status: string;
  version: string;
  environment: string;
  timestamp: string;
  services: {
    database: {
      status: string;
      latency_ms: number;
      engine: string;
    };
    api: {
      status: string;
      uptime_seconds: number;
      platform: string;
      python_version: string;
    };
    job_worker: {
      status: string;
      mode: string;
    };
  };
  system_metrics: {
    cpu_percent: number;
    memory_used_mb: number;
    memory_total_mb: number;
    memory_percent: number;
  };
}

export interface ColumnProfile {
  name: string;
  inferred_type: 'NUMERICAL' | 'CATEGORICAL' | 'DATETIME' | 'BOOLEAN' | 'TEXT' | 'IDENTIFIER' | 'UNKNOWN';
  total_count: number;
  non_null_count: number;
  null_count: number;
  null_percentage: number;
  unique_count: number;
  unique_percentage: number;
  min?: number;
  max?: number;
  mean?: number;
  median?: number;
  std?: number;
  skewness?: number;
  kurtosis?: number;
  histogram?: Array<{ bin_start: number; bin_end: number; count: number }>;
  top_categories?: Array<{ category: string; count: number; percentage: number }>;
  cardinality?: number;
  avg_length?: number;
  min_date?: string;
  max_date?: string;
  date_range_days?: number;
}

export interface DatasetProfile {
  row_count: number;
  column_count: number;
  memory_usage_bytes: number;
  memory_usage_mb: number;
  schema: {
    columns: Record<string, {
      name: string;
      inferred_type: string;
      raw_dtype: string;
      null_count: number;
      null_percentage: number;
      unique_count: number;
      unique_percentage: number;
      is_nullable: boolean;
      sample_values: any[];
    }>;
    total_rows: number;
    total_columns: number;
    column_types_summary: Record<string, number>;
  };
  columns: Record<string, ColumnProfile>;
}

export interface DataQualityReportData {
  score: number;
  grade: string;
  score_breakdown: {
    overall_score: number;
    grade: string;
    components: {
      completeness: number;
      validity: number;
      uniqueness: number;
      consistency: number;
    };
    penalties: Record<string, number>;
  };
  issues: {
    missing_values: Array<{ column: string; count: number; percentage: number; severity: string; inferred_type: string }>;
    duplicates: { count: number; percentage: number };
    constant_columns: string[];
    near_constant_columns: string[];
    outliers: Array<{ column: string; count: number; percentage: number; lower_bound: number; upper_bound: number; method: string }>;
    invalid_values: Array<{ column: string; issue: string }>;
  };
  recommendations: Array<{
    id: string;
    category: string;
    priority: 'HIGH' | 'MEDIUM' | 'LOW';
    title: string;
    description: string;
    affected_columns: string[];
    action_type: string;
  }>;
}

export interface EDAResponseData {
  summary: {
    row_count: number;
    column_count: number;
    numerical_columns_count: number;
    categorical_columns_count: number;
    datetime_columns_count: number;
    high_correlation_pairs_count: number;
  };
  correlations: {
    numerical_columns: string[];
    matrix: Record<string, Record<string, number>>;
    significant_correlations: Array<{
      feature_a: string;
      feature_b: string;
      pearson_r: number;
      abs_r: number;
      strength: string;
      direction: string;
    }>;
  };
  distributions: Record<string, any>;
}

export interface InsightItem {
  id: string;
  category: string;
  title: string;
  description: string;
  severity: string;
  confidence: number;
  verified: boolean;
  metric_evidence: Record<string, any>;
  recommendation?: string | null;
}

export interface VisualizationSpec {
  id: string;
  title: string;
  chart_type: string;
  x_field: string;
  y_field?: string | null;
  description: string;
  data: any[];
}

export interface PaginatedPreviewData {
  page: number;
  page_size: number;
  total_rows: number;
  total_pages: number;
  total_columns: number;
  columns: string[];
  dtypes: Record<string, string>;
  inferred_types: Record<string, string>;
  rows: Record<string, any>[];
}

// ----------------------------------------------------
// Phase 3: Autonomous Machine Learning & MLOps Types
// ----------------------------------------------------

export interface TargetSuggestion {
  column: string;
  confidence: number;
  suggested_type: string;
  unique_count: number;
  null_count: number;
  reasons: string[];
  sample_values: string[];
}

export interface TargetSuggestionsResponse {
  dataset_id: string;
  total_columns: number;
  suggestions: TargetSuggestion[];
}

export interface LeakageWarning {
  column: string;
  risk_type: string;
  severity: 'CRITICAL' | 'WARNING' | 'INFO';
  description: string;
  recommendation: string;
  correlation?: number | null;
}

export interface LeakageAuditResponse {
  has_leakage_risk: boolean;
  has_critical_risk: boolean;
  target_column: string;
  warning_count: number;
  warnings: LeakageWarning[];
  recommended_drop_columns: string[];
}

export interface MLTrainRequest {
  dataset_id: string;
  target_column: string;
  candidate_algorithms?: string[];
  excluded_columns?: string[];
  cv_splits?: number;
  optimize_hyperparameters?: boolean;
  optuna_trials?: number;
}

export interface MLModelSummary {
  id: string;
  project_id: string;
  dataset_id: string;
  name: string;
  algorithm: string;
  problem_type: string;
  target_column: string;
  version: string;
  lifecycle_stage: 'CANDIDATE' | 'STAGING' | 'PRODUCTION' | 'ARCHIVED';
  primary_metric_name: string;
  primary_metric_value: number;
  delta_improvement_pct?: number | null;
  training_duration_seconds?: number | null;
  created_at: string;
}

export interface MLModelDetail extends MLModelSummary {
  all_metrics: Record<string, any>;
  cv_scores?: Record<string, any> | null;
  confusion_matrix?: {
    labels: any[];
    matrix: number[][];
  } | null;
  roc_curve?: Array<{
    fpr: number;
    tpr: number;
    threshold: number;
  }> | null;
  actual_vs_pred?: Array<{
    actual: number;
    predicted: number;
  }> | null;
  hyperparameters?: Record<string, any> | null;
  feature_importance?: Array<{
    rank: number;
    feature: string;
    importance_score: number;
    percentage: number;
  }> | null;
  feature_names?: string[] | null;
  model_card?: {
    model_name: string;
    problem_type: string;
    target_column: string;
    features: string[];
    metrics: Record<string, any>;
    dataset_summary: Record<string, any>;
    baseline_comparison?: Record<string, any> | null;
    markdown_card?: string;
  } | null;
  artifact_path?: string | null;
  status: string;
}

export interface SinglePredictRequest {
  features: Record<string, any>;
}

export interface ContributingFeature {
  feature: string;
  feature_value: any;
  impact: number;
  direction: 'positive' | 'negative';
}

export interface PredictResponse {
  model_id: string;
  model_version: string;
  prediction: any;
  probability?: number | null;
  probabilities?: Record<string, number> | null;
  contributing_features: ContributingFeature[];
  latency_ms: number;
}

// ----------------------------------------------------
// Phase 4: Advanced AI Engine — Forecasting & Anomaly Detection
// ----------------------------------------------------

export interface TimeSeriesCandidate {
  column_name: string;
  confidence_score: number;
  is_parsed: boolean;
  reasons: string[];
}

export interface TemporalProfile {
  time_column: string;
  total_observations: number;
  valid_timestamps: number;
  min_timestamp: string;
  max_timestamp: string;
  is_chronological: boolean;
  duplicate_timestamps: number;
  inferred_frequency: string;
  median_interval_seconds: number;
  is_regular_interval: boolean;
  estimated_missing_intervals: number;
  trend: 'UPWARD' | 'DOWNWARD' | 'STATIONARY' | 'UNKNOWN';
  seasonality_period?: number | null;
  autocorrelation_lag1?: number | null;
}

export interface TimeSeriesDetectResponse {
  dataset_id: string;
  candidates: TimeSeriesCandidate[];
  profile?: TemporalProfile | null;
}

export interface ForecastTrainRequest {
  project_id: string;
  dataset_id: string;
  time_column: string;
  target_column: string;
  horizon?: number;
  candidate_models?: string[];
  covariates?: string[];
}

export interface ForecastIntervalPoint {
  step: number;
  timestamp: string;
  prediction: number;
  lower_80: number;
  upper_80: number;
  lower_95: number;
  upper_95: number;
  uncertainty_sigma: number;
  estimation_method: string;
  disclaimer: string;
}

export interface ForecastModelResponse {
  id: string;
  project_id: string;
  dataset_id: string;
  time_column: string;
  target_column: string;
  horizon: number;
  best_model_name: string;
  lifecycle_stage: string;
  metrics: {
    mae: number;
    rmse: number;
    mape: number;
    smape: number;
    r2: number;
    mase?: number | null;
  };
  leaderboard: Array<{
    model_name: string;
    status: string;
    mae: number;
    rmse: number;
    mape?: number;
    smape?: number;
    r2?: number;
    training_duration_seconds?: number;
    loss_curves?: {
      train_loss: number[];
      val_loss: number[];
    } | null;
    error?: string;
  }>;
  temporal_profile?: TemporalProfile | null;
  future_forecast: ForecastIntervalPoint[];
  recent_history?: Array<{ timestamp: string; actual: number }> | null;
  validation_comparison?: Array<{ timestamp: string; actual: number; predicted: number }> | null;
  feature_importances?: Record<string, number> | null;
  training_duration_seconds?: number | null;
  created_at?: string | null;
}

export interface DynamicForecastResponse {
  model_id: string;
  best_model_name: string;
  horizon: number;
  forecast: ForecastIntervalPoint[];
}

export interface AnomalyEventItem {
  id: string;
  index: number;
  timestamp: string;
  metric_name: string;
  actual: number;
  expected: number;
  deviation: number;
  anomaly_score: number;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  detection_methods: string[];
  explanation: string;
  recommendation: string;
}

export interface ChangePointItem {
  index: number;
  timestamp: string;
  disparity_score: number;
  previous_mean: number;
  new_mean: number;
  magnitude: number;
  percentage_change: number;
  regime_type: 'UPWARD_SHIFT' | 'DOWNWARD_SHIFT';
  confidence: number;
}

export interface AnomalySummary {
  total_observations: number;
  total_anomalies: number;
  critical_count: number;
  high_count: number;
  medium_count: number;
  low_count: number;
  anomaly_percentage: number;
  change_points_count: number;
}

export interface AnomalyReportResponse {
  id: string;
  project_id: string;
  dataset_id: string;
  analyzed_metric: string;
  dataset_summary: AnomalySummary;
  anomalies: AnomalyEventItem[];
  change_points: ChangePointItem[];
  timeline?: Array<{
    timestamp: string;
    actual: number;
    expected: number;
    is_anomaly: boolean;
    severity?: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
    score?: number;
  }> | null;
  created_at?: string | null;
}

// ==========================================
// Multi-Agent Swarm & Critic Types
// ==========================================
export interface SwarmStepTrace {
  step_id: string;
  agent_id: string;
  agent_name: string;
  status: 'queued' | 'running' | 'completed' | 'verified' | 'rejected' | 'failed';
  started_at: string;
  completed_at?: string | null;
  duration_ms: number;
  summary: string;
  tools_used: string[];
  reasoning?: string | null;
  data_artifacts: Record<string, any>;
}

export interface CriticAuditItem {
  claim: string;
  metric_name: string;
  reported_value: any;
  ground_truth_value: any;
  verified: boolean;
  status: 'SUPPORTED' | 'PARTIALLY_SUPPORTED' | 'REJECTED';
  delta?: number | null;
  confidence: number;
  reason: string;
}

export interface CriticVerificationResult {
  status: 'SUPPORTED' | 'PARTIALLY_SUPPORTED' | 'REJECTED';
  audit_count: number;
  supported_count: number;
  rejected_count: number;
  overall_confidence: number;
  gatekeeper_verdict: 'PASS' | 'PASS_WITH_WARNINGS' | 'REJECT';
  audits: CriticAuditItem[];
}

export interface RecommendationItem {
  id: string;
  title: string;
  category: 'strategic' | 'risk_mitigation' | 'operational' | 'model_deployment';
  action: string;
  expected_impact: string;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  confidence: number;
  supporting_metrics: string[];
}

export interface SwarmRunResponse {
  job_id: string;
  project_id: string;
  dataset_id: string;
  status: 'running' | 'completed' | 'failed' | 'rejected_by_critic';
  progress_percentage: number;
  current_step?: string | null;
  trace: SwarmStepTrace[];
  critic_verdict?: CriticVerificationResult | null;
  recommendations: RecommendationItem[];
  report_markdown?: string | null;
  created_at: string;
  completed_at?: string | null;
}

export interface SwarmRunRequest {
  project_id: string;
  dataset_id: string;
  target_column?: string | null;
  time_column?: string | null;
  goal?: string;
  run_automl?: boolean;
  run_forecasting?: boolean;
  run_anomalies?: boolean;
  run_rag?: boolean;
  document_query?: string | null;
}

// ==========================================
// Document Intelligence & RAG Types
// ==========================================
export interface RAGDocument {
  document_id: string;
  project_id: string;
  filename: string;
  file_type: string;
  file_size: number;
  sha256_hash?: string | null;
  page_count: number;
  chunk_count: number;
  status: string;
  created_at?: string | null;
}

export interface RAGCitation {
  citation_id: string;
  citation_tag: string;
  label: string;
  claim?: string;
  document_id: string;
  chunk_id: string;
  filename: string;
  page: number;
  section?: string;
  confidence: number;
  snippet: string;
}

export interface RAGQueryResponse {
  query: string;
  answer: string;
  grounded: boolean;
  citations: RAGCitation[];
  evidence: Array<Record<string, any>>;
  duration_ms: number;
}

export interface HybridSearchResultItem {
  chunk_id: string;
  document_id: string;
  text: string;
  score?: number;
  dense_score?: number;
  bm25_score?: number;
  metadata?: Record<string, any>;
}

export interface HybridSearchResponse {
  query: string;
  total_found: number;
  results: HybridSearchResultItem[];
  duration_ms: number;
}

export interface GraphNodeData {
  id: string;
  name: string;
  node_type: string;
  properties: Record<string, any>;
  project_id?: string | null;
}

export interface GraphEdgeData {
  id: string;
  source_id: string;
  target_id: string;
  relation: string;
  weight: number;
  properties: Record<string, any>;
}

export interface KnowledgeGraphData {
  project_id?: string | null;
  nodes: GraphNodeData[];
  edges: GraphEdgeData[];
  metrics: {
    node_count: number;
    edge_count: number;
    density: number;
    node_type_distribution: Record<string, number>;
    relation_distribution: Record<string, number>;
    key_entities: Array<{
      node_id: string;
      name: string;
      type: string;
      score: number;
    }>;
    is_connected: boolean;
    backend?: string;
  };
}

export interface GraphBuildResult {
  project_id: string;
  dataset_id?: string | null;
  nodes_created: number;
  edges_created: number;
  total_nodes: number;
  total_edges: number;
  duration_ms: number;
  status: string;
}

export interface GraphRAGResult {
  answer: string;
  grounded: boolean;
  seed_entities: Array<{
    id: string;
    name: string;
    node_type: string;
    properties: Record<string, any>;
  }>;
  facts: string[];
  subgraph: {
    nodes: GraphNodeData[];
    edges: GraphEdgeData[];
    node_count: number;
    edge_count: number;
  };
  paths: Array<{
    nodes: GraphNodeData[];
    edges: GraphEdgeData[];
    score: number;
    description: string;
  }>;
  duration_ms: number;
}


