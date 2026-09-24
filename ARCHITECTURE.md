# NEXUS Architecture & Engineering Specification

## 1. System Philosophy: Deterministic Computation + Agentic Reasoning

In enterprise data platforms, LLMs must **never** be used as raw calculators or ungrounded predictors. NEXUS strictly enforces the separation between **computation** (deterministic Python, pandas, scikit-learn, statsmodels) and **reasoning** (agent planning, hypothesis formation, synthesis, explanation).

```
RAW DATA ──► PROFILING ──► DATA QUALITY ──► AUTOML / FORECAST ──► METRIC EXTRACTION ──► CRITIC VERIFICATION ──► DECISION INTELLIGENCE
   ▲                                                                                          │
   └────────────────────── [RE-ANALYZE IF REJECTED BY CRITIC] ────────────────────────────────┘
```

---

## 2. Layered Architecture

### Presentation Tier (`apps/web`)
* Built with **React 18**, **TypeScript**, and **Vite**.
* Styled using **TailwindCSS** with dark-mode first design tokens, glassmorphism paneling, and micro-interactions.
* Data visualizations powered by **Recharts** and **Plotly.js**.
* Live agent execution visualizer showing DAG state transitions and tool outputs.

### API Gateway Tier (`apps/api`)
* **FastAPI** application with asynchronous request handling.
* **JWT Authentication** (Bearer tokens with HMAC-SHA256 signature verification and bcrypt password hashing).
* Automated OpenAPI/Swagger documentation generation.
* Global exception handling and structured JSON error envelopes.
* Dual database support: SQLite for instant, zero-dependency local development and PostgreSQL 16 for production deployments.

### Autonomous Agent Swarm (`services/agents`)
* Implemented as a stateful directed acyclic graph (DAG) with **LangGraph**.
* Dedicated agent roles:
  * **Data Agent**: Data hygiene, missing value strategies, outlier identification.
  * **Data Scientist Agent**: Statistical distributions, hypothesis generation, correlation analysis.
  * **ML Engineer Agent**: Problem classification, model selection, hyperparameter tuning, metric benchmarking.
  * **Forecasting Agent**: Autoregressive modeling, trend/seasonality decomposition, prediction intervals.
  * **Anomaly Agent**: Unsupervised outlier ranking with Isolation Forests.
  * **Research / RAG Agent**: Document retrieval and chunk citation grounding.
  * **Critic Agent**: Strict gatekeeper that mathematically validates all claims against raw computed metrics.
  * **Recommendation Agent**: Formulates decision actions with quantified risk bounds.
  * **Report Agent**: Compiles executive HTML/PDF analytical dossiers.

### Machine Learning & Data Engines (`services/ml`)
* **AutoML Engine**: Trains and compares Logistic/Linear Regression, Random Forest, XGBoost, LightGBM, SVM, and Neural Networks across 5-fold cross-validation.
* **Evaluation Matrix**: Multi-metric evaluation (Accuracy, F1, Precision, Recall, ROC-AUC, PR-AUC for classification; MAE, MSE, RMSE, R², MAPE for regression).
* **Explainability**: SHAP (SHapley Additive exPlanations) values computed from trained models to generate transparent feature importance scores.

---

## 3. Database Schema Design (Key Tables)

| Table | Purpose | Primary Key | Key Relationships |
| :--- | :--- | :--- | :--- |
| `users` | User accounts, credentials, and roles | `UUID` | 1:N with `projects`, `audit_logs` |
| `organizations` | Tenant isolation and team workspaces | `UUID` | 1:N with `users`, `projects` |
| `projects` | Grouping container for datasets and models | `UUID` | 1:N with `datasets`, `models`, `jobs` |
| `datasets` | Ingested data sources (CSV/Excel/JSON) | `UUID` | 1:N with `dataset_versions` |
| `dataset_versions` | Immutable snapshots of dataset versions | `UUID` | 1:N with `data_quality_reports`, `eda_reports` |
| `analysis_jobs` | Asynchronous task execution tracking | `UUID` | 1:N with `agent_runs` |
| `ml_models` | Registered ML models and artifacts | `UUID` | 1:N with `model_versions`, `predictions` |
| `anomalies` | Detected anomalous records and scores | `UUID` | N:1 with `dataset_versions` |
| `forecasts` | Time-series projected intervals | `UUID` | N:1 with `dataset_versions` |
| `agent_runs` | Multi-agent execution traces and logs | `UUID` | 1:N with `agent_messages` |
