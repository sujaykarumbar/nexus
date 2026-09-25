# 🚀 NEXUS — Autonomous Multi-Agent Data Intelligence & Prediction Platform

NEXUS is an end-to-end, production-grade autonomous data intelligence and prediction platform. Acting as an **AI-powered automated data scientist**, NEXUS takes raw datasets (CSV, Excel, JSON, Parquet, streaming feeds) and autonomously executes:

1. **Automated Ingestion & Schema Profiling**: Type inference, missingness mapping, cardinality and outlier analysis.
2. **Data Quality Scoring**: Rigorous, deterministic quality scoring (0-100) based on invalid, outlier, and missing records.
3. **Automated EDA (Exploratory Data Analysis)**: Feature correlations, distributions, skewness, and class imbalance calculations.
4. **AutoML Engine**: Problem classification, model selection (Linear/Logistic, Random Forest, XGBoost, LightGBM, Neural Nets), cross-validation, and multi-metric benchmarking.
5. **Time-Series Forecasting & Anomaly Detection**: Trend decomposition, stationarity testing, prediction intervals, and Isolation Forest anomaly ranking.
6. **Multi-Agent Orchestration**: Graph-based LangGraph DAG orchestrating Data, ML, Forecasting, RAG, and Critic agents.
7. **Critic Verification Layer**: Strict mathematical evidence checking that prevents hallucinated claims.
8. **Decision Intelligence**: Actionable recommendations with quantified risk bounds and confidence intervals.
9. **Interactive Command Center UI**: Modern React + TypeScript interface with dark mode, live agent execution traces, and model comparison labs.

---

## 🏛️ System Architecture

```
                               ┌────────────────────────────────┐
                               │   NEXUS Command Center (UI)    │
                               │      React + TypeScript        │
                               └───────────────┬────────────────┘
                                               │ (REST / WebSocket)
                                               ▼
                               ┌────────────────────────────────┐
                               │       API Gateway (FastAPI)    │
                               │   Auth • Jobs • Profiling • ML │
                               └───────────────┬────────────────┘
                                               │
               ┌───────────────────────────────┼───────────────────────────────┐
               ▼                               ▼                               ▼
       ┌───────────────┐               ┌───────────────┐               ┌───────────────┐
       │ Multi-Agent   │               │ Data & AutoML │               │ RAG & Graph   │
       │ Orchestrator  │               │ Engine        │               │ Engine        │
       └───────┬───────┘               └───────┬───────┘               └───────┬───────┘
               │                               │                               │
               └───────────────────────────────┼───────────────────────────────┘
                                               ▼
                               ┌────────────────────────────────┐
                               │       Storage & Persistence    │
                               │  PostgreSQL • Redis • Chroma   │
                               └────────────────────────────────┘
```

---

## ⚡ Quick Start (Phase 1)

### Prerequisites
* Python 3.11+
* Node.js 18+ and npm 9+
* Docker & Docker Compose (Optional for full containerization)

### Local Development Setup

1. **Clone & Setup Backend**:
   ```bash
   cd nexus
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate

   pip install -r requirements.txt
   ```

2. **Run Backend API**:
   ```bash
   uvicorn apps.api.main:app --reload --host 127.0.0.1 --port 8000
   ```
   * Interactive OpenAPI Docs: `http://localhost:8000/docs`
   * Health Check: `http://localhost:8000/api/v1/health`

3. **Run Frontend Application**:
   ```bash
   cd apps/web
   npm install
   npm run dev
   ```
   * Access the UI at: `http://localhost:5173`

4. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

---

## 🧪 Testing

Run backend tests with pytest:
```bash
pytest tests/backend -v
```

---

## 🗺️ Engineering Roadmap — 100% Completed
- [x] **Phase 1**: Architecture, Monorepo, FastAPI backend, JWT Auth, Database models, React Command Center UI, Docker setup.
- [x] **Phase 2**: Dataset Ingestion, Validation, Profiling & Automated EDA Engine.
- [x] **Phase 3**: AutoML Engine, Bayesian Hyperparameter Tuning & Model Comparison.
- [x] **Phase 4**: Time-Series Forecasting, Anomaly Detection & SHAP Explainability.
- [x] **Phase 5**: LangGraph Multi-Agent Swarm with Strict Critic Verification.
- [x] **Phase 6**: Document Intelligence & Production RAG with Evidence Citations.
- [x] **Phase 7**: Knowledge Graph Extraction & GraphRAG (Neo4j).
- [x] **Phase 8**: Real-Time Streaming, Async Event Bus & Anomaly Detection Simulator.
- [x] **Phase 9**: MLOps Engine, Model Version Registry, Data Drift (PSI/KS) & Pipeline Scheduler.
- [x] **Phase 10**: Deep Health Observability, Executive Intelligence Dossiers & Production Hardening.
