# GraphFraud-X

> **Live Demo Dashboard:** [https://graphfraud-x.vercel.app](https://graphfraud-x.vercel.app)  

**Graph Neural Network (GraphSAGE) Financial Fraud Detection & Coordinated Ring Intelligence Platform**

GraphFraud-X models financial networks across **50,000 monitored accounts** and **1,000,000+ transactions** as a heterogeneous interaction graph to uncover coordinated money-laundering syndicates that traditional tabular models miss.

---

## Key Performance Metrics

| Metric | Score |
| :--- | :--- |
| **Test Accuracy** | **98.42%** |
| **ROC-AUC** | **0.9974** |
| **PR-AUC** | **0.9881** |
| **Decision Threshold** | `0.30` (Calibrated for Fraud Recall) |
| **Detected Fraud Rings** | **2,652 Coordinated Clusters** |

---

## System Architecture

* **Graph Machine Learning (`models/`, `pipeline/`):** PyTorch Geometric `GraphSAGE` architecture trained on **27 engineered behavioral & NetworkX structural features** (node degrees, clustering coefficients, ring sizes, and transaction velocity) with class-weighted loss.
* **Persistent Storage (`api/database.py`, `pipeline/init_db.py`):** SQLite + SQLAlchemy relational layer (`data/graphfraud.db`) storing accounts, GNN risk scores, network edges, 1M+ transaction records, daily/weekly activity trends, and real-time inference logs.
* **REST API (`api/app.py`):** FastAPI backend serving sub-second inductive account scoring, ring topology subgraphs, suspicious counterparty transaction histories, and fraud cluster analytics.
* **Interactive SOC Dashboard (`frontend/`):** React, Tailwind CSS, and Cytoscape.js executive console for live network ring exploration, daily/weekly fraud analytics, account investigation, and real-time GNN risk scoring.

---

## Project Structure

```text
GraphFraud-X/
├── api/
│   ├── app.py              # FastAPI endpoints & inductive GNN inference
│   └── database.py         # SQLAlchemy ORM models & SQLite session manager
├── config/
│   └── settings.py         # Model hyperparameters & artifact paths
├── frontend/               # React + Vite + Tailwind CSS + Cytoscape dashboard
├── models/
│   └── gnn.py              # FraudGraphSAGE PyTorch Geometric architecture
├── pipeline/
│   └── init_db.py          # CSV-to-SQLite database migration pipeline
└── saved_models/           # Trained GraphSAGE weights & feature scaler
```
---

## Local Setup & Execution

### 1. Backend & Database Setup
```bash
# Activate virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate
pip install fastapi uvicorn sqlalchemy pandas numpy scikit-learn torch torch-geometric joblib

# Initialize SQLite database (data/graphfraud.db)
python -m pipeline.init_db

# Start FastAPI server
uvicorn api.app:app --reload
```
API Health Check: http://127.0.0.1:8000/api/v1/health

---
### 2. Frontend Dashboard Setup

```bash
cd frontend
npm install
npm run dev
```
Dashboard URL: http://localhost:5174

---
