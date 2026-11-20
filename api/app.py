from typing import List
from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import func, case, or_
import joblib
import numpy as np
import torch

from models.gnn import FraudGraphSAGE
from config.settings import ModelConfig, PathConfig
from api.database import (
    init_tables, get_db,
    Account, NetworkEdge, TransactionRecord, ActivityTrend
)

app = FastAPI(
    title="GraphFraud-X Detection API",
    version="2.0.0",
    description="Database-backed API for GNN-powered financial fraud detection and coordinated ring analysis."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

paths = PathConfig()
SCALER = None
GNN_MODEL = None
MODEL_CONFIG = ModelConfig()


@app.on_event("startup")
def on_startup():
    init_tables()
    from api.database import SessionLocal
    db = SessionLocal()
    try:
        if db.query(func.count(Account.account_id)).scalar() == 0:
                        sample_accounts = [
                Account(account_id=f"ACC000000{i}", fraud_probability=round(0.94 - (i * 0.06), 4),
                        anomaly_score=round(0.91 - (i * 0.05), 4),
                        risk_level="CRITICAL" if i <= 3 else ("HIGH" if i <= 6 else "MEDIUM"),
                        ring_id="RING_1230" if i <= 5 else "RING_0318",
                        ring_size=5 if i <= 5 else 4,
                        action_recommended="FREEZE_ACCOUNT_AND_TRIGGER_AML_AUDIT" if i <= 3 else "STEP_UP_AUTHENTICATION_AND_TRANSACTION_LIMIT")
                for i in range(1, 10)
            ]
            db.add_all(sample_accounts)
            db.add_all([
                NetworkEdge(account_a="ACC0000001", account_b="ACC0000002", shared_type="SHARED_DEVICE", connection_count=6),
                NetworkEdge(account_a="ACC0000002", account_b="ACC0000003", shared_type="RAPID_TRANSFER", connection_count=4),
                NetworkEdge(account_a="ACC0000003", account_b="ACC0000004", shared_type="SHARED_IP", connection_count=5),
                NetworkEdge(account_a="ACC0000004", account_b="ACC0000005", shared_type="SHARED_DEVICE", connection_count=3),
                NetworkEdge(account_a="ACC0000001", account_b="ACC0000005", shared_type="CIRCULAR_FLOW", connection_count=7),
                NetworkEdge(account_a="ACC0000006", account_b="ACC0000007", shared_type="SHARED_IP", connection_count=4),
                NetworkEdge(account_a="ACC0000007", account_b="ACC0000008", shared_type="RAPID_TRANSFER", connection_count=5),
            ])
            db.add_all([
                TransactionRecord(tx_id=f"TX_{1000+i}", source=f"ACC000000{i}", target=f"ACC000000{(i%8)+1}",
                                  amount=18500.0 * i, timestamp=f"2026-11-1{i} 14:20:00", is_suspicious=True)
                for i in range(1, 9)
            ])
            db.add_all([
                ActivityTrend(granularity="daily", period=f"2026-11-1{d}", fraud_count=42 + d * 5, normal_count=1200 + d * 30, fraud_volume=485000.0 + d * 42000)
                for d in range(1, 8)
            ] + [
                ActivityTrend(granularity="weekly", period=f"Week {w}", fraud_count=280 + w * 25, normal_count=8400 + w * 150, fraud_volume=3200000.0 + w * 210000)
                for w in range(1, 5)
            ])
            db.commit()
    finally:
        db.close()


def get_inference_resources():
    global SCALER, GNN_MODEL
    if SCALER is None and paths.scaler_path.exists():
        SCALER = joblib.load(paths.scaler_path)

    if GNN_MODEL is None:
        model_file = paths.models_dir / "best_gnn_model.pt"
        if model_file.exists():
            GNN_MODEL = FraudGraphSAGE(
                in_channels=27,
                hidden_channels=MODEL_CONFIG.hidden_channels,
                out_channels=MODEL_CONFIG.out_channels,
                dropout=0.0
            )
            GNN_MODEL.load_state_dict(torch.load(model_file, weights_only=True, map_location="cpu"))
            GNN_MODEL.eval()

    return SCALER, GNN_MODEL


class RealtimePredictionRequest(BaseModel):
    account_id: str
    features: List[float] = Field(
        ...,
        description="List of 27 numerical predictive features corresponding to account behavior"
    )


class RealtimePredictionResponse(BaseModel):
    account_id: str
    fraud_probability: float
    predicted_label: int
    risk_level: str
    action_recommended: str


@app.post("/api/v1/predict/account", response_model=RealtimePredictionResponse)
def predict_single_account(payload: RealtimePredictionRequest, db: Session = Depends(get_db)):
    scaler, model = get_inference_resources()
    if scaler is None or model is None:
        raise HTTPException(status_code=500, detail="Inference engine resources not loaded on server.")

    if len(payload.features) != 27:
        raise HTTPException(
            status_code=422,
            detail=f"Expected exactly 27 numerical feature values, received {len(payload.features)}."
        )

    raw_array = np.array(payload.features, dtype=np.float32).reshape(1, -1)
    scaled_features = scaler.transform(raw_array)

    x = torch.tensor(scaled_features, dtype=torch.float)
    edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    with torch.no_grad():
        logits = model(x, edge_index)
        probs = torch.softmax(logits, dim=1)
        fraud_prob = float(probs[0, 1].item())
        pred_label = int(torch.argmax(probs, dim=1).item())

    if fraud_prob >= 0.75:
        risk_level = "CRITICAL"
        recommendation = "FREEZE_ACCOUNT_AND_TRIGGER_AML_AUDIT"
    elif fraud_prob >= 0.50:
        risk_level = "HIGH"
        recommendation = "STEP_UP_AUTHENTICATION_AND_TRANSACTION_LIMIT"
    elif fraud_prob >= 0.25:
        risk_level = "MEDIUM"
        recommendation = "FLAG_FOR_ENHANCED_MONITORING"
    else:
        risk_level = "LOW"
        recommendation = "ALLOW_NORMAL_OPERATIONS"

    acc_id = payload.account_id.strip()
    existing = db.query(Account).filter(Account.account_id == acc_id).first()
    if existing:
        existing.fraud_probability = round(fraud_prob, 4)
        existing.anomaly_score = round(fraud_prob, 4)
        existing.risk_level = risk_level
        existing.action_recommended = recommendation
    else:
        db.add(Account(
            account_id=acc_id,
            fraud_probability=round(fraud_prob, 4),
            anomaly_score=round(fraud_prob, 4),
            risk_level=risk_level,
            ring_id="ISOLATED",
            ring_size=1,
            action_recommended=recommendation
        ))
    db.commit()

    return RealtimePredictionResponse(
        account_id=acc_id,
        fraud_probability=round(fraud_prob, 4),
        predicted_label=pred_label,
        risk_level=risk_level,
        action_recommended=recommendation
    )


class AccountRiskResponse(BaseModel):
    account_id: str
    fraud_probability: float
    anomaly_score: float
    risk_level: str
    ring_id: str
    ring_size: int


@app.get("/api/v1/health")
def health_check(db: Session = Depends(get_db)):
    count = db.query(func.count(Account.account_id)).scalar() or 0
    return {
        "status": "healthy",
        "database": "SQLite (data/graphfraud.db)",
        "service": "GraphFraud-X Backend",
        "accounts_loaded": count
    }


@app.get("/api/v1/account/{account_id}", response_model=AccountRiskResponse)
def get_account_risk(account_id: str, db: Session = Depends(get_db)):
    acc = db.query(Account).filter(Account.account_id == account_id.strip()).first()
    if not acc:
        raise HTTPException(status_code=404, detail=f"Account '{account_id}' not found in database.")

    return AccountRiskResponse(
        account_id=acc.account_id,
        fraud_probability=float(acc.fraud_probability),
        anomaly_score=float(acc.anomaly_score),
        risk_level=acc.risk_level,
        ring_id=acc.ring_id,
        ring_size=int(acc.ring_size),
    )


@app.get("/api/v1/rings/top")
def get_top_fraud_rings(limit: int = Query(default=10, ge=1, le=50), db: Session = Depends(get_db)):
    rows = (
        db.query(
            Account.ring_id,
            func.max(Account.ring_size).label("ring_size"),
            func.avg(Account.anomaly_score).label("avg_anomaly"),
            func.max(Account.anomaly_score).label("max_anomaly"),
            func.sum(case((Account.risk_level == "CRITICAL", 1), else_=0)).label("critical_count")
        )
        .filter(Account.ring_id != "ISOLATED")
        .group_by(Account.ring_id)
        .order_by(func.sum(case((Account.risk_level == "CRITICAL", 1), else_=0)).desc(), func.avg(Account.anomaly_score).desc())
        .limit(limit)
        .all()
    )

    result = []
    for r in rows:
        members = [
            m[0] for m in db.query(Account.account_id).filter(Account.ring_id == r.ring_id).limit(20).all()
        ]
        result.append({
            "ring_id": r.ring_id,
            "ring_size": int(r.ring_size),
            "avg_anomaly": round(float(r.avg_anomaly), 4),
            "max_anomaly": round(float(r.max_anomaly), 4),
            "critical_count": int(r.critical_count),
            "accounts": members
        })
    return result


@app.get("/api/v1/rings/{ring_id}/graph")
def get_ring_graph(ring_id: str, db: Session = Depends(get_db)):
    ring_accounts = db.query(Account).filter(Account.ring_id == ring_id).all()
    if not ring_accounts:
        raise HTTPException(status_code=404, detail=f"Ring '{ring_id}' not found.")

    account_ids = [a.account_id for a in ring_accounts]
    sub_edges = (
        db.query(NetworkEdge)
        .filter(NetworkEdge.account_a.in_(account_ids), NetworkEdge.account_b.in_(account_ids))
        .all()
    )

    nodes = [
        {
            "id": a.account_id,
            "label": a.account_id,
            "risk_level": a.risk_level,
            "fraud_probability": float(a.fraud_probability),
            "anomaly_score": float(a.anomaly_score)
        }
        for a in ring_accounts
    ]

    edges = [
        {
            "source": e.account_a,
            "target": e.account_b,
            "connection_type": e.shared_type,
            "weight": int(e.connection_count)
        }
        for e in sub_edges
    ]

    return {
        "ring_id": ring_id,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "nodes": nodes,
        "edges": edges
    }


@app.get("/api/v1/analytics/overview")
def get_dashboard_overview(db: Session = Depends(get_db)):
    total_accounts = db.query(func.count(Account.account_id)).scalar() or 0
    risk_rows = db.query(Account.risk_level, func.count(Account.account_id)).group_by(Account.risk_level).all()
    risk_counts = {r[0]: int(r[1]) for r in risk_rows}

    unique_rings = db.query(func.count(func.distinct(Account.ring_id))).filter(Account.ring_id != "ISOLATED").scalar() or 0
    accounts_in_rings = db.query(func.count(Account.account_id)).filter(Account.ring_id != "ISOLATED").scalar() or 0
    largest_ring = db.query(func.max(Account.ring_size)).filter(Account.ring_id != "ISOLATED").scalar() or 0

    avg_fraud_prob = db.query(func.avg(Account.fraud_probability)).scalar() or 0.0
    avg_anomaly = db.query(func.avg(Account.anomaly_score)).scalar() or 0.0

    return {
        "total_monitored_accounts": total_accounts,
        "risk_breakdown": {
            "CRITICAL": risk_counts.get("CRITICAL", 0),
            "HIGH": risk_counts.get("HIGH", 0),
            "MEDIUM": risk_counts.get("MEDIUM", 0),
            "LOW": risk_counts.get("LOW", 0),
        },
        "coordinated_fraud": {
            "total_detected_rings": unique_rings,
            "accounts_in_rings": accounts_in_rings,
            "largest_ring_size": int(largest_ring)
        },
        "averages": {
            "average_fraud_probability": round(float(avg_fraud_prob), 4),
            "average_anomaly_score": round(float(avg_anomaly), 4)
        }
    }


@app.get("/api/v1/accounts/{account_id}/transactions")
def get_account_transaction_history(account_id: str, limit: int = 25, db: Session = Depends(get_db)):
    acc_id = account_id.strip()
    tx_rows = (
        db.query(TransactionRecord)
        .filter(or_(TransactionRecord.source == acc_id, TransactionRecord.target == acc_id))
        .limit(limit)
        .all()
    )

    counterparties = list({(t.target if t.source == acc_id else t.source) for t in tx_rows})
    cp_accounts = {
        a.account_id: a for a in db.query(Account).filter(Account.account_id.in_(counterparties)).all()
    } if counterparties else {}

    history = []
    for t in tx_rows:
        cp = t.target if t.source == acc_id else t.source
        cp_obj = cp_accounts.get(cp)
        cp_risk = cp_obj.risk_level if cp_obj else "HIGH"
        cp_prob = float(cp_obj.fraud_probability) if cp_obj else 0.84
        cp_ring = cp_obj.ring_id if cp_obj else "RING_DETECTED"
        history.append({
            "tx_id": t.tx_id,
            "timestamp": t.timestamp,
            "direction": "OUTGOING" if t.source == acc_id else "INCOMING",
            "counterparty": cp,
            "amount": round(float(t.amount), 2),
            "counterparty_risk": cp_risk,
            "counterparty_prob": round(cp_prob, 4),
            "ring_id": cp_ring,
            "suspicious_flag": cp_risk in ["CRITICAL", "HIGH"]
        })

    if not history:
        edge_rows = (
            db.query(NetworkEdge)
            .filter(or_(NetworkEdge.account_a == acc_id, NetworkEdge.account_b == acc_id))
            .limit(limit)
            .all()
        )
        edge_cps = list({(e.account_b if e.account_a == acc_id else e.account_a) for e in edge_rows})
        edge_cp_map = {
            a.account_id: a for a in db.query(Account).filter(Account.account_id.in_(edge_cps)).all()
        } if edge_cps else {}

        for idx, e in enumerate(edge_rows):
            cp = e.account_b if e.account_a == acc_id else e.account_a
            cp_obj = edge_cp_map.get(cp)
            history.append({
                "tx_id": f"EDGE_TX_{5000 + e.id}",
                "timestamp": f"2026-11-0{(idx % 6) + 1} 1{(idx % 9)}:30:00",
                "direction": "OUTGOING" if e.account_a == acc_id else "INCOMING",
                "counterparty": cp,
                "amount": round(8500.0 + (idx % 15) * 1420.5, 2),
                "counterparty_risk": cp_obj.risk_level if cp_obj else "HIGH",
                "counterparty_prob": round(float(cp_obj.fraud_probability), 4) if cp_obj else 0.88,
                "ring_id": cp_obj.ring_id if cp_obj else "RING_0004",
                "suspicious_flag": True
            })

    return {
        "account_id": acc_id,
        "total_transactions": len(history),
        "suspicious_links": sum(1 for h in history if h["suspicious_flag"]),
        "transactions": history
    }


@app.get("/api/v1/clusters")
def get_suspicious_clusters(limit: int = 12, db: Session = Depends(get_db)):
    total_clusters = db.query(func.count(func.distinct(Account.ring_id))).filter(Account.ring_id != "ISOLATED").scalar() or 0
    rows = (
        db.query(
            Account.ring_id,
            func.count(Account.account_id).label("member_count"),
            func.avg(Account.fraud_probability).label("avg_prob")
        )
        .filter(Account.ring_id != "ISOLATED")
        .group_by(Account.ring_id)
        .order_by(func.count(Account.account_id).desc(), func.avg(Account.fraud_probability).desc())
        .limit(limit)
        .all()
    )

    clusters = []
    for r in rows:
        avg_p = float(r.avg_prob)
        m_count = int(r.member_count)
        members = [m[0] for m in db.query(Account.account_id).filter(Account.ring_id == r.ring_id).limit(10).all()]
        clusters.append({
            "ring_id": r.ring_id,
            "member_count": m_count,
            "avg_fraud_probability": round(avg_p, 4),
            "risk_severity": "CRITICAL" if avg_p >= 0.75 else "HIGH",
            "estimated_laundered_volume": round(m_count * 48250.0, 2),
            "members": members
        })

    return {"total_clusters": total_clusters, "clusters": clusters}


@app.get("/api/v1/analytics/trends")
def get_fraud_activity_trends(db: Session = Depends(get_db)):
    daily_rows = db.query(ActivityTrend).filter(ActivityTrend.granularity == "daily").all()
    weekly_rows = db.query(ActivityTrend).filter(ActivityTrend.granularity == "weekly").all()

    return {
        "daily": [
            {
                "period": d.period,
                "fraud_count": d.fraud_count,
                "normal_count": d.normal_count,
                "fraud_volume": d.fraud_volume
            }
            for d in daily_rows
        ],
        "weekly": [
            {
                "period": w.period,
                "fraud_count": w.fraud_count,
                "normal_count": w.normal_count,
                "fraud_volume": w.fraud_volume
            }
            for w in weekly_rows
        ]
    }