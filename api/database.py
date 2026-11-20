from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.orm import declarative_base, sessionmaker

DB_DIR = Path("data")
DB_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DB_DIR / 'graphfraud.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Account(Base):
    __tablename__ = "accounts"

    account_id = Column(String, primary_key=True, index=True)
    fraud_probability = Column(Float, nullable=False, default=0.0)
    anomaly_score = Column(Float, nullable=False, default=0.0)
    risk_level = Column(String, index=True, nullable=False, default="LOW")
    ring_id = Column(String, index=True, nullable=False, default="ISOLATED")
    ring_size = Column(Integer, nullable=False, default=1)
    action_recommended = Column(String, nullable=True)


class NetworkEdge(Base):
    __tablename__ = "network_edges"

    id = Column(Integer, primary_key=True, autoincrement=True)
    account_a = Column(String, index=True, nullable=False)
    account_b = Column(String, index=True, nullable=False)
    shared_type = Column(String, default="transaction")
    connection_count = Column(Integer, default=1)


class TransactionRecord(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tx_id = Column(String, index=True, nullable=False)
    source = Column(String, index=True, nullable=False)
    target = Column(String, index=True, nullable=False)
    amount = Column(Float, nullable=False, default=0.0)
    timestamp = Column(String, nullable=False)
    is_suspicious = Column(Boolean, default=False)


class ActivityTrend(Base):
    __tablename__ = "activity_trends"

    id = Column(Integer, primary_key=True, autoincrement=True)
    granularity = Column(String, index=True, nullable=False)  # 'daily' or 'weekly'
    period = Column(String, nullable=False)
    fraud_count = Column(Integer, nullable=False, default=0)
    normal_count = Column(Integer, nullable=False, default=0)
    fraud_volume = Column(Float, nullable=False, default=0.0)


def init_tables():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
