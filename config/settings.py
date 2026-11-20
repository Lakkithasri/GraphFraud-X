from pathlib import Path
from dataclasses import dataclass, field
from typing import List

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class PathConfig:
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    models_dir: Path = BASE_DIR / "saved_models"
    
    account_profiles_path: Path = raw_data_dir / "account_profiles.csv"
    network_edges_path: Path = raw_data_dir / "network_edges.csv"
    graph_data_path: Path = processed_data_dir / "graph_data.pt"
    scaler_path: Path = processed_data_dir / "scaler.pkl"
    node_mapping_path: Path = processed_data_dir / "node_mapping.json"

@dataclass
class DataConfig:
    target_column: str = "is_fraudster"
    id_column: str = "account_id"
    # Strict leakage prevention: these cannot be seen prior to label resolution
    leakage_columns: List[str] = field(default_factory=lambda: [
        "account_id",
        "fraud_count",
        "fraud_amount",
        "fraud_rate",
        "is_fraudster"
    ])
    categorical_columns: List[str] = field(default_factory=lambda: [
        "home_country",
        "account_type"
    ])
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = 42

@dataclass
class ModelConfig:
    hidden_channels: int = 128
    out_channels: int = 2
    num_layers: int = 2
    dropout: float = 0.3
    learning_rate: float = 0.005
    weight_decay: float = 5e-4
    epochs: int = 200
    early_stopping_patience: int = 20