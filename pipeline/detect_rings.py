import json
import logging
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import torch

from config.settings import PathConfig, ModelConfig
from models.gnn import FraudGraphSAGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FraudRingDetector:
    def __init__(self, paths: PathConfig, model_config: ModelConfig):
        self.paths = paths
        self.model_config = model_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self):
        # 1. Load artifacts
        logger.info("Loading graph artifact, model weights, and node mappings...")
        data = torch.load(self.paths.graph_data_path, weights_only=False).to(self.device)
        with open(self.paths.node_mapping_path, "r") as f:
            account_to_idx = json.load(f)
        idx_to_account = {v: k for k, v in account_to_idx.items()}

        # 2. Load trained GNN model
        model = FraudGraphSAGE(
            in_channels=data.num_features,
            hidden_channels=self.model_config.hidden_channels,
            out_channels=self.model_config.out_channels,
            dropout=self.model_config.dropout
        ).to(self.device)
        
        best_model_path = self.paths.models_dir / "best_gnn_model.pt"
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        model.eval()

        # 3. Generate model probability predictions
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

        # 4. Construct NetworkX graph for topology and community detection
        logger.info("Constructing graph topology for ring analysis...")
        G = nx.Graph()
        edge_index_cpu = data.edge_index.cpu().numpy()
        edges = list(zip(edge_index_cpu[0], edge_index_cpu[1]))
        G.add_edges_from(edges)

        # 5. Connected components analysis (Coordinated Rings)
        connected_components = list(nx.connected_components(G))
        node_to_ring = {}
        node_to_ring_size = {}

        for ring_idx, comp in enumerate(connected_components):
            comp_size = len(comp)
            ring_name = f"RING_{ring_idx + 1:04d}" if comp_size > 1 else "ISOLATED"
            for node_id in comp:
                node_to_ring[node_id] = ring_name
                node_to_ring_size[node_id] = comp_size

        # 6. Build Consolidated Account Risk Intelligence Table
        records = []
        for idx in range(data.num_nodes):
            acc_id = idx_to_account.get(idx, f"ACC_{idx}")
            fraud_prob = float(probs[idx])
            ring_id = node_to_ring.get(idx, "ISOLATED")
            ring_size = node_to_ring_size.get(idx, 1)

            # Composite Anomaly Score: Blends GNN representation probability with topological ring exposure
            ring_factor = min(ring_size / 10.0, 1.0)
            anomaly_score = float(0.7 * fraud_prob + 0.3 * ring_factor)

            # Assign categorical risk levels
            if anomaly_score >= 0.75:
                risk_level = "CRITICAL"
            elif anomaly_score >= 0.50:
                risk_level = "HIGH"
            elif anomaly_score >= 0.25:
                risk_level = "MEDIUM"
            else:
                risk_level = "LOW"

            records.append({
                "account_id": acc_id,
                "node_idx": idx,
                "fraud_probability": round(fraud_prob, 4),
                "anomaly_score": round(anomaly_score, 4),
                "risk_level": risk_level,
                "ring_id": ring_id,
                "ring_size": ring_size
            })

        df_risk = pd.DataFrame(records)
        
        # 7. Persist intelligence artifacts
        output_csv = self.paths.processed_data_dir / "account_risk_intelligence.csv"
        df_risk.to_csv(output_csv, index=False)
        logger.info(f"Risk intelligence dataset saved to: {output_csv}")

        # Summary statistics
        logger.info("=== Anomaly Risk Breakdown ===")
        logger.info(df_risk["risk_level"].value_counts().to_string())
        
        num_rings = sum(1 for c in connected_components if len(c) > 1)
        logger.info(f"Detected {num_rings} multi-account coordinated fraud rings.")


if __name__ == "__main__":
    detector = FraudRingDetector(PathConfig(), ModelConfig())
    detector.run()