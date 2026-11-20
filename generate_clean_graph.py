import json
import logging
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def build_review_ready_graph(
    num_normal: int = 6800,
    num_fraud: int = 1200,
    output_path: str = "data/processed/graph_data.pt",
    seed: int = 42
):
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    total_nodes = num_normal + num_fraud
    logger.info(f"Generating realistic graph: Total {total_nodes} nodes (Normal={num_normal}, Fraud={num_fraud})")
    
    # 1. Base Features (Shared financial baseline)
    X = np.random.normal(loc=0.0, scale=1.0, size=(total_nodes, 18))
    
    # Fraud nodes get realistic behavioral signal
    fraud_indices = np.arange(num_normal, total_nodes)
    X[fraud_indices, 0] += 0.95  # Velocity
    X[fraud_indices, 3] += 1.10  # Amount burst
    X[fraud_indices, 7] += 0.85  # Risk score
    
    # 2. Add realistic label overlap (fraudsters looking normal, merchants looking suspicious)
    # 6% of normal nodes behave like fraud (high frequency business accounts)
    false_alarm_norm = np.random.choice(num_normal, size=int(0.06 * num_normal), replace=False)
    X[false_alarm_norm, 0] += 0.90
    X[false_alarm_norm, 3] += 1.05
    
    # 8% of fraud nodes mimic pure normal users (low amount sleepers)
    camo_fraud = np.random.choice(fraud_indices, size=int(0.08 * num_fraud), replace=False)
    X[camo_fraud] = np.random.normal(loc=0.0, scale=0.9, size=(len(camo_fraud), 18))
    
    scaler = StandardScaler()
    x = torch.tensor(scaler.fit_transform(X), dtype=torch.float)
    y = torch.tensor([0] * num_normal + [1] * num_fraud, dtype=torch.long)
    
    # 3. Topology: Connected transactions with cross-edges
    edges = []
    # Normal P2P
    for i in range(num_normal):
        k = np.random.randint(2, 5)
        neighbors = np.random.choice(num_normal, size=k, replace=False)
        for n in neighbors:
            if i != n:
                edges.append((i, n))
                
    # Fraud Rings
    ring_size = 10
    num_rings = num_fraud // ring_size
    for r in range(num_rings):
        base = num_normal + r * ring_size
        for j in range(ring_size):
            u = base + j
            v = base + ((j + 1) % ring_size)
            edges.append((u, v))
            edges.append((v, u))
            
    # Bridge transactions (Normal users transacting with Fraud accounts)
    for _ in range(500):
        u = np.random.choice(fraud_indices)
        v = np.random.randint(0, num_normal)
        edges.append((u, v))
        edges.append((v, u))
        
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    edge_index = torch.unique(edge_index, dim=1)
    
    # 4. Stratified Splits (70% Train, 15% Val, 15% Test)
    normal_idx = np.arange(num_normal)
    fraud_idx = np.arange(num_normal, total_nodes)
    np.random.shuffle(normal_idx)
    np.random.shuffle(fraud_idx)
    
    n_tr, n_va = int(0.70 * num_normal), int(0.15 * num_normal)
    f_tr, f_va = int(0.70 * num_fraud), int(0.15 * num_fraud)
    
    train_idx = np.concatenate([normal_idx[:n_tr], fraud_idx[:f_tr]])
    val_idx = np.concatenate([normal_idx[n_tr:n_tr+n_va], fraud_idx[f_tr:f_tr+f_va]])
    test_idx = np.concatenate([normal_idx[n_tr+n_va:], fraud_idx[f_tr+f_va:]])
    
    train_mask = torch.zeros(total_nodes, dtype=torch.bool)
    val_mask = torch.zeros(total_nodes, dtype=torch.bool)
    test_mask = torch.zeros(total_nodes, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    graph_data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph_data, output_path)
    logger.info(f"Graph artifact saved to: {output_path}")

if __name__ == "__main__":
    build_review_ready_graph()