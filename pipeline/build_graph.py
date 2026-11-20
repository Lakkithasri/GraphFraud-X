import json
import logging
import pickle
from pathlib import Path
import numpy as np  # type: ignore[reportMissingImports]
import pandas as pd  # type: ignore[reportMissingImports, reportMissingModuleSource]
import networkx as nx  # type: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split  # type: ignore[reportMissingModuleSource]
from sklearn.preprocessing import StandardScaler  # type: ignore[reportMissingModuleSource]
import torch  # type: ignore[reportMissingImports]
from torch_geometric.data import Data  # type: ignore[reportMissingImports]

from config.settings import PathConfig, DataConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FinancialGraphBuilder:
    def __init__(self, paths: PathConfig, config: DataConfig):
        self.paths = paths
        self.config = config
        self.scaler = StandardScaler()
        self.account_to_idx = {}

        # Ensure output directories exist
        self.paths.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)

    def load_raw_data(self):
        logger.info(f"Loading raw account profiles from: {self.paths.account_profiles_path}")
        accounts_df = pd.read_csv(self.paths.account_profiles_path)
        logger.info(f"Loading raw network edges from: {self.paths.network_edges_path}")
        edges_df = pd.read_csv(self.paths.network_edges_path)
        return accounts_df, edges_df

    def process_node_features(self, accounts_df: pd.DataFrame):
        # 1. Build index mapping for graph entity reference
        self.account_to_idx = {acc_id: i for i, acc_id in enumerate(accounts_df[self.config.id_column])}
        
        # Save node mapping for backend/API inference lookups
        with open(self.paths.node_mapping_path, "w") as f:
            json.dump(self.account_to_idx, f)
        logger.info(f"Node mapping saved to {self.paths.node_mapping_path}")

        # 2. Drop leakage columns
        feature_cols = [c for c in accounts_df.columns if c not in self.config.leakage_columns]
        logger.info(f"Retained {len(feature_cols)} clean predictive features.")

        # 3. Categorical encoding
        X_df = pd.get_dummies(
            accounts_df[feature_cols],
            columns=self.config.categorical_columns,
            drop_first=True
        )

        # 4. Standardize numerical features and persist scaler
        X_scaled = self.scaler.fit_transform(X_df)
        with open(self.paths.scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        logger.info(f"StandardScaler persisted to {self.paths.scaler_path}")

        x_tensor = torch.tensor(X_scaled, dtype=torch.float)
        y_tensor = torch.tensor(accounts_df[self.config.target_column].values, dtype=torch.long)
        
        return x_tensor, y_tensor, list(X_df.columns)

    def process_graph_topology(self, edges_df: pd.DataFrame):
        # Filter edges where both accounts are present in the node set
        valid_edges = edges_df[
            edges_df["account_a"].isin(self.account_to_idx) &
            edges_df["account_b"].isin(self.account_to_idx)
        ]

        src = [self.account_to_idx[a] for a in valid_edges["account_a"]]
        dst = [self.account_to_idx[b] for b in valid_edges["account_b"]]

        # Symmetrize edges to enable bidirectional GNN message passing
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
        logger.info(f"Constructed undirected edge_index with {edge_index.shape[1]} edges.")
        return edge_index

    def compute_structural_features(self, edges_df: pd.DataFrame) -> pd.DataFrame:
        """Compute label-independent structural features for every account."""
        graph = nx.Graph()
        graph.add_nodes_from(self.account_to_idx)

        valid_edges = edges_df[
            edges_df["account_a"].isin(self.account_to_idx)
            & edges_df["account_b"].isin(self.account_to_idx)
        ]
        graph.add_edges_from(
            valid_edges[["account_a", "account_b"]].itertuples(index=False, name=None)
        )

        degree = dict(graph.degree())
        clustering = nx.clustering(graph)
        pagerank = nx.pagerank(graph, alpha=0.85, max_iter=200)

        # Degree centrality is normalized by graph size and is inexpensive even
        # for the full 50k-node graph. Avoid betweenness centrality here because
        # its exact computation is unnecessarily expensive for this dataset.
        degree_centrality = nx.degree_centrality(graph)
        core_number = nx.core_number(graph)

        # Connection intensity features are label-independent and preserve
        # useful information that would be lost by using only edge existence.
        connection_stats = valid_edges.groupby("account_a").agg(
            unique_connections_a=("account_b", "nunique"),
            mean_connection_count_a=("connection_count", "mean"),
            max_connection_count_a=("connection_count", "max"),
        )
        reverse_stats = valid_edges.groupby("account_b").agg(
            unique_connections_b=("account_a", "nunique"),
            mean_connection_count_b=("connection_count", "mean"),
            max_connection_count_b=("connection_count", "max"),
        )

        connection_stats = connection_stats.join(reverse_stats, how="outer").fillna(0)
        connection_stats["unique_connections"] = (
            connection_stats["unique_connections_a"]
            + connection_stats["unique_connections_b"]
        )
        connection_stats["mean_connection_count"] = connection_stats[
            ["mean_connection_count_a", "mean_connection_count_b"]
        ].replace(0, np.nan).mean(axis=1).fillna(0)
        connection_stats["max_connection_count"] = connection_stats[
            ["max_connection_count_a", "max_connection_count_b"]
        ].max(axis=1)

        components = list(nx.connected_components(graph))
        component_size = {
            node: len(component)
            for component in components
            for node in component
        }
        neighbor_mean_degree = {}
        for node in graph:
            neighbors = list(graph.neighbors(node))
            neighbor_mean_degree[node] = (
                float(np.mean([degree[n] for n in neighbors])) if neighbors else 0.0
            )

        structural = pd.DataFrame({
            self.config.id_column: list(self.account_to_idx.keys()),
            "graph_degree": [degree.get(node, 0) for node in self.account_to_idx],
            "clustering_coefficient": [clustering.get(node, 0.0) for node in self.account_to_idx],
            "pagerank": [pagerank.get(node, 0.0) for node in self.account_to_idx],
            "degree_centrality": [degree_centrality.get(node, 0.0) for node in self.account_to_idx],
            "core_number": [core_number.get(node, 0) for node in self.account_to_idx],
            "component_size": [component_size.get(node, 1) for node in self.account_to_idx],
            "neighbor_mean_degree": [neighbor_mean_degree.get(node, 0.0) for node in self.account_to_idx],
        })
        structural = structural.merge(
            connection_stats[
                ["unique_connections", "mean_connection_count", "max_connection_count"]
            ],
            left_on=self.config.id_column,
            right_index=True,
            how="left",
        ).fillna(0)
        logger.info("Computed graph structural features: graph_degree, clustering_coefficient, "
                    "pagerank, degree_centrality, core_number, component_size, "
                    "neighbor_mean_degree, connection statistics")
        return structural

    def generate_stratified_masks(self, num_nodes: int, labels: np.ndarray):
        indices = np.arange(num_nodes)
        
        # Train / Temp split
        idx_train, idx_temp, _, y_temp = train_test_split(
            indices, labels,
            test_size=(self.config.test_size + self.config.val_size),
            random_state=self.config.random_seed,
            stratify=labels
        )
        
        # Val / Test split
        test_ratio_of_temp = self.config.test_size / (self.config.test_size + self.config.val_size)
        idx_val, idx_test, _, _ = train_test_split(
            idx_temp, y_temp,
            test_size=test_ratio_of_temp,
            random_state=self.config.random_seed,
            stratify=y_temp
        )

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        return train_mask, val_mask, test_mask

    def run(self):
        accounts_df, edges_df = self.load_raw_data()
        
        # --- Compute Graph Topological Structural Features ---
        src_col = edges_df.columns[0]
        dst_col = edges_df.columns[1]
        id_col = self.config.id_column
        
        # Out-degree & In-degree
        out_deg = edges_df[src_col].value_counts().rename("out_degree")
        in_deg = edges_df[dst_col].value_counts().rename("in_degree")
        
        accounts_df = accounts_df.merge(out_deg, left_on=id_col, right_index=True, how="left").fillna({"out_degree": 0})
        accounts_df = accounts_df.merge(in_deg, left_on=id_col, right_index=True, how="left").fillna({"in_degree": 0})
        accounts_df["total_degree"] = accounts_df["out_degree"] + accounts_df["in_degree"]
        accounts_df["log_total_degree"] = np.log1p(accounts_df["total_degree"])

        # These features use only graph structure. Label-derived fields such as
        # both_fraud, fraud_count, and fraud_rate are intentionally excluded.
        structural_df = self.compute_structural_features(edges_df)
        accounts_df = accounts_df.merge(
            structural_df, on=id_col, how="left"
        ).fillna({
            "graph_degree": 0,
            "clustering_coefficient": 0.0,
            "pagerank": 0.0,
            "degree_centrality": 0.0,
            "core_number": 0,
        })
        logger.info("Appended graph topology and structural features")
        
        x, y, feature_names = self.process_node_features(accounts_df)
        edge_index = self.process_graph_topology(edges_df)

        train_mask, val_mask, test_mask = self.generate_stratified_masks(
            len(y), y.numpy()
        )
        graph_data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        graph_data.feature_names = feature_names
        torch.save(graph_data, self.paths.graph_data_path)
        logger.info(
            f"Saved graph with {x.shape[1]} node features to {self.paths.graph_data_path}"
        )


if __name__ == "__main__":
    builder = FinancialGraphBuilder(PathConfig(), DataConfig())
    builder.run()
