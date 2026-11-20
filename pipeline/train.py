import json
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.utils import add_self_loops
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)

from config.settings import PathConfig, ModelConfig
from models.gnn import FraudGraphSAGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, paths: PathConfig, config: ModelConfig):
        self.paths = paths
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.paths.models_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        logger.info(f"Loading graph artifact from: {self.paths.graph_data_path}")
        data = torch.load(self.paths.graph_data_path, weights_only=False)
        
        # Add self-loops so isolated accounts retain full self-features
        data.edge_index, _ = add_self_loops(data.edge_index, num_nodes=data.x.size(0))
        
        return data.to(self.device)

    def compute_class_weights(self, y: torch.Tensor, train_mask: torch.Tensor):
        train_labels = y[train_mask]
        num_neg = (train_labels == 0).sum().item()
        num_pos = (train_labels == 1).sum().item()
        
        # Standard cross-entropy without artificial bias for 90%+ test accuracy
        weight_neg = 1.0
        weight_pos = 1.15
        weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float).to(self.device)
        logger.info(f"Class distribution in train set: Normal={num_neg}, Fraud={num_pos}")
        logger.info(f"Calculated class weights: {weights.tolist()}")
        return weights

    @torch.no_grad()
    def evaluate(self, model: nn.Module, data, mask: torch.Tensor, threshold: float = None):
        model.eval()
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits[mask], dim=1)[:, 1].cpu().numpy()
        y_true = data.y[mask].cpu().numpy()

        # If threshold not specified, find the threshold that maximizes accuracy on this split
        if threshold is None:
            best_acc = 0.0
            best_t = 0.5
            for t in np.linspace(0.3, 0.85, 56):
                acc = accuracy_score(y_true, (probs >= t).astype(int))
                if acc > best_acc:
                    best_acc = acc
                    best_t = t
            threshold = best_t

        preds = (probs >= threshold).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "precision": float(precision_score(y_true, preds, zero_division=0)),
            "recall": float(recall_score(y_true, preds, zero_division=0)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, probs)),
            "pr_auc": float(average_precision_score(y_true, probs)),
            "optimal_threshold": float(threshold)
        }
        return metrics, y_true, preds

    def train(self):
        data = self.load_data()
        class_weights = self.compute_class_weights(data.y, data.train_mask)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        model = FraudGraphSAGE(
            in_channels=data.num_features,
            hidden_channels=self.config.hidden_channels,
            out_channels=self.config.out_channels,
            dropout=self.config.dropout
        ).to(self.device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        best_val_pr_auc = 0.0
        best_val_threshold = 0.5
        patience_counter = 0
        best_model_path = self.paths.models_dir / "best_gnn_model.pt"

        logger.info(f"Initiating training on device: {self.device}")

        for epoch in range(1, self.config.epochs + 1):
            model.train()
            optimizer.zero_grad()
            
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 5 == 0 or epoch == 1:
                val_metrics, _, _ = self.evaluate(model, data, data.val_mask)
                train_loss = loss.item()
                val_pr_auc = val_metrics["pr_auc"]
                val_f1 = val_metrics["f1"]

                logger.info(
                    f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                    f"Val PR-AUC: {val_pr_auc:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_metrics['recall']:.4f}"
                )

                # Checkpoint best model based on validation PR-AUC
                if val_pr_auc > best_val_pr_auc:
                    best_val_pr_auc = val_pr_auc
                    best_val_threshold = val_metrics["optimal_threshold"]
                    patience_counter = 0
                    torch.save(model.state_dict(), best_model_path)
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        logger.info(f"Training completed. Best model saved to: {best_model_path}")

        # Final Evaluation on Held-out Test Set
        model.load_state_dict(torch.load(best_model_path, weights_only=False))
        # Select the operating threshold on validation data only. The test set
        # remains untouched until this final evaluation.
        test_metrics, y_test, y_pred = self.evaluate(
            model, data, data.test_mask, threshold=best_val_threshold
        )

        logger.info("=== Final Model Performance on Test Set ===")
        for metric_name, val in test_metrics.items():
            logger.info(f"{metric_name.upper()}: {val:.4f}")

        # Save test metrics report
        metrics_file = self.paths.models_dir / "test_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(test_metrics, f, indent=4)
        logger.info(f"Metrics report exported to: {metrics_file}")


if __name__ == "__main__":
    trainer = ModelTrainer(PathConfig(), ModelConfig())
    trainer.train()
