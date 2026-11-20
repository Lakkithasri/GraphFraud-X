import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FraudGraphSAGE(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        
        # Linear projection to preserve raw node features for isolated accounts
        self.skip1 = nn.Linear(in_channels, hidden_channels)
        
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.norm2 = nn.LayerNorm(hidden_channels)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, out_channels)
        )
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1 + Residual Skip
        h = self.conv1(x, edge_index) + self.skip1(x)
        h = self.norm1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # Layer 2 + Residual Skip
        h = self.conv2(h, edge_index) + h
        h = self.norm2(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        return self.classifier(h)