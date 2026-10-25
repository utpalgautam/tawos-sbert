"""Shared three-layer CNN regression head over a 384-d SBERT vector."""
import torch
import torch.nn as nn


class SbertCNN(nn.Module):
    def __init__(self, channels=(128, 64, 32), kernel_size=3, padding=1, dropout=0.2, output_dim=1):
        super().__init__()
        c1, c2, c3 = channels
        self.conv1 = nn.Conv1d(1, c1, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size, padding=padding)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(c3, output_dim)

    def forward(self, x):                          # x: (B, 384)
        x = x.unsqueeze(1)                         # (B, 1, 384)
        x = self.pool(torch.relu(self.conv1(x)))   # (B, 128, 192)
        x = self.pool(torch.relu(self.conv2(x)))   # (B, 64, 96)
        x = torch.relu(self.conv3(x))              # (B, 32, 96)
        x = self.gap(x).squeeze(-1)                # (B, 32)
        return self.fc(self.drop(x)).squeeze(-1)   # (B,)


def build_model(cfg):
    c = cfg["cnn"]
    return SbertCNN(c["conv_channels"], c["kernel_size"], c["padding"], c["dropout"], c["output_dim"])
