import torch
import torch.nn as nn

class SBERT_CNN(nn.Module):
    """
    Three-layer 1D CNN regression architecture operating over SBERT sentence embeddings (dim=384).
    Follows Section 8 of COMMON_PIPELINE.md:
      Input: (B, 384) -> reshaped to (B, 1, 384)
      Conv1D 1: 1 -> 128 channels, kernel=3, padding=1 -> ReLU -> MaxPool1d(2) -> (B, 128, 192)
      Conv1D 2: 128 -> 64 channels, kernel=3, padding=1 -> ReLU -> MaxPool1d(2) -> (B, 64, 96)
      Conv1D 3: 64 -> 32 channels, kernel=3, padding=1 -> ReLU -> (B, 32, 96)
      AdaptiveAvgPool1d(1) -> (B, 32, 1) -> Flatten -> (B, 32)
      Dropout: p=0.2
      Linear: 32 -> 1 -> (B, 1)
    """
    def __init__(self, in_features: int = 384, dropout: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.conv_block = nn.Sequential(
            # Layer 1
            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (B, 128, 192)
            # Layer 2
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # (B, 64, 96)
            # Layer 3
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),  # (B, 32, 96)
            # Pooling & Flatten
            nn.AdaptiveAvgPool1d(1),  # (B, 32, 1)
            nn.Flatten(),  # (B, 32)
            nn.Dropout(p=dropout),
            nn.Linear(32, 1)  # (B, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 384) or (B, 1, 384)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, 384)
        out = self.conv_block(x)  # (B, 1)
        return out.squeeze(-1)  # (B,)
