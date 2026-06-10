"""DouZero neural network architecture (placeholder)."""

import torch
import torch.nn as nn


class ShengJiNet(nn.Module):
    """Neural network for Sheng Ji (拖拉机) game (TODO: implement)."""

    def __init__(self, role: str = "farmer", input_dim: int = 567):
        super().__init__()
        self.role = role
        # TODO: Implement architecture based on DouZero paper
        # Suggested structure:
        # - Embedding layer for card representations
        # - ResNet blocks for feature extraction
        # - Policy and value heads
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returns action logits."""
        return self.net(x)
