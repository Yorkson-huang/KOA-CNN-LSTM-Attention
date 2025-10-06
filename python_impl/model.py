"""Neural network modules for the KOA-CNN-LSTM-Attention Python port."""
from __future__ import annotations

import math

import torch
from torch import nn


class SelfAttentionBlock(nn.Module):
    """Single-head self-attention emulating MATLAB's selfAttentionLayer."""

    def __init__(self, embed_dim: int, key_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.inner_dim = num_heads * key_dim

        self.query = nn.Linear(embed_dim, self.inner_dim)
        self.key = nn.Linear(embed_dim, self.inner_dim)
        self.value = nn.Linear(embed_dim, self.inner_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = inputs.shape
        head_dim = self.key_dim

        def _reshape(tensor: torch.Tensor) -> torch.Tensor:
            return (
                tensor.view(batch, seq_len, self.num_heads, head_dim)
                .transpose(1, 2)
                .contiguous()
            )

        q = _reshape(self.query(inputs))
        k = _reshape(self.key(inputs))
        v = _reshape(self.value(inputs))

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.inner_dim)
        return context


class KOACNNLSTMAttention(nn.Module):
    """CNN + LSTM + Attention network used inside the KOA optimizer."""

    def __init__(self, kernel_size: int, num_neurons: int) -> None:
        super().__init__()
        padding = max(kernel_size // 2, 0)

        self.conv = nn.Conv2d(1, 3, kernel_size, padding=padding)
        self.batch_norm = nn.BatchNorm2d(3, momentum=0.9)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)

        self.flattened_size = 3 * math.ceil(18 / 3) * math.ceil(24 / 3)
        self.lstm = nn.LSTM(
            input_size=self.flattened_size,
            hidden_size=num_neurons,
            batch_first=True,
        )
        self.attention = SelfAttentionBlock(embed_dim=num_neurons, key_dim=24, num_heads=1)
        self.fc = nn.Linear(24, 24)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, 18, 24)
        x = inputs.unsqueeze(1)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.pool(x)

        batch = x.size(0)
        x = x.view(batch, 1, -1)

        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        output = self.fc(attn_out[:, -1, :])
        return output


__all__ = [
    "KOACNNLSTMAttention",
]

