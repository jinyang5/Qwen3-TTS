"""Factorized voice/style/text conditioning modules."""

from __future__ import annotations

import torch
from torch import nn


def _validate_inputs(hidden_states: torch.Tensor, segment_ids: torch.Tensor, hidden_size: int) -> None:
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape [batch, seq_len, hidden_size], got {tuple(hidden_states.shape)}"
        )
    if segment_ids.ndim != 2:
        raise ValueError(f"segment_ids must have shape [batch, seq_len], got {tuple(segment_ids.shape)}")
    if hidden_states.shape[:2] != segment_ids.shape:
        raise ValueError(
            "hidden_states and segment_ids batch/sequence dimensions must match: "
            f"{tuple(hidden_states.shape[:2])} != {tuple(segment_ids.shape)}"
        )
    if hidden_states.shape[-1] != hidden_size:
        raise ValueError(
            f"expected hidden size {hidden_size}, got {hidden_states.shape[-1]}"
        )


class PromptSegmentEmbedding(nn.Module):
    """Add a learned segment/type embedding without changing tensor shape."""

    def __init__(self, hidden_size: int = 2048, num_segments: int = 4):
        super().__init__()
        if hidden_size <= 0 or num_segments < 4:
            raise ValueError("hidden_size must be positive and num_segments must be at least 4")
        self.hidden_size = hidden_size
        self.num_segments = num_segments
        self.embedding = nn.Embedding(num_segments, hidden_size)
        self.reset_identity_parameters()

    def reset_identity_parameters(self) -> None:
        nn.init.zeros_(self.embedding.weight)

    def forward(self, hidden_states: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
        _validate_inputs(hidden_states, segment_ids, self.hidden_size)
        if segment_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("segment_ids must use an integer dtype")
        if segment_ids.numel() and (
            int(segment_ids.min()) < 0 or int(segment_ids.max()) >= self.num_segments
        ):
            raise ValueError(f"segment_ids must be in [0, {self.num_segments - 1}]")
        return hidden_states + self.embedding(segment_ids.to(hidden_states.device)).to(hidden_states.dtype)


class _ResidualBottleneckAdapter(nn.Module):
    def __init__(self, hidden_size: int, bottleneck_size: int):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.activation = nn.SiLU()
        self.up = nn.Linear(bottleneck_size, hidden_size)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(hidden_states)))


class VoiceStyleConditioningAdapter(nn.Module):
    """Apply independent near-identity residual adapters by prompt segment."""

    def __init__(
        self,
        hidden_size: int = 2048,
        bottleneck_size: int = 256,
        adapt_text: bool = False,
    ):
        super().__init__()
        if hidden_size <= 0 or bottleneck_size <= 0:
            raise ValueError("hidden_size and bottleneck_size must be positive")
        self.hidden_size = hidden_size
        self.bottleneck_size = bottleneck_size
        self.adapt_text = adapt_text
        self.voice_adapter = _ResidualBottleneckAdapter(hidden_size, bottleneck_size)
        self.style_adapter = _ResidualBottleneckAdapter(hidden_size, bottleneck_size)
        self.text_adapter = (
            _ResidualBottleneckAdapter(hidden_size, bottleneck_size) if adapt_text else None
        )
        self.reset_identity_parameters()

    def reset_identity_parameters(self) -> None:
        for adapter in (self.voice_adapter, self.style_adapter, self.text_adapter):
            if adapter is not None:
                nn.init.zeros_(adapter.up.weight)
                nn.init.zeros_(adapter.up.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        segment_ids: torch.Tensor,
        enabled: bool = True,
    ) -> torch.Tensor:
        _validate_inputs(hidden_states, segment_ids, self.hidden_size)
        if not enabled:
            return hidden_states
        if segment_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("segment_ids must use an integer dtype")
        segment_ids = segment_ids.to(hidden_states.device)
        residual = torch.zeros_like(hidden_states)
        for segment_id, adapter in (
            (1, self.voice_adapter),
            (2, self.style_adapter),
            (3, self.text_adapter),
        ):
            if adapter is None:
                continue
            mask = (segment_ids == segment_id).unsqueeze(-1).to(hidden_states.dtype)
            residual = residual + adapter(hidden_states) * mask
        return hidden_states + residual


__all__ = ["PromptSegmentEmbedding", "VoiceStyleConditioningAdapter"]
