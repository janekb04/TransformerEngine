# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
from typing import Optional
import torch
import transformer_engine.pytorch as te
import nvtx


def speedometer(
    module: torch.nn.Module,
    input: torch.Tensor,
    output_grad: torch.Tensor,
    forward_kwargs: dict = {},
    fp8_autocast_kwargs: Optional[dict] = None,
    timing_iters: int = 50,
    warmup_iters: int = 50,
) -> float:
    """Measure average run time for a PyTorch module

    Performs forward and backward passes.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if fp8_autocast_kwargs is None:
        fp8_autocast_kwargs = {"enabled": False}

    def _benchmark(iters: int):
        for i in range(iters):
            with nvtx.annotate(f"iter_{i}"):
                with nvtx.annotate("forward"):
                    with te.fp8_autocast(**fp8_autocast_kwargs):
                        output = module(input, **forward_kwargs)
                with nvtx.annotate("backward"):
                    output.backward(output_grad)

    # Warmup runs
    torch.cuda.synchronize()
    with nvtx.annotate("warmup"):
        _benchmark(warmup_iters)

    # Timing runs
    start.record()
    with nvtx.annotate("timing"):
        _benchmark(timing_iters)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)/timing_iters
    return elapsed_ms

class DotProductAttention(torch.nn.Module):
    """Attention operation in Transformer layer

    Built with plain PyTorch modules.

    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.projection_size = kv_channels * num_attention_heads
        self.hidden_size_per_attention_head = kv_channels
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.dropout = torch.nn.Dropout(attention_dropout)

    def masked_softmax(self, inp: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        if mask is not None:
            inp.masked_fill_(mask, -10000.0)
        return torch.nn.Softmax(dim=-1)(inp)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b = query.size(1)
        np = query.size(2)
        sq = query.size(0)
        sk = key.size(0)
        hn = value.size(3)

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query = query.view(sq, b * np, -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(sk, b * np, -1)

        bmm1 = (
            torch.bmm(query.transpose(0, 1), key.transpose(0, 1).transpose(1, 2)) / self.norm_factor
        )

        # change view to [b, np, sq, sk]
        attention_scores = bmm1.view(b, np, sq, sk)

        attention_probs = self.masked_softmax(attention_scores, attention_mask)

        attention_probs = self.dropout(attention_probs)

        # change view [sk, b * np, hn]
        value = value.view(sk, b * np, -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(b * np, sq, -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(b, np, sq, hn)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        context = context.view(sq, b, self.projection_size)

        return context
