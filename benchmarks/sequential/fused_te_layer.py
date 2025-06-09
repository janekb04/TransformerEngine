# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
import transformer_engine.pytorch as te
import utils

class FusedTETransformerLayer(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_eps: float = 1e-5,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = hidden_size // num_attention_heads
        self.ln_qkv = te.LayerNormLinear(hidden_size, 3 * hidden_size, eps=layernorm_eps, bias=True)
        self.attention = utils.DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=self.kv_channels,
            attention_dropout=attention_dropout,
        )
        self.projection = te.Linear(hidden_size, hidden_size, bias=True)
        self.dropout = torch.nn.Dropout(hidden_dropout)
        self.ln_mlp = te.LayerNormMLP(hidden_size, ffn_hidden_size, eps=layernorm_eps, bias=True)
        
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: torch.Tensor
    ):
        res = x
        qkv = self.ln_qkv(x)
        
        # Split qkv into query, key and value
        qkv = qkv.view(qkv.size(0), qkv.size(1), self.num_attention_heads, 3 * self.kv_channels)
        q, k, v = torch.split(qkv, qkv.size(3) // 3, dim=3)
        
        x = self.attention(q, k, v, attention_mask)
        x = self.projection(x)
        x = self.dropout(x)
        x = res + x
        res = x
        x = self.ln_mlp(x)
        
        return x + res