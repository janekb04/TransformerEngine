# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import transformer_engine.pytorch as te
import torch
from torch import nn
from fused_te_layer import FusedTETransformerLayer
from sequential_te_layer import SequentialTETransformerLayer
from utils import speedometer

# Configuration
HIDDEN_SIZE = 4096
SEQUENCE_LENGTH = 2048
BATCH_SIZE = 4
FFN_HIDDEN_SIZE = 16384
NUM_ATTENTION_HEADS = 32
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")

# Transformer layers to compare
fused_te_transformer_layer = FusedTETransformerLayer(
    HIDDEN_SIZE,
    FFN_HIDDEN_SIZE,
    NUM_ATTENTION_HEADS
)
fused_te_transformer_layer.to(dtype=DTYPE).cuda()

sequential_te_transformer_layer = SequentialTETransformerLayer(
    HIDDEN_SIZE,
    FFN_HIDDEN_SIZE,
    NUM_ATTENTION_HEADS    
)
sequential_te_transformer_layer.to(dtype=DTYPE).cuda()

# Synthetic data
x = torch.rand(SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE).cuda().to(dtype=DTYPE)
dy = torch.rand(SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE).cuda().to(dtype=DTYPE)

# Test layers
TEST_ITERS = 10
for i in range(TEST_ITERS):
    print()
    print(f"Test iter {i}")
    print("---------------")
    print("Fused TE Layer")
    fused_te_mean_ms = speedometer(
        fused_te_transformer_layer,
        x,
        dy,
        forward_kwargs = { "attention_mask": None },
    )
    print(f"Mean time: {fused_te_mean_ms:.2f} ms")

    print("Sequential TE Layer")
    sequential_te_mean_ms = speedometer(
        sequential_te_transformer_layer,
        x,
        dy,
        forward_kwargs = { "attention_mask": None },
    )
    print(f"Mean time: {sequential_te_mean_ms:.2f} ms")

    time_percent_difference = (fused_te_mean_ms / sequential_te_mean_ms - 1) * 100
    print()
    print(f"Sequential TE Layer ran {abs(time_percent_difference):.2f}% {"slower" if time_percent_difference < 0 else "faster"} than Fused TE Layer")