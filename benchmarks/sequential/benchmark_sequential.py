# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
from torch import nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling
from fused_te_layer import FusedTETransformerLayer
from sequential_te_layer import SequentialTETransformerLayer
from utils import speedometer
import nvtx

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

builtin_te_transformer_layer = te.TransformerLayer(
    HIDDEN_SIZE,
    FFN_HIDDEN_SIZE,
    NUM_ATTENTION_HEADS
)
builtin_te_transformer_layer.to(dtype=DTYPE).cuda()

# Synthetic data
x = torch.rand(SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE).cuda().to(dtype=DTYPE)
dy = torch.rand(SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE).cuda().to(dtype=DTYPE)

# Setup fp8 autocast kwargs
fp8_format = Format.HYBRID
fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
fp8_autocast_kwargs = { "enabled": True, "fp8_recipe": fp8_recipe }

# Test layers
def _test_layer(layer: nn.Module, name: str, fp8_autocast_kwargs: dict | None):
    print(name)
    with nvtx.annotate(name):
        mean_ms = speedometer(
            layer,
            x,
            dy,
            { "attention_mask": None },
            fp8_autocast_kwargs
        )
    print(f"Mean time: {mean_ms:.2f} ms")

def test_layer_with_without_fp8(layer: nn.Module, name: str):
    _test_layer(layer, name, None)
    _test_layer(layer, name + " (FP8)", fp8_autocast_kwargs)

test_layer_with_without_fp8(fused_te_transformer_layer, "Fused TE Layer")
test_layer_with_without_fp8(sequential_te_transformer_layer, "Sequential TE Layer")
test_layer_with_without_fp8(builtin_te_transformer_layer, "Builtin TE TransformerLayer")