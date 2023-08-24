from __future__ import annotations
import torch
from enum import Enum
from typing import Sequence

class QKVLayout(Enum):
    NOT_INTERLEAVED = 0
    QKV_INTERLEAVED = 1
    KV_INTERLEAVED = 2

class BiasType(Enum):
    NO_BIAS = 0
    PRE_SCALE_BIAS = 1
    POST_SCALE_BIAS = 2

class MaskType(Enum):
    NO_MASK = 0
    PADDING_MASK = 1
    CAUSAL_MASK = 2

class FusedAttnBackend(Enum):
    No_Backend = -1
    F16_max512_seqlen = 0
    F16_arbitrary_seqlen = 1
    FP8 = 2

class DType(Enum):
    Byte = 0
    Int32 = 1
    Int64 = 2
    Float32 = 3
    Float16 = 4
    BFloat16 = 5
    Float8E4M3 = 6
    Float8E5M2 = 7

class Tensor:
    dtype: DType
    shape: Sequence[int]
    data: torch.Tensor
    amax: torch.Tensor
    scale: torch.Tensor
    scale_inv: torch.Tensor
    def __init__(
        self,
        dtype: DType,
        data: torch.Tensor,
        amax: torch.Tensor,
        scale: torch.Tensor,
        scale_inv: torch.Tensor,
    ) -> None: ...

def gelu(input: Tensor, output: Tensor) -> None: ...
def dgelu(grad: Tensor, input: Tensor, output: Tensor) -> None: ...
def geglu(input: Tensor, output: Tensor) -> None: ...
def dgeglu(grad: Tensor, input: Tensor, output: Tensor) -> None: ...
def relu(input: Tensor, output: Tensor) -> None: ...
def drelu(grad: Tensor, input: Tensor, output: Tensor) -> None: ...
def swiglu(input: Tensor, output: Tensor) -> None: ...
def dswiglu(grad: Tensor, input: Tensor, output: Tensor) -> None: ...
def reglu(input: Tensor, output: Tensor) -> None: ...
def dreglu(grad: Tensor, input: Tensor, output: Tensor) -> None: ...
def fp8_quantize(input: Tensor, output: Tensor) -> None: ...
def fp8_dequantize(input: Tensor, output: Tensor) -> None: ...
def get_fused_attn_backend(
    q_dtype: DType,
    kv_dtype: DType,
    qkv_layout: QKVLayout,
    bias_type: BiasType,
    attn_mask_type: MaskType,
    dropout: float,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    head_dim: int,
) -> FusedAttnBackend: ...
def fused_attn_fwd_qkvpacked(
    QKV: Tensor,
    Bias: Tensor,
    S: Tensor,
    O: Tensor,
    Aux_CTX_Tensors: Sequence[Tensor],
    cu_seqlens: Tensor,
    rng_state: Tensor,
    max_seqlen: int,
    is_training: bool,
    attn_scale: float,
    dropout: float,
    qkv_layout: QKVLayout,
    bias_type: BiasType,
    attn_mask_type: MaskType,
    workspace: Tensor,
) -> None: ...
def fused_attn_bwd_qkvpacked(
    QKV: Tensor,
    O: Tensor,
    dO: Tensor,
    S: Tensor,
    dP: Tensor,
    Aux_CTX_Tensors: Sequence[Tensor],
    dQKV: Tensor,
    dBias: Tensor,
    cu_seqlens: Tensor,
    max_seqlen: int,
    attn_scale: float,
    dropout: float,
    qkv_layout: QKVLayout,
    bias_type: BiasType,
    attn_mask_type: MaskType,
    workspace: Tensor,
) -> None: ...
def fused_attn_fwd_kvpacked(
    Q: Tensor,
    KV: Tensor,
    Bias: Tensor,
    S: Tensor,
    O: Tensor,
    Aux_CTX_Tensors: Sequence[Tensor],
    cu_seqlens_q: Tensor,
    cu_seqlens_kv: Tensor,
    rng_state: Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    is_training: bool,
    attn_scale: float,
    dropout: float,
    qkv_layout: QKVLayout,
    bias_type: BiasType,
    attn_mask_type: MaskType,
    workspace: Tensor,
) -> None: ...
def fused_attn_bwd_kvpacked(
    Q: Tensor,
    KV: Tensor,
    O: Tensor,
    dO: Tensor,
    S: Tensor,
    dP: Tensor,
    Aux_CTX_Tensors: Sequence[Tensor],
    dQ: Tensor,
    dKV: Tensor,
    dBias: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_kv: Tensor,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    attn_scale: float,
    dropout: float,
    qkv_layout: QKVLayout,
    bias_type: BiasType,
    attn_mask_type: MaskType,
    workspace: Tensor,
) -> None: ...
def cublas_gemm(
    A: Tensor,
    B: Tensor,
    D: Tensor,
    bias: Tensor,
    pre_gelu_out: Tensor,
    transa: bool,
    transb: bool,
    grad: bool,
    workspace: Tensor,
    accumulate: bool,
    use_split_accumulator: bool,
    math_sm_count: int,
) -> None: ...
def layernorm_fwd(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    epsilon: float,
    z: Tensor,
    mu: Tensor,
    rsigma: Tensor,
    multiprocessorCount: int,
    workspace: Tensor,
    barrier: Tensor,
) -> None: ...
def layernorm1p_fwd(
    x: Tensor,
    gamma: Tensor,
    beta: Tensor,
    epsilon: float,
    z: Tensor,
    mu: Tensor,
    rsigma: Tensor,
    multiprocessorCount: int,
    workspace: Tensor,
    barrier: Tensor,
) -> None: ...
def layernorm_bwd(
    dz: Tensor,
    x: Tensor,
    mu: Tensor,
    rsigma: Tensor,
    gamma: Tensor,
    dx: Tensor,
    dgamma: Tensor,
    dbeta: Tensor,
    dgamma_part: Tensor,
    dbeta_part: Tensor,
    multiprocessorCount: int,
    workspace: Tensor,
    barrier: Tensor,
) -> None: ...
def layernorm1p_bwd(
    dz: Tensor,
    x: Tensor,
    mu: Tensor,
    rsigma: Tensor,
    gamma: Tensor,
    dx: Tensor,
    dgamma: Tensor,
    dbeta: Tensor,
    dgamma_part: Tensor,
    dbeta_part: Tensor,
    multiprocessorCount: int,
    workspace: Tensor,
    barrier: Tensor,
) -> None: ...
def rmsnorm_fwd(
    x: Tensor,
    gamma: Tensor,
    epsilon: float,
    z: Tensor,
    rsigma: Tensor,
    multiprocessorCount: int,
    workspace: Tensor,
    barrier: Tensor,
) -> None: ...
def rmsnorm_bwd(
    dz: Tensor,
    x: Tensor,
    rsigma: Tensor,
    gamma: Tensor,
    dx: Tensor,
    dgamma: Tensor,
    dgamma_part: Tensor,
    multiprocessorCount: int,
    workspace: Tensor,
    barrier: Tensor,
) -> None: ...
def scaled_softmax_forward(
    input: Tensor, softmax_results: Tensor, scale_factor: float
) -> None: ...
def scaled_softmax_backward(
    incoming_grads: Tensor,
    softmax_results: Tensor,
    output_grads: Tensor,
    scale_factor: float,
) -> None: ...
def scaled_masked_softmax_forward(
    input: Tensor, mask: Tensor, softmax_results: Tensor, scale_factor: float
) -> None: ...
def scaled_masked_softmax_backward(
    incoming_grads: Tensor,
    softmax_results: Tensor,
    output_grads: Tensor,
    scale_factor: float,
) -> None: ...
def scaled_upper_triang_masked_softmax_forward(
    input: Tensor, softmax_results: Tensor, scale_factor: float
) -> None: ...
def scaled_upper_triang_masked_softmax_backward(
    incoming_grads: Tensor,
    softmax_results: Tensor,
    output_grads: Tensor,
    scale_factor: float,
) -> None: ...
def cast_transpose(
    input: Tensor, cast_output: Tensor, transposed_output: Tensor
) -> None: ...
def transpose(input: Tensor, transposed_output: Tensor) -> None: ...
def cast_transpose_dbias(
    input: Tensor,
    cast_output: Tensor,
    transposed_output: Tensor,
    dbias: Tensor,
    workspace: Tensor,
) -> None: ...
def fp8_transpose_dbias(
    input: Tensor, transposed_output: Tensor, dbias: Tensor, workspace: Tensor
) -> None: ...
def cast_transpose_dbias_dgelu(
    input: Tensor,
    gelu_input: Tensor,
    cast_output: Tensor,
    transposed_output: Tensor,
    dbias: Tensor,
    workspace: Tensor,
) -> None: ...
def dgeglu_cast_transpose(
    input: Tensor, geglu_input: Tensor, cast_output: Tensor, transposed_output: Tensor
) -> None: ...
def multi_cast_transpose(
    input_list: Sequence[Tensor],
    cast_output_list: Sequence[Tensor],
    transposed_output_list: Sequence[Tensor],
) -> None: ...
