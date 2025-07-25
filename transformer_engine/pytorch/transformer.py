# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer."""
import os
import warnings
from contextlib import nullcontext
from typing import Callable, List, Optional, Tuple, Union

import torch

from transformer_engine.pytorch import torch_version
from transformer_engine.pytorch.module import LayerNormMLP, LayerNorm, RMSNorm
from transformer_engine.debug.pytorch.debug_state import TEDebugState
from transformer_engine.pytorch.attention.multi_head_attention import MultiheadAttention
from transformer_engine.pytorch.attention.inference import InferenceParams
from transformer_engine.pytorch.jit import (
    set_jit_fusion_options,
    warmup_jit_bias_dropout_add_all_dtypes,
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from transformer_engine.pytorch.utils import (
    cast_if_needed,
    get_default_init_method,
    torch_get_autocast_gpu_dtype,
)
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    LayerTypes,
    dist_group_type,
)
from transformer_engine.pytorch.distributed import get_distributed_world_size
from transformer_engine.pytorch.export import is_in_onnx_export_mode
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule


warnings.filterwarnings("module", category=DeprecationWarning, module="transformer")


__all__ = ["TransformerLayer"]


class DropPath(torch.nn.Module):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """DropPath FWD"""

        if self.drop_prob == 0.0 or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_state.shape[0],) + (1,) * (hidden_state.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=hidden_state.dtype, device=hidden_state.device
        )
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output


class TransformerLayer(torch.nn.Module):
    r"""
    TransformerLayer is made up of an attention block and a feedforward network (MLP).
    This standard layer is based on the paper "Attention Is All You Need".

    .. note::

        Argument :attr:`attention_mask` in the `forward` call is only used when
        :attr:`self_attn_mask_type` includes `"padding"` or `"arbitrary"`.

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    ffn_hidden_size : int
                     intermediate size to which input samples are projected.
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    num_gqa_groups : int, default = `None`
                         number of GQA groups in the transformer layer.
                         Grouped Query Attention is described in
                         `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                         This only affects the keys and values, not the querys.
                         GQA-1 is equivalent to Multi-Query Attention
                         (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                         is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    layernorm_epsilon : float, default = 1e-5
                       a value added to the denominator of layer normalization
                       for numerical stability.
    hidden_dropout: float, default = 0.1
                   dropout probability for the dropout op after FC2 layer.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    init_method : Callable, default = `None`
                 used for initializing weights of QKV and FC1 weights in the following way:
                 `init_method(weight)`. When set to `None`, defaults to
                 `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
                              used for initializing weights of PROJ and FC2 in the following way:
                              `output_layer_init_method(weight)`. When set to `None`, defaults to
                              `torch.nn.init.normal_(mean=0.0, std=0.023)`.
    apply_residual_connection_post_layernorm : bool, default = `False`
                                              if set to `True`, residual connections are taken
                                              from the output of layer norm (default is taken
                                              from input of layer norm)
    layer_number: int, default = `None`
                 layer number of the current `TransformerLayer` when multiple such modules are
                 concatenated to form a transformer block.
    output_layernorm: bool, default = `False`
                     if set to `True`, layer normalization is applied on the output side,
                     after the final dropout-add. default behavior is to apply layer
                     normalization on the input side, before the QKV transformation.
    parallel_attention_mlp: bool, default = `False`
                           if set to `True`, self-attention and feedforward network are computed
                           based on the same input (in parallel) instead of sequentially.
                           Both blocks have an independent normalization.
                           This architecture is used in `Falcon` models.
    layer_type: {'encoder', 'decoder'}, default = `encoder`
               if set to `decoder`, an additional cross-attn block is added after self-attn.
               This can be used for structures like `T5` Transformer in conjunction with the
               `encoder` option.
    kv_channels: int, default = `None`
                number of query-key-value channels per attention head. defaults to
                :attr:`hidden_size` / :attr:`num_attention_heads` if `None`.
    self_attn_mask_type: {'no_mask', 'padding', 'causal', 'padding_causal', 'causal_bottom_right',
                        'padding_causal_bottom_right', 'arbitrary'},
                        default = `causal`
                        type of attention mask passed into softmax operation for encoder.
                        Overridden by :attr:`self_attn_mask_type` in the `forward` method.
                        The forward arg is useful for dynamically changing mask types, e.g.
                        a different mask for training and inference. The init arg is useful
                        for cases involving compilation/tracing, e.g. ONNX export.
    window_size: Optional[Tuple[int, int]], default = `None`
                sliding window size for local attention in encoder, where query at position i
                attends to keys in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k
                - seqlen_q + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean
                no sliding window and causal mask specifically. Both `causal` and
                `causal_bottom_right` masks map to `window_size = (-1, 0)` and Transformer Engine
                distinguishes them based on `self_attn_mask_type` or `enc_dec_attn_mask_type`.
                Similar to :attr:`self_attn_mask_type`, `window_size` can be overridden by
                :attr:`window_size` in `forward` as well.
    enc_dec_attn_mask_type: {'no_mask', 'causal', 'padding', 'padding_causal', 'arbitrary'},
                           default = `no_mask`
                           type of attention mask passed into softmax operation for decoder.
    enc_dec_window_size: Optional[Tuple[int, int]], default = `None`
                        sliding window size for local attention in decoder.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    qkv_weight_interleaved : bool, default = `True`
                            if set to `False`, the QKV weight is interpreted as a concatenation of
                            query, key, and value weights along the `0th` dimension. The default
                            interpretation is that the individual `q`, `k`, and `v` weights for each
                            attention head are interleaved. This parameter is set to `False` when
                            using :attr:`fuse_qkv_params=False`.
    rotary_pos_interleaved : bool, default = `False`
                            whether to use interleaved rotary position embeddings.
    bias : bool, default = `True`
          if set to `False`, the transformer layer will not learn any additive biases.
    activation : str, default = 'gelu'
          Type of activation used in MLP block.
          Options are: 'gelu', 'relu', 'reglu', 'geglu', 'swiglu', 'qgelu' and 'srelu'.
    device : Union[torch.device, str], default = "cuda"
          The device on which the parameters of the model will be allocated. It is the user's
          responsibility to ensure all parameters are moved to the GPU before running the
          forward pass.
    attn_input_format: {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
                         This controls whether the dimensions of the
                         intermediate hidden states is 'sequence first' ('sbhd'), 'batch first' ('bshd'),
                         or 'token first' ('thd'). `s` stands for the sequence length, `b` batch size,
                         `t` the total number of tokens, `h` the number of heads, `d` head size.
                         Note that these formats are very closely
                         related to the `qkv_format` in the `MultiHeadAttention`
                         and `DotProductAttention` modules.
    name: str, default = `None`
        name of the module, currently used for debugging purposes.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, QKV and FC1 layers are used as Column Parallel
                      whereas PROJ and FC2 is used as Row Parallel as described
                      `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    tp_size : int, default = 1
             used as TP (tensor parallel) world size when TP groups are not formed during
             initialization. In this case, users must call the
             `set_tensor_parallel_group(tp_group)` method on the initialized module before the
             forward pass to supply the tensor parallel group needed for tensor and sequence
             parallel collectives.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.
    params_dtype : torch.dtype, default = `torch.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    seq_length: int
               sequence length of input samples. Needed for JIT Warmup, a technique where jit
               fused functions are warmed up before training to ensure same kernels are used for
               forward propogation and activation recompute phase.
    micro_batch_size: int
                     batch size per training step. Needed for JIT Warmup, a technique where jit
                     fused functions are warmed up before training to ensure same kernels are
                     used for forward propogation and activation recompute phase.
    drop_path_rate: float, default = 0.0
                   when > 0.0, applies stochastic depth per sample in
                   the main path of the residual block.
    fuse_qkv_params: bool, default = 'False'
                    if set to `True`, `TransformerLayer` module exposes a single fused
                    parameter for query-key-value. This enables optimizations such as QKV
                    fusion without concatentations/splits and also enables the argument
                    `fuse_wgrad_accumulation`.
    qk_norm_type: Optional[str], default = None
                    type of normalization to apply to query and key tensors.
                    Options: None, 'L2Normalization', 'RMSNorm', 'LayerNorm'. When None, no normalization is applied.
                    When 'L2Normalization', L2 normalization is applied to query and key tensors.
                    When 'RMSNorm', RMS normalization is applied to query and key tensors.
                    When 'LayerNorm', layer normalization is applied to query and key tensors.
                    Normalization is applied after RoPE (if applicable) but before attention computation
                    when `qk_norm_before_rope` is False. This follows the e.g. Llama4 approach for
                    QK normalization to improve training stability and model performance.
    qk_norm_eps: float, default = 1e-6
                    epsilon value for normalization of query and key tensors.
                    Only used when `qk_norm_type` is not None.
    qk_norm_before_rope: bool, default = `False`
                    if set to `True`, query and key normalization is applied before rotary position
                    embedding. When `False` (default), normalization is applied after RoPE.
                    This parameter allows supporting different architectural variants that apply
                    QK normalization at different points.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        num_gqa_groups: Optional[int] = None,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        layer_number: Optional[int] = None,
        kv_channels: Optional[int] = None,
        self_attn_mask_type: str = "causal",
        window_size: Optional[Tuple[int, int]] = None,
        enc_dec_attn_mask_type: str = "no_mask",
        enc_dec_window_size: Optional[Tuple[int, int]] = None,
        tp_group: Optional[dist_group_type] = None,
        tp_size: int = 1,
        params_dtype: Optional[torch.dtype] = None,
        get_rng_state_tracker: Optional[Callable] = None,
        fuse_wgrad_accumulation: bool = False,
        seq_length: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
        sequence_parallel: bool = False,
        apply_residual_connection_post_layernorm: bool = False,
        output_layernorm: bool = False,
        parallel_attention_mlp: bool = False,
        layer_type: str = "encoder",
        drop_path_rate: float = 0.0,
        set_parallel_mode: bool = False,
        fuse_qkv_params: bool = False,
        rotary_pos_interleaved: bool = False,
        zero_centered_gamma: bool = False,
        qkv_weight_interleaved: bool = True,
        ub_tp_comm_overlap: bool = False,
        ub_overlap_ag: bool = True,
        ub_overlap_rs: bool = True,
        ub_overlap_rs_dgrad: bool = False,
        ub_bulk_dgrad: bool = True,
        ub_bulk_wgrad: bool = True,
        bias: bool = True,
        activation: str = "gelu",
        normalization: str = "LayerNorm",
        device: Union[torch.device, str] = "cuda",
        attn_input_format: str = "sbhd",
        name: str = None,
        qk_norm_type: Optional[str] = None,
        qk_norm_eps: float = 1e-6,
        qk_norm_before_rope: bool = False,
    ) -> None:
        super().__init__()

        self.self_attn_mask_type = self_attn_mask_type
        self.window_size = window_size
        self.enc_dec_attn_mask_type = enc_dec_attn_mask_type
        self.enc_dec_window_size = enc_dec_window_size
        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        ub_bulk_wgrad = ub_tp_comm_overlap and ub_bulk_wgrad
        ub_bulk_dgrad = ub_tp_comm_overlap and ub_bulk_dgrad
        ub_overlap_ag = ub_tp_comm_overlap and ub_overlap_ag
        ub_overlap_rs = ub_tp_comm_overlap and ub_overlap_rs
        ub_overlap_rs_dgrad = ub_tp_comm_overlap and ub_overlap_rs_dgrad

        bias_dropout_fusion = bool(int(os.getenv("NVTE_BIAS_DROPOUT_FUSION", "1")))
        self.layer_number = layer_number
        self.output_layernorm = output_layernorm
        self.layer_type = layer_type
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm

        if parallel_attention_mlp:
            assert self.layer_type == "encoder", "parallel_attention requires layer_type='encoder'"
            assert not self.apply_residual_connection_post_layernorm, (
                "parallel_attention and apply_residual_connection_post_layernorm "
                "not supported simultaneously."
            )
            assert (
                not self.output_layernorm
            ), "parallel_attention and output_layernorm not supported simultaneously"

        self.parallel_attention_mlp = parallel_attention_mlp

        assert layer_type in LayerTypes, f"layer_type {layer_type} not supported"

        if not fuse_qkv_params:
            assert (
                not fuse_wgrad_accumulation
            ), "Gradient accumulation fusion requires single QKV parameter."

        if not fuse_qkv_params:
            qkv_weight_interleaved = False

        self.kv_channels = kv_channels if kv_channels else (hidden_size // num_attention_heads)

        if init_method is None:
            init_method = get_default_init_method()
        if output_layer_init_method is None:
            output_layer_init_method = get_default_init_method()

        self.tp_size = tp_size if tp_group is None else get_distributed_world_size(tp_group)
        self.sequence_parallel = (self.tp_size > 1) and sequence_parallel
        self.seq_length = seq_length

        self.get_rng_state_tracker = get_rng_state_tracker

        self.attn_input_format = attn_input_format

        self.name = name

        attention_args = (
            hidden_size,
            num_attention_heads,
            self.kv_channels,
            attention_dropout,
            layernorm_epsilon,
            init_method,
            output_layer_init_method,
        )
        common_attention_kwargs = {
            "layer_number": layer_number,
            "tp_group": tp_group,
            "tp_size": self.tp_size,
            "num_gqa_groups": num_gqa_groups,
            "fuse_wgrad_accumulation": fuse_wgrad_accumulation,
            "get_rng_state_tracker": get_rng_state_tracker,
            "sequence_parallel": self.sequence_parallel,
            "params_dtype": params_dtype,
            "return_layernorm_output": apply_residual_connection_post_layernorm,
            "set_parallel_mode": set_parallel_mode,
            "fuse_qkv_params": fuse_qkv_params,
            "zero_centered_gamma": zero_centered_gamma,
            "qkv_weight_interleaved": qkv_weight_interleaved,
            "rotary_pos_interleaved": rotary_pos_interleaved,
            "ub_bulk_wgrad": ub_bulk_wgrad,
            "ub_bulk_dgrad": ub_bulk_dgrad,
            "ub_overlap_ag": ub_overlap_ag,
            "ub_overlap_rs": ub_overlap_rs,
            "ub_overlap_rs_dgrad": ub_overlap_rs_dgrad,
            "qkv_format": self.attn_input_format,
            "seq_length": seq_length,
            "micro_batch_size": micro_batch_size,
        }

        self.self_attention = MultiheadAttention(
            *attention_args,
            **common_attention_kwargs,
            input_layernorm=not output_layernorm,
            attention_type="self",
            bias=bias,
            return_bias=not self.parallel_attention_mlp,
            normalization=normalization,
            device=device,
            qk_norm_type=qk_norm_type,
            qk_norm_eps=qk_norm_eps,
            qk_norm_before_rope=qk_norm_before_rope,
            name=name + ".self_attention" if name is not None else None,
        )

        if layer_type == "decoder":
            self.inter_attention = MultiheadAttention(
                *attention_args,
                **common_attention_kwargs,
                attn_mask_type=enc_dec_attn_mask_type,
                input_layernorm=True,
                attention_type="cross",
                bias=bias,
                return_bias=True,
                normalization=normalization,
                device=device,
                qk_norm_type=qk_norm_type,
                qk_norm_eps=qk_norm_eps,
                qk_norm_before_rope=qk_norm_before_rope,
                name=name + ".inter_attention" if name is not None else None,
            )

        # LayerNorm -> activation(Linear + Bias) -> Linear
        # parallel_mode not supported for LayerNormMLP,
        # FC1 is CPL and FC2 is RPL
        # In the case of GLU activation, FC1 handles both
        # Linear layers before the activation
        self.layernorm_mlp = LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            eps=layernorm_epsilon,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            tp_group=tp_group,
            tp_size=self.tp_size,
            get_rng_state_tracker=get_rng_state_tracker,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            bias=bias,
            return_bias=not self.parallel_attention_mlp,
            sequence_parallel=self.sequence_parallel,
            params_dtype=params_dtype,
            return_layernorm_output=apply_residual_connection_post_layernorm,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            set_parallel_mode=set_parallel_mode,
            zero_centered_gamma=zero_centered_gamma,
            ub_bulk_wgrad=ub_bulk_wgrad,
            ub_bulk_dgrad=ub_bulk_dgrad,
            ub_overlap_rs_dgrad=ub_overlap_rs_dgrad,
            ub_overlap_rs=ub_overlap_rs,
            ub_overlap_ag=ub_overlap_ag,
            activation=activation,
            normalization=normalization,
            device=device,
            name=name + ".layernorm_mlp" if name is not None else None,
        )

        self.hidden_dropout = hidden_dropout
        self.bias_dropout_fusion = bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Set bias+dropout+add fusion grad_enable execution handler.
        use_nvfuser = torch_version() >= (1, 10, 0) and torch_version() < (2, 2, 0)
        self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad

        if self.bias_dropout_fusion:
            set_jit_fusion_options()
            if seq_length and micro_batch_size:
                if self.sequence_parallel:
                    seq_length = seq_length // self.tp_size
                warmup_jit_bias_dropout_add_all_dtypes(hidden_size, seq_length, micro_batch_size)

        norm_module = {
            "LayerNorm": LayerNorm,
            "RMSNorm": RMSNorm,
        }
        if self.output_layernorm:
            self.layernorm = norm_module[normalization](
                hidden_size,
                eps=layernorm_epsilon,
                sequence_parallel=self.sequence_parallel,
                params_dtype=params_dtype,
                zero_centered_gamma=zero_centered_gamma,
                device=device,
            )

    def set_tensor_parallel_group(self, tp_group: Union[dist_group_type, None]) -> None:
        """
        Set the tensor parallel group for the given
        module before executing the forward pass.

        Parameters
        ----------
        tp_group : ProcessGroup, default = `None`
                  tensor parallel process group.
        """
        # Deep iterate but skip self to avoid infinite recursion.
        for index, child in enumerate(self.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_tensor_parallel_group"):
                child.set_tensor_parallel_group(tp_group)

    def reset_fp8_meta_tensors(self) -> None:
        """Set TP group"""
        # Deep iterate but skip self to avoid infinite recursion.
        for index, child in enumerate(self.modules()):
            if index == 0:
                continue
            if hasattr(child, "reset_fp8_meta_tensors"):
                child.reset_fp8_meta_tensors()

    def set_context_parallel_group(
        self,
        cp_group: Union[dist_group_type, List[dist_group_type], None],
        cp_global_ranks: List[int],
        cp_stream: torch.cuda.Stream,
        cp_comm_type: str = "p2p",
    ) -> None:
        """
        Set the context parallel attributes for the given
        module before executing the forward pass.

        Parameters
        ----------
        cp_group : Union[ProcessGroup, List[ProcessGroup]]
                  context parallel process group.
                  ProcessGroup is for cp_comm_type of "p2p", "all_gather", and "a2a".
                  List[ProcessGroup] is for cp_comm_type of "a2a+p2p", where cp_group[0]
                  and cp_group[1] are for a2a and p2p communications respectively.
        cp_global_ranks : List[int]
                         list of global ranks in the context group.
        cp_stream : torch.cuda.Stream
                   cuda stream for context parallel execution.
        cp_comm_type : str, default = `p2p`
                      inter-gpu communication type for context parallelism.
                      Can be "p2p" or "all_gather" or "a2a", or "a2a+p2p".
                      "p2p": Exchange KV chunks with P2P communications in ring topology.
                             P2P is async and can be overlapped with attention compute.
                      "all_gather": All-gather to get full sequence of KV before attention.
                                    The all-gather is not async, and cannot be overlapped.
                      "a2a": Like DeepSpeed Ulysses, scatter attention heads across the CP
                             group, and gather to get full sequence of QKV.
                      "a2a+p2p": hierarchical CP implementation. First applying a2a to QKV
                      across each CP sub-group (e.g., via NVLink), then exchanging KV with
                      p2p between sub-groups (e.g., via IBLink).
        """
        # Deep iterate but skip self to avoid infinite recursion.
        for index, child in enumerate(self.modules()):
            if index == 0:
                continue
            if hasattr(child, "set_context_parallel_group"):
                child.set_context_parallel_group(cp_group, cp_global_ranks, cp_stream, cp_comm_type)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        self_attn_mask_type: Optional[str] = None,
        window_size: Optional[Tuple[int, int]] = None,
        encoder_output: Optional[torch.Tensor] = None,
        enc_dec_attn_mask: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        enc_dec_attn_mask_type: Optional[str] = None,
        enc_dec_window_size: Optional[Tuple[int, int]] = None,
        is_first_microbatch: Optional[bool] = None,
        checkpoint_core_attention: bool = False,
        inference_params: Optional[InferenceParams] = None,
        rotary_pos_emb: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[torch.Tensor] = None,
        alibi_slopes: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        cu_seqlens_q_padded: Optional[torch.Tensor] = None,
        cu_seqlens_kv_padded: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        fast_zero_fill: bool = True,
        pad_between_seqs: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Transformer Layer: attention block and a feedforward network (MLP)

        .. note::

            Argument :attr:`attention_mask` is only used when :attr:`self_attn_mask_type`
            includes `"padding"` or `"arbitrary"`.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Input tensor.
        attention_mask : Optional[torch.Tensor], default = `None`
            Boolean tensor used to mask out self-attention softmax input. It should be
            in [batch_size, 1, 1, seqlen_q] for padding masks, and broadcastable
            to [batch_size, num_heads, max_seqlen_q, max_seqlen_kv] for "`arbitrary`"
            mask. It should be `None` for causal masks and "`no_mask`" type.
            A `True` value means the corresponding position is masked out and
            a `False` means that position is allowed to participate in attention.
        self_attn_mask_type: {'no_mask', 'causal', 'padding', 'padding_causal',
            'causal_bottom_right', 'padding_causal_bottom_right','arbitrary'},
            default = `causal`
            Type of attention mask passed into softmax operation for encoder.
            By default, causal masks are aligned to the top left corner of
            the softmax matrix. When "`bottom_right`" is specified in the mask type,
            causal masks are aligned to the bottom right corner.
        window_size: Optional[Tuple[int, int]], default = `None`
            Sliding window size for local attention in encoder.
        encoder_output : Optional[torch.Tensor], default = `None`
            Output of the encoder block to be fed into the decoder block if using
            `layer_type="decoder"`.
        enc_dec_attn_mask : Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]],
            default = `None`. Boolean tensors used to mask out inter-attention softmax input if
            using `layer_type="decoder"`. It should be a tuple of two masks in
            [batch_size, 1, 1, seqlen_q] and [batch_size, 1, 1, seqlen_kv] for padding masks.
            It should be broadcastable to [batch_size, num_heads, max_seqlen_q, max_seqlen_kv]
            for "`arbitrary`" mask. It should be `None` for causal masks and "`no_mask`".
            A `True` value means the corresponding position is masked out and a `False`
            means that position is allowed to participate in attention.
        enc_dec_attn_mask_type: {'no_mask', 'causal', 'padding', 'padding_causal', 'arbitrary'},
            default = `None`
            Type of attention mask passed into softmax operation for decoder.
        enc_dec_window_size: Optional[Tuple[int, int]], default = `None`
            Sliding window size for local attention in decoder.
        is_first_microbatch : {True, False, None}, default = None
            During training using either gradient accumulation or
            pipeline parallelism a minibatch of data is further split
            into microbatches. Between the microbatches of the same minibatch
            the model weights are not updated. Setting this parameter indicates
            whether the current microbatch is the first in a minibatch or not.
            When set, this parameter enables additional optimizations:

            * during FP8 training, it allows caching of the FP8 versions of
              the weights
            * it also allows skipping gradient accumulation during the
              first microbatch (since it is the first gradient being
              produced)
        checkpoint_core_attention: bool, default = `False`
            If true, forward activations for core attention are recomputed
            during the backward pass in order to save memory that would
            otherwise be occupied to store the forward activations until
            backprop.
        rotary_pos_emb: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], default = `None`
            Embeddings for query and key tensors for applying rotary position
            embedding. By default no input embedding is applied.
        core_attention_bias_type: str, default = `no_bias`
            Bias type, {`no_bias`, `pre_scale_bias`, `post_scale_bias`, `alibi`}
        core_attention_bias: Optional[torch.Tensor], default = `None`
            Bias tensor for Q * K.T
        alibi_slopes: Optional[torch.Tensor], default = `None`
            ALiBi slopes in FP32 and shape [nheads] or [batch_size, nheads].
            It adds a bias of (-alibi_slope * (i + seqlen_k - seqlen_q - j))
            to the attention score of query i and key j.
        cu_seqlens_q: Optional[torch.Tensor], default = `None`
            Cumulative sum of sequence lengths (without offset) in a batch for `query_layer`,
            with shape [batch_size + 1] and dtype torch.int32.
            Used by encoders, or decoders' self-attention.
        cu_seqlens_kv: Optional[torch.Tensor], default = `None`
            Cumulative sum of sequence lengths (without offset) in a batch for `key_layer`
            and `value_layer`, with shape [batch_size + 1] and dtype torch.int32.
            Used by decoders' cross-attention.
        cu_seqlens_q_padded: Optional[torch.Tensor], default = `None`
            Cumulative sum of sequence lengths (with offset) in a batch for `query_layer`,
            with shape [batch_size + 1] and dtype torch.int32. Set to `cu_seqlens_q` if None.
            Used by encoders, or decoders' self-attention.
        cu_seqlens_kv_padded: Optional[torch.Tensor], default = `None`
            Cumulative sum of sequence lengths (with offset) in a batch for `key_layer`
            and `value_layer`, with shape [batch_size + 1] and dtype torch.int32.
            Set to `cu_seqlens_kv` if None. Used by decoders' cross-attention.
        max_seqlen_q: Optional[int], default = `None`
            Maximum sequence length in `query_layer`.
            Calculated from `cu_seqlens_q_padded` if not provided.
        max_seqlen_kv: Optional[int], default = `None`
            Maximum sequence length in `key_layer` and `value_layer`.
            Calculated from `cu_seqlens_kv_padded` if not provided.
        fast_zero_fill: bool, default = `True`
            Whether to set output tensors to 0 or not before use.
        inference_params: InferenceParams, default = None
            Inference parameters that are passed to the main model in order
            to efficiently calculate and store the context during inference.
        pad_between_seqs: Optional[bool], default = `None`
            If None, inferred from qkv_format, cu_seqlens and cu_seqlens_padded.
            If true, there are padding tokens between individual sequences in a packed batch,
            i.e. qkv_format = 'thd'.
        """

        if self_attn_mask_type is None:
            self_attn_mask_type = self.self_attn_mask_type
        if window_size is None:
            window_size = self.window_size
        if enc_dec_attn_mask_type is None:
            enc_dec_attn_mask_type = self.enc_dec_attn_mask_type
        if enc_dec_window_size is None:
            enc_dec_window_size = self.enc_dec_window_size

        assert (
            self_attn_mask_type in AttnMaskTypes
        ), f"self_attn_mask_type {self_attn_mask_type} not supported"
        assert (
            enc_dec_attn_mask_type in AttnMaskTypes
        ), f"enc_dec_attn_mask_type {enc_dec_attn_mask_type} not supported"

        hidden_states = hidden_states.contiguous()

        if self.sequence_parallel and self.seq_length is not None:
            assert (
                hidden_states.shape[0] == self.seq_length // self.tp_size
            ), "Sequence dimension must be split across TP group when using sequence parallel."

        if (
            "padding" in self_attn_mask_type or self_attn_mask_type == "arbitrary"
        ) and attention_mask is not None:
            assert all(
                attention_mask[i].dtype == torch.bool for i in range(len(attention_mask))
            ), "Attention mask must be a boolean tensor or a list/tuple of two boolean tensors"
        if (
            "padding" in enc_dec_attn_mask_type or enc_dec_attn_mask_type == "arbitrary"
        ) and enc_dec_attn_mask is not None:
            assert all(
                enc_dec_attn_mask[i].dtype == torch.bool for i in range(len(enc_dec_attn_mask))
            ), "Encoder-decoder attention mask must be boolean tensor(s)"

        if TEDebugState.debug_enabled:
            TransformerEngineBaseModule._validate_name(self)

        # For AMP
        if torch.is_autocast_enabled():
            hidden_states = cast_if_needed(hidden_states, torch_get_autocast_gpu_dtype())

        # Self attention.
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask=attention_mask,
            attn_mask_type=self_attn_mask_type,
            window_size=window_size,
            inference_params=inference_params,
            is_first_microbatch=is_first_microbatch,
            checkpoint_core_attention=checkpoint_core_attention,
            rotary_pos_emb=rotary_pos_emb,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias=core_attention_bias,
            alibi_slopes=alibi_slopes,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_q,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_kv_padded=cu_seqlens_q_padded,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_q,
            fast_zero_fill=fast_zero_fill,
            pad_between_seqs=pad_between_seqs,
        )

        if self.apply_residual_connection_post_layernorm and not self.output_layernorm:
            attention_output, attention_bias, residual = self_attention_outputs
            hidden_states = self._bias_dropout_add(
                attention_output, attention_bias, residual, self.drop_path
            )
        elif not self.parallel_attention_mlp:
            attention_output, attention_bias = self_attention_outputs
            hidden_states = self._bias_dropout_add(
                attention_output, attention_bias, hidden_states, self.drop_path
            )

        # Cross attention.
        if self.layer_type == "decoder":
            inter_attention_outputs = self.inter_attention(
                hidden_states,
                attention_mask=enc_dec_attn_mask,
                attn_mask_type=enc_dec_attn_mask_type,
                window_size=enc_dec_window_size,
                encoder_output=encoder_output,
                inference_params=inference_params,
                is_first_microbatch=is_first_microbatch,
                checkpoint_core_attention=checkpoint_core_attention,
                rotary_pos_emb=rotary_pos_emb,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
                alibi_slopes=alibi_slopes,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_q_padded=cu_seqlens_q_padded,
                cu_seqlens_kv_padded=cu_seqlens_kv_padded,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                fast_zero_fill=fast_zero_fill,
                pad_between_seqs=pad_between_seqs,
            )
            if self.apply_residual_connection_post_layernorm:
                attention_output, attention_bias, residual = inter_attention_outputs
            else:
                attention_output, attention_bias = inter_attention_outputs
                residual = hidden_states

            hidden_states = self._bias_dropout_add(attention_output, attention_bias, residual)

        # MLP.
        mlp_outputs = self.layernorm_mlp(
            hidden_states,
            is_first_microbatch=is_first_microbatch,
        )
        if self.apply_residual_connection_post_layernorm:
            mlp_output, mlp_bias, residual = mlp_outputs
            output = self._bias_dropout_add(mlp_output, mlp_bias, residual, self.drop_path)
        elif self.parallel_attention_mlp:
            output = self._bias_dropout_add(
                self_attention_outputs, mlp_outputs, hidden_states, self.drop_path
            )
        else:
            mlp_output, mlp_bias = mlp_outputs
            output = self._bias_dropout_add(mlp_output, mlp_bias, hidden_states, self.drop_path)

        # For BERT like architectures.
        if self.output_layernorm:
            output = self.layernorm(output)

        # output: [s, b, h]
        return output

    def _bias_dropout_add(self, hidden_state, bias, residual, drop_path=None):
        if (
            drop_path is None
            and bias is not None
            and bias.numel() != 0
            and not is_in_onnx_export_mode()
        ):
            if self.bias_dropout_fusion:
                if self.training:
                    bias_dropout_add_func = bias_dropout_add_fused_train
                else:
                    bias_dropout_add_func = bias_dropout_add_fused_inference
            else:
                bias_dropout_add_func = get_bias_dropout_add(self.training)

            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(hidden_state, bias, residual, self.hidden_dropout)
        else:
            if bias is not None and bias.numel() != 0:
                hidden_state = hidden_state + bias
            out = torch.nn.functional.dropout(
                hidden_state, p=self.hidden_dropout, training=self.training
            )
            if drop_path is not None:
                out = drop_path(out)
            output = residual + out

        return output
