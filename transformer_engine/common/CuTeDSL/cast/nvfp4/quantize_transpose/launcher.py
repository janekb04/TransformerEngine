# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Compilation and TVM-FFI registration for NVFP4 quantize-transpose."""

import logging

from cutlass import cute
from cutlass import BFloat16, Float4E2M1FN, Float8E4M3FN, Float32, Int64
import tvm_ffi

from transformer_engine.common.CuTeDSL.cast.nvfp4.utils import (
    NVFP4_SCALE_PAD_INNER,
    NVFP4_SCALE_PAD_OUTER,
    NVFP4_SHAPE_ALIGNMENT,
)
from .config import NVFP4QuantizeConfig
from .kernel import NVFP4QuantizeTransposeTuned1DKernel

logger = logging.getLogger("transformer_engine.cutedsl.nvfp4")


def compile_cutedsl_function_from_cfg(cfg: NVFP4QuantizeConfig):
    """Uses cute.compile to AOT-compile the variant of the NVFP4QuantizeTransposeTuned1DKernel specified by cfg"""

    kernel_obj = NVFP4QuantizeTransposeTuned1DKernel(cfg)

    def _gmem(dtype, shape, stride_order, align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=stride_order,
            memspace=cute.AddressSpace.gmem,
            assumed_align=align,
        )

    # Flattened dims of the input tensor.
    sym_M = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)
    sym_N = cute.sym_int32(divisibility=NVFP4_SHAPE_ALIGNMENT)

    scale_row_shape = (
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_OUTER),
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_INNER),
    )
    scale_col_shape = (
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_OUTER),
        cute.sym_int32(divisibility=NVFP4_SCALE_PAD_INNER),
    )

    # Fake tensors with divisibility constraints for tracing the kernel body.
    in_fake = _gmem(BFloat16, (sym_M, sym_N), stride_order=(1, 0), align=16)
    out_row_fake = _gmem(Float4E2M1FN, (sym_M, sym_N), stride_order=(1, 0), align=16)
    scale_row_fake = _gmem(Float8E4M3FN, scale_row_shape, stride_order=(1, 0), align=4)

    out_col_fake = (
        _gmem(Float4E2M1FN, (sym_N, sym_M), stride_order=(1, 0), align=16)
        if cfg.RETURN_TRANSPOSE
        else None
    )
    scale_col_fake = (
        _gmem(Float8E4M3FN, scale_col_shape, stride_order=(1, 0), align=4)
        if cfg.RETURN_TRANSPOSE
        else None
    )

    # The row-scaled variant takes per-row and per-column scaling factors instead of a single
    # per-tensor one per direction.
    amax_row_fake = (
        _gmem(Float32, (sym_M,), stride_order=(0,), align=4)
        if cfg.ROW_SCALED_NVFP4
        else _gmem(Float32, (1,), stride_order=(0,), align=4)
    )
    if cfg.RETURN_TRANSPOSE:
        amax_col_fake = _gmem(
            Float32, (sym_N,) if cfg.ROW_SCALED_NVFP4 else (1,), stride_order=(0,), align=4
        )
    else:
        amax_col_fake = None

    # The cast-noop flag, present only in the variant the dispatcher hands one to.
    noop_fake = _gmem(Float32, (1,), stride_order=(0,), align=4) if cfg.IS_NOOP else None

    # RNG (seed, offset) for stochastic rounding.
    rng_state_fake = (
        _gmem(Int64, (2,), stride_order=(0,), align=8) if cfg.USE_STOCHASTIC_ROUNDING else None
    )

    compiled = cute.compile(
        kernel_obj,
        in_fake,  # mX
        out_row_fake,  # mO_row
        scale_row_fake,  # mS_row
        out_col_fake,  # mO_col
        scale_col_fake,  # mS_col
        amax_row_fake,  # mAmaxRow
        amax_col_fake,  # mAmaxCol
        noop_fake,  # mNoop
        rng_state_fake,  # mRngState
        cute.runtime.make_fake_stream(),  # stream
        options="--enable-tvm-ffi",
    )
    return compiled


def get_nvfp4_quantization_function(
    fn_name: str,
    use_stochastic_rounding: bool,
    use_fast_math: bool,
    row_scaled_nvfp4: bool,
    return_transpose: bool,
    is_noop: bool,
) -> bool:
    """Interface to call from the C++ side to try to register a variant of the NVFP4QuantizeTransposeTuned1DKernel
    with tvm-ffi. Returns if the variant is supported and registered."""

    if tvm_ffi.get_global_func(fn_name, allow_missing=True) is not None:
        return True

    cfg = NVFP4QuantizeConfig(
        use_stochastic_rounding=use_stochastic_rounding,
        use_fast_math=use_fast_math,
        row_scaled_nvfp4=row_scaled_nvfp4,
        return_transpose=return_transpose,
        is_noop=is_noop,
    )

    logger.debug(
        "Compiling CuTeDSL NVFP4 quantization kernel for %s",
        f"NVFP4QuantizeConfig({use_stochastic_rounding=}, {use_fast_math=}, {row_scaled_nvfp4=},"
        f" {return_transpose=}, {is_noop=})",
    )
    try:
        compiled = compile_cutedsl_function_from_cfg(cfg)
    # cute.compile raises DSLBaseError subclasses, which derive from Exception, not from
    # RuntimeError. Any failure here just means the C++ dispatcher falls back to CUDA.
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning(
            "CuTeDSL NVFP4 kernel compilation failed, falling back to the CUDA C++ kernel: %s",
            e,
        )
        return False

    # Register the underlying native tvm-ffi function, if available
    native = getattr(compiled, "__tvm_ffi_object__", lambda: None)() or compiled

    tvm_ffi.register_global_func(fn_name, native, override=True)

    return True
