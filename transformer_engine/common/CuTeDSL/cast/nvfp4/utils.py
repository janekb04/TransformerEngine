# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Common utilities for the NVFP4 kernels."""

import cutlass
from cutlass import cute
from cutlass import BFloat16, Float32, Float8E4M3FN, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from transformer_engine.common.CuTeDSL.utils import (
    BFLOAT16_MAX,
    FLOAT4E2M1_MAX,
    FLOAT8E4M3_MAX,
    FLOAT32_MAX,
    select_f32,
)


# Number of elements per NVFP4 quantization block.
NVFP4_BLOCK_SCALING_SIZE = 16
# Input row/col divisibility the CUDA tuned-1D kernel requires (required for 16B TMA alignment).
NVFP4_SHAPE_ALIGNMENT = 32
# Padding TE applies to the scale tensors' outer / inner dim (from NVFP4Quantizer::get_scale_shape).
NVFP4_SCALE_PAD_OUTER = 128
NVFP4_SCALE_PAD_INNER = 4


@cute.jit
def compute_global_encode_sf(global_amax: Float32) -> Float32:
    """like compute_global_encode_scaling_factor_FP4 in core_nvfp4.cuh"""
    global_encode_scale = FLOAT8E4M3_MAX * FLOAT4E2M1_MAX / global_amax
    global_encode_scale = cute.arch.fmin(global_encode_scale, FLOAT32_MAX)
    # it seems CuTe DSL wouldn't automatically elide this branch
    degenerate = (global_amax == 0.0) | (global_encode_scale == 0.0)
    return select_f32(degenerate, Float32(1.0), global_encode_scale)


@cute.jit
def compute_block_decode_sf(block_amax: Float32, global_encode_sf: Float32) -> Float8E4M3FN:
    "like quantization_and_transposition_SF::compute_decoding_scaling_factor in core_nvfp4.cuh"
    block_decode_sf = block_amax * (global_encode_sf * (1.0 / FLOAT4E2M1_MAX))
    block_decode_sf = cute.arch.fmin(block_decode_sf, FLOAT32_MAX)
    return block_decode_sf.to(Float8E4M3FN)


@cute.jit
def compute_block_encode_sf(
    block_decode_sf: Float8E4M3FN,
    global_encode_sf: Float32,
    sf_type: cutlass.Constexpr[type],
):
    "like compute_nvfp4_scaling_coefficient in quantize_tranpose_nvfp4_tuned_1D.cuh"
    if cutlass.const_expr(sf_type == Float32):
        global_decode_sf = 1.0 / global_encode_sf
        block_encode_sf = 1.0 / (block_decode_sf.to(Float32) * global_decode_sf)
        return cute.arch.fmin(block_encode_sf, FLOAT32_MAX)
    if cutlass.const_expr(sf_type == BFloat16):
        block_encode_sf = global_encode_sf / block_decode_sf.to(Float32)
        block_encode_sf = cute.arch.fmin(block_encode_sf, BFLOAT16_MAX)
        return block_encode_sf.to(BFloat16)
    raise ValueError("Unsupported scaling-factor type. Only FP32 and BF16 are supported.")


@dsl_user_op
def mul_cvt_bf16x8_to_fp4x8(
    v01: Uint32,
    v23: Uint32,
    v45: Uint32,
    v67: Uint32,
    coeff: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """like mul_cvt_bf16_to_fp4_8x_round_to_nearest<bf16> in ptx.cuh"""
    asm = (
        "{\n"
        ".reg.f32 zero;\n\t"
        "mov.b32 zero, 0;\n\t"
        ".reg.b16 c;\n\t"
        # cast coeff to bf16 to perform fma in bf16
        "cvt.rn.bf16.f32 c, $5;\n\t"
        ".reg.b16 h0, h1, h2, h3, h4, h5, h6, h7;\n\t"
        "mov.b32 {h0, h1}, $1;\n\t"
        "mov.b32 {h2, h3}, $2;\n\t"
        "mov.b32 {h4, h5}, $3;\n\t"
        "mov.b32 {h6, h7}, $4;\n\t"
        # scale values by scaling factor
        ".reg.f32 v0, v1, v2, v3, v4, v5, v6, v7;\n\t"
        "fma.rn.f32.bf16 v0, h0, c, zero;\n\t"
        "fma.rn.f32.bf16 v1, h1, c, zero;\n\t"
        "fma.rn.f32.bf16 v2, h2, c, zero;\n\t"
        "fma.rn.f32.bf16 v3, h3, c, zero;\n\t"
        "fma.rn.f32.bf16 v4, h4, c, zero;\n\t"
        "fma.rn.f32.bf16 v5, h5, c, zero;\n\t"
        "fma.rn.f32.bf16 v6, h6, c, zero;\n\t"
        "fma.rn.f32.bf16 v7, h7, c, zero;\n\t"
        # convert scaled values to fp4e2m1 with RN
        ".reg.b8 f0, f1, f2, f3;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f0, v1, v0;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f1, v3, v2;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f2, v5, v4;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 f3, v7, v6;\n\t"
        "mov.b32 $0, {f0, f1, f2, f3};\n\t"
        "}"
    )
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                v01.ir_value(loc=loc, ip=ip),
                v23.ir_value(loc=loc, ip=ip),
                v45.ir_value(loc=loc, ip=ip),
                v67.ir_value(loc=loc, ip=ip),
                coeff.ir_value(loc=loc, ip=ip),
            ],
            asm,
            "=r,r,r,r,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_cvt_bf16x8_to_fp4x8_sr(
    v01: Uint32,
    v23: Uint32,
    v45: Uint32,
    v67: Uint32,
    coeff: Float32,
    rbits03: Uint32,
    rbits47: Uint32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """like mul_cvt_bf16_to_fp4_8x_stochastic_rounding<bf16> in ptx.cuh."""
    asm = (
        "{\n"
        ".reg.f32 zero;\n\t"
        "mov.b32 zero, 0;\n\t"
        ".reg.b16 c;\n\t"
        # cast coeff to bf16 to perform fma in bf16
        "cvt.rn.bf16.f32 c, $5;\n\t"
        ".reg.b16 h0, h1, h2, h3, h4, h5, h6, h7;\n\t"
        "mov.b32 {h0, h1}, $1;\n\t"
        "mov.b32 {h2, h3}, $2;\n\t"
        "mov.b32 {h4, h5}, $3;\n\t"
        "mov.b32 {h6, h7}, $4;\n\t"
        # scale values by scaling factor
        ".reg.f32 v0, v1, v2, v3, v4, v5, v6, v7;\n\t"
        "fma.rn.f32.bf16 v0, h0, c, zero;\n\t"
        "fma.rn.f32.bf16 v1, h1, c, zero;\n\t"
        "fma.rn.f32.bf16 v2, h2, c, zero;\n\t"
        "fma.rn.f32.bf16 v3, h3, c, zero;\n\t"
        "fma.rn.f32.bf16 v4, h4, c, zero;\n\t"
        "fma.rn.f32.bf16 v5, h5, c, zero;\n\t"
        "fma.rn.f32.bf16 v6, h6, c, zero;\n\t"
        "fma.rn.f32.bf16 v7, h7, c, zero;\n\t"
        # convert scaled values to fp4e2m1 with SR
        ".reg.b16 b03, b47;\n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b03, {v3, v2, v1, v0}, $6;\n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b47, {v7, v6, v5, v4}, $7;\n\t"
        "mov.b32 $0, {b03, b47};\n\t"
        "}"
    )
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                v01.ir_value(loc=loc, ip=ip),
                v23.ir_value(loc=loc, ip=ip),
                v45.ir_value(loc=loc, ip=ip),
                v67.ir_value(loc=loc, ip=ip),
                coeff.ir_value(loc=loc, ip=ip),
                rbits03.ir_value(loc=loc, ip=ip),
                rbits47.ir_value(loc=loc, ip=ip),
            ],
            asm,
            "=r,r,r,r,r,f,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_f32x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int64:
    """mov.b64 $return, {lo, hi};"""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [lo.ir_value(loc=loc, ip=ip), hi.ir_value(loc=loc, ip=ip)],
            "mov.b64 $0, {$1, $2};",
            "=l,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def make_mul2_cvt_bf16x8(mode: str, use_sr: bool):
    """Build the exact (f32-coefficient) scale-and-convert of eight bf16 elements: widen each
    selected half exactly (a bf16 is the top 16 bits of an f32), scale with `mul.rn.f32x2`
    (per-lane identical to the CUDA kernel's `mul.rn.f32`, at half the FP32 instruction
    count), and convert with the same e2m1 cvt as the CUDA kernel. The multiply is a plain
    mul rather than the bf16 path's fma-against-zero, which is exactly what the CUDA kernel
    pairs with an f32 coefficient -- a plain mul preserves the sign of a -0 product where the
    fma flushes it to +0, and E2M1 has a signed zero, so the instruction has to match.

    mode selects which bf16 of each input register is an element and their order:
      * "interleave": elements are (lo0, hi0, lo1, hi1, ...) -- the rowwise pass's 8 adjacent
        elements in four pair-registers.
      * "lo" / "hi": elements are the low (resp. high) halves of eight row registers -- one
        column of the colwise pass, so the caller passes 8 registers split across two calls.

    The f32 coefficient arrives packed twice into a b64 (pack_f32x2(c, c)), so both lanes see
    the same coefficient and the per-element results are bit-identical to the scalar code.
    """
    if mode == "interleave":
        n_in = 4
        widen = "".join(
            f"shl.b32 e{2 * i}, ${i + 1}, 16;\n\tand.b32 e{2 * i + 1}, ${i + 1}, 0xffff0000;\n\t"
            for i in range(4)
        )
    elif mode in ("lo", "hi"):
        n_in = 8
        if mode == "lo":
            widen = "".join(f"shl.b32 e{i}, ${i + 1}, 16;\n\t" for i in range(8))
        else:
            widen = "".join(f"and.b32 e{i}, ${i + 1}, 0xffff0000;\n\t" for i in range(8))
    else:
        raise ValueError(mode)

    coeff_idx = n_in + 1
    body = (
        "{\n.reg .b32 e0, e1, e2, e3, e4, e5, e6, e7;\n\t"
        + widen
        + ".reg .b64 p0, p1, p2, p3;\n\t"
        "mov.b64 p0, {e0, e1};\n\t"
        "mov.b64 p1, {e2, e3};\n\t"
        "mov.b64 p2, {e4, e5};\n\t"
        "mov.b64 p3, {e6, e7};\n\t"
        f"mul.rn.f32x2 p0, p0, ${coeff_idx};\n\t"
        f"mul.rn.f32x2 p1, p1, ${coeff_idx};\n\t"
        f"mul.rn.f32x2 p2, p2, ${coeff_idx};\n\t"
        f"mul.rn.f32x2 p3, p3, ${coeff_idx};\n\t"
        ".reg .f32 a0, a1, a2, a3, a4, a5, a6, a7;\n\t"
        "mov.b64 {a0, a1}, p0;\n\t"
        "mov.b64 {a2, a3}, p1;\n\t"
        "mov.b64 {a4, a5}, p2;\n\t"
        "mov.b64 {a6, a7}, p3;\n\t"
    )
    if use_sr:
        body += (
            ".reg .b16 b03, b47;\n\t"
            f"cvt.rs.satfinite.e2m1x4.f32 b03, {{a3, a2, a1, a0}}, ${coeff_idx + 1};\n\t"
            f"cvt.rs.satfinite.e2m1x4.f32 b47, {{a7, a6, a5, a4}}, ${coeff_idx + 2};\n\t"
            "mov.b32 $0, {b03, b47};\n\t"
            "}"
        )
        constraints = "=r," + "r," * n_in + "l,r,r"
    else:
        body += (
            ".reg .b8 f0, f1, f2, f3;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 f0, a1, a0;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 f1, a3, a2;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 f2, a5, a4;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 f3, a7, a6;\n\t"
            "mov.b32 $0, {f0, f1, f2, f3};\n\t"
            "}"
        )
        constraints = "=r," + "r," * n_in + "l"

    @dsl_user_op
    def op(*args, loc=None, ip=None) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [a.ir_value(loc=loc, ip=ip) for a in args],
                body,
                constraints,
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    return op


@dsl_user_op
def st_global_b64(addr, val, *, loc=None, ip=None):
    """st.global.b64 [addr], val;"""
    llvm.inline_asm(
        None,
        [addr.ir_value(loc=loc, ip=ip), val.ir_value(loc=loc, ip=ip)],
        "st.global.b64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
