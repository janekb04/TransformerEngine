# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the CuTeDSL NVFP4 quantize-transpose kernel.

Each test checks if the correct kernel runs (CUDA C++ or CuTe DSL) for it and if the two kernels'
outputs match bitwise. Covers cases supported by the NVFP4QuantizeTransposeTuned1DKernel.
"""

import ctypes
import itertools
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

import transformer_engine.pytorch as te
from transformer_engine.common import (
    _get_shared_object_file,
    _load_tvm_ffi_library,
    _register_cutedsl_backends,
)
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.constants import NVFP4_BLOCK_SCALING_SIZE

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)

CORE_LIB = ctypes.CDLL(str(_get_shared_object_file("core")))
backend_available = recipe_available and _load_tvm_ffi_library() and _register_cutedsl_backends()

pytestmark = [
    pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe),
    pytest.mark.skipif(
        not backend_available, reason="the CuTeDSL backend could not be registered via tvm-ffi"
    ),
]

if backend_available:
    from transformer_engine.common.CuTeDSL.cast.nvfp4.quantize_transpose import (
        get_nvfp4_quantization_function,
    )
    from transformer_engine.common.CuTeDSL.cast.nvfp4.quantize_transpose.kernel import (
        NVFP4QuantizeTransposeTuned1DKernel,
    )
    from transformer_engine.common.CuTeDSL.cast.nvfp4.utils import NVFP4_SHAPE_ALIGNMENT

SEED = 1234


@dataclass(frozen=True)
class Config:
    """The configuration for a case, which determines the kernel variant to run"""
    shape: tuple
    dtype: torch.dtype
    rowwise: bool = False
    columnwise: bool = False
    with_2d_quantization: bool = False
    stochastic_rounding: bool = False
    row_scaled_nvfp4: bool = False
    fast_math: bool = False
    noop_flag: bool = False  # is a non-nullptr noop tensor passed, regardless of its value


def flat_2d_dims(shape):
    return math.prod(shape[:-1]), shape[-1]


def ceil_div(numerator, denominator):
    return -(-numerator // denominator)


def make_quantizer(cfg):
    return NVFP4Quantizer(
        rowwise=cfg.rowwise,
        columnwise=cfg.columnwise,
        with_2d_quantization=cfg.with_2d_quantization,
        stochastic_rounding=cfg.stochastic_rounding,
        row_scaled_nvfp4=cfg.row_scaled_nvfp4,
    )


# ---------------------------------------------------------------------------
# Backend selection and kernel observation
# ---------------------------------------------------------------------------


@contextmanager
def cutedsl_backend_enabled(enabled):
    CORE_LIB.nvte_set_cutedsl_quant_backend(int(enabled))
    try:
        yield
    finally:
        from_env = os.getenv("NVTE_ENABLE_CUTEDSL_QUANT_BACKEND", "0")
        CORE_LIB.nvte_set_cutedsl_quant_backend(int(not from_env.startswith("0")))


@contextmanager
def fast_math_enabled(enabled):
    """TE reads NVTE_USE_FAST_MATH on every quantize call."""
    previous = os.environ.get("NVTE_USE_FAST_MATH")
    os.environ["NVTE_USE_FAST_MATH"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            del os.environ["NVTE_USE_FAST_MATH"]
        else:
            os.environ["NVTE_USE_FAST_MATH"] = previous


def should_cutedsl_kernel_run(cfg):
    """Conditions from cast/dispatch/quantize.cuh and cast/nvfp4/quantize_transpose_nvfp4_cutedsl.cuh."""
    rows, cols = flat_2d_dims(cfg.shape)
    return (
        cfg.dtype is torch.bfloat16
        and rows % NVFP4_SHAPE_ALIGNMENT == 0
        and cols % NVFP4_SHAPE_ALIGNMENT == 0
        and cfg.rowwise
        and not cfg.with_2d_quantization
    )


def does_cutedsl_kernel_run(quantization):
    """The compiled kernel's symbol contains the name of the class it is traced from."""
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        quantization()
        torch.cuda.synchronize()
    return any(
        NVFP4QuantizeTransposeTuned1DKernel.__name__ in event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA and event.device_time_total > 0
    )


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------


def make_random_input(cfg, seed=SEED):
    """Uniform noise, scaled by a random power of two per block. Neighbouring blocks then get
    different E4M3 block scales and the largest elements of a block saturate E2M1. TE requires
    both flat dims to be whole numbers of blocks."""
    rows, cols = flat_2d_dims(cfg.shape)
    blocks = (rows, cols // NVFP4_BLOCK_SCALING_SIZE)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.empty((*blocks, NVFP4_BLOCK_SCALING_SIZE), device="cuda").uniform_(
        -6.0, 6.0, generator=generator
    )
    exponents = torch.randint(-6, 7, (*blocks, 1), device="cuda", generator=generator).float()
    return (values * torch.exp2(exponents)).to(cfg.dtype).view(cfg.shape)


def make_prefilled_output(cfg, fill_byte):
    """Fills the output buffers with fill_byte. The two runs of a comparison use different
    values, so bytes that no kernel writes do not compare equal."""
    out = make_quantizer(cfg).make_empty(cfg.shape, dtype=cfg.dtype, device="cuda")
    for buffer in (
        out._rowwise_data,
        out._columnwise_data,
        out._rowwise_scale_inv,
        out._columnwise_scale_inv,
    ):
        if buffer is not None:
            buffer.view(torch.uint8).fill_(fill_byte)
    return out


def strip_padding_bytes(out, cfg):
    """Returns the buffers to compare. The scale buffers are allocated with padding that no
    kernel writes, so only their valid region is taken."""
    rows, cols = flat_2d_dims(cfg.shape)
    result = {}
    if cfg.rowwise:
        result["rowwise data"] = out._rowwise_data.view(torch.uint8).clone()
        result["rowwise scales"] = out._rowwise_scale_inv.view(torch.uint8)[
            :rows, : ceil_div(cols, NVFP4_BLOCK_SCALING_SIZE)
        ].clone()
    if cfg.columnwise:
        result["columnwise data"] = out._columnwise_data.view(torch.uint8).clone()
        result["columnwise scales"] = out._columnwise_scale_inv.view(torch.uint8)[
            :cols, : ceil_div(rows, NVFP4_BLOCK_SCALING_SIZE)
        ].clone()
    for name, amax in (
        ("rowwise amax", out._amax_rowwise),
        ("columnwise amax", out._amax_columnwise),
    ):
        if amax is not None:
            result[name] = amax.clone()
    return result


def quantize_and_verify_kernel(cfg, x, *, cutedsl_enabled, out=None, noop_value=0.0):
    out = make_prefilled_output(cfg, 0x00) if out is None else out
    flag = torch.full((1,), noop_value, device="cuda") if cfg.noop_flag else None
    with cutedsl_backend_enabled(cutedsl_enabled), fast_math_enabled(cfg.fast_math):
        # Stochastic rounding takes its Philox state from the default CUDA generator.
        torch.cuda.manual_seed(SEED)
        ran = does_cutedsl_kernel_run(lambda: out.quantize_(x, noop_flag=flag))
    expected = cutedsl_enabled and should_cutedsl_kernel_run(cfg)
    assert ran == expected, (
        "the CuTeDSL kernel ran where the CUDA kernel was expected to"
        if ran
        else "the CuTeDSL kernel was expected to run, but the quantization fell back to CUDA"
    )
    return out


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

# The five bits NVFP4QuantConfig is keyed on, in to_key() order. Each name is also a Config field.
VARIANT_FLAGS = ("stochastic_rounding", "fast_math", "row_scaled_nvfp4", "columnwise", "noop_flag")


def assert_cutedsl_variant_compiles(variant):
    """The key is the one NVFP4QuantConfig::to_key builds, so this registers the kernel that the
    C++ dispatcher will look up."""
    flags = [variant[name] for name in VARIANT_FLAGS]
    key = "cutedsl_nvfp4_" + "_".join("1" if flag else "0" for flag in flags)
    assert get_nvfp4_quantization_function(key, *flags), f"no CuTeDSL kernel compiled for {key}"


def assert_backends_agree(cfg, x=None):
    assert should_cutedsl_kernel_run(cfg), "this config never reaches the CuTeDSL kernel"
    x = make_random_input(cfg) if x is None else x
    cuda = quantize_and_verify_kernel(
        cfg, x, cutedsl_enabled=False, out=make_prefilled_output(cfg, 0x00)
    )
    cutedsl = quantize_and_verify_kernel(
        cfg, x, cutedsl_enabled=True, out=make_prefilled_output(cfg, 0xFF)
    )
    expected = strip_padding_bytes(cuda, cfg)
    actual = strip_padding_bytes(cutedsl, cfg)
    for name, expected_bytes in expected.items():
        assert torch.equal(actual[name], expected_bytes), f"{name} differs from the CUDA kernel's"


# ---------------------------------------------------------------------------
# Test all kernel variants
# ---------------------------------------------------------------------------

VARIANTS = [
    variant
    for variant in (
        dict(zip(VARIANT_FLAGS, flags))
        for flags in itertools.product([False, True], repeat=len(VARIANT_FLAGS))
    )
    # NVFP4Quantizer rejects row scaling with stochastic rounding, so nothing reaches these.
    if not (variant["row_scaled_nvfp4"] and variant["stochastic_rounding"])
]


def variant_id(variant):
    return "-".join(name for name, on in variant.items() if on) or "plain"


@pytest.mark.parametrize("variant", VARIANTS, ids=variant_id)
def test_kernel_variants(variant):
    assert_cutedsl_variant_compiles(variant)
    assert_backends_agree(Config(shape=(256, 1024), dtype=torch.bfloat16, rowwise=True, **variant))


# ---------------------------------------------------------------------------
# Test different shapes
# ---------------------------------------------------------------------------

# All flat dims are aligned; the shapes differ in how they fit the kernel's 128x128 chunk.
SHAPES = [
    (32, 32),  # one partial chunk in both dimensions
    (32, 1024),  # a single row of chunks
    (1024, 32),  # a single column of chunks
    (96, 160),  # partial along both edges
    (512, 512),
    (8, 32, 1024),  # rank > 2, flattened to (256, 1024)
    (8192, 7168),  # many chunks in both dimensions
]


def shape_id(shape):
    return "x".join(str(dim) for dim in shape)


@pytest.mark.parametrize("shape", SHAPES, ids=shape_id)
@pytest.mark.parametrize("columnwise", [False, True], ids=["rowwise", "transpose"])
def test_shapes(shape, columnwise):
    assert_backends_agree(
        Config(shape=shape, dtype=torch.bfloat16, rowwise=True, columnwise=columnwise)
    )


# ---------------------------------------------------------------------------
# Test configs, which should fall back to CUDA
# ---------------------------------------------------------------------------

CUDA_FALLBACK_CONFIGS = {
    "unaligned-rows": Config(shape=(48, 1024), dtype=torch.bfloat16, rowwise=True),
    "unaligned-cols": Config(shape=(256, 400), dtype=torch.bfloat16, rowwise=True),
    "fp32-input": Config(shape=(256, 1024), dtype=torch.float32, rowwise=True),
    "fp16-input": Config(shape=(256, 1024), dtype=torch.float16, rowwise=True),
    "columnwise-only": Config(shape=(256, 1024), dtype=torch.bfloat16, columnwise=True),
    "2d-quantization": Config(
        shape=(256, 1024),
        dtype=torch.bfloat16,
        rowwise=True,
        columnwise=True,
        with_2d_quantization=True,
    ),
}


@pytest.mark.parametrize("cfg", CUDA_FALLBACK_CONFIGS.values(), ids=CUDA_FALLBACK_CONFIGS.keys())
def test_cuda_fallback_configs(cfg):
    assert not should_cutedsl_kernel_run(cfg)
    quantize_and_verify_kernel(cfg, make_random_input(cfg), cutedsl_enabled=True)


# ---------------------------------------------------------------------------
# Test numerical edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["zero-blocks", "negative-zero-blocks", "all-zero"])
@pytest.mark.parametrize("use_fast_math", [False, True], ids=["exact", "fastmath"])
def test_degenerate_inputs(kind, use_fast_math):
    """A zero block gives a zero block amax, which both kernels pass to a reciprocal and clamp;
    an all-zero tensor does the same for the encode scale. Negative zero separates the two forms
    of the scaling multiply: the fast-math path uses an fma whose +0 addend turns a -0 product
    into +0, and E2M1 encodes -0 as 0x8 and +0 as 0x0."""
    cfg = Config(
        shape=(256, 1024),
        dtype=torch.bfloat16,
        rowwise=True,
        columnwise=True,
        fast_math=use_fast_math,
    )
    x = make_random_input(cfg)
    if kind == "all-zero":
        x.zero_()
    else:
        x.view(-1, NVFP4_BLOCK_SCALING_SIZE)[::3] = -0.0 if kind == "negative-zero-blocks" else 0.0
    assert_backends_agree(cfg, x)


# ---------------------------------------------------------------------------
# Test noop tensor use
# ---------------------------------------------------------------------------


def test_cast_noop_flag():
    """Checks that the flag suppressed the writes, and that this input would otherwise have
    changed the output. The amax pass runs before the kernel and the flag does not gate it."""
    cfg = Config(
        shape=(256, 1024), dtype=torch.bfloat16, rowwise=True, columnwise=True, noop_flag=True
    )
    x = make_random_input(cfg)
    previous_input = make_random_input(cfg, seed=SEED + 1)

    x_quantized = strip_padding_bytes(quantize_and_verify_kernel(cfg, x, cutedsl_enabled=True), cfg)
    out = quantize_and_verify_kernel(cfg, previous_input, cutedsl_enabled=True)
    before_noop_call = strip_padding_bytes(out, cfg)
    after_noop_call = strip_padding_bytes(
        quantize_and_verify_kernel(cfg, x, cutedsl_enabled=True, out=out, noop_value=1.0), cfg
    )

    assert not torch.equal(
        before_noop_call["rowwise data"], x_quantized["rowwise data"]
    ), "the two inputs quantize to the same bytes, so the flag would suppress nothing visible"
    overwritten = {
        name
        for name, before in before_noop_call.items()
        if not torch.equal(after_noop_call[name], before)
    }
    assert not overwritten - {
        "rowwise amax",
        "columnwise amax",
    }, f"the noop flag did not suppress writes to {sorted(overwritten)}"


# ---------------------------------------------------------------------------
# Test, independent of the CUDA kernel, that the quantization seems reasonable
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_fast_math", [False, True], ids=["exact", "fastmath"])
def test_dequantized_output_tracks_input(use_fast_math):
    """Elementwise tolerances do not apply, since an element that is small relative to its block
    amax quantizes to zero. The bound is on the relative norm of the whole tensor."""
    cfg = Config(shape=(256, 1024), dtype=torch.bfloat16, rowwise=True, fast_math=use_fast_math)
    x = make_random_input(cfg)
    out = quantize_and_verify_kernel(cfg, x, cutedsl_enabled=True)
    error = torch.linalg.vector_norm(out.dequantize(dtype=torch.float32) - x.float())
    error /= torch.linalg.vector_norm(x.float())
    assert error < 0.2, f"relative error {error:.4f} is too large for an NVFP4 round trip"
