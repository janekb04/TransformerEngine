# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Cross-backend bit-exactness tests for the CuTeDSL NVFP4 quantize-transpose kernel, from JAX.

JAX companion to tests/pytorch/nvfp4/test_nvfp4_cutedsl_backend.py: the CuTeDSL dispatch lives in
TE/common, so this checks that the JAX FFI path reaches it and produces the same bytes as the CUDA
kernel. The kernel's own coverage (fast math, row scaling, the cast-noop flag, zero blocks) is
exercised from PyTorch, which is where those configurations are reachable.
"""

import ctypes
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transformer_engine.common import (
    _get_shared_object_file,
    _load_tvm_ffi_library,
    _register_cutedsl_backends,
)
from transformer_engine.jax import cpp_extensions as tex
from transformer_engine.jax.quantize import (
    QuantizerFactory,
    QuantizeLayout,
    ScaledTensor1x,
    ScalingMode,
    helper,
)

tvm_ffi = pytest.importorskip("tvm_ffi")

recipe_available, reason_for_no_recipe = helper.is_scaling_mode_supported(
    ScalingMode.NVFP4_1D_SCALING
)

# The already-loaded core lib (dlopen refcounts: this returns the same handle, so the call
# mutates the same dispatcher singleton the quantize ops read).
CORE_LIB = ctypes.CDLL(str(_get_shared_object_file("core")))
if not hasattr(CORE_LIB, "nvte_set_cutedsl_quant_backend"):
    raise RuntimeError(
        "libtransformer_engine.so lacks nvte_set_cutedsl_quant_backend -- rebuild the "
        "Transformer Engine core library."
    )

# TE registers the CuTeDSL entrypoints at import time only when NVTE_ENABLE_CUTEDSL_QUANT_BACKEND
# is set. These tests choose the backend through the C++ setter instead, so they wire up the
# Python side regardless of the environment. Both calls are idempotent.
backend_available = _load_tvm_ffi_library() and _register_cutedsl_backends()

pytestmark = [
    pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe),
    pytest.mark.skipif(
        not backend_available, reason="the CuTeDSL backend could not be registered via tvm-ffi"
    ),
]

# Elements sharing one E4M3 block scale.
NVFP4_BLOCK_SIZE = 16
# Both flat dims must be multiples of 32 for the tuned-1D path the CuTeDSL kernel implements.
MATRIX_SIZES = [
    (128, 128),
    (256, 1024),
    (512, 512),
    (8192, 7168),
]
# QuantizeLayout.COLWISE is absent on purpose: _quantize_dbias_impl diverts colwise-only to a
# pure-JAX implementation before reaching the FFI, so it cannot exercise the kernel.
Q_LAYOUTS = [QuantizeLayout.ROWWISE, QuantizeLayout.ROWWISE_COLWISE]

get_shape_id = lambda s: f"{s[0]}x{s[1]}"
get_layout_id = lambda l: "rowwise" if l == QuantizeLayout.ROWWISE else "rowwise-colwise"


def set_cutedsl_backend(enabled):
    CORE_LIB.nvte_set_cutedsl_quant_backend(1 if enabled else 0)


@pytest.fixture(scope="module", autouse=True)
def _restore_backend_choice_from_env():
    """Restore the flag that decides the CuTeDSL / CUDA backend choice when this module is done."""
    yield
    flag = os.getenv("NVTE_ENABLE_CUTEDSL_QUANT_BACKEND")
    set_cutedsl_backend(flag is not None and not flag.startswith("0"))


def cutedsl_key(q_layout, stochastic_rounding=False):
    """Mirror of NVFP4QuantConfig::to_key: the name the CuTeDSL backend registers its compiled
    kernel under. JAX drives neither fast math, row scaling nor the cast-noop flag, so only the
    stochastic-rounding and transpose flags vary."""
    flags = (stochastic_rounding, False, False, q_layout.has_colwise, False)
    return "cutedsl_nvfp4_" + "_".join("1" if f else "0" for f in flags)


def make_quantizer(q_layout, sr_rng_state=None):
    return QuantizerFactory.create(
        scaling_mode=ScalingMode.NVFP4_1D_SCALING,
        q_dtype=jnp.float4_e2m1fn,
        q_layout=q_layout,
        stochastic_rounding_rng_state=sr_rng_state,
    )


def make_input(shape, seed=0):
    """Uniform noise with a per-scaling-block power-of-two magnitude, so that neighbouring blocks
    land on different E4M3 block scales and the elements at the top of a block saturate E2M1."""
    rows, cols = shape
    k_value, k_exponent = jax.random.split(jax.random.PRNGKey(seed))
    values = jax.random.uniform(k_value, (rows, cols), jnp.float32, -6.0, 6.0)
    exponents = jax.random.randint(k_exponent, (rows, cols // NVFP4_BLOCK_SIZE, 1), -6, 7).astype(
        jnp.float32
    )
    values = (values.reshape(rows, -1, NVFP4_BLOCK_SIZE) * jnp.exp2(exponents)).reshape(rows, cols)
    return values.astype(jnp.bfloat16)


def scaled_tensors(out):
    """The one or two ScaledTensor1x a quantize call produced, by direction."""
    tensors = [out] if isinstance(out, ScaledTensor1x) else [out.rowwise_tensor, out.colwise_tensor]
    return {"colwise" if t.is_colwise else "rowwise": t for t in tensors}


def quantized_parts(out):
    """Pull the bytes both backends must agree on onto the host.

    Materializing here is what makes the backend toggle safe: JAX dispatch is asynchronous, so
    the FFI handler that reads the toggle may not have run yet when the Python call returns.
    ScaledTensor1x carries scale_inv already trimmed to the unpadded shape (see its
    __post_init__), so there is no uninitialized scale padding to exclude. The raw bytes are
    taken rather than the values, so the comparison is exact for FP4 and E4M3 alike.
    """
    parts = {}
    for name, tensor in scaled_tensors(out).items():
        parts[f"{name} data"] = np.asarray(tensor.data).view(np.uint8)
        parts[f"{name} scales"] = np.asarray(tensor.scale_inv).view(np.uint8)
    return parts


def quantize_with(backend, x, q_layout, sr_rng_state=None):
    """Quantize with one backend selected, and return the result materialized on the host."""
    set_cutedsl_backend(backend == "cutedsl")
    try:
        out = tex.quantize(x, quantizer=make_quantizer(q_layout, sr_rng_state))
        parts = quantized_parts(out)
        dequantized = {
            name: np.asarray(tensor.dequantize().astype(jnp.float32))
            for name, tensor in scaled_tensors(out).items()
        }
        return parts, dequantized
    finally:
        set_cutedsl_backend(False)


def assert_registered(key, tag):
    """Guard against a silent CUDA fallback: every config here is one the CuTeDSL backend serves,
    so its kernel must have been registered under the config key. If not, the comparison above
    was CUDA against itself."""
    assert (
        tvm_ffi.get_global_func(key, allow_missing=True) is not None
    ), f"{tag}: CuTeDSL kernel not registered under {key}, so the backend fell back to CUDA"


@pytest.mark.parametrize("shape", MATRIX_SIZES, ids=get_shape_id)
@pytest.mark.parametrize("q_layout", Q_LAYOUTS, ids=get_layout_id)
def test_quantize(q_layout, shape):
    """The CuTeDSL and CUDA backends produce bit-identical output for the same input."""
    x = make_input(shape)
    tag = f"{get_shape_id(shape)}/{get_layout_id(q_layout)}"

    cuda_parts, _ = quantize_with("cuda", x, q_layout)
    cutedsl_parts, _ = quantize_with("cutedsl", x, q_layout)

    assert_registered(cutedsl_key(q_layout), tag)
    for name, cuda_bytes in cuda_parts.items():
        assert np.array_equal(cutedsl_parts[name], cuda_bytes), f"{tag}: {name} differs"


@pytest.mark.parametrize("q_layout", Q_LAYOUTS, ids=get_layout_id)
def test_stochastic_rounding(q_layout):
    """Stochastic rounding is not compared byte for byte against CUDA: which random bits an
    element gets follows from the work decomposition, so the two implementations may legitimately
    disagree. What is checked is the property that defines it. The input is built so every scaling
    block carries one exact 6.0 (making the encode coefficient exactly 1.0 in both directions) and
    probe values of 2.75, which sit 3/4 of the way from 2 to 3 on the E2M1 lattice, so a correct
    rounder sends them to 3 with probability 0.75. Determinism for a fixed RNG state is asserted
    too, since it is what makes an SR run reproducible.
    """
    rows, cols = 256, 1024
    tag = f"sr/{get_layout_id(q_layout)}"
    x = jnp.full((rows, cols), 2.75, jnp.bfloat16)
    x = x.at[::NVFP4_BLOCK_SIZE, :].set(6.0).at[:, ::NVFP4_BLOCK_SIZE].set(6.0)
    probe = np.ones((rows, cols), dtype=bool)
    probe[::NVFP4_BLOCK_SIZE, :] = False
    probe[:, ::NVFP4_BLOCK_SIZE] = False

    rng_state = jnp.asarray([[1, 2, 3, 4]], dtype=jnp.uint32)
    parts, dequantized = quantize_with("cutedsl", x, q_layout, rng_state)
    parts_again, _ = quantize_with("cutedsl", x, q_layout, rng_state)

    assert_registered(cutedsl_key(q_layout, stochastic_rounding=True), tag)
    for name, first in parts.items():
        assert np.array_equal(parts_again[name], first), f"{tag}: {name} is not deterministic"

    for name, values in dequantized.items():
        # The colwise direction holds the transposed tensor, so the probe mask follows the shape.
        picked = values[probe if values.shape == probe.shape else probe.T]
        # The block scales work out to exactly 1.0, so a probe dequantizes to its E2M1 value.
        assert np.all((picked == 2.0) | (picked == 3.0)), f"{tag}: {name} probes left {{2.0, 3.0}}"
        up_fraction = np.mean(picked == 3.0)
        assert (
            abs(up_fraction - 0.75) < 0.02
        ), f"{tag}: {name} rounded 2.75 up with frequency {up_fraction:.4f}, expected 0.75"
