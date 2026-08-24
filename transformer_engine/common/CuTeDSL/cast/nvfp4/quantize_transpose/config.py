# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Configuration for the variants of the NVFP4QuantizeTransposeTuned1DKernel"""


class NVFP4QuantizeConfig:
    """Instantiation parameters of the CuTe DSL kernel.

    Analogous to the template parameters of the CUDA kernel: each config is a separately
    compiled kernel variant.

    use_stochastic_rounding: use RN or SR rounding?
    use_fast_math: perform scaling of values in bf16 instead of fp32?
    row_scaled_nvfp4: use per-row (and per-column) instead of per-tensor fp32 scaling factors?
    return_transpose: also return the quantized transposition of the input?
    is_noop: is a cast-noop flag tensor passed to the kernel?
    """

    def __init__(
        self,
        use_stochastic_rounding: bool,
        use_fast_math: bool,
        row_scaled_nvfp4: bool,
        return_transpose: bool,
        is_noop: bool,
    ):
        self.USE_STOCHASTIC_ROUNDING = use_stochastic_rounding
        self.USE_FAST_MATH = use_fast_math
        self.ROW_SCALED_NVFP4 = row_scaled_nvfp4
        self.RETURN_TRANSPOSE = return_transpose
        self.IS_NOOP = is_noop
