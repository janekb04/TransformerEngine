# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Philox random-number generation for CuTeDSL kernels. Based on curanddx.hpp"""

import os

from cutlass import Uint32, Uint64

from transformer_engine.common.CuTeDSL.utils import (
    bool_to_u64,
    u64_hi32,
    u64_lo32,
    umulhi_u32,
)


# per envvars.rst
NUM_PHILOX_ROUNDS = int(os.environ.get("NVTE_BUILD_NUM_PHILOX_ROUNDS", "10"))

# per curanddx.hpp
PHILOX_W32_0 = 0x9E3779B9
PHILOX_W32_1 = 0xBB67AE85
PHILOX_M4x32_0 = 0xD2511F53
PHILOX_M4x32_1 = 0xCD9E8D57


class PhiloxRng:
    """Trace-time replica of the curanddx.hpp philox4x32 generator"""

    def __init__(self, seed: Uint64, subsequence: Uint64, offset: Uint64):
        self._key = (u64_lo32(seed), u64_hi32(seed))
        self._ctr_lo = offset
        self._ctr_hi = subsequence
        self._buf = []

    def _generate4(self):
        c = [
            u64_lo32(self._ctr_lo),
            u64_hi32(self._ctr_lo),
            u64_lo32(self._ctr_hi),
            u64_hi32(self._ctr_hi),
        ]
        k0, k1 = self._key
        for round_idx in range(NUM_PHILOX_ROUNDS):
            rk0 = k0 + Uint32(round_idx * PHILOX_W32_0)
            rk1 = k1 + Uint32(round_idx * PHILOX_W32_1)
            # hi and lo multiplies should fuse into a single wide instruction
            lo0 = Uint32(PHILOX_M4x32_0) * c[0]
            hi0 = umulhi_u32(Uint32(PHILOX_M4x32_0), c[0])
            lo1 = Uint32(PHILOX_M4x32_1) * c[2]
            hi1 = umulhi_u32(Uint32(PHILOX_M4x32_1), c[2])
            # should become a LOP3
            c = [hi1 ^ c[1] ^ rk0, lo1, hi0 ^ c[3] ^ rk1, lo0]
        self._buf.extend(c)
        new_lo = self._ctr_lo + Uint64(1)
        self._ctr_hi = self._ctr_hi + bool_to_u64(new_lo == Uint64(0))
        self._ctr_lo = new_lo

    def get_rbits(self) -> Uint32:
        """Fetch the next Uint32 of random bits"""
        if not self._buf:
            self._generate4()
        return self._buf.pop(0)

    def prefetch(self, words: int):
        """Pregenerate fours of Uint32 so at least `words` are available"""
        while len(self._buf) < words:
            self._generate4()
