# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Device and host-launch logic for the tuned NVFP4 quantize-transpose kernel.

The kernel is bitwise-identical to quantize_transpose_nvfp4_tuned_1D_kernel.
"""

from typing import Optional

import cutlass
from cutlass import cute
from cutlass import (
    BFloat16,
    Float4E2M1FN,
    Float8E4M3FN,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint32,
    Uint64,
)
from cuda.bindings.driver import CUstream  # pylint: disable=no-name-in-module

from transformer_engine.common.CuTeDSL import iket
from transformer_engine.common.CuTeDSL.cast.nvfp4.utils import (
    NVFP4_BLOCK_SCALING_SIZE,
    NVFP4_SCALE_PAD_INNER,
    NVFP4_SCALE_PAD_OUTER,
    compute_block_decode_sf,
    compute_block_encode_sf,
    compute_global_encode_sf,
    mul2_cvt_bf16x8_hi_to_fp4x8,
    mul2_cvt_bf16x8_hi_to_fp4x8_sr,
    mul2_cvt_bf16x8_lo_to_fp4x8,
    mul2_cvt_bf16x8_lo_to_fp4x8_sr,
    mul2_cvt_bf16x8_to_fp4x8,
    mul2_cvt_bf16x8_to_fp4x8_sr,
    mul_cvt_bf16x8_to_fp4x8,
    mul_cvt_bf16x8_to_fp4x8_sr,
    pack_f32x2,
    st_global_b64,
)
from transformer_engine.common.CuTeDSL.philox_rng import PhiloxRng
from transformer_engine.common.CuTeDSL.utils import (
    CUTEDSL_DEBUG_LOGGING,
    fabs_f32,
    make_prmt_u32,
    pack_u32x2,
    packed16_kit,
    validate_tensor,
)

# Object containing some functions operating on bf16x2 values
bf16_kit = packed16_kit(BFloat16)

# {a_hi, a_lo}, {b_hi, b_lo} -> {a_lo, a_lo}
prmt_lo_u32 = make_prmt_u32(0x5410)
# {a_hi, a_lo}, {b_hi, b_lo} -> {a_hi, a_hi}
prmt_hi_u32 = make_prmt_u32(0x7632)

# The colwise pass picks its converter by wave: wave 0 owns the low bf16 of each of the eight
# row registers it holds, wave 1 the high one.
mul2_cvt_col = (mul2_cvt_bf16x8_lo_to_fp4x8, mul2_cvt_bf16x8_hi_to_fp4x8)
mul2_cvt_col_sr = (mul2_cvt_bf16x8_lo_to_fp4x8_sr, mul2_cvt_bf16x8_hi_to_fp4x8_sr)


def _abs_max_tree(vals):
    """Reduce vals with abs_max_x2 in a tree fashion."""
    while len(vals) > 1:
        vals = [
            bf16_kit.abs_max_x2(vals[2 * i], vals[2 * i + 1]) for i in range(len(vals) // 2)
        ] + ([vals[-1]] if len(vals) % 2 else [])
    return vals[0]


class NVFP4QuantizeTransposeTuned1DKernel:
    """Quantizes a bf16 tensor to NVFP4. Supported variants described by NVFP4QuantizeConfig.

    Each thread block processes a 128x128 _chunk_ of the input tensor as four 64x64 _tiles_,
    walked sequentially with PREFETCH_STAGES tiles prefetched beyond the current one. TMA loads
    a tile to SMEM, the CTA quantizes it into staged SMEM output buffers, and TMA stores those while
    the next tile loads. Block scale factors are staged in SMEM for multiple tiles and
    flushed with vectorized predicated stores.
    """

    CHUNK_DIM_Y = 128
    CHUNK_DIM_X = 128
    TILE_DIM = 64
    THREADS = 128
    PREFETCH_STAGES = 1

    # Derived tiling constants (names follow the CUDA kernel).
    NUM_BUFFERS = PREFETCH_STAGES + 1
    STAGES_Y = CHUNK_DIM_Y // TILE_DIM  # 2
    STAGES_X = CHUNK_DIM_X // TILE_DIM  # 2
    STAGES = STAGES_Y * STAGES_X  # 4
    SCALES_PER_TILE = TILE_DIM // NVFP4_BLOCK_SCALING_SIZE  # 4
    SCALES_PER_CHUNK_X = CHUNK_DIM_X // NVFP4_BLOCK_SCALING_SIZE  # 8
    SCALES_PER_CHUNK_Y = CHUNK_DIM_Y // NVFP4_BLOCK_SCALING_SIZE  # 8
    THREADS_X_ROWWISE = TILE_DIM // NVFP4_BLOCK_SCALING_SIZE  # 4
    THREADS_Y_ROWWISE = THREADS // THREADS_X_ROWWISE  # 32
    ITERATIONS_ROWWISE = TILE_DIM // THREADS_Y_ROWWISE  # 2
    PACK_SIZE = 8  # elements per vectorized SMEM access
    WAVES = NVFP4_BLOCK_SCALING_SIZE // PACK_SIZE  # 2
    # Threads that span the 32 4-byte SMEM banks at 16 bf16 per thread (the rowwise stagger).
    THREADS_PER_BANK = (32 * 4 * 8) // 4 // NVFP4_BLOCK_SCALING_SIZE  # 16
    assert NUM_BUFFERS <= STAGES  # otherwise, prefetch loop would read OOB

    def __init__(self, cfg):
        self.cfg = cfg

        # The switches below are per-variant tuning choices, measured on a GB200.

        # Stage the input through SMEM with the TMA 128B swizzle: it removes the read passes'
        # bank conflicts (an SMEM row of a 64x64 bf16 tile is exactly 128B, so rows alias
        # banks) at the cost of an XOR in every SMEM address. Only pays for RN + transpose.
        self.SWIZZLE_INPUT = cfg.RETURN_TRANSPOSE and not cfg.USE_STOCHASTIC_ROUNDING

        # A single staged-output buffer saves 2-4KB SMEM per CTA (one more resident CTA per
        # SM) at the cost of a second barrier per stage. That trade wins for the SR and
        # row-scaled configs, the barrier-light double-buffered loop for the rest.
        self.NUM_BUFFERS_OUT = 1 if (cfg.USE_STOCHASTIC_ROUNDING or cfg.ROW_SCALED_NVFP4) else 2

        # Single-buffered outputs put a barrier at the top of every stage, which makes it safe
        # to flush the rowwise scales one 64-row half at a time; halving the staging is what
        # fits the extra CTA per SM.
        self.HALF_SS_ROW = self.NUM_BUFFERS_OUT == 1
        self.SS_ROW_ROWS = self.TILE_DIM if self.HALF_SS_ROW else self.CHUNK_DIM_Y

        # Random words generated ahead of each stage's mbarrier wait (SR only): exactly one
        # stage's consumption per thread, so the Philox ALU work hides under the TMA wait.
        self.RNG_PREFETCH = 16 if cfg.RETURN_TRANSPOSE else 8

    # Host-side kernel launch
    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (M, N) bf16 input
        mO_row: cute.Tensor,  # (M, N) fp4 rowwise output
        mS_row: cute.Tensor,  # (roundup(M, 128), roundup(ceil(N / 16), 4)) e4m3 scales
        mO_col: Optional[cute.Tensor],  # (N, M) fp4 transposed output
        mS_col: Optional[cute.Tensor],  # (roundup(N, 128), roundup(ceil(M / 16), 4)) e4m3 scales
        mAmaxRow: cute.Tensor,  # (M,) f32 per-row amax if ROW_SCALED_NVFP4, else (1,) global amax
        mAmaxCol: Optional[cute.Tensor],  # (N,) f32 per-column amax if ROW_SCALED_NVFP4, else (1,)
        mNoop: Optional[cute.Tensor],  # (1,) f32 cast-noop flag, present iff IS_NOOP
        mRngState: Optional[cute.Tensor],  # (2,) i64 Philox {seed, offset}
        stream: CUstream,
    ):
        """AOT-compiled host entrypoint. `quantize_transpose_nvfp4_cutedsl.cuh` passes these
        arguments in this exact order via tvm-ffi, and the config fixes which of the optional
        ones are present (see `compile_cutedsl_function_from_cfg`).

        M, N are the input's *flattened* 2D dims, both multiples of NVFP4_SHAPE_ALIGNMENT; a
        rank > 2 input already arrives flattened.
        All tensors are row-major (with rightmost stride 1).
        FP4 extents are logical element counts, not the halved extents of the uint8 buffer TE
        actually allocates.
        """
        if cutlass.const_expr(CUTEDSL_DEBUG_LOGGING):
            cute.printf(
                "[CuTeDSL] NVFP4QuantizeTransposeTuned1DKernel.__call__() with config:"
                f" {self.cfg}\n"
            )

        ## Validation

        # Validate input and output tensor layouts
        M, N = mX.shape
        mX_layout = cute.make_ordered_layout((M, N), order=(1, 0))
        mO_row_layout = mX_layout
        mO_col_layout = cute.make_ordered_layout((N, M), order=(1, 0))
        validate_tensor(mX, mX_layout, BFloat16)
        validate_tensor(mO_row, mO_row_layout, Float4E2M1FN)
        validate_tensor(mO_col, mO_col_layout, Float4E2M1FN)

        # Validate scaling factor tensor layouts
        mS_row_layout = cute.make_ordered_layout(
            (
                cute.round_up(M, NVFP4_SCALE_PAD_OUTER),
                cute.round_up(cute.ceil_div(N, NVFP4_BLOCK_SCALING_SIZE), NVFP4_SCALE_PAD_INNER),
            ),
            order=(1, 0),
        )
        mS_col_layout = cute.make_ordered_layout(
            (
                cute.round_up(N, NVFP4_SCALE_PAD_OUTER),
                cute.round_up(cute.ceil_div(M, NVFP4_BLOCK_SCALING_SIZE), NVFP4_SCALE_PAD_INNER),
            ),
            order=(1, 0),
        )
        validate_tensor(mS_row, mS_row_layout, Float8E4M3FN)
        validate_tensor(mS_col, mS_col_layout, Float8E4M3FN)

        # Validate amax tensor layouts
        if cutlass.const_expr(self.cfg.ROW_SCALED_NVFP4):
            validate_tensor(mAmaxRow, cute.make_layout((M,)), Float32)
            validate_tensor(mAmaxCol, cute.make_layout((N,)), Float32)
        else:
            validate_tensor(mAmaxRow, cute.make_layout((1,)), Float32)
            validate_tensor(mAmaxCol, cute.make_layout((1,)), Float32)

        # Validate RNG state tensor layout
        validate_tensor(mRngState, cute.make_layout((2,)), Int64)

        # Validate cast-noop flag tensor layout
        validate_tensor(mNoop, cute.make_layout((1,)), Float32)

        ## TMA descriptors
        # The FP4 outputs are moved as bytes: two elements to a byte, so the byte view of the
        # (M, N) rowwise output is (M, N/2) and a 64x64-element tile is a 64x32-byte box. The
        # 32-divisibility of both dims is what keeps every row stride a multiple of 16B as TMA
        # requires: 2*N bytes for the bf16 input, N/2 and M/2 for the fp4 outputs.
        mO_row_bytes = cute.recast_tensor(mO_row, Uint8)

        tile_in_layout = cute.make_ordered_layout((self.TILE_DIM, self.TILE_DIM), order=(1, 0))
        if cutlass.const_expr(self.SWIZZLE_INPUT):
            # The TMA 128B swizzle (Swizzle<3,4,3> on byte offsets): the 16-byte unit index
            # within a 128B row is XORed with row % 8. The compute passes fold the same XOR
            # into their manual indexing.
            tile_in_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 4, 3), 0, tile_in_layout
            )
        tile_out_layout = cute.make_ordered_layout(
            (self.TILE_DIM, self.TILE_DIM // 2), order=(1, 0)
        )

        op_load = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        op_store = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_in, tma_view_in = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_load, mX, tile_in_layout, (self.TILE_DIM, self.TILE_DIM), num_multicast=1
        )
        tma_atom_row, tma_view_row = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op_store,
            mO_row_bytes,
            tile_out_layout,
            (self.TILE_DIM, self.TILE_DIM // 2),
            num_multicast=1,
        )
        if cutlass.const_expr(self.cfg.RETURN_TRANSPOSE):
            mO_col_bytes = cute.recast_tensor(mO_col, Uint8)
            tma_atom_col, tma_view_col = cute.nvgpu.cpasync.make_tiled_tma_atom(
                op_store,
                mO_col_bytes,
                tile_out_layout,
                (self.TILE_DIM, self.TILE_DIM // 2),
                num_multicast=1,
            )
        else:
            tma_atom_col = None
            tma_view_col = None

        ## Grid: one CTA per chunk; X indexes columns and Y rows, as in the CUDA kernel's
        ## ctaid_X / ctaid_Y.
        grid = [
            cute.ceil_div(N, self.CHUNK_DIM_X),
            cute.ceil_div(M, self.CHUNK_DIM_Y),
            1,
        ]

        self.kernel(
            mX,
            mS_row,
            mS_col,
            mAmaxRow,
            mAmaxCol,
            mNoop,
            mRngState,
            tma_atom_in,
            tma_view_in,
            tma_atom_row,
            tma_view_row,
            tma_atom_col,
            tma_view_col,
        ).launch(grid=grid, block=[self.THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mS_row: cute.Tensor,
        mS_col: Optional[cute.Tensor],
        mAmaxRow: cute.Tensor,
        mAmaxCol: Optional[cute.Tensor],
        mNoop: Optional[cute.Tensor],
        mRngState: Optional[cute.Tensor],
        tma_atom_in: cute.CopyAtom,
        tma_view_in: cute.Tensor,
        tma_atom_row: cute.CopyAtom,
        tma_view_row: cute.Tensor,
        tma_atom_col: Optional[cute.CopyAtom],
        tma_view_col: Optional[cute.Tensor],
    ):
        """Device entry for the NVFP4 tuned-1D quantize-transpose kernel."""
        # Whether a cast-noop flag was passed at all is fixed when the variant is traced, so
        # only the IS_NOOP=True variant carries the load and the check.
        skip_execution = cutlass.const_expr(self.cfg.IS_NOOP) and mNoop[0] == Float32(1.0)
        if not skip_execution:
            self.kernel_main(
                mX,
                mS_row,
                mS_col,
                mAmaxRow,
                mAmaxCol,
                mRngState,
                tma_atom_in,
                tma_view_in,
                tma_atom_row,
                tma_view_row,
                tma_atom_col,
                tma_view_col,
            )

    @cute.jit
    def kernel_main(
        self,
        mX: cute.Tensor,
        mS_row: cute.Tensor,
        mS_col: Optional[cute.Tensor],
        mAmaxRow: cute.Tensor,
        mAmaxCol: Optional[cute.Tensor],
        mRngState: Optional[cute.Tensor],
        tma_atom_in: cute.CopyAtom,
        tma_view_in: cute.Tensor,
        tma_atom_row: cute.CopyAtom,
        tma_view_row: cute.Tensor,
        tma_atom_col: Optional[cute.CopyAtom],
        tma_view_col: Optional[cute.Tensor],
    ):
        """The kernel body, split out so the cast-noop flag can skip it wholesale."""
        # -- Trace time --
        cfg = self.cfg
        TILE = self.TILE_DIM

        # Shared memory layout
        if cutlass.const_expr(cfg.RETURN_TRANSPOSE):

            @cute.struct
            class SharedStorage:  # pylint: disable=missing-class-docstring
                mbar_storage: cute.struct.MemRange[cute.Int64, self.NUM_BUFFERS]
                sX: cute.struct.Align[
                    # 1024B: the TMA 128B swizzle's period; SW128 requires the staging base
                    # aligned to it or the hardware XOR pattern is phase-shifted.
                    cute.struct.MemRange[BFloat16, TILE * TILE * self.NUM_BUFFERS],
                    1024,
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE * (TILE // 2) * self.NUM_BUFFERS_OUT], 128
                ]
                sO_col: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE * (TILE // 2) * self.NUM_BUFFERS_OUT], 128
                ]
                sS_row: cute.struct.Align[
                    cute.struct.MemRange[Float8E4M3FN, self.SS_ROW_ROWS * self.SCALES_PER_CHUNK_X],
                    16,
                ]
                sS_col: cute.struct.Align[
                    cute.struct.MemRange[Float8E4M3FN, self.CHUNK_DIM_X * self.SCALES_PER_CHUNK_Y],
                    16,
                ]

        else:

            @cute.struct
            class SharedStorage:  # pylint: disable=missing-class-docstring
                mbar_storage: cute.struct.MemRange[cute.Int64, self.NUM_BUFFERS]
                sX: cute.struct.Align[
                    cute.struct.MemRange[BFloat16, TILE * TILE * self.NUM_BUFFERS], 1024
                ]
                sO_row: cute.struct.Align[
                    cute.struct.MemRange[Uint8, TILE * (TILE // 2) * self.NUM_BUFFERS_OUT], 128
                ]
                sS_row: cute.struct.Align[
                    cute.struct.MemRange[Float8E4M3FN, self.SS_ROW_ROWS * self.SCALES_PER_CHUNK_X],
                    16,
                ]

        # "Allocate" shared memory
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # Row-scaled: per-row (and, with a transpose, per-column) encode scales, computed once
        # per chunk (see below).
        if cutlass.const_expr(cfg.ROW_SCALED_NVFP4):
            sSEnc = smem.allocate_tensor(
                Float32, cute.make_layout((self.CHUNK_DIM_Y,)), byte_alignment=16
            )
        else:
            sSEnc = None
        if cutlass.const_expr(cfg.ROW_SCALED_NVFP4 and cfg.RETURN_TRANSPOSE):
            sSEncCol = smem.allocate_tensor(
                Float32, cute.make_layout((self.CHUNK_DIM_X,)), byte_alignment=16
            )
        else:
            sSEncCol = None

        # Create views into the shared memory
        def buffered_tile(inner_cols, buffers):
            return cute.make_layout(
                ((TILE, inner_cols), buffers),
                stride=((inner_cols, 1), TILE * inner_cols),
            )

        # A swizzled view for TMA (matching the layout the input TMA atom was built with) and
        # a raw base pointer for the compute passes, which fold the swizzle XOR into their
        # manual flat indexing instead of going through a swizzled pointer.
        if cutlass.const_expr(self.SWIZZLE_INPUT):
            sX = storage.sX.get_tensor(
                buffered_tile(TILE, self.NUM_BUFFERS), swizzle=cute.make_swizzle(3, 4, 3)
            )
        else:
            sX = storage.sX.get_tensor(buffered_tile(TILE, self.NUM_BUFFERS))
        sX_base = storage.sX.data_ptr()
        sO_row = storage.sO_row.get_tensor(buffered_tile(TILE // 2, self.NUM_BUFFERS_OUT))
        sS_row = storage.sS_row.get_tensor(
            cute.make_layout(
                (self.SS_ROW_ROWS, self.SCALES_PER_CHUNK_X), stride=(self.SCALES_PER_CHUNK_X, 1)
            )
        )
        if cutlass.const_expr(cfg.RETURN_TRANSPOSE):
            sO_col = storage.sO_col.get_tensor(buffered_tile(TILE // 2, self.NUM_BUFFERS_OUT))
            sS_col = storage.sS_col.get_tensor(
                cute.make_layout(
                    (self.CHUNK_DIM_X, self.SCALES_PER_CHUNK_Y),
                    stride=(self.SCALES_PER_CHUNK_Y, 1),
                )
            )
        else:
            sO_col = None
            sS_col = None

        # Bind GMEM and SMEM tensor views for TMA
        gX_tiled = cute.zipped_divide(tma_view_in, (TILE, TILE))
        tXsX, tXgX = cute.nvgpu.cpasync.tma_partition(
            tma_atom_in, 0, cute.make_layout(1), sX, gX_tiled
        )
        gO_row_tiled = cute.zipped_divide(tma_view_row, (TILE, TILE // 2))
        tOsO_row, tOgO_row = cute.nvgpu.cpasync.tma_partition(
            tma_atom_row, 0, cute.make_layout(1), sO_row, gO_row_tiled
        )
        if cutlass.const_expr(cfg.RETURN_TRANSPOSE):
            gO_col_tiled = cute.zipped_divide(tma_view_col, (TILE, TILE // 2))
            tOsO_col, tOgO_col = cute.nvgpu.cpasync.tma_partition(
                tma_atom_col, 0, cute.make_layout(1), sO_col, gO_col_tiled
            )

        # -- Runtime --
        rows, cols = mX.shape
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()  # (chunk_x, chunk_y)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        ik_kernel = iket.range_push("Kernel")
        ik_prologue = iket.range_push("Prologue")

        # Prefetch TMA descriptor of the input
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_in)

        # Row-scaled: each thread derives one chunk row's (and column's) encode scale from its
        # amax, once per chunk instead of once per owned block, since a block's scale is consumed
        # by several (thread, stage) pairs. The mbarrier-init syncthreads below publishes the
        # slabs before their first use. An out-of-bounds row/column clamps to the last valid
        # amax; nothing it produces is observable -- its data is dropped by the output TMA and
        # its scales are not flushed.
        if cutlass.const_expr(cfg.ROW_SCALED_NVFP4):
            enc_row = bidy * self.CHUNK_DIM_Y + tidx
            sSEnc[tidx] = compute_global_encode_sf(mAmaxRow[cutlass.min(enc_row, rows - 1)])
        if cutlass.const_expr(cfg.ROW_SCALED_NVFP4 and cfg.RETURN_TRANSPOSE):
            enc_col = bidx * self.CHUNK_DIM_X + tidx
            sSEncCol[tidx] = compute_global_encode_sf(mAmaxCol[cutlass.min(enc_col, cols - 1)])

        # Initialize mbarriers for TMA G2S input tensor copy
        mbar = storage.mbar_storage.data_ptr()
        if warp_idx == 0:
            with cute.arch.elect_one():
                for b in cutlass.range_constexpr(self.NUM_BUFFERS):
                    cute.arch.mbarrier_init(mbar + b, 1)
                # release mbarrier_init from elected thread to all threads in cluster (here: CTA)
                cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()  # acquire mbarrier_init by CTA (cluster)

        # Prefetch NUM_BUFFERS=PREFETCH_STAGES tiles + the first tile to process
        tx_count = TILE * TILE * 2  # each load is a TILE x TILE x bf16 (2 bytes) tile
        if warp_idx == 0:
            for s in cutlass.range_constexpr(self.NUM_BUFFERS):
                tile_coord = (
                    bidy * self.STAGES_Y + s // self.STAGES_X,
                    bidx * self.STAGES_X + s % self.STAGES_X,
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar + s, tx_count)
                cute.copy(
                    tma_atom_in,
                    tXgX[(None, tile_coord)],
                    tXsX[(None, s)],
                    tma_bar_ptr=mbar + s,
                )

        # Compute global encode scales from supplied amax
        if cutlass.const_expr(not cfg.ROW_SCALED_NVFP4):
            S_enc_rowwise = compute_global_encode_sf(mAmaxRow[0])
        else:
            # Per-row encode scales are drawn from sSEnc inside the rowwise pass instead.
            S_enc_rowwise = Float32(1.0)
        if cutlass.const_expr(cfg.RETURN_TRANSPOSE and not cfg.ROW_SCALED_NVFP4):
            S_enc_colwise = compute_global_encode_sf(mAmaxCol[0])
        else:
            # Per-column encode scales are drawn from sSEncCol inside the colwise pass instead.
            S_enc_colwise = Float32(1.0)

        # Construct RNG state for SR
        rng = None
        if cutlass.const_expr(cfg.USE_STOCHASTIC_ROUNDING):
            grid_dim_x, _, _ = cute.arch.grid_dim()
            # Contrary to CUDA C++ version, calculate in Int64 for correctness
            rng_sequence = (
                Int64(tidx)
                + Int64(bidx) * self.THREADS
                + Int64(bidy) * Int64(grid_dim_x) * self.THREADS
            )
            rng = PhiloxRng(
                seed=Uint64(mRngState[0].ir_value()),
                subsequence=Uint64(rng_sequence.ir_value()),
                offset=Uint64(mRngState[1].ir_value()),
            )

        iket.range_pop(ik_prologue)

        # Main loop over tiles/stages in chunk
        for stage in cutlass.range_constexpr(self.STAGES):
            stage_y = stage // self.STAGES_X
            stage_x = stage % self.STAGES_X
            parity = (stage // self.NUM_BUFFERS) % 2
            buf = stage % self.NUM_BUFFERS
            buf_out = stage % self.NUM_BUFFERS_OUT

            ik_stage = iket.range_push("Stage")

            if cutlass.const_expr(self.NUM_BUFFERS_OUT == 1):
                ik_store_wait = iket.range_push("StoreWait", level=2)
                # Single-buffered outputs: the previous stage's TMA store must have finished
                # reading the buffer before this stage's quantization overwrites it.
                if warp_idx == 0:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.sync_threads()
                iket.range_pop(ik_store_wait)

            # Generate this stage's random words before blocking on the tile's arrival: the
            # Philox ALU work then overlaps the TMA wait instead of serializing with the
            # conversions. Word values and consumption order are unchanged (bitwise-equal SR).
            if cutlass.const_expr(cfg.USE_STOCHASTIC_ROUNDING):
                ik_philox = iket.range_push("Philox", level=2)
                rng.prefetch(self.RNG_PREFETCH)
                iket.range_pop(ik_philox)

            # Wait for TMA G2S tile load to complete
            ik_tma_wait = iket.range_push("TmaWait", level=2)
            cute.arch.mbarrier_wait(mbar + buf, parity)
            iket.range_pop(ik_tma_wait)

            # Unswizzled per-buffer view for the compute passes (they add the XOR themselves).
            sX_tile = cute.make_tensor(
                sX_base + buf * (TILE * TILE),
                cute.make_ordered_layout((TILE, TILE), order=(1, 0)),
            )

            ik_rowwise = iket.range_push("Rowwise", level=2)
            self.rowwise_tile(
                sX_tile,
                sO_row[(None, buf_out)],
                sS_row,
                sSEnc,
                S_enc_rowwise,
                stage_y,
                stage_x,
                rng,
            )
            iket.range_pop(ik_rowwise)
            if cutlass.const_expr(cfg.RETURN_TRANSPOSE):
                ik_colwise = iket.range_push("Colwise", level=2)
                self.colwise_tile(
                    sX_tile,
                    sO_col[(None, buf_out)],
                    sS_col,
                    sSEncCol,
                    S_enc_colwise,
                    stage_y,
                    stage_x,
                    rng,
                )
                iket.range_pop(ik_colwise)

            ik_sync = iket.range_push("StageSync", level=2)
            # Make the SMEM output writes visible to the TMA async proxy. The dedicated
            # fence.proxy.async.shared::cta, not the far costlier generic membar
            # cute.arch.fence_proxy would emit.
            cute.arch.fence_view_async_shared()
            if cutlass.const_expr(self.NUM_BUFFERS_OUT > 1):
                # Before the barrier: wait for every prior TMA store to have finished reading
                # its staged output buffer -- the buffer stage t+1 writes is the one that
                # store(t-1) read. (With a single buffer this wait already happened at the
                # top of the stage.)
                if warp_idx == 0:
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
            # The barrier broadcasts the store-wait to all threads, orders this stage's SMEM
            # output writes before the TMA store below, and makes the input buffer's reads
            # visible before it is refilled below.
            cute.arch.sync_threads()
            iket.range_pop(ik_sync)

            if cutlass.const_expr(self.HALF_SS_ROW and stage_x == self.STAGES_X - 1):
                ik_flush_row = iket.range_push("FlushRow", level=2)
                # This 64-row half of the chunk is complete: flush its staged rowwise scales
                # now (the sync above ordered every thread's scale writes). The next stage's
                # top-of-stage barrier orders this flush before the staging is overwritten.
                self.flush_scales(
                    sS_row,
                    mS_row,
                    bidy * self.CHUNK_DIM_Y + stage_y * self.TILE_DIM,
                    bidx * self.SCALES_PER_CHUNK_X,
                    rows,
                    cols,
                    self.SS_ROW_ROWS,
                )
                iket.range_pop(ik_flush_row)

            if warp_idx == 0:
                ik_tma_store = iket.range_push("TmaStore", level=2)
                # Store this tile's outputs. The transposed tile of chunk-relative tile (y, x)
                # lands at tile (x, y) of the transposed tensor.
                cute.copy(
                    tma_atom_row,
                    tOsO_row[(None, buf_out)],
                    tOgO_row[
                        (
                            None,
                            (bidy * self.STAGES_Y + stage_y, bidx * self.STAGES_X + stage_x),
                        )
                    ],
                )
                if cutlass.const_expr(cfg.RETURN_TRANSPOSE):
                    cute.copy(
                        tma_atom_col,
                        tOsO_col[(None, buf_out)],
                        tOgO_col[
                            (
                                None,
                                (bidx * self.STAGES_X + stage_x, bidy * self.STAGES_Y + stage_y),
                            )
                        ],
                    )
                cute.arch.cp_async_bulk_commit_group()
                iket.range_pop(ik_tma_store)

                # Refill the buffer this stage just consumed: every thread's reads of it
                # finished before the syncthreads above.
                if cutlass.const_expr(stage + self.NUM_BUFFERS < self.STAGES):
                    ik_refill = iket.range_push("TmaRefill", level=2)
                    next_stage = stage + self.NUM_BUFFERS
                    tile_coord = (
                        bidy * self.STAGES_Y + next_stage // self.STAGES_X,
                        bidx * self.STAGES_X + next_stage % self.STAGES_X,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar + buf, tx_count)
                    cute.copy(
                        tma_atom_in,
                        tXgX[(None, tile_coord)],
                        tXsX[(None, buf)],
                        tma_bar_ptr=mbar + buf,
                    )
                    iket.range_pop(ik_refill)

            iket.range_pop(ik_stage)

        ik_epilogue = iket.range_push("Epilogue")
        ## Epilogue: flush the staged scales not already flushed mid-loop. The last stage's
        ## syncthreads ordered every thread's scale writes before this point.
        if cutlass.const_expr(not self.HALF_SS_ROW):
            ik_flush_row_end = iket.range_push("FlushRow", level=2)
            self.flush_scales(
                sS_row,
                mS_row,
                bidy * self.CHUNK_DIM_Y,
                bidx * self.SCALES_PER_CHUNK_X,
                rows,
                cols,
                self.SS_ROW_ROWS,
            )
            iket.range_pop(ik_flush_row_end)
        if cutlass.const_expr(cfg.RETURN_TRANSPOSE):
            ik_flush_col = iket.range_push("FlushCol", level=2)
            self.flush_scales(
                sS_col,
                mS_col,
                bidx * self.CHUNK_DIM_X,
                bidy * self.SCALES_PER_CHUNK_Y,
                cols,
                rows,
                self.CHUNK_DIM_X,
            )
            iket.range_pop(ik_flush_col)

        ik_drain = iket.range_push("StoreDrain", level=2)
        # Wait for in-flight TMA stores before the kernel returns.
        cute.arch.cp_async_bulk_wait_group(0, read=False)
        iket.range_pop(ik_drain)
        iket.range_pop(ik_epilogue)
        iket.range_pop(ik_kernel)

    @cute.jit
    def rowwise_tile(
        self,
        sX_tile: cute.Tensor,  # (TILE, TILE) bf16 SMEM input tile
        sO_row_tile: cute.Tensor,  # (TILE, TILE/2) u8 SMEM staged rowwise output
        sS_row: cute.Tensor,  # (SS_ROW_ROWS, SCALES_PER_CHUNK_X) e4m3 SMEM staged scales
        sSEnc: Optional[cute.Tensor],  # (CHUNK_DIM_Y,) f32 per-row encode scales (ROW_SCALED)
        S_enc: Float32,  # global rowwise encode scale (ignored when ROW_SCALED)
        stage_y: cutlass.Constexpr,
        stage_x: cutlass.Constexpr,
        rng,
    ):
        """Quantize one SMEM tile rowwise: a thread owns one 16-element scaling block per
        iteration, so the block amax never leaves the thread. Mirrors rowwise_scaling in the
        CUDA kernel, including the bank-group stagger of the two 16-byte reads per block."""
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        bank_group = lane // self.THREADS_PER_BANK  # which 16-byte wave this thread starts on
        tid_y = tidx // self.THREADS_X_ROWWISE
        tid_x = tidx % self.THREADS_X_ROWWISE

        if cutlass.const_expr(cfg.USE_FAST_MATH):
            sf_type = BFloat16
        else:
            sf_type = Float32

        # The tile as (row * 8-element group, element): a flat first mode so a slice needs only
        # one dynamic index, each group one 16-byte vectorized SMEM access. A fresh view on the
        # tile's pointer rather than cute.composition, whose right operand maps into the tile's
        # colexicographic element order, not its memory order.
        groups_per_row = self.TILE_DIM // self.PACK_SIZE
        sX_groups = cute.make_tensor(
            sX_tile.iterator,
            cute.make_layout(
                (self.TILE_DIM * groups_per_row, self.PACK_SIZE), stride=(self.PACK_SIZE, 1)
            ),
        )
        # The staged output as u32: one conversion of 8 elements fills one u32.
        sO_u32 = cute.make_tensor(
            cute.recast_ptr(sO_row_tile.iterator, dtype=Uint32),
            cute.make_layout((self.TILE_DIM * self.TILE_DIM // 8,), stride=(1,)),
        )
        u32_per_row = self.TILE_DIM // 8

        for it in cutlass.range_constexpr(self.ITERATIONS_ROWWISE):
            row = tid_y + it * self.THREADS_Y_ROWWISE

            # Read this thread's scaling block, one staggered 8-element wave at a time.
            frg = []
            frg_u32 = []
            for w in cutlass.range_constexpr(self.WAVES):
                frg.append(cute.make_rmem_tensor(self.PACK_SIZE, BFloat16))
                frg_u32.append(
                    cute.make_tensor(
                        cute.recast_ptr(frg[w].iterator, dtype=Uint32),
                        cute.make_layout((self.PACK_SIZE // 2,), stride=(1,)),
                    )
                )
            for w in cutlass.range_constexpr(self.WAVES):
                group = tid_x * self.WAVES + ((w + bank_group) % self.WAVES)
                # Under the TMA 128B swizzle, 16B group g of row r physically sits at
                # group g ^ (r % 8).
                if cutlass.const_expr(self.SWIZZLE_INPUT):
                    group = group ^ (row % 8)
                cute.autovec_copy(sX_groups[row * groups_per_row + group, None], frg[w])

            # Block amax as packed bf16 pairs (max.xorsign.abs.bf16x2), tree-reduced.
            amax_2x = _abs_max_tree(
                [
                    Int32(frg_u32[w][j].ir_value())
                    for w in range(self.WAVES)
                    for j in range(self.PACK_SIZE // 2)
                ]
            )
            block_amax = cute.arch.fmax(
                fabs_f32(bf16_kit.x2_lo_to_f32(amax_2x)),
                fabs_f32(bf16_kit.x2_hi_to_f32(amax_2x)),
            )

            # The encode scale: global, or this row's own under row-scaled quantization.
            if cutlass.const_expr(cfg.ROW_SCALED_NVFP4):
                S_enc_block = sSEnc[stage_y * self.TILE_DIM + row]
            else:
                S_enc_block = S_enc

            block_decode_sf = compute_block_decode_sf(block_amax, S_enc_block)
            # A halved staging buffer holds only the current 64-row half, indexed by the
            # tile-relative row.
            if cutlass.const_expr(self.HALF_SS_ROW):
                ss_row = row
            else:
                ss_row = stage_y * self.TILE_DIM + row
            sS_row[ss_row, stage_x * self.SCALES_PER_TILE + tid_x] = block_decode_sf
            coeff = compute_block_encode_sf(block_decode_sf, S_enc_block, sf_type)

            # Scale and convert one wave (8 elements, one u32 of nibbles) at a time, storing
            # to the same staggered group the wave was read from. Each coefficient type gets
            # the multiply the CUDA kernel pairs it with: an fma against a zero addend for the
            # bf16 one (it flushes a -0 product to +0, and E2M1 has a signed zero, so the
            # instruction has to match) and a plain multiply -- as mul.rn.f32x2 -- for f32.
            for w in cutlass.range_constexpr(self.WAVES):
                group = tid_x * self.WAVES + ((w + bank_group) % self.WAVES)
                if cutlass.const_expr(cfg.USE_STOCHASTIC_ROUNDING):
                    rbits03 = rng.get_rbits()
                    rbits47 = rng.get_rbits()
                    if cutlass.const_expr(cfg.USE_FAST_MATH):
                        out = mul_cvt_bf16x8_to_fp4x8_sr(
                            frg_u32[w][0],
                            frg_u32[w][1],
                            frg_u32[w][2],
                            frg_u32[w][3],
                            coeff.to(Float32),
                            rbits03,
                            rbits47,
                        )
                    else:
                        out = mul2_cvt_bf16x8_to_fp4x8_sr(
                            frg_u32[w][0],
                            frg_u32[w][1],
                            frg_u32[w][2],
                            frg_u32[w][3],
                            pack_f32x2(coeff, coeff),
                            rbits03,
                            rbits47,
                        )
                else:
                    if cutlass.const_expr(cfg.USE_FAST_MATH):
                        out = mul_cvt_bf16x8_to_fp4x8(
                            frg_u32[w][0],
                            frg_u32[w][1],
                            frg_u32[w][2],
                            frg_u32[w][3],
                            coeff.to(Float32),
                        )
                    else:
                        out = mul2_cvt_bf16x8_to_fp4x8(
                            frg_u32[w][0],
                            frg_u32[w][1],
                            frg_u32[w][2],
                            frg_u32[w][3],
                            pack_f32x2(coeff, coeff),
                        )
                sO_u32[row * u32_per_row + group] = out

    @cute.jit
    def colwise_tile(
        self,
        sX_tile: cute.Tensor,  # (TILE, TILE) bf16 SMEM input tile
        sO_col_tile: cute.Tensor,  # (TILE, TILE/2) u8 SMEM staged transposed output
        sS_col: cute.Tensor,  # (CHUNK_DIM_X, SCALES_PER_CHUNK_Y) e4m3 SMEM staged scales
        sSEnc: Optional[cute.Tensor],  # (CHUNK_DIM_X,) f32 per-column encode scales (ROW_SCALED)
        S_enc: Float32,  # global colwise encode scale (ignored when ROW_SCALED)
        stage_y: cutlass.Constexpr,
        stage_x: cutlass.Constexpr,
        rng,
    ):
        """Quantize one SMEM tile columnwise into the transposed staged output: a thread owns
        two adjacent columns of one 16-row scaling block, read as bf16 pairs, with the block row
        staggered by warp so consecutive lanes hit different SMEM rows (conflict-free 4-byte
        reads). Mirrors colwise_scaling in the CUDA kernel."""
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        warp = tidx // 32
        tid_y = (lane // 2 + warp) % self.SCALES_PER_TILE  # which 16-row block

        if cutlass.const_expr(cfg.USE_FAST_MATH):
            sf_type = BFloat16
        else:
            sf_type = Float32

        # The tile as (row, column pair): a pair-register read per row.
        sX_pairs = cute.make_tensor(
            cute.recast_ptr(sX_tile.iterator, dtype=Uint32),
            cute.make_layout((self.TILE_DIM * (self.TILE_DIM // 2),), stride=(1,)),
        )
        pairs_per_row = self.TILE_DIM // 2
        # The transposed staged output as u64: one 16-element block of a transposed row is 8
        # bytes, one two-conversion store.
        sO_u64 = cute.make_tensor(
            cute.recast_ptr(sO_col_tile.iterator, dtype=Int64),
            cute.make_layout((self.TILE_DIM * self.TILE_DIM // 16,), stride=(1,)),
        )
        u64_per_row = self.TILE_DIM // 16

        # Read the two columns' 16-row block as pairs; both block amaxes at once in the packed
        # halves, tree-reduced.
        row0 = tid_y * NVFP4_BLOCK_SCALING_SIZE
        pairs = []
        for i in cutlass.range_constexpr(NVFP4_BLOCK_SCALING_SIZE):
            pair_idx = lane
            # Under the TMA 128B swizzle, 4B pair p of row r sits at pair p ^ ((r % 8) << 2)
            # (the XOR acts on the 16B-unit part of the pair index). row0 is a multiple of
            # 16, so r % 8 == i % 8, a constant per unrolled step.
            if cutlass.const_expr(self.SWIZZLE_INPUT):
                pair_idx = lane ^ ((i % 8) * 4)
            pairs.append(sX_pairs[(row0 + i) * pairs_per_row + pair_idx])
        amax_2x = _abs_max_tree([Int32(p.ir_value()) for p in pairs])
        block_amax = [
            fabs_f32(bf16_kit.x2_lo_to_f32(amax_2x)),
            fabs_f32(bf16_kit.x2_hi_to_f32(amax_2x)),
        ]

        for w in cutlass.range_constexpr(2):
            col = stage_x * self.TILE_DIM + 2 * lane + w
            # The encode scale: global, or this column's own under row-scaled quantization.
            if cutlass.const_expr(cfg.ROW_SCALED_NVFP4):
                S_enc_block = sSEnc[col]
            else:
                S_enc_block = S_enc

            block_decode_sf = compute_block_decode_sf(block_amax[w], S_enc_block)
            sS_col[col, stage_y * self.SCALES_PER_TILE + tid_y] = block_decode_sf
            coeff = compute_block_encode_sf(block_decode_sf, S_enc_block, sf_type)

            outs = []
            if cutlass.const_expr(cfg.USE_FAST_MATH):
                coeff_f32 = coeff.to(Float32)
                # Repack this column's elements into adjacent bf16 pairs for the fma-based
                # conversion: bytes of (row 2j, row 2j+1) picked by a byte permute.
                prmt = prmt_lo_u32 if w == 0 else prmt_hi_u32
                packed = [
                    prmt(pairs[2 * j], pairs[2 * j + 1])
                    for j in range(NVFP4_BLOCK_SCALING_SIZE // 2)
                ]
                for e in cutlass.range_constexpr(NVFP4_BLOCK_SCALING_SIZE // 8):
                    if cutlass.const_expr(cfg.USE_STOCHASTIC_ROUNDING):
                        rbits03 = rng.get_rbits()
                        rbits47 = rng.get_rbits()
                        outs.append(
                            mul_cvt_bf16x8_to_fp4x8_sr(
                                packed[4 * e],
                                packed[4 * e + 1],
                                packed[4 * e + 2],
                                packed[4 * e + 3],
                                coeff_f32,
                                rbits03,
                                rbits47,
                            )
                        )
                    else:
                        outs.append(
                            mul_cvt_bf16x8_to_fp4x8(
                                packed[4 * e],
                                packed[4 * e + 1],
                                packed[4 * e + 2],
                                packed[4 * e + 3],
                                coeff_f32,
                            )
                        )
            else:
                # Packed mul.rn.f32x2 scaling of this column's halves of eight row registers;
                # no repacking needed, the widen mode selects the half.
                coeff2 = pack_f32x2(coeff, coeff)
                for e in cutlass.range_constexpr(NVFP4_BLOCK_SCALING_SIZE // 8):
                    if cutlass.const_expr(cfg.USE_STOCHASTIC_ROUNDING):
                        rbits03 = rng.get_rbits()
                        rbits47 = rng.get_rbits()
                        outs.append(
                            mul2_cvt_col_sr[w](*pairs[8 * e : 8 * e + 8], coeff2, rbits03, rbits47)
                        )
                    else:
                        outs.append(mul2_cvt_col[w](*pairs[8 * e : 8 * e + 8], coeff2))

            sO_u64[(2 * lane + w) * u64_per_row + tid_y] = pack_u32x2(outs[0], outs[1])

    @cute.jit
    def flush_scales(
        self,
        sS: cute.Tensor,  # (slab_rows, 8) e4m3 SMEM staged scales
        mS: cute.Tensor,  # padded gmem scale tensor
        chunk_row0: Int32,  # global scale row the slab starts at
        chunk_sf_col0: Int32,  # global scale column of the chunk's first scale
        outer: Int32,  # valid rows of the scale tensor (input rows / cols)
        inner_elems: Int32,  # input elements along the scaled direction (cols / rows)
        # -> valid scale columns = inner_elems / 16
        slab_rows: cutlass.Constexpr,  # rows staged in sS (64 for a half-chunk, else 128)
    ):
        """Write staged scale bytes out, one slab row per thread, predicated against the
        input's real extents: the scale tensors' padding is left untouched and rows/columns
        past the input's edge are skipped, like the CUDA kernel's Vec-based scale store.

        A chunk owns 8 scale bytes of each of its 128 scale rows, so this store is
        sector-amplified either way; the 8-byte-per-row path is what keeps that amplification
        from throttling the kernel on L2 write sectors. It needs an 8B-aligned row address
        (true whenever the padded scale stride is a multiple of 8), else two 4-byte stores."""
        tidx, _, _ = cute.arch.thread_idx()
        # A halved slab has fewer rows than there are threads; excess threads must not even
        # read it.
        in_slab = tidx < slab_rows if cutlass.const_expr(slab_rows < self.THREADS) else tidx >= 0
        row_global = chunk_row0 + tidx
        width = self.SCALES_PER_CHUNK_X  # == SCALES_PER_CHUNK_Y == 8 scale bytes per row
        count = cutlass.min(
            Int32(width),
            (inner_elems - chunk_sf_col0 * NVFP4_BLOCK_SCALING_SIZE) // NVFP4_BLOCK_SCALING_SIZE,
        )
        frg = cute.make_rmem_tensor(width, Float8E4M3FN)
        if in_slab:
            cute.autovec_copy(sS[cutlass.min(tidx, slab_rows - 1), None], frg)
        frg_u32 = cute.make_tensor(
            cute.recast_ptr(frg.iterator, dtype=Uint32),
            cute.make_layout((width // 4,), stride=(1,)),
        )
        # A u32 view of the scale tensor: its padded inner extent (= the row stride) is a
        # multiple of NVFP4_SCALE_PAD_INNER = 4, so rows stay u32-aligned, and a chunk's
        # first scale column is a multiple of 8.
        scale_stride = cute.size(mS.shape[1])
        mS_u32 = cute.make_tensor(
            cute.recast_ptr(mS.iterator, dtype=Uint32),
            cute.make_layout((mS.shape[0], scale_stride // 4), stride=(scale_stride // 4, 1)),
        )
        row_addr = mS.iterator.toint() + (
            Int64(row_global) * Int64(scale_stride) + Int64(chunk_sf_col0)
        )
        if in_slab and (row_global < outer):
            if count == width:
                if row_addr % 8 == 0:
                    st_global_b64(row_addr, pack_u32x2(frg_u32[0], frg_u32[1]))
                else:
                    for c4 in cutlass.range_constexpr(width // 4):
                        mS_u32[row_global, chunk_sf_col0 // 4 + c4] = frg_u32[c4]
            else:
                for c in cutlass.range_constexpr(width):
                    if c < count:
                        mS[row_global, chunk_sf_col0 + c] = frg[c]
