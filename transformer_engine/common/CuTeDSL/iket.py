# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Helper for using IKET (In-Kernel Event Tracing) for profiling CuTeDSL kernels.

Controlled with NVTE_IKET_ENABLED.
0 (default) disables profiling.
Higher levels enable more events.
"""

import os

LEVEL = int(os.environ.get("NVTE_IKET_ENABLED", "0"))

# What `range_push` hands back when it emitted. Opaque: only ever tested against None.
_PUSHED = object()


def _impl():
    # pylint: disable=import-outside-toplevel
    # iket is experimental, import only when necessary
    from cutlass.cute.experimental import iket as _iket

    return _iket


def range_push(event_name: str, level: int = 1):
    """Begin range"""
    if LEVEL < level:
        return None
    _impl().range_push(event_name)
    return _PUSHED


def range_pop(handle) -> None:
    """End range"""
    if handle is None:
        return
    _impl().range_pop()


def mark(event_name: str, level: int = 1) -> None:
    """Record single event"""
    if LEVEL < level:
        return
    _impl().mark(event_name)
