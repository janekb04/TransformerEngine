from __future__ import annotations
import torch
from .. import cpp_extensions as _nvte
from .dtype import torch_to_te_dtype
from ..utils import torch_op


@torch_op
def make_nvte_tensor(t: torch.Tensor):
    return _nvte.Tensor(
        torch_to_te_dtype(t.dtype),
        t.data,
        torch.Tensor().cuda(),
        torch.Tensor().cuda(),
        torch.Tensor().cuda(),
    )
