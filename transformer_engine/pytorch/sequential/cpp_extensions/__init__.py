from typing import Any
from .real import *

from . import printing

raw_tensor = globals().pop("Tensor")


class __TensorImpostor:
    def __getattribute__(self, __name: str) -> Any:
        if __name == "__repr__":
            return printing.tensor_repr  # type: ignore
        else:
            return getattr(raw_tensor, __name)

    def __call__(
        self,
        dtype: Any,
        data: torch.Tensor,
        amax: torch.Tensor,
        scale: torch.Tensor,
        scale_inv: torch.Tensor,
    ):
        return raw_tensor(dtype.value, data, amax, scale, scale_inv)  # type: ignore

    def dtype(self, self_: Any):  # type: ignore
        raw_dtype = raw_tensor.dtype(self_)  # type: ignore
        return DType(raw_dtype)  # type: ignore


Tensor = __TensorImpostor()
