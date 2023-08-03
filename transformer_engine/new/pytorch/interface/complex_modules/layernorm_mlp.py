from typing_extensions import deprecated
from ..meta_modules.sequential import Sequential
from ..base_modules import ComputePipelineModuleBase
from ..atomic_modules import LayerNorm, Linear, ReLU, GELU


@deprecated
class LayerNormMLP(ComputePipelineModuleBase):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        eps: float = 1e-5,
        zero_centered_gamma: bool = False,
        bias: bool = True,
        activation: ReLU | GELU = GELU(),
    ):
        super().__init__(
            *Sequential(
                LayerNorm(in_features, eps, zero_centered_gamma),
                Linear(in_features, out_features, bias=bias),
                activation,
                Linear(in_features, out_features, bias=bias),
            ).ops
        )
