from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Generator, final
from transformer_engine.pytorch.sequential_new.common_back.enums import DType, PType
from .generic_environment import ExecutionEnv
from transformer_engine.pytorch.sequential_new.common_back.generic_tensor import (
    GenericTensor,
    TensorDescriptor,
    ParamInitializer,
)
from .enums import DType, DTypeInfer, PType
from .generic_tensor import GenericTensor
from . import generic_tensor as f

ExecutionFlow = list["Op"]
Parallelism = tuple[PType, PType]


# Op Protocol
class Unset:
    pass


class Op(ABC):
    """
    An Op represents a transformation applied to the main data tensor.
    It is akin to an nn.Module + autograd.Function pair in PyTorch.

    Inside ComputePipeline, the following methods are called in the order
    in which they are declared below.
    """

    name: str
    __environment: ExecutionEnv | Unset
    __dist_group_size: int | Unset | None
    __input_type: DType | DTypeInfer
    __output_type: DType | DTypeInfer
    __parallellism: Parallelism | Unset
    __input_shape: tuple[int, ...] | Unset

    @property
    def environment(self):
        assert not isinstance(self.__environment, Unset)
        return self.__environment

    @property
    def dist_group_size(self):
        assert not isinstance(self.__dist_group_size, Unset)
        if self.__dist_group_size is None:
            raise ValueError(
                "Operation is not distributed, but distributed group size is requested"
            )
        else:
            return self.__dist_group_size

    @property
    def raw_input_type(self):
        return self.__input_type

    @raw_input_type.setter
    def raw_input_type(self, value: DType):
        self.__input_type = value

    @property
    def raw_output_type(self):
        return self.__output_type

    @raw_output_type.setter
    def raw_output_type(self, value: DType):
        self.__output_type = value

    @property
    def input_type(self):
        assert isinstance(self.__input_type, DType)
        return self.__input_type

    @property
    def output_type(self):
        assert isinstance(self.__output_type, DType)
        return self.__output_type

    @property
    def parallellism(self):
        assert not isinstance(self.__parallellism, Unset)
        return self.__parallellism

    @property
    def input_shape(self):
        assert not isinstance(self.__input_shape, Unset)
        return self.__input_shape

    def __init__(
        self,
        relative_name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
    ):
        self.name = relative_name
        self.__input_type = input_type
        self.__output_type = output_type
        self.__environment = Unset()
        self.__dist_group_size = Unset()
        self.__parallellism = Unset()
        self.__input_shape = Unset()

    def set_parent_name(self, parent_module_name: str):
        self.name = parent_module_name + "." + self.name
        return self

    def set_environment(self, env: ExecutionEnv):
        self.__environment = env
        self.__dist_group_size = (
            self.__environment.distributed_group.size()
            if self.__environment.distributed_group is not None
            else None
        )
        return self

    def set_types_inferred(
        self, inferred_input_type: DType, inferred_output_type: DType
    ):
        self.__input_type = inferred_input_type
        self.__output_type = inferred_output_type
        return self

    @abstractmethod
    def describe_parallellism(self) -> list[ExecutionFlow]:
        raise NotImplementedError()

    def set_parallelism(self, chosen_parallelism: Parallelism):
        self.__parallellism = chosen_parallelism
        return self

    def set_input_shape(self, input_shape: tuple[int, ...]):
        self.__input_shape = input_shape
        self._pre_describe_tensors()
        return self

    @abstractmethod
    def _pre_describe_tensors(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def describe_params(self) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

    @abstractmethod
    def describe_activation_shape(self) -> tuple[int, ...]:
        raise NotImplementedError()

    @abstractmethod
    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

    @abstractmethod
    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

    def set_tensors_allocated(self, **tensors: GenericTensor):
        for name, tensor in tensors.items():
            setattr(self, name, tensor)

    @abstractmethod
    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, None]:
        raise NotImplementedError()

    @abstractmethod
    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        raise NotImplementedError()


# Base classes
class ParameterFreeOp(Op):
    """
    Base class for ops that do not have any trainable parameters.
    """

    def describe_params(self) -> dict[str, TensorDescriptor]:
        return {}


class ShapePreserveOp(Op):
    """
    Base class for ops, whose activation's shape is the same as the input's.
    """

    def describe_activation_shape(self):
        return self.input_shape


class NoSupplementaryTensorsOp(Op):
    """
    Base class for ops that do not need any additional tensors.
    """

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {}

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()


# Parallelism base classes


class ParallelismClass:
    NORMAL = (PType.NA, PType.NA)
    ROWP = (PType.NRS, PType.NRS)
    COLP = (PType.NCS, PType.NCS)
    CGEMM = (PType.NA, PType.NCS)
    RGEMM = (PType.NCS, PType.PA)
    S = (PType.NA, PType.NRS)
    RS = (PType.PA, PType.NRS)
    AG = (PType.NRS, PType.NA)
    AR = (PType.PA, PType.NA)


def __single_parallel(parallelism: Parallelism, op: Op):
    op = deepcopy(op)
    op.set_parallelism(parallelism)
    return [op]


_normal = partial(__single_parallel, ParallelismClass.NORMAL)
_row_parallel = partial(__single_parallel, ParallelismClass.ROWP)
_column_parallel = partial(__single_parallel, ParallelismClass.COLP)


class PointwiseOp(Op):
    def describe_parallellism(self):
        return [_normal(self), _row_parallel(self), _column_parallel(self)]


class RowwiseOp(Op):
    def describe_parallellism(self):
        return [_normal(self), _row_parallel(self)]


class ColumnwiseOp(Op):
    def describe_parallellism(self):
        return [_normal(self), _column_parallel(self)]


class EnvObliviousOp(Op):
    def _pre_describe_tensors(self) -> None:
        return


class NonParallelOp(EnvObliviousOp):
    def describe_parallellism(self):
        return [_normal(self)]


# Identity
@final
class Identity(
    PointwiseOp,
    ParameterFreeOp,
    ShapePreserveOp,
    NoSupplementaryTensorsOp,
    EnvObliviousOp,
):
    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, None]:
        grad = yield x
        yield grad

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        return x


# Transpose
@final
class Transpose(NonParallelOp, ParameterFreeOp):
    act: GenericTensor

    def _act_shape(self):
        return self.input_shape[:-2] + (self.input_shape[-1], self.input_shape[-2])

    def describe_activation_shape(self) -> tuple[int, ...]:
        return self._act_shape()

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        if self.input_shape != self._act_shape():
            return {"act": TensorDescriptor(self._act_shape(), None, self.output_type)}
        else:
            return {}

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        if self.input_shape != self._act_shape():
            f.transpose(x, self.act)
            grad = yield self.act
            f.transpose(grad, x)
            yield x
        else:
            f.transpose(x, x)
            grad = yield x
            f.transpose(grad, grad)
            yield grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        if self.input_shape != self._act_shape():
            f.transpose(x, self.act)
            return self.act
        else:
            f.transpose(x, x)
            return x


# Normalization
@final
class LayerNorm(RowwiseOp, ShapePreserveOp, EnvObliviousOp):
    weight: GenericTensor
    weight_grad: GenericTensor
    bias: GenericTensor
    bias_grad: GenericTensor
    act: GenericTensor
    mu: GenericTensor
    rsigma: GenericTensor

    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        param_type: DType,
        features: int,
        eps: float,
        zero_centered_gamma: bool,
    ):
        super().__init__(name, input_type, output_type)
        self.param_type = param_type
        self.features = features
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma

    def describe_params(
        self,
    ) -> dict[str, TensorDescriptor]:
        return {
            "weight": TensorDescriptor(
                (self.features,),
                f.zeros if self.zero_centered_gamma else f.ones,
                self.param_type,
            ),
            "bias": TensorDescriptor((self.features,), f.zeros, self.param_type),
        }

    def describe_supplementary_tensors_training(
        self,
    ) -> dict[str, TensorDescriptor]:
        return {
            "act": TensorDescriptor(self.input_shape, None, self.output_type),
            "mu": TensorDescriptor((self.input_shape[-2],), None, DType.FP32),
            "rsigma": TensorDescriptor((self.input_shape[-2],), None, DType.FP32),
        }

    def describe_supplementary_tensors_inference(
        self,
    ) -> dict[str, TensorDescriptor]:
        if self.output_type != self.input_type:
            return {"act": TensorDescriptor(self.input_shape, None, self.output_type)}
        else:
            return {}

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.layer_norm(
            x,
            self.weight,
            self.bias,
            self.eps,
            self.zero_centered_gamma,
            self.act,
            self.mu,
            self.rsigma,
        )
        grad = yield self.act
        f.dlayer_norm(
            grad,
            x,
            self.weight,
            self.zero_centered_gamma,
            self.mu,
            self.rsigma,
            grad,
            self.weight_grad,
            self.bias_grad,
        )
        yield grad

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        if self.output_type != self.input_type:
            f.layer_norm_inf(
                x,
                self.weight,
                self.bias,
                self.eps,
                self.zero_centered_gamma,
                self.act,
            )
            return self.act
        else:
            f.layer_norm_inf(
                x,
                self.weight,
                self.bias,
                self.eps,
                self.zero_centered_gamma,
                x,
            )
            return x


# Linear
_rgemm = partial(__single_parallel, ParallelismClass.RGEMM)
_cgemm = partial(__single_parallel, ParallelismClass.CGEMM)


def _rgemm_rs(op: Gemm) -> list[Op]:
    op = deepcopy(op)
    op.set_parallelism(ParallelismClass.RGEMM)
    rs = (
        ReduceScatter("post-rs", op.output_type, op.output_type)
        .set_parent_name(op.name)
        .set_environment(op.environment)
        .set_types_inferred(op.output_type, op.output_type)
        .set_parallelism(ParallelismClass.RS)
    )
    return _rgemm(op) + [rs]


def _ag_cgemm(op: Gemm) -> list[Op]:
    op = deepcopy(op)
    ag = (
        AllGather("pre-ag", op.input_type, op.input_type)
        .set_parent_name(op.name)
        .set_environment(op.environment)
        .set_types_inferred(op.output_type, op.output_type)
        .set_parallelism(ParallelismClass.AG)
    )
    op.set_parallelism(ParallelismClass.CGEMM)
    return [ag] + _cgemm(op)


@final
class Gemm(Op):
    weight: GenericTensor
    weight_grad: GenericTensor
    weight_t: GenericTensor
    x_t: GenericTensor
    act: GenericTensor

    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        param_type: DType,
        in_features: int,
        out_features: int,
        init_method: ParamInitializer,
    ):
        super().__init__(name, input_type, output_type)
        self.in_features = in_features
        self.out_features = out_features
        self.param_dtype = param_type
        self.init_method = init_method

    def _pre_describe_tensors(self):
        assert self.parallellism in [
            ParallelismClass.NORMAL,
            ParallelismClass.RGEMM,
            ParallelismClass.CGEMM,
        ]

        workers = (
            self.dist_group_size
            if self.parallellism in [ParallelismClass.RGEMM, ParallelismClass.CGEMM]
            else 1
        )
        if self.parallellism == ParallelismClass.RGEMM:
            if self.in_features % workers != 0:
                raise ValueError(
                    "Number of input features must be divisible by the distributed group size"
                )
            self.in_features //= workers
        elif self.parallellism == ParallelismClass.CGEMM:
            if self.out_features % workers != 0:
                raise ValueError(
                    "Number of output features must be divisible by the distributed group size"
                )
            self.out_features //= workers

    def describe_parallellism(self):
        return [
            _normal(self),
            _rgemm(self),
            _cgemm(self),
            _rgemm_rs(self),
            _ag_cgemm(self),
        ]

    def describe_params(self) -> dict[str, TensorDescriptor]:
        return {
            "weight": TensorDescriptor(
                (self.in_features, self.out_features),
                self.init_method,
                self.param_dtype,
            ),
        }

    def _act_shape(self):
        return self.input_shape[:-1] + (self.out_features,)

    def describe_activation_shape(self) -> tuple[int, ...]:
        return self._act_shape()

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {
            "act": TensorDescriptor(self._act_shape(), None, self.output_type),
            "weight_t": TensorDescriptor(
                (self.out_features, self.in_features), None, self.param_dtype
            ),
            "x_t": TensorDescriptor(
                (self.input_shape[-1], self.input_shape[-2]), None, self.input_type
            ),
        }

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        if self.output_type != self.input_type or self.input_shape != self._act_shape():
            return {"act": TensorDescriptor(self._act_shape(), None, self.output_type)}
        else:
            return {}

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.transpose(self.weight, self.weight_t)
        f.gemm(x, self.weight, self.act)
        grad = yield self.act
        f.gemm(grad, self.weight_t, x)
        f.gemm(grad, self.x_t, self.weight_grad)
        yield x

    def inference(self, x: GenericTensor) -> GenericTensor:
        if self.output_type != self.input_type or self.input_shape != self._act_shape():
            f.gemm(x, self.weight, self.act)
            return self.act
        else:
            f.gemm(x, self.weight, x)
            return x


@final
class Bias(PointwiseOp, ShapePreserveOp, NoSupplementaryTensorsOp):
    bias: GenericTensor
    bias_grad: GenericTensor

    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        param_type: DType,
        features: int,
        init_method: ParamInitializer,
    ):
        super().__init__(name, input_type, output_type)
        self.param_dtype = param_type
        self.features = features
        self.init_method = init_method

    def _pre_describe_tensors(self):
        assert self.parallellism in [
            ParallelismClass.NORMAL,
            ParallelismClass.ROWP,
            ParallelismClass.COLP,
        ]
        workers = (
            self.dist_group_size if self.parallellism == ParallelismClass.COLP else 1
        )
        if self.features % workers != 0:
            raise ValueError(
                "Number of features must be divisible by the distributed group size"
            )
        self.features //= workers

    def describe_params(self) -> dict[str, TensorDescriptor]:
        return {
            "bias": TensorDescriptor(
                (self.features,), self.init_method, self.param_dtype
            ),
        }

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.add(x, self.bias, x)
        grad = yield x
        f.copy(grad, self.bias_grad)
        yield grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.add(x, self.bias, x)
        return x


# Attention
HEAD_PARALLEL_SCATTERING = (PType.NA, PType.NCS)
HEAD_PARALLEL_GATHERING = (PType.NCS, PType.NA)


@final
class DotProductAttention(ParameterFreeOp, ShapePreserveOp):  # TODO
    features_per_head: int

    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        features_per_head: int,
    ):
        super().__init__(
            name,
            input_type,
            output_type,
        )
        self.features_per_head = features_per_head

    def describe_parallellism(self):
        return [
            _normal(self),
            _column_parallel(self),
            HEAD_PARALLEL_SCATTERING,
            HEAD_PARALLEL_GATHERING,
        ]


# Residual
@final
class ResidualBegin(PointwiseOp, ParameterFreeOp, ShapePreserveOp, EnvObliviousOp):
    end: ResidualEnd | None = None
    fwd_residue: GenericTensor

    def __init__(self, name: str, end: ResidualEnd | None = None):
        super().__init__(name, DType.default, DType.default)
        self.end = end

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        assert self.end is not None
        f.copy(x, self.fwd_residue)
        grad = yield x
        f.add(self.end.bwd_residue, grad, self.end.bwd_residue)
        yield self.end.bwd_residue

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.copy(x, self.fwd_residue)
        return x

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {
            "fwd_residue": TensorDescriptor(self.input_shape, None, self.input_type)
        }

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()


@final
class ResidualEnd(PointwiseOp, ParameterFreeOp, ShapePreserveOp, EnvObliviousOp):
    begin: ResidualBegin
    bwd_residue: GenericTensor

    def __init__(self, name: str, begin: ResidualBegin):
        super().__init__(name, DType.default, DType.default)
        self.begin = begin

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.add(self.begin.fwd_residue, x, self.begin.fwd_residue)
        grad = yield self.begin.fwd_residue
        f.copy(grad, self.bwd_residue)
        yield grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.add(self.begin.fwd_residue, x, self.begin.fwd_residue)
        return self.begin.fwd_residue

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {
            "bwd_residue": TensorDescriptor(self.input_shape, None, self.input_type)
        }

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return {}


# Dropout
@final
class Dropout(
    PointwiseOp,
    ParameterFreeOp,
    ShapePreserveOp,
    NoSupplementaryTensorsOp,
    EnvObliviousOp,
):
    p: float

    def __init__(self, name: str, p: float):
        super().__init__(name, DTypeInfer(), DTypeInfer())
        self.p = p

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.dropout(x, self.p, x)
        grad = yield x
        f.dropout(grad, self.p, grad)
        yield grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.dropout(x, self.p, x)
        return x


# Activation
@final
class Gelu(PointwiseOp, ParameterFreeOp, ShapePreserveOp, EnvObliviousOp):
    act: GenericTensor

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.gelu(x, self.act)
        grad = yield self.act
        f.dgelu(grad, x, grad)
        yield grad

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        if self.output_type != self.input_type:
            f.gelu(x, self.act)
            return self.act
        else:
            f.gelu(x, x)
            return x

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {"act": TensorDescriptor(self.input_shape, None, self.input_type)}

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        if self.output_type != self.input_type:
            return {"act": TensorDescriptor(self.input_shape, None, self.output_type)}
        else:
            return {}


@final
class Relu(PointwiseOp, ParameterFreeOp, ShapePreserveOp, EnvObliviousOp):
    act: GenericTensor

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.relu(x, self.act)
        grad = yield self.act
        f.drelu(grad, x, grad)
        yield grad

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        if self.output_type != self.input_type:
            f.relu(x, self.act)
            return self.act
        else:
            f.relu(x, x)
            return x

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {"act": TensorDescriptor(self.input_shape, None, self.input_type)}

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        if self.output_type != self.input_type:
            return {"act": TensorDescriptor(self.input_shape, None, self.output_type)}
        else:
            return {}


# Cast
@final
class Cast(PointwiseOp, ShapePreserveOp, ParameterFreeOp, EnvObliviousOp):
    act: GenericTensor

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.cast(x, self.act)
        grad = yield self.act
        f.cast(grad, x)
        yield x

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        f.cast(x, self.act)
        return self.act

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {"act": TensorDescriptor(self.input_shape, None, self.output_type)}

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()


# Communication

_scatter = partial(__single_parallel, ParallelismClass.S)
_reduce_scatter = partial(__single_parallel, ParallelismClass.RS)
_all_gather = partial(__single_parallel, ParallelismClass.AG)
_all_reduce = partial(__single_parallel, ParallelismClass.AR)


def row_split_shape(shape: tuple[int, ...], workers: int) -> tuple[int, ...]:
    assert len(shape) >= 2
    assert shape[-2] % workers == 0
    return shape[:-2] + (shape[-2] // workers, shape[-1])


def row_merge_shape(shape: tuple[int, ...], workers: int) -> tuple[int, ...]:
    assert len(shape) >= 2
    return shape[:-2] + (shape[-2] * workers, shape[-1])


@final
class Scatter(ParameterFreeOp, EnvObliviousOp):
    act: GenericTensor

    def describe_parallellism(self) -> list[ExecutionFlow]:
        return [_scatter(self)]

    def describe_activation_shape(self) -> tuple[int, ...]:
        return row_split_shape(self.input_shape, self.dist_group_size)

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {
            "act": TensorDescriptor(
                self.describe_activation_shape(), None, self.output_type
            )
        }

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.scatter(x, self.act, self.environment.distributed_group)
        grad = yield self.act
        f.gather(grad, x, self.environment.distributed_group)
        yield x

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.scatter(x, self.act, self.environment.distributed_group)
        return self.act


@final
class ReduceScatter(ParameterFreeOp, EnvObliviousOp):
    act: GenericTensor

    def describe_parallellism(self) -> list[ExecutionFlow]:
        return [_reduce_scatter(self)]

    def describe_activation_shape(self) -> tuple[int, ...]:
        return row_split_shape(self.input_shape, self.dist_group_size)

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {
            "act": TensorDescriptor(
                self.describe_activation_shape(), None, self.output_type
            )
        }

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.reduce_scatter(x, self.act, self.environment.distributed_group)
        grad = yield self.act
        f.all_gather(grad, x, self.environment.distributed_group)
        yield x

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.reduce_scatter(x, self.act, self.environment.distributed_group)
        return self.act


@final
class AllGather(ParameterFreeOp, EnvObliviousOp):
    act: GenericTensor

    def describe_parallellism(self) -> list[ExecutionFlow]:
        return [_all_gather(self)]

    def describe_activation_shape(self) -> tuple[int, ...]:
        return row_merge_shape(self.input_shape, self.dist_group_size)

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {
            "act": TensorDescriptor(
                self.describe_activation_shape(), None, self.output_type
            )
        }

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.all_gather(x, self.act, self.environment.distributed_group)
        grad = yield self.act
        f.reduce_scatter(grad, x, self.environment.distributed_group)
        yield x

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.all_gather(x, self.act, self.environment.distributed_group)
        return self.act


@final
class AllReduce(ParameterFreeOp, EnvObliviousOp, ShapePreserveOp):
    act: GenericTensor

    def describe_parallellism(self) -> list[ExecutionFlow]:
        return [_all_reduce(self)]

    def describe_supplementary_tensors_training(self) -> dict[str, TensorDescriptor]:
        return {
            "act": TensorDescriptor(
                self.describe_activation_shape(), None, self.output_type
            )
        }

    def describe_supplementary_tensors_inference(self) -> dict[str, TensorDescriptor]:
        return self.describe_supplementary_tensors_training()

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        f.all_reduce(x, self.act, self.environment.distributed_group)
        grad = yield self.act
        yield grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        f.all_reduce(x, self.act, self.environment.distributed_group)
        return self.act
