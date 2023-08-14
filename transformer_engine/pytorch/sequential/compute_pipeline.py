import copy
from functools import reduce
import operator
import transformer_engine_cuda as _nvte  # pylint: disable=import-error
from .utils import set_attribute
from .nvte import is_fp8
from .ops import Op, Grads, Context
from .fusions import FusedOp, get_fused_op_list
from .environment import Environment


class SelfContainedOp(Op):
    def __init__(self, fwds: list[Op], bwds: list[Op]) -> None:
        self.fwds = fwds
        self.bwds = bwds

    def inference(self, x: _nvte.Tensor) -> _nvte.Tensor:
        raise AssertionError("Not used for inference")

    def forward(self, x: _nvte.Tensor):
        full_ctx = Context()
        for op in self.fwds:
            x, ctx = op.forward(x)
            if not isinstance(op, FusedOp):
                op_name = getattr(op, "name")
                ctx = {op_name + name: tensor for name, tensor in ctx.items()}
            full_ctx |= ctx
        return x, full_ctx

    def backward(self, ctx: Context, dy: _nvte.Tensor):
        ctxs = list[Context]()
        for op in self.bwds:
            if isinstance(op, FusedOp):
                ctxs.append(ctx)
            else:
                op_name = getattr(op, "name")
                ctxs.append(
                    {
                        name[len(op_name) :]: tensor
                        for name, tensor in ctx.items()
                        if name.startswith(op_name)
                    }
                )

        full_grads = Grads()
        for op, ctx in list(zip(self.bwds, ctxs))[::-1]:
            dy, grads = op.backward(ctx, dy)
            full_grads += grads
        return dy, full_grads

    def args(self):
        return list(sum((op.args() for op in self.fwds), list[_nvte.Tensor]()))


def force_use_bf16(ops: list[Op]):
    for op in ops:
        attributes = dir(op)
        dtype_attributes = [attr for attr in attributes if attr.endswith("_dtype")]
        for dtype_attribute in dtype_attributes:
            attr_val = getattr(op, dtype_attribute)
            if isinstance(attr_val, _nvte.DType) and is_fp8(attr_val):
                setattr(op, dtype_attribute, _nvte.DType.BFloat16)


def model_parallel_transform(ops: list[Op]):
    raise NotImplementedError()


def name_ops(ops: list[Op]):
    for i, op in enumerate(ops):
        setattr(op, "name", f"{i}({op.__class__.__name__})")


def split_into_self_contained(fwds: list[Op], bwds: list[Op]):
    functions = list[SelfContainedOp]()
    while fwds or bwds:
        fwd = fwds.pop(0)
        unmatched_fwd_ops: set[Op] = {
            *reduce(operator.iadd, [fwd.ops if isinstance(fwd, FusedOp) else [fwd]], [])
        }
        used_forwards = [fwd]
        used_backwards = list[Op]()
        unmatched_bwd_ops: set[Op] = set()
        while unmatched_fwd_ops or unmatched_bwd_ops:
            while unmatched_fwd_ops:
                bwd = bwds.pop(0)
                used_backwards.append(bwd)
                ops = bwd.ops if isinstance(bwd, FusedOp) else [bwd]
                for op in ops:
                    if op in unmatched_fwd_ops:
                        unmatched_fwd_ops.remove(op)
                    else:
                        unmatched_bwd_ops.add(op)
            while unmatched_bwd_ops:
                fwd = fwds.pop(0)
                used_forwards.append(fwd)
                ops = fwd.ops if isinstance(fwd, FusedOp) else [fwd]
                for op in ops:
                    if op in unmatched_bwd_ops:
                        unmatched_bwd_ops.remove(op)
                    else:
                        unmatched_fwd_ops.add(op)
        functions.append(SelfContainedOp(used_forwards, used_backwards))
    return functions


def copy_op_list(ops: list[Op]):
    "Deep copy ops, except for tensors"
    with set_attribute(_nvte.Tensor, "__deepcopy__", lambda self, memo: self):  # type: ignore[unknown-lambda-type]
        return copy.deepcopy(ops)


class ComputePipeline:
    def __init__(self, ops: list[Op], env: Environment):
        ops = copy_op_list(ops)

        name_ops(ops)
        if not env.fp8_enabled:
            force_use_bf16(ops)
        if env.world_size > 1:
            model_parallel_transform(ops)

        self._inf = get_fused_op_list(ops, "inference")

        self.functions = split_into_self_contained(
            get_fused_op_list(ops, "forward"), get_fused_op_list(ops, "backward")
        )
        self.forward = tuple(op for f in self.functions for op in f.fwds)
        self.backward = tuple(op for f in self.functions for op in f.bwds)

    def run_inference(self, x: _nvte.Tensor) -> _nvte.Tensor:
        for op in self._inf:
            x = op.inference(x)
        return x

    def __repr__(self):
        return f"""ComputePipeline(
    forward: {self.forward},
    backward: {self.backward},
)"""
