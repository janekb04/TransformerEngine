from __future__ import annotations
import torch
from torch import autograd
from torch.autograd.function import FunctionCtx
from typing import Final, Sequence
from .persistent import Persistent
from . import nvte
from .ops import Context, Op
from .compute_pipeline import ComputePipeline

FP8Meta = tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class BackwardComm:
    nvte_grad_output: nvte.Tensor | None = None


class ForwardArgs:
    is_exposed_x_squished_now: Final[bool]
    upcoming_backward: Final[BackwardComm | None]
    next_upcoming_backward: Final[BackwardComm]
    op: Final[Op]
    meta_tensor_provider_fwd: Final[Persistent[FP8Meta]]
    meta_tensor_provider_bwd: Final[Persistent[FP8Meta]]

    def __init__(
        self,
        is_exposed_x_squished_now: bool,
        upcoming_backward: BackwardComm | None,
        op: Op,
        meta_tensor_provider_fwd: Persistent[FP8Meta],
        meta_tensor_provider_bwd: Persistent[FP8Meta],
    ):
        self.is_exposed_x_squished_now = is_exposed_x_squished_now
        self.upcoming_backward = upcoming_backward
        self.next_upcoming_backward = BackwardComm()
        self.op = op
        self.meta_tensor_provider_fwd = meta_tensor_provider_fwd
        self.meta_tensor_provider_bwd = meta_tensor_provider_bwd


_args: ForwardArgs


def get_exposed_y_save_for_backward(
    exposed_x: torch.Tensor, nvte_y: nvte.Tensor, output: torch.Tensor
) -> None:
    return None


def get_exposed_y_backward(
    ctx: FunctionCtx, saved: None, *grads: torch.Tensor
) -> torch.Tensor:
    return grads[0]


@nvte.torch_op(
    save_for_backward=get_exposed_y_save_for_backward,
    backward=lambda ctx, saved, *grads: grads[0],
)
def get_exposed_y(exposed_x: torch.Tensor, nvte_y: nvte.Tensor) -> torch.Tensor:
    x_data = exposed_x.data
    exposed_x.data = torch.Tensor().cuda()  # avoid copy
    exposed_y = exposed_x.clone()  # copy history
    exposed_x.data = x_data
    exposed_y.data = nvte_y.data
    return exposed_y


class ComputePipelineFunction(autograd.Function):
    args: ForwardArgs

    @staticmethod
    def forward(  # type: ignore[arg-type]
        ctx: FunctionCtx,
        exposed_x: torch.Tensor,
        *tensor_mess: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nvte_x = nvte.Tensor(*tensor_mess[-4:])
        del tensor_mess

        nvte.set_execution_state("forward", _args.meta_tensor_provider_fwd)
        nvte_y, to_save = _args.op.forward(nvte_x)

        # Expose backward context for tracing
        bwd_ctx: list[torch.Tensor] = []
        for _, tensor in to_save.items():
            bwd_ctx.append(tensor.data)
            bwd_ctx.append(tensor.amax)
            bwd_ctx.append(tensor.scale)
            bwd_ctx.append(tensor.scale_inv)
        ctx.save_for_backward(*bwd_ctx)

        # Save real context
        setattr(ctx, "nvte_ctx", to_save)
        setattr(ctx, "nvte_op", _args.op)
        setattr(ctx, "nvte_meta_tensor_provider_bwd", _args.meta_tensor_provider_bwd)

        # Pytorch will break the computation graph
        # if it will see an output tensor of an integer type.
        # As fp8 tensors internally have dtype int8,
        # we need to pretend that this type is actually different
        # by "squishing" it into a floating point dtype.
        # ("Squishing" because, while the new dtype is larger,
        # the numel() gets smaller).
        # This doesn't work in TorchScript, but this code
        # won't run at inference anyway.

        # Unsquish x if needed:
        if _args.is_exposed_x_squished_now:
            # Intentionally commented out - _unsquish(exposed_x)
            # We don't need to perform the unsquish itself, as this
            # data will not be read anyway.
            # It is only really neccesarry to notify
            # the backward.
            #
            # If the input to the forward was squished,
            # Pytorch will expect its gradient to be squished
            # as well. The backward of this forward will be
            # responsible for producing the gradient of
            # this squished input, so it is responsible for
            # squishing it.
            setattr(ctx, "nvte_squish_outgoing_dgrad", True)
        else:
            setattr(ctx, "nvte_squish_outgoing_dgrad", False)

        # Expose result for Pytorch
        exposed_y = get_exposed_y(exposed_x, nvte_y)  # type: ignore
        assert isinstance(exposed_y, torch.Tensor)

        # Squish y if fp8:
        if exposed_y.data.dtype == torch.int8:
            _squish(exposed_y)
            # Because the output is squished, the gradient also needs to be.
            # The backward of this forward recieves the gradient of the
            # output as its input. So, the backward before it needs
            # to squish it, while the backward coresponding to this
            # forward needs to unsquish it.
            setattr(ctx, "nvte_unsquish_incoming_dgrad", True)
        else:
            setattr(ctx, "nvte_unsquish_incoming_dgrad", False)

        # Save backward comm
        # This object is allows for the current backward to
        # pass data to the next backward (the backward of the
        # preceding operation). This is needed to pass
        # fp8 gradients properly.
        setattr(ctx, "nvte_upcoming_backward_comm", _args.upcoming_backward)
        setattr(ctx, "nvte_preceding_backward_comm", _args.next_upcoming_backward)

        return (exposed_y, nvte_y.data, nvte_y.amax, nvte_y.scale, nvte_y.scale_inv)

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):  # type: ignore[arg-type]
        # The context needs to think that the tensors were read
        _ = ctx.saved_tensors  # type: ignore

        # Get real context
        saved: Context = getattr(ctx, "nvte_ctx")
        op: Op = getattr(ctx, "nvte_op")
        preceding_backward: BackwardComm = getattr(ctx, "nvte_preceding_backward_comm")
        upcoming_backward: BackwardComm | None = getattr(
            ctx, "nvte_upcoming_backward_comm"
        )

        # Get real gradient
        if preceding_backward.nvte_grad_output is None:
            # This is the first backward in the compute pipeline

            grad_output = grad_output.contiguous()  # TODO: try to avoid this

            # Check if incoming gradient needs to be unsquished
            unsquish_incoming_dgrad: bool = getattr(ctx, "nvte_unsquish_incoming_dgrad")
            if unsquish_incoming_dgrad:
                _unsquish(grad_output)
            nvte_grad = nvte.make_nvte_tensor(grad_output)
        else:
            nvte_grad = preceding_backward.nvte_grad_output
        del grad_output

        meta_tensor_provider: Persistent[FP8Meta] = getattr(
            ctx, "nvte_meta_tensor_provider_bwd"
        )
        nvte.set_execution_state("backward", meta_tensor_provider)
        data_grad, param_grads = op.backward(saved, nvte_grad)

        # Store real gradient for next backward in pipeline
        if upcoming_backward is None:
            # This is the last backward in the compute pipeline
            assert not nvte.is_fp8(data_grad)
        else:
            upcoming_backward.nvte_grad_output = data_grad

        # Check that gradients are not fp8 and can be processed by the optimizer
        # TODO: change this when fp8 optimizer comes along
        assert all(not nvte.is_fp8(g) for g in param_grads)

        # Check if outgoing gradient needs to be squished
        exposed_dgrad = data_grad.data
        squish_outgoing_dgrad: bool = getattr(ctx, "nvte_squish_outgoing_dgrad")
        if squish_outgoing_dgrad:
            _squish(exposed_dgrad)

        torch_grads = [exposed_dgrad] + [g.data for g in param_grads]

        return (*torch_grads, None, None, None)


def apply(x: torch.Tensor, pipeline: ComputePipeline, training: bool) -> torch.Tensor:
    nvte_x = nvte.make_nvte_tensor(x)
    if not training:
        y = pipeline.run_inference(nvte_x)
        assert not nvte.is_fp8(y)
        return y.data
    else:
        pipeline.next_iteration()
        for i, contained_op in enumerate(pipeline.functions):
            global _args
            if i == 0:
                _args = ForwardArgs(
                    False,
                    None,
                    contained_op,
                    pipeline.meta_fwd,
                    pipeline.meta_bwd,
                )
            else:
                _args = ForwardArgs(
                    x.dtype != nvte_x.data.dtype,
                    _args.next_upcoming_backward,
                    contained_op,
                    pipeline.meta_fwd,
                    pipeline.meta_bwd,
                )

            nvte_tensors = contained_op.require_grad()
            exposed_tensors: list[torch.Tensor] = []
            for nvte_tensor in nvte_tensors:
                assert not nvte.is_fp8(
                    nvte_tensor
                )  # TODO: change when fp8 optimizer comes along
                exposed_tensors.append(nvte_tensor.data)

            (x, nvte_x_data, nvte_x_amax, nvte_x_scale, nvte_x_scale_inv) = ComputePipelineFunction.apply(  # type: ignore
                x,
                *exposed_tensors,
                *(nvte_x.data, nvte_x.amax, nvte_x.scale, nvte_x.scale_inv),
            )
            assert isinstance(x, torch.Tensor)
            nvte_x = nvte.Tensor(nvte_x_data, nvte_x_amax, nvte_x_scale, nvte_x_scale_inv)  # type: ignore
        return x


# The squish needs to be invertible and
# always reduce the numel() of the tensor by the same
# amount.
#
# If a tensor is to be squished, it must have been
#   1. an fp8 result from forward
#   2. an outgoing gradient
#
# The outgoing gradient could have any type,
# but it is reasonable to assume that if someone is
# using fp8, they are also probably using bfloat16
# rather than float16.
#
# And they probably won't be using float64.
SQUISH_TABLE = {
    torch.int8: torch.float16,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float64,
}
UNSQUISH_TABLE = {v: k for k, v in SQUISH_TABLE.items()}


def _unsquish(t: torch.Tensor):
    assert t.data.dtype in UNSQUISH_TABLE
    t.data = t.data.view(UNSQUISH_TABLE[t.data.dtype])


def _squish(t: torch.Tensor):
    if t.data.dtype in SQUISH_TABLE:
        t.data = t.data.view(SQUISH_TABLE[t.data.dtype])
    else:
        raise RuntimeError("Invalid dtype of gradient for FP8 tensor.")
