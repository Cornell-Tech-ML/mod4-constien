from typing import Tuple

import numpy as np

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    new_height = height // kh
    new_width = width // kw

    input = (
        input.contiguous()
        .view(batch, channel, new_height, kh, new_width, kw)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
        .view(batch, channel, new_height, new_width, kh * kw)
    )

    return input, new_height, new_width


def avgpool2d(t: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Pools together kernel-sized windows of passed tensor using average"""
    batch, channel, _, _ = t.shape

    t, height, width = tile(t, kernel)
    return t.mean(dim=4).view(batch, channel, height, width)


MAX = FastOps.reduce(operators.max, np.iinfo(np.int32).min)


def argmax(t: Tensor, dim: int) -> Tensor:
    """Computes the position of the maximum value in a tensor along the passed dimension"""
    # TODO: Max unique
    return MAX(t, dim) == t


class Max(Function):
    """Maximum function $f(x, y) = max{x, y}$"""

    @staticmethod
    def forward(ctx: Context, t: Tensor, dim: Tensor) -> Tensor:
        """Invoke maximum function saving arguments into context as necessary"""
        d = int(dim.item())
        ctx.save_for_backward(t, d)
        return MAX(t, d)

    @staticmethod
    def backward(ctx: Context, grad: Tensor) -> tuple[Tensor, float]:
        """Compute maximum derivative on arguments in context, scaled by arbitrary input"""
        t, dim = ctx.saved_values
        return argmax(t, dim) * grad, 0.0


def max(t: Tensor, dim: int) -> Tensor:
    """Computes the maximum value of a tensor along the passed dimension"""
    return Max.apply(t, t._ensure_tensor(dim))


def softmax(t: Tensor, dim: int) -> Tensor:
    """Computes the softmax (softmax(x)_{i} := e^{x_i} / sum_{j}{e^x_j}) of a tensor along the passed dimension"""
    t = t.exp()
    return t / t.sum(dim)


def logsoftmax(t: Tensor, dim: int) -> Tensor:
    """Computes the log softmax (logsoftmax(x)_{i} := ln(softmax(x)_{i})) of a tensor along the passed dimension"""
    m = max(t, dim)
    i = t - m
    return i - i.exp().sum(dim).log()


def maxpool2d(t: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Pools together kernel-sized windows of passed tensor using maximum"""
    batch, channel, _, _ = t.shape

    t, height, width = tile(t, kernel)
    return max(t, 4).view(batch, channel, height, width)


def dropout(t: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Dropout of random positions to model background noise - if ignore, randomly sets rate% of values to 0"""
    return t if ignore else (t * (rand(t.shape) > rate))
