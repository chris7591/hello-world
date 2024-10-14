from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        minux_Z = Z - max_Z
        exp_Z = array_api.exp(minux_Z)
        sum_Z = array_api.sum(exp_Z, axis=self.axes)
        log_Z = array_api.log(sum_Z)

        return log_Z + max_Z.reshape(log_Z.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        max_Z = array_api.max(Z.realize_cached_data(), axis=self.axes, keepdims=True)
        exp_Z = exp(Z - max_Z)
        sum_Z = summation(exp_Z, axes=self.axes)

        return (out_grad / sum_Z).reshape(max_Z.shape).broadcast_to(Z.shape) * exp_Z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

