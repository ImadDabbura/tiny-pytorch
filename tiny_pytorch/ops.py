"""Tensor operations module for tiny-pytorch implementation.

This module provides a collection of fundamental tensor operations that form the building blocks
of the computational graph in tiny-pytorch. Each operation is implemented as a class that
inherits from the (TensorOp) base class, with corresponding helper functions for easier usage.

The module includes element-wise operations, matrix operations, and various mathematical
functions commonly used in deep learning.

Classes
-------
ScalarAdd : (TensorOp)
    Addition of a scalar to a tensor.
EWiseAdd : (TensorOp)
    Element-wise addition of two tensors.
ScalarMul : (TensorOp)
    Multiplication of a tensor by a scalar.
EWiseMul : (TensorOp)
    Element-wise multiplication of two tensors.
Negate : (TensorOp)
    Negation of a tensor.
ScalarPower : (TensorOp)
    Raising tensor elements to a scalar power.
EWisePower : (TensorOp)
    Element-wise power operation between two tensors.
ScalarDivide : (TensorOp)
    Division of a tensor by a scalar.
EWiseDivide : (TensorOp)
    Element-wise division of two tensors.
Reshape : (TensorOp)
    Reshaping a tensor to a new shape.
Summation : (TensorOp)
    Summing tensor elements along specified axes.
BroadcastTo : (TensorOp)
    Broadcasting a tensor to a larger shape.
Transpose : (TensorOp)
    Transposing a tensor along specified axes.
MatMul : (TensorOp)
    Matrix multiplication between two tensors.
Log : (TensorOp)
    Natural logarithm of tensor elements.
Exp : (TensorOp)
    Exponential of tensor elements.
ReLU : (TensorOp)
    Rectified Linear Unit activation function.
LogSumExp : (TensorOp)
    Log-sum-exp operation, commonly used in softmax computation.
Stack : (TensorOp)
    Stack a sequence of arrays along a new axis.

Functions
---------
add_scalar(a, scalar)
    Add a scalar to a tensor.
add(a, b)
    Add two tensors element-wise.
mul_scalar(a, scalar)
    Multiply a tensor by a scalar.
multiply(a, b)
    Multiply two tensors element-wise.
negate(a)
    Negate a tensor.
power_scalar(a, scalar)
    Raise tensor elements to a scalar power.
power(a, b)
    Element-wise power operation.
divide_scalar(a, scalar)
    Divide a tensor by a scalar.
divide(a, b)
    Element-wise division of tensors.
reshape(a, shape)
    Reshape a tensor.
summation(a, axes=None)
    Sum tensor elements along specified axes.
broadcast_to(a, shape)
    Broadcast tensor to a larger shape.
transpose(a, axes=None)
    Transpose tensor axes.
matmul(a, b)
    Matrix multiplication.
log(a)
    Natural logarithm.
exp(a)
    Exponential function.
relu(a)
    ReLU activation function.
logsumexp(a, axes=None)
    Log-sum-exp operation.
stack(arrays, axis)
    Stack a sequence of arrays along a new axis.

Notes
-----
All operations support automatic differentiation through their gradient methods,
making them suitable for building and training neural networks.
"""

from __future__ import annotations

from itertools import zip_longest
from typing import Sequence

from . import init
from .backend_selection import NDArray, array_api
from .tensor import Tensor, TensorOp, TensorTuple, TensorTupleOp


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Tensor:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.from_operation(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class ScalarAdd(TensorOp):
    """Add a scalar to a tensor.

    Parameters
    ----------
    scalar : float
        The scalar value to add to the tensor.

    Methods
    -------
    compute(x: NDArray) -> NDArray
        Compute the scalar addition operation.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, x: NDArray):
        return x + self.scalar

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    """Add a scalar value to a tensor.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    scalar : float
        Scalar value to add.

    Returns
    -------
    Tensor
        A new tensor with the scalar added to each element.
    """
    return ScalarAdd(scalar)(a)


class EWiseAdd(TensorOp):
    """Element-wise addition of two tensors.

    Methods
    -------
    compute(x: NDArray, y: NDArray) -> NDArray
        Compute element-wise addition.
    gradient(out_grad: Tensor, out_node: Tensor) -> tuple[Tensor, Tensor]
        Compute the gradient with respect to both inputs.
    """

    def compute(self, x: NDArray, y: NDArray):
        return x + y

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad, out_grad


def add(a, b):
    """Add two tensors element-wise.

    Parameters
    ----------
    a : Tensor
        First input tensor.
    b : Tensor
        Second input tensor.

    Returns
    -------
    Tensor
        Element-wise sum of the input tensors.
    """
    return EWiseAdd()(a, b)


class ScalarMul(TensorOp):
    """Multiply a tensor by a scalar.

    Parameters
    ----------
    scalar : float
        The scalar value to multiply with the tensor.

    Methods
    -------
    compute(x: NDArray) -> NDArray
        Compute the scalar multiplication.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, x: NDArray):
        return self.scalar * x

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad * self.scalar


def mul_scalar(a, scalar):
    """Multiply a tensor by a scalar value.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    scalar : float
        Scalar value to multiply with.

    Returns
    -------
    Tensor
        A new tensor with each element multiplied by the scalar.
    """
    return ScalarMul(scalar)(a)


class EWiseMul(TensorOp):
    """Element-wise multiplication of two tensors.

    Methods
    -------
    compute(x: NDArray, y: NDArray) -> NDArray
        Compute element-wise multiplication.
    gradient(out_grad: Tensor, out_node: Tensor) -> tuple[Tensor, Tensor]
        Compute the gradient with respect to both inputs.
    """

    def compute(self, x: NDArray, y: NDArray):
        return x * y

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad * out_node.inputs[1], out_grad * out_node.inputs[0]


def multiply(a, b):
    """Multiply two tensors element-wise.

    Parameters
    ----------
    a : Tensor
        First input tensor.
    b : Tensor
        Second input tensor.

    Returns
    -------
    Tensor
        Element-wise product of the input tensors.
    """
    return EWiseMul()(a, b)


class Negate(TensorOp):
    """Negate a tensor element-wise.

    Methods
    -------
    compute(x: NDArray) -> NDArray
        Compute the negation operation.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def compute(self, x: NDArray):
        return array_api.negative(x)

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return -out_grad


def negate(a):
    """Negate a tensor element-wise.

    Parameters
    ----------
    a : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        A new tensor with each element negated.
    """
    return Negate()(a)


class ScalarPower(TensorOp):
    """Raise tensor elements to a scalar power.

    Parameters
    ----------
    scalar : float
        The power to raise tensor elements to.

    Methods
    -------
    compute(x: NDArray) -> NDArray
        Compute the power operation.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, x: NDArray):
        return x**self.scalar

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad * self.scalar * out_node.inputs[0] ** (self.scalar - 1)


def power_scalar(a, scalar):
    """Raise tensor elements to a scalar power.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    scalar : float
        Power to raise elements to.

    Returns
    -------
    Tensor
        A new tensor with each element raised to the given power.
    """
    return ScalarPower(scalar)(a)


class EWisePower(TensorOp):
    """Element-wise power operation between two tensors.

    Methods
    -------
    compute(x: NDArray, y: NDArray) -> NDArray
        Compute element-wise power operation.
    gradient(out_grad: Tensor, out_node: Tensor) -> tuple[Tensor, Tensor]
        Compute the gradient with respect to both inputs.
    """

    def compute(self, x: NDArray, y: NDArray):
        return x**y

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        raise NotImplementedError()
        # return (
        #     out_grad
        #     * out_node.inputs[1]
        #     * out_node.inputs[0] ** (out_node.inputs[1] - 1)
        # )


def power(a, b):
    """Raise elements of one tensor to powers specified by another tensor.

    Parameters
    ----------
    a : Tensor
        Base tensor.
    b : Tensor
        Exponent tensor.

    Returns
    -------
    Tensor
        Element-wise power operation result.
    """
    return EWisePower()(a, b)


class ScalarDivide(TensorOp):
    """Divide a tensor by a scalar.

    Parameters
    ----------
    scalar : float
        The scalar value to divide by.

    Methods
    -------
    compute(x: NDArray) -> NDArray
        Compute the scalar division.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, x: NDArray):
        return x / self.scalar

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    """Divide a tensor by a scalar value.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    scalar : float
        Scalar value to divide by.

    Returns
    -------
    Tensor
        A new tensor with each element divided by the scalar.
    """
    return ScalarDivide(scalar)(a)


class EWiseDivide(TensorOp):
    """Element-wise division of two tensors.

    Methods
    -------
    compute(x: NDArray, y: NDArray) -> NDArray
        Compute element-wise division.
    gradient(out_grad: Tensor, out_node: Tensor) -> tuple[Tensor, Tensor]
        Compute the gradient with respect to both inputs.
    """

    def compute(self, x: NDArray, y: NDArray):
        return x / y

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad / out_node.inputs[1], out_grad * (
            -out_node.inputs[0] / out_node.inputs[1] ** 2
        )


def divide(a, b):
    """Divide two tensors element-wise.

    Parameters
    ----------
    a : Tensor
        Numerator tensor.
    b : Tensor
        Denominator tensor.

    Returns
    -------
    Tensor
        Element-wise division of the input tensors.
    """
    return EWiseDivide()(a, b)


class Reshape(TensorOp):
    """Reshape a tensor to a new shape.

    Parameters
    ----------
    shape : tuple
        The target shape for the tensor.

    Methods
    -------
    compute(x: NDArray) -> NDArray
        Compute the reshape operation.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def __init__(self, shape):
        self.shape = shape

    def compute(self, x: NDArray):
        return array_api.reshape(x, self.shape)

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return Reshape(out_node.inputs[0].shape)(out_grad)


def reshape(a, shape):
    """Reshape a tensor to a new shape.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    shape : tuple
        Target shape for the tensor.

    Returns
    -------
    Tensor
        A new tensor with the specified shape.
    """
    return Reshape(shape)(a)


class Summation(TensorOp):
    """Sum tensor elements along specified axes.

    Parameters
    ----------
    axes : tuple or None, optional
        Axes along which to perform summation. If None, sum over all axes.

    Methods
    -------
    compute(a: NDArray) -> NDArray
        Compute the summation operation.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def __init__(self, axes: tuple | None = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        axes = self.axes if self.axes else tuple(range(len(input_shape)))
        tmp_shape = [1 if i in axes else x for i, x in enumerate(input_shape)]
        tmp_out = Reshape(tuple(tmp_shape))(out_grad)
        return BroadcastTo(input_shape)(tmp_out)


def summation(a, axes=None):
    """Sum tensor elements along specified axes.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    axes : tuple or None, optional
        Axes along which to perform summation. If None, sum over all axes.

    Returns
    -------
    Tensor
        Sum of elements along specified axes.
    """
    return Summation(axes)(a)


class BroadcastTo(TensorOp):
    """Broadcast a tensor to a larger shape.

    Parameters
    ----------
    shape : tuple
        Target shape to broadcast to.

    Methods
    -------
    compute(a: NDArray) -> NDArray
        Compute the broadcast operation.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        axes = [
            i
            for i, x in enumerate(zip_longest(input_shape, out_grad.shape))
            if x[0] == 1 or not x[0]
        ]
        out = Summation(tuple(axes))(out_grad)
        out = Reshape(input_shape)(out)
        return out


def broadcast_to(a, shape):
    """Broadcast a tensor to a larger shape.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    shape : tuple
        Target shape to broadcast to.

    Returns
    -------
    Tensor
        Broadcasted tensor with the specified shape.
    """
    return BroadcastTo(shape)(a)


class Transpose(TensorOp):
    """Transpose a tensor along specified axes.

    Parameters
    ----------
    axes : tuple or None, optional
        Permutation of the dimensions. If None, reverse the last two dimensions.

    Methods
    -------
    compute(a: NDArray) -> NDArray
        Compute the transpose operation.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def __init__(self, axes: tuple | None = None):
        self.axes = axes

    def compute(self, a):
        if self.axes is None:
            self.axes = list(range(len(a.shape)))[-2:]
        return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad, node):
        return Transpose(self.axes[::-1])(out_grad)


def transpose(a, axes=None):
    """Transpose a tensor along specified axes.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    axes : tuple or None, optional
        Permutation of the dimensions. If None, reverse the dimensions.

    Returns
    -------
    Tensor
        Transposed tensor.
    """
    return Transpose(axes)(a)


class MatMul(TensorOp):
    """Matrix multiplication between two tensors.

    Methods
    -------
    compute(x: NDArray, y: NDArray) -> NDArray
        Compute matrix multiplication.
    gradient(out_grad: Tensor, out_node: Tensor) -> tuple[Tensor, Tensor]
        Compute the gradient with respect to both inputs.
    """

    def compute(self, x: NDArray, y: NDArray):
        return x @ y

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        x, y = out_node.inputs
        lhs = out_grad @ Transpose()(y)
        rhs = Transpose()(x) @ out_grad
        if lhs.shape != x.shape:
            lhs_sum_axis = len(lhs.shape) - len(x.shape)
            lhs = Summation(axes=tuple(range(lhs_sum_axis)))(lhs)
        if rhs.shape != y.shape:
            rhs_sum_axis = len(rhs.shape) - len(y.shape)
            rhs = Summation(axes=tuple(range(rhs_sum_axis)))(rhs)
        return lhs, rhs


def matmul(a, b):
    """Perform matrix multiplication between two tensors.

    Parameters
    ----------
    a : Tensor
        First input tensor.
    b : Tensor
        Second input tensor.

    Returns
    -------
    Tensor
        Result of matrix multiplication.
    """
    return MatMul()(a, b)


class Log(TensorOp):
    """Natural logarithm of tensor elements.

    Methods
    -------
    compute(x: NDArray) -> NDArray
        Compute natural logarithm.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def compute(self, x: NDArray):
        return array_api.log(x)

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad / out_node.inputs[0]


def log(a):
    """Compute the natural logarithm of tensor elements.

    Parameters
    ----------
    a : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Natural logarithm of input tensor elements.
    """
    return Log()(a)


class Exp(TensorOp):
    """Exponential of tensor elements.

    Methods
    -------
    compute(x: NDArray) -> NDArray
        Compute exponential.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def compute(self, x: NDArray):
        return array_api.exp(x)

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad * Exp()(out_node.inputs[0])


def exp(a):
    """Compute the exponential of tensor elements.

    Parameters
    ----------
    a : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Exponential of input tensor elements.
    """
    return Exp()(a)


class ReLU(TensorOp):
    """Rectified Linear Unit activation function.

    Methods
    -------
    compute(x: NDArray) -> NDArray
        Compute ReLU activation.
    gradient(out_grad: Tensor, out_node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def compute(self, x: NDArray):
        return array_api.maximum(x, 0)

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad * (out_node.realize_cached_data() > 0)


def relu(a):
    """Apply Rectified Linear Unit (ReLU) activation function.

    Parameters
    ----------
    a : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Tensor with ReLU activation applied.
    """
    return ReLU()(a)


class LogSumExp(TensorOp):
    """Log-sum-exp operation, commonly used in softmax computation.

    Parameters
    ----------
    axes : tuple or None, optional
        Axes along which to perform the operation. If None, use all axes.

    Methods
    -------
    compute(Z: NDArray) -> NDArray
        Compute log-sum-exp operation.
    gradient(out_grad: Tensor, node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def __init__(self, axes: tuple | None = None):
        self.axes = axes

    def compute(self, Z):
        self.max = array_api.max(Z, axis=self.axes)
        if self.axes is None:
            axes = tuple(range(len(Z.shape)))
            self.max = array_api.array([self.max], dtype=Z.dtype)
        else:
            axes = self.axes
        self.tmp_shape = [1 if i in axes else x for i, x in enumerate(Z.shape)]
        tmp_max = array_api.reshape(self.max, tuple(self.tmp_shape))
        self.broadcasted_max = array_api.broadcast_to(tmp_max, Z.shape)
        return (
            array_api.log(
                array_api.summation(
                    array_api.exp(Z - self.broadcasted_max), self.axes
                )
            )
            + self.max
        )

    def gradient(self, out_grad, node):
        Z = node.inputs[0] - Tensor(self.broadcasted_max)
        log_sum_exp = BroadcastTo(Z.shape)(
            Reshape(tuple(self.tmp_shape))(Tensor(LogSumExp(self.axes)(Z)))
        )
        log_softmax = Z - log_sum_exp
        return BroadcastTo(Z.shape)(
            Reshape(tuple(self.tmp_shape))(out_grad)
        ) * (Exp()(log_softmax))


def logsumexp(a, axes=None):
    """Compute log-sum-exp along specified axes.

    Parameters
    ----------
    a : Tensor
        Input tensor.
    axes : tuple or None, optional
        Axes along which to perform the operation. If None, use all axes.

    Returns
    -------
    Tensor
        Result of log-sum-exp operation.
    """
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    """Hyperbolic tangent activation function.

    Methods
    -------
    compute(a: NDArray) -> NDArray
        Compute hyperbolic tangent.
    gradient(out_grad: Tensor, node: Tensor) -> Tensor
        Compute the gradient of the operation.
    """

    def compute(self, a: NDArray) -> NDArray:
        return array_api.tanh(a)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return out_grad * (1 - tanh(node.inputs[0] ** 2))


def tanh(a: Tensor) -> Tensor:
    """Compute the hyperbolic tangent of tensor elements.

    Parameters
    ----------
    a : Tensor
        Input tensor.

    Returns
    -------
    Tensor
        Hyperbolic tangent of input tensor elements.
    """
    return Tanh()(a)
