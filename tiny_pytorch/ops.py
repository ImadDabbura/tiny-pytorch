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
from typing import Optional, Sequence

from . import init
from .backend_selection import NDArray, array_api
from .tensor import Tensor, TensorOp, TensorTuple, TensorTupleOp
from .utils import tuplify


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
        return x * self.scalar

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
        lhs, rhs = out_node.inputs
        return out_grad * rhs, out_grad * lhs


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
        return -x

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
        lhs, rhs = out_node.inputs
        return out_grad / rhs, out_grad * (-lhs / rhs**2)


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
        return x.compact().reshape(self.shape)

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        return out_grad.reshape(out_node.inputs[0].shape)


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
        # Our sum method/func in ndarray only sums over either all axes
        # Or one axis -> If we are summing over multiple axes (not all)
        # We need to sum one axis at a time starting from most outer
        # axes to inner axes
        if isinstance(self.axes, (tuple, list)) and len(self.axes) > 1:
            for axis in reversed(sorted(self.axes)):
                a = a.sum(axis=axis)
            return a
        return a.sum(self.axes)

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        axes = (
            tuplify(self.axes)
            if self.axes is not None
            else tuple(range(len(input_shape)))
        )
        new_shape = [1 if i in axes else x for i, x in enumerate(input_shape)]
        return out_grad.reshape(new_shape).broadcast_to(input_shape)


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
        if a.shape == self.shape:
            return a
        return a.broadcast_to(self.shape).compact()

    def gradient(self, out_grad, node):
        input_shape = node.inputs[0].shape
        if input_shape == out_grad.shape:
            return out_grad
        # Assuming here that axis that would be broadcasted should already
        # exist NOT creating new axes -> If axes don't exist, user must first
        # reshape the array to have the same dimensions with 1 for all new axes
        # Then call broadcast
        axes = [
            i
            for i, x in enumerate(zip_longest(input_shape, out_grad.shape))
            if x[0] == 1 or not x[0]
        ]
        return out_grad.sum(axes).reshape(input_shape)


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
        if self.axes:
            ax0, ax1 = self.axes[0], self.axes[1]
        else:
            ax0, ax1 = a.ndim - 2, a.ndim - 1
        permute_axes = list(range(a.ndim))
        permute_axes[ax0], permute_axes[ax1] = ax1, ax0
        return a.permute(permute_axes)

    def gradient(self, out_grad, node):
        return out_grad.transpose(self.axes)


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
        lhs = out_grad @ y.transpose()
        rhs = x.transpose() @ out_grad
        if x.shape < lhs.shape:
            lhs_sum_axis = len(lhs.shape) - len(x.shape)
            lhs = lhs.sum(tuple([i for i in range(lhs_sum_axis)]))
        if y.shape < rhs.shape:
            rhs_sum_axis = len(rhs.shape) - len(y.shape)
            rhs = rhs.sum(tuple([i for i in range(rhs_sum_axis)]))
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
        return out_grad * exp(out_node.inputs[0])


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
        return out_grad * Tensor(
            out_node.realize_cached_data() > 0, device=out_grad.device
        )


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
        self.max = Z.max(axis=self.axes, keepdims=True)
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


class Stack(TensorOp):
    """Stack a sequence of arrays along a new axis.

    Parameters
    ----------
    axis : int
        The axis along which to stack. The new axis will be inserted
        at this position in the result array shape.

    Methods
    -------
    compute(args: list[NDArray]) -> NDArray
        Stack the input arrays along the specified axis.
    gradient(out_grad: Tensor, node: Tensor) -> Tensor
        Compute the gradient of the stack operation (returns split of out_grad along axis).
    """

    def __init__(self, axis: int) -> None:
        self.axis = axis

    def compute(self, *arrays: list[NDArray]) -> NDArray:
        n = len(arrays)
        assert n > 0, "Stack needs at least one array!"
        shape = arrays[0].shape
        for arr in arrays:
            assert (
                shape == arr.shape
            ), "All arrays need to be of the same size!"
        new_shape = list(shape)
        new_shape.insert(self.axis, n)
        slices = [slice(0, s) for s in new_shape]
        out = array_api.empty(new_shape, device=arrays[0].device)
        for i, arr in enumerate(arrays):
            # index at the new dimension is always an integer which
            # is the index of the to-be inserted array in the list of
            # arrays passed as argument
            slices[self.axis] = slice(i, i + 1)
            out[tuple(slices)] = arr
        return out

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        # Gradient of stack is split and vice versa
        return split(out_grad, self.axis)


def stack(arrays: Sequence[Tensor], axis: int) -> Tensor:
    """Stack a sequence of tensors along a new axis.

    Parameters
    ----------
    arrays : list of Tensor
        Sequence of tensors to stack. All tensors must have the same shape.
    axis : int
        The axis along which to stack. The new axis will be inserted at this position in the result tensor shape.

    Returns
    -------
    Tensor
        The stacked tensor with one more dimension than the input tensors.
    """
    return Stack(axis)(*arrays)


class Split(TensorTupleOp):
    """Split a tensor along an axis into a tuple of tensors.

    This operation is the inverse of Stack. It splits a tensor along a specified axis
    into multiple tensors, each with one less dimension than the input tensor.

    Parameters
    ----------
    axis : int
        The axis along which to split the tensor. The axis dimension will be removed
        from each resulting tensor.

    Methods
    -------
    compute(A: NDArray) -> tuple[NDArray, ...]
        Split the input array along the specified axis.
    gradient(out_grad: TensorTuple, node: Tensor) -> Tensor
        Compute the gradient of the split operation (returns stack of out_grad tensors).
    """

    def __init__(self, axis: int) -> None:
        self.axis = axis

    def compute(self, A: NDArray) -> tuple[NDArray, ...]:
        n = A.shape[self.axis]
        new_shape = list(A.shape)
        new_shape.pop(self.axis)
        slices = [slice(0, s) for s in A.shape]
        splits = []
        for i in range(n):
            slices[self.axis] = slice(i, i + 1)
            splits.append(A[tuple(slices)].compact().reshape(new_shape))
        return tuple(splits)

    def gradient(self, out_grad: TensorTuple, node: Tensor) -> Tensor:
        return stack(out_grad, self.axis)


def split(a: Tensor, axis: int) -> TensorTuple:
    """Split a tensor along an axis into a tuple of tensors.

    This function splits a tensor along a specified axis into multiple tensors.
    Each resulting tensor has one less dimension than the input tensor.

    Parameters
    ----------
    a : Tensor
        Input tensor to split.
    axis : int
        The axis along which to split the tensor. The axis dimension will be removed
        from each resulting tensor.

    Returns
    -------
    TensorTuple
        A tuple of tensors, each with the specified axis dimension removed.
        The number of tensors in the tuple equals the size of the input tensor
        along the specified axis.

    Examples
    --------
    >>> x = Tensor([[1, 2, 3], [4, 5, 6]])
    >>> result = split(x, axis=0)
    >>> len(result)  # Returns 2 tensors
    2
    >>> result[0].shape  # Each tensor has shape (3,)
    (3,)
    """
    return Split(axis)(a)


class Flip(TensorOp):
    """
    Reverse (flip) the order of elements in a tensor along the specified axes.

    Parameters
    ----------
    axes : tuple[int, ...] or None, optional
        Axes along which to flip the tensor. Each axis index must be valid for the tensor's dimensions.
        If None, flip over all axes (reverse the tensor in every dimension).

    Methods
    -------
    compute(a: NDArray) -> NDArray
        Compute the flip operation on the input NDArray.
    gradient(out_grad: Tensor, node: Tensor) -> Tensor
        Compute the gradient of the flip operation (flip the gradient along the same axes).

    Raises
    ------
    numpy.AxisError
        If the number of axes is greater than the number of dimensions, or if any axis is out of bounds.

    Examples
    --------
    >>> x = Tensor([[1, 2], [3, 4]])
    >>> Flip((0,))(x)
    Tensor([[3, 4], [1, 2]])
    >>> Flip((1,))(x)
    Tensor([[2, 1], [4, 3]])
    >>> Flip((0, 1))(x)
    Tensor([[4, 3], [2, 1]])
    >>> Flip()(x)
    Tensor([[4, 3], [2, 1]])
    """

    def __init__(self, axes: tuple[int, ...] | None = None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray:
        return a.flip(self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        return flip(out_grad, self.axes)


def flip(a: Tensor, axes: tuple[int, ...] | None = None) -> Tensor:
    """
    Reverse (flip) the order of elements in a tensor along the specified axes.

    Parameters
    ----------
    a : Tensor
        Input tensor to be flipped.
    axes : tuple[int, ...] or None, optional
        Axes along which to flip the tensor. Each axis index must be valid for the tensor's dimensions.
        If None, flip over all axes (reverse the tensor in every dimension).

    Returns
    -------
    Tensor
        A tensor with the entries reversed along the specified axes.

    Raises
    ------
    numpy.AxisError
        If the number of axes is greater than the number of dimensions, or if any axis is out of bounds.

    Examples
    --------
    >>> x = Tensor([[1, 2], [3, 4]])
    >>> flip(x, (0,))
    Tensor([[3, 4], [1, 2]])
    >>> flip(x, (1,))
    Tensor([[2, 1], [4, 3]])
    >>> flip(x, (0, 1))
    Tensor([[4, 3], [2, 1]])
    >>> flip(x)
    Tensor([[4, 3], [2, 1]])
    """
    return Flip(axes)(a)


class Dilate(TensorOp):
    """Dilate a tensor by inserting zeros between elements along specified axes.

    This operation inserts zeros between elements along the specified axes, effectively
    increasing the size of the tensor in those dimensions. This is commonly used in
    convolutional neural networks for dilated convolutions.

    Parameters
    ----------
    axes : tuple[int, ...]
        The axes along which to apply dilation. Each axis index must be valid for the tensor's dimensions.
    dilation : int
        The dilation factor. For each element in the original tensor, `dilation` zeros
        will be inserted after it along the specified axes.

    Methods
    -------
    compute(a: NDArray) -> NDArray
        Compute the dilation operation on the input NDArray.
    gradient(out_grad: Tensor, node: Tensor) -> Tensor
        Compute the gradient of the dilation operation (returns undilated gradient).

    Examples
    --------
    >>> x = Tensor([[1, 2], [3, 4]])
    >>> Dilate((0,), 1)(x)
    Tensor([[1, 2], [0, 0], [3, 4]])
    >>> Dilate((1,), 1)(x)
    Tensor([[1, 0, 2], [3, 0, 4]])
    >>> Dilate((0, 1), 1)(x)
    Tensor([[1, 0, 2], [0, 0, 0], [3, 0, 4]])
    """

    def __init__(self, axes: tuple[int, ...], dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray) -> NDArray:
        new_shape = tuple(
            (
                a.shape[i] * (self.dilation + 1)
                if i in self.axes and i < a.ndim
                else a.shape[i]
            )
            for i in range(a.ndim)
        )
        arr = a.device.full(new_shape, 0)
        slices = tuple(
            (
                slice(0, arr.shape[i], self.dilation + 1)
                if i in self.axes and i < a.ndim
                else slice(0, n)
            )
            for i, n in enumerate(a.shape)
        )
        arr[tuple(slices)] = a
        return arr

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor:
        # For now, return the gradient as-is since UnDilate is not implemented
        # TODO: Implement UnDilate class for proper gradient computation
        return out_grad


def dilate(a: Tensor, axes: tuple[int, ...], dilation: int) -> Tensor:
    """Dilate a tensor by inserting zeros between elements along specified axes.

    This function inserts zeros between elements along the specified axes, effectively
    increasing the size of the tensor in those dimensions. This is commonly used in
    convolutional neural networks for dilated convolutions.

    Parameters
    ----------
    a : Tensor
        Input tensor to be dilated.
    axes : tuple[int, ...]
        The axes along which to apply dilation. Each axis index must be valid for the tensor's dimensions.
    dilation : int
        The dilation factor. For each element in the original tensor, `dilation` zeros
        will be inserted after it along the specified axes.

    Returns
    -------
    Tensor
        A dilated tensor with zeros inserted along the specified axes.

    Examples
    --------
    >>> x = Tensor([[1, 2], [3, 4]])
    >>> dilate(x, (0,), 1)
    Tensor([[1, 2], [0, 0], [3, 4]])
    >>> dilate(x, (1,), 1)
    Tensor([[1, 0, 2], [3, 0, 4]])
    >>> dilate(x, (0, 1), 1)
    Tensor([[1, 0, 2], [0, 0, 0], [3, 0, 4]])
    """
    return Dilate(axes, dilation)(a)
