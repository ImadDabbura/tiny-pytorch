"""Core data structures for multi-dimensional tensors.

This module provides the fundamental Tensor class and related components that form
the backbone of the tiny-pytorch framework. It implements automatic differentiation,
computation graph management, and tensor operations with support for multiple
backends and devices.

The module includes the core Tensor class, operation abstractions, and gradient
computation utilities that enable building and training neural networks with
automatic differentiation capabilities.

Key Features
-----------
- Automatic differentiation with gradient tracking
- Computation graph construction and management
- Support for multiple backends (NumPy, CPU, CUDA)
- Lazy evaluation mode for memory efficiency
- Tensor operations with automatic broadcasting
- Gradient computation and backpropagation
- Device and dtype management

Classes
-------
Op
    Base class for all tensor operations. Defines the interface for operations
    that can be applied to tensors to create new tensors in the computation graph.
TensorOp : Op
    Base class for operations that produce single tensors.
TensorTupleOp : Op
    Base class for operations that produce tuples of tensors.
Tensor
    Multi-dimensional tensor with automatic differentiation support.
    The core data structure for representing inputs, outputs, and intermediate
    results in neural network computations.
TensorTuple : Tensor
    Specialized tensor class for representing tuples of tensors.

Functions
---------
compute_gradients(out_tensor, out_grad) -> None
    Compute gradients for all tensors in the computation graph.
find_topo_sort(node_list) -> list[Tensor]
    Find topological sort of tensors in the computation graph.
_topo_sort_dfs(node, visited, topo_list) -> None
    Depth-first search for topological sorting.

Notes
-----
The Tensor system implements automatic differentiation through a computation graph
where each tensor operation creates a new tensor node that tracks its inputs and
the operation that produced it. When backward() is called on a tensor, gradients
are computed and propagated through the graph using the chain rule.

The system supports both eager and lazy evaluation modes. In eager mode (default),
tensor values are computed immediately. In lazy mode, computation is deferred
until the tensor value is actually needed.

All tensor operations are designed to work seamlessly with the automatic
differentiation system, automatically tracking gradients when requires_grad=True.

Examples
--------
>>> import tiny_pytorch as tp
>>>
>>> # Create tensors
>>> x = tp.Tensor([1, 2, 3], requires_grad=True)
>>> y = tp.Tensor([4, 5, 6], requires_grad=True)
>>>
>>> # Perform operations
>>> z = x * y + 2  # Automatic gradient tracking
>>> loss = z.sum()
>>>
>>> # Compute gradients
>>> loss.backward()
>>> print(x.grad)  # Gradient with respect to x
>>> print(y.grad)  # Gradient with respect to y
>>>
>>> # Use different devices
>>> x_cpu = tp.Tensor([1, 2, 3], device=tp.cpu())
>>> x_cuda = tp.Tensor([1, 2, 3], device=tp.cuda())
"""

from __future__ import annotations

import numpy as np

import tiny_pytorch

from . import init
from .backend_selection import Device, NDArray, array_api, default_device
from .utils import listify, tuplify

LAZY_MODE = False  # Default mode is eager mode


class Op:
    """Base class for all tensor operations.

    This class defines the interface that all tensor operations must implement.
    Operations are callable objects that can be applied to tensors to create
    new tensors in the computation graph.

    Methods
    -------
    __call__(*args)
        Apply the operation to the given arguments.
    compute(*args)
        Compute the actual operation on the underlying arrays.
    gradient(out_grad, out_node)
        Compute the gradient of the operation.
    """

    def __call__(self, *args):
        """Apply the operation to the given arguments.

        Parameters
        ----------
        *args : Tensor
            Input tensors to the operation.

        Returns
        -------
        Tensor
            Result of applying the operation to the inputs.
        """
        return Tensor.from_operation(self, args)

    def compute(self, *args: tuple[NDArray]):
        """Compute the actual operation on the underlying arrays.

        Parameters
        ----------
        *args : tuple[NDArray]
            Input arrays to the operation.

        Returns
        -------
        NDArray
            Result of the operation.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def gradient(self, out_grad: Tensor, out_node: Tensor):
        """Compute the gradient of the operation.

        Parameters
        ----------
        out_grad : Tensor
            Gradient of the output with respect to the final result.
        out_node : Tensor
            The output tensor of this operation.

        Returns
        -------
        Tensor or tuple[Tensor]
            Gradient(s) with respect to the input(s) of this operation.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError()

    def gradient_as_tuple(
        self, out_grad: Tensor, node: Tensor
    ) -> tuple[Tensor]:
        """Convenience method to always return a tuple from gradient call"""
        out = self.gradient(out_grad, node)
        return tuplify(out)


class TensorOp(Op):
    """Op class specialized to output tensors, will be alterate subclasses for other structures"""

    def __call__(self, *args):
        return Tensor.from_operation(self, args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.from_operation(self, args)


class Tensor:
    """
    Tensor is the fundamental data structure in tiny_pytorch. It is a multi-dimensional array of numerical values used to represent inputs, outputs, and intermediate results in a computation graph.

    Attributes
    ----------
    cached_data : list[object]
        The cached data of the tensor.
    inputs : list[Tensor]
        The input tensors to the operation that produced this tensor.
    op : Op
        The operation that produced this tensor.
    requires_grad : bool
        If True, the tensor will track gradients.
    """

    def __init__(
        self,
        array,
        *,
        device: Device | None = None,
        dtype: str | None = None,
        requires_grad: bool = True,
    ) -> None:
        """
        Construct a Tensor by copying `array`.

        Parameters
        ----------
        array : object
            The array to be copied.
        device : Device, optional
            The device on which to place the tensor. Default is None.
        dtype : str, optional
            The data type of the tensor. Default is None.
        requires_grad : bool, optional
            If True, the tensor will track gradients. Default is True.
        """
        if isinstance(array, Tensor):
            device = array.device if not device else device
            dtype = array.dtype if not dtype else dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # Use numpy as brige
                cached_data = self._from_numpy_array(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = default_device() if not device else device
            cached_data = self._from_numpy_array(
                array, device=device, dtype=dtype
            )
        self._init(cached_data=cached_data, requires_grad=requires_grad)

    def _init(
        self,
        inputs: list[Tensor] | None = None,
        op: Op | None = None,
        *,
        cached_data: list[object] | None = None,
        requires_grad: bool | None = None,
    ):
        self.inputs = listify(inputs)
        self.op = op
        self.cached_data = cached_data
        if requires_grad is None:
            # If any of the input Tensors have requires_grad -> output will requires_grad
            requires_grad = any([x.requires_grad for x in self.inputs])
        self.requires_grad = requires_grad

    def _from_numpy_array(self, array, device, dtype):
        if array_api is np:
            return np.array(array, dtype=dtype)
        return array_api.array(array, device=device, dtype=dtype)

    def realize_cached_data(self):
        """Run computation to get the output if the LAZY MODE is on, else return cached data."""
        if self.cached_data is not None:
            return self.cached_data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def numpy(self):
        """
        Returns `Tensor` as Numpy ndarray. The underlying data will be shared
        between Tensor and the Numpy ndarray.
        """
        data = self.realize_cached_data()
        if array_api is np:
            return data
        return data.numpy()

    @property
    def shape(self):
        """
        Returns the shape of the tensor as a tuple.

        Returns
        -------
        tuple
            Shape of the tensor.
        """
        return self.realize_cached_data().shape

    @property
    def ndim(self):
        """
        Returns the number of dimensions of the tensor.

        Returns
        -------
        int
            Number of dimensions of the tensor.
        """
        return self.realize_cached_data().ndim

    @property
    def dtype(self):
        """
        Returns the data type of the tensor.

        Returns
        -------
        dtype : numpy.dtype
            The data type of the tensor.
        """
        return self.realize_cached_data().dtype

    @property
    def device(self):
        """
        Returns the device on which the tensor is stored.

        Returns
        -------
        device : Device
            The device on which the tensor is stored.
        """
        if array_api is np:
            return default_device()
        return self.realize_cached_data().device

    @property
    def data(self):
        """Returns a detached Tensor with the original data."""
        return self.detach()

    @data.setter
    def data(self, data):
        assert isinstance(
            data, Tensor
        ), f"data must be of type `Tensor`, {type(data)} is given"
        assert self.dtype == data.dtype, (
            f"data must be of the same type as `Tensor`, "
            f"{self.dtype} != {data.dtype}"
        )
        self.cached_data = data.realize_cached_data()

    @classmethod
    def from_constant(cls, data, requires_grad: bool = False):
        """Creates a leaf node Tensor from the given `data`."""
        tensor = Tensor.__new__(cls)
        tensor._init(
            cached_data=data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @classmethod
    def from_operation(cls, op: Op, inputs: tuple[Tensor]):
        """
        Creates a node Tensor by applying the `op` operation on the `inputs`
        Tensors.
        """
        tensor = Tensor.__new__(cls)
        tensor._init(inputs, op)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    def detach(self):
        """
        Returns a new Tensor with no history (detached from the computation
        graph). The returned Tensor will share the same data with the
        original one.
        """
        return Tensor.from_constant(self)

    def is_leaf(self):
        """
        All Tensors that have `requires_grad` set to `False` OR they were
        created by the user and were not the result of an operation are
        considered leaf Tensors.
        """
        return self.op is None

    def __repr__(self):
        """String representation of the tensor.

        Returns
        -------
        str
            String representation showing the tensor data.
        """
        return f"tiny_pytorch.Tensor({str(self.realize_cached_data())})"

    def __str__(self):
        """String representation of the tensor.

        Returns
        -------
        str
            String representation of the tensor data.
        """
        return str(self.realize_cached_data())

    def __add__(self, other):
        """Add another tensor or scalar to this tensor.

        Parameters
        ----------
        other : Tensor or scalar
            The tensor or scalar to add.

        Returns
        -------
        Tensor
            Result of the addition operation.
        """
        if isinstance(other, Tensor):
            return tiny_pytorch.ops.EWiseAdd()(self, other)
        return tiny_pytorch.ops.ScalarAdd(other)(self)

    def __neg__(self):
        """Negate this tensor.

        Returns
        -------
        Tensor
            Negated tensor.
        """
        return tiny_pytorch.ops.Negate()(self)

    def __sub__(self, other):
        """Subtract another tensor or scalar from this tensor.

        Parameters
        ----------
        other : Tensor or scalar
            The tensor or scalar to subtract.

        Returns
        -------
        Tensor
            Result of the subtraction operation.
        """
        if isinstance(other, Tensor):
            return tiny_pytorch.ops.EWiseAdd()(self, -other)
        return tiny_pytorch.ops.ScalarAdd(-other)(self)

    def __mul__(self, other):
        """Multiply this tensor by another tensor or scalar.

        Parameters
        ----------
        other : Tensor or scalar
            The tensor or scalar to multiply by.

        Returns
        -------
        Tensor
            Result of the multiplication operation.
        """
        if isinstance(other, Tensor):
            return tiny_pytorch.ops.EWiseMul()(self, other)
        return tiny_pytorch.ops.ScalarMul(other)(self)

    def __pow__(self, other):
        """Raise this tensor to the power of another tensor or scalar.

        Parameters
        ----------
        other : Tensor or scalar
            The exponent.

        Returns
        -------
        Tensor
            Result of the power operation.
        """
        if isinstance(other, Tensor):
            return tiny_pytorch.ops.EWisePower()(self, other)
        return tiny_pytorch.ops.ScalarPower(other)(self)

    def __truediv__(self, other):
        """Divide this tensor by another tensor or scalar.

        Parameters
        ----------
        other : Tensor or scalar
            The tensor or scalar to divide by.

        Returns
        -------
        Tensor
            Result of the division operation.
        """
        if isinstance(other, Tensor):
            return tiny_pytorch.ops.EWiseDivide()(self, other)
        return tiny_pytorch.ops.ScalarDivide(other)(self)

    def __matmul__(self, other):
        """Matrix multiplication with another tensor.

        Parameters
        ----------
        other : Tensor
            The tensor to multiply with.

        Returns
        -------
        Tensor
            Result of the matrix multiplication.
        """
        return tiny_pytorch.ops.MatMul()(self, other)

    def sum(self, axes=None):
        """
        Returns the sum of elements over specified axes.

        Parameters
        ----------
        axes : None or int or tuple of ints, optional
            Axis or axes along which a sum is performed. The default is to sum all of the elements of the input tensor.

        Returns
        -------
        Tensor
            A new tensor with the sum of elements over specified axes.
        """
        return tiny_pytorch.ops.Summation(axes)(self)

    def reshape(self, shape):
        """
        Reshapes the tensor to the specified shape.

        Parameters
        ----------
        shape : tuple of ints
            The new shape of the tensor.

        Returns
        -------
        Tensor
            A new tensor with the specified shape.
        """
        return tiny_pytorch.ops.Reshape(shape)(self)

    def broadcast_to(self, shape):
        """
        Broadcasts the tensor to the specified shape.

        Parameters
        ----------
        shape : tuple of ints
            The new shape of the tensor.

        Returns
        -------
        Tensor
            A new tensor with the specified shape.
        """
        return tiny_pytorch.ops.BroadcastTo(shape)(self)

    def transpose(self, axes=None):
        """
        Transposes the tensor according to the specified axes.

        Parameters
        ----------
        axes : tuple of ints, optional
            By default, reverse the dimensions, otherwise permute the axes according to the values given.

        Returns
        -------
        Tensor
            A new tensor with the specified axes transposed.
        """
        return tiny_pytorch.ops.Transpose(axes)(self)

    def backward(self, out_grad: Tensor | None = None):
        """
        Computes the gradients of the tensor with respect to the output gradient.

        Parameters
        ----------
        out_grad : Tensor, optional
            The gradient of the output with respect to which the gradients are computed. If None, a tensor of ones is used.

        Returns
        -------
        None
            This method updates the `grad` attribute of the tensor and its dependencies with the computed gradients.
        """
        out_grad = (
            out_grad
            if out_grad
            else Tensor(
                init.ones(*self.shape, dtype=self.dtype, device=self.device)
            )
        )
        compute_gradients(self, out_grad)

    def _compute_gradients(self, out_grad):
        pass

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__


class TensorTuple(Tensor):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return tiny_pytorch.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "tiny_pytorch.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return tiny_pytorch.ops.make_tuple(
            *[self[i] + other[i] for i in range(len(self))]
        )

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return TensorTuple.make_const(self.realize_cached_data())


def compute_gradients(out_tensor, out_grad):
    """Compute gradients for all nodes in the computation graph.

    This function implements reverse-mode automatic differentiation by
    traversing the computation graph in reverse topological order and
    computing gradients for each node.

    Parameters
    ----------
    out_tensor : Tensor
        The output tensor for which gradients are computed.
    out_grad : Tensor
        The gradient of the output with respect to the final result.

    Notes
    -----
    This function modifies the `grad` attribute of tensors in the computation
    graph. It stores the computed result in the grad field of each tensor.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: dict[Tensor, list[Tensor]] = {}
    node_to_output_grads_list[out_tensor] = [out_grad]

    # Traverse graph in reverse topological order given
    # the output_node that we are taking gradient wrt.
    reverse_topo_order = find_topo_sort([out_tensor])[::-1]

    for out_tensor in reverse_topo_order:
        out_grad = sum(node_to_output_grads_list[out_tensor])
        out_tensor.grad = out_grad
        if out_tensor.op:
            partial_adjoints = tuplify(
                out_tensor.op.gradient(out_grad, out_tensor)
            )
            for input_node, partial_adjoint in zip(
                out_tensor.inputs, partial_adjoints
            ):
                node_to_output_grads_list.setdefault(input_node, []).append(
                    partial_adjoint
                )


def find_topo_sort(node_list: list[Tensor]) -> list[Tensor]:
    """Find topological sort of nodes in the computation graph.

    Given a list of nodes, return a topological sort list of nodes ending in them.
    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    Parameters
    ----------
    node_list : list[Tensor]
        List of tensors to sort topologically.

    Returns
    -------
    list[Tensor]
        Topologically sorted list of tensors.
    """
    visited = []
    topo_list = []
    for output_node in node_list:
        for input_node in _topo_sort_dfs(output_node, visited, topo_list):
            topo_list.append(input_node)
    return topo_list


def _topo_sort_dfs(node, visited, topo_list):
    """Perform post-order DFS for topological sorting.

    Parameters
    ----------
    node : Tensor
        Current node in the DFS traversal.
    visited : list[Tensor]
        List of already visited nodes.
    topo_list : list[Tensor]
        List to collect nodes in topological order.

    Yields
    ------
    Tensor
        Nodes in post-order DFS traversal.
    """
    for input_node in node.inputs:
        if input_node not in visited:
            yield from _topo_sort_dfs(input_node, visited, topo_list)
    visited.append(node)
    if node not in topo_list:
        yield node
