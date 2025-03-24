"""Utility functions for tiny-pytorch.

This module provides various utility functions used throughout the tiny-pytorch
framework. It includes helper functions for data type conversion, object
manipulation, and other common operations needed by the framework's components.

Functions
---------
listify
    Convert any object into a list.
tuplify
    Convert any object into a tuple.
setify
    Convert any object into a set.

Notes
-----
The utility functions in this module are designed to be simple, reusable helpers
that support the core functionality of the framework. They handle common
operations like type conversion and data manipulation that are needed across
multiple modules.
"""

from collections.abc import Iterable


def listify(obj):
    """Convert any object into a list.

    Parameters
    ----------
    obj : object
        Object to convert to list.

    Returns
    -------
    list
        List representation of input object.

    Notes
    -----
    The function handles the following cases:

    - None -> empty list
    - list -> returned as-is
    - string -> single-item list containing the string
    - iterable -> converted to list
    - any other object -> single-item list containing the object

    Examples
    --------
    >>> listify(None)
    []
    >>> listify([1,2,3])
    [1,2,3]
    >>> listify("abc")
    ["abc"]
    >>> listify(range(3))
    [0,1,2]
    >>> listify(5)
    [5]
    """
    if obj is None:
        return []
    elif isinstance(obj, list):
        return obj
    elif isinstance(obj, str):
        return [obj]
    elif isinstance(obj, Iterable):
        return list(obj)
    return [obj]


def tuplify(obj):
    """Convert any object into a tuple.

    Parameters
    ----------
    obj : object
        Object to convert to tuple.

    Returns
    -------
    tuple
        Tuple representation of input object.

    Notes
    -----
    The function handles the following cases:

    - None -> empty tuple
    - tuple -> returned as-is
    - list -> converted to tuple
    - string -> single-item tuple containing the string
    - iterable -> converted to tuple
    - any other object -> single-item tuple containing the object

    Examples
    --------
    >>> tuplify(None)
    ()
    >>> tuplify([1,2,3])
    (1,2,3)
    >>> tuplify("abc")
    ("abc",)
    >>> tuplify(range(3))
    (0,1,2)
    >>> tuplify(5)
    (5,)
    """
    if isinstance(obj, tuple):
        return obj
    return tuple(listify(obj))


def setify(obj):
    """Convert any object into a set.

    Parameters
    ----------
    obj : object
        Object to convert to set.

    Returns
    -------
    set
        Set representation of input object.

    Notes
    -----
    The function handles the following cases:

    - None -> empty set
    - set -> returned as-is
    - list/tuple -> converted to set
    - string -> single-item set containing the string
    - iterable -> converted to set
    - any other object -> single-item set containing the object

    Examples
    --------
    >>> setify(None)
    set()
    >>> setify([1,2,2,3])
    {1,2,3}
    >>> setify("abc")
    {"abc"}
    >>> setify(range(3))
    {0,1,2}
    >>> setify(5)
    {5}
    """
    if isinstance(obj, set):
        return obj
    return set(listify(obj))
