from functools import wraps
from types import FunctionType
from typing import Union

import lab as B

from . import _dispatch

__all__ = ["uprank", "num_elements", "expand"]


@_dispatch
def uprank(x):
    """Ensure that the rank of `x` is 2.

    Args:
        x (tensor): Tensor to ensure the rank of.

    Returns:
        tensor: `x` with rank at least 2.
    """
    # Simply return non-numerical inputs.
    return x


@_dispatch
def uprank(x: B.Numeric):
    return B.uprank(x)


@_dispatch
def uprank(f: FunctionType):
    """A decorator to ensure that the rank of the arguments is two."""

    @wraps(f)
    def wrapped_f(*args):
        return f(*[uprank(x) for x in args])

    return wrapped_f


@_dispatch
def num_elements(x):
    """Determine the number of elements in an input.

    Deals with scalars, vectors, matrices, and batches of matrices.

    Args:
        x (tensor): Input.

    Returns:
        int: Number of elements.
    """
    return B.shape_matrix(x, 0)


@_dispatch
def expand(xs: Union[tuple, list]):
    """Expand a sequence to the same element repeated twice if there is only one
    element.

    Args:
        xs (tuple or list): Sequence to expand.

    Returns:
        tuple or list: `xs * 2` or `xs`.
    """
    return xs * 2 if len(xs) == 1 else xs
