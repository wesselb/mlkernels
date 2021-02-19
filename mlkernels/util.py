from functools import wraps
from types import FunctionType

import lab as B

from . import _dispatch

__all__ = ["uprank", "num_elements", "expand"]


@_dispatch(object)
def uprank(x):
    """Ensure that the rank of `x` is 2.

    Args:
        x (tensor): Tensor to ensure the rank of.

    Returns:
        tensor: `x` with rank at least 2.
    """
    # Simply return non-numerical inputs.
    return x


@_dispatch(B.Numeric)
def uprank(x):
    return B.uprank(x)


@_dispatch(FunctionType)
def uprank(f):
    """A decorator to ensure that the rank of the arguments is two."""

    @wraps(f)
    def wrapped_f(*args):
        return f(*[uprank(x) for x in args])

    return wrapped_f


@_dispatch(object)
def num_elements(x):
    """Determine the number of elements in an input.

    Deals with scalars, vectors, and matrices.

    Args:
        x (tensor): Input.

    Returns:
        int: Number of elements.
    """
    shape = B.shape(x)
    if shape == ():
        return 1
    else:
        return shape[0]


@_dispatch({tuple, list})
def expand(xs):
    """Expand a sequence to the same element repeated twice if there is only one
    element.

    Args:
        xs (tuple or list): Sequence to expand.

    Returns:
        tuple or list: `xs * 2` or `xs`.
    """
    return xs * 2 if len(xs) == 1 else xs
