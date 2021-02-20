import lab as B
import numpy as np
from algebra import DerivativeFunction
from algebra.util import identical
from matrix import Dense
from plum import convert, Dispatcher, Self

from . import _dispatch
from .. import Kernel
from ..util import num_elements, uprank, expand

__all__ = ["perturb", "DerivativeKernel"]


def dkx(k_elwise, i):
    """Construct the derivative of a kernel with respect to its first
    argument.

    Args:
        k_elwise (function): Function that performs element-wise computation
            of the kernel.
        i (int): Dimension with respect to which to compute the derivative.

    Returns:
        function: Derivative of the kernel with respect to its first argument.
    """

    @uprank
    def _dkx(x, y):
        import tensorflow as tf

        with tf.GradientTape() as t:
            # Get the numbers of inputs.
            nx = num_elements(x)
            ny = num_elements(y)

            # Copy the input `ny` times to efficiently compute many derivatives.
            xis = tf.identity_n([x[:, i : i + 1]] * ny)
            t.watch(xis)

            # Tile inputs for batched computation.
            x = B.tile(x, ny, 1)
            y = B.reshape(B.tile(y, 1, nx), ny * nx, -1)

            # Insert tracked dimension, which is different for every tile.
            xi = B.concat(*xis, axis=0)
            x = B.concat(x[:, :i], xi, x[:, i + 1 :], axis=1)

            # Perform the derivative computation.
            out = B.dense(k_elwise(x, y))
            grads = t.gradient(out, xis, unconnected_gradients="zero")
            return B.concat(*grads, axis=1)

    return _dkx


def dkx_elwise(k_elwise, i):
    """Construct the element-wise derivative of a kernel with respect to
    its first argument.

    Args:
        k_elwise (function): Function that performs element-wise computation
            of the kernel.
        i (int): Dimension with respect to which to compute the derivative.

    Returns:
        function: Element-wise derivative of the kernel with respect to its
            first argument.
    """

    @uprank
    def _dkx_elwise(x, y):
        import tensorflow as tf

        with tf.GradientTape() as t:
            xi = x[:, i : i + 1]
            t.watch(xi)
            x = B.concat(x[:, :i], xi, x[:, i + 1 :], axis=1)
            out = B.dense(k_elwise(x, y))
            return t.gradient(out, xi, unconnected_gradients="zero")

    return _dkx_elwise


def dky(k_elwise, i):
    """Construct the derivative of a kernel with respect to its second
    argument.

    Args:
        k_elwise (function): Function that performs element-wise computation
            of the kernel.
        i (int): Dimension with respect to which to compute the derivative.

    Returns:
        function: Derivative of the kernel with respect to its second argument.
    """

    @uprank
    def _dky(x, y):
        import tensorflow as tf

        with tf.GradientTape() as t:
            # Get the numbers of inputs.
            nx = num_elements(x)
            ny = num_elements(y)

            # Copy the input `nx` times to efficiently compute many derivatives.
            yis = tf.identity_n([y[:, i : i + 1]] * nx)
            t.watch(yis)

            # Tile inputs for batched computation.
            x = B.reshape(B.tile(x, 1, ny), nx * ny, -1)
            y = B.tile(y, nx, 1)

            # Insert tracked dimension, which is different for every tile.
            yi = B.concat(*yis, axis=0)
            y = B.concat(y[:, :i], yi, y[:, i + 1 :], axis=1)

            # Perform the derivative computation.
            out = B.dense(k_elwise(x, y))
            grads = t.gradient(out, yis, unconnected_gradients="zero")
            return B.transpose(B.concat(*grads, axis=1))

    return _dky


def dky_elwise(k_elwise, i):
    """Construct the element-wise derivative of a kernel with respect to
    its second argument.

    Args:
        k_elwise (function): Function that performs element-wise computation
            of the kernel.
        i (int): Dimension with respect to which to compute the derivative.

    Returns:
        function: Element-wise derivative of the kernel with respect to its
            second argument.
    """

    @uprank
    def _dky_elwise(x, y):
        import tensorflow as tf

        with tf.GradientTape() as t:
            yi = y[:, i : i + 1]
            t.watch(yi)
            y = B.concat(y[:, :i], yi, y[:, i + 1 :], axis=1)
            out = B.dense(k_elwise(x, y))
            return t.gradient(out, yi, unconnected_gradients="zero")

    return _dky_elwise


def perturb(x):
    """Slightly perturb a tensor.

    Args:
        x (tensor): Tensor to perturb.

    Returns:
        tensor: `x`, but perturbed.
    """
    dtype = convert(B.dtype(x), B.NPDType)
    if dtype == np.float64:
        return 1e-20 + x * (1 + 1e-14)
    elif dtype == np.float32:
        return 1e-20 + x * (1 + 1e-7)
    else:
        raise ValueError(f"Cannot perturb a tensor of data type {B.dtype(x)}.")


class DerivativeKernel(Kernel, DerivativeFunction):
    """Derivative of kernel."""

    _dispatch = Dispatcher(in_class=Self)

    @property
    def _stationary(self):
        # NOTE: In the one-dimensional case, if derivatives with respect to both
        # arguments are taken, then the result is in fact stationary.
        return False

    @_dispatch(Self)
    def __eq__(self, other):
        identical_derivs = identical(expand(self.derivs), expand(other.derivs))
        return self[0] == other[0] and identical_derivs


@_dispatch(DerivativeKernel, B.Numeric, B.Numeric)
def pairwise(k, x, y):
    i, j = expand(k.derivs)
    k = k[0]

    # Prevent that `x` equals `y` to stabilise nested gradients.
    y = perturb(y)

    if i is not None and j is not None:
        # Derivative with respect to both `x` and `y`.
        return Dense(dky(dkx_elwise(k.elwise, i), j)(x, y))

    elif i is not None and j is None:
        # Derivative with respect to `x`.
        return Dense(dkx(k.elwise, i)(x, y))

    elif i is None and j is not None:
        # Derivative with respect to `y`.
        return Dense(dky(k.elwise, j)(x, y))

    else:
        raise RuntimeError("No derivative specified.")


@_dispatch(DerivativeKernel, B.Numeric, B.Numeric)
def elwise(k, x, y):
    i, j = expand(k.derivs)
    k = k[0]

    # Prevent that `x` equals `y` to stabilise nested gradients.
    y = perturb(y)

    if i is not None and j is not None:
        # Derivative with respect to both `x` and `y`.
        return dky_elwise(dkx_elwise(k.elwise, i), j)(x, y)

    elif i is not None and j is None:
        # Derivative with respect to `x`.
        return dkx_elwise(k.elwise, i)(x, y)

    elif i is None and j is not None:
        # Derivative with respect to `y`.
        return dky_elwise(k.elwise, j)(x, y)

    else:
        raise RuntimeError("No derivative specified.")
