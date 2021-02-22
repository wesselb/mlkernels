import lab as B
from algebra import Function, get_algebra

from . import _dispatch

__all__ = ["Kernel", "pairwise", "elwise", "periodicise"]


class Kernel(Function):
    """Kernel function.

    Kernels can be added and multiplied.
    """

    def __call__(self, *args, **kw_args):
        return self.pairwise(*args, **kw_args)

    def pairwise(self, *args, **kw_args):
        """Construct the kernel matrix between all `x` and `y`.

        This method does *not* preserve matrix structure and simply returns a tensor.

        Args:
            x (input): First argument.
            y (input, optional): Second argument. Defaults to first argument.

        Returns:
            matrix: Kernel matrix.
        """
        return B.dense(pairwise(self, *args, **kw_args))

    def elwise(self, *args, **kw_args):
        """Construct the kernel vector `x` and `y` element-wise.

        This method does *not* preserve matrix structure and simply returns a tensor.

        Args:
            x (input): First argument.
            y (input, optional): Second argument. Defaults to first argument.

        Returns:
            tensor: Kernel vector as a rank 2 column vector.
        """
        return B.dense(elwise(self, *args, **kw_args))

    def periodic(self, period=1):
        """Map to a periodic space.

        Args:
            period (tensor, optional): Period. Defaults to `1`.

        Returns:
            :class:`.kernel.Kernel`: Periodic version of the kernel.
        """
        return periodicise(self, period)

    @property
    def stationary(self):
        """Stationarity of the kernel."""
        try:
            return self._stationary_cache
        except AttributeError:
            self._stationary_cache = self._stationary
            return self._stationary_cache

    @property
    def _stationary(self):
        return False


# Register the algebra.
@get_algebra.extend(Kernel)
def get_algebra(a):
    return Kernel


@_dispatch(Kernel, object, object)
def pairwise(k, x, y):
    """Construct the kernel matrix between all `x` and `y`.

    This method does preserve matrix structure and *may* return a structured matrix.

    Args:
        k (:class:`.Kernel`): Kernel.
        x (input): First argument.
        y (input, optional): Second argument. Defaults to first argument.

    Returns:
        matrix or :class:`matrix.AbstractMatrix`: Kernel matrix.
    """
    raise RuntimeError(
        f'For kernel "{k}", could not resolve arguments "{x}" and "{y}".'
    )


@_dispatch(Kernel, object)
def pairwise(k, x):
    return pairwise(k, x, x)


@_dispatch(Kernel)
def pairwise(k):
    def call(*args, **kw_args):
        return pairwise(k, *args, **kw_args)

    return call


@_dispatch(Kernel, object, object)
def elwise(k, x, y):
    """Construct the kernel vector `x` and `y` element-wise.

    This method does preserve matrix structure and *may* return a structured matrix.

    Args:
        kernel (:class:`.Kernel`): Kernel.
        x (input): First argument.
        y (input, optional): Second argument. Defaults to first argument.

    Returns:
        matrix or :class:`matrix.AbstractMatrix`: Kernel vector as a rank 2 column
            vector.
    """
    # TODO: Throw warning.
    return B.expand_dims(B.diag(pairwise(k, x, y)), axis=1)


@_dispatch(Kernel, object)
def elwise(k, x):
    return elwise(k, x, x)


@_dispatch(Kernel)
def elwise(k):
    def call(*args, **kw_args):
        return elwise(k, *args, **kw_args)

    return call


@_dispatch(Kernel, object)
def periodicise(k, period):  # pragma: no cover
    """Map a kernel to a periodic space.

    Args:
        k (:class:`.kernel.Kernel`): Kernel to make periodic.
        period (tensor, optional): Period. Defaults to `1`.

    Returns:
        :class:`.kernel.Kernel`: Periodic version of the kernel.
    """
    # This method will be overwritten. Hence, this should never run.
    raise NotImplementedError(f"Cannot make {k} periodic with period {period}.")
