import numpy as np
import lab as B
from mlkernels import Kernel, elwise
from numpy.testing import assert_allclose

__all__ = ["approx", "standard_kernel_tests"]


def approx(x, y, rtol=1e-12, atol=1e-12):
    assert_allclose(B.dense(x), B.dense(y), rtol=rtol, atol=atol)


def standard_kernel_tests(k, shapes=None, dtype=np.float64):
    if shapes is None:
        shapes = [
            ((10, 2), (5, 2)),
            ((10, 1), (5, 1)),
            ((10,), (5, 1)),
            ((10, 1), (5,)),
            ((10,), (5,)),
            ((10,), ()),
            ((), (5,)),
            ((), ()),
        ]

    # Check various shapes of arguments.
    for shape1, shape2 in shapes:
        x1 = B.randn(dtype, *shape1)
        x2 = B.randn(dtype, *shape2)

        # Check that the kernel computes consistently.
        approx(k(x1, x2), B.transpose(reversed(k)(x2, x1)))
        approx(k(x1), B.transpose(reversed(k)(x1)))

        x2 = B.randn(dtype, *shape1)

        # Check `elwise`. Note that the element-wise computation is more accurate,
        # which is why we allow a discrepancy a bit larger than the square root of
        # the machine epsilon.
        errors = {"rtol": 2e-7, "atol": 1e-12}

        approx(k.elwise(x1, x2), B.diag(k(x1, x2))[:, None], **errors)
        # Check against fallback brute force computation.
        approx(
            k.elwise(x1, x2), elwise.invoke(Kernel, object, object)(k, x1, x2), **errors
        )

        # Check the one-argument calls for `elwise`.
        approx(k.elwise(x1)[:, 0], B.diag(k(x1)), **errors)
        # This actually should not trigger the fallback.
        approx(k.elwise(x1), elwise.invoke(Kernel, object)(k, x1))
        approx(k.elwise(x1), elwise.invoke(Kernel, object, object)(k, x1, x1), **errors)
