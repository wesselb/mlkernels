import numpy as np
import lab as B
from mlkernels import Kernel, elwise, pairwise
from matrix import AbstractMatrix
from numpy.testing import assert_allclose

__all__ = ["approx", "standard_kernel_tests"]


def approx(x, y, rtol=1e-12, atol=1e-12):
    assert_allclose(B.dense(x), B.dense(y), rtol=rtol, atol=atol)


def standard_kernel_tests(k, shapes=None, dtype=np.float64, f1=None, f2=None):
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

        # Check that a different way of computing the same thing gives the same thing.
        if f1:
            approx(f1(x1, x2), f2(x1, x2))
            approx(f1(x1), f2(x1))

        # Check `pairwise`.

        # Check that types of the output are right. Strictly, the `pairwise` calls
        # do not have to return a structured matrix, but here we require that they do.
        assert isinstance(pairwise(k, x1, x2), AbstractMatrix)
        assert isinstance(pairwise(k, x1), AbstractMatrix)
        assert isinstance(k.pairwise(x1, x2), B.Numeric)
        assert isinstance(k.pairwise(x1), B.Numeric)

        # Check argument duplication.
        approx(pairwise(k, x1, x1), pairwise(k, x1))

        # Check that the kernel computes consistently.
        approx(pairwise(k, x1, x2), B.transpose(pairwise(reversed(k), x2, x1)))
        approx(pairwise(k, x1), B.transpose(pairwise(reversed(k), x1)))

        # Check `elwise`.

        x2 = B.randn(dtype, *shape1)

        # Check that types of the output are right.
        assert isinstance(elwise(k, x1, x2), (B.Numeric, AbstractMatrix))
        assert isinstance(elwise(k, x1), (B.Numeric, AbstractMatrix))
        assert isinstance(k.elwise(x1, x2), B.Numeric)
        assert isinstance(k.elwise(x1), B.Numeric)

        # Check argument duplication.
        approx(elwise(k, x1, x1), elwise(k, x1))

        # Note that the element-wise computation is more accurate: we allow a
        # discrepancy a bit larger than the square root of the machine epsilon.
        errors = {"rtol": 2e-7, "atol": 1e-12}

        # Check the two-argument calls for `elwise`.
        approx(elwise(k, x1, x2), B.diag(k(x1, x2))[:, None], **errors)
        # Check against fallback brute force computation.
        approx(
            elwise(k, x1, x2),
            elwise.invoke(Kernel, object, object)(k, x1, x2),
            **errors
        )

        # Check the one-argument calls for `elwise`.
        approx(elwise(k, x1)[:, 0], B.diag(k(x1)), **errors)
        # This actually should not trigger the fallback, so we do not use `errors`.
        approx(elwise(k, x1), elwise.invoke(Kernel, object)(k, x1))
        approx(
            elwise(k, x1), elwise.invoke(Kernel, object, object)(k, x1, x1), **errors
        )
