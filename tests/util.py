import lab as B
import numpy as np
from numpy.testing import assert_allclose

from mlkernels import Kernel, elwise, pairwise

__all__ = ["approx", "standard_kernel_tests"]


def approx(x, y, rtol=1e-12, atol=1e-12):
    assert_allclose(B.dense(x), B.dense(y), rtol=rtol, atol=atol)


def standard_kernel_tests(
    k,
    shapes=None,
    batch_shapes=True,
    dtype=np.float64,
    f1=None,
    f2=None,
):
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
        if batch_shapes:
            shapes += [
                ((3, 10, 2), (3, 5, 2)),
                ((10, 2), (3, 5, 2)),
                ((3, 10, 2), (5, 2)),
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

        # Check argument duplication.
        approx(pairwise(k, x1, x1), pairwise(k, x1))

        # Check that the kernel computes consistently.
        approx(pairwise(k, x1, x2), B.transpose(pairwise(reversed(k), x2, x1)))
        approx(pairwise(k, x1), B.transpose(pairwise(reversed(k), x1)))

        # Check `elwise`.

        x2 = B.randn(dtype, *shape1)

        # Check argument duplication.
        approx(elwise(k, x1, x1), elwise(k, x1))

        # Note that the element-wise computation is more accurate: we allow a
        # discrepancy a bit larger than the square root of the machine epsilon.
        errors = {"rtol": 2e-7, "atol": 1e-12}

        # Check the two-argument calls for `elwise`.
        approx(elwise(k, x1, x2), B.diag(k(x1, x2))[..., None], **errors)
        # Check against fallback brute force computation.
        approx(
            elwise(k, x1, x2),
            elwise.invoke(Kernel, object, object)(k, x1, x2),
            **errors,
        )

        # Check the one-argument calls for `elwise`.
        approx(elwise(k, x1)[..., 0], B.diag(k(x1)), **errors)
        # This actually should not trigger the fallback, so we do not use `errors`.
        approx(elwise(k, x1), elwise.invoke(Kernel, object)(k, x1))
        approx(
            elwise(k, x1),
            elwise.invoke(Kernel, object, object)(k, x1, x1),
            **errors,
        )

        # Check shapes.

        x1 = B.randn(dtype, *shape1)
        x2 = B.randn(dtype, *shape2)
        x3 = B.randn(dtype, *shape1)

        # Check matrix shape.
        pairwise_shape = (B.shape_matrix(x1, 0), B.shape_matrix(x2, 0))
        assert B.shape_matrix(pairwise(k, x1, x2)) == pairwise_shape
        assert B.shape_matrix(elwise(k, x1, x3)) == (B.shape_matrix(x1, 0), 1)

        # Check batch shape.
        pairwise_shape = B.shape_batch_broadcast(x1, x2)
        assert B.shape_batch(pairwise(k, x1, x2)) == pairwise_shape
        assert B.shape_batch(elwise(k, x1, x3)) == B.shape_batch(x1)
