import lab as B
import pytest

from mlkernels import Kernel, pairwise, elwise, EQ


def test_corner_cases():
    with pytest.raises(RuntimeError):
        # Cannot resolve the arguments for this kernel, because it has no
        # implementation.
        Kernel()(1.0)


def test_construction():
    x1 = B.randn(10)
    x2 = B.randn(10)

    k = EQ()

    k(x1)
    k(x1, x2)

    pairwise(k, x1)
    pairwise(k, x1, x2)
    pairwise(k)(x1)
    pairwise(k)(x1, x2)

    k.elwise(x1)
    k.elwise(x1, x2)

    elwise(k, x1)
    elwise(k, x1, x2)
    elwise(k)(x1)
    elwise(k)(x1, x2)
