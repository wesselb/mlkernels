import lab as B
import numpy as np

from mlkernels import EQ, Matern12
from ..util import approx, standard_kernel_tests


def test_selected():
    k = (2 * EQ().stretch(5)).select(0)

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert EQ().select(0) == EQ().select(0)
    assert EQ().select(0) != EQ().select(1)
    assert EQ().select(0) != Matern12().select(0)

    # Standard tests:
    standard_kernel_tests(k)

    # Verify that the kernel has the right properties.
    k = (2 * EQ().stretch(5)).select([2, 3])
    assert k.stationary

    k = (2 * EQ().stretch(np.array([1, 2, 3]))).select([0, 2])
    assert k.stationary

    k = (2 * EQ().periodic(np.array([1, 2, 3]))).select([1, 2])
    assert k.stationary

    k = (2 * EQ().stretch(np.array([1, 2, 3]))).select([0, 2], [1, 2])
    assert not k.stationary

    k = (2 * EQ().periodic(np.array([1, 2, 3]))).select([0, 2], [1, 2])
    assert not k.stationary

    # Test computation of the kernel.
    k1 = EQ().select([1, 2])
    k2 = EQ()
    x = B.randn(10, 3)
    approx(k1(x), k2(x[:, [1, 2]]))
