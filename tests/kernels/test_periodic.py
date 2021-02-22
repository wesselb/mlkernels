import lab as B
import numpy as np

from mlkernels import EQ, Matern12, ZeroKernel
from ..util import standard_kernel_tests


def test_periodic():
    k = EQ().stretch(2).periodic(3)

    # Verify that the kernel has the right properties.
    assert str(k) == "(EQ() > 2) per 3"
    assert k.stationary

    # Test equality.
    assert EQ().periodic(2) == EQ().periodic(2)
    assert EQ().periodic(2) != EQ().periodic(3)
    assert Matern12().periodic(2) != EQ().periodic(2)

    # Standard tests:
    standard_kernel_tests(
        k,
        f1=lambda *xs: EQ().periodic(1.0)(*xs),
        f2=lambda *xs: EQ().periodic(1.0)(*(x + 5.0 for x in xs)),
    )

    k = 5 * k.stretch(5)

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Check passing in a list.
    k = EQ().periodic(np.array([1, 2]))
    k(B.randn(10, 2))

    # Check periodication of a zero.
    k = ZeroKernel()
    assert k.periodic(3) is k
