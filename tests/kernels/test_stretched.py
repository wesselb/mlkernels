import lab as B
import numpy as np

from mlkernels import EQ, Matern12
from ..util import standard_kernel_tests


def test_stretched():
    k = EQ().stretch(2)

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert EQ().stretch(2) == EQ().stretch(2)
    assert EQ().stretch(2) != EQ().stretch(3)
    assert EQ().stretch(2) != Matern12().stretch(2)

    # Standard tests:
    standard_kernel_tests(
        k,
        f1=lambda *xs: EQ().stretch(2.0)(*xs),
        f2=lambda *xs: EQ()(*(x / 2.0 for x in xs)),
    )

    k = EQ().stretch(1, 2)

    # Verify that the kernel has the right properties.
    assert not k.stationary

    # Check passing in a list.
    k = EQ().stretch(np.array([1, 2]))
    k(B.randn(10, 2))
