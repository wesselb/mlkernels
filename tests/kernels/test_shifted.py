import lab as B
import numpy as np

from mlkernels import EQ, DecayingKernel, Linear, ShiftedKernel

from ..util import approx, standard_kernel_tests


def test_shifted():
    k = ShiftedKernel(2 * EQ(), 5)

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert Linear().shift(2) == Linear().shift(2)
    assert Linear().shift(2) != Linear().shift(3)
    assert Linear().shift(2) != DecayingKernel(1, 1).shift(2)

    # Standard tests:
    standard_kernel_tests(
        k,
        f1=lambda *xs: Linear().shift(5)(*xs),
        f2=lambda *xs: Linear()(*(x - 5 for x in xs)),
    )

    k = (2 * EQ()).shift(5, 6)

    # Verify that the kernel has the right properties.
    assert not k.stationary

    # Check passing in a list.
    k = Linear().shift(np.array([1, 2]))
    k(B.randn(10, 2))
