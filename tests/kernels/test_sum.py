import lab as B

from mlkernels import EQ, RQ, Linear
from ..util import standard_kernel_tests, approx


def test_sum():
    k1 = EQ().stretch(2)
    k2 = 3 * RQ(1e-2).stretch(5)
    k = k1 + k2

    # Verify that the kernel has the right properties.
    assert k.stationary

    # Test equality.
    assert EQ() + Linear() == EQ() + Linear()
    assert EQ() + Linear() == Linear() + EQ()
    assert EQ() + Linear() != EQ() + RQ(1e-1)
    assert EQ() + Linear() != RQ(1e-1) + Linear()

    # Standard tests:
    standard_kernel_tests(
        k,
        f1=lambda *xs: (EQ() + RQ(1e-1))(*xs),
        f2=lambda *xs: EQ()(*xs) + RQ(1e-1)(*xs),
    )
