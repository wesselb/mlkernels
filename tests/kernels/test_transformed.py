import lab as B

from mlkernels import Linear, EQ, Matern12
from ..util import approx, standard_kernel_tests


def test_transform():
    k = Linear().transform(lambda x: x - 5)

    # Verify that the kernel has the right properties.
    assert not k.stationary

    def f1(x):
        return x

    def f2(x):
        return x**2

    # Test equality.
    assert EQ().transform(f1) == EQ().transform(f1)
    assert EQ().transform(f1) != EQ().transform(f2)
    assert EQ().transform(f1) != Matern12().transform(f1)

    # Standard tests:
    standard_kernel_tests(k)

    # Test computation of the kernel.
    k = Linear()
    x1, x2 = B.randn(10, 2), B.randn(10, 2)
    k2 = k.transform(lambda x: x**2)
    k3 = k.transform(lambda x: x**2, lambda x: x - 5)
    approx(k(x1**2, x2**2), k2(x1, x2))
    approx(k(x1**2, x2 - 5), k3(x1, x2))
