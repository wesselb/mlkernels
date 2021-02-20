import lab as B

from mlkernels import EQ, DecayingKernel, Linear

from ..util import approx, standard_kernel_tests


def test_reversal():
    x1 = B.randn(10, 2)
    x2 = B.randn(5, 2)
    x3 = B.randn()

    # Test with a stationary and non-stationary kernel.
    for k in [EQ(), Linear()]:
        approx(k(x1), reversed(k)(x1))
        approx(k(x3), reversed(k)(x3))
        approx(k(x1, x2), reversed(k)(x1, x2))
        approx(k(x1, x2), reversed(k)(x2, x1).T)

        # Test double reversal does the right thing.
        approx(k(x1), reversed(reversed(k))(x1))
        approx(k(x3), reversed(reversed(k))(x3))
        approx(k(x1, x2), reversed(reversed(k))(x1, x2))
        approx(k(x1, x2), reversed(reversed(k))(x2, x1).T)

    # Verify that the kernel has the right properties.
    k = reversed(EQ())
    assert k.stationary

    k = reversed(Linear())
    assert not k.stationary
    assert str(k) == "Reversed(Linear())"

    # Check equality.
    assert reversed(Linear()) == reversed(Linear())
    assert reversed(Linear()) != Linear()
    assert reversed(Linear()) != reversed(EQ())
    assert reversed(Linear()) != reversed(DecayingKernel(1, 1))

    # Standard tests:
    standard_kernel_tests(k)
