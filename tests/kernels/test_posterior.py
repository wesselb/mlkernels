import lab as B

from mlkernels import PosteriorKernel, EQ
from ..util import standard_kernel_tests, approx


def test_posterior():
    z = B.randn(3, 2)
    k = PosteriorKernel(EQ(), EQ(), EQ(), z, EQ()(z))

    # Check that the kernel computes correctly.
    approx(k(z), B.zeros(3, 3), atol=1e-11)

    # Verify that the kernel has the right properties.
    assert not k.stationary
    assert str(k) == "PosteriorKernel()"

    # Standard tests:
    standard_kernel_tests(
        k,
        shapes=[
            ((10, 2), (5, 2)),
            # Add in batch shapes.
            ((3, 10, 2), (3, 5, 2)),
            ((3, 10, 2), (5, 2)),
            ((10, 2), (3, 5, 2)),
        ],
    )
