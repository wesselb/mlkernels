import lab as B
import numpy as np
import pytest

from mlkernels import EQ, Delta
from mlkernels.util import num_elements

from ..util import approx, standard_kernel_tests


def test_delta_properties():
    k = Delta()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "Delta()"

    # Check equality.
    assert Delta() == Delta()
    assert Delta() != Delta(epsilon=k.epsilon * 10)
    assert Delta() != EQ()


@pytest.mark.parametrize(
    "x1, x2",
    [
        (np.array(0), np.array(1)),
        (B.randn(10), B.randn(5)),
        (B.randn(10, 1), B.randn(5, 1)),
        (B.randn(10, 2), B.randn(5, 2)),
    ],
)
def test_delta_evaluations(x1, x2):
    k = Delta()
    n1 = num_elements(x1)
    n2 = num_elements(x2)

    # Test uniqueness checks.
    approx(k(x1), B.eye(n1))
    approx(k(x1, x2), B.zeros(n1, n2))

    # Standard tests:
    standard_kernel_tests(k)
