import lab as B

from mlkernels import ZeroMean, num_elements
from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, m_square


def test_one(x):
    m = ZeroMean()

    # Check that the mean computes correctly.
    approx(m(x), B.zeros(num_elements(x), 1))

    # Verify that the mean has the right properties.
    assert str(m) == "0"

    # Test equality.
    assert ZeroMean() == ZeroMean()
    assert ZeroMean() != m_square
