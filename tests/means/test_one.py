import lab as B

from mlkernels import OneMean, num_elements
from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, m_square


def test_one(x):
    m = OneMean()

    # Check that the mean computes correctly.
    approx(m(x), B.ones(num_elements(x), 1))

    # Verify that the mean has the right properties.
    assert str(m) == "1"

    # Test equality.
    assert OneMean() == OneMean()
    assert OneMean() != m_square
