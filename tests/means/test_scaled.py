import lab as B

from mlkernels import num_elements
from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, m_square, m_cube


def test_scaled(x):
    # Test correctness.
    approx((2 * m_square)(x), 2 * m_square(x))
    assert B.shape((2 * m_square)(x)) == (num_elements(x), 1)

    # Test equality.
    assert 2 * m_square == m_square * 2
    assert 2 * m_square == m_square + m_square
    assert 2 * m_square != m_square * 3
    assert 2 * m_square != m_square + m_cube
