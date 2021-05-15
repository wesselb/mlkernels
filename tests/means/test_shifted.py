from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, m_square, m_cube


def test_shifted(x):
    # Test correctness.
    approx(m_square.shift(5)(x), m_square(x - 5))

    # Test equality.
    assert m_square.shift(5) == m_square.shift(5)
    assert m_square.shift(5) != m_square.shift(6)
