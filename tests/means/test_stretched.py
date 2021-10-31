import lab as B

from mlkernels import num_elements
from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, m_square, m_cube


def test_stretched(x):
    # Test correctness.
    approx(m_square.stretch(5)(x), m_square(x / 5))
    assert B.shape(m_square.stretch(5)(x)) == (num_elements(x), 1)

    # Test equality.
    assert m_square.stretch(5) == m_square.stretch(5)
    assert m_square.stretch(5) != m_square.stretch(6)
