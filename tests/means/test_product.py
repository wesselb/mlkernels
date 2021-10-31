import lab as B

from mlkernels import num_elements
from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, m_square, m_cube


def test_product(x):
    # Test correctness.
    approx((m_square * m_cube)(x), m_square(x) * m_cube(x))
    assert B.shape((m_square * m_cube)(x)) == (num_elements(x), 1)

    # Test equality.
    assert m_square * m_cube == m_square * m_cube
    assert m_square * m_cube == m_cube * m_square
    assert m_square * m_cube != m_cube * m_cube
    assert m_square * m_cube != m_square * m_square
