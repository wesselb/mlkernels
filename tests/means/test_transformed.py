import lab as B

from mlkernels import num_elements
from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, m_square, m_cube


def test_stretched(x):
    # Test correctness.
    approx(m_square.transform(lambda y: y**2)(x), m_square(x**2))
    assert B.shape(m_square.transform(lambda y: y**2)(x)) == (num_elements(x), 1)

    def f(x):
        return x**2

    # Test equality.
    assert m_square.transform(f) == m_square.transform(f)
    assert m_square.transform(f) != m_square.transform(lambda x: x**3)
