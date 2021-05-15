from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, m_square, m_cube


def test_stretched(x):
    # Test correctness.
    approx(m_square.transform(lambda y: y ** 2)(x), m_square(x ** 2))

    def f(x):
        return x ** 2

    # Test equality.
    assert m_square.transform(f) == m_square.transform(f)
    assert m_square.transform(f) != m_square.transform(lambda x: x ** 3)
