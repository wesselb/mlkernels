from random import shuffle

import lab as B

from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, m_square


def test_selected(x):
    # Test correctness.
    inds = list(range(B.shape(B.uprank(x), 1)))
    shuffle(inds)
    approx(m_square.select(inds[:2])(x), m_square(B.uprank(x)[:, inds[:2]]))

    # Test equality.
    assert m_square.select([0]) == m_square.select([0])
    assert m_square.select([0]) != m_square.select([1])
