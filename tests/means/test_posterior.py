import lab as B
import pytest

from mlkernels import PosteriorMean, EQ
from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, m_square


def test_posterior(x):
    z = B.randn(3, B.shape(B.uprank(x), 1))
    y = B.randn(3, 1)
    m = PosteriorMean(m_square, m_square, EQ(), z, EQ()(z), y)

    # Check correctness.
    approx(m(z), y, rtol=1e-6)  # This is numerically a bit iffy.
    with pytest.raises(AssertionError):
        approx(m(x), y)

    # Check properties.
    assert str(m) == "PosteriorMean()"
