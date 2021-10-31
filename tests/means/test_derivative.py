import lab as B
import tensorflow as tf

from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, m_square, m_cube


def test_derivative(x):
    # Check that the mean computes correctly.
    x = tf.constant(x)
    approx(m_square.diff(0)(x), 2 * B.uprank(x)[:, :1])
    approx(m_cube.diff(0)(x), 3 * B.uprank(x)[:, :1] ** 2)
