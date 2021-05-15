import lab as B
from algebra import DerivativeFunction

from . import _dispatch
from ..mean import Mean
from ..util import uprank

__all__ = ["DerivativeMean"]


class DerivativeMean(Mean, DerivativeFunction):
    """Derivative of mean."""

    @_dispatch
    @uprank
    def __call__(self, x: B.Numeric):
        import tensorflow as tf

        i = self.derivs[0]
        with tf.GradientTape() as t:
            xi = x[:, i : i + 1]
            t.watch(xi)
            x = B.concat(x[:, :i], xi, x[:, i + 1 :], axis=1)
            out = B.dense(self[0](x))
            return t.gradient(out, xi, unconnected_gradients="zero")
