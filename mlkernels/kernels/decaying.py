import lab as B
from algebra.util import identical
from matrix import Dense

from . import _dispatch
from .. import Kernel

__all__ = ["DecayingKernel"]


class DecayingKernel(Kernel):
    """Decaying kernel.

    This kernel is a multi-dimensional version of equation (6) in
         https://arxiv.org/pdf/1406.3896.pdf

    Args:
        alpha (tensor): Shape of the gamma distribution governing the distribution of
            decaying exponentials.
        beta (tensor): Rate of the gamma distribution governing the distribution of
            decaying exponentials.
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def _compute_beta_raised(self):
        return B.power(self.beta * self.beta, self.alpha / 2)

    def render(self, formatter):
        return f"DecayingKernel({formatter(self.alpha)}, {formatter(self.beta)})"

    @_dispatch
    def __eq__(self, other: "DecayingKernel"):
        return identical(self.alpha, other.alpha) and identical(self.beta, other.beta)


@_dispatch
def pairwise(k: DecayingKernel, x: B.Numeric, y: B.Numeric):
    pw_sums_raised = B.power(B.pw_sums2(B.add(x, k.beta), y), k.alpha / 2)
    return Dense(B.divide(k._compute_beta_raised(), pw_sums_raised))


@_dispatch
def elwise(k: DecayingKernel, x: B.Numeric, y: B.Numeric):
    return B.divide(
        k._compute_beta_raised(),
        B.power(B.ew_sums2(B.add(x, k.beta), y), k.alpha / 2),
    )
