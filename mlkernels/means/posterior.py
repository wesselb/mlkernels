import lab as B
from matrix import AbstractMatrix
from plum import convert

from . import _dispatch
from ..mean import Mean

__all__ = ["PosteriorMean"]


class PosteriorMean(Mean):
    """Posterior mean.

    Args:
        m_i (:class:`.mean.Mean`): Mean of process corresponding to the input.
        m_z (:class:`.mean.Mean`): Mean of process corresponding to the data.
        k_zi (:class:`.kernel.Kernel`): Kernel between processes corresponding to the
            data and the input respectively.
        z (input): Locations of data.
        K_z (matrix): Kernel matrix of data.
        y (vector): Observations to condition on.
    """

    def __init__(self, m_i, m_z, k_zi, z, K_z, y):
        self.m_i = m_i
        self.m_z = m_z
        self.k_zi = k_zi
        self.z = z
        self.K_z = convert(K_z, AbstractMatrix)
        self.y = B.uprank(y)

    @_dispatch
    def __call__(self, x):
        diff = B.subtract(self.y, self.m_z(self.z))
        return B.add(self.m_i(x), B.iqf(self.K_z, self.k_zi(self.z, x), diff))
