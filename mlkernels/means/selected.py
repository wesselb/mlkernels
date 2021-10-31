import lab as B
from algebra import SelectedFunction

from . import _dispatch
from ..mean import Mean
from ..util import uprank

__all__ = ["SelectedMean"]


class SelectedMean(Mean, SelectedFunction):
    """Mean with particular input dimensions selected."""

    @_dispatch
    @uprank
    def __call__(self, x):
        return self[0](B.take(x, self.dims[0], axis=-1))
