# noinspection PyUnresolvedReferences
import lab.autograd as B
from algebra import Element
from autograd.numpy.numpy_boxes import ArrayBox
from matrix import AbstractMatrix

# noinspection PyUnresolvedReferences
from . import *

_dispatch = Dispatcher()

# Save original methods.
__mul__array_box = ArrayBox.__mul__
__add__array_box = ArrayBox.__add__


@_dispatch(ArrayBox, object)
def __mul__(self, other):
    return __mul__array_box(self, other)


@_dispatch(ArrayBox, {Element, AbstractMatrix})
def __mul__(self, other):
    return other.__rmul__(self)


@_dispatch(ArrayBox, object)
def __add__(self, other):
    return __add__array_box(self, other)


@_dispatch(ArrayBox, {Element, AbstractMatrix})
def __add__(self, other):
    return other.__radd__(self)


# Assign dispatchable methods.
ArrayBox.__mul__ = __mul__
ArrayBox.__add__ = __add__
