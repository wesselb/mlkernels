from numbers import Number
import lab as B
import pytest
from mlkernels import Mean, TensorProductMean
from plum import Dispatcher

__all__ = ["x", "f_square", "m_square", "f_cube", "m_cube"]

_dispatch = Dispatcher()


@pytest.fixture(params=[(), (10,), (10, 1), (10, 2)])
def x(request):
    yield B.randn(*request.param)


def f_square(x):
    return B.sum(B.uprank(x) ** 2, axis=1)[:, None]


m_square = TensorProductMean(f_square)


@_dispatch
def f_cube(x):
    return B.sum(B.uprank(x) ** 3, axis=1)[:, None]


m_cube = TensorProductMean(f_cube)
