import lab as B
import pytest

from mlkernels import (
    Kernel,
    pairwise,
    elwise,
    EQ,
    RQ,
    Matern12,
    Matern32,
    Matern52,
    Delta,
    Linear,
)
from .util import approx


def test_corner_cases():
    with pytest.raises(RuntimeError):
        # Cannot resolve the arguments for this kernel, because it has no
        # implementation.
        Kernel()(1.0)


def test_construction():
    x1 = B.randn(10)
    x2 = B.randn(10)

    k = EQ()

    k(x1)
    k(x1, x2)

    pairwise(k, x1)
    pairwise(k, x1, x2)

    k.elwise(x1)
    k.elwise(x1, x2)

    elwise(k, x1)
    elwise(k, x1, x2)


def test_basic_arithmetic():
    k1 = EQ()
    k2 = RQ(1e-1)
    k3 = Matern12()
    k4 = Matern32()
    k5 = Matern52()
    k6 = Delta()
    k7 = Linear()
    xs1 = B.randn(10, 2), B.randn(20, 2)
    xs2 = B.randn(), B.randn()

    approx(k6(xs1[0]), k6(xs1[0], xs1[0]))
    approx((k1 * k2)(*xs1), k1(*xs1) * k2(*xs1))
    approx((k1 * k2)(*xs2), k1(*xs2) * k2(*xs2))
    approx((k3 + k4)(*xs1), k3(*xs1) + k4(*xs1))
    approx((k3 + k4)(*xs2), k3(*xs2) + k4(*xs2))
    approx((5.0 * k5)(*xs1), 5.0 * k5(*xs1))
    approx((5.0 * k5)(*xs2), 5.0 * k5(*xs2))
    approx((5.0 + k7)(*xs1), 5.0 + k7(*xs1))
    approx((5.0 + k7)(*xs2), 5.0 + k7(*xs2))
    approx(k1.stretch(2.0)(*xs1), k1(xs1[0] / 2.0, xs1[1] / 2.0))
    approx(k1.stretch(2.0)(*xs2), k1(xs2[0] / 2.0, xs2[1] / 2.0))
    approx(k1.periodic(1.0)(*xs1), k1.periodic(1.0)(xs1[0], xs1[1] + 5.0))
    approx(k1.periodic(1.0)(*xs2), k1.periodic(1.0)(xs2[0], xs2[1] + 5.0))
