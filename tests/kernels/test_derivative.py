import lab as B
import numpy as np
import pytest
import tensorflow as tf

from mlkernels import EQ, Matern12, Linear, perturb, pairwise, elwise
from ..util import approx, standard_kernel_tests


def test_derivative():
    k = EQ().diff(0)

    # Check that the kernel has the right properties.
    assert not k.stationary

    # Test equality.
    assert EQ().diff(0) == EQ().diff(0)
    assert EQ().diff(0) != EQ().diff(1)
    assert Matern12().diff(0) != EQ().diff(0)

    # Standard tests:
    for k in [EQ().diff(0), EQ().diff(None, 0), EQ().diff(0, None)]:
        standard_kernel_tests(k, dtype=tf.float64, batch_shapes=False)
    # Batch mode inputs is not yet implemented.
    with pytest.raises(NotImplementedError):
        k(np.random.randn(3, 5, 1))
    with pytest.raises(NotImplementedError):
        k.elwise(np.random.randn(3, 5, 1))

    # Check that a derivative must be specified.
    with pytest.raises(RuntimeError):
        pairwise(EQ().diff(None, None), np.array([1.0]))
    with pytest.raises(RuntimeError):
        elwise(EQ().diff(None, None), np.array([1.0]))


def test_derivative_eq():
    # Test derivative of kernel `EQ()`.
    k = EQ()
    x1 = B.randn(tf.float64, 10, 1)
    x2 = B.randn(tf.float64, 5, 1)

    # Test derivative with respect to first input.
    approx(k.diff(0, None)(x1, x2), -k(x1, x2) * (x1 - B.transpose(x2)))
    approx(k.diff(0, None)(x1), -k(x1) * (x1 - B.transpose(x1)))

    # Test derivative with respect to second input.
    approx(k.diff(None, 0)(x1, x2), -k(x1, x2) * (B.transpose(x2) - x1))
    approx(k.diff(None, 0)(x1), -k(x1) * (B.transpose(x1) - x1))

    # Test derivative with respect to both inputs.
    ref = k(x1, x2) * (1 - (x1 - B.transpose(x2)) ** 2)
    approx(k.diff(0, 0)(x1, x2), ref)
    approx(k.diff(0)(x1, x2), ref)
    ref = k(x1) * (1 - (x1 - B.transpose(x1)) ** 2)
    approx(k.diff(0, 0)(x1), ref)
    approx(k.diff(0)(x1), ref)


def test_derivative_linear():
    # Test derivative of kernel `Linear()`.
    k = Linear()
    x1 = B.randn(tf.float64, 10, 1)
    x2 = B.randn(tf.float64, 5, 1)

    # Test derivative with respect to first input.
    approx(k.diff(0, None)(x1, x2), B.ones(tf.float64, 10, 5) * B.transpose(x2))
    approx(k.diff(0, None)(x1), B.ones(tf.float64, 10, 10) * B.transpose(x1))

    # Test derivative with respect to second input.
    approx(k.diff(None, 0)(x1, x2), B.ones(tf.float64, 10, 5) * x1)
    approx(k.diff(None, 0)(x1), B.ones(tf.float64, 10, 10) * x1)

    # Test derivative with respect to both inputs.
    ref = B.ones(tf.float64, 10, 5)
    approx(k.diff(0, 0)(x1, x2), ref)
    approx(k.diff(0)(x1, x2), ref)
    ref = B.ones(tf.float64, 10, 10)
    approx(k.diff(0, 0)(x1), ref)
    approx(k.diff(0)(x1), ref)


@pytest.mark.parametrize(
    "x,result",
    [
        (np.float64([1]), np.float64([1e-20 + 1 + 1e-14])),
        (np.float32([1]), np.float32([1e-20 + 1 + 1e-7])),
    ],
)
def test_perturb(x, result):
    assert perturb(x) == result  # Test NumPy.
    assert perturb(tf.constant(x)).numpy() == result  # Test TF.


def test_perturb_type_check():
    with pytest.raises(ValueError):
        perturb(0)


@pytest.mark.parametrize("dtype", [tf.float32, tf.float64])
def test_nested_derivatives(dtype):
    x = B.randn(dtype, 10, 2)

    res = EQ().diff(0, 0).diff(0, 0)(x)
    assert ~B.isnan(res[0, 0])

    res = EQ().diff(1, 1).diff(1, 1)(x)
    assert ~B.isnan(res[0, 0])
