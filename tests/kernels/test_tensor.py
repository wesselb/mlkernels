import lab as B

from mlkernels import TensorProductKernel, EQ
from ..util import approx, standard_kernel_tests


def test_tensor_product():
    k = TensorProductKernel(lambda x: B.sum(x**2, axis=-1))

    # Verify that the kernel has the right properties.
    assert not k.stationary
    assert str(k) == "<lambda>"

    # Test equality.
    assert k == k
    assert k != TensorProductKernel(lambda x: x)
    assert k != EQ()

    # Standard tests:
    standard_kernel_tests(k, batch_shapes=False)
    # The sum over `axis = -1` will not work in batch mode. We test one implementation
    # of `TensorProductKernel` which does work in batch mode.
    standard_kernel_tests(TensorProductKernel(lambda x: B.sum(x, axis=-1)[..., None]))

    # Test computation of the kernel.
    k = TensorProductKernel(lambda x: x)
    x1 = B.linspace(0, 1, 100)[:, None]
    x2 = B.linspace(0, 1, 50)[:, None]
    approx(k(x1), x1 * x1.T)
    approx(k(x1, x2), x1 * x2.T)

    k = TensorProductKernel(lambda x: x**2)
    approx(k(x1), x1**2 * (x1**2).T)
    approx(k(x1, x2), (x1**2) * (x2**2).T)
