from mlkernels import TensorProductMean

from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, f_square


def test_tensor(x):
    m = TensorProductMean(f_square)

    # Test correctness.
    approx(m(x), f_square(x))

    # Test properties.
    assert str(m) == "f_square"
