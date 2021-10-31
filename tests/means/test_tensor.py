import lab as B

from mlkernels import TensorProductMean, num_elements
from ..util import approx

# noinspection PyUnresolvedReferences
from ..util_mean import x, f_square


def test_tensor(x):
    m = TensorProductMean(f_square)

    # Test correctness.
    approx(m(x), f_square(x))
    assert B.shape(m(x)) == (num_elements(x), 1)

    # Test properties.
    assert str(m) == "f_square"
