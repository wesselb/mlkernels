import pytest
from mlkernels import Mean


def test_corner_cases():
    with pytest.raises(RuntimeError):
        # Cannot resolve the arguments for this kernel, because it has no
        # implementation.
        Mean()(1.0)
