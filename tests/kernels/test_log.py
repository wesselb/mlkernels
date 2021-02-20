from mlkernels import EQ, LogKernel

from ..util import standard_kernel_tests


def test_log_kernel():
    k = LogKernel()

    # Verify that the kernel has the right properties.
    assert k.stationary
    assert str(k) == "LogKernel()"

    # Test equality.
    assert LogKernel() == LogKernel()
    assert LogKernel() != EQ()

    # Standard tests:
    standard_kernel_tests(k)
