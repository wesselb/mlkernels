from mlkernels import EQ, DecayingKernel

from ..util import standard_kernel_tests


def test_decaying_kernel():
    k = DecayingKernel(3.0, 4.0)

    # Verify that the kernel has the right properties.
    assert not k.stationary
    assert str(k) == "DecayingKernel(3.0, 4.0)"

    # Test equality.
    assert DecayingKernel(3.0, 4.0) == DecayingKernel(3.0, 4.0)
    assert DecayingKernel(3.0, 4.0) != DecayingKernel(3.0, 5.0)
    assert DecayingKernel(3.0, 4.0) != DecayingKernel(4.0, 4.0)
    assert DecayingKernel(3.0, 4.0) != EQ()

    # Standard tests:
    standard_kernel_tests(k)
