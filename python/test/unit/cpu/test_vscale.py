"""Tests for SVE vscale infrastructure in the CPU backend.

These tests verify that the compilation pipeline correctly propagates host CPU
features to the LLVM target machine and sets the ``vscale_range`` function
attribute when SVE is available.  This attribute is required for LLVM to accept
scalable vector types in subsequent VLA code generation.
"""

import os

import pytest
import torch

import triton
import triton.language as tl
from triton._C.libtriton import cpu


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_cpu():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "cpu"


def has_sve():
    """Return True if the host CPU supports SVE."""
    if not is_cpu():
        return False
    cpu_features = cpu.llvm.get_cpu_features()
    return "sve" in cpu_features


@pytest.mark.parametrize("vec_lib", ["libsleef", None])
def test_vscale_range_attribute(vec_lib, device):
    """Verify that vscale_range is set on kernel functions when SVE is available.

    When SVE is present, every function in the generated LLVM IR module should
    carry the ``vscale_range(min, max)`` attribute so that LLVM knows the
    function may use scalable vectors.  When SVE is not present, the attribute
    should be absent.
    """
    if not is_cpu():
        pytest.skip("This test is CPU-specific")

    @triton.jit
    def kernel(x_ptr, y_ptr, N: tl.constexpr):
        idx = tl.arange(0, N)
        x = tl.load(x_ptr + idx)
        y = tl.exp(x)
        tl.store(y_ptr + idx, y)

    x = torch.rand(128, dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    meta = kernel[(1, )](x, y, N=128, vec_lib=vec_lib)

    llir = meta.asm["llir"]
    if has_sve():
        assert "vscale_range" in llir, ("Expected vscale_range attribute in LLVM IR when SVE is available, "
                                        "but it was not found.  This indicates that the target machine was "
                                        "not configured with SVE features or setVscaleRangeForSVE() was "
                                        "not called.")
    else:
        assert "vscale_range" not in llir, ("vscale_range attribute found in LLVM IR even though SVE is not "
                                            "available.  The attribute should only be set when SVE is present.")
