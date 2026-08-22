import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import numpy_random

# This matrix preserves the CPU configurations selected when these tests were
# isolated: shapes with both M and N non-unit had previously failed. The K=64
# last-dimension case remains as regression coverage for a former xfail.
flip_cases = [
    pytest.param(1, 16, 64, 0),
    pytest.param(1, 16, 64, 1),
    pytest.param(1, 16, 64, 2),
    pytest.param(1, 16, 64, -2),
    pytest.param(32, 1, 2, 0),
    pytest.param(32, 1, 2, 1),
    pytest.param(32, 1, 2, 2),
    pytest.param(32, 1, 2, -2),
]


@pytest.mark.cpu
@pytest.mark.parametrize("M, N, K, dim", flip_cases)
@pytest.mark.parametrize("dtype_str", ["int32", "float16", "float32", "bfloat16"])
def test_cpu_flip(M, N, K, dtype_str, dim, device):

    @triton.jit
    def flip_kernel(X, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, dim: tl.constexpr):
        offx = tl.arange(0, M) * N * K
        offy = tl.arange(0, N) * K
        offz = tl.arange(0, K)
        off3d = offx[:, None, None] + offy[None, :, None] + offz[None, None, :]
        x = tl.load(X + off3d)
        x = tl.flip(x, dim)
        tl.store(Z + off3d, x)

    x = torch.from_numpy(numpy_random((M, N, K), dtype_str=dtype_str)).to(device)
    expected = torch.flip(x, (dim, ))
    actual = torch.empty_like(x, device=device)
    flip_kernel[(1, )](x, actual, M, N, K, dim, num_warps=8)
    assert (expected == actual).all(), (expected, actual)
