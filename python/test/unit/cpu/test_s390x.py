"""Vector codegen smoke test for the s390x CPU backend."""

import os

import pytest


def _running_on_s390x():
    if os.environ.get("TRITON_INTERPRET", "0") == "1":
        return False
    try:
        import triton
    except ImportError:
        return False
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cpu" and target.arch == "s390x"


@pytest.mark.skipif(not _running_on_s390x(), reason="s390x CPU backend only")
def test_s390x_vector_add_and_dot(device):
    import torch

    import triton
    import triton.language as tl

    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, BLOCK: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        x = tl.load(x_ptr + offs)
        y = tl.load(y_ptr + offs)
        tl.store(out_ptr + offs, x + y)

    @triton.jit
    def dot_kernel(a_ptr, b_ptr, c_ptr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
        offs_m = tl.arange(0, M)
        offs_n = tl.arange(0, N)
        offs_k = tl.arange(0, K)
        a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :])
        b = tl.load(b_ptr + offs_k[:, None] * N + offs_n[None, :])
        c = tl.dot(a, b)
        tl.store(c_ptr + offs_m[:, None] * N + offs_n[None, :], c)

    block = 128
    x = torch.randn(block, dtype=torch.float32, device=device)
    y = torch.randn(block, dtype=torch.float32, device=device)
    out = torch.empty_like(x)
    add_meta = add_kernel[(1, )](x, y, out, BLOCK=block)
    torch.testing.assert_close(out, x + y)
    add_asm = add_meta.asm["asm"]
    # Vector load / single-precision add from the z/Architecture vector facility.
    assert "vfasb" in add_asm or "\tvl\t" in add_asm or "\tvl " in add_asm

    m = n = k = 16
    a = torch.randn((m, k), dtype=torch.float32, device=device)
    b = torch.randn((k, n), dtype=torch.float32, device=device)
    c = torch.empty((m, n), dtype=torch.float32, device=device)
    dot_meta = dot_kernel[(1, )](a, b, c, M=m, N=n, K=k)
    torch.testing.assert_close(c, a @ b, atol=1e-4, rtol=1e-4)
    # Vector FP multiply-and-add, the VXE FMA the dot lowering emits.
    assert "vfmasb" in dot_meta.asm["asm"]
