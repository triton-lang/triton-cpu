import os
import platform
import re
import pytest
import torch

import triton
import triton.language as tl
from triton._C.libtriton import cpu
"""
This test is only for the CPU backend.
"""


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_cpu():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "cpu"


def is_x86():
    return is_cpu() and \
        triton.runtime.driver.active.get_current_target().arch == "x86_64"


def test_scalar_pointer_arith(device):

    @triton.jit
    def kernel(src, dst, BLOCK_SIZE: tl.constexpr):
        offs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + offs)
        tl.store(dst + offs, x)

    src = torch.rand((128, ), dtype=torch.float32, device=device)
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, BLOCK_SIZE=128)
    assert (src == res).all()

    # Check TTCIR doesn't have pointer extraction from a tensor.
    ttcir = meta.asm["ttcir"]
    assert ttcir.count("extract") == 0


@pytest.mark.parametrize("size", [32, 128, 135])
@pytest.mark.parametrize("tile_size", [16])
def test_optimize_tile_mask(size, tile_size, device):
    pytest.xfail("masked vector ops no longer eliminated by upstream canonicalizers")

    @triton.jit
    def kernel(src, dst, size, TILE_SIZE: tl.constexpr):
        for i in range(0, tl.cdiv(size, TILE_SIZE)):
            offs = tl.arange(0, TILE_SIZE) + i * TILE_SIZE
            mask = offs < size
            x = tl.load(src + offs, mask=mask, other=0)
            tl.store(dst + offs, x, mask=mask)

    src = torch.rand((size, ), dtype=torch.float32, device='cpu')
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, size, TILE_SIZE=tile_size)
    assert (src == res).all()

    # Check number of masked loads and stores.
    tttcir = meta.asm["tttcir"]
    masked_loads = tttcir.count("maskedload")
    masked_stores = tttcir.count("maskedstore")
    if size % tile_size == 0:
        assert masked_loads == 0
        assert masked_stores == 0
    else:
        assert masked_loads == 1
        assert masked_stores == 1


# Regression test for compilation failure in masks optimization
def test_vec_cdiv(device):

    @triton.jit
    def kernel(in_ptr, out_ptr):
        offs = tl.arange(0, 16)
        x = tl.load(in_ptr + offs)
        res = (x + 15) // 16
        tl.store(out_ptr + offs, res)

    arg0 = torch.zeros((16, ), dtype=torch.int32)
    arg1 = torch.empty_like(arg0)
    kernel[(1, )](arg0, arg1)


@pytest.mark.cpu
def test_cpu_triple_matches_compiled_target(device):
    if device != "cpu":
        pytest.skip("CPU target triple test requires --device cpu.")

    @triton.jit
    def kernel(dst):
        tl.store(dst, 1)

    output = torch.empty((), dtype=torch.int32, device=device)
    compiled = kernel[(1, )](output)
    assert output.item() == 1

    triple = cpu.llvm.get_cpu_triple()
    target = re.search(r'target triple = "([^"]+)"', compiled.asm["llir"])
    assert target is not None
    assert target.group(1) == triple
    machine = platform.machine().lower()
    # On macOS, both Python and LLVM can report arm64. Apply aliases to both
    # values to avoid comparing aarch64 with arm64 and failing incorrectly.
    arch_aliases = {"arm64": "aarch64", "amd64": "x86_64"}
    expected_arch = arch_aliases.get(machine, machine)
    reported_arch = triple.split("-")[0]
    assert arch_aliases.get(reported_arch, reported_arch) == expected_arch


@pytest.mark.cpu
@pytest.mark.parametrize("allow_fp_fusion", [False, True])
@pytest.mark.parametrize("default_fp_fusion", [False, True])
@pytest.mark.parametrize("enable_fp_fusion", [None, False, True])
@pytest.mark.parametrize("enable_fast_math", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("explicit_fma", [False, True])
@pytest.mark.parametrize("subtract", [False, True])
def test_cpu_fp_fusion(allow_fp_fusion, default_fp_fusion, enable_fp_fusion, enable_fast_math, dtype, scalar,
                       explicit_fma, subtract, device, fresh_knobs, tmp_path):
    if device != "cpu":
        pytest.skip("CPU fusion test requires --device cpu.")

    @triton.jit
    def kernel(A, B, C, Out, N, SCALAR: tl.constexpr, EXPLICIT_FMA: tl.constexpr, SUBTRACT: tl.constexpr,
               BLOCK: tl.constexpr):
        if SCALAR:
            a = tl.load(A)
            b = tl.load(B)
            c = tl.load(C)
            if EXPLICIT_FMA:
                result = tl.fma(a, b, -c if SUBTRACT else c)
            elif SUBTRACT:
                result = a * b - c
            else:
                result = a * b + c
            tl.store(Out, result)
        else:
            offsets = tl.arange(0, BLOCK)
            a = tl.load(A + offsets, offsets < N, other=0)
            b = tl.load(B + offsets, offsets < N, other=0)
            c = tl.load(C + offsets, offsets < N, other=0)
            if EXPLICIT_FMA:
                result = tl.fma(a, b, -c if SUBTRACT else c)
            elif SUBTRACT:
                result = a * b - c
            else:
                result = a * b + c
            tl.store(Out + offsets, result, offsets < N)

    fresh_knobs.cache.home_dir = str(tmp_path)
    fresh_knobs.language.allow_fp_fusion = allow_fp_fusion
    fresh_knobs.language.default_fp_fusion = default_fp_fusion
    effective_fusion = allow_fp_fusion and (default_fp_fusion if enable_fp_fusion is None else enable_fp_fusion)
    size = 1 if scalar else 17
    epsilon = torch.finfo(dtype).eps
    a = torch.full((size, ), 1 + epsilon, dtype=dtype, device=device)
    b = torch.full((size, ), 1 - epsilon, dtype=dtype, device=device)
    c = torch.full((size, ), 1 if subtract else -1, dtype=dtype, device=device)
    storage = torch.full((size + 8, ), 123, dtype=dtype, device=device)
    output = storage[4:-4]
    options = {"enable_fast_math": enable_fast_math, "num_cpu_threads": 1, "vec_lib": None}
    if enable_fp_fusion is not None:
        options["enable_fp_fusion"] = enable_fp_fusion
    compiled = kernel[(1, )](a, b, c, output, size, SCALAR=scalar, EXPLICIT_FMA=explicit_fma, SUBTRACT=subtract,
                             BLOCK=32, **options)

    fused = -(epsilon**2)
    if explicit_fma:
        torch.testing.assert_close(output, torch.full_like(output, fused), rtol=0, atol=0)
    elif effective_fusion:
        assert torch.all((output == 0) | (output == fused))
    else:
        assert torch.all(output == 0)
        assert not torch.signbit(output).any()
    assert torch.all(storage[:4] == 123)
    assert torch.all(storage[-4:] == 123)
    assert compiled.metadata.enable_fp_fusion == effective_fusion

    features = cpu.llvm.get_cpu_features()
    if "neon" in features or "fma" in features:
        # Include destructive SVE forms as well as scalar and x86 FMA.
        fma_pattern = r"\b(fn?m(add|sub|ad|sb)|fn?ml[as]|vfn?m(add|sub)\w*)\b"
        has_fma = re.search(fma_pattern, compiled.asm["asm"]) is not None
        assert has_fma == (explicit_fma or effective_fusion)
