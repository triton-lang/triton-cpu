from __future__ import annotations

import numpy as np
import pytest
import torch

import triton
import triton.language as tl


def annotated_function(return_type=None, **arg_types):

    def decorator(func):
        func.__annotations__ = {**arg_types, "return": return_type}
        return func

    return decorator


# Keep CPU-specific expected failures local: nonzero fp16 and bf16 scalar
# annotations remain NYI, while zero and wider floating-point values must pass.
@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [
    pytest.param(tl.float16, id="float16"),
    pytest.param(tl.bfloat16, id="bfloat16"),
    pytest.param(tl.float32, id="float32"),
    pytest.param(tl.float64, id="float64"),
])
@pytest.mark.parametrize("test_val", [0.0, 42.0, float("inf"), float("nan")])
def test_cpu_float_annotation(device, dtype, test_val):
    if dtype in (tl.float16, tl.bfloat16) and test_val != 0.0:
        pytest.xfail("Nonzero float16 and bfloat16 scalar annotations are NYI in the CPU backend")

    @triton.jit
    @annotated_function(val=dtype)
    def kernel(ptr, val):
        tl.static_assert(val.dtype == dtype)
        tl.store(ptr, val)

    ptr = torch.empty(1, device=device, dtype=torch.float32)
    compiled = kernel[(1, )](ptr, test_val)
    np.testing.assert_allclose(ptr.cpu().numpy(), [test_val], atol=1e-6)

    if dtype == tl.float16:
        assert "%val: f16" in compiled.asm["ttir"]
        assert "arith.extf %val : f16 to f32" in compiled.asm["ttir"]
    elif dtype == tl.bfloat16:
        assert "%val: bf16" in compiled.asm["ttir"]
        assert "arith.extf %val : bf16 to f32" in compiled.asm["ttir"]
    elif dtype == tl.float32:
        assert "%val: f32" in compiled.asm["ttir"]
    elif dtype == tl.float64:
        assert "%val: f64" in compiled.asm["ttir"]
        assert "arith.truncf %val : f64 to f32" in compiled.asm["ttir"]
