import contextlib
import importlib.util
import sys
from pathlib import Path

import pytest

import triton
import triton.language as tl

shared_test_path = Path(__file__).parents[1] / "language" / "test_conversions.py"
shared_test_spec = importlib.util.spec_from_file_location("shared_test_conversions", shared_test_path)
assert shared_test_spec is not None and shared_test_spec.loader is not None
conversions = importlib.util.module_from_spec(shared_test_spec)
sys.modules[shared_test_spec.name] = conversions
shared_test_spec.loader.exec_module(conversions)

# CPU supports a narrower FP8 matrix than GPU backends. Unsupported source
# formats are checked as compile errors below; downcasts cover e5, e4nv, and
# e5b16, while FP8 downcast clamping remains NYI and is excluded from CPU CI.


def require_cpu(device):
    if device != "cpu":
        pytest.skip("CPU conversion tests require --device cpu.")


@pytest.mark.cpu
@pytest.mark.parametrize("src_dtype", ["float8e4b8", "float8e4b15"])
def test_cpu_unsupported_fp8_upcast(src_dtype, device):
    require_cpu(device)

    with pytest.raises(triton.CompilationError, match="not supported in this architecture"):
        conversions.launch_exhaustive_populate(getattr(tl, src_dtype), 0, 65536, False, 8, 0x7f, device=device)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [tl.float8e5, tl.float8e5b16, tl.float8e4nv, tl.float8e4b8, tl.float8e4b15])
def test_cpu_fp8_dot_compile_support(fresh_triton_cache, dtype, device):
    require_cpu(device)

    supported_dtypes = [tl.float8e5, tl.float8e5b16, tl.float8e4nv]

    @triton.jit
    def dtype_kernel(dtype: tl.constexpr):
        a = tl.full((64, 64), 0.0, dtype)
        tl.dot(a, a)

    if dtype in supported_dtypes:
        ctx = contextlib.nullcontext()
    else:
        ctx = pytest.raises(triton.CompilationError)

    with ctx as exc_info:
        triton.compile(
            triton.compiler.ASTSource(fn=dtype_kernel, signature={"dtype": "constexpr"}, constexprs={"dtype": dtype}))

    if dtype not in supported_dtypes:
        assert "not supported in this architecture" in str(exc_info.value.__cause__)


@pytest.mark.cpu
@pytest.mark.parametrize("src_dtype, dst_dtype, rounding, max_repr", [
    ("float32", "float8e5", "rtne", 0x47600000),
    ("float32", "float8e5", "rtz", 0x47600000),
    ("float32", "float8e4nv", "rtne", 0x43e00000),
    ("float32", "float8e5b16", "rtne", 0x47600000),
    ("bfloat16", "float8e5", "rtne", 0x4760),
    ("bfloat16", "float8e4nv", "rtne", 0x43e0),
    ("float16", "float8e5", "rtne", 0x7b00),
    ("float16", "float8e4nv", "rtne", 0x5f00),
    ("bfloat16", "float8e5b16", "rtne", 0x4760),
    ("float16", "float8e5b16", "rtne", 0x7b00),
])
def test_cpu_typeconvert_downcast(src_dtype, dst_dtype, rounding, max_repr, device):
    require_cpu(device)

    exponent_bits, mantissa_bits, exponent_bias = {
        "float8e5": (5, 2, 15),
        "float8e4nv": (4, 3, 7),
        "float8e5b16": (5, 2, 16),
    }[dst_dtype]

    # Exhaustive conversion is expensive on CPU. Sampling 16 high-byte
    # offsets retains the coverage used by the CPU backend test suite.
    for offset in range(16):
        conversions.downcast_test(
            getattr(tl, src_dtype),
            getattr(tl, dst_dtype),
            rounding,
            exponent_bits,
            mantissa_bits,
            exponent_bias,
            max_repr,
            offset,
            device=device,
        )
