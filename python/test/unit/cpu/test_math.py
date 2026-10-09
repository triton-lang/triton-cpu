import inspect
import os
import re
import pytest
import torch

import triton
import triton.language as tl
from triton._C.libtriton import cpu, ir
from triton.language.extra import libdevice
from itertools import chain, product


def get_native_vector_size_in_bits():
    """
    Returns the fixed vector size used by external SLEEF calls and fallbacks.
    Assuming x86 always uses "auto dispatch" with 512-bit vectors for Sleef.
    """
    cpu_features = cpu.llvm.get_cpu_features()
    if "neon" in cpu_features:
        return 128
    return 512


def has_sve():
    return bool({"sve", "sve2"} & cpu.llvm.get_cpu_features())


def is_interpreter():
    return os.environ.get('TRITON_INTERPRET', '0') == '1'


def is_cpu():
    return not is_interpreter() and \
        triton.runtime.driver.active.get_current_target().backend == "cpu"


float_dtypes = ['float32', 'float64']
lib_prefix = {
    "libsleef": "Sleef",
    "libmvec": "_ZGV",
}
arch = triton.runtime.driver.active.get_current_target().arch

vec_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
scalar_sizes = [1, 4, 16, 64]


def check_num_vec_calls(meta, vec_lib, dtype_str, size, is_always_extern=False):
    # Check generated code calls vector math function
    # FP16 and BF16 are cast to FP32 for math ops
    elem_size = 8 if dtype_str == "float64" else 4
    data_size = size * elem_size

    vec_size = get_native_vector_size_in_bits() / 8  # bytes
    # 128-bit vector is the smallest supported by Sleef for both x86 and arm
    smallest_vec_size = 128 / 8  # bytes
    if vec_lib == "libsleef" and has_sve() and not is_always_extern and data_size >= smallest_vec_size:
        # The scalable loop has one static call site, regardless of how many
        # hardware vectors it processes at runtime.
        num_vec_calls = 1
    elif data_size > vec_size:
        num_vec_calls = data_size // vec_size
    elif data_size >= smallest_vec_size:
        num_vec_calls = 1
    else:
        num_vec_calls = 1 if is_always_extern else 0
    assert meta.asm["asm"].count(lib_prefix[vec_lib]) == num_vec_calls


@pytest.mark.parametrize("vec_lib, size",
                         chain(product(["libsleef", "libmvec"], vec_sizes), product([None], scalar_sizes)))
@pytest.mark.parametrize("dtype_str", float_dtypes)
@pytest.mark.parametrize("math_fn", ["cos", "exp", "exp2", "log", "log2", "sin"])
def test_tensor_math_fn(vec_lib, dtype_str, math_fn, size, device):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")
    if vec_lib == "libmvec" and arch != "x86_64":
        pytest.skip("Vectorized libm calls are supported for x86 target only.")

    @triton.jit
    def kernel(src, dst, MATH_FN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + idxs)
        y = getattr(x, MATH_FN)()
        tl.store(dst + idxs, y)

    src = torch.rand((size, ), dtype=getattr(torch, dtype_str), device=device)
    res = torch.empty(src.shape, dtype=getattr(torch, dtype_str), device=device)
    meta = kernel[(1, )](src, res, MATH_FN=math_fn, BLOCK_SIZE=size, vec_lib=vec_lib)
    ref = getattr(src, math_fn)()
    torch.testing.assert_close(ref, res)

    if vec_lib is not None:
        check_num_vec_calls(meta, vec_lib, dtype_str, size)


@pytest.mark.parametrize("vec_lib, size",
                         chain(product(["libsleef", "libmvec"], vec_sizes), product([None], scalar_sizes)))
@pytest.mark.parametrize("dtype_str", float_dtypes)
@pytest.mark.parametrize("math_fn", [
    "acos", "acosh", "asin", "asinh", "atan", "atanh", "cbrt", "ceil", "cos", "cosh", "erf", "exp", "exp2", "expm1",
    "floor", "fmod", "isnan", "isinf", "log", "log1p", "log2", "log10", "pow", "rsqrt", "signbit", "sin", "sinh",
    "sqrt", "tan", "tanh", "trunc"
])
def test_libdevice_math_fn(vec_lib, dtype_str, math_fn, size, device):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")
    if vec_lib == "libmvec" and arch != "x86_64":
        pytest.skip("Vectorized libm calls are supported for x86 target only.")
    if math_fn in {"ceil", "fmod", "pow"}:
        if vec_lib != "libsleef":
            pytest.skip("extern_elementwise only supports libsleef")
        if dtype_str not in {"float32", "torch.float64"}:
            pytest.skip(f"{math_fn} only supports fp32, fp64")

    @triton.jit
    def unary_kernel(src, dst, MATH_FN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(src + idxs)
        y = getattr(libdevice, MATH_FN)(x)
        tl.store(dst + idxs, y)

    @triton.jit
    def binary_kernel(x_ptr, y_ptr, out_ptr, MATH_FN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + idxs)
        y = tl.load(y_ptr + idxs)
        result = getattr(libdevice, MATH_FN)(x, y)
        tl.store(out_ptr + idxs, result)

    signature = inspect.signature(getattr(libdevice, math_fn))
    num_params = len(signature.parameters)
    inputs = [torch.rand((size, ), dtype=getattr(torch, dtype_str), device=device) for _ in range(num_params)]
    # Customize inputs
    if math_fn == "acosh":
        inputs[0] = inputs[0].abs() + 1
    if math_fn == "isnan" or math_fn == "isinf":
        indices = torch.randint(low=0, high=size, size=(size // 2, ), device=device)
        src = inputs[0]
        for i in indices:
            if math_fn == "isnan":
                src[i] = float("nan")
            else:
                src[i] = float(("+" if i % 2 else "-") + "inf")

    # Generate reference output
    if math_fn == "cbrt":
        ref = inputs[0].pow(1 / 3)
    else:
        ref = getattr(inputs[0], math_fn)(*inputs[1:])

    res = torch.empty(inputs[0].shape, dtype=ref.dtype, device=device)
    kernel = unary_kernel if num_params == 1 else binary_kernel
    meta = kernel[(1, )](*inputs, res, MATH_FN=math_fn, BLOCK_SIZE=size, vec_lib=vec_lib)
    torch.testing.assert_close(ref, res)

    if vec_lib is None:
        return

    # These are not implemented via extern library calls
    native_impls = {
        "libmvec": {"expm1", "floor", "isnan", "isinf", "rsqrt", "signbit", "sqrt", "trunc"},
        "libsleef": {"isnan", "isinf", "rsqrt", "signbit"},
    }
    # These are always implemented with extern library calls
    always_extern = {"ceil", "fmod", "pow"}
    if math_fn not in native_impls[vec_lib]:
        check_num_vec_calls(meta, vec_lib, dtype_str, size, is_always_extern=math_fn in always_extern)
    else:
        assert meta.asm["asm"].count(lib_prefix[vec_lib]) == 0


@pytest.mark.parametrize("dtype_str", ["float16", "bfloat16", "float32", "float64"])
@pytest.mark.parametrize("size", [4, 16, 64, 256])
@pytest.mark.parametrize("math_fn, ulp_suffix", [("sin", "_u10"), ("sqrt", "_u05"), ("floor", "_")])
def test_sleef_sve_vscale_reshaped_math(dtype_str, size, math_fn, ulp_suffix, device):
    if not is_cpu() or not has_sve():
        pytest.skip("This test requires an SVE CPU")

    @triton.jit
    def kernel(src, dst, MATH_FN: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        x = tl.reshape(tl.load(src + idxs), (2, BLOCK_SIZE // 2))
        y = getattr(libdevice, MATH_FN)(x)
        tl.store(dst + idxs, tl.reshape(y, (BLOCK_SIZE, )))

    # Distinct values across the entire block catch incorrect loop strides or
    # shape restoration, including a final partial hardware vector.
    src = torch.linspace(0.125, 3.125, size, device=device).to(getattr(torch, dtype_str))
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, MATH_FN=math_fn, BLOCK_SIZE=size, vec_lib="libsleef")
    ref = getattr(src, math_fn)()
    torch.testing.assert_close(res, ref)

    # Half and bfloat16 math is promoted before calling the SVE library.
    is_double = dtype_str == "float64"
    lanes, llvm_dtype, precision = (2, "double", "dx") if is_double else (4, "float", "fx")
    llir = meta.asm["llir"]
    assert "llvm.vscale" in llir
    assert f"<vscale x {lanes} x {llvm_dtype}>" in llir
    assert f"Sleef_{math_fn}{precision}{ulp_suffix}sve" in llir
    check_num_vec_calls(meta, "libsleef", dtype_str, size)


@pytest.mark.parametrize("dtype_str, size", [("float32", 1), ("float32", 2), ("float64", 1)])
def test_sleef_sve_small_vector_fallback(dtype_str, size, device):
    if not is_cpu() or not has_sve():
        pytest.skip("This test requires an SVE CPU")

    @triton.jit
    def kernel(src, dst, BLOCK_SIZE: tl.constexpr):
        idxs = tl.arange(0, BLOCK_SIZE)
        tl.store(dst + idxs, tl.sin(tl.load(src + idxs)))

    src = torch.linspace(0.125, 1.125, size, device=device).to(getattr(torch, dtype_str))
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, BLOCK_SIZE=size, vec_lib="libsleef")
    torch.testing.assert_close(res, src.sin())
    assert "llvm.vscale" not in meta.asm["llir"]
    assert "Sleef_" not in meta.asm["llir"]


def test_sleef_sve_chained_math_stack_usage(device):
    if not is_cpu() or not has_sve():
        pytest.skip("This test requires an SVE CPU")

    @triton.jit
    def kernel(src, dst, BLOCK_SIZE: tl.constexpr, DEPTH: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        first = tl.sin(tl.load(src + offsets))
        result = first
        for _ in tl.static_range(DEPTH - 1):
            result = tl.sin(result)
        # Keep the first result live across the remaining math operations.
        tl.store(dst + offsets, result + first)

    size = 256
    src = torch.linspace(0.125, 3.125, size, dtype=torch.float32, device=device)
    res = torch.empty_like(src)
    frames = []
    for depth in (1, 8):
        meta = kernel[(1, )](src, res, BLOCK_SIZE=size, DEPTH=depth, vec_lib="libsleef")
        first = src.sin()
        ref = first
        for _ in range(depth - 1):
            ref = ref.sin()
        torch.testing.assert_close(res, ref + first)

        llir = meta.asm["llir"]
        if depth > 1:
            # LLVM can turn the single entry-block scope into static allocas.
            assert "llvm.stacksave" in llir
            assert "llvm.stackrestore" in llir
        assert meta.asm["asm"].count("Sleef_sinfx_u10sve") == depth
        assembly = meta.asm["asm"]
        assert ".cfi_def_cfa" in assembly, "Expected AArch64 stack-frame unwind information"
        prologue = assembly.split(".cfi_def_cfa", 1)[0]
        frame_bytes = sum(map(int, re.findall(r"\[sp,\s*#-(\d+)\]!", prologue)))
        for amount, shift in re.findall(r"sub\s+sp,\s*sp,\s*#(\d+)(?:,\s*lsl\s*#(\d+))?", prologue):
            frame_bytes += int(amount) << int(shift or 0)
        assert frame_bytes > 0
        frames.append(frame_bytes)

    # Dynamic scopes add at most one scratch pair at a time. The fixed frame
    # can spill a live earlier result, but must not grow by a full input/output
    # buffer pair for every added math operation.
    buffer_pair_bytes = 2 * size * src.element_size()
    assert frames[1] <= frames[0] + buffer_pair_bytes, frames


@pytest.mark.parametrize("mode", [0, 1])
def test_sleef_sve_scratch_scopes_in_loop_branch(mode, device):
    if not is_cpu() or not has_sve():
        pytest.skip("This test requires an SVE CPU")

    @triton.jit(do_not_specialize=["count", "mode"])
    def kernel(src, dst, count, mode, BLOCK_SIZE: tl.constexpr):
        offsets = tl.arange(0, BLOCK_SIZE)
        initial = tl.load(src + offsets)
        result = initial
        for _ in range(count):
            if mode == 0:
                result = tl.sin(result)
            else:
                result = tl.cos(result)
            result = tl.sqrt(result * result + 1.0)
        tl.store(dst + offsets, result + initial)

    size = 256
    count = 5
    src = torch.linspace(0.125, 3.125, size, dtype=torch.float32, device=device)
    res = torch.empty_like(src)
    meta = kernel[(1, )](src, res, count, mode, BLOCK_SIZE=size, vec_lib="libsleef")
    ref = src
    for _ in range(count):
        ref = ref.sin() if mode == 0 else ref.cos()
        ref = (ref * ref + 1.0).sqrt()
    torch.testing.assert_close(res, ref + src)
    assert "llvm.stacksave" in meta.asm["llir"]
    assert "llvm.stackrestore" in meta.asm["llir"]


def run_math_to_vec_lib_pass(tmp_path, source, features):
    context = ir.context()
    ir.load_dialects(context)
    cpu.load_dialects(context)
    path = tmp_path / "math.mlir"
    path.write_text(source)
    module = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(context)
    cpu.passes.ttcpuir.add_math_to_vec_lib(pm, cpu.passes.ttcpuir.VecLib.libsleef, set(features))
    pm.run(module, "math_to_vec_lib_test")
    return str(module)


def math_vector_ir(math_fn, dtype, shape):
    count = 1
    for dim in shape.split("x"):
        count *= int(dim)
    flat_type = f"vector<{count}x{dtype}>"
    vec_type = f"vector<{shape}x{dtype}>"
    return f"""
module {{
  llvm.func @math_kernel(%arg: {flat_type}) -> {flat_type} {{
    %x = vector.shape_cast %arg : {flat_type} to {vec_type}
    %y = math.{math_fn} %x : {vec_type}
    %result = vector.shape_cast %y : {vec_type} to {flat_type}
    llvm.return %result : {flat_type}
  }}
}}
"""


@pytest.mark.parametrize("features", [("sve", ), ("neon", "sve"), ("sve2", )])
@pytest.mark.parametrize("dtype, shape", [("f32", "5"), ("f64", "3"), ("f32", "2x4"), ("f64", "2x3"), ("f16", "2x4"),
                                          ("bf16", "2x4")])
def test_sleef_sve_vscale_lowering(features, dtype, shape, tmp_path):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")

    # Feature overrides exercise the compiler on hosts without SVE. Odd fixed
    # vector lengths require a mask even at the minimum SVE vector length.
    result = run_math_to_vec_lib_pass(tmp_path, math_vector_ir("sin", dtype, shape), features)
    promoted_dtype = "f64" if dtype == "f64" else "f32"
    lanes, precision = (2, "dx") if dtype == "f64" else (4, "fx")
    scalable_type = f"vector<[{lanes}]x{promoted_dtype}>"
    symbol = f"Sleef_sin{precision}_u10sve"
    assert f"@{symbol}({scalable_type}) -> {scalable_type}" in result
    assert result.count(f"call @{symbol}") == 1
    assert "vector.vscale" in result
    assert "scf.for" in result
    assert "vector.create_mask" in result
    assert "vector.maskedload" in result
    assert "vector.maskedstore" in result
    assert "math.sin" not in result
    if dtype in {"f16", "bf16"}:
        assert "arith.extf" in result
        assert "arith.truncf" in result


def test_sleef_sve_scratch_scopes_lowering(tmp_path):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")

    source = """
module {
  llvm.func @math_kernel(%arg: vector<16xf32>) -> vector<16xf32> {
    %first = math.sin %arg : vector<16xf32>
    %second = math.cos %first : vector<16xf32>
    %result = arith.addf %first, %second : vector<16xf32>
    llvm.return %result : vector<16xf32>
  }
}
"""
    result = run_math_to_vec_lib_pass(tmp_path, source, {"neon", "sve"})
    lines = result.splitlines()
    # Scratch allocation and all buffer accesses must lie inside the
    # corresponding stack scope.
    allocas = set(re.findall(r"(%\w+) = memref.alloca", result))
    saves = {
        pointer: i
        for i, line in enumerate(lines)
        for pointer in re.findall(r"(%\w+) = llvm.intr.stacksave : !llvm.ptr", line)
    }
    restores = {
        pointer: i
        for i, line in enumerate(lines)
        for pointer in re.findall(r"llvm.intr.stackrestore (%\w+) : !llvm.ptr", line)
    }
    assert len(saves) == 2
    assert saves.keys() == restores.keys()
    covered_allocas = set()
    intervals = []
    for saved, start in saves.items():
        end = restores[saved]
        allocations = [(pointer, i) for i, line in enumerate(lines) if start < i < end
                       for pointer in re.findall(r"(%\w+) = memref.alloca", line)]
        assert len(allocations) == 2
        for buffer, allocation in allocations:
            accesses = [
                i for i, line in enumerate(lines)
                if re.search(r"vector\.(?:masked)?(?:load|store) .*" + re.escape(buffer) + r"\[", line)
            ]
            assert accesses, buffer
            assert allocation < min(accesses) <= max(accesses) < end
            covered_allocas.add(buffer)
        intervals.append((start, end))
    assert covered_allocas == allocas

    # The scratch scope of the first math evaluation must finish before
    # the second evaluation starts, even while its SSA result remains live.
    intervals.sort()
    assert intervals[0][1] < intervals[1][0]
    loads = re.findall(r"(%\w+) = vector.load", result)
    assert len(loads) == 2
    assert f"arith.addf {loads[0]}, {loads[1]}" in result


@pytest.mark.parametrize("math_fn, suffix", [("sqrt", "_u05sve"), ("floor", "_sve"), ("expm1", "_u10sve"),
                                             ("trunc", "_sve")])
def test_sleef_sve_additional_math_lowering(math_fn, suffix, tmp_path):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")

    result = run_math_to_vec_lib_pass(tmp_path, math_vector_ir(math_fn, "f32", "5"), {"neon", "sve"})
    assert f"@Sleef_{math_fn}fx{suffix}(vector<[4]xf32>) -> vector<[4]xf32>" in result
    assert "vector.vscale" in result
    assert f"math.{math_fn}" not in result


@pytest.mark.parametrize("features, bits", [(("neon", ), 128), (("avx", ), 256)])
@pytest.mark.parametrize("dtype, precision, elem_bits", [("f32", "f", 32), ("f64", "d", 64)])
def test_sleef_fixed_vector_fallback_lowering(features, bits, dtype, precision, elem_bits, tmp_path):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")

    result = run_math_to_vec_lib_pass(tmp_path, math_vector_ir("sin", dtype, "16"), features)
    lanes = bits // elem_bits
    symbol = f"Sleef_sin{precision}{lanes}_u10"
    assert f"@{symbol}(vector<{lanes}x{dtype}>) -> vector<{lanes}x{dtype}>" in result
    assert result.count(f"call @{symbol}") == 16 // lanes
    assert "vector.vscale" not in result
    assert "scf.for" not in result
    assert "sve(" not in result


@pytest.mark.parametrize("dtype, shape", [("f32", "1"), ("f32", "2"), ("f64", "1")])
def test_sleef_sve_small_vector_lowering(dtype, shape, tmp_path):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")

    result = run_math_to_vec_lib_pass(tmp_path, math_vector_ir("sin", dtype, shape), {"neon", "sve"})
    assert "math.sin" in result
    assert "vector.vscale" not in result
    assert "Sleef_" not in result


def test_sleef_sve_explicit_extern_fixed_abi(tmp_path):
    if not is_cpu():
        pytest.skip("This test is CPU-specific")

    source = """
module {
  llvm.func @math_kernel(%arg: vector<16xf32>) -> vector<16xf32> {
    %result = triton_cpu.extern_elementwise %arg {symbol = "Sleef_ceilf%(numel)", pure = true}
      : (vector<16xf32>) -> vector<16xf32>
    llvm.return %result : vector<16xf32>
  }
}
"""
    result = run_math_to_vec_lib_pass(tmp_path, source, {"neon", "sve"})
    assert "@Sleef_ceilf4(vector<4xf32>) -> vector<4xf32>" in result
    assert result.count("call @Sleef_ceilf4") == 4
    assert "vector.vscale" not in result
    assert "sve(" not in result
