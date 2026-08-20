import sys
import uuid

import torch
from torch.testing import assert_close

import triton
import triton.language as tl


@triton.jit
def kernel_device_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.device_print("x: ", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_device_print_cast(BLOCK: tl.constexpr):
    x = tl.arange(0, BLOCK) + 128
    tl.device_print("x: ", x.to(tl.uint8))


@triton.jit
def kernel_device_print_hex(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.device_print("x: ", x, hex=True)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    # Triton should add a space after this prefix.
    print("x:", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_device_print_scalars(SCALAR, INT, FLOAT):
    x = tl.load(SCALAR)
    # Triton should add a space after this prefix.
    print("x:", x)
    print("int:", INT)
    print("float:", FLOAT)


@triton.jit
def kernel_device_print_large(
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    x = tl.full([BLOCK_M, BLOCK_N], 1, tl.int32)
    # Triton should change this prefix to "x: ".
    tl.device_print("x ", x)


@triton.jit
def kernel_print_multiple_args(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.full((BLOCK, ), 1, tl.int32)
    print("", x, y)


@triton.jit
def kernel_device_print_multiple_args(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.full((BLOCK, ), 1, tl.int32)
    tl.device_print("", x, y)
    tl.store(Y + tl.arange(0, BLOCK), y)


@triton.jit
def kernel_static_print(X, Y, BLOCK: tl.constexpr, PLACEHOLDER: tl.constexpr):
    # This function takes an extra value as a tl.constexpr so this kernel is not
    # cached. This way the static print is run every time.
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.static_print("", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_no_arg_print():
    print("", tl.program_id(0))


@triton.jit
def kernel_print_no_arg():
    print("no arg")


@triton.jit
def kernel_print_pointer(X, Y, BLOCK: tl.constexpr):
    tl.device_print("ptr ", X + tl.arange(0, BLOCK))


@triton.jit
def kernel_print_2d_tensor(X, Y, BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr):
    off_x = tl.arange(0, BLOCK_SIZE_X)
    off_y = tl.arange(0, BLOCK_SIZE_Y)
    x = tl.load(X + off_x[:, None] * BLOCK_SIZE_Y + off_y[None, :])
    tl.device_print("", x)


@triton.jit
def kernel_noop_pointer(_ptr):
    pass


@triton.jit
def kernel_noop():
    pass


class RefCountedZero(int):
    pass


class Pointer:

    def __init__(self, value):
        self.value = value
        self.dtype = torch.float32

    def data_ptr(self):
        return self.value


class RaisingPointer:
    dtype = torch.float32

    def __init__(self):
        self.calls = 0

    def data_ptr(self):
        self.calls += 1
        if self.calls == 1:
            return 0
        raise RuntimeError("data_ptr sentinel error")


def test_pointer_refcount(device: str):
    zero = RefCountedZero(0)
    pointer = Pointer(zero)
    kernel_noop_pointer[(1, )](pointer, num_warps=1, num_cpu_threads=1)
    initial_refcount = sys.getrefcount(zero)
    for _ in range(10):
        kernel_noop_pointer[(1, )](pointer, num_warps=1, num_cpu_threads=1)
    final_refcount = sys.getrefcount(zero)
    if final_refcount != initial_refcount:
        raise RuntimeError(f"data_ptr return reference leaked: {initial_refcount} -> {final_refcount}")


def test_pointer_error(device: str):
    kernel_noop_pointer[(1, )](RaisingPointer(), num_warps=1, num_cpu_threads=1)


def test_hook_refcount(device: str):
    result = object()

    def hook(_metadata):
        return result

    triton.knobs.runtime.launch_enter_hook = hook
    triton.knobs.runtime.launch_exit_hook = hook
    kernel_noop[(1, )](num_warps=1, num_cpu_threads=1)
    initial_refcount = sys.getrefcount(result)
    for _ in range(10):
        kernel_noop[(1, )](num_warps=1, num_cpu_threads=1)
    final_refcount = sys.getrefcount(result)
    if final_refcount != initial_refcount:
        raise RuntimeError(f"launch hook return reference leaked: {initial_refcount} -> {final_refcount}")


@triton.jit(noinline=True)
def print_context_from_subfunction(s0, s1, s2, s3, s4, s5):
    pid = tl.program_id(0) + 10 * tl.program_id(1)
    num_programs = tl.num_programs(0) + 10 * tl.num_programs(1)
    encoded_context = pid + 100 * num_programs
    # Keep all six sentinel arguments live so that the regression test catches
    # them being mistaken for pid and num_programs arguments.
    sentinel_sum = s0 + s1 + s2 + s3 + s4 + s5
    print("context:", encoded_context + 0 * sentinel_sum)


@triton.jit
def kernel_print_from_subfunction(SENTINELS):
    print_context_from_subfunction(
        tl.load(SENTINELS + 0),
        tl.load(SENTINELS + 1),
        tl.load(SENTINELS + 2),
        tl.load(SENTINELS + 3),
        tl.load(SENTINELS + 4),
        tl.load(SENTINELS + 5),
    )


def test_print_from_subfunction(device: str):
    sentinels = torch.tensor([11, 12, 13, 14, 15, 16], dtype=torch.int32, device=device)
    kernel_print_from_subfunction[(2, 3)](sentinels, num_warps=1, num_cpu_threads=1)


def test_print(func: str, data_type: str, device: str):
    if device != "cpu":
        raise ValueError(f"CPU print helper received unexpected device: {device}")

    N = 128  # This value should match test_cpu_print in test_cpu_subprocess.py.
    SCALAR = 42
    # The CPU target has no GPU warp. A logical size of one preserves the
    # launch geometry used by the CPU print lowering.
    num_warps = N

    x = torch.arange(0, N, dtype=torch.int32, device=device).to(getattr(torch, data_type))
    y = torch.zeros((N, ), dtype=x.dtype, device=device)
    if func == "device_print":
        kernel_device_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_scalars":
        scalar = torch.tensor(SCALAR, dtype=x.dtype, device=device)
        kernel_device_print_scalars[(1, )](scalar, SCALAR, 3.14, num_warps=num_warps)
    elif func == "device_print_negative":
        x = -x
        kernel_device_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_uint":
        x = torch.arange((1 << 31), (1 << 31) + N, device=device).to(getattr(torch, data_type))
        kernel_device_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_uint_cast":
        kernel_device_print_cast[(1, )](num_warps=num_warps, BLOCK=N)
    elif func == "print":
        kernel_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_large":
        kernel_device_print_large[(1, 2)](BLOCK_M=64, num_warps=num_warps, BLOCK_N=N)
    elif func == "print_multiple_args":
        kernel_print_multiple_args[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_multiple_args":
        kernel_device_print_multiple_args[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "static_print":
        kernel_static_print[(1, )](x, y, num_warps=num_warps, BLOCK=N, PLACEHOLDER=uuid.uuid4())
    elif func == "no_arg_print":
        kernel_no_arg_print[(1, )](num_warps=num_warps)
    elif func == "print_no_arg":
        kernel_print_no_arg[(1, )](num_warps=num_warps)
    elif func == "device_print_hex":
        kernel_device_print_hex[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_pointer":
        kernel_print_pointer[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_2d_tensor":
        block_size_x = N
        block_size_y = 1
        x_2d_tensor = x.reshape((block_size_x, block_size_y))
        kernel_print_2d_tensor[(1, )](x_2d_tensor, y, num_warps=num_warps, BLOCK_SIZE_X=block_size_x,
                                      BLOCK_SIZE_Y=block_size_y)
    else:
        raise ValueError(f"Unknown kernel: {func}")

    excluded_funcs = {
        "print_no_arg", "no_arg_print", "device_print_large", "print_multiple_args", "device_print_multiple_args",
        "device_print_pointer", "device_print_scalars", "device_print_2d_tensor", "device_print_uint_cast"
    }
    if func not in excluded_funcs:
        assert_close(y, x)


if __name__ == "__main__":
    fn = globals()[sys.argv[1]]
    fn(*sys.argv[2:])
