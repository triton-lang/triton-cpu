"""
Matrix Multiplication
=====================
In this tutorial, matmul on CPU with different input layouts is tested.

This tutorial is optimized for AMX-enabled CPUs.

"""

# %%
# Kernels
# -------

import torch

import triton
import triton.language as tl
import os

DTYPE = os.getenv("DTYPE", "float32")
in_dtype = getattr(torch, DTYPE)
out_dtype = torch.float32 if in_dtype.is_floating_point else torch.int32
# Choose block size depending on dtype. We have more register
# capacity for bfloat16/float16 compared to float32.
BLOCK_SIZE_M = 8 if DTYPE == "float32" else 32
BLOCK_SIZE_N = 32
BLOCK_SIZE_K = 8 if DTYPE == "float32" else 64 // in_dtype.itemsize
GROUP_SIZE_M = 8


# This kernel is used for blocked encoding of input tensors for matmul.
#
# Blocked encoding is used to transform 2D tensor [M, N] into 4D tensor
# [M / BLOCK_SIZE_M, N / BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_N].
# This makes following access to blocks in matmul more efficient because
# each block is placed into a contiguous memory fragment and is likely
# to fit a single memory page.
#
# If TRANSPOSED_B is set to True then head dimensions of the RHS
# tensor are transposed. It provides contiguos placement for a column
# of blocks.
#
# If PACKED_B is set to True then B is VNNI encoded. Only works when
# BLOCKED_B is True.
#
# If TRANSPOSED_BLOCK_A is set to True then tail dimensions of the LHS
# tensor are transposed. Transposed LHS block better matches FMA lowering
# used by Triton CPU backend which processes RHS block row-by-row and LHS
# block column-by-column.
@triton.jit
def block_transpose_combined_kernel(in_a, out_a, in_b, out_b, M, N, K, BLOCK_SIZE_M: tl.constexpr,
                                    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
                                    BLOCKED_A: tl.constexpr, TRANSPOSED_BLOCK_A: tl.constexpr, BLOCKED_B: tl.constexpr,
                                    TRANSPOSED_B: tl.constexpr, PACKED_B: tl.constexpr):
    tl.static_assert(BLOCKED_A or not TRANSPOSED_BLOCK_A)
    tl.static_assert(BLOCKED_B or not TRANSPOSED_B)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    in_block_m = first_pid_m + (pid % group_size_m)
    in_block_n = (pid % num_pid_in_group) // group_size_m

    if BLOCKED_A:
        a_out_block_m = in_block_m
        A_OUT_BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE_K if TRANSPOSED_BLOCK_A else BLOCK_SIZE_M
        A_OUT_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_M if TRANSPOSED_BLOCK_A else BLOCK_SIZE_K
        A_OUT_BLOCKS_M = M // BLOCK_SIZE_M
        A_OUT_BLOCKS_K = K // BLOCK_SIZE_K
        A_OUT_STRIDE_M: tl.constexpr = A_OUT_BLOCK_SIZE_K
        A_OUT_STRIDE_BLOCK_M = BLOCK_SIZE_M * K
        A_OUT_STRIDE_BLOCK_K: tl.constexpr = BLOCK_SIZE_M * BLOCK_SIZE_K
        for in_block_k in tl.range(in_block_n, A_OUT_BLOCKS_K, N // BLOCK_SIZE_N):
            a_out_block_k = in_block_k
            a_in_desc = tl.make_tensor_descriptor(
                base=in_a, shape=(M, K), strides=(K, 1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K))
            a_out_desc = tl.make_tensor_descriptor(
                base=out_a, shape=(A_OUT_BLOCKS_M, A_OUT_BLOCKS_K, A_OUT_BLOCK_SIZE_M, A_OUT_BLOCK_SIZE_K),
                strides=(A_OUT_STRIDE_BLOCK_M, A_OUT_STRIDE_BLOCK_K, A_OUT_STRIDE_M, 1),
                block_shape=(1, 1, A_OUT_BLOCK_SIZE_M, A_OUT_BLOCK_SIZE_K))
            val = a_in_desc.load((in_block_m * BLOCK_SIZE_M, in_block_k * BLOCK_SIZE_K))
            if TRANSPOSED_BLOCK_A:
                val = val.T
            val = tl.reshape(val, (1, 1, A_OUT_BLOCK_SIZE_M, A_OUT_BLOCK_SIZE_K))
            a_out_desc.store((a_out_block_m, a_out_block_k, 0, 0), val)

    if BLOCKED_B:
        B_PACKED_NUM: tl.constexpr = 32 // in_b.type.element_ty.primitive_bitwidth if PACKED_B else 1
        PACKED_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // B_PACKED_NUM if PACKED_B else BLOCK_SIZE_K
        PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N * B_PACKED_NUM if PACKED_B else BLOCK_SIZE_N
        B_OUT_BLOCKS_K = N // BLOCK_SIZE_N if TRANSPOSED_B else K // BLOCK_SIZE_K
        B_OUT_BLOCKS_N = K // BLOCK_SIZE_K if TRANSPOSED_B else N // BLOCK_SIZE_N
        B_OUT_STRIDE_BLOCK_K = (K * BLOCK_SIZE_N if TRANSPOSED_B else BLOCK_SIZE_K * N)
        B_OUT_STRIDE_BLOCK_N: tl.constexpr = BLOCK_SIZE_K * BLOCK_SIZE_N
        for in_block_k in tl.range(in_block_m, K // BLOCK_SIZE_K, M // BLOCK_SIZE_M):
            b_out_block_k = in_block_n if TRANSPOSED_B else in_block_k
            b_out_block_n = in_block_k if TRANSPOSED_B else in_block_n
            b_in_desc = tl.make_tensor_descriptor(
                base=in_b, shape=(K, N), strides=(N, 1), block_shape=(1, BLOCK_SIZE_N))
            b_out_desc = tl.make_tensor_descriptor(
                base=out_b, shape=(B_OUT_BLOCKS_K, B_OUT_BLOCKS_N, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N),
                strides=(B_OUT_STRIDE_BLOCK_K, B_OUT_STRIDE_BLOCK_N, PACKED_BLOCK_SIZE_N, 1),
                block_shape=(1, 1, 1, PACKED_BLOCK_SIZE_N))
            for i in tl.range(0, BLOCK_SIZE_K // B_PACKED_NUM):
                row1 = b_in_desc.load(
                    (in_block_k * BLOCK_SIZE_K + i * B_PACKED_NUM, in_block_n * BLOCK_SIZE_N)).reshape((BLOCK_SIZE_N, ))
                if B_PACKED_NUM > 1:
                    row2 = b_in_desc.load(
                        (in_block_k * BLOCK_SIZE_K + i * B_PACKED_NUM + 1, in_block_n * BLOCK_SIZE_N)).reshape((BLOCK_SIZE_N, ))
                    if B_PACKED_NUM > 2:
                        row3 = b_in_desc.load(
                            (in_block_k * BLOCK_SIZE_K + i * B_PACKED_NUM + 2, in_block_n * BLOCK_SIZE_N)).reshape((BLOCK_SIZE_N, ))
                        row4 = b_in_desc.load(
                            (in_block_k * BLOCK_SIZE_K + i * B_PACKED_NUM + 3, in_block_n * BLOCK_SIZE_N)).reshape((BLOCK_SIZE_N, ))
                        row1 = tl.ravel(tl.join(row1, row3))
                        row2 = tl.ravel(tl.join(row2, row4))
                    row1 = tl.ravel(tl.join(row1, row2))
                b_out_desc.store(
                    (b_out_block_k, b_out_block_n, i, 0), row1.reshape((1, 1, 1, PACKED_BLOCK_SIZE_N)))


# Matmul kernel that computes a single output block [BLOCK_SIZE_M, BLOCK_SIZE_N]. LHS can be in the
# rowmajor, blocked, or blocked transposed encoding. RHS can be in rowmajor, blocked, or transposed
# blocked encoding.
#
# To cover all input layouts, we use 4D block pointers that address a single input block
# [1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N], we choose strides for these block pointers
# appropriately to keep navigation bentween blocks similar for all input encodings.
#
# E.g. for rowmajor LHS we use BLOCK_SIZE_K stride to move to the next block over K axis, but
# for blocked encoding we use BLOCK_SIZE_M * BLOCK_SIZE_K stride. In both cases we then can
# advance using the same (0, 1, 0, 0) offset in the loop.
#
# Reshape is used to remove the heading (1, 1) dimensions, but CPU backend folds it with the load
# operation and it doesn't prevent direct vector loads from the input memory.
@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                  BLOCK_SIZE_K: tl.constexpr,
                  # number of blocks in a group
                  GROUP_SIZE_M: tl.constexpr, BLOCKED_A: tl.constexpr, TRANSPOSED_BLOCK_A: tl.constexpr,
                  BLOCKED_B: tl.constexpr, TRANSPOSED_B: tl.constexpr, PACKED_B: tl.constexpr, OUT_DTYPE: tl.constexpr):
    # TRANSPOSED_BLOCK_A means that each block in A is transposed.
    # It is allowed only for blocked input.
    assert (BLOCKED_A or not TRANSPOSED_BLOCK_A)
    # TRANSPOSED_B means that blocks of B are reordered but blocks
    # itself are not transpoed. It is allowed only for blocked input.
    assert (BLOCKED_B or not TRANSPOSED_B)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    block_m = first_pid_m + (pid % group_size_m)
    block_n = (pid % num_pid_in_group) // group_size_m

    A_BLOCK_SIZE_M: tl.constexpr = BLOCK_SIZE_K if TRANSPOSED_BLOCK_A else BLOCK_SIZE_M
    A_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_M if TRANSPOSED_BLOCK_A else BLOCK_SIZE_K
    A_BLOCKS_M = M // BLOCK_SIZE_M
    A_BLOCKS_K = K // BLOCK_SIZE_K
    a_stride_k: tl.constexpr = 1
    a_stride_m = A_BLOCK_SIZE_K if BLOCKED_A else K
    a_stride_block_k = A_BLOCK_SIZE_M * A_BLOCK_SIZE_K if BLOCKED_A else A_BLOCK_SIZE_K
    a_stride_block_m = BLOCK_SIZE_M * K

    B_PACKED_NUM: tl.constexpr = 32 // b_ptr.type.element_ty.primitive_bitwidth if PACKED_B else 1
    PACKED_BLOCK_SIZE_K: tl.constexpr = BLOCK_SIZE_K // B_PACKED_NUM if PACKED_B else BLOCK_SIZE_K
    PACKED_BLOCK_SIZE_N: tl.constexpr = BLOCK_SIZE_N * B_PACKED_NUM if PACKED_B else BLOCK_SIZE_N
    assert BLOCKED_B or not TRANSPOSED_B
    b_stride_n: tl.constexpr = 1
    b_stride_k = PACKED_BLOCK_SIZE_N if BLOCKED_B else N * B_PACKED_NUM
    if TRANSPOSED_B:
        b_stride_block_n = BLOCK_SIZE_N * K
        b_stride_block_k = BLOCK_SIZE_K * BLOCK_SIZE_N
    else:
        b_stride_block_n = BLOCK_SIZE_K * BLOCK_SIZE_N if BLOCKED_B else PACKED_BLOCK_SIZE_N
        b_stride_block_k = BLOCK_SIZE_K * N

    a_desc = tl.make_tensor_descriptor(base=a_ptr, shape=(A_BLOCKS_M, A_BLOCKS_K, A_BLOCK_SIZE_M, A_BLOCK_SIZE_K),
                                       strides=(a_stride_block_m, a_stride_block_k, a_stride_m, a_stride_k),
                                       block_shape=(1, 1, A_BLOCK_SIZE_M, A_BLOCK_SIZE_K))
    b_desc = tl.make_tensor_descriptor(base=b_ptr,
                                       shape=(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N),
                                       strides=(b_stride_block_k, b_stride_block_n, b_stride_k, b_stride_n),
                                       block_shape=(1, 1, PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N))
    c_desc = tl.make_tensor_descriptor(base=c_ptr, shape=(M, N), strides=(N, 1), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N))

    c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=OUT_DTYPE)
    for block_k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = a_desc.load((block_m, block_k, 0, 0)).reshape((A_BLOCK_SIZE_M, A_BLOCK_SIZE_K))
        b = b_desc.load((block_k, block_n, 0, 0)).reshape((PACKED_BLOCK_SIZE_K, PACKED_BLOCK_SIZE_N))

        if TRANSPOSED_BLOCK_A:
            a = a.T

        if PACKED_B:
            b = tl.extra.cpu.vnni_decode(b)

        c += tl.dot(a, b, out_dtype=OUT_DTYPE)

    c_desc.store((block_m * BLOCK_SIZE_M, block_n * BLOCK_SIZE_N), c)


def matmul(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, ab: torch.Tensor, bb: torch.Tensor, M, N, K, PREPACKED,
           BLOCKED_A, TRANSPOSED_BLOCK_A, BLOCKED_B, TRANSPOSED_B, PACKED_B, num_cpu_threads=0):
    #TODO: Currently masked load is not supported yet.
    assert (M % BLOCK_SIZE_M == 0) and (N % BLOCK_SIZE_N == 0) and (
        K % BLOCK_SIZE_K == 0), "Masking currently not supported, Matrix dimensions must be multiples of block size"
    # 1D launch kernel where each block gets its own program.
    grid = ((M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N), )
    if (BLOCKED_A or BLOCKED_B) and not PREPACKED:
        block_transpose_combined_kernel[grid](
            a, ab, b, bb,  #
            M, N, K,  #
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
            GROUP_SIZE_M=GROUP_SIZE_M,  #
            BLOCKED_A=BLOCKED_A, TRANSPOSED_BLOCK_A=TRANSPOSED_BLOCK_A,  #
            BLOCKED_B=BLOCKED_B, TRANSPOSED_B=TRANSPOSED_B, PACKED_B=PACKED_B)
        if BLOCKED_A:
            a = ab
        if BLOCKED_B:
            b = bb
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
        GROUP_SIZE_M=GROUP_SIZE_M,  #
        BLOCKED_A=BLOCKED_A, TRANSPOSED_BLOCK_A=TRANSPOSED_BLOCK_A,  #
        BLOCKED_B=BLOCKED_B, TRANSPOSED_B=TRANSPOSED_B, PACKED_B=PACKED_B,  #
        OUT_DTYPE=tl.float32 if a.dtype.is_floating_point else tl.int32, num_cpu_threads=num_cpu_threads)
    return c


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation.
torch.manual_seed(0)

triton.runtime.driver.set_active_to_cpu()

if in_dtype.is_floating_point:
    a = torch.randn((512, 512), device='cpu', dtype=in_dtype)
    b = torch.randn((512, 512), device='cpu', dtype=in_dtype)
else:
    a = torch.randint(0, 5, (512, 512), device='cpu', dtype=in_dtype)
    b = torch.randint(0, 5, (512, 512), device='cpu', dtype=in_dtype)
c = torch.empty((512, 512), device='cpu', dtype=out_dtype)
torch_output = torch.matmul(a.to(out_dtype), b.to(out_dtype))
rtol = 0
a_tmp = torch.zeros((512 * 512 + (512 // BLOCK_SIZE_M) * (512 // BLOCK_SIZE_K) * 64), device='cpu', dtype=in_dtype)
b_tmp = torch.zeros((512 * 512 + (512 // BLOCK_SIZE_K) * (512 // BLOCK_SIZE_N) * 64), device='cpu', dtype=in_dtype)
triton_output = matmul(a, b, c, a_tmp, b_tmp, 512, 512, 512, True, False, False, False, False, False)
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ TritonCPU and TorchCPU match")
else:
    print("❌ TritonCPU and TorchCPU differ, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')
    assert False
triton_output = matmul(a, b, c, a_tmp, b_tmp, 512, 512, 512, False, True, True, True, True, DTYPE != "float32")
if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
    print("✅ TritonCPU pre-packed and TorchCPU match")
else:
    print("❌ TritonCPU pre-packed and TorchCPU differ, the maximum difference is "
          f'{torch.max(torch.abs(triton_output - torch_output))}')
    assert False

# %%
# Benchmark
# ---------
#
# Square Matrix Performance
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now compare the performance of our kernel against that of Pytorch. Here we focus on square matrices,
# but feel free to arrange this script as you wish to benchmark any other matrix shape.


def encode_triton_provider(blocked_a, transposed_a, blocked_b, transposed_b, packed_b, prepack, single_thread, dtype):
    assert dtype == 'float32' or dtype == 'bfloat16' or dtype == 'float16' or dtype == 'int8'
    return f"triton-cpu{'-ba' if blocked_a else ''}{'-ta' if transposed_a else ''}{'-bb' if blocked_b else ''}{'-tb' if transposed_b else ''}{'-pb' if packed_b else ''}{'-prepack' if prepack else ''}{'-st' if single_thread else ''}-{dtype}"


def encode_torch_provider(single_thread, dtype):
    assert dtype == 'float32' or dtype == 'bfloat16' or dtype == 'float16' or dtype == 'int8'
    return f"torch-cpu-native{'-st' if single_thread else ''}-{dtype}"


def decode_provider(provider):
    if '-bfloat16' in provider:
        dtype = torch.bfloat16
    if '-float16' in provider:
        dtype = torch.float16
    elif '-float32' in provider:
        dtype = torch.float32
    elif '-int8' in provider:
        dtype = torch.int8

    # ukernel = None
    # if '-ukNone' in provider:
    #     ukernel = None
    # if '-ukOneDNN' in provider:
    #     ukernel = "OneDNN"
    # if '-ukXSMM' in provider:
    #     ukernel = "XSMM"

    if 'triton-cpu' in provider:
        backend = 'triton-cpu'
    elif 'torch-cpu-native' in provider:
        backend = 'torch-cpu-native'
    elif 'torch-cpu-compile' in provider:
        backend = 'torch-cpu-compile'
    return backend, '-ba' in provider, '-ta' in provider, '-bb' in provider, '-tb' in provider, '-pb' in provider, '-prepack' in provider, '-st' in provider, dtype


BLOCK_TRANSPOSE_A_OPTS = [(False, False)]
BLOCK_TRANSPOSE_PACK_B_OPTS = [(True, True, True), (True, True, False), (False, False, False)]
PREPACK_OPTS = [False, True]
SINGLE_THREAD_OPTS = [False]
DTYPE_OPTS = [DTYPE]
# contolled via TRITON_CPU_UKERNELS_LIB
# UKERNEL_OPTS = [None, "OneDNN", "XSMM"]
LINE_VALS = [
    encode_triton_provider(blocked_a, transposed_a, blocked_b, transposed_b, packed_b, prepack, single_thread, dtype)
    for single_thread in SINGLE_THREAD_OPTS
    for blocked_a, transposed_a in BLOCK_TRANSPOSE_A_OPTS
    for blocked_b, transposed_b, packed_b in BLOCK_TRANSPOSE_PACK_B_OPTS
    for prepack in PREPACK_OPTS
    for dtype in DTYPE_OPTS
    # for ukernel in UKERNEL_OPTS
    if (blocked_a or blocked_b or not prepack) and (not packed_b or dtype != "float32")
] + [encode_torch_provider(single_thread, dtype) for dtype in DTYPE_OPTS for single_thread in SINGLE_THREAD_OPTS]
LINE_NAMES = LINE_VALS
LINE_STYLES = None

default_num_threads = torch.get_num_threads()


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 21)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=LINE_VALS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=LINE_STYLES,  # Line styles.
        ylabel='GFLOPS',  # Label name for the y-axis.
        plot_name=
        # Name for the plot. Used also as a file name for saving the plot.
        f'matmul-performance-{DTYPE} (BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, BLOCK_SIZE_K={BLOCK_SIZE_K}, GROUP_SIZE_M={GROUP_SIZE_M})',
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(M, N, K, provider):

    device = 'cpu' if 'cpu' in provider else 'cuda'
    backend, blocked_a, transposed_a, blocked_b, transposed_b, packed_b, prepack, single_thread, dtype = decode_provider(
        provider)
    if dtype.is_floating_point:
        a = torch.randn((M, K), device=device, dtype=dtype)
        b = torch.randn((K, N), device=device, dtype=dtype)
    else:
        a = torch.randint(0, 5, (M, K), device=device, dtype=dtype)
        b = torch.randint(0, 5, (K, N), device=device, dtype=dtype)

    if single_thread:
        torch.set_num_threads(1)
    else:
        torch.set_num_threads(default_num_threads)

    if backend == 'triton-cpu':
        c = torch.zeros((M, N), device=a.device, dtype=out_dtype)
        a_tmp = torch.zeros((M * K + (M // BLOCK_SIZE_M) * (K // BLOCK_SIZE_K) * 64), device=device, dtype=dtype)
        b_tmp = torch.zeros((K * N + (K // BLOCK_SIZE_K) * (N // BLOCK_SIZE_N) * 64), device=device, dtype=dtype)
        c = torch.zeros((M, N), device=a.device, dtype=out_dtype)
        if prepack and (blocked_a or blocked_b):
            grid = ((M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N), )
            block_transpose_combined_kernel[grid](
                a, a_tmp, b, b_tmp,  #
                M, N, K,  #
                BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
                GROUP_SIZE_M=GROUP_SIZE_M,  #
                BLOCKED_A=blocked_a, TRANSPOSED_BLOCK_A=transposed_a,  #
                BLOCKED_B=blocked_b, TRANSPOSED_B=transposed_b, PACKED_B=packed_b)
            if blocked_a:
                a = a_tmp
            if blocked_b:
                b = b_tmp
    else:
        c = torch.zeros((M, N), device=a.device, dtype=dtype)

    quantiles = [0.5, 0.2, 0.8]
    if backend == 'torch-cpu-native':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b, out=c), quantiles=quantiles)
    elif backend == 'torch-cpu-compile':
        compiled = torch.compile(torch.matmul)
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: compiled(a, b, out=c), quantiles=quantiles)
    elif backend == 'triton-cpu':
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: matmul(a, b, c, a_tmp, b_tmp, M, N, K, prepack, blocked_a, transposed_a, blocked_b, transposed_b,
                           packed_b, num_cpu_threads=int(single_thread)), quantiles=quantiles,
            measure_time_with_hooks=True, rep=1000)
    perf = lambda ms: 2 * M * N * K * 1e-9 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


# %%
# We can now run the decorated function above. Pass `print_data=True` to see the performance number, `show_plots=True` to plot them, and/or
# `save_path='/path/to/results/' to save them to disk along with raw CSV data:
benchmark.run(print_data=True, show_plots=True)
