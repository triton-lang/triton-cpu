# %%
# Kernels
# -------

import torch

import triton
import triton.language as tl

import argparse
import functools
import re
import sys

from gilbert_d2xy import gilbert_d2xy


# Transforms the B matrix into a tensor of shape:
#
#  (BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K, BLOCK_SIZE_N)
#
# Data is blocked into contiguous chunks of memory. Neighboring blocks in the K
# dimension will also be neighboring in memory.
@triton.jit
def block_transpose_pack_kernel(in_ptr, out_ptr, sfc_map_ptr, N, K,
                                    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_k = tl.load(sfc_map_ptr + 2 * pid)
    block_n = tl.load(sfc_map_ptr + 2 * pid + 1)

    in_desc = tl.make_tensor_descriptor(
        base=in_ptr,
        shape=(K, N),
        strides=(N, 1),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N))
    out_desc = tl.make_tensor_descriptor(
        base=out_ptr,
        shape=(N // BLOCK_SIZE_N, K // BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_SIZE_N),
        strides=(BLOCK_SIZE_N * K, BLOCK_SIZE_K * BLOCK_SIZE_N, BLOCK_SIZE_N, 1),
        block_shape=(1, 1, BLOCK_SIZE_K, BLOCK_SIZE_N))
    
    block = in_desc.load((block_k * BLOCK_SIZE_K, block_n * BLOCK_SIZE_N)).reshape((1, 1, BLOCK_SIZE_K, BLOCK_SIZE_N))
    out_desc.store((block_n, block_k, 0, 0), block)


# Matmul kernel using the space curve filling approach in https://arxiv.org/abs/2601.16294v1,
# based on the generalized hilbert curve implementation from https://github.com/jakubcerveny/gilbert
#
# Each program computes a single output tile with the 2D coordinates derived from the precomputed SFC mapping.
# If `BLOCKING_FACTOR_K == 1`, then program handles all `BLOCKS_K = K // BLOCK_SIZE_K` blocks along the common dimension,
# otherwise the program performs a partial accumulation of the K blocks in the half-open interval:
#    [ ik * (BLOCKS_K // BLOCKING_FACTOR_K), (ik + 1) * (BLOCKS_K // BLOCKING_FACTOR_K) )
#
@triton.jit
def sfc_kernel(a_ptr, b_ptr, c_ptr, sfc_map_ptr, M, N, K, ik,
               BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
               BLOCK_SIZE_K: tl.constexpr,
               DTYPE: tl.constexpr, ACC_DTYPE: tl.constexpr,
               BLOCKING_FACTOR_K: tl.constexpr):
    BLOCKS_M = M // BLOCK_SIZE_M
    BLOCKS_N = N // BLOCK_SIZE_N
    BLOCKS_K = K // BLOCK_SIZE_K
    BLOCKS_K_PER_PROG = BLOCKS_K // BLOCKING_FACTOR_K
    pid = tl.program_id(axis=0)
    block_m = tl.load(sfc_map_ptr + 2 * pid)
    block_n = tl.load(sfc_map_ptr + 2 * pid + 1)
    block_k = ik * BLOCKS_K_PER_PROG

    a_desc = tl.make_tensor_descriptor(base=a_ptr,
                                       shape=(BLOCKS_M, BLOCKS_K, BLOCK_SIZE_M, BLOCK_SIZE_K),
                                       strides=(BLOCK_SIZE_M * K, BLOCK_SIZE_K, K, 1),
                                       block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_K))
    
    b_desc = tl.make_tensor_descriptor(base=b_ptr,
                                       shape=(BLOCKS_N, BLOCKS_K, BLOCK_SIZE_K, BLOCK_SIZE_N),
                                       strides=(BLOCK_SIZE_N * K, BLOCK_SIZE_K * BLOCK_SIZE_N, BLOCK_SIZE_N, 1),
                                       block_shape=(1, 1, BLOCK_SIZE_K, BLOCK_SIZE_N))
    
    c_desc = tl.make_tensor_descriptor(base=c_ptr,
                                       shape=(BLOCKS_M, BLOCKS_N, BLOCK_SIZE_M, BLOCK_SIZE_N),
                                       strides=(BLOCK_SIZE_M * N, BLOCK_SIZE_N, N, 1),
                                       block_shape=(1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N))

    if ik == 0:
        c0 = tl.zeros((1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=DTYPE)
        c_desc.store([block_m, block_n, 0, 0], c0)

    c = c_desc.load([block_m, block_n, 0, 0]).reshape((BLOCK_SIZE_M, BLOCK_SIZE_N)).to(ACC_DTYPE)

    for block_ki in range(block_k, block_k + BLOCKS_K_PER_PROG):
        a = a_desc.load([block_m, block_ki, 0, 0]).reshape((BLOCK_SIZE_M, BLOCK_SIZE_K))
        b = b_desc.load([block_n, block_ki, 0, 0]).reshape((BLOCK_SIZE_K, BLOCK_SIZE_N))

        c = tl.dot(a, b, acc=c, out_dtype=ACC_DTYPE)

    c = c.to(DTYPE).reshape((1, 1, BLOCK_SIZE_M, BLOCK_SIZE_N))
    c_desc.store([block_m, block_n, 0, 0], c)


@functools.lru_cache
def make_sfc_tensor(x, y, dtype=torch.int32, device='cpu'):
    gilbert = (gilbert_d2xy(i, x, y) for i in range(x * y))
    return torch.tensor([c for xy in gilbert for c in xy], dtype=dtype, device=device)


def matmul(a: torch.Tensor, b: torch.Tensor, M, N, K, blocking_factor_k=1):
    assert (M % BLOCK_SIZE_M == 0) and (N % BLOCK_SIZE_N == 0) and (K % BLOCK_SIZE_K == 0), \
           "Masking currently not supported, matrix dimensions must be multiples of block size"

    sfc_map_mn = make_sfc_tensor(M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    sfc_map_kn = make_sfc_tensor(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N)

    bp = torch.empty((N // BLOCK_SIZE_N, K // BLOCK_SIZE_K, BLOCK_SIZE_K, BLOCK_SIZE_N), device=b.device, dtype=b.dtype)
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    tt_dtype = tl.bfloat16 if a.dtype == torch.bfloat16 else tl.int8
    tt_acc_dtype = tl.float32 if a.dtype == torch.bfloat16 else tl.int32

    block_transpose_pack_kernel[((K // BLOCK_SIZE_K) * (N // BLOCK_SIZE_N),)](
        b, bp, sfc_map_kn, N, K, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        assume_in_bounds=True)
    
    for ik in range(blocking_factor_k):
        sfc_kernel[((M // BLOCK_SIZE_M) * (N // BLOCK_SIZE_N),)](
            a, bp, c,  #
            sfc_map_mn, #
            M, N, K,  #
            ik,  #
            BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,  #
            DTYPE=tt_dtype, ACC_DTYPE=tt_acc_dtype,  #
            BLOCKING_FACTOR_K=blocking_factor_k,
            assume_in_bounds=True)
    return c


# %%
# Driver
# ---------

parser = argparse.ArgumentParser()
parser.add_argument('--target', choices=['amx', 'avx512', 'avx_ne_convert'], default='amx')
parser.add_argument('--dtype', choices=['bfloat16', 'int8'], default='bfloat16')
parser.add_argument('--bench', action='store_true')
parser.add_argument('-M', type=int, nargs='+', default=[512])
parser.add_argument('-N', type=int, nargs='+', default=[1024])
parser.add_argument('-K', type=int, nargs='+', default=[256])

args = parser.parse_args()

dtype_str = args.dtype
dtype = torch.bfloat16 if dtype_str == 'bfloat16' else torch.int8
acc_dtype = torch.float32 if dtype == torch.bfloat16 else torch.int32
if args.target == 'amx':
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 32 if dtype == torch.bfloat16 else 64
elif args.target == 'avx512':
    if dtype != torch.bfloat16:
        parser.error("AVX-512 target only supports bfloat16 data type")
    BLOCK_SIZE_M = 4
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 2
elif args.target == 'avx_ne_convert':
    if dtype != torch.bfloat16:
        parser.error("AVX-NE-CONVERT target only supports bfloat16 data type")
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_N = 32
    BLOCK_SIZE_K = 1

torch.manual_seed(0)
torch.set_printoptions(
    threshold=float("inf"),
    linewidth=1000,
)

triton.runtime.driver.set_active_to_cpu()


# %%
# Unit test
# ---------
M = args.M[0]
N = args.N[0]
K = args.K[0]

if dtype.is_floating_point:
    a = torch.randn((M, K), device='cpu', dtype=dtype)
    b = torch.randn((K, N), device='cpu', dtype=dtype)
else:
    a = torch.randint(-4, 4, (M, K), device='cpu', dtype=dtype)
    b = torch.randint(-4, 4, (K, N), device='cpu', dtype=dtype)

print(f"Running unit test with "
        f"M={M}, N={N}, K={K} dtype={dtype_str} "
        f"BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, BLOCK_SIZE_K={BLOCK_SIZE_K}...")

torch_output = torch.matmul(a, b)
sfc_map_mn = make_sfc_tensor(M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
sfc_map_kn = make_sfc_tensor(K // BLOCK_SIZE_K, N // BLOCK_SIZE_N)

triton_output = matmul(a, b, M=M, N=N, K=K, blocking_factor_k=1)
# print(triton_output)

if torch.allclose(triton_output, torch_output, atol=1e-5, rtol=1e-2):
    print("✅ TritonCPU pre-packed SFC and TorchCPU match")
else:
    print("⚠️ TritonCPU pre-packed SFC and TorchCPU differ, the maximum difference is "
        f'{torch.max(torch.abs(triton_output - torch_output))}')


# %%
# Benchmark
# ---------

if not args.bench:
    sys.exit(0)

def calculate_layers(M, N, K, dtype: torch.dtype, low_limit_in_gb=5.0):
    size_total = lambda n_layers: dtype.itemsize * n_layers * (M * K + K * N + M * N) / (1024.0 * 1024.0 * 1024.0)

    n_layers = 1
    while size_total(n_layers) < low_limit_in_gb:
        n_layers += 1
    
    return n_layers


def encode_triton_provider(sfc_bfk):
    return f"triton-cpu{f'-sfc{sfc_bfk}' if sfc_bfk > 0 else ''}-{dtype_str}"


def encode_torch_provider():
    return f"torch-cpu-native-{dtype_str}"


def decode_provider(provider):
    if 'triton-cpu' in provider:
        backend = 'triton-cpu'
    elif 'torch-cpu-native' in provider:
        backend = 'torch-cpu-native'

    sfc_bfk = 0
    if m := re.search(r'-sfc(\d+)', provider):
        sfc_bfk = int(m.group(1))

    return backend, sfc_bfk

SFC_BFK_OPTS = [1, 2, 4, 8]
LINE_VALS = [
    encode_triton_provider(sfc_bfk)
    for sfc_bfk in SFC_BFK_OPTS
] + [encode_torch_provider()]
LINE_NAMES = LINE_VALS
LINE_STYLES = None

X_VALS = [(M, N, K) for M in args.M for N in args.N for K in args.K]

if dtype.is_floating_point:
    rand_a = torch.randn((max(M * K * calculate_layers(M, N, K, dtype) for M, N, K in X_VALS),), device='cpu', dtype=dtype)
    rand_b = torch.randn((max(K * N * calculate_layers(M, N, K, dtype) for M, N, K in X_VALS),), device='cpu', dtype=dtype)
else:
    rand_a = torch.randint(-4, 4, (max(M * K * calculate_layers(M, N, K, dtype) for M, N, K in X_VALS),), device='cpu', dtype=dtype)
    rand_b = torch.randint(-4, 4, (max(K * N * calculate_layers(M, N, K, dtype) for M, N, K in X_VALS),), device='cpu', dtype=dtype)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
        x_vals=X_VALS,  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=LINE_VALS,  # Possible values for `line_arg`.
        line_names=LINE_NAMES,  # Label name for the lines.
        styles=LINE_STYLES,  # Line styles.
        ylabel='GFLOPS',  # Label name for the y-axis.
        plot_name=
        # Name for the plot. Used also as a file name for saving the plot.
        f'matmul-performance-{dtype_str} (BLOCK_SIZE_M={BLOCK_SIZE_M}, BLOCK_SIZE_N={BLOCK_SIZE_N}, BLOCK_SIZE_K={BLOCK_SIZE_K}',
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    ))
def benchmark(M, N, K, provider):
    backend, sfc_bfk = decode_provider(provider)

    print(f"Running {provider} with M={M}, N={N}, K={K}...")

    # Make sure we have enough independent matmuls so that tensors are always cold.
    n_layers = calculate_layers(M, N, K, dtype)
    a = rand_a[:n_layers * M * K].reshape(n_layers, M, K)
    b = rand_b[:n_layers * K * N].reshape(n_layers, K, N)

    quantiles = [0.5, 0.2, 0.8]
    if backend == 'torch-cpu-native':
        def doit():
            for i in range(n_layers):
                torch.matmul(a[i % n_layers], b[i % n_layers])
        ms, min_ms, max_ms = triton.testing.do_bench(doit, quantiles=quantiles, rep=10)
    elif backend == 'triton-cpu':
        def doit():
            for i in range(n_layers):
                matmul(a[i % n_layers], b[i % n_layers], M, N, K, blocking_factor_k=sfc_bfk)
        ms, min_ms, max_ms = triton.testing.do_bench(doit, quantiles=quantiles, measure_time_with_hooks=True, rep=10)
    perf = lambda ms: 2 * n_layers * M * N * K * 1e-9 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


benchmark.run(print_data=True, show_plots=True)
