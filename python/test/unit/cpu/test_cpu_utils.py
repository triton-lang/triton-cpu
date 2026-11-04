import pytest
import torch

import triton
import triton.language as tl


@pytest.mark.parametrize("dtype, vnni_factor", [(torch.bfloat16, 2), (torch.int8, 4)], ids=["bf16", "int8"])
@pytest.mark.parametrize("K, N", [(4, 1), (8, 4), (16, 8), (32, 16)])
def test_cpu_vnni_decode(dtype, vnni_factor, K, N, device):

    @triton.jit
    def decode_kernel(packed_ptr, decoded_ptr, K: tl.constexpr, N: tl.constexpr, VNNI_FACTOR: tl.constexpr):
        packed_rows = tl.arange(0, K // VNNI_FACTOR)
        packed_cols = tl.arange(0, N * VNNI_FACTOR)
        packed_offsets = packed_rows[:, None] * (N * VNNI_FACTOR) + packed_cols[None, :]
        packed = tl.load(packed_ptr + packed_offsets)
        decoded = tl.extra.cpu.vnni_decode(packed)
        rows = tl.arange(0, K)
        cols = tl.arange(0, N)
        tl.store(decoded_ptr + rows[:, None] * N + cols[None, :], decoded)

    values = torch.arange(K * N, device=device, dtype=torch.int32) % 256 - 128
    expected = values.to(dtype).reshape(K, N)
    packed = expected.reshape(K // vnni_factor, vnni_factor, N).permute(0, 2, 1).contiguous()
    packed = packed.reshape(K // vnni_factor, N * vnni_factor)
    actual = torch.empty_like(expected)
    decode_kernel[(1, )](packed, actual, K, N, vnni_factor)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
