import itertools
import math

import pytest
import torch

import triton
import triton.language as tl


@pytest.mark.cpu
@pytest.mark.interpreter
@pytest.mark.parametrize("dtype_str", ["int32", "int8"])
@pytest.mark.parametrize("shape", [(2, 2, 2, 16), (2, 4, 2, 16)])
@pytest.mark.parametrize("perm", list(itertools.permutations([0, 1, 2, 3])))
def test_cpu_trans_4d(dtype_str, shape, perm, device, with_allocator):

    @triton.jit
    def kernel(In, Out, in_shape1: tl.constexpr, in_shape2: tl.constexpr, in_shape3: tl.constexpr,
               in_shape4: tl.constexpr, ou_shape1: tl.constexpr, ou_shape2: tl.constexpr, ou_shape3: tl.constexpr,
               ou_shape4: tl.constexpr, trans1: tl.constexpr, trans2: tl.constexpr, trans3: tl.constexpr,
               trans4: tl.constexpr):
        in_desc = tl.make_tensor_descriptor(
            base=In,
            shape=[in_shape1, in_shape2, in_shape3, in_shape4],
            strides=[in_shape4 * in_shape3 * in_shape2, in_shape4 * in_shape3, in_shape4, 1],
            block_shape=[in_shape1, in_shape2, in_shape3, in_shape4],
        )
        out_desc = tl.make_tensor_descriptor(
            base=Out,
            shape=[ou_shape1 * ou_shape2 * ou_shape3 * ou_shape4],
            strides=[1],
            block_shape=[ou_shape1 * ou_shape2 * ou_shape3 * ou_shape4],
        )
        val = in_desc.load([0, 0, 0, 0]).permute((trans1, trans2, trans3, trans4))
        out_desc.store([0], val.reshape(out_desc.block_shape))

    input = torch.arange(math.prod(shape), dtype=getattr(torch, dtype_str), device=device).reshape(shape)
    expected = torch.permute(input, perm)
    actual = torch.zeros(expected.shape, dtype=input.dtype, device=device)

    kernel[(1, )](input, actual, *shape, *[shape[axis] for axis in perm], *perm, num_warps=8)

    torch.testing.assert_close(actual, expected)
