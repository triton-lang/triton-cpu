import importlib.util
import pathlib
import sys

import pytest
import torch

import triton
import triton.language as tl
from triton._internal_testing import tma_dtypes
from triton.tools.tensor_descriptor import TensorDescriptor

_SHARED_TEST_PATH = pathlib.Path(__file__).parents[1] / "language" / "test_tensor_descriptor.py"
_SHARED_TEST_SPEC = importlib.util.spec_from_file_location("_shared_tensor_descriptor_tests", _SHARED_TEST_PATH)
assert _SHARED_TEST_SPEC is not None and _SHARED_TEST_SPEC.loader is not None
_SHARED_TESTS = importlib.util.module_from_spec(_SHARED_TEST_SPEC)
sys.modules[_SHARED_TEST_SPEC.name] = _SHARED_TESTS
_SHARED_TEST_SPEC.loader.exec_module(_SHARED_TESTS)

_test_functional_interface = _SHARED_TESTS.test_tensor_descriptor_functional_interface
_test_load = _SHARED_TESTS.test_tensor_descriptor_load
_test_load3d = _SHARED_TESTS.test_tensor_descriptor_load3d
_test_load_nd = _SHARED_TESTS.test_tensor_descriptor_load_nd
_test_store = _SHARED_TESTS.test_tensor_descriptor_store
_test_store3d = _SHARED_TESTS.test_tensor_descriptor_store3d
_test_store_nd = _SHARED_TESTS.test_tensor_descriptor_store_nd

pytestmark = pytest.mark.cpu

# These wrappers preserve the CPU subset selected when the tests were split:
# one CTA and block dimensions no larger than 32. Larger blocks and multi-CTA
# configurations had previously been skipped by the shared tests.


@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16), (8, 32)])
def test_cpu_tensor_descriptor_load(dtype_str, M_BLOCK, N_BLOCK, device):
    _test_load(dtype_str, 1, M_BLOCK, N_BLOCK, device)


@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("M_BLOCK,N_BLOCK", [(2, 16), (8, 16), (8, 32)])
def test_cpu_tensor_descriptor_store(dtype_str, M_BLOCK, N_BLOCK, device):
    _test_store(dtype_str, 1, M_BLOCK, N_BLOCK, device)


@pytest.mark.parametrize("dtype_str", tma_dtypes)
def test_cpu_tensor_descriptor_functional_interface(dtype_str, device):
    _test_functional_interface(dtype_str, device)


@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("K_BLOCK", [16, 32])
def test_cpu_tensor_descriptor_load3d(dtype_str, K_BLOCK, device):
    _test_load3d(dtype_str, K_BLOCK, device)


@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("K_BLOCK", [16, 32])
def test_cpu_tensor_descriptor_store3d(dtype_str, K_BLOCK, device):
    _test_store3d(dtype_str, K_BLOCK, device)


@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("INNER_BLOCK", [16, 32])
def test_cpu_tensor_descriptor_load_nd(dtype_str, ndim, INNER_BLOCK, device):
    _test_load_nd(dtype_str, 1, ndim, INNER_BLOCK, device)


@pytest.mark.parametrize("dtype_str", tma_dtypes)
@pytest.mark.parametrize("ndim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("INNER_BLOCK", [16, 32])
def test_cpu_tensor_descriptor_store_nd(dtype_str, ndim, INNER_BLOCK, device):
    _test_store_nd(dtype_str, 1, ndim, INNER_BLOCK, device)


def test_cpu_tensor_descriptor_padding(device):
    # The original CPU path exercised only descriptors created in the kernel;
    # keep padding coverage on that path rather than a host-passed descriptor.

    @triton.jit
    def device_descriptor_load(in_ptr, out_ptr, IM, IN, OM, ON, M_BLOCK: tl.constexpr, N_BLOCK: tl.constexpr,
                               padding: tl.constexpr):
        desc = tl.make_tensor_descriptor(
            in_ptr,
            shape=[IM, IN],
            strides=[IN, 1],
            block_shape=[M_BLOCK, N_BLOCK],
            padding_option=padding,
        )

        moffset = tl.program_id(0) * M_BLOCK
        noffset = tl.program_id(1) * N_BLOCK
        value = desc.load([moffset, noffset])

        offsets_m = moffset + tl.arange(0, M_BLOCK)
        offsets_n = noffset + tl.arange(0, N_BLOCK)
        tl.store(out_ptr + offsets_m[:, None] * ON + offsets_n[None, :], value)

    def alloc_fn(size: int, alignment: float, stream: float):
        return torch.ones(size, device=device, dtype=torch.float32)

    triton.set_allocator(alloc_fn)

    input_shape = (48, 48)
    output_shape = (64, 64)
    block_shape = (32, 32)
    inp = torch.arange(input_shape[0] * input_shape[1], device=device, dtype=torch.float32)
    inp = inp.reshape(input_shape)
    out = torch.zeros(output_shape, device=device, dtype=torch.float32)

    host_descriptor = TensorDescriptor(inp, inp.shape, inp.stride(), list(block_shape), padding="nan")
    assert tuple(host_descriptor.shape) == input_shape
    assert tuple(host_descriptor.block_shape) == block_shape

    grid = tuple(triton.cdiv(size, block) for size, block in zip(output_shape, block_shape))
    device_descriptor_load[grid](inp, out, *input_shape, *output_shape, *block_shape, "nan")

    expected = torch.zeros(output_shape, device=device, dtype=torch.float32)
    expected[:input_shape[0], :input_shape[1]] = inp
    expected[:, input_shape[1]:] = float("nan")
    expected[input_shape[0]:, :] = float("nan")
    torch.testing.assert_close(expected, out, equal_nan=True)
