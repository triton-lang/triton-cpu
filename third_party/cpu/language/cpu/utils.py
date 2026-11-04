from triton import jit
import triton.language as tl
from triton.language.core import builtin


@jit
def _vnni_decode2(arg0):
    tl.static_assert(len(arg0.shape) == 2)
    tmp = arg0.reshape((arg0.shape[0], arg0.shape[1] // 2, 2))
    tmp1, tmp2 = tl.split(tmp)
    return tl.join(tmp1.T, tmp2.T).reshape((arg0.shape[1] // 2, arg0.shape[0] * 2)).T


@jit
def _vnni_decode4(arg0):
    tl.static_assert(len(arg0.shape) == 2)
    tmp = arg0.reshape((arg0.shape[0], arg0.shape[1] // 2, 2))
    tmp02, tmp13 = tl.split(tmp)
    tmp02_2 = tmp02.reshape((tmp02.shape[0], tmp02.shape[1] // 2, 2))
    tmp13_2 = tmp13.reshape((tmp13.shape[0], tmp13.shape[1] // 2, 2))
    tmp0, tmp2 = tl.split(tmp02_2)
    tmp1, tmp3 = tl.split(tmp13_2)
    tmp4 = tl.join(tmp0.T, tmp2.T)
    tmp4_2 = tl.reshape(tmp4, (tmp4.shape[0], tmp4.shape[1] * 2))
    tmp5 = tl.join(tmp1.T, tmp3.T)
    tmp5_2 = tl.reshape(tmp5, (tmp5.shape[0], tmp5.shape[1] * 2))
    tmp6 = tl.join(tmp4_2, tmp5_2)
    return tmp6.reshape((arg0.shape[1] // 4, arg0.shape[0] * 4)).T


@builtin
def vnni_decode(arg0, _semantic=None, _generator=None):
    bitwidth = arg0.dtype.primitive_bitwidth
    if bitwidth == 16:
        decoded = _generator.call_JitFunction(_vnni_decode2, (arg0, ), kwargs={})
    elif bitwidth == 8:
        decoded = _generator.call_JitFunction(_vnni_decode4, (arg0, ), kwargs={})
    else:
        raise ValueError("Expected 8-bit or 16-bit values for vnni_decode")
    return decoded
