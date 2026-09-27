import pytest

pytestmark = pytest.mark.cpu


def test_arm_feat_bf16_native(device):
    """The native feature query should return without raising."""
    if device != "cpu":
        pytest.skip("CPU capability tests require --device cpu.")

    from triton._C.libtriton import cpu

    supported = cpu.has_arm_feat_bf16()
    print(f"Arm FEAT_BF16 supported: {supported}")
