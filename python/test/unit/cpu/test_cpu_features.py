import platform
import subprocess

import pytest

pytestmark = pytest.mark.cpu


def test_cpu_features(device):
    """The backend and LLVM codegen should use the same Arm feature query."""
    if device != "cpu":
        pytest.skip("CPU capability tests require --device cpu.")

    from triton._C.libtriton import cpu
    from triton.backends.compiler import GPUTarget
    from triton.backends.cpu.compiler import CPUBackend

    features = cpu.llvm.get_cpu_features()
    print(f"CPU features: {sorted(features)}")
    print(f"Arm FEAT_BF16 supported: {'bf16' in features}")
    assert isinstance(features, set)
    assert all(isinstance(feature, str) for feature in features)

    arch = cpu.llvm.get_cpu_triple().split("-")[0]
    if arch in ("aarch64", "arm64"):
        backend = CPUBackend(GPUTarget("cpu", arch, 0))
        assert backend.cpu_features == features
        if platform.system() == "Darwin":
            native = subprocess.run(["sysctl", "-n", "hw.optional.arm.FEAT_BF16"], capture_output=True, text=True)
            if native.returncode == 0 and native.stdout.strip() == "1":
                assert "bf16" in features
