import os
import subprocess
import sys
import textwrap
from collections import Counter

import pytest

dir_path = os.path.dirname(os.path.realpath(__file__))
print_path = os.path.join(dir_path, "cpu_print_helper.py")
torch_types = ["int8", "uint8", "int16", "int32", "long", "float16", "float32", "float64"]


@pytest.mark.cpu
@pytest.mark.parametrize("func_type, data_type", [(fn, data_type)
                                                  for fn in ["device_print", "device_print_scalars"]
                                                  for data_type in torch_types] + [
                                                      ("print", "int32"),
                                                      ("static_print", "int32"),
                                                      ("no_arg_print", "int32"),
                                                      ("print_no_arg", "int32"),
                                                      ("device_print_large", "int32"),
                                                      ("print_multiple_args", "int32"),
                                                      ("device_print_multiple_args", "int32"),
                                                      ("device_print_hex", "int16"),
                                                      ("device_print_hex", "int32"),
                                                      ("device_print_hex", "int64"),
                                                      ("device_print_pointer", "int32"),
                                                      ("device_print_negative", "int32"),
                                                      ("device_print_uint", "uint32"),
                                                      ("device_print_uint_cast", "uint8"),
                                                      ("device_print_2d_tensor", "int32"),
                                                  ])
def test_cpu_print(func_type: str, data_type: str, device: str):
    if device != "cpu":
        pytest.skip("CPU print tests require --device cpu.")
    if data_type == "float16" or func_type in ["device_print_pointer", "device_print_large"]:
        pytest.skip("Printing float16, pointers, and large tensors is not yet supported on CPU.")

    env = os.environ.copy()
    env.pop("TRITON_INTERPRET", None)
    env["TRITON_DEFAULT_BACKEND"] = "cpu"
    proc = subprocess.run(
        [sys.executable, print_path, "test_print", func_type, data_type, device],
        capture_output=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr.decode("UTF-8", errors="replace")

    _check_cpu_print(proc.stdout.decode("UTF-8"), func_type, data_type, 128, 42)


@pytest.mark.cpu
def test_cpu_launcher_pointer_refcount(device: str):
    if device != "cpu":
        pytest.skip("CPU launcher tests require --device cpu.")

    env = os.environ.copy()
    env.pop("TRITON_INTERPRET", None)
    env["TRITON_DEFAULT_BACKEND"] = "cpu"
    proc = subprocess.run(
        [sys.executable, print_path, "test_pointer_refcount", device],
        capture_output=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr.decode("UTF-8", errors="replace")


@pytest.mark.cpu
def test_cpu_launcher_pointer_error(device: str):
    if device != "cpu":
        pytest.skip("CPU launcher tests require --device cpu.")

    env = os.environ.copy()
    env.pop("TRITON_INTERPRET", None)
    env["TRITON_DEFAULT_BACKEND"] = "cpu"
    proc = subprocess.run(
        [sys.executable, print_path, "test_pointer_error", device],
        capture_output=True,
        env=env,
    )
    assert proc.returncode != 0
    assert proc.stderr.decode("UTF-8", errors="replace").rstrip().endswith("RuntimeError: data_ptr sentinel error")


@pytest.mark.cpu
def test_cpu_launcher_hook_refcount(device: str):
    if device != "cpu":
        pytest.skip("CPU launcher tests require --device cpu.")

    env = os.environ.copy()
    env.pop("TRITON_INTERPRET", None)
    env["TRITON_DEFAULT_BACKEND"] = "cpu"
    proc = subprocess.run(
        [sys.executable, print_path, "test_hook_refcount", device],
        capture_output=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr.decode("UTF-8", errors="replace")


@pytest.mark.cpu
def test_cpu_print_from_subfunction(device: str):
    if device != "cpu":
        pytest.skip("CPU print tests require --device cpu.")

    env = os.environ.copy()
    env.pop("TRITON_INTERPRET", None)
    env["TRITON_DEFAULT_BACKEND"] = "cpu"
    proc = subprocess.run(
        [sys.executable, print_path, "test_print_from_subfunction", device],
        capture_output=True,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr.decode("UTF-8", errors="replace")

    actual = proc.stdout.decode("UTF-8")
    expected = [f"({x}, {y}, 0) context: {3200 + x + 10 * y}" for y in range(3) for x in range(2)]
    assert actual.endswith("\n")
    assert Counter(actual.splitlines()) == Counter(expected)


def _check_cpu_print(actual, func_type, data_type, N, SCALAR_VAL):
    # An example of tensor printing is:
    # (0, 0, 0) x: [  0,   1,   2,   3,   4,   5,   6,   7,
    #                 8,   9,  10,  11,  12,  13,  14,  15,
    #                 ...
    #               120, 121, 122, 123, 124, 125, 126, 127]
    PID_PREFIX = "(0, 0, 0)"
    NEWLINE_WITH_PADDING = "\n" + " " * len(PID_PREFIX + " x: [")
    if func_type in ("print", "device_print", "device_print_uint", "device_print_uint_cast"):
        expected = PID_PREFIX + " x: ["
        for i in range(N):
            if func_type == "device_print_uint_cast":
                offset = 128  # tl.arange(0, BLOCK) + 128
            elif data_type == "uint32":
                offset = 1 << 31
            else:
                offset = 0
            expected += f"{i + offset:3}"
            if data_type.startswith("float"):
                expected += ".0000"
            if i == N - 1:
                continue
            expected += ","
            expected += NEWLINE_WITH_PADDING if i % 8 == 7 else " "
        expected += "]"
    elif func_type == "device_print_scalars":
        expected = f"{PID_PREFIX} x: {SCALAR_VAL}"
        if data_type.startswith("float"):
            expected += ".000000"
        expected += f"\n{PID_PREFIX} int: {SCALAR_VAL}"
        expected += f"\n{PID_PREFIX} float: 3.140000"
    elif func_type == "device_print_negative":
        expected = PID_PREFIX + " x: ["
        for i in range(N):
            expected += f"{-i:4}"
            if i == N - 1:
                continue
            expected += ","
            expected += NEWLINE_WITH_PADDING if i % 8 == 7 else " "
        expected += "]"
    elif func_type == "device_print_hex":
        expected = PID_PREFIX + " x: ["
        for i in range(N):
            if data_type.endswith("8"):
                expected += f"0x{i:02x}"
            elif data_type.endswith("16"):
                expected += f"0x{i:04x}"
            elif data_type.endswith("32"):
                expected += f"0x{i:08x}"
            elif data_type.endswith("64"):
                expected += f"0x{i:016x}"
            if i == N - 1:
                continue
            expected += ","
            expected += NEWLINE_WITH_PADDING if i % 8 == 7 else " "
        expected += "]"
    elif func_type == "static_print":
        expected = f" int32[constexpr[{N}]]"
    elif func_type == "no_arg_print":
        expected = f"{PID_PREFIX}: 0"
    elif func_type == "print_no_arg":
        expected = f"{PID_PREFIX} no arg"
    elif func_type in ("print_multiple_args", "device_print_multiple_args"):
        expected = ""
        for k in range(2):
            expected += PID_PREFIX + ": ["
            for i in range(N):
                expected += f"{i:3}" if k == 0 else "1"
                if i == N - 1:
                    continue
                expected += ","
                if i % 8 == 7:
                    expected += "\n" + " " * len(PID_PREFIX + ": [")
                else:
                    expected += " "
            expected += "]"
            if k == 0:
                expected += "\n"
    elif func_type == "device_print_2d_tensor":
        # CPU prints a shape-(N, 1) tensor because it has no GPU warp.
        expected = PID_PREFIX + ": ["
        for i in range(N):
            expected += f"[{i:3}]"
            if i == N - 1:
                continue
            expected += ","
            expected += "\n" + " " * len(PID_PREFIX + ": [")
        expected += "]"
    else:
        raise ValueError(f"Unexpected CPU print test case: {func_type}")

    assert actual.endswith("\n")
    assert actual[:-1] == expected


@pytest.mark.cpu
@pytest.mark.parametrize("unload_order", [(0, 1), (1, 0)])
def test_cpu_module_unload_releases_library(device, tmp_path, unload_order):
    if device != "cpu":
        pytest.skip("CPU module unloading requires --device cpu.")

    from triton.runtime.build import _find_compiler

    source = tmp_path / "unload.c"
    binary = tmp_path / "unload.so"
    marker = tmp_path / "unloaded.txt"
    source.write_text(textwrap.dedent("""\
        #include <stdio.h>
        #include <stdlib.h>

        int unload_test_kernel(void) { return 42; }

        __attribute__((destructor)) static void record_unload(void) {
            FILE *file = fopen(getenv("TRITON_CPU_UNLOAD_TEST_MARKER"), "a");
            if (file) {
                fputs("unloaded\\n", file);
                fclose(file);
            }
        }
    """))
    subprocess.run([_find_compiler("c"), "-shared", "-fPIC", str(source), "-o", str(binary)], check=True)

    script = textwrap.dedent("""\
        import ctypes
        import sys
        from pathlib import Path
        from triton.backends.cpu.driver import CPUUtils

        binary, marker, first, second = sys.argv[1:]
        marker = Path(marker)
        utils = CPUUtils()
        modules = [utils.load_binary("unload_test_kernel", Path(binary).read_bytes(), 0, 0) for _ in range(2)]
        first, second = int(first), int(second)
        for module, pointer, *_ in modules:
            assert ctypes.CFUNCTYPE(ctypes.c_int)(pointer)() == 42
        utils.unload_module(modules[first][0])
        assert marker.read_text().splitlines() == ["unloaded"]
        assert ctypes.CFUNCTYPE(ctypes.c_int)(modules[second][1])() == 42
        utils.unload_module(modules[second][0])
        assert marker.read_text().splitlines() == ["unloaded", "unloaded"]
        for module, *_ in modules:
            utils.unload_module(module)
        assert marker.read_text().splitlines() == ["unloaded", "unloaded"]
    """)
    env = os.environ.copy()
    env["TRITON_CPU_UNLOAD_TEST_MARKER"] = str(marker)
    proc = subprocess.run([sys.executable, "-c", script, str(binary), str(marker), *map(str, unload_order)],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr
