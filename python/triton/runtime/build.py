from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import io
import logging
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import re

from types import ModuleType

from .cache import get_cache_manager
from .. import knobs


@contextlib.contextmanager
def quiet():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr


def _is_apple_clang():
    if platform.system() != "Darwin":
        return False
    res = subprocess.run(["clang", "--version"], capture_output=True, text=True)
    if res.returncode != 0:
        return False
    return "Apple clang" in res.stdout


def _build(name: str, src: str, srcdir: str, library_dirs: list[str], include_dirs: list[str], libraries: list[str],
           ccflags: list[str]) -> str:
    if impl := knobs.build.impl:
        return impl(name, src, srcdir, library_dirs, include_dirs, libraries)
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    system = platform.system()
    machine = platform.machine()
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    cc = os.environ.get("CC")
    if cc is None:
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError(
                "Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl.")
    scheme = sysconfig.get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    custom_backend_dirs = knobs.build.backend_dirs
    include_dirs = include_dirs + [srcdir, py_include_dir, *custom_backend_dirs]
    # for -Wno-psabi, see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=111047
    cc_cmd = [cc, src, "-O3", "-shared", "-fPIC", "-Wno-psabi", "-o", so]

    libraries += ["gcc"]
    # Use dynamic lookup to load Python library on Mac
    if system == "Darwin":
        cc_cmd += ["-undefined", "dynamic_lookup"]
        # Don't use libgcc on clang + macos
        if "clang" in cc:
            libraries.remove("gcc")

    cc_cmd += [_library_flag(lib) for lib in libraries]
    cc_cmd += [f"-L{dir}" for dir in library_dirs]
    cc_cmd += [f"-I{dir}" for dir in include_dirs if dir is not None]
    for dir in library_dirs:
        cc_cmd.extend(["-Wl,-rpath", dir])
    # CPU backend uses C++ (driver.cpp). Some old version compilers need a specific C++17 flag.
    if src.endswith(".cpp") or src.endswith(".cc"):
        cc_cmd += ["-std=c++17"]
        if not os.environ.get("TRITON_DISABLE_OPENMP", None):
            libomp_path = os.environ.get("TRITON_LOCAL_LIBOMP_PATH", None)
            if _is_apple_clang():
                if libomp_path:
                    cc_cmd += ["-Xclang"]
                    cc_cmd += ["-fopenmp"]
                    cc_cmd += [f"-I{libomp_path}/include"]
                    cc_cmd += [f"-L{libomp_path}/lib"]
                    cc_cmd += ["-lomp"]
                else:
                    print("Warning: TRITON_LOCAL_LIBOMP_PATH is not set for Apple clang. OpenMP is disabled.")
            else:
                cc_cmd += ["-fopenmp"]
                if libomp_path:
                    print("Info: Ignoring TRITON_LOCAL_LIBOMP_PATH for non-Apple clang compiler")
    if src.endswith(".s"):
        # This is required to properly parse .file directives
        cc_cmd += ["-g"]
        if system == "Linux" and machine in ("aarch64", "arm64"):
            # On Arm backend, some CPU (neoverse-v2) needs to be specified through -mcpu
            cc_cmd += ["-mcpu=native"]
    cc_cmd.extend(ccflags)
    subprocess.check_call(cc_cmd, stdout=subprocess.DEVNULL)
    return so


def _library_flag(lib: str) -> str:
    # Match .so files with optional version numbers (e.g., .so, .so.1, .so.513.50.1)
    if re.search(r'\.so(\.\d+)*$', lib) or lib.endswith(".a"):
        return f"-l:{lib}"
    return f"-l{lib}"


@functools.lru_cache
def platform_key() -> str:
    from platform import machine, system, architecture
    return ",".join([machine(), system(), *architecture()])


def _load_module_from_path(name: str, path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"Failed to load newly compiled {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compile_module_from_src(src: str, name: str, library_dirs: list[str] | None = None,
                            include_dirs: list[str] | None = None, libraries: list[str] | None = None,
                            ccflags: list[str] | None = None) -> ModuleType:
    key = hashlib.sha256((src + platform_key()).encode("utf-8")).hexdigest()
    cache = get_cache_manager(key)
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    cache_path = cache.get_file(f"{name}{suffix}")

    if cache_path is not None:
        try:
            return _load_module_from_path(name, cache_path)
        except (RuntimeError, ImportError):
            log = logging.getLogger(__name__)
            log.warning(f"Triton cache error: compiled module {name}.so could not be loaded")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, name + ".c")
        with open(src_path, "w") as f:
            f.write(src)
        so = _build(name, src_path, tmpdir, library_dirs or [], include_dirs or [], libraries or [], ccflags or [])
        with open(so, "rb") as f:
            cache_path = cache.put(f.read(), f"{name}{suffix}", binary=True)

    return _load_module_from_path(name, cache_path)
