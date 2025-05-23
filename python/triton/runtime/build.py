import contextlib
import sys
import platform
import io
import sysconfig
import os
import shutil
import subprocess


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


def _build(name, src, srcdir, library_dirs, include_dirs, libraries):
    suffix = sysconfig.get_config_var('EXT_SUFFIX')
    system = platform.system()
    machine = platform.machine()
    so = os.path.join(srcdir, '{name}{suffix}'.format(name=name, suffix=suffix))
    # try to avoid setuptools if possible
    cc = os.environ.get("CC")
    if cc is None:
        # TODO: support more things here.
        clang = shutil.which("clang")
        gcc = shutil.which("gcc")
        cc = gcc if gcc is not None else clang
        if cc is None:
            raise RuntimeError("Failed to find C compiler. Please specify via CC environment variable.")
    # This function was renamed and made public in Python 3.10
    if hasattr(sysconfig, 'get_default_scheme'):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == 'posix_local':
        scheme = 'posix_prefix'
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    custom_backend_dirs = set(os.getenv(var) for var in ('TRITON_CUDACRT_PATH', 'TRITON_CUDART_PATH'))
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
    cc_cmd += [f'-l{lib}' for lib in libraries]
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
    ret = subprocess.check_call(cc_cmd)
    if ret != 0:
        raise RuntimeError("Failed to compile so.")
    return so
