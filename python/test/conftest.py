import importlib.util
import os
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "gsan_fine_granularity(reason): requires sub-16-byte GSan tracking")
    # If pytest-sugar is not active, enable instafail
    if not config.pluginmanager.hasplugin("sugar"):
        config.option.instafail = True


# Exercise synchronization/ordering at the default granularity. Tests whose
# behavior depends on cell size explicitly parametrize this fixture indirectly.
@pytest.fixture(params=[4], ids=lambda granularity: f"granularity-{granularity}")
def shadow_granularity(request):
    marker = request.node.get_closest_marker("gsan_fine_granularity")
    if request.param == 16 and marker is not None:
        pytest.skip(marker.args[0])
    return request.param


@pytest.fixture
def with_allocator():
    import triton
    from triton.runtime._allocation import NullAllocator
    from triton._internal_testing import default_alloc_fn

    triton.set_allocator(default_alloc_fn)
    try:
        yield
    finally:
        triton.set_allocator(NullAllocator())


def pytest_collection_modifyitems(config, items):
    """On the CPU backend, run only the allowlisted subset of test_core.py.

    Selection data lives in unit/cpu/test_core_cpu_allowlist.py so the shared
    test_core.py needs no CPU-specific marks and stays identical to upstream.
    """
    if os.environ.get("TRITON_INTERPRET") == "1":
        return
    try:
        device = config.getoption("device")
    except ValueError:
        device = None
    # Some CI jobs select the CPU backend implicitly (GPU-less runners) and
    # only pass --device cpu, while others set TRITON_DEFAULT_BACKEND=cpu.
    if os.environ.get("TRITON_DEFAULT_BACKEND") != "cpu" and device != "cpu":
        return
    allowlist_path = Path(__file__).parent / "unit" / "cpu" / "test_core_cpu_allowlist.py"
    spec = importlib.util.spec_from_file_location("test_core_cpu_allowlist", allowlist_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    allowlisted = module.CPU_TEST_CORE_ALLOWLIST

    kept, deselected, seen = [], [], set()
    saw_test_core = False
    for item in items:
        parts = item.nodeid.split("::")
        if not parts[0].endswith("test_core.py"):
            kept.append(item)
            continue
        saw_test_core = True
        func = parts[1].split("[")[0] if len(parts) > 1 else ""
        if func in allowlisted:
            seen.add(func)
            kept.append(item)
        else:
            deselected.append(item)
    if saw_test_core:
        missing = sorted(set(allowlisted) - seen)
        if missing:
            raise ValueError(f"CPU allowlist entries matched no collected test in test_core.py: {missing}")
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = kept
