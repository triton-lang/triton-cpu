"""Skip or xfail the tests in known-failures.txt.

Not in .github/s390x-ci, whose build.py would shadow PyPA build on PYTHONPATH.
Marks rather than deselects, so python/test/conftest.py sees every allowlisted test.
"""

from pathlib import Path

import pytest

KNOWN_FAILURES = Path(__file__).parents[1] / "known-failures.txt"
MODES = ("skip", "xfail")


def _load(path):
    entries = {}
    default = reason = "known s390x failure"
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            reason = default
            continue
        if line.startswith("#"):
            reason = line.lstrip("#").strip() or reason
            continue
        mode, _, node_id = line.partition(" ")
        node_id = node_id.strip()
        if mode not in MODES or not node_id:
            raise pytest.UsageError(f"{path}:{lineno}: expected 'skip|xfail <node id>', got {raw!r}")
        if node_id in entries:
            raise pytest.UsageError(f"{path}:{lineno}: duplicate entry {node_id}")
        entries[node_id] = (mode, reason)
    return entries


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(items):
    entries = _load(KNOWN_FAILURES)
    for item in items:
        key = item.nodeid if item.nodeid in entries else item.nodeid.split("[", 1)[0]
        if key not in entries:
            continue
        mode, reason = entries[key]
        if mode == "skip":
            item.add_marker(pytest.mark.skip(reason=f"s390x: {reason}"))
        else:
            item.add_marker(pytest.mark.xfail(reason=f"s390x: {reason}", strict=False))
