#!/usr/bin/env bash

set -euxo pipefail

# PyPI has no s390x wheels for these. Build in dependency order with find-links so
# numpy is built once; test-requirements.txt pins pandas, the tutorials list does not.
mkdir -p /wheelhouse
export PIP_FIND_LINKS=/wheelhouse
wheel=(python3 -m pip wheel --no-cache-dir --wheel-dir /wheelhouse -c /tmp/requirements/test-requirements.txt)
"${wheel[@]}" numpy
"${wheel[@]}" markupsafe msgpack pillow
"${wheel[@]}" pandas scipy matplotlib
"${wheel[@]}" llnl-hatchet

"${wheel[@]}" \
    -r /tmp/requirements/requirements.txt \
    -r /tmp/requirements/test-requirements.txt \
    -r /tmp/requirements/tutorials-requirements.txt
