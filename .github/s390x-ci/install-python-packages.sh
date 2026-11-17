#!/usr/bin/env bash

set -euxo pipefail

: "${PYTORCH_VERSION:?PYTORCH_VERSION must be set}"
: "${PYTORCH_INDEX_URL:?PYTORCH_INDEX_URL must be set}"

requirements=(
    -r /tmp/requirements/requirements.txt
    -r /tmp/requirements/test-requirements.txt
    -r /tmp/requirements/tutorials-requirements.txt
)

python3 -m pip install --no-cache-dir --no-index --find-links /wheelhouse "${requirements[@]}"

python3 -m pip install --no-cache-dir --index-url "${PYTORCH_INDEX_URL}" "torch==${PYTORCH_VERSION}"

# CI-only: lets the s390x jobs stop one slow compile from using up the job.
python3 -m pip install --no-cache-dir "pytest-timeout==2.4.0"

python3 - <<'PY'
import importlib.metadata
import os

import hatchet
import matplotlib
import numpy
import pandas
import pytest_timeout
import scipy
import torch

expected = os.environ["PYTORCH_VERSION"]
actual = importlib.metadata.version("torch")
if actual != expected:
    raise SystemExit(f"installed torch=={actual}, expected torch=={expected}")
for module in (torch, numpy, scipy, pandas, matplotlib, hatchet):
    print(f"{module.__name__}: {getattr(module, '__version__', 'unknown')}")
PY

python3 -m pip uninstall -y triton pytorch-triton || true
python3 -m pip freeze > /opt/python-packages.txt

# A packaged Triton would shadow the one CI builds.
! python3 -c 'import triton'
