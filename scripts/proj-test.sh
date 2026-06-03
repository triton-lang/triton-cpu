#!/usr/bin/env bash

set -e
NPROC=64

echo "===================================== Run python unit tests"

python -m pytest -s -n $NPROC --device cpu -m cpu\
    python/test/unit/language/test_core.py \
    python/test/unit/language/test_tensor_descriptor.py
python -m pytest -s -n $NPROC --device cpu \
    python/test/unit/cpu/test_math.py \
    python/test/unit/cpu/test_opt.py \
    python/test/unit/language/test_annotations.py \
    python/test/unit/language/test_block_pointer.py \
    python/test/unit/language/test_compile_errors.py \
    python/test/unit/language/test_conversions.py \
    python/test/unit/language/test_decorator.py \
    python/test/unit/language/test_pipeliner.py \
    python/test/unit/language/test_random.py \
    python/test/unit/language/test_standard.py \
    python/test/unit/runtime/test_autotuner.py \
    python/test/unit/runtime/test_bindings.py \
    python/test/unit/runtime/test_cache.py \
    python/test/unit/runtime/test_driver.py \
    python/test/unit/runtime/test_launch.py \
    python/test/unit/runtime/test_subproc.py \
    python/test/unit/test_debug_dump.py

echo "===================================== Run lit tests"

LIT_TEST_DIR="build/$(ls build | grep -i cmake)/test"
if [ ! -d "${LIT_TEST_DIR}" ]; then
    echo "Could not find '${LIT_TEST_DIR}'" ; exit -1
fi
lit -v "${LIT_TEST_DIR}/TritonCPU"

echo "===================================== Run dot tests"

export OMP_NUM_THREADS=$NPROC

DTYPE=bfloat16 python python/tutorials/cpu-blocked-matmul.py
DTYPE=float32 python python/tutorials/cpu-blocked-matmul.py
DTYPE=float16 python python/tutorials/cpu-blocked-matmul.py

python python/tutorials/cpu-sfc-matmul.py --dtype bfloat16 --target amx \
  --bench -M 512 1024 2048 -N 512 1024 2048 -K 512 1024 2048
python python/tutorials/cpu-sfc-matmul.py --dtype bfloat16 --target avx512 \
  --bench -M 512 1024 2048 -N 512 1024 2048 -K 512 1024 2048

echo "===================================== Run dot tests (XSMM)"

export TRITON_CPU_UKERNELS_LIB=XSMM OMP_STACKSIZE=64M

python -m pytest -s -n $NPROC --device cpu \
    python/test/unit/language/test_core.py -m cpu
DTYPE=bfloat16 python python/tutorials/cpu-blocked-matmul.py
DTYPE=float32 python python/tutorials/cpu-blocked-matmul.py
DTYPE=float16 python python/tutorials/cpu-blocked-matmul.py

echo "✅ Success"
