#!/usr/bin/env bash

set -euxo pipefail

: "${LLVM_COMMIT_HASH:?LLVM_COMMIT_HASH must be set}"

export LLVM_PROJECT_PATH=/tmp/llvm-project
export LLVM_BUILD_PATH="${LLVM_PROJECT_PATH}/build"
export LLVM_INSTALL_PATH=/opt/llvm
export LLVM_BUILD_TYPE=Release
export TRITON_BUILD_WITH_CLANG_LLD=1

# build-llvm-project.sh would otherwise clone the full history.
git clone --depth 1 --revision="${LLVM_COMMIT_HASH}" https://github.com/llvm/llvm-project "${LLVM_PROJECT_PATH}"

# Keep the default targets: libtriton links the NVPTX and AMDGPU code generators.
build-llvm-project.sh
ninja -C "${LLVM_BUILD_PATH}" install-distribution-stripped

for tool in FileCheck clang++ llc llvm-config mlir-tblgen not opt; do
    test -x "${LLVM_INSTALL_PATH}/bin/${tool}"
done
test -d "${LLVM_INSTALL_PATH}/lib/cmake/mlir"
test -d "${LLVM_INSTALL_PATH}/lib/cmake/lld"
targets="$("${LLVM_INSTALL_PATH}/bin/llvm-config" --targets-built)"
for target in SystemZ NVPTX AMDGPU; do
    [[ " ${targets} " == *" ${target} "* ]]
done

# Ubuntu packages no libomp for s390x. A standalone runtimes build installs
# headers to CMAKE_INSTALL_INCLUDEDIR, so point it at clang's resource directory.
OPENMP_BUILD_PATH="${LLVM_PROJECT_PATH}/build-openmp"
CLANG_RESOURCE_DIR="$("${LLVM_INSTALL_PATH}/bin/clang" -print-resource-dir)"
OPENMP_LIB_PATH="${LLVM_INSTALL_PATH}/lib/$("${LLVM_INSTALL_PATH}/bin/clang" -print-target-triple)"
cmake -G Ninja -S "${LLVM_PROJECT_PATH}/runtimes" -B "${OPENMP_BUILD_PATH}" \
    -DCMAKE_BUILD_TYPE="${LLVM_BUILD_TYPE}" \
    -DCMAKE_C_COMPILER="${LLVM_BUILD_PATH}/bin/clang" \
    -DCMAKE_CXX_COMPILER="${LLVM_BUILD_PATH}/bin/clang++" \
    -DCMAKE_INSTALL_PREFIX="${LLVM_INSTALL_PATH}" \
    -DCMAKE_INSTALL_INCLUDEDIR="${CLANG_RESOURCE_DIR}/include" \
    -DLLVM_BINARY_DIR="${LLVM_BUILD_PATH}" \
    -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=ON \
    -DLLVM_ENABLE_RUNTIMES=openmp \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLIBOMP_INSTALL_ALIASES=OFF \
    -DLIBOMP_OMPD_SUPPORT=OFF
ninja -C "${OPENMP_BUILD_PATH}" install
test -f "${OPENMP_LIB_PATH}/libomp.so"

cat > /tmp/omp-check.c <<'EOF'
#include <omp.h>
#include <stdio.h>
int main(void) {
    int threads = 0;
#pragma omp parallel
#pragma omp atomic
    threads++;
    printf("libomp threads: %d\n", threads);
    return threads == 4 ? 0 : 1;
}
EOF
"${LLVM_INSTALL_PATH}/bin/clang" -fopenmp /tmp/omp-check.c -o /tmp/omp-check
OMP_NUM_THREADS=4 LD_LIBRARY_PATH="${OPENMP_LIB_PATH}" /tmp/omp-check
LD_LIBRARY_PATH="${OPENMP_LIB_PATH}" ldd /tmp/omp-check | grep -q "${OPENMP_LIB_PATH}/libomp.so"
rm -f /tmp/omp-check /tmp/omp-check.c

# CI compares this with cmake/llvm-info.json to catch a stale image.
echo "${LLVM_COMMIT_HASH}" > "${LLVM_INSTALL_PATH}/llvm-commit"

rm -rf "${LLVM_PROJECT_PATH}"
