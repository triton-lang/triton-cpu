#!/usr/bin/env bash

# saving path
HERE=$PWD

echo "===================================== Setting Up Python venv"
source .venv/bin/activate

echo "===================================== Config LLVM"
LLVM_DIR="llvm-project-triton-cpu"
if [ ! -d ${HERE}/../${LLVM_DIR} ]; then
  echo "Failed to find LLVM: ${LLVM_DIR}"
  exit 1
fi
export PATH=~/${LLVM_DIR}/build/bin:$PATH && export LD_LIBRARY_PATH=~/${LLVM_DIR}/build/lib:$LD_LIBRARY_PATH

echo "===================================== Config LIBXSMM"
LIBXSMM_DIR="libxsmm-triton-cpu"
LIBXSMM_INSTALL_DIR="${LIBXSMM_DIR}-install"
if [ ! -d ${HERE}/../${LIBXSMM_INSTALL_DIR} ]; then
  echo "Failed to find LIBXSMM install: ${LIBXSMM_INSTALL_DIR}"
  exit 1
fi

echo "===================================== Config env vars"
export LLVM_BUILD_DIR=$PWD/../${LLVM_DIR}/build
export TRITON_BUILD_WITH_CCACHE=false
export TRITON_BUILD_WITH_CLANG_LLD=true
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR
export XSMM_INSTALL=$PWD/../${LIBXSMM_INSTALL_DIR}
export XSMM_LIBRARY_DIRS=$XSMM_INSTALL/lib
export XSMM_INCLUDE_DIRS=$XSMM_INSTALL/include
