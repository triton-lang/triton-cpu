#!/usr/bin/env bash

# saving path
HERE=$PWD

echo "===================================== Configuring Checkout"
git submodule init
git submodule update

echo "===================================== Setting Up Python venv"
uv venv --allow-existing
source .venv/bin/activate
uv pip install -r ./python/requirements.txt
uv pip install -r ./python/test-requirements.txt
uv pip install torch matplotlib

echo "===================================== Building LLVM"
LLVM_DIR="llvm-project-triton-cpu"
if [ ! -d ${HERE}/../${LLVM_DIR} ]; then
  pushd ./../
  git clone https://github.com/llvm/llvm-project.git ${LLVM_DIR}
  cd ${LLVM_DIR}
  git checkout `cat ${HERE}/cmake/llvm-hash.txt`
  mkdir -p build
  pushd build
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=True -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_USE_LINKER=lld -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld;clang" -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" ../llvm
  ninja
  popd
  popd
else
  echo "LLVM already built, skipping."
fi
export PATH=~/${LLVM_DIR}/build/bin:$PATH && export LD_LIBRARY_PATH=~/${LLVM_DIR}/build/lib:$LD_LIBRARY_PATH

echo "===================================== Building LIBXSMM"
LIBXSMM_DIR="libxsmm-triton-cpu"
LIBXSMM_INSTALL_DIR="${LIBXSMM_DIR}-install"
if [ ! -d ${HERE}/../${LIBXSMM_INSTALL_DIR} ]; then
  pushd ./../
  git clone https://github.com/libxsmm/libxsmm.git ${LIBXSMM_DIR}
  cd ${LIBXSMM_DIR}
  git checkout 63251ac815e88593d06f36b3e07180f117abae37
  BLAS=0 make PREFIX=../${LIBXSMM_INSTALL_DIR} install
  popd
else
  echo "LIBXSMM already built, skipping."
fi

export LLVM_BUILD_DIR=$PWD/../${LLVM_DIR}/build
export TRITON_BUILD_WITH_CCACHE=false
export TRITON_BUILD_WITH_CLANG_LLD=true
export LLVM_INCLUDE_DIRS=$LLVM_BUILD_DIR/include
export LLVM_LIBRARY_DIR=$LLVM_BUILD_DIR/lib
export LLVM_SYSPATH=$LLVM_BUILD_DIR
export XSMM_INSTALL=$PWD/../${LIBXSMM_INSTALL_DIR}
export XSMM_LIBRARY_DIRS=$XSMM_INSTALL/lib
export XSMM_INCLUDE_DIRS=$XSMM_INSTALL/include

echo "===================================== Build"
uv pip install -vvv -e .
if [ $? != 0 ]; then
  exit 1
fi

echo "===================================== CMake Tests"
ctest --test-dir build/cmake*
if [ $? != 0 ]; then
  exit 1
fi
