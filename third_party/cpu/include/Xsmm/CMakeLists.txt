set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls --name TritonCPUXsmm)
add_public_tablegen_target(TritonCPUXsmmPassIncGen)

set(LLVM_TARGET_DEFINITIONS XsmmEnum.td)
mlir_tablegen(XsmmEnum.h.inc -gen-enum-decls)
mlir_tablegen(XsmmEnum.cpp.inc -gen-enum-defs)
add_public_tablegen_target(TritonCPUXsmmAttrDefIncGen)
