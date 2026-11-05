#ifndef TRITONCPU_DIALECT
#define TRITONCPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonCPU_Dialect : Dialect {
  let name = "triton_cpu";

  let cppNamespace = "::mlir::triton::cpu";

  // Preserve the existing attr-dict syntax for inherent attributes.
  let useStrictPropertiesInAssemblyFormat = 0;

  let hasOperationAttrVerify = 1;

  let description = [{
    Triton CPU Dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "tensor::TensorDialect",
    "mlir::memref::MemRefDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif
