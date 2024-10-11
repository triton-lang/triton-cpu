#ifndef TRITONCPU_XSMM_PASSES
#define TRITONCPU_XSMM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertVectorToXsmm : Pass<"triton-cpu-convert-vector-to-xsmm", "mlir::ModuleOp"> {
 let summary = "Convert vector to xsmm";
 let description = [{
   Convert vector operations to XSMM operations.
 }];
 let dependentDialects = ["func::FuncDialect",
                          "memref::MemRefDialect",
                          "vector::VectorDialect",
			                    "LLVM::LLVMDialect"];
}

#endif
