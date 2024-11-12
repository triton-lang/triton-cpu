#ifndef TRITONCPU_XSMM_PASSES
#define TRITONCPU_XSMM_PASSES

include "mlir/Pass/PassBase.td"

def ConvertVectorToXsmm : Pass<"triton-cpu-convert-vector-to-xsmm", "mlir::ModuleOp"> {
 let summary = "Convert vector to xsmm";
 let description = [{
   Convert vector operations to XSMM operations.
 }];
 let dependentDialects = ["arith::ArithDialect",
                          "func::FuncDialect",
                          "memref::MemRefDialect",
                          "vector::VectorDialect",
			                    "LLVM::LLVMDialect"];
}

def ConvertTritonToXsmm : Pass<"triton-cpu-convert-triton-to-xsmm", "mlir::ModuleOp"> {
 let summary = "Convert triton to xsmm";
 let description = [{
   Convert triton operations to XSMM operations.
 }];
 let dependentDialects = ["arith::ArithDialect",
                          "func::FuncDialect",
                          "memref::MemRefDialect",
                          "triton::cpu::TritonCPUDialect",
			                    "LLVM::LLVMDialect"];
}

def LoopToBrgemmXsmm : Pass<"triton-cpu-loop-to-brgemm-xsmm", "mlir::ModuleOp"> {
 let summary = "Redution loop GEMM to BRGEMM";
 let description = [{
   Collapse reduction loop over GEMM to XSMM BRGEMM kernel.
 }];
 let dependentDialects = ["arith::ArithDialect",
                          "func::FuncDialect",
                          "memref::MemRefDialect",
                          "triton::cpu::TritonCPUDialect",
			                    "LLVM::LLVMDialect"];
}

#endif
