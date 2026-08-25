#ifndef TRITON_CONVERSION_TRITONCPU_TO_LLVM_UTILITY_H
#define TRITON_CONVERSION_TRITONCPU_TO_LLVM_UTILITY_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::triton::cpu {

inline constexpr unsigned kNumProgramDims = 3;
inline constexpr unsigned kNumProgramContextArgs = 2 * kNumProgramDims;

Value getProgramId(mlir::FunctionOpInterface funcOp, int axis);
Value getNumPrograms(mlir::FunctionOpInterface funcOp, int axis);

} // namespace mlir::triton::cpu

#endif
