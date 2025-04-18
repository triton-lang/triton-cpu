#ifndef TRITONCPU_CONVERSION_TRITONCPUTOLLVM_PASSES_H
#define TRITONCPU_CONVERSION_TRITONCPUTOLLVM_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace cpu {

enum class VecLib {
  Mvec,
  Sleef,
};

#define GEN_PASS_DECL
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createFuncOpToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createMemoryOpToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createGetProgramIdOpToLLVMPass();
std::unique_ptr<OperationPass<triton::FuncOp>> createLowerMultiReductionPass();
std::unique_ptr<OperationPass<ModuleOp>> createAtomicOpsToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createDebugOpsToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createUkernelOpsToOneDNNLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createUkernelOpsToXSMMLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createMathToVecLibPass(VecLib lib = VecLib::Sleef,
                       std::set<std::string> cpu_features = {});

#define GEN_PASS_REGISTRATION
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"

} // namespace cpu
} // namespace triton

} // namespace mlir

#endif
