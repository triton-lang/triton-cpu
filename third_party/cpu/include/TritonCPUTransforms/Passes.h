#ifndef TritonCPUTransforms_CONVERSION_PASSES_H
#define TritonCPUTransforms_CONVERSION_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {
namespace cpu {

#define GEN_PASS_DECL
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertUnsupportedOps();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertUnsupportedOps(bool promoteBf16ToFp32,
                            bool convertMixedPrecisionMatmul,
                            bool promoteLibMathToFp32);
std::unique_ptr<OperationPass<ModuleOp>> createDecomposeFpConversions();
std::unique_ptr<OperationPass<ModuleOp>>
createDecomposeFpConversions(bool decomposeBf16Conversions,
                             bool decomposeFp8Conversions);
std::unique_ptr<OperationPass<ModuleOp>> createOptimizeMasks();

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotProduct();

#define GEN_PASS_REGISTRATION
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"

} // namespace cpu
} // namespace triton

} // namespace mlir

#endif
