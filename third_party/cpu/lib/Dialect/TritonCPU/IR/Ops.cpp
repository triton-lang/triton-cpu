#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "mlir/IR/Builders.h"

#define GET_OP_CLASSES
#include "cpu/include/Dialect/TritonCPU/IR/Ops.cpp.inc"

// enum attribute definitions
#include "cpu/include/Dialect/TritonCPU/IR/OpsEnums.cpp.inc"

namespace mlir::triton::cpu {

LogicalResult DescriptorLoadOp::verify() {
  auto baseTy = cast<MemRefType>(getBase().getType());
  auto resultTy = cast<VectorType>(getType());
  if (getIndices().size() != baseTy.getRank())
    return emitOpError("expects one index per memref dimension");
  if (resultTy.isScalable() || resultTy.getRank() > baseTy.getRank())
    return emitOpError(
        "expects a fixed vector with rank at most the memref rank");
  if (resultTy.getElementType() != baseTy.getElementType() ||
      getPadding().getType() != resultTy.getElementType())
    return emitOpError(
        "expects matching memref, vector and padding element types");
  return success();
}

LogicalResult DescriptorStoreOp::verify() {
  auto baseTy = cast<MemRefType>(getBase().getType());
  auto valueTy = cast<VectorType>(getValue().getType());
  if (getIndices().size() != baseTy.getRank())
    return emitOpError("expects one index per memref dimension");
  if (valueTy.isScalable() || valueTy.getRank() != baseTy.getRank())
    return emitOpError(
        "expects a fixed vector with the same rank as the memref");
  if (valueTy.getElementType() != baseTy.getElementType())
    return emitOpError("expects matching memref and vector element types");
  return success();
}

LogicalResult PrintOp::verify() {
  if (getOperands().size() > 1)
    return emitOpError("expects at most one operand");
  return success();
}

void ExternElementwiseOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  if (getPure())
    return;
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(),
                       SideEffects::DefaultResource::get());
}

LogicalResult
DotOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        PropertyRef properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  // type is the same as the accumulator
  auto accTy = cast<VectorType>(operands[2].getType());
  inferredReturnTypes.push_back(accTy);
  return success();
}

} // namespace mlir::triton::cpu
