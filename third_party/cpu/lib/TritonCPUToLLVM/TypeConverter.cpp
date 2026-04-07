#include "TypeConverter.h"

#include "mlir/Dialect/X86/X86Dialect.h"

using namespace mlir;
using namespace mlir::triton;

TritonCPUToLLVMTypeConverter::TritonCPUToLLVMTypeConverter(
    MLIRContext *ctx, LowerToLLVMOptions &option,
    const DataLayoutAnalysis *analysis)
    : LLVMTypeConverter(ctx, option, analysis) {
  addConversion([&](triton::PointerType type) -> std::optional<Type> {
    return convertTritonPointerType(type);
  });
  addConversion([this](RankedTensorType type) -> std::optional<Type> {
    return convertTritonTensorType(type);
  });
  addConversion([&](x86::amx::TileType type) {
    return LLVM::LLVMX86AMXType::get(type.getContext());
  });
  addConversion([&](TensorDescType type) -> std::optional<Type> {
    return convertTritonTensorDescType(type);
  });
}

Type TritonCPUToLLVMTypeConverter::convertTritonPointerType(
    triton::PointerType type) {
  return LLVM::LLVMPointerType::get(type.getContext());
}

Type TritonCPUToLLVMTypeConverter::convertTritonTensorType(
    RankedTensorType type) {
  if (isa<PointerType>(type.getElementType()))
    return VectorType::get(type.getShape(),
                           IntegerType::get(type.getContext(), 64));
  llvm_unreachable("No tensor types are expected in TTCIR");
}

Type TritonCPUToLLVMTypeConverter::convertTritonTensorDescType(
    TensorDescType type) {
  auto ctx = type.getContext();
  auto rank = type.getBlockType().getRank();
  auto i64Ty = IntegerType::get(ctx, 64);

  // Mimic a MemRef descriptor.
  SmallVector<Type, 5> types;
  types.push_back(LLVM::LLVMPointerType::get(ctx)); // allocated ptr (unused)
  types.push_back(LLVM::LLVMPointerType::get(ctx)); // aligned ptr (= base ptr)
  types.push_back(i64Ty);                           // offset (unused)
  types.push_back(
      LLVM::LLVMArrayType::get(ctx, i64Ty, rank)); // sizes (= shape)
  types.push_back(LLVM::LLVMArrayType::get(ctx, i64Ty, rank)); // strides
  return LLVM::LLVMStructType::getLiteral(ctx, types);
}
