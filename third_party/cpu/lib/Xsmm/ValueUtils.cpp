//===- ValueUtils.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ValueUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace utils {

// Returns true if the value is a constant float or integer.
bool isValConstZero(Value val) {
  return matchPattern(val, m_AnyZeroFloat()) || matchPattern(val, m_Zero());
}

// Returns true if the attribute represent "all zeros"
static bool isZeroAttr(Attribute attribute) {
  return TypeSwitch<Attribute, bool>(attribute)
      .Case<FloatAttr>([](auto attr) { return attr.getValueAsDouble() == 0.0; })
      .Case<IntegerAttr>([](auto attr) { return attr.getInt() == 0; })
      .Case<DenseElementsAttr>([](auto attr) {
        if (!attr.getElementType().isIntOrFloat())
          return false;
        if (!attr.isSplat())
          return false;
        auto splat = attr.template getSplatValue<Attribute>();
        return isZeroAttr(splat);
      })
      .Default([](auto attr) { return false; });
}

// Prototypes
bool isZeroOp(Operation *);

// Returns true if the value represents a zero filled tensor.
// Recurse into isZeroOp for defining ops if not immediately obvious
// Looks past linalg generic's argument (which don't have defining ops)
bool isZeroTensor(Value val) {
  if (!val)
    return false;
  if (isValConstZero(val))
    return true;

  Operation *defOp = nullptr;

  // Block arguments don't have a defining op, but they do have an op arg
  if (auto arg = dyn_cast<BlockArgument>(val)) {
    // We need to find the argument to the linalg on the same order as this one
    auto *linalgOp = arg.getParentRegion()->getParentOp();
    if (!isa<linalg::GenericOp>(linalgOp))
      return false;
    auto index = arg.getArgNumber();
    auto linalgArg = linalgOp->getOperand(index);
    defOp = linalgArg.getDefiningOp();
  } else {
    defOp = val.getDefiningOp();
  }
  return isZeroOp(defOp);
}

// Returns true if the operation represents a zero filled tensor
// Recurses into isZeroTensor for operands and isZeroAttr for attributes
bool isZeroOp(Operation *defOp) {
  if (!defOp)
    return false;

  return TypeSwitch<Operation *, bool>(defOp)
      .Case<arith::ConstantOp>([&](auto op) {
        // Dense attributes don't match APFloat.isZero()
        auto attr = op.getValue();
        return isZeroAttr(attr);
      })
      .Case<linalg::FillOp, linalg::CopyOp>([&](auto op) {
        if (op.getInputs().size() != 1)
          return false;
        return isZeroTensor(op.getInputs()[0]);
      })
      .Case<memref::CopyOp, memref::SubViewOp, tensor::CastOp,
            tensor::ExtractSliceOp>(
          [&](auto op) { return isZeroTensor(op.getSource()); })
      .Case<memref::GetGlobalOp>([&](auto op) {
        auto name = op.getName();
        auto module = defOp->getParentOfType<ModuleOp>();
        auto global = module.lookupSymbol<memref::GlobalOp>(name);
        auto attr = global.getInitialValueAttr();
        return isZeroAttr(attr);
      })
      .Default([&](Operation *op) { return false; });
}

FailureOr<SmallVector<int64_t>> getStaticStrides(Value value) {
  auto valueType = value.getType();
  if (!isa<MemRefType>(valueType))
    return failure();
  auto memrefType = cast<MemRefType>(valueType);
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memrefType, strides, offset))) {
    return failure();
  }
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      })) {
    return failure();
  }
  return strides;
}

std::pair<Value, Value> getPtrAndOffset(OpBuilder &builder, Value operand,
                                        Location loc) {
  auto memrefType = dyn_cast<MemRefType>(operand.getType());
  assert(memrefType && "Expect a memref value");
  MemRefType baseMemrefType = MemRefType::get({}, memrefType.getElementType());
  Type basePtrType = builder.getIndexType();
  Type offsetType = builder.getIndexType();
  SmallVector<Type> sizesTypes(memrefType.getRank(), offsetType);
  SmallVector<Type> stridesTypes(memrefType.getRank(), offsetType);
  auto meta = builder.create<memref::ExtractStridedMetadataOp>(
      loc, baseMemrefType, offsetType, sizesTypes, stridesTypes, operand);
  Value alignedPointerAsIndex =
      builder.create<memref::ExtractAlignedPointerAsIndexOp>(loc, basePtrType,
                                                             operand);
  Value alignedPointerAsI64 = builder.create<arith::IndexCastOp>(
      loc, builder.getIntegerType(64), alignedPointerAsIndex);
  // TODO: non-POD will require an LLVMTypeConverter.
  Value alignedPointer = builder.create<LLVM::IntToPtrOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()),
      alignedPointerAsI64);
  Value offset = meta.getOffset();
  return std::make_pair(alignedPointer, offset);
}

} // namespace utils
} // namespace mlir
