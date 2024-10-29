//===- XsmmUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "XsmmUtils.h"
#include "ValueUtils.h"
#include "VnniUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Compiler.h"

#include <functional>
#include <optional>

#define DEBUG_TYPE "xsmm-utils"

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace xsmm {
namespace utils {

// Callable object to verify if `operand` has static shape.
struct HasStaticShape {
  HasStaticShape() = default;
  HasStaticShape(SmallVectorImpl<int64_t> *shape) : shape(shape){};

  bool operator()(Value operand, Operation *op) const {
    auto operandType = operand.getType();
    if (auto shapedType = dyn_cast_or_null<ShapedType>(operandType)) {
      if (!shapedType.hasStaticShape())
        return false;
      if (shape) {
        for (int64_t shapeOnDim : shapedType.getShape())
          shape->push_back(shapeOnDim);
      }
    }
    return true;
  }
  SmallVectorImpl<int64_t> *shape = nullptr;
};

// Callable object to verify if `operand` has static strides.
// If `operand` is a tensor type or a scalar, return true.
struct HasStaticStrides {
  HasStaticStrides() = default;
  HasStaticStrides(SmallVector<int64_t> *strides) : strides(strides){};

  bool operator()(Value operand, Operation *op) const {
    auto operandType = operand.getType();
    SmallVector<int64_t> strides;
    if (auto memRefType = dyn_cast_or_null<MemRefType>(operandType)) {
      int64_t offset;
      if (failed(getStridesAndOffset(memRefType, strides, offset)))
        return false;
      if (llvm::any_of(strides, [](int64_t stride) {
            return stride == ShapedType::kDynamic;
          })) {
        return false;
      }
      if (this->strides)
        this->strides->append(strides.begin(), strides.end());
    }
    return true;
  }
  SmallVectorImpl<int64_t> *strides = nullptr;
};

// Structural matcher.
static FailureOr<ContractionDimensions>
checkStructure(Operation *contractOp, SmallVector<Value> &inputs,
               SmallVector<Value> &outputs, ArrayRef<AffineMap> indexingMap) {
  if (!HasStaticShape()(inputs[0], inputs[0].getDefiningOp()) ||
      !HasStaticShape()(inputs[1], inputs[1].getDefiningOp()) ||
      !HasStaticShape()(inputs[2], inputs[2].getDefiningOp()) ||
      (outputs[0] != nullptr &&
       !HasStaticShape()(outputs[0], outputs[0].getDefiningOp())) ||
      !HasStaticStrides()(inputs[0], inputs[0].getDefiningOp()) ||
      !HasStaticStrides()(inputs[1], inputs[1].getDefiningOp()) ||
      !HasStaticStrides()(inputs[2], inputs[2].getDefiningOp()) ||
      (outputs[0] != nullptr &&
       !HasStaticStrides()(outputs[0], outputs[0].getDefiningOp()))) {
    return failure();
  }
  return inferContractionDims(indexingMap);
}

// Return the position of `dim` in the codomain of `operand`.
std::optional<unsigned> getPosInCodomain(unsigned dim, Value operand,
                                         Operation *contractOp, AffineMap map) {
  return map.getResultPosition(getAffineDimExpr(dim, contractOp->getContext()));
}

static SmallVector<int64_t, 4>
createFlatListOfOperandStaticDims(Operation *contractOp) {
  SmallVector<int64_t, 4> res;
  for (OpOperand &opOperand : contractOp->getOpOperands())
    llvm::append_range(
        res, dyn_cast<ShapedType>(opOperand.get().getType()).getShape());
  return res;
}

static SmallVector<int64_t, 4>
computeStaticLoopSizes(Operation *contractOp, ArrayRef<AffineMap> maps) {
  AffineMap map = concatAffineMaps(maps);
  unsigned numDims = map.getNumDims(), numRes = map.getNumResults();
  SmallVector<int64_t, 4> allShapeSizes =
      createFlatListOfOperandStaticDims(contractOp);
  SmallVector<int64_t, 4> res(numDims, 0);
  for (unsigned idx = 0; idx < numRes; ++idx) {
    auto result = map.getResult(idx);
    if (auto d = dyn_cast<AffineDimExpr>(result))
      res[d.getPosition()] = allShapeSizes[idx];
  }
  return res;
}

static FailureOr<SmallVector<int64_t>>
getVNNIStaticStrides(MemRefType valueType) {
  SmallVector<int64_t> strides;
  int64_t offset;
  SmallVector<int64_t> shape;
  for (size_t i = 0; i < valueType.getShape().size(); i++) {
    shape.push_back(valueType.getShape()[i]);
  }
  auto temp = shape[shape.size() - 1];
  shape[shape.size() - 1] = shape[shape.size() - 2];
  shape[shape.size() - 2] = temp;
  auto memrefType = MemRefType::get(shape, valueType.getElementType());
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

// Access matcher.
FailureOr<xsmm::BrgemmInfo>
checkAccess(PatternRewriter &rewriter, Operation *contractOp, unsigned m,
            unsigned n, SmallVector<unsigned, 2> kVector,
            std::optional<unsigned> batchPos, SmallVector<Value> inputs,
            ArrayRef<AffineMap> indexingMap) {
  MLIRContext *ctx = contractOp->getContext();

  Value operandA = inputs[0];
  Value operandB = inputs[1];
  Value operandC = inputs[2];

  // Assume that if the last two dimensions are reductions, it is VNNI format.
  // TODO: Add proper checks for VNNI.
  auto dims = inferContractionDims(indexingMap);
  auto numDims = indexingMap[0].getNumDims();
  bool isVnni =
      dims->k.back() == (numDims - 1) && dims->k.end()[-2] == (numDims - 2);

  unsigned k;
  if (*xsmm::utils::getPosInCodomain(kVector[0],
                                     contractOp->getOpOperand(1).get(),
                                     contractOp, indexingMap[1]) <
          *xsmm::utils::getPosInCodomain(n, contractOp->getOpOperand(1).get(),
                                         contractOp, indexingMap[1]) ||
      kVector.size() == 1) {
    k = kVector[0];
  } else if (kVector.size() > 1) {
    k = kVector[1];
  }

  auto checkStridesAndGetLda = [&](unsigned minorDim, unsigned majorDim,
                                   Value operand, AffineMap map,
                                   int operandIndex) -> FailureOr<int64_t> {
    auto minorDimPosInCodomain =
        xsmm::utils::getPosInCodomain(minorDim, operand, contractOp, map);
    auto majorDimPosInCodomain =
        xsmm::utils::getPosInCodomain(majorDim, operand, contractOp, map);
    if (!minorDimPosInCodomain || !majorDimPosInCodomain) {
      return failure();
    }
    auto dataType = xsmm::utils::getDataType(rewriter, operand.getType());
    FailureOr<SmallVector<int64_t>> stridesOnOperand;
    if (isVnni && dataType == DataTypeAttr::get(ctx, xsmm::DataType::BF16) &&
        operandIndex == 1) {
      stridesOnOperand =
          getVNNIStaticStrides(dyn_cast<MemRefType>(operand.getType()));
    } else {
      stridesOnOperand = mlir::utils::getStaticStrides(operand);
    }
    if (failed(stridesOnOperand) ||
        (isVnni && dataType == DataTypeAttr::get(ctx, xsmm::DataType::BF16) &&
         operandIndex == 0 &&
         (*stridesOnOperand)[*minorDimPosInCodomain] != 2) ||
        (isVnni && (dataType != DataTypeAttr::get(ctx, xsmm::DataType::BF16) &&
                    (*stridesOnOperand)[*minorDimPosInCodomain] != 1))) {
      return failure();
    }
    if (isVnni && dataType == DataTypeAttr::get(ctx, xsmm::DataType::BF16) &&
        operandIndex == 1) {
      return (*stridesOnOperand)[*majorDimPosInCodomain + 1];
    } else {
      return (*stridesOnOperand)[*majorDimPosInCodomain];
    }
  };
  // A(m, k)
  auto lda = checkStridesAndGetLda(k, m, operandA, indexingMap[0], 0);
  if (failed(lda)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute lda\n");
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "A: OK\n");

  // B(k, n)
  auto ldb = checkStridesAndGetLda(n, k, operandB, indexingMap[1], 1);
  if (failed(ldb)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute ldb\n");

    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "B: OK\n");

  // C(m, n)
  auto ldc = checkStridesAndGetLda(n, m, operandC, indexingMap[2], 2);
  if (failed(ldc)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to compute ldc\n");
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Strides on "
                             "C: OK\n");

  int64_t strideA = 1;
  int64_t strideB = 1;
  if (batchPos) {
    auto batchPosCodomainA = getPosInCodomain(batchPos.value(), operandA,
                                              contractOp, indexingMap[0]);
    auto stridesOnA = ::mlir::utils::getStaticStrides(operandA);
    strideA = (*stridesOnA)[*batchPosCodomainA];

    auto batchPosCodomainB = getPosInCodomain(batchPos.value(), operandB,
                                              contractOp, indexingMap[1]);
    auto stridesOnB = ::mlir::utils::getStaticStrides(operandB);
    strideB = (*stridesOnB)[*batchPosCodomainB];
  }

  auto loops = computeStaticLoopSizes(contractOp, indexingMap);
  int64_t batchVal = (batchPos) ? loops[batchPos.value()] : 0;

  auto loopsK = 1;
  for (auto kItr : kVector)
    loopsK *= loops[kItr];

  xsmm::BrgemmInfo info{loops[m], loops[n], loopsK,  batchVal, *lda,
                        *ldb,     *ldc,     strideA, strideB,  isVnni};
  return info;
}

// Check if the given
// generic is mappable to a
// brgemm xsmm op.
// - It is a contraction,
// with:
// -- 1 m and 1 n and 2 k
// dimensions.
// -- m appears on the LHS
// and OUT but not in RHS.
// -- n appears on the RHS
// and OUT but not in LHS.
// -- k and k' appear on the
// RHS and LHS but not OUT.
// -- the stride of the
// minor dimension for A, k
// is 1.
// -- the stride of the
// minor dimension for B, n
// is 1.
// -- the stride of the
// minor dimension for C, n
// is 1.
FailureOr<BrgemmInfo> isMappableToBrgemm(PatternRewriter &rewriter,
                                         Operation *contractOp,
                                         SmallVector<Value> &inputs,
                                         SmallVector<Value> &output,
                                         ArrayRef<AffineMap> indexingMap) {
  auto contractionDims =
      checkStructure(contractOp, inputs, output, indexingMap);
  if (failed(contractionDims)) {
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] Failed "
                               "on "
                               "checkStructure"
                               "\n");
    return failure();
  }
  unsigned m = contractionDims->m.back();
  unsigned n = contractionDims->n.back();
  SmallVector<unsigned, 2> kVector;
  std::optional<unsigned> batch;
  if (contractionDims->k.size() >= 2) {
    for (size_t i = 1; i < contractionDims->k.size(); i++)
      kVector.push_back(contractionDims->k[i]);
  } else {
    for (size_t i = 0; i < contractionDims->k.size(); i++)
      kVector.push_back(contractionDims->k[i]);
  }

  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] Candidate "
                             "dims: "
                          << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] m: "
                          << m << "\n");
  LLVM_DEBUG(llvm::dbgs() << "[isMappableToBrge"
                             "mm] n: "
                          << n << "\n");
  if (batch)
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] batch: "
                            << batch << "\n");
  else
    LLVM_DEBUG(llvm::dbgs() << "[isMappableToBr"
                               "gemm] no batch "
                               "dim\n");
  auto retval = checkAccess(rewriter, contractOp, m, n, kVector, batch, inputs,
                            indexingMap);
  if (failed(retval)) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to check access\n");
    return failure();
  }
  return retval;
}

DataTypeAttr getDataType(RewriterBase &rewriter, Type type) {
  auto elemType = getElementTypeOrSelf(type);
  if (elemType.isBF16())
    return DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::BF16);
  return DataTypeAttr::get(rewriter.getContext(), xsmm::DataType::F32);
}

FailureOr<UnaryInfo> getUnaryInfo(Value input, Value output,
                                  UnaryFlags inputFlag) {
  Type outputType = output.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = cast<ShapedType>(outputType);
  if (outputShapedType.getRank() != 2 || !outputShapedType.hasStaticShape() ||
      !isa<FloatType>(outputShapedType.getElementType())) {
    return failure();
  }

  UnaryInfo unaryInfo;
  unaryInfo.m = outputShapedType.getShape()[0];
  unaryInfo.n = outputShapedType.getShape()[1];

  int64_t ldi = 1;
  if (ShapedType inputShapedType = dyn_cast<ShapedType>(input.getType())) {
    auto stridesOnInput = mlir::utils::getStaticStrides(input);
    if (failed(stridesOnInput) || stridesOnInput->back() != 1 ||
        !inputShapedType.hasStaticShape()) {
      return failure();
    }

    // If we are broascasting a row into cols, the leading
    // dimension is 1, same for scalar broadcast.
    if (inputFlag == UnaryFlags::BCAST_ROW ||
        inputFlag == UnaryFlags::BCAST_SCALAR) {
      ldi = 1;
    }
    // If we are broascasting a col into rows, the leading
    // dimension is the size of the tensor.
    else if (inputFlag == UnaryFlags::BCAST_COL) {
      ldi = inputShapedType.getShape().back();
    } else {
      ldi = stridesOnInput->front();
    }
  }
  auto stridesOnOutput = mlir::utils::getStaticStrides(output);
  if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
    return failure();

  unaryInfo.ldi = ldi;
  unaryInfo.ldo = stridesOnOutput->front();
  return unaryInfo;
}

FailureOr<BinaryInfo> getBinaryInfo(Value lhs, BinaryFlags lhsFlag, Value rhs,
                                    BinaryFlags rhsFlag, Value output) {
  Type outputType = output.getType();

  assert(isa<ShapedType>(outputType));
  auto outputShapedType = cast<ShapedType>(outputType);
  if (outputShapedType.getRank() != 2 || !outputShapedType.hasStaticShape() ||
      !isa<FloatType>(outputShapedType.getElementType())) {
    return failure();
  }

  BinaryInfo binaryInfo;
  binaryInfo.m = outputShapedType.getShape()[0];
  binaryInfo.n = outputShapedType.getShape()[1];

  int64_t ldiLhs = 1;
  if (ShapedType lhsShapedType = dyn_cast<ShapedType>(lhs.getType())) {
    auto stridesOnLhs = mlir::utils::getStaticStrides(lhs);
    if (failed(stridesOnLhs) || stridesOnLhs->back() != 1 ||
        !lhsShapedType.hasStaticShape()) {
      return failure();
    }

    if (lhsFlag == BinaryFlags::BCAST_SCALAR_IN_0 ||
        lhsFlag == BinaryFlags::BCAST_ROW_IN_0) {
      ldiLhs = 1;
    } else if (lhsFlag == BinaryFlags::BCAST_COL_IN_0) {
      ldiLhs = lhsShapedType.getShape().back();
    } else {
      ldiLhs = stridesOnLhs->front();
    }
  }

  int64_t ldiRhs = 1;
  if (ShapedType rhsShapedType = dyn_cast<ShapedType>(rhs.getType())) {
    auto stridesOnRhs = mlir::utils::getStaticStrides(rhs);
    if (failed(stridesOnRhs) || stridesOnRhs->back() != 1 ||
        !rhsShapedType.hasStaticShape()) {
      return failure();
    }

    if (rhsFlag == BinaryFlags::BCAST_SCALAR_IN_1 ||
        rhsFlag == BinaryFlags::BCAST_ROW_IN_1) {
      ldiRhs = 1;
    } else if (rhsFlag == BinaryFlags::BCAST_COL_IN_1) {
      ldiRhs = rhsShapedType.getShape().back();
    } else {
      ldiRhs = stridesOnRhs->front();
    }
  }

  binaryInfo.ldiLhs = ldiLhs;
  binaryInfo.ldiRhs = ldiRhs;

  auto stridesOnOutput = mlir::utils::getStaticStrides(output);
  if (failed(stridesOnOutput) || stridesOnOutput->back() != 1)
    return failure();
  binaryInfo.ldo = stridesOnOutput->front();
  return binaryInfo;
}

// Examples:
// If lower=[c], higher=[a, b, c], [c] reshaped into [1, 1, c].
// If lower=[b, c], higher=[a, b, c], [b, c] reshaped into [1, b, c].
// If lower=[a], higher=[a, a], [a] reshaped into [1, a].
// If lower=[a], target=[a, b, a], [a] reshaped into [1, 1, a].
// If lower=[], target=[a, b, c], [] reshaped into [1, 1, 1].
static void
computeBcastShapeInput(ArrayRef<int64_t> higherRankShape,
                       ArrayRef<int64_t> lowerRankShape,
                       SmallVectorImpl<int64_t> &reshapeOutputShape) {
  // Initialize new shapes with [1] * higherRank.
  int64_t higherRank = higherRankShape.size();
  int64_t lowerRank = lowerRankShape.size();

  reshapeOutputShape.assign(higherRank, 1);

  int64_t higherRankDim;
  int64_t lowerRankDim;

  for (int64_t i = higherRank - 1, j = lowerRank - 1; i >= 0 && j >= 0;
       i--, j--) {
    higherRankDim = higherRankShape[i];
    lowerRankDim = lowerRankShape[j];

    if (lowerRankDim == 1 && higherRankDim > 1)
      reshapeOutputShape[i] = 1;
    else if ((lowerRankDim > 1 && higherRankDim == 1) ||
             (lowerRankDim == higherRankDim))
      reshapeOutputShape[i] = lowerRankDim;
    else if (higherRankDim != lowerRankDim)
      assert(false && "bCast semantics for identity op broken");
  }
}

FailureOr<UnaryFlags> getUnaryFlags(Type inputType, Type outputType) {
  assert(isa<ShapedType>(outputType) && "expect shaped type on output");
  assert(cast<ShapedType>(outputType).getRank() == 2 &&
         "expect rank 2 on output");

  if (!isa<ShapedType>(inputType) ||
      cast<ShapedType>(inputType).getRank() == 0) {
    return xsmm::UnaryFlags::BCAST_SCALAR;
  }

  ArrayRef<int64_t> shapeOutput = cast<ShapedType>(outputType).getShape();
  ArrayRef<int64_t> shapeInput = cast<ShapedType>(inputType).getShape();
  assert(shapeOutput.size() >= shapeInput.size() &&
         "output rank must be >= input rank");
  SmallVector<int64_t> bShapeInput;
  computeBcastShapeInput(shapeOutput, shapeInput, bShapeInput);
  assert(shapeOutput.size() == bShapeInput.size());
  shapeInput = bShapeInput;

  // Same shape for input and output, no bcast.
  if (shapeInput == shapeOutput)
    return xsmm::UnaryFlags::NONE;

  // Input is a memref but it is all ones, bcast = scalar.
  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(shapeInput, isOne))
    return xsmm::UnaryFlags::BCAST_SCALAR;

  if (shapeInput[1] == 1 && shapeOutput[1] > 1)
    return xsmm::UnaryFlags::BCAST_ROW;

  if (shapeInput[0] == 1 && shapeOutput[0] > 1)
    return xsmm::UnaryFlags::BCAST_COL;

  return failure();
}

FailureOr<BinaryFlags> getBinFlags(ArrayRef<int64_t> shapeOutput,
                                   ArrayRef<int64_t> shapeOperand,
                                   OperandPos operandNumber) {
  assert(shapeOutput.size() >= shapeOperand.size() &&
         "Output rank must be >= operand rank");
  SmallVector<int64_t> bOperandShape;
  computeBcastShapeInput(shapeOutput, shapeOperand, bOperandShape);
  assert(shapeOutput.size() == bOperandShape.size());
  assert(shapeOutput.size() == 2);
  enum class BCastType { NONE = 0, SCALAR, ROW, COL };
  auto getBCastEnum = [](BCastType bCastType,
                         OperandPos operandPos) -> xsmm::BinaryFlags {
    switch (bCastType) {
    case BCastType::NONE:
      return xsmm::BinaryFlags::NONE;
    case BCastType::SCALAR:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
    case BCastType::ROW:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_ROW_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_ROW_IN_1;
    case BCastType::COL:
      if (operandPos == OperandPos::LHS)
        return xsmm::BinaryFlags::BCAST_COL_IN_0;
      else
        return xsmm::BinaryFlags::BCAST_COL_IN_1;
    }
    assert(false && "unrechable");
    abort();
  };

  if (bOperandShape == shapeOutput)
    return getBCastEnum(BCastType::NONE, operandNumber);

  auto isOne = [](int64_t val) { return val == 1; };
  if (llvm::all_of(bOperandShape, isOne))
    return getBCastEnum(BCastType::SCALAR, operandNumber);

  if (bOperandShape[1] == 1 && shapeOutput[1] > 1)
    return getBCastEnum(BCastType::ROW, operandNumber);

  if (bOperandShape[0] == 1 && shapeOutput[0] > 1)
    return getBCastEnum(BCastType::COL, operandNumber);

  return failure();
}

FailureOr<BinaryFlags> getBinaryFlags(Type operandType, Type outputType,
                                      OperandPos operandNumber) {
  assert(isa<ShapedType>(outputType) && "expect shaped type on output");
  assert(cast<ShapedType>(outputType).getRank() == 2 &&
         "expect rank 2 on output");

  if (!isa<ShapedType>(operandType) ||
      cast<ShapedType>(operandType).getRank() == 0) {
    if (operandNumber == OperandPos::LHS)
      return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
    return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
  }

  enum class BCastType { NONE = 0, SCALAR, ROW, COL };
  auto shapeOutput = cast<MemRefType>(outputType).getShape();
  auto shapeOperand = cast<MemRefType>(operandType).getShape();
  return getBinFlags(shapeOutput, shapeOperand, operandNumber);
}

FailureOr<BinaryFlags> getBinaryFlagsVectorType(Type operandType,
                                                Type outputType,
                                                OperandPos operandNumber) {
  assert(isa<ShapedType>(outputType) && "expect shaped type on output");
  assert(cast<ShapedType>(outputType).getRank() == 2 &&
         "expect rank 2 on output");

  if (!isa<ShapedType>(operandType) ||
      cast<ShapedType>(operandType).getRank() == 0) {
    if (operandNumber == OperandPos::LHS)
      return xsmm::BinaryFlags::BCAST_SCALAR_IN_0;
    return xsmm::BinaryFlags::BCAST_SCALAR_IN_1;
  }

  enum class BCastType { NONE = 0, SCALAR, ROW, COL };
  auto shapeOutput = cast<VectorType>(outputType).getShape();
  auto shapeOperand = cast<MemRefType>(operandType).getShape();
  return getBinFlags(shapeOutput, shapeOperand, operandNumber);
}

FailureOr<int64_t> getLeadingDim(Type type, size_t pos) {
  // Not shaped type, the leading dimension is the single scalar.
  auto memref = dyn_cast<MemRefType>(type);
  if (!memref)
    return 1;
  // For 1d memref we cannot use the stride as leading dimension, but the
  // leading dimension is the dimension itself.
  if (memref.getRank() == 1)
    return memref.getShape()[0];

  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memref, strides, offset)))
    return failure();
  // fail if the strides are non-constant
  if (llvm::any_of(strides, [](int64_t stride) {
        return stride == ShapedType::kDynamic;
      }))
    return failure();
  return strides[pos];
}

static bool isInnerMostDim(OpOperand *operand, unsigned minorDim,
                           vector::ContractionOp contractOp, DataTypeAttr dtype,
                           int operandNumber) {
  auto shapedType = cast<VectorType>(operand->get().getType());
  int64_t rank = shapedType.getRank();
  if (dtype ==
          DataTypeAttr::get(contractOp.getContext(), xsmm::DataType::BF16) &&
      (operandNumber == 1 || operandNumber == 0)) {
    return minorDim == rank - 2;
  }
  return minorDim == rank - 1;
}

// Emit a transpose operation for `operand` by swapping `dim` with `newDim`.
// Emit a transpose operation for `operand` by swapping the dimensions at index
// `dim` with `newDim`.
static void emitTransposeOnOperand(RewriterBase &rewriter,
                                   vector::ContractionOp contractOp,
                                   Value operand, unsigned dim, unsigned newDim,
                                   int operandNumber) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(contractOp);

  Location loc = contractOp.getLoc();
  auto operandType = cast<ShapedType>(operand.getType());
  auto rank = operandType.getRank();
  SmallVector<int64_t> shape = llvm::to_vector(operandType.getShape());
  auto permutation = llvm::to_vector(llvm::seq<int64_t>(0, rank));
  std::swap(permutation[dim], permutation[newDim]);
  assert(isPermutationVector(permutation));
  LLVM_DEBUG(llvm::interleaveComma(
      permutation, llvm::dbgs() << "[emitTransposeOnOperand] Perm: "));
  LLVM_DEBUG(llvm::dbgs() << "\n");

  applyPermutationToVector<int64_t>(shape, permutation);
  auto vectorType = VectorType::get(
      shape, cast<ShapedType>(operand.getType()).getElementType());
  Value transposeResult = rewriter.create<vector::TransposeOp>(
      loc, vectorType, operand, permutation);

  SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
  AffineMap operandMap = indexingMaps[operandNumber];
  LLVM_DEBUG(llvm::dbgs() << "[emitTransposeOnOperand] Old map: " << operandMap
                          << "\n");
  SmallVector<AffineExpr> mapResults = llvm::to_vector(operandMap.getResults());
  applyPermutationToVector<AffineExpr>(mapResults, permutation);
  AffineMap newMap =
      AffineMap::get(operandMap.getNumDims(), operandMap.getNumSymbols(),
                     mapResults, contractOp.getContext());
  LLVM_DEBUG(llvm::dbgs() << "[emitTransposeOnOperand] New map: " << newMap
                          << "\n");
  indexingMaps[operandNumber] = newMap;
  // TODO: We probably cannot update the result in place.
  rewriter.modifyOpInPlace(contractOp, [&]() {
    contractOp->setOperand(operandNumber, transposeResult);
    contractOp.setIndexingMapsAttr(
        ArrayAttr::get(contractOp.getContext(),
                       llvm::to_vector(llvm::map_range(
                           indexingMaps, [](AffineMap map) -> Attribute {
                             return AffineMapAttr::get(map);
                           }))));
  });
}

FailureOr<vector::ContractionOp>
makeMinorDimensionsInnerMost(RewriterBase &rewriter,
                             vector::ContractionOp contractOp, unsigned m,
                             unsigned n, unsigned k, DataTypeAttr type) {
  OpOperand *operandA = &contractOp->getOpOperand(0);
  OpOperand *operandB = &contractOp->getOpOperand(1);
  OpOperand &operandC = contractOp->getOpOperand(2);

  // C(m,n) += A(m,k) * B(k,n)
  // n is expected to be the innermost for C
  // k is expected to be the innermost for A
  // n is expected to be the innermost for B
  auto minorKInCodomainOpA = xsmm::utils::getPosInCodomain(
      k, operandA->get(), contractOp, contractOp.getIndexingMapsArray()[0]);
  auto minorMInCodomainOpA = xsmm::utils::getPosInCodomain(
      m, operandA->get(), contractOp, contractOp.getIndexingMapsArray()[0]);
  if (!minorKInCodomainOpA || !minorMInCodomainOpA) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for A\n");
    return failure();
  }

  auto minorNInCodomainOpB = xsmm::utils::getPosInCodomain(
      n, operandB->get(), contractOp, contractOp.getIndexingMapsArray()[1]);
  auto minorKInCodomainOpB = xsmm::utils::getPosInCodomain(
      k, operandB->get(), contractOp, contractOp.getIndexingMapsArray()[1]);
  if (!minorNInCodomainOpB || !minorKInCodomainOpB) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for B\n");
    return failure();
  }

  auto minorNInCodomainOpC = xsmm::utils::getPosInCodomain(
      n, operandC.get(), contractOp, contractOp.getIndexingMapsArray()[2]);
  auto minorMInCodomainOpC = xsmm::utils::getPosInCodomain(
      m, operandC.get(), contractOp, contractOp.getIndexingMapsArray()[2]);
  if (!minorNInCodomainOpC || !minorMInCodomainOpC) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "[makeMinorDimensionsInnerMost] did not find minor dims for C\n");
    return failure();
  }

  if (!isInnerMostDim(&operandC, *minorNInCodomainOpC, contractOp, type, 2)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for C\n");
    assert(
        isInnerMostDim(&operandC, *minorMInCodomainOpC, contractOp, type, 2));
    if (isInnerMostDim(operandA, *minorKInCodomainOpA, contractOp, type, 0)) {
      emitTransposeOnOperand(rewriter, contractOp, operandA->get(),
                             *minorKInCodomainOpA, *minorMInCodomainOpA, 0);
    }
    if (isInnerMostDim(operandB, *minorNInCodomainOpB, contractOp, type, 1)) {
      emitTransposeOnOperand(rewriter, contractOp, operandB->get(),
                             *minorNInCodomainOpB, *minorKInCodomainOpB, 1);
    }
    // Avoid transpose on the output by swapping A and B.
    OpOperand *operandA = &contractOp->getOpOperand(0);
    OpOperand *operandB = &contractOp->getOpOperand(1);
    SmallVector<AffineMap> indexingMaps = contractOp.getIndexingMapsArray();
    std::swap(indexingMaps[0], indexingMaps[1]);
    rewriter.modifyOpInPlace(contractOp, [&]() {
      Value operandATmp = operandA->get();
      contractOp->setOperand(0, operandB->get());
      contractOp->setOperand(1, operandATmp);
      contractOp.setIndexingMapsAttr(
          ArrayAttr::get(contractOp.getContext(),
                         llvm::to_vector(llvm::map_range(
                             indexingMaps, [](AffineMap map) -> Attribute {
                               return AffineMapAttr::get(map);
                             }))));
    });
    return contractOp;
  }

  if (!isInnerMostDim(operandA, *minorKInCodomainOpA, contractOp, type, 0)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for A\n");
    assert(isInnerMostDim(operandA, *minorMInCodomainOpA, contractOp, type, 0));
    emitTransposeOnOperand(rewriter, contractOp, operandA->get(),
                           *minorKInCodomainOpA, *minorMInCodomainOpA, 0);
  }
  if (!isInnerMostDim(operandB, *minorNInCodomainOpB, contractOp, type, 1)) {
    LLVM_DEBUG(llvm::dbgs()
               << "[makeMinorDimensionsInnerMost] emit transpose for B\n");
    assert(isInnerMostDim(operandB, *minorKInCodomainOpB, contractOp, type, 1));
    emitTransposeOnOperand(rewriter, contractOp, operandB->get(),
                           *minorKInCodomainOpB, *minorNInCodomainOpB, 1);
  }
  return contractOp;
}

bool WithInputs(PatternRewriter &rewriter, Operation *op,
                SmallVector<std::function<bool(Operation *op)>> operations,
                SmallVector<Value> &inputs, SmallVector<Operation *> &opChain) {
  for (size_t i = 0; i < operations.size(); i++) {
    auto input = op->getOperand(i);
    if (!operations[i](input.getDefiningOp()))
      return false;
    if (input.getDefiningOp()->getOperand(0).getDefiningOp() != nullptr) {
      if (!(isa<memref::ExpandShapeOp>(
                input.getDefiningOp()->getOperand(0).getDefiningOp()) ||
            isa<memref::GetGlobalOp>(
                input.getDefiningOp()->getOperand(0).getDefiningOp()) ||
            isa<memref::SubViewOp>(
                input.getDefiningOp()->getOperand(0).getDefiningOp()) ||
            isa<vector::TransferReadOp>(
                input.getDefiningOp()->getOperand(0).getDefiningOp())))
        return false;
    }
    inputs.push_back(input.getDefiningOp()->getOpOperand(0).get());
    opChain.push_back(input.getDefiningOp());
  }
  return true;
}

bool WithOutput(Operation *op, std::function<bool(Operation *op)> operation,
                SmallVector<Value> &output, SmallVector<Operation *> &opChain) {
  // Check on the inner chain of operations in the right order.
  // Make sure all operands are used and chained
  for (auto use : op->getResult(0).getUsers()) {
    if (use != op && operation(use)) {
      if (!isa<memref::SubViewOp>(use->getOperand(1).getDefiningOp()))
        return false;
      output.push_back(use->getOpOperand(1).get());
      opChain.push_back(use);
      return true;
    }
  }
  return false;
}

bool WithOps(Region *region, Operation *op, Operation *currentOp,
             SmallVector<std::function<bool(Operation *op)>> operations,
             SmallVector<Operation *> &opChain) {
  auto &block = region->front();

  llvm::SmallSetVector<Value, 4> chainedValues;

  auto start = block.begin();
  for (auto opItr = block.begin(); opItr != block.end(); opItr++) {
    if (&*opItr != currentOp || !operations[0](&*opItr))
      continue;
    start = opItr;
    opChain.push_back(&*opItr);
    break;
  }
  // Check on the inner chain of operations in the right order.
  // Make sure all operands are used and chained
  for (auto check : operations) {
    Operation *innerOp = &*start;
    // Must be right op in right order
    if (start == block.end() || !check(innerOp))
      return false;
    start++;
    // At least one operand must come from args or a previous op
    bool consumesValueFromChain = false;
    if (chainedValues.empty()) {
      consumesValueFromChain = true;
    } else {
      for (auto operand : innerOp->getOperands()) {
        if (chainedValues.contains(operand)) {
          chainedValues.remove(operand);
          consumesValueFromChain = true;
        }
      }
    }

    // Operation isn't in the chain
    if (!consumesValueFromChain)
      return false;

    for (auto ret : innerOp->getResults()) {
      chainedValues.insert(ret);
    }
  }
  return true;
}

bool isTwoDTransposeOp(vector::TransposeOp transposeOp) {
  if (!(dyn_cast<VectorType>(transposeOp.getOperand().getType()).getRank() ==
            2 &&
        dyn_cast<VectorType>(transposeOp.getResult().getType()).getRank() ==
            2) ||
      !(isa<scf::ForallOp>(transposeOp->getParentOp()) &&
        dyn_cast<scf::ForallOp>(transposeOp->getParentOp()).getRank() == 2))
    return false;
  return true;
}

// Extract the operands to be used in the function call. For each memref operand
// extract the aligned pointer and the offset.
SmallVector<Value> getOperands(OpBuilder &builder, Location loc,
                               ValueRange operands, IntegerAttr dataTypeAttr,
                               std::optional<IntegerAttr> outDataTypeAttr) {
  SmallVector<Value> res;
  IntegerType integer64 = IntegerType::get(builder.getContext(), 64);
  res.push_back(
      builder.create<arith::ConstantOp>(loc, integer64, dataTypeAttr));
  if (outDataTypeAttr)
    res.push_back(
        builder.create<arith::ConstantOp>(loc, integer64, *outDataTypeAttr));

  for (Value operand : operands) {
    auto memrefType = dyn_cast<MemRefType>(operand.getType());
    if (!memrefType) {
      res.push_back(operand);
      continue;
    }
    auto [ptr, offset] = ::mlir::utils::getPtrAndOffset(builder, operand, loc);
    res.push_back(ptr);
    res.push_back(offset);
  }
  return res;
}

SmallVector<Type> extractInvokeOperandTypes(OpBuilder &builder,
                                            ValueRange operands) {
  SmallVector<Type> results;
  for (Value operand : operands) {
    Type operandType = operand.getType();
    if (auto memrefType = dyn_cast<MemRefType>(operandType)) {
      // TODO: non-POD will require an LLVMTypeConverter.
      Type basePtrType = LLVM::LLVMPointerType::get(builder.getContext());
      results.push_back(basePtrType);
      results.push_back(builder.getIndexType()); // offset
    } else {
      results.push_back(operand.getType());
    }
  }
  return results;
}

int64_t getOredFlags(ArrayAttr flags) {
  int64_t oredFlag = 0;
  for (auto flag : flags) {
    int64_t intAttr = dyn_cast<DataTypeAttr>(flag).getInt();
    // LIBXSMM is col-major, swap A and B flags.
    if (auto gemmFlag = dyn_cast_or_null<xsmm::GemmFlagsAttr>(flag)) {
      if (gemmFlag.getValue() == GemmFlags::VNNI_A)
        intAttr = static_cast<int64_t>(GemmFlags::VNNI_B);
      if (gemmFlag.getValue() == GemmFlags::VNNI_B)
        intAttr = static_cast<int64_t>(GemmFlags::VNNI_A);
    }
    oredFlag |= intAttr;
  }
  return oredFlag;
}

func::CallOp buildDispatchCall(RewriterBase &rewriter, Location loc,
                               ArrayRef<Value> dispatchOperands,
                               ArrayRef<Type> dispatchOperandTypes,
                               ModuleOp module, FlatSymbolRefAttr fnName) {
  auto libFnType = rewriter.getFunctionType(
      dispatchOperandTypes, IntegerType::get(rewriter.getContext(), 64));

  if (!module.lookupSymbol(fnName.getAttr())) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, fnName.getValue(), libFnType);
    funcOp.setPrivate();
  }

  func::CallOp call = rewriter.create<func::CallOp>(
      loc, fnName.getValue(), IntegerType::get(rewriter.getContext(), 64),
      dispatchOperands);
  return call;
}

func::CallOp buildInvokeCall(RewriterBase &rewriter, Location loc,
                             ModuleOp module, SmallVector<Value> operandRange,
                             StringRef invokeName, DataTypeAttr dtype,
                             std::optional<DataTypeAttr> outDtype) {
  SmallVector<Type> operandTypes;
  // Extra operands for datatypes.
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  operandTypes.push_back(integer64);
  if (outDtype)
    operandTypes.push_back(integer64);
  operandTypes.append(
      xsmm::utils::extractInvokeOperandTypes(rewriter, operandRange));
  auto libFnType = rewriter.getFunctionType(operandTypes, {});
  FlatSymbolRefAttr fnName =
      SymbolRefAttr::get(rewriter.getContext(), invokeName);

  if (!module.lookupSymbol(fnName)) {
    OpBuilder::InsertionGuard guard(rewriter);
    // Insert before module terminator.
    rewriter.setInsertionPoint(module.getBody(),
                               std::prev(module.getBody()->end()));
    func::FuncOp funcOp =
        rewriter.create<func::FuncOp>(loc, invokeName, libFnType);
    funcOp.setPrivate();
  }

  func::CallOp call = rewriter.create<func::CallOp>(
      loc, fnName, TypeRange(),
      xsmm::utils::getOperands(rewriter, loc, operandRange, dtype, outDtype));

  return call;
}

std::pair<Operation *, Operation *>
buildBrgemmCalls(PatternRewriter &rewriter, Operation *op, ValueRange inputs,
                 xsmm::BrgemmInfo brgemmInfo, SmallVector<Attribute> flags) {
  assert(inputs.size() == 3 && "Expects three inputs for BRGEMM call");
  auto m = brgemmInfo.m;
  auto n = brgemmInfo.n;
  auto k = brgemmInfo.k;
  auto batch = brgemmInfo.batch;
  int64_t lda = brgemmInfo.lda;
  int64_t ldb = brgemmInfo.ldb;
  int64_t ldc = brgemmInfo.ldc;
  int64_t strideA = brgemmInfo.strideA;
  int64_t strideB = brgemmInfo.strideB;
  auto loc = op->getLoc();
  auto dtype = xsmm::utils::getDataType(rewriter, inputs[0].getType());
  auto outDtype = xsmm::utils::getDataType(rewriter, inputs[2].getType());
  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  SmallVector<Value, 11> dispatchOperands;
  SmallVector<Type, 11> dispatchOperandTypes;
  // Dispatch the data type.
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(dtype)));
  dispatchOperandTypes.push_back(integer64);
  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, cast<TypedAttr>(outDtype)));
  dispatchOperandTypes.push_back(integer64);

  ArrayAttr brgemmFlags = rewriter.getArrayAttr(flags);
  SmallVector<Value, 10> invokeOperands;
  std::string dispatchName = "xsmm_gemm_dispatch";
  std::string invokeName = "xsmm_gemm_invoke";

  if (batch != 0) {
    dispatchName = "xsmm_brgemm_dispatch";
    invokeName = "xsmm_brgemm_invoke";
  }

  auto dims = SmallVector<int64_t>{m, n, k, lda, ldb, ldc};
  if (batch != 0) {
    dims.append({strideA, strideB});
  }
  for (size_t idx = 0; idx < dims.size(); idx++) {
    dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, dims[idx])));
    dispatchOperandTypes.push_back(integer64);
  }
  // Dispatch the flags. Pass to the library the already ored-flag to
  // avoid changing the interface every time we add a new flag. Flags
  // are assumed to be verified before (i.e., op verifier).
  int64_t oredFlag = xsmm::utils::getOredFlags(brgemmFlags);

  dispatchOperands.push_back(rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), oredFlag)));
  dispatchOperandTypes.push_back(integer64);
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto dispatched = xsmm::utils::buildDispatchCall(
      rewriter, loc, dispatchOperands, dispatchOperandTypes, module,
      SymbolRefAttr::get(op->getContext(), dispatchName));
  SmallVector<Value, 6> operandRange;
  operandRange.push_back(dispatched.getResult(0));
  for (auto operand : inputs) {
    operandRange.push_back(operand);
  }
  if (batch != 0) {
    Value batchDim = rewriter.create<arith::ConstantOp>(
        loc, integer64, rewriter.getIntegerAttr(integer64, batch));
    operandRange.push_back(batchDim);
  }
  auto invokeCall = xsmm::utils::buildInvokeCall(
      rewriter, loc, module, operandRange, invokeName, dtype, outDtype);
  return std::make_pair(&*dispatched, &*invokeCall);
}

} // namespace utils
} // namespace xsmm
} // namespace mlir
