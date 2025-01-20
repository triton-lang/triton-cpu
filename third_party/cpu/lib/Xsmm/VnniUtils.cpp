//===- VNNIUtils.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VnniUtils.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"

#include "libxsmm.h"

namespace mlir {
namespace vnni {
namespace utils {

std::optional<int64_t> getVnniBlockingFactor(Type type) {
  auto elementType = getElementTypeOrSelf(type);
  if (elementType.isBF16())
    return libxsmm_cpuid_dot_pack_factor(LIBXSMM_DATATYPE_BF16);
  return std::nullopt;
}

// Until we have a better way to express the VNNI layout (see: #563), it is up
// to the callee to specify the expected rank in the VNNI layout as the rank
// depends on the operations we are dealing with.
bool isInVnniLayout(VnniOperandRank expectedRank, MemRefType memref) {
  if (memref.getRank() != static_cast<int64_t>(expectedRank) ||
      !memref.getElementType().isBF16()) {
    return false;
  }
  return memref.getShape().back() == vnni::utils::getVnniBlockingFactor(memref);
}

bool isInVnniLayout(int64_t expectedRank, VectorType vector) {
  if (vector.getRank() != expectedRank || !vector.getElementType().isBF16()) {
    return false;
  }
  return vector.getShape().back() == vnni::utils::getVnniBlockingFactor(vector);
}

// Until we have a better way to express the VNNI layout (see: #563), it is up
// to the callee to specify the expected rank in the VNNI layout as the rank
// depends on the operations we are dealing with.
bool isInVnniLayout(VnniOperandRank expectedRank, VectorType vector) {
  return isInVnniLayout((int64_t)expectedRank, vector);
}

FailureOr<AffineDimExpr> isInVnniLayout(linalg::GenericOp linalgOp,
                                        AffineMap map, int64_t blockingFactor) {
  ArrayRef<AffineExpr> results = map.getResults();
  SmallVector<mlir::utils::IteratorType> iteratorTypes =
      linalgOp.getIteratorTypesArray();

  AffineExpr vnniDim = results.back();
  auto dimExpr = dyn_cast<AffineDimExpr>(vnniDim);
  if (!dimExpr || iteratorTypes[dimExpr.getPosition()] !=
                      mlir::utils::IteratorType::reduction) {
    return failure();
  }

  for (auto result : results) {
    auto blockeDim = dyn_cast<AffineBinaryOpExpr>(result);
    if (!blockeDim)
      continue;
    if (blockeDim.getKind() != AffineExprKind::FloorDiv)
      continue;
    auto lhsDim = dyn_cast<AffineDimExpr>(blockeDim.getLHS());
    auto rhsCst = dyn_cast<AffineConstantExpr>(blockeDim.getRHS());
    if (!lhsDim || !rhsCst)
      continue;
    if (iteratorTypes[lhsDim.getPosition()] !=
        mlir::utils::IteratorType::reduction)
      continue;
    if (rhsCst.getValue() != blockingFactor)
      continue;
    return lhsDim;
  }
  return failure();
}

} // namespace utils
} // namespace vnni
} // namespace mlir
