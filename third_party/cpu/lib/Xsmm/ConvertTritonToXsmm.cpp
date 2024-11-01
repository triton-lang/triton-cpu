//===- ConvertTritonToXsmm.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "cpu/include/Xsmm/Passes.h"

#include "ValueUtils.h"
#include "VnniUtils.h"
#include "XsmmUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#include "cpu/include/Analysis/TensorPtrShapeInfo.h"

#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::func;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTTRITONTOXSMM
#include "cpu/include/Xsmm/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

namespace {

// Helper from MemoryOpConversion.
// Extract memref out of block pointer.
static Value extractMemRef(PatternRewriter &rewriter, Value ptr,
                           ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis) {
  Location loc = ptr.getLoc();
  MLIRContext *ctx = ptr.getContext();

  auto tensorTy = dyn_cast<RankedTensorType>(
      dyn_cast<PointerType>(ptr.getType()).getPointeeType());
  auto elemTy = tensorTy.getElementType();
  auto shapeInfo = shapeAnalysis.getPtrShapeInfo(ptr);
  Type memRefTy;
  if (shapeInfo && shapeInfo->getRank() > 0) {
    auto layout = StridedLayoutAttr::get(ctx, 0, shapeInfo->getStrides());
    memRefTy = MemRefType::get(shapeInfo->getShape(), elemTy, layout);
  } else {
    SmallVector<int64_t> dynVals(tensorTy.getRank(), ShapedType::kDynamic);
    auto layout = StridedLayoutAttr::get(ctx, 0, dynVals);
    memRefTy = MemRefType::get(dynVals, elemTy, layout);
  }
  return rewriter.create<triton::cpu::ExtractMemRefOp>(loc, memRefTy, ptr);
}

static Value getMemrefSource(PatternRewriter &rewriter, Operation *op,
                             TypedValue<RankedTensorType> operand,
                             ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis) {
  Location loc = op->getLoc();
  MLIRContext *ctx = op->getContext();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  RankedTensorType tensorTy = operand.getType();

  if (auto loadOp = dyn_cast_or_null<triton::LoadOp>(operand.getDefiningOp())) {
    auto ptr = loadOp.getPtr();
    if (triton::isTensorPointerType(ptr.getType())) {
      auto memref = extractMemRef(rewriter, ptr, shapeAnalysis);
      auto indices =
          rewriter.create<triton::cpu::ExtractIndicesOp>(loc, ptr).getResults();
      SmallVector<int64_t> strides(tensorTy.getRank(), 1);

      return rewriter.create<memref::SubViewOp>(
          loc, memref, getAsOpFoldResult(indices),
          getAsIndexOpFoldResult(ctx, tensorTy.getShape()),
          getAsIndexOpFoldResult(ctx, strides));
    }
  }

  MemRefType memTy =
      MemRefType::get(tensorTy.getShape(), tensorTy.getElementType());
  auto alloca = rewriter.create<memref::AllocaOp>(loc, memTy);
  rewriter.create<triton::cpu::StoreOp>(loc, operand, alloca);

  return alloca;
}

// Helper to move accumulation buffer outside of GEMM reduction loop.
// Returns new accumulation buffer or std::nullopt, otherwise.
//
// Rewrites the following pattern:
//   %init = ... tensor<...>
//   %0 = scf.for ... iter_args(%acc = %init)
//     %res = GEMM(%A, %B, %acc) -> tensor<...>
//     scf.yield %res
//   consumer(%0)
// into:
//   %init = ... tensor<...>
//   %hoisted = ... memref<...>
//   store %init, %hoisted
//   %unused = %scf.for ... iter_args(%acc = %init)
//     %res = GEMM(%A, %B, %acc)
//     scf.yield %acc
//   %0 = load(%hoisted) -> tensor<...>
//   consumer(%0)
//
// This rewrite should be used as a part of contraction to memref conversion.
static std::optional<Value>
hoistAccumulationBuffer(PatternRewriter &rewriter, Operation *op,
                        TypedValue<RankedTensorType> operand,
                        ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis) {
  Location loc = op->getLoc();

  // Check if there is any loop around the contraction and if the operand
  // comes from loop's arguments.
  auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
  BlockArgument blockArg = dyn_cast<BlockArgument>(operand);
  if (!forOp || !blockArg)
    return std::nullopt;
  OpOperand *loopArg = forOp.getTiedLoopInit(blockArg);
  if (!loopArg)
    return std::nullopt;

  // The accumulation iter_arg can be safely moved outside the loop only
  // for the following chain: iter_arg -> contraction -> yield
  // and there are no other users.
  Value res = op->getResults()[0];
  if (!operand.hasOneUse() || !res.hasOneUse() ||
      !isa<scf::YieldOp>(*res.getUsers().begin()))
    return std::nullopt;

  // Create a buffer outside the loop.
  Value accBuf = getMemrefSource(
      rewriter, forOp, dyn_cast<TypedValue<RankedTensorType>>(loopArg->get()),
      shapeAnalysis);

  // For simplicity, feed the iter_arg directly into loop yield terminator.
  // Canonicalizer will folded them away later.
  rewriter.replaceAllUsesWith(res, operand);

  // Replace the corresponding loop result with the latest value read from the
  // accumulation buffer.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(forOp);

  auto loadOp =
      rewriter.create<triton::cpu::LoadOp>(loc, operand.getType(), accBuf);
  rewriter.replaceAllUsesWith(forOp.getTiedLoopResult(blockArg),
                              loadOp.getResult());

  return accBuf;
}

struct DotToXsmm : public OpRewritePattern<triton::DotOp> {
  using OpRewritePattern::OpRewritePattern;

  DotToXsmm(MLIRContext *ctx,
            ModuleTensorPtrShapeInfoAnalysis &shapeInfoAnalysis)
      : OpRewritePattern<triton::DotOp>(ctx), shapeAnalysis(shapeInfoAnalysis) {
  }

  LogicalResult matchAndRewrite(triton::DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    Location loc = dotOp.getLoc();
    MLIRContext *ctx = dotOp.getContext();

    // Dot op computes standard (batch) GEMM.
    SmallVector<AffineMap> indexingMaps;
    TypedValue<RankedTensorType> res = dotOp.getD();
    uint32_t rank = res.getType().getRank();
    if (rank == 2) {
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(3, {0, 2}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(3, {2, 1}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(3, {0, 1}, ctx));
    } else if (rank == 3) {
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(4, {0, 1, 3}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(4, {0, 3, 2}, ctx));
      indexingMaps.push_back(
          AffineMap::getMultiDimMapWithTargets(4, {0, 1, 2}, ctx));
    }
    if (indexingMaps.size() == 0)
      return rewriter.notifyMatchFailure(dotOp, "unsupported indexing maps");

    TypedValue<RankedTensorType> lhs = dotOp.getA();
    TypedValue<RankedTensorType> rhs = dotOp.getB();
    TypedValue<RankedTensorType> acc = dotOp.getC();

    SmallVector<Attribute> flags;
    Value lhsBuf = getMemrefSource(rewriter, dotOp, lhs, shapeAnalysis);
    Value rhsBuf = getMemrefSource(rewriter, dotOp, rhs, shapeAnalysis);
    std::optional<Value> hoistedAcc =
        hoistAccumulationBuffer(rewriter, dotOp, acc, shapeAnalysis);
    Value accBuf = hoistedAcc
                       ? *hoistedAcc
                       : getMemrefSource(rewriter, dotOp, acc, shapeAnalysis);
    SmallVector<Value> inputs{lhsBuf, rhsBuf, accBuf};
    SmallVector<Value> outputs{nullptr};

    // Rewrite matmul into a BRGEMM.
    // This allows for additional reduction dimension tiling driven
    // by a microkernel.
    //
    // TODO: Expand heuristics about brgemm rewrite profitability.
    // TODO: Allow for batch dimension.
    int64_t kDim = lhs.getType().getShape().back();
    auto accShape = acc.getType().getShape();
    constexpr int64_t kTile = 32;
    int64_t numTiles = kDim / kTile;
    if (rank == 2 && (kDim % kTile) == 0 && numTiles > 1) {
      // Split reduction dimension into tiles.
      // The number of tiles represents the batch dimension.
      inputs[0] = rewriter.create<memref::ExpandShapeOp>(
          loc, SmallVector<int64_t>{accShape[0], numTiles, kTile}, inputs[0],
          SmallVector<ReassociationIndices>{{0}, {1, 2}});
      inputs[1] = rewriter.create<memref::ExpandShapeOp>(
          loc, SmallVector<int64_t>{numTiles, kTile, accShape[1]}, inputs[1],
          SmallVector<ReassociationIndices>{{0, 1}, {2}});

      // Update maps with BRGEMM indexing.
      auto mapA = AffineMap::getMultiDimMapWithTargets(4, {1, 0, 3}, ctx);
      auto mapB = AffineMap::getMultiDimMapWithTargets(4, {0, 3, 2}, ctx);
      auto mapC = AffineMap::getMultiDimMapWithTargets(4, {1, 2}, ctx);
      indexingMaps = SmallVector<AffineMap>{mapA, mapB, mapC};
    }

    // TODO: Perform this check much earlier before any rewrites.
    auto brgemmInfo = xsmm::utils::isMappableToBrgemm(rewriter, dotOp, inputs,
                                                      outputs, indexingMaps);
    if (failed(brgemmInfo)) {
      assert(false); // FIXME: getMemrefSource above already modified IR...
      // return rewriter.notifyMatchFailure(dotOp, "not mappable to XSMM");
    }

    auto xsmmFuncs = xsmm::utils::buildBrgemmCalls(rewriter, dotOp, inputs,
                                                   indexingMaps, flags);

    if (hoistedAcc) {
      // Hoisting already updated all uses correctly.
      // Only remove the original contraction.
      rewriter.eraseOp(dotOp);
    } else {
      // Load back the result to bring it back to tensor semantics.
      auto loadOp =
          rewriter.create<triton::cpu::LoadOp>(loc, res.getType(), accBuf);
      rewriter.replaceOp(dotOp, loadOp);
    }

    return success();
  }

private:
  ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis;
};

struct ConvertTritonToXsmm
    : public triton::cpu::impl::ConvertTritonToXsmmBase<ConvertTritonToXsmm> {
  using ConvertTritonToXsmmBase::ConvertTritonToXsmmBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    ModuleTensorPtrShapeInfoAnalysis shapeInfoAnalysis(mod);

    RewritePatternSet patterns(context);
    patterns.add<DotToXsmm>(context, shapeInfoAnalysis);
    if (failed(mlir::applyPatternsAndFoldGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
