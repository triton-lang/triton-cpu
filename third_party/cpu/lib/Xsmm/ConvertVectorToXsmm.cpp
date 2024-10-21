//===- ConvertVectorToXsmm.cpp ----------------------------------*- C++ -*-===//
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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

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
#define GEN_PASS_DEF_CONVERTVECTORTOXSMM
#include "cpu/include/Xsmm/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

namespace {

static Value getMemrefSource(PatternRewriter &rewriter, Operation *op,
                             TypedValue<Type> operand) {
  Location loc = op->getLoc();
  MLIRContext *ctx = op->getContext();

  if (isa<MemRefType>(operand.getType()))
    return operand;

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(op);

  if (auto readOp =
          dyn_cast_or_null<vector::TransferReadOp>(operand.getDefiningOp())) {
    VectorType vecTy = readOp.getVectorType();
    SmallVector<int64_t> strides(vecTy.getRank(), 1);
    return rewriter.create<memref::SubViewOp>(
        loc, readOp.getSource(), getAsOpFoldResult(readOp.getIndices()),
        getAsIndexOpFoldResult(ctx, vecTy.getShape()),
        getAsIndexOpFoldResult(ctx, strides));
  }

  auto vecTy = dyn_cast<VectorType>(operand.getType());
  assert(vecTy && "Expect vector type operand");
  MemRefType memTy = MemRefType::get(vecTy.getShape(), vecTy.getElementType());
  auto alloca = rewriter.create<memref::AllocaOp>(loc, memTy);
  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(memTy.getRank(), zeroIdx);
  auto write =
      rewriter.create<vector::TransferWriteOp>(loc, operand, alloca, indices);

  return alloca;
}

// Helper to move accumulation buffer outside of GEMM reduction loop.
// Returns new accumulation buffer or std::nullopt, otherwise.
//
// Rewrites the following pattern:
//   %init = ... vector<...>
//   %0 = scf.for ... iter_args(%acc = %init)
//     %res = GEMM(%A, %B, %acc) -> vector<...>
//     scf.yield %res
//   consumer(%0)
// into:
//   %init = ... vector<...>
//   %hoisted = ... memref<...>
//   store %init, %hoisted
//   %unused = %scf.for ... iter_args(%acc = %init)
//     %res = GEMM(%A, %B, %acc)
//     scf.yield %acc
//   %0 = load(%hoisted) -> vector<...>
//   consumer(%0)
//
// This rewrite should be used as a part of contraction to memref conversion.
static std::optional<Value> hoistAccumulationBuffer(PatternRewriter &rewriter,
                                                    Operation *op,
                                                    TypedValue<Type> operand) {
  Location loc = op->getLoc();

  // Expect the contraction op to still be in vector abstraction.
  auto vecTy = dyn_cast<VectorType>(operand.getType());
  if (!vecTy)
    return std::nullopt;

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
  Value accBuf = getMemrefSource(rewriter, forOp, loopArg->get());

  // For simplicity, feed the iter_arg directly into loop yield terminator.
  // Canonicalizer will folded them away later.
  rewriter.replaceAllUsesWith(res, operand);

  // Replace the corresponding loop result with the latest value read from the
  // accumulation buffer.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPointAfter(forOp);

  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(dyn_cast<MemRefType>(accBuf.getType()).getRank(),
                             zeroIdx);
  auto readOp =
      rewriter.create<vector::TransferReadOp>(loc, vecTy, accBuf, indices);
  rewriter.replaceAllUsesWith(forOp.getTiedLoopResult(blockArg),
                              readOp.getResult());

  return accBuf;
}

struct ContractToXsmm : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    Location loc = contractOp.getLoc();

    TypedValue<VectorType> lhs = contractOp.getLhs();
    TypedValue<VectorType> rhs = contractOp.getRhs();
    TypedValue<Type> acc = contractOp.getAcc();

    auto vecTy = dyn_cast<VectorType>(acc.getType());
    if (!vecTy)
      return rewriter.notifyMatchFailure(contractOp,
                                         "expects to accumulate on vector");

    SmallVector<Attribute> flags;
    Value lhsBuf = getMemrefSource(rewriter, contractOp, lhs);
    Value rhsBuf = getMemrefSource(rewriter, contractOp, rhs);
    std::optional<Value> hoistedAcc =
        hoistAccumulationBuffer(rewriter, contractOp, acc);
    Value accBuf =
        hoistedAcc ? *hoistedAcc : getMemrefSource(rewriter, contractOp, acc);

    SmallVector<Value> inputs{lhsBuf, rhsBuf, accBuf};
    SmallVector<Value> outputs{nullptr};
    auto brgemmInfo =
        xsmm::utils::isMappableToBrgemm(rewriter, contractOp, inputs, outputs,
                                        contractOp.getIndexingMapsArray());
    if (failed(brgemmInfo))
      return rewriter.notifyMatchFailure(contractOp, "not mappable to XSMM");
    if (brgemmInfo->isVnni)
      return rewriter.notifyMatchFailure(contractOp, "VNNI support NYI");

    auto xsmmFuncs = xsmm::utils::buildBrgemmCalls(
        rewriter, contractOp, ValueRange{lhsBuf, rhsBuf, accBuf}, *brgemmInfo,
        flags);

    if (hoistedAcc) {
      // Hoisting already updated all uses correctly.
      // Only remove the original contraction.
      rewriter.eraseOp(contractOp);
    } else {
      // Load back the result to bring it back to vector semantics.
      Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      SmallVector<Value> indices(
          dyn_cast<MemRefType>(accBuf.getType()).getRank(), zeroIdx);
      auto readOp =
          rewriter.create<vector::TransferReadOp>(loc, vecTy, accBuf, indices);
      rewriter.replaceOp(contractOp, readOp);
    }

    return success();
  }
};

struct ConvertVectorToXsmm
    : public triton::cpu::impl::ConvertVectorToXsmmBase<ConvertVectorToXsmm> {
  using ConvertVectorToXsmmBase::ConvertVectorToXsmmBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<ContractToXsmm>(context);
    if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                  std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
