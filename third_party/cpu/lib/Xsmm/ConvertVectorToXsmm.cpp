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
  OpBuilder::InsertionGuard g(rewriter);

  if (auto readOp =
          dyn_cast_or_null<vector::TransferReadOp>(operand.getDefiningOp())) {
    VectorType vecTy = readOp.getVectorType();
    SmallVector<int64_t> strides(vecTy.getRank(), 1);
    return rewriter.create<memref::SubViewOp>(
        loc, readOp.getSource(), getAsOpFoldResult(readOp.getIndices()),
        getAsIndexOpFoldResult(ctx, vecTy.getShape()),
        getAsIndexOpFoldResult(ctx, strides));
  }

  rewriter.setInsertionPoint(op);

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
    Value accBuf = getMemrefSource(rewriter, contractOp, acc);
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

    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(dyn_cast<MemRefType>(accBuf.getType()).getRank(),
                               zeroIdx);
    auto readOp =
        rewriter.create<vector::TransferReadOp>(loc, vecTy, accBuf, indices);

    rewriter.replaceOp(contractOp, readOp);

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
