#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::triton::cpu {
#define GEN_PASS_DEF_LOWERTENSORDESCRIPTOROPS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace mlir::triton::cpu

using namespace mlir;
using namespace mlir::triton::cpu;

namespace {

Value createBoundsMask(Value base, ValueRange indices, VectorType vectorTy,
                       PatternRewriter &rewriter, Location loc) {
  auto maskTy = vectorTy.clone(rewriter.getI1Type());
  SmallVector<Value> upperBounds, blockShape;
  for (auto [dim, index] : llvm::enumerate(indices)) {
    Value extent = rewriter.createOrFold<memref::DimOp>(loc, base, dim);
    upperBounds.push_back(arith::SubIOp::create(rewriter, loc, extent, index));
    blockShape.push_back(arith::ConstantIndexOp::create(
        rewriter, loc, vectorTy.getDimSize(dim)));
  }
  Value mask = vector::CreateMaskOp::create(rewriter, loc, maskTy, upperBounds);
  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value allTrue = arith::ConstantOp::create(
      rewriter, loc, maskTy,
      SplatElementsAttr::get(maskTy, rewriter.getBoolAttr(true)));
  for (auto [dim, index] : llvm::enumerate(indices)) {
    // create_mask(shape - index) checks the upper bounds only. Exclude each
    // dimension's negative prefix as well, including rank-reduced dimensions.
    SmallVector<Value> lowerBounds(blockShape);
    lowerBounds[dim] = arith::SubIOp::create(rewriter, loc, zero, index);
    Value belowZero =
        vector::CreateMaskOp::create(rewriter, loc, maskTy, lowerBounds);
    Value nonnegative =
        arith::XOrIOp::create(rewriter, loc, belowZero, allTrue);
    mask = arith::AndIOp::create(rewriter, loc, mask, nonnegative);
  }
  return mask;
}

struct LowerDescriptorLoad : OpRewritePattern<DescriptorLoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto vectorTy = cast<VectorType>(op.getType());
    auto baseTy = cast<MemRefType>(op.getBase().getType());
    auto readTy = vectorTy;
    if (!op.getAssumeInBounds() && vectorTy.getRank() < baseTy.getRank()) {
      SmallVector<int64_t> shape(baseTy.getRank(), 1);
      llvm::copy(vectorTy.getShape(),
                 shape.begin() + baseTy.getRank() - vectorTy.getRank());
      readTy = VectorType::get(shape, vectorTy.getElementType());
    }
    Value mask;
    if (!op.getAssumeInBounds())
      mask = createBoundsMask(op.getBase(), op.getIndices(), readTy, rewriter,
                              loc);
    auto map = AffineMap::getMinorIdentityMap(
        baseTy.getRank(), readTy.getRank(), rewriter.getContext());
    Value result = vector::TransferReadOp::create(
        rewriter, loc, readTy, op.getBase(), op.getIndices(),
        AffineMapAttr::get(map), op.getPadding(), mask,
        rewriter.getBoolArrayAttr(SmallVector<bool>(readTy.getRank(), true)));
    if (readTy != vectorTy)
      result = vector::ShapeCastOp::create(rewriter, loc, vectorTy, result);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct LowerDescriptorStore : OpRewritePattern<DescriptorStoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto vectorTy = cast<VectorType>(op.getValue().getType());
    Value mask;
    if (!op.getAssumeInBounds())
      mask = createBoundsMask(op.getBase(), op.getIndices(), vectorTy, rewriter,
                              loc);
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        op, op.getValue(), op.getBase(), op.getIndices(),
        AffineMapAttr::get(rewriter.getMultiDimIdentityMap(vectorTy.getRank())),
        mask,
        rewriter.getBoolArrayAttr(SmallVector<bool>(vectorTy.getRank(), true)));
    return success();
  }
};

struct LowerTensorDescriptorOps
    : triton::cpu::impl::LowerTensorDescriptorOpsBase<
          LowerTensorDescriptorOps> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerDescriptorLoad, LowerDescriptorStore>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::cpu::createLowerTensorDescriptorOps() {
  return std::make_unique<LowerTensorDescriptorOps>();
}
