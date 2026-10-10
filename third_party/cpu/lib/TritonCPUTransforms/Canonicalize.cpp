#include "cpu/include/TritonCPUTransforms/OptCommon.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CANONICALIZE
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// Fold a transfer_read followed by a shape cast. This handles two cases:
// 1. The shape cast removes leading dimensions with size 1.
// 2. The shape cast adds trailing dimensions with size 1.
struct FoldShapeCastIntoTransferRead
    : public OpRewritePattern<vector::ShapeCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto readOp = op.getSource().getDefiningOp<vector::TransferReadOp>();
    if (!readOp || !readOp->hasOneUse() || readOp.hasOutOfBoundsDim() ||
        !readOp.getPermutationMap().isMinorIdentity() || readOp.getMask())
      return failure();

    VectorType srcTy = op.getSourceVectorType();
    VectorType resTy = op.getResultVectorType();

    MLIRContext *context = getContext();
    AffineMap permMap;

    if (srcTy.getRank() > resTy.getRank()) {
      // Check all removed dimensions have size 1.
      if (!all_of(drop_end(srcTy.getShape(), resTy.getRank()),
                  [](int64_t val) { return val == 1; }))
        return failure();

      // Check shape suffix matches the resulting type.
      if (!equal(
              drop_begin(srcTy.getShape(), srcTy.getRank() - resTy.getRank()),
              resTy.getShape()))
        return failure();

      permMap = AffineMap::getMinorIdentityMap(srcTy.getRank(), resTy.getRank(),
                                               context);
    } else if (srcTy.getRank() < resTy.getRank()) {
      // Check all added dimensions have size 1.
      if (!all_of(drop_begin(resTy.getShape(), srcTy.getRank()),
                  [](int64_t val) { return val == 1; }))
        return failure();

      // Check shape prefix matches the source type.
      if (!equal(drop_end(resTy.getShape(), resTy.getRank() - srcTy.getRank()),
                 srcTy.getShape()))
        return failure();

      SmallVector<AffineExpr> exprs(resTy.getRank(),
                                    getAffineConstantExpr(0, context));
      for (unsigned r = 0; r < srcTy.getRank(); ++r)
        exprs[r] = getAffineDimExpr(r, context);

      // Construct a broadcasting map.
      permMap = AffineMap::get(srcTy.getRank(), 0, exprs, context);
    } else
      return failure();

    SmallVector<bool> inBounds(permMap.getNumResults(), true);
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        op, resTy, readOp.getBase(), readOp.getIndices(), readOp.getPadding(),
        permMap, inBounds);

    return success();
  }
};

// Fold a shape cast that only adds leading dimensions with size 1 into a
// transfer write.
struct FoldShapeCastIntoTransferWrite
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp op,
                                PatternRewriter &rewriter) const override {
    if (op.hasOutOfBoundsDim() || !op.getPermutationMap().isMinorIdentity())
      return failure();

    auto castOp = op.getValueToStore().getDefiningOp<vector::ShapeCastOp>();
    if (!castOp || !castOp->hasOneUse())
      return failure();

    VectorType srcTy = castOp.getSourceVectorType();
    VectorType resTy = castOp.getResultVectorType();

    if (srcTy.getRank() >= resTy.getRank())
      return failure();

    // Check all added dimensions have size 1.
    if (!all_of(drop_end(resTy.getShape(), srcTy.getRank()),
                [](int64_t val) { return val == 1; }))
      return failure();

    // Check shape suffix matches the source type.
    if (!equal(drop_begin(resTy.getShape(), resTy.getRank() - srcTy.getRank()),
               srcTy.getShape()))
      return failure();

    SmallVector<bool> inBounds(srcTy.getRank(), true);
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        op, castOp.getSource(), op.getBase(), op.getIndices(), inBounds);

    return success();
  }
};

// Fold a broadcast into a preceding transfer read, expressed through a
// permutation map.
struct FoldBroadcastIntoRead : public OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    vector::TransferReadOp readOp =
        op.getSource().getDefiningOp<vector::TransferReadOp>();
    if (!readOp || !readOp->hasOneUse() || readOp.hasOutOfBoundsDim() ||
        readOp.getMask())
      return failure();

    VectorType srcTy = dyn_cast<VectorType>(op.getSourceType());
    if (!srcTy)
      return failure();
    VectorType resTy = op.getResultVectorType();

    if (srcTy.getRank() > resTy.getRank())
      return failure();

    SmallVector<int64_t> extSrcShape(resTy.getRank() - srcTy.getRank(), 1);
    extSrcShape.append(srcTy.getShape().begin(), srcTy.getShape().end());

    MLIRContext *context = getContext();
    SmallVector<AffineExpr> exprs;
    unsigned dim = 0;
    for (auto [ext, res] : zip_equal(extSrcShape, resTy.getShape())) {
      if (ext == 1)
        exprs.push_back(getAffineConstantExpr(0, context));
      else
        exprs.push_back(getAffineDimExpr(dim++, context));
    }

    auto permMap = AffineMap::get(srcTy.getRank(), 0, exprs, context)
                       .compose(readOp.getPermutationMap());

    SmallVector<bool> inBounds(permMap.getNumResults(), true);
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        readOp, resTy, readOp.getBase(), readOp.getIndices(),
        readOp.getPadding(), permMap, inBounds);

    return success();
  }
};

// Very early in the pipeline, the ReorderBroadcast pass replaces
// elementwise(broadcast(x)) with broadcast(elementwise(x)) when possible. This
// pattern reverses this for casts (arith.extf/si), to surface more
// opportunities to fuse broadcasts into transfer reads.
template <typename CastOp>
struct HoistBroadcastThroughCast : OpRewritePattern<vector::BroadcastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::BroadcastOp op,
                                PatternRewriter &rewriter) const override {
    auto castOp = op.getSource().getDefiningOp<CastOp>();
    if (!castOp || !castOp->hasOneUse())
      return failure();

    VectorType castSrcTy = dyn_cast<VectorType>(castOp.getIn().getType());
    if (!castSrcTy)
      return failure();

    VectorType oldBcastTy = op.getResultVectorType();
    VectorType newBcastTy =
        oldBcastTy.cloneWith(std::nullopt, castSrcTy.getElementType());

    auto newBcastOp = vector::BroadcastOp::create(rewriter, op.getLoc(),
                                                  newBcastTy, castOp.getIn());

    CastOp newCastOp;
    if constexpr (std::is_same_v<CastOp, arith::ExtFOp>)
      newCastOp = CastOp::create(rewriter, castOp.getLoc(), oldBcastTy,
                                 newBcastOp, castOp.getFastMathFlagsAttr());
    else
      newCastOp =
          CastOp::create(rewriter, castOp.getLoc(), oldBcastTy, newBcastOp);

    rewriter.replaceOp(op, newCastOp);
    return success();
  }
};

// This pattern restores the original form (cast before broadcast), to clean up
// any broadcast that wasn't fused into a transfer read.
template <typename CastOp>
struct HoistCastThroughBroadcast : OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CastOp op,
                                PatternRewriter &rewriter) const override {
    auto bcastOp = op.getIn().template getDefiningOp<vector::BroadcastOp>();
    if (!bcastOp || !bcastOp->hasOneUse())
      return failure();

    VectorType bcastSrcTy = dyn_cast<VectorType>(bcastOp.getSourceType());
    if (!bcastSrcTy)
      return failure();

    VectorType oldCastResTy = dyn_cast<VectorType>(op.getType());
    if (!oldCastResTy)
      return failure();

    VectorType newCastResType =
        bcastSrcTy.cloneWith(std::nullopt, oldCastResTy.getElementType());

    CastOp newCastOp;
    if constexpr (std::is_same_v<CastOp, arith::ExtFOp>)
      newCastOp =
          CastOp::create(rewriter, op.getLoc(), newCastResType,
                         bcastOp.getSource(), op.getFastMathFlagsAttr());
    else
      newCastOp = CastOp::create(rewriter, op.getLoc(), newCastResType,
                                 bcastOp.getSource());

    auto newBcastOp = vector::BroadcastOp::create(rewriter, bcastOp.getLoc(),
                                                  oldCastResTy, newCastOp);

    rewriter.replaceOp(op, newBcastOp);
    return success();
  }
};

struct Canonicalize : public triton::cpu::impl::CanonicalizeBase<Canonicalize> {
  Canonicalize() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);
    patterns
        .add<FoldShapeCastIntoTransferRead, FoldShapeCastIntoTransferWrite,
             FoldBroadcastIntoRead, HoistBroadcastThroughCast<arith::ExtFOp>,
             HoistBroadcastThroughCast<arith::ExtSIOp>>(context

        );

    if (failed(applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();

    // Clean-up any broadcasts that weren't fused into transfer reads.
    RewritePatternSet patterns2(context);
    patterns2.add<HoistCastThroughBroadcast<arith::ExtFOp>,
                  HoistCastThroughBroadcast<arith::ExtSIOp>>(context);

    if (failed(applyPatternsGreedily(mod, std::move(patterns2))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createCanonicalize() {
  return std::make_unique<Canonicalize>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
