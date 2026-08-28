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

// Fold transfer read and the following shape cast that removes heading
// dimensions with size 1.
struct FoldReadShapeCast : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    if (!op->hasOneUse())
      return failure();

    auto permMap = op.getPermutationMap();
    if (!permMap.isMinorIdentity())
      return failure();

    auto reshape = dyn_cast<vector::ShapeCastOp>(*op->user_begin());
    if (!reshape)
      return failure();

    VectorType ty = cast<VectorType>(op.getType());
    VectorType dstTy = cast<VectorType>(reshape.getType());
    if (ty.getRank() <= dstTy.getRank())
      return failure();

    // Check all removed dimensions have size 1.
    if (!all_of(drop_end(ty.getShape(), dstTy.getRank()),
                [](int64_t val) { return val == 1; }))
      return failure();

    // Check shape prefix matches the resulting type.
    if (!equal(drop_begin(ty.getShape(), ty.getRank() - dstTy.getRank()),
               dstTy.getShape()))
      return failure();

    auto inBounds = op.getInBounds();
    if (std::any_of(inBounds.begin(), inBounds.end() - dstTy.getRank(),
                    [](Attribute attr) {
                      return !cast<mlir::BoolAttr>(attr).getValue();
                    }))
      return failure();

    // Fold read and shape cast into a single read.
    auto newPermMap = permMap.getMinorIdentityMap(
        permMap.getNumDims(), dstTy.getRank(), getContext());
    auto newInBounds = rewriter.getArrayAttr(SmallVector<Attribute>(drop_begin(
        op.getInBounds().getValue(), ty.getRank() - dstTy.getRank())));
    auto newRead = vector::TransferReadOp::create(
        rewriter, loc, dstTy, op.getBase(), op.getIndices(), newPermMap,
        op.getPadding(), op.getMask(), newInBounds);
    rewriter.replaceOp(reshape, newRead);
    rewriter.eraseOp(op);

    return success();
  }
};

// Fold a fully in-bounds contiguous multi-dimensional transfer read followed
// by a flattening shape cast into a rank-1 transfer read. This intentionally
// handles only statically provable row-major reads without masks.
struct FlattenContiguousReadShapeCast
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse() || op.getMask())
      return failure();

    auto reshape = dyn_cast<vector::ShapeCastOp>(*op->user_begin());
    if (!reshape)
      return failure();

    auto srcTy = dyn_cast<VectorType>(op.getType());
    auto dstTy = dyn_cast<VectorType>(reshape.getType());
    auto memrefTy = dyn_cast<MemRefType>(op.getBase().getType());
    if (!srcTy || !dstTy || !memrefTy || srcTy.getRank() <= 1 ||
        dstTy.getRank() != 1 ||
        srcTy.getElementType() != dstTy.getElementType() ||
        srcTy.getNumElements() != dstTy.getNumElements())
      return failure();

    auto permMap = op.getPermutationMap();
    if (!permMap.isMinorIdentity() ||
        permMap.getNumResults() != srcTy.getRank() ||
        memrefTy.getRank() < srcTy.getRank())
      return failure();

    auto inBounds = op.getInBounds();
    if (inBounds.size() != srcTy.getRank() ||
        llvm::any_of(inBounds, [](Attribute attr) {
          return !cast<BoolAttr>(attr).getValue();
        }))
      return failure();

    SmallVector<int64_t> strides;
    int64_t offset;
    if (failed(memrefTy.getStridesAndOffset(strides, offset)))
      return failure();

    // The minor memref dimensions read by the transfer must be large enough
    // for the vector tile and have row-major strides for every dimension that
    // advances an address. Size-one vector dimensions do not constrain their
    // corresponding memref stride because their sole index is zero.
    int64_t expectedStride = 1;
    int64_t memrefDim = memrefTy.getRank() - 1;
    for (int64_t vectorDim = srcTy.getRank() - 1; vectorDim >= 0;
         --vectorDim, --memrefDim) {
      int64_t vectorSize = srcTy.getDimSize(vectorDim);
      int64_t memrefSize = memrefTy.getDimSize(memrefDim);
      if (ShapedType::isDynamic(vectorSize) ||
          (!ShapedType::isDynamic(memrefSize) && memrefSize < vectorSize))
        return failure();
      if (vectorSize > 1 &&
          (strides[memrefDim] == ShapedType::kDynamic ||
           strides[memrefDim] != expectedStride))
        return failure();
      expectedStride *= vectorSize;
    }

    auto newPermMap = AffineMap::getMinorIdentityMap(
        permMap.getNumDims(), 1, rewriter.getContext());
    auto newInBounds = rewriter.getBoolArrayAttr({true});
    auto newRead = vector::TransferReadOp::create(
        rewriter, op.getLoc(), dstTy, op.getBase(), op.getIndices(),
        newPermMap, op.getPadding(), Value(), newInBounds);
    rewriter.replaceOp(reshape, newRead);
    rewriter.eraseOp(op);
    return success();
  }
};

struct Canonicalize : public triton::cpu::impl::CanonicalizeBase<Canonicalize> {
  Canonicalize() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<FoldReadShapeCast, FlattenContiguousReadShapeCast>(context);

    if (failed(mlir::applyPatternsGreedily(mod, std::move(patterns))))
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
