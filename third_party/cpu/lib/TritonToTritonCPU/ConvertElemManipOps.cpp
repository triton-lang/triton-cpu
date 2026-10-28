#include "OpTypeConversion.h"
#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTELEMMANIPOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class ElemManipOpConversionTarget : public ConversionTarget {
public:
  explicit ElemManipOpConversionTarget(MLIRContext &ctx,
                                       TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<math::MathDialect>();
    addLegalDialect<ub::UBDialect>();
    addLegalDialect<memref::MemRefDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Deferred maps must retain store-free regions. Convert their tensor
    // manipulation operations after map expansion in the LLVM stage.
    addLegalOp<triton::MapElementwiseOp>();
    markOpRecursivelyLegal<triton::MapElementwiseOp>();

    addIllegalOp<triton::BroadcastOp>();
    addIllegalOp<triton::ExpandDimsOp>();
    addIllegalOp<triton::ReshapeOp>();
    addIllegalOp<triton::TransOp>();
    addIllegalOp<triton::JoinOp>();
    addIllegalOp<triton::SplitOp>();
    addIllegalOp<triton::GatherOp>();
  }
};

struct ReshapeOpConversion : public OpConversionPattern<triton::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(isa<RankedTensorType>(op.getType()));
    auto loc = op.getLoc();
    auto src = rewriter.getRemappedValue(op.getSrc());
    auto srcShape = dyn_cast<VectorType>(src.getType()).getShape();
    auto resTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(op.getType()));
    auto dstShape = resTy.getShape();
    auto elemTy = resTy.getElementType();

    // There are restrictions on how shape can be modified by ShapeCastOp
    // when rank is changed. For now, we simply detect it and handle through
    // a cast to 1D vector. Better solution may be required later.
    if (canCastShape(srcShape, dstShape)) {
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
          op, VectorType::get(dstShape, elemTy), src);
    } else {
      SmallVector<int64_t> tmpShape({resTy.getNumElements()});
      auto tmp = vector::ShapeCastOp::create(
          rewriter, loc, VectorType::get(tmpShape, elemTy), src);
      rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(
          op, VectorType::get(dstShape, elemTy), tmp);
    }
    return success();
  }

private:
  bool canCastShape(ArrayRef<int64_t> src, ArrayRef<int64_t> dst) const {
    if (src.size() == dst.size())
      return true;
    if (src.size() > dst.size())
      return canCastShape(dst, src);

    size_t srcIdx = 0;
    size_t dstIdx = 0;
    while (srcIdx < src.size() && dstIdx < dst.size()) {
      if (src[srcIdx] == 1) {
        ++srcIdx;
      } else {
        // Source dim size should be a product of continuous dest dim sizes.
        int64_t srcSize = src[srcIdx++];
        int64_t dstSize = dst[dstIdx++];
        while (dstSize < srcSize && dstIdx < dst.size())
          dstSize *= dst[dstIdx++];
        if (dstSize != srcSize)
          return false;
      }
    }

    // Skip trailing 1s.
    while (srcIdx < src.size() && src[srcIdx] == 1)
      ++srcIdx;
    while (dstIdx < dst.size() && dst[dstIdx] == 1)
      ++dstIdx;

    return srcIdx == src.size() && dstIdx == dst.size();
  }
};

struct TransOpConversion : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto val = rewriter.getRemappedValue(op.getSrc());
    auto order = op.getOrder();
    SmallVector<int64_t> permutation(order.begin(), order.end());
    rewriter.replaceOpWithNewOp<vector::TransposeOp>(op, val, permutation);
    return success();
  }
};

struct JoinOpConversion : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto lhs = rewriter.getRemappedValue(op.getLhs());
    auto rhs = rewriter.getRemappedValue(op.getRhs());
    auto interleave = vector::InterleaveOp::create(rewriter, loc, lhs, rhs);
    // JoinOp creates a new dimension, but InterleaveOp doubles the final one.
    // Use ShapeCastOp to get the required shape.
    auto resTy = getTypeConverter()->convertType(op.getType());
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, resTy, interleave);
    return success();
  }
};

struct SplitOpConversion : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = rewriter.getRemappedValue(op.getSrc());
    auto srcTy = cast<VectorType>(src.getType());
    auto resTy = getTypeConverter()->convertType(op.getType(0));
    assert(srcTy.getShape().back() == 2);

    SmallVector<Value> results;
    if (srcTy.getRank() == 1) {
      results.push_back(vector::ExtractOp::create(rewriter, loc, src, 0));
      results.push_back(vector::ExtractOp::create(rewriter, loc, src, 1));
      rewriter.replaceOp(op, results);
    } else {
      SmallVector<int64_t> tmpShape(srcTy.getShape().drop_back());
      tmpShape.back() *= 2;
      auto tmp = vector::ShapeCastOp::create(
          rewriter, loc, VectorType::get(tmpShape, srcTy.getElementType()),
          src);
      rewriter.replaceOpWithNewOp<vector::DeinterleaveOp>(op, tmp);
    }
    return success();
  }
};

struct GatherOpConversion : public OpConversionPattern<triton::GatherOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto allocationScope = op->getParentOfType<triton::FuncOp>();
    if (!allocationScope)
      return rewriter.notifyMatchFailure(op,
                                         "requires a parent Triton function");
    auto loc = op.getLoc();
    Value src = adaptor.getSrc();
    Value indices = adaptor.getIndices();
    auto srcTy = cast<VectorType>(src.getType());
    auto indicesTy = cast<VectorType>(indices.getType());
    auto resTy =
        cast<VectorType>(getTypeConverter()->convertType(op.getType()));

    // Scalar memory accesses use one byte per sub-byte integer element, while
    // vector transfers pack them. Materialize these sources as bytes instead.
    Type storageElemTy = srcTy.getElementType();
    if (auto intTy = dyn_cast<IntegerType>(storageElemTy);
        intTy && intTy.getWidth() < 8)
      storageElemTy = rewriter.getI8Type();
    auto srcStorageTy = VectorType::get(srcTy.getShape(), storageElemTy);
    auto resStorageTy = VectorType::get(resTy.getShape(), storageElemTy);
    if (srcStorageTy != srcTy)
      src = arith::ExtUIOp::create(rewriter, loc, srcStorageTy, src);

    // The pinned vector.gather requires a memref or tensor base. Allocate its
    // fixed scratch buffer once at function entry, including for CFG loops.
    // Retain vector ranks to avoid building a very wide flat vector.
    Value buffer;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&allocationScope.getBody().front());
      buffer = memref::AllocaOp::create(
          rewriter, loc, MemRefType::get(srcTy.getShape(), storageElemTy),
          rewriter.getI64IntegerAttr(64));
    }
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> positions(srcTy.getRank(), zero);
    SmallVector<bool> inBounds(srcTy.getRank(), true);
    vector::TransferWriteOp::create(rewriter, loc, src, buffer, positions,
                                    inBounds);
    Value flatBuffer = buffer;
    if (srcTy.getRank() > 1) {
      SmallVector<ReassociationIndices> reassociation(1);
      for (int64_t dim = 0; dim < srcTy.getRank(); ++dim)
        reassociation[0].push_back(dim);
      flatBuffer =
          memref::CollapseShapeOp::create(rewriter, loc, buffer, reassociation);
    }

    // A flat gather must include the unchanged coordinates in every other
    // dimension. Keep its indices in the original result shape so downstream
    // vector lowering can select native-sized gathers without a flat reshape.
    auto indexTy = VectorType::get(resTy.getShape(), rewriter.getI64Type());
    if (indicesTy != indexTy)
      indices = arith::ExtUIOp::create(rewriter, loc, indexTy, indices);
    auto splatIndex = [&](int64_t value) -> Value {
      return arith::ConstantOp::create(
          rewriter, loc, indexTy,
          SplatElementsAttr::get(indexTy, rewriter.getI64IntegerAttr(value)));
    };
    Value validAxis =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::ult, indices,
                              splatIndex(srcTy.getDimSize(op.getAxis())));
    // Use a valid source element for invalid lanes so the memory gather can
    // remain unmasked. Masked gathers may otherwise scalarize into a large
    // control-flow graph on targets without native gather instructions.
    Value safeIndices = arith::SelectOp::create(rewriter, loc, validAxis,
                                                indices, splatIndex(0));

    SmallVector<int64_t> srcStrides(srcTy.getRank(), 1);
    SmallVector<int64_t> resStrides(resTy.getRank(), 1);
    for (int64_t dim = srcTy.getRank() - 2; dim >= 0; --dim) {
      srcStrides[dim] = srcStrides[dim + 1] * srcTy.getDimSize(dim + 1);
      resStrides[dim] = resStrides[dim + 1] * resTy.getDimSize(dim + 1);
    }
    Value linearIndices = safeIndices;
    if (srcStrides[op.getAxis()] != 1)
      linearIndices = arith::MulIOp::create(
          rewriter, loc, linearIndices, splatIndex(srcStrides[op.getAxis()]));

    SmallVector<int64_t> nonAxisOffsets(resTy.getNumElements(), 0);
    for (int64_t dim = 0; dim < resTy.getRank(); ++dim) {
      if (dim == op.getAxis())
        continue;
      for (int64_t i = 0; i < resTy.getNumElements(); ++i)
        nonAxisOffsets[i] +=
            ((i / resStrides[dim]) % resTy.getDimSize(dim)) * srcStrides[dim];
    }
    Value offsets = arith::ConstantOp::create(
        rewriter, loc, indexTy,
        DenseIntElementsAttr::get(indexTy, ArrayRef<int64_t>(nonAxisOffsets)));
    linearIndices =
        arith::AddIOp::create(rewriter, loc, linearIndices, offsets);

    // Keep invalid lanes poison after gathering only from valid scratch-buffer
    // addresses, without introducing a faulting access into a pure operation.
    auto maskTy = VectorType::get(resTy.getShape(), rewriter.getI1Type());
    Value allTrue = arith::ConstantOp::create(
        rewriter, loc, maskTy, DenseElementsAttr::get(maskTy, true));
    Value passthrough = ub::PoisonOp::create(rewriter, loc, resStorageTy);
    Value result = vector::GatherOp::create(
        rewriter, loc, resStorageTy, flatBuffer, ValueRange{zero},
        linearIndices, allTrue, passthrough);
    result =
        arith::SelectOp::create(rewriter, loc, validAxis, result, passthrough);
    if (resStorageTy != resTy)
      result = arith::TruncIOp::create(rewriter, loc, resTy, result);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertElemManipOps
    : public triton::impl::ConvertElemManipOpsBase<ConvertElemManipOps> {
  using ConvertElemManipOpsBase::ConvertElemManipOpsBase;

  ConvertElemManipOps() : ConvertElemManipOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    ElemManipOpConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<OpTypeConversion<triton::BroadcastOp, vector::BroadcastOp>>(
        typeConverter, context);
    patterns.add<OpTypeConversion<triton::ExpandDimsOp, vector::ShapeCastOp>>(
        typeConverter, context);
    patterns.add<ReshapeOpConversion>(typeConverter, context);
    patterns.add<TransOpConversion>(typeConverter, context);
    patterns.add<JoinOpConversion>(typeConverter, context);
    patterns.add<SplitOpConversion>(typeConverter, context);
    patterns.add<GatherOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertElemManipOps() {
  return std::make_unique<ConvertElemManipOps>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
