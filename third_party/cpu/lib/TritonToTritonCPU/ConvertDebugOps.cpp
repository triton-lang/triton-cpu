#include "TypeConverter.h"

#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTDEBUGOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

class DebugOpsConversionTarget : public ConversionTarget {
public:
  explicit DebugOpsConversionTarget(MLIRContext &ctx, TypeConverter &converter)
      : ConversionTarget(ctx) {
    addLegalDialect<mlir::BuiltinDialect>();
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();

    addLegalOp<arith::ConstantOp>();
    addLegalOp<memref::AllocOp>();
    addLegalOp<memref::DeallocOp>();
    addLegalOp<memref::CastOp>();

    addIllegalOp<triton::PrintOp>();
    addIllegalOp<triton::AssertOp>();
  }
};

struct PrintOpConversion : public OpConversionPattern<triton::PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    // It lowers to triton_cpu.print after converting tensor types to vectors.
    // (tt.print doesn't accept vector types, so we have this intermediate op.)
    if (op.getNumOperands() == 0) {
      triton::cpu::PrintOp::create(rewriter, loc, op.getPrefix(), op.getHex(),
                                   ValueRange{}, llvm::SmallVector<int, 0>{});
      rewriter.eraseOp(op);
      return success();
    }

    for (size_t i = 0; i < op.getNumOperands(); i++) {
      Value operand = op.getOperands()[i];
      auto isSigned = {op.getIsSigned()[i]};
      if (!isa<RankedTensorType>(operand.getType())) {
        triton::cpu::PrintOp::create(rewriter, loc, op.getPrefix(), op.getHex(),
                                     rewriter.getRemappedValue(operand),
                                     isSigned);
        continue;
      }

      auto tensorTy = cast<RankedTensorType>(operand.getType());
      auto elemTy = tensorTy.getElementType();
      if (isa<triton::PointerType>(elemTy)) {
        elemTy = rewriter.getI64Type();
      }
      MemRefType memRefTy = MemRefType::get(tensorTy.getShape(), elemTy);

      Value allocVal = memref::AllocOp::create(rewriter, loc, memRefTy,
                                               rewriter.getI64IntegerAttr(64));

      Value vec = rewriter.getRemappedValue(operand);
      VectorType vecTy = cast<VectorType>(vec.getType());

      Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
      SmallVector<Value> indices(vecTy.getRank(), zeroIdx);

      vector::TransferWriteOp::create(rewriter, loc, vec, allocVal, indices);

      Value allocUnrankedVal = memref::CastOp::create(
          rewriter, loc,
          UnrankedMemRefType::get(elemTy, memRefTy.getMemorySpace()), allocVal);

      triton::cpu::PrintOp::create(rewriter, loc, op.getPrefix(), op.getHex(),
                                   allocUnrankedVal, isSigned);

      memref::DeallocOp::create(rewriter, loc, allocVal);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct AssertOpConversion : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value acc = arith::ConstantOp::create(rewriter, loc, i1_ty,
                                          rewriter.getOneAttr(i1_ty));
    Value condition = rewriter.getRemappedValue(op.getCondition());
    SmallVector<bool> dimsToReduce(
        cast<VectorType>(condition.getType()).getRank(), true);
    condition = vector::MultiDimReductionOp::create(rewriter, loc, condition,
                                                    acc, dimsToReduce,
                                                    vector::CombiningKind::AND);
    rewriter.replaceOpWithNewOp<triton::cpu::AssertOp>(op, condition,
                                                       op.getMessage());
    return success();
  }
};

struct ConvertDebugOps
    : public triton::impl::ConvertDebugOpsBase<ConvertDebugOps> {
  using ConvertDebugOpsBase::ConvertDebugOpsBase;

  ConvertDebugOps() : ConvertDebugOpsBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    TritonToTritonCPUTypeConverter typeConverter;
    DebugOpsConversionTarget convTarget(*context, typeConverter);
    RewritePatternSet patterns(context);
    patterns.add<PrintOpConversion>(typeConverter, context);
    patterns.add<AssertOpConversion>(typeConverter, context);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDebugOps() {
  return std::make_unique<ConvertDebugOps>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
