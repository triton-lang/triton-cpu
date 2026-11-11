#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_MAPELEMENTWISETOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;
using namespace mlir::triton;

namespace {

// CPU tensors become vectors, with pointer elements represented by i64. LLVM
// represents the leading dimensions of a multidimensional vector as arrays.
VectorType getCPUVectorType(RankedTensorType tensorType) {
  Type elementType = tensorType.getElementType();
  if (isa<PointerType>(elementType))
    elementType = IntegerType::get(tensorType.getContext(), 64);
  return VectorType::get(tensorType.getShape(), elementType);
}

SmallVector<int64_t> getOuterPosition(int64_t linearPosition,
                                      ArrayRef<int64_t> shape) {
  SmallVector<int64_t> position(shape.size() - 1);
  linearPosition /= shape.back();
  for (int64_t dim = static_cast<int64_t>(position.size()) - 1; dim >= 0;
       --dim) {
    position[dim] = linearPosition % shape[dim];
    linearPosition /= shape[dim];
  }
  return position;
}

struct MapElementwiseOpConversion
    : public OpConversionPattern<triton::MapElementwiseOp> {
  using OpConversionPattern::OpConversionPattern;

  void initialize() {
    // Inlining can expose nested maps. Their nesting depth decreases with each
    // rewrite, so recursively legalizing the cloned operations terminates.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult
  matchAndRewrite(triton::MapElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto *converter = getTypeConverter();
    int64_t numElements =
        cast<RankedTensorType>(op.getOperand(0).getType()).getNumElements();
    int64_t pack = op.getPack();
    if (pack <= 0 || numElements % pack != 0)
      return op.emitOpError(
          "pack size must divide the number of elements in the tensor");

    // Make all block signatures and returned scalars LLVM-compatible before
    // using the shared helper, which joins the region with LLVM branches.
    if (failed(rewriter.convertRegionTypes(&op.getScalarOp(), *converter)))
      return rewriter.notifyMatchFailure(op, "could not convert scalar region");
    for (Block &block : op.getScalarOp()) {
      if (auto ret = dyn_cast<MapElementwiseReturnOp>(block.getTerminator())) {
        SmallVector<Value> values;
        if (failed(rewriter.getRemappedValues(ret.getOperands(), values)))
          return failure();
        rewriter.modifyOpInPlace(ret, [&] { ret->setOperands(values); });
      }
    }

    SmallVector<SmallVector<Value>> inputs;
    for (auto [original, converted] :
         llvm::zip(op.getOperands(), adaptor.getOperands())) {
      auto type = cast<RankedTensorType>(original.getType());
      auto shape = type.getShape();
      int64_t rowSize = shape.empty() ? 1 : shape.back();
      SmallVector<Value> elements;
      for (int64_t i = 0; i < numElements; ++i) {
        Value row = converted;
        if (shape.size() > 1)
          row = LLVM::ExtractValueOp::create(rewriter, loc, converted,
                                             getOuterPosition(i, shape));
        Value position =
            LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                     rewriter.getI32IntegerAttr(i % rowSize));
        Value scalar =
            LLVM::ExtractElementOp::create(rewriter, loc, row, position);
        if (isa<PointerType>(type.getElementType()))
          scalar = LLVM::IntToPtrOp::create(
              rewriter, loc, converter->convertType(type.getElementType()),
              scalar);
        elements.push_back(scalar);
      }
      inputs.push_back(std::move(elements));
    }

    SmallVector<SmallVector<Value>> outputs(op.getNumResults());
    // Like the GPU lowering, inline one scalar region per pack and preserve
    // operand-major arguments and result-major returned values. CPU blocks can
    // contain more packs than a GPU thread; very large maps may need a loop in
    // a future optimization rather than duplicating the scalar body here.
    for (int64_t offset = 0; offset < numElements; offset += pack) {
      SmallVector<Value> arguments;
      for (const auto &elements : inputs)
        llvm::append_range(arguments, ArrayRef(elements).slice(offset, pack));
      auto values = inlineRegion<MapElementwiseReturnOp>(
          rewriter, op.getScalarOp(), arguments, loc);
      for (auto [resultIndex, elements] : llvm::enumerate(outputs))
        llvm::append_range(elements,
                           ArrayRef(values).slice(resultIndex * pack, pack));
    }

    SmallVector<Value> results;
    for (auto [original, elements] : llvm::zip(op.getResults(), outputs)) {
      auto type = cast<RankedTensorType>(original.getType());
      auto shape = type.getShape();
      int64_t rowSize = shape.empty() ? 1 : shape.back();
      auto vectorType = getCPUVectorType(type);
      Type convertedType = converter->convertType(type);
      Type rowType = converter->convertType(
          VectorType::get({rowSize}, vectorType.getElementType()));
      Value result = LLVM::PoisonOp::create(rewriter, loc, convertedType);
      for (int64_t offset = 0; offset < numElements; offset += rowSize) {
        Value row = LLVM::PoisonOp::create(rewriter, loc, rowType);
        for (int64_t i = 0; i < rowSize; ++i) {
          Value scalar = elements[offset + i];
          if (isa<PointerType>(type.getElementType()))
            scalar = LLVM::PtrToIntOp::create(rewriter, loc,
                                              rewriter.getI64Type(), scalar);
          Value position =
              LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                       rewriter.getI32IntegerAttr(i));
          row = LLVM::InsertElementOp::create(rewriter, loc, row, scalar,
                                              position);
        }
        result =
            shape.size() <= 1
                ? row
                : LLVM::InsertValueOp::create(rewriter, loc, result, row,
                                              getOuterPosition(offset, shape));
      }
      results.push_back(result);
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct MapElementwiseToLLVM
    : public triton::impl::MapElementwiseToLLVMBase<MapElementwiseToLLVM> {
  using MapElementwiseToLLVMBase::MapElementwiseToLLVMBase;

  void runOnOperation() override {
    auto maps = getOperation().walk(
        [](MapElementwiseOp) { return WalkResult::interrupt(); });
    if (!maps.wasInterrupted())
      return;

    MLIRContext *context = &getContext();
    LowerToLLVMOptions options(context);
    TritonCPUToLLVMTypeConverter converter(context, options);
    // Only this pass accepts the tensor/vector bridges left around a deferred
    // tt.map_elementwise. Other CPU LLVM passes retain their TTCIR invariant.
    converter.addConversion([&](RankedTensorType type) -> Type {
      return converter.convertType(getCPUVectorType(type));
    });

    ConversionTarget target(*context);
    // Scalar operations cloned by inlineRegion are lowered by subsequent
    // passes. They must be explicitly legal as newly generated IR, too.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<UnrealizedConversionCastOp>();
    target.addIllegalOp<MapElementwiseOp>();
    target.addIllegalDialect<cf::ControlFlowDialect>();

    RewritePatternSet patterns(context);
    patterns.add<MapElementwiseOpConversion>(converter, context);
    cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
    ConversionConfig config;
    // Region cloning must observe converted arguments and return operands,
    // rather than replacements deferred until the conversion completes.
    config.allowPatternRollback = false;
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns), config)))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::cpu {
std::unique_ptr<OperationPass<ModuleOp>> createMapElementwiseToLLVMPass() {
  return std::make_unique<MapElementwiseToLLVM>();
}
} // namespace mlir::triton::cpu
