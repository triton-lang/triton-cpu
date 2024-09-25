#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

Value getProgramId(mlir::FunctionOpInterface funcOp, int axis) {
  auto args = funcOp.getArguments();
  assert(funcOp && args.size() >= 8);
  assert(axis >= 0 && axis < 3);

  // The first three of the last 8 args are x, y, z program ids.
  auto argIdx = args.size() - 8 + axis;
  assert(argIdx < args.size() && "out-of-bounds arg index");
  assert(args[argIdx].getType().isInteger(32) && "unexpected arg type");
  return args[argIdx];
}

struct GetProgramIdOpConversion
    : public ConvertOpToLLVMPattern<triton::GetProgramIdOp> {
  explicit GetProgramIdOpConversion(LLVMTypeConverter &typeConverter,
                                    const TargetInfoBase &targetInfo,
                                    PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<triton::GetProgramIdOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (triton::gpu::isCPUMode()) {
      auto funcOp = op->getParentOfType<FunctionOpInterface>();
      assert(funcOp && "expected LLVM::FuncOp as a parent of GetProgramIdOp");
      rewriter.replaceOp(op, getProgramId(funcOp, op.getAxisAsInt()));
      return success();
    }
    Value programId = targetInfo.programId(rewriter, op->getLoc(),
                                           op->getParentOfType<ModuleOp>(),
                                           op.getAxisAsInt());
    rewriter.replaceOp(op, programId);
    return success();
  }

private:
  const TargetInfoBase &targetInfo;
};

} // namespace

void mlir::triton::populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               const TargetInfoBase &targetInfo,
                                               PatternBenefit benefit) {
  patterns.add<GetProgramIdOpConversion>(typeConverter, targetInfo, benefit);
}
