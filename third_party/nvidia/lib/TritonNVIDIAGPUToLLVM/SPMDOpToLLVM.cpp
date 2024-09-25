#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

namespace {

using namespace mlir;
using namespace mlir::triton;

Value getNumPrograms(mlir::FunctionOpInterface funcOp, int axis) {
  auto args = funcOp.getArguments();
  assert(funcOp && args.size() >= 8);
  assert(axis >= 0 && axis < 3);

  // The first three of the last 8 args are x, y, z program ids.
  auto argIdx = args.size() - 5 + axis;
  assert(argIdx < args.size() && "out-of-bounds arg index");
  assert(args[argIdx].getType().isInteger(32) && "unexpected arg type");
  return args[argIdx];
}

struct GetNumProgramsOpConversion
    : public ConvertOpToLLVMPattern<triton::GetNumProgramsOp> {
  using ConvertOpToLLVMPattern<
      triton::GetNumProgramsOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (triton::gpu::isCPUMode()) {
      auto funcOp = op->getParentOfType<FunctionOpInterface>();
      assert(funcOp && "expected LLVM::FuncOp as a parent of GetProgramIdOp");
      rewriter.replaceOp(op, getNumPrograms(funcOp, op.getAxisAsInt()));
      return success();
    }

    // It is not easy to get the compute capability here, so we use numCTAs to
    // decide the semantic of GetNumProgramsOp. If numCTAs = 1, then
    // GetNumProgramsOp is converted to "%nctaid", otherwise it is converted to
    // "%nclusterid".
    auto moduleOp = op->getParentOfType<ModuleOp>();
    assert(moduleOp && "Parent ModuleOp not found for GetProgramIdOp");
    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(moduleOp);

    Location loc = op->getLoc();
    assert(op.getAxisAsInt() < 3);
    std::string sreg = numCTAs == 1 ? "%nctaid." : "%nclusterid.";
    sreg.append(1, 'x' + op.getAxisAsInt()); // 0 -> 'x', 1 -> 'y', 2 -> 'z'

    Value numPrograms = LLVM::NVIDIA::getSRegValue(rewriter, loc, sreg);
    rewriter.replaceOp(op, numPrograms);
    return success();
  }
};

} // namespace

void mlir::triton::NVIDIA::populateSPMDOpToLLVMPattern(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetNumProgramsOpConversion>(typeConverter, benefit);
}
