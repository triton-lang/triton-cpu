#include "TypeConverter.h"
#include "Utility.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/VectorPattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_FUNCOPTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct FuncOpConversion : public ConvertOpToLLVMPattern<triton::FuncOp> {
  FuncOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern(converter, benefit) {}

  /// Only retain those attributes that are not constructed by
  /// `LLVMFuncOp::build`. If `filterArgAttrs` is set, also filter out argument
  /// attributes.
  static void filterFuncAttributes(triton::FuncOp op, bool filterArgAttrs,
                                   SmallVectorImpl<NamedAttribute> &result) {

    for (const auto &attr : op->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == op.getFunctionTypeAttrName() ||
          attr.getName() == "std.varargs" ||
          (filterArgAttrs && attr.getName() == op.getArgAttrsAttrName()))
        continue;
      result.push_back(attr);
    }
  }

  triton::FuncOp amendProgramIdArgs(triton::FuncOp funcOp,
                                    ConversionPatternRewriter &rewriter) const {
    auto loc = funcOp.getLoc();
    auto ctx = funcOp->getContext();
    SmallVector<Type, cpu::kNumProgramContextArgs> programContextTypes = {
        i32_ty, i32_ty, i32_ty, ui32_ty, ui32_ty, ui32_ty};

    // 1. Modify the function type to add new arguments.
    auto funcTy = funcOp.getFunctionType();
    auto amendedInputTy = llvm::to_vector<4>(funcTy.getInputs());
    amendedInputTy.append(programContextTypes);
    auto amendedFuncTy = FunctionType::get(funcTy.getContext(), amendedInputTy,
                                           funcTy.getResults());
    // 2. Modify the argument attributes to add new arguments.
    SmallVector<NamedAttribute> amendedAttrs;
    filterFuncAttributes(funcOp, /*filterArgAttrs=*/true, amendedAttrs);
    SmallVector<Attribute> amendedArgAttrs;
    if (funcOp.getAllArgAttrs()) {
      amendedArgAttrs = llvm::to_vector<4>(funcOp.getAllArgAttrs());
      amendedArgAttrs.append(cpu::kNumProgramContextArgs,
                             DictionaryAttr::get(ctx));
      amendedAttrs.push_back(
          rewriter.getNamedAttr(funcOp.getArgAttrsAttrName(),
                                rewriter.getArrayAttr(amendedArgAttrs)));
    }
    // 3. Add new arguments to the region.
    auto amendedFuncOp =
        triton::FuncOp::create(rewriter, funcOp.getLoc(), funcOp.getName(),
                               amendedFuncTy, amendedAttrs);
    auto &region = funcOp.getBody();
    for (Type type : programContextTypes)
      region.addArgument(type, loc);
    rewriter.inlineRegionBefore(region, amendedFuncOp.getBody(),
                                amendedFuncOp.end());
    return amendedFuncOp;
  }

  LogicalResult
  matchAndRewrite(triton::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Defined Triton functions need pid and num_programs arguments because
    // operations such as program_id and print may also occur in subfunctions.
    bool isExternal = funcOp.isExternal();
    auto modifiedFuncOp = funcOp;
    if (!isExternal)
      modifiedFuncOp = amendProgramIdArgs(modifiedFuncOp, rewriter);

    LLVM::LLVMFuncOp newFuncOp = *mlir::convertFuncOpToLLVMFuncOp(
        modifiedFuncOp, rewriter, *getTypeConverter());
    if (!newFuncOp)
      return failure();

    // Workaround: Prevent LLVM codegen from emitting the .prefalign directive,
    // which not all assemblers support.
    newFuncOp.setAlignment(128);

    if (!isExternal)
      rewriter.eraseOp(modifiedFuncOp);
    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct ReturnOpConversion : public ConvertOpToLLVMPattern<triton::ReturnOp> {
  using ConvertOpToLLVMPattern<triton::ReturnOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::ReturnOp newOp;
    if (adaptor.getOperands().size() < 2) {
      // Single or no return value.
      newOp =
          LLVM::ReturnOp::create(rewriter, op.getLoc(), adaptor.getOperands());
    } else {
      // Pack the results into a struct.
      auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
      auto packedResultsTy = this->getTypeConverter()->packFunctionResults(
          funcOp.getResultTypes());
      Value packedResults =
          LLVM::UndefOp::create(rewriter, op.getLoc(), packedResultsTy);
      auto loc = op.getLoc();
      auto b = TritonLLVMOpBuilder(loc, rewriter);
      for (auto it : llvm::enumerate(adaptor.getOperands())) {
        packedResults = b.insert_val(packedResultsTy, packedResults, it.value(),
                                     it.index());
      }
      newOp = LLVM::ReturnOp::create(rewriter, op.getLoc(), packedResults);
    }
    newOp->setAttrs(op->getAttrs());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

// CallOpInterfaceLowering is adapted from
// https://github.com/llvm/llvm-project/blob/fae656b2dd80246c3c6f01e9c77c49560368752c/mlir/lib/Conversion/FuncToLLVM/FuncToLLVM.cpp#L485
struct CallOpConversion : public ConvertOpToLLVMPattern<triton::CallOp> {
  CallOpConversion(LLVMTypeConverter &converter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::CallOp>(converter, benefit) {}

  LogicalResult
  matchAndRewrite(triton::CallOp callOp,
                  typename triton::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto promotedOperands = promoteOperands(callOp, adaptor, rewriter);
    if (failed(promotedOperands))
      return failure();
    auto newCallOp =
        convertCallOpToLLVMCallOp(callOp, *promotedOperands, rewriter);
    if (!newCallOp)
      return failure();
    auto results = getCallOpResults(callOp, newCallOp, rewriter);
    rewriter.replaceOp(callOp, results);
    return success();
  }

private:
  FailureOr<SmallVector<Value, 4>>
  promoteOperands(triton::CallOp callOp,
                  typename triton::CallOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const {
    auto promotedOperands = this->getTypeConverter()->promoteOperands(
        callOp.getLoc(), /*opOperands=*/callOp->getOperands(),
        adaptor.getOperands(), rewriter);

    auto callee = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
        callOp, callOp.getCalleeAttr());
    if (!callee) {
      callOp.emitOpError("expected an LLVM function callee");
      return failure();
    }
    if (callee.isExternal())
      return promotedOperands;

    auto caller = callOp->getParentOfType<LLVM::LLVMFuncOp>();
    if (!caller) {
      callOp.emitOpError("expected an enclosing LLVM function");
      return failure();
    }

    // FuncOpConversion runs before CallOpConversion, so this call's enclosing
    // function should already have the program-context arguments appended.
    auto callerArgs = caller.getArguments();
    if (callerArgs.size() < cpu::kNumProgramContextArgs) {
      callOp.emitOpError("caller is missing pid and num_programs arguments");
      return failure();
    }

    auto expectedNumArgs =
        promotedOperands.size() + cpu::kNumProgramContextArgs;
    if (callee.getFunctionType().getNumParams() != expectedNumArgs) {
      callOp.emitOpError("callee has an unexpected number of arguments");
      return failure();
    }

    // Forward pid_x/y/z and num_programs_x/y/z to the Triton subfunction.
    for (unsigned i = callerArgs.size() - cpu::kNumProgramContextArgs;
         i < callerArgs.size(); ++i)
      promotedOperands.push_back(callerArgs[i]);

    return promotedOperands;
  }

  LLVM::CallOp
  convertCallOpToLLVMCallOp(triton::CallOp callOp,
                            ArrayRef<Value> promotedOperands,
                            ConversionPatternRewriter &rewriter) const {
    // Pack the result types into a struct.
    Type packedResult = nullptr;
    unsigned numResults = callOp.getNumResults();
    auto resultTypes = llvm::to_vector<4>(callOp.getResultTypes());

    if (numResults != 0) {
      if (!(packedResult =
                this->getTypeConverter()->packFunctionResults(resultTypes)))
        return nullptr;
    }
    auto newCallOp = LLVM::CallOp::create(rewriter, callOp.getLoc(),
                                          packedResult ? TypeRange(packedResult)
                                                       : TypeRange(),
                                          promotedOperands, callOp->getAttrs());
    newCallOp.getProperties().setOpBundleSizes(
        rewriter.getDenseI32ArrayAttr({}));
    newCallOp.getProperties().setOperandSegmentSizes(
        {static_cast<int>(promotedOperands.size()), 0});
    return newCallOp;
  }

  SmallVector<Value>
  getCallOpResults(triton::CallOp callOp, LLVM::CallOp newCallOp,
                   ConversionPatternRewriter &rewriter) const {
    auto numResults = callOp.getNumResults();
    SmallVector<Value> results;
    if (numResults < 2) {
      // If < 2 results, packing did not do anything and we can just return.
      results.append(newCallOp.result_begin(), newCallOp.result_end());
    } else {
      // Otherwise, it had been converted to an operation producing a structure.
      // Extract individual results from the structure and return them as list.
      results.reserve(numResults);
      for (unsigned i = 0; i < numResults; ++i) {
        results.push_back(LLVM::ExtractValueOp::create(
            rewriter, callOp.getLoc(), newCallOp->getResult(0), i));
      }
    }
    return results;
  }
};

struct FuncOpToLLVM : public triton::impl::FuncOpToLLVMBase<FuncOpToLLVM> {
  using FuncOpToLLVMBase::FuncOpToLLVMBase;

  FuncOpToLLVM() : FuncOpToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    // Lower tt.func
    RewritePatternSet funcPatterns(context);
    funcPatterns.add<FuncOpConversion>(typeConverter,
                                       /*benefit=*/1);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          funcPatterns);
    if (failed(
            applyPartialConversion(mod, convTarget, std::move(funcPatterns))))
      return signalPassFailure();

    // Lower tt.call, tt.return
    int benefit = 10;
    RewritePatternSet patterns(context);
    patterns.add<ReturnOpConversion>(typeConverter, benefit);
    patterns.add<CallOpConversion>(typeConverter, benefit);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createFuncOpToLLVMPass() {
  return std::make_unique<FuncOpToLLVM>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
