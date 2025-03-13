#include "TypeConverter.h"
#include "Utility.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUOps.h.inc"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#if defined(XSMM_AVAILABLE)
#include "libxsmm_typedefs.h"
#endif
void assert_on_xsmm_missing() {
#if !defined(XSMM_AVAILABLE)
  assert(false && "No XSMM with uKernels available. Pass will be redundant.");
#endif
}

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_UKERNELOPSTOXSMMLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

inline Value intLLVMConst(Location loc, Type ty, int64_t val,
                          PatternRewriter &rewriter) {
  return rewriter.create<LLVM::ConstantOp>(
      loc, IntegerAttr::get(getElementTypeOrSelf(ty), val));
}

static inline int64_t getXSMMDataTypeVal(Type ty) {
#if defined(XSMM_AVAILABLE)
  ty = getElementTypeOrSelf(ty);

  // Float types
  if (ty.isF32())
    return static_cast<int64_t>(LIBXSMM_DATATYPE_F32);
  if (ty.isBF16())
    return static_cast<int64_t>(LIBXSMM_DATATYPE_BF16);
  if (ty.isF16())
    return static_cast<int64_t>(LIBXSMM_DATATYPE_F16);
#endif
  assert_on_xsmm_missing();
  llvm_unreachable("Unexpected type for conversion to XSMM type.");
}

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

LLVM::LLVMFuncOp getFuncDecl(ConversionPatternRewriter &rewriter,
                             StringRef funcName, SmallVector<Type> argsType,
                             Type resultType) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *ctx = rewriter.getContext();

  auto funcType =
      LLVM::LLVMFunctionType::get(resultType, argsType, /*isVarArg*/ false);

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                           funcType);
}

struct BrgemmCreateConversion : public ConvertOpToLLVMPattern<BrgemmCreate> {
  using ConvertOpToLLVMPattern<BrgemmCreate>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BrgemmCreate brgemmOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = brgemmOp.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

    std::string dispatchName = "xsmm_brgemm_dispatch";

    if (getElementTypeOrSelf(brgemmOp.getDtypeA()) !=
        getElementTypeOrSelf(brgemmOp.getDtypeB()))
      return rewriter.notifyMatchFailure(
          brgemmOp, "expects the same element type for input operands");

    auto inXsmmType = b.i64_val(getXSMMDataTypeVal(adaptor.getDtypeA()));
    auto outXsmmType = b.i64_val(getXSMMDataTypeVal(adaptor.getDtypeC()));

    auto brgemmArgs = SmallVector<Value>{adaptor.getM(),     adaptor.getN(),
                                         adaptor.getKK(),    adaptor.getLda(),
                                         adaptor.getLdb(),   adaptor.getLdc(),
                                         adaptor.getStepA(), adaptor.getStepB(),
                                         inXsmmType,         outXsmmType};
    SmallVector<Type> brgemmArgTypes{i64_ty, i64_ty, i64_ty, i64_ty, i64_ty,
                                     i64_ty, i64_ty, i64_ty, i64_ty, i64_ty};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(
            rewriter, dispatchName, brgemmArgTypes,
            getTypeConverter()->convertType(brgemmOp.getResult().getType())),
        brgemmArgs);

    rewriter.replaceOp(brgemmOp, dispatched.getResult());
    return success();
  };
};

struct BrgemmExecuteConversion : public ConvertOpToLLVMPattern<BrgemmExecute> {
  using ConvertOpToLLVMPattern<BrgemmExecute>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BrgemmExecute brgemmOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = brgemmOp.getLoc();
    auto ctx = rewriter.getContext();

    std::string invokeName = "xsmm_brgemm_invoke";

    auto brgemm_kernel_hash_ptr = rewriter.create<LLVM::IntToPtrOp>(
        loc, ptr_ty(ctx), adaptor.getBrgemmKernelHash());

    auto brgemmArgs = SmallVector<Value>{
        brgemm_kernel_hash_ptr,
        MemRefDescriptor(adaptor.getAPtr())
            .bufferPtr(rewriter, loc, *getTypeConverter(),
                       cast<MemRefType>(brgemmOp.getAPtr().getType())),
        MemRefDescriptor(adaptor.getBPtr())
            .bufferPtr(rewriter, loc, *getTypeConverter(),
                       cast<MemRefType>(brgemmOp.getBPtr().getType())),
        MemRefDescriptor(adaptor.getCPtr())
            .bufferPtr(rewriter, loc, *getTypeConverter(),
                       cast<MemRefType>(brgemmOp.getCPtr().getType())),
        adaptor.getNumBatches()};

    auto brgemmArgTypes = SmallVector<Type>{ptr_ty(ctx), ptr_ty(ctx),
                                            ptr_ty(ctx), ptr_ty(ctx), i64_ty};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(rewriter, invokeName, brgemmArgTypes, void_ty(ctx)),
        brgemmArgs);

    rewriter.replaceOp(brgemmOp, dispatched);
    return success();
  };
};

struct UkernelOpsToXSMMLLVM
    : public triton::cpu::impl::UkernelOpsToXSMMLLVMBase<UkernelOpsToXSMMLLVM> {
  using UkernelOpsToXSMMLLVMBase::UkernelOpsToXSMMLLVMBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget conversionTarget(*context);

    RewritePatternSet patterns(context);

    patterns.add<BrgemmCreateConversion, BrgemmExecuteConversion>(
        typeConverter);

    if (failed(applyPartialConversion(mod, conversionTarget,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // anonymous namespace

namespace mlir::triton::cpu {

std::unique_ptr<OperationPass<ModuleOp>> createUkernelOpsToXSMMLLVMPass() {
  return std::make_unique<UkernelOpsToXSMMLLVM>();
}

} // namespace mlir::triton::cpu
