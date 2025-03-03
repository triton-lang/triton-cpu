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

#if defined(ONEDNN_AVAILABLE)
#include "oneapi/dnnl/dnnl_types.h"
#endif

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_UKERNELOPSTOONEDNNLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

#if defined(ONEDNN_AVAILABLE)
#include "oneapi/dnnl/dnnl_config.h"
#endif
void assert_on_onednn_missing() {
#if !defined(DNNL_EXPERIMENTAL_UKERNEL)
  assert(false && "No OneDNN with uKernels available. Pass will be redundant.");
#endif
}

inline Value intLLVMConst(Location loc, Type ty, int64_t val,
                          PatternRewriter &rewriter) {
  return rewriter.create<LLVM::ConstantOp>(
      loc, IntegerAttr::get(getElementTypeOrSelf(ty), val));
}

static inline int64_t getDnnlDataTypeVal(Type ty) {
#if defined(DNNL_EXPERIMENTAL_UKERNEL)
  ty = getElementTypeOrSelf(ty);
  if (ty.isF32())
    return static_cast<int64_t>(dnnl_f32);
  if (ty.isF64())
    return static_cast<int64_t>(dnnl_f64);
  if (ty.isBF16())
    return static_cast<int64_t>(dnnl_bf16);
  if (ty.isF16())
    return static_cast<int64_t>(dnnl_f16);
#endif
  assert_on_onednn_missing();
  llvm_unreachable("Unexpected type for conversion to DNNL type.");
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

    std::string dispatchName = "create_brgemm";

    auto lhsDnnType = b.i64_val(getDnnlDataTypeVal(adaptor.getDtypeA()));
    auto rhsDnnType = b.i64_val(getDnnlDataTypeVal(adaptor.getDtypeB()));
    auto accDnnType = b.i64_val(getDnnlDataTypeVal(adaptor.getDtypeC()));

    auto brgemmArgs = SmallVector<Value>{adaptor.getM(),
                                         adaptor.getN(),
                                         adaptor.getKK(),
                                         adaptor.getBatchSize(),
                                         adaptor.getLda(),
                                         adaptor.getLdb(),
                                         adaptor.getLdc(),
                                         lhsDnnType,
                                         rhsDnnType,
                                         accDnnType,
                                         adaptor.getSkipPacking()};
    SmallVector<Type> brgemmArgTypes{i64_ty, i64_ty, i64_ty, i64_ty,
                                     i64_ty, i64_ty, i64_ty, i64_ty,
                                     i64_ty, i64_ty, i1_ty};

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

    std::string invokeName = "brgemm_execute";

    auto brgemm_kernel_hash_ptr = rewriter.create<LLVM::IntToPtrOp>(
        loc, ptr_ty(ctx), adaptor.getBrgemmKernelHash());

    auto brgemmArgs = SmallVector<Value>{
        // tf_kernel_hash_ptr,
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
        adaptor.getStepA(),
        adaptor.getStepB(),
        adaptor.getBlockedBsize(),
        adaptor.getNumBatches(),
        adaptor.getSkipPacking()};

    auto brgemmArgTypes = SmallVector<Type>{
        ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx), i64_ty,
        i64_ty,      i64_ty,      i64_ty,      i1_ty};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(rewriter, invokeName, brgemmArgTypes, void_ty(ctx)),
        brgemmArgs);

    rewriter.replaceOp(brgemmOp, dispatched);
    return success();
  };
};

struct UkernelOpsToOneDNNLLVM
    : public triton::cpu::impl::UkernelOpsToOneDNNLLVMBase<
          UkernelOpsToOneDNNLLVM> {
  UkernelOpsToOneDNNLLVM() = default;

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

std::unique_ptr<OperationPass<ModuleOp>> createUkernelOpsToOneDNNLLVMPass() {
  return std::make_unique<UkernelOpsToOneDNNLLVM>();
}

} // namespace mlir::triton::cpu
