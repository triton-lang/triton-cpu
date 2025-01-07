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
#define GEN_PASS_DEF_ONEDNNOPSTOLLVM
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

struct TransformCreateConversion
    : public ConvertOpToLLVMPattern<TransformCreate> {
  using ConvertOpToLLVMPattern<TransformCreate>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransformCreate transformOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = transformOp.getLoc();

    std::string dispatchName = "create_transform_ukernel";

    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
    auto inDnnType = intLLVMConst(
        loc, integer64, getDnnlDataTypeVal(transformOp.getInDt()), rewriter);
    auto outDnnType = intLLVMConst(
        loc, integer64, getDnnlDataTypeVal(transformOp.getOutDt()), rewriter);
    auto transformArgs = SmallVector<Value>{
        adaptor.getK(),     adaptor.getN(), adaptor.getInLd(),
        adaptor.getOutLd(), inDnnType,      outDnnType};
    auto transformArgTypes =
        SmallVector<Type>{i64_ty, i64_ty, i64_ty, i64_ty, i64_ty, i64_ty};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(
            rewriter, dispatchName, transformArgTypes,
            getTypeConverter()->convertType(transformOp.getResult().getType())),
        transformArgs);

    rewriter.replaceOp(transformOp, dispatched.getResult());
    return success();
  };
};

struct TransformCallConversion : public ConvertOpToLLVMPattern<TransformCall> {
  using ConvertOpToLLVMPattern<TransformCall>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransformCall transformOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = transformOp.getLoc();
    auto ctx = rewriter.getContext();

    std::string invokeName = "call_transform";

    auto transformArgs =
        SmallVector<Value>{adaptor.getKernelHash(), adaptor.getInPtr(),
                           adaptor.getOutBlockedPtr()};
    auto transformArgTypes = SmallVector<Type>{
        adaptor.getKernelHash().getType(), adaptor.getInPtr().getType(),
        adaptor.getOutBlockedPtr().getType()};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(rewriter, invokeName, transformArgTypes, void_ty(ctx)),
        transformArgs);

    rewriter.replaceOp(transformOp, dispatched);
    return success();
  };
};

struct BrgemmCreateConversion : public ConvertOpToLLVMPattern<BrgemmCreate> {
  using ConvertOpToLLVMPattern<BrgemmCreate>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BrgemmCreate brgemmOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = brgemmOp.getLoc();
    IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);

    std::string dispatchName = "create_brgemm_ukernel";

    auto lhsDnnType = intLLVMConst(
        loc, integer64, getDnnlDataTypeVal(adaptor.getDtypeA()), rewriter);
    auto rhsDnnType = intLLVMConst(
        loc, integer64, getDnnlDataTypeVal(adaptor.getDtypeB()), rewriter);
    auto accDnnType = intLLVMConst(
        loc, integer64, getDnnlDataTypeVal(adaptor.getDtypeC()), rewriter);

    auto brgemmArgs =
        SmallVector<Value>{adaptor.getM(),   adaptor.getN(),
                           adaptor.getKK(),  adaptor.getBatchSize(),
                           adaptor.getLda(), adaptor.getLdb(),
                           adaptor.getLdc(), lhsDnnType,
                           rhsDnnType,       accDnnType};
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

struct BrgemmCallConversion : public ConvertOpToLLVMPattern<BrgemmCall> {
  using ConvertOpToLLVMPattern<BrgemmCall>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BrgemmCall brgemmOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = brgemmOp.getLoc();
    auto ctx = rewriter.getContext();

    std::string invokeName = "call_brgemm";

    auto kernel_hash_ptr = rewriter.create<LLVM::IntToPtrOp>(
        loc, ptr_ty(ctx), adaptor.getKernelHash());

    auto brgemmArgs = SmallVector<Value>{
        kernel_hash_ptr,
        MemRefDescriptor(adaptor.getAPtr()).alignedPtr(rewriter, loc),
        MemRefDescriptor(adaptor.getBPtr()).alignedPtr(rewriter, loc),
        MemRefDescriptor(adaptor.getCPtr()).alignedPtr(rewriter, loc),
        MemRefDescriptor(adaptor.getScratchpad()).alignedPtr(rewriter, loc),
        adaptor.getStepA(),
        adaptor.getStepB(),
        adaptor.getNumBatches()};

    auto brgemmArgTypes =
        SmallVector<Type>{ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx),
                          ptr_ty(ctx), i64_ty,      i64_ty,      i64_ty};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(rewriter, invokeName, brgemmArgTypes, void_ty(ctx)),
        brgemmArgs);

    rewriter.replaceOp(brgemmOp, dispatched);
    return success();
  };
};

struct CallBrgemmWithTransformConversion
    : public ConvertOpToLLVMPattern<CallBrgemmWithTransform> {
  using ConvertOpToLLVMPattern<CallBrgemmWithTransform>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CallBrgemmWithTransform brgemmOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = brgemmOp.getLoc();
    auto ctx = rewriter.getContext();

    std::string invokeName = "call_all";

    auto tf_kernel_hash_ptr = rewriter.create<LLVM::IntToPtrOp>(
        loc, ptr_ty(ctx), adaptor.getTransformKernelHash());
    auto brgemm_kernel_hash_ptr = rewriter.create<LLVM::IntToPtrOp>(
        loc, ptr_ty(ctx), adaptor.getBrgemmKernelHash());

    auto brgemmArgs = SmallVector<Value>{
        tf_kernel_hash_ptr,
        brgemm_kernel_hash_ptr,
        MemRefDescriptor(adaptor.getAPtr()).alignedPtr(rewriter, loc),
        MemRefDescriptor(adaptor.getBPtr()).alignedPtr(rewriter, loc),
        MemRefDescriptor(adaptor.getCPtr()).alignedPtr(rewriter, loc),
        adaptor.getStepA(),
        adaptor.getStepB(),
        adaptor.getBlockedBsize(),
        adaptor.getNumBatches()};

    auto brgemmArgTypes = SmallVector<Type>{
        ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx),
        i64_ty,      i64_ty,      i64_ty,      i64_ty};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(rewriter, invokeName, brgemmArgTypes, void_ty(ctx)),
        brgemmArgs);

    rewriter.replaceOp(brgemmOp, dispatched);
    return success();
  };
};

struct ConfigureHWConversion : public ConvertOpToLLVMPattern<ConfigureHW> {
  using ConvertOpToLLVMPattern<ConfigureHW>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConfigureHW configureHwOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = configureHwOp.getLoc();
    auto ctx = rewriter.getContext();

    std::string invokeName = "prepare_hw_context";

    auto configureArgs = SmallVector<Value>{adaptor.getBrgemmKernelHash()};
    auto configureArgTypes = SmallVector<Type>{
        getTypeConverter()->convertType(configureHwOp.getOperand().getType())};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(rewriter, invokeName, configureArgTypes, void_ty(ctx)),
        configureArgs);

    rewriter.replaceOp(configureHwOp, dispatched);
    return success();
  };
};

struct ReleaseHWConversion : public ConvertOpToLLVMPattern<ReleaseHW> {
  using ConvertOpToLLVMPattern<ReleaseHW>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReleaseHW releaseHwOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = releaseHwOp.getLoc();
    auto ctx = rewriter.getContext();

    std::string invokeName = "release_hw_context";

    SmallVector<Value> releaseArgs{};
    SmallVector<Type> releaseArgTypes{};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(rewriter, invokeName, releaseArgTypes, void_ty(ctx)),
        releaseArgs);

    rewriter.replaceOp(releaseHwOp, dispatched);
    return success();
  };
};

struct OneDNNOpsToLLVM
    : public triton::cpu::impl::OneDNNOpsToLLVMBase<OneDNNOpsToLLVM> {
  OneDNNOpsToLLVM() = default;
  OneDNNOpsToLLVM(bool canReplace) { this->canReplace = canReplace; }

  void runOnOperation() override {
    if (!canReplace) {
      LDBG("Pass disabled.");
      return;
    }

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget conversionTarget(*context);

    RewritePatternSet patterns(context);

    patterns.add<TransformCreateConversion, TransformCallConversion,
                 BrgemmCreateConversion, BrgemmCallConversion,
                 ConfigureHWConversion, ReleaseHWConversion,
                 CallBrgemmWithTransformConversion>(typeConverter);

    if (failed(applyPartialConversion(mod, conversionTarget,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // anonymous namespace

namespace mlir::triton::cpu {

std::unique_ptr<OperationPass<ModuleOp>> createOneDNNOpsToLLVMPass() {
  return std::make_unique<OneDNNOpsToLLVM>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createOneDNNOpsToLLVMPass(bool isReplacementToOneDnnPossible) {
  if (isReplacementToOneDnnPossible) {
    assert_on_onednn_missing();
  }
  return std::make_unique<OneDNNOpsToLLVM>(isReplacementToOneDnnPossible);
}

} // namespace mlir::triton::cpu
