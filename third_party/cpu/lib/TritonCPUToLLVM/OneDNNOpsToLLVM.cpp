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

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ONEDNNOPSTOLLVM
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

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
    auto ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    std::string dispatchName = "create_transform_ukernel";

    // Value k_int = transformOp.getK();
    // auto val_ty = k_int.getType();
    // if (val_ty.isIndex()) {
    // Value k_int =
    //     rewriter.create<arith::IndexCastOp>(loc, i64_ty, transformOp.getK())
    //         .getResult();
    // }

    // llvm::errs() << "k int: " << k_int << "\n";

    auto transformArgs = SmallVector<Value>{
        adaptor.getK(),     adaptor.getN(),    adaptor.getInLd(),
        adaptor.getOutLd(), adaptor.getInDt(), adaptor.getOutDt()};
    auto transformArgTypes =
        SmallVector<Type>{i64_ty, i64_ty, i64_ty, i64_ty, i64_ty, i64_ty};
    // auto transformArgTypes = SmallVector<Type>{
    //     transformOp.getK().getType(),    transformOp.getN().getType(),
    //     transformOp.getInLd().getType(), transformOp.getOutLd().getType(),
    //     transformOp.getInDt().getType(), transformOp.getOutDt().getType()};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(
            rewriter, dispatchName, transformArgTypes,
            getTypeConverter()->convertType(transformOp.getResult().getType())),
        transformArgs);

    // transformOp.getResult().replaceAllUsesWith(dispatched.getResult());
    // llvm::errs() << "dispatched llvm call: " << dispatched << "\n";

    auto mod = transformOp->getParentOfType<ModuleOp>();

    // llvm::errs() << "[Fail] Mod op: "
    //              << "=============================\n"
    //              << mod << "\n =============================\n";

    // rewriter.replaceAllOpUsesWith(transformOp, dispatched.getResult());
    rewriter.replaceOp(transformOp, dispatched.getResult());
    return success();
  };
};

struct TransformCallConversion : public ConvertOpToLLVMPattern<TransformCall> {
  using ConvertOpToLLVMPattern<TransformCall>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransformCall transformOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // llvm::errs() << "invoke orig op call: " << transformOp << "\n";

    auto loc = transformOp.getLoc();
    auto ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // std::string dispatchName = "create_transform_ukernel";
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
    // llvm::errs() << "invoked llvm call: " << dispatched << "\n";

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
    auto ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    if (brgemmOp.getResult().getUses().empty()) {
      // llvm::errs() << "!!!!!!!!!uses empty!!!!!!!!!\n";
      auto mod = brgemmOp->getParentOfType<ModuleOp>();

      // llvm::errs() << "[Fail] Brgemm op: "
      //              << "=============================\n"
      //              << mod << "\n =============================\n";
    }

    std::string dispatchName = "create_brgemm_ukernel";

    // Value batch_size_int = brgemmOp.getBatchSize();
    // auto val_ty = batch_size_int.getType();
    // if (val_ty.isIndex()) {
    // Value batch_size_int = // brgemmOp.getOperand(3);
    //     rewriter.create<arith::IndexCastOp>(loc, i64_ty,
    //     brgemmOp.getOperand(3))
    //         .getResult();
    // }

    // llvm::errs() << "bs size: " << batch_size_int << "\n";

    // Value ldc_int = brgemmOp.getLdc();
    // val_ty = ldc_int.getType();
    // if (val_ty.isIndex()) {
    // Value ldc_int = // brgemmOp.getOperand(6);
    //     rewriter.create<arith::IndexCastOp>(loc, i64_ty,
    //     brgemmOp.getOperand(6))
    //         .getResult();
    // }

    // llvm::errs() << "ldc: " << ldc_int << "\n";

    auto brgemmArgs = SmallVector<Value>{
        adaptor.getM(),         adaptor.getN(),      adaptor.getKK(),
        adaptor.getBatchSize(), adaptor.getLda(),    adaptor.getLdb(),
        adaptor.getLdc(),       adaptor.getDtypeA(), adaptor.getDtypeB(),
        adaptor.getDtypeC()};
    SmallVector<Type> brgemmArgTypes{i64_ty, i64_ty, i64_ty, i64_ty, i64_ty,
                                     i64_ty, i64_ty, i64_ty, i64_ty, i64_ty};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(
            rewriter, dispatchName, brgemmArgTypes,
            getTypeConverter()->convertType(brgemmOp.getResult().getType())),
        brgemmArgs);

    // brgemmOp.getResult().replaceAllUsesWith(dispatched.getResult());

    // llvm::errs() << "brgem res: " << brgemmOp.getResult() << "\n";
    // llvm::errs() << "brgemm result uses: \n";
    // for (auto &use : brgemmOp.getResult().getUses()) {
    //  llvm::errs() << "\t" << use.get() << "\n";
    //}
    // llvm::errs() << "uses done ------ \n";
    // llvm::errs() << "brgemm dispatched llvm call: " << dispatched << "\n";
    //  rewriter.replaceAllOpUsesWith(brgemmOp, dispatched.getResult());

    rewriter.replaceOp(brgemmOp, dispatched.getResult());
    return success();
  };
};

struct BrgemmCallConversion : public ConvertOpToLLVMPattern<BrgemmCall> {
  using ConvertOpToLLVMPattern<BrgemmCall>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BrgemmCall brgemmOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // llvm::errs() << "invoke orig op call: " << brgemmOp << "\n";

    auto loc = brgemmOp.getLoc();
    auto ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // auto brgem_kernel_params_op =
    //     adaptor.getKernelHash()
    //         .getDefiningOp<triton::cpu::BrgemmCreate>();
    // if (brgem_kernel_params_op == nullptr) {
    //   return failure();
    // }

    // std::string dispatchName = "create_Brgemm_ukernel";
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
    // auto unranked =
    //     getTypeConverter()->convertType(brgemmOp.getOperand(0).getType());
    // auto brgemmArgTypes = SmallVector<Type>{
    //     unranked, unranked, unranked, unranked, unranked,
    // };
    auto brgemmArgTypes =
        SmallVector<Type>{ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx),
                          ptr_ty(ctx), i64_ty,      i64_ty,      i64_ty};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(rewriter, invokeName, brgemmArgTypes, void_ty(ctx)),
        brgemmArgs);
    // llvm::errs() << "invoked llvm call: " << dispatched << "\n";

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
    // llvm::errs() << "invoke orig op call: " << brgemmOp << "\n";

    auto loc = brgemmOp.getLoc();
    auto ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // auto brgem_kernel_params_op =
    //     adaptor.getKernelHash()
    //         .getDefiningOp<triton::cpu::BrgemmCreate>();
    // if (brgem_kernel_params_op == nullptr) {
    //   return failure();
    // }

    // std::string dispatchName = "create_Brgemm_ukernel";
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
        MemRefDescriptor(adaptor.getScratchpad()).alignedPtr(rewriter, loc),
        adaptor.getStepA(),
        adaptor.getStepB(),
        adaptor.getBlockedBsize(),
        adaptor.getNumBatches()};
    // auto unranked =
    //     getTypeConverter()->convertType(brgemmOp.getOperand(0).getType());
    // auto brgemmArgTypes = SmallVector<Type>{
    //     unranked, unranked, unranked, unranked, unranked,
    // };
    auto brgemmArgTypes = SmallVector<Type>{
        ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx), ptr_ty(ctx),
        ptr_ty(ctx), i64_ty,      i64_ty,      i64_ty,      i64_ty};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(rewriter, invokeName, brgemmArgTypes, void_ty(ctx)),
        brgemmArgs);
    // llvm::errs() << "invoked llvm call: " << dispatched << "\n";

    rewriter.replaceOp(brgemmOp, dispatched);
    return success();
  };
};

struct ConfigureHWConversion : public ConvertOpToLLVMPattern<ConfigureHW> {
  using ConvertOpToLLVMPattern<ConfigureHW>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConfigureHW configureHwOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // llvm::errs() << "invoke orig op call: " << configureHwOp << "\n";

    auto loc = configureHwOp.getLoc();
    auto ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // std::string dispatchName = "create_Brgemm_ukernel";
    std::string invokeName = "prepare_hw_context";

    auto configureArgs = SmallVector<Value>{adaptor.getBrgemmKernelHash()};
    auto configureArgTypes = SmallVector<Type>{
        getTypeConverter()->convertType(configureHwOp.getOperand().getType())};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(rewriter, invokeName, configureArgTypes, void_ty(ctx)),
        configureArgs);
    // llvm::errs() << "invoked llvm call: " << dispatched << "\n";

    rewriter.replaceOp(configureHwOp, dispatched);
    return success();
  };
};

struct ReleaseHWConversion : public ConvertOpToLLVMPattern<ReleaseHW> {
  using ConvertOpToLLVMPattern<ReleaseHW>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReleaseHW releaseHwOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // llvm::errs() << "invoke orig op call: " << releaseHwOp << "\n";

    auto loc = releaseHwOp.getLoc();
    auto ctx = rewriter.getContext();
    auto typeConverter = getTypeConverter();

    // std::string dispatchName = "create_Brgemm_ukernel";
    std::string invokeName = "release_hw_context";

    SmallVector<Value> releaseArgs{};
    SmallVector<Type> releaseArgTypes{};

    auto dispatched = LLVM::createLLVMCallOp(
        rewriter, loc,
        getFuncDecl(rewriter, invokeName, releaseArgTypes, void_ty(ctx)),
        releaseArgs);
    // llvm::errs() << "invoked llvm call: " << dispatched << "\n";

    rewriter.replaceOp(releaseHwOp, dispatched);
    return success();
  };
};

struct OneDNNOpsToLLVM
    : public triton::impl::OneDNNOpsToLLVMBase<OneDNNOpsToLLVM> {
  using OneDNNOpsToLLVMBase::OneDNNOpsToLLVMBase;

  OneDNNOpsToLLVM() : OneDNNOpsToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget conversionTarget(*context);

    RewritePatternSet patterns(context);
    // mlir::arith::populateArithToLLVMConversionPatterns(typeConverter,
    // patterns);

    patterns.add<TransformCreateConversion, TransformCallConversion,
                 BrgemmCreateConversion, BrgemmCallConversion,
                 ConfigureHWConversion, ReleaseHWConversion,
                 CallBrgemmWithTransformConversion>(typeConverter);
    // patterns.add<BrgemmOpConversion>(typeConverter);
    // patterns.add<ConfigOpConversion>(typeConverter);

    if (failed(applyPartialConversion(mod, conversionTarget,
                                      std::move(patterns)))) {

      // llvm::errs() << "[Fail] Mod op: "
      //              << "=============================\n"
      //              << mod << "\n =============================\n";
      return signalPassFailure();
    } else {
      // llvm::errs() << "[Succ] Mod op: "
      //              << "=============================\n"
      //              << mod << "\n =============================\n";
    }
  }
};

} // anonymous namespace

namespace mlir::triton::cpu {

std::unique_ptr<OperationPass<ModuleOp>> createOneDNNOpsToLLVMPass() {
  return std::make_unique<OneDNNOpsToLLVM>();
}

} // namespace mlir::triton::cpu
