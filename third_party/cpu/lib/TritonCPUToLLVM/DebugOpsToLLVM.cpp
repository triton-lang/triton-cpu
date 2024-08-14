#include "TypeConverter.h"
#include "Utility.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUOps.h.inc"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_DEBUGOPSTOLLVM
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

// TODO: This code is the same as the GPU-backend code. Consider refactoring.
std::string getFormatSubstr(Value value, bool hex = false,
                            std::optional<int> width = std::nullopt) {
  Type type = value.getType();
  if (isa<LLVM::LLVMPointerType>(type)) {
    return "%p";
  }
  // Hex is "0x%0nx" or "0x%0nllx", where n is the number of hex digits in the
  // type (so 4 for fp16, 8 for int32, 16 for int64).
  if (hex) {
    // Ignore `width` for `hex` values, pad to typeWidth.
    std::string ret = "0x%0" + std::to_string(type.getIntOrFloatBitWidth() / 4);
    if (type.getIntOrFloatBitWidth() > 32) {
      ret += "ll";
    }
    ret += "x";
    return ret;
  }

  std::string prefix = "%";
  if (width.has_value()) {
    prefix += std::to_string(*width);
  } else if (hex) {
    prefix += "0";
    prefix += std::to_string(value.getType().getIntOrFloatBitWidth() / 4);
  }

  if (type.isBF16() || type.isF16() || type.isF32() || type.isF64()) {
    return prefix + "f";
  } else if (type.isSignedInteger()) {
    if (type.getIntOrFloatBitWidth() == 64)
      return prefix + "lli";
    else
      return prefix + "i";
  } else if (type.isUnsignedInteger() || type.isSignlessInteger()) {
    if (type.getIntOrFloatBitWidth() == 64)
      return prefix + "llu";
    else
      return prefix + "u";
  }
  assert(false && "not supported type");
  return "";
}

LLVM::LLVMFuncOp getPrintFuncDecl(ConversionPatternRewriter &rewriter,
                                  bool printf) {
  auto moduleOp = rewriter.getBlock()->getParent()->getParentOfType<ModuleOp>();
  StringRef funcName = printf ? "printf" : "triton_vector_print";
  Operation *funcOp = moduleOp.lookupSymbol(funcName);
  if (funcOp)
    return cast<LLVM::LLVMFuncOp>(*funcOp);

  auto *ctx = rewriter.getContext();
  SmallVector<Type> argsType;
  if (printf)
    argsType = {ptr_ty(ctx)};
  else
    argsType = {i32_ty,      i32_ty, i32_ty, ptr_ty(ctx),
                ptr_ty(ctx), i32_ty, i32_ty, i64_ty};

  auto funcType =
      LLVM::LLVMFunctionType::get(i32_ty, argsType, /*isVarArg*/ printf);

  ConversionPatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(moduleOp.getBody());

  return rewriter.create<LLVM::LLVMFuncOp>(UnknownLoc::get(ctx), funcName,
                                           funcType);
}

void llPrintf(StringRef prefix, std::array<Value, 3> pid,
              std::optional<Value> arg, ConversionPatternRewriter &rewriter,
              bool hex = false) {
  assert(!prefix.empty() && "printf with empty string not supported");
  auto loc = UnknownLoc::get(rewriter.getContext());

  std::string formatStr;
  llvm::raw_string_ostream os(formatStr);
  os << "(" << getFormatSubstr(pid[0]) << ", " << getFormatSubstr(pid[1])
     << ", " << getFormatSubstr(pid[2]) << ")" << prefix;
  if (arg.has_value())
    os << getFormatSubstr(arg.value(), hex);

  llvm::SmallString<64> formatStrNewline(formatStr);
  formatStrNewline.push_back('\n');
  formatStrNewline.push_back('\0');
  Value formatStrValue =
      LLVM::addStringToModule(loc, rewriter, "printfFormat_", formatStrNewline);

  SmallVector<Value> allArgs{formatStrValue};
  for (auto elem : pid)
    allArgs.push_back(elem);
  if (arg.has_value())
    allArgs.push_back(arg.value());
  call(getPrintFuncDecl(rewriter, true), allArgs);
}

void llVectorPrint(std::array<Value, 3> pid, StringRef prefix, Value ptr,
                   bool isInteger, uint32_t bitWidth, int64_t numElem,
                   ConversionPatternRewriter &rewriter) {
  assert(!prefix.empty());
  auto loc = UnknownLoc::get(rewriter.getContext());

  llvm::SmallString<64> prefixStr(prefix);
  prefixStr.push_back('\0');
  Value prefixValue =
      LLVM::addStringToModule(loc, rewriter, "vectorPrintPrefix_", prefixStr);

  SmallVector<Value> allArgs;
  for (auto elem : pid)
    allArgs.push_back(elem);
  allArgs.push_back(prefixValue);
  allArgs.push_back(ptr);
  allArgs.push_back(i32_val(isInteger));
  allArgs.push_back(i32_val(bitWidth));
  allArgs.push_back(i64_val(numElem));
  call(getPrintFuncDecl(rewriter, false), allArgs);
}

bool usePrintf(triton::cpu::PrintOp op) {
  // Simply use printf if no operand or the operand is scalar.
  if (op.getNumOperands() == 0)
    return true;

  // tt.print is already decomposed to triton_cpu.print per value.
  assert(op.getNumOperands() == 1);
  Type oprType = op.getOperands()[0].getType();
  return (oprType.isIntOrIndexOrFloat() || isa<triton::PointerType>(oprType));
}

struct PrintOpConversion : public ConvertOpToLLVMPattern<triton::cpu::PrintOp> {
  using ConvertOpToLLVMPattern<triton::cpu::PrintOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::cpu::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    auto getPid = [&](int axis) {
      return getProgramId(op->getParentOfType<LLVM::LLVMFuncOp>(), axis);
    };
    std::array<Value, 3> pid = {getPid(0), getPid(1), getPid(2)};

    if (usePrintf(op)) {
      if (op.getNumOperands() == 0) {
        llPrintf(op.getPrefix(), pid, std::nullopt, rewriter);
      } else {
        Value llOpr = adaptor.getOperands()[0];
        llPrintf(op.getPrefix(), pid, llOpr, rewriter, op.getHex());
      }
    } else {
      Value llOpr = adaptor.getOperands()[0];
      auto vecShapedType = cast<ShapedType>(op.getOperands()[0].getType());
      // Currently, we only support 1D vector printing.
      if (vecShapedType.getRank() == 1) {

        // To get the pointer of the vector, create an alloca and store it.
        auto ptrType = ptr_ty(rewriter.getContext());
        auto ptr = rewriter.create<LLVM::AllocaOp>(loc, ptrType,
                                                   llOpr.getType(), i32_val(1));
        rewriter.create<LLVM::StoreOp>(loc, llOpr, ptr);

        // TODO: Consider passing an encoded element type information instead of
        // booleans and separate bit width.
        llVectorPrint(pid, op.getPrefix(), ptr,
                      vecShapedType.getElementType().isInteger(),
                      vecShapedType.getElementTypeBitWidth(),
                      vecShapedType.getNumElements(), rewriter);
      } else {
        // TODO: support 2D+ vector printing.
        std::string msg{op.getPrefix()};
        llvm::raw_string_ostream os(msg);
        os << "<<not implemented for '" << llOpr.getType() << "'>>";
        llPrintf(msg, pid, std::nullopt, rewriter);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

using BarrierOp = mlir::gpu::BarrierOp;

// This is part of the DebugOps pass because gpu::barrier is generated by
// tl.debug_barrier.
struct BarrierOpConversion : public ConvertOpToLLVMPattern<BarrierOp> {
  explicit BarrierOpConversion(LLVMTypeConverter &typeConverter)
      : mlir::ConvertOpToLLVMPattern<BarrierOp>(typeConverter) {}

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Just make it a no-op for now
    rewriter.eraseOp(op);
    return success();
  }
};

struct DebugOpsToLLVM
    : public triton::impl::DebugOpsToLLVMBase<DebugOpsToLLVM> {
  using DebugOpsToLLVMBase::DebugOpsToLLVMBase;

  DebugOpsToLLVM() : DebugOpsToLLVMBase() {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mlir::LowerToLLVMOptions option(context);
    TritonCPUToLLVMTypeConverter typeConverter(context, option);
    TritonLLVMConversionTarget convTarget(*context);

    RewritePatternSet patterns(context);
    patterns.add<PrintOpConversion>(typeConverter);
    patterns.add<BarrierOpConversion>(typeConverter);
    // patterns.add<AssertOpConversion>(typeConverter);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // anonymous namespace

namespace mlir::triton::cpu {

std::unique_ptr<OperationPass<ModuleOp>> createDebugOpsToLLVMPass() {
  return std::make_unique<DebugOpsToLLVM>();
}

} // namespace mlir::triton::cpu
