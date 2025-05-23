#include "TypeConverter.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_MATHTOVECLIB
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

template <typename OpT> struct VecOpToFp32 : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  VecOpToFp32(MLIRContext *context) : OpRewritePattern<OpT>(context) {}

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy)
      return failure();

    Type elemTy = vecTy.getElementType();
    if (!elemTy.isBF16() && !elemTy.isF16())
      return failure();

    Type fp32VecTy = vecTy.cloneWith(std::nullopt, rewriter.getF32Type());
    SmallVector<Value> fp32Ops;
    for (auto operand : op->getOperands())
      fp32Ops.push_back(
          rewriter.create<arith::ExtFOp>(loc, fp32VecTy, operand));
    auto newOp = rewriter.create<OpT>(loc, fp32VecTy, fp32Ops);
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, vecTy, newOp);
    return success();
  }
};

// Decompose vector operation to single-dimensional vector operations
// with a AVX512 for x86 or NEON for ARM.
template <typename OpT>
struct DecomposeToNativeVecs : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;
  // CPU SIMD vector size in bits
  size_t vec_bits;

  DecomposeToNativeVecs(MLIRContext *context,
                        size_t native_vec_size_in_bits = 512)
      : OpRewritePattern<OpT>(context), vec_bits(native_vec_size_in_bits) {}

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy)
      return failure();

    Type elemTy = vecTy.getElementType();
    if (!elemTy.isF32() && !elemTy.isF64())
      return failure();

    int64_t numElems = vecTy.getNumElements();
    if (numElems * elemTy.getIntOrFloatBitWidth() < 128)
      return failure();

    // Produce a new shape where trailing dimensions wouldn't exceed the native
    // vector size.
    auto shape = vecTy.getShape();
    SmallVector<int64_t> newShape(1, 1);
    int64_t elemsPerVec = vec_bits / elemTy.getIntOrFloatBitWidth();
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
      int64_t size = shape[i];
      if (newShape.size() > 1) {
        newShape.insert(newShape.begin(), size);
      } else {
        int64_t combined = newShape[0] * size;
        if (combined > elemsPerVec) {
          newShape[0] = elemsPerVec;
          newShape.insert(newShape.begin(), combined / elemsPerVec);
        } else {
          newShape[0] = combined;
        }
      }
    }
    if (newShape == shape)
      return failure();

    // Convert input operand to the new shape.
    SmallVector<Value> reshapedInputs;
    for (auto operand : op->getOperands()) {
      auto operandTy = cast<VectorType>(operand.getType());
      auto newOperandTy = VectorType::get(newShape, operandTy.getElementType());
      reshapedInputs.push_back(
          rewriter.create<vector::ShapeCastOp>(loc, newOperandTy, operand));
    }

    // Decompose the original operation to a set of operations on native
    // vectors.
    auto newOpTy = VectorType::get(newShape, elemTy);
    auto subResTy = VectorType::get(newShape.back(), elemTy);
    Value newRes = rewriter.create<arith::ConstantOp>(
        loc, SplatElementsAttr::get(newOpTy, rewriter.getFloatAttr(elemTy, 0)));
    auto strides = computeStrides(newShape);
    // Remove the last stride to produce sub-vector indices.
    strides.pop_back();
    for (int64_t idx = 0; idx < numElems; idx += newShape.back()) {
      auto indices = delinearize(idx, strides);
      SmallVector<Value> subInputs(reshapedInputs.size());
      std::transform(reshapedInputs.begin(), reshapedInputs.end(),
                     subInputs.begin(), [&](auto val) {
                       return rewriter.create<vector::ExtractOp>(loc, val,
                                                                 indices);
                     });
      Value subRes =
          rewriter.create<OpT>(loc, subResTy, subInputs, op->getAttrs());
      newRes = rewriter.create<vector::InsertOp>(loc, subRes, newRes, indices);
    }

    // Reshape the result back to the original type.
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, vecTy, newRes);
    return success();
  }
};

using ExternElementwiseOp = triton::cpu::ExternElementwiseOp;

/*
 * libsleef does not contain implementations for 2-element vectors, so we pad
 * any such vectors to size 4 instead.
 */
struct PadSmallVecsForSleef : public OpRewritePattern<ExternElementwiseOp> {
public:
  using OpRewritePattern<ExternElementwiseOp>::OpRewritePattern;

  PadSmallVecsForSleef(MLIRContext *context)
      : OpRewritePattern<ExternElementwiseOp>(context) {}

  LogicalResult matchAndRewrite(ExternElementwiseOp op,
                                PatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy)
      return failure();

    Type elemTy = vecTy.getElementType();
    if (!elemTy.isF32() && !elemTy.isF64())
      return failure();

    int64_t numElems = vecTy.getNumElements();
    if (numElems >= 4)
      return failure();

    // Create a single-element vector for shuffle to use
    auto paddingVec = rewriter.create<vector::SplatOp>(
        loc, b.undef(elemTy), VectorType::get({1}, elemTy));
    // Assign indices such that shuffle will pad the original vector with
    // elements from the paddingVec
    SmallVector<int64_t> indices(4);
    for (int i = 0; i < 4; ++i) {
      if (i < numElems)
        indices[i] = i;
      else
        indices[i] = numElems;
    }
    SmallVector<Value> newOperands;
    for (auto argVal : op.getOperands()) {
      auto shuf =
          rewriter.create<vector::ShuffleOp>(loc, argVal, paddingVec, indices);
      newOperands.push_back(shuf.getResult());
    }
    // Update return type of extern call
    auto newVecTy = VectorType::get({4}, elemTy);
    auto extern_elem = rewriter.create<ExternElementwiseOp>(
        loc, newVecTy, newOperands, op.getSymbol(), op.getPure());
    indices.resize(numElems);
    // Truncate result to original size
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(op, extern_elem.getResult(),
                                                   paddingVec, indices);
    return success();
  }
};

using GetVecFnNameFn = std::function<std::string(
    unsigned /*bitwidth*/, unsigned /*numel*/, ValueRange /*operands*/)>;

class MvecNameGenerator {
public:
  explicit MvecNameGenerator(StringRef baseName) : baseName(baseName) {}

  std::string operator()(unsigned bitwidth, unsigned numel,
                         ValueRange operands) const {
    if (bitwidth != 32 && bitwidth != 64)
      return "";
    unsigned vecSize = numel * bitwidth;
    std::string isaPrefix;
    if (vecSize == 128) {
      isaPrefix = "b";
    } else if (vecSize == 256) {
      isaPrefix = "d";
    } else if (vecSize == 512) {
      isaPrefix = "e";
    } else {
      return "";
    }
    std::string fnName = "_ZGV" + isaPrefix + "N" + std::to_string(numel);
    for (auto operand : operands)
      fnName += "v";
    return fnName + "_" + baseName + (bitwidth == 32 ? "f" : "");
  }

private:
  std::string baseName;
};

class SleefNameGenerator {
public:
  SleefNameGenerator(StringRef baseName, unsigned ulp = 10)
      : baseName(baseName), ulpSuffix(4, '\0') {
    if (ulp == 0) {
      ulpSuffix = "";
    } else {
      char buf[5]; // 4 char suffix + '\0' added by snprintf
      snprintf(buf, 5, "_u%02u", ulp);
      ulpSuffix = buf;
    }
  }

  std::string operator()(unsigned bitwidth, unsigned numel,
                         ValueRange /*operands*/) const {
    if (bitwidth != 32 && bitwidth != 64)
      return "";
    unsigned vecSize = numel * bitwidth;
    if (vecSize < 128)
      return "";
    return "Sleef_" + baseName + (bitwidth == 32 ? "f" : "d") +
           std::to_string(numel) + ulpSuffix;
  }

private:
  std::string baseName;
  std::string ulpSuffix;
};

template <typename OpT>
struct OpToVecLibConversion : public OpRewritePattern<OpT> {
public:
  using OpRewritePattern<OpT>::OpRewritePattern;

  virtual std::string getVecFnName(OpT op, unsigned bitwidth,
                                   unsigned numel) const = 0;

  LogicalResult matchAndRewrite(OpT op, PatternRewriter &rewriter) const {
    VectorType vecTy = dyn_cast<VectorType>(op.getType());
    if (!vecTy || vecTy.getRank() > 1)
      return failure();

    auto fnName = getVecFnName(op, vecTy.getElementTypeBitWidth(),
                               vecTy.getNumElements());
    if (fnName.empty())
      return failure();

    auto module = SymbolTable::getNearestSymbolTable(op);
    auto opFunc = dyn_cast_or_null<SymbolOpInterface>(
        SymbolTable::lookupSymbolIn(module, fnName));
    // Generate function declaration if it doesn't exists yet.
    if (!opFunc) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&module->getRegion(0).front());
      auto fnTy = FunctionType::get(
          rewriter.getContext(), op->getOperandTypes(), op->getResultTypes());
      opFunc =
          rewriter.create<func::FuncOp>(rewriter.getUnknownLoc(), fnName, fnTy);
      opFunc.setPrivate();
      opFunc->setAttr(LLVM::LLVMDialect::getReadnoneAttrName(),
                      UnitAttr::get(rewriter.getContext()));
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, fnName, op.getType(),
                                              op->getOperands());
    return success();
  }
};

template <typename OpT>
struct VecOpToVecLibConversion : public OpToVecLibConversion<OpT> {
public:
  VecOpToVecLibConversion(MLIRContext *context, GetVecFnNameFn getVecFnName)
      : OpToVecLibConversion<OpT>(context), getVecFnNameImpl(getVecFnName) {}

  std::string getVecFnName(OpT op, unsigned bitwidth,
                           unsigned numel) const override {
    return getVecFnNameImpl(bitwidth, numel, op->getOperands());
  }

private:
  GetVecFnNameFn getVecFnNameImpl;
};

struct ExternElementwiseOpConversion
    : public OpToVecLibConversion<triton::cpu::ExternElementwiseOp> {
  using OpToVecLibConversion::OpToVecLibConversion;

  std::string getVecFnName(triton::cpu::ExternElementwiseOp op,
                           unsigned bitwidth, unsigned numel) const override {
    auto fnName = op.getSymbol();
    auto numelIdx = fnName.find("%(numel)");
    if (numelIdx == StringRef::npos)
      return fnName.str();
    return (fnName.take_front(numelIdx) + Twine(numel) +
            fnName.drop_front(numelIdx + 8))
        .str();
  }
};

template <typename OpTy>
void populatePatternsForOp(RewritePatternSet &patterns,
                           GetVecFnNameFn getVecFnName,
                           size_t vec_size_in_bits = 512) {
  patterns.add<VecOpToFp32<OpTy>>(patterns.getContext());
  patterns.add<DecomposeToNativeVecs<OpTy>>(patterns.getContext(),
                                            vec_size_in_bits);
  patterns.add<VecOpToVecLibConversion<OpTy>>(patterns.getContext(),
                                              getVecFnName);
}

struct MathToVecLibPass
    : public mlir::triton::cpu::impl::MathToVecLibBase<MathToVecLibPass> {
  MathToVecLibPass() = default;
  // Default to 128-bit if no features are specified.
  size_t vec_size_in_bits = 128;

  explicit MathToVecLibPass(VecLib lib, std::set<std::string> cpu_features) {
    this->lib = lib;
    update_vec_size(cpu_features);
  }

  void update_vec_size(std::set<std::string> &cpu_features) {
    // TODO:
    //  Refactor this as an independent function.
    //  And improve this to support other x86 SIMD ISAs and also for arm SVE
    //  (VLA)
    for (auto feature : cpu_features) {
      if (feature == "avx512f") {
        vec_size_in_bits = std::max<size_t>(vec_size_in_bits, 512);
      } else if (feature == "avx") {
        vec_size_in_bits = std::max<size_t>(vec_size_in_bits, 256);
      } else if (feature == "sse") {
        vec_size_in_bits = std::max<size_t>(vec_size_in_bits, 128);
      } else if (feature == "neon") {
        // Arm NEON is fixed 128-bit SIMD ISA.
        vec_size_in_bits = 128;
        break;
      }
    }
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet patterns(context);

    if (!cpu_features.empty()) {
      std::set<std::string> cpu_features_set{cpu_features.begin(),
                                             cpu_features.end()};
      update_vec_size(cpu_features_set);
    }

    switch (lib) {
    case VecLib::Mvec: {
      populateCommonPatterns<MvecNameGenerator>(patterns);
      break;
    }
    case VecLib::Sleef: {
      populateCommonPatterns<SleefNameGenerator>(patterns);
      populatePatternsForOp<math::ExpM1Op>(
          patterns, SleefNameGenerator("expm1"), vec_size_in_bits);
      populatePatternsForOp<math::FloorOp>(
          patterns, SleefNameGenerator("floor", /*ulp=*/0), vec_size_in_bits);
      populatePatternsForOp<math::SqrtOp>(
          patterns, SleefNameGenerator("sqrt", /*ulp=*/5), vec_size_in_bits);
      populatePatternsForOp<math::TruncOp>(
          patterns, SleefNameGenerator("trunc", /*ulp=*/0), vec_size_in_bits);
      break;
    }
    }

    patterns.add<DecomposeToNativeVecs<ExternElementwiseOp>>(
        patterns.getContext(), vec_size_in_bits);
    patterns.add<PadSmallVecsForSleef>(patterns.getContext());
    patterns.add<ExternElementwiseOpConversion>(patterns.getContext());

    if (failed(applyPatternsGreedily(op, std::move(patterns))))
      signalPassFailure();
  }

  template <typename VecFnNameGenerator>
  void populateCommonPatterns(RewritePatternSet &patterns) const {
    populatePatternsForOp<math::AcosOp>(patterns, VecFnNameGenerator("acos"),
                                        vec_size_in_bits);
    populatePatternsForOp<math::AcoshOp>(patterns, VecFnNameGenerator("acosh"),
                                         vec_size_in_bits);
    populatePatternsForOp<math::AsinOp>(patterns, VecFnNameGenerator("asin"),
                                        vec_size_in_bits);
    populatePatternsForOp<math::AsinhOp>(patterns, VecFnNameGenerator("asinh"),
                                         vec_size_in_bits);
    populatePatternsForOp<math::AtanOp>(patterns, VecFnNameGenerator("atan"),
                                        vec_size_in_bits);
    populatePatternsForOp<math::AtanhOp>(patterns, VecFnNameGenerator("atanh"),
                                         vec_size_in_bits);
    populatePatternsForOp<math::CbrtOp>(patterns, VecFnNameGenerator("cbrt"),
                                        vec_size_in_bits);
    populatePatternsForOp<math::CosOp>(patterns, VecFnNameGenerator("cos"),
                                       vec_size_in_bits);
    populatePatternsForOp<math::CoshOp>(patterns, VecFnNameGenerator("cosh"),
                                        vec_size_in_bits);
    populatePatternsForOp<math::ErfOp>(patterns, VecFnNameGenerator("erf"),
                                       vec_size_in_bits);
    populatePatternsForOp<math::ExpOp>(patterns, VecFnNameGenerator("exp"),
                                       vec_size_in_bits);
    populatePatternsForOp<math::Exp2Op>(patterns, VecFnNameGenerator("exp2"),
                                        vec_size_in_bits);
    populatePatternsForOp<math::LogOp>(patterns, VecFnNameGenerator("log"),
                                       vec_size_in_bits);
    populatePatternsForOp<math::Log2Op>(patterns, VecFnNameGenerator("log2"),
                                        vec_size_in_bits);
    populatePatternsForOp<math::Log10Op>(patterns, VecFnNameGenerator("log10"),
                                         vec_size_in_bits);
    populatePatternsForOp<math::Log1pOp>(patterns, VecFnNameGenerator("log1p"),
                                         vec_size_in_bits);
    populatePatternsForOp<math::SinOp>(patterns, VecFnNameGenerator("sin"),
                                       vec_size_in_bits);
    populatePatternsForOp<math::SinhOp>(patterns, VecFnNameGenerator("sinh"),
                                        vec_size_in_bits);
    populatePatternsForOp<math::TanOp>(patterns, VecFnNameGenerator("tan"),
                                       vec_size_in_bits);
    populatePatternsForOp<math::TanhOp>(patterns, VecFnNameGenerator("tanh"),
                                        vec_size_in_bits);
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>>
createMathToVecLibPass(VecLib lib, std::set<std::string> cpu_features) {
  return std::make_unique<MathToVecLibPass>(lib, cpu_features);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
