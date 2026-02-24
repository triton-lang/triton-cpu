#include "TypeConverter.h"

#include "cpu/include/Analysis/TensorPtrShapeInfo.h"
#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "cpu/include/ScalarizePass/ScalarizeInterface.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTMEMORYOPS
#include "cpu/include/TritonToTritonCPU/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

template <typename OpT>
struct MemoryOpConversion : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;
  using OpConversionPattern<OpT>::getContext;
  using OpConversionPattern<OpT>::getTypeConverter;

  MemoryOpConversion(ModuleAxisInfoAnalysis &axisInfoAnalysis,
                     ModuleTensorPtrShapeInfoAnalysis &shapeInfoAnalysis,
                     TypeConverter &typeConverter, MLIRContext *context,
                     bool useGatherScatter)
      : OpConversionPattern<OpT>(typeConverter, context),
        axisAnalysis(axisInfoAnalysis), shapeAnalysis(shapeInfoAnalysis) {
    this->useGatherScatter = useGatherScatter;
  }

  Value extractScalarPointer(Location loc, Value ptrs,
                             ArrayRef<int64_t> indices,
                             ConversionPatternRewriter &rewriter) const {
    // If we build a vector of pointers and the extract a pointer from it, then
    // compiler doesn't always optimize it to a simple scalar pointer
    // computation. Here we try to follow a data flow of the tensor to rebuild a
    // scalar pointer for more efficient resulting code.
    if (canComputeScalarValue(ptrs)) {
      return computeScalarValue(ptrs.getDefiningOp(), ptrs, indices, rewriter);
    }

    // Fall back to a scalar pointer extraction from the vector.
    Value ptr = vector::ExtractOp::create(
        rewriter, loc, rewriter.getRemappedValue(ptrs), indices);
    auto ptrTy = dyn_cast<RankedTensorType>(ptrs.getType()).getElementType();
    ptr = IntToPtrOp::create(rewriter, loc, ptrTy, ptr);
    return ptr;
  }

  Value extractMemRef(Location loc, Value ptr,
                      ConversionPatternRewriter &rewriter) const {
    auto tensorTy = dyn_cast<RankedTensorType>(
        dyn_cast<PointerType>(ptr.getType()).getPointeeType());
    auto elemTy = tensorTy.getElementType();
    auto shapeInfo = shapeAnalysis.getPtrShapeInfo(ptr);
    Type memRefTy;
    if (shapeInfo && shapeInfo->getRank() > 0) {
      auto layout =
          StridedLayoutAttr::get(getContext(), 0, shapeInfo->getStrides());
      memRefTy = MemRefType::get(shapeInfo->getShape(), elemTy, layout);
    } else {
      SmallVector<int64_t> dynVals(tensorTy.getRank(), ShapedType::kDynamic);
      auto layout = StridedLayoutAttr::get(getContext(), 0, dynVals);
      memRefTy = MemRefType::get(dynVals, elemTy, layout);
    }
    return ExtractMemRefOp::create(rewriter, loc, memRefTy, ptr);
  }

  Value convertOtherVal(triton::LoadOp loadOp,
                        ConversionPatternRewriter &rewriter) const {
    if (loadOp.getOther())
      return rewriter.getRemappedValue(loadOp.getOther());

    auto resTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(loadOp.getType()));
    return arith::ConstantOp::create(
        rewriter, loadOp.getLoc(), resTy,
        SplatElementsAttr::get(resTy,
                               rewriter.getZeroAttr(resTy.getElementType())));
  }

  Value createAlloca(Location loc, MemRefType ty, Operation *before,
                     ConversionPatternRewriter &rewriter) const {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(before);
    return memref::AllocaOp::create(
        rewriter, loc, ty, rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
  }

  // If tensor is not null and its element cannot be recomputed in a scalar
  // loop, then store it to a temporary buffer.
  Value maybeStoreVecToTempBuf(Location loc, Value vals, Value zeroIdx,
                               Operation *allocaPoint,
                               ConversionPatternRewriter &rewriter) const {
    if (!vals || canComputeScalarValue(vals))
      return nullptr;

    auto vec = rewriter.getRemappedValue(vals);
    auto vecTy = cast<VectorType>(vec.getType());
    auto elemTy = vecTy.getElementType();
    // Memref of i1 assumes one element per byte when we load/store element,
    // but vector store (through transfer write) would write 1 bit per element.
    if (elemTy.isInteger(1)) {
      elemTy = rewriter.getI8Type();
      vec = arith::ExtUIOp::create(
          rewriter, loc, VectorType::get(vecTy.getShape(), elemTy), vec);
    }
    auto memRefTy = MemRefType::get(vecTy.getShape(), elemTy);
    Value memRef = createAlloca(vals.getLoc(), memRefTy, allocaPoint, rewriter);
    SmallVector<Value> indices(vecTy.getRank(), zeroIdx);
    vector::TransferWriteOp::create(rewriter, vals.getLoc(), vec, memRef,
                                    indices);
    return memRef;
  }

protected:
  ModuleAxisInfoAnalysis &axisAnalysis;
  ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis;
  bool useGatherScatter;
};

struct LoadOpConversion : public MemoryOpConversion<triton::LoadOp> {
  using MemoryOpConversion::MemoryOpConversion;

  static Value
  getPaddingValue(Location loc, Type type,
                  const std::optional<triton::PaddingOption> &padding,
                  ConversionPatternRewriter &rewriter) {
    auto padding_option = padding.value_or(PaddingOption::PAD_ZERO);

    TypedAttr attr;
    switch (padding_option) {
    case PaddingOption::PAD_ZERO:
      attr = rewriter.getZeroAttr(type);
      break;
    case PaddingOption::PAD_NAN:
      assert(!type.isIntOrIndex());
      auto apNaN =
          llvm::APFloat::getNaN(cast<FloatType>(type).getFloatSemantics());
      attr = FloatAttr::get(type, apNaN);
      break;
    }

    return arith::ConstantOp::create(rewriter, loc, attr);
  }

  LogicalResult
  matchAndRewrite(triton::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto mask = loadOp.getMask();
    auto ptr = loadOp.getPtr();
    auto boundaryChecks = loadOp.getBoundaryCheck();

    if (!triton::isTensorPointerType(ptr.getType())) {
      auto axisInfo = axisAnalysis.getAxisInfo(ptr);
      if (isContiguousRowMajorAccess(axisInfo, loadOp)) {
        return lowerToContiguousRowMajor(loadOp, rewriter);
      }
      if (useGatherScatter && succeeded(lowerToGather(loadOp, rewriter))) {
        return success();
      }
      return lowerToScalarLoads(loadOp, rewriter);
    }

    // TODO: support masks.
    if (mask) {
      llvm_unreachable("unsupported load op");
    }

    auto memRef = extractMemRef(loc, ptr, rewriter);
    auto rank = dyn_cast<MemRefType>(memRef.getType()).getRank();
    auto resTy = dyn_cast<VectorType>(
        getTypeConverter()->convertType(loadOp.getResult().getType()));
    auto indices = ExtractIndicesOp::create(rewriter, loc, ptr).getResults();
    SmallVector<bool, 4> inBounds(rank, true);
    for (auto dim : boundaryChecks) {
      inBounds[dim] = false;
    }
    Value padding = getPaddingValue(loc, resTy.getElementType(),
                                    loadOp.getPadding(), rewriter);
    auto vecRead = vector::TransferReadOp::create(rewriter, loc, resTy, memRef,
                                                  indices, padding, inBounds);
    rewriter.replaceOp(loadOp, vecRead);
    return success();
  }

  LogicalResult
  lowerToContiguousRowMajor(triton::LoadOp loadOp,
                            ConversionPatternRewriter &rewriter) const {
    // This is an experimental code that covers only a simple case of axis info
    // usage to demostrate load by tensor of pointers transformation into vector
    // loads.
    // TODO: Support more cases.
    // TODO: Make separate pass to produce block pointer stores?
    auto loc = loadOp.getLoc();
    auto vecTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(loadOp.getType()));
    auto shape = vecTy.getShape();

    auto strides = computeStrides(shape);
    int64_t numElems = vecTy.getNumElements();
    Type subVecTy = VectorType::get(shape.back(), vecTy.getElementType());
    Type memRefTy = MemRefType::get(shape.back(), vecTy.getElementType());
    Value mask = loadOp.getMask() ? rewriter.getRemappedValue(loadOp.getMask())
                                  : nullptr;
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value defaultVal = convertOtherVal(loadOp, rewriter);
    Value res = defaultVal;
    for (int64_t idx = 0; idx < numElems; idx += shape.back()) {
      auto indices = delinearize(idx, strides);
      SmallVector<int64_t> subIndices(indices.begin(),
                                      indices.begin() + indices.size() - 1);
      auto ptr = extractScalarPointer(loc, loadOp.getPtr(), indices, rewriter);
      Value memRef =
          triton::cpu::PtrToMemRefOp::create(rewriter, loc, memRefTy, ptr);
      Value vec;
      if (mask) {
        Value subMask = mask;
        Value passThru = defaultVal;
        if (shape.size() > 1) {
          subMask = vector::ExtractOp::create(rewriter, loc, mask, subIndices);
          passThru =
              vector::ExtractOp::create(rewriter, loc, defaultVal, subIndices);
        }
        vec = vector::MaskedLoadOp::create(rewriter, loc, subVecTy, memRef,
                                           zeroIdx, subMask, passThru);
      } else {
        vec = vector::LoadOp::create(rewriter, loc, subVecTy, memRef, zeroIdx);
      }

      if (shape.size() > 1) {
        res = vector::InsertOp::create(rewriter, loc, vec, res, subIndices);
      } else {
        res = vec;
      }
    }

    rewriter.replaceOp(loadOp, res);
    return success();
  }

  LogicalResult lowerToGather(triton::LoadOp loadOp,
                              ConversionPatternRewriter &rewriter) const {
    auto loc = loadOp.getLoc();
    auto vecTy = dyn_cast<VectorType>(
        getTypeConverter()->convertType(loadOp.getResult().getType()));
    auto shape = vecTy.getShape();

    auto [basePtr, offset] = getMemoryBaseOffset(loadOp);

    if (!basePtr || !offset)
      return failure();

    auto pointeeType =
        dyn_cast<PointerType>(basePtr.getType()).getPointeeType();

    auto gatherBase = triton::cpu::PtrToMemRefOp::create(
        rewriter, loc, MemRefType::get({}, pointeeType), basePtr);
    auto gatherIndices = SmallVector<Value>();
    auto gatherIndexVec = rewriter.getRemappedValue(offset);

    Value gatherMask;
    if (auto loadMask = loadOp.getMask()) {
      gatherMask = rewriter.getRemappedValue(loadMask);
    } else {
      auto maskType = VectorType::get(shape, rewriter.getI1Type());
      gatherMask = arith::ConstantOp::create(
          rewriter, loc, maskType, DenseElementsAttr::get(maskType, true));
    }

    auto passThru = convertOtherVal(loadOp, rewriter);

    auto gatherOp = vector::GatherOp::create(rewriter, loc, vecTy, gatherBase,
                                             gatherIndices, gatherIndexVec,
                                             gatherMask, passThru);
    rewriter.replaceOp(loadOp, gatherOp);
    return success();
  }

  LogicalResult lowerToScalarLoads(triton::LoadOp loadOp,
                                   ConversionPatternRewriter &rewriter) const {
    // Scalar loads and boundary checks are not expected.
    assert(loadOp.getBoundaryCheck().empty());
    assert(isa<RankedTensorType>(loadOp.getType()));

    auto loc = loadOp.getLoc();
    auto vecTy =
        dyn_cast<VectorType>(getTypeConverter()->convertType(loadOp.getType()));

    auto ptrs = rewriter.getRemappedValue(loadOp.getPtr());
    auto mask = loadOp.getMask() ? rewriter.getRemappedValue(loadOp.getMask())
                                 : nullptr;
    auto ptrTy =
        dyn_cast<RankedTensorType>(loadOp.getPtr().getType()).getElementType();
    auto cache = loadOp.getCache();
    auto evict = loadOp.getEvict();
    auto isVolatile = loadOp.getIsVolatile();

    auto loadOne = [=, &rewriter](ArrayRef<int64_t> indices, Value dst) {
      Value ptr = vector::ExtractOp::create(rewriter, loc, ptrs, indices);
      ptr = IntToPtrOp::create(rewriter, loc, ptrTy, ptr);
      Value val =
          triton::LoadOp::create(rewriter, loc, ptr, cache, evict, isVolatile);
      return vector::InsertOp::create(rewriter, loc, val, dst, indices);
    };

    Value dst = convertOtherVal(loadOp, rewriter);
    int64_t numElems = vecTy.getNumElements();
    auto strides = computeStrides(vecTy.getShape());
    for (auto idx = 0; idx < numElems; ++idx) {
      auto indices = delinearize(idx, strides);
      if (!mask) {
        dst = loadOne(indices, dst);
        continue;
      }
      // Create a conditional block for load if there is a mask.
      auto predicate = vector::ExtractOp::create(rewriter, loc, mask, indices);
      auto ifOp = scf::IfOp::create(
          rewriter, loc, predicate,
          [&](OpBuilder &builder, Location loc) {
            auto result = loadOne(indices, dst).getResult();
            scf::YieldOp::create(rewriter, loc, result);
          },
          [&](OpBuilder &builder, Location loc) {
            scf::YieldOp::create(rewriter, loc, dst);
          });
      dst = ifOp.getResult(0);
    }

    rewriter.replaceOp(loadOp, dst);

    return success();
  }
};

struct StoreOpConversion : public MemoryOpConversion<triton::StoreOp> {
  using MemoryOpConversion::MemoryOpConversion;

  LogicalResult
  matchAndRewrite(triton::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();
    auto mask = storeOp.getMask();
    auto ptr = storeOp.getPtr();
    auto boundaryChecks = storeOp.getBoundaryCheck();

    if (!triton::isTensorPointerType(ptr.getType())) {
      auto axisInfo = axisAnalysis.getAxisInfo(ptr);
      if (isContiguousRowMajorAccess(axisInfo, storeOp)) {
        return lowerToContiguousRowMajor(storeOp, rewriter);
      }
      if (useGatherScatter && succeeded(lowerToScatter(storeOp, rewriter))) {
        return success();
      }
      return lowerToScalarStores(storeOp, rewriter);
    }

    // TODO: support masks.
    if (mask) {
      llvm_unreachable("unsupported store op");
    }

    auto value = rewriter.getRemappedValue(storeOp.getValue());
    auto memRef = extractMemRef(loc, ptr, rewriter);
    auto rank = dyn_cast<MemRefType>(memRef.getType()).getRank();
    auto indices = ExtractIndicesOp::create(rewriter, loc, ptr).getResults();
    SmallVector<bool, 4> inBounds(rank, true);
    for (auto dim : boundaryChecks) {
      inBounds[dim] = false;
    }
    auto vecWrite = vector::TransferWriteOp::create(rewriter, loc, value,
                                                    memRef, indices, inBounds);
    rewriter.replaceOp(storeOp, vecWrite);
    return success();
  }

  LogicalResult
  lowerToContiguousRowMajor(triton::StoreOp storeOp,
                            ConversionPatternRewriter &rewriter) const {
    // This is an experimental code that covers only a simple case of axis info
    // usage to demostrate load by tensor of pointers transformation into vector
    // loads.
    // TODO: Support more cases.
    // TODO: Make separate pass to produce block pointer stores instead?
    auto loc = storeOp.getLoc();
    auto vals = rewriter.getRemappedValue(storeOp.getValue());
    auto vecTy = dyn_cast<VectorType>(vals.getType());
    auto shape = vecTy.getShape();

    auto strides = computeStrides(shape);
    int64_t numElems = vecTy.getNumElements();
    Type memRefTy = MemRefType::get(shape.back(), vecTy.getElementType());
    Value mask = storeOp.getMask()
                     ? rewriter.getRemappedValue(storeOp.getMask())
                     : nullptr;
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    for (int64_t idx = 0; idx < numElems; idx += shape.back()) {
      auto indices = delinearize(idx, strides);
      auto ptr = extractScalarPointer(loc, storeOp.getPtr(), indices, rewriter);
      Value memRef =
          triton::cpu::PtrToMemRefOp::create(rewriter, loc, memRefTy, ptr);
      indices.pop_back();
      auto val = vector::ExtractOp::create(rewriter, loc, vals, indices);

      if (mask) {
        Value subMask = mask;
        if (shape.size() > 1) {
          SmallVector<int64_t> subIndices = indices;
          subIndices.pop_back();
          subMask = vector::ExtractOp::create(rewriter, loc, mask, indices);
        }
        vector::MaskedStoreOp::create(rewriter, loc, memRef, zeroIdx, subMask,
                                      val);
      } else {
        vector::StoreOp::create(rewriter, loc, val, memRef, zeroIdx);
      }
    }

    rewriter.eraseOp(storeOp);
    return success();
  }

  LogicalResult lowerToScatter(triton::StoreOp storeOp,
                               ConversionPatternRewriter &rewriter) const {
    auto loc = storeOp.getLoc();
    auto vals = rewriter.getRemappedValue(storeOp.getValue());
    auto vecTy = dyn_cast<VectorType>(vals.getType());
    auto shape = vecTy.getShape();

    auto [basePtr, offset] = getMemoryBaseOffset(storeOp);

    if (!basePtr || !offset)
      return failure();

    auto strides = computeStrides(shape);
    int64_t numElems = vecTy.getNumElements();
    Type memRefTy = MemRefType::get(shape.back(), vecTy.getElementType());
    Value mask = storeOp.getMask()
                     ? rewriter.getRemappedValue(storeOp.getMask())
                     : nullptr;

    for (int64_t idx = 0; idx < numElems; idx += shape.back()) {
      auto indices = delinearize(idx, strides);
      indices.pop_back();

      auto val = vector::ExtractOp::create(rewriter, loc, vals, indices);
      auto indexVec = vector::ExtractOp::create(
          rewriter, loc, rewriter.getRemappedValue(offset), indices);
      Value scatterMask;

      if (mask) {
        scatterMask = vector::ExtractOp::create(rewriter, loc, mask, indices);
      } else {
        // Create a mask with all true values if no mask is provided.
        auto maskType = VectorType::get({shape.back()}, rewriter.getI1Type());
        scatterMask = arith::ConstantOp::create(
            rewriter, loc, maskType, DenseElementsAttr::get(maskType, true));
      }

      auto scatterBase = triton::cpu::PtrToMemRefOp::create(
          rewriter, loc, MemRefType::get({}, vecTy.getElementType()), basePtr);
      auto scatterIndices = SmallVector<Value>();

      vector::ScatterOp::create(rewriter, loc, TypeRange{}, scatterBase,
                                scatterIndices, indexVec, scatterMask, val,
                                IntegerAttr{});
    }

    rewriter.eraseOp(storeOp);
    return success();
  }

  LogicalResult lowerToScalarStores(triton::StoreOp storeOp,
                                    ConversionPatternRewriter &rewriter) const {
    // Scalar stores and boundary checks are not expected.
    assert(storeOp.getBoundaryCheck().empty());
    assert(isa<RankedTensorType>(storeOp.getValue().getType()));

    auto loc = storeOp.getLoc();
    auto tensorTy = dyn_cast<RankedTensorType>(storeOp.getPtr().getType());

    auto ptrs = rewriter.getRemappedValue(storeOp.getPtr());
    auto mask = storeOp.getMask() ? rewriter.getRemappedValue(storeOp.getMask())
                                  : nullptr;
    auto vals = rewriter.getRemappedValue(storeOp.getValue());
    auto ptrTy = tensorTy.getElementType();
    auto cache = storeOp.getCache();
    auto evict = storeOp.getEvict();

    auto storeOne = [=, &rewriter](ArrayRef<int64_t> indices) {
      Value ptr = vector::ExtractOp::create(rewriter, loc, ptrs, indices);
      ptr = IntToPtrOp::create(rewriter, loc, ptrTy, ptr);
      Value val = vector::ExtractOp::create(rewriter, loc, vals, indices);
      triton::StoreOp::create(rewriter, loc, ptr, val, cache, evict);
    };

    int64_t numElems = tensorTy.getNumElements();
    auto strides = computeStrides(tensorTy.getShape());
    for (auto idx = 0; idx < numElems; ++idx) {
      auto indices = delinearize(idx, strides);
      if (!mask) {
        storeOne(indices);
        continue;
      }
      // Create a conditional block for store if there is a mask.
      auto predicate = vector::ExtractOp::create(rewriter, loc, mask, indices);
      scf::IfOp::create(rewriter, loc, predicate,
                        [&](OpBuilder &builder, Location loc) {
                          storeOne(indices);
                          scf::YieldOp::create(rewriter, loc);
                        });
    }

    rewriter.eraseOp(storeOp);

    return success();
  }
};

struct CpuStoreOpConversion : public MemoryOpConversion<triton::cpu::StoreOp> {
  using MemoryOpConversion::MemoryOpConversion;

  LogicalResult
  matchAndRewrite(triton::cpu::StoreOp storeOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = storeOp.getLoc();
    auto value = rewriter.getRemappedValue(storeOp.getSrc());
    auto memRef = storeOp.getDst();
    auto rank = dyn_cast<MemRefType>(memRef.getType()).getRank();
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> indices(rank, zeroIdx);
    auto vecWrite =
        vector::TransferWriteOp::create(rewriter, loc, value, memRef,
                                        indices); //, inBounds);
    rewriter.replaceOp(storeOp, vecWrite);
    return success();
  }
};

struct CpuLoadOpConversion : public MemoryOpConversion<triton::cpu::LoadOp> {
  using MemoryOpConversion::MemoryOpConversion;

  LogicalResult
  matchAndRewrite(triton::cpu::LoadOp loadOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = loadOp.getLoc();
    auto memRef = loadOp.getSrc();
    auto rank = dyn_cast<MemRefType>(memRef.getType()).getRank();
    auto resTy = dyn_cast<VectorType>(
        getTypeConverter()->convertType(loadOp.getResult().getType()));
    Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<Value> indices(resTy.getRank(), zeroIdx);
    auto vecRead = vector::TransferReadOp::create(
        rewriter, loc, resTy, memRef, indices,
        arith::getZeroConstant(rewriter, loc, resTy.getElementType()));
    rewriter.replaceOp(loadOp, vecRead);
    return success();
  }
};

class MemoryOpConversionTarget : public ConversionTarget {
public:
  explicit MemoryOpConversionTarget(MLIRContext &ctx) : ConversionTarget(ctx) {
    addLegalDialect<vector::VectorDialect>();
    addLegalDialect<arith::ArithDialect>();
    addLegalDialect<scf::SCFDialect>();
    addLegalDialect<memref::MemRefDialect>();
    addLegalDialect<TritonDialect>();
    addLegalDialect<TritonCPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addIllegalOp<mlir::triton::cpu::StoreOp, mlir::triton::cpu::LoadOp>();

    // Allow only scalar loads and stores.
    addDynamicallyLegalOp<triton::LoadOp>([](triton::LoadOp loadOp) {
      return loadOp.getType().isIntOrIndexOrFloat();
    });
    addDynamicallyLegalOp<triton::StoreOp>([](triton::StoreOp storeOp) {
      return storeOp.getValue().getType().isIntOrIndexOrFloat();
    });
  }
};

struct ConvertMemoryOps
    : public triton::cpu::impl::ConvertMemoryOpsBase<ConvertMemoryOps> {
  ConvertMemoryOps() = default;

  ConvertMemoryOps(bool useGatherScatter) {
    this->useGatherScatter = useGatherScatter;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    ModuleTensorPtrShapeInfoAnalysis shapeInfoAnalysis(mod);
    MemoryOpConversionTarget convTarget(*context);
    TritonToTritonCPUTypeConverter pointerConverter;
    RewritePatternSet patterns(context);
    patterns.add<LoadOpConversion, StoreOpConversion, CpuStoreOpConversion,
                 CpuLoadOpConversion>(axisInfoAnalysis, shapeInfoAnalysis,
                                      pointerConverter, context,
                                      useGatherScatter);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertMemoryOps() {
  return std::make_unique<ConvertMemoryOps>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMemoryOps(bool useGatherScatter) {
  return std::make_unique<ConvertMemoryOps>(useGatherScatter);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
