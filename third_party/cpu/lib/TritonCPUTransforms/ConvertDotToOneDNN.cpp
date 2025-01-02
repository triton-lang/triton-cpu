// save diagnostic state
// #pragma GCC diagnostic push

// turn off the specific warning. Can also use "-Wall"
// #pragma GCC diagnostic ignored "-Wall"
// #pragma GCC diagnostic ignored "-Weffc++"
// #pragma GCC diagnostic ignored "-pedantic"

#include "cpu/include/TritonCPUTransforms/OptCommon.h"

#include "cpu/include/Analysis/TensorPtrShapeInfo.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"

#include "include/triton/Analysis/Utility.h"
#include "oneapi/dnnl/dnnl_types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#include "ConvertDotOp/ConvertDotCommon.h"

#include <iostream>
#include <tuple>
#include <utility>

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTTOONEDNN
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir
// #pragma GCC diagnostic pop

// #define DEBUG_TYPE "triton-cpu-dot-to-onednn"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

static inline int64_t getDnnlDataTypeVal(Type ty) {
  ty = getElementTypeOrSelf(ty);
  if (ty.isF32())
    return static_cast<int64_t>(dnnl_f32);
  if (ty.isF64())
    return static_cast<int64_t>(dnnl_f64);
  if (ty.isBF16())
    return static_cast<int64_t>(dnnl_bf16);
  if (ty.isF16())
    return static_cast<int64_t>(dnnl_f16);
  llvm_unreachable("Unexpected type for conversion to DNNL type.");
}

// This structure is used to hold candidates for conversion to ukernel calls.
struct DotOpCandidate {
  // Operation to convert.
  triton::cpu::DotOp op;

  // Block sizes.
  int64_t blockM;
  int64_t blockN;
  int64_t blockK;
  // If accumulator is updated in a loop, then this flag indicates if we
  // should keep it in tiles the whole loop and move back to vectors only
  // after the loop.
  bool isAccLoopCarried = false;
  bool canFuseLoop = false;

  // If output buffer is used then keep the original vector store here.
  Operation *origStore = nullptr;

  // If input data is available in memory then input buffers hold it.
  MemBuffer lhsBuf;
  MemBuffer rhsBuf;
  // If result is written to a memory, then we can use it directly for
  // ukernel calls.
  MemBuffer outBuf;
};

// Check if vector transfer read/write operation uses a mask
// or involves a bounds check.
template <typename T> bool hasMaskOrBoundsCheck(T op) {
  auto inBounds = op.getInBounds();
  Value mask = op.getMask();
  bool hasBoundsCheck =
      std::any_of(inBounds.begin(), inBounds.end(), [](Attribute attr) {
        return !cast<mlir::BoolAttr>(attr).getValue();
      });
  llvm::errs() << "mask: " << mask << " bounds check: " << hasBoundsCheck
               << "\n";
  return hasBoundsCheck || mask;
}

bool isLoopInvariant(SmallVector<Value> vals, LoopLikeOpInterface loopLike) {
  for (Value val : vals) {
    LDBG("Checking value for invariance: " << val);
    if (!loopLike.isDefinedOutsideOfLoop(val)) {
      LDBG("  Not invariant");
      return false;
    }
  }
  return true;
}

bool checkElemTypes(Type lhsElemTy, Type rhsElemTy, Type accElemTy,
                    Type resElemTy) {
  // Integer types are not supported yet.
  if (lhsElemTy.isInteger() || rhsElemTy.isInteger() || resElemTy.isInteger()) {
    LDBG("Drop candidate. Integer types are not supported.");
    return false;
  }

  // FP8 input is not supported yet.
  if (lhsElemTy.getIntOrFloatBitWidth() == 8 ||
      rhsElemTy.getIntOrFloatBitWidth() == 8) {
    LDBG("Drop candidate. FP8 input is not supported.");
    return false;
  }

  // FP64 result is not supported.
  if (accElemTy.getIntOrFloatBitWidth() == 64 ||
      resElemTy.getIntOrFloatBitWidth() == 64) {
    LDBG("Drop candidate. FP64 result is not supported.");
    return false;
  }

  return true;
}

bool checkInputShapes(VectorType lhsTy, VectorType resTy,
                      DotOpCandidate &candidate) {
  if (lhsTy.getRank() != 2)
    return false;

  candidate.blockM = resTy.getDimSize(0);
  candidate.blockN = resTy.getDimSize(1);
  candidate.blockK = lhsTy.getDimSize(1);

  return true;
}

// Check if specified ContractionOp can be lowered to AMX operations.
// If conversion is possible, then true is returned and candidate
// structure is filled with detailed transformation info.
bool isOneDNNCandidate(triton::cpu::DotOp op, bool supportInt8,
                       bool supportFp16, bool supportBf16,
                       DotOpCandidate &candidate) {
  // MLIRContext *ctx = op.getContext();
  VectorType lhsTy = cast<VectorType>(op.getA().getType());
  VectorType rhsTy = cast<VectorType>(op.getB().getType());
  VectorType accTy = cast<VectorType>(op.getC().getType());
  VectorType resTy = cast<VectorType>(op.getType());

  LDBG("Considering candidate op: " << op);

  if (accTy.getRank() != 2) {
    LDBG("  Drop candidate. Only 2D case is supported.");
    return false;
  }

  // Check input/output types.
  if (!checkElemTypes(lhsTy.getElementType(), rhsTy.getElementType(),
                      accTy.getElementType(), resTy.getElementType()))
    return false;

  // Check input shapes.
  if (!checkInputShapes(lhsTy, resTy, candidate))
    return false;

  candidate.op = op;
  candidate.isAccLoopCarried = isLoopCarriedAcc(op.getC());
  candidate.lhsBuf = findInputBuffer(op.getA(), true);
  candidate.rhsBuf = findInputBuffer(op.getB(), false);

  // Check if we can fuse dot op loop into a single brgemm call.
  if (candidate.isAccLoopCarried && !candidate.lhsBuf.step.empty() &&
      !candidate.rhsBuf.step.empty()) {
    SmallVector<Value> valsToCheck;
    valsToCheck.append(candidate.lhsBuf.step);
    valsToCheck.append(candidate.rhsBuf.step);

    auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
    candidate.canFuseLoop = isLoopInvariant(valsToCheck, forOp);
  }

  // We don't need this I guess
  // findOutputBuffer(op.getResult(), candidate);

  return true;
}

MemBuffer allocateTmpBufferLocal(Location loc, VectorType vecTy,
                                 Operation *allocaPoint,
                                 PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(allocaPoint);
  auto memRefTy = MemRefType::get(vecTy.getShape(), vecTy.getElementType());
  Value memRef = rewriter.create<memref::AllocaOp>(
      loc, memRefTy, rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(2, zeroIdx);
  return {memRef, indices};
}

// Prepare temporary buffers to be used for tile loads. If the original
// value can be directly loaded to tiles from its original memory, then
// use it instead. Return empty buffer if source value is all zeros and
// skipForZeros is set.
//
// If interleave flag is set, then pre-pack RHS before sotring. See
// interleaveAndStore for more details.
MemBuffer prepareTensorBuffer(PatternRewriter &rewriter, Location loc,
                              Value val,
                              memref::ExtractStridedMetadataOp metadata,
                              bool readOnly, Operation *allocaPoint,
                              Value transform = nullptr) {
  LDBG("Preparing buffer (interleave=" << (transform == nullptr)
                                       << ") for a vector: " << val
                                       << " readOnly: " << readOnly);
  auto valLoad = val.getDefiningOp<vector::TransferReadOp>();
  // some extra conditions required
  if (valLoad) {
    LDBG("Lhs should take src memref!\n");
    Value memRef = valLoad.getSource();
    ValueRange indices = valLoad.getIndices();
    if (!transform) {
      LDBG("  Reusing the original memref for a buffer: " << memRef);
      auto vecTy = cast<VectorType>(val.getType());
      // auto memRefTy = MemRefType::get(vecTy.getShape(),
      // vecTy.getElementType());
      auto ctx = rewriter.getContext();
      // Value memRef_view = rewriter.create<memref::SubViewOp>(
      //     loc, memRefTy, memRef, {0, 0}, indices,
      //     metadata.getStrides());
      SmallVector<int64_t> strides(vecTy.getRank(), 1);

      Value memRef_view = rewriter.create<memref::SubViewOp>(
          loc, memRef, getAsOpFoldResult(indices),
          getAsIndexOpFoldResult(ctx, vecTy.getShape()),
          getAsIndexOpFoldResult(ctx, strides));
      return {memRef_view, indices};
    }

    // Just a stub, dont know why we should use Vector TYpe here
    // auto vecTy = cast<VectorType>(val.getType());
    // auto transf_op = transform.getDefiningOp<triton::cpu::TransformCreate>();

    assert("Used( " && false);
    // MemBuffer buf = allocateTmpBuffer(
    //     loc,
    //     {transf_op.getBlockK(), transf_op.getOutLd(), transf_op.getNCalls()},
    //     vecTy.getElementType(), allocaPoint, rewriter);

    // LDBG("  Reusing the original memref for a buffer: " << memRef);
    // transformIntoBuf(loc, transform, memRef, buf.memRef, rewriter);
    // return buf;
  }

  auto vecTy = cast<VectorType>(val.getType());
  MemBuffer buf = allocateTmpBufferLocal(loc, vecTy, allocaPoint, rewriter);

  if (transform) {
    LDBG("Unhandled case for transform: " << val);
    assert(false);
    return {};
  }
  auto rank = dyn_cast<MemRefType>(buf.memRef.getType()).getRank();
  SmallVector<bool, 4> inBounds(rank, false);
  rewriter.create<vector::TransferWriteOp>(loc, val, buf.memRef, buf.indices,
                                           inBounds);

  return buf;
}

// Return a buffer where the final result should be stored. If result can
// be directly stored to the output memory, then it is used as an output
// buffer. Otherwise, re-use accumulator buffer or create a new one.
MemBuffer prepareResultBuffer(Location loc, Value val, const MemBuffer &accBuf,
                              const MemBuffer &outBuf, Operation *allocaPoint,
                              PatternRewriter &rewriter) {
  if (!outBuf.empty()) {
    LDBG("Output memory will be used for direct tile stores. Outbuf in "
         "candidate: "
         << outBuf.memRef.getType());
    return outBuf;
  }

  if (!accBuf.empty()) {
    LDBG("Result will be stored to accumulator buffer.");
    return accBuf;
  }

  LDBG("Allocating buffer for the result.");
  return allocateTmpBufferLocal(loc, cast<VectorType>(val.getType()),
                                allocaPoint, rewriter);
}

void replaceLoop(DotOpCandidate &candidate,
                 ModuleTensorPtrShapeInfoAnalysis &shapeAnalysis,
                 PatternRewriter &rewriter) {
  triton::cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();

  auto extractMemref = [&](Value ptr) {
    auto tensorTy = dyn_cast<RankedTensorType>(
        dyn_cast<PointerType>(ptr.getType()).getPointeeType());
    auto elemTy = tensorTy.getElementType();
    auto shapeInfo = shapeAnalysis.getPtrShapeInfo(ptr);
    Type memRefTy;
    if (shapeInfo && shapeInfo->getRank() > 0) {
      auto layout = StridedLayoutAttr::get(ctx, 0, shapeInfo->getStrides());
      memRefTy = MemRefType::get(shapeInfo->getShape(), elemTy, layout);
    } else {
      SmallVector<int64_t> dynVals(tensorTy.getRank(), ShapedType::kDynamic);
      auto layout = StridedLayoutAttr::get(ctx, 0, dynVals);
      memRefTy = MemRefType::get(dynVals, elemTy, layout);
    }
    return rewriter.create<ExtractMemRefOp>(loc, memRefTy, ptr);
  };

  // VectorType lhsTy = cast<VectorType>(candidate.op.getA().getType());
  // VectorType rhsTy = cast<VectorType>(candidate.op.getB().getType());
  auto addSubView = [&](Value vecVal, ValueRange indices, Value memRef) {
    LDBG("  Reusing the original memref for a buffer: " << memRef);
    auto vecTy = cast<VectorType>(vecVal.getType());
    // auto memRefTy = MemRefType::get(vecTy.getShape(),
    // vecTy.getElementType());
    auto ctx = rewriter.getContext();
    SmallVector<int64_t> strides(vecTy.getRank(), 1);

    Value memRef_view = rewriter.create<memref::SubViewOp>(
        loc, memRef, getAsOpFoldResult(indices),
        getAsIndexOpFoldResult(ctx, vecTy.getShape()),
        getAsIndexOpFoldResult(ctx, strides));
    return memRef_view;
  };

  auto lhsTritonPtr = candidate.lhsBuf.origBlockPtr;
  auto lhsMemRef = extractMemref(lhsTritonPtr);
  // auto lhsRank = dyn_cast<MemRefType>(memRef.getType()).getRank();
  auto lhsIndices =
      rewriter.create<ExtractIndicesOp>(loc, lhsTritonPtr).getResults();
  auto lhsSubView = addSubView(candidate.op.getA(), lhsIndices, lhsMemRef);
  candidate.lhsBuf.memRef = lhsSubView;
  candidate.lhsBuf.indices = lhsIndices;

  auto rhsTritonPtr = candidate.rhsBuf.origBlockPtr;
  auto rhsMemRef = extractMemref(rhsTritonPtr);
  // auto rhsRank = dyn_cast<MemRefType>(memRef.getType()).getRank();
  auto rhsIndices =
      rewriter.create<ExtractIndicesOp>(loc, rhsTritonPtr).getResults();
  auto rhsSubView = addSubView(candidate.op.getB(), rhsIndices, rhsMemRef);
  candidate.rhsBuf.memRef = rhsSubView;
  candidate.rhsBuf.indices = rhsIndices;
}

Value computeStepInBytes(Location loc, memref::ExtractStridedMetadataOp meta,
                         ArrayRef<Value> steps, PatternRewriter &rewriter) {
  Value res = index_cst(0);
  if (steps.empty())
    return res;

  SmallVector<Value> strides = meta.getStrides();
  for (uint i = 0; i < strides.size(); i++) {
    Value stride = strides[i];
    Value step = steps[i];
    if (!step.getType().isIndex())
      step = op_index_cast(rewriter.getIndexType(), step);
    res = op_addi(res, op_muli(step, stride));
  }

  Value dtSize = index_cst(
      getElementTypeOrSelf(meta.getBaseBuffer()).getIntOrFloatBitWidth() / 8);
  res = op_muli(res, dtSize);
  return res;
}

// 1. DotOp is replaced with BRGEMM call (aka loop collapse)
//   CONDITIONS:
//     - Acc is loop-carried
//     - Input buffers should exists, have steps and basic block pointers
//     - Buffer steps and block pointers should be loop invariants
//   - All generation goes out of the loop
//   - The original dot op result uses are replaced with its acc operand (to
//   make it dead code)

// 2. DotOp is replaced with GEMM call
// a) Acc is loop-carried
//   - Create buf for acc before the loop
//     -- OPT: use output buffer instead of the temporary one
//   - Put init acc values into the buf before the loop
//   - Load acc from buf after the loop and replace loop result uses with loaded
//   acc
//   - The original dot op result uses are replaced with its acc operand (to
//   make it dead code)
//   -- OPT: Remove the original store if output buffer was used for acc

// b) Acc is not loop-carried
//   - Create buf for acc before the loop
//   - Put acc value into the buf before the dot op
//   - Load acc from buf after GEMM call and replace orig dot op result uses
//   with loaded acc
//   - The original dot op is removed

LogicalResult
convertCandidate(DotOpCandidate &candidate,
                 ModuleTensorPtrShapeInfoAnalysis &shapeInfoAnalysis,
                 PatternRewriter &rewriter) {
  triton::cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  // MLIRContext *ctx = op.getContext();

  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  VectorType resTy = cast<VectorType>(op.getResult().getType());
  Type resElemTy = resTy.getElementType();
  // FloatType float32 = FloatType::getF32(rewriter.getContext());
  // IndexType indexType = rewriter.getIndexType();

  scf::ForOp forOp = dyn_cast<scf::ForOp>(op->getParentOp());
  Value numBatches = index_cst(1);
  if (candidate.isAccLoopCarried && candidate.canFuseLoop) {
    // We can fully replace the loop with one op.

    // Initial tile values are loaded before the loop and then directly
    // used within the loop. Later, new iter values will be added to
    // add loop carried-dependencies for accumulator tiles and accInitTiles
    // will be used as initializers for them.
    rewriter.setInsertionPoint(forOp);
    replaceLoop(candidate, shapeInfoAnalysis, rewriter);
    LDBG("Loading accumulator to tiles before the loop.");

    numBatches = op_divui(op_subi(forOp.getUpperBound(), forOp.getLowerBound()),
                          forOp.getStep());
    numBatches = op_index_cast(rewriter.getIndexType(), numBatches);
  }

  // If we don't work with a loop and want to directly store tiles into output
  // memory, then use the original store as insertion point to have its buffer
  // values available for generated code.
  if (!candidate.isAccLoopCarried && !candidate.outBuf.empty())
    rewriter.setInsertionPoint(candidate.origStore);

  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  ModuleOp module = op->template getParentOfType<ModuleOp>();

  auto blockM = int_cst(integer64, candidate.blockM);
  auto blockN = int_cst(integer64, candidate.blockN);
  auto blockK = int_cst(integer64, candidate.blockK);

  if (candidate.lhsBuf.empty()) {
    candidate.lhsBuf =
        storeToTmpBuffer(loc, candidate.op.getA(), allocaPoint, rewriter);
  }

  if (candidate.rhsBuf.empty()) {
    candidate.rhsBuf =
        storeToTmpBuffer(loc, candidate.op.getB(), allocaPoint, rewriter);
  }

  Value accToStore = op.getC();
  if (candidate.isAccLoopCarried) {
    LDBG("Setting insertion op to forOp. (accToStore)");
    forOp = cast<scf::ForOp>(op->getParentOp());
    accToStore = getInitAccValue(accToStore);
  }

  MemBuffer accBuf;
  {
    // If accumulator is bufferized then we should move initial values before
    // the loop.
    OpBuilder::InsertionGuard g(rewriter);
    if (candidate.isAccLoopCarried) {
      LDBG("String Setting insertion op to forOp. (accBuf)");
      rewriter.setInsertionPoint(forOp);
    }
    // Currently, acc always needs to be FP32.
    accToStore = maybeCast(loc, accToStore, rewriter.getF32Type(), rewriter);
    accBuf =
        prepareTensorBuffer(rewriter, loc, accToStore, {}, false, allocaPoint);
  }

  MemBuffer resBuf = prepareResultBuffer(
      loc, op.getResult(), accBuf, candidate.outBuf, allocaPoint, rewriter);

  auto metadataA = rewriter.create<memref::ExtractStridedMetadataOp>(
      loc, candidate.lhsBuf.memRef);
  auto metadataB = rewriter.create<memref::ExtractStridedMetadataOp>(
      loc, candidate.rhsBuf.memRef);
  auto metadataAcc =
      rewriter.create<memref::ExtractStridedMetadataOp>(loc, accBuf.memRef);

  Value lda = metadataA.getStrides()[metadataA.getStrides().size() - 2];
  Value ldb = metadataB.getStrides()[metadataB.getStrides().size() - 2];
  Value ldc = metadataAcc.getStrides()[metadataAcc.getStrides().size() - 2];

  auto lhsDnnType = int_cst(integer64, getDnnlDataTypeVal(op.getA().getType()));
  auto rhsDnnType = int_cst(integer64, getDnnlDataTypeVal(op.getB().getType()));
  auto accDnnType =
      int_cst(integer64, getDnnlDataTypeVal(rewriter.getF32Type()));

  Value brgemm = rewriter.create<triton::cpu::BrgemmCreate>(
      loc, rewriter.getIndexType(), blockM, blockN, blockK, numBatches, lda,
      ldb, ldc, lhsDnnType, rhsDnnType, accDnnType);

  auto rhsTypeSize =
      int_cst(integer64, op.getB().getType().getElementTypeBitWidth() / 8);
  Value rhsBlockSizeInBytes = op_muli(op_muli(blockN, blockK), rhsTypeSize);

  Value transform = rewriter.create<triton::cpu::TransformCreate>(
      loc, rewriter.getIndexType(), blockK, blockN, ldb, blockN, rhsDnnType,
      rhsDnnType);

  LDBG("[prepareResultBuffer] prepared acc buf: " << accBuf.memRef);
  LDBG("[prepareResultBuffer] prepared res buf: " << resBuf.memRef);
  LDBG("lhsBuf: {   memref "
       << candidate.lhsBuf.memRef << "\n "
       << "           indices " << candidate.lhsBuf.indices.size() << "\n"
       << "              step " << candidate.lhsBuf.step.size() << "\n"
       << "          blockptr " << candidate.lhsBuf.origBlockPtr << "\n"
       << "        transposed " << candidate.lhsBuf.transposed << "\n} \n");
  LDBG("rhsBuf: {   memref "
       << candidate.rhsBuf.memRef << "\n "
       << "           indices " << candidate.rhsBuf.indices.size() << "\n"
       << "              step " << candidate.rhsBuf.step.size() << "\n"
       << "          blockptr " << candidate.rhsBuf.origBlockPtr << "\n"
       << "        transposed " << candidate.rhsBuf.transposed << "\n} \n");

  Value lhsStepInBytes =
      computeStepInBytes(loc, metadataA, candidate.lhsBuf.step, rewriter);
  Value rhsStepInBytes =
      computeStepInBytes(loc, metadataB, candidate.rhsBuf.step, rewriter);

  rewriter.create<triton::cpu::CallBrgemmWithTransform>(
      loc, transform, brgemm, candidate.lhsBuf.memRef, candidate.rhsBuf.memRef,
      resBuf.memRef, accBuf.memRef, lhsStepInBytes, rhsStepInBytes,
      rhsBlockSizeInBytes, numBatches);

  if (candidate.isAccLoopCarried && candidate.canFuseLoop) {
    LDBG("Loading the result to a vector to replace orig op result.");
    auto rank = dyn_cast<MemRefType>(resBuf.memRef.getType()).getRank();
    SmallVector<bool, 4> inBounds(rank, false);
    Value newVal = rewriter.create<vector::TransferReadOp>(
        loc, cast<VectorType>(toFp32(resTy)), resBuf.memRef, resBuf.indices,
        inBounds);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, resElemTy, rewriter);
    // rewriter.replaceOp(op, newVal);
    // rewriter.eraseOp(*op.getResult().user_begin());
    // rewriter.eraseOp(op);
    LDBG("Printing module before replace:");
    LDBG(module);

    rewriter.replaceOp(forOp, ValueRange{newVal, candidate.lhsBuf.memRef,
                                         candidate.rhsBuf.memRef});
    return success();
  }

  if (candidate.isAccLoopCarried) {
    rewriter.setInsertionPointAfter(forOp);
    auto rank = dyn_cast<MemRefType>(resBuf.memRef.getType()).getRank();
    SmallVector<bool, 4> inBounds(rank, false);
    Value newVal = rewriter.create<vector::TransferReadOp>(
        loc, cast<VectorType>(toFp32(resTy)), resBuf.memRef, resBuf.indices,
        inBounds);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, resElemTy, rewriter);
    int resIdx = op.getResult().getUses().begin()->getOperandNumber();
    Value loopRes = forOp.getResult(resIdx);
    loopRes.replaceAllUsesWith(newVal);
    rewriter.replaceOp(op, op.getC());
    return success();
  }
  if (candidate.outBuf.empty()) {
    LDBG("Loading the result to a vector to replace orig op result.");
    Value newVal = rewriter.create<vector::TransferReadOp>(
        loc, cast<VectorType>(toFp32(resTy)), resBuf.memRef, resBuf.indices);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, resElemTy, rewriter);
    op.getResult().replaceAllUsesWith(newVal);
    rewriter.eraseOp(op);
    // rewriter.replaceOp(op, newVal);
    // rewriter.eraseOp(*op.getResult().user_begin());
    // rewriter.eraseOp(op);
    LDBG("Printing module before replace:");
    LDBG(module);
  } else {
    LDBG("Removing original operation and its use.");
    rewriter.eraseOp(*op.getResult().user_begin());
    rewriter.eraseOp(op);
  }

  return success();
}

struct ConvertDotToOneDNN
    : public triton::cpu::impl::ConvertDotToOneDNNBase<ConvertDotToOneDNN> {
  ConvertDotToOneDNN() = default;

  void runOnOperation() override {

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    ModuleTensorPtrShapeInfoAnalysis shapeInfoAnalysis(mod);

    SmallVector<DotOpCandidate, 2> candidates;
    mod->walk([this, &candidates](triton::cpu::DotOp op) {
      DotOpCandidate candidate;
      if (isOneDNNCandidate(op, convertInt8, convertFp16, convertBf16,
                            candidate)) {
        LLVM_DEBUG({
          LDBG("Found OneDNN candidate");
          LDBG("  Op: " << candidate.op);
          LDBG("  blockM: " << candidate.blockM);
          LDBG("  blockN: " << candidate.blockN);
          LDBG("  blockK: " << candidate.blockK);
          LDBG("  isAccLoopCarried: " << candidate.isAccLoopCarried);
          LDBG("  canFuseLoop: " << candidate.canFuseLoop);
          LDBG("  Has output buffer: " << (bool)!candidate.outBuf.empty());
        });
        candidates.push_back(candidate);
      }
      return WalkResult::advance();
    });

    for (auto &candidate : candidates) {
      LDBG("Starting conversion of candidate: " << candidate.op);
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.op);
      if (succeeded(convertCandidate(candidate, shapeInfoAnalysis, rewriter))) {
        LDBG("Conversion succeeded!");
      } else {
        LDBG("Conversion failed!");
      }
    }
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToOneDNN() {
  return std::make_unique<ConvertDotToOneDNN>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
