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

static inline int64_t getDnnlDataTypeVal(RewriterBase &rewriter,
                                         Attribute attr) {
  auto context = rewriter.getContext();
  auto tattr = dyn_cast_or_null<TypeAttr>(attr);
  assert(tattr);
  if (tattr == TypeAttr::get(FloatType::getF32(context))) {
    return static_cast<int64_t>(dnnl_f32);
  } else if (tattr == TypeAttr::get(FloatType::getF64(context))) {
    return static_cast<int64_t>(dnnl_f64);
  } else if (tattr == TypeAttr::get(FloatType::getBF16(context))) {
    return static_cast<int64_t>(dnnl_bf16);
  } else if (tattr == TypeAttr::get(FloatType::getF16(context))) {
    return static_cast<int64_t>(dnnl_f16);
  } else if (tattr == TypeAttr::get(
                          IntegerType::get(context, 32, IntegerType::Signed))) {
    return static_cast<int64_t>(dnnl_s32);
  } else if (tattr ==
             TypeAttr::get(IntegerType::get(context, 8, IntegerType::Signed))) {
    return static_cast<int64_t>(dnnl_s8);
  } else if (tattr == TypeAttr::get(IntegerType::get(context, 8,
                                                     IntegerType::Unsigned))) {
    return static_cast<int64_t>(dnnl_u8);
  }
  return static_cast<int64_t>(dnnl_data_type_undef);
}

// This structure is used to hold candidates for conversion to AMX
// Mul[F|I]Op operations.
struct AmxDotOpCandidate {
  // Operation to convert.
  triton::cpu::DotOp op;

  scf::ForOp forOp = nullptr;
  // Available LHS, RHS, and accumulator types are limited in AMX and we might
  // require additional casts. Here we keep actual element types used by LHS,
  // RHS, and accumulator in AMX tiles.
  Type lhsTileElemTy;
  Type rhsTileElemTy;
  Type accTileElemTy;
  // AMX tile row size is limited by 64 bytes, so M and N dimensions are limited
  // by 16 because accumulator always has 4-byte elements. K dimension for tiles
  // is limited by 64 / <size of input element>. Here we keep actual tile sizes.
  int64_t tileM;
  int64_t tileN;
  int64_t tileK;
  // If accumulator is updated in a loop, then this flag indicates if we
  // should keep it in tiles the whole loop and move back to vectors only
  // after the loop.
  bool isAccLoopCarried = false;
  bool isInputsAccessWithInvariants = false;
  // If we want to keep accumulator in tiles but it's too big, then we might
  // keep it bufferized instead.
  bool canDotOpMovedOutCycle = false;

  // If output buffer is used then keep the original vector store here.
  Operation *origStore = nullptr;

  // If resulting tiles are not required to be trasfered to vectors and can be
  // directly stored to the output memory instead, then this field holds a
  // buffer to use.
  MemBuffer outBuf;
  MemBuffer lhsBuf;
  MemBuffer rhsBuf;
};

// Return a value that holds the resulting loop carried accumulator value.
// It's one of ForOp results.
// Value getResValueForLoopCarriedAcc(vector::ContractionOp op) {
//   Value updAcc = op.getResult();
//   auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
//   auto &use = *updAcc.getUses().begin();
//   return forOp.getResult(use.getOperandNumber());
// }

// Choose tile and block sizes for the candidate. Tile sizes are determined
// by input shapes and types. Block sizes are chosen to minimize number of
// tile loads/stores including tile register spills.
void setupBlockAndTileSizes(ArrayRef<int64_t> lhsShape,
                            ArrayRef<int64_t> rhsShape,
                            AmxDotOpCandidate &candidate) {
  int64_t m = lhsShape[0];
  int64_t n = rhsShape[1];
  int64_t k = rhsShape[0];
  int64_t tileM = m;
  int64_t tileN = n;
  int64_t tileK = k;

  candidate.tileM = tileM;
  candidate.tileN = tileN;
  candidate.tileK = tileK;
}

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

// Check if a value is used only for a store and that this store can be
// replaced with tile stores. In this case fill appropriate fields in the
// candidate structure.
// void findOutputBuffer(Value val, AmxDotOpCandidate &candidate) {
//   if (val.hasOneUse() && false) {
//     auto store = dyn_cast<vector::TransferWriteOp>(*val.user_begin());
//     if (store && !hasMaskOrBoundsCheck(store))
//       candidate.outBuf = MemBuffer{store.getSource(), store.getIndices()};
//     candidate.origStore = store;
//     LDBG("[findOutputBuffer] cand outbuf: " << candidate.outBuf.memRef);
//   }
// }

// Check if val is available in memory.
void findInputBuffer(Value val, bool allowTransposed, MemBuffer &buf,
                     scf::ForOp &ptrToForOp) {

  if (allowTransposed) {
    auto transposeOp = val.getDefiningOp<vector::TransposeOp>();
    if (transposeOp) {
      val = transposeOp.getVector();
      buf.transposed = true;
    }
  }

  auto valLoad = val.getDefiningOp<vector::TransferReadOp>();
  if (!valLoad || hasMaskOrBoundsCheck(valLoad)) {
    LDBG("Couldn't find a buffer with input: " << val);
    return;
  }

  buf.memRef = valLoad.getSource();
  buf.indices = valLoad.getIndices();
  LLVM_DEBUG(
      DBGS() << "Found buffer with input: " << val << "\n";
      DBGS() << "  MemRef: " << buf.memRef << "\n"; DBGS() << "  Indices: ";
      llvm::interleaveComma(buf.indices, llvm::dbgs()); llvm::dbgs() << "\n");

  auto forOp = dyn_cast<scf::ForOp>(valLoad->getParentOp());
  if (!forOp) {
    LDBG("  Skip steps. Not in a for-loop.");
    return;
  }

  if (!ptrToForOp)
    ptrToForOp = forOp;

  auto extractMemRef = buf.memRef.getDefiningOp<ExtractMemRefOp>();
  if (!extractMemRef) {
    LDBG("  Skip steps. No ExtractMemRefOp.");
    return;
  }

  ExtractIndicesOp extractIndices;
  for (auto index : buf.indices) {
    auto def = index.getDefiningOp<ExtractIndicesOp>();
    if (!def || (extractIndices && def != extractIndices)) {
      LDBG("  Skip steps. No ExtractIndicesOp.");
      return;
    }
    extractIndices = def;
  }

  if (extractMemRef.getSrc() != extractIndices.getSrc()) {
    LDBG("  Skip steps. Mismatched ExtractMemRefOp and ExtractIndicesOp.");
    return;
  }

  BlockArgument blockPtrArg = dyn_cast<BlockArgument>(extractMemRef.getSrc());
  if (!blockPtrArg) {
    LDBG("  Skip steps. No block pointer arg.");
    return;
  }

  OpOperand *yieldOp = forOp.getTiedLoopYieldedValue(blockPtrArg);
  if (!yieldOp) {
    LDBG("  Skip steps. No block pointer in yield.");
    return;
  }

  auto advance = yieldOp->get().getDefiningOp<AdvanceOp>();
  if (!advance) {
    LDBG("  Skip steps. No AdvanceOp.");
    return;
  }

  if (advance.getPtr() != blockPtrArg) {
    LDBG("  Skip steps. AdvanceOp doesn't use block pointer arg.");
    return;
  }

  // CHeck that steaps are loop invariants.
  buf.step = advance.getOffsets();
  LLVM_DEBUG(DBGS() << "  Step: ";
             llvm::interleaveComma(advance.getOffsets(), llvm::dbgs());
             llvm::dbgs() << "\n");

  // Add code for invariant check
  OpOperand *tritonPtr = forOp.getTiedLoopInit(blockPtrArg);
  if (!tritonPtr) {
    LDBG("  Skip triton ptr. No Tied block ptr.");
    return;
  }

  buf.origTritonBlockPtr = tritonPtr->get();
}

static bool canBeHoisted(Operation *op,
                         function_ref<bool(OpOperand &)> condition) {
  // Do not move terminators.
  if (op->hasTrait<OpTrait::IsTerminator>())
    return false;

  // Walk the nested operations and check that all used values are either
  // defined outside of the loop or in a nested region, but not at the level of
  // the loop body.
  auto walkFn = [&](Operation *child) {
    for (OpOperand &operand : child->getOpOperands()) {
      // Ignore values defined in a nested region.
      if (op->isAncestor(operand.get().getParentRegion()->getParentOp()))
        continue;
      if (!condition(operand))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };
  return !op->walk(walkFn).wasInterrupted();
}

static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value)> definedOutside) {
  return canBeHoisted(
      op, [&](OpOperand &operand) { return definedOutside(operand.get()); });
}

bool isOpsLoopInvariantCode(SmallVector<Operation *, 4> ops,
                            LoopLikeOpInterface loopLike) {

  auto region = loopLike.getLoopRegions();
  auto definedOutside = [&](Value value) {
    // return isDefinedOutsideRegion(value, region);
    return loopLike.isDefinedOutsideOfLoop(value);
  };

  for (Operation *op : ops) {
    LLVM_DEBUG(llvm::dbgs() << "Original loop:\n"
                            << *op->getParentOp() << "\n");

    LLVM_DEBUG(llvm::dbgs() << "Checking op: " << *op << "\n");
    if (!canBeHoisted(op, definedOutside)) {
      return false;
    }
  }
  return true;
}

// Check if specified ContractionOp can be lowered to AMX operations.
// If conversion is possible, then true is returned and candidate
// structure is filled with detailed transformation info.
bool isOneDNNCandidate(triton::cpu::DotOp op, bool supportInt8,
                       bool supportFp16, bool supportBf16,
                       AmxDotOpCandidate &candidate) {
  // MLIRContext *ctx = op.getContext();
  VectorType lhsTy = cast<VectorType>(op.getA().getType());
  VectorType rhsTy = cast<VectorType>(op.getB().getType());
  VectorType accTy = cast<VectorType>(op.getC().getType());
  // VectorType resTy = cast<VectorType>(op.getType());

  LDBG("Considering candidate op: " << op);

  if (accTy.getRank() != 2) {
    LDBG("  Drop candidate. Only 2D case is supported.");
    return false;
  }

  if (accTy.getElementType().isInteger()) {
    LDBG("  Drop candidate. Integer type is not supported.");
    return false;
  }

  candidate.lhsTileElemTy = lhsTy.getElementType();
  candidate.rhsTileElemTy = rhsTy.getElementType();
  candidate.accTileElemTy = accTy.getElementType();

  candidate.op = op;
  setupBlockAndTileSizes(lhsTy.getShape(), rhsTy.getShape(), candidate);

  candidate.isAccLoopCarried = isLoopCarriedAcc(op.getC());

  // if (lhsTy.getElementType() == candidate.lhsElemTy)
  findInputBuffer(op.getA(), true, candidate.lhsBuf, candidate.forOp);

  // if (rhsTy.getElementType() == candidate.rhsElemTy)
  findInputBuffer(op.getB(), false, candidate.rhsBuf, candidate.forOp);

  // maybe we should just check that loop exists
  if (candidate.isAccLoopCarried) {
    // candidate.lhsBuf.indices;
    if (candidate.lhsBuf.indices.empty() || candidate.rhsBuf.indices.empty() ||
        candidate.lhsBuf.step.empty() || candidate.rhsBuf.step.empty()) {
      // can't do
      return true;
    }

    SmallVector<Value, 4> toCHeckVals;
    toCHeckVals.append(candidate.lhsBuf.indices);
    toCHeckVals.append(candidate.lhsBuf.step);
    toCHeckVals.append(candidate.rhsBuf.indices);
    toCHeckVals.append(candidate.rhsBuf.step);

    SmallVector<Operation *, 4> toCheckInvariance;
    auto toOp = [](Value val) -> Operation * { return val.getDefiningOp(); };
    std::transform(toCHeckVals.begin(), toCHeckVals.end(),
                   toCheckInvariance.begin(), toOp);

    candidate.isInputsAccessWithInvariants =
        isOpsLoopInvariantCode(toCheckInvariance, candidate.forOp);
  };

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

void checkInputBtw(SmallVector<Value> &inps, PatternRewriter &rewriter) {
  for (uint i = 0; i < inps.size(); i++) {
    auto inp = inps[i];
    Type inpT = inp.getType();
    if (!inpT.isInteger()) {
      continue;
    }
    if (inpT.getIntOrFloatBitWidth() != 64) {
      LDBG("Type: " << inpT << " val: " << inp);
      inps[i] = rewriter.create<arith::IndexCastOp>(
          inp.getLoc(), IndexType::get(rewriter.getContext()), inp);
      LDBG("created: " << inps[i]);
    }
  }
}

void replaceLoop(AmxDotOpCandidate &candidate,
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

  auto lhsTritonPtr = candidate.lhsBuf.origTritonBlockPtr;
  auto lhsMemRef = extractMemref(lhsTritonPtr);
  // auto lhsRank = dyn_cast<MemRefType>(memRef.getType()).getRank();
  auto lhsIndices =
      rewriter.create<ExtractIndicesOp>(loc, lhsTritonPtr).getResults();
  auto lhsSubView = addSubView(candidate.op.getA(), lhsIndices, lhsMemRef);
  candidate.lhsBuf.memRef = lhsSubView;
  candidate.lhsBuf.indices = lhsIndices;

  auto rhsTritonPtr = candidate.rhsBuf.origTritonBlockPtr;
  auto rhsMemRef = extractMemref(rhsTritonPtr);
  // auto rhsRank = dyn_cast<MemRefType>(memRef.getType()).getRank();
  auto rhsIndices =
      rewriter.create<ExtractIndicesOp>(loc, rhsTritonPtr).getResults();
  auto rhsSubView = addSubView(candidate.op.getB(), rhsIndices, rhsMemRef);
  candidate.rhsBuf.memRef = rhsSubView;
  candidate.rhsBuf.indices = rhsIndices;
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
convertCandidate(AmxDotOpCandidate &candidate,
                 ModuleTensorPtrShapeInfoAnalysis &shapeInfoAnalysis,
                 PatternRewriter &rewriter) {
  triton::cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  // MLIRContext *ctx = op.getContext();

  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  // FloatType float32 = FloatType::getF32(rewriter.getContext());
  // IndexType indexType = rewriter.getIndexType();

  scf::ForOp forOp = dyn_cast<scf::ForOp>(op->getParentOp());
  Value num_batches = rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), 1));
  if (candidate.isAccLoopCarried && candidate.isInputsAccessWithInvariants) {
    // We can fully replace the loop with one op.

    // Initial tile values are loaded before the loop and then directly
    // used within the loop. Later, new iter values will be added to
    // add loop carried-dependencies for accumulator tiles and accInitTiles
    // will be used as initializers for them.
    rewriter.setInsertionPoint(forOp);
    replaceLoop(candidate, shapeInfoAnalysis, rewriter);
    LDBG("Loading accumulator to tiles before the loop.");

    auto loopRange = rewriter.create<arith::SubIOp>(loc, forOp.getUpperBound(),
                                                    forOp.getLowerBound());
    num_batches =
        rewriter.create<arith::DivUIOp>(loc, loopRange, forOp.getStep());
  }
  llvm::errs() << "Num batches: " << num_batches << "\n";

  // If we don't work with a loop and want to directly store tiles into output
  // memory, then use the original store as insertion point to have its buffer
  // values available for generated code.
  if (!candidate.isAccLoopCarried && !candidate.outBuf.empty())
    rewriter.setInsertionPoint(candidate.origStore);

  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  ModuleOp module = op->template getParentOfType<ModuleOp>();

  auto block_m = rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), candidate.tileM));
  auto block_n = rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), candidate.tileN));
  auto block_k = rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), candidate.tileK));

  auto block_k_ind = rewriter.create<arith::IndexCastOp>(
      loc, IndexType::get(rewriter.getContext()), block_k);

  if (candidate.lhsBuf.empty()) {
    candidate.lhsBuf =
        storeToTmpBuffer(loc, candidate.op.getA(), allocaPoint, rewriter);
  }

  if (candidate.rhsBuf.empty()) {
    candidate.rhsBuf =
        storeToTmpBuffer(loc, candidate.op.getB(), allocaPoint, rewriter);
  }

  Value acc = maybeCast(loc, op.getC(), candidate.accTileElemTy, rewriter);
  Value accToStore = acc;

  LDBG("isAccHoisted: " << candidate.isAccLoopCarried << " is inps invariants: "
                        << candidate.isInputsAccessWithInvariants);
  if (candidate.isAccLoopCarried) {
    LDBG("Setting insertion op to forOp. (accToStore)");
    forOp = cast<scf::ForOp>(op->getParentOp());
    accToStore = getInitAccValue(acc);
  }

  MemBuffer accBuf;
  {
    // If accumulator is bufferized then we should move initial values before
    // the loop.
    OpBuilder::InsertionGuard g(rewriter);
    if (candidate.isAccLoopCarried) {
      LDBG("Setting insertion op to forOp. (accBuf)");
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
  lda = rewriter.create<arith::IndexCastOp>(loc, integer64, lda);
  ldb = rewriter.create<arith::IndexCastOp>(loc, integer64, ldb);
  ldc = rewriter.create<arith::IndexCastOp>(loc, integer64, ldc);

  // dtypeA, dtypeB, dtypeC
  SmallVector<Attribute, 2> brgemmDtypes{
      TypeAttr::get(getElementTypeOrSelf(op.getA().getType())),
      TypeAttr::get(getElementTypeOrSelf(op.getB().getType())),
      TypeAttr::get(rewriter.getF32Type())};

  LDBG("ElemTypes: " << brgemmDtypes[0] << ", " << brgemmDtypes[1] << ", "
                     << brgemmDtypes[2]);

  auto dtypes = rewriter.getArrayAttr(brgemmDtypes);

  auto dtypeAAttr = IntegerAttr::get(rewriter.getI64Type(),
                                     getDnnlDataTypeVal(rewriter, dtypes[0]));
  auto dtypeBAttr = IntegerAttr::get(rewriter.getI64Type(),
                                     getDnnlDataTypeVal(rewriter, dtypes[1]));
  auto dtypeCAttr = IntegerAttr::get(rewriter.getI64Type(),
                                     getDnnlDataTypeVal(rewriter, dtypes[2]));

  auto a_dt = rewriter.create<arith::ConstantOp>(loc, integer64, dtypeAAttr);
  auto b_dt = rewriter.create<arith::ConstantOp>(loc, integer64, dtypeBAttr);
  auto c_dt = rewriter.create<arith::ConstantOp>(loc, integer64, dtypeCAttr);

  Value num_batches_ind = num_batches;
  if (!num_batches.getType().isIndex()) {
    num_batches_ind = rewriter.create<arith::IndexCastOp>(
        loc, IndexType::get(rewriter.getContext()), num_batches);
  }

  Value brgemm = rewriter.create<triton::cpu::BrgemmCreate>(
      loc, rewriter.getIndexType(), block_m, block_n, block_k, num_batches_ind,
      lda, ldb, ldc, a_dt, b_dt, c_dt);

  auto in_ld = block_n;

  auto b_data_type_size = rewriter.create<arith::ConstantOp>(
      loc, integer64,
      IntegerAttr::get(rewriter.getI64Type(),
                       op.getB().getType().getElementTypeBitWidth() / 8));
  auto b_data_type_size_ind = rewriter.create<arith::IndexCastOp>(
      loc, IndexType::get(rewriter.getContext()), b_data_type_size);

  Value block_k_ver = block_k;
  Value b_data_type_size_ver = b_data_type_size;
  if (!ldb.getType().isInteger()) {
    block_k_ver = block_k_ind;
    b_data_type_size_ver = b_data_type_size_ind;
  }
  auto blocked_B_size = rewriter.create<arith::MulIOp>(loc, ldb, block_k_ver);
  blocked_B_size =
      rewriter.create<arith::MulIOp>(loc, blocked_B_size, b_data_type_size_ver);

  Value transform = rewriter.create<triton::cpu::TransformCreate>(
      loc, rewriter.getIndexType(), block_k, block_n, in_ld, ldb, b_dt, b_dt);

  LDBG("[prepareResultBuffer] prepared acc buf: " << accBuf.memRef);
  LDBG("[prepareResultBuffer] prepared res buf: " << resBuf.memRef);
  LDBG("lhsBuf: {   memref "
       << candidate.lhsBuf.memRef << "\n "
       << "           indices " << candidate.lhsBuf.indices.size() << "\n"
       << "              step " << candidate.lhsBuf.step.size() << "\n"
       << "          blockptr " << candidate.lhsBuf.origTritonBlockPtr << "\n"
       << "        transposed " << candidate.lhsBuf.transposed << "\n} \n");
  LDBG("rhsBuf: {   memref "
       << candidate.rhsBuf.memRef << "\n "
       << "           indices " << candidate.rhsBuf.indices.size() << "\n"
       << "              step " << candidate.rhsBuf.step.size() << "\n"
       << "          blockptr " << candidate.rhsBuf.origTritonBlockPtr << "\n"
       << "        transposed " << candidate.rhsBuf.transposed << "\n} \n");

  auto zero = rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), 0));

  auto lhsMemrefType = dyn_cast<MemRefType>(candidate.lhsBuf.memRef.getType());
  auto lhsStrides = metadataA.getStrides();
  SmallVector<Value> lhsSteps;
  if (candidate.lhsBuf.step.empty()) {
    lhsSteps = SmallVector<Value>(lhsStrides.size(), zero);
  } else {
    lhsSteps = candidate.lhsBuf.step;
  }
  Value lhsAccSize = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value lhsDtSize = rewriter.create<arith::ConstantIndexOp>(
      loc, lhsMemrefType.getElementTypeBitWidth() / 8);

  for (uint i = 0; i < metadataA.getStrides().size(); i++) {
    Value cast_lhs_stride = lhsStrides[i];
    Value cast_lhs_steps = lhsSteps[i];
    if (!cast_lhs_stride.getType().isIndex()) {
      cast_lhs_stride = rewriter.create<arith::IndexCastOp>(
          loc, IndexType::get(rewriter.getContext()), cast_lhs_stride);
    }
    if (!cast_lhs_steps.getType().isIndex()) {
      cast_lhs_steps = rewriter.create<arith::IndexCastOp>(
          loc, IndexType::get(rewriter.getContext()), cast_lhs_steps);
    }
    auto tmp =
        rewriter.create<arith::MulIOp>(loc, cast_lhs_stride, cast_lhs_steps);
    lhsAccSize = rewriter.create<arith::AddIOp>(loc, tmp, lhsAccSize);
  }

  lhsAccSize = rewriter.create<arith::MulIOp>(loc, lhsAccSize, lhsDtSize);

  auto rhsStrides = metadataB.getStrides();
  SmallVector<Value> rhsSteps;
  if (candidate.rhsBuf.step.empty()) {
    rhsSteps = SmallVector<Value>(rhsStrides.size(), zero);
  } else {
    rhsSteps = candidate.rhsBuf.step;
  }
  auto rhsMemrefType = dyn_cast<MemRefType>(candidate.rhsBuf.memRef.getType());
  Value rhsAccSize = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value rhsDtSize = rewriter.create<arith::ConstantIndexOp>(
      loc, rhsMemrefType.getElementTypeBitWidth() / 8);
  for (uint i = 0; i < metadataA.getStrides().size(); i++) {
    Value cast_rhs_stride = rhsStrides[i];
    Value cast_rhs_steps = rhsSteps[i];
    if (!cast_rhs_stride.getType().isIndex()) {
      cast_rhs_stride = rewriter.create<arith::IndexCastOp>(
          loc, IndexType::get(rewriter.getContext()), cast_rhs_stride);
    }
    if (!cast_rhs_steps.getType().isIndex()) {
      cast_rhs_steps = rewriter.create<arith::IndexCastOp>(
          loc, IndexType::get(rewriter.getContext()), cast_rhs_steps);
    }
    auto tmp =
        rewriter.create<arith::MulIOp>(loc, cast_rhs_stride, cast_rhs_steps);
    rhsAccSize = rewriter.create<arith::AddIOp>(loc, tmp, rhsAccSize);
  }
  rhsAccSize = rewriter.create<arith::MulIOp>(loc, rhsAccSize, rhsDtSize);

  SmallVector<Value> sizeAndSteps{lhsAccSize, rhsAccSize, blocked_B_size,
                                  num_batches_ind};
  checkInputBtw(sizeAndSteps, rewriter);

  rewriter.create<triton::cpu::CallBrgemmWithTransform>(
      loc, transform, brgemm, candidate.lhsBuf.memRef, candidate.rhsBuf.memRef,
      resBuf.memRef, accBuf.memRef, sizeAndSteps[0], sizeAndSteps[1],
      sizeAndSteps[2], sizeAndSteps[3]);

  if (candidate.isAccLoopCarried && candidate.isInputsAccessWithInvariants) {
    LDBG("Loading the result to a vector to replace orig op result.");
    auto rank = dyn_cast<MemRefType>(resBuf.memRef.getType()).getRank();
    SmallVector<bool, 4> inBounds(rank, false);
    Value newVal = rewriter.create<vector::TransferReadOp>(
        loc, cast<VectorType>(toFp32(acc.getType())), resBuf.memRef,
        resBuf.indices, inBounds);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, candidate.accTileElemTy, rewriter);
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
        loc, cast<VectorType>(toFp32(acc.getType())), resBuf.memRef,
        resBuf.indices, inBounds);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, candidate.accTileElemTy, rewriter);
    int resIdx = op.getResult().getUses().begin()->getOperandNumber();
    Value loopRes = forOp.getResult(resIdx);
    loopRes.replaceAllUsesWith(newVal);
    rewriter.replaceOp(op, op.getC());
    return success();
  }
  if (candidate.outBuf.empty()) {
    LDBG("Loading the result to a vector to replace orig op result.");
    Value newVal = rewriter.create<vector::TransferReadOp>(
        loc, cast<VectorType>(toFp32(acc.getType())), resBuf.memRef,
        resBuf.indices);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, candidate.accTileElemTy, rewriter);
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

    SmallVector<AmxDotOpCandidate, 2> candidates;
    mod->walk([this, &candidates](triton::cpu::DotOp op) {
      AmxDotOpCandidate candidate;
      if (isOneDNNCandidate(op, convertInt8, convertFp16, convertBf16,
                            candidate)) {
        LLVM_DEBUG({
          LDBG("Found OneDNN candidate");
          LDBG("  Op: " << candidate.op);
          LDBG("  LhsTileElemTy: " << candidate.lhsTileElemTy);
          LDBG("  RhsTileElemTy: " << candidate.rhsTileElemTy);
          LDBG("  AccTileElemTy: " << candidate.accTileElemTy);
          LDBG("  TileM: " << candidate.tileM);
          LDBG("  TileN: " << candidate.tileN);
          LDBG("  TileK: " << candidate.tileK);
          LDBG("  isAccLoopCarried: " << candidate.isAccLoopCarried);
          LDBG("  isInputsAccessWithInvariants: "
               << candidate.isInputsAccessWithInvariants);
          LDBG("  Has output buffer: " << (bool)!candidate.outBuf.empty());
        });
        std::cout << "  isAccLoopCarried: " << std::boolalpha
                  << candidate.isAccLoopCarried << "\n";
        std::cout << "  isInputsAccessWithInvariants: " << std::boolalpha
                  << candidate.isInputsAccessWithInvariants << "\n";
        std::cout << "  Has output buffer: " << std::boolalpha
                  << (bool)!candidate.outBuf.empty() << "\n";
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
