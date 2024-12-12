#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <iostream>
#include <utility>

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTTOFMA
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// This structure is used to hold candidates for conversion to FMA operations.
struct FmaDotOpCandidate {
  // Operation to convert.
  cpu::DotOp op;
  // Here we keep actual element types used by LHS, RHS, and accumulator for
  // computation.
  Type lhsElemTy;
  Type rhsElemTy;
  Type accElemTy;
  // Accumulator size.
  int64_t accVecSize;
  int64_t accRows;
  // If accumulator is updated in a loop, then this flag indicates if we
  // should keep it in registers the whole loop.
  bool keepAccOnRegs = false;
  // Memory buffer holding LHS. Can be empty if LHS is not a result of a
  // simple load.
  MemBuffer lhsBuf;
  // Memory buffer holding RHS. Can be empty if RHS is not a result of a
  // simple load.
  MemBuffer rhsBuf;
};

// Check if input and output types can be handled by FMA (possibly, using
// additional casts for input/output). Returns true if FMA lowering is possible.
// In this case, element type fields of the candidate structure are filled
// with actual types to be used in lowering.
bool checkElemTypes(Type lhsElemTy, Type rhsElemTy, Type accElemTy,
                    Type resElemTy, FmaDotOpCandidate &candidate) {
  MLIRContext *ctx = lhsElemTy.getContext();
  if (lhsElemTy.isInteger() || rhsElemTy.isInteger() || resElemTy.isInteger()) {
    LDBG("Drop candidate because int types are not supported.");
    return false;
  }

  // Find a type to use for computations. Here we assume FMA works on FP32
  // and FP64, so smaller types are promoted. Flags should be added to cover
  // other cases.
  Type commonInputElemTy;
  if (lhsElemTy.isF64() || rhsElemTy.isF64() || resElemTy.isF64())
    commonInputElemTy = Float64Type::get(ctx);
  else
    commonInputElemTy = Float32Type::get(ctx);

  candidate.lhsElemTy = commonInputElemTy;
  candidate.rhsElemTy = commonInputElemTy;
  candidate.accElemTy = commonInputElemTy;

  return true;
}

// Check input shapes. Currently, support only 2D cases and ignore small
// inputs.
bool checkInputShapes(VectorType lhsTy, VectorType resTy) {
  if (lhsTy.getRank() != 2)
    return false;

  if (resTy.getDimSize(1) < 8)
    return false;

  return true;
}

// Check if specified ContractionOp can be lowered to FMA operations.
// If conversion is possible, then true is returned and candidate
// structure is filled with detailed transformation info.
bool isFmaCandidate(cpu::DotOp op, FmaDotOpCandidate &candidate) {
  MLIRContext *ctx = op.getContext();
  VectorType lhsTy = op.getA().getType();
  VectorType rhsTy = op.getB().getType();
  VectorType accTy = op.getC().getType();
  VectorType resTy = op.getType();

  LDBG("Considering candidate op: " << op);

  // Check if input and output types match available hardware capabilities.
  // If check is successful then effective element types are assigned to the
  // candidate.
  if (!checkElemTypes(lhsTy.getElementType(), rhsTy.getElementType(),
                      accTy.getElementType(), resTy.getElementType(),
                      candidate))
    return false;

  // Check input shapes.
  if (!checkInputShapes(lhsTy, resTy))
    return false;

  candidate.op = op;
  candidate.accVecSize = resTy.getDimSize(1);
  candidate.accRows = resTy.getDimSize(0);
  candidate.keepAccOnRegs = isLoopCarriedAcc(op.getC());

  if (lhsTy.getElementType() == candidate.lhsElemTy)
    candidate.lhsBuf = findInputBuffer(op.getA(), true);
  if (rhsTy.getElementType() == candidate.rhsElemTy)
    candidate.rhsBuf = findInputBuffer(op.getB(), false);

  return true;
}

MemBuffer storeToTmpBuffer(Location loc, Value val, Operation *allocaPoint,
                           PatternRewriter &rewriter) {
  LDBG("Storing vector to a temporary buffer: " << val);
  auto vecTy = cast<VectorType>(val.getType());
  MemBuffer buf = allocateTmpBuffer(loc, vecTy, allocaPoint, rewriter);
  rewriter.create<vector::TransferWriteOp>(loc, val, buf.memRef, buf.indices);
  return buf;
}

SmallVector<Value> shiftIndices(Location loc, ArrayRef<Value> indices,
                                bool transposed, int64_t m, int64_t n,
                                PatternRewriter &rewriter) {
  SmallVector<Value> res(indices.begin(), indices.end() - 2);
  if (transposed)
    std::swap(m, n);
  res.push_back(shiftIndex(loc, *(indices.end() - 2), m, rewriter));
  res.push_back(shiftIndex(loc, *(indices.end() - 1), n, rewriter));
  return res;
}

SmallVector<Value> shiftIndices(Location loc, const MemBuffer &buf, int64_t m,
                                int64_t n, PatternRewriter &rewriter) {
  return shiftIndices(loc, buf.indices, buf.transposed, m, n, rewriter);
}

Value loadRow(Location loc, VectorType resTy, const MemBuffer &buf, int64_t m,
              PatternRewriter &rewriter) {
  assert(!buf.empty());
  SmallVector<Value> indices = buf.indices;
  indices[indices.size() - 2] =
      shiftIndex(loc, indices[indices.size() - 2], m, rewriter);
  return rewriter.create<vector::LoadOp>(loc, resTy, buf.memRef, indices);
}

void storeRow(Location loc, const MemBuffer &buf, int64_t rowIdx, Value vec,
              PatternRewriter &rewriter) {
  SmallVector<Value> indices = buf.indices;
  indices[indices.size() - 2] =
      shiftIndex(loc, buf.indices[indices.size() - 2], rowIdx, rewriter);
  rewriter.create<vector::StoreOp>(loc, vec, buf.memRef, indices);
}

void storeRows(Location loc, const MemBuffer &buf,
               const SmallVector<Value> &vecs, PatternRewriter &rewriter) {
  SmallVector<Value> indices = buf.indices;
  for (int64_t m = 0; m < vecs.size(); ++m)
    storeRow(loc, buf, m, vecs[m], rewriter);
}

SmallVector<Value> extractRows(Location loc, Value vec,
                               PatternRewriter &rewriter) {
  VectorType vecTy = cast<VectorType>(vec.getType());
  SmallVector<Value> res;
  for (int64_t m = 0; m < vecTy.getDimSize(0); ++m) {
    auto row =
        rewriter.create<vector::ExtractOp>(loc, vec, SmallVector<int64_t>({m}));
    res.push_back(row);
  }
  return res;
}

Value mergeRows(Location loc, VectorType resTy, const SmallVector<Value> &tiles,
                PatternRewriter &rewriter) {
  Value res =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(resTy));
  for (int64_t m = 0; m < tiles.size(); ++m)
    res = rewriter.create<vector::InsertOp>(loc, tiles[m], res,
                                            SmallVector<int64_t>({m}));
  return res;
}

Value broadcastElem(Location loc, VectorType tileTy, const MemBuffer &buf,
                    int64_t m, int64_t n, PatternRewriter &rewriter) {
  SmallVector<Value> indices = shiftIndices(loc, buf, m, n, rewriter);
  Value scalar = rewriter.create<memref::LoadOp>(loc, buf.memRef, indices);
  return rewriter.create<vector::BroadcastOp>(loc, tileTy, scalar);
}

SmallVector<Value> computePrefetchIndices(Location loc, const MemBuffer &buf,
                                          int64_t iters,
                                          PatternRewriter &rewriter) {
  SmallVector<Value> scaledStep;
  Value itersVal;
  for (auto step : buf.step) {
    if (iters == 1)
      scaledStep.push_back(rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(), step));
    else if (auto cstOp = dyn_cast<arith::ConstantOp>(step.getDefiningOp())) {
      int64_t oldVal = cast<IntegerAttr>(cstOp.getValue()).getInt();
      scaledStep.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, oldVal * iters));
    } else {
      if (!itersVal)
        itersVal =
            rewriter.create<arith::ConstantIntOp>(loc, iters, step.getType());
      scaledStep.push_back(rewriter.create<arith::IndexCastOp>(
          loc, rewriter.getIndexType(),
          rewriter.create<arith::MulIOp>(loc, step.getType(), step, itersVal)));
    }
  }

  SmallVector<Value> res;
  for (int64_t i = 0; i < scaledStep.size(); ++i)
    res.push_back(rewriter.create<arith::AddIOp>(
        loc, buf.indices[i].getType(), buf.indices[i], scaledStep[i]));
  return res;
}

void prefetch(Location loc, const MemBuffer &buf, int64_t m, int64_t n,
              ArrayRef<Value> prefetchIndices, int64_t hint,
              PatternRewriter &rewriter) {
  SmallVector<Value> indices =
      shiftIndices(loc, prefetchIndices, buf.transposed, m, n, rewriter);
  rewriter.create<memref::PrefetchOp>(loc, buf.memRef, indices, false, hint,
                                      true);
}

LogicalResult convertCandidate(FmaDotOpCandidate &candidate,
                               PatternRewriter &rewriter) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  VectorType lhsTy = cast<VectorType>(op.getA().getType());
  VectorType rhsTy = cast<VectorType>(op.getB().getType());
  VectorType accTy = cast<VectorType>(op.getC().getType());
  VectorType resTy = cast<VectorType>(op.getResult().getType());
  VectorType rhsVecTy =
      VectorType::get(candidate.accVecSize, candidate.rhsElemTy);
  VectorType accVecTy =
      VectorType::get(candidate.accVecSize, candidate.accElemTy);

  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  // Cast input data if required and prepare input buffer. It might be temporary
  // buffers with stored vectors or the original input memory.
  MemBuffer lhsBuf = candidate.lhsBuf;
  if (lhsBuf.empty()) {
    Value lhs = maybeCast(loc, op.getA(), candidate.lhsElemTy, rewriter);
    lhsBuf = storeToTmpBuffer(loc, lhs, allocaPoint, rewriter);
  }

  MemBuffer rhsBuf = candidate.rhsBuf;
  if (rhsBuf.empty()) {
    Value rhs = maybeCast(loc, op.getB(), candidate.rhsElemTy, rewriter);
    rhsBuf = storeToTmpBuffer(loc, rhs, allocaPoint, rewriter);
  }

  Value acc = maybeCast(loc, op.getC(), candidate.accElemTy, rewriter);
  Value accToStore = acc;
  scf::ForOp forOp;
  if (candidate.keepAccOnRegs) {
    forOp = cast<scf::ForOp>(op->getParentOp());
    accToStore = getInitAccValue(acc);
  }

  SmallVector<Value> accVecs;
  SmallVector<Value> accInitVecs;
  if (candidate.keepAccOnRegs) {
    // Initial tile values are loaded before the loop and then directly
    // used within the loop. Later, new iter values will be added to
    // add loop carried-dependencies for accumulator tiles and accInitTiles
    // will be used as initializers for them.
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(forOp);
    LDBG("Loading accumulator to tiles before the loop.");
    accInitVecs = extractRows(loc, accToStore, rewriter);
    accVecs = accInitVecs;
  } else {
    accVecs = extractRows(loc, acc, rewriter);
  }

  // Compute indices to be used by prefetch.
  int64_t lhsPrefetchIters =
      std::max(int64_t(128) / lhsTy.getNumElements(), int64_t(1));
  auto lhsPrefetchIndices =
      computePrefetchIndices(loc, candidate.lhsBuf, lhsPrefetchIters, rewriter);
  int64_t rhsPrefetchIters =
      std::max(int64_t(128) / rhsTy.getNumElements(), int64_t(1));
  auto rhsPrefetchIndices =
      computePrefetchIndices(loc, candidate.rhsBuf, rhsPrefetchIters, rewriter);
  Value nextRhsVec = loadRow(loc, rhsVecTy, rhsBuf, 0, rewriter);
  for (int64_t k = 0; k < lhsTy.getDimSize(1); ++k) {
    Value rhsVec = nextRhsVec;

    // Load next vector in advance to hide load latency.
    if (k != lhsTy.getDimSize(1) - 1)
      nextRhsVec = loadRow(loc, rhsVecTy, rhsBuf, k + 1, rewriter);

    // Prefetch RHS to LLC cache.
    if (!rhsPrefetchIndices.empty())
      prefetch(loc, candidate.rhsBuf, k, 0, rhsPrefetchIndices, 1, rewriter);

    Value nextLhsBroadcasted =
        broadcastElem(loc, accVecTy, lhsBuf, 0, k, rewriter);
    for (int64_t m = 0; m < candidate.accRows; ++m) {
      Value lhsBroadcasted = nextLhsBroadcasted;

      // Load next value in advance to hide load latency.
      if (m != candidate.accRows - 1)
        nextLhsBroadcasted =
            broadcastElem(loc, accVecTy, lhsBuf, m + 1, k, rewriter);

      // Prefetch LHS to L1 cache.
      if (!lhsPrefetchIndices.empty()) {
        if ((candidate.lhsBuf.transposed && (m % 8 == 0)) ||
            (!candidate.lhsBuf.transposed && (k % 8 == 0)))
          prefetch(loc, candidate.lhsBuf, m, k, lhsPrefetchIndices, 3,
                   rewriter);
      }

      accVecs[m] = rewriter.create<vector::FMAOp>(loc, rhsVec, lhsBroadcasted,
                                                  accVecs[m]);
    }
  }

  if (candidate.keepAccOnRegs) {
    // In this case we have the whole accumulator/result on tiles. Loop
    // carried dependencies are not in place yet and should be added.
    // After the loop, resulting tiles should either be stored to the
    // output buffer, or moved to a vector through a temporary buffer.

    // We don't need the original accumulator and contraction op anymore.
    // Directly yield orig accumulator value, so it would be later removed
    // as unused. The original contraction can be removed right away.
    int64_t origResIdx = op.getResult().getUses().begin()->getOperandNumber();
    rewriter.replaceOp(op, op.getC());

    // Now, replace the loop with a new one to add loop carried dependency for
    // accumulator tiles.
    LDBG("Rewrite loop to introduce loop carried dependencies for accumulator "
         "tiles.");
    SmallVector<Value> newInitOperands;
    SmallVector<Value> newYieldedValues;
    for (int64_t m = 0; m < candidate.accRows; ++m) {
      LDBG("Initial value\n  " << accInitVecs[m] << "\nis combined with\n  "
                               << accVecs[m]);
      newInitOperands.push_back(accInitVecs[m]);
      newYieldedValues.push_back(accVecs[m]);
    }
    auto newForOp = cast<scf::ForOp>(*forOp.replaceWithAdditionalYields(
        rewriter, newInitOperands, true,
        [&newYieldedValues](OpBuilder &b, Location loc,
                            ArrayRef<BlockArgument> newBBArgs) {
          return newYieldedValues;
        }));

    // The resulting tiles are now in the new loop results.
    auto resVecs = newForOp.getResults().take_back(newYieldedValues.size());
    for (int64_t m = 0; m < candidate.accRows; ++m)
      accVecs[m] = resVecs[m];

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newForOp);
    // Collect all results into a single vector.
    LDBG("Merging resulting rows to replace loop result.");
    VectorType resTy = accTy.cloneWith(std::nullopt, candidate.accElemTy);
    Value newVal = mergeRows(loc, resTy, accVecs, rewriter);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, accTy.getElementType(), rewriter);
    rewriter.replaceAllUsesWith(newForOp.getResult(origResIdx), newVal);
  } else {
    // The result is in the buffer. We should load it and replace the original
    // constraction result.
    LDBG("Merging resulting rows to replace orig op result.");
    VectorType resTy = accTy.cloneWith(std::nullopt, candidate.accElemTy);
    Value newVal = mergeRows(loc, resTy, accVecs, rewriter);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, accTy.getElementType(), rewriter);
    rewriter.replaceOp(op, newVal);
  }

  return success();
}

struct ConvertDotToFMA
    : public triton::cpu::impl::ConvertDotToFMABase<ConvertDotToFMA> {
  ConvertDotToFMA() = default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    SmallVector<FmaDotOpCandidate, 1> candidates;
    mod->walk([this, &candidates](cpu::DotOp op) {
      FmaDotOpCandidate candidate;
      if (isFmaCandidate(op, candidate)) {
        LLVM_DEBUG({
          LDBG("Found FMA candidate");
          LDBG("  Op: " << candidate.op);
          LDBG("  LhsElemTy: " << candidate.lhsElemTy);
          LDBG("  RhsElemTy: " << candidate.rhsElemTy);
          LDBG("  AccElemTy: " << candidate.accElemTy);
          LDBG("  AccVecSize: " << candidate.accVecSize);
          LDBG("  AccRows: " << candidate.accRows);
          LDBG("  KeepAccOnRegs: " << candidate.keepAccOnRegs);
          if (!candidate.lhsBuf.empty()) {
            LDBG("  LhsBuf: " << candidate.lhsBuf.memRef);
            LDBG("  Transposed: " << candidate.lhsBuf.transposed);
          }
          if (!candidate.rhsBuf.empty()) {
            LDBG("  RhsBuf: " << candidate.rhsBuf.memRef);
            LDBG("  Transposed: " << candidate.rhsBuf.transposed);
          }
        });
        candidates.push_back(candidate);
      }
      return WalkResult::advance();
    });

    for (auto &candidate : candidates) {
      LDBG("Starting conversion of candidate: " << candidate.op);
      PatternRewriter rewriter(context);
      rewriter.setInsertionPoint(candidate.op);
      if (succeeded(convertCandidate(candidate, rewriter))) {
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

std::unique_ptr<OperationPass<ModuleOp>> createConvertDotToFMA() {
  return std::make_unique<ConvertDotToFMA>();
}

} // namespace cpu
} // namespace triton
} // namespace mlir
