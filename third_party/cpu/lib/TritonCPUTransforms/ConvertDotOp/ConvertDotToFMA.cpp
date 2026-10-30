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
// inputs. A non-zero native vector width additionally requires the N
// dimension to be a whole number of vector registers.
bool checkInputShapes(VectorType lhsTy, VectorType resTy,
                      unsigned nativeVectorBitWidth) {
  if (lhsTy.getRank() != 2)
    return false;

  int64_t cols = resTy.getDimSize(1);
  if (nativeVectorBitWidth == 0)
    return cols >= 8;

  unsigned elemBits = resTy.getElementType().getIntOrFloatBitWidth();
  if (elemBits == 0 || nativeVectorBitWidth % elemBits != 0)
    return false;
  int64_t nativeElems = nativeVectorBitWidth / elemBits;
  return nativeElems > 0 && cols >= nativeElems && cols % nativeElems == 0;
}

// Check if specified ContractionOp can be lowered to FMA operations.
// If conversion is possible, then true is returned and candidate
// structure is filled with detailed transformation info.
bool isFmaCandidate(cpu::DotOp op, FmaDotOpCandidate &candidate,
                    unsigned nativeVectorBitWidth) {
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
  if (!checkInputShapes(lhsTy, resTy, nativeVectorBitWidth))
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
  return vector::LoadOp::create(rewriter, loc, resTy, buf.memRef, indices);
}

void storeRow(Location loc, const MemBuffer &buf, int64_t rowIdx, Value vec,
              PatternRewriter &rewriter) {
  SmallVector<Value> indices = buf.indices;
  indices[indices.size() - 2] =
      shiftIndex(loc, buf.indices[indices.size() - 2], rowIdx, rewriter);
  vector::StoreOp::create(rewriter, loc, vec, buf.memRef, indices);
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
    auto row = vector::ExtractOp::create(rewriter, loc, vec,
                                         SmallVector<int64_t>({m}));
    res.push_back(row);
  }
  return res;
}

Value mergeRows(Location loc, VectorType resTy, const SmallVector<Value> &tiles,
                PatternRewriter &rewriter) {
  Value res =
      arith::ConstantOp::create(rewriter, loc, rewriter.getZeroAttr(resTy));
  for (int64_t m = 0; m < tiles.size(); ++m)
    res = vector::InsertOp::create(rewriter, loc, tiles[m], res,
                                   SmallVector<int64_t>({m}));
  return res;
}

Value broadcastElem(Location loc, VectorType tileTy, const MemBuffer &buf,
                    int64_t m, int64_t n, PatternRewriter &rewriter) {
  SmallVector<Value> indices = shiftIndices(loc, buf, m, n, rewriter);
  Value scalar = memref::LoadOp::create(rewriter, loc, buf.memRef, indices);
  return vector::BroadcastOp::create(rewriter, loc, tileTy, scalar);
}

SmallVector<Value> computePrefetchIndices(Location loc, const MemBuffer &buf,
                                          int64_t iters,
                                          PatternRewriter &rewriter) {
  SmallVector<Value> scaledStep;
  Value itersVal = arith::ConstantIndexOp::create(rewriter, loc, iters);
  for (auto step : buf.step) {
    if (!step)
      // Index is loop-invariant.
      step = arith::ConstantIndexOp::create(rewriter, loc, 0);
    else if (!step.getType().isIndex())
      step = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                        step);
    scaledStep.push_back(
        rewriter.createOrFold<arith::MulIOp>(loc, step, itersVal));
  }

  SmallVector<Value> res;
  for (auto [index, step] : llvm::zip(buf.indices, scaledStep))
    res.push_back(rewriter.createOrFold<arith::AddIOp>(loc, index, step));
  return res;
}

void prefetch(Location loc, const MemBuffer &buf, int64_t m, int64_t n,
              ArrayRef<Value> prefetchIndices, int64_t hint,
              PatternRewriter &rewriter) {
  SmallVector<Value> indices =
      shiftIndices(loc, prefetchIndices, buf.transposed, m, n, rewriter);
  memref::PrefetchOp::create(rewriter, loc, buf.memRef, indices, false, hint,
                             true);
}

Value asIndex(OpBuilder &b, Location loc, Value value) {
  if (value.getType().isIndex())
    return value;
  return arith::IndexCastOp::create(b, loc, b.getIndexType(), value);
}

Value addOffset(OpBuilder &b, Location loc, Value index, Value delta) {
  return arith::AddIOp::create(b, loc, b.getIndexType(), asIndex(b, loc, index),
                               asIndex(b, loc, delta));
}

Value addConstOffset(OpBuilder &b, Location loc, Value index, int64_t delta) {
  if (auto cst =
          dyn_cast_or_null<arith::ConstantIndexOp>(index.getDefiningOp()))
    return arith::ConstantIndexOp::create(b, loc, cst.value() + delta);
  if (auto cst = dyn_cast_or_null<arith::ConstantOp>(index.getDefiningOp())) {
    if (auto intAttr = dyn_cast<IntegerAttr>(cst.getValue()))
      return arith::ConstantIndexOp::create(b, loc, intAttr.getInt() + delta);
  }
  Value idx = asIndex(b, loc, index);
  if (delta == 0)
    return idx;
  Value off = arith::ConstantIndexOp::create(b, loc, delta);
  return arith::AddIOp::create(b, loc, b.getIndexType(), idx, off);
}

// LHS scalar at logical (row, col). `col` is the reduction index and may be
// the induction variable of the K loop. Transpose swaps which physical
// dimension receives the row, matching shiftIndices().
Value loadLhsScalar(OpBuilder &b, Location loc, const MemBuffer &buf,
                    int64_t row, Value col) {
  SmallVector<Value> indices(buf.indices.begin(), buf.indices.end() - 2);
  Value rowBase = *(buf.indices.end() - 2);
  Value colBase = *(buf.indices.end() - 1);
  if (buf.transposed) {
    indices.push_back(addOffset(b, loc, rowBase, col));
    indices.push_back(addConstOffset(b, loc, colBase, row));
  } else {
    indices.push_back(addConstOffset(b, loc, rowBase, row));
    indices.push_back(addOffset(b, loc, colBase, col));
  }
  return memref::LoadOp::create(b, loc, buf.memRef, indices);
}

// One native-width slice of RHS row `row`, starting at column `col`.
// loadRow() shifts the reduction dimension and leaves the contiguous
// dimension at its base; this applies the same convention.
Value loadRhsTile(OpBuilder &b, Location loc, VectorType tileTy,
                  const MemBuffer &buf, Value row, int64_t col) {
  SmallVector<Value> indices(buf.indices);
  indices[indices.size() - 2] =
      addOffset(b, loc, indices[indices.size() - 2], row);
  indices[indices.size() - 1] =
      addConstOffset(b, loc, indices[indices.size() - 1], col);
  return vector::LoadOp::create(b, loc, tileTy, buf.memRef, indices);
}

// Lower a dot to 128-bit (or other native-width) FMAs. The K loop stays in
// scf.for so a large tile does not unroll into hundreds of thousands of ops.
// Each accumulator panel is one vector register wide and a few rows tall,
// which fits in the 32 z/Architecture vector registers.
LogicalResult lowerDotToNativeVectorFMA(FmaDotOpCandidate &candidate,
                                        const MemBuffer &lhsBuf,
                                        const MemBuffer &rhsBuf,
                                        PatternRewriter &rewriter,
                                        unsigned nativeVectorBitWidth) {
  cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  if (lhsBuf.indices.size() < 2 || rhsBuf.indices.size() < 2)
    return failure();

  unsigned elemBits = candidate.accElemTy.getIntOrFloatBitWidth();
  if (elemBits == 0 || nativeVectorBitWidth % elemBits != 0)
    return failure();
  int64_t nativeElems = nativeVectorBitWidth / elemBits;
  int64_t rows = candidate.accRows;
  int64_t cols = candidate.accVecSize;
  int64_t kDim = cast<VectorType>(op.getA().getType()).getDimSize(1);
  if (nativeElems <= 0 || cols % nativeElems != 0)
    return failure();

  // z15 has 32 vector registers. Eight accumulator rows plus the RHS tile
  // and the broadcast LHS stay well inside that file.
  constexpr int64_t kPanelRows = 8;
  VectorType nativeTy = VectorType::get({nativeElems}, candidate.accElemTy);
  VectorType accTy = cast<VectorType>(op.getC().getType());
  Type origElemTy = accTy.getElementType();
  Value accMatrix = maybeCast(loc, op.getC(), candidate.accElemTy, rewriter);

  Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
  Value cK = arith::ConstantIndexOp::create(rewriter, loc, kDim);

  for (int64_t m0 = 0; m0 < rows; m0 += kPanelRows) {
    int64_t mr = std::min(kPanelRows, rows - m0);
    for (int64_t n0 = 0; n0 < cols; n0 += nativeElems) {
      SmallVector<Value> savedRows;
      SmallVector<Value> initTiles;
      savedRows.reserve(mr);
      initTiles.reserve(mr);
      SmallVector<int64_t, 1> offset{n0};
      SmallVector<int64_t, 1> sizes{nativeElems};
      SmallVector<int64_t, 1> strides{1};
      for (int64_t m = 0; m < mr; ++m) {
        Value row = vector::ExtractOp::create(rewriter, loc, accMatrix,
                                              SmallVector<int64_t>({m0 + m}));
        savedRows.push_back(row);
        initTiles.push_back(vector::ExtractStridedSliceOp::create(
            rewriter, loc, row, offset, sizes, strides));
      }

      int64_t col = n0;
      auto kLoop = scf::ForOp::create(
          rewriter, loc, c0, cK, c1, ValueRange(initTiles),
          [&](OpBuilder &b, Location bodyLoc, Value kIv, ValueRange iterArgs) {
            Value rhs = loadRhsTile(b, bodyLoc, nativeTy, rhsBuf, kIv, col);
            SmallVector<Value> next;
            next.reserve(mr);
            for (int64_t m = 0; m < mr; ++m) {
              Value lhsScalar = loadLhsScalar(b, bodyLoc, lhsBuf, m0 + m, kIv);
              Value lhs =
                  vector::BroadcastOp::create(b, bodyLoc, nativeTy, lhsScalar);
              next.push_back(
                  vector::FMAOp::create(b, bodyLoc, rhs, lhs, iterArgs[m]));
            }
            scf::YieldOp::create(b, bodyLoc, ValueRange(next));
          });

      for (int64_t m = 0; m < mr; ++m) {
        Value updatedRow = vector::InsertStridedSliceOp::create(
            rewriter, loc, kLoop.getResult(m), savedRows[m], offset, strides);
        accMatrix =
            vector::InsertOp::create(rewriter, loc, updatedRow, accMatrix,
                                     SmallVector<int64_t>({m0 + m}));
      }
    }
  }

  rewriter.replaceOp(op, maybeCast(loc, accMatrix, origElemTy, rewriter));
  return success();
}

LogicalResult convertCandidate(FmaDotOpCandidate &candidate,
                               PatternRewriter &rewriter,
                               unsigned nativeVectorBitWidth) {
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

  if (nativeVectorBitWidth != 0)
    return lowerDotToNativeVectorFMA(candidate, lhsBuf, rhsBuf, rewriter,
                                     nativeVectorBitWidth);

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

      accVecs[m] = vector::FMAOp::create(rewriter, loc, rhsVec, lhsBroadcasted,
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
  explicit ConvertDotToFMA(unsigned nativeVectorBitWidth) {
    this->nativeVectorBitWidth = nativeVectorBitWidth;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    SmallVector<FmaDotOpCandidate, 1> candidates;
    mod->walk([this, &candidates](cpu::DotOp op) {
      FmaDotOpCandidate candidate;
      if (isFmaCandidate(op, candidate, nativeVectorBitWidth)) {
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
      if (succeeded(
              convertCandidate(candidate, rewriter, nativeVectorBitWidth))) {
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
  return createConvertDotToFMA(0);
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotToFMA(unsigned nativeVectorBitWidth) {
  return std::make_unique<ConvertDotToFMA>(nativeVectorBitWidth);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
