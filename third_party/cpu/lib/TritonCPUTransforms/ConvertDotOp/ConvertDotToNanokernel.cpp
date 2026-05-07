#include "ConvertDotCommon.h"

#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86/Transforms.h"
#include "mlir/Dialect/X86/X86Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopInvariantCodeMotionUtils.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_CONVERTDOTTONANOKERNEL
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

enum ISAExt : unsigned {
  AVX_NE_CONVERT = 1 << 0,
  AVX512_BF16 = 1 << 1,
  AVX10_2 = 1 << 2,
  AMX_BF16 = 1 << 3,
  AMX_INT8 = 1 << 4,
};

std::string stringifyISAExtMask(unsigned mask) {
  std::string res;
  if (mask & AVX_NE_CONVERT)
    res += "AVX_NE_CONVERT ";
  if (mask & AVX512_BF16)
    res += "AVX512_BF16 ";
  if (mask & AVX10_2)
    res += "AVX10.2 ";
  if (mask & AMX_BF16)
    res += "AMX_BF16 ";
  if (mask & AMX_INT8)
    res += "AMX_INT8 ";
  return res.empty() ? "None" : res.substr(0, res.size() - 1);
}

unsigned parseCPUFeatures(const std::string &cpuFeatures) {
  unsigned mask = 0;
  if (cpuFeatures.find("avxneconvert") != std::string::npos)
    mask |= AVX_NE_CONVERT;
  if (cpuFeatures.find("avx512bf16") != std::string::npos)
    mask |= AVX512_BF16;
  if (cpuFeatures.find("avx10.2") != std::string::npos)
    mask |= AVX10_2;
  if (cpuFeatures.find("amx-bf16") != std::string::npos)
    mask |= AMX_BF16;
  if (cpuFeatures.find("amx-int8") != std::string::npos)
    mask |= AMX_INT8;
  return mask;
}

// This structure is used to hold candidates for application of a nanokernel
// pattern.
struct DotOpCandidate {
  ISAExt target;

  // Operation to convert (and intermediate contraction op).
  triton::cpu::DotOp dot;
  vector::ContractionOp contract;

  // Block sizes.
  int64_t blockM;
  int64_t blockN;
  int64_t blockK;

  // Accumulation loop (original and spliced).
  scf::ForOp accLoop;
  scf::ForOp splicedAccLoop;

  // Vector transfer ops for inputs/output.
  vector::TransferReadOp lhsRead;
  vector::TransferReadOp rhsRead;
  vector::TransferReadOp accRead;
  vector::TransferWriteOp accWrite;

  // Temporary buffer for accumulator.
  memref::AllocaOp accBuffer;
};

unsigned checkElemTypes(Type lhsElemTy, Type rhsElemTy, Type accElemTy,
                        Type resElemTy, unsigned mask) {
  if (accElemTy != resElemTy) {
    // Constrain the two for simplicity.
    LDBG("  Drop candidate. Expect same accumulation and result type.");
    return 0;
  }

  if (lhsElemTy != rhsElemTy) {
    LDBG("  Drop candidate. Mixed type input is not supported");
    return 0;
  }

  if (lhsElemTy.isBF16() && accElemTy.isF32())
    return mask & (AVX_NE_CONVERT | AVX512_BF16 | AMX_BF16);

  if (lhsElemTy.isInteger(8) && accElemTy.isInteger(32))
    return mask & (AVX10_2 | AMX_INT8);

  LDBG("  Drop candidate. Unsupported type combination");
  return 0;
}

unsigned checkInputShapes(VectorType lhsTy, VectorType resTy,
                          DotOpCandidate &candidate, unsigned mask) {
  candidate.blockM = resTy.getDimSize(0);
  candidate.blockN = resTy.getDimSize(1);
  candidate.blockK = lhsTy.getDimSize(1);

  auto shapeUnrollsTo = [&candidate](int64_t m, int64_t n, int64_t k,
                                     int64_t numRegs) {
    return candidate.blockM % m == 0 && candidate.blockN % n == 0 &&
           candidate.blockK == k &&
           (candidate.blockM / m) * (candidate.blockN / n) <= numRegs;
  };
  auto shapeEquals = [&candidate](int64_t m, int64_t n, int64_t k) {
    return candidate.blockM == m && candidate.blockN == n &&
           candidate.blockK == k;
  };

  if (shapeUnrollsTo(1, 8, 1, 12))
    return mask & AVX_NE_CONVERT;

  if (shapeUnrollsTo(1, 16, 2, 24))
    return mask & AVX512_BF16;

  if (shapeUnrollsTo(1, 16, 4, 24))
    return mask & AVX10_2;

  // AMX lowering currently matches only the 2x2 register tiling.
  if (shapeEquals(32, 32, 32))
    return mask & AMX_BF16;

  if (shapeEquals(32, 32, 64))
    return mask & AMX_INT8;

  LDBG("  Drop candidate. Unsupported shapes");
  return 0;
}

// Check if specified DotOp can be lowered to a vector contraction on which the
// nanokernel patterns for the given CPU features apply.
// If conversion is possible, then true is returned and candidate structure is
// filled with detailed transformation info.
bool isNanokernelCandidate(triton::cpu::DotOp op, DotOpCandidate &candidate,
                           unsigned mask) {
  VectorType lhsTy = cast<VectorType>(op.getA().getType());
  VectorType rhsTy = cast<VectorType>(op.getB().getType());
  VectorType accTy = cast<VectorType>(op.getC().getType());
  VectorType resTy = cast<VectorType>(op.getType());

  LDBG("Considering candidate op: " << op);

  if (accTy.getRank() != 2) {
    LDBG("  Drop candidate. Only 2D case is supported.");
    return false;
  }

  mask = checkElemTypes(lhsTy.getElementType(), rhsTy.getElementType(),
                        accTy.getElementType(), resTy.getElementType(), mask);
  mask = checkInputShapes(lhsTy, resTy, candidate, mask);
  if (llvm::isPowerOf2_32(mask) != 1) {
    LDBG("  Drop candidate. No uniquely matching ISA extension for the op's "
         "types and shapes.");
    return false;
  }

  if (!isLoopCarriedAcc(op.getC())) {
    LDBG("  Drop candidate. Only loop-carried accumulator case is supported.");
    return false;
  }

  auto accLoop = cast<scf::ForOp>(op->getParentOp());

  auto lhsRead = op.getA().getDefiningOp<vector::TransferReadOp>();
  auto rhsRead = op.getB().getDefiningOp<vector::TransferReadOp>();
  if (!lhsRead || !rhsRead) {
    LDBG("  Drop candidate. Expected vector.transfer_read ops feeding the dot, "
         "but got: "
         << op.getA() << " and " << op.getB());
    return false;
  }

  auto accRead = accLoop.getTiedLoopInit(cast<BlockArgument>(op.getC()))
                     ->get()
                     .getDefiningOp<vector::TransferReadOp>();

  auto accWrite = accLoop.getResults().front().hasOneUse()
                      ? dyn_cast<vector::TransferWriteOp>(
                            *accLoop.getResults().front().getUsers().begin())
                      : nullptr;

  if (accRead && accWrite &&
      (accRead.getBase() != accWrite.getBase() ||
       !llvm::equal(accRead.getIndices(), accWrite.getIndices()))) {
    LDBG("  Cannot use existing vector.transfer_read/write due to mismatch in "
         "memref or indices.");
    accRead = nullptr;
    accWrite = nullptr;
  }

  auto checkTransferOp = [&](auto transferOp) {
    if (!transferOp)
      return true; // No op to check, so no reason to drop the candidate.

    bool hasNonIdentityPermutation =
        !transferOp.getPermutationMap().isMinorIdentity();
    bool hasMask = transferOp.getMask() != nullptr;

    ArrayAttr inBounds = transferOp.getInBounds();
    bool hasBoundsCheck =
        std::any_of(inBounds.begin(), inBounds.end(), [](Attribute attr) {
          return !cast<mlir::BoolAttr>(attr).getValue();
        });

    if (hasNonIdentityPermutation || hasMask || hasBoundsCheck) {
      LDBG("  Drop candidate. Transfer op has a permutation map, mask or needs "
           "bounds checking: "
           << transferOp);
      return false;
    }
    return true;
  };

  if (!checkTransferOp(lhsRead) || !checkTransferOp(rhsRead) ||
      !checkTransferOp(accRead) || !checkTransferOp(accWrite))
    return false;

  candidate.target = static_cast<ISAExt>(mask);
  candidate.dot = op;
  candidate.accLoop = accLoop;
  candidate.lhsRead = lhsRead;
  candidate.rhsRead = rhsRead;
  candidate.accRead = accRead;
  candidate.accWrite = accWrite;

  return true;
}

void makeAccBuffer(DotOpCandidate &candidate, PatternRewriter &rewriter) {
  if (candidate.accRead && candidate.accWrite)
    return; // Nothing to do.

  Location loc = candidate.accLoop.getLoc();
  auto accTy = cast<VectorType>(candidate.dot.getC().getType());
  auto allocaTy = MemRefType::get(accTy.getShape(), accTy.getElementType());

  // Before the loop...
  rewriter.setInsertionPoint(candidate.accLoop);
  candidate.accBuffer = memref::AllocaOp::create(rewriter, loc, allocaTy);

  SmallVector<Value> zeroIndices(
      accTy.getRank(), arith::ConstantIndexOp::create(rewriter, loc, 0));
  Value padding = arith::ConstantOp::create(
      rewriter, loc, rewriter.getZeroAttr(accTy.getElementType()));

  OpOperand *initOperand = candidate.accLoop.getTiedLoopInit(
      cast<BlockArgument>(candidate.dot.getC()));
  Value init = initOperand->get();

  vector::TransferWriteOp::create(rewriter, loc, init, candidate.accBuffer,
                                  zeroIndices);
  candidate.accRead = vector::TransferReadOp::create(
      rewriter, loc, accTy, candidate.accBuffer, zeroIndices, padding);
  initOperand->set(candidate.accRead);

  // After the loop...
  rewriter.setInsertionPointAfter(candidate.accLoop);

  Value result = candidate.accLoop.getTiedLoopResult(initOperand);

  candidate.accWrite = vector::TransferWriteOp::create(
      rewriter, loc, result, candidate.accBuffer, zeroIndices);
  Value remat = vector::TransferReadOp::create(
      rewriter, loc, accTy, candidate.accBuffer, zeroIndices, padding);

  rewriter.replaceAllUsesExcept(result, remat, candidate.accWrite);
}

template <typename TransferOp>
void moveIndicesToSubview(TransferOp op, PatternRewriter &rewriter) {
  if (llvm::all_of(op.getIndices(), isZeroInteger))
    return;

  auto *ctx = rewriter.getContext();
  auto loc = op.getLoc();

  rewriter.setInsertionPoint(op);

  auto memrefTy = cast<MemRefType>(op.getBase().getType());
  auto vecTy = op.getVectorType();
  SmallVector<int64_t> shape(memrefTy.getRank(), 1);
  SmallVector<int64_t> strides(memrefTy.getRank(), 1);

  int64_t startDim = memrefTy.getRank() - vecTy.getRank();
  assert(startDim >= 0);

  for (int64_t i = 0; i < vecTy.getRank(); ++i)
    shape[startDim + i] = vecTy.getShape()[i];

  Value memrefView = memref::SubViewOp::create(
      rewriter, loc, op.getBase(), getAsOpFoldResult(op.getIndices()),
      getAsIndexOpFoldResult(ctx, shape), getAsIndexOpFoldResult(ctx, strides));

  Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> zeros(memrefTy.getRank(), zero);

  op.getBaseMutable().assign(memrefView);
  op.getIndicesMutable().assign(zeros);
}

void moveIndicesToSubview(DotOpCandidate &candidate,
                          PatternRewriter &rewriter) {
  moveIndicesToSubview(candidate.lhsRead, rewriter);
  moveIndicesToSubview(candidate.rhsRead, rewriter);
  moveIndicesToSubview(candidate.accRead, rewriter);
  moveIndicesToSubview(candidate.accWrite, rewriter);
}

void convertToContract(DotOpCandidate &candidate, PatternRewriter &rewriter) {
  auto op = candidate.dot;
  auto *ctx = op.getContext();
  auto loc = op.getLoc();

  rewriter.setInsertionPoint(op);

  Value a = op.getA();
  Value b = op.getB();
  Value c = op.getC();
  VectorType aType = cast<VectorType>(a.getType());
  VectorType bType = cast<VectorType>(b.getType());
  VectorType cType = cast<VectorType>(c.getType());

  unsigned rank = cType.getRank();
  assert(rank == 2 && "Only 2D case is supported");
  auto aMap = AffineMap::getMultiDimMapWithTargets(3, {0, 2}, ctx);
  auto bMap = AffineMap::getMultiDimMapWithTargets(3, {2, 1}, ctx);
  auto cMap = AffineMap::getMultiDimMapWithTargets(3, {0, 1}, ctx);
  auto iteratorTypes = rewriter.getArrayAttr(
      {vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
       vector::IteratorTypeAttr::get(ctx, vector::IteratorType::parallel),
       vector::IteratorTypeAttr::get(ctx, vector::IteratorType::reduction)});
  candidate.contract = rewriter.replaceOpWithNewOp<vector::ContractionOp>(
      op, a, b, c, rewriter.getAffineMapArrayAttr({aMap, bMap, cMap}),
      iteratorTypes);
  candidate.dot = {};
}

void performRegisterTiling(DotOpCandidate &candidate,
                           PatternRewriter &rewriter) {

  int vnniFactor =
      32 / candidate.contract.getLhs().getType().getElementTypeBitWidth();

  static constexpr auto *unrollShapeAttrName = "unroll_shape";

  if (candidate.target & AVX_NE_CONVERT) {
    candidate.contract->setAttr(unrollShapeAttrName,
                                rewriter.getI64ArrayAttr({1, 8, 1}));
    candidate.lhsRead->setAttr(unrollShapeAttrName,
                               rewriter.getI64ArrayAttr({1, 1}));
    candidate.rhsRead->setAttr(unrollShapeAttrName,
                               rewriter.getI64ArrayAttr({1, 8}));
    candidate.accRead->setAttr(unrollShapeAttrName,
                               rewriter.getI64ArrayAttr({1, 8}));
    candidate.accWrite->setAttr(unrollShapeAttrName,
                                rewriter.getI64ArrayAttr({1, 8}));
  } else if (candidate.target & (AVX512_BF16 | AVX10_2)) {
    candidate.contract->setAttr(unrollShapeAttrName,
                                rewriter.getI64ArrayAttr({1, 16, vnniFactor}));
    candidate.lhsRead->setAttr(unrollShapeAttrName,
                               rewriter.getI64ArrayAttr({1, vnniFactor}));
    candidate.rhsRead->setAttr(unrollShapeAttrName,
                               rewriter.getI64ArrayAttr({vnniFactor, 16}));
    candidate.accRead->setAttr(unrollShapeAttrName,
                               rewriter.getI64ArrayAttr({1, 16}));
    candidate.accWrite->setAttr(unrollShapeAttrName,
                                rewriter.getI64ArrayAttr({1, 16}));
  } else if (candidate.target & (AMX_BF16 | AMX_INT8)) {
    candidate.contract->setAttr(
        unrollShapeAttrName,
        rewriter.getI64ArrayAttr({16, 16, 16 * vnniFactor}));
    candidate.lhsRead->setAttr(unrollShapeAttrName,
                               rewriter.getI64ArrayAttr({16, 16 * vnniFactor}));
    candidate.rhsRead->setAttr(unrollShapeAttrName,
                               rewriter.getI64ArrayAttr({16 * vnniFactor, 16}));
    candidate.accRead->setAttr(unrollShapeAttrName,
                               rewriter.getI64ArrayAttr({16, 16}));
    candidate.accWrite->setAttr(unrollShapeAttrName,
                                rewriter.getI64ArrayAttr({16, 16}));
  } else {
    llvm_unreachable("Unsupported ISA extension for register tiling");
  }

  vector::UnrollVectorOptions unrollOptions;
  unrollOptions.setFilterConstraint(
      [&](Operation *op) { return success(op->hasAttr(unrollShapeAttrName)); });
  unrollOptions.setNativeShapeFn(
      [&](Operation *op) -> std::optional<SmallVector<int64_t>> {
        assert(op->hasAttr(unrollShapeAttrName));
        auto vals = op->getAttrOfType<ArrayAttr>(unrollShapeAttrName)
                        .getAsValueRange<IntegerAttr>();
        SmallVector<int64_t> shape = llvm::to_vector(llvm::map_range(
            vals, [](const APInt &v) { return v.getSExtValue(); }));

        return shape;
      });

  RewritePatternSet patterns(candidate.contract.getContext());
  vector::populateVectorUnrollPatterns(patterns, unrollOptions);
  auto res = applyPatternsGreedily(
      candidate.contract->getParentOfType<triton::FuncOp>(),
      std::move(patterns));
  (void)res;
  assert(succeeded(res) && "Register tiling patterns failed to apply");
}

void spliceAccumulatorAndConvertToIndex(DotOpCandidate &candidate,
                                        PatternRewriter &rewriter) {
  scf::ForOp oldFor = candidate.accLoop;
  BlockArgument iterArg = cast<BlockArgument>(candidate.contract.getAcc());

  // Collect the extract/insert slice ops on the accumulator from users of iter
  // arg resp. yield operand. We checked earlier that the loop is an accumulator
  // loop for the contraction, so we know that these are present and have the
  // expected form and thus `assert` (or `cast`) liberally.
  llvm::SmallDenseMap<ArrayAttr, vector::ExtractStridedSliceOp> extractOps;
  llvm::SmallDenseMap<ArrayAttr, vector::InsertStridedSliceOp> insertOps;

  for (auto user : iterArg.getUsers()) {
    auto extractSliceOp = cast<vector::ExtractStridedSliceOp>(user);
    extractOps[extractSliceOp.getOffsets()] = extractSliceOp;
  }

  auto insert = oldFor.getTiedLoopYieldedValue(iterArg)
                    ->get()
                    .getDefiningOp<vector::InsertStridedSliceOp>();
  assert(insert);
  for (;;) {
    insertOps[insert.getOffsets()] = insert;
    Operation *dest = insert.getDest().getDefiningOp();
    if (isa_and_present<arith::ConstantOp, ub::PoisonOp>(dest))
      break; // Chain ends if we reach a constant.
    insert = cast<vector::InsertStridedSliceOp>(dest);
  }

  // Sort the offsets to ensure deterministic order of the new iter args.
  SmallVector<ArrayAttr> offsets{extractOps.keys()};
  llvm::stable_sort(offsets, [](ArrayAttr a, ArrayAttr b) {
    for (auto [av, bv] : llvm::zip_equal(a.getAsValueRange<IntegerAttr>(),
                                         b.getAsValueRange<IntegerAttr>())) {
      if (av.getZExtValue() != bv.getZExtValue())
        return av.getZExtValue() < bv.getZExtValue();
    }
    return false;
  });

  assert(llvm::all_of(
      offsets, [&](ArrayAttr offset) { return insertOps.contains(offset); }));

  // Drop one and add |offsets| iter args.
  unsigned numNewArgs = oldFor.getNumRegionIterArgs() -
                        oldFor.getNumInductionVars() - 1 + offsets.size();

  // Determine index of the iter arg to drop.
  unsigned dropIdx = iterArg.getArgNumber() - oldFor.getNumInductionVars();

  // Get init values for the new loop: Use all old inits except the one
  // corresponding to the dropped iter arg, and hoist the extract_slices op
  // before the loop.
  rewriter.setInsertionPoint(oldFor);
  SmallVector<Value> newInits;
  newInits.reserve(numNewArgs);
  for (auto [i, oldInitArg] : llvm::enumerate(oldFor.getInitArgs()))
    if (i != dropIdx)
      newInits.push_back(oldInitArg);

  Value oldInit = oldFor.getTiedLoopInit(iterArg)->get();
  for (ArrayAttr offset : offsets) {
    auto oldExtractOp = extractOps[offset];
    auto newExtractOp = rewriter.createOrFold<vector::ExtractStridedSliceOp>(
        oldExtractOp.getLoc(), oldExtractOp.getType(), oldInit,
        oldExtractOp.getOffsets(), oldExtractOp.getSizes(),
        oldExtractOp.getStrides());
    newInits.push_back(newExtractOp);
  }

  auto indexCast = [&](Value val) -> Value {
    if (val.getType().isIndex())
      return val;
    return arith::IndexCastOp::create(rewriter, oldFor.getLoc(),
                                      rewriter.getIndexType(), val);
  };

  // Create the new loop with the new init values.
  auto newFor = scf::ForOp::create(
      rewriter, oldFor.getLoc(), indexCast(oldFor.getLowerBound()),
      indexCast(oldFor.getUpperBound()), indexCast(oldFor.getStep()), newInits);

  Block *oldBody = oldFor.getBody();
  Block *newBody = newFor.getBody();

  // Clone into the new loop's body.
  rewriter.setInsertionPointToStart(newBody);

  // Set up IR mapping.
  //  1) Adapt to the potential type change of the induction variable.
  //  2) Users of the extract_slice ops shall use the new spliced iter args
  //  instead.
  IRMapping map;
  if (oldFor.getInductionVar().getType().isIndex())
    map.map(oldFor.getInductionVar(), newFor.getInductionVar());
  else {
    // Make sure users of an index_cast op use the new IV directly. Otherwise we
    // might end up with an index->i32->index cast chain that doesn't easily
    // fold away.
    for (auto *user : oldFor.getInductionVar().getUsers())
      if (auto cast = dyn_cast<arith::IndexCastOp>(user))
        map.map(cast, newFor.getInductionVar());

    // Fall-back mapping.
    map.map(oldFor.getInductionVar(),
            arith::IndexCastOp::create(rewriter, oldFor.getLoc(),
                                       oldFor.getInductionVar().getType(),
                                       newFor.getInductionVar()));
  }

  unsigned newArgIdx = 0;
  for (auto [i, oldIterArg] : llvm::enumerate(oldFor.getRegionIterArgs()))
    if (i != dropIdx)
      map.map(oldIterArg, newFor.getRegionIterArg(newArgIdx++));

  for (auto [i, offset] : llvm::enumerate(offsets))
    map.map(extractOps[offset], newFor.getRegionIterArg(newArgIdx + i));

  for (Operation &op : oldBody->without_terminator())
    // Skip ops that already have a mapping, i.e. the extract_slice ops.
    if (op.getNumResults() == 0 || !map.contains(op.getResult(0)))
      rewriter.clone(op, map);

  // Construct the new yield op, following the same logic as for the init
  // values.
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(numNewArgs);
  for (auto [i, oldOperand] : llvm::enumerate(oldYield.getOperands()))
    if (i != dropIdx)
      newYieldOperands.push_back(map.lookup(oldOperand));
  for (auto offset : offsets)
    newYieldOperands.push_back(map.lookup(insertOps[offset].getValueToStore()));
  scf::YieldOp::create(rewriter, oldYield.getLoc(), newYieldOperands);

  // Finally replace the results. Reassemble the vector again after the loop
  // from the spliced results.
  rewriter.setInsertionPointAfter(newFor);
  unsigned newResPos = 0;
  for (auto [i, oldResult] : llvm::enumerate(oldFor.getResults()))
    if (i != dropIdx)
      oldResult.replaceAllUsesWith(newFor.getResult(newResPos++));

  Value newVec = oldInit;
  for (auto [i, offset] : llvm::enumerate(offsets))
    newVec = vector::InsertStridedSliceOp::create(
        rewriter, oldFor.getLoc(), newFor.getResult(newResPos + i), newVec,
        offset, insertOps[offset].getStrides());
  oldFor->getResult(dropIdx).replaceAllUsesWith(newVec);

  rewriter.eraseOp(oldFor);

  candidate.splicedAccLoop = newFor;
  candidate.accLoop = {};
}

LogicalResult applyNanokernelPatterns(DotOpCandidate &candidate,
                                      PatternRewriter &rewriter) {
  RewritePatternSet patterns(candidate.contract.getContext());
  if (candidate.target & AVX_NE_CONVERT)
    x86::populateVectorContractBF16ToFMAPatterns(patterns);
  else if (candidate.target & (AVX512_BF16 | AVX10_2))
    x86::populateVectorContractToPackedTypeDotProductPatterns(patterns);
  else if (candidate.target & (AMX_BF16 | AMX_INT8))
    x86::populateVectorContractToAMXDotProductPatterns(patterns);
  else
    llvm_unreachable(
        "Unsupported target ISA extension for nanokernel patterns");

  return applyPatternsGreedily(
      candidate.contract->getParentOfType<triton::FuncOp>(),
      std::move(patterns));
}

LogicalResult convertCandidate(DotOpCandidate &candidate,
                               PatternRewriter &rewriter) {
  // Introduce temporary buffer if needed.
  makeAccBuffer(candidate, rewriter);

  // Insert memref.subview ops to ensure all transfer ops have zero indices.
  moveIndicesToSubview(candidate, rewriter);

  // 1:1 replacement of triton_cpu.dot -> vector.contract.
  convertToContract(candidate, rewriter);

  // Apply target-specific register tiling via upstream vector-dialect unroll
  // patterns.
  performRegisterTiling(candidate, rewriter);

  // Splice the accumulation loop and convert it to use an index induction
  // variable instead.
  spliceAccumulatorAndConvertToIndex(candidate, rewriter);

  // LICM as preparation: The contraction lowering patterns are sensitive to
  // unexpected ops in the accumulator loop.
  moveLoopInvariantCode(candidate.splicedAccLoop);

  // Finally, apply the upstream patterns.
  return applyNanokernelPatterns(candidate, rewriter);
}

struct ConvertDotToNanokernel
    : public triton::cpu::impl::ConvertDotToNanokernelBase<
          ConvertDotToNanokernel> {
  ConvertDotToNanokernel(std::string cpuFeatures) {
    this->cpuFeatures = cpuFeatures;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    if (cpuFeatures.empty()) {
      LDBG("No CPU features specified. Skipping pass.");
      return;
    }
    unsigned mask = parseCPUFeatures(cpuFeatures);
    LDBG("Available CPU features: " << stringifyISAExtMask(mask));

    SmallVector<DotOpCandidate, 2> candidates;
    mod->walk([&](triton::cpu::DotOp op) {
      DotOpCandidate candidate;
      if (isNanokernelCandidate(op, candidate, mask)) {
        LLVM_DEBUG({
          LDBG("Found nanokernel candidate");
          LDBG("  Target: " << stringifyISAExtMask(candidate.target));
          LDBG("  Op: " << candidate.dot);
          LDBG("  blockM: " << candidate.blockM);
          LDBG("  blockN: " << candidate.blockN);
          LDBG("  blockK: " << candidate.blockK);
          LDBG("  LHS read: " << candidate.lhsRead);
          LDBG("  RHS read: " << candidate.rhsRead);
          if (!candidate.accRead || !candidate.accWrite)
            LDBG("  Accumulating into temporary buffer");
          else {
            LDBG("  Acc read: " << candidate.accRead);
            LDBG("  Acc write: " << candidate.accWrite);
          }
        });
        candidates.push_back(candidate);
      }
    });

    for (auto &candidate : candidates) {
      LDBG("Starting conversion of candidate: " << candidate.dot);
      PatternRewriter rewriter(context);
      if (succeeded(convertCandidate(candidate, rewriter))) {
        LDBG("Conversion succeeded!");
      } else {
        LDBG("Conversion failed!");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

namespace mlir::triton::cpu {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertDotToNanokernel(std::string cpuFeatures) {
  return std::make_unique<ConvertDotToNanokernel>(cpuFeatures);
}

} // namespace mlir::triton::cpu
