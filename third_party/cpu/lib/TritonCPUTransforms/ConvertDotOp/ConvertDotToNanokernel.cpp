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

// This structure is used to hold candidates for application of a nanokernel
// pattern.
struct DotOpCandidate {
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
};

bool checkElemTypes(Type lhsElemTy, Type rhsElemTy, Type accElemTy,
                    Type resElemTy, InstructionSet target) {
  if (accElemTy != resElemTy) {
    // Constrain the two for simplicity.
    LDBG("  Drop candidate. Expect same accumulation and result type.");
    return false;
  }

  if (lhsElemTy != rhsElemTy) {
    LDBG("  Drop candidate. Mixed type input is not supported");
    return false;
  }

  if (target == InstructionSet::AMX)
    return lhsElemTy.isBF16() && accElemTy.isF32();

  LDBG("  Drop candidate. Unsupported type combination");
  return false;
}

bool checkInputShapes(VectorType lhsTy, VectorType resTy,
                      DotOpCandidate &candidate, InstructionSet target) {
  candidate.blockM = resTy.getDimSize(0);
  candidate.blockN = resTy.getDimSize(1);
  candidate.blockK = lhsTy.getDimSize(1);

  if (target == InstructionSet::AMX)
    // Optimal configuration with 2x2 accumulator tiles.
    return candidate.blockM == 32 && candidate.blockN == 32 &&
           candidate.blockK == 32;

  LDBG("  Drop candidate. Unsupported shapes");
  return false;
}

// Check if specified DotOp can be lowered to a vector contraction on which the
// nanokernel patterns for the given target instruction set apply.
// If conversion is possible, then true is returned and candidate structure is
// filled with detailed transformation info.
bool isNanokernelCandidate(triton::cpu::DotOp op, DotOpCandidate &candidate,
                           InstructionSet target) {
  VectorType lhsTy = cast<VectorType>(op.getA().getType());
  VectorType rhsTy = cast<VectorType>(op.getB().getType());
  VectorType accTy = cast<VectorType>(op.getC().getType());
  VectorType resTy = cast<VectorType>(op.getType());

  LDBG("Considering candidate op: " << op);

  if (accTy.getRank() != 2) {
    LDBG("  Drop candidate. Only 2D case is supported.");
    return false;
  }

  if (!checkElemTypes(lhsTy.getElementType(), rhsTy.getElementType(),
                      accTy.getElementType(), resTy.getElementType(), target))
    return false;

  if (!checkInputShapes(lhsTy, resTy, candidate, target))
    return false;

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
  if (!accRead) {
    LDBG("  Drop candidate. Expected vector.transfer_read op initializing the "
         "accumulator, but got: "
         << accLoop.getTiedLoopInit(cast<BlockArgument>(op.getC()))->get());
    return false;
  }

  auto accWrite = accLoop.getResults().front().hasOneUse()
                      ? dyn_cast<vector::TransferWriteOp>(
                            *accLoop.getResults().front().getUsers().begin())
                      : nullptr;
  if (!accWrite) {
    LDBG("  Drop candidate. Expected single vector.transfer_write op storing "
         "the accumulator");
    return false;
  }

  auto checkTransferOp = [&](auto transferOp) {
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

  candidate.dot = op;
  candidate.accLoop = accLoop;
  candidate.lhsRead = lhsRead;
  candidate.rhsRead = rhsRead;
  candidate.accRead = accRead;
  candidate.accWrite = accWrite;

  return true;
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

void performRegisterTiling(DotOpCandidate &candidate, InstructionSet target,
                           PatternRewriter &rewriter) {

  llvm::SmallDenseMap<Operation *, SmallVector<int64_t>> regTileSizes;

  assert(target == InstructionSet::AMX);
  regTileSizes[candidate.contract] = {16, 16, 32};
  regTileSizes[candidate.lhsRead] = {16, 32};
  regTileSizes[candidate.rhsRead] = {32, 16};
  regTileSizes[candidate.accRead] = {16, 16};
  regTileSizes[candidate.accWrite] = {16, 16};

  vector::UnrollVectorOptions unrollOptions;
  unrollOptions.setFilterConstraint(
      [&](Operation *op) { return success(regTileSizes.contains(op)); });
  unrollOptions.setNativeShapeFn(
      [&](Operation *op) -> std::optional<SmallVector<int64_t>> {
        return regTileSizes.lookup(op);
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
    auto newExtractOp = vector::ExtractStridedSliceOp::create(
        rewriter, oldExtractOp.getLoc(), oldExtractOp.getType(), oldInit,
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
                                      InstructionSet target,
                                      PatternRewriter &rewriter) {
  assert(target == InstructionSet::AMX);
  RewritePatternSet patterns(candidate.contract.getContext());
  x86::populateVectorContractToAMXDotProductPatterns(patterns);
  return applyPatternsGreedily(
      candidate.contract->getParentOfType<triton::FuncOp>(),
      std::move(patterns));
}

LogicalResult convertCandidate(DotOpCandidate &candidate, InstructionSet target,
                               PatternRewriter &rewriter) {
  // Insert memref.subview ops to ensure all transfer ops have zero indices.
  moveIndicesToSubview(candidate, rewriter);

  // 1:1 replacement of triton_cpu.dot -> vector.contract.
  convertToContract(candidate, rewriter);

  // Apply target-specific register tiling via upstream vector-dialect unroll
  // patterns.
  performRegisterTiling(candidate, target, rewriter);

  // Splice the accumulation loop and convert it to use an index induction
  // variable instead.
  spliceAccumulatorAndConvertToIndex(candidate, rewriter);

  // LICM as preparation: The contraction lowering patterns are sensitive to
  // unexpected ops in the accumulator loop.
  moveLoopInvariantCode(candidate.splicedAccLoop);

  // Finally, apply the upstream patterns.
  return applyNanokernelPatterns(candidate, target, rewriter);
}

struct ConvertDotToNanokernel
    : public triton::cpu::impl::ConvertDotToNanokernelBase<
          ConvertDotToNanokernel> {
  ConvertDotToNanokernel(InstructionSet target) { this->target = target; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    mod.dump();

    if (this->target != InstructionSet::AMX) {
      LDBG("Currently only AMX target is supported. Skipping pass.");
      return;
    }

    SmallVector<DotOpCandidate, 2> candidates;
    mod->walk([&candidates, this](triton::cpu::DotOp op) {
      DotOpCandidate candidate;
      if (isNanokernelCandidate(op, candidate, this->target)) {
        LLVM_DEBUG({
          LDBG("Found nanokernel candidate");
          LDBG("  Op: " << candidate.dot);
          LDBG("  blockM: " << candidate.blockM);
          LDBG("  blockN: " << candidate.blockN);
          LDBG("  blockK: " << candidate.blockK);
          LDBG("  LHS read: " << candidate.lhsRead);
          LDBG("  RHS read: " << candidate.rhsRead);
          LDBG("  Acc read: " << candidate.accRead);
          LDBG("  Acc write: " << candidate.accWrite);
        });
        candidates.push_back(candidate);
      }
    });

    for (auto &candidate : candidates) {
      LDBG("Starting conversion of candidate: " << candidate.dot);
      PatternRewriter rewriter(context);
      if (succeeded(convertCandidate(candidate, target, rewriter))) {
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
createConvertDotToNanokernel(InstructionSet target) {
  return std::make_unique<ConvertDotToNanokernel>(target);
}

} // namespace mlir::triton::cpu
