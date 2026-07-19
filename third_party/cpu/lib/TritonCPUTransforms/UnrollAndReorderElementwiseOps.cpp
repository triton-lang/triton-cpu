#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_UNROLLANDREORDERELEMENTWISEOPS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "triton-cpu-unroll-and-reorder-elementwise-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {
struct VRFInfo {
  unsigned nRegisters;
  unsigned registerWidthBits;
};
} // namespace

// Guesses the vector register file info based on the CPU features string.
// TODO: Consider using DLTI in the pipeline.
static VRFInfo getVRFInfo(std::string cpuFeatures) {
  if (cpuFeatures.find("avx512") != std::string::npos)
    return {32, 512};

  // Assume AVX2 if not AVX512.
  return {16, 256};
}

static VectorType getVectorType(Operation *op) {
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op))
    return writeOp.getVectorType();
  return cast<VectorType>(op->getResult(0).getType());
}

static SmallVector<int64_t> getUnrollingShape(VectorType vecTy,
                                              VRFInfo &vrfInfo) {
  // NB: Triton semantics guarantee that shapes are powers of 2, and we can
  // safely assume that the vector register width and element type width are
  // powers of 2 as well.

  // Heuristic: use half of the VRF's capacity to avoid spilling.
  unsigned nAvailRegs = vrfInfo.nRegisters / 2;

  SmallVector<int64_t> regShape(vecTy.getShape());
  SmallVector<int64_t> unrollShape(vecTy.getRank(), 1);

  // Convert the rightmost dimension from elements to a number of vector
  // registers. We checked earlier that this dimension spans at least one full
  // register.
  unsigned nElemsPerReg =
      vrfInfo.registerWidthBits / vecTy.getElementTypeBitWidth();
  assert(regShape.back() % nElemsPerReg == 0);
  regShape.back() /= nElemsPerReg;

  // Iterate over the shape in reverse order and keep track of the cumulative
  // product representing the number of registers required to hold the current
  // sub-shape.
  unsigned nReqRegs = 1;
  for (auto regShapeIt = regShape.rbegin(),
            unrollShapeIt = unrollShape.rbegin();
       regShapeIt != regShape.rend(); ++regShapeIt, ++unrollShapeIt) {
    // If we match or exceed the threshold in this dimension, determine the
    // maximum size that'll fit. The other leading dimensions will remain 1.
    if (*regShapeIt * nReqRegs >= nAvailRegs) {
      assert(nAvailRegs % nReqRegs == 0);
      *unrollShapeIt = nAvailRegs / nReqRegs;
      break;
    }

    // Otherwise, keep this dimension as-is and update the number of required
    // registers.
    *unrollShapeIt = *regShapeIt;
    nReqRegs *= *regShapeIt;
  }

  // Convert the rightmost dimension back to the number of elements.
  unrollShape.back() *= nElemsPerReg;
  return unrollShape;
}

// Attempts to find a common unrolling shape for all ops in the DAG. The suffix
// of the common shape divides the shape of each op's vector type. Returns
// failure() if no common shape can be found.
static FailureOr<SmallVector<int64_t>>
getUnrollingShape(ArrayRef<Operation *> ops, VRFInfo &vrfInfo) {
  SmallVector<int64_t> unrollShape;
  for (auto *op : ops) {
    auto us = getUnrollingShape(getVectorType(op), vrfInfo);

    if (unrollShape.empty()) {
      unrollShape = us;
      continue;
    }

    if (unrollShape.size() > us.size())
      us.insert(us.begin(), unrollShape.size() - us.size(), 1);
    else if (unrollShape.size() < us.size())
      unrollShape.insert(unrollShape.begin(), us.size() - unrollShape.size(),
                         1);

    if (std::lexicographical_compare(us.begin(), us.end(), unrollShape.begin(),
                                     unrollShape.end()))
      unrollShape = us;
  }

  // Check divisibility.
  for (auto *op : ops) {
    for (auto [vs, us] : llvm::zip(getVectorType(op).getShape(), unrollShape))
      if (vs % us != 0) {
        LDBG("  Could not determine common unrolling shape.");
        return failure();
      }
  }

  LLVM_DEBUG(DBGS() << "  Found common unrolling shape: ";
             llvm::interleaveComma(unrollShape, llvm::dbgs());
             llvm::dbgs() << "\n");
  return unrollShape;
}

// DFS to discover a DAG of operations that operate on a vector type that is
// guaranteed to cause spilling.
static void buildElementwiseDAG(Operation *op, VRFInfo &vrfInfo,
                                SetVector<Operation *> &dag) {
  if (dag.contains(op))
    return;

  for (Value operand : op->getOperands()) {
    auto vecTy = dyn_cast<VectorType>(operand.getType());
    if (!vecTy)
      continue;

    // The simplest of heuristics: if the vector type is smaller than half of
    // the VRF's capacity, then we assume that it is safe to operate on it
    // without spilling.
    if (vecTy.getNumElements() * vecTy.getElementTypeBitWidth() <=
        vrfInfo.nRegisters / 2 * vrfInfo.registerWidthBits)
      continue;

    // To keep things simple, enforce that the shape in the rightmost dimension
    // is a multiple of the number of elements that can fit in a vector
    // register.
    unsigned nElemsPerReg =
        vrfInfo.registerWidthBits / vecTy.getElementTypeBitWidth();
    if (vecTy.getShape().back() % nElemsPerReg != 0)
      continue;

    Operation *predOp = operand.getDefiningOp();
    if (!predOp)
      continue;

    // Only consider arith and math ops, as well as selected vector ops.
    if (!isa<arith::ArithDialect, math::MathDialect>(predOp->getDialect()) &&
        !isa<vector::ShapeCastOp, vector::BroadcastOp, vector::TransferReadOp>(
            predOp))
      continue;

    buildElementwiseDAG(predOp, vrfInfo, dag);
  }

  // Record nodes in post-order.
  dag.insert(op);
}

static constexpr auto *unrollShapeAttrName = "unroll_shape";
static constexpr auto *unrollOrderAttrName = "unroll_order";

namespace {
// The upstream vector unroll patterns don't always propagate discardable
// attributes to the unrolled ops. This should be fixed upstream, but until
// then, this listener looks for the canonical chain of insert_strided_slice ops
// and propagates the unroll_order attribute to the clones.
struct PropagateOrderAttrListener : public virtual RewriterBase::Listener {
  template <typename OpTy>
  void propagateAttrToUnrolledOps(Operation *op, IntegerAttr attr,
                                  Value newValue) {
    if (auto oldOp = dyn_cast<OpTy>(op)) {
      Value it = newValue;
      vector::InsertStridedSliceOp insertOp;
      OpTy newOp;
      while ((insertOp = it.getDefiningOp<vector::InsertStridedSliceOp>()) &&
             (newOp = insertOp.getValueToStore().getDefiningOp<OpTy>())) {
        newOp->setAttr(unrollOrderAttrName, attr);
        it = insertOp.getDest();
      }
    }
  }

  void notifyOperationReplaced(Operation *op, ValueRange newValues) override {
    auto unrollOrderAttr = op->getAttrOfType<IntegerAttr>(unrollOrderAttrName);
    if (!unrollOrderAttr || newValues.size() != 1)
      return;
    Value newValue = newValues.front();
    propagateAttrToUnrolledOps<vector::TransferReadOp>(op, unrollOrderAttr,
                                                       newValue);
    propagateAttrToUnrolledOps<vector::ShapeCastOp>(op, unrollOrderAttr,
                                                    newValue);
  }
};
} // namespace

// Apply the upstream vector unroll patterns for ops that have the unroll_shape
// attribute and are contained in the given scf.execute_region.
static LogicalResult unrollOpsIn(scf::ExecuteRegionOp exec) {
  vector::UnrollVectorOptions unrollOptions;
  // Op has attribute and is contained in the execute_region.
  unrollOptions.setFilterConstraint([&](Operation *op) {
    return success(op->hasAttr(unrollShapeAttrName) &&
                   op->getParentOfType<scf::ExecuteRegionOp>() == exec);
  });
  // Use the shape for the attribute.
  unrollOptions.setNativeShapeFn(
      [&](Operation *op) -> std::optional<SmallVector<int64_t>> {
        auto vals = op->getAttrOfType<ArrayAttr>(unrollShapeAttrName)
                        .getAsValueRange<IntegerAttr>();
        SmallVector<int64_t> shape = llvm::to_vector(llvm::map_range(
            vals, [](const APInt &v) { return v.getSExtValue(); }));

        return shape;
      });

  RewritePatternSet patterns(exec.getContext());
  vector::populateVectorUnrollPatterns(patterns, unrollOptions);
  GreedyRewriteConfig config;
  PropagateOrderAttrListener listener;
  config.setListener(&listener);
  return applyPatternsGreedily(exec->getParentOfType<triton::FuncOp>(),
                               std::move(patterns), config);
}

// Main driver. Checks whether the DAG rooted in `writeOp` is amenable for
// unrolling and reordering, and if so, performs the transformation.
//
// Return value is success() if the transformation was either not applicable or
// succeeded, and failure() if the transformation failed mid-way.
static LogicalResult rewriteElementwiseDAG(vector::TransferWriteOp writeOp,
                                           VRFInfo &vrfInfo,
                                           PatternRewriter &rewriter) {
  LDBG("Attempt to rewrite elementwise DAG rooted in " << writeOp);
  SetVector<Operation *> dag;
  buildElementwiseDAG(writeOp, vrfInfo, dag);

  if (dag.size() <= 1) {
    LDBG("  No suitable elementwise DAG detected, giving up.");
    return success();
  }
  LDBG("  Discovered DAG of size " << dag.size() << ".");

  bool allUsersInDAG = llvm::all_of(dag, [&dag](Operation *node) {
    return isa<arith::ConstantOp, vector::TransferWriteOp>(node) ||
           llvm::all_of(node->getUsers(),
                        [&dag](Operation *user) { return dag.contains(user); });
  });
  if (!allUsersInDAG) {
    LDBG("  Elementwise DAG has external users, giving up.");
    return success();
  }

  // Determine an unrolling shape that is suitable for all ops.
  SmallVector<Operation *> ops = dag.takeVector();
  auto maybeUnrollShape = getUnrollingShape(ops, vrfInfo);
  if (failed(maybeUnrollShape))
    return success();
  ArrayRef<int64_t> unrollShape = *maybeUnrollShape;

  rewriter.setInsertionPoint(writeOp);
  Location loc = writeOp.getLoc();

  // Create an scf.execute_region to contain the DAG's ops.
  auto exec = scf::ExecuteRegionOp::create(rewriter, loc, {});
  Block *execBlock = rewriter.createBlock(&exec.getRegion());
  rewriter.setInsertionPointToStart(execBlock);

  // Clone ops into the execute_region.
  IRMapping mapping;
  unsigned order = 0;
  for (auto *op : ops) {
    Operation *cloned = rewriter.clone(*op, mapping);
    auto rank = getVectorType(cloned).getRank();
    cloned->setAttr(unrollShapeAttrName,
                    rewriter.getI64ArrayAttr(unrollShape.take_back(rank)));
    cloned->setAttr(unrollOrderAttrName, rewriter.getI64IntegerAttr(order++));
  }

  // Insert terminator and replace the original write op with the
  // execute_region.
  scf::YieldOp::create(rewriter, loc, ValueRange{});
  rewriter.replaceOp(writeOp, exec.getResults());

  // Unroll the ops in the execute_region.
  if (failed(unrollOpsIn(exec))) {
    LDBG("  Error: Failed to unroll ops in execute_region.");
    return failure();
  }

  // Sort the ops into buckets, according to their unroll_order attribute:
  // 0:    Ops that don't have an order attribute and are not
  //       vector.transfer.write ops.
  // 1..N: Ops that have an order attribute.
  // N+1:  vector.transfer.write ops.
  SmallVector<SmallVector<Operation *>> buckets(order + 2);
  Operation *terminator = execBlock->getTerminator();
  for (Operation &op : execBlock->getOperations()) {
    if (&op == terminator)
      continue;
    if (isa<vector::TransferWriteOp>(op)) {
      buckets.back().push_back(&op);
      continue;
    }

    auto orderAttr = op.getAttrOfType<IntegerAttr>(unrollOrderAttrName);
    unsigned bucketIdx = orderAttr ? orderAttr.getInt() + 1 : 0;
    buckets[bucketIdx].push_back(&op);
  }

  // We'll bring ops in the desired order by subsequently moving them before the
  // terminator.
  for (auto *op : buckets.front())
    op->moveBefore(terminator);

  // Remove special buckets: The first one we just handled, and any empty
  // buckets (that corresponded to constants or folded operations).
  buckets.erase(buckets.begin());
  buckets.erase(std::remove_if(buckets.begin(), buckets.end(),
                               [](auto &bucket) { return bucket.empty(); }),
                buckets.end());

  if (buckets.empty()) {
    LDBG("  Error: All buckets were empty.");
    return failure();
  }

  // All remaining buckets should contain the same number of ops, equivalent to
  // the unroll factor.
  unsigned unrollFactor = buckets.back().size();
  if (unrollFactor == 0 || !llvm::all_of(buckets, [&](auto &bucket) {
        return bucket.size() == unrollFactor;
      })) {
    LDBG("  Error: Unexpected number of ops in buckets.");
    return failure();
  }

  for (unsigned u = 0; u < unrollFactor; ++u)
    for (unsigned b = 0; b < buckets.size(); ++b) {
      Operation *op = buckets[b][u];
      op->moveBefore(terminator);
      op->removeAttr(unrollShapeAttrName);
      op->removeAttr(unrollOrderAttrName);
    }

  LDBG("  Success.");
  return success();
}

namespace {

struct UnrollAndReorderElementwiseOps
    : public triton::cpu::impl::UnrollAndReorderElementwiseOpsBase<
          UnrollAndReorderElementwiseOps> {
  UnrollAndReorderElementwiseOps(std::string cpuFeatures) {
    this->cpuFeatures = cpuFeatures;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    VRFInfo vrfInfo = getVRFInfo(cpuFeatures);
    LDBG("Vectorizing elementwise ops for VRF size "
         << vrfInfo.nRegisters << "x" << vrfInfo.registerWidthBits);

    PatternRewriter rewriter(context);
    auto res = mod->walk([&](vector::TransferWriteOp write) {
      if (failed(rewriteElementwiseDAG(write, vrfInfo, rewriter)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (res.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace

namespace mlir::triton::cpu {

std::unique_ptr<OperationPass<ModuleOp>>
createUnrollAndReorderElementwiseOps(std::string cpuFeatures) {
  return std::make_unique<UnrollAndReorderElementwiseOps>(cpuFeatures);
}

} // namespace mlir::triton::cpu
