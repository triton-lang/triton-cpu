#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_VECTORIZEELEMENTWISEOPS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "triton-cpu-vectorize-elementwise-ops"
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

// DFS to discover a DAG of operations that operand on a vector type that is
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

    buildElementwiseDAG(predOp, vrfInfo, dag);
  }

  // Record nodes in post-order.
  dag.insert(op);
}

static constexpr auto *unrollShapeAttrName = "unroll_shape";

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
  return applyPatternsGreedily(exec->getParentOfType<triton::FuncOp>(),
                               std::move(patterns));
}

static void rewriteElementwiseDAG(vector::TransferWriteOp writeOp,
                                  VRFInfo &vrfInfo, PatternRewriter &rewriter) {
  LDBG("Attempt to rewrite elementwise DAG rooted in " << writeOp);
  SetVector<Operation *> dag;
  buildElementwiseDAG(writeOp, vrfInfo, dag);

  LDBG("  Discovered DAG of size " << dag.size() << ".");
  if (dag.size() <= 1)
    return;

  bool allUsersInDAG = llvm::all_of(dag, [&dag](Operation *node) {
    return isa<arith::ConstantOp, vector::TransferWriteOp>(node) ||
           llvm::all_of(node->getUsers(),
                        [&dag](Operation *user) { return dag.contains(user); });
  });
  if (!allUsersInDAG) {
    LDBG("  Elementwise DAG has external users, aborting.");
    return;
  }

  // Determine an unrolling shape that is suitable for all ops.
  SmallVector<Operation *> ops = dag.takeVector();
  auto maybeUnrollShape = getUnrollingShape(ops, vrfInfo);
  if (failed(maybeUnrollShape))
    return;
  ArrayRef<int64_t> unrollShape = *maybeUnrollShape;

  rewriter.setInsertionPoint(writeOp);
  Location loc = writeOp.getLoc();

  // Create an scf.execute_region to contain the DAG's ops.
  auto exec = scf::ExecuteRegionOp::create(rewriter, loc, {});
  Block *execBlock = rewriter.createBlock(&exec.getRegion());
  rewriter.setInsertionPointToStart(execBlock);

  // Clone ops into the execute_region and determine unrolling shape.
  IRMapping mapping;
  for (auto *op : ops) {
    Operation *cloned = rewriter.clone(*op, mapping);
    auto rank = getVectorType(cloned).getRank();
    cloned->setAttr(unrollShapeAttrName,
                    rewriter.getI64ArrayAttr(unrollShape.take_back(rank)));
  }

  // Insert terminator and replace the original write op with the
  // execute_region.
  scf::YieldOp::create(rewriter, loc, ValueRange{});
  rewriter.replaceOp(writeOp, exec.getResults());

  // Unroll the ops in the execute_region.
  if (failed(unrollOpsIn(exec))) {
    LDBG("  Failed to unroll ops in execute_region.");
    return;
  }

  // Remove the unroll_shape attribute and reorder ops in the execute_region.
  // The idea is to produce chains from sources to the sinks (= unrolled
  // transfer_write ops) that do not exceed the VRF's capacity.
  SmallVector<Operation *> unrolledOps;
  llvm::transform(execBlock->getOperations(), std::back_inserter(unrolledOps),
                  [](Operation &op) { return &op; });
  Operation *lastInvOp = nullptr;
  for (auto *unrOp : unrolledOps) {
    if (unrOp == execBlock->getTerminator())
      continue;

    unrOp->removeAttr(unrollShapeAttrName);

    Operation *lastOperandOp = nullptr;
    for (Value operand : unrOp->getOperands()) {
      Operation *operandOp = operand.getDefiningOp();
      if (!operandOp || operandOp->getBlock() != execBlock)
        continue;
      if (!lastOperandOp || lastOperandOp->isBeforeInBlock(operandOp))
        lastOperandOp = operandOp;
    }
    if (lastOperandOp)
      unrOp->moveAfter(lastOperandOp);
    else {
      // Make sure invariant ops moved to the top remain in their original
      // order.
      if (lastInvOp)
        unrOp->moveAfter(lastInvOp);
      else
        unrOp->moveBefore(&execBlock->front());
      lastInvOp = unrOp;
    }
  }
  LDBG("  Success.");
}

namespace {

struct VectorizeElementwiseOps
    : public triton::cpu::impl::VectorizeElementwiseOpsBase<
          VectorizeElementwiseOps> {
  VectorizeElementwiseOps(std::string cpuFeatures) {
    this->cpuFeatures = cpuFeatures;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    VRFInfo vrfInfo = getVRFInfo(cpuFeatures);
    LDBG("Vectorizing elementwise ops for VRF size "
         << vrfInfo.nRegisters << "x" << vrfInfo.registerWidthBits);

    PatternRewriter rewriter(context);
    mod->walk([&](vector::TransferWriteOp write) {
      rewriteElementwiseDAG(write, vrfInfo, rewriter);
    });
  }
};

} // namespace

namespace mlir::triton::cpu {

std::unique_ptr<OperationPass<ModuleOp>>
createVectorizeElementwiseOps(std::string cpuFeatures) {
  return std::make_unique<VectorizeElementwiseOps>(cpuFeatures);
}

} // namespace mlir::triton::cpu
