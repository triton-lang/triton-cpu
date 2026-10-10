#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_RETILEELEMENTWISEOPS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "triton-cpu-retile-elementwise-ops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {
struct VRFInfo {
  unsigned nRegisters;
  unsigned registerBitWidth;
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

static SmallVector<int64_t> getUnrollingShape(ArrayRef<int64_t> shape,
                                              unsigned elemBitWidth,
                                              VRFInfo &vrfInfo) {
  // NB: Triton semantics guarantee that shapes are powers of 2, and we can
  // safely assume that the vector register width and element type width are
  // powers of 2 as well.

  // Heuristic: use half of the VRF's capacity to avoid spilling.
  unsigned nAvailRegs = vrfInfo.nRegisters / 2;

  SmallVector<int64_t> regShape(shape);
  SmallVector<int64_t> unrollShape(shape.size(), 1);

  // Convert the rightmost dimension from elements to a number of vector
  // registers. We checked earlier that this dimension spans at least one full
  // register.
  unsigned nElemsPerReg = vrfInfo.registerBitWidth / elemBitWidth;
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

// Discovers a DAG of elementwise operations (limited to arith and math
// dialects) with the specified shape. The leaves are either
// vector.transfer_read ops, or vector-typed constants from a splat value.
// Returns failure if an unsupported operation is encountered.
static LogicalResult buildElementwiseDAG(Operation *op, ArrayRef<int64_t> shape,
                                         SetVector<Operation *> &dag) {
  if (dag.contains(op))
    return success();
  if (op->getNumResults() != 1)
    return failure();
  VectorType vecTy = dyn_cast<VectorType>(op->getResult(0).getType());
  if (!vecTy || !llvm::equal(vecTy.getShape(), shape))
    return failure();

  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    if (readOp.hasOutOfBoundsDim() || readOp.getMask())
      return failure();
    dag.insert(op);
    return success();
  }

  if (!isa<arith::ArithDialect, math::MathDialect>(op->getDialect()))
    return failure();

  // Currently only handle constant ops that are splats.
  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    if (auto attr = dyn_cast<DenseElementsAttr>(constOp.getValue()))
      if (attr.isSplat()) {
        dag.insert(op);
        return success();
      }
    return failure();
  }

  for (Value operand : op->getOperands()) {
    Operation *predOp = operand.getDefiningOp();
    if (!predOp || failed(buildElementwiseDAG(predOp, shape, dag)))
      return failure();
  }

  // Record nodes in post-order.
  dag.insert(op);
  return success();
}

// Main driver. Retiles an elementwise DAG rooted in a vector.transfer_write op:
//   1. Attempts to discover an elementwise DAG from the value being written.
//   2. Determines a shape suitable for no-spill unrolling, based on the VRF
//      info.
//   3. Inserts a loop nest around shrunken clones of the elementwise
//      operations.
//
// Return value is success() if the transformation was either not applicable or
// succeeded, and failure() if the transformation failed mid-way.
static LogicalResult rewriteElementwiseDAG(vector::TransferWriteOp writeOp,
                                           VRFInfo &vrfInfo,
                                           PatternRewriter &rewriter) {
  LDBG("Attempt to retile elementwise DAG rooted in " << writeOp);

  ArrayRef<int64_t> dagShape = writeOp.getVectorType().getShape();
  SetVector<Operation *> dag;

  Operation *valueOp = writeOp.getValueToStore().getDefiningOp();
  if (!valueOp || failed(buildElementwiseDAG(valueOp, dagShape, dag)) ||
      dag.empty()) {
    LDBG("  No suitable elementwise DAG detected, giving up.");
    return success();
  }
  dag.insert(writeOp);

  LDBG("  Discovered DAG of size " << dag.size() << ".");

  unsigned elemBitWidth = 0;
  for (Operation *node : dag) {
    if (!(isa<arith::ConstantOp, vector::TransferWriteOp>(node) ||
          llvm::all_of(node->getUsers(), [&dag](Operation *user) {
            return dag.contains(user);
          }))) {
      LDBG("  Node " << node << " has external users, giving up.");
      return success();
    }
    elemBitWidth =
        std::max(elemBitWidth,
                 getVectorType(node).getElementType().getIntOrFloatBitWidth());
  }

  if (dagShape.back() < vrfInfo.registerBitWidth / elemBitWidth) {
    LDBG("  Minor dimension does not cover at least one element, giving up.");
    return success();
  }

  SmallVector<int64_t> unrShape =
      getUnrollingShape(dagShape, elemBitWidth, vrfInfo);
  if (llvm::equal(dagShape, unrShape)) {
    LDBG("  Unrolling shape is identical to DAG shape, nothing to do.");
    return success();
  }

  rewriter.setInsertionPoint(writeOp);
  auto uloc = rewriter.getUnknownLoc();

  // Set up the loop nest according to the unrolling shape. Single-iteration
  // loops will be canonicalized away later.
  Value c0 = arith::ConstantIndexOp::create(rewriter, uloc, 0);
  SmallVector<scf::ForOp> loops;
  SmallVector<Value> ivs;
  for (auto [dagDim, unrDim] : llvm::zip(dagShape, unrShape)) {
    if (!loops.empty())
      rewriter.setInsertionPoint(loops.back().getBody()->getTerminator());

    auto lb = c0;
    auto ub = arith::ConstantIndexOp::create(rewriter, uloc, dagDim);
    auto step = arith::ConstantIndexOp::create(rewriter, uloc, unrDim);

    auto loop =
        scf::ForOp::create(rewriter, uloc, lb, ub, step, ValueRange{},
                           [&](OpBuilder &b, Location, Value iv, ValueRange) {
                             ivs.push_back(iv);
                             scf::YieldOp::create(b, uloc, ValueRange{});
                           });
    loops.push_back(loop);
  }

  // Construct the innermost loop's body.
  rewriter.setInsertionPoint(loops.back().getBody()->getTerminator());

  // Helper to apply loop induction variables to a set of indices.
  auto addIVs = [&](SmallVector<Value> &indices, Location loc) {
    auto idxIt = indices.rbegin(), idxEnd = indices.rend();
    auto ivIt = ivs.rbegin(), ivEnd = ivs.rend();
    for (; idxIt != idxEnd && ivIt != ivEnd; ++idxIt, ++ivIt)
      *idxIt = arith::AddIOp::create(rewriter, loc, *idxIt, *ivIt);
  };

  IRMapping mapping;
  for (Operation *node : dag) {
    VectorType unrTy =
        VectorType::get(unrShape, getVectorType(node).getElementType());
    Location loc = node->getLoc();

    if (auto readOp = dyn_cast<vector::TransferReadOp>(node)) {
      SmallVector<Value> indices = readOp.getIndices();
      addIVs(indices, loc);

      auto unrReadOp = vector::TransferReadOp::create(
          rewriter, loc, unrTy, readOp.getBase(), indices, readOp.getPadding(),
          readOp.getPermutationMap(), readOp.getInBoundsValues());
      mapping.map(readOp.getResult(), unrReadOp.getResult());
      continue;
    }

    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(node)) {
      SmallVector<Value> indices = writeOp.getIndices();
      addIVs(indices, loc);
      auto unrWriteOp = vector::TransferWriteOp::create(
          rewriter, loc, mapping.lookup(writeOp.getValue()), writeOp.getBase(),
          indices, writeOp.getPermutationMap(), writeOp.getInBoundsValues());
      continue;
    }

    if (auto constOp = dyn_cast<arith::ConstantOp>(node)) {
      auto attr = dyn_cast<DenseElementsAttr>(constOp.getValue());
      assert(attr && attr.isSplat()); // checked earlier
      attr = attr.resizeSplat(unrTy);
      auto unrConstOp = arith::ConstantOp::create(rewriter, loc, unrTy, attr);
      mapping.map(constOp.getResult(), unrConstOp.getResult());
      continue;
    }

    // "Clone" elementwise ops with the new, smaller shape.
    SmallVector<Value> unrOperands;
    for (auto operand : node->getOperands())
      unrOperands.push_back(mapping.lookup(operand));

    Operation *newNode = rewriter.create(
        loc, rewriter.getStringAttr(node->getName().getStringRef()),
        unrOperands, TypeRange{unrTy}, node->getAttrs());
    mapping.map(node->getResult(0), newNode->getResult(0));
  }

  // Clean up. We established earlier that none of the ops have users outside
  // the DAG.
  for (auto &node : llvm::reverse(dag))
    if (!isa<arith::ConstantOp>(node))
      rewriter.eraseOp(node);

  LDBG("  Success.");
  return success();
}

namespace {

struct RetileElementwiseOps
    : public triton::cpu::impl::RetileElementwiseOpsBase<RetileElementwiseOps> {
  RetileElementwiseOps(std::string cpuFeatures) {
    this->cpuFeatures = cpuFeatures;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    VRFInfo vrfInfo = getVRFInfo(cpuFeatures);
    LDBG("Retiling elementwise ops for VRF size " << vrfInfo.nRegisters << "x"
                                                  << vrfInfo.registerBitWidth);

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
createRetileElementwiseOps(std::string cpuFeatures) {
  return std::make_unique<RetileElementwiseOps>(cpuFeatures);
}

} // namespace mlir::triton::cpu
