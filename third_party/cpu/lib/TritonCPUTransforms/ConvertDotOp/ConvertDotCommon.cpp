#include "ConvertDotCommon.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace mlir {
namespace triton {
namespace cpu {

bool isLoopCarriedAcc(Value acc) {
  LDBG("Check if accumulator can be held in tiles: " << acc);
  if (!acc.hasOneUse()) {
    LDBG("  No. Has multiple uses.");
    for (auto op : acc.getUsers())
      LDBG("    " << *op);
    return false;
  }

  auto blockArg = dyn_cast<BlockArgument>(acc);
  if (!blockArg) {
    LDBG("  No. Not a block argument.");
    return false;
  }

  auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  if (!forOp) {
    LDBG("  No. Not in a for-loop.");
    return false;
  }

  Value updAcc = acc.getUsers().begin()->getResult(0);
  if (!updAcc.hasOneUse()) {
    LDBG("  No. Has multiple uses.");
    return false;
  }

  auto &updAccUse = *updAcc.getUses().begin();
  if (!isa<scf::YieldOp>(updAccUse.getOwner()) ||
      updAccUse.getOperandNumber() !=
          (blockArg.getArgNumber() - forOp.getNumInductionVars())) {
    LDBG("  No. Loop carried dependency not detected.");
    return false;
  }

  LDBG("  Yes.");
  return true;
}

Value getInitAccValue(Value val) {
  auto blockArg = cast<BlockArgument>(val);
  auto forOp = cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  int initValIdx = blockArg.getArgNumber() - forOp.getNumInductionVars();
  return forOp.getInitArgs()[initValIdx];
}

namespace {

// Check if val is a result of transpose operation. If it is, then return
// a source of that transpose operation. Otherwise, return nullptr.
Value getTransposedSrc(Value val) {
  auto transposeOp = val.getDefiningOp<vector::TransposeOp>();
  if (transposeOp)
    return transposeOp.getVector();
  return nullptr;
}

// We are looking for the following sequence:
//   %tmp1, %tmp2 = vector.deinterleave %src
//   %tmp3 = vector.transpose %tmp1, [1, 0]
//   %tmp4 = vector.transpose %tmp2, [1, 0]
//   %tmp5 = vector.interleave %tmp3, %tmp4
//   %val = vector.transpose %tmp5, [1, 0]
// and return %src if pattern matching succeeds.
Value getVnniSrcImpl2(Value val) {
  auto transposedVal = getTransposedSrc(val);
  if (!transposedVal)
    return nullptr;

  auto interleave = transposedVal.getDefiningOp<vector::InterleaveOp>();
  if (!interleave)
    return nullptr;

  auto tmp1 = getTransposedSrc(interleave.getLhs());
  auto tmp2 = getTransposedSrc(interleave.getRhs());
  if (!tmp1 || !tmp2)
    return nullptr;

  auto deinterleave1 = tmp1.getDefiningOp<vector::DeinterleaveOp>();
  auto deinterleave2 = tmp2.getDefiningOp<vector::DeinterleaveOp>();
  if (!deinterleave1 || deinterleave1 != deinterleave2 ||
      deinterleave1.getResult(0) != tmp1 || deinterleave2.getResult(1) != tmp2)
    return nullptr;

  return deinterleave1.getSource();
}

// We are looking for the following sequence:
//   %src02, %src13 = vector.deinterleave %src
//   %src0, %src2 = vector.deinterleave %src02
//   %src1, %src3 = vector.deinterleave %src13
//   %t0 = vector.transpose %src0, [1, 0]
//   %t1 = vector.transpose %src1, [1, 0]
//   %t2 = vector.transpose %src2, [1, 0]
//   %t3 = vector.transpose %src3, [1, 0]
//   %i02 = vector.interleave %t0, %t2
//   %i13 = vector.interleave %t1, %t3
//   %i = vector.interleave %i02, %i13
//   %val = vector.transpose %i, [1, 0]
// and return %src if pattern matching succeeds.
Value getVnniSrcImpl4(Value val) {
  auto transposedVal = getTransposedSrc(val);
  if (!transposedVal)
    return nullptr;

  auto interleave = transposedVal.getDefiningOp<vector::InterleaveOp>();
  if (!interleave)
    return nullptr;

  auto interleave02 = interleave.getLhs().getDefiningOp<vector::InterleaveOp>();
  auto interleave13 = interleave.getRhs().getDefiningOp<vector::InterleaveOp>();
  if (!interleave02 || !interleave13)
    return nullptr;

  auto transposed0 = getTransposedSrc(interleave02.getLhs());
  auto transposed1 = getTransposedSrc(interleave13.getLhs());
  auto transposed2 = getTransposedSrc(interleave02.getRhs());
  auto transposed3 = getTransposedSrc(interleave13.getRhs());
  if (!transposed0 || !transposed1 || !transposed2 || !transposed3)
    return nullptr;

  auto deinterleave02 = transposed0.getDefiningOp<vector::DeinterleaveOp>();
  auto deinterleave13 = transposed1.getDefiningOp<vector::DeinterleaveOp>();
  if (!deinterleave02 || !deinterleave13 ||
      deinterleave02.getResult(0) != transposed0 ||
      deinterleave13.getResult(0) != transposed1 ||
      deinterleave02.getResult(1) != transposed2 ||
      deinterleave13.getResult(1) != transposed3)
    return nullptr;

  auto deinterleave =
      deinterleave02.getSource().getDefiningOp<vector::DeinterleaveOp>();
  if (!deinterleave ||
      deinterleave.getResult(0) != deinterleave02.getSource() ||
      deinterleave.getResult(1) != deinterleave13.getSource())
    return nullptr;

  return deinterleave.getSource();
}

} // namespace

Value getVnniSrc(Value val) {
  Type elemTy = getElementTypeOrSelf(val.getType());

  // VNNI encoding is used for 8-bit and 16-bit values only.
  switch (elemTy.getIntOrFloatBitWidth()) {
  case 16:
    return getVnniSrcImpl2(val);
  case 8:
    return getVnniSrcImpl4(val);
  default:
    return nullptr;
  }
}

MemBuffer findInputBuffer(Value val, bool allowTransposed, bool allowVnni) {
  MemBuffer buf;

  if (allowTransposed) {
    auto transposed = getTransposedSrc(val);
    if (transposed) {
      val = transposed;
      buf.transposed = true;
    }
  } else if (allowVnni) {
    auto vnniVal = getVnniSrc(val);
    if (vnniVal) {
      val = vnniVal;
      buf.vnni = true;
    }
  }

  auto valLoad = val.getDefiningOp<vector::TransferReadOp>();
  if (!valLoad || hasMaskOrBoundsCheck(valLoad)) {
    LDBG("Couldn't find a buffer with input: " << val);
    return buf;
  }

  buf.memRef = valLoad.getBase();
  buf.indices = valLoad.getIndices();
  LLVM_DEBUG(
      DBGS() << "Found buffer with input: " << val << "\n";
      DBGS() << "  is Transposed: " << (buf.transposed ? "true" : "false")
             << "\n";
      DBGS() << "  is VNNI: " << (buf.vnni ? "true" : "false") << "\n";
      DBGS() << "  MemRef: " << buf.memRef << "\n"; DBGS() << "  Indices: ";
      llvm::interleaveComma(buf.indices, llvm::dbgs()); llvm::dbgs() << "\n");

  auto forOp = dyn_cast<scf::ForOp>(valLoad->getParentOp());
  if (!forOp) {
    LDBG("  Skip steps. Not in a for-loop.");
    return buf;
  }

  SmallVector<Value> step;
  for (auto index : buf.indices) {
    // The Triton kernel won't be using the `index` type, so if there is a cast,
    // look through it.
    if (auto cast =
            dyn_cast_if_present<arith::IndexCastOp>(index.getDefiningOp()))
      index = cast.getIn();

    // Represent loop-invariant values and constants as a null `Value`.
    if (forOp.isDefinedOutsideOfLoop(index) ||
        getConstantIntValue(index).has_value()) {
      step.emplace_back();
      continue;
    }

    // If the index is the induction var, then use the loop's step.
    // TODO: Make this more sophisticated.
    if (index == forOp.getInductionVar()) {
      step.push_back(forOp.getStep());
      continue;
    }
    LDBG("  Skip steps. Relation to induction variable not understood.");
    return buf;
  }
  buf.step = step;
  LLVM_DEBUG(DBGS() << "  step: ";
             llvm::interleaveComma(buf.step, llvm::dbgs());
             llvm::dbgs() << "\n");
  return buf;
}

Value maybeCast(Location loc, Value val, Type dstElemTy,
                PatternRewriter &rewriter) {
  VectorType srcTy = cast<VectorType>(val.getType());
  if (srcTy.getElementType() == dstElemTy)
    return val;

  VectorType dstTy = srcTy.cloneWith(std::nullopt, dstElemTy);
  if (srcTy.getElementType().isInteger()) {
    if (srcTy.getElementTypeBitWidth() < dstTy.getElementTypeBitWidth())
      return arith::ExtSIOp::create(rewriter, loc, dstTy, val);
    return arith::TruncIOp::create(rewriter, loc, dstTy, val);
  }

  if (srcTy.getElementTypeBitWidth() < dstTy.getElementTypeBitWidth())
    return arith::ExtFOp::create(rewriter, loc, dstTy, val,
                                 arith::FastMathFlagsAttr{});
  return arith::TruncFOp::create(rewriter, loc, dstTy, val);
}

MemBuffer allocateTmpBufferStack(Location loc, VectorType vecTy,
                                 Operation *allocaPoint,
                                 PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(allocaPoint);
  auto memRefTy = MemRefType::get(vecTy.getShape(), vecTy.getElementType());
  Value memRef = memref::AllocaOp::create(
      rewriter, loc, memRefTy,
      rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
  Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  SmallVector<Value> indices(2, zeroIdx);
  return {memRef, indices};
}

Value shiftIndex(Location loc, Value index, int64_t offs,
                 PatternRewriter &rewriter) {
  if (!offs)
    return index;

  // Do constant folding right away here for better code readability
  // after the pass.
  auto cstOp = dyn_cast<arith::ConstantOp>(index.getDefiningOp());
  if (cstOp) {
    int64_t oldVal = cast<IntegerAttr>(cstOp.getValue()).getInt();
    return arith::ConstantIndexOp::create(rewriter, loc, oldVal + offs);
  }

  Value offsVal = arith::ConstantIndexOp::create(rewriter, loc, offs);
  return arith::AddIOp::create(rewriter, loc, index.getType(), index, offsVal);
}

MemBuffer storeToTmpBuffer(Location loc, Value val, Operation *allocaPoint,
                           PatternRewriter &rewriter) {
  LDBG("Storing vector to a temporary buffer: " << val);
  auto vecTy = cast<VectorType>(val.getType());
  MemBuffer buf = allocateTmpBufferStack(loc, vecTy, allocaPoint, rewriter);
  op_write(val, buf.memRef, buf.indices);
  return buf;
}

} // namespace cpu
} // namespace triton
} // namespace mlir
