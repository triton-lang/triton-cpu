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
Value getVnniSrcImpl(Value val) {
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

} // namespace

Value getVnniSrc(Value val) {
  Type elemTy = getElementTypeOrSelf(val.getType());

  // VNNI encoding is used for 8-bit and 16-bit values only.
  if (elemTy.getIntOrFloatBitWidth() > 16)
    return nullptr;

  // For 16-bit values VNNI encoding is a single interleave of
  // subsequenct rows. For 8-bit values, it's applied twice.
  Value encoded = getVnniSrcImpl(val);
  if (encoded && elemTy.getIntOrFloatBitWidth() == 8)
    encoded = getVnniSrcImpl(encoded);

  return encoded;
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

  buf.memRef = valLoad.getSource();
  buf.indices = valLoad.getIndices();
  LLVM_DEBUG(
      DBGS() << "Found buffer with input: " << val << "\n";
      DBGS() << "  MemRef: " << buf.memRef << "\n"; DBGS() << "  Indices: ";
      llvm::interleaveComma(buf.indices, llvm::dbgs()); llvm::dbgs() << "\n");

  auto forOp = dyn_cast<scf::ForOp>(valLoad->getParentOp());
  if (!forOp) {
    LDBG("  Skip steps. Not in a for-loop.");
    return buf;
  }

  auto extractMemRef = buf.memRef.getDefiningOp<ExtractMemRefOp>();
  if (!extractMemRef) {
    LDBG("  Skip steps. No ExtractMemRefOp.");
    return buf;
  }

  ExtractIndicesOp extractIndices;
  for (auto index : buf.indices) {
    auto def = index.getDefiningOp<ExtractIndicesOp>();
    if (!def || (extractIndices && def != extractIndices)) {
      LDBG("  Skip steps. No ExtractIndicesOp.");
      return buf;
    }
    extractIndices = def;
  }

  if (extractMemRef.getSrc() != extractIndices.getSrc()) {
    LDBG("  Skip steps. Mismatched ExtractMemRefOp and ExtractIndicesOp.");
    return buf;
  }

  BlockArgument blockPtrArg = dyn_cast<BlockArgument>(extractMemRef.getSrc());
  if (!blockPtrArg) {
    LDBG("  Skip steps. No block pointer arg.");
    return buf;
  }

  OpOperand *yieldOp = forOp.getTiedLoopYieldedValue(blockPtrArg);
  if (!yieldOp) {
    LDBG("  Skip steps. No block pointer in yield.");
    return buf;
  }

  auto advance = yieldOp->get().getDefiningOp<AdvanceOp>();
  if (!advance) {
    LDBG("  Skip steps. No AdvanceOp.");
    return buf;
  }

  if (advance.getPtr() != blockPtrArg) {
    LDBG("  Skip steps. AdvanceOp doesn't use block pointer arg.");
    return buf;
  }

  buf.step = advance.getOffsets();
  LLVM_DEBUG(DBGS() << "  Step: ";
             llvm::interleaveComma(buf.step, llvm::dbgs());
             llvm::dbgs() << "\n");
  buf.origBlockPtr = forOp.getTiedLoopInit(blockPtrArg)->get();

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
      return rewriter.create<arith::ExtSIOp>(loc, dstTy, val);
    return rewriter.create<arith::TruncIOp>(loc, dstTy, val);
  }

  if (srcTy.getElementTypeBitWidth() < dstTy.getElementTypeBitWidth())
    return rewriter.create<arith::ExtFOp>(loc, dstTy, val);
  return rewriter.create<arith::TruncFOp>(loc, dstTy, val);
}

MemBuffer allocateTmpBufferStack(Location loc, VectorType vecTy,
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

Value shiftIndex(Location loc, Value index, int64_t offs,
                 PatternRewriter &rewriter) {
  if (!offs)
    return index;

  // Do constant folding right away here for better code readability
  // after the pass.
  auto cstOp = dyn_cast<arith::ConstantOp>(index.getDefiningOp());
  if (cstOp) {
    int64_t oldVal = cast<IntegerAttr>(cstOp.getValue()).getInt();
    return rewriter.create<arith::ConstantIndexOp>(loc, oldVal + offs);
  }

  Value offsVal = rewriter.create<arith::ConstantIndexOp>(loc, offs);
  return rewriter.create<arith::AddIOp>(loc, index.getType(), index, offsVal);
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
