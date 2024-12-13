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

#define DEBUG_TYPE "triton-cpu-dot-to-onednn"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {
// This struct describes buffers used to load/store AMX tiles.
struct AmxBuffer {
  Value memRef;
  SmallVector<Value, 2> indices;
  SmallVector<Value, 2> step;

  Value origTritonBlockPtr;

  bool empty() const { return !memRef; }
};

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
  // We have a limited number of available tiles, so if input/output is too
  // big to fit available tiles, we need to split them into blocks. Here we
  // keep a number of tiles in accumulator block. K dimension for input blocks
  // is always 1 tile now.
  int64_t tilesInBlockM;
  int64_t tilesInBlockN;
  // If accumulator is updated in a loop, then this flag indicates if we
  // should keep it in tiles the whole loop and move back to vectors only
  // after the loop.
  bool keepAccOnTiles = false;
  // If we want to keep accumulator in tiles but it's too big, then we might
  // keep it bufferized instead.
  bool keepAccInBuf = false;
  // If output buffer is used then keep the original vector store here.
  Operation *origStore = nullptr;

  // If resulting tiles are not required to be trasfered to vectors and can be
  // directly stored to the output memory instead, then this field holds a
  // buffer to use.
  AmxBuffer outBuf;
  AmxBuffer lhsBuf;
  AmxBuffer rhsBuf;
};

// Check if input and output types can be handled by AMX (possibly, using
// additional casts for input/output). Returns tru if AMX usage is possible.
// In this case, tile element type fields of the candidate structure are
// filled with actual types to be used in lowering.
bool checkElemTypes(Type lhsElemTy, Type rhsElemTy, Type accElemTy,
                    Type resElemTy, AmxDotOpCandidate &candidate) {
  MLIRContext *ctx = lhsElemTy.getContext();

  // This function indicates when VNNI granularity packing is expected by the
  // kernel.
  //
  // Note: used in benchdnn only, not used inside the library.
  // Note: for `bf32` (or brgattr.fpmath_mode_ == bf16) the function returns
  //   `true` because the data transformation to vnni layout is internal and
  //   transparent to the user.

  if (lhsElemTy.getIntOrFloatBitWidth() < 16 &&
      rhsElemTy.getIntOrFloatBitWidth() < 16) {
    LDBG("Packing will be applied for types smaller than 16 bits.");
    // Do something to insert transform and use it result
  }

  // For integer case only i8 is allowed for LHS and RHS.
  if (lhsElemTy.getIntOrFloatBitWidth() < 32 &&
      rhsElemTy.getIntOrFloatBitWidth() < 32) {
    LDBG("Packing can be applied for types smaller than 32 bits.");

    // 16 bit case
    if (!lhsElemTy.isInteger() && !rhsElemTy.isInteger()) {
      // Transform is conditional, maybe should wrapped with get_B_pack check
    } else {
      // Transform required
    }
  }
  // Transform is not required

  // Accumulator should be i32. If it's smaller, we will use casts.
  if (!accElemTy.isInteger() || accElemTy.getIntOrFloatBitWidth() > 32 ||
      !resElemTy.isInteger() || resElemTy.getIntOrFloatBitWidth() > 32) {
    LDBG("Drop candidate with unsupported output integer type.");
    return false;
  }

  candidate.lhsTileElemTy = IntegerType::get(ctx, 8);
  candidate.rhsTileElemTy = IntegerType::get(ctx, 8);
  candidate.accTileElemTy = IntegerType::get(ctx, 32);

  // Try to find common input type.
  Type commonInputElemTy = lhsElemTy;

  candidate.lhsTileElemTy = commonInputElemTy;
  candidate.rhsTileElemTy = commonInputElemTy;
  candidate.accTileElemTy = Float32Type::get(ctx);

  return true;

  if (lhsElemTy.getIntOrFloatBitWidth() == 16) {
    commonInputElemTy = lhsElemTy;
    if (rhsElemTy.getIntOrFloatBitWidth() == 16 &&
        rhsElemTy != commonInputElemTy) {
      LDBG("Drop candidate with mismatched input types.");
      return false;
    }
  } else if (rhsElemTy.getIntOrFloatBitWidth() == 16)
    commonInputElemTy = rhsElemTy;

  // Accumulator type should be FP32, we can use casts if it is smaller.
  if (accElemTy.getIntOrFloatBitWidth() > 32) {
    LDBG("Drop candidate with unsupported accumulator type.");
    return false;
  }

  candidate.lhsTileElemTy = commonInputElemTy;
  candidate.rhsTileElemTy = commonInputElemTy;
  candidate.accTileElemTy = Float32Type::get(ctx);

  return true;
}

// Check if accumulator value is updated in a loop and has no other
// usages than a dot op, that updates it. Tile loads/stores and casts
// for such accumulators can be done outside of the loop.
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

  // blockArg.getArgNumber();

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

// Return a value that holds the resulting loop carried accumulator value.
// It's one of ForOp results.
Value getResValueForLoopCarriedAcc(vector::ContractionOp op) {
  Value updAcc = op.getResult();
  auto forOp = dyn_cast<scf::ForOp>(op->getParentOp());
  auto &use = *updAcc.getUses().begin();
  return forOp.getResult(use.getOperandNumber());
}

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
void findOutputBuffer(Value val, AmxDotOpCandidate &candidate) {
  if (val.hasOneUse() && false) {
    auto store = dyn_cast<vector::TransferWriteOp>(*val.user_begin());
    if (store) // && !hasMaskOrBoundsCheck(store))
      candidate.outBuf = AmxBuffer{store.getSource(), store.getIndices()};
    candidate.origStore = store;
    LDBG("[findOutputBuffer] cand outbuf: " << candidate.outBuf.memRef);
  }
}

// Check if val is available in memory.
void findInputBuffer(Value val, bool allowTransposed, AmxBuffer &buf) {
  // if (allowTransposed) {
  //   auto transposeOp = val.getDefiningOp<vector::TransposeOp>();
  //   if (transposeOp) {
  //     val = transposeOp.getVector();
  //     buf.transposed = true;
  //   }
  // }

  auto valLoad = val.getDefiningOp<vector::TransferReadOp>();
  (void)hasMaskOrBoundsCheck(valLoad);
  if (!valLoad) {
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
  OpOperand *tritonPtr = forOp.getTiedLoopInit(blockPtrArg);
  if (!tritonPtr) {
    LDBG("  Skip triton ptr. No Tied block ptr.");
    return;
  }
  buf.origTritonBlockPtr = tritonPtr->get();
}

// Check if specified ContractionOp can be lowered to AMX operations.
// If conversion is possible, then true is returned and candidate
// structure is filled with detailed transformation info.
bool isAmxCandidate(triton::cpu::DotOp op, bool supportInt8, bool supportFp16,
                    bool supportBf16, AmxDotOpCandidate &candidate) {
  MLIRContext *ctx = op.getContext();
  VectorType lhsTy = cast<VectorType>(op.getA().getType());
  VectorType rhsTy = cast<VectorType>(op.getB().getType());
  VectorType accTy = cast<VectorType>(op.getC().getType());
  VectorType resTy = cast<VectorType>(op.getType());

  LDBG("Considering candidate op: " << op);

  candidate.lhsTileElemTy = lhsTy.getElementType();
  candidate.rhsTileElemTy = rhsTy.getElementType();
  candidate.accTileElemTy = accTy.getElementType();

  candidate.op = op;
  setupBlockAndTileSizes(lhsTy.getShape(), rhsTy.getShape(), candidate);
  candidate.keepAccOnTiles = isLoopCarriedAcc(op.getC());

  // if (lhsTy.getElementType() == candidate.lhsElemTy)
  findInputBuffer(op.getA(), true, candidate.lhsBuf);

  // if (rhsTy.getElementType() == candidate.rhsElemTy)
  findInputBuffer(op.getB(), false, candidate.rhsBuf);

  findOutputBuffer(op.getResult(), candidate);

  return true;
}

// Cast vector to a specified element type using ext or trunc
// operations. Return the originl value if it already matches
// required element type.
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

// Get initial value for a loop-carried accumulator.
Value getInitAccValue(Value val) {
  auto blockArg = cast<BlockArgument>(val);
  auto forOp = cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
  int initValIdx = blockArg.getArgNumber() - forOp.getNumInductionVars();
  return forOp.getInitArgs()[initValIdx];
}

VectorType getSwizzledRhsTileType(VectorType origTileType) {
  int64_t rowsPerGroup = 32 / origTileType.getElementTypeBitWidth();
  SmallVector<int64_t> shape({origTileType.getDimSize(0) / rowsPerGroup,
                              origTileType.getDimSize(1) * rowsPerGroup});
  return origTileType.cloneWith(shape, origTileType.getElementType());
}

AmxBuffer allocateTmpBuffer(Location loc, VectorType vecTy,
                            Operation *allocaPoint, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(allocaPoint);
  auto memRefTy = MemRefType::get(vecTy.getShape(), vecTy.getElementType());
  Value memRef = rewriter.create<memref::AllocOp>(
      loc, memRefTy, rewriter.getIntegerAttr(rewriter.getI64Type(), 64));
  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value, 2> indices(2, zeroIdx);
  return {memRef, indices};
}

AmxBuffer allocateTmpBuffer(Location loc, ArrayRef<Value> shape, Type elemType,
                            Operation *allocaPoint, PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(allocaPoint);
  SmallVector<int64_t, 3> memref_shape;
  SmallVector<Value, 3> memref_val;
  for (auto &sh : shape) {
    auto cst = sh.getDefiningOp<arith::ConstantOp>();
    if (cst) {
      llvm::errs() << cst.getValue() << "\n";
      memref_shape.emplace_back(cast<IntegerAttr>(cst.getValue()).getInt());
    } else {
      memref_shape.emplace_back(ShapedType::kDynamic);
      memref_val.emplace_back(sh);
    }
  }
  auto memRefTy = MemRefType::get(memref_shape, elemType);
  Value memRef = rewriter.create<memref::AllocOp>(loc, memRefTy, memref_val);
  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value, 2> indices(2, zeroIdx);
  return {memRef, indices};
}

// In AMX, element values shoud be packed to 32-bit groups that would be
// multiplied elementwise with following accumulation. It means that RHS
// needs to be pre-packed. E.g. for the following input
//   B(0,0) B(0,1) B(0,2) ... B(0,15)
//   B(1,0) B(1,1) B(1,2) ... B(1,15)
//   B(2,0) B(2,1) B(2,2) ... B(2,15)
//   B(3,0) B(3,1) B(3,2) ... B(3,15)
// and BF16/FP16 type we need to transform it to
//   B(0,0) B(1,0) B(0,1), B(1,1) ... B(0,15) B(1,15)
//   B(2,0) B(3,0) B(2,1), B(3,1) ... B(2,15) B(3,15)
// so that original columns are 32-bits now. In case of int8 type, the
// result would be:
//   B(0,0) B(1,0) B(2,0), B(3,0) ... B(0,15) B(1,15), B(2,15) B(3,15)
void transformIntoBuf(Location loc, Value transform_hash, Value inputBuf,
                      Value outBuf, PatternRewriter &rewriter) {
  // TODO add check that all is memref

  // K, N, in_pack, in_ld, out_ld, in_dt, out_dt
  // Value transform =
  rewriter.create<triton::cpu::TransformCall>(loc, transform_hash, inputBuf,
                                              outBuf);
}

AmxBuffer getMemrefFromVectorOperand(Value val) {
  auto valLoad = val.getDefiningOp<vector::TransferReadOp>();
  // some extra conditions required
  if (valLoad) {
    LDBG("Lhs should take src memref!\n");
    Value memRef = valLoad.getSource();
    ValueRange indices = valLoad.getIndices();

    LDBG("  Reusing the original memref for a buffer: " << memRef);
    return {memRef, indices};
  }
  assert(false && "Not Transfer Read");
}
// Prepare temporary buffers to be used for tile loads. If the original
// value can be directly loaded to tiles from its original memory, then
// use it instead. Return empty buffer if source value is all zeros and
// skipForZeros is set.
//
// If interleave flag is set, then pre-pack RHS before sotring. See
// interleaveAndStore for more details.
AmxBuffer prepareTensorBuffer(PatternRewriter &rewriter, Location loc,
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
      auto memRefTy = MemRefType::get(vecTy.getShape(), vecTy.getElementType());
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
    auto vecTy = cast<VectorType>(val.getType());
    auto transf_op = transform.getDefiningOp<triton::cpu::TransformCreate>();

    assert("Used( " && false);
    // AmxBuffer buf = allocateTmpBuffer(
    //     loc,
    //     {transf_op.getBlockK(), transf_op.getOutLd(), transf_op.getNCalls()},
    //     vecTy.getElementType(), allocaPoint, rewriter);

    // LDBG("  Reusing the original memref for a buffer: " << memRef);
    // transformIntoBuf(loc, transform, memRef, buf.memRef, rewriter);
    // return buf;
  }

  auto vecTy = cast<VectorType>(val.getType());
  AmxBuffer buf = allocateTmpBuffer(loc, vecTy, allocaPoint, rewriter);

  if (transform) {
    LDBG("Unhandled case for transform: " << val);
    assert(false);
    return {};
  }

  rewriter.create<vector::TransferWriteOp>(loc, val, buf.memRef, buf.indices);

  return buf;
}

// Return a buffer where the final result should be stored. If result can
// be directly stored to the output memory, then it is used as an output
// buffer. Otherwise, re-use accumulator buffer or create a new one.
AmxBuffer prepareResultBuffer(Location loc, Value val, const AmxBuffer &accBuf,
                              const AmxBuffer &outBuf, Operation *allocaPoint,
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
  return allocateTmpBuffer(loc, cast<VectorType>(val.getType()), allocaPoint,
                           rewriter);
}

void checkInputBtw(SmallVector<Value> &inps, PatternRewriter &rewriter) {
  for (auto i = 0; i < inps.size(); i++) {
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

  VectorType lhsTy = cast<VectorType>(candidate.op.getA().getType());
  VectorType rhsTy = cast<VectorType>(candidate.op.getB().getType());
  auto addSubView = [&](Value vecVal, ValueRange indices, Value memRef) {
    LDBG("  Reusing the original memref for a buffer: " << memRef);
    auto vecTy = cast<VectorType>(vecVal.getType());
    auto memRefTy = MemRefType::get(vecTy.getShape(), vecTy.getElementType());
    auto ctx = rewriter.getContext();
    // Value memRef_view = rewriter.create<memref::SubViewOp>(
    //     loc, memRefTy, memRef, {0, 0}, indices,
    //     metadata.getStrides());
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

LogicalResult
convertCandidate(AmxDotOpCandidate &candidate,
                 ModuleTensorPtrShapeInfoAnalysis &shapeInfoAnalysis,
                 PatternRewriter &rewriter) {
  triton::cpu::DotOp op = candidate.op;
  Location loc = op.getLoc();
  MLIRContext *ctx = op.getContext();

  IntegerType integer64 = IntegerType::get(rewriter.getContext(), 64);
  FloatType float32 = FloatType::getF32(rewriter.getContext());
  IndexType indexType = rewriter.getIndexType();

  scf::ForOp forOp = dyn_cast<scf::ForOp>(op->getParentOp());
  Value num_batches = rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), 1));
  if (candidate.keepAccOnTiles && forOp) {
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

  // If we don't work with a loop and want to directly store tiles into output
  // memory, then use the original store as insertion point to have its buffer
  // values available for generated code.
  if (!candidate.keepAccInBuf && !candidate.keepAccOnTiles &&
      !candidate.outBuf.empty())
    rewriter.setInsertionPoint(candidate.origStore);

  Operation *allocaPoint = op;
  while (!isa<triton::FuncOp>(allocaPoint->getParentOp()))
    allocaPoint = allocaPoint->getParentOp();

  ModuleOp module = op->template getParentOfType<ModuleOp>();

  auto getMemrefMetadata = [&](Value operand) {
    auto memrefType = dyn_cast<MemRefType>(operand.getType());
    assert(memrefType && "Expect a memref value");
    MemRefType baseMemrefType =
        MemRefType::get({}, memrefType.getElementType());
    SmallVector<Type> sizesTypes(memrefType.getRank(), indexType);
    SmallVector<Type> stridesTypes(memrefType.getRank(), indexType);
    return rewriter.create<memref::ExtractStridedMetadataOp>(
        loc, baseMemrefType, /*offsetType=*/indexType, sizesTypes, stridesTypes,
        operand);
  };

  auto metadataA = getMemrefMetadata(candidate.lhsBuf.memRef);
  auto metadataB = getMemrefMetadata(candidate.rhsBuf.memRef);

  // auto posMInA = 0; //*getPosInCodomain(posM, indexingMaps[0], ctx);
  // auto posNInB = 1; //*getPosInCodomain(posN, indexingMaps[1], ctx);
  // auto posKInA = 1; //*getPosInCodomain(posK, indexingMaps[0], ctx);
  // auto posKInB = 0; //*getPosInCodomain(posK, indexingMaps[1], ctx);

  // auto m = metadataA.getSizes()[posMInA];
  // auto n = metadataB.getSizes()[posNInB];
  // auto k = metadataA.getSizes()[posKInA];

  // auto posLeadingDimA = posMInA; // TODO: account for transposes...
  // auto lda_dyn = metadataA.getStrides()[posLeadingDimA];
  // auto posLeadingDimB = posKInB; // TODO: account for transposes...
  // auto ldb_dyn = m;              // metadataB.getStrides()[posLeadingDimB];
  // // auto posLeadingDimC = 0; // *getPosInCodomain( posM, indexingMaps[2],
  // ctx);
  // // TODO: account for transposes... auto ldc =
  // metadataC.getStrides()[posLeadingDimC];

  // M, N, K_k, batch_size, lda, ldb, ldc, dtypeA, dtypeB, dtypeC
  // they are in the same order with BrgemmDispatchOp inputs;
  // llvm::errs() << "Inps: M - " << candidate.tileM << ", N - " <<
  // candidate.tileN
  //              << ", K - " << candidate.tileK << " lhsType - " << lhsTy
  //              << " rhsType - " << rhsTy << "\n";

  llvm::errs() << "M N K - " << candidate.tileM << " " << candidate.tileN << " "
               << candidate.tileK << "\n";
  auto block_m = rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), candidate.tileM));
  auto block_n = rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), candidate.tileN));
  auto block_k = rewriter.create<arith::ConstantOp>(
      loc, integer64, IntegerAttr::get(rewriter.getI64Type(), candidate.tileK));

  auto block_k_ind = rewriter.create<arith::IndexCastOp>(
      loc, IndexType::get(rewriter.getContext()), block_k);

  // auto batch_size = rewriter.create<arith::DivUIOp>(loc, k, block_k_ind);

  auto lda = block_n;
  auto ldb = block_m;
  auto ldc = block_n;

  // dtypeA, dtypeB, dtypeC
  SmallVector<Attribute, 2> brgemmDtypes{
      TypeAttr::get(getElementTypeOrSelf(op.getA().getType())),
      TypeAttr::get(getElementTypeOrSelf(op.getB().getType())),
      TypeAttr::get(getElementTypeOrSelf(op.getD().getType()))};

  LDBG("ElemTypes: " << brgemmDtypes[0] << ", " << brgemmDtypes[1] << ", "
                     << brgemmDtypes[2]);

  // create dispatch op
  // auto flags = rewriter.getArrayAttr(brgemmFlags);
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

  auto memrefElemType =
      dyn_cast<MemRefType>(candidate.lhsBuf.memRef.getType()).getElementType();

  Value num_batches_ind = num_batches;
  if (!num_batches.getType().isIndex()) {
    num_batches_ind = rewriter.create<arith::IndexCastOp>(
        loc, IndexType::get(rewriter.getContext()), num_batches);
  }
  // Value k = rewriter.create<arith::MulIOp>(loc, block_k_ind,
  // num_batches_ind);

  Value brgemm = rewriter.create<triton::cpu::BrgemmCreate>(
      loc, rewriter.getIndexType(), block_m, block_n, block_k, num_batches_ind,
      lda, ldb, ldc, a_dt, b_dt, c_dt);

  // We will check if packing required and insert transform call if needed
  // const bool need_pack = brg.get_B_pack_type() == pack_type::pack32;

  auto in_ld = block_n;
  // auto ldb = operands[1];

  auto b_data_type_size = rewriter.create<arith::ConstantOp>(
      loc, integer64,
      IntegerAttr::get(rewriter.getI64Type(),
                       op.getB().getType().getElementTypeBitWidth() / 8));
  // auto ldb_ind = rewriter.create<arith::IndexCastOp>(
  //     loc, IndexType::get(rewriter.getContext()), ldb);
  auto b_data_type_size_ind = rewriter.create<arith::IndexCastOp>(
      loc, IndexType::get(rewriter.getContext()), b_data_type_size);

  Value block_k_ver = block_k;
  Value b_data_type_size_ver = b_data_type_size;
  if (!ldb.getType().isInteger()) {
    block_k_ver = block_k_ind;
    b_data_type_size_ver = b_data_type_size_ind;
  }
  auto blocked_B_size = rewriter.create<arith::MulIOp>(loc, ldb, block_k_ver);
  LDBG("blocked_B_size [1]: " << blocked_B_size);
  blocked_B_size =
      rewriter.create<arith::MulIOp>(loc, blocked_B_size, b_data_type_size_ver);
  LDBG("blocked_B_size [2]: " << blocked_B_size);

  Value transform = rewriter.create<triton::cpu::TransformCreate>(
      loc, rewriter.getIndexType(), block_k, block_n, in_ld, ldb, b_dt, b_dt);

  Value acc = maybeCast(loc, op.getC(), candidate.accTileElemTy, rewriter);
  Value accToStore = acc;

  LDBG("keepAcconBuff: " << candidate.keepAccInBuf
                         << " keppOnTiles: " << candidate.keepAccOnTiles);
  if (candidate.keepAccInBuf || candidate.keepAccOnTiles) {
    LDBG("Setting insertion op to forOp. (accToStore)");
    forOp = cast<scf::ForOp>(op->getParentOp());
    accToStore = getInitAccValue(acc);
  }

  AmxBuffer accBuf;
  {
    // If accumulator is bufferized then we should move initial values before
    // the loop.
    OpBuilder::InsertionGuard g(rewriter);
    if (candidate.keepAccOnTiles) {
      LDBG("Setting insertion op to forOp. (accBuf)");
      rewriter.setInsertionPoint(forOp);
    }
    accBuf =
        prepareTensorBuffer(rewriter, loc, accToStore, {}, false, allocaPoint);
  }

  AmxBuffer resBuf = prepareResultBuffer(
      loc, op.getResult(), accBuf, candidate.outBuf, allocaPoint, rewriter);

  LDBG("[prepareResultBuffer] prepared acc buf: " << accBuf.memRef);
  LDBG("[prepareResultBuffer] prepared res buf: " << resBuf.memRef);

  SmallVector<SmallVector<Value>> accTiles;
  if (candidate.keepAccOnTiles)
    assert("keepAcc is true, Move Loop req");

  // SmallVector<AffineMap> indexingMaps = op.getIndexingMapsArray();

  // auto loopRange = rewriter.create<arith::SubIOp>(loc, forOp.getUpperBound(),
  //                                                 forOp.getLowerBound());
  // Value num_batches =
  //     rewriter.create<arith::DivUIOp>(loc, loopRange, forOp.getStep());

  // auto num_batches = rewriter.create<arith::ConstantOp>(
  //     loc, indexType, rewriter.getIndexAttr(1));
  auto stepA = rewriter.create<arith::ConstantOp>(loc, indexType,
                                                  rewriter.getIndexAttr(0));
  auto stepB = rewriter.create<arith::ConstantOp>(loc, indexType,
                                                  rewriter.getIndexAttr(0));

  auto lhsSteps = candidate.lhsBuf.step;
  auto lhsMemrefType = dyn_cast<MemRefType>(candidate.lhsBuf.memRef.getType());
  auto lhsStrides = metadataA.getStrides();
  Value lhsAccSize = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value lhsDtSize = rewriter.create<arith::ConstantIndexOp>(
      loc, lhsMemrefType.getElementTypeBitWidth() / 8);
  for (int i = 0; i < metadataA.getStrides().size(); i++) {
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

  auto rhsSteps = candidate.rhsBuf.step;
  auto rhsStrides = metadataB.getStrides();
  auto rhsMemrefType = dyn_cast<MemRefType>(candidate.rhsBuf.memRef.getType());
  Value rhsAccSize = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value rhsDtSize = rewriter.create<arith::ConstantIndexOp>(
      loc, rhsMemrefType.getElementTypeBitWidth() / 8);
  for (int i = 0; i < metadataA.getStrides().size(); i++) {
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

  if (candidate.outBuf.empty()) {
    LDBG("Loading the result to a vector to replace orig op result.");
    Value newVal = rewriter.create<vector::TransferReadOp>(
        loc, cast<VectorType>(acc.getType()), resBuf.memRef, resBuf.indices);
    // We might need to cast back to the original type.
    newVal = maybeCast(loc, newVal, candidate.accTileElemTy, rewriter);
    // rewriter.replaceOp(op, newVal);
    // rewriter.eraseOp(*op.getResult().user_begin());
    // rewriter.eraseOp(op);
    LDBG("Printing module before replace:");
    LDBG(module);

    rewriter.replaceOp(forOp, ValueRange{newVal, candidate.lhsBuf.memRef,
                                         candidate.rhsBuf.memRef});
  } else {
    LDBG("Removing original operation and its use.");
    rewriter.eraseOp(*op.getResult().user_begin());
    rewriter.eraseOp(op);
  }

  LDBG(module);

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
      if (isAmxCandidate(op, convertInt8, convertFp16, convertBf16,
                         candidate)) {
        LLVM_DEBUG({
          LDBG("Found AMX candidate");
          LDBG("  Op: " << candidate.op);
          LDBG("  LhsTileElemTy: " << candidate.lhsTileElemTy);
          LDBG("  RhsTileElemTy: " << candidate.rhsTileElemTy);
          LDBG("  AccTileElemTy: " << candidate.accTileElemTy);
          LDBG("  TileM: " << candidate.tileM);
          LDBG("  TileN: " << candidate.tileN);
          LDBG("  TileK: " << candidate.tileK);
          LDBG("  TilesInBlockM: " << candidate.tilesInBlockM);
          LDBG("  TilesInBlockN: " << candidate.tilesInBlockN);
          LDBG("  KeepAccOnTiles: " << candidate.keepAccOnTiles);
          LDBG("  KeepAccInBuf: " << candidate.keepAccInBuf);
          LDBG("  Has output buffer: " << (bool)!candidate.outBuf.empty());
        });
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
