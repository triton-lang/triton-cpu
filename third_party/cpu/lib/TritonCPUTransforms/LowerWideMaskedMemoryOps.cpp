#include "cpu/include/TritonCPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace triton {
namespace cpu {
#define GEN_PASS_DEF_LOWERWIDEMASKEDMEMORYOPS
#include "cpu/include/TritonCPUTransforms/Passes.h.inc"
} // namespace cpu
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::cpu;

namespace {

// Lanes per native register, or 0 when the vector is not a static 1-D vector
// spanning several registers.
int64_t getChunkSize(VectorType vecTy, unsigned nativeBitWidth) {
  if (vecTy.getRank() != 1 || vecTy.isScalable() ||
      !vecTy.getElementType().isIntOrFloat())
    return 0;
  unsigned elemBits = vecTy.getElementTypeBitWidth();
  if (elemBits == 0 || nativeBitWidth % elemBits != 0)
    return 0;
  int64_t chunk = nativeBitWidth / elemBits;
  int64_t width = vecTy.getDimSize(0);
  if (width <= chunk || width % chunk != 0)
    return 0;
  return chunk;
}

// The op may sit in a loop, so put the buffer in the function entry block
// where it is allocated once per call. memref.alloca_scope would not work:
// SCF-to-CF later turns the chunk loop into blocks it cannot hold.
Value createEntryAlloca(OpBuilder &b, Location loc, Operation *op,
                        MemRefType bufTy) {
  auto func = op->getParentOfType<FunctionOpInterface>();
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(&func.getFunctionBody().front());
  return memref::AllocaOp::create(b, loc, bufTy);
}

// The full-width access to this buffer sits in a block that mirrors the
// unmasked fast path on the base. Greedy drivers that run while memrefs are
// still unlowered (convert-vector-to-llvm) merge such blocks and pass the
// memref to an llvm.cond_br as a block argument, which does not verify. A rank
// different from the base keeps the two accesses distinct.
Value createDataAlloca(OpBuilder &b, Location loc, Operation *op,
                       VectorType vecTy, Value base) {
  SmallVector<int64_t> shape;
  if (cast<MemRefType>(base.getType()).getRank() == 1)
    shape.push_back(1);
  shape.push_back(vecTy.getDimSize(0));
  return createEntryAlloca(b, loc, op,
                           MemRefType::get(shape, vecTy.getElementType()));
}

SmallVector<Value> bufIndices(Value buf, Value zero, Value iv) {
  SmallVector<Value> res(cast<MemRefType>(buf.getType()).getRank() - 1, zero);
  res.push_back(iv);
  return res;
}

// The flat form ConvertMemoryOps emits: a 1-D index vector covering the same
// lanes as the data vector. Returns the chunk width, or 0 if not applicable.
int64_t getWideIndexChunk(VectorType dataTy, VectorType indexTy,
                          unsigned nativeBitWidth) {
  int64_t chunk = getChunkSize(dataTy, nativeBitWidth);
  if (!chunk || indexTy.getRank() != 1 ||
      indexTy.getDimSize(0) != dataTy.getDimSize(0) ||
      !indexTy.getElementType().isInteger())
    return 0;
  return chunk;
}

using ChunkBodyFn = function_ref<void(OpBuilder &b, Location loc, Value iv,
                                      Value chunkMask, Value chunkIndices)>;

// Loop over `mask` in native-width chunks, calling `body` with each chunk's
// lane offset and mask. The mask and the optional gather/scatter `indices` are
// vector values that a dynamic chunk cannot be sliced out of, so they are
// spilled to entry-block buffers once and reloaded one register per iteration.
// i1 vectors are bit-packed in memory while memref<i1> is byte-addressed, so
// the mask is spilled as bytes.
void emitChunkLoop(OpBuilder &b, Location loc, Operation *op, Value zero,
                   int64_t chunk, Value mask, Value indices, ChunkBodyFn body) {
  auto maskTy = cast<VectorType>(mask.getType());
  auto bytesTy = maskTy.clone(b.getI8Type());
  Value maskBuf = createEntryAlloca(
      b, loc, op, MemRefType::get(maskTy.getShape(), b.getI8Type()));
  Value indexBuf;
  if (indices) {
    auto indexTy = cast<VectorType>(indices.getType());
    indexBuf = createEntryAlloca(
        b, loc, op,
        MemRefType::get(indexTy.getShape(), indexTy.getElementType()));
    vector::StoreOp::create(b, loc, indices, indexBuf, ValueRange{zero});
  }
  Value bytes = arith::ExtUIOp::create(b, loc, bytesTy, mask);
  vector::StoreOp::create(b, loc, bytes, maskBuf, ValueRange{zero});

  Value width = arith::ConstantIndexOp::create(b, loc, maskTy.getDimSize(0));
  Value step = arith::ConstantIndexOp::create(b, loc, chunk);
  scf::ForOp::create(
      b, loc, zero, width, step, ValueRange{},
      [&](OpBuilder &b, Location loc, Value iv, ValueRange) {
        auto chunkBytesTy = VectorType::get({chunk}, b.getI8Type());
        Value chunkBytes =
            vector::LoadOp::create(b, loc, chunkBytesTy, maskBuf, iv);
        Value zeros =
            arith::ConstantOp::create(b, loc, b.getZeroAttr(chunkBytesTy));
        Value chunkMask = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::ne, chunkBytes, zeros);
        Value chunkIndices;
        if (indexBuf) {
          auto chunkIndexTy = VectorType::get(
              {chunk}, cast<VectorType>(indices.getType()).getElementType());
          chunkIndices =
              vector::LoadOp::create(b, loc, chunkIndexTy, indexBuf, iv);
        }
        body(b, loc, iv, chunkMask, chunkIndices);
        scf::YieldOp::create(b, loc);
      });
}

SmallVector<Value> offsetLastIndex(OpBuilder &b, Location loc,
                                   ValueRange indices, Value iv) {
  SmallVector<Value> res(indices);
  res.back() = arith::AddIOp::create(b, loc, res.back(), iv);
  return res;
}

// Rewrite a wide masked load as
//   if (all lanes active) plain vector load
//   else loop over native-width chunks doing small masked loads
// LLVM scalarizes llvm.masked.load on targets without masked loads, which
// for a vector<8192xf32> is 8192 conditional blocks and heavy spilling.
struct LoopWideMaskedLoad : public OpRewritePattern<vector::MaskedLoadOp> {
  LoopWideMaskedLoad(MLIRContext *context, unsigned nativeBitWidth)
      : OpRewritePattern(context), nativeBitWidth(nativeBitWidth) {}

  LogicalResult matchAndRewrite(vector::MaskedLoadOp op,
                                PatternRewriter &rewriter) const override {
    VectorType vecTy = op.getVectorType();
    int64_t chunk = getChunkSize(vecTy, nativeBitWidth);
    if (!chunk || op.getIndices().empty() ||
        !op->getParentOfType<FunctionOpInterface>())
      return failure();

    Location loc = op.getLoc();
    Value base = op.getBase();
    SmallVector<Value> indices(op.getIndices());
    auto chunkTy = VectorType::get({chunk}, vecTy.getElementType());
    Value buf = createDataAlloca(rewriter, loc, op, vecTy, base);

    Value allActive = vector::ReductionOp::create(
        rewriter, loc, vector::CombiningKind::AND, op.getMask());
    auto ifOp = scf::IfOp::create(
        rewriter, loc, allActive,
        [&](OpBuilder &b, Location loc) {
          Value vec = vector::LoadOp::create(b, loc, vecTy, base, indices);
          scf::YieldOp::create(b, loc, vec);
        },
        [&](OpBuilder &b, Location loc) {
          Value zero = arith::ConstantIndexOp::create(b, loc, 0);
          vector::StoreOp::create(b, loc, op.getPassThru(), buf,
                                  bufIndices(buf, zero, zero));
          emitChunkLoop(b, loc, op, zero, chunk, op.getMask(), Value(),
                        [&](OpBuilder &b, Location loc, Value iv,
                            Value chunkMask, Value) {
                          Value chunkPassThru = vector::LoadOp::create(
                              b, loc, chunkTy, buf, bufIndices(buf, zero, iv));
                          Value vec = vector::MaskedLoadOp::create(
                              b, loc, chunkTy, base,
                              offsetLastIndex(b, loc, indices, iv), chunkMask,
                              chunkPassThru);
                          vector::StoreOp::create(b, loc, vec, buf,
                                                  bufIndices(buf, zero, iv));
                        });
          Value vec = vector::LoadOp::create(b, loc, vecTy, buf,
                                             bufIndices(buf, zero, zero));
          scf::YieldOp::create(b, loc, vec);
        });
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }

  unsigned nativeBitWidth;
};

// Same as LoopWideMaskedLoad, for stores.
struct LoopWideMaskedStore : public OpRewritePattern<vector::MaskedStoreOp> {
  LoopWideMaskedStore(MLIRContext *context, unsigned nativeBitWidth)
      : OpRewritePattern(context), nativeBitWidth(nativeBitWidth) {}

  LogicalResult matchAndRewrite(vector::MaskedStoreOp op,
                                PatternRewriter &rewriter) const override {
    Value value = op.getValueToStore();
    auto vecTy = cast<VectorType>(value.getType());
    int64_t chunk = getChunkSize(vecTy, nativeBitWidth);
    if (!chunk || op.getIndices().empty() ||
        !op->getParentOfType<FunctionOpInterface>())
      return failure();

    Location loc = op.getLoc();
    Value base = op.getBase();
    SmallVector<Value> indices(op.getIndices());
    auto chunkTy = VectorType::get({chunk}, vecTy.getElementType());
    Value buf = createDataAlloca(rewriter, loc, op, vecTy, base);

    Value allActive = vector::ReductionOp::create(
        rewriter, loc, vector::CombiningKind::AND, op.getMask());
    scf::IfOp::create(
        rewriter, loc, allActive,
        [&](OpBuilder &b, Location loc) {
          vector::StoreOp::create(b, loc, value, base, indices);
          scf::YieldOp::create(b, loc);
        },
        [&](OpBuilder &b, Location loc) {
          Value zero = arith::ConstantIndexOp::create(b, loc, 0);
          vector::StoreOp::create(b, loc, value, buf,
                                  bufIndices(buf, zero, zero));
          emitChunkLoop(b, loc, op, zero, chunk, op.getMask(), Value(),
                        [&](OpBuilder &b, Location loc, Value iv,
                            Value chunkMask, Value) {
                          Value chunkVal = vector::LoadOp::create(
                              b, loc, chunkTy, buf, bufIndices(buf, zero, iv));
                          vector::MaskedStoreOp::create(
                              b, loc, base,
                              offsetLastIndex(b, loc, indices, iv), chunkMask,
                              chunkVal);
                        });
          scf::YieldOp::create(b, loc);
        });
    rewriter.eraseOp(op);
    return success();
  }

  unsigned nativeBitWidth;
};

// LLVM expands llvm.masked.gather/scatter into one conditional access per
// lane, so a wide gather/scatter has the same scalarization problem as a wide
// masked load. There is no useful all-active fast path, so always loop.
struct LoopWideGather : public OpRewritePattern<vector::GatherOp> {
  LoopWideGather(MLIRContext *context, unsigned nativeBitWidth)
      : OpRewritePattern(context), nativeBitWidth(nativeBitWidth) {}

  LogicalResult matchAndRewrite(vector::GatherOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<MemRefType>(op.getBase().getType()))
      return failure();
    VectorType vecTy = op.getVectorType();
    int64_t chunk =
        getWideIndexChunk(vecTy, op.getIndexVectorType(), nativeBitWidth);
    if (!chunk || !op->getParentOfType<FunctionOpInterface>())
      return failure();

    Location loc = op.getLoc();
    auto chunkTy = VectorType::get({chunk}, vecTy.getElementType());
    Value buf = createDataAlloca(rewriter, loc, op, vecTy, op.getBase());
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    vector::StoreOp::create(rewriter, loc, op.getPassThru(), buf,
                            bufIndices(buf, zero, zero));
    emitChunkLoop(
        rewriter, loc, op, zero, chunk, op.getMask(), op.getIndices(),
        [&](OpBuilder &b, Location loc, Value iv, Value chunkMask,
            Value chunkIndices) {
          Value chunkPass = vector::LoadOp::create(b, loc, chunkTy, buf,
                                                   bufIndices(buf, zero, iv));
          Value vec = vector::GatherOp::create(b, loc, chunkTy, op.getBase(),
                                               op.getOffsets(), chunkIndices,
                                               chunkMask, chunkPass);
          vector::StoreOp::create(b, loc, vec, buf, bufIndices(buf, zero, iv));
        });
    rewriter.replaceOp(op, vector::LoadOp::create(rewriter, loc, vecTy, buf,
                                                  bufIndices(buf, zero, zero)));
    return success();
  }

  unsigned nativeBitWidth;
};

// Same as LoopWideGather, for scatter.
struct LoopWideScatter : public OpRewritePattern<vector::ScatterOp> {
  LoopWideScatter(MLIRContext *context, unsigned nativeBitWidth)
      : OpRewritePattern(context), nativeBitWidth(nativeBitWidth) {}

  LogicalResult matchAndRewrite(vector::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (!isa<MemRefType>(op.getBase().getType()))
      return failure();
    VectorType vecTy = op.getVectorType();
    int64_t chunk =
        getWideIndexChunk(vecTy, op.getIndexVectorType(), nativeBitWidth);
    if (!chunk || !op->getParentOfType<FunctionOpInterface>())
      return failure();

    Location loc = op.getLoc();
    auto chunkTy = VectorType::get({chunk}, vecTy.getElementType());
    Value buf = createDataAlloca(rewriter, loc, op, vecTy, op.getBase());
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    vector::StoreOp::create(rewriter, loc, op.getValueToStore(), buf,
                            bufIndices(buf, zero, zero));
    emitChunkLoop(rewriter, loc, op, zero, chunk, op.getMask(), op.getIndices(),
                  [&](OpBuilder &b, Location loc, Value iv, Value chunkMask,
                      Value chunkIndices) {
                    Value chunkVal = vector::LoadOp::create(
                        b, loc, chunkTy, buf, bufIndices(buf, zero, iv));
                    vector::ScatterOp::create(
                        b, loc, TypeRange{}, op.getBase(), op.getOffsets(),
                        chunkIndices, chunkMask, chunkVal, IntegerAttr{});
                  });
    rewriter.eraseOp(op);
    return success();
  }

  unsigned nativeBitWidth;
};

struct LowerWideMaskedMemoryOps
    : public triton::cpu::impl::LowerWideMaskedMemoryOpsBase<
          LowerWideMaskedMemoryOps> {
  LowerWideMaskedMemoryOps() = default;

  LowerWideMaskedMemoryOps(unsigned nativeVectorBitWidth) {
    this->nativeVectorBitWidth = nativeVectorBitWidth;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    RewritePatternSet patterns(context);
    patterns.add<LoopWideMaskedLoad, LoopWideMaskedStore, LoopWideGather,
                 LoopWideScatter>(context, nativeVectorBitWidth);
    if (failed(mlir::applyPatternsGreedily(mod, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir {
namespace triton {
namespace cpu {

std::unique_ptr<OperationPass<ModuleOp>>
createLowerWideMaskedMemoryOps(unsigned nativeVectorBitWidth) {
  return std::make_unique<LowerWideMaskedMemoryOps>(nativeVectorBitWidth);
}

} // namespace cpu
} // namespace triton
} // namespace mlir
