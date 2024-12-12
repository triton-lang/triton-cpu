#include "cpu/include/TritonCPUTransforms/OptCommon.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonCPU/IR/Dialect.h"

#define DEBUG_TYPE "triton-cpu-dot-conversion"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace triton {
namespace cpu {

// This structure describes input/output buffer.
struct MemBuffer {
  Value memRef;
  SmallVector<Value> indices;
  // If buffer is accessed in a loop and indices are advanced
  // on each iteration, then step can hold those index offsets.
  // Empty step doesn't mean indices are loop invariant.
  SmallVector<Value> step;
  // True if buffer holds transposed value.
  bool transposed = false;

  bool empty() const { return !memRef; }
};

// Check if accumulator value is updated in a loop and has no other
// usages than a dot op that updates it. Loads, stores, and casts
// for such accumulator can be done outside of the loop.
bool isLoopCarriedAcc(Value acc);

// Get initial value for a loop-carried accumulator.
Value getInitAccValue(Value val);

// Check if vector transfer read/write operation uses a mask
// or involves a bounds check.
template <typename T> bool hasMaskOrBoundsCheck(T op) {
  auto inBounds = op.getInBounds();
  Value mask = op.getMask();
  bool hasBoundsCheck =
      std::any_of(inBounds.begin(), inBounds.end(), [](Attribute attr) {
        return !cast<mlir::BoolAttr>(attr).getValue();
      });
  return hasBoundsCheck || mask;
}

// Search for a buffer holding required value. If allowTransposed is true,
// then buffer is allowed to hold both transposed and not transposed value.
// Return empty buffer if no memory holding value was found.
MemBuffer findInputBuffer(Value val, bool allowTransposed = false);

// Cast vector to a specified element type using ext or trunc
// operations. Return the original value if it already matches
// the required element type.
Value maybeCast(Location loc, Value val, Type dstElemTy,
                PatternRewriter &rewriter);

// Allocate temporary buffer on stack for specified vector type.
MemBuffer allocateTmpBuffer(Location loc, VectorType vecTy,
                            Operation *allocaPoint, PatternRewriter &rewriter);

// Move index by specified offset. Do constannt folding if possible.
Value shiftIndex(Location loc, Value index, int64_t offs,
                 PatternRewriter &rewriter);

} // namespace cpu
} // namespace triton
} // namespace mlir
