#ifndef TRITON_CPU_TRANSFORMS_ARM_NEON_BFMMLA_H
#define TRITON_CPU_TRANSFORMS_ARM_NEON_BFMMLA_H

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>

namespace mlir::triton::cpu::bfmmla {

enum class Form { MM, MMT };

struct Candidate {
  Form form;
  int64_t m;
  int64_t n;
  int64_t k;
};

// Inspect only the contraction's structure and masking context. No IR is
// modified and no host capabilities are queried. Unsupported contractions
// return failure with debug logging, without emitting user-facing diagnostics.
FailureOr<Candidate> matchCandidate(vector::ContractionOp op);

// SBGemm means matrix multiplication with BF16 inputs and FP32 outputs.
struct LowerVectorContractSBGemm
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override;

  // Emit and return the assembled result for an eligible MMT contraction.
  // Tests also exercise this directly to check packing before greedy folding.
  Value rewrite(vector::ContractionOp op, const Candidate &candidate,
                PatternRewriter &rewriter) const;
};

} // namespace mlir::triton::cpu::bfmmla

#endif // TRITON_CPU_TRANSFORMS_ARM_NEON_BFMMLA_H
