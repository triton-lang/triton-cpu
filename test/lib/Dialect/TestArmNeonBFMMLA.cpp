#include "cpu/lib/TritonCPUTransforms/ArmNeonBFMMLA.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct TestArmNeonBFMMLAMatcherPass
    : public PassWrapper<TestArmNeonBFMMLAMatcherPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestArmNeonBFMMLAMatcherPass);

  StringRef getArgument() const final { return "test-arm-neon-bfmmla-matcher"; }
  StringRef getDescription() const final {
    return "test ArmNeon BFMMLA eligibility without rewriting IR";
  }

  void runOnOperation() override {
    namespace bfmmla = triton::cpu::bfmmla;
    getOperation().walk([&](vector::ContractionOp op) {
      auto candidate = bfmmla::matchCandidate(op);
      llvm::outs() << op.getLoc() << ": ";
      if (failed(candidate)) {
        llvm::outs() << "rejected\n";
        return;
      }
      llvm::outs() << (candidate->form == bfmmla::Form::MM ? "MM" : "MMT")
                   << " M=" << candidate->m << " N=" << candidate->n
                   << " K=" << candidate->k << '\n';
    });
    markAllAnalysesPreserved();
  }
};

struct TestArmNeonBFMMLAEmitterPass
    : public PassWrapper<TestArmNeonBFMMLAEmitterPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestArmNeonBFMMLAEmitterPass);

  StringRef getArgument() const final { return "test-arm-neon-bfmmla-emitter"; }
  StringRef getDescription() const final {
    return "test ArmNeon BFMMLA emission without greedy folding";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arm_neon::ArmNeonDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    namespace bfmmla = triton::cpu::bfmmla;
    PatternRewriter rewriter(&getContext());
    bfmmla::LowerVectorContractSBGemm pattern(&getContext());
    getOperation().walk([&](vector::ContractionOp op) {
      auto candidate = bfmmla::matchCandidate(op);
      if (failed(candidate) || candidate->form != bfmmla::Form::MMT)
        return;

      // Exercise the emitter directly so greedy rewrites cannot hide packing
      // masks. The emitter may still eagerly fold trivial tile assembly.
      rewriter.setInsertionPoint(op);
      rewriter.replaceOp(op, pattern.rewrite(op, *candidate, rewriter));
    });
  }
};

} // namespace

namespace mlir::test {
void registerTestArmNeonBFMMLAMatcherPass() {
  PassRegistration<TestArmNeonBFMMLAMatcherPass>();
  PassRegistration<TestArmNeonBFMMLAEmitterPass>();
}
} // namespace mlir::test
