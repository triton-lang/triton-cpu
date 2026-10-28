#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::cpu {
#define GEN_PASS_DEF_LOWERSVE
#include "cpu/include/TritonCPUToLLVM/Passes.h.inc"
} // namespace mlir::triton::cpu

using namespace mlir;

namespace {

struct LowerSVE : public mlir::triton::cpu::impl::LowerSVEBase<LowerSVE> {
  using LowerSVEBase::LowerSVEBase;

  void runOnOperation() override {}
};

} // namespace

namespace mlir::triton::cpu {

std::unique_ptr<OperationPass<triton::FuncOp>> createLowerSVEPass() {
  return std::make_unique<LowerSVE>();
}

} // namespace mlir::triton::cpu
