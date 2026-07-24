// RUN: triton-opt %s -split-input-file -triton-cpu-debug-ops-to-llvm | FileCheck %s

// tl.debug_barrier -> ttg.barrier is erased on CPU (no-op); guards the op-name
// fix that previously left it unlowered and broke MLIR->LLVM translation.

// CHECK-LABEL: @debug_barrier
// CHECK-NOT: ttg.barrier
// CHECK: tt.return
module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, "ttg.compute-capability" = 80} {
  tt.func @debug_barrier() {
    ttg.barrier local
    tt.return
  }
}
