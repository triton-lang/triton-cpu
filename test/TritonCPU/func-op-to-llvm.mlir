// RUN: triton-opt %s -triton-cpu-func-op-to-llvm | FileCheck %s

module {
  tt.func public @kernel(%arg0: i32) attributes {noinline = false} {
    tt.call @middle() : () -> ()
    %0 = tt.call @external(%arg0) : (i32) -> i32
    tt.return
  }

  tt.func private @middle() attributes {noinline = true} {
    tt.call @leaf() : () -> ()
    tt.return
  }

  tt.func private @leaf() attributes {noinline = true} {
    tt.return
  }

  tt.func private @external(%arg0: i32) -> i32
}

// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME: %[[VALUE:[^:]+]]: i32,
// CHECK-SAME: %[[KPID0:[^:]+]]: i32, %[[KPID1:[^:]+]]: i32, %[[KPID2:[^:]+]]: i32,
// CHECK-SAME: %[[KNUM0:[^:]+]]: i32, %[[KNUM1:[^:]+]]: i32, %[[KNUM2:[^:]+]]: i32)
// CHECK: llvm.call @middle(%[[KPID0]], %[[KPID1]], %[[KPID2]], %[[KNUM0]], %[[KNUM1]], %[[KNUM2]])
// CHECK: llvm.call @external(%[[VALUE]]) : (i32) -> i32

// CHECK-LABEL: llvm.func @middle(
// CHECK-SAME: %[[MPID0:[^:]+]]: i32, %[[MPID1:[^:]+]]: i32, %[[MPID2:[^:]+]]: i32,
// CHECK-SAME: %[[MNUM0:[^:]+]]: i32, %[[MNUM1:[^:]+]]: i32, %[[MNUM2:[^:]+]]: i32)
// CHECK: llvm.call @leaf(%[[MPID0]], %[[MPID1]], %[[MPID2]], %[[MNUM0]], %[[MNUM1]], %[[MNUM2]])

// CHECK-LABEL: llvm.func @leaf(
// CHECK-SAME: %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32,
// CHECK-SAME: %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32)

// CHECK-LABEL: llvm.func @external(i32) -> i32
