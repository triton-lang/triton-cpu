// RUN: triton-opt %s --triton-cpu-lower-sve | FileCheck %s

module {
  // CHECK-LABEL: tt.func @outer_add_reduction(
  // CHECK-NOT: vector.multi_reduction
  // CHECK: vector.vscale
  // CHECK: scf.for {{.*}} iter_args(%[[ACC0:.*]] = {{.*}}, %[[ACC1:.*]] = {{.*}}) -> (f32, f32) {
  // CHECK: %[[LOADED:.*]] = vector.maskedload {{.*}} : memref<74xf32>, vector<[8]xi1>, vector<[8]xf32>
  // CHECK-NEXT: %[[EVEN:.*]], %[[ODD:.*]] = vector.deinterleave %[[LOADED]] : vector<[8]xf32> -> vector<[4]xf32>
  // CHECK-NEXT: %[[SUM0:.*]] = vector.reduction <add>, %[[EVEN]], %[[ACC0]] : vector<[4]xf32> into f32
  // CHECK-NEXT: %[[SUM1:.*]] = vector.reduction <add>, %[[ODD]], %[[ACC1]] : vector<[4]xf32> into f32
  // CHECK-NEXT: scf.yield %[[SUM0]], %[[SUM1]] : f32, f32
  tt.func @outer_add_reduction(%arg: vector<37x2xf32>) -> vector<2xf32> {
    %zero = arith.constant dense<0.0> : vector<2xf32>
    %result = vector.multi_reduction <add>, %arg, %zero [0] : vector<37x2xf32> to vector<2xf32>
    tt.return %result : vector<2xf32>
  }

  // CHECK-LABEL: tt.func @unsupported_reduction(
  // CHECK: vector.multi_reduction
  tt.func @unsupported_reduction(%arg: vector<37x3xf32>) -> vector<3xf32> {
    %zero = arith.constant dense<0.0> : vector<3xf32>
    %result = vector.multi_reduction <add>, %arg, %zero [0] : vector<37x3xf32> to vector<3xf32>
    tt.return %result : vector<3xf32>
  }
}
