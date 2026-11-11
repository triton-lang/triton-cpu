// RUN: triton-opt %s -split-input-file -triton-cpu-convert-reduction -canonicalize
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-reduction="use-reduction-op=true use-multidim-reduction-op=true" | FileCheck %s

// Regression test: Check that we handle consecutive calls to tt.reduce with
// different types & number of arguments.

module {
  tt.func public @triton_(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xi32>) {
    %0 = "tt.reduce"(%arg0) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: f32):
      tt.reduce.return %arg3 : f32
    }) : (tensor<1x4xf32>) -> tensor<1xf32>
    %1:2 = "tt.reduce"(%arg0, %arg1) <{axis = 1 : i32}> ({
    ^bb0(%arg3: f32, %arg4: i32, %arg5: f32, %arg6: i32):
      tt.reduce.return %arg3, %arg4 : f32, i32
    }) : (tensor<1x4xf32>, tensor<1x4xi32>) -> (tensor<1xf32>, tensor<1xi32>)
    tt.return
  }
}

// -----

module {
  // CHECK-LABEL: @integer_initializers
  // CHECK: %[[SMIN:.*]] = arith.constant -9223372036854775808 : i64
  // CHECK: vector.reduction <maxsi>, {{.*}}, %[[SMIN]]
  // CHECK: %[[SMAX:.*]] = arith.constant 9223372036854775807 : i64
  // CHECK: vector.reduction <minsi>, {{.*}}, %[[SMAX]]
  // CHECK: %[[UMAX:.*]] = arith.constant -1 : i64
  // CHECK: vector.reduction <minui>, {{.*}}, %[[UMAX]]
  tt.func public @integer_initializers(%input: tensor<4xi64>) -> (i64, i64, i64) {
    %max = "tt.reduce"(%input) <{axis = 0 : i32}> ({
    ^bb0(%lhs: i64, %rhs: i64):
      %value = arith.maxsi %lhs, %rhs : i64
      tt.reduce.return %value : i64
    }) : (tensor<4xi64>) -> i64
    %min = "tt.reduce"(%input) <{axis = 0 : i32}> ({
    ^bb0(%lhs: i64, %rhs: i64):
      %value = arith.minsi %lhs, %rhs : i64
      tt.reduce.return %value : i64
    }) : (tensor<4xi64>) -> i64
    %min_unsigned = "tt.reduce"(%input) <{axis = 0 : i32}> ({
    ^bb0(%lhs: i64, %rhs: i64):
      %value = arith.minui %lhs, %rhs : i64
      tt.reduce.return %value : i64
    }) : (tensor<4xi64>) -> i64
    tt.return %max, %min, %min_unsigned : i64, i64, i64
  }
}

// -----

module {
  // CHECK-LABEL: @floating_initializers
  // CHECK: %[[PINF:.*]] = arith.constant 0x7C00 : f16
  // CHECK: vector.reduction <minnumf>, {{.*}}, %[[PINF]]
  // CHECK: %[[NINF:.*]] = arith.constant 0xFF800000 : f32
  // CHECK: vector.reduction <maxnumf>, {{.*}}, %[[NINF]]
  // CHECK: %[[NAN:.*]] = arith.constant 0x7FF8000000000000 : f64
  // CHECK: vector.reduction <minimumf>, {{.*}}, %[[NAN]]
  // CHECK: %[[NAN_SPLAT:.*]] = arith.constant dense<0x7FC0> : vector<2xbf16>
  // CHECK: vector.multi_reduction <maximumf>, {{.*}}, %[[NAN_SPLAT]]
  tt.func public @floating_initializers(%half: tensor<4xf16>, %single: tensor<4xf32>, %double: tensor<4xf64>, %brain: tensor<2x4xbf16>) -> (f16, f32, f64, tensor<2xbf16>) {
    %min_num = "tt.reduce"(%half) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f16, %rhs: f16):
      %value = arith.minnumf %lhs, %rhs : f16
      tt.reduce.return %value : f16
    }) : (tensor<4xf16>) -> f16
    %max_num = "tt.reduce"(%single) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f32, %rhs: f32):
      %value = arith.maxnumf %lhs, %rhs : f32
      tt.reduce.return %value : f32
    }) : (tensor<4xf32>) -> f32
    %min = "tt.reduce"(%double) <{axis = 0 : i32}> ({
    ^bb0(%lhs: f64, %rhs: f64):
      %value = arith.minimumf %lhs, %rhs : f64
      tt.reduce.return %value : f64
    }) : (tensor<4xf64>) -> f64
    %max = "tt.reduce"(%brain) <{axis = 1 : i32}> ({
    ^bb0(%lhs: bf16, %rhs: bf16):
      %value = arith.maximumf %lhs, %rhs : bf16
      tt.reduce.return %value : bf16
    }) : (tensor<2x4xbf16>) -> tensor<2xbf16>
    tt.return %min_num, %max_num, %min, %max : f16, f32, f64, tensor<2xbf16>
  }
}
