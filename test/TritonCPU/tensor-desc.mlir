// RUN: triton-opt %s -split-input-file -triton-cpu-convert-memory-ops -canonicalize | FileCheck %s --check-prefixes=ALL,CHECK
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-memory-ops="assume-in-bounds=true" -canonicalize | FileCheck %s --check-prefixes=ALL,ASSUMEINBOUNDS

// Check that bounds checking semantics of tensor descriptor loads/stores are
// correctly propagated to the generated vector.transfer_read/write ops.

// ALL-LABEL: copy_trans
tt.func public @copy_trans(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // ALL: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32

  %c1_i64 = arith.constant 1 : i64
  %c8_i32 = arith.constant 8 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c8_i32 : i32
  %2 = tt.get_program_id y : i32
  %3 = arith.muli %2, %c8_i32 : i32
  %4 = arith.extsi %arg3 : i32 to i64
  %5 = tt.make_tensor_descriptor %arg0, [%arg2, %arg3], [%4, %c1_i64] : <f32>, <8x8xf32>
  %6 = arith.extsi %arg5 : i32 to i64
  %7 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%6, %c1_i64] : <f32>, <8x8xf32>

  // CHECK: %[[READ:.+]] = vector.transfer_read %{{.+}}[%{{.+}}, %{{.+}}], %[[ZERO]] :  memref<?x?xf32, strided<[?, 1]>>, vector<8x8xf32>
  // ASSUMEINBOUNDS: %[[READ:.+]] = vector.transfer_read %{{.+}}[%{{.+}}, %{{.+}}], %[[ZERO]] {in_bounds = [true, true]} :  memref<?x?xf32, strided<[?, 1]>>, vector<8x8xf32>
  %8 = tt.descriptor_load %5[%1, %3] : !tt.tensordesc<8x8xf32> -> tensor<8x8xf32>

  %9 = tt.trans %8 {order = array<i32: 1, 0>} : tensor<8x8xf32> -> tensor<8x8xf32>

  // CHECK: vector.transfer_write %{{.+}}, %{{.+}}[%{{.+}}, %{{.+}}]  : vector<8x8xf32>,  memref<?x?xf32, strided<[?, 1]>>
  // ASSUMEINBOUNDS: vector.transfer_write %{{.+}}, %{{.+}}[%{{.+}}, %{{.+}}] {in_bounds = [true, true]} : vector<8x8xf32>,  memref<?x?xf32, strided<[?, 1]>>
  tt.descriptor_store %7[%1, %3], %9 : !tt.tensordesc<8x8xf32>, tensor<8x8xf32>
  tt.return
}

// -----

// Same check for a 4D tensor descriptor; here, the rank reduction on the
// descriptor load needs to be reversed through an explicit vector.shape_cast,
// as vector.transfer_read doesn't do bounds checking on the leading/non-vector
// dimensions.

// ALL-LABEL: copy_trans_blocked
tt.func public @copy_trans_blocked(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // ALL: %[[ZERO:.+]] = arith.constant 0.000000e+00 : f32

  %c0_i32 = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c8_i64 = arith.constant 8 : i64
  %c8_i32 = arith.constant 8 : i32
  %0 = tt.get_program_id x : i32
  %1 = tt.get_program_id y : i32
  %2 = arith.divsi %arg2, %c8_i32 : i32
  %3 = arith.divsi %arg3, %c8_i32 : i32
  %4 = arith.muli %arg3, %c8_i32 : i32
  %5 = arith.extsi %4 : i32 to i64
  %6 = arith.extsi %arg3 : i32 to i64
  %7 = tt.make_tensor_descriptor %arg0, [%2, %3, %c8_i32, %c8_i32], [%5, %c8_i64, %6, %c1_i64] : <f32>, <1x1x8x8xf32>
  %8 = arith.divsi %arg4, %c8_i32 : i32
  %9 = arith.divsi %arg5, %c8_i32 : i32
  %10 = arith.muli %arg5, %c8_i32 : i32
  %11 = arith.extsi %10 : i32 to i64
  %12 = arith.extsi %arg5 : i32 to i64
  %13 = tt.make_tensor_descriptor %arg1, [%8, %9, %c8_i32, %c8_i32], [%11, %c8_i64, %12, %c1_i64] : <f32>, <1x1x8x8xf32>

  // CHECK: %[[READ:.+]] = vector.transfer_read %{{.+}}[%{{.+}}, %{{.+}}, %c0, %c0], %[[ZERO]] {in_bounds = [false, false, true, true]} : memref<?x?x8x8xf32, strided<[?, 8, ?, 1]>>, vector<1x1x8x8xf32>
  // CHECK: vector.shape_cast %[[READ]] : vector<1x1x8x8xf32> to vector<8x8xf32>
  // ASSUMEINBOUNDS: %[[READ:.+]] = vector.transfer_read %{{.+}}[%{{.+}}, %{{.+}}, %c0, %c0], %[[ZERO]] {in_bounds = [true, true]} : memref<?x?x8x8xf32, strided<[?, 8, ?, 1]>>, vector<8x8xf32>
  %14 = tt.descriptor_load %7[%0, %1, %c0_i32, %c0_i32] : !tt.tensordesc<1x1x8x8xf32> -> tensor<8x8xf32>

  %15 = tt.trans %14 {order = array<i32: 1, 0>} : tensor<8x8xf32> -> tensor<8x8xf32>
  %16 = tt.reshape %15 : tensor<8x8xf32> -> tensor<1x1x8x8xf32>

  // CHECK: vector.transfer_write %{{.+}}, %{{.+}}[%{{.+}}, %{{.+}}, %c0, %c0] {in_bounds = [false, false, true, true]} : vector<1x1x8x8xf32>, memref<?x?x8x8xf32, strided<[?, 8, ?, 1]>>
  // ASSUMEINBOUNDS: vector.transfer_write %{{.+}}, %{{.+}}[%{{.+}}, %{{.+}}, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x8x8xf32>, memref<?x?x8x8xf32, strided<[?, 8, ?, 1]>>
  tt.descriptor_store %13[%0, %1, %c0_i32, %c0_i32], %16 : !tt.tensordesc<1x1x8x8xf32>, tensor<1x1x8x8xf32>
  tt.return
}

// -----

// Check NaN padding.

tt.func public @copy_nan(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) {
  // ALL: %[[NAN:.+]] = arith.constant 0x7FC00000 : f32

  %c1_i64 = arith.constant 1 : i64
  %c8_i32 = arith.constant 8 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c8_i32 : i32
  %2 = tt.get_program_id y : i32
  %3 = arith.muli %2, %c8_i32 : i32
  %4 = arith.extsi %arg3 : i32 to i64
  %5 = tt.make_tensor_descriptor %arg0, [%arg2, %arg3], [%4, %c1_i64] {padding = 2 : i32} : <f32>, <8x8xf32>
  %6 = arith.extsi %arg5 : i32 to i64
  %7 = tt.make_tensor_descriptor %arg1, [%arg4, %arg5], [%6, %c1_i64] : <f32>, <8x8xf32>

  // ALL: vector.transfer_read %{{.+}}[%{{.+}}, %{{.+}}], %[[NAN]]
  %8 = tt.descriptor_load %5[%1, %3] : !tt.tensordesc<8x8xf32> -> tensor<8x8xf32>

  tt.descriptor_store %7[%1, %3], %8 : !tt.tensordesc<8x8xf32>, tensor<8x8xf32>
  tt.return
}
