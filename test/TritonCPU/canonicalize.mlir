// RUN: triton-opt %s -split-input-file -triton-cpu-canonicalize | FileCheck %s

// Fold transfer read and shape cast.

// CHECK-LABEL: @fold_transfer_read_shape_cast
// CHECK:       %[[VAL:.+]] = vector.transfer_read
// CHECK:       vector.transfer_write %[[VAL]]

module {
  tt.func public @fold_transfer_read_shape_cast(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i32 = arith.constant 16 : i32
    %c16_i64 = arith.constant 16 : i64
    %c256_i64 = arith.constant 256 : i64
    %c512_i64 = arith.constant 512 : i64
    %in_p = tt.make_tensor_descriptor %arg0, [%c2_i32, %c2_i32, %c16_i32, %c16_i32], [%c512_i64, %c256_i64, %c16_i64, %c1_i64] : <bf16>, <1x1x16x16xbf16>
    %out_p = tt.make_tensor_descriptor %arg1, [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <bf16>, <16x16xbf16>
    %memref1 = triton_cpu.extract_memref %in_p : <1x1x16x16xbf16> -> memref<2x2x16x16xbf16, strided<[512, 256, 16, 1]>>
    %val1 = vector.transfer_read %memref1[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<2x2x16x16xbf16, strided<[512, 256, 16, 1]>>, vector<1x1x16x16xbf16>
    %val2 = vector.shape_cast %val1 : vector<1x1x16x16xbf16> to vector<16x16xbf16>
    %memref2 = triton_cpu.extract_memref %out_p : <16x16xbf16> -> memref<16x16xbf16, strided<[16, 1]>>
    vector.transfer_write %val2, %memref2[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xbf16>, memref<16x16xbf16, strided<[16, 1]>>
    tt.return
  }
}

// -----

// The rank-reduced read needs a mask of the new rank as well.
// CHECK-LABEL: @fold_masked_transfer_read_shape_cast
// CHECK: %[[MASK:.+]] = vector.shape_cast %arg3 : vector<1x16xi1> to vector<16xi1>
// CHECK: %[[MASKED_VALUE:.+]] = vector.transfer_read %arg0[%arg1, %arg2], %{{.+}}, %[[MASK]] {in_bounds = [true]} : memref<?x?xf32>, vector<16xf32>
// CHECK-NOT: vector.shape_cast
// CHECK: tt.return %[[MASKED_VALUE]]
tt.func public @fold_masked_transfer_read_shape_cast(%base: memref<?x?xf32>, %i: index, %j: index, %mask: vector<1x16xi1>) -> vector<16xf32> {
  %zero = arith.constant 0.0 : f32
  %read = vector.transfer_read %base[%i, %j], %zero, %mask {in_bounds = [true, true]} : memref<?x?xf32>, vector<1x16xf32>
  %result = vector.shape_cast %read : vector<1x16xf32> to vector<16xf32>
  tt.return %result : vector<16xf32>
}
