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

// Flatten a fully in-bounds contiguous read to a rank-1 wide read.

// CHECK-LABEL: @flatten_contiguous_transfer_read
// CHECK: %[[WIDE:.+]] = vector.transfer_read %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}], %{{.+}} {in_bounds = [true]} : memref<1x64x2xbf16, strided<[?, 2, 1], offset: ?>>, vector<32xbf16>
// CHECK-NOT: vector.shape_cast
// CHECK: vector.store %[[WIDE]]
module {
  tt.func public @flatten_contiguous_transfer_read(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %src = triton_cpu.ptr_to_memref %arg0 : <bf16> -> memref<1x64x2xbf16, strided<[?, 2, 1], offset: ?>>
    %read = vector.transfer_read %src[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x64x2xbf16, strided<[?, 2, 1], offset: ?>>, vector<1x16x2xbf16>
    %flat = vector.shape_cast %read : vector<1x16x2xbf16> to vector<32xbf16>
    %dst = triton_cpu.ptr_to_memref %arg1 : <bf16> -> memref<32xbf16>
    vector.store %flat, %dst[%c0] : memref<32xbf16>, vector<32xbf16>
    tt.return
  }
}

// -----

// Do not flatten a read whose row stride contains gaps.

// CHECK-LABEL: @keep_noncontiguous_transfer_read
// CHECK: %[[READ:.+]] = vector.transfer_read %{{.+}}[%{{.+}}, %{{.+}}, %{{.+}}], %{{.+}} {in_bounds = [true, true, true]} : memref<1x16x2xbf16, strided<[64, 4, 1]>>, vector<1x16x2xbf16>
// CHECK: %[[FLAT:.+]] = vector.shape_cast %[[READ]] : vector<1x16x2xbf16> to vector<32xbf16>
// CHECK: vector.store %[[FLAT]]
module {
  tt.func public @keep_noncontiguous_transfer_read(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}) {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %src = triton_cpu.ptr_to_memref %arg0 : <bf16> -> memref<1x16x2xbf16, strided<[64, 4, 1]>>
    %read = vector.transfer_read %src[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x16x2xbf16, strided<[64, 4, 1]>>, vector<1x16x2xbf16>
    %flat = vector.shape_cast %read : vector<1x16x2xbf16> to vector<32xbf16>
    %dst = triton_cpu.ptr_to_memref %arg1 : <bf16> -> memref<32xbf16>
    vector.store %flat, %dst[%c0] : memref<32xbf16>, vector<32xbf16>
    tt.return
  }
}
