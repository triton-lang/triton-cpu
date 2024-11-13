// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-onednn -canonicalize | FileCheck %s

// Replacement of a contraction operation with a single tile_mulf operation.

// CHECK-LABEL: @test_single_mulf
// CHECK:       %[[RHS_BUF:.+]] = memref.alloca() {alignment = 64 : i64} : memref<16x32xbf16>
// CHECK:       %[[OUT_MEMREF:.+]] = triton_cpu.extract_memref %2 : <tensor<16x16xf32>> -> memref<16x16xf32, strided<[16, 1]>>
// CHECK-NEXT:  %[[OUT_INDICES:.+]]:2 = triton_cpu.extract_indices %2 : <tensor<16x16xf32>> -> index, index
// CHECK:       %[[ACC:.+]] = amx.tile_zero : vector<16x16xf32>
// CHECK-NEXT:  %[[LHS:.+]] = amx.tile_load %3[%4#0, %4#1]
// CHECK-NEXT:  %[[RHS:.+]] = amx.tile_load %[[RHS_BUF]][%c0{{.*}}, %c0{{.*}}]
// CHECK-NEXT:  %[[RES:.+]] = amx.tile_mulf %[[LHS]], %[[RHS]], %[[ACC]] : vector<16x32xbf16>, vector<16x32xbf16>, vector<16x16xf32>
// CHECK-NEXT:  amx.tile_store %[[OUT_MEMREF]][%[[OUT_INDICES]]#0, %[[OUT_INDICES]]#1], %[[RES]] : memref<16x16xf32, strided<[16, 1]>>, vector<16x16xf32>

#loc = loc(unknown)
#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  tt.func public @test_single_mulf(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : bf16 loc(#loc)
    %cst_0 = arith.constant dense<0.000000e+00> : vector<16x16xf32> loc(#loc)
    %c16_i64 = arith.constant 16 : i64 loc(#loc)
    %c32_i64 = arith.constant 32 : i64 loc(#loc)
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %0 = tt.make_tensor_ptr %arg0, [%c16_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xbf16>> loc(#loc)
    %1 = tt.make_tensor_ptr %arg1, [%c32_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xbf16>> loc(#loc)
    %2 = tt.make_tensor_ptr %arg2, [%c16_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>> loc(#loc)
    %3 = triton_cpu.extract_memref %0 : <tensor<16x32xbf16>> -> memref<16x32xbf16, strided<[32, 1]>> loc(#loc)
    %4:2 = triton_cpu.extract_indices %0 : <tensor<16x32xbf16>> -> index, index loc(#loc)
    %5 = vector.transfer_read %3[%4#0, %4#1], %cst {in_bounds = [true, true]} : memref<16x32xbf16, strided<[32, 1]>>, vector<16x32xbf16> loc(#loc)
    %6 = triton_cpu.extract_memref %1 : <tensor<32x16xbf16>> -> memref<32x16xbf16, strided<[16, 1]>> loc(#loc)
    %7:2 = triton_cpu.extract_indices %1 : <tensor<32x16xbf16>> -> index, index loc(#loc)
    %8 = vector.transfer_read %6[%7#0, %7#1], %cst {in_bounds = [true, true]} : memref<32x16xbf16, strided<[16, 1]>>, vector<32x16xbf16> loc(#loc)
    %9 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %8, %cst_0 : vector<16x32xbf16>, vector<32x16xbf16> into vector<16x16xf32> loc(#loc)
    %10 = triton_cpu.extract_memref %2 : <tensor<16x16xf32>> -> memref<16x16xf32, strided<[16, 1]>> loc(#loc)
    %11:2 = triton_cpu.extract_indices %2 : <tensor<16x16xf32>> -> index, index loc(#loc)
    vector.transfer_write %9, %10[%11#0, %11#1] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32, strided<[16, 1]>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

