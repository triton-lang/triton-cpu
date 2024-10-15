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

// #loc = loc(unknown)
// #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// module {
//   tt.func public @test_single_mulf(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
//     %cst = arith.constant 0.000000e+00 : bf16 loc(#loc)
//     %cst_0 = arith.constant dense<0.000000e+00> : vector<16x16xf32> loc(#loc)
//     %c16_i64 = arith.constant 16 : i64 loc(#loc)
//     %c32_i64 = arith.constant 32 : i64 loc(#loc)
//     %c1_i64 = arith.constant 1 : i64 loc(#loc)
//     %c0_i32 = arith.constant 0 : i32 loc(#loc)
//     %0 = tt.make_tensor_ptr %arg0, [%c16_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xbf16>> loc(#loc)
//     %1 = tt.make_tensor_ptr %arg1, [%c32_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xbf16>> loc(#loc)
//     %2 = tt.make_tensor_ptr %arg2, [%c16_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xf32>> loc(#loc)
//     %3 = triton_cpu.extract_memref %0 : <tensor<16x32xbf16>> -> memref<16x32xbf16, strided<[32, 1]>> loc(#loc)
//     %4:2 = triton_cpu.extract_indices %0 : <tensor<16x32xbf16>> -> index, index loc(#loc)
//     %5 = vector.transfer_read %3[%4#0, %4#1], %cst {in_bounds = [true, true]} : memref<16x32xbf16, strided<[32, 1]>>, vector<16x32xbf16> loc(#loc)
//     %6 = triton_cpu.extract_memref %1 : <tensor<32x16xbf16>> -> memref<32x16xbf16, strided<[16, 1]>> loc(#loc)
//     %7:2 = triton_cpu.extract_indices %1 : <tensor<32x16xbf16>> -> index, index loc(#loc)
//     %8 = vector.transfer_read %6[%7#0, %7#1], %cst {in_bounds = [true, true]} : memref<32x16xbf16, strided<[16, 1]>>, vector<32x16xbf16> loc(#loc)
//     %9 = vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %5, %8, %cst_0 : vector<16x32xbf16>, vector<32x16xbf16> into vector<16x16xf32> loc(#loc)
//     %10 = triton_cpu.extract_memref %2 : <tensor<16x16xf32>> -> memref<16x16xf32, strided<[16, 1]>> loc(#loc)
//     %11:2 = triton_cpu.extract_indices %2 : <tensor<16x16xf32>> -> index, index loc(#loc)
//     vector.transfer_write %9, %10[%11#0, %11#1] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32, strided<[16, 1]>> loc(#loc)
//     tt.return loc(#loc)
//   } loc(#loc)
// } loc(#loc)

// -----// IR Dump Before OneDNNOpsToLLVM (triton-cpu-onednn-ops-to-llvm) ('builtin.module' operation) //----- //
#loc = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5906:0)
module {
  tt.func public @matmul_blocked_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5906:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5906:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5906:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5906:0), %arg4: i32 {tt.divisibility = 16 : i32} loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5906:0), %arg5: i32 {tt.divisibility = 16 : i32} loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5906:0), %arg6: i32 {tt.divisibility = 16 : i32} loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5906:0), %arg7: i32 {tt.divisibility = 16 : i32} loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5906:0), %arg8: i32 {tt.divisibility = 16 : i32} loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5906:0)) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f32 loc(#loc1)
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x16xf32> loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c1_i64 = arith.constant 1 : i64 loc(#loc1)
    %c16_i32 = arith.constant 16 : i32 loc(#loc1)
    %c32_i32 = arith.constant 32 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = tt.get_program_id y : i32 loc(#loc3)
    %2 = arith.muli %0, %c32_i32 : i32 loc(#loc4)
    %3 = arith.muli %1, %c16_i32 : i32 loc(#loc5)
    %4 = arith.extsi %arg3 : i32 to i64 loc(#loc6)
    %5 = arith.extsi %arg5 : i32 to i64 loc(#loc6)
    %6 = arith.extsi %arg6 : i32 to i64 loc(#loc6)
    %7 = tt.make_tensor_ptr %arg0, [%4, %5], [%6, %c1_i64], [%2, %c0_i32] {order = array<i32: 1, 0>} : <tensor<32x16xf32>> loc(#loc6)
    %8 = arith.extsi %arg4 : i32 to i64 loc(#loc7)
    %9 = arith.extsi %arg7 : i32 to i64 loc(#loc7)
    %10 = tt.make_tensor_ptr %arg1, [%5, %8], [%9, %c1_i64], [%c0_i32, %3] {order = array<i32: 1, 0>} : <tensor<16x16xf32>> loc(#loc7)
    %11 = arith.divsi %arg5, %c16_i32 : i32 loc(#loc20)
    %12:3 = scf.for %arg9 = %c0_i32 to %11 step %c1_i32 iter_args(%arg10 = %cst_0, %arg11 = %7, %arg12 = %10) -> (vector<32x16xf32>, !tt.ptr<tensor<32x16xf32>>, !tt.ptr<tensor<16x16xf32>>)  : i32 {
      %17 = triton_cpu.extract_memref %arg11 : <tensor<32x16xf32>> -> memref<?x?xf32, strided<[?, 1]>> loc(#loc11)
      %18:2 = triton_cpu.extract_indices %arg11 : <tensor<32x16xf32>> -> index, index loc(#loc11)
      %19 = vector.transfer_read %17[%18#0, %18#1], %cst : memref<?x?xf32, strided<[?, 1]>>, vector<32x16xf32> loc(#loc11)
      %20 = triton_cpu.extract_memref %arg12 : <tensor<16x16xf32>> -> memref<?x?xf32, strided<[?, 1]>> loc(#loc12)
      %21:2 = triton_cpu.extract_indices %arg12 : <tensor<16x16xf32>> -> index, index loc(#loc12)
      %22 = vector.transfer_read %20[%21#0, %21#1], %cst : memref<?x?xf32, strided<[?, 1]>>, vector<16x16xf32> loc(#loc12)
      %23 = triton_cpu.dot %19, %22, %arg10, inputPrecision = tf32 : vector<32x16xf32> * vector<16x16xf32> -> vector<32x16xf32> loc(#loc13)
      %24 = tt.advance %arg11, [%c0_i32, %c16_i32] : <tensor<32x16xf32>> loc(#loc14)
      %25 = tt.advance %arg12, [%c16_i32, %c0_i32] : <tensor<16x16xf32>> loc(#loc15)
      scf.yield %23, %24, %25 : vector<32x16xf32>, !tt.ptr<tensor<32x16xf32>>, !tt.ptr<tensor<16x16xf32>> loc(#loc16)
    } loc(#loc10)
    %13 = arith.extsi %arg8 : i32 to i64 loc(#loc17)
    %14 = tt.make_tensor_ptr %arg2, [%4, %8], [%13, %c1_i64], [%2, %3] {order = array<i32: 1, 0>} : <tensor<32x16xf32>> loc(#loc17)
    %15 = triton_cpu.extract_memref %14 : <tensor<32x16xf32>> -> memref<?x?xf32, strided<[?, 1]>> loc(#loc18)
    %16:2 = triton_cpu.extract_indices %14 : <tensor<32x16xf32>> -> index, index loc(#loc18)
    vector.transfer_write %12#0, %15[%16#0, %16#1] : vector<32x16xf32>, memref<?x?xf32, strided<[?, 1]>> loc(#loc18)
    tt.return loc(#loc19)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5916:26)
#loc3 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5917:26)
#loc4 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5927:29)
#loc5 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5928:29)
#loc6 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5931:106)
#loc7 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5933:106)
#loc8 = loc("/home/jovyan/triton-cpu/python/triton/language/standard.py":40:28)
#loc9 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5935:36)
#loc10 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5935:51)
#loc11 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5936:20)
#loc12 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5937:20)
#loc13 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5938:32)
#loc14 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5939:44)
#loc15 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5940:44)
#loc16 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5940:8)
#loc17 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5951:36)
#loc18 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5952:26)
#loc19 = loc("/home/jovyan/triton-cpu/python/test/unit/language/test_core.py":5952:4)
#loc20 = loc(callsite(#loc8 at #loc9))
