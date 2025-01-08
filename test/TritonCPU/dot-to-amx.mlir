// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-amx="convert-bf16=true convert-fp16=true convert-i8=true" -canonicalize | FileCheck %s

// Replacement of a contraction operation with a single tile_mulf operation.

// CHECK-LABEL: @test_single_mulf
// CHECK:       %[[RHS_BUF:.+]] = memref.alloca() {alignment = 64 : i64} : memref<16x32xbf16>
// CHECK:       %[[OUT_MEMREF:.+]] = triton_cpu.extract_memref %2 : <tensor<16x16xf32>> -> memref<16x16xf32, strided<[16, 1]>>
// CHECK-NEXT:  %[[OUT_INDICES:.+]]:2 = triton_cpu.extract_indices %2 : <tensor<16x16xf32>> -> index, index
// CHECK:       %[[ACC:.+]] = amx.tile_zero : !amx.tile<16x16xf32>
// CHECK-NEXT:  %[[LHS:.+]] = amx.tile_load %3[%4#0, %4#1]
// CHECK-NEXT:  %[[RHS:.+]] = amx.tile_load %[[RHS_BUF]][%c0{{.*}}, %c0{{.*}}]
// CHECK-NEXT:  %[[RES:.+]] = amx.tile_mulf %[[LHS]], %[[RHS]], %[[ACC]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:  amx.tile_store %[[OUT_MEMREF]][%[[OUT_INDICES]]#0, %[[OUT_INDICES]]#1], %[[RES]] : memref<16x16xf32, strided<[16, 1]>>, !amx.tile<16x16xf32>

#loc = loc(unknown)
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
    %9 = triton_cpu.dot %5, %8, %cst_0, inputPrecision = ieee : vector<16x32xbf16> * vector<32x16xbf16> -> vector<16x16xf32> loc(#loc)
    %10 = triton_cpu.extract_memref %2 : <tensor<16x16xf32>> -> memref<16x16xf32, strided<[16, 1]>> loc(#loc)
    %11:2 = triton_cpu.extract_indices %2 : <tensor<16x16xf32>> -> index, index loc(#loc)
    vector.transfer_write %9, %10[%11#0, %11#1] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32, strided<[16, 1]>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

// -----

// Replacement of a contraction operation with multiple tile_muli operations.

// CHECK-LABEL: @test_single_tile_two_muli
// CHECK:       %[[RHS_BUF:.+]] = memref.alloca() {alignment = 64 : i64} : memref<32x64xi8>
// CHECK:       %[[OUT_MEMREF:.+]] = triton_cpu.extract_memref %2 : <tensor<16x16xi32>> -> memref<16x16xi32, strided<[16, 1]>>
// CHECK-NEXT:  %[[OUT_INDICES:.+]]:2 = triton_cpu.extract_indices %2 : <tensor<16x16xi32>> -> index, index
// CHECK:       %[[ACC:.+]] = amx.tile_zero : !amx.tile<16x16xi32>
// CHECK-NEXT:  %[[LHS1:.+]] = amx.tile_load %3[%4#0, %4#1]
// CHECK-NEXT:  %[[RHS1:.+]] = amx.tile_load %[[RHS_BUF]][%c0{{.*}}, %c0{{.*}}]
// CHECK-NEXT:  %[[RES1:.+]] = amx.tile_muli %[[LHS1]], %[[RHS1]], %[[ACC]] : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
// CHECK-NEXT:  %[[IDX1:.+]] = arith.addi %4#1, %c64{{.*}} : index
// CHECK-NEXT:  %[[LHS2:.+]] = amx.tile_load %3[%4#0, %[[IDX1]]] : memref<16x128xi8, strided<[128, 1]>> into !amx.tile<16x64xi8>
// CHECK-NEXT:  %[[RHS2:.+]] = amx.tile_load %[[RHS_BUF]][%c16{{.*}}, %c0{{.*}}] : memref<32x64xi8> into !amx.tile<16x64xi8>
// CHECK-NEXT:  %[[RES2:.+]] = amx.tile_muli %[[LHS2]], %[[RHS2]], %[[RES1]] : !amx.tile<16x64xi8>, !amx.tile<16x64xi8>, !amx.tile<16x16xi32>
// CHECK-NEXT:  amx.tile_store %[[OUT_MEMREF]][%[[OUT_INDICES]]#0, %[[OUT_INDICES]]#1], %[[RES2]] : memref<16x16xi32, strided<[16, 1]>>, !amx.tile<16x16xi32>

#loc = loc(unknown)
module {
  tt.func public @test_single_tile_two_muli(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32} loc(unknown), %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32} loc(unknown), %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %c0_i8 = arith.constant 0 : i8 loc(#loc)
    %cst = arith.constant dense<0> : vector<16x16xi32> loc(#loc)
    %c16_i64 = arith.constant 16 : i64 loc(#loc)
    %c128_i64 = arith.constant 128 : i64 loc(#loc)
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %0 = tt.make_tensor_ptr %arg0, [%c16_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x128xi8>> loc(#loc)
    %1 = tt.make_tensor_ptr %arg1, [%c128_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<128x16xi8>> loc(#loc)
    %2 = tt.make_tensor_ptr %arg2, [%c16_i64, %c16_i64], [%c16_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x16xi32>> loc(#loc)
    %3 = triton_cpu.extract_memref %0 : <tensor<16x128xi8>> -> memref<16x128xi8, strided<[128, 1]>> loc(#loc)
    %4:2 = triton_cpu.extract_indices %0 : <tensor<16x128xi8>> -> index, index loc(#loc)
    %5 = vector.transfer_read %3[%4#0, %4#1], %c0_i8 {in_bounds = [true, true]} : memref<16x128xi8, strided<[128, 1]>>, vector<16x128xi8> loc(#loc)
    %6 = triton_cpu.extract_memref %1 : <tensor<128x16xi8>> -> memref<128x16xi8, strided<[16, 1]>> loc(#loc)
    %7:2 = triton_cpu.extract_indices %1 : <tensor<128x16xi8>> -> index, index loc(#loc)
    %8 = vector.transfer_read %6[%7#0, %7#1], %c0_i8 {in_bounds = [true, true]} : memref<128x16xi8, strided<[16, 1]>>, vector<128x16xi8> loc(#loc)
    %9 = triton_cpu.dot %5, %8, %cst, inputPrecision = ieee : vector<16x128xi8> * vector<128x16xi8> -> vector<16x16xi32> loc(#loc)
    %10 = triton_cpu.extract_memref %2 : <tensor<16x16xi32>> -> memref<16x16xi32, strided<[16, 1]>> loc(#loc)
    %11:2 = triton_cpu.extract_indices %2 : <tensor<16x16xi32>> -> index, index loc(#loc)
    vector.transfer_write %9, %10[%11#0, %11#1] {in_bounds = [true, true]} : vector<16x16xi32>, memref<16x16xi32, strided<[16, 1]>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

// -----

// Replacement of a contraction operation with multiple tile_mulf operations
// and multiple output tiles.

// CHECK-LABEL: @test_two_tiles_four_mulf
// CHECK:       %[[RHS_BUF:.+]] = memref.alloca() {alignment = 64 : i64} : memref<32x64xbf16>
// CHECK:       %[[OUT_MEMREF:.+]] = triton_cpu.extract_memref %2 : <tensor<16x32xf32>> -> memref<16x32xf32, strided<[32, 1]>>
// CHECK-NEXT:  %[[OUT_INDICES:.+]]:2 = triton_cpu.extract_indices %2 : <tensor<16x32xf32>> -> index, index
// CHECK:       %[[ACC1:.+]] = amx.tile_zero : !amx.tile<16x16xf32>
// CHECK-NEXT:  %[[ACC2:.+]] = amx.tile_zero : !amx.tile<16x16xf32>
// CHECK-NEXT:  %[[LHS1:.+]] = amx.tile_load %3[%4#0, %4#1] : memref<16x64xbf16, strided<[64, 1]>> into !amx.tile<16x32xbf16>
// CHECK-NEXT:  %[[RHS1:.+]] = amx.tile_load %[[RHS_BUF]][%c0{{.*}}, %c0{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:  %[[RES1:.+]] = amx.tile_mulf %[[LHS1]], %[[RHS1]], %[[ACC1]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:       %[[RHS2:.+]] = amx.tile_load %[[RHS_BUF]][%c0{{.*}}, %c32{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:  %[[RES2:.+]] = amx.tile_mulf %[[LHS1]], %[[RHS2]], %[[ACC2]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:       %[[IDX1:.+]] = arith.addi %4#1, %c32{{.*}} : index
// CHECK-NEXT:  %[[LHS2:.+]] = amx.tile_load %3[%4#0, %[[IDX1]]] : memref<16x64xbf16, strided<[64, 1]>> into !amx.tile<16x32xbf16>
// CHECK:       %[[RHS3:.+]] = amx.tile_load %[[RHS_BUF]][%c16{{.*}}, %c0{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:  %[[RES3:.+]] = amx.tile_mulf %[[LHS2]], %[[RHS3]], %[[RES1]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:  amx.tile_store %[[OUT_MEMREF]][%[[OUT_INDICES]]#0, %[[OUT_INDICES]]#1], %[[RES3]] : memref<16x32xf32, strided<[32, 1]>>, !amx.tile<16x16xf32>
// CHECK:       %[[RHS4:.+]] = amx.tile_load %[[RHS_BUF]][%c16{{.*}}, %c32{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:  %[[RES4:.+]] = amx.tile_mulf %[[LHS2]], %[[RHS4]], %[[RES2]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK:       %[[IDX2:.+]] = arith.addi %[[OUT_INDICES]]#1, %c16{{.*}} : index
// CHECK-NEXT:  amx.tile_store %[[OUT_MEMREF]][%[[OUT_INDICES]]#0, %[[IDX2]]], %[[RES4]] : memref<16x32xf32, strided<[32, 1]>>, !amx.tile<16x16xf32>

#loc = loc(unknown)
module {
  tt.func public @test_two_tiles_four_mulf(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32} loc(unknown), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : bf16 loc(#loc)
    %cst_0 = arith.constant dense<0.000000e+00> : vector<16x32xf32> loc(#loc)
    %c32_i64 = arith.constant 32 : i64 loc(#loc)
    %c16_i64 = arith.constant 16 : i64 loc(#loc)
    %c64_i64 = arith.constant 64 : i64 loc(#loc)
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %0 = tt.make_tensor_ptr %arg0, [%c16_i64, %c64_i64], [%c64_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x64xbf16>> loc(#loc)
    %1 = tt.make_tensor_ptr %arg1, [%c64_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xbf16>> loc(#loc)
    %2 = tt.make_tensor_ptr %arg2, [%c16_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<16x32xf32>> loc(#loc)
    %3 = triton_cpu.extract_memref %0 : <tensor<16x64xbf16>> -> memref<16x64xbf16, strided<[64, 1]>> loc(#loc)
    %4:2 = triton_cpu.extract_indices %0 : <tensor<16x64xbf16>> -> index, index loc(#loc)
    %5 = vector.transfer_read %3[%4#0, %4#1], %cst {in_bounds = [true, true]} : memref<16x64xbf16, strided<[64, 1]>>, vector<16x64xbf16> loc(#loc)
    %6 = triton_cpu.extract_memref %1 : <tensor<64x32xbf16>> -> memref<64x32xbf16, strided<[32, 1]>> loc(#loc)
    %7:2 = triton_cpu.extract_indices %1 : <tensor<64x32xbf16>> -> index, index loc(#loc)
    %8 = vector.transfer_read %6[%7#0, %7#1], %cst {in_bounds = [true, true]} : memref<64x32xbf16, strided<[32, 1]>>, vector<64x32xbf16> loc(#loc)
    %9 = triton_cpu.dot %5, %8, %cst_0, inputPrecision = ieee : vector<16x64xbf16> * vector<64x32xbf16> -> vector<16x32xf32> loc(#loc)
    %10 = triton_cpu.extract_memref %2 : <tensor<16x32xf32>> -> memref<16x32xf32, strided<[32, 1]>> loc(#loc)
    %11:2 = triton_cpu.extract_indices %2 : <tensor<16x32xf32>> -> index, index loc(#loc)
    vector.transfer_write %9, %10[%11#0, %11#1] {in_bounds = [true, true]} : vector<16x32xf32>, memref<16x32xf32, strided<[32, 1]>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

// -----

// More complicated case with a loop, input casts, and accumulator that
// cannot fit tile register file.

// CHECK-LABEL: @test_loop_acc_two_blocks
// CHECK:       %[[LHS_BUF:.+]] = memref.alloca() {alignment = 64 : i64} : memref<64x64xbf16>
// CHECK:       %[[RHS_BUF:.+]] = memref.alloca() {alignment = 64 : i64} : memref<32x64xbf16>
// CHECK:       %[[ACC_BUF:.+]] = memref.alloca() {alignment = 64 : i64} : memref<64x32xf32>
// CHECK:       vector.transfer_write %cst{{.+}}, %[[ACC_BUF]][%c0{{.*}}, %c0{{.*}}] {in_bounds = [true, true]}  : vector<64x32xf32>, memref<64x32xf32>
// CHECK:       %3:2 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %0, %arg5 = %1) -> (!tt.ptr<tensor<64x64xf8E5M2>>, !tt.ptr<tensor<64x32xf8E5M2>>) : i32
// CHECK:         %[[LHS:.+]] = vector.transfer_read %{{.+}}[%{{.+}}#0, %{{.+}}#1], %{{.+}} {in_bounds = [true, true]} : memref<64x128xf8E5M2, strided<[128, 1]>>, vector<64x64xf8E5M2>
// CHECK:         %[[RHS:.+]] = vector.transfer_read %{{.+}}[%{{.+}}#0, %{{.+}}#1], %{{.+}} {in_bounds = [true, true]} : memref<128x32xf8E5M2, strided<[32, 1]>>, vector<64x32xf8E5M2>
// CHECK-NEXT:    %[[LHS1:.+]] = arith.extf %[[LHS]] : vector<64x64xf8E5M2> to vector<64x64xbf16>
// CHECK-NEXT:    vector.transfer_write %[[LHS1]], %[[LHS_BUF]][%c0{{.*}}, %c0{{.*}}] {in_bounds = [true, true]} : vector<64x64xbf16>, memref<64x64xbf16>
// CHECK-NEXT:    %[[RHS1:.+]] = arith.extf %[[RHS]] : vector<64x32xf8E5M2> to vector<64x32xbf16>
// CHECK-COUNT-32: vector.store %{{.+}}, %[[RHS_BUF]][%{{.+}}, %{{.+}}] : memref<32x64xbf16>, vector<64xbf16>
// CHECK-NEXT:    %[[ACC_0_0:.+]] = amx.tile_load %[[ACC_BUF]][%c0{{.*}}, %c0{{.*}}] : memref<64x32xf32> into !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[ACC_0_1:.+]] = amx.tile_load %[[ACC_BUF]][%c0{{.*}}, %c16{{.*}}] : memref<64x32xf32> into !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[ACC_1_0:.+]] = amx.tile_load %[[ACC_BUF]][%c16{{.*}}, %c0{{.*}}] : memref<64x32xf32> into !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[ACC_1_1:.+]] = amx.tile_load %[[ACC_BUF]][%c16{{.*}}, %c16{{.*}}] : memref<64x32xf32> into !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[LHS_0_0:.+]] = amx.tile_load %[[LHS_BUF]][%c0{{.*}}, %c0{{.*}}] : memref<64x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[LHS_1_0:.+]] = amx.tile_load %[[LHS_BUF]][%c16{{.*}}, %c0{{.*}}] : memref<64x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[RHS_0_0:.+]] = amx.tile_load %[[RHS_BUF]][%c0{{.*}}, %c0{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[TMP_0_0:.+]] = amx.tile_mulf %[[LHS_0_0]], %[[RHS_0_0]], %[[ACC_0_0]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[TMP_1_0:.+]] = amx.tile_mulf %[[LHS_1_0]], %[[RHS_0_0]], %[[ACC_1_0]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[RHS_0_1:.+]] = amx.tile_load %[[RHS_BUF]][%c0{{.*}}, %c32{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[TMP_0_1:.+]] = amx.tile_mulf %[[LHS_0_0]], %[[RHS_0_1]], %[[ACC_0_1]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[TMP_1_1:.+]] = amx.tile_mulf %[[LHS_1_0]], %[[RHS_0_1]], %[[ACC_1_1]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[LHS_0_1:.+]] = amx.tile_load %[[LHS_BUF]][%c0{{.*}}, %c32{{.*}}] : memref<64x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[LHS_1_1:.+]] = amx.tile_load %[[LHS_BUF]][%c16{{.*}}, %c32{{.*}}] : memref<64x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[RHS_1_0:.+]] = amx.tile_load %[[RHS_BUF]][%c16{{.*}}, %c0{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[RES_0_0:.+]] = amx.tile_mulf %[[LHS_0_1]], %[[RHS_1_0]], %[[TMP_0_0]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    amx.tile_store %[[ACC_BUF]][%c0{{.*}}, %c0{{.*}}], %[[RES_0_0]] : memref<64x32xf32>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[RES_1_0:.+]] = amx.tile_mulf %[[LHS_1_1]], %[[RHS_1_0]], %[[TMP_1_0]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    amx.tile_store %[[ACC_BUF]][%c16{{.*}}, %c0{{.*}}], %[[RES_1_0]] : memref<64x32xf32>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[RHS_1_1:.+]] = amx.tile_load %[[RHS_BUF]][%c16{{.*}}, %c32{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[RES_0_1:.+]] = amx.tile_mulf %[[LHS_0_1]], %[[RHS_1_1]], %[[TMP_0_1]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    amx.tile_store %[[ACC_BUF]][%c0{{.*}}, %c16{{.*}}], %[[RES_0_1]] : memref<64x32xf32>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[RES_1_1:.+]] = amx.tile_mulf %[[LHS_1_1]], %[[RHS_1_1]], %[[TMP_1_1]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    amx.tile_store %[[ACC_BUF]][%c16{{.*}}, %c16{{.*}}], %[[RES_1_1]] : memref<64x32xf32>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[ACC_2_0:.+]] = amx.tile_load %[[ACC_BUF]][%c32{{.*}}, %c0{{.*}}] : memref<64x32xf32> into !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[ACC_2_1:.+]] = amx.tile_load %[[ACC_BUF]][%c32{{.*}}, %c16{{.*}}] : memref<64x32xf32> into !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[ACC_3_0:.+]] = amx.tile_load %[[ACC_BUF]][%c48{{.*}}, %c0{{.*}}] : memref<64x32xf32> into !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[ACC_3_1:.+]] = amx.tile_load %[[ACC_BUF]][%c48{{.*}}, %c16{{.*}}] : memref<64x32xf32> into !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[LHS_2_0:.+]] = amx.tile_load %[[LHS_BUF]][%c32{{.*}}, %c0{{.*}}] : memref<64x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[LHS_3_0:.+]] = amx.tile_load %[[LHS_BUF]][%c48{{.*}}, %c0{{.*}}] : memref<64x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[RHS_0_0:.+]] = amx.tile_load %[[RHS_BUF]][%c0{{.*}}, %c0{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[TMP_2_0:.+]] = amx.tile_mulf %[[LHS_2_0]], %[[RHS_0_0]], %[[ACC_2_0]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[TMP_3_0:.+]] = amx.tile_mulf %[[LHS_3_0]], %[[RHS_0_0]], %[[ACC_3_0]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[RHS_0_1:.+]] = amx.tile_load %[[RHS_BUF]][%c0{{.*}}, %c32{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[TMP_2_1:.+]] = amx.tile_mulf %[[LHS_2_0]], %[[RHS_0_1]], %[[ACC_2_1]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[TMP_3_1:.+]] = amx.tile_mulf %[[LHS_3_0]], %[[RHS_0_1]], %[[ACC_3_1]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[LHS_2_1:.+]] = amx.tile_load %[[LHS_BUF]][%c32{{.*}}, %c32{{.*}}] : memref<64x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[LHS_3_1:.+]] = amx.tile_load %[[LHS_BUF]][%c48{{.*}}, %c32{{.*}}] : memref<64x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[RHS_1_0:.+]] = amx.tile_load %[[RHS_BUF]][%c16{{.*}}, %c0{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[RES_2_0:.+]] = amx.tile_mulf %[[LHS_2_1]], %[[RHS_1_0]], %[[TMP_2_0]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    amx.tile_store %[[ACC_BUF]][%c32{{.*}}, %c0{{.*}}], %[[RES_2_0]] : memref<64x32xf32>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[RES_3_0:.+]] = amx.tile_mulf %[[LHS_3_1]], %[[RHS_1_0]], %[[TMP_3_0]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    amx.tile_store %[[ACC_BUF]][%c48{{.*}}, %c0{{.*}}], %[[RES_3_0]] : memref<64x32xf32>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[RHS_1_1:.+]] = amx.tile_load %[[RHS_BUF]][%c16{{.*}}, %c32{{.*}}] : memref<32x64xbf16> into !amx.tile<16x32xbf16>
// CHECK-NEXT:    %[[RES_2_1:.+]] = amx.tile_mulf %[[LHS_2_1]], %[[RHS_1_1]], %[[TMP_2_1]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    amx.tile_store %[[ACC_BUF]][%c32{{.*}}, %c16{{.*}}], %[[RES_2_1]] : memref<64x32xf32>, !amx.tile<16x16xf32>
// CHECK-NEXT:    %[[RES_3_1:.+]] = amx.tile_mulf %[[LHS_3_1]], %[[RHS_1_1]], %[[TMP_3_1]] : !amx.tile<16x32xbf16>, !amx.tile<16x32xbf16>, !amx.tile<16x16xf32>
// CHECK-NEXT:    amx.tile_store %[[ACC_BUF]][%c48{{.*}}, %c16{{.*}}], %[[RES_3_1]] : memref<64x32xf32>, !amx.tile<16x16xf32>
// CHECK:       %[[RES:.+]] = vector.transfer_read %[[ACC_BUF]][%c0{{.*}}, %c0{{.*}}], %{{.*}} {in_bounds = [true, true]} : memref<64x32xf32>, vector<64x32xf32>

#loc = loc(unknown)
module {
  tt.func public @test_loop_acc_two_blocks(%arg0: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} loc(unknown), %arg1: !tt.ptr<f8E5M2> {tt.divisibility = 16 : i32} loc(unknown), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc(unknown)) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : f8E5M2 loc(#loc)
    %c2_i32 = arith.constant 2 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %c64_i32 = arith.constant 64 : i32 loc(#loc)
    %cst_0 = arith.constant dense<0.000000e+00> : vector<64x32xf32> loc(#loc)
    %c32_i64 = arith.constant 32 : i64 loc(#loc)
    %c64_i64 = arith.constant 64 : i64 loc(#loc)
    %c128_i64 = arith.constant 128 : i64 loc(#loc)
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %0 = tt.make_tensor_ptr %arg0, [%c64_i64, %c128_i64], [%c128_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x64xf8E5M2>> loc(#loc)
    %1 = tt.make_tensor_ptr %arg1, [%c128_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf8E5M2>> loc(#loc)
    %2 = tt.make_tensor_ptr %arg2, [%c64_i64, %c32_i64], [%c32_i64, %c1_i64], [%c0_i32, %c0_i32] {order = array<i32: 1, 0>} : <tensor<64x32xf32>> loc(#loc)
    %3:3 = scf.for %arg3 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg4 = %cst_0, %arg5 = %0, %arg6 = %1) -> (vector<64x32xf32>, !tt.ptr<tensor<64x64xf8E5M2>>, !tt.ptr<tensor<64x32xf8E5M2>>)  : i32 {
      %6 = triton_cpu.extract_memref %arg5 : <tensor<64x64xf8E5M2>> -> memref<64x128xf8E5M2, strided<[128, 1]>> loc(#loc)
      %7:2 = triton_cpu.extract_indices %arg5 : <tensor<64x64xf8E5M2>> -> index, index loc(#loc)
      %8 = vector.transfer_read %6[%7#0, %7#1], %cst {in_bounds = [true, true]} : memref<64x128xf8E5M2, strided<[128, 1]>>, vector<64x64xf8E5M2> loc(#loc)
      %9 = triton_cpu.extract_memref %arg6 : <tensor<64x32xf8E5M2>> -> memref<128x32xf8E5M2, strided<[32, 1]>> loc(#loc)
      %10:2 = triton_cpu.extract_indices %arg6 : <tensor<64x32xf8E5M2>> -> index, index loc(#loc)
      %11 = vector.transfer_read %9[%10#0, %10#1], %cst {in_bounds = [true, true]} : memref<128x32xf8E5M2, strided<[32, 1]>>, vector<64x32xf8E5M2> loc(#loc)
      %12 = triton_cpu.dot %8, %11, %arg4, inputPrecision = ieee : vector<64x64xf8E5M2> * vector<64x32xf8E5M2> -> vector<64x32xf32> loc(#loc)
      %13 = tt.advance %arg5, [%c0_i32, %c64_i32] : <tensor<64x64xf8E5M2>> loc(#loc)
      %14 = tt.advance %arg6, [%c64_i32, %c0_i32] : <tensor<64x32xf8E5M2>> loc(#loc)
      scf.yield %12, %13, %14 : vector<64x32xf32>, !tt.ptr<tensor<64x64xf8E5M2>>, !tt.ptr<tensor<64x32xf8E5M2>> loc(#loc)
    } loc(#loc)
    %4 = triton_cpu.extract_memref %2 : <tensor<64x32xf32>> -> memref<64x32xf32, strided<[32, 1]>> loc(#loc)
    %5:2 = triton_cpu.extract_indices %2 : <tensor<64x32xf32>> -> index, index loc(#loc)
    vector.transfer_write %3#0, %4[%5#0, %5#1] {in_bounds = [true, true]} : vector<64x32xf32>, memref<64x32xf32, strided<[32, 1]>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

// -----

// A case with VNNI pre-encoded RHS that can be directly accessed from the input memory.
// We expect both LHS and RHS tiles to be directly loaded from the input mmemory.

// CHECK-LABEL: @test_loop_pre_encoded_direct
// CHECK:       %[[LHS_MEMREF:.+]] = triton_cpu.extract_memref
// CHECK-NEXT:  %[[LHS_INDICES:.+]]:4 = triton_cpu.extract_indices
// CHECK:       %[[RHS_MEMREF:.+]] = triton_cpu.extract_memref
// CHECK-NEXT:  %[[RHS_INDICES:.+]]:4 = triton_cpu.extract_indices
// CHECK:       amx.tile_load %[[LHS_MEMREF]][%[[LHS_INDICES]]#0, %[[LHS_INDICES]]#1, %[[LHS_INDICES]]#2, %[[LHS_INDICES]]#3]
// CHECK:       amx.tile_load %[[RHS_MEMREF]][%[[RHS_INDICES]]#0, %[[RHS_INDICES]]#1, %[[RHS_INDICES]]#2, %[[RHS_INDICES]]#3]
#loc = loc(unknown)
module {
  tt.func public @test_loop_pre_encoded_direct(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : bf16 loc(#loc)
    %c31_i32 = arith.constant 31 : i32 loc(#loc)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc)
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xf32> loc(#loc)
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %c64_i64 = arith.constant 64 : i64 loc(#loc)
    %c16_i64 = arith.constant 16 : i64 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c32_i64 = arith.constant 32 : i64 loc(#loc)
    %c32_i32 = arith.constant 32 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %0 = tt.get_program_id x : i32 loc(#loc)
    %1 = arith.divsi %arg4, %c32_i32 : i32 loc(#loc)
    %2 = arith.divsi %0, %1 : i32 loc(#loc)
    %3 = arith.remsi %0, %1 : i32 loc(#loc)
    %4 = arith.muli %arg5, %c32_i32 : i32 loc(#loc)
    %5 = arith.divsi %arg3, %c32_i32 : i32 loc(#loc)
    %6 = arith.divsi %arg5, %c32_i32 : i32 loc(#loc)
    %7 = arith.extsi %5 : i32 to i64 loc(#loc)
    %8 = arith.extsi %6 : i32 to i64 loc(#loc)
    %9 = arith.extsi %4 : i32 to i64 loc(#loc)
    %10 = arith.extsi %arg5 : i32 to i64 loc(#loc)
    %11 = tt.make_tensor_ptr %arg0, [%7, %8, %c32_i64, %c32_i64], [%9, %c32_i64, %10, %c1_i64], [%2, %c0_i32, %c0_i32, %c0_i32] {order = array<i32: 3, 2, 1, 0>} : <tensor<1x1x32x32xbf16>> loc(#loc)
    %12 = arith.extsi %1 : i32 to i64 loc(#loc)
    %13 = tt.make_tensor_ptr %arg1, [%8, %12, %c16_i64, %c64_i64], [%c1024_i64, %9, %c64_i64, %c1_i64], [%c0_i32, %3, %c0_i32, %c0_i32] {order = array<i32: 3, 2, 1, 0>} : <tensor<1x1x16x64xbf16>> loc(#loc)
    %14 = arith.muli %2, %c32_i32 : i32 loc(#loc)
    %15 = arith.muli %3, %c32_i32 : i32 loc(#loc)
    %16 = arith.extsi %arg3 : i32 to i64 loc(#loc)
    %17 = arith.extsi %arg4 : i32 to i64 loc(#loc)
    %18 = tt.make_tensor_ptr %arg2, [%16, %17], [%17, %c1_i64], [%14, %15] {order = array<i32: 1, 0>} : <tensor<32x32xf32>> loc(#loc)
    %19 = arith.addi %arg5, %c31_i32 : i32 loc(#loc)
    %20 = arith.divsi %19, %c32_i32 : i32 loc(#loc)
    %21:3 = scf.for %arg6 = %c0_i32 to %20 step %c1_i32 iter_args(%arg7 = %cst_0, %arg8 = %11, %arg9 = %13) -> (vector<32x32xf32>, !tt.ptr<tensor<1x1x32x32xbf16>>, !tt.ptr<tensor<1x1x16x64xbf16>>)  : i32 {
      %24 = triton_cpu.extract_memref %arg8 : <tensor<1x1x32x32xbf16>> -> memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>> loc(#loc)
      %25:4 = triton_cpu.extract_indices %arg8 : <tensor<1x1x32x32xbf16>> -> index, index, index, index loc(#loc)
      %26 = vector.transfer_read %24[%25#0, %25#1, %25#2, %25#3], %cst {in_bounds = [true, true]} : memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>>, vector<32x32xbf16> loc(#loc)
      %27 = triton_cpu.extract_memref %arg9 : <tensor<1x1x16x64xbf16>> -> memref<?x?x16x64xbf16, strided<[1024, ?, 64, 1]>> loc(#loc)
      %28:4 = triton_cpu.extract_indices %arg9 : <tensor<1x1x16x64xbf16>> -> index, index, index, index loc(#loc)
      %29 = vector.transfer_read %27[%28#0, %28#1, %28#2, %28#3], %cst {in_bounds = [true, true]} : memref<?x?x16x64xbf16, strided<[1024, ?, 64, 1]>>, vector<16x64xbf16> loc(#loc)
      %res1, %res2 = vector.deinterleave %29 : vector<16x64xbf16> -> vector<16x32xbf16> loc(#loc)
      %30 = vector.transpose %res1, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16> loc(#loc)
      %31 = vector.transpose %res2, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16> loc(#loc)
      %32 = vector.interleave %30, %31 : vector<32x16xbf16> -> vector<32x32xbf16> loc(#loc)
      %33 = vector.transpose %32, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16> loc(#loc)
      %34 = triton_cpu.dot %26, %33, %arg7, inputPrecision = tf32 : vector<32x32xbf16> * vector<32x32xbf16> -> vector<32x32xf32> loc(#loc)
      %35 = tt.advance %arg8, [%c0_i32, %c1_i32, %c0_i32, %c0_i32] : <tensor<1x1x32x32xbf16>> loc(#loc)
      %36 = tt.advance %arg9, [%c1_i32, %c0_i32, %c0_i32, %c0_i32] : <tensor<1x1x16x64xbf16>> loc(#loc)
      scf.yield %34, %35, %36 : vector<32x32xf32>, !tt.ptr<tensor<1x1x32x32xbf16>>, !tt.ptr<tensor<1x1x16x64xbf16>> loc(#loc)
    } loc(#loc)
    %22 = triton_cpu.extract_memref %18 : <tensor<32x32xf32>> -> memref<?x?xf32, strided<[?, 1]>> loc(#loc)
    %23:2 = triton_cpu.extract_indices %18 : <tensor<32x32xf32>> -> index, index loc(#loc)
    vector.transfer_write %21#0, %22[%23#0, %23#1] {in_bounds = [true, true]} : vector<32x32xf32>, memref<?x?xf32, strided<[?, 1]>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

// -----

// A case with VNNI pre-encoded RHS that cannot be directly accessed from the input memory.
// We expect LHS to be directly loaded from the input mmemory and RHS to be loaded through
// a temporary buffer without additional encoding.


// CHECK-LABEL: @test_loop_pre_encoded_indirect
// CHECK:       %[[RHS_BUF:.+]] = memref.alloca() {alignment = 64 : i64} : memref<16x64xbf16>
// CHECK:       %[[LHS_MEMREF:.+]] = triton_cpu.extract_memref
// CHECK-NEXT:  %[[LHS_INDICES:.+]]:4 = triton_cpu.extract_indices
// CHECK:       %[[RHS_MEMREF:.+]] = triton_cpu.extract_memref
// CHECK-NEXT:  %[[RHS_INDICES:.+]]:4 = triton_cpu.extract_indices
// CHECK-NEXT:  %[[RHS:.+]] = vector.transfer_read %[[RHS_MEMREF]][%[[RHS_INDICES]]#0, %[[RHS_INDICES]]#1, %[[RHS_INDICES]]#2, %[[RHS_INDICES]]#3]
// CHECK:       vector.transfer_write %[[RHS]], %[[RHS_BUF]][%c0, %c0] {in_bounds = [true, true]}
// CHECK:       amx.tile_load %[[LHS_MEMREF]][%[[LHS_INDICES]]#0, %[[LHS_INDICES]]#1, %[[LHS_INDICES]]#2, %[[LHS_INDICES]]#3]
// CHECK:       amx.tile_load %[[RHS_BUF]][%c0, %c0]
#loc = loc(unknown)
module {
  tt.func public @test_loop_pre_encoded_indirect(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant 0.000000e+00 : bf16 loc(#loc)
    %c31_i32 = arith.constant 31 : i32 loc(#loc)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc)
    %cst_0 = arith.constant dense<0.000000e+00> : vector<32x32xf32> loc(#loc)
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %c64_i64 = arith.constant 64 : i64 loc(#loc)
    %c16_i64 = arith.constant 16 : i64 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c32_i64 = arith.constant 32 : i64 loc(#loc)
    %c32_i32 = arith.constant 32 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %0 = tt.get_program_id x : i32 loc(#loc)
    %1 = arith.divsi %arg4, %c32_i32 : i32 loc(#loc)
    %2 = arith.divsi %0, %1 : i32 loc(#loc)
    %3 = arith.remsi %0, %1 : i32 loc(#loc)
    %4 = arith.muli %arg5, %c32_i32 : i32 loc(#loc)
    %5 = arith.divsi %arg3, %c32_i32 : i32 loc(#loc)
    %6 = arith.divsi %arg5, %c32_i32 : i32 loc(#loc)
    %7 = arith.extsi %5 : i32 to i64 loc(#loc)
    %8 = arith.extsi %6 : i32 to i64 loc(#loc)
    %9 = arith.extsi %4 : i32 to i64 loc(#loc)
    %10 = arith.extsi %arg5 : i32 to i64 loc(#loc)
    %11 = tt.make_tensor_ptr %arg0, [%7, %8, %c32_i64, %c32_i64], [%9, %c32_i64, %10, %c1_i64], [%2, %c0_i32, %c0_i32, %c0_i32] {order = array<i32: 3, 2, 1, 0>} : <tensor<1x1x32x32xbf16>> loc(#loc)
    %12 = arith.extsi %1 : i32 to i64 loc(#loc)
    %13 = tt.make_tensor_ptr %arg1, [%8, %12, %c16_i64, %c64_i64], [%c1024_i64, %9, %c64_i64, %c1_i64], [%c0_i32, %3, %c0_i32, %c0_i32] {order = array<i32: 3, 2, 1, 0>} : <tensor<1x1x16x64xbf16>> loc(#loc)
    %14 = arith.muli %2, %c32_i32 : i32 loc(#loc)
    %15 = arith.muli %3, %c32_i32 : i32 loc(#loc)
    %16 = arith.extsi %arg3 : i32 to i64 loc(#loc)
    %17 = arith.extsi %arg4 : i32 to i64 loc(#loc)
    %18 = tt.make_tensor_ptr %arg2, [%16, %17], [%17, %c1_i64], [%14, %15] {order = array<i32: 1, 0>} : <tensor<32x32xf32>> loc(#loc)
    %19 = arith.addi %arg5, %c31_i32 : i32 loc(#loc)
    %20 = arith.divsi %19, %c32_i32 : i32 loc(#loc)
    %21:3 = scf.for %arg6 = %c0_i32 to %20 step %c1_i32 iter_args(%arg7 = %cst_0, %arg8 = %11, %arg9 = %13) -> (vector<32x32xf32>, !tt.ptr<tensor<1x1x32x32xbf16>>, !tt.ptr<tensor<1x1x16x64xbf16>>)  : i32 {
      %24 = triton_cpu.extract_memref %arg8 : <tensor<1x1x32x32xbf16>> -> memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>> loc(#loc)
      %25:4 = triton_cpu.extract_indices %arg8 : <tensor<1x1x32x32xbf16>> -> index, index, index, index loc(#loc)
      %26 = vector.transfer_read %24[%25#0, %25#1, %25#2, %25#3], %cst {in_bounds = [true, true]} : memref<?x?x32x32xbf16, strided<[?, 32, ?, 1]>>, vector<32x32xbf16> loc(#loc)
      %27 = triton_cpu.extract_memref %arg9 : <tensor<1x1x16x64xbf16>> -> memref<?x?x16x64xbf16, strided<[1024, ?, 64, 1]>> loc(#loc)
      %28:4 = triton_cpu.extract_indices %arg9 : <tensor<1x1x16x64xbf16>> -> index, index, index, index loc(#loc)
      %29 = vector.transfer_read %27[%28#0, %28#1, %28#2, %28#3], %cst {in_bounds = [false, false]} : memref<?x?x16x64xbf16, strided<[1024, ?, 64, 1]>>, vector<16x64xbf16> loc(#loc)
      %res1, %res2 = vector.deinterleave %29 : vector<16x64xbf16> -> vector<16x32xbf16> loc(#loc)
      %30 = vector.transpose %res1, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16> loc(#loc)
      %31 = vector.transpose %res2, [1, 0] : vector<16x32xbf16> to vector<32x16xbf16> loc(#loc)
      %32 = vector.interleave %30, %31 : vector<32x16xbf16> -> vector<32x32xbf16> loc(#loc)
      %33 = vector.transpose %32, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16> loc(#loc)
      %34 = triton_cpu.dot %26, %33, %arg7, inputPrecision = tf32 : vector<32x32xbf16> * vector<32x32xbf16> -> vector<32x32xf32> loc(#loc)
      %35 = tt.advance %arg8, [%c0_i32, %c1_i32, %c0_i32, %c0_i32] : <tensor<1x1x32x32xbf16>> loc(#loc)
      %36 = tt.advance %arg9, [%c1_i32, %c0_i32, %c0_i32, %c0_i32] : <tensor<1x1x16x64xbf16>> loc(#loc)
      scf.yield %34, %35, %36 : vector<32x32xf32>, !tt.ptr<tensor<1x1x32x32xbf16>>, !tt.ptr<tensor<1x1x16x64xbf16>> loc(#loc)
    } loc(#loc)
    %22 = triton_cpu.extract_memref %18 : <tensor<32x32xf32>> -> memref<?x?xf32, strided<[?, 1]>> loc(#loc)
    %23:2 = triton_cpu.extract_indices %18 : <tensor<32x32xf32>> -> index, index loc(#loc)
    vector.transfer_write %21#0, %22[%23#0, %23#1] {in_bounds = [true, true]} : vector<32x32xf32>, memref<?x?xf32, strided<[?, 1]>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)

// -----

// A case with int8 VNNI pre-encoded RHS that can be directly accessed from the input memory.
// We expect both LHS and RHS tiles to be directly loaded from the input mmemory.

// CHECK-LABEL: @test_loop_int8_pre_encoded
// CHECK:       %[[LHS_MEMREF:.+]] = triton_cpu.extract_memref
// CHECK-NEXT:  %[[LHS_INDICES:.+]]:4 = triton_cpu.extract_indices
// CHECK:       %[[RHS_MEMREF:.+]] = triton_cpu.extract_memref
// CHECK-NEXT:  %[[RHS_INDICES:.+]]:4 = triton_cpu.extract_indices
// CHECK:       amx.tile_load %[[LHS_MEMREF]][%[[LHS_INDICES]]#0, %[[LHS_INDICES]]#1, %[[LHS_INDICES]]#2, %[[LHS_INDICES]]#3]
// CHECK:       amx.tile_load %[[RHS_MEMREF]][%[[RHS_INDICES]]#0, %[[RHS_INDICES]]#1, %[[RHS_INDICES]]#2, %[[RHS_INDICES]]#3]
#loc = loc(unknown)
module {
  tt.func public @test_loop_int8_pre_encoded(%arg0: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32 {tt.divisibility = 16 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i8 = arith.constant 0 : i8 loc(#loc)
    %c31_i32 = arith.constant 31 : i32 loc(#loc)
    %c1024_i64 = arith.constant 1024 : i64 loc(#loc)
    %cst = arith.constant dense<0> : vector<32x32xi32> loc(#loc)
    %c1_i64 = arith.constant 1 : i64 loc(#loc)
    %c128_i64 = arith.constant 128 : i64 loc(#loc)
    %c8_i64 = arith.constant 8 : i64 loc(#loc)
    %c0_i32 = arith.constant 0 : i32 loc(#loc)
    %c32_i64 = arith.constant 32 : i64 loc(#loc)
    %c32_i32 = arith.constant 32 : i32 loc(#loc)
    %c1_i32 = arith.constant 1 : i32 loc(#loc)
    %0 = tt.get_program_id x : i32 loc(#loc)
    %1 = arith.divsi %arg4, %c32_i32 : i32 loc(#loc)
    %2 = arith.divsi %0, %1 : i32 loc(#loc)
    %3 = arith.remsi %0, %1 : i32 loc(#loc)
    %4 = arith.muli %arg5, %c32_i32 : i32 loc(#loc)
    %5 = arith.divsi %arg3, %c32_i32 : i32 loc(#loc)
    %6 = arith.divsi %arg5, %c32_i32 : i32 loc(#loc)
    %7 = arith.extsi %5 : i32 to i64 loc(#loc)
    %8 = arith.extsi %6 : i32 to i64 loc(#loc)
    %9 = arith.extsi %4 : i32 to i64 loc(#loc)
    %10 = arith.extsi %arg5 : i32 to i64 loc(#loc)
    %11 = tt.make_tensor_ptr %arg0, [%7, %8, %c32_i64, %c32_i64], [%9, %c32_i64, %10, %c1_i64], [%2, %c0_i32, %c0_i32, %c0_i32] {order = array<i32: 3, 2, 1, 0>} : <tensor<1x1x32x32xi8>> loc(#loc)
    %12 = arith.extsi %1 : i32 to i64 loc(#loc)
    %13 = tt.make_tensor_ptr %arg1, [%8, %12, %c8_i64, %c128_i64], [%c1024_i64, %9, %c128_i64, %c1_i64], [%c0_i32, %3, %c0_i32, %c0_i32] {order = array<i32: 3, 2, 1, 0>} : <tensor<1x1x8x128xi8>> loc(#loc)
    %14 = arith.muli %2, %c32_i32 : i32 loc(#loc)
    %15 = arith.muli %3, %c32_i32 : i32 loc(#loc)
    %16 = arith.extsi %arg3 : i32 to i64 loc(#loc)
    %17 = arith.extsi %arg4 : i32 to i64 loc(#loc)
    %18 = tt.make_tensor_ptr %arg2, [%16, %17], [%17, %c1_i64], [%14, %15] {order = array<i32: 1, 0>} : <tensor<32x32xi32>> loc(#loc)
    %19 = arith.addi %arg5, %c31_i32 : i32 loc(#loc)
    %20 = arith.divsi %19, %c32_i32 : i32 loc(#loc)
    %21:3 = scf.for %arg6 = %c0_i32 to %20 step %c1_i32 iter_args(%arg7 = %cst, %arg8 = %11, %arg9 = %13) -> (vector<32x32xi32>, !tt.ptr<tensor<1x1x32x32xi8>>, !tt.ptr<tensor<1x1x8x128xi8>>)  : i32 {
      %24 = triton_cpu.extract_memref %arg8 : <tensor<1x1x32x32xi8>> -> memref<?x?x32x32xi8, strided<[?, 32, ?, 1]>> loc(#loc)
      %25:4 = triton_cpu.extract_indices %arg8 : <tensor<1x1x32x32xi8>> -> index, index, index, index loc(#loc)
      %26 = vector.transfer_read %24[%25#0, %25#1, %25#2, %25#3], %c0_i8 {in_bounds = [true, true]} : memref<?x?x32x32xi8, strided<[?, 32, ?, 1]>>, vector<32x32xi8> loc(#loc)
      %27 = triton_cpu.extract_memref %arg9 : <tensor<1x1x8x128xi8>> -> memref<?x?x8x128xi8, strided<[1024, ?, 128, 1]>> loc(#loc)
      %28:4 = triton_cpu.extract_indices %arg9 : <tensor<1x1x8x128xi8>> -> index, index, index, index loc(#loc)
      %30 = vector.transfer_read %27[%28#0, %28#1, %28#2, %28#3], %c0_i8 {in_bounds = [true, true]} : memref<?x?x8x128xi8, strided<[1024, ?, 128, 1]>>, vector<8x128xi8> loc(#loc)
      %res1, %res2 = vector.deinterleave %30 : vector<8x128xi8> -> vector<8x64xi8> loc(#loc)
      %31 = vector.transpose %res1, [1, 0] : vector<8x64xi8> to vector<64x8xi8> loc(#loc)
      %32 = vector.transpose %res2, [1, 0] : vector<8x64xi8> to vector<64x8xi8> loc(#loc)
      %33 = vector.interleave %31, %32 : vector<64x8xi8> -> vector<64x16xi8> loc(#loc)
      %34 = vector.transpose %33, [1, 0] : vector<64x16xi8> to vector<16x64xi8> loc(#loc)
      %res1_0, %res2_1 = vector.deinterleave %34 : vector<16x64xi8> -> vector<16x32xi8> loc(#loc)
      %35 = vector.transpose %res1_0, [1, 0] : vector<16x32xi8> to vector<32x16xi8> loc(#loc)
      %36 = vector.transpose %res2_1, [1, 0] : vector<16x32xi8> to vector<32x16xi8> loc(#loc)
      %37 = vector.interleave %35, %36 : vector<32x16xi8> -> vector<32x32xi8> loc(#loc)
      %38 = vector.transpose %37, [1, 0] : vector<32x32xi8> to vector<32x32xi8> loc(#loc)
      %39 = triton_cpu.dot %26, %38, %arg7, inputPrecision = tf32 : vector<32x32xi8> * vector<32x32xi8> -> vector<32x32xi32> loc(#loc)
      %40 = tt.advance %arg8, [%c0_i32, %c1_i32, %c0_i32, %c0_i32] : <tensor<1x1x32x32xi8>> loc(#loc)
      %41 = tt.advance %arg9, [%c1_i32, %c0_i32, %c0_i32, %c0_i32] : <tensor<1x1x8x128xi8>> loc(#loc)
      scf.yield %39, %40, %41 : vector<32x32xi32>, !tt.ptr<tensor<1x1x32x32xi8>>, !tt.ptr<tensor<1x1x8x128xi8>> loc(#loc)
    } loc(#loc)
    %22 = triton_cpu.extract_memref %18 : <tensor<32x32xi32>> -> memref<?x?xi32, strided<[?, 1]>> loc(#loc)
    %23:2 = triton_cpu.extract_indices %18 : <tensor<32x32xi32>> -> index, index loc(#loc)
    vector.transfer_write %21#0, %22[%23#0, %23#1] {in_bounds = [true, true]} : vector<32x32xi32>, memref<?x?xi32, strided<[?, 1]>> loc(#loc)
    tt.return loc(#loc)
  } loc(#loc)
} loc(#loc)
