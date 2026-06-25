// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=amx-bf16,amx-int8 -cse  | FileCheck %s --check-prefixes=AMX,ALL
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=avx512bf16 -cse  | FileCheck %s --check-prefixes=AVX512,ALL
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=avx10.2 -cse  | FileCheck %s --check-prefixes=AVX10_2,ALL
// RUN: triton-opt %s -split-input-file -triton-cpu-convert-dot-to-nanokernel=cpu-features=avxneconvert -cse  | FileCheck %s --check-prefixes=AVX_NE_CONVERT,ALL

// ALL-LABEL: amx_single_mulf
// AMX: x86.amx.tile_mulf

tt.func public @amx_single_mulf(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>) {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant dense<0.000000e+00> : vector<16x16xf32>
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %c32_i64 = arith.constant 32 : i64
  %0 = tt.make_tensor_descriptor %arg0, [%c16_i32, %c32_i32], [%c32_i64, %c1_i64] : <bf16>, <16x32xbf16>
  %1 = tt.make_tensor_descriptor %arg1, [%c32_i32, %c16_i32], [%c16_i64, %c1_i64] : <bf16>, <32x16xbf16>
  %2 = tt.make_tensor_descriptor %arg2, [%c16_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <16x16xf32>
  %3 = triton_cpu.extract_memref %0 : <16x32xbf16> -> memref<16x32xbf16, strided<[32, 1]>>
  %5 = vector.transfer_read %3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<16x32xbf16, strided<[32, 1]>>, vector<16x32xbf16>
  %6 = triton_cpu.extract_memref %1 : <32x16xbf16> -> memref<32x16xbf16, strided<[16, 1]>>
  %8 = vector.transfer_read %6[%c0, %c0], %cst {in_bounds = [true, true]} : memref<32x16xbf16, strided<[16, 1]>>, vector<32x16xbf16>
  %9 = triton_cpu.dot %5, %8, %cst_0, inputPrecision = ieee : vector<16x32xbf16> * vector<32x16xbf16> -> vector<16x16xf32>
  %10 = triton_cpu.extract_memref %2 : <16x16xf32> -> memref<16x16xf32, strided<[16, 1]>>
  vector.transfer_write %9, %10[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xf32>, memref<16x16xf32, strided<[16, 1]>>
  tt.return
}

// -----

// ALL-LABEL: amx_single_muli
// AMX: x86.amx.tile_muli

tt.func public @amx_single_muli(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i32>) {
  %c0_i8 = arith.constant 0 : i8
  %cst = arith.constant dense<0> : vector<16x16xi32>
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c64_i32 = arith.constant 64 : i32
  %c16_i64 = arith.constant 16 : i64
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %0 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <i8>, <16x64xi8>
  %1 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c16_i32], [%c16_i64, %c1_i64] : <i8>, <64x16xi8>
  %2 = tt.make_tensor_descriptor %arg2, [%c64_i32, %c16_i32], [%c16_i64, %c1_i64] : <i32>, <64x16xi32>
  %3 = triton_cpu.extract_memref %0 : <16x64xi8> -> memref<16x64xi8, strided<[64, 1]>>
  %5 = vector.transfer_read %3[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<16x64xi8, strided<[64, 1]>>, vector<16x64xi8>
  %6 = triton_cpu.extract_memref %1 : <64x16xi8> -> memref<64x16xi8, strided<[16, 1]>>
  %8 = vector.transfer_read %6[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<64x16xi8, strided<[16, 1]>>, vector<64x16xi8>
  %9 = triton_cpu.dot %5, %8, %cst, inputPrecision = ieee : vector<16x64xi8> * vector<64x16xi8> -> vector<16x16xi32>
  %10 = triton_cpu.extract_memref %2 : <64x16xi32> -> memref<64x16xi32, strided<[16, 1]>>
  vector.transfer_write %9, %10[%c0, %c0] {in_bounds = [true, true]} : vector<16x16xi32>, memref<64x16xi32, strided<[16, 1]>>
  tt.return
}

// -----

// ALL-LABEL: amx_single_muli_vnni
// AMX: x86.amx.tile_muli

tt.func public @amx_single_muli_vnni(%a_ptr: !tt.ptr<i8>, %b_ptr: !tt.ptr<i8>, %c_ptr: !tt.ptr<i32>, %M: i32, %N: i32, %K: i32) {
  %a = arith.constant 0 : i8
  %a_0 = arith.constant 0 : index
  %c = arith.constant 0 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i32 = arith.constant 64 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1_i32 = arith.constant 1 : i32
  %m = tt.get_program_id x : i32
  %n = tt.get_program_id y : i32
  %k = tt.get_program_id z : i32
  %a_desc = arith.extsi %K : i32 to i64
  %a_desc_1 = tt.make_tensor_descriptor %a_ptr, [%c1_i32, %c1_i32, %c16_i32, %c64_i32], [%a_desc, %a_desc, %a_desc, %c1_i64] : <i8>, <1x1x16x64xi8>
  %b_desc = arith.extsi %N : i32 to i64
  %b_desc_2 = tt.make_tensor_descriptor %b_ptr, [%c1_i32, %c1_i32, %c16_i32, %c64_i32], [%b_desc, %b_desc, %b_desc, %c1_i64] : <i8>, <1x1x16x64xi8>
  %c_desc = tt.make_tensor_descriptor %c_ptr, [%M, %N], [%b_desc, %c1_i64] : <i32>, <16x16xi32>
  %c_3 = arith.muli %m, %c16_i32 : i32
  %c_4 = arith.muli %n, %c16_i32 : i32
  %c_desc_5 = triton_cpu.extract_memref %c_desc : <16x16xi32> -> memref<?x?xi32, strided<[?, 1]>>
  %c_6 = arith.index_cast %c_3 : i32 to index
  %c_7 = arith.index_cast %c_4 : i32 to index
  %c_8 = vector.transfer_read %c_desc_5[%c_6, %c_7], %c {in_bounds = [true, true]} : memref<?x?xi32, strided<[?, 1]>>, vector<16x16xi32>
  %a_desc_9 = triton_cpu.extract_memref %a_desc_1 : <1x1x16x64xi8> -> memref<1x1x16x64xi8, strided<[?, ?, ?, 1]>>
  %a_10 = arith.index_cast %m : i32 to index
  %a_11 = arith.index_cast %k : i32 to index
  %a_12 = vector.transfer_read %a_desc_9[%a_10, %a_11, %a_0, %a_0], %a {in_bounds = [true, true]} : memref<1x1x16x64xi8, strided<[?, ?, ?, 1]>>, vector<16x64xi8>
  %b_desc_13 = triton_cpu.extract_memref %b_desc_2 : <1x1x16x64xi8> -> memref<1x1x16x64xi8, strided<[?, ?, ?, 1]>>
  %b = arith.index_cast %n : i32 to index
  %b_14 = vector.transfer_read %b_desc_13[%a_11, %b, %a_0, %a_0], %a {in_bounds = [true, true]} : memref<1x1x16x64xi8, strided<[?, ?, ?, 1]>>, vector<16x64xi8>
  %b_16, %b_17 = vector.deinterleave %b_14 : vector<16x64xi8> -> vector<16x32xi8>
  %b_18 = vector.transpose %b_16, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
  %b_19 = vector.transpose %b_17, [1, 0] : vector<16x32xi8> to vector<32x16xi8>
  %b_20 = vector.interleave %b_18, %b_19 : vector<32x16xi8> -> vector<32x32xi8>
  %b_21 = vector.transpose %b_20, [1, 0] : vector<32x32xi8> to vector<32x32xi8>
  %b_22, %b_23 = vector.deinterleave %b_21 : vector<32x32xi8> -> vector<32x16xi8>
  %b_24 = vector.transpose %b_22, [1, 0] : vector<32x16xi8> to vector<16x32xi8>
  %b_25 = vector.transpose %b_23, [1, 0] : vector<32x16xi8> to vector<16x32xi8>
  %b_26 = vector.interleave %b_24, %b_25 : vector<16x32xi8> -> vector<16x64xi8>
  %b_27 = vector.transpose %b_26, [1, 0] : vector<16x64xi8> to vector<64x16xi8>
  %c_28 = triton_cpu.dot %a_12, %b_27, %c_8, inputPrecision = tf32 : vector<16x64xi8> * vector<64x16xi8> -> vector<16x16xi32>
  vector.transfer_write %c_28, %c_desc_5[%c_6, %c_7] {in_bounds = [true, true]} : vector<16x16xi32>, memref<?x?xi32, strided<[?, 1]>>
  tt.return
}

// -----

// ALL-LABEL: avx512_pair_dot
// AVX512-COUNT-2: x86.avx512.dot

tt.func public @avx512_pair_dot(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>) {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1x32xf32>
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c2_i64 = arith.constant 2 : i64
  %c32_i64 = arith.constant 32 : i64
  %0 = tt.make_tensor_descriptor %arg0, [%c1_i32, %c2_i32], [%c2_i64, %c1_i64] : <bf16>, <1x2xbf16>
  %1 = tt.make_tensor_descriptor %arg1, [%c2_i32, %c32_i32], [%c32_i64, %c1_i64] : <bf16>, <2x32xbf16>
  %2 = tt.make_tensor_descriptor %arg2, [%c1_i32, %c32_i32], [%c32_i64, %c1_i64] : <f32>, <1x32xf32>
  %3 = triton_cpu.extract_memref %0 : <1x2xbf16> -> memref<1x2xbf16, strided<[2, 1]>>
  %5 = vector.transfer_read %3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x2xbf16, strided<[2, 1]>>, vector<1x2xbf16>
  %6 = triton_cpu.extract_memref %1 : <2x32xbf16> -> memref<2x32xbf16, strided<[32, 1]>>
  %8 = vector.transfer_read %6[%c0, %c0], %cst {in_bounds = [true, true]} : memref<2x32xbf16, strided<[32, 1]>>, vector<2x32xbf16>
  %9 = triton_cpu.dot %5, %8, %cst_0, inputPrecision = ieee : vector<1x2xbf16> * vector<2x32xbf16> -> vector<1x32xf32>
  %10 = triton_cpu.extract_memref %2 : <1x32xf32> -> memref<1x32xf32, strided<[32, 1]>>
  vector.transfer_write %9, %10[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xf32>, memref<1x32xf32, strided<[32, 1]>>
  tt.return
}

// -----

// ALL-LABEL: avx10_2_pair_dot
// AVX10_2-COUNT-2: x86.avx10.dot.i8

tt.func public @avx10_2_pair_dot(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i32>) {
  %cst = arith.constant 0 : i8
  %cst_0 = arith.constant dense<0> : vector<1x32xi32>
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c32_i64 = arith.constant 32 : i64
  %0 = tt.make_tensor_descriptor %arg0, [%c1_i32, %c4_i32], [%c4_i64, %c1_i64] : <i8>, <1x4xi8>
  %1 = tt.make_tensor_descriptor %arg1, [%c4_i32, %c32_i32], [%c32_i64, %c1_i64] : <i8>, <4x32xi8>
  %2 = tt.make_tensor_descriptor %arg2, [%c1_i32, %c32_i32], [%c32_i64, %c1_i64] : <i32>, <1x32xi32>
  %3 = triton_cpu.extract_memref %0 : <1x4xi8> -> memref<1x4xi8, strided<[4, 1]>>
  %5 = vector.transfer_read %3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x4xi8, strided<[4, 1]>>, vector<1x4xi8>
  %6 = triton_cpu.extract_memref %1 : <4x32xi8> -> memref<4x32xi8, strided<[32, 1]>>
  %8 = vector.transfer_read %6[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x32xi8, strided<[32, 1]>>, vector<4x32xi8>
  %9 = triton_cpu.dot %5, %8, %cst_0, inputPrecision = ieee : vector<1x4xi8> * vector<4x32xi8> -> vector<1x32xi32>
  %10 = triton_cpu.extract_memref %2 : <1x32xi32> -> memref<1x32xi32, strided<[32, 1]>>
  vector.transfer_write %9, %10[%c0, %c0] {in_bounds = [true, true]} : vector<1x32xi32>, memref<1x32xi32, strided<[32, 1]>>
  tt.return
}

// -----

// ALL-LABEL: avxneconvert_pair_fma
// AVX_NE_CONVERT-COUNT-2: vector.fma

tt.func public @avxneconvert_pair_fma(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>) {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant dense<0.000000e+00> : vector<1x16xf32>
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %c1_i64 = arith.constant 1 : i64
  %c16_i64 = arith.constant 16 : i64
  %0 = tt.make_tensor_descriptor %arg0, [%c1_i32, %c1_i32], [%c1_i64, %c1_i64] : <bf16>, <1x1xbf16>
  %1 = tt.make_tensor_descriptor %arg1, [%c1_i32, %c16_i32], [%c16_i64, %c1_i64] : <bf16>, <1x16xbf16>
  %2 = tt.make_tensor_descriptor %arg2, [%c1_i32, %c16_i32], [%c16_i64, %c1_i64] : <f32>, <1x16xf32>
  %3 = triton_cpu.extract_memref %0 : <1x1xbf16> -> memref<1x1xbf16, strided<[1, 1]>>
  %5 = vector.transfer_read %3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x1xbf16, strided<[1, 1]>>, vector<1x1xbf16>
  %6 = triton_cpu.extract_memref %1 : <1x16xbf16> -> memref<1x16xbf16, strided<[16, 1]>>
  %8 = vector.transfer_read %6[%c0, %c0], %cst {in_bounds = [true, true]} : memref<1x16xbf16, strided<[16, 1]>>, vector<1x16xbf16>
  %9 = triton_cpu.dot %5, %8, %cst_0, inputPrecision = ieee : vector<1x1xbf16> * vector<1x16xbf16> -> vector<1x16xf32>
  %10 = triton_cpu.extract_memref %2 : <1x16xf32> -> memref<1x16xf32, strided<[16, 1]>>
  vector.transfer_write %9, %10[%c0, %c0] {in_bounds = [true, true]} : vector<1x16xf32>, memref<1x16xf32, strided<[16, 1]>>
  tt.return
}
