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

// ALL-LABEL: amx_four_muli
// AMX-COUNT-4: x86.amx.tile_muli

tt.func public @amx_four_muli(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i32>) {
  %c0_i8 = arith.constant 0 : i8
  %cst = arith.constant dense<0> : vector<64x16xi32>
  %c0 = arith.constant 0 : index
  %c16_i32 = arith.constant 16 : i32
  %c64_i32 = arith.constant 64 : i32
  %c16_i64 = arith.constant 16 : i64
  %c1_i64 = arith.constant 1 : i64
  %0 = tt.make_tensor_descriptor %arg0, [%c64_i32, %c64_i32], [%c64_i64, %c1_i64] : <i8>, <64x64xi8>
  %1 = tt.make_tensor_descriptor %arg1, [%c64_i32, %c16_i32], [%c16_i64, %c1_i64] : <i8>, <64x16xi8>
  %2 = tt.make_tensor_descriptor %arg2, [%c64_i32, %c16_i32], [%c16_i64, %c1_i64] : <i32>, <64x16xi32>
  %3 = triton_cpu.extract_memref %0 : <64x64xi8> -> memref<64x64xi8, strided<[64, 1]>>
  %5 = vector.transfer_read %3[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<64x64xi8, strided<[64, 1]>>, vector<64x64xi8>
  %6 = triton_cpu.extract_memref %1 : <64x16xi8> -> memref<64x16xi8, strided<[16, 1]>>
  %8 = vector.transfer_read %6[%c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<64x16xi8, strided<[16, 1]>>, vector<64x16xi8>
  %9 = triton_cpu.dot %5, %8, %cst, inputPrecision = ieee : vector<64x64xi8> * vector<64x16xi8> -> vector<64x16xi32>
  %10 = triton_cpu.extract_memref %2 : <64x16xi32> -> memref<64x16xi32, strided<[16, 1]>>
  vector.transfer_write %9, %10[%c0, %c0] {in_bounds = [true, true]} : vector<64x16xi32>, memref<64x16xi32, strided<[16, 1]>>
  tt.return
}

// -----

// ALL-LABEL: amx_four_muli_vnni
// AMX-COUNT-4: x86.amx.tile_muli

tt.func public @amx_four_muli_vnni(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i32>) {
  %c0_i8 = arith.constant 0 : i8
  %cst = arith.constant dense<0> : vector<32x32xi32>
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c16_i32 = arith.constant 16 : i32
  %c32_i32 = arith.constant 32 : i32
  %c64_i32 = arith.constant 64 : i32
  %c128_i32 = arith.constant 128 : i32
  %c1_i64 = arith.constant 1 : i64
  %c32_i64 = arith.constant 32 : i64
  %c64_i64 = arith.constant 64 : i64
  %c128_i64 = arith.constant 128 : i64
  %0 = tt.make_tensor_descriptor %arg0, [%c1_i32, %c32_i32, %c64_i32], [%c64_i64, %c64_i64, %c1_i64] : <i8>, <1x32x64xi8>
  %1 = tt.make_tensor_descriptor %arg1, [%c1_i32, %c16_i32, %c128_i32], [%c64_i64, %c128_i64, %c1_i64] : <i8>, <1x16x128xi8>
  %2 = tt.make_tensor_descriptor %arg2, [%c32_i32, %c32_i32], [%c32_i64, %c1_i64] : <i32>, <32x32xi32>
  %3 = triton_cpu.extract_memref %0 : <1x32x64xi8> -> memref<1x32x64xi8, strided<[64, 64, 1]>>
  %5 = vector.transfer_read %3[%c0, %c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<1x32x64xi8, strided<[64, 64, 1]>>, vector<32x64xi8>
  %6 = triton_cpu.extract_memref %1 : <1x16x128xi8> -> memref<1x16x128xi8, strided<[128, 128, 1]>>
  %8 = vector.transfer_read %6[%c0, %c0, %c0], %c0_i8 {in_bounds = [true, true]} : memref<1x16x128xi8, strided<[128, 128, 1]>>, vector<16x128xi8>
  %res1, %res2 = vector.deinterleave %8 : vector<16x128xi8> -> vector<16x64xi8>
  %33 = vector.transpose %res1, [1, 0] : vector<16x64xi8> to vector<64x16xi8>
  %34 = vector.transpose %res2, [1, 0] : vector<16x64xi8> to vector<64x16xi8>
  %35 = vector.interleave %33, %34 : vector<64x16xi8> -> vector<64x32xi8>
  %36 = vector.transpose %35, [1, 0] : vector<64x32xi8> to vector<32x64xi8>
  %res1_0, %res2_1 = vector.deinterleave %36 : vector<32x64xi8> -> vector<32x32xi8>
  %37 = vector.transpose %res1_0, [1, 0] : vector<32x32xi8> to vector<32x32xi8>
  %38 = vector.transpose %res2_1, [1, 0] : vector<32x32xi8> to vector<32x32xi8>
  %39 = vector.interleave %37, %38 : vector<32x32xi8> -> vector<32x64xi8>
  %40 = vector.transpose %39, [1, 0] : vector<32x64xi8> to vector<64x32xi8>
  %9 = triton_cpu.dot %5, %40, %cst, inputPrecision = ieee : vector<32x64xi8> * vector<64x32xi8> -> vector<32x32xi32>
  %10 = triton_cpu.extract_memref %2 : <32x32xi32> -> memref<32x32xi32, strided<[32, 1]>>
  vector.transfer_write %9, %10[%c0, %c0] {in_bounds = [true, true]} : vector<32x32xi32>, memref<32x32xi32, strided<[32, 1]>>
  tt.return
}

// -----

// ALL-LABEL: avx512_two_dot
// AVX512-COUNT-2: x86.avx512.dot

tt.func public @avx512_two_dot(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>) {
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

// ALL-LABEL: avx10_2_eight_dot
// AVX10_2-COUNT-8: x86.avx10.dot.i8

tt.func public @avx10_2_eight_dot(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<i32>) {
  %cst = arith.constant 0 : i8
  %cst_0 = arith.constant dense<0> : vector<4x32xi32>
  %c0 = arith.constant 0 : index
  %c1_i32 = arith.constant 1 : i32
  %c4_i32 = arith.constant 4 : i32
  %c32_i32 = arith.constant 32 : i32
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c32_i64 = arith.constant 32 : i64
  %0 = tt.make_tensor_descriptor %arg0, [%c4_i32, %c4_i32], [%c4_i64, %c1_i64] : <i8>, <4x4xi8>
  %1 = tt.make_tensor_descriptor %arg1, [%c4_i32, %c32_i32], [%c32_i64, %c1_i64] : <i8>, <4x32xi8>
  %2 = tt.make_tensor_descriptor %arg2, [%c4_i32, %c32_i32], [%c32_i64, %c1_i64] : <i32>, <4x32xi32>
  %3 = triton_cpu.extract_memref %0 : <4x4xi8> -> memref<4x4xi8, strided<[4, 1]>>
  %5 = vector.transfer_read %3[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x4xi8, strided<[4, 1]>>, vector<4x4xi8>
  %6 = triton_cpu.extract_memref %1 : <4x32xi8> -> memref<4x32xi8, strided<[32, 1]>>
  %8 = vector.transfer_read %6[%c0, %c0], %cst {in_bounds = [true, true]} : memref<4x32xi8, strided<[32, 1]>>, vector<4x32xi8>
  %9 = triton_cpu.dot %5, %8, %cst_0, inputPrecision = ieee : vector<4x4xi8> * vector<4x32xi8> -> vector<4x32xi32>
  %10 = triton_cpu.extract_memref %2 : <4x32xi32> -> memref<4x32xi32, strided<[32, 1]>>
  vector.transfer_write %9, %10[%c0, %c0] {in_bounds = [true, true]} : vector<4x32xi32>, memref<4x32xi32, strided<[32, 1]>>
  tt.return
}

// -----

// ALL-LABEL: avxneconvert_two_fma
// AVX_NE_CONVERT-COUNT-2: vector.fma

tt.func public @avxneconvert_two_fma(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<f32>) {
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
