// RUN: triton-opt %s -split-input-file -triton-cpu-unroll-and-reorder-elementwise-ops=cpu-features=avx512 -canonicalize -cse  | FileCheck %s --check-prefixes=CHECK,AVX512
// RUN: triton-opt %s -split-input-file -triton-cpu-unroll-and-reorder-elementwise-ops=cpu-features=avx2 -canonicalize -cse  | FileCheck %s --check-prefixes=CHECK,AVX2

// CHECK-LABEL: eltwise_kernel
// CHECK:         vector.transfer_read
// CHECK-SAME:    vector<32xbf16>
// CHECK:         vector.transfer_read
// AVX512-SAME:   vector<8x32xbf16>
// AVX2-SAME:     vector<2x32xbf16>
// CHECK:         arith.extf
// AVX512-SAME:   vector<8x32xbf16> to vector<8x32xf32>
// AVX2-SAME:     vector<2x32xbf16> to vector<2x32xf32>
// CHECK:         vector.broadcast
// AVX512-SAME:   vector<32xf32> to vector<8x32xf32>
// AVX2-SAME:     vector<32xf32> to vector<2x32xf32>
// CHECK:         arith.addf
// AVX512-SAME:   vector<8x32xf32>
// AVX2-SAME:     vector<2x32xf32>
// CHECK:         arith.subf
// AVX512-SAME:   vector<8x32xf32>
// AVX2-SAME:     vector<2x32xf32>
// CHECK:         arith.mulf
// AVX512-SAME:   vector<8x32xf32>
// AVX2-SAME:     vector<2x32xf32>
// CHECK:         arith.truncf
// AVX512-SAME:   vector<8x32xf32> to vector<8x32xbf16>
// AVX2-SAME:     vector<2x32xf32> to vector<2x32xbf16>
// CHECK:         vector.transfer_write
// AVX512-SAME:   vector<8x32xbf16>
// AVX2-SAME:     vector<2x32xbf16>

tt.func public @eltwise_kernel(%arg0: !tt.ptr<bf16>, %arg1: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %cst_0 = arith.constant dense<2.000000e+00> : vector<32x32xf32>
  %cst_1 = arith.constant dense<1.500000e+00> : vector<32x32xf32>
  %c1_i64 = arith.constant 1 : i64
  %c32_i32 = arith.constant 32 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c32_i32 : i32
  %2 = tt.get_program_id y : i32
  %3 = arith.muli %2, %c32_i32 : i32
  %4 = arith.extsi %arg4 : i32 to i64
  %5 = tt.make_tensor_descriptor %arg0, [%arg3, %arg4], [%4, %c1_i64] : <bf16>, <32x32xbf16>
  %6 = tt.make_tensor_descriptor %arg1, [%arg4], [%c1_i64] : <bf16>, <32xbf16>
  %7 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%4, %c1_i64] : <bf16>, <32x32xbf16>
  %8 = triton_cpu.extract_memref %5 : <32x32xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
  %9 = arith.index_cast %1 : i32 to index
  %10 = arith.index_cast %3 : i32 to index
  %11 = vector.transfer_read %8[%9, %10], %cst {in_bounds = [true, true]} : memref<?x?xbf16, strided<[?, 1]>>, vector<32x32xbf16>
  %12 = arith.extf %11 : vector<32x32xbf16> to vector<32x32xf32>
  %13 = triton_cpu.extract_memref %6 : <32xbf16> -> memref<?xbf16, strided<[1]>>
  %14 = vector.transfer_read %13[%10], %cst {in_bounds = [true]} : memref<?xbf16, strided<[1]>>, vector<32xbf16>
  %15 = arith.extf %14 : vector<32xbf16> to vector<32xf32>
  %16 = vector.broadcast %15 : vector<32xf32> to vector<32x32xf32>
  %17 = arith.addf %12, %16 : vector<32x32xf32>
  %18 = arith.subf %17, %cst_1 : vector<32x32xf32>
  %19 = arith.mulf %18, %cst_0 : vector<32x32xf32>
  %20 = arith.truncf %19 : vector<32x32xf32> to vector<32x32xbf16>
  %21 = triton_cpu.extract_memref %7 : <32x32xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
  vector.transfer_write %20, %21[%9, %10] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<?x?xbf16, strided<[?, 1]>>
  tt.return
}

// -----

// CHECK-LABEL: eltwise_kernel2
// CHECK:         vector.transfer_read
// AVX512-SAME:   vector<256xf32>
// AVX2-SAME:     vector<64xf32>
// CHECK:         math.exp
// AVX512-SAME:   vector<256xf32>
// AVX2-SAME:     vector<64xf32>
// CHECK:         math.erf
// AVX512-SAME:   vector<256xf32>
// AVX2-SAME:     vector<64xf32>
// CHECK:         arith.addf
// AVX512-SAME:   vector<256xf32>
// AVX2-SAME:     vector<64xf32>
// CHECK:         arith.addf
// AVX512-SAME:   vector<256xf32>
// AVX2-SAME:     vector<64xf32>
// CHECK:         vector.transfer_write
// AVX512-SAME:   vector<256xf32>
// AVX2-SAME:     vector<64xf32>

tt.func public @eltwise_kernel2(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) {
  %cst = arith.constant 0.000000e+00 : f32
  %c1_i64 = arith.constant 1 : i64
  %c4096_i32 = arith.constant 4096 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c4096_i32 : i32
  %2 = tt.make_tensor_descriptor %arg0, [%arg2], [%c1_i64] : <f32>, <4096xf32>
  %3 = tt.make_tensor_descriptor %arg1, [%arg2], [%c1_i64] : <f32>, <4096xf32>
  %4 = triton_cpu.extract_memref %2 : <4096xf32> -> memref<?xf32, strided<[1]>>
  %5 = arith.index_cast %1 : i32 to index
  %6 = vector.transfer_read %4[%5], %cst {in_bounds = [true]} : memref<?xf32, strided<[1]>>, vector<4096xf32>
  %7 = math.exp %6 : vector<4096xf32>
  %8 = math.erf %6 : vector<4096xf32>
  %9 = arith.addf %7, %8 : vector<4096xf32>
  %10 = arith.addf %9, %6 : vector<4096xf32>
  %11 = triton_cpu.extract_memref %3 : <4096xf32> -> memref<?xf32, strided<[1]>>
  vector.transfer_write %10, %11[%5] {in_bounds = [true]} : vector<4096xf32>, memref<?xf32, strided<[1]>>
  tt.return
}

// -----

// Negative test: vector.transpose is not an elementwise operation.

// CHECK-LABEL: negative_eltwise_kernel3
// CHECK:         vector.transpose %{{.+}}, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>

tt.func public @negative_eltwise_kernel3(%arg0: !tt.ptr<bf16>, %arg2: !tt.ptr<bf16>, %arg3: i32, %arg4: i32) {
  %cst = arith.constant 0.000000e+00 : bf16
  %c1_i64 = arith.constant 1 : i64
  %c32_i32 = arith.constant 32 : i32
  %0 = tt.get_program_id x : i32
  %1 = arith.muli %0, %c32_i32 : i32
  %2 = tt.get_program_id y : i32
  %3 = arith.muli %2, %c32_i32 : i32
  %4 = arith.extsi %arg4 : i32 to i64
  %5 = tt.make_tensor_descriptor %arg0, [%arg3, %arg4], [%4, %c1_i64] : <bf16>, <32x32xbf16>
  %7 = tt.make_tensor_descriptor %arg2, [%arg3, %arg4], [%4, %c1_i64] : <bf16>, <32x32xbf16>
  %8 = triton_cpu.extract_memref %5 : <32x32xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
  %9 = arith.index_cast %1 : i32 to index
  %10 = arith.index_cast %3 : i32 to index
  %11 = vector.transfer_read %8[%9, %10], %cst {in_bounds = [true, true]} : memref<?x?xbf16, strided<[?, 1]>>, vector<32x32xbf16>
  %12 = vector.transpose %11, [1, 0] : vector<32x32xbf16> to vector<32x32xbf16>
  %21 = triton_cpu.extract_memref %7 : <32x32xbf16> -> memref<?x?xbf16, strided<[?, 1]>>
  vector.transfer_write %12, %21[%9, %10] {in_bounds = [true, true]} : vector<32x32xbf16>, memref<?x?xbf16, strided<[?, 1]>>
  tt.return
}
